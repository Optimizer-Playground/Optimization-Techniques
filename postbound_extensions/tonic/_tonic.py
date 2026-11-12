"""Implementation of the TONIC algorithm for learned operator selections.

Copyright (C) 2026 Rico Bergmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import itertools
import json
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path
from typing import Literal, assert_never

import postbound as pb
from postbound.qal import SqlQuery

from ..util import wrap_logger


def _traverse_join_tree(node: pb.JoinTree) -> Generator[pb.JoinTree, None, None]:
    if node.is_scan():
        yield node
        return

    yield from _traverse_join_tree(node.outer_child)
    yield node.inner_child


def _traverse_query_plan(
    node: pb.QueryPlan,
) -> Generator[tuple[pb.QueryPlan, pb.QueryPlan | None], None, None]:
    if node.is_scan():
        yield node, None

    if node.is_join():
        assert node.outer_child is not None and node.inner_child is not None
        yield from _traverse_query_plan(node.outer_child)
        yield node.inner_child, node

    if node.is_auxiliary():
        assert node.input_node is not None
        yield from _traverse_query_plan(node.input_node)


def _cardinalities_only(parameters: pb.PlanParameterization) -> pb.PlanParameterization:
    scrubbed = pb.PlanParameterization()
    scrubbed.cardinalities = dict(parameters.cardinalities)
    return scrubbed


def _reorient_join_tree(join_tree: pb.JoinTree) -> pb.JoinTree:
    if join_tree.is_scan():
        return join_tree

    outer = _reorient_join_tree(join_tree.outer_child)
    inner = _reorient_join_tree(join_tree.inner_child)

    n_outer, n_inner = len(outer.tables()), len(inner.tables())
    if n_inner > n_outer:
        outer, inner = inner, outer
    elif n_inner == n_outer:
        # deterministic tie-break
        outer_tabs = sorted(outer.tables())
        inner_tabs = sorted(inner.tables())
        if inner_tabs < outer_tabs:
            outer, inner = inner, outer

    return pb.JoinTree.create_join(outer, inner)


def _reorient_query_plan(plan: pb.QueryPlan) -> pb.QueryPlan:
    if plan.is_scan():
        return plan

    if plan.is_join():
        assert plan.outer_child is not None and plan.inner_child is not None
        outer = _reorient_query_plan(plan.outer_child)
        inner = _reorient_query_plan(plan.inner_child)

        n_outer, n_inner = len(outer.tables()), len(inner.tables())

        if n_inner > n_outer:
            outer, inner = inner, outer
        elif n_inner == n_outer:
            # deterministic tie-break
            outer_tabs = sorted(outer.tables())
            inner_tabs = sorted(inner.tables())
            if inner_tabs < outer_tabs:
                outer, inner = inner, outer

        return pb.QueryPlan(
            plan.node_type,
            operator=plan.operator,
            children=[outer, inner],
            plan_params=plan.params,
            estimates=plan.estimates,
            measures=plan.measures,
            subplan=plan.subplan,
        )

    assert plan.input_node is not None
    input_node = _reorient_query_plan(plan.input_node)
    return pb.QueryPlan(
        plan.node_type,
        operator=plan.operator,
        children=[input_node],
        plan_params=plan.params,
        estimates=plan.estimates,
        measures=plan.measures,
        subplan=plan.subplan,
    )


BatchedFeedback = Mapping[frozenset[pb.TableReference], Mapping[pb.JoinOperator, pb.Cost]]


class QepsIdentifier:
    @staticmethod
    def empty() -> QepsIdentifier:
        return QepsIdentifier([], additional=[])

    @staticmethod
    def base(intermediate: Iterable[pb.TableReference]) -> QepsIdentifier:
        return QepsIdentifier(intermediate, additional=[])

    def __init__(
        self,
        intermediate: Iterable[pb.TableReference],
        *,
        additional: Iterable[pb.TableReference],
    ) -> None:
        self._identifier = frozenset(intermediate)
        self._additional = frozenset(additional)

    @property
    def intermediate(self) -> frozenset[pb.TableReference]:
        return self._identifier

    def is_plain(self) -> bool:
        return len(self._additional) == 1

    def add_to(self, other: QepsIdentifier) -> QepsIdentifier:
        combined = self._identifier.union(other._identifier)
        return QepsIdentifier(combined, additional=self.intermediate)

    def __json__(self) -> pb.util.jsondict:
        return {"identifier": self._identifier, "additional": self._additional}

    def __hash__(self) -> int:
        return hash(self._identifier)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QepsIdentifier):
            return NotImplemented
        return self._identifier == other._identifier

    def __repr__(self) -> str:
        return f"QepsIdentifier({self._identifier})"

    def __str__(self) -> str:
        return f"QepsIdentifier({self._identifier})"


class FilterAwareQepsIdentifier(QepsIdentifier):
    @staticmethod
    def infer(intermediate: Iterable[pb.TableReference], *, query: pb.qal.SelectStatement) -> FilterAwareQepsIdentifier:
        subquery = pb.transform.extract_subquery(query, intermediate)
        if not subquery.tables():
            return FilterAwareQepsIdentifier(intermediate, None, additional=[])

        filter_pred = subquery.where_clause.root if subquery.where_clause else None
        return FilterAwareQepsIdentifier(intermediate, filter_pred, additional=[])

    @staticmethod
    def base(
        intermediate: Iterable[pb.TableReference],
        filter_pred: pb.qal.AbstractPredicate | None = None,
    ) -> FilterAwareQepsIdentifier:
        return FilterAwareQepsIdentifier(intermediate, filter_pred, additional=[])

    def __init__(
        self,
        intermediate: Iterable[pb.TableReference],
        filter_pred: pb.qal.AbstractPredicate | None,
        *,
        additional: Iterable[pb.TableReference],
    ) -> None:
        super().__init__(intermediate, additional=additional)
        self._filter_pred = filter_pred

    def add_to(self, other: QepsIdentifier) -> FilterAwareQepsIdentifier:
        combined = self._identifier.union(other._identifier)
        return FilterAwareQepsIdentifier(combined, self._filter_pred, additional=self.intermediate)

    def __json__(self) -> pb.util.jsondict:
        return {
            "identifier": self._identifier,
            "filter": self._filter_pred,
            "additional": self._additional,
        }

    def __hash__(self) -> int:
        return hash((self._identifier, self._filter_pred))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FilterAwareQepsIdentifier):
            return NotImplemented
        return self._identifier == other._identifier and self._filter_pred == other._filter_pred

    def __repr__(self) -> str:
        return f"FilterAwareQepsIdentifier({self._identifier}, {self._filter_pred})"

    def __str__(self) -> str:
        return f"FilterAwareQepsIdentifier({self._identifier}, {self._filter_pred})"


def load_qeps_id_json(
    json_data: str | dict,
) -> QepsIdentifier | FilterAwareQepsIdentifier:
    """Creates a QEPS identifier from its JSON representation.

    Whether the identifier is filter-aware or not is determined automatically from the JSON data.
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    identifier = [pb.parser.load_table_json(raw) for raw in json_data["identifier"]]
    additional = [pb.parser.load_table_json(raw) for raw in json_data.get("additional", [])]

    # We cannot simply do json_data.get("filter") here, b/c a None value is also used if the the QEPS ID is
    # filter-aware, but the current ID does not have a filter. Duplicate checks or Python 3.15 sentinel are required.
    if "filter" not in json_data:
        return QepsIdentifier(identifier, additional=additional)

    raw_filter = json_data.get("filter")
    filter_pred = pb.parser.load_predicate_json(raw_filter)
    return FilterAwareQepsIdentifier(identifier, filter_pred, additional=additional)


class QepsNode(ABC):
    @property
    @abstractmethod
    def identifier(self) -> QepsIdentifier:
        raise NotImplementedError

    @abstractmethod
    def follow(
        self,
        intermediate: Iterable[pb.TableReference],
        *,
        query: pb.qal.SelectStatement | None,
    ) -> QepsNode:
        raise NotImplementedError

    @abstractmethod
    def recommend(
        self,
        current_recommendation: pb.PhysicalOperatorAssignment,
        intermediate: pb.JoinTree,
        *,
        query: pb.qal.SelectStatement | None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def feedback(
        self,
        intermediate: pb.QueryPlan,
        *,
        operator: pb.JoinOperator,
        cost: pb.Cost,
        query: pb.qal.SelectStatement | None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def feedback_batch(
        self,
        intermediate: pb.JoinTree,
        *,
        batch: BatchedFeedback,
        query: pb.qal.SelectStatement | None,
    ) -> None:
        raise NotImplementedError


class PlainQeps(QepsNode):
    @staticmethod
    def empty(gamma: float) -> PlainQeps:
        return PlainQeps(QepsIdentifier.empty(), gamma=gamma)

    def __init__(self, identifier: QepsIdentifier, *, gamma: float) -> None:
        self._identifier = identifier
        self._gamma = gamma

        self._cost_summary: dict[pb.JoinOperator, pb.Cost] = {}
        self._children: dict[QepsIdentifier, QepsNode] = {}

    @property
    def identifier(self) -> QepsIdentifier:
        return self._identifier

    def follow(
        self,
        intermediate: Iterable[pb.TableReference],
        *,
        query: pb.qal.SelectStatement | None,
    ) -> QepsNode:
        child_identifier = (
            QepsIdentifier.base(intermediate).add_to(self._identifier)
            if query is None
            else FilterAwareQepsIdentifier.infer(intermediate, query=query).add_to(self._identifier)
        )
        child = self._children.get(child_identifier)
        if child is not None:
            return child

        if child_identifier.is_plain():
            child = PlainQeps(child_identifier, gamma=self._gamma)
        else:
            child = SubqueryQeps(child_identifier, gamma=self._gamma)

        self._children[child_identifier] = child
        return child

    def recommend(
        self,
        current_recommendation: pb.PhysicalOperatorAssignment,
        intermediate: pb.JoinTree,
        *,
        query: pb.qal.SelectStatement | None,
    ) -> None:
        if not self._cost_summary:
            return

        best_op = pb.util.argmin(self._cost_summary)
        current_recommendation.add(best_op, self._identifier.intermediate)

    def feedback(
        self,
        intermediate: pb.QueryPlan,
        *,
        operator: pb.JoinOperator,
        cost: pb.Cost,
        query: pb.qal.SelectStatement | None,
    ) -> None:
        current_cost = self._cost_summary.get(operator)
        if current_cost is None:
            self._cost_summary[operator] = cost
            return

        self._cost_summary[operator] = cost + self._gamma * current_cost

    def feedback_batch(
        self,
        intermediate: pb.JoinTree,
        *,
        batch: BatchedFeedback,
        query: pb.qal.SelectStatement | None,
    ) -> None:
        batch_cost = batch.get(self._identifier.intermediate)
        if batch_cost is None:
            return

        for op, cost in batch_cost.items():
            current_cost = self._cost_summary.get(op)
            if current_cost is None:
                self._cost_summary[op] = cost
            else:
                self._cost_summary[op] = cost + self._gamma * current_cost

    def __json__(self) -> pb.util.jsondict:
        return {
            "identifier": self._identifier,
            "cost_summary": [{"operator": op, "cost": cost} for op, cost in self._cost_summary.items()],
            "gamma": self._gamma,
            "children": list(self._children.values()),
        }


class SubqueryQeps(QepsNode):
    def __init__(self, identifier: QepsIdentifier, *, gamma: float) -> None:
        self._identifier = identifier
        self._gamma = gamma

        self._subquery_root: QepsNode = PlainQeps.empty(gamma=gamma)
        self._cost_summary: dict[pb.JoinOperator, pb.Cost] = {}
        self._children: dict[QepsIdentifier, QepsNode] = {}

    @property
    def identifier(self) -> QepsIdentifier:
        return self._identifier

    def follow(
        self,
        intermediate: Iterable[pb.TableReference],
        *,
        query: pb.qal.SelectStatement | None,
    ) -> QepsNode:
        child_identifier = (
            QepsIdentifier.base(intermediate).add_to(self._identifier)
            if query is None
            else FilterAwareQepsIdentifier.infer(intermediate, query=query).add_to(self._identifier)
        )
        child = self._children.get(child_identifier)
        if child is not None:
            return child

        if child_identifier.is_plain():
            child = PlainQeps(child_identifier, gamma=self._gamma)
        else:
            child = SubqueryQeps(child_identifier, gamma=self._gamma)

        self._children[child_identifier] = child
        return child

    def recommend(
        self,
        current_recommendation: pb.PhysicalOperatorAssignment,
        intermediate: pb.JoinTree,
        *,
        query: pb.qal.SelectStatement | None,
    ) -> None:
        _generate_recommendations(
            current_recommendation,
            qeps=self._subquery_root,
            join_order=intermediate,
            query=query,
        )

        if not self._cost_summary:
            return

        best_op = pb.util.argmin(self._cost_summary)
        current_recommendation.add(best_op, set(self._identifier.intermediate))

    def feedback(
        self,
        intermediate: pb.QueryPlan,
        *,
        operator: pb.JoinOperator,
        cost: pb.Cost,
        query: pb.qal.SelectStatement | None,
    ) -> None:
        subquery_qeps = self._subquery_root
        for inner_node, join_node in _traverse_query_plan(intermediate):
            subquery_qeps = subquery_qeps.follow(inner_node.tables(), query=query)
            if join_node is None:
                continue
            assert isinstance(join_node.operator, pb.JoinOperator)
            subquery_qeps.feedback(
                inner_node,
                operator=join_node.operator,
                cost=join_node.estimated_cost,
                query=query,
            )

        current_cost = self._cost_summary.get(operator)
        if current_cost is None:
            self._cost_summary[operator] = cost
            return

        self._cost_summary[operator] = cost + self._gamma * current_cost

    def feedback_batch(
        self,
        intermediate: pb.JoinTree,
        *,
        batch: BatchedFeedback,
        query: pb.qal.SelectStatement | None,
    ) -> None:
        subquery_qeps = self._subquery_root
        for node in _traverse_join_tree(intermediate):
            subquery_qeps = subquery_qeps.follow(node.tables(), query=query)
            subquery_qeps.feedback_batch(node, batch=batch, query=query)

        batch_cost = batch.get(self._identifier.intermediate)
        if batch_cost is None:
            return

        for op, cost in batch_cost.items():
            current_cost = self._cost_summary.get(op)
            if current_cost is None:
                self._cost_summary[op] = cost
            else:
                self._cost_summary[op] = cost + self._gamma * current_cost

    def __json__(self) -> pb.util.jsondict:
        return {
            "identifier": self._identifier,
            "cost_summary": [{"operator": op, "cost": cost} for op, cost in self._cost_summary.items()],
            "gamma": self._gamma,
            "subquery": self._subquery_root,
            "children": list(self._children.values()),
        }


def load_qeps_json(json_data: str | dict) -> QepsNode:
    """Creates a QEPS node from its JSON representation.

    Whether the node is filter-aware or not is determined automatically from the JSON data.
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    identifier = load_qeps_id_json(json_data["identifier"])
    gamma = json_data["gamma"]
    cost_summary = {
        pb.opt.read_operator_json(summary_json["operator"]): summary_json["cost"]
        for summary_json in json_data["cost_summary"]
    }
    if any(not isinstance(op, pb.JoinOperator) for op in cost_summary):
        raise ValueError("Cost summary contains non-join operators.")

    children = [load_qeps_json(child) for child in json_data["children"]]

    raw_subquery = json_data.get("subquery")
    if raw_subquery is None:
        qeps = PlainQeps(identifier, gamma=gamma)
        qeps._cost_summary = cost_summary  # type: ignore -- guarded by isinstance check above
        qeps._children = {child.identifier: child for child in children}
        return qeps

    subquery = load_qeps_json(raw_subquery)
    qeps = SubqueryQeps(identifier, gamma=gamma)
    qeps._cost_summary = cost_summary  # type: ignore -- guarded by isinstance check above
    qeps._subquery_root = subquery
    qeps._children = {child.identifier: child for child in children}
    return qeps


def _generate_recommendations(
    assignment: pb.PhysicalOperatorAssignment,
    *,
    qeps: QepsNode,
    join_order: pb.JoinTree,
    query: pb.qal.SelectStatement | None,
):
    for intermediate in _traverse_join_tree(join_order):
        qeps = qeps.follow(intermediate.tables(), query=query)
        qeps.recommend(assignment, intermediate, query=query)


def load_tonic_json(json_data: str | dict, *, database: pb.Database) -> TonicOperators:
    """Creates a TONIC instance from its JSON representation."""
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data

    gamma = json_data["gamma"]
    filter_aware = json_data["filter_aware"]
    retrain = json_data.get("retrain", True)
    prediction_target = json_data.get("prediction_target", "true_cost")
    exploration = json_data.get("exploration", "linear")
    analyze_timeout_ms = json_data.get("analyze_timeout_ms", None)

    tonic = TonicOperators(
        database=database,
        gamma=gamma,
        filter_aware=filter_aware,
        retrain=retrain,
        prediction_target=prediction_target,
        exploration=exploration,
        analyze_timeout_ms=analyze_timeout_ms,
    )

    qeps = load_qeps_json(json_data["qeps"])
    tonic._qeps = qeps  # type: ignore -- guarded by isinstance check above

    return tonic


ExplorationMode = Literal["exhaustive", "random", "linear", "off"]


class TonicOperators(pb.OperatorSelection):
    @staticmethod
    def pre_trained(
        archive: Path | str,
        *,
        database: pb.Database,
        retrain: bool = True,
    ) -> TonicOperators:
        archive = Path(archive)
        if not archive.is_file():
            raise FileNotFoundError(f"Archive file not found: {archive}")

        with archive.open("r", encoding="utf-8") as f:
            json_data = json.load(f)
        tonic = load_tonic_json(json_data, database=database)
        tonic.retrain = retrain
        return tonic

    @staticmethod
    def load_or_build(
        archive: Path | str,
        *,
        database: pb.Database | None = None,
        gamma: float = 0.8,
        filter_aware: bool = False,
        sample_plans: Mapping[pb.SqlQuery, pb.QueryPlan] | None = None,
        retrain: bool = True,
        verbose: bool | pb.util.Logger = False,
    ) -> TonicOperators:
        logger = wrap_logger(verbose)
        archive = Path(archive)

        if archive.is_file():
            logger(f"Loading pre-trained TONIC model from {archive}")
            return TonicOperators.pre_trained(archive, database=database or pb.db.current_database(), retrain=retrain)

        tonic = TonicOperators(database=database, gamma=gamma, filter_aware=filter_aware, retrain=True)

        if sample_plans:
            logger(f"Training TONIC model from {len(sample_plans)} sample plans")
            for query, plan in sample_plans.items():
                tonic.learn_from_feedback(query, plan, exec_time=plan.execution_time)
        tonic.retrain = retrain

        logger(f"Storing trained TONIC model to {archive}")
        tonic.store(archive)

        return tonic

    def __init__(
        self,
        database: pb.Database | None = None,
        *,
        gamma: float = 0.8,
        filter_aware: bool = False,
        retrain: bool = True,
        exploration: int | ExplorationMode = "linear",
        prediction_target: Literal["true_cost", "execution_time"] = "true_cost",
        analyze_timeout_ms: float | None = None,
        log: pb.util.Logger | None = None,
    ) -> None:
        super().__init__()
        self._database = database or pb.db.current_database()
        self._qeps = PlainQeps.empty(gamma=gamma)
        self._gamma = gamma
        self._filter_aware = filter_aware
        self._retrain = retrain
        self._prediction_target = prediction_target

        self._exploration_mode = "random" if isinstance(exploration, int) else exploration
        if isinstance(exploration, int):
            self._exploration_target = exploration
        elif exploration == "random":
            self._exploration_target = 10
        else:
            self._exploration_target = -1
        self._analyze_timeout_ms = analyze_timeout_ms

        self._allowed_join_ops = [op for op in pb.JoinOperator if self._database.hinting().supports_hint(op)]
        self._log = log or pb.util.standard_logger(False)

    @property
    def retrain(self) -> bool:
        return self._retrain

    @retrain.setter
    def retrain(self, value: bool) -> None:
        self._retrain = value

    @property
    def exploration_mode(self) -> ExplorationMode:
        return self._exploration_mode

    @exploration_mode.setter
    def exploration_mode(self, value: ExplorationMode) -> None:
        self._exploration_mode = value
        if self._exploration_mode == "random" and self._exploration_target <= 0:
            self._exploration_target = 10

    @property
    def exploration_target(self) -> int:
        return self._exploration_target

    @exploration_target.setter
    def exploration_target(self, value: int) -> None:
        if self._exploration_mode != "random":
            raise ValueError("Exploration target can only be set in 'random' exploration mode.")
        if value <= 0:
            raise ValueError("Exploration target must be a positive integer.")
        self._exploration_target = value

    def select_physical_operators(
        self, query: pb.SqlQuery, join_order: pb.JoinTree | None
    ) -> pb.PhysicalOperatorAssignment:
        if not pb.qal.is_select_query(query):
            raise pb.qal.QueryTypeError.expected_select(query)

        join_order = join_order or self._fallback_native_join_order(query)
        join_order = _reorient_join_tree(join_order)
        assignment = pb.PhysicalOperatorAssignment()

        query_ctx = query if self._filter_aware else None
        _generate_recommendations(assignment, qeps=self._qeps, join_order=join_order, query=query_ctx)
        return assignment

    def learn_from_feedback(
        self,
        query: SqlQuery,
        result_set: pb.db.ResultSet | pb.QueryPlan,
        *,
        exec_time: pb.TimeMs = float("nan"),
    ) -> pb.train.TrainingMetrics:
        if not pb.qal.is_select_query(query):
            raise pb.qal.QueryTypeError.expected_select(query)
        if not self._retrain:
            return {"status": "skipped", "details": "disabled"}

        if isinstance(result_set, pb.QueryPlan):
            plan = result_set
        else:
            plan = self._database.optimizer().parse_plan(result_set, query=query)

        if not plan:
            return {"status": "failure", "details": "could not parse plan"}

        match self._exploration_mode:
            case "random":
                return self._explore_random(plan, query=query)
            case "exhaustive":
                return self._explore_exhaustive(plan, query=query)
            case "linear":
                return self._explore_linear(plan, query=query)
            case "off":
                return self._explore_single(plan, query=query)
            case _:
                assert_never(self._exploration_mode)
                raise ValueError(f"Invalid exploration mode: {self._exploration_mode}. ")

    def store(self, archive: Path | str) -> None:
        archive = Path(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("w", encoding="utf-8") as f:
            pb.util.to_json_dump(self, f)

    def _explore_single(self, plan: pb.QueryPlan, *, query: pb.qal.SelectStatement) -> pb.train.TrainingMetrics:
        train_start = time.perf_counter_ns()

        match self._prediction_target:
            case "true_cost":
                true_card_query = self._database.hinting().generate_hints(query, plan.with_actual_card())
                plan = self._database.optimizer().query_plan(true_card_query)

            case "execution_time":
                plan = plan.with_runtime_as_cost()

            case _:
                assert_never(self._prediction_target)
                raise ValueError(f"Invalid prediction target: {self._prediction_target}. ")

        self._add_feedback(plan, query=query)

        train_end = time.perf_counter_ns()
        training_time_ms = (train_end - train_start) / 1_000_000
        return {
            "status": "success",
            "mode": "single",
            "training_time_ms": training_time_ms,
            "n_explored": 1,
            "n_timeout": 0,
        }

    def _explore_exhaustive(self, plan: pb.QueryPlan, *, query: pb.qal.SelectStatement) -> pb.train.TrainingMetrics:
        all_joins = [join.tables() for join in plan.find_all_nodes(pb.QueryPlan.is_join)]
        operators = [self._allowed_join_ops] * len(all_joins)

        join_order = pb.jointree_from_plan(plan)
        plan_params = _cardinalities_only(pb.parameters_from_plan(plan, target_cardinality="actual"))

        exploration_start = time.perf_counter_ns()

        n_explored, n_timeout = 0, 0
        for selection in itertools.product(*operators):
            assignment = pb.PhysicalOperatorAssignment()
            for intermediate, operator in zip(all_joins, selection):
                assignment.add(operator, intermediate)

            match self._prediction_target:
                case "true_cost":
                    hinted_query = self._database.hinting().generate_hints(
                        query,
                        join_order=join_order,
                        physical_operators=assignment,
                        plan_parameters=plan_params,
                    )
                    current_plan = self._database.optimizer().query_plan(hinted_query)

                case "execution_time":
                    current_plan = self._analyze_plan(query, join_order=join_order, operators=assignment)
                    current_plan = current_plan.with_runtime_as_cost() if current_plan is not None else None

                case _:
                    assert_never(self._prediction_target)
                    raise ValueError(f"Invalid prediction target: {self._prediction_target}. ")

            if current_plan is None:
                self._log(f"Timeout for query {query} with join {intermediate} = {operator}. Skipping.")
                n_timeout += 1
                continue
            else:
                n_explored += 1

            self._add_feedback(current_plan, query=query)

        exploration_end = time.perf_counter_ns()
        exploration_time_ms = (exploration_end - exploration_start) / 1_000_000
        return {
            "status": "success",
            "mode": "exhaustive",
            "exploration_time_ms": exploration_time_ms,
            "n_explored": n_explored,
            "n_timeout": n_timeout,
        }

    def _explore_random(self, plan: pb.QueryPlan, *, query: pb.qal.SelectStatement) -> pb.train.TrainingMetrics:
        all_joins = [join.tables() for join in plan.find_all_nodes(pb.QueryPlan.is_join)]

        join_order = pb.jointree_from_plan(plan)
        plan_params = _cardinalities_only(pb.parameters_from_plan(plan, target_cardinality="actual"))

        n_combinations = len(self._allowed_join_ops) ** len(all_joins)
        if n_combinations <= self._exploration_target:
            self._log(
                f"Too few distinct join operator combinations for query {query}. Falling back to exhaustive exploration."
            )
            return self._explore_exhaustive(plan, query=query)

        exploration_start = time.perf_counter_ns()

        n_explored, n_timeout = 0, 0
        for _ in range(self._exploration_target):
            assignment = pb.PhysicalOperatorAssignment()
            for intermediate in all_joins:
                operator = random.choice(self._allowed_join_ops)
                assignment.add(operator, intermediate)

            match self._prediction_target:
                case "true_cost":
                    hinted_query = self._database.hinting().generate_hints(
                        query,
                        join_order=join_order,
                        physical_operators=assignment,
                        plan_parameters=plan_params,
                    )
                    current_plan = self._database.optimizer().query_plan(hinted_query)

                case "execution_time":
                    current_plan = self._analyze_plan(query, join_order=join_order, operators=assignment)
                    current_plan = current_plan.with_runtime_as_cost() if current_plan is not None else None

                case _:
                    assert_never(self._prediction_target)
                    raise ValueError(f"Invalid prediction target: {self._prediction_target}. ")

            if current_plan is None:
                self._log(f"Timeout for query {query} with join {intermediate} = {operator}. Skipping.")
                n_timeout += 1
                continue
            else:
                n_explored += 1

            self._add_feedback(current_plan, query=query)

        exploration_end = time.perf_counter_ns()
        exploration_time_ms = (exploration_end - exploration_start) / 1_000_000
        return {
            "status": "success",
            "mode": "random",
            "exploration_time_ms": exploration_time_ms,
            "n_explored": n_explored,
            "n_timeout": n_timeout,
        }

    def _explore_linear(self, plan: pb.QueryPlan, *, query: pb.qal.SelectStatement) -> pb.train.TrainingMetrics:
        all_joins = {
            frozenset(join.tables()): join.operator
            for join in plan.find_all_nodes(pb.QueryPlan.is_join)
            if join.operator is not None
        }

        join_order = pb.jointree_from_plan(plan)
        plan_params = _cardinalities_only(pb.parameters_from_plan(plan, target_cardinality="actual"))

        exploration_start = time.perf_counter_ns()

        base_assignment = pb.PhysicalOperatorAssignment()
        for join, op in all_joins.items():
            base_assignment.add(op, join)
        match self._prediction_target:
            case "true_cost":
                hinted_query = self._database.hinting().generate_hints(
                    query,
                    join_order=join_order,
                    physical_operators=base_assignment,
                    plan_parameters=plan_params,
                )
                base_plan = self._database.optimizer().query_plan(hinted_query)

            case "execution_time":
                # the base plan was already executed
                base_plan = plan.with_runtime_as_cost()

            case _:
                assert_never(self._prediction_target)
                raise ValueError(f"Invalid prediction target: {self._prediction_target}. ")

        cost_vector: dict[frozenset[pb.TableReference], dict[pb.JoinOperator, float]] = {}
        for node in base_plan.find_all_nodes(pb.QueryPlan.is_join):
            if node.operator is None:
                continue
            assert isinstance(node.operator, pb.JoinOperator)
            intermediate = frozenset(node.tables())

            cost_vector[intermediate] = {node.operator: node.estimated_cost}

        n_explored, n_timeout = 1, 0  # base plan is already explored
        for join, current_op in all_joins.items():
            for operator in self._allowed_join_ops:
                if operator == current_op:
                    continue

                assignment = base_assignment.clone()
                assignment.add(operator, join)  # overwrite the operator for this join

                match self._prediction_target:
                    case "true_cost":
                        hinted_query = self._database.hinting().generate_hints(
                            query,
                            join_order=join_order,
                            physical_operators=assignment,
                            plan_parameters=plan_params,
                        )
                        current_plan = self._database.optimizer().query_plan(hinted_query)

                    case "execution_time":
                        current_plan = self._analyze_plan(query, join_order=join_order, operators=assignment)
                        current_plan = current_plan.with_runtime_as_cost() if current_plan is not None else None

                    case _:
                        assert_never(self._prediction_target)
                        raise ValueError(f"Invalid prediction target: {self._prediction_target}. ")

                if current_plan is None:
                    self._log(f"Timeout for query {query} with join {join} = {operator}. Skipping.")
                    n_timeout += 1
                    continue
                else:
                    n_explored += 1

                join_node = current_plan.find_first_node(
                    lambda n: n.is_join() and frozenset(n.tables()) == join  # noqa: B023 - this is a false positive
                )
                assert join_node is not None
                cost_vector[join][operator] = join_node.estimated_cost

        current_qeps = self._qeps
        query_ctx = query if self._filter_aware else None

        join_order = _reorient_join_tree(join_order)
        for node in _traverse_join_tree(join_order):
            current_qeps = current_qeps.follow(node.tables(), query=query_ctx)
            current_qeps.feedback_batch(node, batch=cost_vector, query=query_ctx)

        exploration_end = time.perf_counter_ns()
        exploration_time_ms = (exploration_end - exploration_start) / 1_000_000
        return {
            "status": "success",
            "mode": "linear",
            "exploration_time_ms": exploration_time_ms,
            "n_explored": n_explored,
            "n_timeout": n_timeout,
        }

    def _add_feedback(self, plan: pb.QueryPlan, *, query: pb.qal.SelectStatement) -> pb.train.TrainingMetrics:
        current_qeps = self._qeps
        query_ctx = query if self._filter_aware else None

        plan = _reorient_query_plan(plan)
        for inner_node, join_node in _traverse_query_plan(plan):
            current_qeps = current_qeps.follow(inner_node.tables(), query=query_ctx)
            if join_node is None:
                continue
            assert isinstance(join_node.operator, pb.JoinOperator)
            current_qeps.feedback(
                inner_node,
                operator=join_node.operator,
                cost=join_node.estimated_cost,
                query=query_ctx,
            )

    def _fallback_native_join_order(self, query: pb.qal.SelectStatement) -> pb.JoinTree:
        plan = self._database.optimizer().query_plan(query)
        return pb.jointree_from_plan(plan)

    def _analyze_plan(
        self,
        query: pb.SqlQuery,
        *,
        join_order: pb.JoinTree,
        operators: pb.PhysicalOperatorAssignment,
    ) -> pb.QueryPlan | None:
        hinted_query = self._database.hinting().generate_hints(
            query, join_order=join_order, physical_operators=operators
        )

        if self._analyze_timeout_ms is not None:
            return self._analyze_with_timeout(hinted_query)
        return self._analyze_no_timeout(hinted_query)

    def _analyze_no_timeout(self, query: pb.SqlQuery) -> pb.QueryPlan:
        return self._database.optimizer().analyze_plan(query)

    def _analyze_with_timeout(self, query: pb.SqlQuery) -> pb.QueryPlan | None:
        if not isinstance(self._database, pb.db.TimeoutSupport):
            raise NotImplementedError("Database does not support query timeouts.")

        assert self._analyze_timeout_ms is not None
        query = pb.transform.as_explain_analyze(query)
        result_set = self._database.execute_with_timeout(query, timeout=self._analyze_timeout_ms / 1000)
        if result_set is None:
            return None

        return self._database.optimizer().parse_plan(result_set, query=query)

    def __json__(self) -> pb.util.jsondict:
        return {
            "qeps": self._qeps,
            "gamma": self._gamma,
            "filter_aware": self._filter_aware,
            "retrain": self._retrain,
            "prediction_target": self._prediction_target,
            "exploration": self._exploration_target if self._exploration_mode == "random" else self._exploration_mode,
            "analyze_timeout_ms": self._analyze_timeout_ms,
        }
