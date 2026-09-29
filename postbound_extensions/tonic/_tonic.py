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

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path
from typing import Literal

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


class QepsIdentifier:
    @staticmethod
    def empty() -> QepsIdentifier:
        return QepsIdentifier([])

    def __init__(self, intermediate: Iterable[pb.TableReference]) -> None:
        self._identifier = frozenset(intermediate)

    @property
    def intermediate(self) -> frozenset[pb.TableReference]:
        return self._identifier

    def is_plain(self) -> bool:
        return len(self._identifier) == 1

    def combine_with(self, other: QepsIdentifier) -> QepsIdentifier:
        combined = self._identifier.union(other._identifier)
        return QepsIdentifier(combined)

    def __json__(self) -> pb.util.jsondict:
        return {"identifier": self._identifier}

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
    def infer(
        intermediate: Iterable[pb.TableReference], *, query: pb.qal.SelectStatement
    ) -> FilterAwareQepsIdentifier:
        subquery = pb.transform.extract_subquery(query, intermediate)
        if len(subquery.tables()):
            return FilterAwareQepsIdentifier(intermediate, None)

        filter_pred = subquery.where_clause.root if subquery.where_clause else None
        return FilterAwareQepsIdentifier(intermediate, filter_pred)

    def __init__(
        self,
        intermediate: Iterable[pb.TableReference],
        filter_pred: pb.qal.AbstractPredicate | None,
    ) -> None:
        super().__init__(intermediate)
        self._filter_pred = filter_pred

    def combine_with(self, other: QepsIdentifier) -> FilterAwareQepsIdentifier:
        combined = self._identifier.union(other._identifier)
        return FilterAwareQepsIdentifier(combined, self._filter_pred)

    def __json__(self) -> pb.util.jsondict:
        return {"identifier": self._identifier, "filter": self._filter_pred}

    def __hash__(self) -> int:
        return hash((self._identifier, self._filter_pred))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FilterAwareQepsIdentifier):
            return NotImplemented
        return (
            self._identifier == other._identifier
            and self._filter_pred == other._filter_pred
        )

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
    raw_filter = json_data.get("filter")
    if raw_filter is None:
        return QepsIdentifier(identifier)

    filter_pred = pb.parser.load_predicate_json(raw_filter)
    return FilterAwareQepsIdentifier(identifier, filter_pred)


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
            QepsIdentifier(intermediate).combine_with(self._identifier)
            if query is None
            else FilterAwareQepsIdentifier.infer(
                intermediate, query=query
            ).combine_with(self._identifier)
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
        print(self._identifier, "::", self._cost_summary)
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

    def __json__(self) -> pb.util.jsondict:
        return {
            "identifier": self._identifier,
            "cost_summary": self._cost_summary,
            "gamma": self._gamma,
            "children": self._children,
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
            QepsIdentifier(intermediate).combine_with(self._identifier)
            if query is None
            else FilterAwareQepsIdentifier.infer(
                intermediate, query=query
            ).combine_with(self._identifier)
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

    def __json__(self) -> pb.util.jsondict:
        return {
            "identifier": self._identifier,
            "cost_summary": self._cost_summary,
            "gamma": self._gamma,
            "subquery": self._subquery_root,
            "children": self._children,
        }


def load_qeps_json(json_data: str | dict) -> QepsNode:
    """Creates a QEPS node from its JSON representation.

    Whether the node is filter-aware or not is determined automatically from the JSON data.
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    identifier = load_qeps_id_json(json_data["identifier"])
    gamma = json_data["gamma"]
    cost_summary = {
        pb.opt.read_operator_json(k): v for k, v in json_data["cost_summary"].items()
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

    tonic = TonicOperators(
        database=database,
        gamma=gamma,
        filter_aware=filter_aware,
        retrain=retrain,
        prediction_target=prediction_target,
    )

    qeps = load_qeps_json(json_data["qeps"])
    tonic._qeps = qeps  # type: ignore -- guarded by isinstance check above

    return tonic


def _scrub_query(query: pb.qal.SelectStatement) -> pb.qal.SelectStatement:
    renamings = {tab: tab.drop_alias() for tab in query.tables()}
    return pb.transform.rename_table(query, renamings)


def _scrub_join_tree(join_tree: pb.JoinTree) -> pb.JoinTree:
    if join_tree.is_scan():
        scrubbed = join_tree.base_table.drop_alias()
        return pb.JoinTree.create_scan(scrubbed)

    scrubbed_outer = _scrub_join_tree(join_tree.outer_child)
    scrubbed_inner = _scrub_join_tree(join_tree.inner_child)
    return pb.JoinTree.create_join(scrubbed_outer, scrubbed_inner)


def _scrub_query_plan(plan: pb.QueryPlan) -> pb.QueryPlan:
    if plan.is_scan():
        assert plan.base_table is not None
        scrubbed = plan.base_table.drop_alias()
        return pb.QueryPlan(
            plan.node_type,
            base_table=scrubbed,
            estimates=plan.estimates,
            measures=plan.measures,
        )

    if plan.is_join():
        assert plan.outer_child is not None and plan.inner_child is not None
        scrubbed_outer = _scrub_query_plan(plan.outer_child)
        scrubbed_inner = _scrub_query_plan(plan.inner_child)
        return pb.QueryPlan(
            plan.node_type,
            children=[scrubbed_outer, scrubbed_inner],
            estimates=plan.estimates,
            measures=plan.measures,
        )

    assert plan.input_node is not None
    scrubbed_input = _scrub_query_plan(plan.input_node)
    return pb.QueryPlan(
        plan.node_type,
        children=[scrubbed_input],
        estimates=plan.estimates,
        measures=plan.measures,
    )


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
            return TonicOperators.pre_trained(
                archive, database=database or pb.db.current_database(), retrain=retrain
            )

        tonic = TonicOperators(
            database=database, gamma=gamma, filter_aware=filter_aware, retrain=True
        )

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
        prediction_target: Literal["true_cost", "execution_time"] = "true_cost",
    ) -> None:
        super().__init__()
        self._database = database or pb.db.current_database()
        self._qeps = PlainQeps.empty(gamma=gamma)
        self._gamma = gamma
        self._filter_aware = filter_aware
        self._retrain = retrain
        self._prediction_target = prediction_target

    @property
    def retrain(self) -> bool:
        return self._retrain

    @retrain.setter
    def retrain(self, value: bool) -> None:
        self._retrain = value

    def select_physical_operators(
        self, query: pb.SqlQuery, join_order: pb.JoinTree | None
    ) -> pb.PhysicalOperatorAssignment:
        if not pb.qal.is_select_query(query):
            raise pb.qal.QueryTypeError.expected_select(query)

        join_order = join_order or self._fallback_native_join_order(query)
        assignment = pb.PhysicalOperatorAssignment()

        query_ctx = query if self._filter_aware else None
        _generate_recommendations(
            assignment, qeps=self._qeps, join_order=join_order, query=query_ctx
        )
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

        current_qeps = self._qeps
        query_ctx = query if self._filter_aware else None

        train_start = time.perf_counter_ns()

        match self._prediction_target:
            case "true_cost":
                true_card_query = self._database.hinting().generate_hints(
                    query, plan.with_actual_card()
                )
                plan = self._database.optimizer().query_plan(true_card_query)

            case "execution_time":
                plan = plan.with_runtime_as_cost()

            case _:
                raise ValueError(
                    f"Invalid prediction target: {self._prediction_target}. "
                )

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
        train_end = time.perf_counter_ns()
        training_time_ms = (train_end - train_start) / 1_000_000
        return {"status": "success", "training_time_ms": training_time_ms}

    def store(self, archive: Path | str) -> None:
        archive = Path(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("w", encoding="utf-8") as f:
            pb.util.to_json_dump(self, f)

    def _fallback_native_join_order(self, query: pb.qal.SelectStatement) -> pb.JoinTree:
        plan = self._database.optimizer().query_plan(query)
        return pb.jointree_from_plan(plan)

    def __json__(self) -> pb.util.jsondict:
        return {
            "qeps": self._qeps,
            "gamma": self._gamma,
            "filter_aware": self._filter_aware,
            "retrain": self._retrain,
            "prediction_target": self._prediction_target,
        }
