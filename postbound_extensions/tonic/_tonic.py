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

import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable

import postbound as pb
from postbound.qal import SqlQuery


def _traverse_join_tree(node: pb.JoinTree) -> Generator[pb.JoinTree, None, None]:
    if node.is_scan():
        yield node
        return

    yield from _traverse_join_tree(node.outer_child)
    yield node.inner_child


def _traverse_query_plan(node: pb.QueryPlan) -> Generator[tuple[pb.QueryPlan, pb.QueryPlan | None], None, None]:
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
        if len(subquery.tables()):
            return FilterAwareQepsIdentifier(intermediate, None)

        filter_pred = subquery.where_clause.root if subquery.where_clause else None
        return FilterAwareQepsIdentifier(intermediate, filter_pred)

    def __init__(self, intermediate: Iterable[pb.TableReference], filter_pred: pb.qal.AbstractPredicate | None) -> None:
        super().__init__(intermediate)
        self._filter_pred = filter_pred

    def combine_with(self, other: QepsIdentifier) -> FilterAwareQepsIdentifier:
        combined = self._identifier.union(other._identifier)
        return FilterAwareQepsIdentifier(combined, self._filter_pred)

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


class QepsNode(ABC):
    @abstractmethod
    def follow(self, intermediate: Iterable[pb.TableReference], *, query: pb.qal.SelectStatement | None) -> QepsNode:
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

    def follow(self, intermediate: Iterable[pb.TableReference], *, query: pb.qal.SelectStatement | None) -> QepsNode:
        child_identifier = (
            QepsIdentifier(intermediate).combine_with(self._identifier)
            if query is None
            else FilterAwareQepsIdentifier.infer(intermediate, query=query).combine_with(self._identifier)
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


class SubqueryQeps(QepsNode):
    def __init__(self, identifier: QepsIdentifier, *, gamma: float) -> None:
        self._identifier = identifier
        self._gamma = gamma

        self._subquery_root = PlainQeps.empty(gamma=gamma)
        self._cost_summary: dict[pb.JoinOperator, pb.Cost] = {}
        self._children: dict[QepsIdentifier, QepsNode] = {}

    def follow(self, intermediate: Iterable[pb.TableReference], *, query: pb.qal.SelectStatement | None) -> QepsNode:
        child_identifier = (
            QepsIdentifier(intermediate).combine_with(self._identifier)
            if query is None
            else FilterAwareQepsIdentifier.infer(intermediate, query=query).combine_with(self._identifier)
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
            current_recommendation, qeps=self._subquery_root, join_order=intermediate, query=query
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
            subquery_qeps.feedback(inner_node, operator=join_node.operator, cost=join_node.estimated_cost, query=query)

        current_cost = self._cost_summary.get(operator)
        if current_cost is None:
            self._cost_summary[operator] = cost
            return

        self._cost_summary[operator] = cost + self._gamma * current_cost


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


class TonicOperators(pb.OperatorSelection):
    def __init__(self, database: pb.Database | None = None, *, gamma: float = 0.8, filter_aware: bool = False) -> None:
        super().__init__()
        self._database = database or pb.db.current_database()
        self._qeps = PlainQeps.empty(gamma=gamma)
        self._filter_aware = filter_aware

    def select_physical_operators(
        self, query: pb.SqlQuery, join_order: pb.JoinTree | None
    ) -> pb.PhysicalOperatorAssignment:
        if not pb.qal.is_select_query(query):
            raise pb.qal.QueryTypeError.expected_select(query)

        join_order = join_order or self._fallback_native_join_order(query)
        assignment = pb.PhysicalOperatorAssignment()

        provide_query = query if self._filter_aware else None
        _generate_recommendations(assignment, qeps=self._qeps, join_order=join_order, query=provide_query)
        return assignment

    def learn_from_feedback(
        self, query: SqlQuery, result_set: pb.db.ResultSet | pb.QueryPlan, *, exec_time: pb.TimeMs
    ) -> pb.train.TrainingMetrics:
        if not pb.qal.is_select_query(query):
            raise pb.qal.QueryTypeError.expected_select(query)

        if isinstance(result_set, pb.QueryPlan):
            plan = result_set
        else:
            try:
                plan = self._database.optimizer().parse_plan(result_set, query=query)
            except Exception as e:
                return {"status": "failure", "details": e}

        current_qeps = self._qeps
        provide_query = query if self._filter_aware else None

        train_start = time.perf_counter_ns()
        for inner_node, join_node in _traverse_query_plan(plan):
            current_qeps = current_qeps.follow(inner_node.tables(), query=provide_query)
            if join_node is None:
                continue
            assert isinstance(join_node.operator, pb.JoinOperator)
            current_qeps.feedback(
                inner_node, operator=join_node.operator, cost=join_node.estimated_cost, query=provide_query
            )
        train_end = time.perf_counter_ns()
        training_time_ms = (train_end - train_start) / 1_000_000
        return {"status": "success", "training_time_ms": training_time_ms}

    def _fallback_native_join_order(self, query: pb.qal.SelectStatement) -> pb.JoinTree:
        plan = self._database.optimizer().query_plan(query)
        return pb.jointree_from_plan(plan)
