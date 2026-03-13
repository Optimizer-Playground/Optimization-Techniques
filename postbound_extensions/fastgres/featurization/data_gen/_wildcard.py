
from __future__ import annotations

from pathlib import Path
from typing import Collection, Sequence

import postbound as pb
from postbound.qal import BinaryPredicate, BetweenPredicate, InPredicate, UnaryPredicate, CompoundPredicate, \
    AbstractPredicate, LogicalOperator, SimpleFilter
from tqdm import tqdm

from ._statistics import StatisticsComponent
from .._dbutil import DatabaseConnection
from ..._util import load_json, save_json, min_max_encode


class WildcardCollector(pb.qal.PredicateVisitor[Collection[pb.qal.SimpleFilter]]):

    def visit_binary_predicate(
            self, predicate: BinaryPredicate, *args, **kwargs
    ) -> Collection[pb.qal.SimpleFilter]:

        if not (SimpleFilter.can_wrap(predicate) and predicate.is_filter()):
            return []

        valid_operators = [LogicalOperator.Like, LogicalOperator.NotLike, LogicalOperator.ILike,
                           LogicalOperator.NotILike]
        wrapped: SimpleFilter = SimpleFilter.wrap(predicate)
        if wrapped.operation not in valid_operators:
            return []

        return [wrapped]

    def visit_between_predicate(
            self, predicate: BetweenPredicate, *args, **kwargs
    ) -> Collection[pb.qal.SimpleFilter]:
        return []

    def visit_in_predicate(
            self, predicate: InPredicate, *args, **kwargs
    ) -> Collection[pb.qal.SimpleFilter]:
        return []

    def visit_unary_predicate(
            self, predicate: UnaryPredicate, *args, **kwargs
    ) -> Collection[pb.qal.SimpleFilter]:
        return []

    def visit_or_predicate(
            self,
            predicate: CompoundPredicate,
            components: Sequence[AbstractPredicate],
            *args,
            **kwargs,
    ) -> Collection[pb.qal.SimpleFilter]:
        return []

    def visit_not_predicate(
            self,
            predicate: CompoundPredicate,
            child_predicate: AbstractPredicate,
            *args,
            **kwargs,
    ) -> Collection[pb.qal.SimpleFilter]:
        return []

    def visit_and_predicate(
            self,
            predicate: CompoundPredicate,
            components: Sequence[AbstractPredicate],
            *args,
            **kwargs,
    ) -> Collection[pb.qal.SimpleFilter]:
        return pb.util.flatten(pred.accept_visitor(self) for pred in components)


class FastgresWildcardComponent(StatisticsComponent):

    def __init__(self):
        self._wildcard_dict = None

    def to_dict(self):
        return self._wildcard_dict

    def build(self, dbc: DatabaseConnection, **kwargs):
        workload: pb.Workload = kwargs.get("workload")
        if not isinstance(workload, pb.Workload):
            raise ValueError(f"Provided workload is not a PostBound Workload: {workload}")
        if len(workload.queries()) == 0:
            raise ValueError("Provided workload is empty}")
        wildcard_dict = dict()
        for parsed in tqdm(workload.queries(), desc="Creating Wildcard Dictionary"):
            # parsed: SqlQuery = pb.parse_query(query_string)
            wc_predicates = parsed.where_clause.predicate.accept_visitor(WildcardCollector())

            for wc_predicate in wc_predicates:
                table_ref = wc_predicate.column.table
                column_ref = wc_predicate.column

                table_name = table_ref.full_name
                column_name = column_ref.name

                like_value = wc_predicate.value

                try:
                    table_cardinality = dbc.cardinality(table_ref)
                except KeyError as e:
                    tqdm.write(f"Table {table_name} caused a cardinality lookup error.")
                    raise e

                predicate = pb.qal.BinaryPredicate(
                    wc_predicate.operation,
                    first_argument=pb.qal.ColumnExpression(column_ref),
                    second_argument=pb.qal.StaticValueExpression(value=like_value)
                )
                like_cardinality = dbc.like_cardinality(column_ref, predicate)

                wildcard_dict.setdefault(table_name, {})['max'] = table_cardinality
                like_cardinality = 0 if not like_cardinality else like_cardinality
                wildcard_dict.setdefault(table_name, {}).setdefault(column_name, {})[like_value] = like_cardinality
        self._wildcard_dict = wildcard_dict

    def save(self, dir_path: Path) -> None:
        if self._wildcard_dict is None:
            raise ValueError("Wildcard component is not built!")
        save_json(self._wildcard_dict, dir_path / "wildcard_dict.json")

    @classmethod
    def load(cls, dir_path: Path) -> FastgresWildcardComponent:
        obj = cls()
        obj._wildcard_dict = load_json(dir_path / "wildcard_dict.json")
        return obj

    def transform(self, table: str, col: str, value: str):
        min_v = 0
        max_v = self._wildcard_dict[table]['max']
        offset = 0.001
        round_values = 4
        return min_max_encode(self._wildcard_dict[table][col][value], min_v, max_v, offset, round_values)
