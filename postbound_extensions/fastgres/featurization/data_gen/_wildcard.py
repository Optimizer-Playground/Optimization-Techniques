
from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import postbound as pb
from postbound.qal import (
    AndPredicate,
    BetweenPredicate,
    BinaryOperator,
    BinaryPredicate,
    InPredicate,
    NotPredicate,
    OrPredicate,
    SimpleFilter,
    UnaryPredicate,
)
from tqdm import tqdm

from ..._util import load_json, min_max_encode, save_json
from .._dbutil import DatabaseConnection
from ._statistics import StatisticsComponent


class WildcardCollector(pb.qal.PredicateVisitor[Collection[pb.qal.SimpleFilter]]):

    def visit_binary_predicate(
            self, predicate: BinaryPredicate, *args, **kwargs
    ) -> Collection[pb.qal.SimpleFilter]:

        if not (SimpleFilter.can_wrap(predicate) and predicate.is_filter()):
            return []

        valid_operators = [BinaryOperator.Like, BinaryOperator.NotLike, BinaryOperator.ILike,
                           BinaryOperator.NotILike]
        wrapped: SimpleFilter = SimpleFilter.wrap(predicate)
        if wrapped.operator not in valid_operators:
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
            predicate: OrPredicate,
            *args,
            **kwargs,
    ) -> Collection[pb.qal.SimpleFilter]:
        return []

    def visit_not_predicate(
            self,
            predicate: NotPredicate,
            *args,
            **kwargs,
    ) -> Collection[pb.qal.SimpleFilter]:
        return []

    def visit_and_predicate(
            self,
            predicate: AndPredicate,
            *args,
            **kwargs,
    ) -> Collection[pb.qal.SimpleFilter]:
        return [item for child in predicate.children for item in child.accept_visitor(self)]


class FastgresWildcardComponent(StatisticsComponent):

    def __init__(self):
        self._wildcard_dict: dict[str, Any] | None = None

    def to_dict(self):
        return self._wildcard_dict

    def build(self, dbc: DatabaseConnection, **kwargs):
        workload = kwargs.get("workload")
        if not isinstance(workload, pb.Workload):
            raise ValueError(f"Provided workload is not a PostBound Workload: {workload}")
        if len(workload.queries()) == 0:
            raise ValueError("Provided workload is empty}")
        wildcard_dict = dict()
        for parsed in tqdm(workload.queries(), desc="Creating Wildcard Dictionary"):
            # parsed: SqlQuery = pb.parse_query(query_string)
            if not isinstance(parsed, pb.qal.SelectStatement):
                raise ValueError("FastGRES only supports SELECT statements")
            if parsed.where_clause is None:
                continue
            wc_predicates = parsed.where_clause.root.accept_visitor(WildcardCollector())

            for wc_predicate in wc_predicates:
                table_ref = wc_predicate.column.table
                column_ref = wc_predicate.column
                if table_ref is None:
                    raise ValueError(f"Cannot collect wildcard statistics for an unbound column: {column_ref}")

                table_name = table_ref.full_name
                column_name = column_ref.name

                like_value = wc_predicate.value

                try:
                    table_cardinality = dbc.cardinality(table_ref)
                except KeyError as e:
                    tqdm.write(f"Table {table_name} caused a cardinality lookup error.")
                    raise e

                predicate = wc_predicate.unwrap()
                if not isinstance(predicate, BinaryPredicate):
                    raise ValueError(f"Expected a binary LIKE predicate, got {predicate}")
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
        if self._wildcard_dict is None:
            raise ValueError("Wildcard statistics must be built or loaded first")
        min_v = 0
        max_v = self._wildcard_dict[table]['max']
        offset = 0.001
        round_values = 4
        return min_max_encode(self._wildcard_dict[table][col][value], min_v, max_v, offset, round_values)
