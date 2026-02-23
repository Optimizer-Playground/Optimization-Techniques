from __future__ import annotations

import functools
import operator
from collections.abc import Iterable
from typing import Optional

import networkx as nx
import postbound as pb

from ._compress import merge_constants
from ._piecewise_fns import PiecewiseConstantFn


class AlphaStep:
    def __init__(
        self, relations: Iterable[PiecewiseConstantFn], *, n_distinct: int
    ) -> None:
        self._fns = list(relations)
        self._combined = functools.reduce(operator.mul, self._fns).cut_at(n_distinct)

    @property
    def n_distinct(self) -> int:
        return self._combined.n_distinct

    @property
    def output(self) -> PiecewiseConstantFn:
        return self._combined

    def cardinality(self) -> int:
        return self._combined.cardinality()

    def __call__(self) -> PiecewiseConstantFn:
        return self._combined


class BetaStep:
    def __init__(
        self,
        star_joins: Iterable[PiecewiseConstantFn],
        *,
        projection: Optional[PiecewiseConstantFn] = None,
        n_distinct: int,
    ) -> None:
        if projection is None:
            candidates = sorted(star_joins, key=lambda join: join.n_distinct)
            projection, star_joins = candidates[0], candidates[1:]

        self._proj = projection
        self._proj_cumulative = self._proj.integ()
        self._star_joins = list(star_joins)
        self._join_cumulative = [star_join.integ() for star_join in self._star_joins]

        cards: list[int] = []
        for i in range(self._proj.n_distinct):
            card = self(i)
            if card == 0:
                break
            cards.append(card)
        self._combined = merge_constants(cards)

    @property
    def n_distinct(self) -> int:
        return self._proj.n_distinct

    @property
    def output(self) -> PiecewiseConstantFn:
        return self._combined

    def cardinality(self) -> int:
        return self._combined.cardinality()

    def __call__(self, i: int) -> int:
        card = self._proj(i)
        for star_join, join_cumulative in zip(self._star_joins, self._join_cumulative):
            bucket = join_cumulative.invert_at(self._proj_cumulative(i))
            card *= star_join(bucket)
        return card


def decompose_query(
    query: pb.SqlQuery, *, statistics: dict[pb.ColumnReference, PiecewiseConstantFn]
) -> AlphaStep | BetaStep:
    join_graph = nx.Graph()
    join_graph.add_nodes_from(query.tables(), node_type="base_table")

    for i, eqc in enumerate(pb.qal.determine_join_equivalence_classes(query.joins())):
        join_graph.add_node(i, node_type="join")
        join_graph.add_edges_from((i, col.table, {"join_col": col}) for col in eqc)

    # TODO:
    # 1. derive spanning tree from join graph
    # 2. map node items to alpha/beta steps


def fdsb(root: AlphaStep | BetaStep) -> pb.Cardinality:
    card = root.cardinality()
    return pb.Cardinality(card)
