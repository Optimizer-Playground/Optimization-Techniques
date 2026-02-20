from __future__ import annotations

import functools
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import networkx as nx
import postbound as pb

from ._piecewise_fns import PiecewiseConstantFn, PiecewiseLinearFn


class Step(Protocol):
    def __call__(self) -> PiecewiseConstantFn:
        pass


class LeafStep:
    def __init__(self, fn: PiecewiseConstantFn) -> None:
        self._fn = fn

    def __call__(self) -> PiecewiseConstantFn:
        return self._fn


class AlphaStep:
    def __init__(self, functions: Iterable[Step]) -> None:
        self._fns = list(functions)
        self._out: PiecewiseConstantFn | None = None

    def cardinality(self) -> int:
        final_fn = self()
        return final_fn.cardinality()

    def __call__(self) -> PiecewiseConstantFn:
        if self._out is None:
            self._out = functools.reduce(operator.mul, [fn() for fn in self._fns])
        return self._out


@dataclass
class _Join:
    fn: PiecewiseConstantFn
    cdf: PiecewiseLinearFn
    inv: PiecewiseLinearFn

    @staticmethod
    def of(fn: Step):
        cdf = fn().integrate()
        return _Join(fn(), cdf, cdf.invert())


class BetaStep:
    def __init__(
        self,
        joins: Iterable[Step],
    ) -> None:
        joins: list[_Join] = [_Join.of(fn) for fn in joins]
        sorted(joins, key=lambda j: j.fn.n_distinct)

        self._target = self._joins[0]
        self._joins = self._joins[1:]
        self._n_distinct = self._target.fn.n_distinct

    def cardinality(self) -> int:
        final_fn = self()
        return final_fn.cardinality()

    def __call__(self) -> PiecewiseConstantFn:
        distincts: list[int] = []

        for i in range(self._n_distinct):
            freq = self._target.fn(i)

            for star_fn in self._joins:
                star, inv = star_fn.fn, star_fn.inv
                freq *= star(inv(self._target.cdf(i)))

            distincts.append(freq)

        return PiecewiseConstantFn.unit_segments(distincts)


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
