from __future__ import annotations

import functools
import operator
from collections.abc import Iterable
from typing import Optional

import networkx as nx
import numpy as np
import postbound as pb

from ._piecewise_fns import FunctionLike, PiecewiseConstantFn, PiecewiseLinearFn


class AlphaStep:
    def __init__(self, relations: Iterable[FunctionLike]) -> None:
        self._fns: list[FunctionLike] = list(relations)
        self._combined: PiecewiseConstantFn | None = None
        if all(isinstance(rel, PiecewiseConstantFn) for rel in self._fns):
            self._combined = functools.reduce(operator.mul, self._fns)  # type: ignore
        else:
            self._combined = None

    @property
    def n_distinct(self) -> int:
        if self._combined is not None:
            return self._combined.n_distinct
        return min(fn.n_distinct for fn in self._fns)

    def cardinality(self) -> int:
        if self._combined is not None:
            return self._combined.cardinality()

        vals = np.arange(self.n_distinct)
        total_freqs = self(vals)
        return np.sum(total_freqs)

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        if self._combined is not None:
            return self._combined(vals)

        res = np.ones_like(vals)
        for fn in self._fns:
            res *= fn(vals)
        return res


class BetaStep:
    def __init__(
        self,
        star_joins: Iterable[FunctionLike],
        *,
        projection: Optional[PiecewiseConstantFn] = None,
    ) -> None:
        # TODO: implementation

        self._proj: PiecewiseConstantFn
        self._star_joins: list[PiecewiseConstantFn]
        self._star_join_lookups: list[PiecewiseLinearFn]

    @property
    def n_distinct(self) -> int:
        return self._proj.n_distinct

    def cardinality(self) -> int:
        args = np.arange(self._proj.n_distinct)
        return np.sum(self(args))

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        res = self._proj(vals)
        for i, star_join in enumerate(self._star_joins):
            lookup_fn = self._star_join_lookups[vals]
            idx = lookup_fn(vals)
            res *= star_join(idx)
        return res


def _create_alpha_step(
    join_node: int,
    *,
    graph: nx.Graph,
    anchor: Optional[pb.TableReference],
    stats: dict[pb.ColumnReference, PiecewiseConstantFn],
) -> AlphaStep:
    functions: list[FunctionLike] = []
    for neighbor, data in graph.adj[join_node].items():
        if neighbor == anchor:
            continue
        functions.append(
            _create_spanning_tree(neighbor, graph=graph, anchor=join_node, stats=stats)
        )

    if anchor is None:
        return AlphaStep(functions)

    edge = graph.edges[join_node, anchor]
    join_col = edge["join_col"]
    functions.append(stats[join_col])
    return AlphaStep(functions)


def _create_beta_step(
    join_node: int,
    *,
    graph: nx.Graph,
    anchor: Optional[pb.TableReference],
    stats: dict[pb.ColumnReference, PiecewiseConstantFn],
) -> BetaStep:
    star_joins: list[FunctionLike] = []
    for neighbor, data in graph.adj[join_node].items():
        if neighbor == anchor:
            continue
        star_joins.append(
            _create_spanning_tree(neighbor, graph=graph, anchor=join_node, stats=stats)
        )

    if anchor is None:
        return BetaStep(star_joins)

    edge = graph.edges[join_node, anchor]
    join_col = edge["join_col"]
    return BetaStep(star_joins, projection=stats[join_col])


def _create_spanning_tree(
    node: pb.TableReference,
    *,
    graph: nx.Graph,
    anchor: Optional[int],
    stats: dict[pb.ColumnReference, PiecewiseConstantFn],
) -> AlphaStep | BetaStep:
    joins = graph.adj[node]

    # TODO


def decompose_query(
    query: pb.SqlQuery, *, statistics: dict[pb.ColumnReference, PiecewiseConstantFn]
) -> AlphaStep | BetaStep:
    join_graph = nx.Graph()
    join_graph.add_nodes_from(query.tables(), node_type="base_table")

    for i, eqc in enumerate(pb.qal.determine_join_equivalence_classes(query.joins())):
        join_graph.add_node(i, node_type="join")
        join_graph.add_edges_from((i, col.table, {"join_col": col}) for col in eqc)

    root = None  # TODO
    return _create_spanning_tree(root, graph=join_graph, anchor=None, stats=statistics)


def fdsb(root: AlphaStep | BetaStep) -> pb.Cardinality:
    card = root.cardinality()
    return pb.Cardinality(card)
