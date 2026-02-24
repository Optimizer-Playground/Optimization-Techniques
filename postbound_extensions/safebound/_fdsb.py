from __future__ import annotations

import functools
import operator
from collections.abc import Iterable
from typing import Optional

import networkx as nx
import numpy as np
import postbound as pb

from ._piecewise_fns import FunctionLike, PiecewiseConstantFn


class AlphaStep:
    def __init__(self, relations: Iterable[FunctionLike]) -> None:
        self._fns: list[FunctionLike] = list(relations)
        self._combined: PiecewiseConstantFn | None = None
        if all(isinstance(rel, PiecewiseConstantFn) for rel in self._fns):
            self._combined = functools.reduce(operator.mul, self._fns)  # type: ignore
            self._n_distinct = self._combined.n_distinct  # type: ignore
        else:
            self._combined = None
            self._n_distinct = min(fn.n_distinct for fn in self._fns)

    @property
    def n_distinct(self) -> int:
        return self._n_distinct

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray:
        if self._combined is not None:
            return self._combined.evaluate_at(vals)

        res = np.ones_like(vals)
        for fn in self._fns:
            res *= fn.evaluate_at(vals)
        return res

    def cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        if self._combined is not None:
            return self._combined.cumulative_at(vals)

        max_val = np.max(vals)
        extended_vals = np.arange(max_val + 1)
        evals = self.evaluate_at(extended_vals)
        cumulative = np.cumsum(evals)
        return cumulative[vals]

    def invert_cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        if self._combined is not None:
            return self._combined.invert_cumulative_at(vals)

        # TODO: can we be smarter here than to just materialize everything?
        # Maybe we could do some binary search-style approximation of where
        # we need to start materializing.
        # E.g., start with n_distinct / 2 and determine the cumulative value
        # Then, bisect and evaluate 3/4 * n_distinct and 1/4 * n_distinct, etc.
        cumulative = self.cumulative_at(np.arange(self._n_distinct))
        idx = np.searchsorted(cumulative, vals)
        return np.where(idx >= self._n_distinct, 0, idx)

    def cardinality(self) -> int:
        if self._combined is not None:
            return self._combined.cardinality()

        vals = np.arange(self.n_distinct)
        total_freqs = self(vals)
        return np.sum(total_freqs)

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        return self.evaluate_at(vals)


class BetaStep:
    def __init__(
        self,
        star_joins: Iterable[FunctionLike],
        *,
        projection: Optional[FunctionLike] = None,
    ) -> None:
        if projection is None:
            star_joins = list(star_joins)
            sorted(star_joins, key=lambda j: j.n_distinct)
            projection, star_joins = star_joins[0], star_joins[1:]

        self._proj = projection
        self._star_joins = list(star_joins)

    @property
    def n_distinct(self) -> int:
        return self._proj.n_distinct

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray:
        res = self._proj.evaluate_at(vals)
        proj_freqs = self._proj.cumulative_at(vals)
        for star_join in self._star_joins:
            idx = star_join.invert_cumulative_at(proj_freqs)
            res *= star_join.evaluate_at(idx)
        return res

    def cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        max_val = np.max(vals)
        extended_vals = np.arange(max_val + 1)
        evals = self.evaluate_at(extended_vals)
        cumulative = np.cumsum(evals)
        return cumulative[vals]

    def invert_cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        cumulative = self.cumulative_at(np.arange(self.n_distinct))
        idx = np.searchsorted(cumulative, vals)
        return np.where(idx >= self.n_distinct, 0, idx)

    def cardinality(self) -> int:
        cards = self.evaluate_at(np.arange(self.n_distinct))
        return np.sum(cards)

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        return self.evaluate_at(vals)


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

    root: pb.TableReference = None  # TODO
    return _create_spanning_tree(root, graph=join_graph, anchor=None, stats=statistics)


def fdsb(root: AlphaStep | BetaStep) -> pb.Cardinality:
    card = root.cardinality()
    return pb.Cardinality(card)
