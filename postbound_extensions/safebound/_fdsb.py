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
        self._fns: tuple[FunctionLike, ...] = tuple(relations)
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

    def __hash__(self) -> int:
        if self._combined is not None:
            return hash(self._combined)
        return hash(self._fns)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._fns == other._fns


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

        self._proj: FunctionLike = projection
        self._star_joins: tuple[FunctionLike, ...] = tuple(star_joins)

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

    def __hash__(self) -> int:
        return hash((self._proj, self._star_joins))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._proj == other._proj
            and self._star_joins == other._star_joins
        )


def _generate_alpha(
    join_graph: nx.Graph,
    join: int,
    *,
    statistics: dict[pb.ColumnReference, PiecewiseConstantFn],
) -> tuple[AlphaStep | None, set]:
    free_nodes = []
    total_nodes = 0
    for node, edge in join_graph.adj[join].items():
        total_nodes += 1
        if len(join_graph.adj[node]) == 1:
            free_nodes.append((node, edge.get("join_col", node)))

    if len(free_nodes) < total_nodes - 1:
        return None, set()

    relations = [
        node if join_col is None else statistics[join_col]
        for node, join_col in free_nodes
    ]
    step = AlphaStep(relations)
    return step, {node[0] for node in free_nodes}


def _generate_beta(
    join_graph: nx.Graph,
    node,
    *,
    statistics: dict[pb.ColumnReference, PiecewiseConstantFn],
) -> tuple[BetaStep | None, set]:
    # XXX: make sure to only include each node in one step!
    pass


def decompose_query(
    query: pb.SqlQuery, *, statistics: dict[pb.ColumnReference, PiecewiseConstantFn]
) -> AlphaStep | BetaStep:
    join_graph = nx.Graph()
    join_graph.add_nodes_from(query.tables(), node_type="base_table")

    for i, eqc in enumerate(pb.qal.determine_join_equivalence_classes(query.joins())):
        join_graph.add_node(i, node_type="join")
        join_graph.add_edges_from((i, col.table, {"join_col": col}) for col in eqc)

    while len(join_graph) > 1:
        alpha_steps: dict[int, AlphaStep] = {}
        nodes_to_pop: set[pb.TableReference] = set()

        for join, data in join_graph.nodes(data=True):
            if data["node_type"] != "join":
                continue

            step, removed = _generate_alpha(join_graph, join, statistics=statistics)
            if step is None:
                # not suitable for an alpha step
                continue

            alpha_steps[join] = step
            nodes_to_pop |= removed

        # TODO:
        # - insert alpha steps
        # - remove leaf nodes + joins
        # - re-connect alpha steps

        beta_steps: list[BetaStep] = []
        nodes_to_pop.clear()
        for table, data in join_graph.nodes(data=True):
            if data["node_type"] not in ("base_table", "step"):
                continue

            step, removed = _generate_beta(join_graph, table, statistics=statistics)
            if step is None:
                # not suitable for beta step
                continue

            beta_steps.append(step)
            nodes_to_pop |= removed

        # TODO:
        # - insert beta steps
        # - remove leaf nodes + joins
        # - re-connect beta steps

    root = next(iter(join_graph.nodes()))
    return root


def fdsb(root: AlphaStep | BetaStep) -> pb.Cardinality:
    card = root.cardinality()
    return pb.Cardinality(card)
