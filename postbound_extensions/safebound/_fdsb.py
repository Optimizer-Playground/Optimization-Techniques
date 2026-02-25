from __future__ import annotations

import functools
import operator
from collections.abc import Iterable
from typing import Literal, Optional

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

    def columns(self) -> set[pb.ColumnReference]:
        return pb.util.set_union(fn.columns() for fn in self._fns)

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

    def inspect(self) -> str:
        lines = self._inspect_internal()
        return "\n".join(lines)

    def _inspect_internal(self) -> list[str]:
        padding = "  "
        col_desc = ", ".join(str(col) for col in self.columns())
        lines: list[str] = [f"ɑ step ({col_desc})"]

        for fn in self._fns:
            if isinstance(fn, PiecewiseConstantFn):
                desc = f"PCF({fn.column})" if fn.column else "anonymous PCF"
                lines.append(f"{padding}+- join with {desc}")
                continue

            if not isinstance(fn, (AlphaStep, BetaStep)):
                raise ValueError(f"Unknown function type {type(fn).__name__}: {fn}")

            nested_inspect = fn._inspect_internal()
            nested_inspect[0] = f"{padding} +- {nested_inspect[0]}"
            for i in range(1, len(nested_inspect)):
                nested_inspect[i] = f"{padding}    {nested_inspect[i]}"
            lines.extend(nested_inspect)

        return lines

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
        projection: Optional[PiecewiseConstantFn] = None,
    ) -> None:
        if projection is None:
            min_distinct = np.inf
            smallest_proj: PiecewiseConstantFn | None = None
            retained_joins: list[FunctionLike] = []
            for candidate in star_joins:
                if not isinstance(candidate, PiecewiseConstantFn):
                    retained_joins.append(candidate)
                    continue

                if candidate.n_distinct < min_distinct:
                    min_distinct = candidate.n_distinct
                    smallest_proj = candidate
                else:
                    retained_joins.append(candidate)

            if smallest_proj is None:
                raise ValueError(
                    "At least one PiecewiseConstantFn required for a beta step!"
                )
            projection = smallest_proj
            star_joins = retained_joins

        self._proj: PiecewiseConstantFn = projection
        self._star_joins: tuple[FunctionLike, ...] = tuple(star_joins)
        if len(self._star_joins) < 1:
            raise ValueError("At least one dimension join required for a beta step!")

    @property
    def n_distinct(self) -> int:
        return self._proj.n_distinct

    def columns(self) -> set[pb.ColumnReference]:
        return self._proj.columns() | pb.util.set_union(
            star_join.columns() for star_join in self._star_joins
        )

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

    def inspect(self) -> str:
        lines = self._inspect_internal()
        return "\n".join(lines)

    def _inspect_internal(self) -> list[str]:
        padding = "  "
        col_desc = ", ".join(str(col) for col in self.columns())
        lines: list[str] = [f"β step ({col_desc})"]

        if self._proj.column:
            lines.append(f"{padding}[project on {self._proj.column}]")
        else:
            lines.append(f"{padding}[anonymous projection]")

        for star_join in self._star_joins:
            if isinstance(star_join, PiecewiseConstantFn):
                desc = (
                    f"PCF({star_join.column})" if star_join.column else "anonymous PCF"
                )
                lines.append(f"{padding}+- join with {desc}")
                continue

            if not isinstance(star_join, (AlphaStep, BetaStep)):
                raise ValueError(
                    f"Unknown function type {type(star_join).__name__}: {star_join}"
                )

            nested_inspect = star_join._inspect_internal()
            nested_inspect[0] = f"{padding} +- {nested_inspect[0]}"
            for i in range(1, len(nested_inspect)):
                nested_inspect[i] = f"{padding}    {nested_inspect[i]}"
            lines.extend(nested_inspect)

        return lines

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
    node: FunctionLike | pb.TableReference,
    *,
    statistics: dict[pb.ColumnReference, PiecewiseConstantFn],
) -> tuple[BetaStep | None, set]:
    projection: FunctionLike | None = None
    star_joins: list[tuple[pb.TableReference | FunctionLike, FunctionLike]] = []
    invalid_beta = False

    for neighbor in join_graph.adj[node]:
        if join_graph.degree[neighbor] > 1:
            # Only the projection node is allowed to have degree
            # greater 1 (because it connects to the rest of our graph)
            # If we found a node with higher degree, it has to be our
            # projection.
            # But since each beta step only has exactly one projection,
            # we can break as soon as we encouter another such node.
            # In this case, the current node is not ready yet to be
            # included in a beta step and needs to be re-visited in
            # a later iteration. When that happens, the graph might
            # have shrunk enough s.t. we can include the node.
            if projection is not None:
                invalid_beta = True
                break
            base_tab, edge = next(
                (node, edge)
                for node, edge in join_graph.adj[neighbor]
                if node != neighbor
            )
            join_col = edge["join_col"]
            projection = statistics[join_col]
            continue

        node_info = join_graph.nodes[neighbor]
        match node_info["node_type"]:
            case "join":
                # traditional star join table
                base_tab, edge = next(
                    (node, edge)
                    for node, edge in join_graph.adj[neighbor]
                    if node != neighbor
                )
                join_col = edge["join_col"]
                star_joins.append((base_tab, statistics[join_col]))
            case "alpha_step" | "beta_step":
                star_joins.append((neighbor, neighbor))
            case _:
                assert False, f"Unexpected node type {node_info}"

    if invalid_beta or not star_joins:
        return None, set()

    step = BetaStep([star_join[1] for star_join in star_joins], projection=projection)
    return step, {star_join[0] for star_join in star_joins}


def _prune_danling_joins(join_graph: nx.Graph) -> None:
    dangling = []
    for node, data in join_graph.nodes():
        if data["node_type"] != "join":
            continue
        degree = join_graph.degree[node]
        if degree <= 1:
            dangling.append(node)
    join_graph.remove_nodes_from(dangling)


def _restitch_graph(
    join_graph: nx.Graph,
    steps: dict[AlphaStep | BetaStep, set] | tuple[AlphaStep | BetaStep, set],
    *,
    step_kind: Literal["alpha", "beta"],
) -> None:
    if isinstance(steps, tuple):
        steps = {steps[0]: steps[1]}

    node_type = "alpha_step" if step_kind.startswith("alpha") else "beta_step"
    nodes_to_drop = pb.util.set_union(steps.values())
    for step, joined_nodes in steps.items():
        assert len(joined_nodes) >= 2
        join_graph.add_node(step, node_type=node_type)

        for node in joined_nodes:
            neighbors = set(join_graph.adj[node])
            neighbors -= nodes_to_drop
            join_graph.add_edges_from((step, neighbor) for neighbor in neighbors)

        join_graph.remove_nodes_from(joined_nodes)

    _prune_danling_joins(join_graph)


def decompose_query(
    query: pb.SqlQuery, *, statistics: dict[pb.ColumnReference, PiecewiseConstantFn]
) -> AlphaStep | BetaStep:
    join_graph = nx.Graph()
    join_graph.add_nodes_from(query.tables(), node_type="base_table")

    for i, eqc in enumerate(pb.qal.determine_join_equivalence_classes(query.joins())):
        join_graph.add_node(i, node_type="join")
        join_graph.add_edges_from((i, col.table, {"join_col": col}) for col in eqc)

    while len(join_graph) > 1:
        # BetaStep is just to keep the type checker quiet
        alpha_steps: dict[AlphaStep | BetaStep, set] = {}

        for join, data in join_graph.nodes(data=True):
            if data["node_type"] != "join":
                continue

            step, removed = _generate_alpha(join_graph, join, statistics=statistics)
            if step is None:
                # not suitable for an alpha step
                continue

            alpha_steps[step] = removed | {join}

        _restitch_graph(join_graph, alpha_steps, step_kind="alpha")

        beta_candidates = set(
            node
            for node, data in join_graph.nodes(data=True)
            if data["node_type"] != join
        )
        beta_found = False
        while beta_candidates:
            table = beta_candidates.pop()
            step, removed = _generate_beta(join_graph, table, statistics=statistics)
            if step is None:
                continue

            _restitch_graph(join_graph, (step, removed), step_kind="beta")
            beta_candidates.difference_update(removed)
            beta_found = True

        if not alpha_steps and not beta_found:
            raise RuntimeError(
                "No more alpha/beta steps found but join graph still has unprocessed tables. "
                "This is a programming error. Please consider filing an issue on Github."
            )

    root = next(iter(join_graph.nodes()))
    return root


def fdsb(root: AlphaStep | BetaStep) -> pb.Cardinality:
    card = root.cardinality()
    return pb.Cardinality(card)
