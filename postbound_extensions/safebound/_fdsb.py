from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import overload

import networkx as nx
import numpy as np
import postbound as pb

from ._piecewise_fns import FunctionLike, PiecewiseConstantFn


class AlphaStep:
    def __init__(self, relations: Iterable[FunctionLike]) -> None:
        self._fns: tuple[FunctionLike, ...] = tuple(relations)
        if len(self._fns) < 2:
            raise ValueError("alpha-step requires at least two PCFs to join")
        self._combined: PiecewiseConstantFn | None = None
        if all(isinstance(rel, PiecewiseConstantFn) for rel in self._fns):
            self._combined = self._fns[0]  # type: ignore
            for pcf in self._fns[1:]:
                self._combined *= pcf
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
        lines: list[str] = ["ɑ step"]

        for fn in self._fns:
            if isinstance(fn, PiecewiseConstantFn):
                desc = f"PCF({fn.column})" if fn.column else "anonymous PCF"
                lines.append(f"{padding}+- {desc}")
                continue

            if not isinstance(fn, (AlphaStep, BetaStep)):
                raise ValueError(f"Unknown function type {type(fn).__name__}: {fn}")

            nested_inspect = fn._inspect_internal()
            nested_inspect[0] = f"{padding} +- {nested_inspect[0]}"
            for i in range(1, len(nested_inspect)):
                nested_inspect[i] = f"{padding}    {nested_inspect[i]}"
            lines.extend(nested_inspect)

        return lines

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("AlphaStep(...)")
            return

        lines = self._inspect_internal()
        head, tail = lines[0], lines[1:]
        p.text(head)
        for line in tail:
            p.breakable()
            p.text(line)

    def __json__(self) -> pb.util.jsondict:
        return {
            "step_type": "alpha",
            "relations": self._fns,
        }

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        return self.evaluate_at(vals)

    def __hash__(self) -> int:
        if self._combined is not None:
            return hash(self._combined)
        return hash(self._fns)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._fns == other._fns

    def __repr__(self) -> str:
        components = ", ".join(repr(fn) for fn in self._fns)
        return f"AlphaStep(relations=[{components}])"

    def __str__(self) -> str:
        components = ", ".join(str(fn) for fn in self._fns)
        return f"AlphaStep({components})"


def _split_star_joins(
    star_joins: Iterable[FunctionLike],
) -> tuple[PiecewiseConstantFn, Sequence[FunctionLike]]:
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
            f"At least one PiecewiseConstantFn required for a beta step: {star_joins}"
        )
    return smallest_proj, retained_joins


@dataclass
class DimensionJoin:
    fact_pcf: PiecewiseConstantFn
    dimension_pcf: FunctionLike

    def columns(self) -> set[pb.ColumnReference]:
        return self.fact_pcf.columns() | self.dimension_pcf.columns()


class BetaStep:
    def __init__(
        self,
        pcfs: Iterable[DimensionJoin],
        *,
        projection: PiecewiseConstantFn,
    ) -> None:
        self._proj: PiecewiseConstantFn = projection
        self._dimension_joins = tuple(pcfs)
        if len(self._dimension_joins) < 1:
            raise ValueError("At least one dimension join required for a beta step!")

    @property
    def n_distinct(self) -> int:
        return self._proj.n_distinct

    def columns(self) -> set[pb.ColumnReference]:
        return self._proj.columns() | pb.util.set_union(
            dim_join.columns() for dim_join in self._dimension_joins
        )

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray:
        res = self._proj.evaluate_at(vals)
        cumulative = self._proj.cumulative_at(vals)

        for join in self._dimension_joins:
            fact_pcf, dim_pcf = join.fact_pcf, join.dimension_pcf
            dimension_idx = fact_pcf.invert_cumulative_at(cumulative)
            res *= dim_pcf.evaluate_at(dimension_idx)

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
        lines: list[str] = []

        if self._proj.column:
            lines.append(f"β step on {self._proj.column}")
        else:
            lines.append("β step [anonymous projection]")

        lines.append("")

        for dim_join in self._dimension_joins:
            fact_pcf = dim_join.fact_pcf
            if not isinstance(fact_pcf, PiecewiseConstantFn):
                raise ValueError(
                    "Expected PiecewiseConstantFn for fact side of dimension join, "
                    f"got {type(fact_pcf)}"
                )

            lines.append(f"{padding}+- fact col {fact_pcf.column}::")
            dim_pcf = dim_join.dimension_pcf
            match dim_pcf:
                case PiecewiseConstantFn():
                    lines.append(f"{padding}{padding}{dim_pcf}")
                case AlphaStep() | BetaStep():
                    nested = dim_pcf._inspect_internal()
                    nested = [f"{padding}{padding}{line}" for line in nested]
                    lines.extend(nested)
                case _:
                    raise ValueError(
                        "Expected PiecewiseConstantFn or step function for dimension side of "
                        f"dimension join, got {type(dim_pcf)}"
                    )

            lines.append("")

        return lines

    def _repr_pretty(self, p, cycle):
        if cycle:
            p.text("BetaStep(...)")
            return

        lines = self._inspect_internal()
        head, tail = lines[0], lines[1:]
        p.text(head)
        for line in tail:
            p.breakable()
            p.text(line)

    def __json__(self) -> pb.util.jsondict:
        return {
            "step_type": "beta",
            "projection": self._proj,
            "star_joins": self._star_joins,
        }

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        return self.evaluate_at(vals)

    def __hash__(self) -> int:
        return hash((self._proj, self._dimension_joins))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._proj == other._proj
            and self._dimension_joins == other._dimension_joins
        )

    def __repr__(self) -> str:
        components = ", ".join(repr(fn) for fn in self._dimension_joins)
        return f"BetaStep(relations=[{components}], projection={repr(self._proj)})"

    def __str__(self) -> str:
        components = ", ".join(str(fn) for fn in self._dimension_joins)
        return f"BetaStep({components}) -> {self._proj}"


@overload
def _adj_flow(
    join_graph: nx.Graph, node: int, *, source: pb.TableReference
) -> Generator[pb.TableReference, None, None]: ...


@overload
def _adj_flow(
    join_graph: nx.Graph, node: pb.TableReference, *, source: int
) -> Generator[int, None, None]: ...


def _adj_flow(
    join_graph: nx.Graph,
    node: int | pb.TableReference,
    *,
    source: int | pb.TableReference,
) -> Generator[int | pb.TableReference, None, None]:
    for neighbor in join_graph.adj[node]:
        if neighbor == source:
            continue
        yield neighbor


def _create_alpha_step(
    join_graph: nx.Graph,
    join: int,
    *,
    source: pb.TableReference,
    include_source: bool,
    statistics: Mapping[pb.ColumnReference, PiecewiseConstantFn],
) -> AlphaStep:
    relations: list[FunctionLike] = []
    for neighbor in _adj_flow(join_graph, join, source=source):
        if join_graph.degree[neighbor] == 1:
            edge = join_graph.edges[join, neighbor]
            join_col = edge["join_col"]
            pcf = statistics[join_col]
            relations.append(pcf)
            continue

        beta_step = _create_beta_step(
            join_graph, neighbor, project_on=join, statistics=statistics
        )
        relations.append(beta_step)

    if include_source:
        source_edge = join_graph.edges[join, source]
        join_col = source_edge["join_col"]
        source_pcf = statistics[join_col]
        relations.append(source_pcf)

    return AlphaStep(relations)


def _dispatch_on(
    join_graph: nx.Graph,
    join: int,
    *,
    source: pb.TableReference,
    statistics: Mapping[pb.ColumnReference, PiecewiseConstantFn],
) -> FunctionLike:
    deg = join_graph.degree[join]

    if deg > 2:
        return _create_alpha_step(
            join_graph, join, source=source, include_source=False, statistics=statistics
        )

    assert deg == 2
    target = next(_adj_flow(join_graph, join, source=source))
    edge = join_graph.edges[join, target]
    join_col = edge["join_col"]
    return statistics[join_col]


def _create_beta_step(
    join_graph: nx.Graph,
    node: pb.TableReference,
    *,
    project_on: int,
    statistics: Mapping[pb.ColumnReference, PiecewiseConstantFn],
) -> BetaStep:
    dimension_joins: list[DimensionJoin] = []
    for join in _adj_flow(join_graph, node, source=project_on):
        fact_edge = join_graph.edges[node, join]
        join_col = fact_edge["join_col"]
        fact_pcf = statistics[join_col]
        dimension_pcf = _dispatch_on(
            join_graph, join, source=node, statistics=statistics
        )
        dimension_join = DimensionJoin(fact_pcf, dimension_pcf)
        dimension_joins.append(dimension_join)

    proj_edge = join_graph.edges[node, project_on]
    join_col = proj_edge["join_col"]
    proj_pcf = statistics[join_col]
    return BetaStep(dimension_joins, projection=proj_pcf)


def _select_acyclic_root(join_graph) -> pb.TableReference:
    return next(node for node in join_graph.nodes if join_graph.degree[node] == 1)


def decompose_acyclic(
    join_graph: nx.Graph,
    root: pb.TableReference,
    *,
    statistics: Mapping[pb.ColumnReference, PiecewiseConstantFn],
) -> AlphaStep:
    join: int = next(iter(join_graph.adj[root]))
    return _create_alpha_step(
        join_graph, join, source=root, include_source=True, statistics=statistics
    )


def _fdsb_graph(query: pb.SqlQuery) -> nx.Graph:
    join_graph = nx.Graph()
    join_graph.add_nodes_from(
        (tab.drop_alias() for tab in query.tables()), node_type="base_table"
    )

    for i, eqc in enumerate(pb.qal.determine_join_equivalence_classes(query.joins())):
        join_graph.add_node(i, node_type="join")
        join_graph.add_edges_from(
            (i, col.table.drop_alias(), {"join_col": col.drop_table_alias()})
            for col in eqc
        )
    return join_graph


def decompose_query(
    query: pb.SqlQuery,
    *,
    statistics: Mapping[pb.ColumnReference, PiecewiseConstantFn],
    verbose: bool | pb.util.Logger = False,
) -> AlphaStep | BetaStep:
    join_graph = _fdsb_graph(query)
    if not nx.is_tree(join_graph):
        raise ValueError(f"Decomposition only works for acyclic queries, not '{query}'")
    root = _select_acyclic_root(join_graph)
    return decompose_acyclic(join_graph, root, statistics=statistics)


def fdsb(
    query: pb.SqlQuery, *, statistics: Mapping[pb.ColumnReference, PiecewiseConstantFn]
) -> pb.Cardinality:
    join_graph = _fdsb_graph(query)

    if nx.is_tree(join_graph):
        root = _select_acyclic_root(join_graph)
        decomposed = decompose_acyclic(join_graph, root, statistics=statistics)
        cardinality = decomposed.cardinality()
        return pb.Cardinality(cardinality)

    best_bound = np.inf
    for acyclic in nx.SpanningTreeIterator(join_graph):
        root = _select_acyclic_root(acyclic)
        decomposed = decompose_acyclic(acyclic, root, statistics=statistics)
        cardinality = decomposed.cardinality()
        best_bound = min(cardinality, best_bound)

    return pb.Cardinality(best_bound)
