from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass
from typing import Optional, overload

import networkx as nx
import numpy as np
import postbound as pb

from ._piecewise_fns import FunctionLike, PiecewiseConstantFn


class AlphaStep:
    """An alpha-step models the join of multiple relations on a single join column.

    For example, the following joins could be modeled by an alpha step: *R.a = S.b AND S.b = T.c*.
    The internals of an alpha step are pretty straightforward: for each of the PCF indexes, it
    computes the product of all PCF frequencies. See the actual SafeBound paper for the precise
    formulas, etc.

    Each alpha step provides the following high level functionality:

    - `cardinality` calculates the upper bound on the output relation, if this step were the final
      join in the decomposed query plan
    - `evaluate_at` calculates the output frequency of the given PCF indexes
    - `cumulative_at` calculates the cumulative output frequency of the given PCF indexes
    - `invert_cumulative_at` calcualtes the PCF indexes of the given cumulative output frequencies

    Parameters
    ----------
    relations : Iterable[FunctionLike]
        The PCFs and/or nested steps that are joined together in this alpha step

    See Also
    --------
    BetaStep : used to model star joins
    """

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
        """Get the number of distinct values in the output PCF."""
        return self._n_distinct

    def columns(self) -> set[pb.ColumnReference]:
        """Provides all join columns that are part of this alpha step, including nested steps."""
        return pb.util.set_union(fn.columns() for fn in self._fns)

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the output frequencies of the given PCF indexes.

        While this description might seem a bit cryptic, remember that SafeBound assumes that the
        i-th most frequent value of one join column is always joined with the i-th most frequent
        value of the partner join column. Using this function, we compute the output frequencies
        of a batch of PCF indexes (the most frequent values at different positions).

        Since all join columns are by definition equality-joined, this output frequency boils down
        to the multiplication of all frequencies at the same index.
        """
        if self._combined is not None:
            return self._combined.evaluate_at(vals)

        res = np.ones_like(vals)
        for fn in self._fns:
            res *= fn.evaluate_at(vals)
        return res

    def cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the cumulative output frequencies of the given PCF indexes.

        This method is similar to `evaluate_at` with the key difference that we compute the
        cumulative frequencies (i.e. including all higher-frequency values) instead of the
        frequencies at the specific indexes.
        """
        if self._combined is not None:
            return self._combined.cumulative_at(vals)

        max_val = np.max(vals)
        extended_vals = np.arange(max_val + 1)
        evals = self.evaluate_at(extended_vals)
        cumulative = np.cumsum(evals)
        return cumulative[vals]

    def invert_cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the PCF indexes that reach the given cumulative frequencies.

        Basically, this method functions as the inverse to `cumulative_at`. Instead of computing the
        cumulative frequencies at specific indexes, it computes the indexes at specific cumulative
        frequencies.
        """
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
        """Computes the upper bound of the output relation's cardinality."""
        if self._combined is not None:
            return self._combined.cardinality()

        vals = np.arange(self.n_distinct)
        total_freqs = self(vals)
        return np.sum(total_freqs)

    def inspect(self) -> str:
        """Provides a human-readable representation of this alpha step as well as its inputs."""
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


@dataclass
class DimensionJoin:
    """A dimension join models a single join of a complex star join.

    For example, the star join *R.a = S.b AND R.c = T.d* has two dimension joins: *R.a = S.b* and
    *R.c = T.d*. Here, *R.a* and *R.c* would be the fact PCFs and *S.b* and *T.c* would be the
    dimension PCFs.
    """

    fact_pcf: PiecewiseConstantFn
    dimension_pcf: FunctionLike

    def columns(self) -> set[pb.ColumnReference]:
        return self.fact_pcf.columns() | self.dimension_pcf.columns()


class BetaStep:
    """A beta-step models a star join between one fact table and (one or multiple) dimension tables.

    For example, the following joins could be modeled by a beta step: *R.a = S.b AND R.c = T.d*.

    In contrast to the `AlphaStep`, each beta step additionally has a projection on a join column
    that is not involved in any of the dimension joins.
    Basically, the beta step computes the frequency increase on the projected column based on the
    frequency increases of the dimension joins. See the actual SafeBound paper for the precise
    formulas.

    Each beta step provides the following high level functionality:

    - `cardinality` calculates the upper bound on the output relation, if this step were the final
      join in the decomposed query plan
    - `evaluate_at` calculates the output frequency of the given PCF indexes
    - `cumulative_at` calculates the cumulative output frequency of the given PCF indexes
    - `invert_cumulative_at` calcualtes the PCF indexes of the given cumulative output frequencies

    Parameters
    ----------
    pcfs : Iterable[DimensionJoin]
        The individual dimension joins to perform. It is assumed that all of these joins operate on
        distinct columns and all fact table PCFs belong to the same table.
    projection : PiecewiseConstantFn
        The PCF to project the cardinality increases on. It is assumed that this PCF belongs to the
        same table as all of the fact PCFs, but that the join columns are disjunct.

    See Also
    --------
    AlphaStep : used to model joins on the same column
    """

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
        """Get the number of distinct values in the output PCF."""
        return self._proj.n_distinct

    def columns(self) -> set[pb.ColumnReference]:
        """Provides all join columns that are part of this alpha step, including nested steps."""
        return self._proj.columns() | pb.util.set_union(
            dim_join.columns() for dim_join in self._dimension_joins
        )

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the output frequencies of the given PCF indexes.

        While this description might seem a bit cryptic, remember that SafeBound assumes that the
        i-th most frequent value of one join column is always joined with the i-th most frequent
        value of the partner join column. Using this function, we compute the output frequencies
        of a batch of PCF indexes (the most frequent values at different positions).

        The beta step basically computes the frequency increase on the projected PCF based on the
        frequency increases caused by each dimension join.
        """
        res = self._proj.evaluate_at(vals)
        cumulative = self._proj.cumulative_at(vals)

        for join in self._dimension_joins:
            fact_pcf, dim_pcf = join.fact_pcf, join.dimension_pcf
            dimension_idx = fact_pcf.invert_cumulative_at(cumulative)
            res *= dim_pcf.evaluate_at(dimension_idx)

        return res

    def cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the cumulative output frequencies of the given PCF indexes.

        This method is similar to `evaluate_at` with the key difference that we compute the
        cumulative frequencies (i.e. including all higher-frequency values) instead of the
        frequencies at the specific indexes.
        """
        max_val = np.max(vals)
        extended_vals = np.arange(max_val + 1)
        evals = self.evaluate_at(extended_vals)
        cumulative = np.cumsum(evals)
        return cumulative[vals]

    def invert_cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the PCF indexes that reach the given cumulative frequencies.

        Basically, this method functions as the inverse to `cumulative_at`. Instead of computing the
        cumulative frequencies at specific indexes, it computes the indexes at specific cumulative
        frequencies.
        """
        cumulative = self.cumulative_at(np.arange(self.n_distinct))
        idx = np.searchsorted(cumulative, vals)
        return np.where(idx >= self.n_distinct, 0, idx)

    def cardinality(self) -> int:
        """Computes the upper bound of the output relation's cardinality."""
        cards = self.evaluate_at(np.arange(self.n_distinct))
        return np.sum(cards)

    def inspect(self) -> str:
        """Provides a human-readable representation of this beta step as well as its inputs."""
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
            "dimension_joins": self._dimension_joins,
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
    """Provides all neighbors of the given node that differ from the source.

    Graphically speaking, we traverse the `join_graph` from `source` to `node`. Now, we need to
    figure out to which nodes we can move next. This is exactly, what this function does.
    """
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
    """Creates an alpha step that combines all nodes connected to given join node.

    The source node is not included in the alpha step, unless `include_source` explicitly requests
    it.

    All nodes must have a PCF entry in the `statistics`.

    This function automatically recurses in all connected nodes, creating intermediate beta steps
    as necessary.
    """
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


def _create_beta_step(
    join_graph: nx.Graph,
    node: pb.TableReference,
    *,
    project_on: int,
    statistics: Mapping[pb.ColumnReference, PiecewiseConstantFn],
) -> BetaStep:
    """Creates a beta step that performs the star join on the given table.

    The beta steps projection PCF is inferred from the outgoing join edge to the `project_on` join
    node.

    All nodes must have a PCF entry in the `statistics`.

    This function automatically recurses in all connected nodes, creating intermediate alpha steps
    and beta steps as necessary.
    """
    dimension_joins: list[DimensionJoin] = []
    for join in _adj_flow(join_graph, node, source=project_on):
        fact_edge = join_graph.edges[node, join]
        join_col = fact_edge["join_col"]
        fact_pcf = statistics[join_col]

        deg = join_graph.degree[join]
        if deg > 2:
            # The current join node for our beta step is connected to multiple other nodes.
            # In other words, multiple relations join on the same dimension.
            # We need to create an alpha step for them.
            # This recurses in the join partners to create additional steps as necessary.
            dimension_pcf = _create_alpha_step(
                join_graph,
                join,
                source=node,
                include_source=False,
                statistics=statistics,
            )
        else:
            # The current join node must be connected to exactly two tables: our current node
            # that we create the beta step for, and exactly one partner node.
            # Together, they form a basic dimension join. All we need to do, is extract the PCF
            # of our partner.
            assert deg == 2
            join_partner = next(_adj_flow(join_graph, join, source=node))

            if join_graph.degree[join_partner] == 1:
                # Partner only takes part in one join - our current join.
                # Extract its PCF and be done with it.
                partner_edge = join_graph.edges[join, join_partner]
                partner_col = partner_edge["join_col"]
                dimension_pcf = statistics[partner_col]
            else:
                # Partner takes part in additional joins. This is a beta-step situation once again.
                dimension_pcf = _create_beta_step(
                    join_graph, join_partner, project_on=join, statistics=statistics
                )

        dimension_join = DimensionJoin(fact_pcf, dimension_pcf)
        dimension_joins.append(dimension_join)

    proj_edge = join_graph.edges[node, project_on]
    join_col = proj_edge["join_col"]
    proj_pcf = statistics[join_col]
    return BetaStep(dimension_joins, projection=proj_pcf)


def _select_acyclic_root(join_graph: nx.Graph) -> pb.TableReference:
    """Provides an arbitrary table from the join graph which can be processed in an alpha-step.

    Notes
    -----
    Currently we use the first applicable table that we encouter when iterating the join graph,
    but this is an implementation detail that might change in the future.
    """
    return next(node for node in join_graph.nodes if join_graph.degree[node] == 1)


def decompose_acyclic(
    join_graph: nx.Graph,
    root: Optional[pb.TableReference] = None,
    *,
    statistics: Mapping[pb.ColumnReference, PiecewiseConstantFn],
) -> AlphaStep:
    """Transforms an acyclic SQL query into a tree structure for bound estimation.

    The SQL query must be given by its join graph representation, as created by `fdsb_graph`. Do not
    use the output of ``query.predicates().join_graph()` here! Optionally, a root can be given
    explicitly. This must be a table with a degree of 1 in the join graph. Otherwise, an arbitrary
    (suitable) root is selected. The statistics must contain a PCF for each join column.

    Notes
    -----
    The transformation from SQL query into a tree of alpha/beta steps was not detailed in the
    original SafeBound paper. A reverse engineering of the reference implementation also did not
    prove successfull. Therefore, we develop our own decomposition algorithm it works as follows:

    We always use an alpha step as the root node. This is because a join is (by definition)
    symmetric, i.e. there is always at least on other partner relation. This maps nicely to the
    properties of the alpha step. In contrast, the beta step always requires a projection onto some
    target column. This works nicely for queries like *SELECT R.a FROM R, S WHERE R.b = S.c* (as
    demonstrated in the original SafeBound paper), but we cannot assume that we always have such a
    projection target available (e.g., for the query *SELECT \\* FROM R, S WHERE R.a = S.b*).

    We use an arbitrary join from the input query as our tree root. Afterwards, we greedily process
    all of the relations that take part in this join. If they are plain relations (i.e. they take
    part in no other joins), we can "consume" as part of the alpha step. Such a relation does not
    need to be considered further.

    If the relation takes part in additional joins, it needs to be modelled as a beta step. The key
    idea is to have this beta step project onto the column that is part of the current alpha step.

    The relations that take part in the beta step can be processed using the same distinction: If
    they only take part in the beta step join, they can be consumed completely. If multiple
    relations join on the same dimension, they are first combined in an alpha step. If the take
    part in additional joins, they are processed in a beta step. This maps to a nice recursive
    processing scheme.

    One open question is whether all decompositions of an input query produce the same bound. The
    original SafeBound paper did not discuss different decomposition algorithms and only assumed
    their existence.

    See Also
    --------
    decompose_query : higher-level entry point which takes care of necessary transformations
    fdsb : the highest-level entry point with support for cyclic queries
    """
    root = root or _select_acyclic_root(join_graph)
    join: int = next(iter(join_graph.adj[root]))
    return _create_alpha_step(
        join_graph, join, source=root, include_source=True, statistics=statistics
    )


def fdsb_graph(query: pb.SqlQuery) -> nx.Graph:
    """Transforms an SQL query into a graph representation tailored to the FDSB algorithm.

    Notes
    -----
    The resulting graph is essentially a bipartite graph with one set of nodes (distinguished by the
    *node_type* attribute) correponding to the base tables, and the other color corresponding to
    join equivalence classes. All relations that are part of a join equivalence class are connected
    to the correponding join node. Edges are annotated with the *join_col*  that takes part in the
    join equivalence class.
    """
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
    """Decomposes an acyclic query into its alpha/beta step tree.

    In contrast to `decompose_acyclic`, this method automatically takes care of the translation from
    SQL query to join graph, and it selects a suitable root node for the decomposition.
    Still, this function is limited to acyclic queries. For cyclic queries, there exists no single
    tree decomposition. Therefore, this situation can only be handled by the `fdsb` function.

    The statistics must contain on PCF for each join column.

    See Also
    --------
    decompose_acyclic : the decomposition work-horse
    fdsb : the high-level entry point with support for cyclic queries
    """
    join_graph = fdsb_graph(query)
    if not nx.is_tree(join_graph):
        raise ValueError(f"Decomposition only works for acyclic queries, not '{query}'")
    root = _select_acyclic_root(join_graph)
    return decompose_acyclic(join_graph, root, statistics=statistics)


def fdsb(
    query: pb.SqlQuery, *, statistics: Mapping[pb.ColumnReference, PiecewiseConstantFn]
) -> pb.Cardinality:
    """Decomposes a query and computes its upper bound.

    This is the highest-level function for bound computation. It does not give access to the
    underlying tree decomposition. Instead, it directly provides the bound. This allows to handle
    both acyclic and cyclic queries: for acyclic queries, the bound can be computed directly from
    the (single) decomposition. For cyclic queries, all possible decomposition based on the
    different spanning trees are calculated. Afterwards, the minimum bound of all decompositions is
    determined. This is the same approach as used in the original SafeBound paper.

    Note that this is the only function that is capable of handling cyclic queries.

    The statistics must contain on PCF for each join column.

    See Also
    --------
    decompose_acyclic : for details on the underlying decomposition algorithm
    """
    join_graph = fdsb_graph(query)

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
