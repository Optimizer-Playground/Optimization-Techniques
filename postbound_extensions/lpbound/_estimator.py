from __future__ import annotations

from collections.abc import Iterable
from typing import override

import postbound as pb

from ._linprog import LpBerge, LpFlow, is_berge_acyclic, lp_join_graph
from ._norms import LpBoundCatalog


class LpBoundEstimator(pb.CardinalityEstimator):
    """LpBound is an upper bound-driven cardinality estimator.

    At its core, LpBound uses Lp norms of most frequent value lists (hence the name) over the join columns to calculate
    the upper bounds. The norms are stored in a `LpBoundCatalog` and feed into a linear program that links information
    theory (specifically entropy) with the Lp norms. The estimator itself acts primarily as a coordinator
    that retrieves the norms from the catalog, decomposes the input query, and constructs the linear program.
    Depending on the query shape, two different linear programs are used: `LpBerge` is used for Berge-acyclic queries,
    whereas `LpFlow` is used for all other queries. LPflow achieves the increased generality by using significantly
    more variables and constraints: LPberge is linear in the query size while LPflow is quadratic.

    Parameters
    ----------
    catalog: LpBoundCatalog
        The catalog that stores the Lp norms. The estimator relies on the catalog to retrieve the relevant norms for a
        given query.

    See Also
    --------
    LpBoundCatalog : The catalog that stores the Lp norms
    lp_join_graph : A utility to deconstruct a query into the LpBound-specific graph structure
    is_berge_acyclic : A utility function to check if a query graph is Berge-acyclic
    LpBerge :
        The linear program that is used for Berge-acyclic queries.
        It is significantly smaller than LpFlow and thus faster to solve.
    LpFlow : The linear program that can be used for all queries

    References
    ----------
    .. Haozhe Zhang et al.: "LpBound: Pessimistic Cardinality Estimation using lp-Norms of Degree Sequences"
      (SIGMOD 2025) https://doi.org/10.1145/3725321
    """

    def __init__(self, catalog: LpBoundCatalog) -> None:
        super().__init__()
        self._catalog = catalog

    @override
    def calculate_estimate(
        self, query: pb.SqlQuery, intermediate: pb.TableReference | Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        subquery = pb.transform.extract_subquery(query, intermediate)
        if len(subquery.tables()) == 1:
            return pb.Cardinality.unknown()
        if not pb.qal.is_select_query(subquery):
            raise pb.qal.QueryTypeError.expected_select(subquery)

        subquery = pb.transform.infer_between_predicates(subquery)
        join_graph = lp_join_graph(subquery)
        statistics = self._catalog.retrieve_norms(subquery)

        if is_berge_acyclic(join_graph):
            lp_prog = (
                LpBerge(join_graph)
                .add_statistics_constraints(statistics)
                .add_additivity_constraints()
                .finalize()
            )  # fmt: skip
        else:
            lp_prog = (
                LpFlow(join_graph)
                .add_statistics_constraints(statistics)
                .finalize()
            )  # fmt: skip

        return lp_prog.compute_bound()
