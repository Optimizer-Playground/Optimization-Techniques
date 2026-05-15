from __future__ import annotations

from collections.abc import Iterable

import postbound as pb

from ._catalog import SafeBoundCatalog
from ._fdsb import fdsb


class SafeBoundEstimator(pb.CardinalityEstimator):
    """SafeBound is an upper bound-driven cardinality estimator.

    At its core, SafeBound uses (compressed versions) of most frequent value lists over the join
    columns to calculate the upper bounds. The MCVs are encoded as *piecewise constant functions*
    that are stored in a `SafeBoundCatalog`. The estimator itself acts primarily as a coordinator
    that retrieves the PCFs from the catalog, decomposes the input query, and invokes the FDSB
    algorithm to compute the bound.

    Parameters
    ----------
    catalog: SafeBoundCatalog
        The catalog that stores the piecewise constant functions. The estimator relies on the
        catalog to retrieve the relevant PCFs for a given query.

    See Also
    --------
    SafeBoundCatalog : The catalog that stores the piecewise constant functions
    fdsb : The algorithm that handles the query decomposition and the actual bound calculation

    References
    ----------
    .. Kyle Deeds et al.: "SafeBound: A Practical System for Generating Cardinality Bounds"
      (SIGMOD 2023) https://doi.org/10.1145/3588907
    """

    def __init__(self, catalog: SafeBoundCatalog) -> None:
        super().__init__()
        self._catalog = catalog

    def calculate_estimate(
        self,
        query: pb.SqlQuery,
        intermediate: pb.TableReference | Iterable[pb.TableReference],
    ) -> pb.Cardinality:
        subquery = pb.transform.extract_query_fragment(query, intermediate)
        if subquery is None:
            raise ValueError("No valid subquery found")

        stats = self._catalog.retrieve_stats(subquery)
        return fdsb(subquery, statistics=stats)

    def describe(self) -> pb.util.jsondict:
        return {"name": "SafeBound", "catalog": self._catalog}
