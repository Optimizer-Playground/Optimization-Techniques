from __future__ import annotations

from collections.abc import Iterable

import postbound as pb

from ._catalog import SafeBoundCatalog
from ._fdsb import decompose_query, fdsb


class SafeBoundEstimator(pb.CardinalityEstimator):
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
        decomposition = decompose_query(query, statistics=stats)
        return fdsb(decomposition)

    def describe(self) -> pb.util.jsondict:
        return {"name": "SafeBound", "catalog": self._catalog}
