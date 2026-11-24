from __future__ import annotations

from collections.abc import Iterable
from typing import override

import postbound as pb

from ._linprog import LpBerge, LpFlow, is_berge_acyclic, lp_join_graph
from ._norms import LpNormRepo


class LpBoundEstimator(pb.CardinalityEstimator):
    def __init__(self, catalog: LpNormRepo) -> None:
        super().__init__()
        self._catalog = catalog

    @override
    def calculate_estimate(
        self, query: pb.SqlQuery, intermediate: pb.TableReference | Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        subquery = pb.transform.extract_subquery(query, intermediate)
        if len(subquery.tables()) == 1:
            return pb.Cardinality.unknown()

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
