from __future__ import annotations

from collections.abc import Iterable

import postbound as pb


class SafeBoundEstimator(pb.CardinalityEstimator):
    def __init__(self) -> None:
        super().__init__()

    def calculate_estimate(
        self,
        query: pb.SqlQuery,
        intermediate: pb.TableReference | Iterable[pb.TableReference],
    ) -> pb.Cardinality:
        pass

    def describe(self) -> pb.util.jsondict:
        return {"name": "SafeBound"}
