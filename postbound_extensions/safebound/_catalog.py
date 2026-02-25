from __future__ import annotations

from pathlib import Path

import postbound as pb

from ._piecewise_fns import PiecewiseConstantFn


class SafeBoundCatalog:
    @staticmethod
    def online(workload: pb.Workload, database: pb.Database) -> SafeBoundCatalog:
        pass

    @staticmethod
    def load(archive: Path | str) -> SafeBoundCatalog:
        pass

    @staticmethod
    def load_or_build(
        archive: Path | str, *, workload: pb.Workload, database: pb.Database
    ) -> SafeBoundCatalog:
        archive = Path(archive)
        if archive.is_file():
            return SafeBoundCatalog.load(archive)

        catalog = SafeBoundCatalog.online(workload, database)
        catalog.store(archive)
        return catalog

    def __init__(self) -> None:
        pass

    def retrieve_stats(
        self, query: pb.SqlQuery
    ) -> dict[pb.ColumnReference, PiecewiseConstantFn]:
        pass

    def store(self, archive: Path | str) -> None:
        pass
