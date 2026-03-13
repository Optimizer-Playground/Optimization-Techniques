from ..pg_lib._libraries import (PG12_HINTS, PG13_HINTS, PG14_HINTS, PG15_HINTS, PG16_HINTS,
                                 PG17_HINTS, CORE_HINT_LIBRARY)
from ._postgres_hints import PARTITION_HINTS, MISC_HINTS, BACKEND_HINTS
from .._hint_library import HintLibrary


def get_available_library(postgres_version: str,
                          use_partition_hints: bool = False,
                          use_misc: bool = True,
                          use_backend: bool = False) -> HintLibrary:
    hints = []
    pg_v = None
    if "12" in postgres_version:
        hints = PG12_HINTS
        pg_v = 12
    if "13" in postgres_version:
        hints = PG13_HINTS
        pg_v = 13
    if "14" in postgres_version:
        hints = PG14_HINTS
        pg_v = 14
    if "15" in postgres_version:
        hints = PG15_HINTS
        pg_v = 15
    if "16" in postgres_version:
        hints = PG16_HINTS
        pg_v = 16
    if "17" in postgres_version:
        hints = PG17_HINTS
        pg_v = 17
    if not hints or pg_v is None:
        raise ValueError(f"Unknown PostgreSQL version {postgres_version}")

    if use_partition_hints:
        hints.extend(PARTITION_HINTS)
    if use_misc:
        hints.extend(MISC_HINTS)
    if use_backend and pg_v > 13:
        hints.extend(BACKEND_HINTS)

    return HintLibrary(hints)


def get_default_library() -> HintLibrary:
    return CORE_HINT_LIBRARY
