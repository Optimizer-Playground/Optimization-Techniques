import re

from .._hint_library import HintLibrary
from ._libraries import CORE_HINT_LIBRARY
from ._postgres_hints import (
    BACKEND_HINTS,
    MISC_HINTS,
    MISC_HINTS_PG19,
    PARTITION_HINTS,
    PG12_HINTS,
    PG13_HINTS,
    PG14_HINTS,
    PG15_HINTS,
    PG16_HINTS,
    PG17_HINTS,
    PG18_HINTS,
    PG19_HINTS,
)


def get_available_library(postgres_version: str,
                          use_partition_hints: bool = False,
                          use_misc: bool = True,
                          use_backend: bool = False) -> HintLibrary:
    """Select PG12–PG19 hints, including version-specific declared defaults.

    Accepts a major version, a server version, or PostgreSQL's version() banner.
    Optional hints are appended to a fresh list so calls cannot alter libraries.
    """
    version_hints = {
        12: PG12_HINTS, 13: PG13_HINTS, 14: PG14_HINTS, 15: PG15_HINTS,
        16: PG16_HINTS, 17: PG17_HINTS, 18: PG18_HINTS, 19: PG19_HINTS,
    }
    match = re.match(r"(?:PostgreSQL\s+)?(\d+)(?=\.|\s|$|alpha|beta|rc|devel)",
                     postgres_version.strip(), re.IGNORECASE)
    pg_v = int(match[1]) if match else None
    if pg_v not in version_hints:
        raise ValueError(f"Unknown PostgreSQL version {postgres_version}")
    hints = list(version_hints[pg_v])

    if use_partition_hints:
        hints.extend(PARTITION_HINTS)
    if use_misc:
        hints.extend(MISC_HINTS_PG19 if pg_v >= 19 else MISC_HINTS)
    if use_backend and pg_v > 13:
        hints.extend(BACKEND_HINTS)

    return HintLibrary(hints)


def get_default_library() -> HintLibrary:
    return CORE_HINT_LIBRARY
