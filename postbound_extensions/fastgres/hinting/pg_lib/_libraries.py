from ..pg_lib._postgres_hints import (CORE_HINTS, PG12_HINTS, PG13_HINTS, PG14_HINTS, PG15_HINTS,
                                      PG16_HINTS, PG17_HINTS)
from .._hint_library import HintLibrary

CORE_HINT_LIBRARY = HintLibrary(CORE_HINTS)
PG_12_LIBRARY = HintLibrary(PG12_HINTS)
PG_13_LIBRARY = HintLibrary(PG13_HINTS)
PG_14_LIBRARY = HintLibrary(PG14_HINTS)
PG_15_LIBRARY = HintLibrary(PG15_HINTS)
PG_16_LIBRARY = HintLibrary(PG16_HINTS)
PG_17_LIBRARY = HintLibrary(PG17_HINTS)
