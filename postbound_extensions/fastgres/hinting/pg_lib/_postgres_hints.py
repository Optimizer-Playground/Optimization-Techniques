from __future__ import annotations
from .._hint import Hint

# pg12
INDEX_ONLY_SCAN = Hint("INDEX_ONLY_SCAN", "enable_indexonlyscan", True)
SEQ_SCAN = Hint("SEQ_SCAN", "enable_seqscan", True)
INDEX_SCAN = Hint("INDEX_SCAN", "enable_indexscan", True)
NESTED_LOOP_JOIN = Hint("NESTED_LOOP_JOIN", "enable_nestloop", True)
MERGE_JOIN = Hint("MERGE_JOIN", "enable_mergejoin", True)
HASH_JOIN = Hint("HASH_JOIN", "enable_hashjoin", True)
TID_SCAN = Hint("TID_SCAN", "enable_tidscan", True)
SORT = Hint("SORT", "enable_sort", True)
PARALLEL_HASH = Hint("PARA_HASH", "enable_parallel_hash", True)
PARALLEL_APPEND = Hint("PARA_APPEND", "enable_parallel_append", True)
MATERIALIZATION = Hint("MATERIALIZATION", "enable_material", True)
HASH_AGGREGATION = Hint("HASH_AGG", "enable_hashagg", True)
GATHER_MERGE = Hint("GATHER_MERGE", "enable_gathermerge", True)
BITMAP_SCAN = Hint("BITMAP_SCAN", "enable_bitmapscan", True)

# pg13
INCREMENTAL_SORT = Hint("INC_SORT", "enable_incremental_sort", True)

# pg14
MEMOIZE = Hint("MEMOIZE", "enable_memoize", True)

# pg15
# pg16
PRESORTED_AGGREGATION = Hint("PRESORT_AGG", "enable_presorted_aggregate", True)

# pg17
GROUP_BY_REORDER = Hint("GROUP_BY_REORDER", "enable_group_by_reordering", True)

# partitions
PARTITION_WISE_AGGREGATE = Hint("PART_AGG", "enable_partitionwise_aggregate", False)
PARTITION_WISE_JOIN = Hint("PART_JOIN", "enable_partitionwise_join", False)
PARTITION_PRUNING = Hint("PART_PRUNING", "enable_partition_pruning", True)

# multi backend
ASYNC_APPEND = Hint("ASYNC_APPEND", "enable_async_append", True)

# GEQO
GEQO = Hint("GEQO", "geqo", True)

# JIT
JIT = Hint("JIT", "jit", True)

CORE_HINTS = [
    INDEX_ONLY_SCAN,
    SEQ_SCAN,
    INDEX_SCAN,
    NESTED_LOOP_JOIN,
    MERGE_JOIN,
    HASH_JOIN,
]

PG12_HINTS = [
    *CORE_HINTS,
    TID_SCAN,
    SORT,
    PARALLEL_HASH,
    PARALLEL_APPEND,
    MATERIALIZATION,
    HASH_AGGREGATION,
    GATHER_MERGE,
    BITMAP_SCAN,
]
PG13_HINTS = [*PG12_HINTS, INCREMENTAL_SORT]
PG14_HINTS = [*PG13_HINTS, MEMOIZE]
PG15_HINTS = [*PG14_HINTS]
PG16_HINTS = [*PG15_HINTS, PRESORTED_AGGREGATION]
PG17_HINTS = [*PG16_HINTS, GROUP_BY_REORDER]

# additional hints
PARTITION_HINTS = [PARTITION_WISE_AGGREGATE, PARTITION_WISE_JOIN, PARTITION_PRUNING]
BACKEND_HINTS = [ASYNC_APPEND]
MISC_HINTS = [GEQO, JIT]
