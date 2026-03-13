
def get_hint_abbreviations():
    abbreviations = {
            "enable_indexonlyscan": "IDO-S",
            "enable_seqscan": "Seq-S",
            "enable_indexscan": "ID-S",
            "enable_nestloop": "NL-J",
            "enable_mergejoin": "M-J",
            "enable_hashjoin": "H-J",
            "enable_tidscan": "Tid-J",
            "enable_sort": "Sort",
            "enable_parallel_hash": "Para-Hash",
            "enable_parallel_append": "Para-A",
            "enable_material": "Mat",
            "enable_hashagg": "Hash-Agg",
            "enable_gathermerge": "Gather-M",
            "enable_bitmapscan": "BM-S",
            # pg12
            "enable_incremental_sort": "Inc-Sort",
            # pg14
            "enable_memoize": "Mem",
            # pg15/16
            "enable_presorted_aggregate": "PSort-Agg",
            # partitions
            "enable_partitionwise_aggregate": "Part-Agg",
            "enable_partitionwise_join": "Part-J",
            "enable_partition_pruning": "Part-P",
            # multi backend
            "enable_async_append": "Async-A",
            "geqo": "GEQO"
        }
    return abbreviations
