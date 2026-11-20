from __future__ import annotations

from collections.abc import Mapping, Sequence

import postbound as pb

DatabaseCacheState = Mapping[pb.TableReference, float]


def infer_cache_states(
    plans: Sequence[pb.QueryPlan], database: pb.Database, *, log: pb.util.Logger
) -> list[DatabaseCacheState]:
    """Determines the state of the shared buffer of the `database` for all relevant relations in each of the `plans`.

    All plans are evaluated against the same snapshot of the shared buffer. This is important when comparing the
    candidate plans of a single optimization decision: each plan should see the same cache state.
    """
    if not isinstance(database, pb.postgres.PostgresDatabase):
        return [{} for _ in plans]

    statistics_catalog = database.statistics()
    # buffer_state() scans the entire shared buffer, so we only do this once per decision.
    # Relations without any buffered pages are not contained in the result.
    buffer_state = statistics_catalog.buffer_state()
    page_counts: dict[str, int] = {}
    return [
        _infer_cache_state(plan, statistics_catalog, buffer_state=buffer_state, page_counts=page_counts, log=log)
        for plan in plans
    ]


def infer_cache_state(plan: pb.QueryPlan, database: pb.Database, *, log: pb.util.Logger) -> DatabaseCacheState:
    """Determines the current state of the shared buffer of the `database` for all relevant relations in the `plan`."""
    return infer_cache_states([plan], database, log=log)[0]


def _infer_cache_state(
    plan: pb.QueryPlan,
    statistics_catalog: pb.postgres.PostgresStatistics,
    *,
    buffer_state: Mapping[str, int],
    page_counts: dict[str, int],
    log: pb.util.Logger,
) -> DatabaseCacheState:
    scans = plan.find_all_nodes(pb.QueryPlan.is_scan)
    cache_state: dict[pb.TableReference, float] = {}
    for node in scans:
        if not node.base_table:
            log(f"Node {node} has no base table. Skipping cache state inference for this node.")
            continue

        match node.operator:
            case pb.ScanOperator.SequentialScan:
                rels = [node.base_table.full_name]
            case pb.ScanOperator.IndexScan:
                rels = [node.params.index, node.base_table.full_name]
            case pb.ScanOperator.IndexOnlyScan:
                rels = [node.params.index]
            case pb.ScanOperator.BitmapScan:
                rel_set = set()
                for child in node:
                    if child.base_table is not None:
                        rel_set.add(child.base_table.full_name)
                    if child.params.index:
                        rel_set.add(child.params.index)
                rels = list(rel_set)
            case _:
                raise ValueError(f"Unknown scan operator: {node.operator}")

        pct_sum = 0
        buffered_rels = 0
        for rel in rels:
            total_pages = page_counts.get(rel)
            if total_pages is None:
                total_pages = statistics_catalog.n_pages(rel)
                page_counts[rel] = total_pages
            if not total_pages:
                continue

            buffered_pages = buffer_state.get(rel, 0)
            pct_sum += min(buffered_pages / total_pages, 1.0)
            buffered_rels += 1

        if not buffered_rels:
            continue
        cache_state[node.base_table] = pct_sum / buffered_rels

    return cache_state


def determine_cache_pct(node: pb.QueryPlan, cache_state: DatabaseCacheState, *, log: pb.util.Logger) -> float:
    """Extracts the cache percentage for a node from the `cache_state`.

    If no cache state is found, a fallback value is used.
    """

    if not node.is_scan():
        return 0

    cache_pct = cache_state.get(node.base_table)
    if cache_pct is not None:
        return cache_pct

    # magic constants as last-resort-never-use-fallback
    log(f"No cache state found for node {node}. Falling back to magic values!")
    match node.operator:
        case pb.ScanOperator.SequentialScan:
            return 0.4
        case pb.ScanOperator.IndexScan | pb.ScanOperator.IndexOnlyScan:
            return 0.8
        case pb.ScanOperator.BitmapScan:
            return 0.6
        case _:
            raise ValueError(f"Unknown scan operator: {node}")
