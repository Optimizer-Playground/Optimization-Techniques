# Audit scope: SafeBound

- **Family:** cardinality-bounds
- **What the optimizer produces:** an upper bound on the cardinality of each (sub-)query, from
  compressed degree sequences (FDSB).
- **Critical outcome:** issues that lead to different upper bounds being calculated, above all
  bounds below the true cardinality.
- **Paper:** `SafeBound-Paper.pdf` (not present in this checkout; add it before auditing)
- **Reference implementation:** `reference-implementation/` (not present in this checkout; add it
  before auditing)

## Database and workload

- **Database:** Stats
- **Workload:** Stats-CEB, via `pb.workloads.stats()`
- **Backend for measurements:** DuckDB (`stats.duckdb`) for true cardinalities; Postgres
  (`.psycopg_connection_stats`) for building the catalog and for any runtime benchmark.
- **Benchmarks run on:** Postgres

## Focus areas

1. Selection of the degree sequences to generate
2. Compression of the degree sequences (conversion to piecewise-constant functions)
3. Combination of correlated degree sequences for filter predicates
4. Construction of the FDSB inference tree and the inference logic

## Excluded

- Everything listed as a non-goal in the README, notably group compression and PK-FK pre-joins.

## Notes

- The original audit used a pre-generated Stats catalog in a `generated` directory, which is not
  present in this checkout. Any pre-generated catalog may bake in defects from before the last
  fixes; rebuild it before measuring tightness.
- Previous audit: `safebound-audit.md` (2026-11).
