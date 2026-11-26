# Audit scope: LpBound

- **Family:** cardinality-bounds
- **What the optimizer produces:** an upper bound on the cardinality of each (sub-)query, obtained
  as the optimum of a linear program (LPberge or LPflow) over lp-norms of (conditioned) degree
  sequences.
- **Critical outcome:** issues that lead to different upper bounds being calculated, above all
  bounds below the true cardinality (soundness), then bounds that are looser than the reference's
  on the same statistics (tightness).
- **Paper:** `LpBound-Paper.pdf`
- **Reference implementation:** `reference-implementation/`; the code path behind the paper's
  results: `benchmarks/experiments/accuracy_acyclic.py` (→ `src/lpbound/acyclic/lpbound.py`:
  `build_lpbound_statistics`, `estimate`) for acyclic workloads and
  `benchmarks/experiments/accuracy_cyclic.py` (→ `src/lpbound/cyclic/`) for cyclic ones, configured
  by `src/lpbound/config/lpbound_config.py`. The C++ solver is only needed for timing, not accuracy.

## Database and workload

- **Database:** Stats (primary); IMDB (secondary, only to exercise LPflow on cyclic queries)
- **Workload:** Stats-CEB, via `pb.workloads.stats()`; JOB-light, via `pb.workloads.job_light()`;
  for focus area 5, the cyclic JOB queries via `pb.workloads.job()`
- **Backend for measurements:** DuckDB (`stats.duckdb`, `imdb.duckdb`) for true cardinalities and
  for running the reference (it builds its statistics on DuckDB); Postgres
  (`.pg-connect-stats.toml`) for building our catalog through `LpBoundCatalog.load_or_build`, as
  the README does.
- **Benchmarks run on:** not required (accuracy only)

## Focus areas

1. Computation of the lp-norms: which degree sequences are built (join columns, per relation),
   which norms (p range, l0/l∞), NULL handling.
2. Conditioned norms for filter predicates: MCV-based equality norms, the defaults for non-MCV
   values, the histogram hierarchy for ranges, our half-open range extension, and how several
   predicates on one relation are combined.
3. Construction of the join graph (`lp_join_graph`) and the LP formulations (`LpBerge`, `LpFlow`):
   constraints, variables, objective, and the choice between them (`is_berge_acyclic`).
4. Sub-query estimation through `LpBoundEstimator` (the entry point an optimizer uses), including
   catalog persistence (`store`/`load`/`load_or_build` give identical bounds).
5. (lower priority) LPflow on cyclic queries.

## Excluded

- Everything listed as a non-goal in the README: predicate propagation (pre-computed PK/FK joins),
  prefix sharing of degree sequences, `GROUP BY` queries.
- Estimation and statistics-computation time (no C++ solver comparison).
- Standalone tools (query sampler, training-data generators) and PostBOUND itself.
- The SafeBound module, except the parts LpBound imports and calls on its own code path.

## Notes

- Our implementation reuses parts of `postbound_extensions/safebound/` (degree-sequence, MCV and
  histogram fetching); those parts are in scope as far as LpBound calls them. The SafeBound
  scope refers to a `safebound-audit.md`, which is not in this checkout.
- The README states a median accuracy of 30.80 (with predicate propagation) vs 109.85 (without)
  on Stats; the reference ships its Stats estimates in
  `results/accuracy_acyclic/stats/lpbound_stats_full_estimations.csv` and true cardinalities in
  `truecardinality_stats_full_estimations.csv`.
- To compare like with like, run the reference with predicate propagation disabled as well as in
  the authors' configuration.
- True cardinalities: `models/true-cards/stats-cards.parquet` and
  `models/true-cards/intermediate-cards-stats.csv` (verify before use).
- No pre-built LpBound catalog exists in `models/`; the audit builds one.
