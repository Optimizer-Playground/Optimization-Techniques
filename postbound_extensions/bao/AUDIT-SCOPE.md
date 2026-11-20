# Audit scope: Bao

- **Family:** learned-plan-selection
- **What the optimizer produces:** a hint set (arm) per query, selected by a tree-convolutional
  value network, retrained online from executed queries.
- **Critical outcome:** issues that lead to different hint sets being predicted.
- **Paper:** `BAO-Paper.pdf`
- **Reference implementation:** `reference-implementation/`; the code path behind the paper's
  results: `bao_server/` (`model.py`, `featurize.py`, `train.py`, `main.py`), the Postgres
  extension `pg_extension/` (hint sets, plan JSON, buffer state), and `run_queries.py` as the
  driver.

## Database and workload

- **Database:** IMDB
- **Workload:** JOB, via `pb.workloads.job()`
- **Backend for measurements:** Postgres (`.psycopg_connection_job`). Bao's hint sets are Postgres
  planner switches and its featurization uses Postgres plan operators and buffer-cache state, so
  DuckDB cannot be used.
- **Benchmarks run on:** Postgres

## Focus areas

1. Plan featurization
2. Network architecture
3. Transformations and scaling applied to inputs and outputs
4. The retraining setup (the online feedback loop)
5. (optional, if other results do not make sense) generation of the training data through Bao's
   own calibration path (`calibrate`), not the standalone query sampler

## Excluded

- Everything listed as a non-goal in the README.
- Runtime labels measured through `EXPLAIN ANALYZE`: PostBOUND cannot measure uninstrumented
  runtimes, and all queries carry the same kind of overhead.

## Notes

- Previous audit: `bao-audit.md` (2026-11); its fixes landed in commit `4d48b6f`.
- The cache feature (issue 4 of the previous audit) was redesigned afterwards: `BaoOptimizer`
  registers the buffer state at plan time and stores it with each experience sample; a missing
  cache state is an error. Dropping the cache feature entirely was evaluated as an alternative
  and was neutral within seed noise on JOB and Stack, for both our implementation and the
  reference.
