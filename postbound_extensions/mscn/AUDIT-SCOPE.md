# Audit scope: MSCN

- **Family:** learned-cardinality
- **What the optimizer produces:** a cardinality estimate per (sub-)query from a multi-set
  convolutional network.
- **Critical outcome:** issues that change the estimates, measured as q-error on JOB-light.
- **Paper:** `MSCN-Paper.pdf`
- **Reference implementation:** `reference-implementation/`; the code path behind the paper's
  results: `train.py`, `mscn/model.py`, `mscn/data.py`, `mscn/util.py`, with the paper's
  configuration (100 000 queries split 90/10, 100 epochs, batch 1024, 256 hidden units, lr 0.001).

## Database and workload

- **Database:** IMDB
- **Workload:** JOB-light, via `pb.workloads.job_light()`
- **Backend for measurements:** DuckDB (`imdb.duckdb`); cardinality work does not need Postgres.
- **Benchmarks run on:** not required

## Focus areas

1. Query featurization
2. Network architecture
3. Transformations and scaling applied to inputs and outputs

## Excluded

- Everything listed as a non-goal in the README, most notably sample bitmaps.
- The query sampler (`tools/query-sampler.py`, `postbound_extensions/sampler/`): it is a standalone
  tool, not part of MSCN.

## Notes

- Our IMDB instance is a newer snapshot than the paper's; the reference's labels do not reproduce.
  Compare both implementations on the reference's own `data/train.csv` and JOB-light labels.
- Previous audit: `mscn-audit.md` (2026-11); its fixes landed in commit `43282d0`.
