# Audit scope template

Each optimizer that gets audited has an `AUDIT-SCOPE.md` in its directory
(`postbound_extensions/<optimizer>/AUDIT-SCOPE.md`). It holds everything that differs between
audits; everything that stays the same lives in the skill. Keep it short: a reader should see
at a glance what was asked of the audit.

```markdown
# Audit scope: <Optimizer name>

- **Family:** learned-plan-selection | learned-cardinality | cardinality-bounds | generic
- **What the optimizer produces:** <the output whose correctness matters, e.g. "a hint set per query">
- **Critical outcome:** <the deviation to pay special attention to, e.g. "issues that lead to
  different hint sets being predicted">
- **Paper:** <file name in the directory>
- **Reference implementation:** <path>; the code path behind the paper's results: <entry points>

Both are required. If either is missing, the audit stops and the user decides how to proceed.

## Database and workload

- **Database:** <IMDB | Stats | ...>
- **Workload:** <JOB | JOB-light | Stats-CEB | ...>, accessed via <e.g. `pb.workloads.job()`>
- **Backend for measurements:** <DuckDB | Postgres>, and why
- **Benchmarks (if any) run on:** <Postgres | not required>

## Focus areas

In priority order. The family checklist adds to these.

1. <area>
2. <area>
3. (optional) <area>

## Excluded

- Everything listed as a non-goal in the README.
- Standalone tools (query sampler, training-data generators) and PostBOUND itself, unless listed
  as a focus area above.
- <further exclusions specific to this audit>

## Notes

<Anything the auditor would otherwise have to rediscover, e.g. pre-built catalogs, data files,
known environment differences.>
```
