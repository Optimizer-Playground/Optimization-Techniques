# Environments, data and known traps

Facts in this file were true when the earlier audits ran. Verify them before relying on them, and
update this file when one turns out to be stale or when an audit discovers a new trap.

## Databases and connections

| database | DuckDB file (repository root) | Postgres connection file (repository root)              |
| -------- | ----------------------------- | ------------------------------------------------------- |
| IMDB     | `imdb.duckdb`                 | `.psycopg_connection_job` or `.pg-connect-imdb.toml`    |
| Stats    | `stats.duckdb`                | `.psycopg_connection_stats` or `.pg-connect-stats.toml` |
| Stack    | –                             | `.psycopg_connection_stack` or `.pg-connect-stack.toml` |

- DuckDB: `pb.duckdb.connect("imdb.duckdb")`. Postgres: `pb.postgres.connect(config_file=".psycopg_connection_job")`
  (check the current PostBOUND signature).
- Never print or copy the contents of the connection files into the report.
- Stack is not available on all systems.

## Which backend to use

- **Prefer DuckDB** for anything that only needs data: true cardinalities, sampling, label
  computation. It is much faster.
- **Use Postgres** when the optimizer is Postgres-specific (planner switches/hints, `EXPLAIN` plan
  operators, buffer-cache features) or when the scope asks for runtime benchmarks. State the reason
  in the report.

## Workloads

- JOB: `pb.workloads.job()` (113 queries); JOB-light: `pb.workloads.job_light()` (70 queries);
  Stats: `pb.workloads.stats()` (146 queries). Check the current PostBOUND API if a name fails.
- Sub-queries for cardinality work: enumerate all connected sub-joins with ≥ 2 tables. On Stats this
  gives 1566; a few time out or exceed memory even on DuckDB.

## Known traps

- **IMDB snapshot:** our IMDB instance is newer than the papers' (4 736 114 titles vs 2 528 312).
  Labels shipped with reference implementations (MSCN `train.csv`, JOB-light cardinalities) do not
  reproduce on it. Use the reference's data for both implementations instead.
- **Stats** true cardinalities computed on DuckDB match the SafeBound reference's `Stats_Sizes.csv`.
- **PostBOUND units:** `execution_time` in plans is in **seconds**. Several defects came from code
  treating it as milliseconds.
- **PostBOUND API changes:** version 0.23 changed predicate visitors (`visit_and/or/not_predicate`
  no longer receive `components`; `visit_query_predicates` takes predicates, not a query). Code
  written against older versions may raise `TypeError` only on queries with conjunctions.
- **Postgres 18** includes `BUFFERS` in `EXPLAIN ANALYZE` by default, so analyzed plans carry
  buffer hit counters that plain `EXPLAIN` plans do not.
- **`EXPLAIN ANALYZE` overhead** inflates runtimes by a median of 12–22% on JOB (up to ~3×).
  PostBOUND can only measure runtimes through `EXPLAIN ANALYZE`, so every optimizer's labels and
  rewards carry this overhead. This is an accepted limitation, not an audit finding; use the same
  timing method for all variants you compare.
- **PostBOUND estimates under `Gather`** are multiplied by `workers + 1`, unlike raw `Plan Rows`.
- **pg_lab hinting** occasionally rejects a valid plan (`DatabaseServerError`); treat these as
  tooling failures, not optimizer defects, and count them.
- **Warm vs cold cache:** do a warm-up run before measured runs, and record the cache state when
  the optimizer uses cache features.
- **Run-order cache warming:** when the same query runs under several variants in a fixed order,
  later variants profit from pages the earlier ones loaded, even with identical plans. Take one
  cache snapshot per decision, deduplicate identical plans, and randomize or rotate the order.
- **Postgres timeouts on parallel queries:** a `statement_timeout` that cancels a parallel query
  has crashed the backend (segfault) in later statements. Reconnect after every timeout.
- All commands under MacOS should be executed using `caffeinate` to prevent the system from sleeping
  and interrupting the audit. On Linux, use `systemd-inhibit` or `nohup` to prevent sleep.

## Where to put audit artifacts

Store scripts, recorded data and results in a persistent directory outside the repository:

```
~/.claude/projects/<project-dir-name>/artifacts/<YYYY-MM-DD>-<optimizer>-audit/
├── README.md     # what each script does, how to rerun it
├── scripts/
├── data/         # recorded measurements (e.g. pickled plans and runtimes per query × variant)
└── results/
```

`<project-dir-name>` is the repository path with `/` replaced by `-`. Do not use the session
scratchpad under `/tmp`: it has been wiped between sessions before, losing an audit's data.

The report is committed to the repository and read by people on other machines, so it must not
contain machine-specific paths. Refer to the artifacts directory only by its name
(`<YYYY-MM-DD>-<optimizer>-audit`), and give the user the full path in your final message instead.

## Troubleshooting

In case of (larger) discrepancies between the expected environment and the actual environment
while running the audit, **pause and ask the user how to proceed.**

Situations that require user intervention include for example a database server that does not
seem to be running (e.g., no running Postgres server), or a missing workloads
(e.g., `pb.workloads.job()` not returning any queries). Smaller issues like inconsistent
connection files (e.g. `.pg-connect-imdb.toml` vs. `.pg-connect-job.toml` vs. `.psycopg_connection_imdb`)
can be reconciled without asking the user.
