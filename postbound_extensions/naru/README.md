# Naru

- **Original Paper:** _Deep Unsupervised Cardinality Estimation_
- **Authors:** Zongheng Yang, Eric Liang, Amog Kamsetty, Chenggang Wu, Yan Duan,
  Xi Chen, Pieter Abbeel, Joseph M. Hellerstein, Sanjay Krishnan, Ion Stoica
- **Published:** PVLDB 13(3), 2019
- **Link:** <https://doi.org/10.14778/3368289.3368294>
- **Reference Implementation:** <https://github.com/naru-project/naru>

Naru is an unsupervised, data-driven cardinality estimator. For each table, it
learns the joint distribution of all columns with a deep autoregressive model
(MADE/ResMADE), i.e. without any independence assumptions. The selectivity of a
filter predicate is obtained by integrating the learned distribution over the
query region using _progressive sampling_, a Monte Carlo scheme that uses the
model's conditional distributions to steer the samples into the high-density
parts of the query region. Naru is trained directly on the table data, no training
queries are required.

We implement Naru based on the original paper and the reference implementation.
The autoregressive model has been re-used from the reference implementation (with
some fixes), while the remaining parts have been implemented from scratch (see
[Authors and License](#authors-and-license) below).

> [!CAUTION]
> The Naru estimator is an experimental addition to the playground that was entirely
> AI-generated. We use it as a viability study to assess whether we can implement
> related work in an automated fashion. Our process is described below.
> Please proceed with care.

## High-level Interface

Naru implements the PostBOUND [`CardinalityEstimator`](https://postbound.readthedocs.io/en/latest/api/core/optimization-pipelines.html#postbound.CardinalityEstimator)
interface. The `NaruEstimator` acts as the central entrypoint. It manages one
`NaruTableModel` per table. Each table model consists of the dictionary encodings
of its columns (`ColumnDomain`) and the autoregressive model (`MADE`).

**Naru only estimates base tables.** For all intermediates that contain joins,
the estimator returns an unknown cardinality and PostBOUND falls back to the
native estimates of the target database (see
[Deviations](#deviations-from-the-original-paper) for the reasoning).

Which tables and columns are modeled is described by a `ModelSpec`. Similar to the
other estimators of the playground, the spec can be inferred in different ways:

- `NaruEstimator.online()` (`ModelSpec.from_schema()`) models all columns of all
  tables of the database. This is the setting of the paper: Naru approximates the
  full joint distribution of each table and does not need to know anything about
  the workload. The only exception are _free text_ columns (text columns with more
  than `max_domain_size` distinct values, e.g. comments or post bodies), whose
  dictionary would essentially be a copy of the column. Predicates on such columns
  cannot be estimated.
- `NaruEstimator.infer_from()` (`ModelSpec.from_workload()`) only models the tables
  and columns that are filtered in a specific workload. This leads to much smaller
  models that can be trained faster and are typically more accurate.
- `NaruEstimator.from_spec()` uses an arbitrary spec.

All strategies fetch the column domains and training data via PostBOUND (exact
statistics through `pb.db.PreciseStatistics` and plain `SELECT` queries) and train
the models directly. The architecture, training, and inference parameters are
controlled by `NaruHyperParams`. Since Naru learns from the database rather than
from queries, it also implements PostBOUND's data-driven training protocol
(`fit_database()`): an estimator that is created without any models is trained
once the benchmarking tools hand it the target database.

Filter predicates are translated into query regions by `predicate_regions()`.
Predicates on the same column are evaluated directly on the column's dictionary,
which supports all predicates with static values (ranges, `IN`, `BETWEEN`, `LIKE`,
`IS NULL`, arbitrary `AND`/`OR`/`NOT` nesting, ...) under SQL's three-valued
logic. Disjunctions and negations across multiple columns are expanded using the
inclusion-exclusion principle. By default, the estimator returns an unknown
cardinality for tables with predicates that it cannot evaluate (e.g. predicates
on columns that are not modeled, or comparisons between two columns). Pass
`unsupported="ignore"` to drop such conjuncts instead.

## Storing Trained Models

Training a model for each table takes some time, so it is recommended to persist
the estimator once training is completed. The `store()` method writes a catalog
(_naru-catalog.json_) along with the column domains (Parquet files) and model
weights (PyTorch state dicts). `NaruEstimator.pre_trained()` re-creates the
estimator from such a catalog. The torch device is not persisted and can be chosen
when loading the model.

The generally recommended pattern is to use `NaruEstimator.load_or_build()`. It
checks, whether an estimator has already been persisted at the given location. If
it has, it is simply loaded. Otherwise, a new estimator is built (tailored to the
workload if one is given, otherwise for the entire schema) and stored.

## Example

```py
#
# Step 1: generic setup
#
# As usual for PostBOUND, connect to our target database and load the workload
import postbound as pb
import postbound_extensions as pbx

duck_instance = pb.duckdb.connect("stats.duckdb")
stats = pb.workloads.stats()

#
# Step 2: create the estimator
#
# We use our default hyperparameters (see NaruHyperParams.default() for details).
# Training is much faster on a GPU, e.g. device="cuda" or device="mps".
hyper_params = pbx.naru.NaruHyperParams.default().replace(device="mps")

# If no estimator has been stored yet, train one tailored to the workload.
# Otherwise, load the existing one. Omit the workload to model the entire schema.
naru = pbx.naru.NaruEstimator.load_or_build(
    "models/naru/stats",
    database=duck_instance,
    workload=stats,
    hyper_params=hyper_params,
    verbose=True,
)

#
# Step 3: Evaluation
#
# We can already use the estimator to calculate the cardinality of base tables:
query = stats["q-10"]
for table in query.tables():
    print(table, naru.calculate_estimate(query, table))

# Joins are left to the native optimizer
print(naru.calculate_estimate(query, query.tables()))  # unknown

# Or we can run the PostBOUND benchmarking utilities
pipeline = pb.MultiStageOptimizationPipeline(duck_instance).use(naru).build()
pb.bench.execute_workload(
    stats,
    on=pipeline,
    query_preparation={"prewarm": True},
    name="naru-test",
    progressive_output="naru-benchmark.parquet",
    logger="tqdm",
)
```

## Deviations from the Original Paper

One of the goals of the Naru implementation using PostBOUND is to enable benchmarks
across a wider range of datasets and workloads. While doing so, we noticed a number
of differences between what the paper describes and what the reference
implementation actually does. The following sections document these differences
as well as our own generalizations.

### Joins

The paper (Section 4.1) states that Naru can estimate joins by training a model
on the (materialized or sampled) result of a join. The reference implementation
only supports single tables. Applying the paper's approach to workloads like Stats
would require one model per join graph (the Stats workload alone contains 178 distinct
join graphs among its intermediates, with unfiltered join results of up to 10^12
tuples) and an efficient join sampler. The follow-up work (NeuroCard) is a different
system and therefore out of scope. We stick to what Naru actually demonstrated:
**our Naru estimates base tables only** and returns an unknown cardinality for every
intermediate with joins. Combining the per-table estimates with a join formula would
mix additional (non-Naru) assumptions into the results.

### Paper features that the reference implementation does not provide

The following features are described in the paper but are missing or restricted
in the reference implementation. We implement them as described in the paper:

- **Valid regions with multiple predicates per column.** The paper models each
  filter as a valid region _R_i_ of the column (Section 2.2), e.g. an interval
  _[l_i, r_i]_ or an `IN` list. The reference implementation only keeps a single
  predicate per column: `a >= 1 AND a <= 5` silently becomes `a <= 5`, since later
  predicates overwrite earlier ones. Only `=`, `<`, `<=`, `>`, `>=` are supported.
  This matters a lot for Stats, where most filters are two-sided ranges. We evaluate
  arbitrary single-column predicates on the column dictionary and intersect them.
- **Disjunctions** are supported via inclusion-exclusion in the paper (Section
  2.2), but not in the reference implementation. We implement inclusion-exclusion,
  including negations under SQL's three-valued logic.
- **Information-theoretic column orderings** (_MutInfo_ and _PMutInfo_, Section
  5.3) are not part of the reference implementation (only manually specified
  orderings are supported). We implement both. Like the paper's evaluation, we use
  the natural ordering by default.
- **Encoding strategy.** The paper (Section 4.2) describes one-hot encodings for
  small domains, embeddings of size _h = 64_ for large domains, and embedding reuse
  for the output layer. The reference implementation defaults to a binary input
  encoding, one-hot outputs, and an embedding size of 32. We follow the paper by
  default and support all encodings of the reference implementation.
- **Wildcard-skipping.** The paper masks each column of each training tuple with
  probability _w/n_, where _w ~ Unif[0, n)_ is drawn per tuple (Section 5.2). The
  reference implementation draws one masking probability per column and batch. We
  follow the paper.

### Bugs in the reference implementation that we fixed

- With direct input/output connections and the natural ordering, only a subset of
  the input units is wired to the output layer (the mask uses the slice
  `curr:dist_size` instead of `curr:curr + dist_size`).
- With direct input/output connections, the mask is always derived from the natural
  ordering. If the model is trained with a different (fixed) ordering, the direct
  connections violate the autoregressive property.
- Columns whose domain size equals the embedding size crash the embedding code
  (inconsistent `<` and `<=` checks).
- Progressive sampling with wildcard-skipping fails for queries with a single
  filter (it indexes the second sampled column unconditionally).
- Estimates are rounded up to the next integer (and stored as 32-bit integers).
  We return the unrounded estimate, i.e. Naru can also estimate empty results.
- If a query contains multiple predicates on the same column (e.g. a range
  `a > 1 AND a < 5`), all but the last predicate are silently dropped. We support
  arbitrary predicate trees (see the comparison below for the impact).

The learning rate is not specified in the paper. The reference implementation
creates its Adam optimizer with a learning rate of 2e-4, but overwrites it with
a constant 1e-2 in every step unless a warmup is specified. Our learning rate and
training budget were chosen based on the entropy gap of the trained models (i.e.
without looking at any queries), see `NaruHyperParams.default()`.

### Our generalizations

- **Large domains.** The paper dictionary-encodes all distinct values of a column,
  which is lossless. For columns with hundreds of thousands of distinct values
  (e.g. timestamps or keys), this leads to huge embedding matrices and output layers
  and makes training prohibitively slow. Therefore, columns with more than
  `max_domain_size` distinct values (default: 1024) are partitioned into equi-depth
  buckets. Since the exact frequency of each value is still known, each bucket
  receives a fractional weight during progressive sampling: the fraction of its
  rows that satisfies the predicate (instead of the 0/1 validity of the paper).
  The only additional assumption is that within a bucket, the predicate is independent
  of all other columns. Setting `max_domain_size=None` restores the paper's exact
  encoding.
- **Data types.** We support numeric, temporal, text and boolean columns. Text
  ranges use Python's string ordering, which might differ from the database
  collation. Columns with other data types are not modeled.
- **NULL values** are represented by a dedicated token and handled according to
  SQL's three-valued logic (e.g. `IS NULL`, `NOT (a = 1)`).
- **Free text columns** are not modeled in the schema-based setting (see above).
- **Deterministic estimates.** Progressive sampling is seeded, so the same query
  always receives the same estimate.

At the same time, we currently do not support the following:

- The Transformer architecture. The paper found MADE and ResMADE to be preferable
  due to their efficiency at the same accuracy (Section 6.7, Table 7).
- Training with multiple orderings (order-agnostic training), which the paper
  mentions as possible future work.
- Incremental model updates on data changes. Simply rebuild the estimator.

## Development Process

> [!NOTE]
> This section was written by us, not by the AI assistant.

The Naru estimator differs from the remaining papers in the playground in that it
was entirely generated by an AI assistant (Claude Opus 5.5). The assistant was
tasked with developing a PostBOUND implementation of the Naru estimator. It received
the original paper and the reference implementation as context and was instructed
to use a similar style like the MSCN and SafeBound estimators.
Further, we sensitized the assistant to look for discrepancies between claims in
the paper and what was actually implemented in the reference implementation. In
case of severe discrepancies, we instructed the assistant to escalate the issue.
This mechanism was used to decide on the handling of joins (only described in the
paper but never implemented). All other issues were resolved by the assistant itself.

Once the implementation was finished, we instructed the assistant to compare the
accuracy against the reference implementation as a means of validation. The sections
below summarize the results of the evaluation. They were again generated by the
assistant.

## Evaluation on Stats

We evaluated our implementation on the Stats database (DuckDB) using the default
hyperparameters. Training ran on an Apple M4 Pro (MPS) and took 20-75 seconds per
table (about 3 minutes for the workload-tailored and 5 minutes for the schema-based
estimator). Estimation takes 2-12 ms per table (median).

**Base tables of the Stats workload.** The 146 Stats queries contain 350 distinct
filtered base-table intermediates. Q-errors against the true cardinalities:

| Estimator | Median | 95th | 99th | Max |
| --------- | ------ | ---- | ---- | --- |
| DuckDB (native) | 4.91 | 38.8 | 85.7 | 156.9 |
| Naru, tailored to the workload (`infer_from`) | 1.00 | 1.18 | 1.31 | 1.34 |
| Naru, entire schema (`online`) | 1.00 | 1.12 | 1.50 | 1.77 |
| Naru, workload + _MutInfo_ ordering | 1.00 | 1.11 | 1.18 | 1.39 |
| Naru, workload without wildcard-skipping | 1.00 | 1.06 | 1.91 | 1.95 |

Note that most base-table filters of the Stats workload are not very selective,
which makes this setting rather easy for Naru (and DuckDB often falls back to a
fixed selectivity of 20%).

**Paper-style queries.** To stress the models, we also generated 1,200 queries over
the Stats tables following Section 6.1.3 of the paper (5-11 filters per query,
literals drawn from a random tuple), see _localtest-naru-synthetic.py_. Using the
schema-based estimator:

| Estimator | Selectivity | Queries | Median | 95th | 99th | Max |
| --------- | ----------- | ------- | ------ | ---- | ---- | --- |
| DuckDB | > 2% | 111 | 2.75 | 28.3 | 121 | 184 |
| Naru | > 2% | 111 | 1.04 | 1.16 | 1.24 | 1.32 |
| DuckDB | 0.5% - 2% | 32 | 12.9 | 42.6 | 65.2 | 74.0 |
| Naru | 0.5% - 2% | 32 | 1.07 | 1.20 | 1.26 | 1.26 |
| DuckDB | <= 0.5% | 1057 | 2.00 | 8065 | 34861 | 65612 |
| Naru | <= 0.5% | 1057 | 1.00 | 7.50 | 53.3 | 303 |

The remaining tail errors are mostly caused by queries that select a single tuple
via (near-)unique key columns, such as `Id` or `AccountId`. The schema-based model
includes these columns (bucketized), and the paper's datasets did not contain
such columns. Finer buckets (`max_domain_size=8192`) did not help at the same
training budget; they made the tail worse.

**End-to-end.** Injecting Naru's base-table estimates into DuckDB (all join
estimates remain DuckDB's own) reduced the total runtime of the Stats workload from
152 s to 75 s in a single run with a 120 s timeout. 23 queries became more than
10% faster (e.g. q-107: 18.6 s to 1.6 s) and 4 queries became slower (at most
1.3 s to 1.8 s). See _localtest-naru-e2e.py_.

**Other workloads.** As a smoke test for different schemas and predicate types,
we also trained workload-tailored models for 12 IMDB tables of the Join Order
Benchmark. This exercises `LIKE`, `IN`, `OR`, `BETWEEN`, and `IS NULL` on text columns
with millions of distinct values. We used a deliberately small training budget
(50,000 sampled rows, 300 steps per table, CPU). All 152 distinct base-table filters
could be estimated, with a median q-error of 1.00, a 90th percentile of 1.24, and
a maximum of 7.0.

## Comparison with the Reference Implementation

We compared our implementation against the reference implementation on the
base-table filters of Stats (352 distinct filters) and JOB-light (46 distinct
filters on IMDB), see _localtest-naru-reference.py_. Both implementations modeled
the same columns (all filtered columns of the workload), used the same architecture
(ResMADE, 5 x 256 units, direct I/O connections, wildcard-skipping), the same
training budget (20 epochs, at least 6,000 steps, batch size 1,024), and 2,000
progressive samples. Otherwise, the reference ran with the settings of its README:
binary input / one-hot output encoding, exact domains, and its warmup schedule.
We loaded the reference code unmodified, except for patching the single-filter crash
listed above (in memory). For JOB-light, both implementations were trained on the
same 300,000-row random sample per table: the reference needs one output unit per
distinct value, and with larger samples, training on `company_id` and `keyword_id`
alone took several hours. Truth is the exact `COUNT(*)` on the full tables.

| Workload | Estimator | Median | 90th | 95th | 99th | Max | Training | Parameters |
| -------- | --------- | ------ | ---- | ---- | ---- | --- | -------- | ---------- |
| Stats | ours | 1.00 | 1.07 | 1.18 | 1.31 | 1.34 | 3 min | 5.1 M |
| Stats | reference | 1.01 | 1.08 | 1.13 | 2.01 | 2.08 | 87 min | 170 M |
| JOB-light | ours | 1.04 | 1.12 | 1.16 | 5.36 | 8.70 | 2 min | 3.4 M |
| JOB-light | reference | 1.09 | 4.98 | 7.20 | 9.20 | 10.5 | 14 min | 37 M |

On filters with at most one predicate per column, both implementations are about
equally accurate (Stats: max 1.32 vs. 2.08; JOB-light: 95th percentile 1.22 vs.
2.52, max 8.7 vs. 5.0). The large reference errors on JOB-light come from range
predicates on `production_year`, where the reference only keeps the last bound
(e.g. `> 1980 AND < 1984` becomes `< 1984`, q-error 10.5 vs. 1.08 for ours). On
Stats, dropping a bound changes the true cardinality for 147 of 179 such filters,
but the dropped bounds rarely restrict much (at most by 1.29x), so the effect is
small there. The reference's tail on Stats comes from `votes.BountyAmount` (99.5%
NULL), which it overestimates by about 2x, compared to 1.2-1.3x for ours.

The remaining differences:

- Our domains are computed on the full table even if the model is trained on a
  sample, so values that are missing from the sample still have known frequencies.
  The reference only knows values that are in the training data (e.g.
  `keyword_id = 8200` with 5 matching rows: estimate 0 vs. 5 for ours).
- Our largest error is `mi_idx.info_type_id = 113` (10 of 2.4 M rows; estimate 87,
  reference 35). Such rare values only appear a handful of times in a 300K sample,
  so both models are limited by their training data.
- Training time and model size mainly come from the reference's one-hot output
  over exact domains (e.g. 54 M parameters and 23 min for `postHistory`, which has
  190K distinct timestamps). Ours caps each column at 1,024 equi-depth buckets.
- Estimation latency is similar on JOB-light (about 2 ms median). On Stats, the
  reference takes 77 ms vs. 3 ms (median), mostly because of its large output
  layers.

## Authors and License

This implementation was generated by Claude Opus 5.5 (Anthropic), supervised by
Rico Bergmann (see [Development Process](#development-process)). This includes the
PostBOUND adapter, the predicate translation, the column domains, the column
orderings, and progressive sampling (the latter is based on the reference
implementation).
The MADE/ResMADE model has been re-used from the reference implementation by the
Naru authors, which is licensed under the Apache License 2.0. We license the
complete Naru implementation under the same terms (see _LICENSE_) and make it
available as part of the [Optimizer Collection project](https://github.com/Optimizer-Playground/Optimization-Techniques).
The project as a whole is licensed under GPL-3.0 terms.

## References

Zongheng Yang et al.: _Deep Unsupervised Cardinality Estimation_ (PVLDB 13(3),
2019); DOI [10.14778/3368289.3368294](https://doi.org/10.14778/3368289.3368294)

Reference implementation: <https://github.com/naru-project/naru>
