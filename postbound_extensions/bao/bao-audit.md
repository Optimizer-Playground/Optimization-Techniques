# Audit Prompt

> Please perform an audit of our implementation of the BAO learned optimizer.
>
> The optimizer is implemented in the `postbound_extensions/bao/` directory. Its README provides a high-level description and outlines important non-goals of the implementation, including features that have deliberately not been implemented. **Those should not be part of the audit.**
> In the directory, you will also find the original paper that described BAO (`postbound_extensions/bao/BAO-Paper.pdf`) and the original reference implementation that was published alongside the paper (`postbound_extensions/bao/reference-implementation/`).
>
> The audit should focus on the logical correctness of:
>
> - the plan featurization
> - the network architecture
> - any transformation or scaling that is applied to both input or output
> - the retraining setup
> - (optional if other results do not make sense) the generation of the training data
> - any other aspects that you consider important
>
> Focus the audit on larger deviations from the original paper (or the reference implementation), as well as subtle inconsistencies that turn out to have a large impact. Minor discrepancies should not be analyzed in detail. **Pay special attention to issues that might lead to different hint sets being predicted.**
>
> Should the original paper and the reference implementation disagree on certain parts of the algorithm, describe the discrepancy and its impact on our implementation.
>
> Use the IMDB database and JOB workload as discussed in the reference implementation. You can access them via PostBOUND (`pb.workloads.job()`, `pb.duckdb.connect(...)` on `imdb.duckdb`, `pb.postgres.connect(...)` with `.psycopg_connection_imdb`). In general, the DuckDB version should be preferred due to its superior computation times.
>
> It is very important for the audit to be thorough. So take your time and investigate any potential problems deeply.

The Postgres connection file is named `.psycopg_connection_job` in this checkout (there is no `.psycopg_connection_imdb`). All measurements had to run on Postgres: Bao's hint sets are Postgres planner switches, and the featurizer's operator vocabulary and cache features are Postgres-specific, so DuckDB could not be used for any part of this audit.

# Claude Response

As shipped, our Bao implementation cannot learn anything on IMDB.

- **`BaoFeaturizer.online()` crashes** on IMDB (issue 1). With that crash worked around, its normalization bounds are `log(0) = −inf`, so every cardinality and cost feature is `NaN`.
- **`BaoFeaturizer.infer_from_sample()` has the same `min_card = 0` problem.** It also takes runtime bounds in seconds with a minimum of 0, so all training labels become `NaN`.
- **Both factories therefore produce a model whose loss is `NaN` from the first epoch** and whose predictions are all `NaN`. `torch.argmin` then returns index 0, so Bao silently always selects arm 0, the unmodified Postgres plan.

The default construction path (`BaoOptimizer(db)`, `load_or_build`) uses `online()`.

With the normalization repaired, the model trains, and the network itself matches the reference exactly. Four further problems change which hint set gets selected:

- The cache feature means different things at training time and at inference time (issue 4).
- Arm predictions depend on which other arms are scored in the same batch (issue 5).
- Timed-out executions never reach the experience (issue 10).
- A fresh optimizer picks arms with an untrained, randomly initialized network instead of falling back to Postgres (issue 8).

With these fixed (normalization, cache feature, timeouts, label scaling), our model ranks arms as well as the reference does on the same data (pairwise accuracy 0.795 vs 0.789).

In a replay of the online loop over 500 JOB queries:

- the shipped loop (`NaN` fixed only) is 15% slower than plain Postgres;
- the fully fixed loop is 9% faster than Postgres after warm-up, where the reference is 4% faster.

The retraining and persistence machinery has three independent defects (issue 11): `load_or_build` never finds the model that `store` wrote, a reloaded optimizer never retrains, and stored experience cannot be read back. The training-data tooling (issue 12) produces runtimes in seconds labelled as milliseconds, only default-arm plans, and featurizers that crash on JOB.

I didn't modify the repository. The probe scripts, the recorded measurements, the CV predictions and the simulation logs were kept in a temporary directory and have not been preserved.

## Results on JOB (113 queries × the 5 default arms, Postgres 18)

The workload is fixed: arm 0 (Postgres default) takes 196.0 s, and a per-query oracle takes 153.4 s. The best single non-default arm is arm 4 (no merge join), which takes 199.0 s.

**Offline, query-disjoint 5-fold cross-validation.** Each model trains on all 5 arms of ~90 queries and selects arms for the other ~23. I used 8 seeds per row and identical folds for every row.

| configuration                                                   | pairwise ranking accuracy¹ | selected total ÷ Postgres (mean ± sd) | arm differs batched vs. one-by-one² | arm agreement: with itself across seeds / with reference |
| --------------------------------------------------------------- | -------------------------- | ------------------------------------- | ----------------------------------- | -------------------------------------------------------- |
| **as shipped** (either featurizer factory)                      | – (loss `NaN`)             | **1.00 (always arm 0)**               | –                                   | –                                                        |
| V1: only the `NaN` normalization fixed (issue 2)                | 0.739                      | 1.07 ± 0.04                           | 11.5 / 113                          | 0.64 / 0.42                                              |
| V1 + timeouts kept as 2 × timeout (issue 10)                    | 0.745                      | 1.00 ± 0.05                           | 15.8                                | 0.61 / 0.41                                              |
| V2: V1 + plan-time cache feature in training (issue 4)          | 0.777                      | 1.10 ± 0.09                           | 18.4                                | 0.63 / 0.45                                              |
| V3: V2 + timeouts kept                                          | 0.794                      | 1.02 ± 0.07                           | 17.8                                | 0.62 / 0.46                                              |
| V4: V3 + reference label scaling (issue 6)                      | **0.795**                  | 1.01 ± 0.06                           | 9.1                                 | 0.62 / 0.47                                              |
| reference implementation (same data, same folds)                | **0.789**                  | 1.04 ± 0.05                           | **0**                               | 0.59 / –                                                 |

¹ Share of correctly ordered arm pairs within a test query, counting only pairs whose runtimes differ by more than 1.2×. This is much less noisy than end-to-end totals.
² Out of 113 queries: how often `argmin` changes when the five arms are scored one at a time instead of as one batch (`select_physical_operators` scores them as a batch).

End-to-end totals are within seed noise for every configuration, the reference included. With roughly 450 training plans, no model reliably beats Postgres offline. The ranking accuracy is the robust signal, and it moves only with issues 4 and 10.

Different seeds of the *same* model agree on only about 60% of the selected arms. Our model agrees with the reference on 42–47%, which is below that noise floor, but the gap shrinks with each fix.

**Online loop simulation.** This replays the actual feedback loop on the recorded outcomes. The sequence has 500 queries drawn from JOB with `random.Random(42).choices`, as in the reference's `run_queries.py`. Only the selected arm's outcome enters the experience. 3 seeds per row:

On this sequence, Postgres alone takes 189 s for queries 1–100 and 810 s for queries 101–500 (999 s in total). The per-query oracle takes 154 s / 630 s.

| policy                                                                                                                  | q1–100 ÷ PG | q101–500 ÷ PG | total ÷ PG      | arm histogram q1–100 (arms 0–4) |
| ----------------------------------------------------------------------------------------------------------------------- | ----------- | ------------- | --------------- | ------------------------------- |
| **as shipped**                                                                                                          | 1.00        | 1.00          | **1.00** (arm 0 only) | [100, 0, 0, 0, 0]         |
| shipped loop, only the `NaN` fix (random-init model first, n = 100, analyzed-plan cache feature, timeouts dropped)      | 1.32 ± 0.19 | 1.11 ± 0.13   | **1.15 ± 0.07** | [20, 21, 15, 28, 15]            |
| same, but retraining every 25 queries                                                                                   | 1.32 ± 0.17 | 0.99 ± 0.01   | 1.06 ± 0.03     | [26, 11, 15, 14, 32]            |
| all fixes (Postgres until the first model, plan-time cache, timeouts kept, [0, 1] labels), n = 25                       | 1.35 ± 0.10 | **0.91 ± 0.02** | 0.99 ± 0.01   | [46, 6, 13, 6, 28]              |
| same + bootstrap (paper's Thompson sampling)                                                                            | 1.19 ± 0.08 | 1.00 ± 0.06   | 1.03 ± 0.06     | [45, 3, 18, 6, 27]              |
| reference, n = 25 (as in `run_queries.py`)                                                                              | 1.21 ± 0.16 | 0.96 ± 0.02   | 1.01 ± 0.05     | [54, 6, 14, 5, 20]              |
| reference + bootstrap                                                                                                   | 1.38 ± 0.19 | 0.96 ± 0.02   | 1.04 ± 0.02     | [46, 5, 18, 7, 22]              |

What the simulation shows:

- **The shipped retraining setup is the worst policy, even with the `NaN` bug fixed:** 15% slower than Postgres overall and 11% slower after warm-up.
  - During the first 100 queries the random network spreads its picks evenly over all arms (issue 8).
  - After that, the train/inference cache mismatch (issue 4) and the missing timeouts (issue 10) keep it behind the reference.
- **With the fixes, our loop beats the reference after warm-up** (0.91 vs 0.96 × Postgres). It is the only policy that clearly beats Postgres on queries 101–500.
- **Bootstrapping (the paper's Thompson sampling) does not help over this short horizon**, for either implementation.
- **All policies lose time in queries 1–100.** Models trained on 25–75 samples are worse than Postgres; this is inherent to Bao's design, not specific to our code.

## How the evidence was produced

- **Measurements:** I ran all 113 JOB queries under each of the 5 default arms on the Postgres 18 IMDB instance (4.7 M titles, `shared_buffers` 128 MB, warm OS cache). For every (query, arm) I recorded:
  - the hinted query, as PostBOUND/pg_lab generates it for Bao;
  - the `pg_buffercache` state at planning time;
  - the plan-time `EXPLAIN (FORMAT JSON)`;
  - a warm-up and a measured `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`, with a 30 s timeout.

  2 of 565 runs timed out (25c and 31a, both arm 2). Run-to-run noise is small: the measured/warm-up ratio has a median of 0.99 and an IQR of 0.97–1.02. I also re-ran every (query, arm) without instrumentation (issue 13).
- **Hint sets:** I parsed all 26 arms of `pg_extension/bao_planner.h:set_arm_options` and compared them with `default_hint_sets()`. They are identical, and the first 5 are the reference's default `bao_num_arms = 5`. The selected plans are reproduced when PostBOUND re-hints them for execution in 563/565 cases. The other 2 are pg_lab failures, not Bao bugs.
- **Ours:** I trained through the shipped `BaoOptimizer._train` loop and `BaoModel`, and featurized through the shipped `BaoFeaturizer.encode_plan` and `DatabaseCacheState.determine_cache_pct`. `DatabaseCacheState` was fed the buffer state recorded at planning time instead of the live one; that is the only change.
  - Because `online()` crashes (issue 1), I replicated it with a float `max_card`.
  - "V1" sets `min_card = min_cost = 1`; everything else is as shipped.
  - Fixes for the other issues were applied in the harness, not in the repository.
- **Reference:** I used the reference's own `bao_server/model.py` (`BaoRegression`), `featurize.py` and `TreeConvolution`, with `have_cache_data=True`, unmodified. The plan JSON was converted to exactly what `pg_extension/bao_planner.h:emit_json` sends: node type, relation name for all scans, cost, rows, and only `lefttree`/`righttree`. The buffer map is the `pg_buffercache` relation→block count at planning time, as `bao_bufferstate.h` produces it. Timeouts are recorded as 2 × timeout, as in `reg_blocker.py`.
- **Not done:** I did not run the real online loop against Postgres; the simulation replays the recorded outcomes. I did not run the triggered-exploration/regression blocker, which the README does not cover. DuckDB was not usable (see above).

## Where the paper and the reference disagree

| topic | paper | reference | ours |
| ----- | ----- | --------- | ---- |
| Thompson sampling | train on a bootstrap of the experience (Sec. 3.1.2), so each retrain samples θ | trains on the full experience, no resampling (`train.py`) | no resampling, i.e. the reference (issue 9) |
| window / retrain frequency | k = 2000, n = 100 (Sec. 6.1) | experience unbounded; `run_queries.py` retrains every 25 queries | k = 2000, n = 100 (paper) |
| before the first model | "can start out using only the traditional optimizer" | Postgres plan (arm 0) until a model exists; the first 25 queries are pure Postgres | the untrained random network chooses (issue 8) |
| binarization | full plan, `null` child under unary nodes, with a `null` one-hot (Fig. 3, 4) | strips to joins and scans; unary nodes are skipped | full plan (paper), but `null` children are all-zero vectors instead of the `null` one-hot (issue 3) |
| cache feature | per scan, the percentage of the file in cache | log/min-max of the *absolute* number of buffered blocks of the *heap relation* (no index), per leaf | percentage, the average of the index and heap fraction at planning time; but see issue 4 for training |
| activation | ReLU | LeakyReLU | LeakyReLU (reference) |
| stopping | 100 epochs or < 1% improvement over 10 epochs | 100 epochs or its own rule (`loss < 0.1` and no improvement over the 10th-last epoch) | always all epochs |
| label transform | not specified | `log1p`, then MinMax to [0, 1] fitted on the experience at each retrain | `log1p`, then MinMax onto [log min, log max] of a fixed range (issue 6) |
| feature normalization | not specified | log(x+1) min-max, refit on the experience at each retrain | fixed bounds from the featurizer (issue 2) |
| hint sets | 48 | 26 defined, 5 used by default | the same 26, 5 used by default (docstring says 26) |
| reward | execution time | executor time without instrumentation (`queryDesc->totaltime`) | `EXPLAIN ANALYZE` time or benchmark wall time (issue 13) |
| timeouts | not discussed | exploration records 2 × timeout | dropped (issue 10) |

Impact of the paper/reference disagreements:

- **Bootstrap:** this is the paper's only exploration mechanism. Without it, Bao is a greedy policy and explores only through weight initialization and shuffling.
- **Cache-feature definition:** a percentage vs an absolute block count is a legitimate choice. It does not explain any of the measured differences once issue 4 is fixed.
- **Binarization:** following the paper here is a documented deviation and fine by itself. It is, however, the root cause of issue 5, which the reference avoids by construction.

## A. Plan featurization and input scaling

**1. `BaoFeaturizer.online()` crashes on IMDB — critical.**

- **Where:** `_featurizer.py:321` computes `max_card = int(np.prod(largest_cards))`. For IMDB this is cast_info × movie_info × movie_keyword = 1.19·10²², which exceeds int64.
- **What happens:** `np.log` (`_featurizer.py:463`) receives a Python `int` and raises `TypeError: loop of ufunc does not support argument 0 of type int`.
- **Scope:** `online()` is the default featurizer of `BaoOptimizer.__init__`, `load_or_build` and `BaoFeaturizer.load_or_build`, so none of them can construct an optimizer on IMDB. Stats does not hit this because its tables are small.

**2. Both featurizer factories normalize with `log(0)`, so every cardinality and cost feature is `NaN` — critical.**

- **Where:** `online()` passes `min_card=0, min_cost=0` (`_featurizer.py:331,333`), and `__init__` takes `np.log` of them (`:462,464`), giving `−inf`. `encode_cardinality` then computes `(log(c+1) + inf) / (log(max) + inf)`, which is `inf/inf = NaN`. The same holds for cost.
- **`infer_from_sample` hits the same problem:** on our 563 analyzed JOB plans it derives `min_card = 0` (`_featurizer.py:380`). It also reads `node.execution_time`, which PostBOUND reports in **seconds**, as `min_runtime_ms`/`max_runtime_ms` (`:384–385`); here 0.0 and 23.5. With `log(0)` as the lower scaler bound, `transform_runtime` returns `NaN` for every label.
- **Evidence:**
  - On 1a, all node encodings carry `NaN` in the card/cost slots, and the five arm predictions are `[nan, nan, nan, nan, nan]`.
  - Training through `_train` logs `loss = nan` from epoch 1.
  - `torch.argmin` over `NaN` returns 0, so `select_physical_operators` and `optimize_query` always return arm 0. Nothing fails visibly: Bao behaves exactly like Postgres.
- **Fix:** use `log1p` consistently for the bounds and the values (the reference uses min/max of `log(x+1)`), or floor the minima at 1. The reference also refits these bounds on each retrain. With fixed bounds and the online maxima (1.2·10²² rows, 8.9·10¹⁹ cost), JOB's features occupy only [0, ~0.4]. That is usable but compresses the range.

**3. Dummy (`null`) children are zero vectors, not the `null` one-hot (`_featurizer.py:588–590`) — minor.**

- **What happens:** the `__bao_null__` category is added to the encoder but never emitted, because `_featurize_qep` short-circuits dummies to `torch.zeros`. Fig. 4 of the paper sets the `null` bit.
- **Impact:** the network can still distinguish unary nodes by their non-zero left child, so the effect is small. It is listed only because the README claims to follow the paper here.

**4. The cache feature has different semantics in training and at inference — high impact on ranking.**

- **Where:** `DatabaseCacheState.determine_cache_pct` (`_featurizer.py:198–203`) uses the plan's measured `cache_hits / (hits + misses)` whenever the plan carries buffer counts. Otherwise it uses the buffer-pool fraction of the relation and index at planning time.
- **What happens:**
  - Every experience sample comes from an executed `EXPLAIN ANALYZE` plan: `calibrate`, `bench_feedback`, and `_retrain_offline` with the sampler output. Postgres 18 includes `BUFFERS` by default, so the model is trained on **hit ratios observed during execution**.
  - Candidate plans at selection time are plain `EXPLAIN` plans, so the model is queried with the **buffer-pool fraction** before execution.
  - The paper and the reference use the plan-time cache state for both.
- **Measured on identical scan nodes (4 865 pairs, all JOB × arm plans):**

  | scan operator    | training value (hit ratio) | inference value (buffer fraction) |
  | ---------------- | -------------------------- | --------------------------------- |
  | Index Scan       | 0.86                       | 0.16                              |
  | Bitmap Heap Scan | 0.86                       | 0.10                              |
  | Index Only Scan  | 0.96                       | 0.26                              |
  | Seq Scan         | 0.43                       | 0.38                              |

  The median is 0.84 in training vs 0.04 at inference, the mean absolute difference is 0.35, and the correlation is 0.41.
- **Why this matters for arm selection:** index and bitmap scans are what distinguishes the nested-loop arms. The model therefore systematically sees index-heavy candidate plans with cache values it has never been trained on.
- **Evidence:** training on the plan-time feature instead raises pairwise accuracy from 0.739 to 0.777, the largest single improvement in the CV.
- **Fix:** featurize experience with the cache state captured when the plan was chosen. Store it alongside the plan, as the reference stores `Buffers` with each experience row. Don't use the execution's hit counters.

**5. Arm predictions depend on the batch, because our trees differ in size per arm — changes the selected arm for 4–20% of queries.**

- **Mechanism:** `TreeLayerNorm` normalizes over all positions, padding included. Padding positions become `bias` after each `BinaryTreeConv`, and `DynamicPooling` takes the max over them.
  - Scoring all arms of a query in one batch (`_bao.py:685, 699`) therefore pads every tree to the largest arm's size, and the smaller trees' predictions change.
  - This is inherited from TreeConvolution and present in the reference too. There it is harmless: its trees contain only joins and scans, so all arms of a query have the same number of nodes (0 of 113 queries differ).
  - Ours keep Hash, Sort, Materialize, Memoize, Gather, Aggregate and dummy nodes. All arms differ in size for 110 of 113 queries, by a median of 12 and up to 32 nodes.
- **Evidence:** `argmin` changes between batched and one-by-one scoring for 5–23 queries per trained model (mean 9–18 per configuration, over 40 models), and never for the reference (0 in all 8). It also slightly lowers ranking accuracy (0.795 batched vs 0.786 one-by-one for V4).
- **Fix:** score each candidate plan individually or mask the padding in `TreeLayerNorm`/`DynamicPooling`. Note that training batches mix sizes as well.

## B. Output (label) scaling

**6. The runtime scaler maps to [log min, log max] instead of [0, 1] (`_featurizer.py:470–474`) — moderate.**

- **What happens:** `MinMaxScaler((self._min_runtime_log, self._max_runtime_log))` uses the log bounds as the *output* range. With `online()` (1 ms … 1 h), labels land in [0, 15.1], and the transform is just `log1p` up to a tiny affine distortion. Labels are also not fitted to the experience: the reference fits MinMax(0, 1) on the experience labels at every retrain.
- **Impact:** argmin selection is unaffected by the monotone transform itself. Training dynamics do change: switching to the reference scaling roughly halves the batch-dependence of issue 5 (17.8 → 9.1 flipped queries) at equal ranking accuracy.

**7. Milliseconds vs seconds.** `infer_from_sample` reads seconds (issue 2), and the sampler writes seconds into `runtime_ms` (issue 12). `add_experience` and `bench_feedback` correctly convert to ms. Mixing sampler-calibrated experience with online feedback therefore mixes units by a factor of 1000, inside the same `log1p` label space.

## C. Network architecture

The model is the reference's `BaoNet`, layer for layer:

- three `BinaryTreeConv` layers (256 → 128 → 64), with `TreeLayerNorm` and LeakyReLU in between;
- `DynamicPooling`;
- Linear 64 → 32, LeakyReLU, Linear 32 → 1.

The TreeConvolution utilities (`_util.py`, `_tcnn.py`) are a faithful port. Training is the same as the reference: Adam with default learning rate, batch size 16, MSE, 100 epochs, and a fresh network on every retrain.

- **LeakyReLU vs ReLU:** the paper says ReLU; we follow the reference. This is negligible.
- **No early stopping:** both the paper and the reference stop early. Without it, 100 epochs on a small experience overfits more, but the effect is within seed noise in the CV.

## D. Retraining setup (the RL loop)

**8. Cold start: a fresh optimizer chooses arms with a randomly initialized network — high impact on early hint sets.**

- **Where:** `optimize_query` / `select_physical_operators` always use `self._tcnn`, which `__init__` creates with random weights. There is no fallback to Postgres before the first training.
- **Reference and paper:** the reference returns arm 0 until a model has been trained (`main.py:select_plan`), and `run_queries.py` runs the first 25 queries with Postgres. The paper describes starting with the traditional optimizer.
- **Impact:** with the default `retraining_frequency = 100`, the first 100 queries are optimized by a random network. In the simulation it picks arms almost uniformly ([20, 21, 15, 28, 15] for arms 0–4), including arm 2, which alone is 2.3× slower than Postgres over JOB. Queries 1–100 then take 1.32 × Postgres.

**9. No Thompson sampling: training never bootstraps the experience.**

- **What's missing:** `_train` uses every sample exactly once per epoch. The paper's exploration mechanism is training on |E| samples drawn with replacement.
- **Context:** the reference omits this too, and the README does not list it as a non-goal or deviation. Since the README states "in case of contradictions we follow the paper", this is an undocumented choice for the reference.
- **Simulated effect:** over 500 queries, bootstrapping made no measurable difference for the reference (0.96 × Postgres after warm-up, with or without). It was worse for ours (1.00 vs 0.91), so adding it is not a quick win. It is still the paper's algorithm, and its benefit is expected over longer horizons with workload drift.

**10. Timeouts and failed executions never enter the experience; `calibrate`'s timeout is 10⁶× too long.**

- **Where:**
  - `calibrate` skips executions whose result is `None` (`_bao.py:830`), and `bench_feedback` returns early for any `status != "ok"` (`:859`).
  - `calibrate` passes `timeout=1000 * timeout_ms` (`:818`), while `PostgresInterface.execute_with_timeout` takes seconds. Passing `timeout_ms=30_000` therefore yields a 3·10⁷ s timeout.
- **Why it matters:** the most expensive plans, which are exactly the ones Bao must learn to avoid, are never shown to the model. The reference records 2 × timeout for them (`reg_blocker.py`).
- **Evidence:** keeping the 2 JOB timeouts as 2 × 30 s improves pairwise accuracy (0.777 → 0.794 with plan-time cache features). It also reduces the end-to-end total from 1.07 to 1.00 × Postgres with the shipped cache feature. With a tighter timeout and more aggressive arms, more samples would be lost.

**11. Persistence breaks the loop in three ways.**

- **`load_or_build` never finds the model `store` wrote.**
  - `store(archive_dir)` writes `catalog-<db>.json` (`_bao.py:889`), but `load_or_build` and `pre_trained` look for `catalog.json` (`:555, 614`).
  - With the README's `"models/bao/catalog.json"`, `store` even creates a *directory* called `catalog.json`. Every run then recalibrates from scratch and overwrites the stored model.
  - Reproduced: after `store`, `load_or_build` looked for `persist/catalog.json/catalog.json`, which does not exist.
- **`pre_trained` hard-codes `retrain=False` (`_bao.py:576`).** `load_or_build(..., retrain=True)` therefore returns a frozen optimizer whenever it does find a model. The RL loop then silently stops after a restart.
- **Stored experience cannot be reloaded.** `BaoExperience.load` uses `pb.opt.read_query_plan_json` (`_experience.py:105`), which raises `ValueError: Malformed query plan. Expected physical operator but got: None` for any real Postgres plan with Aggregate/Hash/Gather nodes. As soon as the experience is non-empty, `pre_trained` crashes.

Otherwise the loop matches the paper:

- a bounded deque window (k = 2000);
- a counter-based retrain trigger every n = 100 new samples;
- a fresh network on each retrain;
- `calibrate` keeps the current weights unless `from_scratch=True`.

## E. Training-data generation

**12. The offline calibration data does not fit Bao.**

- **Units:** `tools/query-sampler.py` (via `sampler/_cards.py:124`) writes `plan.root.execution_time` into a column named `runtime_ms`, but the value is in seconds.
- **Column name:** `calibrate` defaults to `plan_column="query_plan"`, while the sampler writes `plan`.
- **Only default-arm plans:** every sampled query runs with the default Postgres optimizer. The model never sees a plan from arms 1–4 and therefore cannot learn to rank them. Bao's experience is supposed to contain the plans chosen under different hint sets.
- **Unknown operators:** the sampler converts queries to `SELECT *` (`as_star_query`). Plans of the original JOB queries contain `Aggregate` and `Index Only Scan` nodes, which never occur in the star versions. A featurizer from `infer_from_sample` on sampler output raises `ValueError: Found unknown categories ['Aggregate']` on the first JOB plan, because the encoder has no `handle_unknown`. Other arms add further operators, e.g. Merge Join and Materialize.
- **Fix:** calibrate by executing the target-style queries under all hint sets, which `calibrate(workload)` does once issue 10 is fixed.

**13. Labels are `EXPLAIN ANALYZE` times — minor on JOB.**

- **What differs:** the reference's reward is the executor time without per-node instrumentation. `calibrate` uses `plan.execution_time` of `EXPLAIN ANALYZE` runs, and `bench_feedback` uses the benchmark's wall time of the analyzed query.
- **Measured:** instrumentation inflates runtimes by a median of 12–22% per arm (p90 1.45–1.67×, max 2.9×).
- **Effect on arm choice:** the fastest arm changes in 41/111 queries. That equals the noise floor between two `EXPLAIN ANALYZE` repetitions (42), and the penalty is a median of 1.04×. On JOB this does not change hint selection beyond noise, but it is a systematic label bias that grows with nested-loop-heavy plans.

## F. Other findings

- **Hint sets:** `default_hint_sets()` is identical to all 26 reference arms, and the default of the first 5 matches the reference's `bao_num_arms = 5`. The constructor docstring says "the same 26 hint sets" (`_bao.py:484`), but the code uses `[:5]`.
- **Plan reproduction:** `optimize_query` returns the arm's plan, which PostBOUND re-hints in pg_lab's full plan mode. This reproduces the identical plan for 563/565 (query, arm) pairs. pg_lab rejects the other 2 (27a arm 0, 33a arm 4) with `DatabaseServerError`; `select_physical_operators` is unaffected.
- **PostBOUND estimates under `Gather`:** these are multiplied by `workers + 1`, including non-parallel inner index scans. This differs from the raw `Plan Rows` the reference sees, but it is consistent between training and inference.
- **`estimate_cost` (CostModel):** returns the network output in label space, not in ms. That is fine for ranking, misleading as a cost.
- **`_retrain_offline` indexes with `plans[i]`/`runtimes[i]`:** this is label-based and raises `KeyError` on DataFrames without a 0…n−1 index.
- **`determine_cache_pct`:** divides by `relpages`, which is 0 for never-analyzed relations (`ZeroDivisionError`). `buffered / relpages` can also exceed 1 with stale statistics.

# Suggested fix order

- [x] Issues 1 and 2: compute bounds in float and with `log1p` (or floor at 1); fix runtime units in `infer_from_sample`. Consider refitting bounds per retrain as in the reference.
- [x] Issue 4: featurize experience with the plan-time cache state stored with each sample, not with the execution's hit counters.
- [x] Issue 10: record timeouts and failures as a penalty (e.g. 2 × timeout); fix the `calibrate` timeout unit.
- [x] Issue 8: return the default arm until a model has been trained.
- [x] Issue 5: score candidate plans individually (or mask padding in `TreeLayerNorm`/`DynamicPooling`).
- [x] Issue 11: align the catalog file name between `store` and `load`, pass `retrain` through `pre_trained`
- [x] Issue 11(2): experience plans round-trip.
- [x] Issue 6: fit the label scaler to [0, 1] on the experience.
- [x] Issue 9: decide on bootstrap (paper) vs. none (reference), and document it in the README.
- [x] Issue 12: generate calibration data with the target query shape under all hint sets, in milliseconds.
- [x] Minor: the `null` one-hot (3), the docstring, `relpages = 0`.
- [x] Minor: early stopping, using the paper's rule (less than 1% loss decrease over 10 epochs, at most 100 epochs).

## Not addressed

- Issue 13 (`EXPLAIN ANALYZE` labels): accepted limitation. PostBOUND cannot measure runtimes without `EXPLAIN ANALYZE`, and all queries go through it.
