# Audit Prompt

> Please perform an audit of our implementation of the MSCN cardinality estimator (`postbound_extensions/mscn/`), comparing it against the original paper (`MSCN-Paper.pdf`) and the reference implementation (`reference-implementation/`). Non-goals listed in the README (most notably bitmap samples) are excluded.
> Focus: query featurization, network architecture, input/output transformations and scaling, (optionally) training-data generation, and anything else important. Use IMDB and JOB-light.

# Claude Response

The network itself, its training loop and the label scaling match the reference implementation. Their validation q-errors are the same. Two problems account for almost all of the JOB-light gap:

- **The learning rate is 10× too small** (issue 1).
- **The featurizer merges predicates on the same column into one set element** (issue 2), which turns every closed range on `production_year` into an input the model has never seen.

With both fixed, our implementation is at least as accurate as the reference implementation in the same setting (no samples). As shipped, its geometric-mean q-error on JOB-light is about 4× worse than the reference's.

I didn't modify the repository. The probe scripts and the per-query predictions of both implementations were kept in a temporary directory and have not been preserved.

## Results on JOB-light (q-error, mean over seeds)

| configuration                                     | runs | median | p90 | p95  | max    | geo. mean | geo. mean, 17 closed-range queries | geo. mean, other 53 queries |
| ------------------------------------------------- | ---- | ------ | --- | ---- | ------ | --------- | ---------------------------------- | --------------------------- |
| **ours as shipped** (lr 1e-4)                     | 3    | 32.0   | 954 | 1713 | 64 851 | **45.3**  | 68.4                               | 40.1                        |
| ours, lr 1e-3 (issue 1)                           | 3    | 9.5    | 741 | 1272 | 17 483 | 16.2      | 277.9                              | 7.8                         |
| ours, lr 1e-3 + mask fix only (issue 3)           | 1    | 7.5    | 618 | 1042 | 3 050  | 13.9      | 109.8                              | 7.2                         |
| ours, lr 1e-3 + one element per predicate (2 + 3) | 1    | 4.9    | 47  | 468  | 1 678  | 7.5       | 8.5                                | 7.2                         |
| **ours, all fixes** (1, 2, 3, 6)                  | 3    | 5.0    | 85  | 414  | 1 617  | **8.5**   | 11.2                               | 7.8                         |
| reference, no samples (our scope)                 | 3    | 8.8    | 144 | 631  | 2 807  | 12.4      | 15.3                               | 11.8                        |
| reference, bitmaps (paper's full model)           | 2    | 2.4    | 32  | 97   | 755    | 4.2       | 4.3                                | 4.1                         |
| _paper, Table 4 (bitmaps)_                        | –    | 3.82   | 78  | 362  | 1 110  | –         | –                                  | –                           |

On the 10 000 held-out validation queries, the reference without samples reaches a median of 1.78–2.15. Ours with lr 1e-3 reaches 1.60–1.82, and as shipped 2.62–2.90. With the fixes, the training pipelines are therefore equivalent. The remaining gap to the paper's numbers comes from the bitmaps, which the README excludes.

Seed variance is large: one JOB-light run with lr 1e-3 but without the merge fix scored 27.0. The all-fixes runs stayed between 7.4 and 10.8.

## How the evidence was produced

- **Dataset:** our IMDB instance (DuckDB and Postgres) is a newer snapshot than the paper's: 4 736 114 titles vs. 2 528 312. None of the reference labels reproduce on it (70/70 JOB-light and 996/1000 sampled training labels differ). So every comparison uses the reference's own **100 000 labelled training queries** (`data/train.csv`) and its JOB-light labels, converted to SQL. Our column encoders were refit to the reference's `column_min_max_vals.csv`. Both implementations thus see the same data, and only featurization and training differ.
- **Reference:** run from a copy with three patches: Python 3.12/numpy 2 compatibility, a seed flag, and a "no samples" mode (zero-width bitmaps, i.e. the paper's _MSCN (no samples)_). Paper configuration: 100 000 queries split 90/10, 100 epochs, batch 1024, 256 hidden units, lr 1e-3.
- **Ours:** trained through the shipped `MscnEstimator.train` and evaluated through `calculate_estimate` on 90 000 queries. The vocabulary comes from `MscnFeaturizer.infer_from_samples(train, workload=JOB-light + synthetic)`. For speed only, I replaced `qerror_loss` with a vectorized version, asserted to give identical values. Fixes were applied as monkeypatches.
- **Not done:** no end-to-end runtime benchmarks on Postgres. I also did not train on the output of our own query sampler; issue 9 is a structural analysis of 1 000 generated queries.

## Where the paper and the reference disagree

| topic                     | paper                                        | reference                                                                                                         | ours                                                       |
| ------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| learning rate             | 0.001 (Sec. 4.6)                             | 0.001                                                                                                             | **0.0001** (issue 1)                                       |
| epochs / queries          | 100 / 100 000                                | CLI defaults 10 / 10 000; README gives 100 / 100 000 for the paper's results                                      | 100 / all given samples, no validation split               |
| empty join/predicate sets | padding is masked out                        | the empty join is its own one-hot category `''`; an empty predicate is a zero vector with mask 1                  | zero embedding (−inf normalizer); equally valid            |
| label transform           | log, min–max over training labels            | `log`, `exp` (predictions rounded to integers)                                                                    | `log1p` for training, `exp` at inference (issue 6)         |
| operator encoding         | one-hot                                      | one-hot over the operators seen in training                                                                       | one-hot, but few-hot after merging (issue 2)               |
| set elements              | one element per predicate                    | one element per predicate                                                                                         | **one per column** (issue 2)                               |
| training queries          | 0–2 joins, uniform; no repeated column       | same (`train.csv`: 26.8 % single-table, 0 repeated columns)                                                       | sampler: ≥ 1 join, 55 % repeated columns (issue 9)         |

The README states that the paper uses a "few-hot encoding scheme" for operators and that we deliberately use pure one-hot. This is inaccurate: the paper (Sec. 3.1, Fig. 2) and the reference both use one-hot. Our implementation is the only one that produces few-hot vectors (issue 2).

## A. Training and output scaling

**1. The learning rate is 0.0001 instead of 0.001 — critical.**

- **Where:** `MscnHyperParams.learning_rate`, `_estimator.py:48`. Commit `a2229ad` ("Align hyperparams with values from the paper") changed it from 0.001 to 0.0001.
- **Paper and reference:** the paper tunes hyperparameters and states that "0.001 [performed] best … we use … a learning rate of 0.001" (Sec. 4.6). The reference also uses 0.001.
- **What happens:** after 100 epochs the model is clearly under-trained. Validation mean q-error is 12–14 instead of 5–6.
- **Impact:** JOB-light geometric mean 45.3 → 16.2; median 32 → 9.5.

**6. Labels are scaled with `log1p` during training but inverted with `exp` at inference.**

- **Where:** `_misc.py:76`. Training uses `log1p`/`expm1` (`normalize_labels`, `unnormalize_torch`), but `unnormalize_labels` uses `exp`.
- **What happens:** every estimate is biased by +1 row.
- **Impact:** negligible here (geometric mean 7.47 vs. 7.47), but the inconsistency should be fixed.

**7. The online/workload card range heuristic** (`infer_from_workload`, `online`).

- **What it does:** `max_card` is the product of the three largest tables, 1.2·10²² on IMDB, i.e. ln ≈ 50.8. Real labels occupy roughly ln 0–23, so they map to only the bottom ~45 % of the sigmoid range.
- **Comparison:** the paper and reference use the observed training min/max. `MscnEstimator.load_or_build` does too (via `infer_from_samples`), so only the `online`/`infer_from_workload` paths are affected.
- **Status:** not measured; probably minor resolution loss.

## B. Featurization

**2. Predicates on the same column are merged into one set element — critical for JOB-light.**

- **Where:** `encode_filter_predicates`, `_featurizer.py:858-872`.
- **What happens:** for a column that appears more than once, the operator one-hots are OR-ed (`np.maximum`) and the values averaged. The code comment calls this "the MSCN strategy", but neither the paper nor the reference does it; both keep every predicate as its own element.
- **Why it hurts:**
  - `production_year > 1980 AND production_year < 1984` becomes `[> <]` with value 0.448. The operator vector `[1,0,1]` never occurs in training.
  - The range width is lost entirely, because only the midpoint survives.
  - Repeated averaging of three or more predicates is also not a true mean.
- **Evidence:**
  - 17 of the 70 JOB-light queries contain such a closed range.
  - With merging, the model predicts 17 k–20 k for 7 of them, whose true sizes range from 75 k to 21 M, and 134 k–270 k for three queries of 2.9 M–207 M.
  - Geometric mean on these 17: **91.8** merged vs. **8.5** separate (seed 1). The other 53 queries are unchanged (7.2).
- **Fix:** emit one row per predicate. The row count must then be bounded by the maximum number of predicates, not `n_columns`.

**3. `predicates_mask` counts raw predicates, not encoded rows.**

- **Where:** `encode_single`, `_featurizer.py:657`, uses `len(predicates)`.
- **What happens:**
  - Merged (issue 2) or skipped predicates leave zero padding rows marked as present. Their `MLP(0)` output is averaged into `w_P`. In q-46 the mask covers 4 rows, but only 3 are populated.
  - When there are more predicates than `n_columns`, `build_mask` fails with a negative padding size.
- **Impact:** with merging in place, fixing only the mask changed little (13.3 → 13.9). It is part of the correct fix for issue 2: the mask must equal the number of emitted rows.

**4. Queries with `IN`, `BETWEEN`, `IS [NOT] NULL` (or any non-binary filter) crash.**

- **Where:** `encode_single` raises `ValueError` (`_featurizer.py:646-649`) before `encode_filter_predicates` can apply its "skip unsupported predicate" logic, so that logic (commits `0ff3644`, `861b563`) is unreachable for these predicates.
- **Impact:** none on JOB-light. MSCN cannot estimate JOB (which uses all of these) or any workload with IN lists.
- **Fix:** either skip these predicates as intended (and exclude them from the mask), or approximate them (e.g. `BETWEEN` as two predicates, which only works once issue 2 is fixed).

**5. Smaller featurization issues** (JOB-light unaffected):

- **LIKE values carry no information.** The text encoder places `'%Star%'` and `'%Wars%'` at the same position (0.00084), because every leading-`%` pattern sorts to the same spot. The README advertises LIKE support; the model can only learn an average LIKE selectivity per column.
- **Text ordering differs between DB and numpy.** `TextEncoder` assumes the database's `ORDER BY` matches numpy's code-point order for `searchsorted`. This holds for DuckDB, but not for Postgres databases with a non-C collation.
- **Constant columns produce NaN.** `NumericEncoder` divides by `max − min`, giving NaN/inf for constant columns; the reference returns 0. A NaN feature makes the whole estimate NaN.
- **Cross products are out of distribution.** Sub-queries without join predicates, such as `{mc, mi}` of a JOB-light star, are encoded as several tables with an empty join set, which never occurs in training. PostBOUND's enumerators normally avoid them.
- **Large queries can crash the padding.** `encode_tables`/`encode_joins` pad to `n_tables`/`n_joins` rows, so a query with more tables (self-joins) or join predicates than the vocabulary size fails.

## C. Network architecture and training loop

Verified line by line against `reference-implementation/mscn/model.py` and `train.py`. They are equivalent:

- **Architecture:** same layers (2-layer ReLU MLPs per set, masked average pooling, concatenation, 2-layer output MLP with sigmoid). Hidden size 256, batch 1024, Adam, mean q-error loss on unnormalized values, no shuffling (as in the reference).
- **Argument order:** `forward` takes `(tables, joins, predicates, …masks)` instead of the reference's `(samples, predicates, joins, …)`. This matches the field order of `FeaturizedQuery`, so the dataset columns are wired correctly.
- **Only behavioural difference:** empty sets get a zero embedding (−inf normalizer) instead of the reference's dummy element. This is equally valid and consistent between training and inference.
- **Confirmation:** with lr 1e-3, our validation q-errors match the reference's (median 1.6–1.8 vs. 1.8–2.2).
- **Leftovers:**
  - `_move_features` sets `requires_grad_(True)` on all inputs (commit `5f9f3c7`, "Apparently we need to set requires_grad_"). This isn't needed and only adds work.
  - The shipped `qerror_loss` loops per sample in Python with a GPU sync each; a vectorized `torch.where` gives identical values and is much faster.

## D. Training-data generation (`tools/query-sampler.py`, `sampler/_qgen.py`)

**8. `--like <workload>` crashes on PostBOUND 0.23.**

- **Where:** `_PredicateCollector` still calls `visit_query_predicates(query)` and expects `components` in `visit_and/or/not_predicate`. This is the same API break the SafeBound audit found.
- **Impact:** the sampler can only run in schema-wide mode. The statistics below used a scratch adapter.

**9. The generated training distribution differs from the paper's in ways that hurt MSCN.** Measured on 1 000 queries with the tool's defaults and `--like job-light`:

- **No single-table queries.** `--min-tables` defaults to 2 and overrides the workload-derived minimum. The paper draws 0–2 joins uniformly (≈ ⅓ base-table queries). An optimizer asks MSCN for every base-table cardinality, so these become out-of-distribution.
- **55 % of queries repeat a filter column.** Columns are drawn with replacement.
  - Some repeats contradict each other, e.g. `info_type_id = 100 AND info_type_id = 101`; their empty results are discarded, so the sampling work is wasted.
  - Others are redundant (`< 1966 AND < 1988`). With issue 2, both kinds turn into few-hot/averaged elements.
  - The paper and `train.csv` never repeat a column.
- **Literal selection depends on the backend.** On Postgres it is uniform over the distinct values in a 1 % `TABLESAMPLE SYSTEM` block sample; on DuckDB it is a single random row (frequency-weighted). The paper uses "an actual value from the column". This shifts the predicate distribution between backends. Not measured.
- **Timeouts silently drop queries** when `--timeout` is used, which biases the sample against large results.

## E. Other findings

- **Store/load round trip** of the featurizer reproduces identical tensors for all JOB-light queries.
- **Minor `store()` issues:**
  - It returns the parsed catalog `dict` instead of the `Path` its signature promises.
  - It rewrites the JSON with `r+` without `truncate()`. This is safe today only because the content always grows.
- **`pre_trained`** calls `torch.load` without `map_location`, so a model trained on a GPU machine can't be loaded on a CPU-only one.
- **Bitmaps (non-goal), for context:** on identical data they improve the reference from 12.4 to 4.2 geometric mean on JOB-light.

# Suggested fix order

- [x] Issue 1: set the default learning rate back to 0.001 (and fix the docstring).
- [x] Issues 2 and 3: one set element per predicate, with the mask equal to the number of emitted rows; size the predicate matrix by the maximum number of predicates rather than `n_columns`.
- [x] Issue 4: actually skip (or approximate) IN/BETWEEN/IS NULL instead of raising.
- [x] Issue 6: use `expm1` in `unnormalize_labels`.
- [x] Minor: constant-column guard in `NumericEncoder`, `map_location` in `pre_trained`, drop `requires_grad_` on inputs, vectorized loss.

After that, retrain any stored MSCN models. Every model trained with the current defaults is affected by issue 1.

## Not addressed

- Issues 8 and 9 (PostBOUND 0.23 visitor adapters in the sampler; single-table queries by default; filter columns without replacement): out of scope. The query sampler is a standalone tool, not part of MSCN.
