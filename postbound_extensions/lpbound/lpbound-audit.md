# Audit Scope

- **Date:** 2026-11-24/25
- **Commit audited:** `076c608` ("[LpBound] Documentation"), working tree otherwise unchanged for `postbound_extensions/lpbound/`
  and `postbound_extensions/safebound/`.

> # Audit scope: LpBound
>
> - **Family:** cardinality-bounds
> - **What the optimizer produces:** an upper bound on the cardinality of each (sub-)query, obtained
>   as the optimum of a linear program (LPberge or LPflow) over lp-norms of (conditioned) degree
>   sequences.
> - **Critical outcome:** issues that lead to different upper bounds being calculated, above all
>   bounds below the true cardinality (soundness), then bounds that are looser than the reference's
>   on the same statistics (tightness).
> - **Paper:** `LpBound-Paper.pdf`
> - **Reference implementation:** `reference-implementation/`; the code path behind the paper's
>   results: `benchmarks/experiments/accuracy_acyclic.py` (→ `src/lpbound/acyclic/lpbound.py`:
>   `build_lpbound_statistics`, `estimate`) for acyclic workloads and
>   `benchmarks/experiments/accuracy_cyclic.py` (→ `src/lpbound/cyclic/`) for cyclic ones, configured
>   by `src/lpbound/config/lpbound_config.py`. The C++ solver is only needed for timing, not accuracy.
>
> ## Database and workload
>
> - **Database:** Stats (primary); IMDB (secondary, only to exercise LPflow on cyclic queries)
> - **Workload:** Stats-CEB, via `pb.workloads.stats()`; JOB-light, via `pb.workloads.job_light()`;
>   for focus area 5, the cyclic JOB queries via `pb.workloads.job()`
> - **Backend for measurements:** DuckDB (`stats.duckdb`, `imdb.duckdb`) for true cardinalities and
>   for running the reference (it builds its statistics on DuckDB); Postgres
>   (`.pg-connect-stats.toml`) for building our catalog through `LpBoundCatalog.load_or_build`, as
>   the README does.
> - **Benchmarks run on:** not required (accuracy only)
>
> ## Focus areas
>
> 1. Computation of the lp-norms: which degree sequences are built (join columns, per relation),
>    which norms (p range, l0/l∞), NULL handling.
> 2. Conditioned norms for filter predicates: MCV-based equality norms, the defaults for non-MCV
>    values, the histogram hierarchy for ranges, our half-open range extension, and how several
>    predicates on one relation are combined.
> 3. Construction of the join graph (`lp_join_graph`) and the LP formulations (`LpBerge`, `LpFlow`):
>    constraints, variables, objective, and the choice between them (`is_berge_acyclic`).
> 4. Sub-query estimation through `LpBoundEstimator` (the entry point an optimizer uses), including
>    catalog persistence (`store`/`load`/`load_or_build` give identical bounds).
> 5. (lower priority) LPflow on cyclic queries.
>
> ## Excluded
>
> - Everything listed as a non-goal in the README: predicate propagation (pre-computed PK/FK joins),
>   prefix sharing of degree sequences, `GROUP BY` queries.
> - Estimation and statistics-computation time (no C++ solver comparison).
> - Standalone tools (query sampler, training-data generators) and PostBOUND itself.
> - The SafeBound module, except the parts LpBound imports and calls on its own code path.
>
> ## Notes
>
> - Our implementation reuses parts of `postbound_extensions/safebound/` (degree-sequence, MCV and
>   histogram fetching); those parts are in scope as far as LpBound calls them. The SafeBound
>   scope refers to a `safebound-audit.md`, which is not in this checkout.
> - The README states a median accuracy of 30.80 (with predicate propagation) vs 109.85 (without)
>   on Stats; the reference ships its Stats estimates in
>   `results/accuracy_acyclic/stats/lpbound_stats_full_estimations.csv` and true cardinalities in
>   `truecardinality_stats_full_estimations.csv`.
> - To compare like with like, run the reference with predicate propagation disabled as well as in
>   the authors' configuration.
> - True cardinalities: `models/true-cards/stats-cards.parquet` and
>   `models/true-cards/intermediate-cards-stats.csv` (verify before use).
> - No pre-built LpBound catalog exists in `models/`; the audit builds one.

**Deviations from the scope that turned out to be necessary:**

- **No cyclic JOB queries exist.** All 70 407 connected JOB sub-queries are Berge-acyclic in our join graph, because
  join equivalence classes merge the table-level cycles. The same holds for all 1566 Stats and all 440 JOB-light
  sub-queries, so LPflow is never used on any workload in scope. Focus area 5 was covered with an independent LPbase
  oracle on random cyclic instances instead (section C).
- **`safebound-audit.md` does exist** in `postbound_extensions/safebound/`; the scope's note is wrong. Its NULL
  findings were re-checked for the functions LpBound calls (section A).
- **JOB-light true cardinalities** were computed on our IMDB snapshot (`imdb.duckdb`, 4 736 114 titles) for all 440
  connected sub-queries of the 70 queries. The reference was fed the same six tables, because its IMDB CSVs are not
  present and the paper's snapshot differs.
- The audit was interrupted once by system sleep after the JOB-light catalog build had finished; it resumed from
  there. Every result below comes from a complete run.

# Audit Report

**The bound computation is correct and sound; the defects that matter are in the entry points.** No bound is below
the true cardinality on any of the 1566 Stats or 440 JOB-light sub-queries. Our LpBerge and LpFlow agree with an
independent LPbase oracle on 1000 random instances (about two thirds cyclic), and every statistic in the Stats catalog
matches an independent recomputation in DuckDB. In the same scope (no predicate propagation, no prefix norms), ours is
tighter than the reference on Stats (full-query geometric mean 180 vs 223) and on par on JOB-light (36.0 vs 35.4).

**The documented usage crashes.** The README example fails with `AttributeError` because `LpBoundCatalog` is not
exported (issue 4). `load_or_build(<directory>)`, the recommended pattern, never finds the catalog it stored: every
later call rebuilds the whole catalog (60 s on Stats, 11 min on JOB-light) and then raises `IsADirectoryError`
(issue 5).

Three tightness issues explain where ours is looser than the reference: a histogram hierarchy that degenerates on
skewed columns (issue 1, up to 1.41× on Stats), two-sided ranges looked up as two half-open ranges (issue 2, up to
1.90× on JOB-light), and a looser non-MCV default (issue 3, up to 2.05× on JOB-light). With all three fixed, ours is at
least as tight as the restricted reference on more than 98 % of sub-queries; the rest is at most 1.23× looser. Their effect on
the geometric means is small (≤ 2 %). The 4.9× gap to the authors' configuration on Stats (36.5 vs 180) is the README
non-goal predicate propagation, as the README states.

I didn't modify the repository. Scripts, catalogs and per-sub-query results are in the artifacts directory
`2026-11-24-lpbound-audit`.

## Results on Stats

Bound ÷ max(1, true cardinality); 1566 connected sub-queries with ≥ 2 tables, 146 full queries. "Looser than ref" is
the number of sub-queries where ours exceeds the reference (`noppnp`) by more than 0.1 %, and the largest ratio.

| configuration                                                     | below true size | all: geo. mean | full: geo. mean | full: median | ours ÷ ref, p10 / p50 / p90 | looser than ref   |
| ----------------------------------------------------------------- | --------------- | -------------- | --------------- | ------------ | --------------------------- | ----------------- |
| as shipped (`load_or_build`; reloaded = fresh)                    | 0 / 1566        | 10.74          | 180.5           | 109.85       | 0.77 / 1.00 / 1.008         | 407 (max 1.41×)   |
| + anchored prefix/suffix buckets (issue 1)                        | 0               | 10.62          | 177.7           | 106.99       | 0.77 / 1.00 / 1.000         | 24 (max 1.008×)   |
| + merged range predicates (issue 2)                               | 0               | 10.62          | 177.7           | 106.99       | 0.77 / 1.00 / 1.000         | 24 (max 1.008×)   |
| _alternative for issue 1: nested levels only_                     | _0_             | _10.69_        | _179.4_         | _107.04_     | _0.77 / 1.00 / 1.005_       | _315 (max 1.06×)_ |
| _alternative for issue 1: reference-style breaks + nested levels_ | _0_             | _10.67_        | _178.9_         | _107.04_     | _0.77 / 1.00 / 1.000_       | _33 (max 1.03×)_  |
| reference, no predicate propagation, no prefix norms (`noppnp`)   | 0               | 12.45          | 223.1           | 120.45       | –                           | –                 |
| reference, `noppnp` without offset histogram layers               | 0               | 12.45          | 223.1           | 120.45       | –                           | –                 |
| reference, no predicate propagation, prefix norms kept            | 0               | 12.30          | 221.8           | 120.45       | –                           | –                 |
| reference, authors' configuration                                 | 0               | 5.74           | 36.5            | 30.80        | –                           | –                 |
| _reference's shipped results file_                                | _0_             | –              | _36.5_          | _30.80_      | –                           | –                 |

- Bounds are deterministic, so there is no run-to-run noise: every difference is real, but the aggregate effects of
  issues 1–3 stay below 2 %.
- Issue 3 has no effect on Stats: all 1226 equality lookups hit an MCV, so the non-MCV default is never used.
- Where ours is tighter than the reference (192 sub-queries as shipped, down to 0.03×), the reason is the
  reference's NULL handling in range buckets (see "Where the paper and the reference disagree").
- Predicate propagation alone explains the gap to the authors' configuration (30.80 → 120.45 median); prefix norms
  change the geometric mean only from 221.8 to 223.1, and the offset layers not at all.

## Results on JOB-light

440 connected sub-queries with ≥ 2 tables of the 70 JOB-light queries, true cardinalities on our IMDB snapshot.

| configuration                                                          | below true size | all: geo. mean | full: geo. mean | full: median | looser than ref  | tighter than ref |
| ---------------------------------------------------------------------- | --------------- | -------------- | --------------- | ------------ | ---------------- | ---------------- |
| as shipped (`load_or_build`; reloaded = fresh)                         | 0 / 440         | 9.60           | 35.98           | 9.42         | 16 (max 2.30×)   | 4                |
| + anchored prefix/suffix buckets (issue 1)                             | 0               | 9.59           | 35.88           | 9.42         | 16 (max 2.05×)   | 4                |
| + merged range predicates (issue 2)                                    | 0               | 9.56           | 35.49           | 9.42         | 14 (max 2.05×)   | 9                |
| + reference-style non-MCV default (issue 3)                            | 0               | 9.46           | 35.26           | 9.42         | 6 (max 1.23×)    | 9                |
| _alternative for issue 1: nested levels only_                          | _0_             | _9.61_         | _36.00_         | _9.42_       | _16 (max 2.35×)_ | _4_              |
| _alternative for issue 1: reference-style breaks + nested, with merge_ | _0_             | _9.58_         | _35.85_         | _10.23_      | _10 (max 2.05×)_ | _1_              |
| reference, `noppnp`                                                    | 0               | 9.47           | 35.43           | 10.23        | –                | –                |
| reference, authors' configuration                                      | 0               | 6.89           | 24.82           | 6.46         | –                | –                |

- Each fix alone, against as shipped: anchored buckets tighten 3 sub-queries (to 0.82×), merging tightens 7 (to
  0.53×), the non-MCV default tightens 8 (geo. mean 0.56×, to 0.49×). None of them loosens any bound.
- The two "alternative" layouts are **not** safe replacements: nested levels alone loosen 4 sub-queries (up to 1.09×)
  and reference-style breaks loosen 3 (up to 1.23×) compared with as shipped. The anchored buckets are additive and
  never loosen a bound.
- The 6 remaining looser sub-queries are closed `production_year` ranges that straddle a bucket boundary, where the
  reference's half-shifted offset layers help (up to 1.23×).

## How the evidence was produced

- **True cardinalities:** Stats from `models/true-cards/intermediate-cards-stats.csv`; they match the reference's
  `truecardinality_stats_full_estimations.csv` on all 146 full queries, and 59 random sub-queries recomputed on DuckDB
  match (one skipped: out of memory). JOB-light computed on `imdb.duckdb` for all 440 sub-queries.
- **Ours:** catalogs built through `LpBoundCatalog.load_or_build(<dir>, database=<Postgres>, workload=...)` with
  `LpBoundSpec.default()` as in the README; estimates from `LpBoundEstimator.calculate_estimate(full_query,
sub_query_tables)`. Every run also builds a fresh in-memory catalog and compares it with the reloaded one.
  **Harness change:** `LpBoundCatalog` is imported from `postbound_extensions.lpbound._norms` (issue 4). Fixes are
  monkeypatches (`patches.py`); catalogs are rebuilt for every histogram or non-MCV variant, the lookup-only merge
  variant reuses the stored catalog.
- **Statistics check (Stats):** every unconditioned (13), MCV (1365) and histogram-bucket (4274, all levels) norm set
  was recomputed in DuckDB from the raw rows; all match to 1e-6. With a 2-entry MCV list the non-MCV fallback dominates
  each of the 1342 non-MCV values for every p. Over all sub-queries, 1226 equality lookups never miss the MCV list and
  no predicate falls back to unconditioned norms.
- **LP oracle:** an independent LPbase (all 2^n entropies, elemental Shannon inequalities, Eq. 18 statistics) on 1000
  random instances (2–4 relations, 2–3 join variables, skewed random data, about one third Berge-acyclic), checked
  against brute-force join sizes. LPbase is never below the true size; our LpBerge and LpFlow equal LPbase up to
  `pb.Cardinality`'s round-to-nearest (|ours − LPbase| ≤ 0.5) on every instance.
- **Reference:** a copy of `reference-implementation/`, Python accuracy path (`build_lpbound_statistics`, `estimate`),
  default `LpBoundConfig` (5000 MCVs, 128 buckets, p ≤ 10, l0 and l∞), on the same sub-query SQL (only `COUNT(*)` →
  `*` and `CAST('…' AS timestamp)` → `'…'`). Runtime patches, none to its files: one DuckDB file per variant; `nopp`
  and `noppnp` empty `fk_pk_joins_dict`; `noppnp` also uses the full instead of the prefix base norms; `nooffset`
  drops the `offset_*` layers. For JOB-light, its DuckDB files were pre-filled with the six tables from `imdb.duckdb`.
  Environment differences: DuckDB 2.0.1 instead of 0.10.1, OR-tools 9.15 instead of 9.12. The authors' configuration
  reproduces the shipped Stats results exactly (geo. mean 36.47, median 30.80).
- **Not done:** runtime benchmarks and estimation times (excluded by the scope); the paper's subgraph-matching
  workload (the reference's C++ LPflow was not built), so LPflow is validated on synthetic instances only.

## Where the paper and the reference disagree

| topic                        | paper                                                                                             | reference                                                                                                                                | ours                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| non-MCV default              | norms of the r largest degrees of all non-MCV join values (r = max join values per non-MCV value) | per-p maximum over the norms of each non-MCV value (Sec. 5's "more expensive alternative")                                               | norms of the rank-wise maximum sequence (SafeBound's default): tighter than the paper, looser than the reference (issue 3) |
| histogram layout             | each layer half the buckets of the one below, covering the domain; smallest enclosing bucket      | quantiles over _distinct_ values (all rows for dates), half-shifted "offset" layers, open-ended outer buckets, one extra full bucket     | equi-depth over rows, closed buckets, levels recomputed independently (issue 1)                                            |
| NULLs in range predicates    | not discussed                                                                                     | buckets with a NULL lower bound also contain rows whose predicate value is NULL (`v.BountyAmount >= 0` gets l1 = 34 789 instead of 1098) | NULLs excluded                                                                                                             |
| two predicates on one column | one range [s, e]                                                                                  | merged into one range                                                                                                                    | separate lookups, minimum (issue 2)                                                                                        |
| choice of LP                 | LPberge if Berge-acyclic, else LPflow                                                             | Python `estimate()` always uses LPberge for non-group-by queries, without an acyclicity check                                            | as the paper                                                                                                               |
| l0 norm                      | only for the prefix optimization                                                                  | computed, not used as an LP constraint                                                                                                   | not computed                                                                                                               |
| MCV selection                | top-k values of the predicate column                                                              | top-k over all rows                                                                                                                      | top-k over rows with a non-NULL join key                                                                                   |
| LP solver                    | HiGHS 1.7.2                                                                                       | C++ path HiGHS; Python accuracy path OR-tools GLOP                                                                                       | SciPy HiGHS                                                                                                                |
| NULL join keys               | not discussed                                                                                     | excluded                                                                                                                                 | excluded                                                                                                                   |

- None of these affect soundness.
- The reference's NULL handling is why ours is tighter on 192 Stats sub-queries; the histogram layout and the
  non-MCV default are why it is looser on others (issues 1 and 3).
- The missing acyclicity check in the reference does not matter here: every sub-query is Berge-acyclic.
- The offset layers make no difference on Stats; on JOB-light they explain the last 6 sub-queries where ours remains
  looser (up to 1.23×).

## A. Statistics (focus area 1)

The norms are what the paper prescribes. For every join column, the catalog stores ||deg_R(\*|X)||_p for p ∈ {1, …,
10, ∞} over rows with a non-NULL join key; all 13 unconditioned Stats norm sets match DuckDB (e.g. `votes.userid`:
34 789 rows, 5543 values, max degree 712). The SafeBound audit's issue 10 (NULL join keys counted as a value) does not
affect LpBound: `fetch_raw_ds`, `fetch_correlated_ds`, `fetch_mcv_ds` and `fetch_non_mcv_ds` all exclude NULL join
keys. l0 is not computed and not needed without prefix norms. Defaults match the paper (5000 MCVs, 128 buckets).

## B. Conditioned norms (focus area 2)

**1. The histogram hierarchy degenerates on skewed columns — high.**

- **Where:** `_norms.py:663-675` (breaks) and `_norms.py:689-702` (coarser levels), since `732ab85`.
- **What happens:** each level computes its breaks independently as equi-depth quantiles over rows and drops
  duplicates. On a column with a few heavy values, every level ends up with nearly the same breaks, so levels neither
  halve nor nest. `posts.favoritecount` (value 0: 885 rows, 1: 6912, 2: 2489, …) gets 15, then 12, … buckets with
  bounds `[0, 1, 2, 3, …]`, and no level has a bucket from 0 to 2. `favoritecount <= 2` therefore falls through to
  the single root bucket: l1 = 12 979 for `posts.owneruserid`, where the matching rows have l1 = 10 085, which the
  reference finds exactly with its open-ended bucket. The `LpBoundSpec` docstring's "2n − 1 conditioned Lp norms per
  range filter column" does not hold either.
- **Evidence:** measured (`06c_hist_favoritecount.txt`, `06_debug_posts_norms.txt`, `06b_debug_ref_norms.txt`).
- **Impact:** sound but looser. As shipped, 407 Stats sub-queries are looser than the reference, by up to 1.41×.
  Adding anchored buckets tightens 529 Stats sub-queries (geo. mean 0.968×, down to 0.71×; full-query geo. mean
  180.5 → 177.7) and 3 JOB-light sub-queries, and leaves only 24 Stats sub-queries looser than the reference, by at
  most 1.008×.
- **Fix:** add anchored buckets [min, bᵢ] and [bᵢ, max] at the finest breaks, and take the minimum with the current
  lookup. This is additive: it never loosens a bound. Rebuilding the levels as nested pairs or with reference-style
  breaks instead is **not** recommended: both loosen some JOB-light bounds (up to 1.09× and 1.23×).

**2. Two-sided ranges on one column are looked up as two half-open ranges — high.**

- **Where:** `_norms.py:1219-1226` (`visit_and_predicate`) with `get_less`/`get_greater` (`_norms.py:395-432`).
- **What happens:** `t.production_year > 1980 AND t.production_year < 1984` becomes min(norms for [1980, max], norms
  for [min, 1984]) instead of the smallest bucket enclosing [1980, 1984] (paper Sec. 5; the reference merges such
  predicates). 1143 of 1566 Stats sub-queries contain such a pair.
- **Evidence:** measured. Merging the ranges in the harness tightens 7 JOB-light sub-queries, down to 0.53×; on top of issue 1 the full-query geo. mean goes 35.88 → 35.49. The worst JOB-light gap to the
  reference (2.30×, the 1980–1984 sub-query) drops to 1.21× with issues 1–3 fixed. No Stats bound changes.
- **Impact:** sound, looser on closed ranges that fit a finer bucket than either half-open range.
- **Fix:** merge simple range predicates per column in `visit_and_predicate` before the lookup.

**3. The non-MCV default is looser than the reference's — high.**

- **Where:** `build_equality_norms` (`_norms.py:276-278`) uses `fetch_non_mcv_ds` (`safebound/_catalog.py:522`),
  the rank-wise maximum degree sequence over all non-MCV values.
- **What happens:** the norms of a rank-wise maximum sequence can be far above the largest norm of any single value.
  For `movie_companies.movie_id` conditioned on `company_id` (5000 MCVs of 234 k values), ours uses l1 = 256 and
  l∞ = 22 for every non-MCV company. The reference takes, per p, the maximum over the individual non-MCV values'
  norms, which is still sound because each non-MCV value's norms are below it. Company 22956 itself has l1 = 88,
  l∞ = 2; for `t.production_year > 2010 AND mc.company_id = 22956` (true size 61) ours bounds 256, the reference 125. The paper's own default (the r largest combined degrees) is looser than ours, so ours is not a deviation from
  the paper.
- **Evidence:** measured (`11_debug_company_id.txt`; patch `nonmcvmax`).
- **Impact:** on JOB-light, the reference-style default tightens 8 sub-queries (geo. mean 0.56×, down to 0.49×) and
  cuts the remaining gap to the reference from 14 sub-queries (max 2.05×) to 6 (max 1.23×). No effect on Stats.
- **Fix:** compute the fallback as the per-p maximum over the norms of each non-MCV value (one SQL aggregate per
  column pair, as in the reference's `generate_nonmcv_max_sql`).

Otherwise the conditioned norms are as the paper describes and sound: all 1365 MCV norm sets and 4274 buckets match
DuckDB; the non-MCV fallback dominates every non-MCV value; conjunctions take the per-p minimum and disjunctions and
`IN` the per-p sum (Minkowski); out-of-domain ranges and `BETWEEN hi AND lo` correctly yield an empty result;
`get_range` always picks a bucket that encloses the range (buckets are closed on both sides); half-open ranges map to
[min, v] and [v, max] as documented.

## C. Join graph and linear programs (focus areas 3 and 5)

`lp_join_graph` builds the bipartite relation/equivalence-class graph, and `is_berge_acyclic` (incidence graph is a
forest) is the correct test. LpBerge has the paper's variables (one per join class, one private and one atom variable
per relation), objective E_Q, the Eq. 18 statistics constraints and the 1 + |V_j| additivity constraints, including
the private variable, which is essential for soundness. LpFlow builds the flow network of Sec. 4.2 with one flow per
variable and the w/p source edges for finite p.

Both were checked against the independent LPbase oracle on 1000 random instances, about two thirds of them not
Berge-acyclic: no disagreement beyond `pb.Cardinality`'s round-to-nearest, and no bound below the brute-force join
size. The rounding is sound because true sizes are integers. Empty filtered relations correctly short-circuit to 0.

## D. Estimator entry points and persistence (focus area 4)

**4. `LpBoundCatalog` is not exported, so the README example crashes — critical.**

- **Where:** `__init__.py:1-17`, never exported since `732ab85`; the README example using
  `pbx.lpbound.LpBoundCatalog` came with `076c608`.
- **What happens:** `pbx.lpbound.LpBoundCatalog.load_or_build(...)` raises
  `AttributeError: module 'postbound_extensions.lpbound' has no attribute 'LpBoundCatalog'`.
- **Evidence:** measured (`01_build_catalog_pg.as-shipped-crash.log`).
- **Impact:** no catalog, and hence no estimator, can be created through the documented API.
- **Fix:** export `LpBoundCatalog` from `__init__.py` and add it to `__all__`.

**5. `load_or_build(<directory>)` never reloads; the second call rebuilds and then crashes — critical.**

- **Where:** `_norms.py:1032-1035` looks for `safebound-catalog.json(.xz)` (copied from SafeBound), while `store`
  writes `lpbound-catalog.json`; `_norms.py:1239` only appends the file name when the directory does not exist yet.
  Both since `732ab85`.
- **What happens:** the first call (path does not exist) builds and stores `<dir>/lpbound-catalog.json`. Every later
  call sees a directory without a `safebound-catalog.*` file, rebuilds the catalog, and `store(<dir>)` then opens the
  existing directory for writing.
- **Evidence:** measured (`04_as_shipped.log`, `04_joblight_as_shipped.log`: the second call attempts a rebuild;
  `10_store_existing_dir.txt`: `IsADirectoryError` for `store(<existing dir>)` and for the full second
  `load_or_build` call).
- **Impact:** the README's recommended pattern costs a full rebuild per process (60 s on Stats, 650 s on JOB-light)
  and then raises. `LpBoundCatalog.load(<dir>)` works, and reloaded catalogs give bounds identical to in-memory ones
  on all 1566 Stats and 440 JOB-light sub-queries, so the persisted format itself is fine.
- **Fix:** look for `lpbound-catalog.json(.xz)` in `load_or_build` (or reuse `load`'s directory logic), and in
  `store` append the file name whenever the path is an existing directory or has no suffix.

## F. Other findings

- **Minor:** `construction_time` is read by `load` (`_norms.py:999`) but never written by `__json__`
  (`_norms.py:1259-1267`), so reloaded catalogs always report `None`.
- **Minor:** the `EqualityConditionedNorms` docstring (`_norms.py:140-150`) says the fallback "represents the
  frequencies of all non-MCV values combined"; it is the rank-wise maximum over non-MCV values.
- **Minor:** the `LpBoundSpec` docstring (`_core.py:18-20`) promises 2n − 1 bucket norm sets per range column; see
  issue 1.
- **Minor:** the README's predicate-propagation statement (30.80 vs 109.85 on Stats) is confirmed; the reference
  without predicate propagation reaches 120.45, i.e. ours is slightly tighter in the same scope.
- **Out of scope (SafeBound):** `SafeBoundCatalog.store` has the same existing-directory crash as issue 5
  (`safebound/_catalog.py:2707`).

# Suggested fix order

- [x] Issue 4: export `LpBoundCatalog` from `postbound_extensions.lpbound`.
- [x] Issue 5: make `load_or_build` find `lpbound-catalog.json(.xz)`; let `store` handle existing directories.
- [x] Issue 2: merge range predicates per column before the histogram lookup.
- [ ] ~~Issue 3: non-MCV fallback as the per-p maximum over the individual non-MCV values' norms.~~
- [ ] ~~Issue 1: add anchored prefix/suffix buckets to the range histograms.~~
- [x] Minor: persist `construction_time`; correct the `EqualityConditionedNorms` and `LpBoundSpec` docstrings.

Issues 1 and 3 change the catalog contents: stored catalogs must be rebuilt after fixing them. Issues 2, 4 and 5 work
with existing catalogs.

## Not addressed

- Issue 1 is a wontfix: the paper only has a minimum description of the bucket
  strategy and the reference goes far beyond that description. Our implementation
  is the more direct translation of the paper's intent, but made practical.
- Issue 3 is a wontfix: we already follow the paper and slightly improve beyond
  it. The paper does not specify which of the two alternatives to use. At the same
  time, the per-norm max is significantly more expensive to compute since it does
  not allow for pre-aggregation as well.
