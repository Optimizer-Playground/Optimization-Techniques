# Audit Prompt

> Please perform an audit of our implementation of the SafeBound algorithm.
>
> The algorithm is implemented in the `postbound_extensions/safebound/` directory. Its README provides a high-level description and outlines important non-goals of the implementation, including features that have deliberately not been implemented. **Those should not be part of the audit.**
> In the directory, you will also find the original paper that described SafeBound (`SafeBound-Paper.pdf`) and the original reference implementation that was published alongside the paper (`reference-implementation/`).
>
> The audit should focus on the logical correctness of:
>
> - the selection of the degree sequences to generate
> - the compression of the degree sequences (i.e. conversion to piecewise-constant functions)
> - the combination of different correlated degree sequences for filter predicates
> - the construction of the FDSB inference tree as well as the corresponding inference logic
> - any other aspects that you consider important
>
> Focus the audit on larger deviations from the original paper (or the reference implementation), as well as subtle inconsistencies that turn out to have a large impact. Minor discrepancies should not be analyzed in detail. **Pay special attention to issues that might lead to different upper bounds being calculated.**
>
> Should the original paper and the reference implementation disagree on certain parts of the algorithm, describe the discrepancy and its impact on our implementation.
>
> Use the Stats database and workload if you need to compare any real measurements. The original paper contains some benchmark results, as does the reference implementation.
> The DuckDB-instance of Stats can be used for fast computation, but all benchmarks should be executed on Postgres. `.pg-connect-stats.toml` contains the necessary connection details.
> Lastly, a pre-generated instance of the SafeBound catalog for Stats is already available in the `generated` directory.
>
> It is very important for the audit to be thorough. So take your time and investigate any potential problems deeply.

The connection file mentioned in the prompt is named `.psycopg_connection_stats` in this checkout. The paper, the reference implementation and the pre-generated catalog were available on the system the audit ran on; they are not part of this repository.

# Claude Response

The implementation, as checked in, does not compute SafeBound bounds. Two defects in the inference code make the output both much too loose and, on 137 of 1566 Stats sub-queries, **below the true cardinality**. Three further soundness defects only show up once those are fixed. With all of them fixed, our bounds track the reference implementation closely.

I didn't modify the repository. The probe scripts and the reference implementation's bounds were kept in a temporary directory and have not been preserved.

## Results on Stats (bound ÷ true cardinality)

| configuration                                            | sub-queries below true size   | full queries: geometric mean | full queries: ours ÷ reference         |
| -------------------------------------------------------- | ----------------------------- | ---------------------------- | -------------------------------------- |
| HEAD as shipped                                          | **137 / 1562** (worst 6·10⁻⁷) | 416                          | 0.0005× … 2200× (10th–90th percentile) |
| HEAD with only the graph fixed (issue 1)                 | 0                             | 17 700                       | 76×                                    |
| all fixes, our sampled integration                       | 0                             | 297                          | 1.29×                                  |
| all fixes, exact integration                             | 0                             | 200                          | **0.87×**                              |
| reference, no grouping, no PK-FK pre-joins               | 0                             | 231                          | –                                      |
| reference, authors' configuration (with PK-FK pre-joins) | 0                             | 54                           | –                                      |

The last row is about 4× tighter than ours. That gap comes from the PK-FK pre-joins, which the README lists as a non-goal.

## How the evidence was produced

- **Exact evaluator:** I wrote an independent implementation of the paper's Algorithm 2 that integrates the piecewise functions exactly. It matches a brute-force evaluation on the explicitly built worst-case instance in all 300 random chain-query tests.
- **True cardinalities:** computed with DuckDB for all 1566 Stats sub-queries with ≥2 tables; 4 hit timeouts or out-of-memory. For the 146 full queries they match the reference's `Stats_Sizes.csv` exactly, so we have the same data.
- **Rebuilt catalog:** a scratch rebuild on Postgres (60 s) with the catalog defects patched, so each defect's effect could be isolated.
- **Reference implementation:** built from your copy and run on all Stats queries and sub-queries. I used the authors' runtime configuration (c=0.01, 64 histogram buckets, 1028 MCVs), both as-is and without group compression and PK-FK pre-joins. The latter matches our scope.
- **Not done:** I measured bounds against true cardinalities rather than running end-to-end Postgres runtime benchmarks.

## Where the paper and the reference disagree

| topic                    | paper                              | reference                                                                                                              | ours                          |
| ------------------------ | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| equality on an MCV value | that value's own degree sequence   | always the maximum with the non-MCV default (looser)                                                                   | paper                         |
| non-MCV default          | maximum of cumulative sequences    | per-rank maximum of degree sequences (ranked _ascending_, still valid but loose), truncated to the largest value count | broken (issue 9)              |
| range predicates         | smallest enclosing bucket          | 64 equi-depth quantiles plus half-shifted intervals; minimum over all enclosing intervals                              | paper (bisection, 128 leaves) |
| IN                       | sum of degree sequences            | sum of _bounds_ over each value combination (tighter)                                                                  | paper                         |
| LIKE fallback            | rows containing no frequent 3-gram | same                                                                                                                   | rows containing a rare 3-gram |
| cyclic queries           | minimum over spanning trees        | arbitrary cycle break                                                                                                  | paper (crashes, issue 5)      |

On LIKE: the paper's and reference's fallback is unsound, because a matching row can also contain frequent 3-grams. Ours is sound, but patterns shorter than 3 characters and `_` wildcards still hit the rare-3-gram PCF, which is unsafe. Stats has no LIKE predicates.

## A. Inference: alpha/beta tree and FDSB evaluation

**1. Only one join equivalence class is ever evaluated — critical, unsafe.**

- **Where:** `_fdsb.py:618` adds edges to `col` instead of `col.table`. This came from commit `7420c0c` (2026-06-12, "Keep table aliases…"), which replaced `col.table.drop_alias()` with `col`.
- **What happens:** table nodes become isolated, so the graph is a forest of one star per equivalence class. `nx.is_tree` is always false, and the spanning-tree iterator returns the forest. Decomposition then starts at the first leaf and only multiplies the PCFs of that one class.
- **Example:** for q-20 (`ph ⋈ p ⋈ u`) the bound is just `ph.postid × p.id`; the `users` join is ignored.
- **Impact:** all 137 violations of HEAD come from this.

**2. `PiecewiseConstantFn.cumulative_at` is mirrored within each segment — unsafe.**

- **Where:** `_piecewise_fns.py:541` uses `(bound[k] − x)·v` where it should use `(x − bound[k−1])·v`.
  - For f = 4 on (0,1], 2 on (1,3] it returns F(0)=4, F(1)=0, F(3)=4 instead of 0, 4, 8.
- **What happens:** `BetaStep.evaluate_at` uses it to compute F_X0(i), so beta outputs are no longer non-increasing.
  - With a single-segment primary-key projection the output is fully reversed. The beta total still comes out right, but the following alpha step then pairs low degrees with high ones.
- **Evidence:**
  - Random chain queries on exact degree sequences: bounds are 0×–7.7× the true degree sequence bound, with 11/300 below it.
  - Fixing only this gives 0/300 violations.
  - With all other fixes applied, q-8, q-17, q-56, q-57, q-58 and q-138 get a bound of **0**. In q-8, the reversed beta on `posts.id` puts zeros on the ranks that `pl.postid` keeps; the exact bound is 5.4·10⁷ against a true size of 481 420.

**3. Beta outputs are integrated by sampling at integer ranks — sound, but loose.**

- **Where:** `cardinality()` evaluates at `np.arange(n_distinct)` (`_fdsb.py:122`, `:306`).
- **What happens:** this is a left Riemann sum on fractional segment boundaries, so the top rank is counted twice. The paper (and the reference) integrate the segments exactly.
- **Impact:** once issue 2 is fixed this stays an upper bound, because the function is non-increasing, but it is 1.5× looser on average and up to 5× on the 90th-percentile full queries.

**4. `invert_cumulative_at` maps out-of-range totals to rank 0 (`_piecewise_fns.py:568`).**

- **What happens:** when the projection column has more rows than the fact column (e.g. because of NULL join keys), the dimension is evaluated at its maximum degree.
- **Paper and reference:** those ranks should contribute 0. The reference additionally truncates all of a table's join-column PCFs to their smallest cardinality (`rightTruncateByRows`); we don't.
- **Impact:** only loose, not unsafe. It turned out negligible on Stats once the other issues were fixed.

**5. Cyclic queries crash once issue 1 is fixed.**

- **Where:** spanning trees of the table/equivalence-class graph can leave a join node with one neighbour, which hits `assert deg == 2` (`_fdsb.py:497`). Reproduced with `c.userid = u.id AND c.postid = p.id AND p.owneruserid = u.id`.
- **Fix:** prune such dangling join nodes before decomposing.
- **Impact on Stats:** none, since none of its sub-queries are cyclic.
- **Paper vs reference:** the paper takes the minimum over all spanning trees (which we do); the reference breaks cycles arbitrarily.

Otherwise the decomposition (`decompose_acyclic` and the alpha/beta construction) is correct. With the graph fixed and exact integration, it matches the brute-force worst-case instance.

## B. Combining conditioned degree sequences

**6. Conjunctions use the pointwise minimum of the degree sequences instead of the minimum of their cumulative sequences — unsafe.**

- **Where:** `_catalog.py:2266`, and also in the LIKE path (`:1590`, `:1672`) via `min_with`.
- **Why it is wrong:** the paper (Sec. 3.3) explicitly requires taking the minimum of the cumulative degree sequences, because compressed degree functions no longer dominate the true ones pointwise. The reference does this (`pointwiseFunctionMin`).
- **Example (q-15, `comments.userid`):**
  - `creationdate ≥ …` gives 1816 compressed ranks with high degrees; `score = 0` gives 2900 ranks.
  - The pointwise minimum is cut off at 1816 ranks and yields 100 723 rows, but 132 504 rows qualify.
- **Evidence:** 23 violations on the fixed catalog and 10 on the generated catalog; the minimum of cumulative sequences gives 0 in both.

**7. Range predicates on the same column are looked up as separate half-open ranges** and then combined; the reference merges them into one interval.

- **Measured:** merging alone was not tighter on Stats (10.9 → 11.1, geometric mean over all sub-queries).
- **Option:** combining both lookups with the cumulative minimum would be at least as good as either. Not a bug.

**8. Integer MCV lookups always miss after `SafeBoundCatalog.load()`.**

- **Where:** JSON object keys come back as strings, and `make_json_parser` maps `integer` to a no-op (`util.py:119`, `_catalog.py:610`).
- **What happens:** `c.score = 0` never matches its MCV entry and falls back to the non-MCV PCF.
- **Effect:** freshly built and reloaded catalogs give different bounds.

The IN and OR handling (sum of degree sequences) follows the paper.

## C. Which degree sequences get built, and their contents

**9. The non-MCV default is wrong and can under-count (`fetch_non_mcv_ds`, `_catalog.py:495`).**

- **What it computes:** the degree sequence of _all rows_ except the 999 rows with the rarest filter values, using `row_number` over rows in ascending order. The paper's default is the maximum over the non-MCV values of their individual degree sequences.
- **Why it under-counts:** if the rare values hold fewer than 999 rows, rows of frequent values are dropped.
- **Measured:**
  - `pl.linktypeid = 1` gets 10 103 rows; 10 186 qualify.
  - `u.downvotes = 0` gets 39 326; 39 578 qualify.
- **Combined with issue 8,** this is what every loaded integer equality predicate uses today.

**10. Conditioned degree sequences count NULL join keys as a value.**

- **Where:** `fetch_correlated_ds` uses `GROUP BY on` without `on IS NOT NULL` (`_catalog.py:432`).
- **What happens:** `votes.userid` is 89% NULL, so it gets a "degree" of up to 293k.
- **Inconsistency:** current PostBOUND excludes NULLs from the unconditioned PCFs. The shipped `generated/` catalog includes them there too, e.g. `votes.userid` totals 328 064 rows instead of 34 789.
- **Impact:** the reference excludes NULLs. Counting them inflates 190 sub-queries, by up to 2.3×.

**11. The histogram hierarchy collapses when the first value holds more than half the rows (`_catalog.py:1404`).**

- **Why:** `cutoff_idx == 0` means no split at all. In Stats the first value is often NULL or 0.
- **Affected columns (root bucket only):** `posts.answercount`, `viewcount`, `favoritecount`, `users.upvotes`, `downvotes`, `views`, `votes.bountyamount`.
- **Related:** NULLs are also folded into the first range bucket (`IS NULL OR <= hi`), although range predicates never match NULL.
- **Impact:** splitting after the heavy value and excluding NULLs halves the full-query geometric mean (391 → 195). This is the largest tightness factor among the catalog fixes.

**12. Compression (`valid_compress`)** is a faithful Algorithm 1 implementation. It is always valid: 8–17 segments for the Stats foreign keys, self-join inflation 1.06–1.15. The reference deviates from the paper in three ways:

- It measures error with the true degree at the compressed position instead of the segment slope.
- It snaps segment boundaries to the ends of runs of equal degrees.
- It compresses unconditioned PCFs at c/4.

At c=0.01 this gives fewer segments and inflation of 1.13–1.27; at c/4 the inflation is 1.05–1.14. The effect on bounds is modest.

## E. Other findings

- **Nothing runs on PostBOUND 0.23 (the latest commit's bump).**
  - `_Predicate2PCF.visit_and/or/not_predicate` still expect `components`, so every AND-filtered query raises `TypeError`.
  - `catalog_from_workload` passes a query to `visit_query_predicates`, which now expects the query's predicates.
  - All measurements above used scratch adapters for these.
- **Dead and broken helpers:** `cut_at` crashes, and `PiecewiseLinearFn.invert` and `__call__` are wrong; none are called.
- **`safebound-accuracy.csv`:** its 0.0 bounds don't reproduce with either catalog under current code, so the file predates later fixes. The older `models/` catalog also gives `posts.id` 91 975 rows, one fewer than the table has.

# Suggested fix order

- [x] Issue 1: connect edges to `col.table`.
- [x] Issue 2: correct `cumulative_at`.
- [x] Issue 6: combine conjunctions by the minimum of cumulative sequences.
- [x] Issues 8 and 9: MCV key parsing and the non-MCV default.
- [x] Issues 10 and 11: NULL keys and the histogram split.
- [x] Issue 3: exact segment-wise integration.
- [x] Issues 4 and 5: saturating inverse and pruning dangling join nodes.
- [x] The PostBOUND 0.23 visitor adapters.

After that, regenerate the catalog. The existing `generated/` catalog bakes in issues 10 and 11.
