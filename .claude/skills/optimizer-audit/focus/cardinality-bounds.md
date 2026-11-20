# Focus: cardinality upper bounds (SafeBound, UES)

The optimizer's output is an upper bound on a (sub-)query's cardinality. The critical questions,
in this order:

1. **Soundness:** is the bound ever below the true cardinality? Any such case is critical.
2. **Tightness:** how far above the true cardinality is it, compared with the reference?

## Checklist

**Statistics selection**

- Which degree sequences / statistics are built, for which columns (join keys, filter columns,
  MCVs, histograms, n-grams)?
- NULL handling: are NULL join keys counted as a value? Do range buckets include NULLs?
- Defaults for values not covered by MCVs: is the default a valid upper bound for every such value?
- Histogram hierarchies: do they degenerate on skewed columns (e.g. a heavy first value)?

**Compression**

- Is the compressed function valid (dominates the true cumulative sequence)? Compare against the
  paper's algorithm and note where the reference deviates from it.

**Combining conditioned statistics**

- Conjunctions: the minimum of **cumulative** sequences, not the pointwise minimum of degree
  sequences (the latter is unsafe after compression).
- Disjunctions and `IN`: sums; `LIKE` fallbacks: is the fallback sound for every pattern shape
  (short patterns, `_` wildcards)?
- Several predicates on the same column (ranges): merged or combined?

**Inference**

- Graph construction: are all join equivalence classes and all tables connected? Is the result
  really a tree for acyclic queries?
- Decomposition and evaluation steps (e.g. alpha/beta): check formulas for cumulative functions
  and their inverses segment by segment, including out-of-range inputs.
- Integration: exact over segments, or sampled (sound but loose, or unsound)?
- Cyclic queries: minimum over spanning trees (paper) vs arbitrary cycle break (reference); do they
  crash?

**Persistence**

- Built vs reloaded catalog give identical bounds (JSON key types, e.g. integer MCV keys returned as
  strings).

## Recommended measurements

- **True cardinalities** for every connected sub-query with ≥ 2 tables (DuckDB), checked against any
  ground truth the reference ships.
- **Violations:** count and worst ratio of bound ÷ true < 1, over all sub-queries.
- **Tightness:** geometric mean of bound ÷ true over full queries and over all sub-queries; ours ÷
  reference per query (10th–90th percentile).
- **Independent oracle:** an exact implementation of the paper's bound algorithm, validated against
  brute force on explicitly constructed worst-case instances (e.g. random chain queries). Use it to
  separate inference defects from statistics defects.
- Rebuild the catalog in the harness with individual defects patched, so each catalog defect's
  effect can be isolated; do not rely only on a pre-generated catalog, which may bake in old defects.
- Run the reference both in the authors' configuration and restricted to our scope (non-goals
  disabled).
