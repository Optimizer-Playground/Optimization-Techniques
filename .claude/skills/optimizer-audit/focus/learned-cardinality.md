# Focus: learned cardinality estimation (MSCN)

The optimizer's output is a cardinality estimate per (sub-)query. The critical question is:
**does a defect change the estimates, and on which kinds of queries?**

## Checklist

**Query featurization**

- One set element per table, join and predicate, as in the paper? Look for merging of elements
  (e.g. several predicates on the same column collapsed into one).
- Operator encoding: one-hot as in the paper, and are combinations produced that never occur in
  training?
- Value normalization per column type: numeric (constant columns → division by zero), text (sort
  order of the database collation vs numpy), dates, LIKE patterns.
- Masks: does each mask equal the number of rows actually emitted, not the number of raw
  predicates?
- Padding sizes: what happens with more tables, joins or predicates than the vocabulary size?
- Unsupported predicates (`IN`, `BETWEEN`, `IS NULL`, ...): skipped as documented, or do they
  crash?
- Sub-queries the optimizer will ask about (base tables, cross products): are they in the training
  distribution?

**Output scaling**

- Label transform and its exact inverse (`log` vs `log1p`, `exp` vs `expm1`).
- Source of the min/max label bounds (observed training labels vs a heuristic range).

**Network and training**

- Layer-by-layer comparison with the reference (set MLPs, pooling, output MLP, sigmoid).
- Hyperparameters against the paper's stated values: learning rate, epochs, batch size, hidden size.
  Check `git log` for commits that changed defaults.
- Loss computed on unnormalized values as in the reference; shuffling; validation split.

**Training-data generation** (only if the scope names the generator as a focus area)

- Distribution of the generated training queries against the paper's: number of joins (including
  single-table queries), repeated filter columns, how literals are chosen, what timeouts drop.
- Does the sampler run on the current PostBOUND version?

**Persistence**

- Featurizer store/load reproduces identical tensors; model loading works on CPU-only machines.

## Recommended measurements

- Train ours and the reference on **the same labelled queries** (if the reference's labels do not
  reproduce on our database, use the reference's own data and refit our encoders to its value
  ranges). Disable reference features that are non-goals for us (e.g. sample bitmaps), and also
  report the full reference for context.
- Metrics: q-error median, p90, p95, max and geometric mean on the target workload, plus validation
  q-error on held-out training-distribution queries (which isolates training from featurization).
- Break the q-error down by query feature that a defect touches (e.g. queries with closed ranges vs
  the rest), so each fix's effect is visible where it applies.
- At least 3 seeds per configuration; report seed variance, which is large for MSCN.
