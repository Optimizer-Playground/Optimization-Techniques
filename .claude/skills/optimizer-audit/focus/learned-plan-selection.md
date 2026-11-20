# Focus: learned plan selection (Bao, FASTgres)

The optimizer's output is a choice among candidate plans or hint sets. The critical question is
always: **does a defect change which candidate gets selected?**

## Checklist

**Action space**

- Do our hint sets (arms) match the reference's, including how many are used by default?
- Are the selected hints reproduced faithfully when the plan is executed (re-hinting round trip)?

**Plan featurization**

- Which plan nodes are kept (full plan vs joins and scans only) and how unary nodes are binarized.
- Operator vocabulary: what happens on an operator never seen during fitting
  (`handle_unknown`)?
- Environment features (e.g. cache state): **is the value computed the same way at training time
  and at inference time?** Compare them on identical plan nodes.
- Cardinality and cost features: source (estimates vs actuals) and consistency between training and
  inference.

**Input and output scaling**

- Normalization bounds: where do they come from (fixed, fitted on a sample, refit per retrain)?
  Look for `log(0)`, integer overflow in products of table sizes, and ranges that compress the
  actual values into a small part of the interval.
- Label transform and its inverse; units of the labels (seconds vs milliseconds) along every path
  that writes experience.

**Network and training**

- Layer-by-layer comparison with the reference (layer sizes, normalization, activations, pooling).
- Optimizer, learning rate, batch size, loss, epochs, early stopping, re-initialization per retrain.
- **Batch dependence:** does a candidate's prediction change with the other candidates in its batch
  (padding, normalization over padded positions)? Compare batched with one-by-one scoring.

**Selection and the feedback loop**

- What happens before the first model is trained (cold start)? The paper and reference fall back to
  the default optimizer.
- Behaviour on `NaN` or tied predictions.
- Experience window size, retraining trigger and frequency, exploration (e.g. Thompson sampling via
  bootstrapping).
- Timeouts and failed executions: do they enter the experience, and with what label?
- Persistence: store → load → continue retraining; experience round trip.

**Training-data generation** (the optimizer's own calibration path; external generators only if
the scope names them)

- Does offline calibration data contain plans from all arms, in the target query shape, in the right
  units, with operators the featurizer knows?

## Recommended measurements

- Record a full **query × arm matrix** once (plan-time plans, plan-time environment state, analyzed
  plans, warm-up and measured runs, timeouts). Every later experiment
  replays from it, so the expensive executions happen only once.
- Baselines from the matrix: default optimizer total, per-query oracle, best single non-default arm.
- **Offline:** query-disjoint k-fold cross-validation, several seeds, identical folds for every
  variant. Metrics:
  - pairwise ranking accuracy within a query, counting only pairs whose runtimes differ by > 1.2×
    (robust at small sample sizes);
  - selected total ÷ default optimizer (mean ± sd);
  - number of queries whose selection changes between batched and one-by-one scoring;
  - arm agreement across seeds (noise floor) and with the reference.
- **Online:** replay the feedback loop on the recorded matrix with a fixed random query sequence
  (as the reference's driver script does), only the selected arm's outcome entering the experience.
  Report warm-up and post-warm-up phases separately, plus the arm histogram.
