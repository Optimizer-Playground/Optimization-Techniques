# Evidence standards

These rules come from the Bao, MSCN and SafeBound audits. Each of them caught a problem that
reading the code alone would have missed or misjudged.

## What counts as evidence

Every issue in the report rests on at least one of:

- **a measurement**: a script in the artifacts directory that reproduces the behaviour, with its
  numbers in the report;
- **a code reading with a concrete trace**: a `file:line` reference plus a worked example of the
  wrong value it produces (e.g. "for f = 4 on (0,1], 2 on (1,3] it returns F(1) = 0 instead of 4").

Label which one it is. An issue that is only suspected, not confirmed, either gets investigated
further or is listed as unconfirmed; it never appears as a finding.

Every claimed impact ("changes the selected arm", "makes the bound unsafe") needs a measurement.
Mechanism alone is not impact.

## Measure what the user actually runs

- **Start with an "as shipped" row**, produced through the real entry points: default constructors,
  factory methods (`online()`, `load_or_build`, `infer_from_*`), the shipped training loop.
  Several critical defects lived only in those paths (a factory that crashed on IMDB, normalization
  bounds of `log(0)`), while the core algorithm was fine.
- **Look for silent failures.** A model that produces `NaN` everywhere and then always selects
  index 0 looks exactly like a working conservative optimizer. Check intermediate values (features,
  losses, predictions), not only final outputs.
- **Exercise persistence.** Store, reload and compare outputs. Past audits found file names that
  `store` and `load` disagree on, JSON keys that come back as strings, and plans that cannot be
  parsed back.
- If a crash prevents measurement, work around it minimally in the harness, document the workaround,
  and report the crash as its own issue.

## Compare against the reference on identical data

- Run the reference implementation yourself instead of comparing against numbers in the paper.
  Patch it only as much as needed to run (Python/numpy compatibility, a seed flag, a switch for a
  feature our implementation excludes), and list every patch.
- Feed both implementations **the same data**. When the reference's labels do not reproduce on our
  database (e.g. a different data snapshot), use the reference's own data for both and refit our
  encoders to it.
- When the paper and the reference disagree, measure the reference as published and, if relevant,
  the reference with the paper's variant.

## Isolate each defect

- Apply fixes as monkeypatches in the harness, never in the repository.
- Add them cumulatively, one per row (V1 = as shipped + fix A, V2 = V1 + fix B, ...), so that each
  row's delta is attributable. Where two fixes interact, also measure them separately.
- Use the same folds, seeds and query sequences for every row.

## Separate signal from noise

- Use at least 3 seeds (more when the result is close) and report mean ± sd.
- **Establish the noise floor** before interpreting a difference: run-to-run repetition of
  measurements, and agreement of the same model with itself across seeds. A difference that is
  within the noise floor is reported as such.
- Prefer metrics that are robust at small sample sizes over end-to-end totals:
  pairwise ranking accuracy instead of total runtime, geometric means and quantiles instead of
  arithmetic means, violation counts for bounds.
- Where possible, build an **independent oracle**: an exact evaluator, a brute-force computation on
  small instances, true cardinalities for all sub-queries. Check the oracle itself first.

## Common defect classes to test deliberately

- Unit mismatches (seconds vs milliseconds), especially across module boundaries.
- `log(0)`, division by zero, integer overflow in products of table sizes, `inf/inf`.
- Features that mean different things at training time and at inference time.
- Outputs that depend on batch composition or padding.
- Inputs never seen in training (unknown categories, operator combinations, value ranges).
- Failures, timeouts and skipped items that silently disappear from training data.
- Off-by-one and mirrored formulas in piecewise or cumulative computations.
- Defaults that differ from the paper's stated hyperparameters; check `git log -S` for when they
  changed.
- README claims about the paper or the reference that are themselves wrong.
