---
name: optimizer-audit
description: Audit a PostBOUND optimizer implementation in postbound_extensions/ for logical correctness against its original paper and reference implementation, with measured evidence on real databases.
argument-hint: <optimizer directory, e.g. bao>
disable-model-invocation: true
---

# Optimizer audit

Audit the implementation in `postbound_extensions/$ARGUMENTS/`.

An audit answers one question: **does our implementation compute what the paper (or the reference
implementation) computes, and where it does not, how much does that change the optimizer's output?**
It is not a code-style review. Minor discrepancies are listed briefly, not analyzed.

Take your time. Thoroughness matters more than speed: investigate every suspicious behaviour until
it is either confirmed with evidence or ruled out.

## 1. Scope

1. Check that both the paper (a PDF in the optimizer directory) and the reference implementation
   (`reference-implementation/`) are present. If either is missing, **stop and ask the user how to
   proceed** before doing anything else, including drafting a scope file.
2. The audit covers the optimizer's own code: its directory, plus shared modules it imports and
   calls on its own code path (e.g. `postbound_extensions/preprocessing/`), as far as it uses them.
   **Standalone tools are out of scope** unless the scope file names them as a focus area: the
   query sampler (`tools/`, `postbound_extensions/sampler/`), training-data generators, and
   PostBOUND itself. If you notice a defect there in passing, mention it in one line under
   "Other findings" without investigating it or adding it to the fix list.
3. Read `postbound_extensions/$ARGUMENTS/AUDIT-SCOPE.md`.
   - If it does not exist, draft one from [scope-template.md](scope-template.md), the optimizer's
     README and its paper, write it to that path, and **stop and ask the user to confirm it**
     before doing any further work.
   - If it exists, proceed without asking.
4. Read the optimizer's README. Everything listed as a non-goal is **out of scope**, as is anything
   the scope file excludes. Documented deviations from the paper are in scope only insofar as the
   README's description of them is wrong or they cause a defect elsewhere.
5. Load the focus checklist for the scope's `family` from `focus/` (see the list below). The scope
   file's own focus items come first; the checklist adds what past audits found worth checking.
6. If a previous audit report exists (`<optimizer>-audit.md` in the optimizer directory), read it.
   Re-verify every item marked as fixed in its fix list, and say in the new report which ones still
   reproduce. Items listed under "Not addressed" are settled decisions; do not raise them again
   unless their stated reason no longer holds.

Focus checklists:

- `learned-plan-selection` → [focus/learned-plan-selection.md](focus/learned-plan-selection.md) (Bao, FASTgres)
- `learned-cardinality` → [focus/learned-cardinality.md](focus/learned-cardinality.md) (MSCN)
- `cardinality-bounds` → [focus/cardinality-bounds.md](focus/cardinality-bounds.md) (SafeBound, UES)
- anything else → [focus/generic.md](focus/generic.md)

## 2. Sources

Read, in this order:

1. the paper (the PDF in the optimizer directory), fully, including the experimental section,
   because hyperparameters and defaults are often only stated there;
2. the reference implementation (`reference-implementation/`), following the code path that
   produced the paper's results, not every script in the repository;
3. our implementation, end to end, starting from the entry points a user actually calls (default
   constructors, `load_or_build`, factory methods), not only the core algorithm.

While reading, keep a running list of every point where **paper and reference disagree**. The report
must contain this table even if our implementation follows one of them consistently.

Their presence was checked in step 1. Do not substitute your own recollection of the paper for the
actual text.

## 3. Evidence

Follow [evidence-standards.md](evidence-standards.md) for what counts as evidence, and
[environments.md](environments.md) for databases, workloads and known traps.

The core rules:

- **Never modify the repository.** Apply fixes as monkeypatches in scratch harnesses.
- **Store every script and dataset in a persistent artifacts directory** (see environments.md), not
  in a temporary scratchpad. The report names that directory, but never contains machine-specific
  paths (see environments.md).
- **Measure "as shipped" first**, through the real entry points, before measuring any fix.
- **Run the reference implementation on the same data** whenever that is feasible. It is the baseline
  that tells you whether a gap is a defect or inherent to the method.
- **Isolate each defect**: add fixes one at a time and report each step as its own row.

## 4. Report

Write the report to `<optimizer>-audit.md` in the optimizer directory, following
[report-template.md](report-template.md) exactly. If a report with that name already exists, write to
`<optimizer>-audit-<YYYY-MM-DD>.md` instead of overwriting it.

Before finishing, check the report against the template's checklist at its end.
