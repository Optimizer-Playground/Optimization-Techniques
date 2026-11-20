# Report template

Earlier reports that follow this structure: `postbound_extensions/bao/bao-audit.md`,
`postbound_extensions/mscn/mscn-audit.md` and `postbound_extensions/safebound/safebound-audit.md`.
They predate this skill, so they start with an "Audit Prompt" section instead of "Audit Scope".
Read one of them before writing if you are unsure about tone or detail.

## Audience and style

- Written for the maintainers of the implementation: they know the code and the paper, but did not
  watch the audit happen.
- Lead with conclusions. Every section starts with its most important sentence.
- Short paragraphs, bullets for parallel facts, tables for comparisons.
- Concrete numbers instead of adjectives ("4× worse geometric mean", not "much worse").
- Reference code as `file.py:line`, relative to the optimizer directory.

## Structure

```markdown
# Audit Scope

<Verbatim copy of AUDIT-SCOPE.md, plus: the date, the commit audited (`git rev-parse --short HEAD`),
and any deviation from the scope that turned out to be necessary, with the reason.>

# Audit Report

<Headline: 1–3 paragraphs. Does the implementation work as shipped? What are the few defects that
account for most of the gap? How does it compare to the reference once they are fixed?>

<One line: "I didn't modify the repository." plus the name of the artifacts directory
(not its absolute path).>

## Results on <workload>

<The main results table. Rows: "as shipped", then cumulative fixes (one per row), then the
reference implementation on the same data, then paper numbers if relevant (in italics).
Columns: the metrics from the focus checklist. Footnotes define any non-obvious metric.>

<A few bullets interpreting the table: what is signal, what is within noise.>

## How the evidence was produced

- **Data / measurements:** what was measured, on which system and configuration, how often,
  and the measured noise floor.
- **Ours:** which entry points were used, and every harness change.
- **Reference:** how it was run, with every patch listed.
- **Not done:** what was not measured and why.

## Where the paper and the reference disagree

| topic | paper | reference | ours |
| ----- | ----- | --------- | ---- |

<Bullets on the impact of the disagreements that matter.>

## A. <Area> … ## F. Other findings

<Group issues by area, in the order of the scope's focus list. Number issues consecutively across
the whole report. Each issue:>

**N. <Claim, stated as the defect> — <severity>.**

- **Where:** `file.py:line`, and the commit that introduced it if known.
- **What happens:** the mechanism, with a concrete worked example.
- **Evidence:** measured numbers, or "by inspection" with a trace.
- **Impact:** what changes in the optimizer's output, quantified.
- **Fix:** the suggested change, in one or two sentences.

<"Otherwise X matches the paper/reference" paragraphs are welcome: they record what was verified.>

# Suggested fix order

- [ ] Issue N: <fix in one line>
- [ ] ...
- [ ] Minor: <collected minor items>

<Follow-up actions that are not code fixes, e.g. "retrain stored models", "regenerate the catalog".>

## Not addressed

<Left empty by the auditor. When the maintainers decide not to fix an issue, they move it here
from the fix list, with the reason (e.g. "out of scope: standalone tool", "accepted limitation of
PostBOUND"). Later audits treat these as settled.>
```

## Severity labels

- **critical**: the optimizer produces wrong or meaningless output on the target workload as shipped
  (crash, NaN, unsafe bound, heavily degraded accuracy).
- **high**: measurably changes the optimizer's output (selected plan, estimate, bound) beyond noise.
- **moderate**: changes internal behaviour or output within noise, or only in configurations not
  measured.
- **minor**: inconsistency without measurable effect; listed briefly, not analyzed.

## Checklist before finishing

- [ ] The scope section names the commit and the date.
- [ ] The results table has an "as shipped" row and a reference row on the same data, or says why not.
- [ ] Every issue has Where / What happens / Evidence / Impact / Fix, and a severity.
- [ ] No issue concerns a README non-goal or a scope exclusion.
- [ ] Every impact claim is measured or explicitly labelled as not measured.
- [ ] Differences within the noise floor are described as such.
- [ ] The paper-vs-reference table is present.
- [ ] The artifacts directory exists and contains a README; the report names it.
- [ ] The report contains no machine-specific paths (home directories, `/tmp`, user names).
- [ ] The repository is unmodified apart from the report (`git status`).
