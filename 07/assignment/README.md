# Assignment 07: Visualization Critique, Redesign, and Explanation

Use prepared synthetic data to move through three visualization roles: inspect
a bounded pattern, diagnose a misleading supplied chart, and communicate one
descriptive finding to a named audience. You will connect each chart to its
question, row grain, variable roles, displayed unit, and evidentiary limit.

Complete and submit the local notebook and its saved artifacts. Graders read
those artifacts without rerunning the notebook. The
fixtures are course-authored, synthetic, nonidentifying, and different from the
Lecture 07 demo data. Do not use Colab, manual uploads, Drive mounts, network access, or `/content` paths.
The portable setup supports both a standalone exported assignment repository
and this full course repository.

## Core vocabulary

A **visualization** maps data values to visible properties so a reader can make
a comparison. The **question** states what the chart should help the reader
compare or understand. The **audience** is who will use it and the context they
bring. An **intended claim** is the bounded descriptive conclusion the final
chart should support. The **displayed unit** names the magnitude reported by an
axis or mark, while the **grain** says what one row and corresponding mark or
position represents.

In this assignment, variables have four roles: **categorical** values identify
groups, **quantitative** values report numerical magnitudes, **ordered** values
have a meaningful sequence, and an **identifier** distinguishes one record.
An **exploratory visualization** is a truthful view used to inspect a pattern
while refining a question. An **explanatory visualization** is a focused chart
that communicates one selected finding to a named audience.

A **mark** is a visible point, line, or rectangle. An **encoding** maps a value
to position, length, color, marker, hatch, or line style. A **redundant
encoding** adds a second cue for the same important category identity. **Visual
integrity** means visible comparisons faithfully represent the data, scale,
context, and claim. **Accessibility** means design choices let more readers
recover the comparison. A Matplotlib **Figure** is the complete saved canvas;
an **Axes** is one plotting area with scales, labels, title, and marks. An
**annotation** attaches focused context to a selected mark or position. A
**text alternative** names the chart, axes, main pattern, and relevant
limitation in text.

## Setup

Use CPython 3.14. From this directory, create and activate a virtual
environment, and install the exact runtime record. If you use the notebook,
open it through Jupyter or the VS Code notebook interface:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Complete
[PLATFORM_CHECK.md](PLATFORM_CHECK.md) before preparing artifacts. If you run
the notebook, its kernel must use this environment.

## Prepared fixtures

- `format_completion.csv`: four rows, one per delivery format and stage;
  categorical format and stage with quantitative prepared completion percent.
- `session_observations.csv`: twelve rows, one per synthetic learning session;
  identifier, categorical pathway, and two quantitative fields.
- `pathway_checkpoints.csv`: eight rows, one per pathway and checkpoint;
  categorical pathway, ordered checkpoint, and quantitative prepared
  completion percent.

The rows describe only the prepared fixtures. They do not establish cause,
population effects, prediction, or general patterns. Do not generate, clean,
join, reshape, or aggregate these complete assignment fixtures.

## Deliverables

Complete every `TODO` in `assignment.ipynb` and commit the notebook source.
Create and commit these six artifact files in the assignment repository:

- `output/exploratory_spec.json`
- `output/critique_redesign.png`
- `output/pathway_explanatory.png`
- `output/explanatory_supporting_data.csv`
- `output/visualization_evidence.json`
- `output/explanatory_text_alternative.txt`

These files are intentionally visible in VS Code Source Control and GitHub
Desktop. Commit and push the artifacts and notebook source. Do not
edit the data fixtures, supplied notebook cells, environment records, checker,
or instructions. Automated grading reads the committed artifacts directly;
notebook execution is optional local QA.

After creating the committed artifacts, use the discoverable student check:

```bash
python check_assignment.py
```

The checker inspects files and committed artifacts without executing notebook code.
Fix each `[FIX]` message, regenerate the artifacts, then check again. It screens
machine-readable requirements; it cannot certify that a chart is clear,
accessible, honest, or visually effective.

## Task 1: bounded exploration

State an exploratory question, the one-session row/mark grain, variable roles,
one observation restricted to the twelve supplied rows, and a limitation that
rejects causal and generalized conclusions. Implement
`build_exploratory_chart(session_table, pathway_order)` as one Altair
scatterplot specification of activities completed against reflection score.
Use typed quantitative positions and nominal pathway color and point-shape
encodings, preserve the caller's two-label order, label units, and include
tooltips. Export `exploratory_chart.to_dict()` as
`output/exploratory_spec.json`, embedding the plotted session rows. It is the
machine-readable Task 1 milestone; no third PNG is required.

## Task 2: critique and redesign

Inspect the supplied four-bar comparison for a learning-support coordinator.
It intentionally has an unsupported causal title, truncated baseline, missing
unit, color-only category encoding, and distracting decoration. Explain each
problem and a repair without changing the prepared values.

Implement `build_critique_redesign(summary_table, format_order, stage_order)`.
Use a zero baseline, explicit percentage unit, course colors plus hatches,
value labels, restrained decoration, and an outside legend. Preserve arbitrary
valid caller labels and order. Save the canonical result as
`output/critique_redesign.png`.

## Task 3: audience-focused explanation

State the question, learning-support coordinator audience and follow-up use,
bounded intended claim, unit, grain, roles, comparison, chart rationale, and
causal limitation. Copy and export the exact supporting data. Implement
`build_explanatory_chart(checkpoint_table, pathway_order)` so the two ordered
paths have redundant color, marker, and line-style cues. Derive the leader,
checkpoint count, final absolute gap, title, and annotation from any valid
two-pathway input; on a final tie, attach the annotation to the second requested
pathway. Save the canonical result as `output/pathway_explanatory.png`.

Export the evidence JSON and a matching text alternative. Semantic fields and
fixture values are machine-checked; prose quality, chart clarity,
accessibility, and visual integrity are reviewed by a person. Finish the
visual-review checklist with observable evidence rather than yes/no answers.

## Scope and assessment boundary

Use the prepared rows directly with one bounded Altair scatterplot and
Matplotlib for the supplied bar and explanatory line charts. GroupBy, aggregation and summary calculations; joining, reshaping
and cleaning; time series; modeling or inference; random or remote data;
dashboards, maps, animations, and additional chart families are outside scope.
Altair tooltips are permitted in Task 1, but must supplement visible labels and context.

Automated grading totals 100 points: 15 for fixtures and reproducibility, 20
for Task 1, 30 for Task 2, 30 for Task 3, and 5 for artifact integrity. There
are no separate human-review points; contract fit, accessibility, annotations,
text alternatives, organization, and limitations remain required deliverables.

## Public automated grading

grading.py is the shared ruleset for students, pytest, and graders. Run
python check_assignment.py [submission_dir], or add --json for a
datasci217/grading-result/v1 result. It reads committed artifacts only, never
notebooks or submission code. The visible tests award fixtures (15), Task 1
(20), Task 2 (30), Task 3 (30), and artifact integrity (5).
