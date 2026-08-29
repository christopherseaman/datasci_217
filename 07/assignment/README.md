# Assignment 07: Visualization Critique, Redesign, and Explanation

Use prepared synthetic data to move through three visualization roles: inspect
a bounded pattern, diagnose a misleading supplied chart, and communicate one
descriptive finding to a named audience. You will connect each chart to its
question, row grain, variable roles, displayed unit, and evidentiary limit.

This is a clean-local-Jupyter assignment. The fixtures are course-authored,
synthetic, nonidentifying, and different from the Lecture 07 demo data. Do not
use Colab, manual uploads, Drive mounts, network access, or `/content` paths.
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

Use CPython 3.12.13. From this directory, create and activate a virtual
environment, install the exact runtime record, and open Jupyter or the VS Code
notebook interface:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Complete
[PLATFORM_CHECK.md](PLATFORM_CHECK.md) before editing the notebook. The Python
program that starts Jupyter and the notebook kernel must use this environment.

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

Complete every `TODO` in `assignment.ipynb`. Restart the kernel and run all 23
cells from top to bottom. Commit the notebook and these five regenerated files
in the assignment repository:

- `output/critique_redesign.png`
- `output/pathway_explanatory.png`
- `output/explanatory_supporting_data.csv`
- `output/visualization_evidence.json`
- `output/explanatory_text_alternative.txt`

These files are intentionally visible in VS Code Source Control and GitHub
Desktop. Commit and push all six deliverables. Do not edit the data fixtures,
supplied notebook cells, environment records, checker, or instructions. Stored
notebook output is useful for human review but is not trusted as execution
evidence; the central grader clears and executes a disposable copy from fresh
state.

After restart-and-run, use the discoverable student check:

```bash
python check_assignment.py
```

The checker inspects files and notebook source without executing notebook code.
Fix each `[FIX]` message, restart and run all, then check again. It screens
machine-readable requirements; it cannot certify that a chart is clear,
accessible, honest, or visually effective.

## Task 1: bounded exploration

State an exploratory question, the one-session row/mark grain, variable roles,
one observation restricted to the twelve supplied rows, and a limitation that
rejects causal and generalized conclusions. Implement
`build_exploratory_chart(session_table, pathway_order)` as one scatterplot of
activities completed against reflection score. Encode pathway with both color
and marker shape, preserve the caller's two-label order, label units, and
return its Figure and Axes without saving a third PNG. Display and inspect the
live result.

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

Export the exact evidence JSON and matching one-paragraph text alternative.
The alternative must name the line chart, both axes and units, both pathways,
the first-to-last pattern, the nine-point final gap, and the limitation that
prepared descriptive rows cannot establish cause. Finish the visual-review
checklist with observable evidence rather than yes/no answers.

## Scope and assessment boundary

Use the prepared rows directly with Matplotlib and one bounded seaborn
scatterplot. GroupBy, aggregation and summary calculations; joining, reshaping
and cleaning; time series; modeling or inference; random or remote data;
interactive charts, dashboards, maps, animations, and additional chart
families are outside scope.

The implementation has a provisional 80-point automated overlay: 10 points for
fixtures and reproducibility, 15 for Task 1, 25 for Task 2, 25 for Task 3, and
5 for artifact integrity. A separate 20-point human review covers contract fit,
visual integrity, accessibility, annotation, text alternative, organization,
and limitations. Course policy will decide how this diagnostic evidence maps
to a grade; the notebook and public checker declare no pass threshold.
