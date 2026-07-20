# Lecture 07 demo implementation blueprint

Status: implementation-ready design handoff; the Lecture 07 core/bonus narrative has passed independent verification. This document authorizes only demo implementation, not Assignment 07 redesign.

## Accepted demo role

The required demonstrations must teach one coherent progression:

1. begin with a question, audience, intended claim, displayed unit, grain, and variable roles before choosing a chart;
2. critique and repair visual integrity and accessibility defects;
3. construct the five core static chart types through explicit Matplotlib `Figure`/`Axes` objects; and
4. use one bounded pandas or seaborn exploratory view before rebuilding one focused explanatory chart with annotation, export, a text alternative, and human visual QA.

The demos consume supplied prepared tables. They do not clean, join, reshape, group, aggregate, resample, correlate, regress, model, or fetch network data.

## Exact package

Replace the current required demo package atomically with:

```text
07/demo/
├── .gitignore
├── .python-version
├── DEMO_GUIDE.md
├── requirements.txt
├── data/
│   ├── followup_summary.csv
│   ├── participant_scores.csv
│   └── program_progress.csv
├── demo1_critique_redesign.ipynb
├── demo2_figure_axes.ipynb
└── demo3_explore_explain.ipynb
```

Delete the three legacy notebooks, paired same-stem Markdown copies, `data_cleaning_viz_demo.md`, and committed generated Altair/interactive/scatter/damped-sine artifacts. Generated images and supporting tables belong under ignored `07/demo/output/`; no generated output is a teaching input.

Use exact candidate records:

```text
.python-version: 3.12.13
requirements.txt:
numpy==2.0.2
pandas==3.0.3
matplotlib==3.10.8
seaborn==0.13.2
```

Jupyter hosting, kernel support, notebook-execution tools, and image-inspection tooling used only by certification are platform/test dependencies rather than lecture imports.

## Pinned prepared fixtures

All three CSVs are course-authored, synthetic, complete, non-identifying, and fixed as literal bytes during implementation. Compute and record their SHA-256 values in the guide and notebooks; no placeholder may remain.

### `followup_summary.csv`

Grain: one row per program and observation period. This is a supplied descriptive summary, not evidence that either program caused a change.

```csv
program,period,follow_up_percent
Standard,Before,68
Standard,After,69
Reminder,Before,67
Reminder,After,74
```

### `program_progress.csv`

Grain: one row per program and study round.

```csv
program,round_number,score
Standard,1,62
Standard,2,65
Standard,3,67
Standard,4,70
Standard,5,72
Guided,1,61
Guided,2,66
Guided,3,71
Guided,4,75
Guided,5,79
```

### `participant_scores.csv`

Grain: one row per participant.

```csv
participant_id,program,practice_hours,score
S01,Standard,2.0,61
S02,Standard,3.0,65
S03,Standard,3.5,66
S04,Standard,4.0,67
S05,Standard,5.0,69
G01,Guided,2.5,64
G02,Guided,3.5,70
G03,Guided,4.0,73
G04,Guided,5.0,77
G05,Guided,6.0,82
```

Each notebook must search upward for its committed fixture(s), reconstruct the exact supplied bytes only when no committed source is present, verify checksums before parsing, and distinguish a missing standalone source from corruption. A corrupted committed fixture must stop execution rather than silently fall back. Repository-root, `07/demo/`, standalone, and nested launch directories must behave equivalently. No manual upload, Drive mount, credential, mutable date, randomness, or network fetch is allowed.

## Notebook-wide contract

Every notebook must have:

- a portable `Python 3` kernelspec, stable globally unique cell IDs, null execution counts, and zero stored outputs;
- a first Markdown cell stating the question/audience/claim role, Colab-first/local-Jupyter equivalence, ephemeral filesystem, privacy rule, fresh-execution rule, and the assignment-Colab pilot boundary;
- one supplied setup cell that conditionally installs only mismatched course packages before the first course-package import, then prints and asserts the candidate versions;
- a deterministic fixture/bootstrap cell with checksum verification;
- explicit output-directory creation and deterministic replacement of stale outputs;
- executable properties for data, chart objects, labels, units, baselines, encodings, annotations, paths, and file dimensions; and
- a final verification cell that can pass only after all prior source executes freshly.

The notebooks should display charts during instruction. Certification may select a headless Matplotlib backend externally, but teaching code must not rely on stored rendering or suppress warnings.

Automated assertions may check observable properties. They must not claim to certify honesty, accessibility, aesthetics, or communicative success; each notebook must distinguish those human judgments explicitly.

## Demo 1: critique and redesign

Canonical filename: `demo1_critique_redesign.ipynb`.

Use `followup_summary.csv`. Before plotting, state:

- question: how the observed follow-up percentages differ by program and period;
- audience: a clinic operations group deciding what deserves follow-up;
- bounded claim: the Reminder rows show a larger observed before-to-after increase in this prepared summary;
- grain/unit: one row and one bar per program-period prepared percentage; and
- roles: program and period are categorical; follow-up percentage is quantitative.

Construct an intentionally flawed grouped bar chart in code with all five documented defects:

1. causal title unsupported by the descriptive table;
2. truncated magnitude baseline that exaggerates length differences;
3. missing percentage unit;
4. program identity encoded by color alone; and
5. distracting decoration or excessive chart furniture.

Ask learners to identify each defect before revealing the repair. Then build one corrected grouped bar chart using an explicit Figure/Axes object, a zero baseline, complete title/labels/unit/context, a colorblind-safe palette plus hatch as redundant program encoding, a clearly associated legend, and a descriptive rather than causal title. Add exact value labels only if they remain uncluttered.

Provide a concise text alternative naming chart type, axes, the observed 1-point versus 7-point changes, and the causal limitation. Save only the corrected chart as `output/followup_redesign.png` at a declared size/DPI. Assertions must cover four fixture rows, exact values, zero lower y-limit, four bars, labels/units, hatch diversity, noncausal title, text-alternative limitation, and a readable nonempty PNG. A human checklist covers comparison fidelity, contrast, legibility, clutter, and claim fit.

## Demo 2: Figure/Axes fundamentals

Canonical filename: `demo2_figure_axes.ipynb`.

Use `program_progress.csv` and `participant_scores.csv`. Define Figure and Axes before using `fig`/`ax`, and state the relevant grain and variable roles before each chart. Construct exactly the five required types without introducing additional chart families:

1. line: prepared score across ordered rounds for two programs, using color plus marker/line style;
2. bar: a supplied literal two-row mean-score table, with a zero baseline;
3. scatter: practice hours versus observed score, using color plus marker for program and making no causal/correlation claim;
4. histogram: participant score with explicit fixed bin edges and count units; and
5. box plot: score by program with median/quartile/IQR/whisker meaning stated and points beyond whiskers described as observations to investigate, not automatic errors.

Do not independently compute the two program means through GroupBy; use the exact supplied prepared values `65.6` and `73.2` as a literal bounded teaching table. Each chart uses one explicit Figure and Axes, complete units, and deterministic ordering. Close figures only after their visible/in-memory instructional checks.

Save one compact `output/core_chart_types.png` gallery only if the implementation can retain readable labels at the declared dimensions; otherwise save only the line chart and treat the other Figure objects as live demonstration state. Do not create five graded-looking artifacts. Assertions must cover exact source shapes, two line identities, redundant encodings, zero bar baseline, two scatter collections, exact histogram edges and observation conservation, two box/median artists, all labels/units, and output readability.

## Demo 3: exploratory to explanatory

Canonical filename: `demo3_explore_explain.ipynb`.

Use the supplied long-form `participant_scores.csv` for one bounded seaborn exploratory scatter and `program_progress.csv` for the final explanatory line chart. No correlation, trend line, uncertainty interval, density estimate, or model is allowed.

The workflow is:

1. restate the exploratory question and row/mark grain;
2. make one seaborn scatter with program represented by both hue and marker style;
3. record only a descriptive observation and explicitly avoid causal or inferential language;
4. restate the final question, coordinator audience, intended descriptive claim, and program-round prepared-summary grain;
5. rebuild the two score paths through Matplotlib Figure/Axes;
6. use color plus marker/line style, full labels and units, a descriptive title, and one annotation for the exact seven-point final-round separation;
7. write `output/explanatory_supporting_data.csv` from the exact plotted columns with `index=False`;
8. write `output/explanatory_text_alternative.txt` containing chart type, axes, main pattern, and causal limitation; and
9. save `output/program_progress_explanatory.png`, then perform executable and human visual QA.

Assertions must cover exact source/output schemas and values, two line identities and redundant encodings, annotation text/position, labels/units, legend, noncausal title, text-alternative limitation, PNG readability, and exact supporting-data readback. The human checklist must cover question/audience/claim fit, chart choice, scale/context, accessibility, annotation usefulness, layout, and whether the text alternative is meaningful rather than copied boilerplate.

## Guide and publication contract

`DEMO_GUIDE.md` must identify the exact three notebooks, objectives, inputs/checksums, outputs, expected visible results, launch paths, instructor prompts, likely failure modes, destructive/repeat-run rehearsal, privacy, and scope boundaries. It must link each notebook through a development Colab badge and state that every badge must move to one immutable release tag and be fresh-run before publication.

The guide's certification table starts with local candidate, fresh Colab, and immutable badge reference all pending. Authorship or stored output is not independent certification.

## Independent QA matrix

After implementation, a reviewer who did not author the notebooks must verify:

- actual fresh-kernel execution of all three notebooks from repository-root, nested `07/demo/`, and standalone layouts;
- a separate progressive execution with lecture warnings promoted to errors;
- each fixture missing and each committed fixture corrupted;
- stale/deleted/corrupt generated outputs and deterministic repeat replacement;
- exact chart-object, label, unit, baseline, redundant-encoding, annotation, supporting-data, and text-alternative contracts;
- readable exported dimensions and a human visual smoke check without clipping;
- portable metadata/state/IDs, exact dependencies, guide claims/checksums, no paired Markdown, and no committed generated artifacts; and
- absence of cleaning/join/reshape/GroupBy/aggregation/pivot/time-series/correlation/regression/modeling/network/upload/Drive scope leakage.

Fresh Colab execution and immutable release-tag badges remain separate publication gates even after local independent QA passes.
