# Assignment 07 Blueprint: From Prepared Data to Accessible Claim

Status: implementation-ready design handoff. This document authorizes a future
Assignment 07 rebuild only; it does not implement or self-certify the student
package. Clean local Jupyter is the required execution path. Assignment Colab
launch, save-back, and submission remain conditional on the course pilot.

## Evidence audit and replacement decision

The legacy Assignment 07 package is not a viable source to retain alongside a
new assignment.

- Its student notebook prescribes broad plotting galleries and six image files
  rather than assessing a question, audience, claim, displayed unit, or
  variable roles.
- Its generator creates mutable-date, random customer, product, and transaction
  data. The assignment then joins those tables even though Lecture 07 consumes
  prepared plotting data.
- The notebook requires correlation, a heatmap, time conversion, GroupBy,
  rolling, resampling, pie charts, a dashboard-like overview, and broad seaborn
  work. Those requirements belong to later lectures or bonus material.
- The tests check mostly that files exist and have broad byte sizes. They cannot
  distinguish an honest chart from a blank or misleading image, do not execute
  chart-object contracts, and do not assess accessibility or explanatory fit.
- The package duplicates the assignment and generator as paired notebook and
  Markdown sources, uses broad lower-bound dependencies, pins a different
  Python in a legacy GitHub Classroom workflow, and downloads mutable remote
  tests.

Replace the package atomically. Do not retain the old generator, paired
Markdown, `.github/` workflow/tests, sales/customer/product fixtures, six-image
checklist, or any legacy output as a second source of truth.

## Fixed role and assessment boundary

Assignment 07 is a practical competence assessment with exactly three tasks
that build in this order:

1. state the contract for a bounded exploratory view and inspect prepared
   session rows without making an inferential or causal claim;
2. critique one course-supplied flawed chart and rebuild it with truthful scale,
   units, restrained design, and redundant encoding; and
3. state a final question/audience/claim contract, construct one accessible
   annotated explanatory chart, and export its exact supporting data and text
   evidence.

The assignment assesses the accepted Lecture 07 capabilities:

- question, audience, intended descriptive claim, displayed unit, grain, and
  categorical/quantitative/ordered/identifier roles;
- exploratory versus explanatory purpose;
- marks and encodings, including redundant encoding;
- Matplotlib Figure/Axes construction and one bounded seaborn scatter;
- visual integrity, bar-baseline reasoning, labels, units, context, and
  descriptive rather than causal language;
- annotation, deterministic export, a text alternative, and human visual QA;
  and
- restart-and-run-all execution from checksum-pinned prepared data.

It does not assess cleaning, missing-data decisions, joins, concatenation,
reshape, GroupBy, aggregation, summary construction, `pivot_table`, dates,
time-series work, correlation, regression, inference, uncertainty, prediction,
modeling, network data, random data, mutable dates, dashboards, animation,
interactive plotting, or performance engineering. The fixtures are already
prepared, complete, and in the plotting grain students need.

## Student repository contract

The future student-facing package must contain exactly this instructional
surface, plus Classroom50-owned metadata added by the delivery system:

```text
07/assignment/
├── .gitignore
├── .python-version
├── PLATFORM_CHECK.md
├── README.md
├── assignment.ipynb
├── check_assignment.py
├── requirements.txt
├── data/
│   ├── fixture.json
│   ├── format_completion.csv
│   ├── pathway_checkpoints.csv
│   └── session_observations.csv
└── output/
    └── .gitkeep
```

The instructor repository may additionally contain `_grader_selftest/` with a
production-grader mirror, adversarial harness, exact grader dependencies, and
maintenance notes. That directory is not copied to student repositories or
production submissions.

There is one canonical notebook and no paired same-stem Markdown source, data
generator, `.github/` grader, solution, or completed starter output.

### Runtime records

`.python-version` contains exactly:

```text
3.12.13
```

`requirements.txt` contains only these deliberate assignment imports:

```text
numpy==2.0.2
pandas==3.0.3
matplotlib==3.10.8
seaborn==0.13.2
```

Do not install pandas 3.0.4. Jupyter, ipykernel, nbformat, nbclient, image
inspection, and grader libraries are host or grader tooling rather than student
runtime imports.

The instructor-only grader candidate additionally pins
`nbclient==0.10.2`, `nbformat==5.10.4`, `ipykernel==6.29.5`, and
`Pillow==12.3.0` alongside the four student runtime packages. These are grader
requirements, not imports students must use in the notebook.

The `.gitignore` release contract is:

```text
.ipynb_checkpoints/
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
```

It must not ignore `output/`, PNG, CSV, JSON, or text files. All five required
outputs must appear in VS Code Source Control or GitHub Desktop so students can
commit them. The required Git path remains GUI-first; no command-line Git task
or command is assessed.

`PLATFORM_CHECK.md` contains the clean-local-Jupyter setup, kernel/interpreter
verification, GUI commit/push, Classroom50 feedback, and resubmission workflow.
It is operational guidance, not a graded visualization task. It must not add a
Colab badge or claim that edits made in Colab are saved to the repository.

## Exact assignment-only prepared fixtures

All fixture files use UTF-8, comma delimiters, LF line endings, the exact column
order shown, and a final newline. They are course-authored, synthetic,
nonidentifying, complete, and distinct from all accepted Lecture 07 demo
fixtures and values. Students do not generate, clean, join, reshape, or
aggregate them.

`data/fixture.json` has this exact semantic content:

```json
{
  "fixture_set_id": "a07-visualization-v1",
  "provenance": "Course-authored synthetic learning-format, session, and pathway records; no real or identifying data.",
  "files": [
    {
      "path": "format_completion.csv",
      "row_grain": "one row per delivery format and stage",
      "row_count": 4,
      "columns": ["format", "stage", "completion_percent"],
      "sha256": "20ad900633154f5f3a2c09cfbc2f890f8423da0897d6345841745332110be66a"
    },
    {
      "path": "pathway_checkpoints.csv",
      "row_grain": "one row per learning pathway and checkpoint",
      "row_count": 8,
      "columns": ["pathway", "checkpoint_number", "completion_percent"],
      "sha256": "ec9a336b7fb97418a6f058704f2509c8cee6b13d744efb7a6e3e99224ef8c258"
    },
    {
      "path": "session_observations.csv",
      "row_grain": "one row per synthetic learning session",
      "row_count": 12,
      "columns": ["session_id", "pathway", "activities_completed", "reflection_score"],
      "sha256": "fc4d69ab836288a2fe9c505c65c08413e137e51ab1914cd0e350f6e6636da096"
    }
  ]
}
```

The implementation must compute and freeze the manifest file's own template
hash after final JSON formatting. No placeholder may remain.

### `format_completion.csv`

- Grain: one row per delivery format and stage.
- Roles: `format` and `stage` are categorical;
  `completion_percent` is quantitative.
- Canonical orders: `Recorded`, `Live`; then `Start`, `Finish`.
- Rows: 4; bytes: 98.
- SHA-256:
  `20ad900633154f5f3a2c09cfbc2f890f8423da0897d6345841745332110be66a`.

```csv
format,stage,completion_percent
Recorded,Start,81
Recorded,Finish,77
Live,Start,82
Live,Finish,80
```

The supplied flawed chart uses these exact values. Recorded changes from 81%
to 77%; Live changes from 82% to 80%; Live finishes three percentage points
higher. The table is descriptive and cannot establish that delivery format
caused the difference.

### `session_observations.csv`

- Grain: one row per synthetic learning session.
- Roles: `session_id` is an identifier; `pathway` is categorical;
  `activities_completed` and `reflection_score` are quantitative.
- Rows: 12; bytes: 309.
- SHA-256:
  `fc4d69ab836288a2fe9c505c65c08413e137e51ab1914cd0e350f6e6636da096`.

```csv
session_id,pathway,activities_completed,reflection_score
I01,Independent,1,52
I02,Independent,2,58
I03,Independent,2,61
I04,Independent,3,65
I05,Independent,4,69
I06,Independent,5,73
F01,Facilitated,1,55
F02,Facilitated,2,62
F03,Facilitated,3,67
F04,Facilitated,3,70
F05,Facilitated,4,75
F06,Facilitated,5,81
```

The bounded exploratory chart may describe the visible co-variation in these
twelve supplied rows. It may not calculate a correlation, fit a line, report
significance or uncertainty, claim cause, predict, or generalize.

### `pathway_checkpoints.csv`

- Grain: one row per learning pathway and checkpoint.
- Roles: `pathway` is categorical; `checkpoint_number` is ordered;
  `completion_percent` is quantitative.
- Canonical pathway order: `Independent`, `Facilitated`.
- Rows: 8; bytes: 181.
- SHA-256:
  `ec9a336b7fb97418a6f058704f2509c8cee6b13d744efb7a6e3e99224ef8c258`.

```csv
pathway,checkpoint_number,completion_percent
Independent,1,58
Independent,2,63
Independent,3,67
Independent,4,70
Facilitated,1,57
Facilitated,2,65
Facilitated,3,72
Facilitated,4,79
```

Independent rises from 58% to 70%; Facilitated rises from 57% to 79%.
Facilitated begins one percentage point lower and finishes nine points higher.
These are supplied prepared summaries, not evidence of a causal pathway effect.

## Portable protected setup

The first code cell is supplied and protected. It must:

1. search upward from the launch directory for either flattened
   `data/fixture.json` or course-root `07/assignment/data/fixture.json`;
2. define `ASSIGNMENT_ROOT`, `DATA_DIR`, and the three fixture paths from the
   discovered manifest; define `OUTPUT_DIR` plus the five exact protected
   output-name literals relative to that discovered assignment root;
3. require the exact fixture-set ID, manifest keys, relative file inventory,
   row counts, ordered columns, safe paths contained in `DATA_DIR`, final
   newlines, and SHA-256 values before pandas reads a CSV;
4. create only `OUTPUT_DIR` and delete only the five named stale outputs before
   student code executes, preserving unrelated files;
5. import and assert Python 3.12.13, NumPy 2.0.2, pandas 3.0.3, Matplotlib
   3.10.8, and seaborn 0.13.2, then define course blue as `#0072B2` and course
   orange as `#D55E00`; and
6. stop with an actionable exception when a version, manifest, fixture, or path
   contract fails.

It must not install packages inside the assignment notebook, embed fallback
fixture bytes, access a network, use an absolute repository path, inspect a
mutable date, prompt for an upload, mount Drive, or rewrite a fixture. The
assignment requires the committed files and local environment. A corrupted or
missing fixture must stop before plotting.

The student load cell reads exact dtypes:

- all identifiers and categorical labels: pandas `string`;
- `checkpoint_number`, `activities_completed`, `reflection_score`, and both
  `completion_percent` fields: `int64`.

It asserts the exact three shapes and column orders without changing the data.

## Definition-before-use contract

The README and protected `a07-terms-data` cell must define these terms in plain
language before any student is asked to use them independently:

| Term | Required meaning before Task 1 |
|---|---|
| visualization | a mapping from data values to visible properties for a reader comparison |
| question | what the chart should help the reader compare or understand |
| audience | who will use the chart and the context they bring |
| intended claim | the bounded descriptive conclusion the final chart should support |
| displayed unit | what magnitude or quantity the axis/mark reports |
| grain | what one row and corresponding mark/position represents |
| categorical, quantitative, ordered, identifier | the four variable roles used in these fixtures |
| exploratory visualization | a truthful view used to inspect a pattern while refining a question |
| explanatory visualization | a focused chart communicating one selected finding to a named audience |
| mark | a visible point, line, or rectangle |
| encoding | a mapping from a value to position, length, color, marker, hatch, or line style |
| redundant encoding | a second visual cue for the same important category identity |
| visual integrity | visible comparisons faithfully represent the data, scale, context, and claim |
| accessibility | design choices that let more readers recover the comparison |
| Figure / Axes | the saved canvas / one plotting area with its scales, labels, title, and marks |
| annotation | focused context attached to a selected mark or position |
| text alternative | text naming the chart, axes, main pattern, and relevant limitation |

The supplied flawed chart appears only after these definitions. The task prompt
may preview later terms, but no unfamiliar API or concept may be required in a
student-editable cell without this definition and the accepted Lecture 07
demos.

## Exact notebook contract

`assignment.ipynb` uses portable `Python 3` kernelspec metadata, notebook-format
major version 4 and minor version 5, the exact 23 cells below, stable globally
unique IDs, null execution counts, and zero stored output in the released
starter. Protected cells are complete; student cells contain actionable TODO
scaffolds but no solution. There is no hidden prerequisite cell.

The header states that clean local Jupyter is required, Assignment Colab is not
yet supported, fixtures contain no real data, students must restart and run all,
stored output is not trusted by automation, and the five generated files are
separate tracked artifacts. Students may retain ordinary freshly rendered
notebook output because a human rubric reviews charts and explanations; the
central grader nevertheless clears and fresh-executes a disposable copy.

### Cells 0–3: orientation, definitions, and load

0. `a07-header` (protected Markdown): title, three-task progression, local-only
   platform boundary, privacy, exact output list, restart/run-all, and GUI Git
   visibility rule.
1. `a07-setup` (protected code): portable discovery, manifest/checksum/version
   validation, imports/constants, exact stale-output removal, and path creation.
2. `a07-terms-data` (protected Markdown): the term ledger above plus compact
   fixture dictionaries, grains, roles, and evidentiary limits.
3. `a07-load` (student code): explicit-dtype reads into `format_completion`,
   `session_observations`, and `pathway_checkpoints`; exact shape/column checks.

### Cells 4–8: Task 1 — contract and bounded exploration

4. `a07-task1-contract` (student Markdown): state what the exploratory chart is
   for, what one row/mark represents, and why it cannot by itself support a
   causal, inferential, predictive, or generalized conclusion.
5. `a07-task1-evidence` (student code): assign nonempty
   `exploration_question`, `exploration_observation`, and
   `exploration_limitation`, plus these exact machine-readable values:

   ```python
   exploration_grain = "one row per synthetic learning session"
   exploration_roles = {
       "session_id": "identifier",
       "pathway": "categorical",
       "activities_completed": "quantitative",
       "reflection_score": "quantitative",
   }
   ```

   The observation remains descriptive and refers only to the twelve supplied
   rows. The limitation rejects cause and generalization without calculating a
   correlation or inferential quantity.
6. `a07-explore-function` (student code): define
   `build_exploratory_chart(session_table, pathway_order)`. It makes one
   defensive copy; constructs one Figure/Axes; calls exactly one bounded
   `sns.scatterplot` with `activities_completed` on x,
   `reflection_score` on y, and `pathway` encoded by both hue and marker style;
   uses the caller's two-label order with the course palette and `o`/`s`
   markers; labels the axes `Activities completed (count)` and
   `Reflection score (points)`; uses title
   `Exploratory view of activities completed and reflection score` and legend
   title `Pathway`; mutates no input; and returns `(figure, axes)` without
   saving an image.
7. `a07-explore-run` (student code): call the function with
   `['Independent', 'Facilitated']`; verify one Axes, twelve points, two colors,
   two marker shapes, exact labels, legend, and source immutability; display it
   as live exploratory state but create no third PNG.
8. `a07-task1-reflection` (student Markdown): explain how the question, grain,
   roles, and visible marks constrain the observation and why the view is not an
   explanatory causal claim.

### Cells 9–14: Task 2 — supplied flawed chart and redesign

9. `a07-task2-context` (protected Markdown): state the supplied comparison,
   coordinator audience, prepared percentage unit, one-format-stage row/bar
   grain, variable roles, and bounded descriptive interpretation.
10. `a07-supplied-flawed` (protected code): build and display the course-supplied
    grouped bar chart from `format_completion` with exactly these visible
    defects:

    1. title `Live delivery caused stronger completion`;
    2. y limits `(76, 83)`, truncating the magnitude baseline;
    3. no y-axis label or percentage unit;
    4. format identity represented by color alone; and
    5. pale-yellow Figure background plus heavy grids on both axes.

    It uses four bars in canonical order and creates no required output file.
    Protected assertions prove the defects are actually present; the chart is
    not merely described in prose.
11. `a07-task2-critique` (student Markdown): explain how each defect could
    mislead or exclude a reader and what a repair must accomplish without
    changing the prepared values.
12. `a07-critique-evidence` (student code): create `critique_entries`, exactly
    five dictionaries in this category order, each with exact keys
    `category`, `problem`, and `repair` and nonempty student-authored text:

    ```text
    unsupported claim
    truncated baseline
    missing unit
    color-only encoding
    distracting decoration
    ```

13. `a07-redesign-function` (student code): define
    `build_critique_redesign(summary_table, format_order, stage_order)`. It
    copies its input, uses one explicit Figure/Axes and four grouped bars,
    preserves caller-provided label order, starts the y-axis at zero, labels
    the axes `Stage` and `Prepared completion (%)`, uses title
    `Prepared completion by delivery format and stage`, uses the course
    blue/orange plus `//` and `\\` hatches, draws exact `81%`, `77%`, `82%`,
    and `80%` value labels on canonical data, removes top/right spines and heavy
    grids/background, and uses a frameless `Delivery format` legend at
    `loc='upper left'`, `bbox_to_anchor=(1.01, 1)`. It mutates no input and
    returns `(figure, axes)`.
14. `a07-redesign-run` (student code): call the function with the canonical
    table and orders, check the exact 81/77/82/80 bar values, zero lower limit,
    labels/unit/title, four bars, two hatches, four value labels, and legend;
    write `output/critique_redesign.png` at Figure size 7.4 by 4.4 inches,
    150 DPI, `bbox_inches='tight'`, replacing any prior file.

### Cells 15–21: Task 3 — explanatory chart and evidence export

15. `a07-task3-contract` (student Markdown): in the student's own words, state
    the final question, learning-support coordinator audience, intended bounded
    claim, displayed unit, grain, roles, comparison, why a line chart matches
    ordered checkpoints, and why the supplied prepared paths do not establish
    cause.
16. `a07-final-contract-values` (student code): assign nonempty `question`,
    `audience`, and `intended_claim`, plus these exact values:

    ```python
    displayed_unit = "prepared completion percent"
    plotting_grain = "one row per pathway and checkpoint"
    variable_roles = {
        "pathway": "categorical",
        "checkpoint_number": "ordered",
        "completion_percent": "quantitative",
    }
    pathway_order = ["Independent", "Facilitated"]
    ```

    The question asks about comparison across four checkpoints. The audience
    names a learning-support coordinator and an actual follow-up use. The claim
    states that Facilitated begins one point lower and finishes nine points
    higher in the prepared summary without using causal, inferential, or
    predictive wording.
17. `a07-supporting-data` (student code): create
    `explanatory_supporting_data` as a copy of the exact plotted columns in the
    exact canonical row order. Assert schema, eight rows, unique
    pathway/checkpoint pairs, pathway order, checkpoint order, and unchanged
    source; write `output/explanatory_supporting_data.csv` with UTF-8,
    `index=False`, `lineterminator='\n'`, and final newline. Exact readback uses
    the same explicit dtype map as the load cell before comparing values and
    dtypes; byte/checksum comparison independently proves serialization.
18. `a07-explanatory-function` (student code): define
    `build_explanatory_chart(checkpoint_table, pathway_order)`. It copies input;
    filters and sorts each supplied pathway by `checkpoint_number`; constructs
    one 8 by 4.8 inch Figure/Axes; draws two paths with the course colors and
    redundant `o`/solid versus `s`/dashed encodings; labels checkpoint and
    prepared completion percent; uses a descriptive data-derived title and
    a `Pathway` legend; sets x ticks to the shared sorted checkpoint values;
    computes the last-checkpoint absolute gap from the supplied rows; annotates
    the higher final point (the second requested pathway on a tie); removes
    top/right spines; mutates no input; and returns
    `(figure, axes, annotation)`. The canonical rows therefore annotate the
    final Facilitated point; no canonical label, checkpoint, value, or gap is
    the function's only logic.
19. `a07-explanatory-run` (student code): call the function with the exact
    supporting table and order. The canonical chart has two lines with x values
    `[1, 2, 3, 4]`, y values `[58, 63, 67, 70]` and `[57, 65, 72, 79]`, exact
    title `Facilitated finishes higher in the prepared 4-checkpoint summary`,
    labels `Checkpoint` and `Prepared completion (%)`, annotation text
    `Checkpoint 4 observed gap: 9 percentage points` attached at `(4, 79)`, and
    a visible legend. Save `output/pathway_explanatory.png` at 150 DPI with
    `bbox_inches='tight'`, replacing any prior file.
20. `a07-evidence-export` (student code): assign one-paragraph
    `explanatory_text_alternative` and construct `visualization_evidence` with
    exactly this shape and no extra keys:

    ```json
    {
      "schema": "datasci217/a07-visualization-evidence/v1",
      "question": "<student text>",
      "audience": "<student text>",
      "intended_claim": "<student text>",
      "displayed_unit": "prepared completion percent",
      "grain": "one row per pathway and checkpoint",
      "variable_roles": {
        "pathway": "categorical",
        "checkpoint_number": "ordered",
        "completion_percent": "quantitative"
      },
      "exploration": {
        "question": "<student text>",
        "grain": "one row per synthetic learning session",
        "variable_roles": {
          "session_id": "identifier",
          "pathway": "categorical",
          "activities_completed": "quantitative",
          "reflection_score": "quantitative"
        },
        "observation": "<student text>",
        "limitation": "<student text>"
      },
      "critique": [
        {"category": "unsupported claim", "problem": "<student text>", "repair": "<student text>"},
        {"category": "truncated baseline", "problem": "<student text>", "repair": "<student text>"},
        {"category": "missing unit", "problem": "<student text>", "repair": "<student text>"},
        {"category": "color-only encoding", "problem": "<student text>", "repair": "<student text>"},
        {"category": "distracting decoration", "problem": "<student text>", "repair": "<student text>"}
      ],
      "text_alternative": "<student text>"
    }
    ```

    Build every field from the named fresh-run variables rather than duplicating
    a second literal answer inside the export cell.
    Write it to `output/visualization_evidence.json` with UTF-8,
    `ensure_ascii=False`, two-space indentation, deterministic insertion order,
    and a final LF using `newline='\n'` or binary bytes. Write the same
    text-alternative value plus exactly one final LF to
    `output/explanatory_text_alternative.txt` with the same cross-platform
    newline control. The text
    alternative must name the line chart, both axes/units, both pathways, the
    first-to-last pattern and nine-point final gap, and the descriptive/causal
    limitation. Fresh readback must equal the in-memory values exactly.
21. `a07-visual-review` (student Markdown): answer the human checklist for
    contract fit, chart choice, integrity, accessibility, annotation, text
    alternative, layout, and remaining limitation. It must not merely state
    `yes`; each response identifies observable evidence in the exported chart.

### Cell 22: supplied final verification

22. `a07-final-verify` (protected code): recheck exact fixture state, all
    executable chart-object proxies, required filenames, PNG signatures and sane
    dimensions, exact supporting CSV bytes, JSON/text readbacks, and source
    immutability. It prints a concise local readiness message, explicitly says
    human visual review and central grading are still required, and instructs
    restart/run-all followed by `python check_assignment.py`. It does not award
    points or claim that automation certified accessibility, honesty, or clarity.

## Exact function interfaces and behavioral variation

The three public functions are:

```python
build_exploratory_chart(session_table, pathway_order)
build_critique_redesign(summary_table, format_order, stage_order)
build_explanatory_chart(checkpoint_table, pathway_order)
```

They must accept any valid prepared table with the documented schema and two
distinct caller-supplied category labels. For this behavioral contract,
"valid prepared" means complete and nonmissing, with the documented column
roles and integer numeric fields, and with no duplicate key at its stated
grain. The session table contains exactly the two requested pathways and at
least one row for each. The redesign table contains the complete unique
two-format by two-stage set of four combinations named by its two order
arguments. The checkpoint table contains exactly the two requested pathways,
at least two checkpoints per pathway, and the same unique checkpoint set for
both pathways. These are caller preconditions, not an invitation to clean
invalid input.

The functions must not depend on canonical IDs, labels, row counts, exact
values, or result direction; mutate input; read files; save files; use global
canonical DataFrames; or perform cleaning, grouping, aggregation, joining,
reshape, statistical calculation, or modeling. File writing belongs in the run
cells so grader-owned calls can inspect alternate Figure/Axes objects without
creating stray artifacts.

Discoverable alternate grader tables change labels, row order, checkpoint
count, numeric values, and which pathway finishes higher while preserving the
documented schemas and complete prepared-data contract. The explanatory title
must use `<leader> finishes higher in the prepared <count>-checkpoint summary`
or `Both pathways finish equally in the prepared <count>-checkpoint summary`
as applicable, deriving both the label and numeric count. Its annotation must
compute the last checkpoint and absolute between-pathway gap from its argument
and attach to the higher final point, using the second requested pathway on a
tie, rather than contain the canonical `4`, `79`, or `9` as its only logic.
This general rule yields the exact canonical title and annotation in cell 19.
The redesign must plot its argument's four values and provided orders rather
than the canonical bytes.

## Required output and deterministic-evidence contract

Completed submissions contain exactly these five regenerated filesystem
artifacts in addition to the notebook. Ordinary embedded `image/png` notebook
display output is not a third PNG file and is ignored as execution evidence:

| Path | Exact machine contract | Human role |
|---|---|---|
| `output/critique_redesign.png` | public signature/IHDR and byte/dimension screening plus central full decode; canonical fresh Figure has four exact bars, zero baseline, labels, hatches, and legend | critique repair, legibility, contrast, clutter, overlap, claim fit |
| `output/pathway_explanatory.png` | public signature/IHDR and byte/dimension screening plus central full decode; canonical fresh Figure has two exact paths, redundant styles, labels, legend, and exact annotation | audience/claim fit, accessibility, annotation usefulness, layout |
| `output/explanatory_supporting_data.csv` | exact 181 bytes and SHA-256 `ec9a336b7fb97418a6f058704f2509c8cee6b13d744efb7a6e3e99224ef8c258` | confirms the final chart's disclosed prepared values |
| `output/visualization_evidence.json` | exact schema/key topology, values equal the fresh namespace, deterministic serialization, final newline | question/audience/claim, critique rationale, limitations |
| `output/explanatory_text_alternative.txt` | exact fresh variable value plus one final newline; content-component checks | meaningful alternative for the final chart |

PNG byte hashes are not fixed across operating systems because fonts,
FreeType/libpng/zlib, configuration, rendering, and metadata can vary. In two
fresh kernels on one fixed grader image, stale replacement and repeat runs must
preserve dimensions and observable chart properties; exact hashes should also
repeat there. The checker reads the PNG signature and
IHDR through the standard library rather than accepting file extension or byte
size alone. Each image must be between 800 and 2,000 pixels wide, between 450
and 1,400 pixels high, and between 10,000 and 2,000,000 bytes. These executable
proxies reject corrupt, tiny, or implausibly dimensioned exports; they cannot
certify that visible content is nonblank or meaningful. Fresh Figure/Axes
inspection and human review own those judgments.

The starter contains only `output/.gitkeep`. Completed outputs are intentionally
not ignored and must be visible in GUI Git. Setup removes only the five named
stale outputs, and rerunning recreates them without deleting unrelated files.

## Protected and student-editable surfaces

Implementation freezes course-owned hashes for:

- `.python-version`, `requirements.txt`, `.gitignore`, `README.md`, and
  `PLATFORM_CHECK.md`;
- `check_assignment.py`;
- `data/fixture.json` and all three CSV fixtures; and
- notebook cells `a07-header`, `a07-setup`, `a07-terms-data`,
  `a07-task2-context`, `a07-supplied-flawed`, and `a07-final-verify`, including
  ID, type, position, and source.

Only the designated notebook cells and five regenerated output files are
student work. Course staff regenerate protected hashes only when intentionally
publishing a new assignment version. The central grader independently owns
expected hashes; editing the public checker cannot weaken production checks.

A public checker cannot securely prove its own integrity from an inline
constant. It may report the template's recorded checker hash as a diagnostic,
but trusted checker-integrity enforcement belongs to Classroom50's central
grader.

## Student-visible public checker

`check_assignment.py` uses only the Python standard library. It must not import
pandas, Matplotlib, seaborn, nbclient, or execute arbitrary notebook source. It
must:

1. locate flattened and course-root layouts portably from `__file__`;
2. validate exact manifest semantics, safe paths, fixture inventory, byte
   hashes, schemas, row counts, and final newlines;
3. parse valid notebook JSON and require exactly the 23 IDs/types/order above,
   portable kernelspec metadata, globally unique IDs, and unedited protected
   cells; submitted execution counts and ordinary stored chart output are
   ignored rather than accepted as execution evidence;
4. detect untouched TODO scaffolds and the three exact public function names;
5. parse student-editable code with `ast` and reject later-scope calls/imports,
   network/random/mutable-date data, dashboards/interactive libraries,
   uploads/Drive, absolute paths, embedded fixture fallbacks, and unexpected
   output paths;
6. require exactly the five output paths plus the template `.gitkeep`, reject
   legacy `q1_`/`q2_`/`q3_` images and unexpected generated artifacts, parse
   both PNG signatures/IHDR, and enforce the stated byte/dimension bounds;
7. verify the supporting CSV exact bytes and checksum;
8. validate the JSON's exact topology, ordered critique categories, scalar
   types, exact machine-readable grain/unit/roles, nonempty authored fields,
   deterministic final newline, and text-file equality; and
9. return nonzero with a small task-grouped set of actionable messages or print
   one readiness summary without claiming a score or visual-quality pass.

AST checks apply to executable student code, not raw Markdown, so legitimate
prompts and explanations of deferred concepts do not trigger false positives.
The checker is preparation feedback, not the production grader, and never
trusts notebook output as execution evidence.

The explicit student-code scope exclusions include:

- `groupby`, `agg`, `aggregate`, `transform`, and `pivot_table`; plus the exact
  summary-producing call-name denylist `corr`, `corrwith`, `cov`, `describe`,
  `mean`, `median`, `sum`, `std`, `var`, `sem`, `quantile`, `nunique`, and
  `value_counts`; the grader may vary values but may not infer additional
  forbidden method names from prose;
- merge/join/concat, `melt`, `pivot`, filling, dropping, deduplication,
  replacement, type-cleaning pipelines, or datetime conversion;
- resample, rolling, expanding, EWM, shift, lag construction, or calendar work;
- scipy, statsmodels, scikit-learn, formula/model APIs, uncertainty or
  significance calculations;
- Altair, Plotly, Bokeh, Holoviews, Panel, Streamlit, animation, maps,
  dashboards, or browser interaction;
- `sns.load_dataset`, requests/urllib/http clients, random generators,
  `datetime.now`, credentials, `/content`, upload prompts, or Drive mounts; and
- pie, heatmap, KDE, regression, violin, pair, joint, or other unrequested chart
  families.

## Classroom50 central grader

Classroom50 is the course-wide delivery system, not a Lecture 04-only system.
The production Assignment 07 grader is teacher-controlled and discoverable; it
contains no solution, credential, confidential record, or test whose value
depends on secrecy. It must not import or trust the editable public checker.

The grader must:

1. independently validate all protected files, fixture bytes, notebook cells,
   exact inventory, output visibility policy, and scope;
2. copy the submission to an isolated temporary directory, remove all five
   outputs, clear submitted execution counts/outputs, append grader-owned checks
   to a disposable notebook copy, and execute from a fresh pinned kernel with
   `MPLBACKEND=Agg` set by the grader for headless rendering;
3. exercise flattened Classroom50, course-root, relocated, nested-working-
   directory, and path-with-spaces layouts without changing the notebook;
4. inspect live Figure/Axes/Annotation objects, input immutability, labels,
   units, titles, scales, values, mark counts, hue/marker/hatch/line-style
   redundancy, legend presence, and canonical annotation; seaborn marker-shape
   checks deduplicate path geometry rather than assume one stored path per
   distinct shape;
5. call all three functions on discoverable alternate complete prepared tables
   with different category labels, row order, checkpoint count, and values,
   including a reversed final leader and a final tie;
6. validate exact canonical supporting CSV bytes, JSON topology and namespace
   equality, text-file equality, full PNG decoding/dimensions with Pillow,
   stale replacement, and clean repeat properties in a second fresh kernel;
   separately plant an unrelated sentinel to prove setup preserves it, while
   still requiring final submission inventory to reject that extra file;
7. run missing-fixture, corrupted-fixture, malformed-notebook, stored-output,
   protected-edit, and corrected-resubmission cases;
8. prove that every committed required artifact equals the artifact regenerated
   by central fresh execution; then use Classroom50's official `REVIEW_URL` to
   inspect the committed notebook Markdown and artifacts for the human rubric,
   without inventing an external review directory, storage service, or bundle;
   and
9. write `./result.json` and produce a grading log with actionable failure
   detail.

The official result object uses the hyphenated
`classroom50/result/v1` contract:

```json
{
  "schema": "classroom50/result/v1",
  "classroom": "...",
  "assignment": "...",
  "submission": "...",
  "commit": "...",
  "release": "...",
  "review": "...",
  "datetime": "...",
  "score": 0,
  "max-score": 80,
  "tests": [
    {
      "test-name": "template and fixture integrity",
      "passed": false,
      "score": 0,
      "max-score": 10
    }
  ]
}
```

Every per-test object has exactly `test-name`, `passed`, `score`, and
`max-score`. The runner supplies nonempty `CLASSROOM`, `ASSIGNMENT`,
`SUBMISSION_TAG`, `COMMIT_URL`, and `RELEASE_URL`; `REVIEW_URL` falls back to
`COMMIT_URL`, and the grader generates UTC `datetime`. These values do not come
from student code. The runner may authoritatively stamp `owner`,
`assignment_type`, and `submitted_by`; the grader does not invent them. Failure
detail belongs in logs or the release/review body rather than incompatible
extra result fields. The grader exits zero when grading completed even when
student tests failed; missing or empty required context and other infrastructure
failures exit nonzero without a misleading result.

The five automated groups sum to 80:

| Test group | Maximum |
|---|---:|
| template, environment, fixture, and notebook integrity | 10 |
| Task 1 contract data and bounded exploratory objects | 15 |
| Task 2 supplied-defect evidence and corrected chart objects | 25 |
| Task 3 final contract, explanatory objects, and exact evidence files | 25 |
| relocation, scope, stale replacement, repeat, and resubmission | 5 |
| **Automated result maximum** | **80** |

The separate human maximum is not fabricated inside `result.json`.

## Adversarial QA matrix

Before release, the self-test and independent reviewer must exercise at least:

- untouched starter, correct solution, multiple partial solutions, and a
  corrected resubmission after prior failure;
- malformed JSON; missing, duplicated, reordered, or changed cell IDs/types;
  edited protected cells/files; and a modified checker;
- missing, extra, renamed, line-ending-changed, or byte-corrupted fixture and an
  edited manifest;
- stored correct-looking charts/evidence with broken source; deleted outputs;
  corrupt or stale PNG/CSV/JSON/text; repeat runs; and unrelated output files;
- wrong working directory, flattened checkout, course-root checkout, relocation,
  nested launch, and paths containing spaces;
- exploratory scatter with wrong axes, dropped rows, one marker/color only,
  extra trend/regression line, calculated correlation, inferential language, or
  a third saved exploratory image;
- critique with fewer/extra/reordered categories, empty problem/repair text,
  critique that changes data, or a supplied chart whose five protected defects
  are no longer executable;
- redesign with truncated baseline, missing percentage unit, causal title,
  color-only identity, unreadable/overlapping legend, wrong values/order,
  non-Figure/Axes API state, hard-coded canonical labels, blank image, or input
  mutation;
- explanatory chart with categorical checkpoints connected in the wrong order,
  wrong lines/values, color-only identity, missing unit/legend, causal title,
  hard-coded nine-point annotation, annotation attached to the wrong mark,
  clipping, blank image, or input mutation;
- supporting CSV with a serialized index, changed/reordered/omitted values,
  wrong header/line ending, or content not equal to the plotted table;
- JSON with missing/extra keys, wrong roles/categories, non-string evidence,
  nondeterministic serialization, or values not equal to the fresh namespace;
- text alternative missing chart type, axes/unit, both pathways, visible pattern,
  nine-point gap, or causal limitation; mismatch between JSON and text file;
- GroupBy/aggregation, joins/concat/reshape/cleaning, time series,
  correlation/regression/inference/modeling, random/mutable/network data,
  dashboards/interactive libraries, uploads/Drive, absolute paths, or legacy
  six-image outputs; and
- a canonical-only solution tested against alternate formats/pathways,
  checkpoint counts, orders, values, reversed final leader, and a final tie.

Automation must not declare a chart honest, accessible, attractive, or useful
merely because these adversarial properties pass.

## Human visual and communication rubric

The provisional human review is 20 points and uses the official Classroom50
review link after automated fresh execution has proved that the committed
artifacts equal the regenerated artifacts:

| Area | Human maximum | Judgment boundary |
|---|---:|---|
| Task 1 contract and exploration | 4 | question/purpose fit; correct grain and roles; bounded observation and meaningful limitation |
| Task 2 critique and redesign | 7 | five accurate explanations/repairs; truthful comparison; readable labels/contrast; redundant encoding; uncluttered, nonoverlapping layout |
| Task 3 explanatory communication | 9 | question/audience/claim fit; chart choice; scale/context; annotation usefulness; accessibility; text-alternative meaning; layout; evidentiary limitation |
| **Human maximum** | **20** | |

Automation owns executable environment/fixture integrity, source scope,
function behavior, nonmutation, data/mark values, observable labels and scales,
artifact structure, serialization, portability, and repeatability. Human review
owns semantic specificity, whether the chart choice and critique make sense,
whether visible design is actually legible and nonmisleading, whether redundant
encodings work at the rendered size, whether annotation helps rather than
obscures, and whether the text alternative communicates rather than satisfies a
keyword list.

The provisional combined diagnostic is therefore 80 automated plus 20 human.
This allocation does not decide the course's competence/pass-fail conversion,
pass threshold, gradebook mapping, late policy, or resubmission policy.

## Platform and publication boundary

- Clean local Jupyter or the VS Code notebook interface is mandatory for the
  initial assignment release.
- Classroom50 applies to the entire course. No GitHub Classroom export or
  legacy workflow is part of this rebuild.
- Student instructions use VS Code Source Control or GitHub Desktop to inspect,
  commit, and push the notebook and all five visible outputs.
- An assignment Colab badge may be added only after repository save-back,
  authoritative submission, feedback, and resubmission pass the course pilot
  for this package. Preserve the same notebook rather than forking a Colab
  edition.
- Classroom50 grader assets are discoverable. Behavioral variation, protected
  central hashes, and human review provide integrity; secrecy does not.
- No duration estimate or timing claim belongs in the notebook, README,
  platform guide, rubric, or grader contract.

## Full legacy disposition

Implementation must:

- rewrite `07/assignment/README.md` around the three exact tasks, definitions,
  fixtures, outputs, local-Jupyter workflow, public checker, GUI Git visibility,
  Classroom50 submission, scope, and hybrid rubric;
- replace `assignment.ipynb` with the exact 23-cell starter above and delete
  `assignment.md`;
- delete both data-generator files and all generated sales/customer/product
  data;
- delete the complete legacy `.github/workflows/` and `.github/test/` trees;
- replace broad dependencies with the exact four-package record;
- remove multi-panel galleries, pie charts, heatmaps, KDE/violin plots,
  correlation, joins, GroupBy, rolling, resampling, dashboard work, random data,
  mutable dates, and all six legacy output names; and
- add only the package, fixtures, functions, artifacts, checker, and central
  grading surfaces specified here.

No implementation should preserve old files for compatibility. History already
retains them.

## Unresolved policy choices

These choices do not block technical implementation and must not be guessed in
student code:

1. how the provisional 80 automated plus 20 human diagnostic maps to the
   historical regular-assignment competence/pass-fail policy, including any
   threshold or grade conversion;
2. the production Classroom50 classroom, assignment, release, review, and
   authoritative-submission metadata sources;
3. the operational route for combining the human 20 points with Classroom50's
   automated result and exporting them to the official grade system;
4. late-submission, resubmission, regrade, and record-retention policy; and
5. whether and when Assignment 07 receives an immutable-release Colab launch
   after the repository-save/Classroom50 pilot passes.

## Implementation and independent acceptance gate

The future implementation may proceed only within this contract. Course staff
must freeze every protected source and exact manifest hash after final prose and
cell sources settle, then run the public-check and central-grader self-tests.

A reviewer who did not implement the package must read every source, execute a
fresh canonical and alternate-data matrix, test every fixture/output failure,
inspect both PNGs at original resolution, verify the human/automation boundary,
exercise the official `result.json` success and failure shapes, run the course
audit and scoped diff gate, and confirm the Lecture 06→07→08 boundary. Fresh
Colab and any immutable badge remain separate external gates rather than
inferred capabilities.

Production-contract correction (2026-07-19): Classroom50 invokes the teacher
bundle's standard-library `autograder.py` with plain Python; it installs exact
sibling requirements before importing the central grader. The accepted student
repository may additionally contain only delivery-owned `.classroom50.yaml` and
`.github/workflows/autograde.yaml`; only the top-level `.git/**` repository
metadata tree is ignored, while every other root/workflow/grader-tree file,
including a nested `ordinary/.git/**` tree, is rejected.
