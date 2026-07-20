# Assignment 08 Blueprint: Grouped Results with an Explicit Grain

Status: independently reviewed, implementation-ready design handoff. This
document authorizes a future Assignment 08 rebuild only; it does not implement
or self-certify the student package. Implementation and post-implementation
acceptance remain separate gates. Clean local Jupyter is required. Assignment
Colab launch, save-back, and submission remain conditional on the course pilot.

## Evidence audit and replacement decision

The legacy Assignment 08 package must be replaced atomically rather than retained
as a compatibility path.

- It begins by merging provider, facility, and encounter tables, then summarizes
  provider attributes at encounter grain. That creates exactly the grain ambiguity
  the verified Lecture 08 sequence now teaches students to prevent.
- It requires group filtering, `GroupBy.apply`, custom statistical summaries,
  MultiIndex operations, stack/unstack, crosstabs, margins, missing-value filling,
  visualization, and performance reporting. Those are bonus, later, or unrelated
  capabilities rather than required Lecture 08 competence.
- Its generator notebook uses random data and requires a second notebook before
  students can begin. The assignment and generator are duplicated as paired
  Markdown sources.
- Its broad lower-bound dependencies permit materially different pandas behavior,
  add plotting/Jupyter/performance packages unrelated to the core task, and do
  not record the course runtime candidate.
- Its GitHub Classroom workflow pins Python 3.11, fetches mutable public tests over
  the network, and grades mostly file existence or nonempty content. It does not
  distrust stored notebook output, fresh-execute a disposable copy, verify result
  grain, or test behavior on alternate prepared data.

Replace `assignment.ipynb`, README, dependencies, checker, platform guidance,
fixtures, and grader surfaces. Delete `assignment.md`, both generator files,
`DATA_SCHEMA.md`, `TIPS.md`, the complete legacy `.github/` tree, and all legacy
outputs. Repository history already preserves them; they must not remain as a
second source of truth.

## Fixed role and assessment boundary

Assignment 08 is a regular, competence-focused assessment with exactly three
cumulative tasks:

1. state input/group/output grain, predict observed groups, and choose `size`,
   selected-column `count`, or selected-column `nunique` from three questions;
2. create a flat named aggregation, add same-index group context with
   `transform`, and produce one bounded flat two-key result; and
3. build exactly one aggregating `pivot_table`, compare every populated cell
   with an equivalent GroupBy result, and preserve an absent combination as
   missing rather than zero.

It assesses only the accepted Lecture 08 capabilities:

- input row grain, grouping key, group, grouping unit, output row grain, and
  observed-group prediction;
- explicit `observed=True`, `sort=True`, and `dropna=True` policies;
- `size`, `count`, and `nunique` chosen from variable meaning and the question;
- flat named aggregation with deliberate `as_index=False`, column names, order,
  and row conservation;
- selected-Series `transform("mean")` with input row count and index preserved;
- one two-key grouped summary with ordinary key columns;
- one aggregating pivot specified by index, columns, values, aggregation function,
  and observed-category policy; and
- deterministic CSV export, schema-aware readback, fresh execution, portable
  paths, and GUI-visible submission artifacts.

The required assignment does not clean, impute, join, concatenate as a new task,
structurally reshape, filter groups, use `GroupBy.apply`, manipulate MultiIndex,
create crosstabs, visualize, analyze dates/time series, calculate statistics or
models, access remote/performance tools, fetch a network source, generate random
data, or depend on a mutable date. It contains no period, resample, lag, rolling,
inference, prediction, or optimization work. Those exclusions preserve the
Lecture 07→08→09 boundary.

## Student repository contract

The future student-facing package contains exactly this instructional surface,
plus Classroom50-owned metadata added by the delivery system:

```text
08/assignment/
├── .gitignore
├── .python-version
├── PLATFORM_CHECK.md
├── README.md
├── assignment.ipynb
├── check_assignment.py
├── requirements.txt
├── data/
│   ├── fixture.json
│   └── support_requests.csv
└── output/
    └── .gitkeep
```

The instructor repository may additionally contain `_grader_selftest/` with a
production-grader mirror, adversarial harness, exact grader dependencies, and
maintenance notes. That directory is not copied to student templates or
production submissions. There is one canonical notebook and no paired Markdown
source, generator, `.github/` grader, solution, or completed starter output.

### Runtime records

`.python-version` contains exactly:

```text
3.12.13
```

`requirements.txt` contains exactly the deliberate student imports:

```text
numpy==2.0.2
pandas==3.0.3
```

Do not install pandas 3.0.4. Jupyter, ipykernel, nbclient, nbformat, and grader
libraries are host/grader tooling rather than student notebook imports. The
instructor-only grader candidate additionally pins `nbclient==0.10.2`,
`nbformat==5.10.4`, and `ipykernel==6.29.5` alongside the two student packages.

The assignment notebook checks the existing environment; it does not install
packages. A failed version check directs the learner to `PLATFORM_CHECK.md`.

### GUI-visible output and ignore policy

`.gitignore` contains exactly:

```text
.ipynb_checkpoints/
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
```

It must not ignore `output/`, CSV, JSON, or notebooks. The five required CSVs
must appear in VS Code Source Control or GitHub Desktop so students can commit
and submit them. `output/.gitkeep` preserves the empty starter directory. The
required Git path remains GUI-first; command-line Git is not assessed.

`PLATFORM_CHECK.md` gives directly actionable clean-local-Jupyter setup, exact
kernel/interpreter verification, restart/run-all, GUI commit/push, Classroom50
feedback, and resubmission steps. It is operational guidance, not a graded
aggregation task. It contains no Colab badge and does not claim that Colab edits
save back to the repository.

## Exact assignment-only prepared fixture

The assignment uses one prepared table so every task builds on the same known
row grain. It is distinct from Lecture 08's encounter demo in domain, identifiers,
labels, values, row count, and missing-value pattern. The table is course-authored,
synthetic, nonidentifying, and contains no real customer or support record.
Students do not generate, clean, join, impute, or structurally reshape it.

All fixture files are UTF-8 with LF line endings and a final newline.

### `data/support_requests.csv`

- Bytes: 469.
- SHA-256:
  `a9136161332c5da9f8f1251d869bbd014ed762751675fb757f81a79cff5352d6`.
- Row grain: one synthetic support request.
- Rows: 15.
- Ordered columns:
  `request_id,center,agent_id,channel,resolution_minutes,satisfaction_score`.

```csv
request_id,center,agent_id,channel,resolution_minutes,satisfaction_score
Q001,Central,A01,Email,30,4
Q002,Central,A01,Chat,20,5
Q003,Central,A02,Email,45,
Q004,Central,A02,Phone,60,4
Q005,Central,A03,Chat,25,3
Q006,Harbor,A04,Email,40,5
Q007,Harbor,A04,Chat,35,
Q008,Harbor,A05,Email,50,4
Q009,Harbor,A05,Chat,45,
Q010,Harbor,A04,Email,30,3
Q011,Ridge,A06,Phone,55,4
Q012,Ridge,A06,Chat,25,5
Q013,Ridge,A07,Phone,65,4
Q014,Ridge,A08,Email,35,3
Q015,Ridge,A08,Chat,30,4
```

The exact semantic categories are:

```python
CENTER_LEVELS = ["Central", "Harbor", "Ridge", "Valley"]
CHANNEL_LEVELS = ["Email", "Phone", "Chat"]
```

`Valley` is deliberately declared but unused. Harbor--Phone is deliberately
absent even though Harbor and Phone each occur elsewhere. Three satisfaction
scores are missing, and agents repeat within centers. Those properties make the
three counting questions and absent-versus-zero distinction observable.

The canonical in-memory dtypes after the student load cell are:

| Column | dtype | Meaning |
|---|---|---|
| `request_id` | pandas `string` | unique row identifier |
| `center` | ordered categorical using `CENTER_LEVELS` | grouping key |
| `agent_id` | pandas `string` | repeatable agent identifier |
| `channel` | ordered categorical using `CHANNEL_LEVELS` | second grouping key |
| `resolution_minutes` | NumPy `int64` | quantitative measurement |
| `satisfaction_score` | pandas nullable `Int64` | optional recorded score |

### `data/fixture.json`

`fixture.json` has exactly these 624 bytes and SHA-256
`b2fee1c48fb678b81318d2f085c42e2f9b480bd6c4eed6f07ef118b9bfd70860`:

```json
{
  "fixture_id": "a08-support-requests-v1",
  "provenance": "Course-authored synthetic support-request records; no real, identifying, or customer data.",
  "path": "support_requests.csv",
  "row_grain": "one row per synthetic support request",
  "row_count": 15,
  "columns": [
    "request_id",
    "center",
    "agent_id",
    "channel",
    "resolution_minutes",
    "satisfaction_score"
  ],
  "sha256": "a9136161332c5da9f8f1251d869bbd014ed762751675fb757f81a79cff5352d6",
  "center_levels": [
    "Central",
    "Harbor",
    "Ridge",
    "Valley"
  ],
  "channel_levels": [
    "Email",
    "Phone",
    "Chat"
  ]
}
```

Implementation must preserve this exact formatting and freeze protected-source
hashes only after the final README, checker, and protected notebook cells settle.

## Portable protected setup

The first code cell is supplied and protected. It must:

1. search upward from the launch directory for either flattened
   `data/fixture.json` or course-root `08/assignment/data/fixture.json`;
2. define `ASSIGNMENT_ROOT`, `DATA_DIR`, `OUTPUT_DIR`, `FIXTURE_PATH`, and the
   exact five output paths from the discovered assignment root;
3. validate the manifest's exact keys, values, safe relative fixture path,
   manifest bytes/hash, CSV bytes/hash, final newline, row count, and ordered
   columns before pandas reads the CSV;
4. create only `OUTPUT_DIR` and delete only the five named stale outputs,
   preserving `.gitkeep` and unrelated files;
5. import and assert Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3; and
6. stop with an actionable exception when the environment, path, manifest, or
   fixture contract fails.

It must not install packages, embed fallback fixture bytes, inspect another
notebook, access a network, use an absolute repository path, depend on
`/content`, inspect a mutable date, prompt for upload, mount Drive, or rewrite a
fixture. A missing or corrupt assignment fixture stops before analysis or output
cleanup. The committed assignment files and local environment are mandatory.

The student load cell reads the exact dtypes above, applies the manifest category
orders without changing values, and verifies shape, columns, unique request IDs,
nonmissing grouping keys, and the exact three missing scores.

## Definition-before-independent-use contract

The README and protected `a08-terms-data` cell define every required term before
student code uses it freely:

| Term | Required plain-language meaning |
|---|---|
| input row grain | what one source row represents |
| grouping key | one column or bounded column combination whose values determine group membership |
| group | the input rows sharing one observed grouping-key value or combination |
| grouping unit | the real-world category or entity represented by one group |
| output row grain | what one result row represents |
| aggregation | reducing the rows in each group to one or more summary values |
| GroupBy object | a pandas object recording how rows are split; not itself a summary table |
| observed-category policy | whether unused categorical levels/combinations appear; required work uses `observed=True` |
| missing-key policy | whether a missing grouping key forms a group; complete prepared keys use `dropna=True` |
| output order | deliberate category-key ordering; required work uses `sort=True` |
| `size` | number of input rows in each group, regardless of missing values in other columns |
| selected-column `count` | number of nonmissing values in the selected column for each group |
| selected-column `nunique` | number of distinct nonmissing values in the selected column for each group |
| named aggregation | an output name paired with its source column and calculation |
| `as_index=False` | keeping grouping keys as ordinary flat result columns |
| `transform` | a within-group calculation returning one same-index value per input row |
| two-key group | rows sharing one observed combination of both key values |
| structural `pivot` | Lecture 06 reshape requiring unique row/column combinations and performing no aggregation |
| aggregating `pivot_table` | grouping repeated combinations and placing their summaries across row and column axes |
| absent combination | no input row has the key combination; it is not a measured zero |

The term cell also states the fixture grain, roles, category orders, missing-score
meaning, unused Valley level, absent Harbor--Phone combination, and why no data
cleaning decision is part of the assignment.

## Exact notebook contract

`assignment.ipynb` uses notebook-format major version 4 and minor version 5, a
portable `Python 3` kernelspec, the exact 25 cells below in order, stable globally
unique IDs, null execution counts, and zero stored output in the released starter.
Protected cells are complete. Student cells contain actionable TODO scaffolds
without solution fragments. There is no hidden prerequisite cell.

The header states that clean local Jupyter is required, Assignment Colab is not
supported, the fixture is synthetic, restart-and-run-all is mandatory, stored
output is not execution evidence, and the notebook plus five generated CSVs are
separate GUI-visible submission artifacts.

### Cells 0--3: orientation, terms, and load

0. `a08-header` (protected Markdown): title, exact three-task progression,
   privacy, local-only platform boundary, five output names, fresh-execution
   rule, and GUI Git visibility.
1. `a08-setup` (protected code): portable path, manifest/checksum/version
   validation, exact stale-output removal, and path constants.
2. `a08-terms-data` (protected Markdown): the complete definition ledger and
   fixture contract above, plus concise `size`/`count`/`nunique`, aggregation/
   transform, and structural-pivot/aggregating-pivot comparisons.
3. `a08-load` (student code): explicit-dtype read into `support_requests`, exact
   ordered categories, shape/schema/ID/key/missing-score assertions, and
   `source_snapshot = support_requests.copy(deep=True)`.

### Cells 4--9: Task 1 — predict grain and choose the count

4. `a08-task1-contract` (student Markdown): in the student's own words, state
   input row grain, grouping key, grouping unit, predicted observed identities
   and count, observed policy, output row grain, and why each count operation
   answers a different question.
5. `a08-task1-values` (student code): assign exactly these machine-readable
   values before grouping:

   ```python
   input_row_grain = "one row per synthetic support request"
   grouping_key = ["center"]
   grouping_unit = "one observed support center"
   predicted_group_identities = ["Central", "Harbor", "Ridge"]
   predicted_group_count = 3
   observed_category_policy = True
   output_row_grain = "one row per observed support center"
   count_plan = {
       "request_count": {
           "question": "How many support-request rows were recorded?",
           "operation": "size",
       },
       "satisfaction_count": {
           "question": "How many requests have a recorded satisfaction score?",
           "operation": "count",
       },
       "unique_agent_count": {
           "question": "How many distinct agents appear?",
           "operation": "nunique",
       },
   }
   ```

6. `a08-count-function` (student code): define
   `build_count_summary(request_table)`. It creates one reusable center GroupBy
   with explicit `observed=True`, `sort=True`, and `dropna=True`; uses `size`,
   selected `satisfaction_score.count`, and selected
   `agent_id.nunique(dropna=True)`; aligns the three same-index Series as columns
   using the already-learned `concat`; resets the grouping key to an ordinary
   column; returns exactly
   `center,request_count,satisfaction_count,unique_agent_count`; and does not
   mutate its argument.
7. `a08-task1-run` (student code): create `center_count_summary`; verify the
   three predicted identities, `ngroups == 3` through an independently created
   bounded diagnostic GroupBy, exact shape/schema/order, `[5, 5, 5]` request
   counts, `[4, 3, 5]` satisfaction counts, `[3, 2, 3]` unique-agent counts,
   request conservation at 15, no Valley row, and source immutability.
8. `a08-task1-save` (student code): write
   `output/center_count_summary.csv` with UTF-8, `index=False`,
   `lineterminator="\n"`, and an explicit empty `na_rep`; read it with schema-
   aware dtypes and assert exact equality after normalizing serialized category
   labels to string.
9. `a08-task1-explain` (student Markdown): explain why missing satisfaction
   changes `count` but not `size`, why repeated agents make `nunique` smaller
   than row count, and why unused Valley is not an observed group.

### Cells 10--16: Task 2 — aggregate, transform, and use two keys

10. `a08-task2-prompt` (protected Markdown): restate the named-aggregation,
    transform, flat-layout, and two-key contracts without introducing another
    concept.
11. `a08-center-summary-function` (student code): define
    `build_center_summary(request_table)`. It uses one explicit center GroupBy
    with `as_index=False`, `observed=True`, `sort=True`, and `dropna=True`, then
    named aggregation to return exactly:

    ```text
    center,request_count,satisfaction_count,unique_agent_count,
    total_resolution_minutes,mean_resolution_minutes
    ```

    It does not round or mutate its argument.
12. `a08-context-function` (student code): define
    `add_center_context(request_table)`. It deep-copies its argument, uses a
    selected-Series center GroupBy with explicit policies and
    `transform("mean")`, then adds `center_mean_resolution_minutes` and
    `difference_from_center_mean`. It returns one row per request with the exact
    original index and leaves its argument unchanged.
13. `a08-two-key-function` (student code): define
    `build_center_channel_summary(request_table)`. It groups on `center` and
    `channel` with `as_index=False` and the three explicit policies, then uses
    named aggregation to return exactly
    `center,channel,request_count,mean_resolution_minutes`. It does not use or
    return a MultiIndex.
14. `a08-task2-run` (student code): create `center_summary`,
    `requests_with_context`, and `center_channel_summary`; verify exact columns,
    order, values, dtypes, three-versus-fifteen aggregate/source row counts,
    same-index transform alignment, eight observed two-key rows, request-count
    conservation at 15, absent Harbor--Phone and Valley, and source immutability.
    It records `aggregate_rows = 3`, `source_rows = 15`,
    `transform_rows = 15`, and
    `transform_index_preserved = True` rather than manufacturing an exception.
15. `a08-task2-save` (student code): write the three Task 2 CSVs with UTF-8,
    LF, explicit empty `na_rep`, and `index=False`; read each with exact schema-
    aware dtypes; normalize serialized categories to strings; and assert exact
    frame equality without rounding.
16. `a08-task2-explain` (student Markdown): explain the output-grain difference
    between `center_summary` and `requests_with_context`, why length and index
    both matter for alignment, what one two-key row represents, and why
    positional assignment of three aggregate rows to fifteen request rows would
    be invalid.

### Cells 17--23: Task 3 — one aggregating pivot and equivalence proof

17. `a08-task3-prompt` (protected Markdown): contrast Lecture 06 structural
    `pivot` with aggregating `pivot_table`; define the displayed row and populated-
    cell meanings; require one pivot and a cell-for-cell GroupBy comparison.
18. `a08-pivot-values` (student code): assign this exact specification before
    constructing the pivot:

    ```python
    pivot_spec = {
        "index": "center",
        "columns": "channel",
        "values": "resolution_minutes",
        "aggfunc": "mean",
        "observed": True,
        "sort": True,
        "dropna": True,
    }
    pivot_display_row_grain = "one observed support center"
    pivot_cell_grain = "one observed center-channel group"
    absent_combination = ["Harbor", "Phone"]
    absent_combination_meaning = "no input row for this center-channel combination"
    ```

    The first five entries name the five pivot choices; `sort` and `dropna`
    record the already-defined ordering and missing-key policies.
19. `a08-pivot-function` (student code): define
    `build_resolution_pivot(request_table)`. Its source contains the assignment's
    only call to `pd.pivot_table`, with literal roles matching `pivot_spec` and
    explicit `observed=True`, `sort=True`, and `dropna=True`. It returns the
    indexed pivot without filling, rounding, resetting, writing a file, or
    mutating its argument.
20. `a08-task3-run` (student code): create `resolution_pivot` and a fresh
    `pivot_reference = build_center_channel_summary(support_requests)`; verify
    row order Central/Harbor/Ridge, column order Email/Phone/Chat, 3-by-3 shape,
    exactly eight populated cells, Harbor--Phone as the only missing cell, no
    Valley row, 37.5 at Central--Email, 60.0 at Ridge--Phone, and equality of
    every reference row with its corresponding pivot cell. It also proves there
    are no measured zero cells and source remains unchanged.
21. `a08-task3-save` (student code): remove the pivot columns-axis name, reset
    `center` to an ordinary first column, write
    `output/mean_resolution_pivot.csv` with UTF-8, LF, `index=False`, and
    `na_rep=""`; schema-aware readback must preserve exactly one missing cell and
    no zero; compare the readback with the serialized in-memory table.
22. `a08-task3-explain` (student Markdown): explain the five pivot choices,
    displayed row grain versus populated-cell grain, why GroupBy equivalence is
    a useful invariant, and why replacing Harbor--Phone with zero would assert a
    measurement that was never observed.
23. `a08-synthesis` (student Markdown): concisely connect the three tasks by
    explaining when a grouped operation changes row grain, when `transform`
    preserves it, and how a wide pivot can hide the two-key group represented by
    each populated cell.

### Cell 24: supplied final verification

24. `a08-final-verify` (protected code): recheck canonical namespace names,
    exact source/summary/context/two-key/pivot shapes, source immutability,
    transform index preservation, group/pivot counts, all five output paths,
    exact readbacks, and no extra generated artifact. It prints one completion
    summary and directs the learner to restart/run-all and then run
    `python check_assignment.py`. It does not award points or claim a human-
    reasoning pass.

## Exact function interfaces and behavioral variation

The five public functions are:

```python
build_count_summary(request_table)
build_center_summary(request_table)
add_center_context(request_table)
build_center_channel_summary(request_table)
build_resolution_pivot(request_table)
```

They accept any complete prepared DataFrame with the documented six-column
schema, nonmissing categorical grouping keys, ordered categorical center/channel
columns, integer resolution minutes, and nullable-integer satisfaction scores.
They must derive labels, counts, values, group order, row count, and index from
their argument. They must not depend on canonical IDs, labels, counts, values,
global `support_requests`, fixture files, or generated outputs; mutate input;
read/write files; clean/impute; join; structurally reshape; filter/apply groups;
plot; or perform time/statistical/modeling work.

File writes remain in run/save cells so the central grader can call the functions
on discoverable alternate in-memory data without creating artifacts. The pivot
function contains the only `pivot_table` call in all student-editable code.

## Exact canonical behavior and artifact bytes

All five outputs use UTF-8, comma delimiters, `lineterminator="\n"`, an explicit
empty `na_rep`, ordered columns, final newline, and `index=False`. Completed
outputs are GUI-visible and deterministic. Under CPython 3.12.13, NumPy 2.0.2,
and pandas 3.0.3 their exact contracts are:

| Artifact | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `center_count_summary.csv` | 3 | 98 | `0735d0647dbbe2199b1de03e1061bf6c3a7a9d15bb553d128bdc1ab295ef2f36` |
| `center_summary.csv` | 3 | 174 | `6c528bd229cd0ce2db2f4c90f09fd2a9ba670fb3aa659951bc113d70a33afad4` |
| `requests_with_context.csv` | 15 | 680 | `391d56794e1537244c8d0b97f39e25e822b1e54d45b049fca98760ba646b1a7a` |
| `center_channel_summary.csv` | 8 | 210 | `41b74a8dac05eff1695e6b972b360bd2b1730e77f5e2060e42801533a07180da` |
| `mean_resolution_pivot.csv` | 3 | 86 | `1274782fc4e773bfd572736c0af106842d92751d672b6ad341207574e636dedf` |

### `output/center_count_summary.csv`

```csv
center,request_count,satisfaction_count,unique_agent_count
Central,5,4,3
Harbor,5,3,2
Ridge,5,5,3
```

### `output/center_summary.csv`

```csv
center,request_count,satisfaction_count,unique_agent_count,total_resolution_minutes,mean_resolution_minutes
Central,5,4,3,180,36.0
Harbor,5,3,2,200,40.0
Ridge,5,5,3,210,42.0
```

### `output/requests_with_context.csv`

```csv
request_id,center,agent_id,channel,resolution_minutes,satisfaction_score,center_mean_resolution_minutes,difference_from_center_mean
Q001,Central,A01,Email,30,4,36.0,-6.0
Q002,Central,A01,Chat,20,5,36.0,-16.0
Q003,Central,A02,Email,45,,36.0,9.0
Q004,Central,A02,Phone,60,4,36.0,24.0
Q005,Central,A03,Chat,25,3,36.0,-11.0
Q006,Harbor,A04,Email,40,5,40.0,0.0
Q007,Harbor,A04,Chat,35,,40.0,-5.0
Q008,Harbor,A05,Email,50,4,40.0,10.0
Q009,Harbor,A05,Chat,45,,40.0,5.0
Q010,Harbor,A04,Email,30,3,40.0,-10.0
Q011,Ridge,A06,Phone,55,4,42.0,13.0
Q012,Ridge,A06,Chat,25,5,42.0,-17.0
Q013,Ridge,A07,Phone,65,4,42.0,23.0
Q014,Ridge,A08,Email,35,3,42.0,-7.0
Q015,Ridge,A08,Chat,30,4,42.0,-12.0
```

### `output/center_channel_summary.csv`

```csv
center,channel,request_count,mean_resolution_minutes
Central,Email,2,37.5
Central,Phone,1,60.0
Central,Chat,2,22.5
Harbor,Email,3,40.0
Harbor,Chat,2,40.0
Ridge,Email,1,35.0
Ridge,Phone,2,60.0
Ridge,Chat,2,27.5
```

### `output/mean_resolution_pivot.csv`

```csv
center,Email,Phone,Chat
Central,37.5,60.0,22.5
Harbor,40.0,,40.0
Ridge,35.0,60.0,27.5
```

Canonical in-memory summary count dtypes preserve nullable `Int64` for
`satisfaction_count`; the other count/total fields are `int64` and mean/context
fields are `float64`. CSV readbacks explicitly restore those intended dtypes.
Serialized categorical labels are compared as pandas string columns after the
in-memory category order has been verified.

The starter contains only `output/.gitkeep`. Protected setup removes only these
five named CSVs. Restart/run-all must recreate deleted outputs, replace stale or
corrupt versions, preserve unrelated output files, and reproduce the hashes on
the pinned grader platform.

## Protected and student-editable surfaces

Implementation freezes course-owned hashes for:

- `.python-version`, `requirements.txt`, `.gitignore`, `README.md`, and
  `PLATFORM_CHECK.md`;
- `check_assignment.py`;
- `data/fixture.json` and `data/support_requests.csv`; and
- notebook cells `a08-header`, `a08-setup`, `a08-terms-data`,
  `a08-task2-prompt`, `a08-task3-prompt`, and `a08-final-verify`, including ID,
  type, position, and exact source.

Only designated student Markdown/code cells and the five regenerated CSVs are
student work. The central grader independently owns protected expectations;
editing the public checker cannot weaken production enforcement. Course staff
regenerate protected hashes only when intentionally releasing a new template.

## Student-visible public checker

`check_assignment.py` uses only the Python standard library. It must not import
pandas/NumPy/nbclient, execute arbitrary notebook source, trust stored output,
read grader secrets, claim a score, or serve as the production grader. It must:

1. locate flattened and course-root assignment layouts from `__file__`;
2. validate exact manifest bytes/semantics/hash, safe fixture path, CSV bytes/
   hash, final newlines, row count, ordered columns, and package inventory;
3. parse notebook JSON and require exactly the 25 IDs/types/order above,
   notebook-format major version 4 and minor version 5, portable kernelspec,
   unique IDs, and unedited protected cells; submitted output/execution counts
   are ignored as evidence;
4. detect untouched TODO scaffolds and all five public function names;
5. parse student code with `ast`; require the one `pd.pivot_table` call and reject
   extra pivot-table calls, structural pivot, later/bonus APIs, disallowed imports,
   absolute/content/Drive/upload/network paths, embedded fixture fallback, random
   or mutable-date data, and unexpected output paths;
6. require exactly the five CSVs plus `.gitkeep` under `output/`; reject legacy
   `q1_`/`q2_`/`q3_` artifacts and unexpected generated files;
7. verify each CSV's exact canonical header, row count, bytes, final newline, and
   SHA-256; and
8. return nonzero with a small task-grouped set of actionable messages, or print
   one readiness summary without claiming a grade or human-reasoning pass.

Source scanning is early feedback, not proof of behavior. The central fresh
grader decides whether required operations were used correctly and whether the
functions generalize to disclosed valid variations.

The AST denylist applies to executable student code, not Markdown explanations.
It covers:

- `GroupBy.apply`, `GroupBy.filter`, crosstab, MultiIndex construction or
  advanced index-level manipulation;
- merge/join, structural `pivot`, `melt`, stack/unstack, group filtering,
  fill/impute/interpolate, drop/deduplicate, replacement, or cleaning pipelines;
- plotting libraries/calls and image output;
- datetime conversion, period, resample, rolling, expanding, EWM, shift, lag,
  calendar, or current-time APIs;
- scipy, statsmodels, scikit-learn, model/formula APIs, correlation, covariance,
  significance, uncertainty, prediction, or optimization;
- SSH/tmux, multiprocessing, Dask, profiling, chunking, performance benchmarks,
  browser/interactive/dashboard libraries; and
- requests/urllib/http clients, `read_html`, `read_json` URLs, random generators,
  credentials, `/content`, upload prompts, or Drive mounts.

## Classroom50 central grader

Classroom50 is the course-wide delivery system, not a Lecture 04-only system.
The Assignment 08 production grader is teacher-controlled and discoverable; it
contains no solution, credential, confidential record, or test whose value
depends on secrecy. It must not import or trust the editable public checker.

The grader must:

1. independently validate protected package files, fixture bytes, notebook
   topology/cells, output visibility policy, and source scope;
2. copy the submission to an isolated temporary directory, remove all five
   output CSVs, clear submitted execution counts/outputs, append grader-owned
   checks to a disposable notebook copy, and execute from a fresh pinned kernel;
3. exercise flattened Classroom50, course-root, relocated, nested-working-
   directory within the assignment tree, and path-with-spaces layouts;
4. verify canonical namespace values, exact operation contracts, function
   signatures, source immutability, schemas/dtypes/order, group identities/counts,
   aggregation conservation, transform row/index preservation, two-key groups,
   pivot occupancy/equivalence, exact output bytes, and schema-aware readbacks;
5. call all five functions on the discoverable alternate complete prepared table
   below, including its shuffled nondefault index, different labels/values/group
   sizes, and absent combination;
6. use AST and behavioral checks together to reject canonical hard-coding,
   incorrect count operations, implicit category/order policies, positional
   transform assignment, output-only solutions, and more than one pivot table;
7. run missing/corrupt fixture, malformed-notebook, protected-edit, stored-output,
   stale/deleted/corrupt output, unrelated-file, repeat, and corrected-
   resubmission cases; separately plant an unrelated output sentinel to prove
   setup preserves it while requiring the final submission inventory to reject
   that extra file until it is removed;
8. direct the bounded human reasoning review to the student-authored Markdown
   through Classroom50's context-supplied `review` URL; fresh execution remains
   the automated behavior evidence, and submitted stored output remains
   untrusted; and
9. write `./result.json` and actionable grading logs.

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
  "max-score": 90,
  "tests": [
    {
      "test-name": "Task 1 grain and count semantics",
      "passed": false,
      "score": 0,
      "max-score": 20
    }
  ]
}
```

Every per-test object has exactly `test-name`, `passed`, `score`, and
`max-score`. The required `classroom`, `assignment`, `submission`, `commit`,
`release`, `review`, and `datetime` metadata come from Classroom50's grading
context, not student code. `owner` and `assignment_type` are optional in the
grader-emitted object because the runner stamps them authoritatively; validation
must not reject their absence, and neither grader nor student code may invent
them. The runner may also add its optional `submitted_by` object. Failure detail
belongs in logs or the release/review body, not incompatible result fields. The
grader exits zero when grading completed even if student tests fail; nonzero is
reserved for grader infrastructure failure.

The automated groups sum to 90:

| Test group | Maximum |
|---|---:|
| template, environment, fixture, notebook, and protected integrity | 10 |
| Task 1 grain prediction and count semantics | 20 |
| Task 2 named aggregation, same-index transform, and two-key result | 35 |
| Task 3 single pivot, GroupBy equivalence, and absent-combination contract | 20 |
| portability, scope, stale/repeat output, and resubmission | 5 |
| **Automated result maximum** | **90** |

The separate human maximum is not fabricated inside `result.json`.

## Discoverable alternate prepared table

The student prompt states that grader-owned calls will vary valid complete data,
labels, row order, row counts, values, and index labels while preserving the
documented schema and categorical contract. The grader bundle publishes this
exact primary alternate table; integrity comes from behavior, not surprise.

It uses ordered centers `Metro, Coast, Hill, Plains`, ordered channels
`Web, Voice, Desk`, and shuffled index labels
`[42, 5, 91, 12, 63, 8, 77, 24, 3, 55]`:

```text
index request_id center agent_id channel resolution_minutes satisfaction_score
42    Z06        Coast  B3       Desk    35                 <NA>
5     Z01        Metro  B1       Web     10                 5
91    Z09        Hill   B6       Voice   70                 4
12    Z04        Metro  B2       Web     14                 3
63    Z08        Hill   B5       Web     50                 3
8     Z02        Metro  B1       Voice   30                 <NA>
77    Z10        Hill   B5       Desk    60                 5
24    Z05        Coast  B3       Web     25                 4
3     Z03        Metro  B2       Desk    20                 4
55    Z07        Coast  B4       Desk    45                 5
```

Required alternate properties are:

- count rows Metro/Coast/Hill: `[4, 3, 3]`; recorded satisfaction `[3, 2, 3]`;
  distinct agents `[2, 2, 2]`;
- center totals `[74, 105, 180]` and means `[18.5, 35.0, 60.0]`;
- transform output index exactly
  `[42, 5, 91, 12, 63, 8, 77, 24, 3, 55]` with means and differences derived
  from the alternate rows rather than canonical literals;
- eight observed two-key rows in declared category order, with Coast--Voice
  absent and unused Plains omitted; and
- a 3-by-3 Web/Voice/Desk pivot whose only missing cell is Coast--Voice and whose
  eight populated cells equal the alternate two-key GroupBy means.

The self-test may add other disclosed property-based variants—different valid
labels, row counts, repeated agents, missing satisfaction positions, group
sizes, values, and nondefault indexes—but may not introduce missing grouping
keys, a new schema, a cleaning rule, or an undisclosed API requirement.

## Adversarial QA matrix

Before release, the self-test and independent reviewer must exercise at least:

- untouched starter, correct solution, multiple partial solutions, and a
  corrected resubmission after failure;
- malformed JSON; missing, duplicated, reordered, or edited cell IDs/types;
  edited protected files/cells; and modified public checker;
- missing, renamed, extra, line-ending-changed, or byte-corrupted fixture and
  edited manifest/category order;
- stored correct-looking tables with broken/unexecuted source; deleted outputs;
  stale/corrupt CSVs; repeat runs; unrelated output file preservation; and extra
  or legacy output names;
- flattened, course-root, relocated, nested-within-assignment, and path-with-
  spaces execution;
- `count` used for request rows, `size` used for recorded satisfaction,
  `count` used for distinct agents, `nunique` on the wrong column, implicit
  `observed`/`sort`/`dropna`, `observed=False`, Valley materialized, wrong group
  identity/order, or lost row conservation;
- dictionary/multi-level aggregation instead of flat named aggregation; wrong
  source column/function/name; implicit or indexed key layout; rounding; source
  mutation; canonical labels/counts/values hard-coded in a reusable function;
- aggregation assigned to source rows; transform on the wrong column/key;
  positional or reset-index alignment; wrong output length/index; same-index
  values hard-coded; or input mutation;
- two-key output with a MultiIndex, unused combinations, Harbor--Phone present,
  missing observed combinations, wrong order/schema/count/mean, or totals not
  conserving 15 requests;
- zero/margins/fill value in the pivot; wrong index/columns/values/aggfunc;
  structural `pivot`; multiple `pivot_table` calls; extra/missing rows or columns;
  failure to compare every populated cell; or canonical-only pivot values;
- GroupBy `apply`/filter, crosstab, stack/unstack, cleaning/imputation, joins,
  visualization, time series, statistics/modeling, remote/performance, network,
  random/mutable data, uploads/Drive, `/content`, or absolute paths; and
- central success and captured-failure `result.json` objects, exact 90-point
  sum, zero exit after completed failing grades, nonzero infrastructure failure,
  score collection, feedback visibility, and corrected resubmission.

The correct solution must pass canonical and alternate behavioral calls. The
untouched starter should fail with a small stable set of task-specific messages.
Automation must not claim that student prose is meaningful merely because it is
nonempty or contains keywords.

## Human reasoning boundary and provisional scoring

For grader/rubric design, use this provisional 90 automated plus 10 human
diagnostic allocation:

| Area | Automated | Human | Total |
|---|---:|---:|---:|
| Task 1: grain prediction and count choices | 20 | 3 | 23 |
| Task 2: aggregation/transform/two-key behavior and reasoning | 35 | 4 | 39 |
| Task 3: pivot/equivalence behavior and missing-combination interpretation | 20 | 3 | 23 |
| shared template, portability, scope, and reproducibility | 15 | 0 | 15 |
| **Diagnostic total** | **90** | **10** | **100** |

Automation owns environment/fixture integrity, required API/source contracts,
function behavior on canonical and alternate prepared tables, nonmutation,
schemas/dtypes/order, group identities/counts, row/index invariants, exact values,
pivot cell equivalence, artifact bytes, portability, and repeatability.

Human review owns only:

- whether the Task 1 grain/group/count explanation connects each question to
  the correct variable meaning rather than paraphrasing function names;
- whether the Task 2 explanation accurately distinguishes reduced aggregate
  grain from same-index transform context and explains the bounded two-key row;
- whether the Task 3 explanation accurately names the pivot choices, distinguishes
  displayed row from cell grain, and explains why absence is not zero; and
- whether the notebook is concise, readable, and contains no identifying or
  sensitive information.

The historical course design describes regular Assignment 08 as competence-
focused pass/fail. This provisional diagnostic does not set a pass threshold,
convert points to pass/fail, establish gradebook weighting, or decide late,
resubmission, or regrade policy. The fixture, functions, artifacts, automated
90-point result, and human 10-point review remain technically separable from
that unresolved policy overlay.

## Platform and publication boundary

- Clean local Jupyter or the VS Code notebook interface is mandatory for the
  initial release.
- Classroom50 applies to the entire course. No GitHub Classroom export, Actions
  workflow, mutable remote-test fetch, or student-editable production grader is
  retained.
- Student instructions use VS Code Source Control or GitHub Desktop to inspect,
  commit, and push `assignment.ipynb` and all five visible CSVs.
- No Assignment 08 Colab badge or claimed Colab submission path is allowed until
  repository save-back, authoritative submission, feedback, and resubmission
  pass the course pilot for notebook assignments. If approved later, preserve
  this same notebook rather than fork a Colab edition.
- Classroom50 grader assets are discoverable. Alternate behavior, central
  protected hashes, and bounded human reasoning review provide integrity;
  secrecy does not.
- No duration estimate, due-date logic, or timing claim belongs in the notebook,
  README, platform guide, rubric, checker, or grader contract.

## Full legacy disposition

Implementation must:

- rewrite `08/assignment/README.md` around the exact three tasks, definitions,
  fixture, outputs, local setup, public checker, GUI Git visibility,
  Classroom50 submission, and conditional Colab status;
- replace `assignment.ipynb` with the exact 25-cell starter and delete
  `assignment.md`;
- delete `data_generator.ipynb`, `data_generator.md`, `DATA_SCHEMA.md`, and
  `TIPS.md`;
- delete the entire legacy `.github/workflows/` and `.github/test/` trees;
- replace broad lower-bound dependencies with the exact two-package record;
- remove all provider/facility/encounter generator schemas and all legacy
  `q1_`/`q2_`/`q3_` output contracts;
- remove joins, standardization/z-scores, group filtering/apply, MultiIndex,
  stack/unstack, crosstab, margins/fill, visualization, time fields, random data,
  remote mutable tests, and performance reporting; and
- add only the student/instructor surfaces, fixture, functions, five CSVs,
  checks, and grading contracts specified here.

No implementation preserves old files for compatibility. Classroom50 course
configuration lives in the course-wide platform area, not `.github/` and not as
a second assignment implementation.

## Unresolved policy choices

These choices do not block technical implementation and must not be guessed in
student code:

1. how the provisional 90 automated plus 10 human diagnostic maps to the
   historical competence/pass-fail policy, including any threshold or
   gradebook conversion;
2. production Classroom50 classroom, assignment, release, review, and
   authoritative-submission metadata sources;
3. how the human 10-point result is combined with Classroom50's automated
   `result.json` and exported to the official grade system;
4. late-submission, resubmission, regrade, and record-retention policy; and
5. whether and when Assignment 08 receives an immutable-release Colab launch
   after the repository-save/Classroom50 pilot passes.

## Implementation and independent acceptance gate

Independent design review has accepted this contract. Implementation may
proceed, but course staff must freeze protected source hashes after final prose
and cell sources settle, implement the exact starter/public checker/central
self-test, and preserve the score-policy separation.

A reviewer who did not implement the package must then inspect every source,
fresh-execute canonical and alternate data in all path layouts, test the full
fixture/output/adversarial matrix, verify official success/failure
`result.json` shapes and exits, check the human/automation boundary, run the
course audit and scoped diff gate, and recheck Lecture 07→08→09 scope. Fresh
Colab execution and any immutable badge remain separate pilot/publication gates,
not inferred assignment capabilities.

Production-contract correction (2026-07-19): Classroom50 invokes the teacher
bundle's standard-library `autograder.py` with plain Python; it installs exact
sibling requirements before importing the central grader. The accepted student
repository may additionally contain only delivery-owned `.classroom50.yaml` and
`.github/workflows/autograde.yaml`; only the top-level `.git/**` repository
metadata tree is ignored, while every other root/workflow/grader-tree file,
including a nested `ordinary/.git/**` tree, is rejected.
