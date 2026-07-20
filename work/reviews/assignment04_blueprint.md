# Assignment 04 implementation blueprint

Status: approved design handoff; local-Jupyter implementation may proceed. The Colab assignment launch/save/submission path remains gated on the Classroom50 notebook pilot.

## Assessment boundary

Assignment 04 assesses only:

1. notebook state and fresh top-to-bottom execution;
2. `Series` and `DataFrame` construction and metadata;
3. bracket selection and Series-versus-DataFrame return types;
4. `.loc` label selection versus `.iloc` position selection;
5. one boolean mask;
6. one arithmetic derived column;
7. deterministic sorting with an explicit unique tie-breaker;
8. pinned, portable CSV input;
9. CSV output with `index=False`; and
10. readback verification.

It must not require cleaning, missing-value decisions, type conversion, dates, joins, concatenation, reshape, GroupBy, aggregation, plotting, modeling, or performance work.

## Student package

```text
04/assignment/
├── .gitignore
├── .python-version
├── PLATFORM_CHECK.md
├── README.md
├── assignment.ipynb
├── check_assignment.py
├── requirements.txt
├── data/
│   ├── fixture.json
│   └── purchases.csv
└── output/
    └── .gitkeep
```

Do not retain a GitHub Classroom Actions workflow, centrally managed grader tests in the student-facing package, a completed notebook, or a second notebook required to generate data.

Environment candidate:

```text
.python-version: 3.12.13
requirements.txt:
numpy==2.0.2
pandas==3.0.3
```

Jupyter hosting, kernel support, pytest, nbclient, and nbformat are platform/grader tooling rather than student code dependencies.

## Exact fixture

Use these canonical bytes for `data/purchases.csv`:

```csv
purchase_id,item,quantity,unit_price
P001,USB Cable,2,8.00
P002,Mouse Pad,1,12.00
P003,Water Bottle,3,10.00
P004,Desk Lamp,2,15.00
P005,Keyboard,1,30.00
P006,Headphones,4,7.50
P007,Webcam Cover,5,3.00
P008,Laptop Stand,2,20.00
P009,Cable Tie,3,5.00
P010,Screen Cloth,1,6.00
P011,USB Hub,2,8.00
P012,Notebook,3,5.00
```

The fixture is course-authored, synthetic, non-identifying, complete, and deliberately contains tied derived totals. `fixture.json` records:

```json
{
  "fixture_id": "a04-purchases-v1",
  "provenance": "course-authored synthetic teaching data",
  "row_count": 12,
  "columns": ["purchase_id", "item", "quantity", "unit_price"],
  "sha256": "<checksum of the canonical committed CSV bytes>"
}
```

Compute the checksum from the canonical bytes during implementation; no placeholder may remain in the released package.

A supplied, non-editable setup cell searches upward for either `data/purchases.csv` in a flattened Classroom50 repository or `04/assignment/data/purchases.csv` in this course repository, verifies the manifest and checksum, creates `output/`, and defines `DATA_PATH`, `OUTPUT_DIR`, `LABELED_OUTPUT_PATH`, and `SELECTED_OUTPUT_PATH`. It must not use an absolute path, Drive mount, manual upload, or network access. The setup is supplied machinery, not assessed student code.

## Notebook tasks

Use a portable `Python 3` kernelspec, stable unique cell IDs, cleared outputs, and null execution counts. State that local Jupyter is required; students restart and run all before submission; stored output is ignored; generated files are separate artifacts; and Colab is not yet an assignment submission path.

### Task 1: repair notebook state

Provide two intentionally misordered cells:

```python
adjusted_rate = base_rate + 2
```

followed later by:

```python
base_rate = 3
```

Students repair visible dependency order so a fresh run produces `base_rate == 3` and `adjusted_rate == 5`. Require a short Markdown explanation of retained kernel state, visible versus execution order, why stored output is not proof, and the repair. Automatic checks inspect the executable outcome; the explanation remains human-reviewed.

### Task 2: labeled pandas objects

Supply:

```python
reading_values = np.array([12.5, 15.0, 11.5, 15.5])

measurement_values = np.array(
    [
        [12, 18],
        [15, 23],
        [10, 17],
        [15, 23],
    ]
)
```

Students create:

- `reading_by_site`, a Series indexed by `north`, `south`, `east`, and `west`, named `reading_c`;
- `measurement_table`, a DataFrame indexed by `site-101` through `site-104`, with index name `record_id` and columns `baseline_c` and `follow_up_c`;
- `baseline_series = measurement_table["baseline_c"]`;
- `baseline_table = measurement_table[["baseline_c"]]`;
- `label_block`, using `.loc` for `site-102` through `site-103`; and
- the equivalent `position_block`, using `.iloc[1:3, 0:2]`.

Students inspect the relevant Series/DataFrame metadata and verify both blocks contain the same values. They write `label_block` to `output/labeled_block.csv`, intentionally preserving its named row index. The output columns are `record_id`, `baseline_c`, and `follow_up_c`.

### Task 3: portable CSV round trip

Students read `purchases = pd.read_csv(DATA_PATH)`, inspect shape/columns/dtypes/head, and create exactly this named mask:

```python
quantity_at_least_two = purchases["quantity"] >= 2
```

They use `.loc` with explicit source columns, add `line_total = quantity * unit_price`, then sort by:

```python
by=["line_total", "purchase_id"]
ascending=[False, True]
```

They write `output/selected_purchases.csv` with `index=False`, read it back into `round_trip`, and verify the schema, nine-row count, mask condition, derived arithmetic, and order. The expected fixture order is:

```text
P008, P003, P004, P006, P001, P011, P007, P009, P012
```

## Submitted artifacts

The required student-authored or generated artifacts are:

1. `assignment.ipynb`;
2. `output/labeled_block.csv`; and
3. `output/selected_purchases.csv`.

Students must not modify the fixture, manifest, environment records, public checker, or platform instructions.

## Public checker

`check_assignment.py` independently derives expected results from the fixture and checks:

1. valid notebook JSON;
2. fixture manifest and checksum;
3. required output files;
4. labeled-block schema and values;
5. selected-purchases schema, membership, order, and arithmetic; and
6. absence of a serialized DataFrame index in `selected_purchases.csv`.

Messages must name the violated contract. The checker must not trust stored notebook output or editable student assertions.

## Central production grader

Classroom50 tests are discoverable and must not be treated as confidential. The teacher-controlled grader should:

1. validate notebook JSON;
2. copy the submission to a temporary directory;
3. delete generated CSVs in the temporary copy;
4. ignore or strip stored outputs and execution counts;
5. append instructor-owned verification to a disposable notebook copy;
6. execute in a fresh kernel;
7. verify required variables and object types;
8. validate newly generated artifacts;
9. repeat from a relocated checkout; and
10. repeat with a second valid fixture/manifest containing different rows, values, source order, and ties.

The alternate fixture catches hard-coded reference output without creating an undisclosed requirement. It should also be treated as discoverable. The grader must emit the required `classroom50/result/v1` result and must not depend on an editable student checker.

Injected checks verify Series/DataFrame metadata, distinct bracket-selection return types, equivalent label/position selections, an index-aligned Boolean Series mask, derived arithmetic, deterministic tie-breaking, readback, and independence from prior files or kernel state.

## Grader QA matrix

Exercise at least:

- untouched starter and correct solution;
- correct source with missing generated files;
- stored output present but broken source;
- `.loc`/`.iloc` stop mistakes;
- Series/DataFrame return-type mistake;
- wrong arithmetic;
- `> 2` rather than `>= 2`;
- missing unique tie-breaker;
- serialized index column;
- hard-coded reference rows;
- absolute or `/content`/Drive-dependent paths;
- edited fixture or manifest;
- malformed notebook JSON or missing cell;
- alternate valid fixture; and
- resubmission after feedback.

The untouched starter should fail with a small, stable set of actionable messages. The correct solution must pass both public and central implementations.

## Human-review boundary

Human review checks only that the state explanation distinguishes source, kernel state, execution order, and stored output; identifies restart-and-run-all; uses understandable task headings; and contains no sensitive information. Object types, selections, arithmetic, ordering, paths, and files are automated.

## Platform disposition

Classroom50 is the delivery system for Assignments 01–11; Assignment 04 is the notebook/autograder pilot, not the point where Classroom50 begins.

Production-contract correction (2026-07-19): Classroom50 invokes the bundle's
`autograder.py` with plain Python, so the standard-library entrypoint provisions
its exact sibling requirements before loading dependency-bearing grader code.
Exact student inventory permits only delivery-owned `.classroom50.yaml` and
`.github/workflows/autograde.yaml` beyond the assignment package, ignores
only the top-level `.git/**` repository metadata tree, and rejects all other
root/workflow/grader-tree files, including a nested `ordinary/.git/**` tree.

For the initial release, require clean local Jupyter, use the supported GUI commit/push workflow, keep platform operations unassessed in `PLATFORM_CHECK.md`, and do not add an assignment Colab badge or claim that Colab submission is supported.

After the pilot proves repository save-back, submission, resubmission, and grader feedback, add one immutable-release Colab launch path for the same notebook. Preserve local Jupyter and do not fork the assignment into local and Colab editions.

## Existing artifact disposition

- Retain only the portable notebook-delivery shape, restart/run-all idea, bounded filter/derived-column work, `index=False`, and generated-output concept.
- Rewrite the README, notebook, dependencies, checks, `.gitignore`, and submission instructions.
- Drop the current generator notebook, near-solution `TIPS.md`, cleaning/GroupBy requirements, output-only tests, broad lower-bound dependencies, mutable legacy workflow, and references to nonexistent files.
