# Assignment 09 Blueprint: Entity-Aware Temporal Structure and Past-Only Evidence

Status: independently accepted implementation blueprint. Clean local Jupyter is
required. Assignment Colab launch, save-back, and submission remain conditional
on the course pilot.

## Evidence audit and replacement decision

The legacy Assignment 09 package must be replaced atomically rather than kept as
a compatibility path.

- Its Markdown generator and notebook generator disagree materially. The
  Markdown describes 200 daily patients with missing visits, 75 variable-stay
  ICU patients, and six surveillance sites over five years; the notebook creates
  50 complete daily patients, 20 complete six-month ICU histories, and three
  sites over three years. The assignment instructions and grading data therefore
  depend on which paired source is executed.
- It requires one generator plus three assignment notebooks, seven outputs, and
  broad plotting/Jupytext/statistics dependencies. This is artifact breadth
  rather than a cumulative competence assessment.
- The resampling task pools patients before or during frequency changes, and the
  rolling task averages all patients into one daily population series. Neither
  requires students to preserve entity boundaries or state the changed grain.
- It manufactures roughly 96 percent missingness by converting monthly means to
  a daily grid, then presents fill/interpolation methods as a generic exercise.
  That is not evidence-based missingness handling.
- It requires centered, expanding, exponentially weighted, and custom windows,
  advanced clock-time selection, a timezone essay, and a plotting gallery.
  Those capabilities are bonus, review-only, or outside the verified Lecture 09
  core.
- The legacy tests mostly check file existence, shallow columns, approximate
  length, and nonempty text/images. Stored outputs can satisfy much of the suite
  without demonstrating a fresh entity-aware temporal workflow.
- The legacy GitHub Classroom workflow downloads mutable tests, pins Python
  3.11, executes paired/generated notebooks, and uses lower-bound-only packages.
  It does not emit the Classroom50 result contract or distrust stored state.

The accepted Lecture 09 narrative and demos instead establish one precise
sequence: describe temporal structure; parse, localize, convert, and sort within
entity; distinguish `asfreq` from measurement-aware `resample`; separate source
missingness from grid-created rows; construct entity-scoped lag, difference, and
two past-only window meanings; audit prediction-time availability; and create a
plausible chronological handoff without fitting a model.

This blueprint replaces the legacy package with one notebook, one small pinned
fixture, six deterministic CSV artifacts, a standard-library public checker,
and a discoverable independent Classroom50 grader/self-test contract.

## Fixed role and assessment boundary

Assignment 09 has exactly three cumulative competence tasks:

1. state the temporal data contract, parse documented local clock text,
   localize once, convert once to UTC, and produce a sorted two-entity panel;
2. create an entity-scoped hourly grid and a measurement-aware two-hour
   summary while distinguishing source-value missingness from grid-created
   rows; and
3. create entity-scoped lag, difference, and past-only observation-count and
   elapsed-time windows, reject unavailable candidates at a supplied prediction
   timestamp, and construct a plausible chronological holdout.

The assignment assesses only verified Lecture 09 capabilities:

- timestamp versus period, single series versus panel, entity/entity key, row
  grain/key, sort keys, regularity, and frequency;
- exact parsing, naive versus timezone-aware values, localization versus
  conversion, UTC representation, and within-entity order;
- `asfreq` versus aggregating `resample`, explicit lowercase `h` aliases,
  measurement meaning, bin boundaries, entity retention, and missingness
  provenance;
- grouped `shift(1)`, grouped `diff()`, a previous-two-observation mean, and a
  left-closed previous-two-hour mean;
- prediction timestamp, information availability, future leakage, rejected
  centered/next candidates, and a chronological holdout; and
- portable paths, exact dependencies, fresh execution, deterministic tracked
  CSVs, schema-aware readback, and GUI-visible submission evidence.

Required work does not fill, interpolate, backward/forward fill, or otherwise
clean measurements; pool entities; compute a lead or negative shift; implement a
centered, custom, expanding, or exponentially weighted window; perform advanced
DST handling or time selection; decompose, forecast, infer, fit, select, or
evaluate a model; add a chart; use remote/network data; generate random data;
inspect a mutable date; or add a performance workflow. Lecture 10 still owns
targets, horizons, formal train/validation/test roles, baselines, metrics,
model selection, and evaluation.

The one `merge(..., validate="one_to_one")` inside the past-feature function is
the accepted Lecture 09 mechanism for returning an elapsed-time result to source
grain. It is bounded prerequisite reuse from Lecture 06, not a new joining task.

## Student repository contract

The future student-facing package contains exactly this instructional surface,
plus Classroom50-owned metadata added by the delivery system:

```text
09/assignment/
├── .gitignore
├── .python-version
├── PLATFORM_CHECK.md
├── README.md
├── assignment.ipynb
├── check_assignment.py
├── requirements.txt
├── data/
│   ├── fixture.json
│   └── zone_co2_readings.csv
└── output/
    └── .gitkeep
```

The instructor repository may additionally contain `_grader_selftest/` with a
production-grader mirror, adversarial harness, exact grader dependencies, and
maintenance notes. That directory is excluded from student templates and
production submissions. There is one canonical notebook and no paired Markdown
source, generator, `.github/` workflow/test tree, solution, or completed starter
output.

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
libraries are host/grader tooling, not student notebook imports. The
instructor-only grader candidate additionally pins `nbclient==0.10.2`,
`nbformat==5.10.4`, and `ipykernel==6.29.5` alongside NumPy and pandas. These
are the same explicit notebook-execution candidates already exercised by the
accepted Lecture 09 local demo gate; the complete course-wide environment still
requires its separate release freeze.

The notebook verifies the existing environment and directs a learner with a
failed check to `PLATFORM_CHECK.md`; it does not install packages.

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

It must not ignore `output/`, CSV, JSON, or notebooks. All six required CSVs
must appear in VS Code Source Control or GitHub Desktop so students can commit
and submit them. `output/.gitkeep` preserves the empty starter directory. The
required Git path remains GUI-first; command-line Git is not assessed.

`PLATFORM_CHECK.md` gives direct clean-local-Jupyter setup, interpreter/kernel
verification, restart-and-run-all, GUI commit/push, Classroom50 feedback, and
resubmission steps. It contains no Colab badge and does not claim that Colab
edits save back to the repository.

## Exact assignment-only prepared fixture

The assignment uses one course-authored, synthetic, nonidentifying indoor-air
sensor table. It is distinct from the Lecture 09 station-temperature demo in
domain, entity labels, timestamp range, source timezone, row count, values,
missingness positions, cutoff, and expected outputs. It contains no real
building, occupant, customer, patient, or identifying data.

Both fixture files use UTF-8, LF line endings, and a final newline. Students do
not generate, download, clean, fill, or rewrite them.

### `data/zone_co2_readings.csv`

- Bytes: 380.
- SHA-256:
  `c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4`.
- Row grain: one recorded CO2 reading for one zone and documented local
  timestamp.
- Rows: 12.
- Ordered columns: `zone,recorded_at,co2_ppm`.
- Source clock zone: `America/New_York`; every supplied January 2026 clock time
  is unambiguous.

```csv
zone,recorded_at,co2_ppm
studio,2026-01-20 12:00,560.0
atrium,2026-01-20 07:00,400.0
studio,2026-01-20 07:00,500.0
atrium,2026-01-20 14:00,500.0
studio,2026-01-20 09:00,520.0
atrium,2026-01-20 10:00,
studio,2026-01-20 14:00,600.0
atrium,2026-01-20 08:00,420.0
studio,2026-01-20 10:00,540.0
atrium,2026-01-20 11:00,460.0
studio,2026-01-20 13:00,580.0
atrium,2026-01-20 13:00,480.0
```

The source order is deliberately interleaved and out of time order. Atrium has
one source row whose CO2 measurement is missing. The canonical prepared UTC
histories are:

```text
atrium: 12:00 400; 13:00 420; 15:00 missing; 16:00 460; 18:00 480; 19:00 500
studio: 12:00 500; 14:00 520; 15:00 540; 17:00 560; 18:00 580; 19:00 600
```

Atrium gaps are `[1, 2, 1, 2, 1]` hours and Studio gaps are
`[2, 1, 2, 1, 1]`; both histories are irregular.

### `data/fixture.json`

`fixture.json` has exactly 473 bytes and SHA-256
`27558bc4da7738775879501a6f11a0a9d874f3948823e54bb5e82ab91a02d703`:

```json
{
  "fixture_id": "a09-temporal-panel-v1",
  "provenance": "Course-authored synthetic indoor-air sensor readings; no real, identifying, or occupant data.",
  "path": "zone_co2_readings.csv",
  "row_grain": "one recorded CO2 reading for one zone and local timestamp",
  "row_count": 12,
  "columns": [
    "zone",
    "recorded_at",
    "co2_ppm"
  ],
  "sha256": "c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4",
  "source_timezone": "America/New_York"
}
```

The implementation must preserve those exact bytes and freeze protected-source
hashes only after final README, checker, and protected notebook cells settle.

## Portable protected setup

The first code cell is supplied and protected. It must:

1. search upward from the launch directory for either flattened
   `data/fixture.json` or course-root `09/assignment/data/fixture.json`;
2. define `ASSIGNMENT_ROOT`, `DATA_DIR`, `OUTPUT_DIR`, `FIXTURE_PATH`, and the
   exact six output paths from the discovered assignment root;
3. validate the manifest's exact bytes/hash, keys/values, safe relative fixture
   path, CSV bytes/hash, final newlines, row count, and ordered columns before
   pandas reads the CSV;
4. create only `OUTPUT_DIR` and, after fixture validation, delete only the six
   named stale outputs while preserving `.gitkeep` and unrelated files;
5. import and assert Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3; and
6. stop with an actionable exception when the environment, path, manifest, or
   fixture contract fails.

It must not install packages, embed fallback fixture bytes, inspect another
notebook, access a network, use an absolute repository path, depend on
`/content`, inspect a mutable date, prompt for upload, mount Drive, or rewrite a
fixture. A missing or corrupt assignment fixture stops before pandas parsing or
owned-output cleanup.

The student load cell reads `zone` and `recorded_at` as pandas `string` and
`co2_ppm` as NumPy `float64`; verifies exact shape, ordered columns, two zone
labels, nonmissing entity/time text, unique source entity-time keys, and one
missing measurement; and saves `raw_snapshot = raw_readings.copy(deep=True)`.

## Definition-before-independent-use contract

The README and protected notebook cells define every required term before its
first demanding use.

### Before Task 1

| Term | Required plain-language meaning |
|---|---|
| temporal ordering | row order determines what counts as previous, inside a window, or available |
| timestamp / period | an instant on a timeline / a span with a start and end |
| entity / entity key | the unit with one ordered history / the column identifying that unit |
| single series / panel | one entity history / multiple entity histories |
| row grain / row key / sort keys | what one row represents / columns identifying it / columns defining computation order |
| regular / irregular | one expected adjacent spacing / varying adjacent gaps |
| frequency | an expected time grid or calendar offset, not proof of an observed value |
| parsing | converting documented text to datetime values |
| naive / timezone-aware timestamp | clock text without an offset / an unambiguous instant with zone/offset |
| localization / conversion | attach the documented source zone / express an aware instant in another zone |
| DatetimeIndex | a pandas index whose labels are datetime values |

### Before Task 2

| Term | Required plain-language meaning |
|---|---|
| upsampling / downsampling | request a finer grid / combine observations into coarser bins |
| `asfreq` / `resample` | conform exact labels without combining / group timestamps into bins for a justified summary |
| source-value missingness | a supplied row exists but its measurement is missing |
| grid-created row | a requested label has no supplied row |
| measurement meaning | what a value represents and how it can legitimately be combined |
| left-closed, left-labeled bin | include the left boundary and use it as the interval label |

### Before Task 3

| Term | Required plain-language meaning |
|---|---|
| lag / lead / difference | earlier same-entity value / later value / current minus previous same-entity value |
| trailing window | a moving summary restricted to the ordered past/current boundary |
| observation-count / elapsed-time window | fixed number of rows / observations inside a stated duration |
| candidate feature | a value that might later be supplied to a prediction procedure |
| prediction timestamp | the instant at which that later prediction would be issued |
| information availability | whether every source value required by a candidate was known then |
| future-derived candidate / future leakage | requires later data / incorrectly treats later data as available |
| chronological holdout | a later block set aside from a strictly earlier block |

The assignment may mention a centered candidate only to define and reject it.
No student code computes that candidate or a lead.

## Exact notebook contract

`assignment.ipynb` uses notebook-format major version 4 and minor version 5, a
portable `Python 3` kernelspec, exactly the 26 cells below in order, stable
globally unique IDs, null execution counts, and zero stored output in the
released starter. Protected cells are complete. Student cells contain
actionable TODO scaffolds without solution fragments. There is no hidden
prerequisite cell.

The header states that clean local Jupyter is required, Assignment Colab is not
supported, data are synthetic, restart-and-run-all is mandatory, stored output
is never execution evidence, and the notebook plus six generated CSVs are
separate GUI-visible submission artifacts.

### Cells 0--3: orientation, terms, and load

0. `a09-header` (protected Markdown): title, exact three-task progression,
   privacy, local-only platform boundary, six output names, fresh-execution
   rule, and GUI Git visibility.
1. `a09-setup` (protected code): portable path, exact environment,
   manifest/checksum validation, output constants, and exact stale-output
   removal.
2. `a09-terms-data` (protected Markdown): Task 1 term ledger, fixture grain,
   keys, source zone, evidentiary limits, and the distinction between source
   clock text and UTC instants.
3. `a09-load` (student code): explicit-dtype read into `raw_readings`, exact
   schema/key/missingness checks, and a deep `raw_snapshot`.

### Cells 4--9: Task 1 -- classify, parse, localize, convert, and sort

4. `a09-task1-contract` (student Markdown): in the student's own words, state
   timestamp versus period meaning, input grain, entity and row keys, panel
   structure, source/output zones, required sort order, and expected regularity.
5. `a09-task1-values` (student code): assign these exact machine-readable values
   before parsing:

   ```python
   temporal_representation = "timestamp"
   input_row_grain = "one recorded CO2 reading for one zone and local timestamp"
   entity_key = ["zone"]
   row_key = ["zone", "recorded_at"]
   sort_keys = ["zone", "recorded_at"]
   series_structure = "panel"
   predicted_entities = ["atrium", "studio"]
   source_timezone = "America/New_York"
   output_timezone = "UTC"
   predicted_source_rows = 12
   predicted_gap_hours = {
       "atrium": [1.0, 2.0, 1.0, 2.0, 1.0],
       "studio": [2.0, 1.0, 2.0, 1.0, 1.0],
   }
   predicted_regularity = {"atrium": "irregular", "studio": "irregular"}
   ```

6. `a09-prepare-function` (student code): define
   `prepare_temporal_panel(reading_table, source_timezone)`. It deep-copies its
   argument; parses `recorded_at` using exact format `%Y-%m-%d %H:%M`; verifies
   that parsed values are naive; localizes once to the caller's zone; converts
   once to UTC; creates `source_row` as NumPy `int64` value 1; stable-sorts by
   `zone,recorded_at`; resets to a zero-based index; validates nonmissing unique
   entity-time keys and monotonic within-zone time; and returns exactly
   `zone,recorded_at,co2_ppm,source_row` without mutating the input.
7. `a09-task1-run` (student code): create `prepared_panel`; verify 12 conserved
   rows, Atrium then Studio, UTC dtype, exact dtypes/order, unique keys,
   within-zone monotonicity, exact gap sequences and irregularity, repeated
   timestamps across zones as valid panel rows, and raw-source immutability.
   Create `indexed_panel = prepared_panel.set_index("recorded_at")`, verify its
   UTC `DatetimeIndex` while keeping `zone` as a column, and use a named one-zone
   subset to demonstrate a single series without changing the full panel.
8. `a09-task1-save` (student code): write
   `output/prepared_panel.csv` with UTF-8, LF, `index=False`, `na_rep=""`, and a
   final newline; restore pandas string, UTC datetime, `float64`, and `int64` on
   readback and assert exact frame equality and bytes.
9. `a09-task1-explain` (student Markdown): explain localization versus
   conversion, why repeated UTC timestamps across zones are valid, why sorting
   by timestamp alone is insufficient, and why varying gaps make both entity
   histories irregular.

### Cells 10--15: Task 2 -- change frequency with measurement meaning

10. `a09-task2-prompt` (protected Markdown): define the complete Task 2 term
    ledger above; state the zone-hour-grid and zone--two-hour output grains;
    explain instantaneous CO2 meaning and the additive source-row counter; and
    state that exact-label hourly `asfreq` requires every source timestamp to
    fall on a whole UTC hour and that the hourly-grid function must reject an
    off-grid label rather than silently lose it. Explicitly prohibit
    fill/interpolation/zero replacement.
11. `a09-hourly-function` (student code): define
    `build_hourly_grid(prepared_table)`. It first validates that every timestamp
    is exactly on a whole UTC hour and raises an actionable `ValueError` for an
    off-grid label. It then uses a temporary UTC datetime index, groups by
    `zone` with `observed=True`, `sort=True`, and `dropna=True`, and uses
    `resample("h").asfreq()` with no fill. After resetting keys to ordinary
    columns, it adds Boolean `grid_created_row` and `source_value_missing` using
    the source marker, returns the six exact columns, and does not mutate input.
12. `a09-summary-function` (student code): define
    `build_two_hour_summary(prepared_table)`. It preserves zone identity, uses
    grouped `resample("2h", closed="left", label="left")`, and named
    aggregation to return exact columns
    `zone,recorded_at,mean_co2_ppm,reading_count`. It averages only `co2_ppm`,
    sums only `source_row`, and does not mutate input.
13. `a09-task2-run` (student code): create `hourly_grid` and
    `two_hour_summary`; verify exact schemas/dtypes/order/values, two entities,
    16 hourly rows, four grid-created rows, one distinct source-value-missing
    row, mutually exclusive provenance flags, 12 source markers, eight two-hour
    bins, reading-count conservation at 12, and the exact missing and populated
    bin values below.
14. `a09-task2-save` (student code): write
    `output/hourly_grid.csv` and `output/two_hour_summary.csv` with exact
    UTF-8/LF/index/empty-missing conventions; perform schema-aware readbacks,
    including UTC timestamps, Boolean flags, and the hourly `source_row`
    promotion to `float64`; assert exact frames and bytes.
15. `a09-task2-explain` (student Markdown): explain `asfreq` versus aggregating
    resample, source versus grid missingness, the state-variable mean and
    additive count, left bin boundaries, and why Atrium's missing 14:00-bin mean
    does not mean that no source row was recorded.

### Cells 16--24: Task 3 -- past-only features and availability

16. `a09-task3-prompt` (protected Markdown): define every Task 3 term above,
    state that both windows exclude the current row, require entity scope, name
    the supplied Studio 18:00 UTC prediction timestamp, and defer formal modeling
    roles to Lecture 10.
17. `a09-features-function` (student code): define
    `build_past_features(prepared_table)`. It preserves source grain/order and
    input immutability; uses grouped `shift(1)` and `diff()`; uses grouped
    `shift(1).rolling(window=2, min_periods=1).mean()` inside a same-index
    transform for the prior-two-observation mean; uses entity-grouped
    `rolling("2h", closed="left", min_periods=1).mean()` for elapsed time; and
    returns that elapsed result to `zone,recorded_at` through a left
    `validate="one_to_one"` merge. It returns exactly seven columns and computes
    no lead, centered, future, custom, EWM, or expanding value.
18. `a09-task3-features-run` (student code): create `temporal_features`; verify
    exact source rows/order, all `float64` measurement/feature dtypes, missing
    first lag/difference per zone, source immutability, and the exact Studio
    17:00/18:00 values below.
19. `a09-availability-values` (student code): assign
    `prediction_zone = "studio"` and
    `prediction_timestamp = pd.Timestamp("2026-01-20 18:00", tz="UTC")`, then
    construct exactly this supplied inventory without computing either rejected
    candidate:

    ```text
    candidate                         latest required UTC  available  decision
    calendar hour                     18:00                True       keep
    previous recorded CO2             17:00                True       keep
    centered three-observation mean   19:00                False      reject
    next recorded CO2                 19:00                False      reject
    ```

    `availability_decisions` has pandas string candidate/decision columns, a UTC
    datetime column, and a NumPy Boolean availability column.
20. `a09-blocks-function` (student code): define
    `build_chronological_blocks(prepared_table, holdout_start)`. It deep-copies
    input, labels rows before the supplied UTC cutoff `earlier` and rows at/after
    it `later_holdout`, preserves exact source columns/order/dtypes, adds one
    pandas string `block`, and performs no target/model/evaluation work.
21. `a09-task3-run` (student code): create `chronological_blocks`; verify the
    availability sequence `keep,keep,reject,reject`, eight earlier and four
    later rows, both zones in each block, maximum earlier timestamp 17:00 UTC,
    minimum holdout timestamp 18:00 UTC, strict separation, and source
    immutability.
22. `a09-task3-save` (student code): write
    `output/temporal_features.csv`,
    `output/availability_decisions.csv`, and
    `output/chronological_blocks.csv` with exact UTF-8/LF/index/missing-value
    conventions; restore exact semantic dtypes on readback and assert exact
    frames and bytes.
23. `a09-task3-explain` (student Markdown): explain observation-count versus
    elapsed-time windows using Studio 17:00, why first values are missing per
    entity, why historical presence is not prediction-time availability, why
    the centered/next candidates are rejected without computation, and why the
    blocks are only a plausible Lecture 10 handoff.
24. `a09-synthesis` (student Markdown): concisely connect structure, frequency,
    provenance, entity boundaries, past-only computation, and availability;
    name one limitation of the synthetic fixture without introducing a model.

### Cell 25: supplied final verification

25. `a09-final-verify` (protected code): recheck fixture integrity, raw
    immutability, exact namespace names, source/grid/resample/feature/
    availability/block invariants, all six paths, exact readbacks, and no extra
    generated artifact. It prints one local-readiness summary and directs the
    learner to restart/run-all and then run `python check_assignment.py`. It
    does not award points or claim a human-reasoning pass.

## Exact function interfaces and behavioral variation

The five public functions are:

```python
prepare_temporal_panel(reading_table, source_timezone)
build_hourly_grid(prepared_table)
build_two_hour_summary(prepared_table)
build_past_features(prepared_table)
build_chronological_blocks(prepared_table, holdout_start)
```

The prepare function accepts any complete raw table with exactly the documented
three-column roles, nonmissing entity/time text, unique entity/time text pairs,
exact `%Y-%m-%d %H:%M` lexical timestamps, and a caller-supplied valid source
timezone. A missing CO2 measurement is permitted. The other functions accept a
valid prepared table with pandas string entity, UTC datetime, `float64` CO2,
`int64` source marker equal to 1, unique entity-time keys, and time sorted within
entity. `build_hourly_grid` has the additional exact-label precondition that
every timestamp falls on a whole UTC hour; it validates this and rejects
off-grid input with `ValueError` before calling `asfreq`, so a source observation
cannot be silently discarded. The two disclosed raw fixtures satisfy this
precondition. These are prepared-data preconditions, not invitations to clean
invalid inputs.

Functions derive labels, rows, timestamps, values, gaps, output length, and
cutoff membership from their arguments. They must not depend on canonical zone
names, timestamp strings, values, row counts, fixture paths, output files, or
global canonical DataFrames; mutate an input; read/write files; fill/clean;
pool entities; access a network; or perform later-scope analysis. File writes
remain in save cells so central alternate calls create no stray artifacts.

The grader publishes an alternate raw table and timezone below. It varies
labels, row order, nondefault index, timezone, row count, timestamps, values,
missing position, cutoff, and expected window contrast. Hidden-by-convention
secrecy is not part of integrity; correct behavior on the disclosed variation
is required.

## Exact canonical behavior and artifact bytes

All six outputs use UTF-8, comma delimiters, `lineterminator="\n"`,
`na_rep=""`, ordered columns, a final newline, and `index=False`. Completed
outputs are GUI-visible and deterministic. Under CPython 3.12.13, NumPy 2.0.2,
and pandas 3.0.3 their exact contracts are:

| Artifact | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `prepared_panel.csv` | 12 | 523 | `e29aa2ac53cffe29c2f412170100e0725ce4b6a2e0cfdd09e5f1cb92fd5fcd64` |
| `hourly_grid.csv` | 16 | 912 | `d4de5178f7ca56960061efb1d263ff4022a9608bc44a6f679c397cb814c150c0` |
| `two_hour_summary.csv` | 8 | 367 | `0805cb42799880b85afdee35b0af36c53d606311e1a83a0a995509c647b9d999` |
| `temporal_features.csv` | 12 | 779 | `a83d7b858bdcf0203f211cd4dbfc907f0530d132a989ee6ead0fc46e6401d0bb` |
| `availability_decisions.csv` | 4 | 310 | `07128f0c67a5765d115c8feb7f3a5ee547450b985b48647c3dcc2324d27a4607` |
| `chronological_blocks.csv` | 12 | 649 | `ddde39feea7e3b864088919675fe279dee82021d514980bfd9271cfacf9ec0d2` |

Canonical semantic dtypes are:

| Output | Exact dtypes in column order |
|---|---|
| prepared | `string`; `datetime64[us, UTC]`; `float64`; `int64` |
| hourly | `string`; `datetime64[us, UTC]`; `float64`; `float64`; `bool`; `bool` |
| two-hour | `string`; `datetime64[us, UTC]`; `float64`; `int64` |
| features | `string`; `datetime64[us, UTC]`; five `float64` columns |
| availability | `string`; `datetime64[us, UTC]`; `bool`; `string` |
| blocks | `string`; `datetime64[us, UTC]`; `float64`; `int64`; `string` |

The hourly grid has 16 rows: eight labels per entity from 12:00 through 19:00
UTC. Grid-created rows are Atrium 14:00/17:00 and Studio 13:00/16:00. Atrium
15:00 is the sole source-value-missing row. Those categories are mutually
exclusive, and 12 nonmissing source markers conserve all source rows.

The exact two-hour summary values are:

```text
zone     UTC bin  mean_co2_ppm  reading_count
atrium   12:00    410.0         2
atrium   14:00    missing       1
atrium   16:00    460.0         1
atrium   18:00    490.0         2
studio   12:00    500.0         1
studio   14:00    530.0         2
studio   16:00    560.0         1
studio   18:00    590.0         2
```

The count total is 12. Atrium's missing 14:00-bin mean comes from its one 15:00
source row whose measurement is missing; it is not an empty interval.

The canonical feature checks include:

- first lag and difference missing for Atrium and Studio;
- Studio 17:00: lag `540.0`, difference `20.0`, previous-two-observation mean
  `530.0`, and previous-two-hour mean `540.0`;
- Studio 18:00: lag `560.0`, difference `20.0`, previous-two-observation mean
  `550.0`, and previous-two-hour mean `560.0`; and
- exactly `keep, keep, reject, reject` in the availability table and block
  counts 8/4 with both entities retained.

The exact serialized output blocks are included below so implementation and
review do not infer row order or datetime formatting.

<details>
<summary>Exact six CSV serializations</summary>

```csv
zone,recorded_at,co2_ppm,source_row
atrium,2026-01-20 12:00:00+00:00,400.0,1
atrium,2026-01-20 13:00:00+00:00,420.0,1
atrium,2026-01-20 15:00:00+00:00,,1
atrium,2026-01-20 16:00:00+00:00,460.0,1
atrium,2026-01-20 18:00:00+00:00,480.0,1
atrium,2026-01-20 19:00:00+00:00,500.0,1
studio,2026-01-20 12:00:00+00:00,500.0,1
studio,2026-01-20 14:00:00+00:00,520.0,1
studio,2026-01-20 15:00:00+00:00,540.0,1
studio,2026-01-20 17:00:00+00:00,560.0,1
studio,2026-01-20 18:00:00+00:00,580.0,1
studio,2026-01-20 19:00:00+00:00,600.0,1
```

```csv
zone,recorded_at,co2_ppm,source_row,grid_created_row,source_value_missing
atrium,2026-01-20 12:00:00+00:00,400.0,1.0,False,False
atrium,2026-01-20 13:00:00+00:00,420.0,1.0,False,False
atrium,2026-01-20 14:00:00+00:00,,,True,False
atrium,2026-01-20 15:00:00+00:00,,1.0,False,True
atrium,2026-01-20 16:00:00+00:00,460.0,1.0,False,False
atrium,2026-01-20 17:00:00+00:00,,,True,False
atrium,2026-01-20 18:00:00+00:00,480.0,1.0,False,False
atrium,2026-01-20 19:00:00+00:00,500.0,1.0,False,False
studio,2026-01-20 12:00:00+00:00,500.0,1.0,False,False
studio,2026-01-20 13:00:00+00:00,,,True,False
studio,2026-01-20 14:00:00+00:00,520.0,1.0,False,False
studio,2026-01-20 15:00:00+00:00,540.0,1.0,False,False
studio,2026-01-20 16:00:00+00:00,,,True,False
studio,2026-01-20 17:00:00+00:00,560.0,1.0,False,False
studio,2026-01-20 18:00:00+00:00,580.0,1.0,False,False
studio,2026-01-20 19:00:00+00:00,600.0,1.0,False,False
```

```csv
zone,recorded_at,mean_co2_ppm,reading_count
atrium,2026-01-20 12:00:00+00:00,410.0,2
atrium,2026-01-20 14:00:00+00:00,,1
atrium,2026-01-20 16:00:00+00:00,460.0,1
atrium,2026-01-20 18:00:00+00:00,490.0,2
studio,2026-01-20 12:00:00+00:00,500.0,1
studio,2026-01-20 14:00:00+00:00,530.0,2
studio,2026-01-20 16:00:00+00:00,560.0,1
studio,2026-01-20 18:00:00+00:00,590.0,2
```

```csv
zone,recorded_at,co2_ppm,co2_lag_1,co2_difference,mean_previous_2_observations,mean_previous_2h
atrium,2026-01-20 12:00:00+00:00,400.0,,,,
atrium,2026-01-20 13:00:00+00:00,420.0,400.0,20.0,400.0,400.0
atrium,2026-01-20 15:00:00+00:00,,420.0,,410.0,420.0
atrium,2026-01-20 16:00:00+00:00,460.0,,,420.0,
atrium,2026-01-20 18:00:00+00:00,480.0,460.0,20.0,460.0,460.0
atrium,2026-01-20 19:00:00+00:00,500.0,480.0,20.0,470.0,480.0
studio,2026-01-20 12:00:00+00:00,500.0,,,,
studio,2026-01-20 14:00:00+00:00,520.0,500.0,20.0,500.0,500.0
studio,2026-01-20 15:00:00+00:00,540.0,520.0,20.0,510.0,520.0
studio,2026-01-20 17:00:00+00:00,560.0,540.0,20.0,530.0,540.0
studio,2026-01-20 18:00:00+00:00,580.0,560.0,20.0,550.0,560.0
studio,2026-01-20 19:00:00+00:00,600.0,580.0,20.0,570.0,570.0
```

```csv
candidate,latest_required_timestamp,available_by_prediction_time,decision
calendar hour,2026-01-20 18:00:00+00:00,True,keep
previous recorded CO2,2026-01-20 17:00:00+00:00,True,keep
centered three-observation mean,2026-01-20 19:00:00+00:00,False,reject
next recorded CO2,2026-01-20 19:00:00+00:00,False,reject
```

```csv
zone,recorded_at,co2_ppm,source_row,block
atrium,2026-01-20 12:00:00+00:00,400.0,1,earlier
atrium,2026-01-20 13:00:00+00:00,420.0,1,earlier
atrium,2026-01-20 15:00:00+00:00,,1,earlier
atrium,2026-01-20 16:00:00+00:00,460.0,1,earlier
atrium,2026-01-20 18:00:00+00:00,480.0,1,later_holdout
atrium,2026-01-20 19:00:00+00:00,500.0,1,later_holdout
studio,2026-01-20 12:00:00+00:00,500.0,1,earlier
studio,2026-01-20 14:00:00+00:00,520.0,1,earlier
studio,2026-01-20 15:00:00+00:00,540.0,1,earlier
studio,2026-01-20 17:00:00+00:00,560.0,1,earlier
studio,2026-01-20 18:00:00+00:00,580.0,1,later_holdout
studio,2026-01-20 19:00:00+00:00,600.0,1,later_holdout
```

</details>

The starter contains only `output/.gitkeep`. Setup removes only the six named
CSVs. A fresh run recreates deleted/stale/corrupt owned outputs, preserves an
unrelated sentinel during setup, and reproduces the hashes on the pinned grader
platform. Final submission inventory rejects the sentinel or any extra file
until the student removes it.

## Protected and student-editable surfaces

Implementation freezes course-owned hashes for:

- `.python-version`, `requirements.txt`, `.gitignore`, `README.md`, and
  `PLATFORM_CHECK.md`;
- `check_assignment.py`;
- `data/fixture.json` and `data/zone_co2_readings.csv`; and
- notebook cells `a09-header`, `a09-setup`, `a09-terms-data`,
  `a09-task2-prompt`, `a09-task3-prompt`, and `a09-final-verify`, including ID,
  type, position, and exact source.

Only designated student Markdown/code cells and the six regenerated CSVs are
student work. The central grader independently owns protected expectations;
editing the public checker cannot weaken production checks. Staff regenerate
protected hashes only when intentionally releasing a new assignment version.

## Student-visible public checker

`check_assignment.py` uses only the Python standard library. It must not import
pandas/NumPy/nbclient, execute arbitrary notebook source, trust stored output,
read grader secrets, claim a score, or serve as production authority. It must:

1. locate flattened and course-root layouts from `__file__`;
2. validate exact manifest and fixture bytes/semantics/hashes, safe paths, final
   newlines, row count, ordered columns, and package inventory;
3. parse notebook JSON and require exactly the 26 IDs/types/order above,
   nbformat 4.5, portable kernelspec, globally unique IDs, and unedited protected
   cells; submitted outputs/execution counts are ignored as evidence;
4. detect untouched TODO scaffolds and all five public function names;
5. parse executable student code with `ast`; require the bounded APIs and
   explicit grouping/time policies; reject pooled operations, negative shifts,
   `center=True`, computed leads/future candidates, extra/unapproved output
   paths, later/bonus APIs, absolute/content/Drive/upload/network paths,
   embedded fixture fallback, random data, or mutable dates;
6. require exactly the six CSVs plus `.gitkeep` under `output/`, reject legacy
   `q1_`/`q2_`/`q3_` artifacts and unexpected generated files;
7. verify every CSV's exact canonical header, row count, bytes, final newline,
   and SHA-256; and
8. return nonzero with a small task-grouped set of actionable messages, or one
   readiness summary without claiming a grade or prose-quality pass.

The checker applies scope scanning to code, not Markdown explanations. The
allow/deny contract must distinguish required calls from prohibited variants:

- permit exact-format `to_datetime`, one localization, one UTC conversion,
  stable sort, bounded `set_index`/`reset_index`, required GroupBy, `resample`,
  `asfreq`, named aggregation, `shift(1)`, `diff`, the two specified `rolling`
  forms, same-index transform, and the one validated merge;
- reject `shift(-1)` or any negative shift, centered windows, expanding/EWM,
  custom rolling apply, interpolation/fill/backfill, frequency inference as a
  substitute for the documented contract, advanced period/offset work, and
  date/time selection breadth;
- reject plotting/image libraries and output; statsmodels/scipy/scikit-learn,
  model/formula/prediction/evaluation APIs; decomposition/forecasting; and
  GroupBy `apply`/filter or advanced MultiIndex manipulation; and
- reject requests/urllib/http clients, remote readers, random generators,
  `datetime.now`/`Timestamp.now`/`today`, credentials, `/content`, uploads, or
  Drive mounts.

Source scanning is early feedback, not behavior proof. The central grader
fresh-executes and calls functions on alternate values.

## Classroom50 central grader

The production grader is teacher-controlled and discoverable, contains no
solution, credential, confidential record, or test dependent on secrecy, and
does not import or trust the editable public checker. It must:

1. independently validate all protected package files, fixture bytes, notebook
   topology/cells, output visibility policy, and source scope;
2. copy the submission to an isolated temporary directory, remove all six
   outputs, clear stored notebook outputs/counts, append grader-owned checks to
   a disposable notebook copy, and execute from a fresh pinned kernel;
3. exercise flattened Classroom50, course-root, relocated,
   nested-within-assignment, and path-with-spaces layouts;
4. verify canonical namespace values, exact function signatures and call
   policies, raw/source immutability, schemas/dtypes/order, timezone and entity
   invariants, grid provenance, resample conservation, past-only boundaries,
   availability, chronological separation, output bytes, and schema-aware
   readbacks;
5. call all five functions on the disclosed alternate complete raw/prepared
   table below, including its shuffled nondefault index, different timezone,
   labels, values, row count, missing position, gaps, and cutoff;
6. use static and behavioral checks together to reject canonical hard-coding,
   pooled entity operations, output-only solutions, positional alignment,
   incorrect window boundaries, future computation, and mutated inputs;
7. run missing/corrupt/line-ending-changed fixture, edited manifest,
   malformed/protected/reordered notebook, stored-output, stale/deleted/corrupt
   output, unrelated-file, repeat, and corrected-resubmission cases; prove setup
   preserves a foreign sentinel while final exact inventory rejects it;
8. direct the bounded human review to student-authored Markdown through the
   context-supplied `review` URL; fresh execution is behavior evidence and
   submitted stored output remains untrusted; and
9. write `./result.json` plus actionable grading logs.

The grader emits exactly the official hyphenated
`classroom50/result/v1` topology:

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
      "test-name": "Task 1 temporal structure and preparation",
      "passed": false,
      "score": 0,
      "max-score": 20
    }
  ]
}
```

The seven metadata values `classroom`, `assignment`, `submission`, `commit`,
`release`, `review`, and `datetime` come from Classroom50 runner context or the
grader clock, never student code. Under the official runner contract current on
2026-07-18, the reference grader reads `CLASSROOM`, `ASSIGNMENT`,
`SUBMISSION_TAG`, `COMMIT_URL`, and `RELEASE_URL`; reads `REVIEW_URL` with
fallback to `COMMIT_URL`; and generates `datetime` from the current UTC clock in
exact `YYYY-MM-DDTHH:MM:SSZ` form. Missing/empty required context is an
infrastructure error. An absent/empty `REVIEW_URL` is not: it takes the
documented commit-URL fallback. The grader writes the result to
`./result.json`. This contract comes directly from the official Classroom50
[Autograders contract and template](https://github.com/foundation50/classroom50/wiki/Autograders#contract),
not invented `CLASSROOM50_*` variables.

Every test has exactly `test-name`, `passed`, `score`, and `max-score`.
Runner-stamped `owner`, `assignment_type`, and `submitted_by` are optional and
must not be required from or invented by grader/student code. Standard
`GITHUB_*`, `OWNER`/`USERNAME`, `ASSIGNMENT_TYPE`, and `PAGES_BASE_URL` may be
present but are not needed to construct this assignment's required result.
Failure detail belongs in logs or the review/release body, not incompatible
result fields. A completed failing grade writes a valid result and exits zero;
nonzero is reserved for grader infrastructure failure.

Automated groups sum to 90:

| Test group | Maximum |
|---|---:|
| template, environment, fixture, notebook, and protected integrity | 10 |
| Task 1 temporal contract, parsing/timezone, entity order, and prepared output | 20 |
| Task 2 entity-scoped grid, provenance, resample meaning, and two outputs | 25 |
| Task 3 past-only features, availability, chronological blocks, and three outputs | 30 |
| portability, scope, stale/repeat output, and resubmission | 5 |
| **Automated result maximum** | **90** |

The separate human maximum is not fabricated inside `result.json`.

## Discoverable alternate prepared table

The grader publishes this alternate raw table. It uses source timezone
`America/Chicago`, labels `gallery` and `lab`, raw index labels
`[31, 4, 88, 12, 57, 6, 73, 20, 2, 45]`, and this shuffled row order:

```text
index  zone     recorded_at       co2_ppm
31     lab      2026-02-10 11:00  850.0
4      gallery  2026-02-10 06:00  700.0
88     lab      2026-02-10 06:00  800.0
12     gallery  2026-02-10 12:00  750.0
57     lab      2026-02-10 08:00  820.0
6      gallery  2026-02-10 09:00  missing
73     lab      2026-02-10 12:00  860.0
20     gallery  2026-02-10 07:00  710.0
2      lab      2026-02-10 09:00  830.0
45     gallery  2026-02-10 10:00  730.0
```

Required alternate properties are:

- prepared UTC order Gallery 12:00/13:00/15:00/16:00/18:00, then Lab
  12:00/14:00/15:00/17:00/18:00, with 10 rows and one source missing value;
- Gallery gaps `[1, 2, 1, 2]`, Lab gaps `[2, 1, 2, 1]`, and both irregular;
- a 14-row hourly grid with four grid-created rows (Gallery 14:00/17:00 and Lab
  13:00/16:00), Gallery 15:00 as the sole source-value-missing row, and ten
  conserved source markers;
- eight two-hour bins: Gallery means/counts
  `(705,2),(missing,1),(730,1),(750,1)` and Lab
  `(800,1),(825,2),(850,1),(860,1)` in 12/14/16/18 UTC order;
- at Lab 17:00, lag `830`, difference `20`, previous-two-observation mean `825`,
  and previous-two-hour mean `830`; and
- cutoff `2026-02-10 17:00 UTC` produces seven earlier and three later rows,
  strict separation, and both zones in both blocks.

The harness may add further disclosed property variations preserving the same
schema/preconditions, but may not introduce a cleaning rule, missing entity/time
key, advanced timezone policy, new API, or later-course objective.

## Adversarial QA matrix

Before release, the author self-test and independent reviewer must exercise at
least:

- untouched starter, correct solution, multiple partials, and corrected
  resubmission;
- malformed JSON; missing, duplicated, reordered, type-changed, or edited cell
  IDs; protected file/cell edits; and a modified public checker;
- missing, renamed, extra, line-ending-changed, or byte-corrupted fixture and an
  edited manifest/timezone;
- stored correct-looking tables with broken/unexecuted source; deleted,
  stale, binary-corrupt, truncated, or legacy outputs; deterministic repeat;
  unrelated-output preservation versus final exact-inventory rejection;
- flattened, course-root, relocated, nested-within-assignment, and spaces path
  layouts;
- parsing without exact format, localize/convert reversed or repeated, wrong
  source timezone, retained naive timestamps, unstable/wrong-key sort, timestamp-
  only duplicate rejection, pooled entity keys, missing source marker, source
  mutation, or canonical labels/rows/times hard-coded;
- global `asfreq`/resample, uppercase obsolete `H`, fill/interpolate/zero,
  an off-grid whole-hour violation that is silently dropped instead of rejected,
  provenance inferred from measurement missingness alone, wrong entity range,
  wrong bin boundary/label, count instead of mean, mean of a label/counter,
  missing entity, wrong row/order/count, or nonconserved readings;
- pooled `shift`/`diff`/rolling, `shift(-1)`, lead computation, current row
  included in a past-only window, row-count window treated as elapsed time,
  elapsed window treated as two rows, wrong `closed` boundary, positional merge,
  lost source order, or input mutation;
- centered/custom/EWM/expanding computation, availability derived from final
  table presence instead of latest-required timestamps, incorrect keep/reject,
  target/horizon/model/split-role/evaluation work, or cutoff that overlaps or
  loses an entity;
- plots/images, advanced DST/time selection/period work, decomposition,
  forecasting, statistics/modeling, GroupBy apply/filter, cleaning, remote/
  performance/network/random/mutable-date data, credentials, uploads/Drive,
  `/content`, or absolute paths; and
- central success and captured-failure result objects, exact 90-point sum,
  completed-failure exit zero, nonzero infrastructure failure, feedback
  visibility, and corrected resubmission.

Correct work must pass canonical and alternate behavior. The starter should fail
with a small stable set of task-specific messages. Automation must not claim
that authored reasoning is meaningful merely because it is nonempty or contains
keywords.

## Human reasoning boundary and provisional scoring

Use a provisional 90 automated plus 10 human diagnostic allocation:

| Area | Automated | Human | Total |
|---|---:|---:|---:|
| Task 1: temporal contract, timezone, entity order, and explanation | 20 | 3 | 23 |
| Task 2: frequency/provenance behavior and measurement rationale | 25 | 3 | 28 |
| Task 3: past-only behavior, availability, holdout, and reasoning | 30 | 4 | 34 |
| shared template, portability, scope, and reproducibility | 15 | 0 | 15 |
| **Diagnostic total** | **90** | **10** | **100** |

Automation owns environment/fixture/protected integrity, function behavior on
canonical and alternate inputs, nonmutation, schemas/dtypes/order, timezone,
entity keys, exact values, row/source conservation, provenance, window
boundaries, availability, chronological separation, artifact bytes,
portability, and repeatability.

Human review owns only:

- whether Task 1 accurately explains timestamp/period, panel/grain/key,
  localization/conversion, sort order, and irregularity;
- whether Task 2 connects `asfreq`/resample and missingness provenance to
  measurement meaning instead of paraphrasing API names;
- whether Task 3 distinguishes the two past-window meanings, explains
  prediction-time availability and rejected future candidates, and keeps the
  chronological split as a bounded handoff rather than an evaluation claim; and
- whether the notebook is concise, readable, and contains no identifying or
  sensitive information.

The historical design describes regular Assignment 09 as competence-focused
and pass/fail, but this diagnostic does not set a threshold, convert points to
pass/fail, establish gradebook weight, or decide late/resubmission/regrade
policy. Fixture, behavior, the automated 90, and human 10 remain technically
separable from that unresolved policy overlay.

## Platform and publication boundary

- Clean local Jupyter or the VS Code notebook interface is mandatory for the
  initial release.
- Classroom50 applies to the entire course. No GitHub Classroom export, Actions
  workflow, mutable test fetch, or student-editable production grader remains.
- Student instructions use VS Code Source Control or GitHub Desktop to inspect,
  commit, and push `assignment.ipynb` and all six visible CSVs.
- No Assignment 09 Colab badge or claimed Colab submission path is allowed until
  repository save-back, authoritative submission, feedback, and resubmission
  pass the notebook-assignment pilot. If approved later, preserve this notebook
  rather than fork a Colab edition.
- Classroom50 grader assets are discoverable. Behavioral variation, central
  protected hashes, and bounded human reasoning review provide integrity;
  secrecy does not.
- No duration estimate, due-date logic, or timing claim belongs in the notebook,
  README, platform guide, rubric, checker, or grader contract.

## Full legacy disposition

Implementation must:

- rewrite `09/assignment/README.md` around the exact three tasks, definitions,
  fixture, outputs, local setup, public checker, GUI Git visibility,
  Classroom50 submission, and conditional Colab status;
- replace the three legacy assignment notebooks with the exact one 26-cell
  `assignment.ipynb` and delete all three paired Markdown sources;
- delete both data-generator sources and all generated patient/ICU/disease data;
- delete the complete legacy `.github/workflows/` and `.github/test/` trees;
- replace broad lower-bound plotting/Jupyter/Jupytext/statsmodels dependencies
  with the exact two-package runtime record;
- remove all legacy q1/q2/q3 report/image contracts, artificial monthly-to-daily
  imputation, pooled-patient calculations, advanced selection, business-date
  schedules, advanced windows, EWM, visualization, and timezone essay breadth;
  and
- add only the student/instructor surfaces, fixture, five functions, six CSVs,
  checks, and grading contracts specified here.

No legacy file remains for compatibility; repository history already preserves
it. Classroom50 course configuration belongs in the course-wide platform area,
not `.github/` and not as a second assignment implementation.

## Unresolved policy choices

These choices do not block technical implementation and must not be guessed in
student code:

1. how the provisional 90 automated plus 10 human diagnostic maps to the
   historical competence/pass-fail policy, including any threshold or gradebook
   conversion;
2. production Classroom50 classroom, assignment, release, review, and
   authoritative-submission metadata sources;
3. how the human 10 is combined with Classroom50's automated result and
   exported to the official grade system;
4. late-submission, resubmission, regrade, and record-retention policy;
5. whether and when Assignment 09 receives an immutable-release Colab launch
   after the repository-save/Classroom50 pilot passes.

## Design evidence and independent acceptance gate

The 380-byte fixture, 473-byte manifest, six canonical serializations, all
hashes, exact dtypes, temporal invariants, and the disclosed alternate behavior
were computed under CPython 3.12.13 with exactly NumPy 2.0.2 and pandas 3.0.3.
The design preserves the verified Lecture 08→09→10 boundary and introduces no
Lecture 10 modeling/evaluation capability.

A reviewer who did not author this blueprint must independently reconstruct the
fixture and manifest bytes; recompute canonical and alternate values under the
pinned candidate; audit content, completeness, extraneous scope, organization,
term-before-use order, notebook feasibility, exact artifacts, checker/grader
contracts, Classroom50 topology, output visibility, path behavior, and the
human/automation boundary; and record an explicit PASS before any file under
`09/assignment/**` is edited.

After implementation, another reviewer who did not implement the package must
inspect every source, fresh-execute canonical and alternate matrices, test all
fixture/output/adversarial cases, verify official result success/failure paths,
run the course audit and scoped diff gate, and recheck the Lecture 08→09→10
scope boundary. Fresh Colab execution remains a separate pilot/publication gate,
not an inferred assignment capability.

Production-contract correction (2026-07-19): Classroom50 invokes the teacher
bundle's standard-library `autograder.py` with plain Python; it installs exact
sibling requirements before importing the central grader. The accepted student
repository may additionally contain only delivery-owned `.classroom50.yaml` and
`.github/workflows/autograde.yaml`; only the top-level `.git/**` repository
metadata tree is ignored, while every other root/workflow/grader-tree file,
including a nested `ordinary/.git/**` tree, is rejected.
