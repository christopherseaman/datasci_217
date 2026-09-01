# Lecture 09 demo implementation blueprint

> Historical review snapshot. Any Classroom50 or GitHub Classroom language
> below records a superseded intermediate delivery plan, not current course
> policy. The release design uses no Classroom service.

Status: independently reviewed implementation-ready design handoff; **PASS
after one narrow output-dtype clarification**. The Lecture 09 core/bonus
narrative and the Lecture 08 to 09 dependency boundary have passed independent
verification. This document authorizes only demo implementation, not Assignment
09 redesign.

## Accepted demo role

The required demonstrations must teach one cumulative progression:

1. classify timestamp/period, regular/irregular, and single-series/panel
   structure; state row grain and keys; then parse, localize, convert, sort, and
   index one entity-aware panel;
2. contrast an entity-scoped hourly `asfreq()` grid with a grouped two-hour
   `resample()` summary chosen from measurement meaning, preserving both
   entities and distinguishing source missingness from grid-created rows; and
3. create entity-scoped lags, differences, and past-only observation-count and
   elapsed-time windows; inventory information availability at a supplied
   prediction timestamp, reject future leakage, and form a plausible
   chronological holdout for the Lecture 10 handoff.

All three notebooks use the same small station-observation fixture. They do not
pool entities, fill or interpolate measurements, compute centered/custom/EWM/
expanding windows, handle advanced daylight-saving-time cases, decompose or
forecast a series, fit a model, introduce statistical inference, fetch runtime
data, or teach visualization. The narrative permits one already-familiar
Lecture 07 line chart in Demo 3, but this design deliberately uses none: the
temporal table contracts are the learning objective.

## Exact package

Replace the current required demo package atomically with:

```text
09/demo/
|-- .gitignore
|-- .python-version
|-- DEMO_GUIDE.md
|-- requirements.txt
|-- data/
|   `-- station_observations.csv
|-- demo1_temporal_structure.ipynb
|-- demo2_frequency_measurement.ipynb
`-- demo3_past_only_features.ipynb
```

Delete the three legacy notebooks, their paired same-stem Markdown sources, the
notebook-form guide, and the legacy Markdown guide. Do not retain the random
patient/ICU/disease generators, mutable current-time examples, broad plotting
gallery, seasonality claims, statistics/modeling dependencies, or Jupytext
authoring path. Generated evidence belongs under ignored `09/demo/output/` and
is never a teaching input.

`.gitignore` must ignore `output/`, notebook checkpoints, Python caches, and
common local virtual-environment directories. It must not hide the notebooks,
guide, requirements, or committed fixture.

Use exact candidate records:

```text
.python-version: 3.12.13
requirements.txt:
numpy==2.0.2
pandas==3.0.3
```

Jupyter hosting, kernels, and notebook-execution tools used only during
certification are platform/test dependencies rather than lecture runtime
imports. Matplotlib, seaborn, Altair, statsmodels, and Jupytext are not required.
The course-wide version set remains a compatibility candidate until the local
and fresh-Colab release gates pass.

## Pinned station fixture

`data/station_observations.csv` is course-authored, synthetic,
non-identifying, and deterministic. Its grain is one recorded temperature
observation for one station at one documented local clock timestamp. The
source order is deliberately interleaved and out of time order so sorting has
an observable purpose. One North source row has a missing measurement.

The file is exactly 310 bytes and has SHA-256
`57dcdb82372805cf1dda83a7c227b463fe997cf1437275d64d01b9719ff26b54`.
Use UTF-8, comma delimiters, `\n` line endings, the exact column order below,
and a final newline:

```csv
station,observed_at,temperature_c
south,2026-01-15 13:00,23.0
north,2026-01-15 08:00,10.0
south,2026-01-15 08:00,20.0
north,2026-01-15 14:00,14.0
south,2026-01-15 10:00,21.0
north,2026-01-15 11:00,
south,2026-01-15 14:00,24.0
north,2026-01-15 09:00,11.0
south,2026-01-15 11:00,22.0
north,2026-01-15 12:00,13.0
```

The source contract documents every clock value as unambiguous
`America/Los_Angeles` local time in January 2026. Parse only the exact
`%Y-%m-%d %H:%M` format, localize once, then convert once to UTC. The parsed
semantic dtypes are:

- `station`: pandas string dtype;
- `observed_at`: timezone-aware datetime dtype in UTC;
- `temperature_c`: NumPy `float64`; and
- supplied `source_row`: NumPy `int64`, with value `1` for all ten source rows.

Each notebook must resolve one demo root before reading or writing. Search the
current directory and its ancestors for a directory containing the supplied
`DEMO_GUIDE.md` and `.python-version`, recognizing both the course layout
`09/demo/` and a flattened standalone directory. If no such root exists, use
the current directory as the ephemeral standalone/Colab root. Read
`data/station_observations.csv` below that root, reconstruct the exact supplied
bytes only when that path is absent, and verify its checksum before parsing. A
present but corrupt fixture must stop execution rather than be replaced or
silently bypassed. Write only below `<resolved-demo-root>/output/`.

Repository root, `09/demo/`, a directory nested below `09/demo/`, and a
standalone launch directory must behave equivalently. No manual upload, Drive
mount, credential, random value, mutable date, or runtime data fetch is allowed.

## Notebook-wide contract

Every notebook must have:

- a portable `Python 3` kernelspec, stable globally unique cell IDs, null
  execution counts, and zero stored outputs;
- a first Markdown cell stating the learning question, input and output grain,
  Colab-first/local-Jupyter equivalence, ephemeral filesystem, privacy rule,
  fresh-execution rule, that Colab edits are not automatically saved to the
  repository, and that assignment use of Colab remains conditional on the
  repository-save/Classroom 50 pilot;
- one supplied setup cell that conditionally installs only mismatched course
  packages before their first import, then prints and asserts Python 3.12.13,
  NumPy 2.0.2, and pandas 3.0.3;
- deterministic fixture/bootstrap logic with exact byte and checksum
  verification, followed by the shared parse/localize/convert/sort preparation
  where needed;
- explicit output-directory creation and deterministic replacement of only the
  stale files owned by that notebook;
- explicit pandas 3 policies: lowercase `h` aliases, stable entity/timestamp
  sorting, and `observed=True`, `sort=True`, and `dropna=True` on required
  GroupBy calls;
- UTF-8 CSV writes with `lineterminator="\n"` and `index=False` after moving
  meaningful index labels into ordinary columns;
- concise instructional Markdown defining every newly demanding term before
  its first code use;
- executable assertions for shapes, dtypes, timezone, entity retention,
  within-entity order, grain, columns, row order, exact values, paths,
  schema-aware readback, and deterministic repeat behavior; and
- a final verification cell that passes only after all prior source executes
  freshly.

Tables may be displayed during instruction, but stored display output is not
execution evidence. Every notebook is self-contained: it must derive its own
state from the committed or reconstructed fixture and must not read another
notebook's generated output.

Use these exact in-memory dtypes before writing and restore the same semantic
dtypes during schema-aware readback:

| output | exact dtypes in column order |
|---|---|
| `prepared_panel.csv` | `string`; `datetime64[us, UTC]`; `float64`; `int64` |
| `hourly_grid.csv` | `string`; `datetime64[us, UTC]`; `float64`; `float64`; `bool`; `bool` |
| `two_hour_summary.csv` | `string`; `datetime64[us, UTC]`; `float64`; `int64` |
| `temporal_features.csv` | `string`; `datetime64[us, UTC]`; five `float64` columns |
| `availability_decisions.csv` | `string`; `datetime64[us, UTC]`; `bool`; `string` |
| `chronological_blocks.csv` | `string`; `datetime64[us, UTC]`; `float64`; `int64`; `string` |

The hourly grid's `source_row` is deliberately `float64`: entity-scoped
`asfreq()` inserts grid rows with missing source markers, so pandas promotes the
source counter from NumPy `int64` rather than inventing an integer source row.

## Demo 1: classify and prepare temporal structure

Canonical filename: `demo1_temporal_structure.ipynb`.

Define timestamp, period, entity/entity key, single series, panel, row grain,
row key, sort keys, regular, irregular, and frequency before using them freely.
Use one `pd.Timestamp("2026-01-15 08:00")` and one
`pd.Period("2026-01-15", freq="D")` to distinguish an instant from a calendar
span without opening advanced period arithmetic. Use one four-label
`pd.date_range(..., periods=4, freq="h", tz="UTC")` reference whose adjacent
gaps are all one hour to make regular spacing concrete before contrasting it
with the irregular station histories.

Before parsing, state and verify this exact source contract:

- the supplied observations are timestamp-based, not period-based;
- input row grain is one recorded temperature observation for one station and
  local timestamp;
- `station` is the entity key;
- `station` plus `observed_at` is the row key;
- the table is a two-entity panel; and
- the required sort keys are `station`, then `observed_at`.

Define parsing, naive timestamp, timezone-aware timestamp, localization,
conversion, and `DatetimeIndex` before their first corresponding operation.
Parse the documented local text exactly, prove it is initially naive, localize
to `America/Los_Angeles`, convert to UTC, then stable-sort and reset the row
index. Set a UTC `DatetimeIndex` in a named view only after preserving the
station key as a column.

The prepared table must have the exact station/time/value sequence:

```text
north: 16:00 10.0; 17:00 11.0; 19:00 missing; 20:00 13.0; 22:00 14.0
south: 16:00 20.0; 18:00 21.0; 19:00 22.0; 21:00 23.0; 22:00 24.0
```

All times above are on 2026-01-15 UTC. Verify ten rows, two entities,
entity-time uniqueness, monotonic time within each station, repeated timestamps
across stations as valid panel rows, and within-station gaps of exactly one and
two hours. Both station histories are therefore irregular. A one-station
subset is a single series; the full table remains a panel.

Write only `output/prepared_panel.csv`, with exact columns
`station,observed_at,temperature_c,source_row`. Perform schema-aware readback
that restores the UTC datetime rather than accepting it as arbitrary text. The
output grain remains one supplied station observation, and all ten source rows
must be conserved.

## Demo 2: change frequency with measurement meaning

Canonical filename: `demo2_frequency_measurement.ipynb`.

Reconstruct the shared prepared panel independently. Define upsampling,
downsampling, `asfreq`, `resample`, source missingness, grid-created
missingness, measurement meaning, and left-closed/left-labeled bin before the
first operation that needs each term. State explicitly that grid creation does
not justify fill, backward fill, interpolation, or zero replacement.

First, group by station and conform each station independently to an hourly
grid with `resample("h").asfreq()` and no fill method. Preserve `station` and
`observed_at` as ordinary output columns after the temporary entity/time index.
Add exact Boolean provenance columns:

```python
hourly_grid["grid_created_row"] = hourly_grid["source_row"].isna()
hourly_grid["source_value_missing"] = (
    hourly_grid["source_row"].eq(1)
    & hourly_grid["temperature_c"].isna()
)
```

The hourly output has exact columns
`station,observed_at,temperature_c,source_row,grid_created_row,source_value_missing`
and 14 rows: seven station-hours per entity. The four grid-created labels are
North 18:00 and 21:00 UTC and South 17:00 and 20:00 UTC. The sole
source-value-missing row is North 19:00 UTC. These categories must be mutually
exclusive, both stations must remain present, and no input row may be dropped.

Second, state that temperature is a state measured at an instant, so a bin mean
answers the bounded question "what was the mean of recorded temperatures in
this interval?" State that `source_row` is an additive reading counter. Build a
grouped `resample("2h", closed="left", label="left")` named aggregation with
result grain one station--two-hour interval and exact columns
`station,observed_at,mean_temperature_c,reading_count`.

The summary has eight ordered rows and these exact values:

```text
station  UTC bin  mean_temperature_c  reading_count
north    16:00    10.5                2
north    18:00    missing             1
north    20:00    13.0                1
north    22:00    14.0                1
south    16:00    20.0                1
south    18:00    21.5                2
south    20:00    23.0                1
south    22:00    24.0                1
```

Explain that North's missing 18:00-bin mean comes from one source row with a
missing temperature, not from an empty station interval. Assert that total
`reading_count` is ten, entity order is North then South, bins are ascending
within entity, and no text/entity field is averaged.

Write exactly `output/hourly_grid.csv` and
`output/two_hour_summary.csv`. Require schema-aware exact readback, including
the UTC timestamps and Boolean provenance columns, and stable bytes on a repeat
run. Do not use the resampled output as an input to Demo 3.

## Demo 3: build past-only features and audit availability

Canonical filename: `demo3_past_only_features.ipynb`.

Reconstruct the shared prepared panel independently. Define lag, lead,
difference, trailing window, observation-count window, and elapsed-time window
before their first use. State that `shift(1)` means one earlier row, not one
hour, and that every operation must remain scoped to `station`.

Create exact columns:

- `temperature_lag_1`: the previous station observation via grouped
  `shift(1)`;
- `temperature_difference`: the within-station difference via grouped
  `diff()`;
- `mean_previous_2_observations`: the mean of up to two previous station rows,
  excluding the current row, with `min_periods=1`; and
- `mean_previous_2h`: the mean of station observations in `[t - 2h, t)`, with
  `min_periods=1`.

Use the narrative's bounded implementations rather than introducing a new
window API: grouped `shift(1).rolling(window=2, min_periods=1).mean()` inside a
same-index `transform` for observation count, and an entity-grouped datetime
index followed by `rolling("2h", closed="left", min_periods=1).mean()` for
elapsed time. Return the elapsed result to the source grain on
`station,observed_at` with a `one_to_one` validated merge and preserve the
prepared order.

The temporal-feature output preserves the ten-row source grain and exact
station/time order from Demo 1. The first lag and difference for each station
must be missing. North 17:00 UTC has lag `10.0`, difference `1.0`, and both
window means `10.0`. South 21:00 UTC has lag `22.0`, difference `1.0`,
prior-two-observation mean `21.5`, and prior-two-hour mean `22.0`. South 22:00
UTC has prior-two-observation mean `22.5` and prior-two-hour mean `23.0`.
These values prove that observation count and elapsed time are not synonyms on
an irregular series. Do not compute a pooled shift, a negative shift, a lead,
or a centered/custom/EWM/expanding window even as a counterexample.

Then define candidate feature as a bounded Lecture 10 preview, prediction
timestamp, information availability, future-derived candidate, future leakage,
and chronological holdout. Use the exact supplied prediction timestamp South
at 2026-01-15 21:00 UTC and construct this explicit inventory without computing
the rejected candidates:

```text
candidate                          latest required UTC  available  decision
calendar hour                      21:00                True       keep
previous observed temperature      19:00                True       keep
centered three-observation mean    22:00                False      reject
next observed temperature          22:00                False      reject
```

Explain that appearance in a completed historical table does not imply
prediction-time availability. The centered and next-observation candidates are
rejected because both require information after 21:00 UTC; the notebook must
not implement either calculation.

Finally label rows before 21:00 UTC `earlier` and rows at or after 21:00 UTC
`later_holdout`. The chronological-block output has ten rows: seven earlier and
three later. Both blocks contain North and South, the maximum earlier timestamp
is 20:00 UTC, and the minimum holdout timestamp is 21:00 UTC. State that this is
only a plausible chronological handoff: Lecture 10 still owns targets,
horizons, formal evaluation roles, baselines, metrics, and model selection.

Write exactly:

- `output/temporal_features.csv`, with columns
  `station,observed_at,temperature_c,temperature_lag_1,temperature_difference,mean_previous_2_observations,mean_previous_2h`;
- `output/availability_decisions.csv`, with columns
  `candidate,latest_required_timestamp,available_by_prediction_time,decision`;
  and
- `output/chronological_blocks.csv`, with the three prepared-panel fields plus
  `source_row,block`.

Require schema-aware exact readback, exact row/column order and values, stable
repeat bytes, and no chart output.

## Guide and publication contract

`DEMO_GUIDE.md` must identify the exact three notebooks, cumulative objectives,
fixture and checksum, generated outputs, expected visible tables, launch paths,
actionable prediction checkpoints with their expected outcomes and concise
explanations, likely failure modes, destructive/repeat-run rehearsal, privacy,
and scope boundaries. Its instructions must address the learner directly and
must not contain instructor talking points, meta-instructions, or lesson-duration
language.

The guide must define terms before asking the learner to use them and preserve
these contrasts explicitly:

- timestamp versus period, single series versus panel, and regular versus
  irregular;
- localization versus conversion;
- `asfreq` versus aggregating `resample`;
- source-value missingness versus grid-created rows;
- observation-count versus elapsed-time windows; and
- historical availability versus prediction-time availability.

Link each notebook through a development Colab badge and state that every badge
must move to one immutable release tag and be fresh-run before publication. The
development URL must use the official course-repository Colab form and must not
imply that Colab edits save back to GitHub. The guide's certification table
starts with local candidate, fresh Colab, and immutable badge-reference rows all
pending, with fields for execution date, exact environment, warnings, and
result. It must not invent a timing claim. Authorship, stored notebook output,
or a development-branch badge is not independent certification.

## Independent QA matrix

After implementation, a reviewer who did not author the notebooks must verify:

- actual fresh-kernel execution of all three notebooks from repository root,
  `09/demo/`, a directory nested below `09/demo/`, and a disposable standalone
  package;
- a separate progressive execution in every layout with notebook-code warnings
  promoted to errors;
- absent fixture reconstruction to the exact 310 bytes and checksum, and
  present-but-corrupt fixture failure before parsing, replacement, or owned-
  output cleanup;
- stale, deleted, and binary-corrupt owned outputs; deterministic replacement;
  stable repeat hashes; and preservation of an unrelated output sentinel;
- exact temporal classification, dtypes/timezone, entity-time uniqueness,
  within-entity ordering/gaps, hourly provenance, resample bin semantics,
  feature values, availability decisions, chronological separation, paths, and
  schema-aware CSV readback;
- exact package tree, dependencies, fixture bytes, guide claims/checksum,
  portable metadata/state, globally unique cell IDs, no paired Markdown or
  notebook guide, and no committed generated artifacts; and
- absence from required code of pooled entity operations, fill/interpolation,
  negative shift/lead, centered/custom/EWM/expanding windows, advanced DST,
  decomposition/seasonality/forecasting, inference/modeling/statistics,
  remote/network data, upload/Drive, credentials, randomness, mutable dates,
  and visualization instruction or outputs.

Fresh Colab execution and immutable release-tag badges remain separate
publication gates even after independent local QA passes. The canonical
repository owner/name and immutable release reference must be confirmed at
publication; no production badge or final version lock is inferred by this
design.

## Design evidence and unresolved choices

The fixture checksum and all stated pandas values were independently computed
under the candidate Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3 environment.
The deliberate no-chart decision removes unnecessary plotting dependencies and
keeps all three demos aligned to the verified LIVE DEMO contracts.

No pedagogical or fixture choice remains unresolved. Publication still requires
the independent local execution matrix, a fresh Colab run, confirmation of the
canonical repository and immutable release tag for badges, and the course-wide
version freeze. Those gates do not authorize Assignment 09 implementation or an
unconditional assignment-Colab path.
