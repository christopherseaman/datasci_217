# Lecture 09 demonstration guide

These three demonstrations build one temporal-data workflow: classify and
prepare entity-specific histories, change frequency without changing
measurement meaning, then create past-only comparisons and audit when their
inputs were available. All examples use one fixed two-station table.

Required Lecture 09 demos are Colab-first and run equivalently in local Jupyter
or the VS Code notebook interface. Colab storage is ephemeral, and changes made
in a notebook opened from GitHub are not automatically saved back to the
repository. Assignment Colab support remains conditional on the repository-save
and Classroom50 pilot; these demo badges do not establish an assignment path.

## Launch the demonstrations

The badges below are development links to `main`, not publication references.
Change all three to the same immutable course release tag and fresh-run each
notebook in Colab before publication.

1. [![Open Demo 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo1_temporal_structure.ipynb) — classify and prepare one two-station temporal panel.
2. [![Open Demo 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo2_frequency_measurement.ipynb) — contrast an hourly grid with a measurement-aware two-hour summary.
3. [![Open Demo 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo3_past_only_features.ipynb) — create entity-scoped past-only values, audit availability, and form a chronological handoff.

For local use, open the repository in VS Code/Jupyter with Python 3.12.13 and
the exact direct dependencies in `requirements.txt`. The supplied setup cell
checks the environment and installs only a mismatched NumPy or pandas candidate
before importing either package. Start Jupyter from repository root, `09/demo/`,
or a directory nested below the demo; every notebook resolves the same demo
root. Restart the kernel and run all cells in order. Stored output is not
execution evidence.

## Fixed teaching input

`data/station_observations.csv` is course-authored, synthetic,
non-identifying, and has grain one recorded temperature observation for one
station and documented local clock timestamp. Its exact size is 310 bytes and
its SHA-256 is
`57dcdb82372805cf1dda83a7c227b463fe997cf1437275d64d01b9719ff26b54`.
The interleaved source order, one- and two-hour gaps, and North's one missing
source temperature are intentional teaching properties.

Every notebook verifies the committed bytes before parsing. If the fixture is
absent in an ephemeral standalone/Colab layout, the notebook reconstructs the
exact supplied bytes. A present but corrupt fixture stops execution and is
never silently replaced. No manual upload, Drive mount, credential, runtime
data fetch, random input, or mutable date is required.

## Demo sequence and expected visible results

### Demo 1: classify and prepare temporal structure

A **timestamp** represents an instant; a **period** represents a calendar span.
An **entity** is the unit with one ordered history, and its **entity key** names
that unit. A **single series** contains one entity; a **panel** contains multiple
entity histories. **Row grain** states what one row represents. A sequence is
**regular** when its adjacent gaps follow one expected spacing and **irregular**
when those gaps vary.

The source is a timestamp-based, two-entity panel with row key
`station,observed_at`. Parse the documented Los Angeles clock readings,
**localize** them by attaching that source zone, **convert** the aware instants
to UTC, then stable-sort by station and time. Localization preserves displayed
source clock values; conversion changes their displayed clock zone while
preserving the instants.

Expected visible properties are ten conserved rows, North then South, and UTC
time ranges 16:00 through 22:00. North's gap sequence is `[1, 2, 1, 2]` hours;
South's is `[2, 1, 2, 1]`. Both histories are irregular. A North-only subset is
one single series; the complete table remains a panel.

Generated output: `output/prepared_panel.csv` — 431 bytes, SHA-256
`a9e2b75c2f4e9f9a3778b53cd87e68d4a559511c368cad80b7153b1109a987ba`.

Learner prediction checkpoints:

| Predict before running | Expected outcome | Why |
|---|---|---|
| Source classification | timestamp-based, irregular, two-entity panel | Rows describe recorded instants, gaps vary, and two stations have separate histories. |
| Row and entity keys | one station observation; entity key `station`; row key `station,observed_at` | Repeated timestamps across stations are valid panel rows, not duplicates. |
| Los Angeles 08:00 in January after conversion | 16:00 UTC | Localization attaches the documented source zone; conversion represents the same instant in UTC. |

### Demo 2: change frequency with measurement meaning

**Upsampling** requests a finer grid; **downsampling** creates coarser bins.
`asfreq()` conforms exact labels to a grid without combining observations.
`resample()` groups timestamps into bins and needs a summary when several
observations can contribute. **Measurement meaning** determines which summary
answers the question.

An hourly entity-scoped `asfreq()` result distinguishes two kinds of missing
displayed values. **Source-value missingness** means a supplied row exists but
its measurement is missing. A **grid-created row** is a requested timestamp
with no supplied row. Grid creation alone does not justify fill, interpolation,
or replacement with zero.

Expected hourly properties are 14 rows, four grid-created rows (North 18:00 and
21:00 UTC; South 17:00 and 20:00 UTC), and one separate source-value-missing
row (North 19:00 UTC). Both stations remain present.

Temperature is a state observed at an instant, so the two-hour mean answers
"what was the mean recorded temperature in this interval?" The source-row
counter is additive. Left-closed, left-labeled bins produce eight station-bin
rows. North 16:00 has mean `10.5` and count `2`; North 18:00 has a missing mean
and count `1`; South 18:00 has mean `21.5` and count `2`. All reading counts sum
to ten.

Generated outputs:

- `output/hourly_grid.csv` — 788 bytes, SHA-256 `7054dfb410b36f35ef53ff4e02cc77fb633ff413e1dcd9f66d1807674053b40e`;
- `output/two_hour_summary.csv` — 361 bytes, SHA-256 `0558659b66336e71c3c67769097aadf4e2616a2d4f913425bb498463528a9d6f`.

Learner prediction checkpoints:

| Predict before running | Expected outcome | Why |
|---|---|---|
| Hourly rows per station | seven | Each entity receives labels from its own 16:00 through 22:00 range. |
| North 19:00 provenance | source row with a missing temperature | Its source marker is present, unlike a grid-created row. |
| North 18:00 two-hour mean/count | missing mean, count `1` | One row was recorded in the bin, but that row's temperature is missing. |

### Demo 3: past-only comparisons and information availability

A **lag** attaches an earlier same-entity observation; a **lead** attaches a
later value. A **difference** compares the current value with the previous
same-entity value. A **trailing window** uses values at or before the current
position. An **observation-count window** selects rows; an **elapsed-time
window** selects observations inside a clock interval. The required windows
exclude the current row.

Expected visible values for South at 21:00 UTC are lag `22.0`, difference
`1.0`, mean of the previous two observations `21.5`, and mean in the previous
two elapsed hours `22.0`. The values differ because the series is irregular.
The first lag and difference for each station are missing, proving that neither
history borrowed a value from the other station.

A **prediction timestamp** is the instant when a later procedure would issue a
prediction. **Information availability** asks whether every required source
value was known by then. A **future-derived candidate** requires a later value;
using it creates **future leakage**. At South 21:00 UTC, calendar hour and the
previous observed temperature are kept. A centered three-observation mean and
the next observed temperature both require 22:00 data and are rejected without
being computed.

A **chronological holdout** sets aside a later time block. The supplied 21:00
UTC cutoff yields seven `earlier` rows and three `later_holdout` rows; both
blocks retain both stations and every earlier timestamp precedes every holdout
timestamp. This is a temporal handoff, not a completed modeling workflow.

Generated outputs:

- `output/temporal_features.csv` — 633 bytes, SHA-256 `5a6524e8dbb37da3cc056cc648e5ff444c12cb485214bef1a95c3a67a22af3ab`;
- `output/availability_decisions.csv` — 326 bytes, SHA-256 `d4125def8dcf8e23b9f33574f1dd9e14a5ed3f92889f88b455509110ad87e505`;
- `output/chronological_blocks.csv` — 535 bytes, SHA-256 `7ea9752756ef882dbe19318bfbb1614c33ff6bbca45a5bbc5effe4bcad065a67`.

Learner prediction checkpoints:

| Predict before running | Expected outcome | Why |
|---|---|---|
| First lag for each station | missing | Each entity begins a new history; no value crosses the station boundary. |
| South 21:00 two window means | `21.5` by prior rows; `22.0` by prior two hours | The 18:00 row is among the prior two observations but outside `[19:00, 21:00)`. |
| Availability decisions | keep, keep, reject, reject | The rejected candidates require a 22:00 value that did not exist at 21:00. |
| Chronological block sizes | seven earlier, three later | The fixed global cutoff preserves strict time order and both entities in both blocks. |

## Rehearsal and likely failure modes

Run the rehearsal from repository root, `09/demo/`, a nested directory, and a
standalone copy. For each notebook:

1. delete its owned generated output and confirm a fresh run recreates it;
2. replace that output with stale or corrupt bytes and confirm a rerun replaces it;
3. run again and confirm same-platform output hashes are unchanged;
4. temporarily remove the fixture and confirm exact-byte reconstruction;
5. temporarily corrupt a present fixture and confirm checksum failure occurs
   before parsing, replacement, or owned-output cleanup; and
6. restore the committed fixture and restart/run all.

Common failures are sorting the panel only by timestamp, using one station's
last row as another station's lag, treating `shift(1)` as one elapsed hour,
confusing a grid-created row with a missing source value, averaging station
labels, including the current row in a past-only window, or accepting a
candidate whose latest required timestamp is later than the prediction time.

Each notebook owns only the outputs listed in its section. Setup replaces stale
copies of those files without deleting another notebook's artifacts or an
unrelated sentinel. Runtime output is ignored and never used as a teaching
input.

## Privacy and scope boundaries

The fixture is synthetic and non-identifying. Do not substitute credentials,
private records, or sensitive stored output in a shared notebook. Required
demos do not pool stations, fill or interpolate measurements, compute leads or
centered/custom/EWM/expanding windows, handle advanced daylight-saving-time
cases, decompose or forecast a series, fit or evaluate a model, introduce
statistical inference, fetch runtime data, or add visualization instruction.
Optional extensions remain in `../BONUS.md`.

## Certification record

Authorship and stored notebook output are not independent certification.

| Gate | Execution date | Exact environment | Warnings | Result/reference |
|---|---|---|---|---|
| Independent local candidate | 2026-07-18 | Python 3.12.13; NumPy 2.0.2; pandas 3.0.3; nbclient 0.10.2; nbformat 5.10.4; ipykernel 6.29.5 | none from notebook code | PASS; see independent evidence in `../../work/2026_refresh_audit.md` |
| Fresh Colab runtime | pending | pending | pending | pending |
| Immutable release-tag badge | pending | pending | pending | pending |

Author-side local evidence is recorded in `../../work/2026_refresh_audit.md`.
Fresh Colab execution, immutable release-tag badges, and the course-wide version
freeze remain separate publication gates.
