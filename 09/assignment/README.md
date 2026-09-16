# Assignment 09: Entity-Aware Temporal Evidence

## Files

```text
assignment/
├── assignment.ipynb     # Provided notebook scaffold to complete
├── data/                # Provided fixtures
├── requirements.txt     # Provided pinned environment
├── check_assignment.py  # Provided completion checker
└── output/              # Generated artifacts to submit
```

## Setup

In VS Code, open the assignment folder and choose **Terminal → New Terminal**. You can also use your native terminal or WSL Ubuntu; change to the assignment directory before running these commands.

From this assignment directory, create a Python 3.13 environment and install the pinned requirements:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. If Python 3.13 is missing, run `uv python install 3.13`. Select this environment as the kernel when opening the notebook in VS Code or Jupyter. The course uses pandas 3.0.5.

Keep the supplied `data/` files unchanged. Open the complete assignment directory; the setup cell locates and verifies its fixtures in either a standalone assignment repository or the course repository. Restore missing or checksum-mismatched fixtures before continuing.

## Data and terms

The fixture is course-authored, synthetic, and non-identifying. One source row represents one recorded CO2 reading for one zone at one documented `America/New_York` local timestamp. `zone` is the entity key; `zone,recorded_at` is the row key.

- Temporal ordering determines what counts as previous, inside a window, or available.
- A timestamp is an instant; a period is a span with a start and end.
- An entity is the unit with one ordered history. A single series has one entity; a panel has several.
- Row grain states what one row represents. Row keys identify rows; sort keys define computation order.
- A regular history has one expected adjacent spacing; an irregular history has varying gaps. Frequency is an expected time grid or calendar offset, not proof of an observation.
- Parsing converts documented text into datetime values. A naive timestamp has no offset; a timezone-aware timestamp identifies an instant.
- Localization attaches the documented source zone. Conversion expresses an aware instant in another zone.
- A DatetimeIndex is an index whose labels are datetime values.

## Question 1: Prepare the temporal panel

### 1.1 Parse and order the readings

State the temporal contract, parse the documented local clock text, localize it to America/New_York, convert to UTC, and sort by zone and time. Preserve source measurements and add `source_row=1`.

> **Checkpoint — `output/prepared_panel.csv`**

## Question 2: Change frequency

### 2.1 Build the hourly grid

Create a separate hourly grid for each zone. Distinguish missing source values from rows introduced by the grid.

> **Checkpoint — `output/hourly_grid.csv`**

### 2.2 Summarize two-hour intervals

Calculate mean recorded CO2 and source-row count in left-closed, left-labeled two-hour bins within each zone.

> **Checkpoint — `output/two_hour_summary.csv`**

## Question 3: Build past-only evidence

### 3.1 Calculate temporal features

Create same-zone lag and difference, then compare means over the previous two observations and the previous two elapsed hours. Both means exclude the current observation.

> **Checkpoint — `output/temporal_features.csv`**

### 3.2 Audit availability and create chronological blocks

Audit the four supplied candidate features at 2026-01-20 18:00 UTC, then label observations before that instant `earlier` and the remaining observations `later_holdout`.

> **Checkpoint — `output/availability_decisions.csv`**

> **Checkpoint — `output/chronological_blocks.csv`**

## Check Your Work

Run this from the assignment directory after saving your artifacts:

```bash
python check_assignment.py
```

Fix each failed check, regenerate the affected files, and run the checker again. It reads saved artifacts without running your code.

### Completion contract

Save these six CSVs in `output/` with the exact columns below in order, no extra index, and UTC timestamps. Preserve required missing measurements. The panel, grid, summary, feature, and block files must have unique zone/time keys sorted by `zone`, then `recorded_at`; values and missingness must match the source-derived results.

| Artifact | Columns in order and completion criteria |
|---|---|
| `output/prepared_panel.csv` | `zone`, `recorded_at`, `co2_ppm`, `source_row`: all supplied readings, localized from America/New_York and converted to UTC; `source_row=1`. |
| `output/hourly_grid.csv` | Those four columns, then `grid_created_row`, `source_value_missing`: each zone's hourly grid from its first to last source timestamp, with separate Boolean flags for new grid rows and missing source measurements. |
| `output/two_hour_summary.csv` | `zone`, `recorded_at`, `mean_co2_ppm`, `reading_count`: left-closed, left-labeled two-hour bins with recorded-value means and source-row counts. |
| `output/temporal_features.csv` | `zone`, `recorded_at`, `co2_ppm`, `co2_lag_1`, `co2_difference`, `mean_previous_2_observations`, `mean_previous_2h`: source rows with same-zone previous value, current-minus-previous difference, and two past-only means (`min_periods=1`). |
| `output/availability_decisions.csv` | `candidate`, `latest_required_timestamp`, `available_by_prediction_time`, `decision`: one row each for `calendar hour`, `previous recorded CO2`, `centered three-observation mean`, `next recorded CO2`. At prediction time 2026-01-20 18:00 UTC, their latest timestamps are 18:00, 17:00, 19:00, 19:00 UTC; availability is True, True, False, False and decisions are keep, keep, reject, reject. |
| `output/chronological_blocks.csv` | `zone`, `recorded_at`, `co2_ppm`, `source_row`, `block`: all prepared rows, labeled `earlier` before 2026-01-20 18:00 UTC and `later_holdout` at or after it. |

Task 1 is worth 30 points, Task 2 is worth 35, and Task 3 is worth 35: 100 total.

## Submit

In VS Code Source Control, inspect your completed notebook and required `output/` files, then commit and push them. Alternatively, use **Add file → Upload files** on the GitHub website and commit the files at their required paths. Keep private data, credentials, virtual environments, and notebook checkpoints out of your submission. GitHub Actions runs the assignment checks automatically on every push. If your fork has Actions disabled, enable it once in the Actions tab. Review the feedback, then regenerate, check, commit, and push corrected artifacts if needed.
