# Assignment 09: Entity-Aware Temporal Evidence

Use the supplied synthetic indoor-air readings to build one reproducible temporal workflow. The three tasks accumulate:

1. state the temporal contract, parse the documented local clock text, localize it once, convert it once to UTC, and sort a two-zone panel;
2. create an entity-scoped hourly grid and a measurement-aware two-hour summary while keeping source-value missingness distinct from grid-created rows; and
3. create entity-scoped lag, difference, and two past-only window meanings, audit candidate availability at a supplied prediction timestamp, and form a chronological handoff.

The assignment requires clean local Jupyter or the VS Code notebook interface. Assignment Colab is not part of this repository contract; stored notebook output is not execution evidence.

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

Task 2 defines upsampling, downsampling, `asfreq`, `resample`, missingness provenance, measurement meaning, and bin boundaries before use. Task 3 defines lag, lead, difference, both window meanings, prediction-time availability, future leakage, and chronological holdout before use.

Do not fill or interpolate measurements, pool zones, compute a lead or centered feature, add charts, fit a model, fetch data, or use mutable/random data.

## Run the assignment

1. Follow [PLATFORM_CHECK.md](PLATFORM_CHECK.md) and open `assignment.ipynb` in the assignment environment.
2. Complete every TODO in order.
3. Restart the kernel and run all cells.
4. Run `python check_assignment.py` from `09/assignment/`.
5. In VS Code Source Control or GitHub Desktop, confirm that the notebook and all six CSVs below are visible, then commit and push them.
6. Commit the completed subtree or its exported assignment repository. The optional Actions workflow is feedback; use instructor or TA review when revising.

Required GUI-visible artifacts:

- `output/prepared_panel.csv`
- `output/hourly_grid.csv`
- `output/two_hour_summary.csv`
- `output/temporal_features.csv`
- `output/availability_decisions.csv`
- `output/chronological_blocks.csv`

The public checker gives structural and artifact feedback only. It does not award a score or judge written reasoning. The central grader clears stored state and fresh-executes a disposable copy. Written explanations receive separate human review.
