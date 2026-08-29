# Time Series Analysis: Temporal Data and Trends

Instructor cues for the core sequence. Use the authoritative explanations and
examples in [README.md](README.md). Optional periods, decomposition, forecasting,
and high-frequency material are in [BONUS.md](BONUS.md).

## Orientation and data types

- Establish the temporal-ordering question: regular versus irregular observations, trends, seasonality, and noise.
- Contrast Python `datetime` for scalar work with pandas parsing and `DatetimeIndex` for labeled time-series operations.
- Prompt: what is gained by parsing dates and sorting the index?
- Reference: README sections “Understanding Time Series Data” and “Date and Time Data Types.”

## Date ranges, frequencies, and lags

- Demonstrate `to_datetime`, `date_range`, and business/calendar frequencies.
- Keep the pandas 3 distinction visible: timestamp offsets include `QE`, `YE`, and lowercase `h`; Period frequencies retain span aliases such as `Q-DEC` and `Y-DEC`.
- Distinguish `infer_freq`, `asfreq` (conform to a grid), and `resample` (bin, then aggregate).
- Connect `shift`, `diff`, and `pct_change` to lagged questions; mention UTC localization/conversion and DST ambiguity.
- Reference: README sections “Date Range Generation” through “Time Zone Handling.”

## LIVE DEMO!

## Indexing and selection

- Ask students to predict exact-date, partial-string, `.loc`, and `.iloc` selections.
- Extend to `between_time`, `at_time`, `truncate`, and sorted-index range selection.
- Reference: README section “Time Series Indexing and Selection.”

## Resampling and frequency conversion

- Frame resampling as time-based grouping: choose a bin and an aggregation that matches the measurement.
- Contrast downsampling with upsampling; `asfreq` selects/conforms labels and does not combine observations.
- Compare mean, sum, and a named multi-aggregation while checking missing bins and nonnumeric columns.
- Reference: README section “Resampling and Frequency Conversion.”

## LIVE DEMO!

## Rolling and exponentially weighted windows

- Compare fixed trailing windows, centered windows, expanding summaries, and EWM responsiveness.
- Discuss `window`, `min_periods`, and leakage: centered windows use future observations.
- Reference: README section “Rolling Window Operations.”

## Visualization and close

- Apply Lecture 07 principles: chronological x-axis, visible gaps, labeled units/time zone, and raw series plus a clearly identified rolling summary.
- Ask which apparent pattern is temporal structure and which could be sampling or missingness.
- Reference: README section “Time Series Visualization.”

## LIVE DEMO!
