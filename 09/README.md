---
notion:
  title_line: "# Time Series Analysis: Temporal Data and Trends"
  role: lecture
  status: mapped
  page_id: "2a8d9fdd-1a1a-80ed-828d-e5feb58d5ed9"
  url: "https://app.notion.com/p/2a8d9fdd1a1a80ed828de5feb58d5ed9"
---

# Time Series Analysis: Temporal Data and Trends

See [BONUS.md](BONUS.md) for the optional extensions.

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo1_datetime_fundamentals.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo2_indexing_resampling.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo3_visualization_automation.ipynb)

**Run locally:**

```shell
curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/09/demo/setup_demo.sh | sh
```

![xkcd 2048: Curve-Fitting. One scatter plot, twelve fitted curves, twelve different messages; not every pattern in a time series is meaningful.](media/xkcd_2048.png)

<callout icon="🧭" color="blue_bg">
	# This Lecture
	## Covers
	- **Time series data and date types**: timestamps, periods, panels, Python `datetime`, and a pandas DatetimeIndex.
	- **Date ranges, selection, and time zones**: frequency grids, calendar slicing, UTC conversion, and clock-change nights.
	- **Resampling**: daily or hourly summaries, bin edges, upsampling, per-patient grids, and gap runs.
	- **Rolling windows and past-only features**: count and time windows, EWM, grouped lags, and availability at prediction time.
	- **Time series visualization**: raw readings with a smoother, gaps, and time zones on the plot.
	- **Readings**: McKinney, _Python for Data Analysis_ (3rd ed.) 9.2, 11.1 to 11.4, 11.6, and 11.7.
	## Enables
	- Parse a text timestamp column from a monitor export into a DatetimeIndex and select one day or one week of readings.
	- Convert two clinics' local visit times to UTC and flag repeated or skipped times on clock-change nights.
	- Resample one patient's heart rate into daily means, maxima, and reading counts, then build an hourly grid per patient and count its gap runs.
	- Build previous-reading and past-only rolling-mean features per patient with `groupby()`, `shift()`, and `rolling(..., closed='left')`, keeping later rows out.
	- Smooth daily readings with a 7-day rolling mean and plot it over the raw values in matplotlib.
</callout>

# Understanding Time Series Data

- **Time series**: one measurement recorded again and again over time, such as an ICU patient's heart rate every minute or a hospital's daily flu admissions.
- Row order carries meaning: "what was the previous reading?" and "how many admissions this week?" need tools that read dates and times.

![Tufte, _The Visual Display of Quantitative Information_, p. 74: Connecticut traffic deaths for 1955 and 1956 alone suggest a crackdown worked; the 1951 to 1959 series shows 1955 was the peak year.](media/tufte_vdqi_p74_traffic_deaths.png)

## Forms of Time

- **Timestamp**: one instant, such as the blood draw at `2024-03-01 08:15`.
- **Period**: a whole span with a start and end, such as "March 2024" in a monthly report.
- **Elapsed time**: time since a starting point, such as hours since admission.

![One admission, three forms of time: the blood draw's timestamp 2024-03-01 08:15, the period March 2024 (shaded), and elapsed time, 12 hours since admission on February 29 at 20:15.](media/forms_of_time.png)

## Regular and Irregular Series

| Type | Spacing | Example |
| --- | --- | --- |
| **Regular** | A repeating rule (daily, hourly, monthly) | Daily patient temperature readings |
| **Irregular** | Variable intervals, recorded when something happens | Clinic visit dates |

![Regular: an hourly monitor on January 5, 2024, with 14 of 24 readings and no rows 08:00 to 17:00. Irregular: one patient charted at 02:00, 03:30, 04:30, 08:00, and 09:00 UTC on November 3, 1 to 3.5 hours apart.](media/regular_irregular.png)

A calendar rule does not always mean equal elapsed time: months have different lengths, and a local day can have 23 or 25 hours when the clocks change.

_Some time series are as regular as a Swiss watch; others are as unpredictable as a toddler's nap schedule._

## Single Series and Panels

- **Single series**: one history, such as one patient's weight.
- **Panel**: one history per **entity** (a patient, sensor, or site) stacked in one table with an entity column such as `patient_id`; most calculations run inside each entity's history with Lecture 08's `groupby()`.
- **Grain**: what one row represents, stated before computing anything: "one heart-rate reading for one patient at one time."

![A panel: three patients' charted heart rates on November 3, 2024, one line per patient_id; P01 rises from 88 to 118 bpm, P02 stays near 70, and P03 falls from 104 to 92.](media/icu_panel_hr_utc.png)

```text
patient_id  recorded_at        heart_rate
P1          2024-03-01 08:00           72   ┐ P1's history
P1          2024-03-01 09:00           80   ┘
P2          2024-03-01 09:00           88   ┐ P2's history
P2          2024-03-01 10:00           86   ┘
```

# Date and Time Data Types

- **Datetime**: one value holding a date and a clock time, which Python can compare, subtract, and print in any format. Text such as `"2023-12-25 14:30"` cannot: subtracting two text times raises `TypeError`.
- **Parsing** turns text into datetimes; **formatting** turns datetimes into text.
- Python's `datetime` module works one value at a time, such as stamping a lab report; pandas converts a whole column.

## Python datetime Module

| Operation | Input | Output |
| --- | --- | --- |
| Parse with `strptime()` | Text `'2024-03-01 14:30'` | A datetime for March 1 at 14:30; arithmetic is now possible |
| Format with `strftime()` | That datetime | Text such as `'March 01, 2024'`; the underlying datetime is unchanged |

### Reference Card: Python `datetime`

- `datetime(year, month, day, hour, minute)`: Create a specific date and time; hour and minute default to 0.
- `datetime.now()`: The current date and time from the computer's clock.
- `datetime.strptime(text, format)`: Parse text into a datetime ("string parse time").
- `dt.strftime(format)`: Format a datetime as text ("string format time").
- `timedelta(days=30)`: A duration (also `hours=`, `weeks=`); add it to a datetime to move it. Subtracting two datetimes returns one, and `.days` reads its whole days.
- Format codes: `%Y` four-digit year, `%m` month 01-12, `%d` day, `%H` 24-hour hour, `%M` minute, `%S` second, `%I` with `%p` 12-hour clock with AM/PM, `%B` full month name.

### Code Snippet: Python `datetime`

```python
from datetime import datetime, timedelta

lab_time = datetime.strptime("2023-12-25 14:30:00", "%Y-%m-%d %H:%M:%S")
print(lab_time.strftime("%B %d, %Y at %I:%M %p"))
print(lab_time + timedelta(days=30))  # 30-day follow-up
age = datetime(2024, 3, 1) - datetime(1990, 5, 15)
print(age.days)  # age in days on March 1, 2024
```

```text
December 25, 2023 at 02:30 PM
2024-01-24 14:30:00
12344
```

## pandas DatetimeIndex

For a whole column, use Lecture 05's `pd.to_datetime()`: its `datetime64` values sort by time rather than character by character, and invalid text becomes `NaT` with `errors='coerce'`.

| Two clinic dates | As text (dtype `str`) | After `pd.to_datetime()` (dtype `datetime64[us]`) |
| --- | --- | --- |
| Sorted | `'3/10/2024'`, then `'3/9/2024'` | `2024-03-09`, then `2024-03-10` |
| Later minus earlier | `TypeError` | `1 days 00:00:00` |

![Four weights from 81.9 to 83.4 kg: as text, '3/9/2024' sorts last and the dates sit evenly spaced; after `pd.to_datetime()`, March 9 comes first and March 12 to 20 spans 8 days.](media/text_vs_datetime_axis.png)

- **`Timestamp`**: one value in the column.
- **`DatetimeIndex`**: timestamps used as row labels, which date selection, resampling, and time windows all read.

### Reference Card: Parsing and Indexing Dates

| Task | Code | Purpose and key arguments | Typical output |
| --- | --- | --- | --- |
| Parse | `pd.to_datetime(s, format='%Y-%m-%d %H:%M')` | Convert text; `format=` states the expected pattern; `errors='coerce'` turns bad values into `NaT` | `datetime64` Series |
| Index | `df.set_index('recorded_at').sort_index()` | Timestamps as row labels, in order | `DataFrame` with `DatetimeIndex` |
| Check | `df.index.is_monotonic_increasing` | Confirm order before slicing | `True` / `False` |
| Parts | `df.index.month`, `.year`, `.hour`, `.day_name()` | Calendar parts from the index | Index of numbers or names |
| Parts | `s.dt.month`, `s.dt.year`, `s.dt.hour` | The same parts from a datetime column | Series |
| Parts | `s.dt.dayofweek` | Day of the week as a number, Monday `0` through Sunday `6`; `.dt.day_name()` spells it out | Series of `int32` |
| Parts | `s.dt.dayofyear` | Day of the year, `1` to 365 (366 in a leap year) | Series of `int32` |
| Round | `s.dt.floor('h')` | Round each time down to the hour (`'D'` for the day) | Series |
| Format | `s.dt.strftime('%Y%m%d%H')` | Write each time as text with `strftime` codes, such as `'2024030108'` for 08:00 on 1 March 2024 | Series of text |
| Duration | `pd.Timedelta(days=2)`, `pd.Timedelta(hours=6)` | pandas' `timedelta`; add it to or subtract it from a timestamp | `Timedelta` |
| Elapsed | `df['end'] - df['start']` | Time between two datetime columns, row by row, such as length of stay; `.dt.total_seconds() / 3600` gives hours and `.dt.days` whole days | `timedelta64` Series |

`pd.to_datetime(format=...)` and `.dt.strftime()` use the same format codes as the `datetime` card above.

### Code Snippet: Text Column to DatetimeIndex

`vitals` contains the three readings below; `recorded_at` starts as text and `heart_rate` is in bpm.

```python
vitals['recorded_at'] = pd.to_datetime(vitals['recorded_at'], format='%Y-%m-%d %H:%M')
vitals = vitals.set_index('recorded_at').sort_index()
display(vitals)
print(vitals.index.hour)
```

| recorded_at | heart_rate |
| --- | --- |
| 2024-03-01 08:00:00 | 72 |
| 2024-03-01 20:00:00 | 76 |
| 2024-03-02 08:00:00 | 88 |

```text
Index([8, 20, 8], dtype='int32', name='recorded_at')
```

<callout icon="⚠️" color="yellow_bg">
	## Sort by time first
	On unsorted rows, a date slice can raise `KeyError`. Run `sort_index()` first. A panel sorted by patient, then time, is not in time order, so select its dates with a mask such as `df[df['recorded_at'].between(start, end)]`.
</callout>

![xkcd 2867: DateTime. T2 minus T1 looks like simple subtraction until time zones and clock changes, later in this lecture, get involved.](media/xkcd_2867.png)

# Date Ranges and Frequencies

- **Frequency**: the repeating rule of a regular series, such as daily or hourly.
- **Alias**: its short text name, such as `'D'` (daily) or `'h'` (hourly).
- pandas uses frequencies to build schedules such as weekly check-in dates, to name a series' spacing, and to define reporting bins.

## Date Range Generation

`pd.date_range(start, end, freq=...)` or `pd.date_range(start, periods=n, freq=...)` builds timestamps at a chosen frequency.

| Schedule starting January 1, 2024 | First three dates |
| --- | --- |
| Monday check-ins (`W-MON`) | January 1, January 8, January 15 |
| Monthly draws (`MS`) | January 1, February 1, March 1 |
| Month-end reports (`ME`) | January 31, February 29, March 31 |

![`pd.date_range('2024-01-01', '2024-03-31', freq=...)`: `W-MON` gives 13 Mondays from January 1, `MS` the first of each month, and `ME` each month's last day, including February 29.](media/date_range_aliases.png)

### Reference Card: Frequency Aliases

| Alias | Meaning | Health example | Older alias that now fails |
| --- | --- | --- | --- |
| `'min'`, `'15min'` | Minutes | Bedside monitor | `'T'` |
| `'h'`, `'2h'` | Hours | Hourly vitals | `'H'` |
| `'D'` | Calendar days | Daily symptom diary | |
| `'W'` / `'W-MON'` | Every Sunday / every Monday (weeks end on that day) | Weekly check-in | |
| `'MS'` / `'ME'` | Month start / month end | Monthly lab draw / monthly report | `'M'` |

pandas 3 rejects the older aliases in the last column with `ValueError`.

### Code Snippet: An Hourly Grid

```python
print(pd.date_range('2024-01-01 08:00', periods=3, freq='h'))
```

```text
DatetimeIndex(['2024-01-01 08:00:00', '2024-01-01 09:00:00',
               '2024-01-01 10:00:00'],
              dtype='datetime64[us]', freq='h')
```

# Time Series Indexing and Selection

- **Time series selection**: picking rows by when they were recorded, such as a whole month or a date range. A sorted DatetimeIndex makes each one a short expression.
- **Partial-string indexing**: on a DatetimeIndex, Lecture 04's `.loc` accepts partial dates; `'2024-03'` selects every row in March 2024.

## Basic Time Series Selection

![Hourly heart rate, January 1 to 28, 2024: a `.loc` date range from January 15 to 17 selects all 72 hours of the three fever days, and `.loc` on January 5 selects that day's 14 rows, none from 08:00 to 17:00.](media/time_series_indexing.png)

### Reference Card: Calendar Selection

- `df.loc['2024-03']`: Every row in March 2024; `'2024'` selects the whole year
- `df.loc['2024-03-01':'2024-03-07']`: A date range, both endpoints included, as with Lecture 04's label slices
- `df.loc['2024-03-01 08:00']`: One timestamp
- `ts['2024-03']`: The same shortcut on a Series; on a DataFrame `df['2024-03']` looks for a _column_ and raises `KeyError`, so use `.loc`

### Code Snippet: Calendar Selection

`study_day` is a Series named `value` that numbers the dates in January and February 2024, starting at 1.

```python
print(study_day.loc['2024-02'].shape)  # leap year: 29 days
display(study_day.loc['2024-01-30':'2024-02-02'])
```

```text
(29,)
2024-01-30    30
2024-01-31    31
2024-02-01    32
2024-02-02    33
Freq: D, Name: value, dtype: int64
```

# Time Zone Handling

- **Time zone**: a region's rule for turning an instant into a wall-clock reading, including when daylight saving time moves the clocks.
- A multi-site trial records each visit on its clinic's wall clock: 09:00 in New York and 09:00 in Chicago are different moments, so ordering the visits needs every timestamp tied to one instant.

## Basic Time Zone Operations

- A **naive** timestamp is a clock reading with no zone attached, like `2024-03-09 09:00`. pandas cannot tell which instant it means.
- An **aware** timestamp carries a zone or UTC offset, like `2024-03-09 09:00-05:00`, so it pins down one instant.
- **UTC** (Coordinated Universal Time) has no daylight saving time, which makes it the safe zone for storing and ordering data.

| New York visit time | Attached offset | Same instant in UTC |
| --- | --- | --- |
| March 9, 2024 at 09:00 | UTC-5 | March 9 at 14:00 |
| March 11, 2024 at 09:00 | UTC-4 | March 11 at 13:00 |

![Four clinic visits on March 15, 2024: by local clock reading, Chicago's 08:30 (CH-01) comes first; in UTC, New York's 09:00 (NY-01, 13:00 UTC) is first and CH-01 (13:30 UTC) second.](media/two_clinics_utc.png)

### Reference Card: Time Zones

- `ts.tz_localize('America/New_York')`: Attach the recording zone to a naive DatetimeIndex; clock times are unchanged.
- `ts.tz_convert('UTC')`: Show aware timestamps in another zone; instants are unchanged. Raises `TypeError` on naive data.
- `df['t'].dt.tz_localize(...)` and `.dt.tz_convert(...)`: The same operations on a datetime column.
- `ts.index.tz` / `s.dt.tz`: The attached zone; `None` means naive.
- `pd.to_datetime(text, utc=True)`: Parse text with an offset, or known to be UTC, straight to aware UTC values.
- `pd.Timestamp('2024-03-09 14:00', tz='UTC')`, `pd.Timestamp.now(tz='UTC')`, `pd.date_range(..., tz='UTC')`: Build aware timestamps and ranges directly; `.tz_convert(...)` works on all of them. Comparing or subtracting an aware column and a naive `Timestamp` raises `TypeError`, so give a cutoff the column's `tz`.
- Zone names: full names from the IANA time-zone database (the standard list of world time zones), such as `'America/New_York'`; `'US/Eastern'` exists only for backward compatibility.

### Code Snippet: Local Clinic Times to UTC

`clinic` is a Series named `value` with readings of 120 and 118, indexed by naive New York times: March 9 and March 11, 2024, both at 09:00.

```python
local = clinic.tz_localize('America/New_York')
display(local)
display(local.tz_convert('UTC'))
```

```text
2024-03-09 09:00:00-05:00    120
2024-03-11 09:00:00-04:00    118
Name: value, dtype: int64
2024-03-09 14:00:00+00:00    120
2024-03-11 13:00:00+00:00    118
Name: value, dtype: int64
```

Daylight saving time began March 10, so the two 9 AM local visits are 14:00 and 13:00 UTC.

<callout icon="⚠️" color="yellow_bg">
	## Localize in the recording zone
	Localizing these New York readings as `'UTC'` raises no error but shifts each instant by 4 or 5 hours. Localize with the recording zone and convert to UTC to store and order. Convert to the reporting zone before grouping by local calendar days or weeks, or displaying local times.
</callout>

![xkcd 1799: Bad Map Projection: Time Zones. Each country is redrawn where its clocks say it should be; a wall-clock reading alone does not pin down where, or when, something happened.](media/xkcd_1799.png)

## Clock Changes: Repeated and Skipped Times

Twice a year, some local clock readings do not name exactly one instant:

- **Nonexistent**: on the spring-forward night (March 10, 2024 in New York), clocks jump from 02:00 to 03:00, so 02:30 never happens.
- **Ambiguous**: on the fall-back night (November 3, 2024), clocks return from 02:00 to 01:00, so 01:30 happens twice, first in daylight time (EDT) and then standard time (EST).

| UTC instant | New York clock | Offset |
| --- | --- | --- |
| 2024-11-03 05:00 | 01:00 EDT | UTC-4 |
| 2024-11-03 06:00 | 01:00 EST | UTC-5 |
| 2024-11-03 07:00 | 02:00 EST | UTC-5 |

![New York clock against UTC: on March 10, 2024 the clock skips 02:00 to 03:00, so 02:30 meets no instant; on November 3 it repeats 01:00 to 02:00, so 01:30 meets two (05:30 and 06:30 UTC); both become NaT.](media/ny_clock_changes_2024.png)

By default, `tz_localize()` raises `ValueError` at these times; the arguments below set them aside as `NaT` instead.

### Reference Card: Daylight-Saving Arguments

- `s.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')`: Repeated (fall-back) and skipped (spring-forward) clock times become `NaT`.
- `aware.isna().sum()`: Count the readings set aside, and report that count before dropping them.
- `pd.date_range(start, end, freq='h', tz='America/New_York', inclusive='left')`: Every elapsed hour from `start` up to, but not including, `end`; a spring-forward day has 23 and a fall-back day has 25.

### Code Snippet: Flag Clock-Change Timestamps

`local` is a Series of naive New York timestamps: March 10, 2024 at 01:00 and 02:30, and November 3 at 01:30 and 03:00.

```python
aware = local.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')
display(aware.dt.tz_convert('UTC'))
print("Set aside:", aware.isna().sum())

spring_day = pd.date_range('2024-03-10', '2024-03-11', freq='h',
                           tz='America/New_York', inclusive='left')
print(len(spring_day))  # hours in that local day
```

```text
0   2024-03-10 06:00:00+00:00
1                         NaT
2                         NaT
3   2024-11-03 08:00:00+00:00
dtype: datetime64[us, UTC]
Set aside: 2
23
```

_Time series analysis is 90% datetime wrangling, 5% actual analysis, and 5% swearing at time zone conversions._

# LIVE DEMO!

[Open Demo 1 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo1_datetime_fundamentals.ipynb)

# Resampling and Frequency Conversion

- **Resampling**: changing a time series' frequency, such as turning minute-by-minute heart rates into one summary per hour for a ward report.
- **Downsampling**: many readings combined into fewer, longer bins.
- **Upsampling**: readings spread over more, shorter slots, most of which start empty.

![Downsampling: 91 daily home weights (gray) become 3 month-end means, 82.02, 82.00, and 82.34 kg, each labeled with its month's last day; March's mean barely shows the 84.4 kg peak.](media/resampling_example.png)

## Basic Resampling

`resample()` works like Lecture 08's `groupby()`: it splits rows into time bins and needs an aggregation such as `.mean()` to combine each bin into one row.

### Reference Card: Resampling Frequencies

- `ts.resample('h').mean()`: Hourly means; each row is labeled with the start of its hour
- `ts.resample('D').mean()`: Daily means, labeled with the date
- `ts.resample('W').mean()`: Weekly means, labeled with the Sunday that ends each week
- `ts.resample('ME').mean()`: Monthly means, labeled with the month's last day
- `df.resample('ME').mean()`: One summary per column; a text column raises `TypeError`, so select numeric columns first or use `.agg()` per column

### Code Snippet: Basic Resampling

`day_num` is a Series named `value` holding 1 to 30 on January 1 to 30, 2023.

```python
display(day_num.resample('W').mean())
display(day_num.resample('ME').mean())   # all 30 days fall in January
```

```text
2023-01-01     1.0
2023-01-08     5.0
2023-01-15    12.0
2023-01-22    19.0
2023-01-29    26.0
2023-02-05    30.0
Freq: W-SUN, Name: value, dtype: float64
2023-01-31    15.5
Freq: ME, Name: value, dtype: float64
```

## How Resampling Draws Bins

`resample('2h')` chops the timeline into two-hour **bins** and aggregates the readings in each. Two arguments decide where a boundary reading goes:

- `closed`: which edge belongs to the bin. With `closed='left'`, a 10:00 reading goes in the 10:00-12:00 bin, not 08:00-10:00.
- `label`: which edge names the bin in the output.

| Reading time | Heart rate | Bin label with `resample('2h')` |
| --- | --- | --- |
| 08:00 | 70 | 08:00 |
| 08:30 | 72 | 08:00 |
| 09:45 | 75 | 08:00 |
| 10:00 | 80 | 10:00 |
| 11:15 | 78 | 10:00 |

![`resample('2h')` on March 1, 2024: each bin includes its left edge and excludes its right, so 10:00 opens the second bin; the means are 72.3 (3 readings, labeled 08:00) and 79.0 (2 readings, labeled 10:00).](media/resample_bins_2h.png)

### Reference Card: Bin Edges

- `ts.resample('2h', closed='left', label='left')`: Each bin includes and is named by its start; the default for clock frequencies (`'min'`, `'h'`, `'D'`) and start-anchored ones (`'MS'`).
- `ts.resample('W', closed='right', label='right')`: Each bin includes and is named by its end; the default for end-anchored frequencies (`'W'`, `'ME'`). So the weekly bin labeled Sunday `2023-01-08` above holds January 2 through 8, and the first bin holds only January 1.

### Code Snippet: Bin Boundaries

`readings` contains the five March 1, 2024 readings above, with a DatetimeIndex.

```python
display(readings.resample('2h').agg(['mean', 'count']))
```

|  | mean | count |
| --- | --- | --- |
| 2024-03-01 08:00:00 | 72.333333 | 3 |
| 2024-03-01 10:00:00 | 79.000000 | 2 |

![xkcd 1985: Meteorologist. Five hourly 20% chances of rain do not say how likely rain is this afternoon; a longer bin needs an aggregation that answers the question being asked.](media/xkcd_1985.png)

# Resampling Summaries, Grids, and Groups

- Any aggregation can summarize a bin: a maximum heart rate flags a crisis, and a reading count shows whether the monitor stayed connected.
- **Grouped resampling**: binning each patient's readings separately, so no bin averages one patient with another.
- Upsampling to an hourly grid adds empty slots between readings, to fill or leave missing.

## Resampling with Different Aggregations

Any aggregation that works after `groupby()` works after `resample()`, including several at once and Lecture 08's named aggregation.

![`resample('D').count()` on an hourly monitor, January 1 to 28, 2024: 24 readings on 27 days and 14 on January 5, when it was unplugged for 10 hours.](media/daily_reading_counts.png)

### Reference Card: Resampling Aggregations

- `ts.resample('D').mean()`, `.sum()`, `.max()`, `.min()`, `.std()`: One summary value per bin
- `ts.resample('D').count()`: Non-missing readings per bin; `0` marks a bin with no data
- `ts.resample('D').agg(['mean', 'std', 'min', 'max'])`: Multiple aggregations
- `df.resample('W').agg({'temperature': ['mean', 'std'], 'heart_rate': 'mean'})`: Different summaries for different columns

### Code Snippet: Resampling Aggregations

`daily` has eight rows, January 1 to 8, 2023, indexed by date: `temperature` (°F) 98.6, 98.4, 98.8, 99.6, 100.2, 99.1, 98.5, 98.4 and `heart_rate` (bpm) 72, 74, 76, 80, 82, 76, 72, 71.

```python
display(daily.resample('W').agg({'temperature': ['mean', 'max'], 'heart_rate': ['mean', 'count']}).round(1))
```

|  | temperature |  | heart_rate |  |
| --- | --- | --- | --- | --- |
|  | mean | max | mean | count |
| 2023-01-01 | 98.6 | 98.6 | 72.0 | 1 |
| 2023-01-08 | 99.0 | 100.2 | 75.9 | 7 |

The `count` column shows that the first weekly mean describes one day and the second seven.

## Upsampling: Filling a Finer Grid

`asfreq()` lays a series onto a regular grid without combining observations: values exactly on the grid are kept, and the other slots become `NaN`. Leave the new slots missing, or fill them with Lecture 05's `ffill()` or `interpolate()`.

![Six temperatures charted every 4 hours on January 15, 2024, on an hourly grid: `asfreq()` leaves 15 of 21 slots NaN, `ffill(limit=2)` leaves 5, and `interpolate()` fills all 21 with straight lines.](media/charted_temp_upsampling.png)

### Reference Card: Upsampling

- `ts.resample('h').asfreq()`: A finer grid; new slots are `NaN`.
- `ts.resample('h').ffill(limit=2)`: Carry the last reading forward, at most 2 slots.
- `ts.resample('h').interpolate()`: Straight-line fill between known readings.

Filled values are estimates, not measurements; keep a flag or the original column if later steps need to know which values were observed. Interpolation uses the later endpoint: the 09:00 estimate below needs the 11:00 reading, so it cannot be used by a score running at 09:00.

### Code Snippet: Upsampling Choices

`pulse` records 70.0 and 76.0 bpm (floats) at 08:00 and 11:00 on March 1, 2024, with a DatetimeIndex.

```python
display(pd.DataFrame({
    'asfreq': pulse.resample('h').asfreq(),
    'ffill': pulse.resample('h').ffill(),
    'interpolate': pulse.resample('h').interpolate(),
}))
```

|  | asfreq | ffill | interpolate |
| --- | --- | --- | --- |
| 2024-03-01 08:00:00 | 70.0 | 70.0 | 70.0 |
| 2024-03-01 09:00:00 | NaN | 70.0 | 72.0 |
| 2024-03-01 10:00:00 | NaN | 70.0 | 74.0 |
| 2024-03-01 11:00:00 | 76.0 | 76.0 | 76.0 |

## Resampling Each Patient Separately

Resample inside each patient's group (Lecture 08's split-apply-combine); the whole table mixes patients in each bin. For the snippet's five readings:

| Two-hour bin | Whole table (mixes patients) | P1 only | P2 only |
| --- | --- | --- | --- |
| 08:00 | 79.0 | 73.5 | 90.0 |
| 10:00 | 87.0 | 80.0 | 94.0 |

![Two-hour bins on March 1, 2024: P1's means are 73.5 and 80.0 bpm and P2's 90.0 and 94.0; the whole-table means, 79.0 and 87.0, fall between the patients and describe neither.](media/per_patient_bins.png)

### Reference Card: Grouped Resampling

- `df.set_index('recorded_at').groupby('patient_id')['heart_rate'].resample('2h').agg(['mean', 'count'])`: One row per patient per two-hour bin, with a `(patient_id, recorded_at)` MultiIndex.
- `df.set_index('recorded_at').groupby('patient_id')[['heart_rate', 'source_row']].resample('h').asfreq()`: Each patient's hourly grid from their first reading's hour to their last; empty hours become `NaN`, and readings off the hour are dropped.
- `df['recorded_at'].eq(df['recorded_at'].dt.floor('h')).all()`: `True` only if every reading sits exactly on the hour; check this before `asfreq()`.
- `df['source_row'] = 1` before `asfreq()`: Grid-created rows get `NaN` in `source_row`, so `grid['source_row'].isna()` tells them apart from real readings with a missing value.
- `df.set_index('recorded_at').groupby('patient_id').resample('2h').agg(mean_hr=('heart_rate', 'mean'), n_rows=('source_row', 'count'))`: Named summaries in Lecture 08's `(column, function)` form; `count()` skips missing values, so count `source_row` to include readings with a missing heart rate.
- `.reset_index()`: Turn the MultiIndex back into ordinary `patient_id` and `recorded_at` columns.

### Code Snippet: Two-Hour Summaries per Patient

`vitals` has `patient_id`, datetime `recorded_at`, and `heart_rate` columns: P1 has 72, 75, and 80 bpm at 08:00, 09:00, and 10:00 on March 1, 2024; P2 has 90 and 94 at 09:00 and 10:00.

```python
per_patient = (
    vitals.set_index('recorded_at').groupby('patient_id')['heart_rate']
    .resample('2h').agg(['mean', 'count']).reset_index()
)
display(per_patient)
```

|  | patient_id | recorded_at | mean | count |
| --- | --- | --- | --- | --- |
| 0 | P1 | 2024-03-01 08:00:00 | 73.5 | 2 |
| 1 | P1 | 2024-03-01 10:00:00 | 80.0 | 1 |
| 2 | P2 | 2024-03-01 08:00:00 | 90.0 | 1 |
| 3 | P2 | 2024-03-01 10:00:00 | 94.0 | 1 |

### Code Snippet: Hourly Grid per Patient

P1 and P2 have two charted readings each; P1's last has no heart rate. In `vitals`, `recorded_at` is datetime.

```python
print(vitals['recorded_at'].eq(vitals['recorded_at'].dt.floor('h')).all())
vitals['source_row'] = 1
grid = (
    vitals.set_index('recorded_at').groupby('patient_id')[['heart_rate', 'source_row']]
    .resample('h').asfreq().reset_index()
)
grid['grid_created'] = grid['source_row'].isna()
grid['value_missing'] = grid['source_row'].notna() & grid['heart_rate'].isna()
display(grid.drop(columns='source_row'))
```

```text
True
```

|  | patient_id | recorded_at | heart_rate | grid_created | value_missing |
| --- | --- | --- | --- | --- | --- |
| 0 | P1 | 2024-03-01 08:00:00 | 72.0 | False | False |
| 1 | P1 | 2024-03-01 09:00:00 | NaN | True | False |
| 2 | P1 | 2024-03-01 10:00:00 | NaN | True | False |
| 3 | P1 | 2024-03-01 11:00:00 | NaN | False | True |
| 4 | P2 | 2024-03-01 09:00:00 | 90.0 | False | False |
| 5 | P2 | 2024-03-01 10:00:00 | 94.0 | False | False |

P1's three `NaN` rows look alike; only the flags show that the grid created 09:00 and 10:00, while 11:00 is a real reading with no heart rate. This grid requires one row per patient and timestamp; duplicate keys must be resolved before `asfreq()` can assign one value to each slot.

### Reference Card: Gap Runs

A **gap run** is a stretch of consecutive grid hours with no reading, such as a monitor unplugged for three hours. With `source_observed` `True` for real readings (the opposite of `grid_created`), a running count of readings stays flat through each run, so its hours share a number:

![`cumsum()` of `source_observed` stays flat through each gap: P1 has runs of 2 and 3 hours (run_id 1 and 2), and P2 has one 1-hour run (run_id 2).](media/gap_run_tiles.png)

- `df['source_observed'] = ~df['grid_created']`: `True` for real readings, including a charted row with a missing value.
- `df['run_id'] = df.groupby('patient_id')['source_observed'].cumsum()`: Running count of each patient's real readings (`cumsum()` counts `True` as 1), flat through a gap.
- `df[~df['source_observed']].groupby(['patient_id', 'run_id']).size()`: Each gap run's length in hours; count the runs and take their `max()` per patient.

### Code Snippet: Count Gap Runs per Patient

`hours` is sorted by patient and hour: P1 has the sequence above, and P2 has `True, True, False, True`.

```python
hours['run_id'] = hours.groupby('patient_id')['source_observed'].cumsum()
gaps = hours[~hours['source_observed']]
runs = gaps.groupby(['patient_id', 'run_id']).size().rename('gap_hours').reset_index()
display(runs)
display(runs.groupby('patient_id')['gap_hours'].agg(['count', 'max']))
```

|  | patient_id | run_id | gap_hours |
| --- | --- | --- | --- |
| 0 | P1 | 1 | 2 |
| 1 | P1 | 2 | 3 |
| 2 | P2 | 2 | 1 |

| patient_id | count | max |
| --- | --- | --- |
| P1 | 2 | 3 |
| P2 | 1 | 1 |

A patient with no gaps has no rows in `runs`; report 0 runs for them rather than leaving them out.

![xkcd 605: Extrapolating. Two readings make a line, not a trend; a rolling mean over several readings is steadier than any single change.](media/xkcd_605.png)

# Rolling Window Operations

- **Rolling window**: a fixed-size frame that slides along a series and computes a statistic at every row, such as the mean of the last seven daily blood pressures, steadier than any single noisy reading.
- `resample('W').mean()` returns one row per week; `rolling(7).mean()` returns one row per original reading.

## Basic Rolling Operations

![Daily mean heart rate (gray dots), a 7-day rolling mean (blue) with its ±1 SD band, and an exponentially weighted mean (EWM, span 7, dashed orange): over the January 15 to 17 fever, EWM rises and falls sooner, peaking at 86.9 bpm against 84.7 on January 17.](media/daily_hr_rolling_ewm.png)

### Reference Card: Rolling Windows

- `ts.rolling(window=5)`: A **count window**: the current row and the 4 before it, however far apart; like `resample()`, it needs an aggregation after it
- `ts.rolling(window=5).mean()`, `.std()`, `.sum()`, `.min()`, `.max()`: One value per row; `NaN` unless the window has 5 nonmissing readings, so the first 4 rows are `NaN`
- `ts.rolling('7D').mean()`: A **time window**: the readings in the 7 days ending at each row, which suits irregular data such as clinic visits; needs a sorted DatetimeIndex; starts once one nonmissing reading is present
- `ts.rolling('2h', closed='left').mean()`: Time window that excludes the current row (a past-only window)
- `ts.rolling(window=7, min_periods=3).mean()`: Start once the window holds 3 readings instead of waiting for all 7
- `ts.rolling(window=7, center=True).mean()`: A **centered window**: 3 rows before, the current row, and 3 after, so each value uses later readings
- `ax.fill_between(ts.index, mean - std, mean + std, alpha=0.2)`: Shade the band in the figure above on a Lecture 07 `Axes`; `alpha` keeps the lines visible through it

A centered window reads future rows: its smooth curve suits describing a completed series, not a summary built only from readings up to the current row.

### Code Snippet: Count Window vs Time Window

`glucose` contains the four readings below, in mg/dL, with a sorted DatetimeIndex.

```python
compare = pd.DataFrame({
    'glucose': glucose,
    'last_2_readings': glucose.rolling(2).mean(),
    'last_2_hours': glucose.rolling('2h').mean(),
})
display(compare)
```

|  | glucose | last_2_readings | last_2_hours |
| --- | --- | --- | --- |
| 2024-03-01 07:00:00 | 110 | NaN | 110.0 |
| 2024-03-01 08:00:00 | 145 | 127.5 | 127.5 |
| 2024-03-01 11:00:00 | 130 | 137.5 | 130.0 |
| 2024-03-01 11:30:00 | 180 | 155.0 | 155.0 |

At 11:00, the count window averages in the 08:00 reading from three hours earlier; the two-hour window sees only 11:00.

## Another Smoother: Exponentially Weighted Means

- **Exponentially weighted moving average (EWM)**: an optional alternative to rolling means that weights recent readings more heavily than older ones.
- `ts.ewm(span=7).mean()` returns one smoothed value per row; a larger `span` means slower decay.

# LIVE DEMO!

[Open Demo 2 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo2_indexing_resampling.ipynb)

# Entity-Aware Features and Past-Only Windows

- **Feature**: an input column for a risk score or model.
- **Past-only feature**: a summary of a patient's history from only that patient's earlier rows, such as the previous heart rate or the mean of the last two hours for an early-warning score.
- **Prediction time**: when the score runs.
- **Future leakage**: reading a row from after the prediction time. It makes the score look better than it will be, with no error to warn you.

## Shifting and Lagging

- **Shifting**: sliding values down or up the rows while the dates stay in place.
- **Lag**: the previous row's value, such as the last visit's blood pressure; a **lead** is the next row's.

![Daily weights, March 6 to 14, 2024: `shift(1)` (lag) moves each value one row later and `shift(-1)` (lead) one row earlier; `diff()`, today minus the lag, crosses the 1.0 kg alert line only on March 10, at +1.3 kg.](media/shifting_lagging.png)

Shifting counts rows, not time: with irregular clinic visits, the "previous" reading can be one week or four weeks back.

### Reference Card: Lagged Features

- `ts.shift(1)`: Lag: each row gets the previous row's value; the first row becomes `NaN`
- `ts.shift(-1)`: Lead: each row gets the next row's value; the last row becomes `NaN`
- `ts.diff()`: Current value minus the previous row's value

### Code Snippet: Lagged Features

`weight` contains the five daily weights below, in kg.

```python
weight['lag_1'] = weight['weight'].shift(1)
weight['lead_1'] = weight['weight'].shift(-1)
weight['diff'] = weight['weight'].diff()
display(weight)
```

|  | weight | lag_1 | lead_1 | diff |
| --- | --- | --- | --- | --- |
| 2023-01-01 | 70.5 | NaN | 70.8 | NaN |
| 2023-01-02 | 70.8 | 70.5 | 70.2 | 0.3 |
| 2023-01-03 | 70.2 | 70.8 | 71.0 | -0.6 |
| 2023-01-04 | 71.0 | 70.2 | 70.9 | 0.8 |
| 2023-01-05 | 70.9 | 71.0 | NaN | -0.1 |

## Grouped Lags and Past-Only Windows

In a panel, a plain `shift(1)` runs down the whole stacked column (`wrong_previous`); grouping by `patient_id` first keeps each shift inside one patient's history (`previous_hr`):

| | patient_id | recorded_at | heart_rate | wrong_previous | previous_hr |
| --- | --- | --- | --- | --- | --- |
| 0 | P1 | 2024-03-01 08:00:00 | 72 | NaN | NaN |
| 1 | P1 | 2024-03-01 09:00:00 | 80 | 72.0 | 72.0 |
| 2 | P1 | 2024-03-01 11:30:00 | 95 | 80.0 | 80.0 |
| 3 | P2 | 2024-03-01 09:00:00 | 88 | 95.0 | NaN |
| 4 | P2 | 2024-03-01 10:00:00 | 86 | 88.0 | 88.0 |
| 5 | P2 | 2024-03-01 11:00:00 | 84 | 86.0 | 86.0 |

![Six readings on March 1, 2024: grouped `shift(1)` (green) stays inside each patient, plain `shift(1)` hands P1's 11:30 reading of 95 to P2's first row, and `rolling('2h', closed='left')` at P1's 11:30 finds no reading from 09:30, so NaN.](media/grouped_shift_leak.png)

Row 3 is the leak: P2's "previous" heart rate is P1's 95. Sort by patient and time first: `shift()` reads row order, so on unsorted rows a patient's "previous" reading can be a later one.

### Reference Card: Past-Only Panel Features

| Task | Code | Result |
| --- | --- | --- |
| Order each history | `df.sort_values(['patient_id', 'recorded_at'])` | Patients together, oldest reading first |
| Previous value | `df.groupby('patient_id')['heart_rate'].shift(1)` | Same-patient previous reading; `NaN` on each patient's first row |
| Next value as target | `df.groupby('patient_id')['heart_rate'].shift(-1)` | Same-patient next reading; never use this later value as a current feature |
| Change | `df.groupby('patient_id')['heart_rate'].diff()` | Current minus previous, same patient |
| Mean of previous n readings | `df.groupby('patient_id')['heart_rate'].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())` | Excludes the current row; aligned to the original rows |
| Mean over previous 2 hours | `df.set_index('recorded_at').groupby('patient_id')['heart_rate'].rolling('2h', closed='left').mean()` | `closed='left'` excludes the current time; `reset_index()`, then `merge(..., validate='one_to_one')` it back |

### Code Snippet: Grouped Lag

`vitals` has the six readings above, with datetime `recorded_at` values.

```python
vitals = vitals.sort_values(['patient_id', 'recorded_at'])
vitals['wrong_previous'] = vitals['heart_rate'].shift(1)
vitals['previous_hr'] = vitals.groupby('patient_id')['heart_rate'].shift(1)
display(vitals)  # the table above
```

### Code Snippet: Grouped Past-Only Windows

Keep the three original columns. `transform()` preserves row alignment; the time window returns keys to merge back (Lecture 06). Each patient and timestamp pair is unique here, so `validate='one_to_one'` checks that every feature matches exactly one reading.

```python
vitals = vitals[['patient_id', 'recorded_at', 'heart_rate']].copy()
vitals['mean_prev_2'] = vitals.groupby('patient_id')['heart_rate'].transform(
    lambda s: s.shift(1).rolling(2, min_periods=1).mean()
)
prev_2h = (
    vitals.set_index('recorded_at').groupby('patient_id')['heart_rate']
    .rolling('2h', closed='left').mean().rename('mean_prev_2h').reset_index()
)
vitals = vitals.merge(prev_2h, on=['patient_id', 'recorded_at'], validate='one_to_one')
display(vitals)
```

|  | patient_id | recorded_at | heart_rate | mean_prev_2 | mean_prev_2h |
| --- | --- | --- | --- | --- | --- |
| 0 | P1 | 2024-03-01 08:00:00 | 72 | NaN | NaN |
| 1 | P1 | 2024-03-01 09:00:00 | 80 | 72.0 | 72.0 |
| 2 | P1 | 2024-03-01 11:30:00 | 95 | 76.0 | NaN |
| 3 | P2 | 2024-03-01 09:00:00 | 88 | NaN | NaN |
| 4 | P2 | 2024-03-01 10:00:00 | 86 | 88.0 | 88.0 |
| 5 | P2 | 2024-03-01 11:00:00 | 84 | 87.0 | 87.0 |

At P1's 11:30 reading, the previous two readings average 76, but no reading falls in the two hours before, so the elapsed-time mean is `NaN`.

![xkcd 2289: Scenario 4. The fourth scenario's curve bends back in time; a feature that reads rows from after the prediction time is the same kind of time travel.](media/xkcd_2289.png)

## Availability at Prediction Time

- A timestamp says when something happened, not when anyone could know it: a lab is resulted after collection, and a monthly case count is published weeks after the month ends.
- **Available**: a feature whose inputs were all reported or otherwise accessible at or before the prediction time; use their availability times, not just when they were measured.
- **Chronological holdout**: build a method on rows before a cutoff and test it on rows from the cutoff onward, the way it would meet future patients.

![Prediction time 11:00 on March 1, 2024: lactate (resulted 08:45) and troponin (10:50) are available, creatinine (resulted 11:15) is not; at the same cutoff, 4 vitals are `earlier` and 2, including P2's 11:00, are `later_holdout`.](media/lab_availability_timeline.png)

### Reference Card: Availability and Chronological Blocks

- `labs['resulted_at'] <= prediction_time`: `True` where the value was known at prediction time.
- `np.where(df['recorded_at'] < cutoff, 'earlier', 'later_holdout')`: Label rows before the cutoff `earlier` (to build a method) and the rest `later_holdout` (to test it); `np.where` is from Lecture 03.

### Code Snippet: Recorded Is Not Available

`labs` has the three tests below, with datetime collection and reporting columns.

```python
prediction_time = pd.Timestamp('2024-03-01 11:00')
labs['available'] = labs['resulted_at'] <= prediction_time
labs['turnaround_h'] = ((labs['resulted_at'] - labs['collected_at']).dt.total_seconds() / 3600).round(2)
display(labs)
```

|  | test | collected_at | resulted_at | available | turnaround_h |
| --- | --- | --- | --- | --- | --- |
| 0 | lactate | 2024-03-01 08:00:00 | 2024-03-01 08:45:00 | True | 0.75 |
| 1 | creatinine | 2024-03-01 09:30:00 | 2024-03-01 11:15:00 | False | 1.75 |
| 2 | troponin | 2024-03-01 10:30:00 | 2024-03-01 10:50:00 | True | 0.33 |

### Code Snippet: Chronological Holdout

```python
cutoff = pd.Timestamp('2024-03-01 11:00')
vitals['block'] = np.where(vitals['recorded_at'] < cutoff, 'earlier', 'later_holdout')
display(vitals[['patient_id', 'recorded_at', 'block']])
```

|  | patient_id | recorded_at | block |
| --- | --- | --- | --- |
| 0 | P1 | 2024-03-01 08:00:00 | earlier |
| 1 | P1 | 2024-03-01 09:00:00 | earlier |
| 2 | P1 | 2024-03-01 11:30:00 | later_holdout |
| 3 | P2 | 2024-03-01 09:00:00 | earlier |
| 4 | P2 | 2024-03-01 10:00:00 | earlier |
| 5 | P2 | 2024-03-01 11:00:00 | later_holdout |

P2's 11:00 reading falls in the holdout, because `<` keeps the cutoff itself out of `earlier`.

A chronological split permits earlier history from the same patient on both sides; it tests later readings, not performance on entirely unseen patients.

# Time Series Visualization

- **Time-series plot**: readings drawn against time on the x-axis, in order, showing trends, seasonal cycles, and gaps that a summary table hides.
- Draw the raw readings faintly with a smoother such as a rolling mean over them, let the line break where data is missing, and name the time zone when it matters.

![Tufte, _The Visual Display of Quantitative Information_, p. 30: New York City's weather for 1980 shows 1,888 numbers, daily highs and lows against the normal range plus monthly precipitation and daily humidity.](media/tufte_vdqi_p30_nyc_weather.jpg)

## Basic Time Series Plots

![Daily influenza-like illness visits (gray) with the 30-day rolling mean (blue), which follows the winter peak without the daily noise](media/viz_ili_rolling.png)

![Bar chart of mean daily visits for each calendar month, highest in January and lowest in July](media/viz_ili_monthly.png)

![Daily flu-like illness visits, December 2022 to March 2023: as reported, the line bridges the February 1 to 14 outage; after `asfreq('D')`, the 14 missing days become NaN and the line breaks.](media/ili_outage_gap.png)

### Reference Card: Time Series Plotting

- `ts.plot()`: Line plot with the DatetimeIndex on the x-axis; returns the `Axes` (`ax = ts.plot()`) for further customization
- `ts.plot(figsize=(12, 6), title='Title', marker='o')`: Figure size, title, and a marker at each reading
- `ts.asfreq('D').plot()`: Inserts `NaN` for missing days, so the line breaks at gaps instead of bridging them; `ts.asfreq('D')` starts the grid at the first reading's clock time, while `ts.resample('D').asfreq()` starts at midnight and turns off-midnight readings into `NaN`; the two match for date-only readings
- `ts.groupby(ts.index.month).mean().plot(kind='bar', ax=ax)`: Calendar-month means as bars, to show a seasonal pattern; `groupby()` also accepts an array of keys as long as the series, such as `ts.index.month`. Draw them on a new `fig, ax = plt.subplots()`: bars on the dated line's Axes raise `AttributeError`. Bars start at zero (Lecture 07), which suits counts; for a reading such as temperature, plot the difference from a baseline such as 98.6 °F
- `ax.axhline(98.6, linestyle='--', label='Normal')`: Horizontal reference line for a clinical threshold
- `ax.set(xlabel='Time (UTC)')`: Name the zone of an aware time axis

### Code Snippet: Time-Series Plotting

`ts` is a year of daily influenza-like illness visits at a clinic, with a DatetimeIndex.

```python
ax = ts.plot(alpha=0.5, label='Daily', color='gray')
ts.rolling(window=30).mean().plot(ax=ax, linewidth=2, label='30-Day Rolling Mean', color='blue')
ax.set(title='Influenza-like Illness Visits with Rolling Mean', xlabel='Date', ylabel='Visits per day')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
monthly_mean = ts.groupby(ts.index.month).mean()
monthly_mean.plot(kind='bar', ax=ax, rot=0, title='Mean Daily Visits by Calendar Month',
                  xlabel='Month', ylabel='Visits per day')
plt.show()
```

# LIVE DEMO!

[Open Demo 3 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo3_visualization_automation.ipynb)
