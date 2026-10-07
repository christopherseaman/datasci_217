---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    name: python
    version: 3.13
---

# Demo 1: Parse, Index, and Select Patient Readings

A heart-failure clinic receives a home-scale log with text timestamps, out-of-order rows, and one impossible date.

- Parse text into dates and find the impossible one.
- Make a sorted `DatetimeIndex` and format an hour key.
- Select a calendar interval.
- All patient data are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

```python
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

## 1. One date at a time with `datetime`

A discharge time arrives as text. `strptime()` parses it, `strftime()` formats it for a letter, and a `timedelta` moves it forward to the follow-up contacts.

```python
discharge = datetime.strptime('2024-03-01 14:30', '%Y-%m-%d %H:%M')
print(discharge)
print(discharge.strftime('%B %d, %Y at %I:%M %p'))

print('7-day phone call:', (discharge + timedelta(days=7)).strftime('%Y-%m-%d'))
print('30-day clinic visit:', (discharge + timedelta(days=30)).strftime('%Y-%m-%d'))

# Subtracting two datetimes gives a timedelta
birth_date = datetime(1958, 7, 4)
age = discharge - birth_date
print('Age in days:', age.days)
print('Age in years:', round(age.days / 365.25, 1))
```

**Expect:** `2024-03-01 14:30:00`, then `March 01, 2024 at 02:30 PM`. The phone call falls on `2024-03-08` and the clinic visit on `2024-03-31` (2024 is a leap year, so February has 29 days). Age: `23982` days, `65.7` years.


## 2. A text column becomes a DatetimeIndex

The home scale app exports its log as text, partly out of order, and one entry has an impossible date: February 30.

```python
raw = pd.DataFrame({
    'patient_id': ['P001'] * 5,
    'recorded_at': ['2024-03-03 07:10', '2024-03-01 07:05', '2024-02-30 07:00',
                    '2024-03-02 07:15', '2024-03-04 06:55'],
    'weight_kg': [82.3, 81.9, 82.0, 82.1, 83.4],
    'resting_hr': [74, 72, 73, 75, 79],
})
display(raw)
display(raw.dtypes)
```

**Expect:** five rows. `patient_id` and `recorded_at` are `str`, `weight_kg` is `float64`, and `resting_hr` is `int64`. Text has no calendar arithmetic; sorting it is chronological only when every date uses a consistent sortable pattern.

`format=` states the pattern the export promises, and `errors='coerce'` turns anything that does not fit into `NaT` instead of stopping.

```python
raw['recorded_at'] = pd.to_datetime(raw['recorded_at'], format='%Y-%m-%d %H:%M', errors='coerce')
display(raw)
print('Unparseable dates:', raw['recorded_at'].isna().sum())
```

**Expect:** row 2 shows `NaT` in `recorded_at`, and `Unparseable dates: 1`.

`.dt.strftime()` formats a whole column with the same codes as `datetime.strftime()`. An hour key such as `2024030307` names the hour each weight was taken, which is handy as a row ID or a file name; `NaT` has no date to format, so its key stays missing.

```python
display(raw['recorded_at'].dt.strftime('%Y%m%d%H'))
```

**Expect:** `2024030307`, `2024030107`, `NaN` (row 2), `2024030207`, and `2024030406`, with `dtype: str`.

Drop the bad row (Lecture 05), make the timestamps the index, and check the order before and after sorting.

```python
log = raw.dropna(subset=['recorded_at']).set_index('recorded_at')
print('Sorted?', log.index.is_monotonic_increasing)

log = log.sort_index()
print('Sorted?', log.index.is_monotonic_increasing)
display(log)
print(log.index.day_name())
```

**Expect:** `Sorted? False`, then `Sorted? True`, and a table of four rows with columns `patient_id`, `weight_kg`, and `resting_hr`, from `2024-03-01 07:05:00` (81.9 kg) to `2024-03-04 06:55:00` (83.4 kg). The day names run `Friday`, `Saturday`, `Sunday`, `Monday`.


## 3. Optional extension: schedules and frequency inference

Prerequisite: BONUS.md, Frequency Inference and Specialized Schedules. Uses the `log` from section 2.

`pd.date_range()` builds a schedule from a frequency alias, and `pd.infer_freq()` reads the spacing back from an index.

```python
weekly_calls = pd.date_range('2024-03-04', periods=4, freq='W-MON')   # Monday phone check-ins
monthly_labs = pd.date_range('2024-03-01', periods=3, freq='MS')      # first-of-month blood tests
clinic_days = pd.bdate_range('2024-03-01', '2024-03-14')              # weekdays the clinic is open
display(weekly_calls)
display(monthly_labs)
print('Clinic days:', len(clinic_days))
print('Weekly calls:', pd.infer_freq(weekly_calls))
print('Home weigh-ins:', pd.infer_freq(log.index))
```

**Expect:** Mondays `2024-03-04`, `03-11`, `03-18`, `03-25`; lab dates `2024-03-01`, `04-01`, `05-01`; `Clinic days: 10` (two weekends skipped). `infer_freq` returns `W-MON` for the calls and `None` for the weigh-ins: the patient steps on the scale at a slightly different minute each morning, so the log has no single frequency.


## 4. Select a calendar interval

Use the sorted `log` from section 2. A partial month selects every reading in that month; a date slice includes both endpoint days.

```python
print(log.loc['2024-03'].shape)
display(log.loc['2024-03-01':'2024-03-02'])
print('Hours:', log.index.hour.tolist())
```

**Expect:** `(4, 3)` for March. The date slice has March 1 at 07:05 (81.9 kg, 72 bpm) and March 2 at 07:15 (82.1 kg, 75 bpm). `Hours: [7, 7, 7, 6]` extracts the clock hour without turning the dates back into text.


## 5. Optional extension: select by clock time

Prerequisite: BONUS.md, Time-of-Day Selection Example. Builds its own data.

A second patient's monitor records heart rate and oxygen saturation (SpO2, in percent) every hour for one week. Heart rate runs about 12 beats per minute lower between midnight and 06:00, while the patient sleeps.

```python
rng = np.random.default_rng(42)
hours = pd.date_range('2024-03-04', periods=24 * 7, freq='h')
icu = pd.DataFrame({
    'heart_rate': rng.normal(84, 5, len(hours)).round().astype(int),
    'spo2': rng.integers(93, 99, len(hours)),
}, index=hours)
icu.loc[icu.index.hour < 6, 'heart_rate'] -= 12
print(icu.shape)
display(icu.head(3))
```

**Expect:** `(168, 2)` and three rows starting at `2024-03-04 00:00:00`.

`between_time()` keeps a clock range on every day, and `at_time()` keeps one clock time.

```python
day_shift = icu.between_time('09:00', '17:00')
overnight = icu.between_time('00:00', '05:00')
print('Day readings:', day_shift.shape, 'mean HR', round(day_shift['heart_rate'].mean(), 1))
print('Overnight readings:', overnight.shape, 'mean HR', round(overnight['heart_rate'].mean(), 1))
display(icu.at_time('12:00').head(3))
```

**Expect:** `Day readings: (63, 2)` (9 hours × 7 days, both ends included) with mean heart rate `83.5`, and `Overnight readings: (42, 2)` with mean `71.7`. The noon table starts `2024-03-04 12:00:00` with heart rate `84`.

"The first three days" needs a strict `<`: a `.loc` slice includes its end point, so it picks up one reading too many.

```python
end_of_day_3 = icu.index.min() + pd.Timedelta(days=3)
too_many = icu.loc[:end_of_day_3]
first_3_days = icu.loc[icu.index < end_of_day_3]
last_3_days = icu.loc[icu.index > icu.index.max() - pd.Timedelta(days=3)]
print('Slice:', too_many.shape, 'ends', too_many.index[-1])
print('Strict <:', first_3_days.shape, 'ends', first_3_days.index[-1])
print('Last 3 days:', last_3_days.shape, 'starts', last_3_days.index[0])
```

**Expect:** `Slice: (73, 2) ends 2024-03-07 00:00:00`, `Strict <: (72, 2) ends 2024-03-06 23:00:00`, and `Last 3 days: (72, 2) starts 2024-03-08 00:00:00`.
