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

# Demo 1: Parse, Index, Shift, and Select Patient Readings

A heart-failure clinic follows one patient's daily home weights, and an ICU monitor records another patient's vitals every hour. You parse text timestamps into a DatetimeIndex, build follow-up schedules, flag a rapid weight gain with lags, select rows by calendar date and by clock time, and resample into daily and two-hour bins. Everything here comes from Lecture 09 up to the first demo break, plus Lectures 01 to 08. Patient values are synthetic.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-25 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, and matplotlib 3.11.1; the whole notebook runs in a few seconds.

## Setup

The first cell installs pandas 3.0.5, the course version. Colab ships an older pandas (2.2). A `.venv` made with `uv venv --seed` includes pip, so the same `%pip` cell works locally too.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.` Locally, with the requirements already installed, the cell changes nothing and you can go on. In Colab, pip may also print a dependency conflict because some preinstalled packages expect pandas 2.2; that is expected, and this demo does not use them. If Colab asks you to restart the session (or says pandas was previously imported), choose **Runtime → Restart session**, then continue with the next cell. You do not need to rerun the install.

```python
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print('pandas', pd.__version__)
print('NumPy', np.__version__)
```

**Expect:** `pandas 3.0.5` and `NumPy 2.3.3` (Colab may show a different NumPy; that is fine).

## 1. One date at a time with `datetime`

A discharge time arrives as text. `strptime()` parses it, `strftime()` formats it for a letter, and a `timedelta` moves it forward to the follow-up contacts.

```python
discharge = datetime.strptime('2024-03-01 14:30', '%Y-%m-%d %H:%M')
print(discharge)
print(discharge.strftime('%B %d, %Y at %I:%M %p'))

# Follow-up contacts after discharge
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

The patient's home scale app exports its log as text, newest first in places, and one entry has an impossible date: February 30.

```python
raw = pd.DataFrame({
    'recorded_at': ['2024-03-03 07:10', '2024-03-01 07:05', '2024-02-30 07:00',
                    '2024-03-02 07:15', '2024-03-04 06:55'],
    'weight_kg': [82.3, 81.9, 82.0, 82.1, 83.4],
})
print(raw.dtypes)
```

**Expect:** `recorded_at` is `str` and `weight_kg` is `float64`. Text cannot be subtracted or sorted by time.

`format=` states the pattern the export promises, and `errors='coerce'` turns anything that does not fit into `NaT` instead of stopping.

```python
raw['recorded_at'] = pd.to_datetime(raw['recorded_at'], format='%Y-%m-%d %H:%M', errors='coerce')
print(raw)
print('Unparseable dates:', raw['recorded_at'].isna().sum())
```

**Expect:** row 2 shows `NaT` in `recorded_at`, and `Unparseable dates: 1`.

Drop the bad row (Lecture 05), make the timestamps the index, and check the order before and after sorting.

```python
log = raw.dropna(subset=['recorded_at']).set_index('recorded_at')
print('Sorted?', log.index.is_monotonic_increasing)

log = log.sort_index()
print('Sorted?', log.index.is_monotonic_increasing)
print(log)
print(log.index.day_name())
```

**Expect:** `Sorted? False`, then `Sorted? True`, and four rows from `2024-03-01 07:05:00` (81.9 kg) to `2024-03-04 06:55:00` (83.4 kg). The day names run `Friday`, `Saturday`, `Sunday`, `Monday`.

## 3. Follow-up schedules and frequencies

`pd.date_range()` builds a schedule from a frequency alias, and `pd.infer_freq()` reads the spacing back from an index.

```python
weekly_calls = pd.date_range('2024-03-04', periods=4, freq='W-MON')   # Monday phone check-ins
monthly_labs = pd.date_range('2024-03-01', periods=3, freq='MS')      # first-of-month blood tests
clinic_days = pd.bdate_range('2024-03-01', '2024-03-14')              # weekdays the clinic is open
print(weekly_calls)
print(monthly_labs)
print('Clinic days:', len(clinic_days))

print('Weekly calls:', pd.infer_freq(weekly_calls))
print('Home weigh-ins:', pd.infer_freq(log.index))
```

**Expect:** Mondays `2024-03-04`, `03-11`, `03-18`, `03-25`; lab dates `2024-03-01`, `04-01`, `05-01`; `Clinic days: 10` (two weekends skipped). `infer_freq` returns `W-MON` for the calls and `None` for the weigh-ins: the patient steps on the scale at a slightly different minute each morning, so the log has no single frequency.

## 4. Daily weights: lags and a fluid-gain alert

Heart-failure patients are commonly told to call the clinic if their weight rises by more than about 1 kg in a day or 2 kg in a week, because fluid is building up. Here is one patient's daily weight for January through March, with a fluid-gain episode in March. `rng` makes the same "random" values on every run (Lecture 03), so your numbers match the ones below.

```python
rng = np.random.default_rng(42)
days = pd.date_range('2024-01-01', '2024-03-31', freq='D')
scale_noise = rng.normal(0, 0.15, len(days))  # mean 0, standard deviation 0.15 kg
weights = pd.Series(82 + scale_noise, index=days)

# Fluid builds up from March 9, then a diuretic brings it back down
weights.loc['2024-03-09':'2024-03-15'] += [0.5, 1.9, 2.6, 2.6, 1.9, 1.1, 0.4]

daily = pd.DataFrame({'weight_kg': weights.round(1)})
print(daily.shape)
print('Frequency:', pd.infer_freq(daily.index))
```

**Expect:** `(91, 1)` (31 + 29 + 31 days) and `Frequency: D`.

`shift(1)` brings each row the previous day's value (a lag), `shift(-1)` the next day's (a lead), and `diff()` subtracts the lag. Because the rows are consecutive days, `shift(7)` is the weight one week earlier.

```python
daily['prev_day'] = daily['weight_kg'].shift(1)
daily['next_day'] = daily['weight_kg'].shift(-1)
daily['change_1d'] = daily['weight_kg'].diff()
daily['change_7d'] = daily['weight_kg'] - daily['weight_kg'].shift(7)
daily['pct_1d'] = daily['weight_kg'].pct_change() * 100
print(daily.head(3).round(2))
print(daily.loc['2024-03-08':'2024-03-13'].round(2))
```

**Expect:** on January 1, `prev_day`, `change_1d`, and `pct_1d` are `NaN` (nothing came before), and `change_7d` stays `NaN` through January 7. In March, the weight climbs from 81.9 kg on March 8 to 83.9 kg on March 10 (`change_1d` 1.3) and 84.4 kg on March 11 and 12 (`change_7d` 2.3).

Apply both alert rules with a boolean filter (Lecture 04).

```python
alerts = daily[(daily['change_1d'] > 1.0) | (daily['change_7d'] > 2.0)]
print(alerts[['weight_kg', 'change_1d', 'change_7d']].round(1))
print('Largest daily change before March 9:', daily.loc[:'2024-03-08', 'change_1d'].abs().max().round(1), 'kg')
```

**Expect:** three alert days: March 10 (the 1-day rule, `1.3` kg) and March 11 and 12 (the 7-day rule, `2.3` kg). Before the episode, no day moved more than `0.4` kg, so ordinary scale noise never trips the rule.

## 5. Select by calendar, then summarize by week and month

Partial dates select whole months, and a date slice keeps both ends. `resample()` then turns the daily rows into weekly and monthly summaries.

```python
print(daily.loc['2024-02'].shape)
print(daily.loc['2024-03-08':'2024-03-16', 'weight_kg'].tolist())

print(daily['weight_kg'].resample('W').mean().head(3).round(2))
print(daily['weight_kg'].resample('ME').agg(['mean', 'max']).round(2))
```

**Expect:** `(29, 6)` for February, then nine March weights from `81.9` to `82.1`. Weekly means are labeled with the Sunday that ends each week: `2024-01-07`, `2024-01-14`, `2024-01-21`. The monthly table has three rows; March's `max` is `84.4`, against `82.3` in January and `82.2` in February.

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily.index, daily['weight_kg'], color='gray', label='Daily weight')
ax.plot(alerts.index, alerts['weight_kg'], 'o', color='red', label='Alert day')  # markers only, no line
ax.set(title='Home weights with fluid-gain alerts', xlabel='Date', ylabel='Weight (kg)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Expect:** a flat gray line near 82 kg with a sharp hump in mid-March, and three red points at the top of the hump (March 10 to 12).

## 6. Hourly ICU vitals: select by clock time

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
print(icu.head(3))
```

**Expect:** `(168, 2)` and three rows starting at `2024-03-04 00:00:00`.

`between_time()` keeps a clock range on every day, and `at_time()` keeps one clock time.

```python
day_shift = icu.between_time('09:00', '17:00')
overnight = icu.between_time('00:00', '05:00')
print('Day readings:', day_shift.shape, 'mean HR', round(day_shift['heart_rate'].mean(), 1))
print('Overnight readings:', overnight.shape, 'mean HR', round(overnight['heart_rate'].mean(), 1))
print(icu.at_time('12:00').head(3))
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

## 7. Resample into daily and two-hour bins

Resampling the hourly monitor to days gives one row per day. Select the numeric columns you want before resampling a DataFrame; a text column would raise `TypeError`.

```python
print(icu[['heart_rate', 'spo2']].resample('D').mean().round(1))
print(icu['heart_rate'].resample('D').agg(['mean', 'min', 'max', 'count']).round(1).head(3))
```

**Expect:** seven rows labeled `2024-03-04` through `2024-03-10`, with daily mean heart rate between `79.8` and `81.9` and SpO2 near 95.5. Every `count` is `24`: the monitor never missed an hour.

A nurse charts respiratory rate at irregular times. Two readings sit near the 08:00 bin boundary, one just before it and one exactly on it.

```python
resp_rate = pd.Series(
    [18, 22, 26, 24, 20],
    index=pd.to_datetime(['2024-03-05 06:00', '2024-03-05 07:59', '2024-03-05 08:00',
                          '2024-03-05 09:30', '2024-03-05 10:00']),
)
print(resp_rate.resample('2h').agg(['mean', 'count']))
print(resp_rate.resample('2h', closed='right', label='right').agg(['mean', 'count']))
```

**Expect:** with the default (left-closed, left-labeled) bins, the 08:00 reading opens the `08:00` bin: counts `2`, `2`, `1` for `06:00`, `08:00`, `10:00`. With `closed='right', label='right'`, each bin ends at its label, so 08:00 joins 07:59 in the bin labeled `08:00`: counts `1`, `2`, `2`. Same readings, different bins: state the rule whenever a report depends on it.
