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

# Demo 1: Audit Two Years of Emergency Department Timestamps

A health system's four emergency departments (EDs) send a two-year visit extract for a staffing review. Every timestamp arrives as text. Parse and audit the timestamps, measure length of stay, find when patients arrive, select calendar weeks, check that every site reported every day, and build the monthly 4-hour report. All visits and patients are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

## Core walkthrough

### 1. Load the extract

This supplied cell downloads `generate_data.py` when the folder lacks it (`urlretrieve(url, path)` saves the file at a web address to `path`), runs its `build()` when a table in `data/` is missing, and reads the ED table. `build()` writes all seven Lecture 09 tables in about 20 seconds, once. It uses a fixed seed, so your values match the ones below.

```python
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd

REPO_RAW = 'https://raw.githubusercontent.com/christopherseaman/datasci_217/main/09/demo'
if not Path('generate_data.py').exists():
    urlretrieve(f'{REPO_RAW}/generate_data.py', 'generate_data.py')
if not Path('data/ed_visits.parquet').exists():
    import generate_data
    generate_data.build()

ed = pd.read_parquet('data/ed_visits.parquet')
print(ed.shape)
display(ed.head(3))
display(ed.dtypes)
```

Expect `(627505, 9)`: one row per visit at the Central, Northside, Eastside, and Harbor EDs from January 2023 through December 2024. `arrival_time` and `departure_time` are `str`, so they cannot be subtracted or sorted by time yet. `esi` is the Emergency Severity Index triage level, 1 (most urgent) to 5, and `arrival_mode` is `ambulance` or `walk-in`.

### 2. Set the report window with `datetime`

The staffing committee meets on January 7, 2025 and wants the last 28 days before the meeting. One date is a job for Python's `datetime`.

```python
meeting = datetime.strptime('January 7, 2025', '%B %d, %Y')
window_start = meeting - timedelta(days=28)
print(meeting)
print('Window starts', window_start.strftime('%A, %B %d, %Y'))
```

Expect `2025-01-07 00:00:00` and `Window starts Tuesday, December 10, 2024`. Step 6 uses both dates to select rows.

### 3. Parse the timestamps and audit what fails

`format=` states the pattern the extract promises, and `errors='coerce'` turns any text that does not fit into `NaT` instead of stopping.

```python
ed['arrival'] = pd.to_datetime(ed['arrival_time'], format='%Y-%m-%d %H:%M', errors='coerce')
ed['departure'] = pd.to_datetime(ed['departure_time'], format='%Y-%m-%d %H:%M', errors='coerce')
print('Arrival text that failed to parse:', ed['arrival'].isna().sum())
print('Departure text missing:', ed['departure_time'].isna().sum(), ' Departure NaT:', ed['departure'].isna().sum())
display(ed.loc[ed['arrival'].isna(), 'arrival_time'].value_counts())
display(ed.groupby('disposition')['departure_time'].apply(lambda s: s.isna().mean()).round(3))
```

Expect `150` arrivals that failed and `7701` departures, all of them already missing in the text. The failures come in five kinds of 30 rows each: an empty string, `UNKNOWN`, two dates that do not exist (`2023-02-29`, a non-leap year, and `2024-06-31`), and the hour `24:10`. A plain `pd.to_datetime()` without `errors='coerce'` would stop at the first one. The missing departures are not spread evenly: `0.462` of patients who left without being seen have none, against `0.003` of discharged patients, because nobody checks out a patient who walked out. A length-of-stay summary that drops missing departures under-counts that group.

### 4. Length of stay and the 4-hour mark

Subtracting two datetime columns gives a `timedelta64` column; `.dt.total_seconds() / 3600` turns it into hours. A departure before its arrival is a data-entry error, so count those before any summary.

```python
ed['los_hours'] = (ed['departure'] - ed['arrival']).dt.total_seconds() / 3600
print('Departure before arrival:', (ed['los_hours'] < 0).sum())
display(ed.loc[ed['los_hours'] < 0, ['arrival_time', 'departure_time', 'los_hours']].head(3).round(1))

valid = ed[ed['los_hours'] >= 0].copy()
valid['over_4h'] = valid['los_hours'] > 4
display(valid.groupby('disposition').agg(visits=('los_hours', 'size'),
                                         median_hours=('los_hours', 'median'),
                                         share_over_4h=('over_4h', 'mean')).round(2))
```

Expect `75` departures before arrival, from `-18.9` to `-22.1` hours in the first three: each departure was typed with the previous day's date. Admitted patients have the longest median stay (`6.88` hours, `0.89` of them over 4 hours), because they wait in the ED for an inpatient bed. Discharged patients have a median of `2.83` hours, with `0.26` over 4 hours.

### 5. When do patients arrive?

Every visit with a parsed arrival counts here, including the ones with no departure. `.dt.hour` reads the clock hour from each arrival, and `.dt.floor('D')` rounds each arrival down to its date. Dividing the visits per hour by the number of days gives the average arrivals in that hour of a day.

```python
arrived = ed.dropna(subset=['arrival'])
n_days = arrived['arrival'].dt.floor('D').nunique()
per_hour = arrived.groupby(arrived['arrival'].dt.hour).size() / n_days
print('Days:', n_days)
print('Busiest hour:', per_hour.idxmax(), round(per_hour.max(), 1), ' Quietest hour:', per_hour.idxmin(), round(per_hour.min(), 1))

fig, ax = plt.subplots(figsize=(9, 3.5))
per_hour.plot(kind='bar', ax=ax, rot=0, color='steelblue')
ax.set(title='Average ED arrivals by hour of day, all four sites, 2023 to 2024',
       xlabel='Hour of arrival', ylabel='Arrivals per day in that hour')
plt.show()
```

Expect `Days: 731` and `Busiest hour: 11 54.0  Quietest hour: 4 11.1`: about five times as many patients arrive late in the morning as before dawn. The bars climb from 06:00 to 11:00, stay high through 19:00, and fall overnight.

The night shift sees a different mix of patients. Compare the share who arrive by ambulance and the share triaged ESI 1 or 2 by hour group.

```python
night = (arrived['arrival'].dt.hour >= 23) | (arrived['arrival'].dt.hour < 7)
display(arrived.groupby(night.map({True: 'night 23:00-06:59', False: 'day 07:00-22:59'})).agg(
    visits=('visit_id', 'size'),
    ambulance_share=('arrival_mode', lambda s: (s == 'ambulance').mean()),
    esi_1_or_2=('esi', lambda s: (s <= 2).mean())).round(3))
```

Expect `533054` day arrivals and `94301` at night. At night `0.202` arrive by ambulance and `0.198` are triaged ESI 1 or 2, against `0.142` and `0.139` by day: fewer people come in overnight, and more of those who do are very sick.

### 6. Select weeks by date

A `DatetimeIndex` lets `.loc` select by calendar. This intentional error shows why the index must be sorted first: the extract is ordered by site, so the arrivals jump back to January 2023 at each new site.

```python
by_time = arrived.set_index('arrival')
print('Sorted?', by_time.index.is_monotonic_increasing)
try:
    by_time.loc['2024-01-08':'2024-01-14']
except KeyError as error:
    print('KeyError:', error)
```

Expect `Sorted? False`, then `KeyError: 'Value based partial slicing on non-monotonic DatetimeIndexes with non-existing keys is not allowed.'`. The fix is `sort_index()`.

```python
by_time = by_time.sort_index()
print('Sorted?', by_time.index.is_monotonic_increasing)
print('January 2024 visits:', len(by_time.loc['2024-01']))

weeks = pd.DataFrame({
    'flu_week': by_time.loc['2024-01-01':'2024-01-07'].groupby('site').size(),
    'summer_week': by_time.loc['2024-07-08':'2024-07-14'].groupby('site').size(),
    'report_window': by_time.loc[window_start:meeting].groupby('site').size(),
})
display(weeks)
```

Expect `Sorted? True` and `January 2024 visits: 30495`. Every site saw more visits in the flu-peak week of January 1 to 7, 2024 than in July 8 to 14 (Central `2898` against `2476`, Eastside `1556` against `1255`). Both slice endpoints are included, so each week holds 7 full days. The `datetime` values from step 2 work as slice endpoints too: the report window runs from December 10 to the end of the data on December 31 (Central `7545` visits).

### 7. Check that every site reported every day

A missing day can mean a quiet day or a broken data feed. `resample('D').count()` counts each site's visits per calendar day, and a day with no rows gets `0`.

```python
all_days = pd.date_range('2023-01-01', '2024-12-31', freq='D')
print(len(all_days), 'days in the extract')
for site in ['Central', 'Northside', 'Eastside', 'Harbor']:
    site_daily = by_time.loc[by_time['site'] == site, 'visit_id'].resample('D').count()
    empty = site_daily[site_daily == 0]
    print(site, len(site_daily), 'days,', len(empty), 'with no visits', list(empty.index.strftime('%Y-%m-%d')))
```

Expect `731 days in the extract`, and `731` daily rows for every site. Central, Northside, and Harbor have `0` days with no visits. Eastside has `9`, August 12 to 20, 2024, in a row: an ED that sees about 170 patients a day did not go quiet for nine days, so its feed failed. A monthly count for Eastside in August 2024 would be low because of the feed, not the patients.

### 8. The monthly 4-hour report

The committee tracks the share of visits that last over 4 hours, one row per month. `valid` from step 4 has `over_4h` as `True` or `False`, so a monthly mean is that share. Index it by arrival time first.

```python
valid_by_time = valid.set_index('arrival').sort_index()
monthly_over_4h = valid_by_time['over_4h'].resample('ME').mean()
display(monthly_over_4h.sort_values(ascending=False).head(3).round(3))
display(monthly_over_4h.sort_values().head(3).round(3))

fig, ax = plt.subplots(figsize=(9, 3.5))
monthly_over_4h.plot(ax=ax, marker='o', color='steelblue')
ax.set(title='Share of ED visits over 4 hours, by month of arrival', xlabel='Month', ylabel='Share over 4 hours')
plt.show()
```

Expect the three worst months to be `2023-12-31` (`0.454`), `2024-01-31` (`0.450`), and `2023-01-31` (`0.416`), and the best near `0.38` in spring and late autumn (`2024-11-30`, `0.380`). The line sits near 0.39 most of the year and jumps in December 2023 and January 2024, the flu peak.

The labels are month ends: each row covers the month that ends on that date. Split by disposition to see who drives the winter rise.

```python
by_disposition = pd.DataFrame({
    disposition: group['over_4h'].resample('ME').mean()
    for disposition, group in valid_by_time.groupby('disposition')
})
display(by_disposition.loc['2023-11':'2024-02', ['admitted', 'discharged']].round(3))
```

Expect admitted patients to rise from `0.888` over 4 hours in November 2023 to `0.961` in January 2024, and discharged patients from `0.260` to `0.301`. Every patient waits longer when the ED is crowded, and admitted patients also wait for an inpatient bed, which fills in flu season.

## Independent practice

Continue on your own after class. These cells reuse `arrived`, `valid`, `by_time`, and `valid_by_time` from the core; if the runtime closed, run the cells above again first.

### 1. Find the busiest single hour and name it

`.dt.floor('h')` rounds each arrival down to its hour, so `value_counts()` counts every clock hour of the two years. `.dt.strftime()` writes each hour as a short key, such as a file name for that hour's staffing note.

```python
busiest = arrived['arrival'].dt.floor('h').value_counts().head(3)
display(busiest)
print(busiest.index.strftime('%Y%m%d%H').tolist())
```

Expect two clock hours during the January 2024 flu peak, `2024-01-07 12:00` (`90` arrivals) and `2024-01-08 12:00` (`87`), then `2024-09-23 16:00` (`85`). Their keys print as `['2024010712', '2024010812', '2024092316']` (year, month, day, hour).

### 2. Weekly volume, labeled by week end

`resample('W')` bins by calendar week; each bin ends on a Sunday and carries that Sunday's date as its label.

```python
weekly = by_time['visit_id'].resample('W').count()
print(weekly.index[:2].strftime('%a %Y-%m-%d').tolist())
display(weekly.sort_values(ascending=False).head(3))
weekend = by_time.index.dayofweek >= 5
print('Mean visits per day:', round(weekly.sum() / 731), ' Weekend share of visits:', round(weekend.mean(), 3))
```

Expect the first two labels, `Sun 2023-01-01` and `Sun 2023-01-08`: the first bin holds only January 1, a Sunday. The three busiest weeks end `2024-01-07` (`7454` visits), `2023-12-31` (`7367`), and `2024-01-14` (`7187`), all at the flu peak. `Mean visits per day: 858`, and weekends hold `0.272` of visits, a little less than their 2 days in 7 (`0.286`).

### 3. Bin edges for a triage block report

The charge nurse counts arrivals in 4-hour blocks and reports each block when it ends. An arrival at exactly 08:00 belongs to the block that just ended, `04:00 to 08:00`. That is `closed='right', label='right'`. Compare it with the default on one day at Central.

```python
central_day = by_time.loc['2024-03-07']
central_day = central_day[central_day['site'] == 'Central']['visit_id']
print('Arrivals at exactly 08:00:', (central_day.index == '2024-03-07 08:00').sum())
display(central_day.resample('4h').count())
display(central_day.resample('4h', closed='right', label='right').count())
```

Expect `Arrivals at exactly 08:00: 1`, then two tables with different counts under the same labels. By default, bins include their start and carry it as the label: `08:00` means 08:00 to 11:59 and holds `84`. With `closed='right', label='right'`, `08:00` means after 04:00 up to and including 08:00 and holds `26`: the 25 arrivals from 04:01 to 07:59 plus the one at 08:00. The last right-closed bin is labeled `2024-03-08 00:00`, the end of March 7. State the rule whenever a report depends on it.

### 4. Series and DataFrame shortcuts

`weekly` is a Series with a DatetimeIndex, so `weekly['2024-03']` selects the weeks ending in March 2024 directly. On a DataFrame, square brackets look for a column; this intentional error shows the `KeyError`.

```python
print(weekly['2024-03'].shape)
try:
    by_time['2024-03']
except KeyError as error:
    print('KeyError:', error)
print(by_time.loc['2024-03'].shape)
```

Expect `(5,)`: the five weeks ending on a Sunday in March 2024. Then `KeyError: '2024-03'`, then the fix with `.loc`: `(25665, 11)`, every March 2024 visit.

### 5. Follow-up schedules for frequent visitors

Patients with 4 or more visits in 2024 are enrolled in a care-management program: phone calls every Monday for four weeks from January 6, 2025, a nurse visit on the first of each month for a quarter, and a report at each month's end.

```python
visits_2024 = by_time.loc['2024', 'patient_id'].value_counts()
print('Patients:', len(visits_2024), ' Frequent visitors:', (visits_2024 >= 4).sum(), ' Their share of 2024 visits:',
      round(visits_2024[visits_2024 >= 4].sum() / visits_2024.sum(), 3))
display(pd.date_range('2025-01-06', periods=4, freq='W-MON'))
display(pd.date_range('2025-01-01', periods=3, freq='MS'))
display(pd.date_range('2025-01-01', periods=3, freq='ME'))
```

Expect `Patients: 192210  Frequent visitors: 12627` and a share of `0.208`: about 7% of 2024's patients make a fifth of its visits. Then the four Mondays January 6 to 27, the month starts January 1, February 1, and March 1, and the month ends January 31, February 28, and March 31.

The older alias `'M'` for month end now fails. This intentional error shows the message.

```python
try:
    pd.date_range('2025-01-01', periods=3, freq='M')
except ValueError as error:
    print('ValueError:', error)
```

Expect `ValueError: Invalid frequency: M. Failed to parse with error message: ValueError("'M' is no longer supported for offsets. Please use 'ME' instead.")`.

### 6. Monthly means of several columns

`df.resample('ME').mean()` summarizes every column. A text column such as `site` raises `TypeError`, so select the numeric columns first. This intentional error shows the message.

```python
try:
    valid_by_time[['site', 'los_hours']].resample('ME').mean()
except TypeError as error:
    print('TypeError:', str(error)[:60])
display(valid_by_time[['los_hours', 'age']].resample('ME').mean().loc['2024-01':'2024-03'].round(2))
```

Expect `TypeError: dtype 'str' does not support operation 'mean'`, then three month-end rows. January 2024 has the longest mean stay (`4.99` hours, against `4.21` in February and `4.08` in March); the mean age barely moves (`46.66` to `45.39`).

## Optional extension: overnight arrivals and frequency checks

Prerequisite: BONUS.md, Time-of-Day Selection Example, and BONUS.md, Frequency Inference and Specialized Schedules. Nothing in the core route or the assignment needs this. These cells use `by_time` and `arrived` from the core.

`between_time()` keeps a clock range on every day; a range that wraps past midnight, such as 23:00 to 06:59, works too. The night shift wants its share of each site's arrivals.

```python
overnight = by_time.between_time('23:00', '06:59')
display((overnight.groupby('site').size() / by_time.groupby('site').size()).round(3))
```

Expect every site between `0.149` and `0.153`: about one arrival in seven comes during the 8 night hours, a third of the day.

`pd.infer_freq()` reads the spacing back from an index. Daily counts with a missing day have no single frequency.

```python
for site in ['Central', 'Eastside']:
    site_daily = arrived[arrived['site'] == site].groupby(arrived['arrival'].dt.floor('D')).size()
    print(site, pd.infer_freq(site_daily.index))
```

Expect `Central D` and `Eastside None`: the nine missing August days break the daily spacing, the same outage step 7 found.
