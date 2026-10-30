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

# Demo 2: Is 4 West Watching Its Patients?

Phase 2 of Dr. Reyes's winter program moves from the ED to 4 West, a 24-bed step-down unit where every patient wears bedside telemetry: a monitor writes heart rate and oxygen saturation (SpO2) every 5 minutes.

**This phase's ask.** The unit's nurse manager, Sam Okoye, and the patient-safety officer meet at the end of the month to decide whether to buy portable telemetry units, which keep a patient monitored during imaging and walks. They want, from last winter's monitor archive (January to March 2024): how much of each stay was monitored, when the gaps happen, which stays went unwatched for 3 hours or more, and a review of stay `S0297`, transferred to the ICU on February 9, including which smoother the unit's heart-rate alert should use. All stays and readings are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

## Core walkthrough

### 1. Load the stays and the monitor feed

This supplied cell downloads `generate_data.py` when the folder lacks it (`urlretrieve(url, path)` saves the file at a web address to `path`), runs its `build()` when a table in `data/` is missing (about 20 seconds, once), and reads three tables. The seed is fixed, so your values match the ones below.

```python
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd

REPO_RAW = 'https://raw.githubusercontent.com/christopherseaman/datasci_217/main/09/demo'
TABLES = ['stays', 'telemetry', 'charted_temps', 'ed_visits']
if not Path('generate_data.py').exists():
    urlretrieve(f'{REPO_RAW}/generate_data.py', 'generate_data.py')
if not all(Path(f'data/{name}.parquet').exists() for name in TABLES):
    import generate_data
    generate_data.build()

stays = pd.read_parquet('data/stays.parquet')
telemetry = pd.read_parquet('data/telemetry.parquet')
temps = pd.read_parquet('data/charted_temps.parquet')
print(stays.shape, telemetry.shape, temps.shape)
display(stays.head(3))
display(telemetry.head(3))
display(stays['outcome'].value_counts())
print('Readmitted patients:', stays['patient_id'].duplicated().sum())
```

Expect `(713, 7) (548923, 5) (11499, 3)`: 713 stays, over half a million monitor readings, and the temperatures nurses charted every 4 hours. Most stays end in discharge (`667`); `25` end in an ICU transfer, and `21` were still on the unit when the archive ends on March 31. `55` stays belong to a patient with an earlier stay, so `stay_id`, not `patient_id`, names one continuous monitoring episode.

### 2. What did the monitor record in each hour of each stay?

The safety officer's unit of review is a stay-hour. Resample inside each stay, as in Lecture 08's split-apply-combine: a bin that mixed two patients would describe neither. Named aggregation gives each summary its own column. A full hour holds 12 readings; `source_row` is a column of ones, so its `count` includes readings whose heart rate is missing.

```python
telemetry['source_row'] = 1
hourly = (
    telemetry.set_index('recorded_at')
    .groupby('stay_id')
    .resample('h')
    .agg(mean_hr=('heart_rate', 'mean'),
         median_hr=('heart_rate', 'median'),
         max_hr=('heart_rate', 'max'),
         min_spo2=('spo2', 'min'),
         n_readings=('source_row', 'count'))
    .reset_index()
)
print(hourly.shape)
display(hourly.head(4).round({'mean_hr': 1}))
```

Expect `(49244, 7)`: one row per stay per clock hour, from each stay's first hour to its last. `S0001`'s first hour, `00:00`, has only `7` readings, because the patient arrived at 00:25; its next hours have `12`. Its `02:00` hour has a `min_spo2` of `79.0` while the mean heart rate stays near 67: a probe slipping off a finger reads falsely low before it goes blank.

A first draft grouped by `patient_id`. Count what that does to the readmitted patients.

```python
by_patient = telemetry.set_index('recorded_at').groupby('patient_id')['source_row'].resample('h').count()
print('Hours grouped by stay:', len(hourly), ' by patient:', len(by_patient))
print('Empty hours by stay:', (hourly['n_readings'] == 0).sum(), ' by patient:', (by_patient == 0).sum())
```

Expect `49244` hours by stay against `76456` by patient, and `1230` empty hours against `28470`. The extra 27,240 empty hours are weeks at home between admissions, which the safety review would have counted as monitoring gaps.

### 3. When do patients come off the monitor?

`pd.cut()` (Lecture 05) sorts each stay-hour by its reading count, and `value_counts(normalize=True)` returns shares in place of counts. The clock hour of each empty stay-hour shows when the gaps happen.

```python
coverage = pd.cut(hourly['n_readings'], bins=[-1, 0, 11, 12], labels=['none', 'partial', 'full'])
display(coverage.value_counts(normalize=True).round(3))

empty_hour = hourly.loc[hourly['n_readings'] == 0, 'recorded_at'].dt.hour
fig, ax = plt.subplots(figsize=(9, 3.5))
empty_hour.value_counts().sort_index().plot(kind='bar', ax=ax, rot=0, color='steelblue')
ax.set(title='4 West stay-hours with no telemetry reading, by clock hour', xlabel='Hour of day', ylabel='Empty stay-hours')
plt.show()
print('Empty hours starting 09:00 to 20:00:', round(empty_hour.between(9, 20).mean(), 3))
```

Expect `0.868` of stay-hours `full` (all 12 readings), `0.107` `partial` (a short break, or the first or last hour of a stay), and `0.025` `none`: 1,230 whole hours with no reading. The bars are low overnight (`9` empty hours at 04:00 and 05:00) and jump at 09:00 to between `58` and `92` an hour until 20:00. `0.773` of empty hours start in that span, when patients go to imaging, walk the hall, or shower.

**What we tell Sam:** 1 stay-hour in 40 has no reading, and three in four of those fall in the daytime, when patients leave the bed. Portable units target exactly those hours.

### 4. Which stays went 3 hours or more without a reading?

The safety officer reviews every stay with a long gap. A **gap run** is a stretch of consecutive empty hours. `observed` is `True` for an hour with at least one reading; its running count per stay (`cumsum()`) stays flat through a gap, so each gap's hours share a `run_id`.

```python
hourly['observed'] = hourly['n_readings'] > 0
hourly['run_id'] = hourly.groupby('stay_id')['observed'].cumsum()
runs = (hourly[~hourly['observed']]
        .groupby(['stay_id', 'run_id'])
        .size()
        .rename('gap_hours')
        .reset_index())
print(len(runs), 'gap runs')

fig, ax = plt.subplots(figsize=(8, 3.5))
runs['gap_hours'].value_counts().sort_index().plot(kind='bar', ax=ax, rot=0, color='steelblue')
ax.set(title='Length of each telemetry gap run', xlabel='Consecutive empty hours', ylabel='Gap runs')
plt.show()
```

Expect `606 gap runs`. The tallest bar is one empty hour (`364` runs), but `130` runs last 3 to 8 hours, long enough to miss a deterioration, and one lasts `11`.

A stay with no gaps has no rows in `runs`. A left merge (Lecture 06) onto every stay keeps those stays, and `fillna(0)` records their zero runs, so the review list counts every stay.

```python
per_stay = runs.groupby('stay_id')['gap_hours'].agg(n_gaps='count', longest_gap='max').reset_index()
audit = stays[['stay_id', 'outcome']].merge(per_stay, on='stay_id', how='left', validate='one_to_one')
audit = audit.fillna({'n_gaps': 0, 'longest_gap': 0})
print('Stays with no empty hour:', (audit['n_gaps'] == 0).sum(), 'of', len(audit))
print('Stays with a gap of 3 hours or more:', (audit['longest_gap'] >= 3).sum())
display(audit.sort_values(['longest_gap', 'n_gaps'], ascending=False).head(3))
display(audit.groupby('outcome')['longest_gap'].agg(stays='size', share_3h_or_more=lambda s: (s >= 3).mean()).round(3))
```

Expect `Stays with no empty hour: 342 of 713` and `111` stays with a gap of 3 hours or more, the safety officer's review list. The top row is `S0299`, with one gap of `11.0` hours. `0.280` of ICU transfers had a 3-hour gap (7 of 25), against `0.151` of discharged stays (101 of 667).

**What we tell Sam:** 111 stays, 1 in 6, had a gap of 3 hours or more. Transfers look worse, but with only 25 of them, two or three stays move that share by about 10 points, so one quarter cannot show that the monitor misses sicker patients more often.

### 5. The temperature grid will not build

The review wants nurse-charted temperatures beside the hourly monitor table, which means laying every stay's 4-hourly temperatures on an hourly grid with `asfreq()`. This intentional error shows the first attempt.

```python
try:
    temps.set_index('charted_at').groupby('stay_id')['temperature_f'].resample('h').asfreq()
except ValueError as error:
    print('ValueError:', error)
```

Expect `ValueError: cannot reindex on an axis with duplicate labels`: some stay has two rows at the same time, and `asfreq()` can put only one value in each slot. Count the repeated keys and look at a few.

```python
repeated = temps.duplicated(['stay_id', 'charted_at'], keep=False)
print('Repeated rows:', temps.duplicated(['stay_id', 'charted_at']).sum(), ' stays:', temps.loc[repeated, 'stay_id'].nunique())
display(temps[repeated].head(6))
print('Share of repeated rounds with a reading of 99.5 or more:',
      round((temps[repeated].groupby(['stay_id', 'charted_at'])['temperature_f'].max() >= 99.5).mean(), 2))
```

Expect `Repeated rows: 254  stays: 122`. The pairs match or differ by a few tenths, such as `S0012`'s `101.2` and `100.9` at midnight on January 2, and `0.86` of the repeated rounds include a reading of 99.5 °F or more: a nurse who sees a fever rechecks it and charts the recheck under the same round. Keep one row per stay and round: the mean of the reading and its recheck.

```python
temps = temps.groupby(['stay_id', 'charted_at'], as_index=False)['temperature_f'].mean()
print(temps.shape, temps.duplicated(['stay_id', 'charted_at']).sum())
```

Expect `(11245, 3) 0`. Now upsample `S0297`'s temperatures to an hourly grid three ways: `asfreq()` leaves the new hours empty, `ffill(limit=2)` carries each round forward for two hours, and `interpolate()` draws a straight line between rounds.

```python
one_temp = temps[temps['stay_id'] == 'S0297'].set_index('charted_at')['temperature_f']
filled = pd.DataFrame({
    'asfreq': one_temp.resample('h').asfreq(),
    'ffill_2': one_temp.resample('h').ffill(limit=2),
    'interpolate': one_temp.resample('h').interpolate(),
})
display(filled.tail(9))

last_day_temp = filled.loc['2024-02-09']
fig, ax = plt.subplots(figsize=(9, 3.5))
last_day_temp['interpolate'].plot(ax=ax, color='#0072B2', label='interpolate()')
last_day_temp['ffill_2'].plot(ax=ax, drawstyle='steps-post', color='#D55E00', label='ffill(limit=2)')
last_day_temp['asfreq'].plot(ax=ax, style='o', color='black', label='Charted (asfreq)')
ax.set(title='Stay S0297: charted temperature on an hourly grid, February 9', xlabel='Time (unit clock)', ylabel='Temperature (°F)')
ax.legend(loc='upper left')
plt.show()
```

Expect the table to end with `S0297`'s last three rounds, `98.3`, `98.4`, and `99.0` °F at 12:00, 16:00, and 20:00. `asfreq` is `NaN` between rounds. `ffill_2` repeats each round for two hours, and the hour before the next round stays `NaN`, so the orange steps break there. `interpolate` climbs from 16:00 and reaches `98.850` at 19:00, a value that needs the 20:00 reading, which no alert running at 19:00 could have seen. Only the black dots were measured.

### 6. Rapid-response review: which smoother should the alert use?

`S0297` went to the ICU at 23:45 on February 9, after more than six hours off the monitor that afternoon. The unit's alert smooths heart rate before comparing it with 120 bpm. Compare a 12-reading count window (one hour when nothing is missing) with a 1-hour time window around the patient's return.

```python
s0297 = telemetry[telemetry['stay_id'] == 'S0297'].set_index('recorded_at')['heart_rate'].dropna()
windows = pd.DataFrame({
    'heart_rate': s0297,
    'last_12_readings': s0297.rolling(12).mean(),
    'last_1_hour': s0297.rolling('1h').mean(),
})
display(windows.loc['2024-02-09 13:00':'2024-02-09 20:30'].round(1))
```

Expect the jump from `13:15` straight to `20:10`. At 20:10, `last_12_readings` is `97.6`, still mostly readings from before the gap, while `last_1_hour` is `120.0`, the 20:10 reading alone. By 20:30 the count window has climbed only to `108.0`; the time window reads `124.0`. A count window follows rows; a time window follows the clock.

Plot the last 24 hours for the review slide: the raw readings, a 1-hour rolling mean with a band of plus or minus one rolling standard deviation, an exponentially weighted mean (EWM) with `span=12`, and a dotted line at the 120 bpm alert. Every reading sits on a 5-minute mark, so `asfreq('5min')` adds `NaN` rows across the gap and the lines break there instead of bridging it.

```python
last_day = s0297.loc['2024-02-08 23:45':]
mean_1h = last_day.rolling('1h').mean().asfreq('5min')
sd_1h = last_day.rolling('1h').std().asfreq('5min')
ewm_12 = last_day.ewm(span=12).mean().asfreq('5min')

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(last_day.index, last_day, '.', color='gray', markersize=3, label='Reading (every 5 min)')
ax.plot(mean_1h.index, mean_1h, color='#0072B2', label='1-hour rolling mean')
ax.fill_between(mean_1h.index, mean_1h - sd_1h, mean_1h + sd_1h, color='#0072B2', alpha=0.2, label='± 1 SD')
ax.plot(ewm_12.index, ewm_12, color='#D55E00', linestyle='--', label='EWM, span 12')
ax.axhline(120, color='black', linestyle=':', label='Alert (120 bpm)')
ax.set(title='Stay S0297: heart rate in the 24 hours before ICU transfer',
       xlabel='Time (unit clock)', ylabel='Heart rate (bpm)')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.show()
print('First hour mean:', round(last_day.iloc[:12].mean(), 1), ' Last hour mean:', round(last_day.iloc[-12:].mean(), 1))
```

Expect `First hour mean: 88.3  Last hour mean: 123.1`. The band drifts from about 80 bpm overnight to about 95 by 13:00. The gray dots stop after 13:15 and resume at 20:10, and both lines break across that gap. After the gap, the 1-hour mean starts near 120, while the dashed EWM starts near 100 and takes under an hour to catch up: like a count window, EWM weights rows, so it still carries the readings from before the gap. The 1-hour mean stays above the alert line for about two hours, dips to about 115, and is back at the line before the transfer.

**What we tell Sam:** after a patient returns to the monitor, a 1-hour time window reaches the alert level at once, while a 12-reading window or an EWM lags by up to an hour. The alert should use the time window, and the nearly 7-hour gap before this transfer is the case for portable units.

## Independent practice

These are follow-up requests from the unit and the program; continue on your own after class. The cells reuse `stays`, `telemetry`, `temps` (one row per round after step 5), `hourly`, and `s0297` from the core; if the runtime closed, run the cells above again first.

### 1. The unit director wants a monthly unit report

A unit-level count is fine to resample across patients: one admission is one event, whoever it is. Index the stays by admission time, then summarize each month with a different aggregation per column.

```python
stays['los_days'] = (stays['left_at'] - stays['admitted_at']).dt.total_seconds() / 86400
stays['icu_transfer'] = stays['outcome'] == 'ICU transfer'
monthly = stays.set_index('admitted_at').resample('ME').agg(
    {'stay_id': 'count', 'los_days': ['mean', 'max'], 'icu_transfer': 'sum', 'age': 'median'})
display(monthly.round(1))
```

Expect three month-end rows, `2024-01-31` to `2024-03-31`, with `250`, `225`, and `238` admissions, mean stays of `3.0`, `2.8`, and `2.8` days, `9`, `5`, and `11` ICU transfers, and a median age of `67` or `68`. `icu_transfer` is `True` or `False`, so its `sum` counts transfers.

### 2. The morning huddle list is too long

The huddle lists any stay whose heart rate topped 120 bpm the day before, and the charge nurse says the list names half the unit. A single 5-minute reading can be a motion artifact: when the patient moves, the monitor can count the jolts as beats. Compare the daily maximum of the raw readings with the daily maximum of the hourly medians from step 2, and count how many ICU transfers each rule would have listed.

```python
raw_daily = telemetry.set_index('recorded_at').groupby('stay_id')['heart_rate'].resample('D').max()
hour_daily = hourly.set_index('recorded_at').groupby('stay_id')['median_hr'].resample('D').max()
outcome = stays.set_index('stay_id')['outcome']
for name, daily in [('raw reading', raw_daily), ('hourly median', hour_daily)]:
    flagged = daily[daily > 120].reset_index()['stay_id'].unique()
    print(f'{name:>13}: {len(flagged)} stays flagged,',
          (outcome[flagged] == 'ICU transfer').sum(), 'of', (outcome == 'ICU transfer').sum(), 'ICU transfers')
both = pd.DataFrame({'raw_max': raw_daily, 'hourly_median_max': hour_daily})
one_reading = both[(both['raw_max'] > 120) & (both['hourly_median_max'] <= 120)]
print('Stay-days flagged by a reading but by no hour:', len(one_reading))
stay, day = one_reading.index[0]
that_day = telemetry[(telemetry['stay_id'] == stay) & (telemetry['recorded_at'].dt.floor('D') == day)]
display(that_day.sort_values('heart_rate', ascending=False).head(4))
```

Expect `raw reading: 364 stays flagged, 21 of 25 ICU transfers` and `hourly median: 19 stays flagged, 12 of 25 ICU transfers`. `495` stay-days are flagged by a reading but by no hour. The example, `S0002` on January 1, has one reading of `142.0` at 23:10 and none above `102.0` the rest of the day. The hourly median cuts the list from 364 stays to 19 but misses 9 more transfers, so neither rule replaces looking at the trend.

### 3. The review wants every stay's temperatures beside the monitor hours

Lay each stay's temperatures on its own hourly grid, now that step 5 left one row per round. `asfreq()` keeps only readings exactly on the grid, so check that first; `source_row` tells grid-created slots from charted rows with no value.

```python
print('All on the hour:', temps['charted_at'].eq(temps['charted_at'].dt.floor('h')).all())
display(temps['charted_at'].dt.hour.value_counts().sort_index())

temps['source_row'] = 1
temp_grid = (
    temps.set_index('charted_at')
    .groupby('stay_id')[['temperature_f', 'source_row']]
    .resample('h')
    .asfreq()
    .reset_index()
)
temp_grid['grid_created'] = temp_grid['source_row'].isna()
temp_grid['value_missing'] = temp_grid['source_row'].notna() & temp_grid['temperature_f'].isna()
print(temp_grid.shape)
display(temp_grid[['grid_created', 'value_missing']].sum())
```

Expect `All on the hour: True`, and fewer `04:00` rounds (`1614`) than any other round (about `1930`): nurses skip the night round for stable patients who are asleep. The grid has `(46035, 6)` rows. `34790` are `grid_created`: three empty hours between rounds, and more where a round was missed. `190` are `value_missing`: a nurse started a row but entered no temperature.

### 4. A reviewer asks whether a centered window flags `S0297` sooner

A centered window looks forward as well as back, which suits a retrospective chart review but not an alert. `min_periods` lets a trailing window start before it is full.

```python
hourly_297 = hourly[hourly['stay_id'] == 'S0297'].set_index('recorded_at')['mean_hr']
compare = pd.DataFrame({
    'hourly_mean': hourly_297,
    'trailing_6h': hourly_297.rolling(6).mean(),
    'trailing_6h_min3': hourly_297.rolling(6, min_periods=3).mean(),
    'centered_6h_min3': hourly_297.rolling(6, min_periods=3, center=True).mean(),
})
display(compare.head(4).round(1))
display(compare.tail(12).round(1))
```

Expect `trailing_6h` to be `NaN` for the stay's first rows, and from 14:00 on February 9 to the end: each 6-hour window there contains an empty hour, whose mean is `NaN`. `trailing_6h_min3` keeps a value through the first rows (`94.0` at 09:00 on February 7, the mean of 3 hours) and returns at 22:00 (`120.2`). `centered_6h_min3` already reads `120.2` at 20:00, two hours earlier, because it averages heart rates from after 20:00. The answer for the reviewer: it looks sooner only because it reads the future.

### 5. Dr. Reyes wants Demo 1's Eastside outage drawn honestly for the committee

Demo 1 found that Eastside's ED sent no visits from August 12 to 20, 2024. Count its visits per date, then compare two ways of filling the calendar.

```python
ed = pd.read_parquet('data/ed_visits.parquet')
ed['arrival'] = pd.to_datetime(ed['arrival_time'], format='%Y-%m-%d %H:%M', errors='coerce')
east = ed[ed['site'] == 'Eastside'].dropna(subset=['arrival'])
east_daily = east.groupby(east['arrival'].dt.floor('D')).size()
print(len(east_daily), 'dates with visits')
print('resample fills the outage with:', east_daily.resample('D').sum().loc['2024-08-15'])
print('asfreq fills it with:', east_daily.asfreq('D').loc['2024-08-15'])
```

Expect `722 dates with visits`. `resample('D').sum()` fills the outage with `0`, as if nobody came, while `asfreq('D')` fills it with `nan`, which a line plot leaves as a break.

```python
summer = east_daily.asfreq('D').loc['2024-06-01':'2024-10-31']
fig, ax = plt.subplots(figsize=(10, 3.5))
summer.plot(ax=ax, color='gray', alpha=0.6, label='Daily visits')
summer.rolling('28D').mean().plot(ax=ax, color='#0072B2', linewidth=2, label='28-day mean')
ax.set(title='Eastside ED visits, June to October 2024', xlabel='Date', ylabel='Visits per day', ylim=(0, 250))
ax.legend(loc='lower left')
plt.show()

fig, ax = plt.subplots(figsize=(8, 3.5))
by_month = east_daily.groupby(east_daily.index.month).mean()
by_month.plot(kind='bar', ax=ax, rot=0, color='steelblue',
              title='Eastside mean daily visits by calendar month, 2023 to 2024', xlabel='Month', ylabel='Visits per day')
plt.show()
```

Expect a daily line between about 130 and 220 visits with a nine-day break in mid-August, and a 28-day mean near 170 that keeps going through the break (a time window averages whatever days were reported). The bars are highest in January and December, the flu season, at about 190 a day, and near 170 the rest of the year. The monthly means use only reported days, so the outage does not pull August down.
