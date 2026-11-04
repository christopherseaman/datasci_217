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

**This phase's ask.** The unit's nurse manager, Sam Okoye, and the patient-safety officer meet at the end of the month to decide whether to buy portable telemetry units, which keep a patient monitored during imaging and walks. They want, from last winter's monitor archive (January to March 2024): how much of each stay was monitored, when the gaps happen, which stays went unwatched for 3 hours or more, and a review of stay `S0023`, transferred to the ICU on January 6, including which smoother the unit's heart-rate alert should use. All stays and readings are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas, pyarrow (for Parquet), and dowhy (for generate_data.py) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0 dowhy==0.14
```

## Core walkthrough

### 1. Load the stays and the monitor feed

This supplied cell downloads `generate_data.py` when the folder lacks it (`urlretrieve(url, path)` saves the file at a web address to `path`), runs its `build()` when a table in `data/` is missing (about 25 seconds, once), and reads three tables. The seed is fixed, so your values match the ones below.

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

Expect `(688, 7) (550659, 5) (11509, 3)`: 688 stays, over half a million monitor readings, and the temperatures nurses charted every 4 hours. Most stays end in discharge (`630`); `37` end in an ICU transfer, and `21` were still on the unit when the archive ends on March 31. `57` stays belong to a patient with an earlier stay, so `stay_id`, not `patient_id`, names one continuous monitoring episode.

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

Expect `(49375, 7)`: one row per stay per clock hour, from each stay's first hour to its last. `S0001`'s first hour, `00:00`, has only `4` readings, because the patient arrived at 00:39; its next hours have `12`.

A first draft grouped by `patient_id`. Count what that does to the readmitted patients.

```python
by_patient = telemetry.set_index('recorded_at').groupby('patient_id')['source_row'].resample('h').count()
print('Hours grouped by stay:', len(hourly), ' by patient:', len(by_patient))
print('Empty hours by stay:', (hourly['n_readings'] == 0).sum(), ' by patient:', (by_patient == 0).sum())
```

Expect `49375` hours by stay against `76697` by patient, and `1260` empty hours against `28902`. The extra 27,642 empty hours are weeks at home between admissions, which the safety review would have counted as monitoring gaps.

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

Expect `0.870` of stay-hours `full` (all 12 readings), `0.105` `partial` (a short break, or the first or last hour of a stay), and `0.026` `none`: 1,260 whole hours with no reading. The bars are low overnight (`11` to `20` empty hours an hour from 02:00 to 08:00), rise at 09:00, and stay between `66` and `90` an hour from 10:00 through 21:00. `0.73` of empty hours start between 09:00 and 20:00, when patients go to imaging, walk the hall, or shower.

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

Expect `615 gap runs`. The tallest bar is one empty hour (`365` runs), but `128` runs last 3 to 8 hours, long enough to miss a deterioration, and two last `9`.

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

Expect `Stays with no empty hour: 318 of 688` and `116` stays with a gap of 3 hours or more, the safety officer's review list. The top row is `S0509`, with two gaps, the longer `9.0` hours. `0.270` of ICU transfers had a 3-hour gap (10 of 37), against `0.165` of discharged stays (104 of 630).

**What we tell Sam:** 116 stays, 1 in 6, had a gap of 3 hours or more. Transfers look worse, but with only 37 of them, four stays move that share by about 10 points, so one quarter cannot show that the monitor misses sicker patients more often.

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

Expect `Repeated rows: 241  stays: 105`. The pairs match or differ by a few tenths, such as `S0005`'s `100.2` and `99.6` at 08:00 on January 3, and `0.87` of the repeated rounds include a reading of 99.5 °F or more: a nurse who sees a fever rechecks it and charts the recheck under the same round. Keep one row per stay and round: the mean of the reading and its recheck.

```python
temps = temps.groupby(['stay_id', 'charted_at'], as_index=False)['temperature_f'].mean()
print(temps.shape, temps.duplicated(['stay_id', 'charted_at']).sum())
```

Expect `(11268, 3) 0`. Now upsample `S0023`'s temperatures to an hourly grid three ways: `asfreq()` leaves the new hours empty, `ffill(limit=2)` carries each round forward for two hours, and `interpolate()` draws a straight line between rounds.

```python
one_temp = temps[temps['stay_id'] == 'S0023'].set_index('charted_at')['temperature_f']
filled = pd.DataFrame({
    'asfreq': one_temp.resample('h').asfreq(),
    'ffill_2': one_temp.resample('h').ffill(limit=2),
    'interpolate': one_temp.resample('h').interpolate(),
})
display(filled.tail(9))

last_day_temp = filled.loc['2024-01-06']
fig, ax = plt.subplots(figsize=(9, 3.5))
last_day_temp['interpolate'].plot(ax=ax, color='#0072B2', label='interpolate()')
last_day_temp['ffill_2'].plot(ax=ax, drawstyle='steps-post', color='#D55E00', label='ffill(limit=2)')
last_day_temp['asfreq'].plot(ax=ax, style='o', color='black', label='Charted (asfreq)')
ax.set(title='Stay S0023: charted temperature on an hourly grid, January 6', xlabel='Time (unit clock)', ylabel='Temperature (°F)')
ax.legend(loc='upper left')
plt.show()
```

Expect the table to end with `S0023`'s last three rounds, `97.4`, `98.6`, and `99.4` °F at 08:00, 12:00, and 16:00. `asfreq` is `NaN` between rounds. `ffill_2` repeats each round for two hours, and the hour before the next round stays `NaN`, so the orange steps break there. `interpolate` climbs from 12:00 and reaches `99.2` at 15:00, a value that needs the 16:00 reading, which no alert running at 15:00 could have seen. Only the black dots were measured.

### 6. Rapid-response review: which smoother should the alert use?

`S0023` went to the ICU at 17:12 on January 6, after almost two and a half hours off the monitor late that morning. The unit's alert smooths heart rate before comparing it with 120 bpm. Compare a 12-reading count window (one hour when nothing is missing) with a 1-hour time window around the patient's return.

```python
s0023 = telemetry[telemetry['stay_id'] == 'S0023'].set_index('recorded_at')['heart_rate'].dropna()
windows = pd.DataFrame({
    'heart_rate': s0023,
    'last_12_readings': s0023.rolling(12).mean(),
    'last_1_hour': s0023.rolling('1h').mean(),
})
display(windows.loc['2024-01-06 10:30':'2024-01-06 14:30'].round(1))
```

Expect the jump from `11:30` straight to `13:55`: the heart rate rose from about 100 to `118.0` while nobody could see it. At 13:55, `last_12_readings` is `98.3`, still mostly readings from before the gap, while `last_1_hour` is `118.0`, the 13:55 reading alone. By 14:30 the count window has climbed only to `116.7`; the time window reads `125.0`. A count window follows rows; a time window follows the clock.

Plot the last 24 hours for the review slide: the raw readings, a 1-hour rolling mean with a band of plus or minus one rolling standard deviation, an exponentially weighted mean (EWM) with `span=12`, and a dotted line at the 120 bpm alert. Every reading sits on a 5-minute mark, so `asfreq('5min')` adds `NaN` rows across the gap and the lines break there instead of bridging it.

```python
last_day = s0023.loc['2024-01-05 17:12':]
mean_1h = last_day.rolling('1h').mean().asfreq('5min')
sd_1h = last_day.rolling('1h').std().asfreq('5min')
ewm_12 = last_day.ewm(span=12).mean().asfreq('5min')

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(last_day.index, last_day, '.', color='gray', markersize=3, label='Reading (every 5 min)')
ax.plot(mean_1h.index, mean_1h, color='#0072B2', label='1-hour rolling mean')
ax.fill_between(mean_1h.index, mean_1h - sd_1h, mean_1h + sd_1h, color='#0072B2', alpha=0.2, label='± 1 SD')
ax.plot(ewm_12.index, ewm_12, color='#D55E00', linestyle='--', label='EWM, span 12')
ax.axhline(120, color='black', linestyle=':', label='Alert (120 bpm)')
ax.set(title='Stay S0023: heart rate in the 24 hours before ICU transfer',
       xlabel='Time (unit clock)', ylabel='Heart rate (bpm)')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.show()
print('First hour mean:', round(last_day.iloc[:12].mean(), 1), ' Last hour mean:', round(last_day.iloc[-12:].mean(), 1))
```

Expect `First hour mean: 93.3  Last hour mean: 136.8`. The band drifts from about 82 bpm overnight to about 97 by 11:30. The gray dots stop after 11:30 and resume at 13:55, and both lines break across that gap. After the gap, the 1-hour mean starts near 118, while the dashed EWM starts near 100 and takes 55 minutes to reach 120: like a count window, EWM weights rows, so it still carries the readings from before the gap. The 1-hour mean stays above the alert line from 14:00 to the transfer and climbs to about 137.

**What we tell Sam:** after a patient returns to the monitor, a 1-hour time window reaches the alert level at once, while a 12-reading window or an EWM lags by up to an hour. The alert should use the time window, and the heart rate that climbed unseen during this patient's 2.5-hour gap is the case for portable units.

## Independent practice

These are follow-up requests from the unit and the program; continue on your own after class. The cells reuse `stays`, `telemetry`, `temps` (one row per round after step 5), `hourly`, and `s0023` from the core; if the runtime closed, run the cells above again first.

### 1. The unit director wants a monthly unit report

A unit-level count is fine to resample across patients: one admission is one event, whoever it is. Index the stays by admission time, then summarize each month with a different aggregation per column.

```python
stays['los_days'] = (stays['left_at'] - stays['admitted_at']).dt.total_seconds() / 86400
stays['icu_transfer'] = stays['outcome'] == 'ICU transfer'
monthly = stays.set_index('admitted_at').resample('ME').agg(
    {'stay_id': 'count', 'los_days': ['mean', 'max'], 'icu_transfer': 'sum', 'age': 'median'})
display(monthly.round(1))
```

Expect three month-end rows, `2024-01-31` to `2024-03-31`, with `247`, `209`, and `232` admissions, mean stays of `3.0`, `3.1`, and `2.8` days, `14`, `11`, and `12` ICU transfers, and a median age of `68` or `69`. `icu_transfer` is `True` or `False`, so its `sum` counts transfers.

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

Expect `raw reading: 366 stays flagged, 34 of 37 ICU transfers` and `hourly median: 29 stays flagged, 22 of 37 ICU transfers`. `493` stay-days are flagged by a reading but by no hour. The example, `S0001` on January 1, has one reading of `145.0` at 06:40 and none above `90.0` the rest of the day. The hourly median cuts the list from 366 stays to 29 but misses 12 more transfers, so neither rule replaces looking at the trend.

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

Expect `All on the hour: True`, and fewer `04:00` rounds (`1566`) than any other round (about `1940`): nurses skip the night round for stable patients who are asleep. The grid has `(46256, 6)` rows. `34988` are `grid_created`: three empty hours between rounds, and more where a round was missed. `172` are `value_missing`: a nurse started a row but entered no temperature.

### 4. A reviewer asks whether a centered window flags `S0023` sooner

A centered window looks forward as well as back, which suits a retrospective chart review but not an alert. `min_periods` lets a trailing window start before it is full.

```python
hourly_023 = hourly[hourly['stay_id'] == 'S0023'].set_index('recorded_at')['mean_hr']
compare = pd.DataFrame({
    'hourly_mean': hourly_023,
    'trailing_6h': hourly_023.rolling(6).mean(),
    'trailing_6h_min3': hourly_023.rolling(6, min_periods=3).mean(),
    'centered_6h_min3': hourly_023.rolling(6, min_periods=3, center=True).mean(),
})
display(compare.head(4).round(1))
display(compare.tail(12).round(1))
```

Expect `trailing_6h` to be `NaN` for the stay's first rows, and from 12:00 on January 6 to the end: each 6-hour window there contains the empty 12:00 hour, whose mean is `NaN`. `trailing_6h_min3` keeps a value through the first rows (`93.6` at 12:00 on January 1, the mean of 3 hours) and through the gap, and first passes 120 at 16:00 (`121.1`). `centered_6h_min3` already reads `121.1` at 14:00, two hours earlier, because it averages heart rates from after 14:00. The answer for the reviewer: it looks sooner only because it reads the future.

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

### 6. The rapid-response committee wants `S0023`'s last day as a table

The committee reads 4-hour blocks, the same span as a nurse's vitals round. `agg()` with a list of functions gives one column per summary, and `count` shows how many readings each block holds.

```python
display(s0023.loc['2024-01-06'].resample('4h').agg(['mean', 'std', 'min', 'max', 'count']).round(1))
```

Expect five blocks, the last one at `16:00` because the transfer came at 17:12. The `08:00` block holds `42` readings and the `12:00` block only `25`: the gap ran from 11:30 to 13:55. The `12:00` block jumps to a mean of `125.3`, with a `max` of `139.0`, and the `16:00` block reaches `137.7`, against means of `82` to `89` earlier in the day.

### 7. The alert team asks whether 120 bpm means the same at 3 a.m. and 3 p.m.

Before changing a threshold, run the time series EDA steps on the series it watches. Start with spacing: `S0023`'s readings are 5 minutes apart except where the monitor went quiet. Then build one unit-wide heart rate per clock hour (the mean of every stay's hourly mean, a unit-level summary like practice 1) and check its level, its hour and weekday profiles, and its autocorrelation.

```python
display(s0023.index.diff().value_counts())

unit_hr = hourly.groupby('recorded_at')['mean_hr'].mean().asfreq('h')
print('Unit hours:', len(unit_hr), ' missing:', unit_hr.isna().sum())
display(unit_hr.resample('ME').mean().round(1))
by_hour = unit_hr.groupby(unit_hr.index.hour).mean()
by_weekday = unit_hr.groupby(unit_hr.index.dayofweek).mean()
print('Lowest hour:', by_hour.idxmin(), round(by_hour.min(), 1), ' highest hour:', by_hour.idxmax(), round(by_hour.max(), 1))
print('Weekday means from', round(by_weekday.min(), 1), 'to', round(by_weekday.max(), 1))
for lag in [1, 12, 24]:
    print(f'lag {lag}:', round(unit_hr.autocorr(lag=lag), 2))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
by_hour.plot(ax=axes[0], marker='o', color='steelblue', xticks=range(0, 24, 3),
             title='4 West mean heart rate by clock hour', xlabel='Hour of day', ylabel='Heart rate (bpm)')
pd.plotting.lag_plot(unit_hr, lag=24, ax=axes[1], s=6, alpha=0.4)
axes[1].set(title='Unit heart rate against the same hour the next day', xlabel='Hour t (bpm)', ylabel='Hour t + 24 (bpm)')
plt.show()
```

Expect `S0023`'s spacing to be `0 days 00:05:00` for `1311` gaps between readings and `7` of 10 minutes, then single breaks from 25 minutes up to `0 days 06:35:00` earlier in the stay, including the `0 days 02:25:00` off the monitor in step 6. The unit series has `2184` hours and `0` missing: some patient is always on the monitor. Monthly means are `83.5`, `81.6`, and `81.9` bpm, so there is no trend. The lowest hour is `3` at `76.3` bpm and the highest is `15` at `88.5`, a 12 bpm daily swing, while the weekday means differ by under 2 bpm (`81.5` to `83.4`). The line climbs from 03:00 to a mid-afternoon peak. Autocorrelation is `0.95` at lag 1, `-0.51` at lag 12 (afternoon against early morning), and `0.88` at lag 24, the tight diagonal in the lag plot.

**What we tell the alert team:** the unit's heart rate follows the clock, not the weekday, and swings about 12 bpm each day. A fixed 120 bpm line sits 44 bpm above the 03:00 mean and 32 above the 15:00 mean, so a night-time rise has further to climb before it alerts; the team can weigh an hour-aware threshold.
