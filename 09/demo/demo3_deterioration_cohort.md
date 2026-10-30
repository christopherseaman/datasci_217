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

# Demo 3: Build a Day-1 Deterioration Cohort from Three Time Zones

A health system wants an early-warning score that runs 24 hours after each ward admission and predicts transfer to the ICU later in the stay. Its hospitals in New York, Chicago, and Phoenix chart vitals on their own wall clocks, and the lab system stores times in UTC. Put every reading on one clock, build past-only features per encounter, keep only the lab results that were known at the prediction time, split the cohort chronologically, and plot one patient's history. All encounters, patients, and values are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

## Core walkthrough

### 1. Load the encounters, vitals, and labs

This supplied cell downloads `generate_data.py` when the folder lacks it (`urlretrieve(url, path)` saves the file at a web address to `path`), runs its `build()` when a table in `data/` is missing (about 20 seconds, once), and reads three tables. The seed is fixed, so your values match the ones below.

```python
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_RAW = 'https://raw.githubusercontent.com/christopherseaman/datasci_217/main/09/demo'
TABLES = ['encounters', 'ward_vitals', 'labs', 'ed_visits']
if not Path('generate_data.py').exists():
    urlretrieve(f'{REPO_RAW}/generate_data.py', 'generate_data.py')
if not all(Path(f'data/{name}.parquet').exists() for name in TABLES):
    import generate_data
    generate_data.build()

encounters = pd.read_parquet('data/encounters.parquet')
vitals = pd.read_parquet('data/ward_vitals.parquet')
labs = pd.read_parquet('data/labs.parquet')
print(encounters.shape, vitals.shape, labs.shape)
display(encounters.head(3))
display(vitals.head(3))
display(labs.head(3))
```

Expect `(10000, 8) (270306, 6) (57934, 5)`: 10,000 ward encounters in 2024, their charted heart rate and SpO2, and their lab orders. Each encounter has the patient's `age` and admitting `diagnosis`. `admitted_at` and `left_at` are already aware UTC datetimes (`datetime64[us, UTC]`). `recorded_local` is text on each hospital's wall clock, in no particular row order. The lab times are text ending in `Z`, which means UTC.

### 2. Put every site's clock on UTC

Each hospital's readings are localized in the zone where they were charted. This intentional error shows what happens when the New York rows meet the night the clocks fell back, November 3, 2024.

```python
ZONES = {'New York': 'America/New_York', 'Chicago': 'America/Chicago', 'Phoenix': 'America/Phoenix'}
new_york = vitals[vitals['site'] == 'New York']
try:
    pd.to_datetime(new_york['recorded_local'], format='%Y-%m-%d %H:%M').dt.tz_localize('America/New_York')
except ValueError as error:
    print('ValueError:', error)
```

Expect `ValueError: Cannot infer dst time from 2024-11-03 01:52:00, try using the 'ambiguous' argument`. That clock time happened twice, and the text does not say which. The fix sets these readings aside as `NaT`, along with any time that never happened, then converts to UTC. Each site is one piece, and `pd.concat()` (Lecture 06) stacks them.

```python
pieces = []
for site, zone in ZONES.items():
    piece = vitals[vitals['site'] == site].copy()
    local = pd.to_datetime(piece['recorded_local'], format='%Y-%m-%d %H:%M')
    piece['recorded_at'] = (local.dt.tz_localize(zone, ambiguous='NaT', nonexistent='NaT')
                            .dt.tz_convert('UTC'))
    pieces.append(piece)
localized = pd.concat(pieces)

set_aside = localized[localized['recorded_at'].isna()]
display(set_aside.groupby('site').size())
display(set_aside.sort_values('recorded_local')[['encounter_id', 'site', 'recorded_local']])

vitals = (localized.dropna(subset=['recorded_at'])
          .sort_values(['encounter_id', 'recorded_at'])
          .reset_index(drop=True))
print(vitals.shape, vitals['recorded_at'].dtype)
```

Expect `16` readings set aside: `12` in Chicago and `4` in New York, none in Phoenix, because Arizona does not change its clocks. The table lists `4` readings typed by hand at `2024-03-10 02:30` to `02:59`, an hour that never happened on the spring-forward night, and `12` charted between `2024-11-03 01:01` and `01:55`, the hour that happened twice. Report that count before dropping them. `vitals` keeps `(270290, 7)` rows, sorted by encounter and then time, in `datetime64[us, UTC]`.

One Chicago patient, encounter `E08290`, was charted through that night.

```python
night = localized[(localized['encounter_id'] == 'E08290')
                  & localized['recorded_local'].between('2024-11-02 23:00', '2024-11-03 03:59')]
display(night.sort_values('recorded_local')[['recorded_local', 'recorded_at', 'heart_rate']])
```

Expect six rows. `00:51` becomes `05:51:00+00:00` and `03:56` becomes `09:56:00+00:00`: on the wall clock they look about 3 hours apart, but 4 hours passed. The `01:24` and `01:55` rows show `NaT`.

### 3. Past-only features within each encounter

The rows stack 10,000 encounters. A plain `shift(1)` runs straight down the column and hands each encounter's first row the previous encounter's last heart rate.

```python
first_row = ~vitals['encounter_id'].duplicated()
leaked = vitals['heart_rate'].shift(1)[first_row].notna().sum()
print('First rows:', first_row.sum(), ' given another encounter\'s heart rate:', leaked)
```

Expect `First rows: 10000  given another encounter's heart rate: 9866`. Grouping by `encounter_id` keeps every feature inside one stay. Each feature reads only the current row and earlier ones.

```python
by_encounter = vitals.groupby('encounter_id')['heart_rate']
vitals['previous_hr'] = by_encounter.shift(1)
vitals['hr_change'] = by_encounter.diff()
vitals['mean_prev_3'] = by_encounter.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

prev_8h = (vitals.set_index('recorded_at')
           .groupby('encounter_id')['heart_rate']
           .rolling('8h', closed='left')
           .mean()
           .rename('mean_prev_8h')
           .reset_index())
vitals = vitals.merge(prev_8h, on=['encounter_id', 'recorded_at'], validate='one_to_one')
print(vitals.shape)
display(vitals.loc[vitals['encounter_id'] == 'E00002',
                   ['recorded_at', 'heart_rate', 'previous_hr', 'hr_change', 'mean_prev_3', 'mean_prev_8h']].head(5))
```

Expect `(270290, 11)`: the merge matched one feature row to each reading. In `E00002`'s first five readings, `previous_hr` and `hr_change` start as `NaN`. At `02:00` on January 2, `mean_prev_3` is `61.333333` (the 52, 62, and 70 readings, back to 14:02 the day before) but `mean_prev_8h` is `70.0`: only the 19:12 reading falls in the 8 hours before. The grouped rolling step takes a few seconds on 270,290 rows.

### 4. What was known at the prediction time?

The score runs 24 hours after admission. Encounters that ended before then (discharged or already transferred) never get a score, so they leave the cohort.

```python
encounters['prediction_time'] = encounters['admitted_at'] + pd.Timedelta(hours=24)
cohort = encounters[encounters['left_at'] > encounters['prediction_time']].copy()
print('Cohort:', len(cohort), 'of', len(encounters))
display(cohort['outcome'].value_counts())
```

Expect `Cohort: 9549 of 10000`, with `390` ICU transfers. `127` of the 517 transfers happened in the first 24 hours, before the score would run.

A lab result can feed the score only if it was reported by the prediction time, whenever its sample was drawn. `pd.to_datetime(..., utc=True)` reads the lab system's UTC text; a blank `resulted_at` (cancelled or never resulted) becomes `NaT`.

```python
labs['collected_at'] = pd.to_datetime(labs['collected_at'], utc=True)
labs['resulted_at'] = pd.to_datetime(labs['resulted_at'], utc=True)
labs['turnaround_h'] = (labs['resulted_at'] - labs['collected_at']).dt.total_seconds() / 3600
display(labs.groupby('test')['turnaround_h'].median().round(2))

cohort_labs = labs.merge(cohort[['encounter_id', 'prediction_time']], on='encounter_id', validate='many_to_one')
cohort_labs['collected_by_then'] = cohort_labs['collected_at'] <= cohort_labs['prediction_time']
cohort_labs['available'] = cohort_labs['resulted_at'] <= cohort_labs['prediction_time']
display(pd.crosstab(cohort_labs['test'], [cohort_labs['collected_by_then'], cohort_labs['available']]))
```

Expect median turnarounds of `0.67` hours for lactate, `1.15` for troponin, `1.80` for creatinine, and `45.23` for a blood culture. In the crosstab, the `True, False` column counts labs drawn before the prediction time but not yet reported: `2136` blood cultures, `1949` creatinines, `49` lactates, and `56` troponins. A filter on `collected_at` would let all 4,190 of them into the score.

### 5. One row per encounter at the prediction time

The feature row for each encounter is its last reading at or before the prediction time. Sorting by encounter and time already happened, so `groupby().tail(1)` (Lecture 08) keeps the latest row. The latest lactate comes from the available results only.

```python
at_or_before = vitals.merge(cohort[['encounter_id', 'prediction_time']], on='encounter_id', validate='many_to_one')
at_or_before = at_or_before[at_or_before['recorded_at'] <= at_or_before['prediction_time']]
last_vitals = at_or_before.groupby('encounter_id').tail(1)

lactate = cohort_labs[(cohort_labs['test'] == 'lactate') & cohort_labs['available']]
last_lactate = (lactate.sort_values('resulted_at').groupby('encounter_id').tail(1)
                [['encounter_id', 'value']].rename(columns={'value': 'last_lactate'}))

features = (cohort[['encounter_id', 'patient_id', 'site', 'prediction_time', 'outcome']]
            .merge(last_vitals[['encounter_id', 'heart_rate', 'previous_hr', 'hr_change', 'mean_prev_3', 'mean_prev_8h']],
                   on='encounter_id', how='left', validate='one_to_one')
            .merge(last_lactate, on='encounter_id', how='left', validate='one_to_one'))
features['icu_transfer'] = features['outcome'] == 'ICU transfer'
print(features.shape)
display(features.head(3).round({'mean_prev_3': 1, 'mean_prev_8h': 1}))
```

Expect `(9549, 12)`: one row per encounter. A missing `last_lactate` means no lactate had been reported by the prediction time, not a normal result.

A first look at signal: the ICU-transfer rate by lactate group. `np.select()` (Lecture 03) names each group; a missing lactate matches neither condition, so it gets the default.

```python
features['lactate_group'] = np.select([features['last_lactate'] > 2, features['last_lactate'] <= 2],
                                       ['over 2', '2 or less'], default='none reported')
display(features.groupby('lactate_group')['icu_transfer'].agg(['count', 'mean']).round(3))
```

Expect transfer rates of `0.052` (`2005` encounters) with a lactate of 2 mmol/L or less, `0.131` (`1074`) over 2, and `0.022` (`6470`) with none reported. A clinician orders a lactate when worried, so whether one was ordered carries signal too.

### 6. A chronological holdout

Build the score on earlier encounters and test it on later ones, the way it would meet future patients. This intentional error compares the aware `prediction_time` column with a naive cutoff.

```python
try:
    features['prediction_time'] < pd.Timestamp('2024-10-01')
except TypeError as error:
    print('TypeError:', error)
```

Expect `TypeError: Invalid comparison between dtype=datetime64[us, UTC] and Timestamp`: an aware column cannot be compared with a naive timestamp. Give the cutoff the column's zone.

```python
cutoff = pd.Timestamp('2024-10-01', tz='UTC')
features['block'] = np.where(features['prediction_time'] < cutoff, 'earlier', 'later_holdout')
display(features.groupby('block')['icu_transfer'].agg(['count', 'sum', 'mean']).round(3))
in_both = features.groupby('patient_id')['block'].nunique()
print('Patients in both blocks:', (in_both == 2).sum(), 'of', len(in_both))
```

Expect `7109` earlier encounters (`292` transfers) and `2440` in the holdout (`98`), both near a `0.040` transfer rate. `133` of 8,556 patients were readmitted across the cutoff and appear in both blocks: this split tests later encounters, not entirely unseen patients.

### 7. Plot one encounter in UTC

`E05701`, a New York patient admitted with a GI bleed, was transferred on July 29. Draw its readings, a 4-hour rolling mean, and a dashed line at 100 bpm, the usual threshold for tachycardia (a fast heart rate). The axis label names the zone.

```python
one = vitals[vitals['encounter_id'] == 'E05701'].set_index('recorded_at')['heart_rate']
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(one.index, one, 'o', color='gray', markersize=4, label='Charted heart rate')
ax.plot(one.index, one.rolling('4h').mean(), color='#0072B2', linewidth=2, label='4-hour rolling mean')
ax.axhline(100, color='#D55E00', linestyle='--', label='Tachycardia (100 bpm)')
ax.set(title='Encounter E05701: heart rate before ICU transfer', xlabel='Time (UTC)', ylabel='Heart rate (bpm)')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.show()
display(encounters.loc[encounters['encounter_id'] == 'E05701', ['site', 'diagnosis', 'admitted_at', 'left_at', 'outcome']])
```

Expect readings every 4 hours between about 75 and 100 bpm for four days, then hourly checks on July 29 as the rolling mean crosses the dashed line and climbs to about 115 bpm before the transfer at `23:54` UTC.

## Independent practice

Continue on your own after class. These cells reuse `vitals`, `last_vitals`, and `ZONES` from the core; if the runtime closed, run the cells above again first.

### 1. Count days on the local clock

A daily report for the New York hospital counts its days on New York's clock. Convert to the reporting zone before `resample('D')`; binning in UTC starts each New York day at 19:00 or 20:00 the evening before. `vitals` is sorted by encounter, not time, so sort the index before slicing dates.

```python
ny = vitals[vitals['site'] == 'New York'].set_index('recorded_at')['heart_rate'].sort_index()
utc_days = ny.loc['2024-11-01':'2024-11-05'].resample('D').count()
local_days = ny.tz_convert('America/New_York').loc['2024-11-01':'2024-11-05'].resample('D').count()
display(pd.DataFrame({'utc_day': utc_days.to_numpy(), 'new_york_day': local_days.to_numpy()},
                     index=['Nov 1', 'Nov 2', 'Nov 3', 'Nov 4', 'Nov 5']))

for day in ['2024-03-10', '2024-11-03', '2024-11-04']:
    hours = pd.date_range(day, pd.Timestamp(day) + pd.Timedelta(days=1), freq='h',
                          tz='America/New_York', inclusive='left')
    print(day, len(hours), 'hours')
```

Expect the columns to disagree every day, because a New York day starts 4 or 5 hours after a UTC day: `406` and `412` on November 1, `372` and `347` on November 2. November 3 differs most, `337` readings on the New York day against `298` on the UTC day: the shifted start accounts for most of the gap, and that local day also held one extra hour. The `date_range` lines print `23`, `25`, and `24` hours: a local daily count on March 10 covers one hour less.

### 2. Plot the ED feed outage honestly

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
monthly = east_daily.groupby(east_daily.index.month).mean()
monthly.plot(kind='bar', ax=ax, rot=0, color='steelblue',
             title='Eastside mean daily visits by calendar month, 2023 to 2024', xlabel='Month', ylabel='Visits per day')
plt.show()
```

Expect a daily line between about 130 and 220 visits with a nine-day break in mid-August, and a 28-day mean near 170 that keeps going through the break (a time window averages whatever days were reported). The bars are highest in January and December, the flu season, at about 190 a day, and near 170 the rest of the year. The monthly means use only reported days, so the outage does not pull August down.

### 3. Audit two leaking candidates

Two tempting features read the row after the current one: the next heart rate (a lead, `shift(-1)`) and a centered 3-reading mean. At each encounter's feature row from step 5, find when that next reading was charted.

```python
vitals['next_time'] = vitals.groupby('encounter_id')['recorded_at'].shift(-1)
check = (last_vitals[['encounter_id', 'recorded_at', 'prediction_time']]
         .merge(vitals[['encounter_id', 'recorded_at', 'next_time']],
                on=['encounter_id', 'recorded_at'], validate='one_to_one'))
hours_late = (check['next_time'] - check['prediction_time']).dt.total_seconds() / 3600
print('Feature rows:', len(check), ' next reading after the prediction time:', (hours_late > 0).sum())
print('Median hours after the prediction time:', round(hours_late.median(), 1))
```

Expect `Feature rows: 9549  next reading after the prediction time: 9428` and a median of `1.9` hours. The other 121 encounters had no later reading. Every feature row is the last reading before the score runs, so a lead or a centered window there reads a value nobody had yet. Use `shift(-1)` only to build a target, never a feature.

### 4. Offsets through the year

The same zone name gives different UTC offsets in winter and summer, except in Phoenix.

```python
for zone in ZONES.values():
    winter = pd.Timestamp('2024-01-15 12:00', tz='UTC').tz_convert(zone)
    summer = pd.Timestamp('2024-07-15 12:00', tz='UTC').tz_convert(zone)
    print(f'{zone:<20} {winter.strftime("%H:%M %z")}  {summer.strftime("%H:%M %z")}')
```

Expect New York at `-0500` in January and `-0400` in July, Chicago at `-0600` and `-0500`, and Phoenix at `-0700` both times.
