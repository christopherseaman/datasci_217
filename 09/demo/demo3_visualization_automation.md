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

# Demo 3: Time Zones, Past-Only Patient Features, and Time-Series Plots

A New York ICU exports charted heart rates on the local wall clock, and the export spans the night the clocks fell back. You convert clinic times to UTC, set aside the clock readings that happened twice, build lags and past-only means inside each patient's history, check which values were known at a prediction time, split the rows into a chronological holdout, and plot the panel and three years of emergency-department visits. Everything here comes from Lecture 09, plus Lectures 01 to 08. Patient values and visit counts are synthetic.


## How to run

Run the cells from top to bottom; after each step, an **Expect** line says what you should see. The notebook builds its own data, so it needs nothing from an earlier demo. Tested 2026-09-30 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, matplotlib 3.11.1, and seaborn 0.13.2. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them.


## Choose Your Route

The **core walkthrough** is the part practiced in class. Work through **independent practice** on your own after class. For a full repeat, restart and run every cell from top to bottom; both routes use the same code below.

| Route | Cells to run |
| --- | --- |
| Core walkthrough | Run Setup, both cells in [3. The night the clocks fell back](#3-the-night-the-clocks-fell-back), both cells in [5. Previous readings within each patient](#5-previous-readings-within-each-patient), [6. Past-only means](#6-past-only-means), the first cell in [7. What was known at the prediction time?](#7-what-was-known-at-the-prediction-time), then [8. A chronological holdout](#8-a-chronological-holdout). |
| Independent practice | After class, compare the clinic clocks and both DST days, audit candidate features, and plot patient/flu histories. Repeat the daily-weight lag/lead alert and its plot; that section labels its optional inference and percentage-change methods as BONUS.md topics. |

**Core checkpoint:** Set aside 1 ambiguous reading; each patient starts with an empty lag. At P01's 08:00 UTC reading, `mean_prev_2` is 97.5 and `mean_prev_2h` is `NaN`; 2 of 4 labs are available at the prediction time; the holdout has 9 earlier and 4 later rows.


## Setup

- **In Colab:** run the install cell below first.
- **Locally:** run these commands in a terminal, then open the `09-demo` folder in VS Code and choose its `.venv` as the notebook kernel.

<!-- #region -->
```shell
curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/09/demo/setup_demo.sh | sh
cd ~/09-demo
uv venv --seed
source .venv/bin/activate
uv sync
```
<!-- #endregion -->

If `~/09-demo` is already set up from Demo 1, just open it and choose its `.venv` kernel.

```python
# Setup: install the course's pandas version
%pip install -q --no-warn-conflicts pandas==3.0.5
```

**Expect:** nothing, or a note to restart the kernel; if Colab asks to restart the session, do it and rerun from the top.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print('pandas', pd.__version__)
print('NumPy', np.__version__)
```

**Expect:** `pandas 3.0.5` and `NumPy 2.3.3` (Colab may show a different NumPy; that is fine).


## 1. One instant on several clinic clocks

A telehealth consult starts at 14:00 UTC. `tz_convert()` shows what each site's wall clock read at that instant. Zone names come from the IANA time-zone database, the standard list of world time zones, such as `'America/New_York'`.

```python
consult = pd.Timestamp('2024-03-15 14:00', tz='UTC')
for zone in ['UTC', 'America/New_York', 'America/Chicago', 'America/Los_Angeles', 'Europe/London']:
    print(f'{zone:<20} {consult.tz_convert(zone)}')
```

**Expect:** the same instant as `10:00-04:00` in New York, `09:00-05:00` in Chicago, `07:00-07:00` in Los Angeles, and `14:00+00:00` in London. The US had already switched to daylight saving time on March 10, and the UK had not yet (it switches on March 31).


## 2. Two clinics' visit times, ordered in UTC

A blood-pressure trial runs clinics in New York and Chicago. Each exports visit times as naive text on its own wall clock. Localize each site in the zone where it recorded, then convert both to UTC, and stack them with `pd.concat()` (Lecture 06).

```python
new_york = pd.DataFrame({'site': 'New York', 'patient_id': ['NY-01', 'NY-02'],
                         'visit_local': ['2024-03-15 09:00', '2024-03-15 10:30'], 'sbp': [142, 128]})
chicago = pd.DataFrame({'site': 'Chicago', 'patient_id': ['CH-01', 'CH-02'],
                        'visit_local': ['2024-03-15 08:30', '2024-03-15 09:00'], 'sbp': [135, 150]})

new_york['visit_utc'] = (pd.to_datetime(new_york['visit_local'], format='%Y-%m-%d %H:%M')
                         .dt.tz_localize('America/New_York').dt.tz_convert('UTC'))
chicago['visit_utc'] = (pd.to_datetime(chicago['visit_local'], format='%Y-%m-%d %H:%M')
                        .dt.tz_localize('America/Chicago').dt.tz_convert('UTC'))
visits = pd.concat([new_york, chicago], ignore_index=True)

display(visits.sort_values('visit_local'))
display(visits.sort_values('visit_utc'))
```

**Expect:** sorted by the local text, Chicago's 08:30 visit comes first. Sorted by `visit_utc`, New York's 09:00 visit (`13:00+00:00`) comes first and Chicago's 08:30 (`13:30+00:00`) second: 08:30 in Chicago happened half an hour after 09:00 in New York.


## 3. The night the clocks fell back

The New York ICU's export covers the night of November 2 to 3, 2024, for three patients. At 02:00 that night, clocks went back to 01:00, so every local time from 01:00 to 01:59 happened twice. One of P02's readings was charted at 01:30, and the export does not say which 01:30 it was.

```python
raw = pd.DataFrame({
    'patient_id': ['P01'] * 5 + ['P02'] * 5 + ['P03'] * 4,
    'recorded_local': [
        '2024-11-02 22:00', '2024-11-02 23:30', '2024-11-03 00:30', '2024-11-03 03:00', '2024-11-03 04:00',
        '2024-11-02 22:30', '2024-11-03 00:00', '2024-11-03 01:30', '2024-11-03 02:00', '2024-11-03 03:30',
        '2024-11-02 23:00', '2024-11-03 00:15', '2024-11-03 02:15', '2024-11-03 04:00',
    ],
    'heart_rate': [88, 94, 101, 112, 118, 72, 70, 68, 71, 69, 104, 99, 95, 92],
})
print(raw.shape)
```

**Expect:** `(14, 3)`. P01's heart rate climbs from 88 to 118 bpm (deteriorating), P02 stays near 70, and P03 falls from 104 to 92 (recovering).

Localize with `ambiguous='NaT'` and `nonexistent='NaT'`, so pandas marks the uncertain reading instead of guessing, and report how many were set aside before dropping them.

```python
local = pd.to_datetime(raw['recorded_local'], format='%Y-%m-%d %H:%M')
aware = local.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')
print('Set aside:', aware.isna().sum())
display(raw[aware.isna()])

raw['recorded_at'] = aware.dt.tz_convert('UTC')
vitals = (raw.dropna(subset=['recorded_at'])[['patient_id', 'recorded_at', 'heart_rate']]
          .sort_values(['patient_id', 'recorded_at'])
          .reset_index(drop=True))
display(vitals)
```

**Expect:** `Set aside: 1`, the P02 row at `2024-11-03 01:30`. `vitals` has 13 rows, sorted by patient and then time, all in UTC. P01's local 00:30 and 03:00 look 2.5 hours apart, but in UTC they are `04:30` and `08:00`: 3.5 hours passed, because the 01:00 hour happened twice.


## 4. Check both clock-change days

A monitor that records every hour produces 24 readings on an ordinary day. On clock-change days, count the elapsed hours in the local day instead of assuming 24.

```python
for day in ['2024-03-10', '2024-11-03', '2024-11-04']:
    hours = pd.date_range(day, pd.Timestamp(day) + pd.Timedelta(days=1), freq='h',
                          tz='America/New_York', inclusive='left')
    print(day, len(hours), 'hours')

# A dose charted at 02:30 on the spring-forward night names a time that never happened
charted = pd.Series(pd.to_datetime(['2024-03-10 01:30', '2024-03-10 02:30', '2024-03-10 03:30']))
spring = charted.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')
display(spring)
print('Set aside:', spring.isna().sum())
```

**Expect:** `2024-03-10 23 hours`, `2024-11-03 25 hours`, and `2024-11-04 24 hours`. The 02:30 dose becomes `NaT` (`Set aside: 1`); the 01:30 and 03:30 doses keep offsets of `-05:00` and `-04:00`, one hour of elapsed time apart.


## 5. Previous readings within each patient

The table stacks three patients. A plain `shift(1)` runs straight down the stacked column; a grouped shift stays inside each patient's history.

```python
vitals['wrong_previous'] = vitals['heart_rate'].shift(1)
by_patient = vitals.groupby('patient_id')['heart_rate']
vitals['previous_hr'] = by_patient.shift(1)
display(vitals[['patient_id', 'heart_rate', 'wrong_previous', 'previous_hr']])
```

**Expect:** the two columns agree except on each patient's first row. Row 5, P02's first reading, has `wrong_previous` `118.0`, P01's last heart rate, and row 9, P03's first, has `69.0` from P02. `previous_hr` is `NaN` on all three first rows.

```python
vitals = vitals.drop(columns='wrong_previous')
vitals['hr_change'] = by_patient.diff()
display(vitals[['patient_id', 'heart_rate', 'previous_hr', 'hr_change']])
```

**Expect:** P01's changes are all positive (`6.0`, `7.0`, `11.0`, `6.0`); P03's are all negative (`-5.0`, `-4.0`, `-3.0`).


## 6. Past-only means

Two summaries of the recent past, both excluding the current reading: the mean of the previous two readings, however far apart, and the mean over the previous two hours of elapsed time.

```python
vitals['mean_prev_2'] = by_patient.transform(
    lambda s: s.shift(1).rolling(2, min_periods=1).mean()
)
prev_2h = (
    vitals.set_index('recorded_at')
    .groupby('patient_id')['heart_rate']
    .rolling('2h', closed='left')
    .mean()
    .rename('mean_prev_2h')
    .reset_index()
)
vitals = vitals.merge(prev_2h, on=['patient_id', 'recorded_at'], validate='one_to_one')
print(vitals.shape)
display(vitals[['patient_id', 'recorded_at', 'heart_rate', 'mean_prev_2', 'mean_prev_2h']])
```

**Expect:** `(13, 7)`: the merge kept one row per reading. At P01's `08:00` reading, `mean_prev_2` is `97.5` (the 94 and 101 readings) but `mean_prev_2h` is `NaN`, because nothing was recorded in the two hours before 08:00 UTC. At P03's `09:00` reading, the two-hour mean is `95.0`, only the 07:15 reading.


## 7. What was known at the prediction time?

An early-warning score runs at 08:00 UTC (03:00 in New York). Lab results count only if they had been reported by then, whenever the sample was drawn. `pd.to_datetime(..., utc=True)` reads times the lab system already stores in UTC.

```python
prediction_time = pd.Timestamp('2024-11-03 08:00', tz='UTC')
labs = pd.DataFrame({
    'patient_id': ['P01', 'P01', 'P02', 'P03'],
    'test': ['lactate', 'blood culture', 'creatinine', 'troponin'],
    'collected_at': pd.to_datetime(['2024-11-03 06:00', '2024-11-03 04:45',
                                    '2024-11-03 07:30', '2024-11-03 07:00'], utc=True),
    'resulted_at': pd.to_datetime(['2024-11-03 06:40', '2024-11-04 14:00',
                                   '2024-11-03 08:20', '2024-11-03 07:45'], utc=True),
})
labs['available'] = labs['resulted_at'] <= prediction_time
display(labs[['patient_id', 'test', 'resulted_at', 'available']])
```

**Expect:** lactate and troponin are `True`; the blood culture (resulted the next day) and the creatinine (drawn at 07:30, resulted at 08:20) are `False`, even though both were collected before 08:00.

The same test applies to features. For P01's reading at 08:00 UTC, each candidate reads rows up to some latest timestamp: the lag reads the 04:30 row, while a lead or a centered window reads the 09:00 row, which did not exist yet.

```python
audit = pd.DataFrame({
    'candidate': ['current heart rate', 'previous heart rate', 'mean of previous 2 readings',
                  'next heart rate', 'centered 3-reading mean'],
    'latest_timestamp': pd.to_datetime(['2024-11-03 08:00', '2024-11-03 04:30', '2024-11-03 04:30',
                                        '2024-11-03 09:00', '2024-11-03 09:00'], utc=True),
})
audit['available'] = audit['latest_timestamp'] <= prediction_time
audit['decision'] = np.where(audit['available'], 'keep', 'reject')
display(audit)
```

**Expect:** the first three candidates are `True` and `keep`; `next heart rate` and `centered 3-reading mean` are `False` and `reject`.


## 8. A chronological holdout

To test a score honestly, build it on the earlier rows and hold out the rows from the prediction time onward, the way it would meet later patients.

```python
vitals['block'] = np.where(vitals['recorded_at'] < prediction_time, 'earlier', 'later_holdout')
display(vitals[['patient_id', 'recorded_at', 'block']])
display(pd.crosstab(vitals['patient_id'], vitals['block']))
```

**Expect:** 9 `earlier` rows and 4 `later_holdout` rows: 3 and 2 for P01 (its 08:00 reading is held out, because `<` keeps the cutoff itself out of `earlier`), 3 and 1 for P02, and 3 and 1 for P03.


## 9. Plot the panel in UTC

One line per patient with seaborn's `hue=` (Lecture 07), a dashed reference line at 100 bpm, the usual threshold for tachycardia (a fast heart rate), and the time zone in the axis label.

```python
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=vitals, x='recorded_at', y='heart_rate', hue='patient_id', marker='o', ax=ax)
ax.axhline(100, color='red', linestyle='--', label='Tachycardia (100 bpm)')
ax.set(title='Charted heart rate, night of November 2 to 3, 2024',
       xlabel='Time (UTC), November 3, 2024', ylabel='Heart rate (bpm)', ylim=(40, 125))
ax.legend()
plt.tight_layout()
plt.show()
```

**Expect:** P01's line climbs across the dashed line between 03:30 and 04:30 UTC and keeps rising to 118; P03's starts above it and falls below; P02's stays near 70. There is no P02 point between 04:00 and 07:00 UTC, where the ambiguous reading was set aside.


## 10. Three years of flu-like illness visits

Independent reference: `infer_freq()` below uses Frequency Inference from BONUS.md; the date plots, reporting counts, and rolling mean use the main lecture.

The emergency department counts visits for influenza-like illness (fever with cough or sore throat) every day. Visits peak each winter. The reporting feed failed for two weeks in February 2023, so those days have no rows at all.

```python
rng = np.random.default_rng(42)
days = pd.date_range('2021-01-01', '2023-12-31', freq='D')
# Practice data only: np.cos makes a wave that repeats every 365.25 days and peaks on day 15 (mid-January)
winter_wave = 20 * np.cos(2 * np.pi * (days.dayofyear - 15) / 365.25)
noise = rng.normal(0, 4, len(days))  # mean 0, standard deviation 4 visits
ili = pd.Series((40 + winter_wave + noise).round().astype(int), index=days)

outage = pd.date_range('2023-02-01', '2023-02-14', freq='D')
ili = ili[~ili.index.isin(outage)]
print(len(ili), 'days reported')
print('Frequency:', pd.infer_freq(ili.index))
display(ili.resample('W').count().loc['2023-01-29':'2023-02-19'])
```

**Expect:** `1081 days reported` (1,095 days minus the 14-day outage), and `Frequency: None`, because the missing days break the daily spacing. The weekly counts of reported days drop to `2`, `0`, and `5` for the weeks ending February 5, 12, and 19. A weekly _sum_ of visits would fall the same way, which is a reporting gap, not fewer patients.

Plotted as is, the line draws straight across the outage. `asfreq('D')` puts the missing days back as `NaN`, and the line breaks there instead.

```python
winter = ili.loc['2022-12-01':'2023-03-31']
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
winter.plot(ax=axes[0], color='gray', title='As reported: the line bridges the outage')
winter.asfreq('D').plot(ax=axes[1], color='gray', title="After asfreq('D'): the gap shows")
for ax in axes:
    ax.set(xlabel='Date', ylabel='Visits per day')
plt.tight_layout()
plt.show()
```

**Expect:** the top panel shows a straight segment from January 31 to February 15; the bottom panel shows the same data with a two-week break.

Over all three years, a 28-day time window smooths the daily counts. A time window uses whatever days were reported in the last 28 days, so the outage does not shift it the way a 28-row window would.

```python
fig, ax = plt.subplots(figsize=(10, 4))
ili.asfreq('D').plot(ax=ax, color='gray', alpha=0.5, label='Daily visits')
ili.rolling('28D').mean().plot(ax=ax, color='blue', linewidth=2, label='28-day mean')
ax.axhline(55, color='red', linestyle='--', label='Winter surge level (55 per day)')
ax.set(title='Influenza-like illness visits', xlabel='Date', ylabel='Visits per day', ylim=(0, 100))
ax.legend()
plt.tight_layout()
plt.show()
```

**Expect:** three winter peaks near 60 visits per day and three summer troughs near 20; the blue 28-day mean crosses the dashed surge line each winter.

Averaging by calendar month shows the seasonal pattern directly. Bars go on a new Axes of their own, and they start at zero, which suits counts.

```python
monthly_mean = ili.groupby(ili.index.month).mean()
display(monthly_mean.round(1))

fig, ax = plt.subplots(figsize=(8, 4))
monthly_mean.plot(kind='bar', ax=ax, rot=0, color='steelblue',
                  title='Mean daily visits by calendar month, 2021 to 2023',
                  xlabel='Month', ylabel='Visits per day')
plt.tight_layout()
plt.show()
```

**Expect:** January is highest (`60.2`) and July lowest (`20.6`); the bars fall from January to July and rise again toward December (`56.8`).


## 11. Daily weights: lags, leads, and a fluid-gain alert

Independent practice: the lag, lead, difference, and alert use the main lecture. The frequency diagnostic and `pct_change()` use the optional Frequency Inference and Percentage Changes topics in BONUS.md. This section builds its own data.

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
display(daily.head(3).round(2))
display(daily.loc['2024-03-08':'2024-03-13'].round(2))
```

**Expect:** on January 1, `prev_day`, `change_1d`, and `pct_1d` are `NaN` (nothing came before), and `change_7d` stays `NaN` through January 7. In March, the weight climbs from 81.9 kg on March 8 to 83.9 kg on March 10 (`change_1d` 1.3) and 84.4 kg on March 11 and 12 (`change_7d` 2.3).

Apply both alert rules with a boolean filter (Lecture 04).

```python
alerts = daily[(daily['change_1d'] > 1.0) | (daily['change_7d'] > 2.0)]
display(alerts[['weight_kg', 'change_1d', 'change_7d']].round(1))
print('Largest daily change before March 9:', daily.loc[:'2024-03-08', 'change_1d'].abs().max().round(1), 'kg')
```

**Expect:** three alert days: March 10 (the 1-day rule, `1.3` kg) and March 11 and 12 (the 7-day rule, `2.3` kg). Before the episode, no day moved more than `0.4` kg, so ordinary scale noise never trips the rule.


## 12. Plot the home-weight alerts

Use `daily` and `alerts` from section 11. Marking the alert days on the raw series shows both the size of the change and when it happened.

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
