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

# Demo 1: Contract first, then matplotlib

Two prepared clinic tables (readings and weekly flu visits), then charts drawn with `fig, ax = plt.subplots()` and Axes methods. The patient IDs and values are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

## Core walkthrough

### 1. Meet the plotting tables

The first table has one row per systolic blood-pressure reading: 20 patients at each of two clinics.

- `rng` makes the same "random" values on every run (Lecture 03), so your numbers match the ones below.
- Each reading rises with age, runs 6 mmHg higher at South, and varies by `rng.standard_normal(40) * 8`: 40 bell-curve values spread around 0 by about 8 mmHg.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown

rng = np.random.default_rng(42)
ages = rng.integers(30, 80, size=40)             # 30 through 79 years
noise = rng.standard_normal(40) * 8              # reading-to-reading variation, mmHg
clinic = np.array(['North'] * 20 + ['South'] * 20)
systolic = 95 + 0.6 * ages + np.where(clinic == 'South', 6, 0) + noise

readings = pd.DataFrame({
    'patient_id': np.arange(1001, 1041),
    'clinic': clinic,
    'age': ages,
    'systolic_bp': systolic.round().astype(int),
})
print(readings.shape)
display(readings.head())
display(readings.dtypes)
```

Expect `(40, 4)`, five rows starting with patient `1001` (North, age 34, 114 mmHg), and three `int64` columns plus `clinic` as `str`.

The second table is already one row per week: flu visits at each clinic over a ten-week season.

```python
weekly = pd.DataFrame({
    'week': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'North': [12, 15, 21, 30, 42, 51, 47, 36, 24, 17],
    'South': [10, 11, 14, 18, 23, 29, 34, 33, 27, 20],
})
display(weekly)
```

Expect ten rows. North peaks at 51 visits in week 6; South peaks at 34 in week 7.

### 2. Customize: units, a legend, and a redundant cue

- **Question**: does BP vary with age and clinic?
- **Marks**: one point is one reading; age and BP are quantitative, clinic is nominal.
- **Claim**: descriptive only; these synthetic rows support no causal claim.
- **Encoding**: each clinic is its own series with a color **and** a marker shape, so the groups stay distinct in grayscale.

```python
north = readings.loc[readings['clinic'] == 'North']
south = readings.loc[readings['clinic'] == 'South']

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(north['age'], north['systolic_bp'], color='#0072B2', marker='o', alpha=0.7, label='North')
ax.scatter(south['age'], south['systolic_bp'], color='#D55E00', marker='s', alpha=0.7, label='South')
ax.set(title='Systolic BP by age at two clinics', xlabel='Age (years)', ylabel='Systolic BP (mmHg)')
ax.grid(alpha=0.3)
ax.legend(title='Clinic', loc='lower right')
plt.show()
print(len(north), 'North points and', len(south), 'South points')
```

Expect blue circles and orange squares that both rise with age, a legend titled Clinic in the empty lower-right corner, and `20 North points and 20 South points`. At similar ages the orange squares tend to sit higher.

## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run the cells above again first.

### 1. Write the contract before plotting

Three `int64` columns do not mean three measures. The contract records what each column means and which job it does, and that is what picks the chart.

```python
contract = {
    'question': 'How do systolic readings compare between the North and South clinics?',
    'audience_and_claim': 'Clinic managers; a descriptive comparison of levels and spread, not a cause',
    'unit_and_grain': 'One source row per systolic reading at one visit; one histogram bar per pressure bin, '
                      'and one box summarizing the readings at each clinic',
    'variables': 'systolic_bp: quantitative measure (y); clinic: categorical group (x); '
                 'age: quantitative, not used here; patient_id: identifier, never averaged or plotted',
}
display(Markdown('\n'.join(f'- **{part}**: {answer}' for part, answer in contract.items())))
```

Expect a four-item list, one per part of the contract.

The question compares a distribution across two groups, so the chart-selection figure in the lecture points to a **box plot** (one box per clinic), with a **histogram** to check the overall shape first. The other questions in this demo pick other charts:

| Question | Data types | Chart |
| --- | --- | --- |
| How are all 40 readings distributed? | One quantitative | Histogram |
| How do readings compare by clinic? | Quantitative by categorical | Box plot |
| Does systolic BP rise with age? | Two quantitative | Scatter plot |
| How many flu visits did each clinic have in total? | Quantitative by categorical, one value each | Bar chart from zero |
| How did weekly flu visits change? | Quantitative over temporal | Line chart |

### 2. An exploratory look: one Figure, four Axes

`plt.subplots(2, 2)` returns one Figure and a 2-D array of Axes, so `axes[0, 1]` is row 0, column 1. The two clinics' readings come from boolean selection with `.loc` (Lecture 04).

```python
north_bp = readings.loc[readings['clinic'] == 'North', 'systolic_bp']
south_bp = readings.loc[readings['clinic'] == 'South', 'systolic_bp']
display(pd.DataFrame({'count': [north_bp.count(), south_bp.count()], 'median': [north_bp.median(), south_bp.median()]}, index=['North', 'South']))
display(weekly[['North', 'South']].sum())
```

Expect medians of 130.5 mmHg (North) and 139.0 mmHg (South) over 20 readings each, and flu-visit totals of 295 (North) and 219 (South).

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
print(type(fig))
print(axes.shape)

axes[0, 0].hist(readings['systolic_bp'], bins=10)
axes[0, 0].set(title='All readings', xlabel='Systolic BP (mmHg)', ylabel='Readings')

axes[0, 1].boxplot([north_bp, south_bp], tick_labels=['North', 'South'])
axes[0, 1].set(title='Readings by clinic', xlabel='Clinic', ylabel='Systolic BP (mmHg)')

axes[1, 0].scatter(readings['age'], readings['systolic_bp'], alpha=0.6)
axes[1, 0].set(title='Systolic BP by age', xlabel='Age (years)', ylabel='Systolic BP (mmHg)')

axes[1, 1].bar(['North', 'South'], [weekly['North'].sum(), weekly['South'].sum()], color='steelblue')
axes[1, 1].set(title='Flu visits, weeks 1-10', xlabel='Clinic', ylabel='Flu visits')

fig.tight_layout()
plt.show()
```

Expect `<class 'matplotlib.figure.Figure'>`, `(2, 2)`, and four panels. The histogram spans about 110 to 156 mmHg, with most readings between 125 and 145. South's box sits higher, with its middle line at the 139.0 median; the two open circles below it are readings beyond its whisker. The scatter drifts upward from left to right. Both bars start at 0 and reach 295 and 219, in one color, because the clinic is already named on the x-axis.

This is exploratory work: quick, but with honest scales and units on every axis.

### 3. An explanatory chart: annotate, declutter, and save

Now one finding for one audience. Write its contract, then build the chart it asks for.

```python
flu_contract = {
    'question': 'When did flu visits peak at each clinic?',
    'audience_and_claim': 'Clinic managers; North peaked a week earlier and higher than South',
    'unit_and_grain': 'One point per clinic per week',
    'variables': 'week: temporal order (x); North and South: quantitative visit counts (y), one line each',
}
display(Markdown('\n'.join(f'- **{part}**: {answer}' for part, answer in flu_contract.items())))
```

Expect a four-item list, one per part of the contract.

The line chart pairs each color with its own marker and line style, points an arrow at North's peak, hides the two frame lines that carry no data, and puts the legend outside the plotting area.

```python
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(weekly['week'], weekly['North'], color='#0072B2', marker='o', linestyle='-', label='North')
ax.plot(weekly['week'], weekly['South'], color='#D55E00', marker='s', linestyle='--', label='South')
ax.annotate('North peak: 51 visits', xy=(6, 51), xytext=(7.2, 56),
            arrowprops=dict(arrowstyle='->'))
ax.set(title='North flu visits peaked a week earlier and higher than South',
       xlabel='Week of season', ylabel='Flu visits', ylim=(0, 60))
ax.set_xticks(weekly['week'])
ax.spines[['top', 'right']].set_visible(False)
ax.legend(title='Clinic', loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

fig.savefig('weekly_flu_visits.png', dpi=150, bbox_inches='tight')
plt.show()

from pathlib import Path

print(Path('weekly_flu_visits.png').exists())
```

Expect a blue solid line with circles and an orange dashed line with squares, an arrow from "North peak: 51 visits" to the week-6 point, ticks at weeks 1 through 10, no top or right frame line, and the legend just outside the right edge. The last line prints `True`: `weekly_flu_visits.png` is in the notebook's folder (in Colab, open the folder icon in the left sidebar).
