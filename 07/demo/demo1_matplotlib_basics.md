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

A clinic network hands you two prepared plotting tables: one row per blood-pressure reading, and one row per week of flu visits. You write the visualization contract first, let it pick the chart, then draw exploratory and explanatory charts with `fig, ax = plt.subplots()` and Axes methods. Everything here comes from Lecture 07 up to the first demo break, plus Lectures 01 to 06. The patient IDs and values are synthetic.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-24 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, and matplotlib 3.11.1; the whole notebook runs in a few seconds.

## Setup

Run this cell first. It installs pandas 3.0.5, the course version, into the notebook's environment: in Colab, which ships an older pandas, and in your local `.venv` alike.

- pip may print a warning that other Colab packages expect a different pandas. That is expected; this demo does not use those packages.
- If Colab asks you to restart after the install, choose **Runtime → Restart session**, then run the notebook from the top.
- Locally, the `.venv` you made with `uv venv --seed` includes pip, so `%pip` installs into it too. When pandas 3.0.5 is already there, the cell only prints `Note: you may need to restart the kernel to use updated packages.`; nothing needs doing.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

```python
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("matplotlib:", matplotlib.__version__)
```

Expect `pandas: 3.0.5`. If Colab shows an older pandas, it was imported before the install finished: restart the session and run all cells again.

## 1. Meet the plotting tables

The first table has one row per systolic blood-pressure reading: 20 patients at the North clinic and 20 at the South clinic. `rng` makes the same "random" values on every run (Lecture 03), so your numbers match the ones below.

```python
rng = np.random.default_rng(42)
ages = rng.integers(30, 80, size=40)             # 30 through 79 years
noise = rng.normal(0, 8, size=40)                # reading-to-reading variation, mmHg
clinic = np.array(['North'] * 20 + ['South'] * 20)
systolic = 95 + 0.6 * ages + np.where(clinic == 'South', 6, 0) + noise

readings = pd.DataFrame({
    'patient_id': np.arange(1001, 1041),
    'clinic': clinic,
    'age': ages,
    'systolic_bp': systolic.round().astype(int),
})
print(readings.shape)
print(readings.head())
print(readings.dtypes)
```

Expect `(40, 4)`, five rows starting with patient `1001`, and three `int64` columns plus `clinic` as `str`.

The second table is already one row per week: flu visits at each clinic over a ten-week season.

```python
weekly = pd.DataFrame({
    'week': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'North': [12, 15, 21, 30, 42, 51, 47, 36, 24, 17],
    'South': [10, 11, 14, 18, 23, 29, 34, 33, 27, 20],
})
print(weekly)
```

Expect ten rows. North peaks at 51 visits in week 6; South peaks at 34 in week 7.

## 2. Write the contract before plotting

Three `int64` columns do not mean three measures. The contract records what each column means and which job it does, and that is what picks the chart.

```python
contract = {
    'question': 'How do systolic readings compare between the North and South clinics?',
    'audience_and_claim': 'Clinic managers; a descriptive comparison of levels and spread, not a cause',
    'unit_and_grain': 'One row and one mark per systolic reading at one visit',
    'variables': 'systolic_bp: quantitative measure (y); clinic: categorical group (x); '
                 'age: quantitative, not used here; patient_id: identifier, never averaged or plotted',
}
for part, answer in contract.items():
    print(f'{part}: {answer}')
```

Expect four lines, one per part of the contract.

The question compares a distribution across two groups, so the chart-selection figure in the lecture points to a **box plot** (one box per clinic), with a **histogram** to check the overall shape first. The other questions in this demo pick other charts:

| Question | Data types | Chart |
| --- | --- | --- |
| How are all 40 readings distributed? | One quantitative | Histogram |
| How do readings compare by clinic? | Quantitative by categorical | Box plot |
| Does systolic BP rise with age? | Two quantitative | Scatter plot |
| How many flu visits did each clinic have in total? | Quantitative by categorical, one value each | Bar chart from zero |
| How did weekly flu visits change? | Quantitative over temporal | Line chart |

## 3. An exploratory look: one Figure, four Axes

`plt.subplots(2, 2)` returns one Figure and a 2-D array of Axes, so `axes[0, 1]` is row 0, column 1. The two clinics' readings come from boolean selection with `.loc` (Lecture 04).

```python
north_bp = readings.loc[readings['clinic'] == 'North', 'systolic_bp']
south_bp = readings.loc[readings['clinic'] == 'South', 'systolic_bp']
print('North median:', north_bp.median(), 'mmHg')
print('South median:', south_bp.median(), 'mmHg')
print('Total flu visits:', weekly['North'].sum(), 'North,', weekly['South'].sum(), 'South')
```

Expect `North median: 130.5 mmHg`, `South median: 139.0 mmHg`, and totals of 295 (North) and 219 (South).

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

## 4. Customize: units, a legend, and a redundant cue

The scatter above cannot tell the clinics apart. Draw each clinic as its own series, giving each a color **and** a marker shape, so the groups stay distinct in grayscale.

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

## 5. An explanatory chart: annotate, declutter, and save

Now one finding for one audience. Write its contract, then build the chart it asks for.

```python
flu_contract = {
    'question': 'When did flu visits peak at each clinic?',
    'audience_and_claim': 'Clinic managers; North peaked a week earlier and higher than South',
    'unit_and_grain': 'One point per clinic per week',
    'variables': 'week: temporal order (x); North and South: quantitative visit counts (y), one line each',
}
for part, answer in flu_contract.items():
    print(f'{part}: {answer}')
```

Expect four lines, one per part of the contract.

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
print(Path('weekly_flu_visits.png').exists())
```

Expect a blue solid line with circles and an orange dashed line with squares, an arrow from "North peak: 51 visits" to the week-6 point, ticks at weeks 1 through 10, no top or right frame line, and the legend just outside the right edge. The last line prints `True`: `weekly_flu_visits.png` is in the notebook's folder (in Colab, open the folder icon in the left sidebar).
