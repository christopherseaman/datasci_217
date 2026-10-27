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

Two prepared clinic tables (2,400 patients and weekly flu visits), then charts drawn with `fig, ax = plt.subplots()` and Axes methods. The patients and values are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

## Core walkthrough

### 1. Meet the plotting tables

This supplied cell builds the patient table. It downloads `generate_data.py` when the folder lacks it (`urlretrieve(url, path)` saves the file at a web address to `path`), runs its `build()` when a table in `data/` is missing, and reads the table. `build()` uses a fixed seed, so your values match the ones below.

```python
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/07/demo"
if not Path("generate_data.py").exists():
    urlretrieve(f"{REPO_RAW}/generate_data.py", "generate_data.py")
if not (Path("data/patients.parquet").exists() and Path("data/clinic_monthly.parquet").exists()):
    import generate_data
    generate_data.build()

patients = pd.read_parquet("data/patients.parquet")
print(patients.shape)
display(patients.head())
display(patients["clinic"].value_counts())
```

Expect `(2400, 8)`, five rows starting with patient `P0001` (North, age 38, systolic BP 99 mmHg), and clinic counts of 900 North, 700 South, 500 East, and 300 West.

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

### 2. Contract first, then a scatter plot

- **Question**: does systolic BP rise with age, and does South run higher than North?
- **Marks**: one point is one patient; age and BP are quantitative, clinic is nominal.
- **Claim**: descriptive only; these synthetic rows support no causal claim.
- **Encoding**: each clinic is its own series with a color **and** a marker shape; `alpha=0.4` lets 1,600 overlapping points show where they pile up.
- **Accessibility**: the marker shape repeats the color, so the groups stay distinct in grayscale. Text alternative: "Systolic BP rises with age at both clinics, and South's points sit a few mmHg higher."

Boolean selection with `.loc` (Lecture 04) splits the two clinics.

```python
north = patients.loc[patients['clinic'] == 'North']
south = patients.loc[patients['clinic'] == 'South']

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(north['age'], north['systolic_bp'], color='#0072B2', marker='o', s=15, alpha=0.4, label='North')
ax.scatter(south['age'], south['systolic_bp'], color='#D55E00', marker='s', s=15, alpha=0.4, label='South')
ax.set(title='Systolic BP by age at two clinics', xlabel='Age (years)', ylabel='Systolic BP (mmHg)')
ax.grid(alpha=0.3)
ax.legend(title='Clinic', loc='upper left')
plt.show()
print(len(north), 'North points and', len(south), 'South points')
```

Expect a cloud of blue circles and orange squares that drifts upward from age 20 to 90, with a few points above 170 mmHg from about age 32 up, a legend titled Clinic in the upper-left corner, and `900 North points and 700 South points`. The cloud is wide: age explains only part of BP.

### 3. A line chart: format strings, an annotation, and a saved file

The weekly table answers "when did flu visits peak at each clinic?" A format string sets the marker and line style in one argument: `'o-'` is circles joined by a solid line, `'s--'` squares joined by a dashed line. Save the Figure **before** `plt.show()`.

```python
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(weekly['week'], weekly['North'], 'o-', color='#0072B2', label='North')
ax.plot(weekly['week'], weekly['South'], 's--', color='#D55E00', label='South')
ax.annotate('North peak: 51 visits', xy=(6, 51), xytext=(7.2, 56),
            arrowprops=dict(arrowstyle='->'))
ax.set(title='North flu visits peaked a week earlier and higher than South',
       xlabel='Week of season', ylabel='Flu visits per week', ylim=(0, 60))
ax.spines[['top', 'right']].set_visible(False)
ax.legend(title='Clinic', loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

Path('output').mkdir(exist_ok=True)
fig.savefig('output/weekly_flu_visits.png', dpi=150, bbox_inches='tight')
plt.show()
print(Path('output/weekly_flu_visits.png').exists())
```

Expect a blue solid line with circles and an orange dashed line with squares, an arrow from "North peak: 51 visits" to the week-6 point, no top or right frame line, and the legend just outside the right edge. The last line prints `True`: `weekly_flu_visits.png` is in the `output/` folder beside the notebook (in Colab, open the folder icon in the left sidebar).

## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run the cells above again first.

### 1. Write the contract before plotting

Four numeric columns do not mean four measures. The contract records what each column means and which job it does, and that is what picks the chart.

```python
contract = {
    'question': 'How do systolic readings compare across the four clinics?',
    'audience_and_claim': 'Clinic managers; a descriptive comparison of levels and spread, not a cause',
    'unit_and_grain': 'One source row per patient; one histogram bar per pressure bin, '
                      'and one box summarizing the patients at each clinic',
    'variables': 'systolic_bp: quantitative measure (y); clinic: categorical group (x); '
                 'age, bmi: quantitative, not used here; patient_id: identifier, never averaged or plotted',
    'accessibility': 'Clinic names sit on the x-axis, so no color is needed; text alternative: '
                     '"Median systolic BP is highest at South, and every clinic has high outliers."',
}
display(Markdown('\n'.join(f'- **{part}**: {answer}' for part, answer in contract.items())))
```

Expect a five-item list, one per part of the contract.

The question compares a distribution across groups, so the chart-selection figure in the lecture points to a **box plot** (one box per clinic), with a **histogram** to check the overall shape first. The other questions in this demo pick other charts:

| Question | Data types | Chart |
| --- | --- | --- |
| How are all 2,400 readings distributed? | One quantitative | Histogram |
| How do readings compare by clinic? | Quantitative by categorical | Box plot |
| Does systolic BP rise with age? | Two quantitative | Scatter plot |
| How many flu visits did each clinic have in total? | Quantitative by categorical, one value each | Bar chart from zero |
| How did weekly flu visits change? | Quantitative over temporal | Line chart |

### 2. An exploratory look: one Figure, four Axes

`plt.subplots(2, 2)` returns one Figure and a 2-D array of Axes, so `axes[0, 1]` is row 0, column 1. The list comprehension (Lecture 03) collects one Series of readings per clinic for the box plot.

```python
clinics = ['North', 'South', 'East', 'West']
bp_by_clinic = [patients.loc[patients['clinic'] == c, 'systolic_bp'] for c in clinics]
display(pd.DataFrame({'count': [s.count() for s in bp_by_clinic],
                      'median': [s.median() for s in bp_by_clinic]}, index=clinics))
display(weekly[['North', 'South']].sum())
```

Expect medians of 122.0 (North), 130.0 (South), 121.0 (East), and 120.0 mmHg (West), and flu-visit totals of 295 (North) and 219 (South).

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
print(type(fig))
print(axes.shape)

axes[0, 0].hist(patients['systolic_bp'], bins=30)
axes[0, 0].set(title='All readings', xlabel='Systolic BP (mmHg)', ylabel='Patients')

axes[0, 1].boxplot(bp_by_clinic, tick_labels=clinics)
axes[0, 1].set(title='Readings by clinic', xlabel='Clinic', ylabel='Systolic BP (mmHg)')

axes[1, 0].scatter(patients['age'], patients['systolic_bp'], s=8, alpha=0.3)
axes[1, 0].set(title='Systolic BP by age', xlabel='Age (years)', ylabel='Systolic BP (mmHg)')

axes[1, 1].bar(['North', 'South'], [weekly['North'].sum(), weekly['South'].sum()], color='steelblue')
axes[1, 1].set(title='Flu visits, weeks 1-10', xlabel='Clinic', ylabel='Flu visits')

fig.tight_layout()
plt.show()
bottom, top = axes[1, 1].get_ylim()
print('Bar baseline:', bottom)
```

Expect `<class 'matplotlib.figure.Figure'>`, `(2, 2)`, and four panels. The histogram starts at 90 mmHg, peaks near 125 mmHg, and has a long right tail out to 200. South's box sits highest; open circles above every box are high outliers. The scatter drifts upward from left to right. Both bars reach 295 and 219, in one color, because the clinic is already named on the x-axis. The last line prints `Bar baseline: 0.0`: the bars start at zero, so their lengths compare honestly.

This is exploratory work: quick, but with honest scales and units on every axis.

### 3. Label the lines directly

Redraw the core flu chart for a reader with no legend to decode. Write its contract first.

```python
flu_contract = {
    'question': 'When did flu visits peak at each clinic?',
    'audience_and_claim': 'Clinic managers; North peaked a week earlier and higher than South',
    'unit_and_grain': 'One point per clinic per week',
    'variables': 'week: temporal order (x); North and South: quantitative visit counts (y), one line each',
    'accessibility': 'Marker, line style, and a name at each line end repeat the color; text alternative: '
                     '"North flu visits peaked at 51 in week 6, South at 34 in week 7."',
}
display(Markdown('\n'.join(f'- **{part}**: {answer}' for part, answer in flu_contract.items())))
```

Expect a five-item list, one per part of the contract.

`ax.text` puts each clinic's name at its line end in place of a legend. `ax.set_xticks(positions, labels)` names every week, `ax.tick_params` turns those labels, and `ax.grid(axis='y')` adds horizontal reference lines only.

```python
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(weekly['week'], weekly['North'], 'o-', color='#0072B2')
ax.plot(weekly['week'], weekly['South'], 's--', color='#D55E00')
ax.text(10.2, 17, 'North', color='#0072B2', va='center')
ax.text(10.2, 20, 'South', color='#D55E00', va='center')
ax.set_xticks(weekly['week'], [f'Week {w}' for w in weekly['week']])
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
ax.set(title='North flu visits peaked a week earlier and higher than South',
       ylabel='Flu visits per week', ylim=(0, 60))
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

Expect the same two lines with "North" and "South" written in their line colors just right of week 10, labels "Week 1" through "Week 10" tilted 45 degrees, light horizontal grid lines, and no legend.

### 4. Two other ways to build a Figure

The pyplot shortcut style draws on whichever Axes is current; it suits a one-panel sketch. `plt.style.use` changes the look of every figure made after it, so the cell switches back to `'default'` at the end.

```python
plt.style.use('ggplot')
plt.figure(figsize=(6, 3.5))
plt.plot(weekly['week'], weekly['North'], 'o-')
plt.title('North weekly flu visits')
plt.xlabel('Week of season')
plt.ylabel('Flu visits per week')
plt.show()
plt.style.use('default')
```

Expect one line on a gray background with white grid lines: the `ggplot` style.

`fig.add_subplot(rows, cols, position)` adds one Axes at a time to an existing Figure; position counts from 1, left to right.

```python
fig = plt.figure(figsize=(9, 3.5))
ax_left = fig.add_subplot(1, 2, 1)
ax_right = fig.add_subplot(1, 2, 2)
ax_left.plot(weekly['week'], weekly['North'], 'o-', color='#0072B2')
ax_left.set(title='North', xlabel='Week of season', ylabel='Flu visits per week')
ax_right.plot(weekly['week'], weekly['South'], 's--', color='#D55E00')
ax_right.set(title='South', xlabel='Week of season', ylabel='Flu visits per week')
fig.tight_layout()
plt.show()
```

Expect two panels side by side on the default white background. Each Axes picks its own y range, so South's 34-visit peak looks as tall as North's 51; `ylim=(0, 60)` on both would make them comparable.
