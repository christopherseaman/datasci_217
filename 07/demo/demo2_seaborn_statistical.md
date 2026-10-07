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

# Demo 2: pandas plotting, seaborn, and density

- seaborn on real health-spending data.
- Quick pandas plots of clinic visit tables.
- A correlation matrix saved to CSV.
- A check on what a seaborn line actually averages.
- Density plots of fasting glucose.

The clinic tables and glucose values are synthetic; `healthexp` is real OECD data, supplied as a CSV for this demo.

Run the cells from top to bottom.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

```python
from pathlib import Path
from urllib.request import urlretrieve

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/07/demo"
if not Path("healthexp.csv").exists():
    urlretrieve(f"{REPO_RAW}/healthexp.csv", "healthexp.csv")
print("Ready: healthexp.csv")
```

Expect `Ready: healthexp.csv`.

## Core walkthrough

### 1. seaborn on real data: health spending and life expectancy

`healthexp` has one row per country per year: health spending per person in US dollars and life expectancy in years, for six countries from 1970 to 2020.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')
health = pd.read_csv('healthexp.csv')
print(health.shape)
display(health.head())
print(health['Country'].nunique(), 'countries,', health['Year'].min(), 'to', health['Year'].max())
display(health.loc[(health['Country'] == 'USA') & (health['Year'] == 2020)])
```

Expect `(274, 4)`, then `6 countries, 1970 to 2020`, and one USA row for 2020 with spending of about 11,860 USD and a life expectancy of 77.0 years.

seaborn takes the DataFrame and column names, and maps each column to an encoding: position (`x=`, `y=`) and color (`hue=`).

```python
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=health, x='Spending_USD', y='Life_Expectancy',
                hue='Country', style='Country', ax=ax)
ax.set(title='Health spending and life expectancy, six countries, 1970-2020',
       xlabel='Health spending per person (USD)', ylabel='Life expectancy (years)')
plt.show()
```

Expect one point per country-year, with color and shape identifying the country. USA observations extend farthest right without having the highest life expectancy. Shared time trends and country differences prevent a causal claim from this comparison.

## Independent practice

Continue on your own after class. If the runtime closed, run the cells above again first.

### 1. pandas `.plot()`: the index becomes the x-axis

One row per week of flu visits, one column per clinic. `df.plot()` draws the index on the x-axis and one line per numeric column.

_Intentional mistake:_ plot the table before setting a meaningful index.

```python
weekly_raw = pd.DataFrame({
    'week': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'North': [12, 15, 21, 30, 42, 51, 47, 36, 24, 17],
    'South': [10, 11, 14, 18, 23, 29, 34, 33, 27, 20],
})
display(weekly_raw.head())
ax = weekly_raw.plot(marker='o', ylabel='Flu visits', title='Wrong: row numbers on x')
plt.show()
print(weekly_raw.index.tolist())
```

Expect three lines, not two. The x-axis runs 0 to 9, the row numbers printed above, and `week` is drawn as a third line climbing from 1 to 10, because it is a numeric column.

**The fix:** make `week` the index with `set_index()` (Lecture 06), assigning the result to a new name.

```python
weekly = weekly_raw.set_index('week')
ax = weekly.plot(marker='o', ylabel='Flu visits', title='Weekly flu visits by clinic')
plt.show()
print(ax.get_xlabel())
```

Expect two lines, North and South, over weeks 1 to 10, and `week` printed as the x-axis label.

### 2. One table, four views

`kind=` picks the mark, and `ax=` draws into one panel of a `plt.subplots()` grid.

```python
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
weekly.plot(ax=axes[0, 0], marker='o', ylabel='Flu visits', title='Line: visits each week')
weekly.plot(kind='bar', ax=axes[0, 1], ylabel='Flu visits', title='Bar: visits each week')
weekly.plot(kind='box', ax=axes[1, 0], ylabel='Flu visits', title='Box: spread of weekly visits')
weekly.plot(kind='hist', bins=8, alpha=0.6, ax=axes[1, 1], title='Histogram: weekly visit counts')
axes[1, 1].set_xlabel('Flu visits in a week')
fig.tight_layout()
plt.show()
```

Expect four panels from the same ten rows. The bar panel has ten pairs of bars, one pair per week, starting at 0. The box panel shows North's box taller and higher than South's, because North's weekly counts vary more.

### 3. Small multiples with a shared y-axis

`subplots=True` gives each clinic its own panel; `sharey=True` puts every panel on one y-scale, so the levels compare directly.

```python
monthly = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'North': [410, 395, 380, 350, 330, 320],
    'South': [260, 270, 265, 280, 290, 300],
    'East': [150, 160, 170, 165, 180, 175],
}).set_index('month')
display(monthly)

axes = monthly.plot(subplots=True, sharey=True, figsize=(8, 7),
                    title='Primary-care visits by month', ylabel='Visits', grid=True)
plt.tight_layout()
plt.show()
print(axes.shape)
print(axes[0].get_ylim() == axes[2].get_ylim())  # get_ylim() reads back the limits set_ylim() sets
```

Expect three stacked panels on the same y-scale: North high and falling, South rising, East lowest. `(3,)` and `True` print: one Axes per clinic, and North's and East's panels share limits.

### 4. Country trends and distributions

Continue comparing the same country-year table with line and box plots.

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.lineplot(data=health, x='Year', y='Spending_USD', hue='Country', legend=False, ax=axes[0])
axes[0].set(ylabel='Health spending per person (USD)')

sns.boxplot(data=health, x='Life_Expectancy', y='Country', ax=axes[1])
axes[1].set(xlabel='Life expectancy (years)', ylabel='')
fig.tight_layout()
plt.show()
```

Expect two panels: health spending rises over the years at all six countries, steepest in the USA; the life-expectancy boxes compare the country-year distributions. These summaries describe observations, not a spending effect.

### 5. A correlation matrix, its heatmap, and a CSV

`corr()` needs numeric columns. `Country` holds words, so asking for every column fails.

_Intentional error:_ correlate the whole table. `try`/`except` (Lecture 02) catches the error so the notebook keeps running.

```python
try:
    health.corr()
except ValueError as error:
    print('ValueError:', error)
```

Expect a `ValueError` that mentions a country name such as `'Germany'`: pandas cannot turn that text into a number.

**The fix:** list the numeric columns yourself.

```python
corr = health[['Year', 'Spending_USD', 'Life_Expectancy']].corr()
display(corr.round(2))
```

Expect a 3-by-3 table with `1.0` down the diagonal, `0.83` for Year and spending, `0.90` for Year and life expectancy, and `0.58` for spending and life expectancy. All three rise together over time; a correlation measures straight-line association, not cause.

A heatmap turns the same numbers into color. For correlations, a two-hue palette centered at 0 over the full range from -1 to 1 keeps the colors honest.

```python
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlations, six countries, 1970-2020')
plt.show()
```

Expect nine cells with their values written in them, all in shades of red because every correlation is positive; the color bar runs from blue at -1 through white at 0 to dark red at 1.

The row labels are data here, so keep them when saving, and name the column they land in.

```python
corr.round(3).to_csv('health_corr.csv', index=True, index_label='feature')
with open('health_corr.csv', encoding='utf-8') as file:
    print(file.read())
```

Expect four lines: a header starting `feature,Year,Spending_USD,Life_Expectancy`, then one line per variable, such as `Spending_USD,0.826,1.0,0.579`.

### 6. Watch the grain: what a seaborn line averages

Three patients each have a systolic reading in each of four weeks. `sns.lineplot()` draws one value per week: the mean of the three readings.

```python
readings = pd.DataFrame({
    'week': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'patient_id': ['P01', 'P02', 'P03'] * 4,
    'systolic_bp': [152, 138, 145, 148, 136, 141, 143, 135, 139, 140, 131, 136],
})
display(readings.head(6))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
sns.scatterplot(data=readings, x='week', y='systolic_bp', ax=axes[0])
axes[0].set(title='One point per reading', xlabel='Week', ylabel='Systolic BP (mmHg)')
axes[0].set_xticks([1, 2, 3, 4])
sns.lineplot(data=readings, x='week', y='systolic_bp', errorbar=None, marker='o', ax=axes[1])
axes[1].set(title='One point per week: the mean', xlabel='Week', ylabel='Mean systolic BP (mmHg)')
axes[1].set_xticks([1, 2, 3, 4])
plt.show()

line_y = axes[1].get_lines()[0].get_ydata()
by_hand = [readings.loc[readings['week'] == w, 'systolic_bp'].mean() for w in [1, 2, 3, 4]]
display(pd.DataFrame({'line_y': line_y, 'mean_by_hand': by_hand}, index=[1, 2, 3, 4]))
```

Expect 12 points on the left and a four-point line on the right. The table lists the line's y-values beside the weekly means computed by hand; the two columns match (145.0, 141.67, 139.0, 135.67). The unit displayed changed from one reading to a weekly mean, so the right panel's axis label says so.

### 7. Density plots: a distribution a mean would hide

Fasting glucose in a clinic that serves people with and without diabetes: 300 readings centered near 95 mg/dL and 100 near 165 mg/dL. `95 + 8 * rng.standard_normal(300)` gives 300 bell-curve values around 95 that spread by about 8 mg/dL, and `pd.concat(..., ignore_index=True)` (Lecture 06) stacks the two groups into one Series.

```python
rng = np.random.default_rng(42)
without_diabetes = pd.Series(95 + 8 * rng.standard_normal(300))   # mg/dL
with_diabetes = pd.Series(165 + 25 * rng.standard_normal(100))    # mg/dL
glucose = pd.concat([without_diabetes, with_diabetes], ignore_index=True)
display(pd.Series({'count': len(glucose), 'mean': round(glucose.mean(), 1), 'median': round(glucose.median(), 1)}, dtype=object))
```

Expect a table with count 400, mean 112.9 mg/dL, and median 97.5 mg/dL. Neither number shows that there are two groups; a density plot does.

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

sns.histplot(x=glucose, kde=True, ax=axes[0])
axes[0].set(title='Histogram + density', xlabel='Fasting glucose (mg/dL)')

sns.kdeplot(x=glucose, bw_adjust=0.3, ax=axes[1], label='bw_adjust=0.3')
sns.kdeplot(x=glucose, bw_adjust=2, ax=axes[1], label='bw_adjust=2')
axes[1].set(title='Bandwidth changes the story', xlabel='Fasting glucose (mg/dL)')
axes[1].legend()

glucose.plot.density(ax=axes[2], title='pandas density (uses SciPy)')
axes[2].set_xlabel('Fasting glucose (mg/dL)')

fig.tight_layout()
plt.show()
```

Expect a tall peak near 95 and a lower, wider hump around 170 to 180 mg/dL in the left and right panels. In the middle, `bw_adjust=0.3` keeps both peaks but adds small wiggles, while `bw_adjust=2` blurs them into one broad hump with a flat shoulder, so the second group is easy to miss. The middle and right y-axes are density, not a count; the left panel keeps the histogram's counts and scales its curve to match.
