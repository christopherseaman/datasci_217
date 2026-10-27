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

- Quick pandas plots of clinic visit tables.
- seaborn on real health-spending data.
- Density plots of fasting glucose.
- Counts, correlations, and heatmaps for 2,400 patients.
- A check on what a seaborn line or bar actually averages.

The patient and clinic tables are synthetic; `healthexp` is real OECD data, supplied as a Parquet file for this demo.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

This supplied cell downloads `healthexp.parquet` and `generate_data.py` when they are missing, runs `generate_data.build()` to write the two synthetic tables into `data/` if they are not there yet, then loads all three tables.

```python
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/07/demo"
for filename in ["healthexp.parquet", "generate_data.py"]:
    if not Path(filename).exists():
        urlretrieve(f"{REPO_RAW}/{filename}", filename)
if not (Path("data/patients.parquet").exists() and Path("data/clinic_monthly.parquet").exists()):
    import generate_data
    generate_data.build()

patients = pd.read_parquet("data/patients.parquet")
clinic_monthly = pd.read_parquet("data/clinic_monthly.parquet")
health = pd.read_parquet("healthexp.parquet")
print(patients.shape, clinic_monthly.shape, health.shape)
```

Expect `(2400, 8) (12, 4) (274, 4)`: 2,400 patients, 12 months of visits, and 274 country-years.

## Core walkthrough

### 1. pandas `.plot()`: the index becomes the x-axis

One row per week of flu visits, one column per clinic. `df.plot()` draws the index on the x-axis and one line per numeric column.

Make `week` the index with `set_index()` (Lecture 06), so each clinic becomes one line.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
weekly_raw = pd.DataFrame({
    'week': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'North': [12, 15, 21, 30, 42, 51, 47, 36, 24, 17],
    'South': [10, 11, 14, 18, 23, 29, 34, 33, 27, 20],
})
display(weekly_raw.head())
weekly = weekly_raw.set_index('week')
ax = weekly.plot(marker='o', ylabel='Flu visits', title='Weekly flu visits by clinic')
plt.show()
print(ax.get_xlabel())
```

Expect two lines, North and South, over weeks 1 to 10, and `week` printed as the x-axis label.

_Intentional mistake:_ compare that result with plotting the table before setting its index.

```python
ax = weekly_raw.plot(marker='o', ylabel='Flu visits', title='Wrong: row numbers on x')
plt.show()
print(weekly_raw.index.tolist())
```

Expect three lines, not two. The x-axis runs 0 to 9, and `week` is drawn as a third line climbing from 1 to 10. The `weekly` table above fixes both problems by making `week` the index.

### 2. seaborn on real data: health spending and life expectancy

`healthexp` has one row per country per year with data: health spending per person in US dollars and life expectancy in years, for six countries from 1970 to 2020 (274 rows; France reports every 5 years before 1990, and Canada starts in 1971).

```python
display(health.head())
print(health['Country'].nunique(), 'countries,', health['Year'].min(), 'to', health['Year'].max())
display(health.loc[(health['Country'] == 'USA') & (health['Year'] == 2020)])
```

Expect `6 countries, 1970 to 2020`, and one USA row for 2020 with spending of about 11,860 USD and a life expectancy of 77.0 years.

seaborn takes the DataFrame and column names, and maps each column to an encoding: position (`x=`, `y=`) and color (`hue=`).

```python
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=health, x='Spending_USD', y='Life_Expectancy',
                hue='Country', style='Country', ax=ax)
ax.set(title='Health spending and life expectancy, six countries, 1970-2020',
       xlabel='Health spending per person (USD)', ylabel='Life expectancy (years)')
ax.legend(title='Country', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
```

Expect one point per country-year, with color and shape identifying the country. USA observations extend farthest right without having the highest life expectancy. Shared time trends and country differences prevent a causal claim from this comparison.

### 3. Density plots: a distribution a mean would hide

`patients['fasting_glucose']` holds one fasting glucose reading per patient, from a clinic network that serves people with and without diabetes.

```python
glucose = patients['fasting_glucose']
display(patients.head())
print(f'{len(glucose)} readings; mean {glucose.mean():.1f}, median {glucose.median():.1f}')
```

Expect `2400 readings; mean 107.4, median 96.0` (mg/dL). Neither number shows that there are two groups; a density plot does.

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

sns.histplot(x=glucose, kde=True, ax=axes[0])
axes[0].set(title='Histogram + density', xlabel='Fasting glucose (mg/dL)')

sns.kdeplot(x=glucose, ax=axes[1], label='default')
sns.kdeplot(x=glucose, bw_adjust=0.3, ax=axes[1], label='bw_adjust=0.3')
sns.kdeplot(x=glucose, bw_adjust=3, ax=axes[1], label='bw_adjust=3')
axes[1].set(title='Bandwidth changes the story', xlabel='Fasting glucose (mg/dL)')
axes[1].legend()

glucose.plot.density(ax=axes[2], title='pandas density (uses SciPy)')
axes[2].set_xlabel('Fasting glucose (mg/dL)')

fig.tight_layout()
plt.show()
```

Expect a tall peak near 95 mg/dL and a low, wide hump near 160 mg/dL in all three panels. In the middle, the default and `bw_adjust=0.3` curves both keep the two peaks, and `bw_adjust=0.3` adds small wiggles; `bw_adjust=3` blurs them into one broad hump with a shoulder, so the second group is easy to miss. The middle and right y-axes are density, not a count; the left panel keeps the histogram's counts and scales its curve to match.

## Independent practice

Continue on your own after class. If the runtime closed, run the cells above again first.

### 1. Count and compare patients

`kind='bar'` draws one group of bars per index value, here one pair per week from the `weekly` table above.

```python
ax = weekly.plot(kind='bar', ylabel='Flu visits', title='Weekly flu visits by clinic')
plt.show()
```

Expect one pair of bars per week, North in blue and South in orange, on a y-axis that starts at 0. North peaks at 51 visits in week 6.

`kind=` picks the mark, and `ax=` draws into one panel of a `plt.subplots()` grid. Each panel answers a different question about the same 2,400 rows.

```python
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
patients['clinic'].value_counts().sort_values().plot.barh(
    ax=axes[0, 0], xlabel='Patients', title='Patients per clinic')
patients.plot(kind='scatter', x='age', y='systolic_bp', alpha=0.2, ax=axes[0, 1],
              xlabel='Age (years)', ylabel='Systolic BP (mmHg)', title='Age and blood pressure')
patients[['systolic_bp']].plot(kind='box', ax=axes[1, 0], ylabel='Systolic BP (mmHg)',
                               title='Spread of systolic BP')
patients['systolic_bp'].plot(kind='hist', bins=23, ax=axes[1, 1], title='Systolic BP readings')
axes[1, 1].set_xlabel('Systolic BP (mmHg)')
fig.tight_layout()
plt.show()
```

Expect four panels. The bars run West (300), East (500), South (700), North (900), largest on top. The scatter is a wide cloud that drifts upward with age; `alpha=0.2` makes overlapping points darker. The box runs from 112 to 136 mmHg with a median of 124, and its circles above 172 mmHg are high outliers (53 patients). The histogram shows the same long right tail.

A Series has one column of values, so `kind='pie'` needs no `y=`: each clinic's count becomes one slice.

```python
ax = patients['clinic'].value_counts().plot(kind='pie', ylabel='', title='Share of patients by clinic')
plt.show()
```

Expect four slices from largest to smallest: North, South, East, West (900, 700, 500, and 300 of 2,400 patients).

### 2. Small multiples with a shared y-axis

`subplots=True` gives each clinic its own panel. Without `sharey=True`, each panel picks its own y-range, so a small clinic looks as busy as a large one.

```python
monthly = clinic_monthly.set_index('month')
display(monthly)

axes = monthly.plot(subplots=True, figsize=(8, 7),
                    title='Primary-care visits by month (own y-scales)', ylabel='Visits', grid=True)
plt.tight_layout()
plt.show()
print(axes[0].get_ylim() == axes[2].get_ylim())  # get_ylim() reads back the limits set_ylim() sets
```

Expect three stacked panels that all look equally busy: each line fills its own panel because each panel picks its own y-range (North 201 to 251 visits, East 25 to 37). `False` prints: North's and East's panels have different limits.

`sharey=True` puts every panel on one y-scale, so the levels compare directly.

```python
axes = monthly.plot(subplots=True, sharey=True, figsize=(8, 7),
                    title='Primary-care visits by month (shared y-scale)', ylabel='Visits', grid=True)
plt.tight_layout()
plt.show()
print(axes.shape)
print(axes[0].get_ylim() == axes[2].get_ylim())
```

Expect the same three panels on one shared scale up to about 260 visits: North high, South in the middle, East near the bottom with an almost flat line. `(3,)` and `True` print: one Axes per clinic, and North's and East's panels share limits.

### 3. Country trends and distributions

Continue comparing the same country-year table with line and box plots. Each line connects straight across a country's missing years.

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.lineplot(data=health, x='Year', y='Spending_USD', hue='Country', style='Country', ax=axes[0])
axes[0].set(ylabel='Health spending per person (USD)')

sns.boxplot(data=health, x='Life_Expectancy', y='Country', ax=axes[1])
axes[1].set(xlabel='Life expectancy (years)', ylabel='')
fig.tight_layout()
plt.show()
```

Expect two panels: the legend identifies each country's line by color and dash pattern; health spending rises over the years at all six countries, steepest in the USA. The life-expectancy boxes compare the country-year distributions. These summaries describe observations, not a spending effect.

### 4. A correlation matrix, its heatmap, and a CSV

`corr()` needs numeric columns. `patients` also holds text columns (`patient_id`, `clinic`, `sex`), so asking for every column fails.

_Intentional error:_ correlate the whole table. `try`/`except` (Lecture 02) catches the error so the notebook keeps running.

```python
try:
    patients.corr()
except ValueError as error:
    print('ValueError:', error)
```

Expect `ValueError: could not convert string to float: 'M'`: pandas cannot turn the text in `sex` into a number.

**The fix:** list the numeric columns yourself.

```python
corr = patients[['age', 'systolic_bp', 'bmi', 'steps_per_day', 'fasting_glucose']].corr()
display(corr)
```

Expect a 5-by-5 table with `1.000000` down the diagonal. Age goes with higher systolic BP (`0.401`) and fewer daily steps (`-0.410`); BMI also goes with fewer steps (`-0.392`). Steps is the only column with negative correlations. A correlation measures straight-line association, not cause.

A heatmap turns the same numbers into color. For correlations, a two-hue palette centered at 0 over the full range from -1 to 1 keeps the colors honest.

```python
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlations among five measures, 2,400 patients')
plt.show()
```

Expect 25 cells with their values written in them: red where two columns rise together, blue in the `steps_per_day` row and column, and pale cells near 0, such as `0.094` for systolic BP and glucose. The color bar runs from blue at -1 through white at 0 to dark red at 1.

The row labels are data here, so keep them when saving, and name the column they land in.

```python
Path('output').mkdir(exist_ok=True)
corr.to_csv('output/health_corr.csv', index=True, index_label='feature')
with open('output/health_corr.csv', encoding='utf-8') as file:
    print(file.read())
```

Expect six lines: a header starting `feature,age,systolic_bp,bmi,steps_per_day,fasting_glucose`, then one line per variable at full precision, such as `age,1.0,0.40107239526711,0.09952032367181005,...`.

`sns.heatmap()` colors any wide table of numbers, not only correlations: here one row per month and one column per clinic.

```python
fig, ax = plt.subplots(figsize=(5, 6))
sns.heatmap(data=monthly, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set(title='Visits by month and clinic', xlabel='Clinic', ylabel='Month')
plt.show()
```

Expect a 12-by-3 grid with each visit count written in its cell (`fmt='d'` prints whole numbers). North's column is the darkest, from 201 in July to 251 in December; East's column stays pale, at 25 to 37.

### 5. Watch the grain: what a seaborn line averages

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

`sns.barplot()` averages the same way: each bar is the mean systolic BP of every patient in one clinic and sex.

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
sns.barplot(data=patients, x='clinic', y='systolic_bp', hue='sex', ax=axes[0])
axes[0].set(title='Mean with 95% CI', ylabel='Mean systolic BP (mmHg)')
axes[0].legend(title='Sex', loc='lower center')
sns.barplot(data=patients, x='clinic', y='systolic_bp', hue='sex', errorbar=None, ax=axes[1])
axes[1].set(title='errorbar=None', ylabel='Mean systolic BP (mmHg)')
axes[1].legend(title='Sex', loc='lower center')
plt.show()
```

Expect eight bars, one per clinic and sex: South's are tallest (about 130 mmHg) and West's lowest (about 120). Each bar is a mean, not one reading. The black lines on the left are 95% confidence intervals for each mean; `errorbar=None` on the right removes them and leaves the means unchanged.
