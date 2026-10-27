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

# Demo 3: Critique, redesign, and share a clinic chart

A draft dashboard chart about flu vaccination misleads its readers.

- Critique it with Tufte's checks, one problem and repair at a time.
- Redesign it so it works without color, and write its text alternative.
- Save the chart record as JSON.
- Draw a line chart with redundant cues and direct labels.
- Build an Altair chart of patient blood pressure from 2,400 synthetic patients and save its Vega-Lite specification.
- Independent practice: a lie-factor calculation, a layered Altair dashboard, small multiples, and a text alternative.

All values are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas, pyarrow (for Parquet), and altair in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0 altair==5.5.0
```

## Core walkthrough

### 1. A chart that gets the numbers right and still misleads

The prepared table has one row per clinic: the percentage of adult patients vaccinated against flu in two seasons. North started sending reminder texts before the 2025-26 season.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

uptake = pd.DataFrame({
    'clinic': ['North', 'South', 'East'],
    'adults_enrolled': [4210, 3875, 2960],
    '2024-25': [56, 57, 61],
    '2025-26': [63, 60, 62],
})
display(uptake)
```

Expect three rows: clinic, adults enrolled, and the vaccinated percentage in each season. North rose from 56% to 63%, South from 57% to 60%, and East from 61% to 62%.

This is the draft from the dashboard. Grouped bars need one x position per clinic (`np.arange`, Lecture 03), with each season's bar shifted half a bar width to either side. Look at the chart before reading the critique.

```python
clinics = uptake['clinic'].tolist()
x = np.arange(len(clinics))  # [0 1 2]: one position per clinic
width = 0.38

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x - width / 2, uptake['2024-25'], width, label='2024-25', color='red',
       hatch='//', edgecolor='black', linewidth=2)
ax.bar(x + width / 2, uptake['2025-26'], width, label='2025-26', color='green',
       hatch='//', edgecolor='black', linewidth=2)
ax.set_xticks(x, clinics)
ax.set_ylim(50, 65)
ax.grid(True, linewidth=2)
ax.set_title('Reminder texts drove a surge in vaccination at North!')
ax.legend()
plt.show()
print(clinics, x)
```

Expect `['North', 'South', 'East'] [0 1 2]` and three pairs of thick-edged bars, every bar striped the same way, with only red versus green telling the seasons apart. North's green bar looks more than twice as tall as its red one, and the y-axis has no label.

### 2. Critique it with Tufte's checks

For each problem, write what is wrong and how to repair it. The truncated baseline is the lie-factor check, and the decoration is the data-ink check; the claim and the color-only encoding come from the contract and the accessibility section.

```python
critique = [
    {'category': 'unsupported claim',
     'problem': 'The title says reminder texts caused the rise, but the table only shows that uptake rose, and it rose at South and East too.',
     'repair': 'Use a descriptive title: uptake rose at all three clinics, most at North.'},
    {'category': 'truncated baseline',
     'problem': "The y-axis starts at 50%, so North's 7-point rise (56% to 63%, 12.5% relative) is drawn as a 117% rise in bar length (lie factor 9.3).",
     'repair': 'Start the bar axis at zero with ax.set_ylim(0, 100).'},
    {'category': 'missing unit',
     'problem': 'The y-axis has no label, so readers cannot tell the values are percentages of adult patients.',
     'repair': 'Label the axis "Adults vaccinated against flu (%)" and write each value on its bar.'},
    {'category': 'color-only encoding',
     'problem': 'Only red versus green separates the seasons, a pair many colorblind readers cannot tell apart.',
     'repair': 'Use colorblind-safe colors plus a different hatch for each season.'},
    {'category': 'distracting decoration',
     'problem': 'Thick black edges, a heavy grid, and the same hatch on every bar carry no data.',
     'repair': 'Drop the edges and grid, hide the top and right spines, and keep hatches only where they mark the season.'},
]
for entry in critique:
    print(entry['category'] + ':', entry['problem'])
    print('    Repair:', entry['repair'])
```

Expect five problems, each followed by its repair, starting with `unsupported claim:`.

### 3. Redesign: zero baseline, redundant cues, labels on the bars

The contract for the redesign:

- **Question**: how did uptake change at each clinic?
- **Audience and claim**: clinic managers; descriptive.
- **Grain**: one bar is one clinic in one season.

Each repair from the critique appears in the code below, reusing `clinics`, `x`, and `width`.

```python
fig, ax = plt.subplots(figsize=(7, 4.5))
before = ax.bar(x - width / 2, uptake['2024-25'], width, label='2024-25',
                color='#E69F00', hatch='..', edgecolor='white')
after = ax.bar(x + width / 2, uptake['2025-26'], width, label='2025-26',
               color='#0072B2', hatch='//', edgecolor='white')
before_labels = ax.bar_label(before, fmt='%d%%')
after_labels = ax.bar_label(after, fmt='%d%%')

ax.set_xticks(x, clinics)
ax.set(ylim=(0, 100), xlabel='Clinic', ylabel='Adults vaccinated against flu (%)',
       title='Flu vaccination rose at all three clinics; North gained the most')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(title='Season', loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

from pathlib import Path

Path('output').mkdir(exist_ok=True)
fig.savefig('output/vaccination_redesign.png', dpi=150, bbox_inches='tight')
plt.show()
bottom, top = ax.get_ylim()                                # reads back the limits set above
print(bottom, top)
print([label.get_text() for label in after_labels])  # each label's text
```

`ax.bar_label()` returns one text label per bar, and `.get_text()` reads a label's words. Expect `0.0 100.0` and `['63%', '60%', '62%']`. Three pairs of bars rise from 0: orange dotted bars for 2024-25 and blue striped bars for 2025-26, each labeled with its value, and the legend just outside the right edge. Here the hatch is data ink: it marks the season, so the pairs stay distinguishable in grayscale.

### 4. Write the text alternative and save the chart record

A **text alternative** states the chart type, axes, main pattern, and a relevant limitation, so a reader who cannot see the chart still gets its comparison.

```python
bar_alt = ('Grouped bar chart of adults vaccinated against flu (%) at three clinics in the 2024-25 '
           'and 2025-26 seasons, on a 0 to 100% axis. All three rose: North from 56% to 63%, '
           'South from 57% to 60%, and East from 61% to 62%. These are clinic records only; they '
           "do not show that North's reminder texts caused its larger rise.")
print(bar_alt)
```

Expect the paragraph printed in full.

The chart record keeps the contract, the critique, and the text alternative beside the saved PNG. `json.dump()` writes it; `json.load()` reads it back.

```python
import json

chart_record = {
    'chart': 'output/vaccination_redesign.png',
    'question': 'How did adult flu vaccination change at each clinic between seasons?',
    'audience_and_claim': 'Clinic managers; uptake rose at all three clinics, most at North',
    'grain': 'one clinic in one season',
    'variables': {
        'clinic': 'categorical group (x)',
        'season': 'ordinal group (color and hatch)',
        'vaccinated_pct': 'quantitative measure (bar length)',
    },
    'critique_of_draft': critique,
    'text_alternative': bar_alt,
}
with open('output/vaccination_record.json', 'w', encoding='utf-8') as file:
    json.dump(chart_record, file, indent=2, ensure_ascii=False)

with open('output/vaccination_record.json', encoding='utf-8') as file:
    saved = json.load(file)
print(list(saved))
print(saved['grain'])
print(len(saved['critique_of_draft']), 'critique entries')
```

Expect the seven keys from `chart` to `text_alternative`, then `one clinic in one season` and `5 critique entries`.

### 5. A line chart with redundant cues and direct labels

A second prepared table: mean HbA1c (%) at four quarterly visits for patients in two diabetes programs. Lower is better. The format strings `'s--'` and `'o-'` give each series its own marker and line style beside its color, and `ax.text()` labels each line at its end, so the chart needs no legend.

```python
visits = [1, 2, 3, 4]
standard = [8.4, 8.2, 8.1, 8.0]
education = [8.5, 8.0, 7.7, 7.5]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(visits, standard, 's--', color='#E69F00')
ax.plot(visits, education, 'o-', color='#0072B2')
ax.text(4.08, standard[-1], 'Standard care', va='center')
ax.text(4.08, education[-1], 'Group education', va='center')
ax.annotate('0.5 points lower at visit 4', xy=(4, 7.5), xytext=(2.3, 7.4),
            arrowprops=dict(arrowstyle='->'))
ax.set(xlabel='Quarterly visit', ylabel='Mean HbA1c (%)', ylim=(7, 9), xlim=(0.8, 4.9),
       title='HbA1c fell further in the group-education program')
ax.set_xticks(visits)
ax.spines[['top', 'right']].set_visible(False)
plt.show()
print(f'{standard[-1] - education[-1]:.1f}')
```

Expect two falling lines, orange dashed squares for Standard care and blue solid circles for Group education, each labeled at its right end, and an arrow from the annotation to the last blue point. `0.5` prints. This y-axis starts at 7 rather than 0: position, not length, encodes the value, and the axis labels make the range clear.

### 6. Load the patient table

This supplied cell downloads the course's data generator if it is missing, builds `data/` with it when either table is missing, and reads both Parquet files. `generate_data.build()` uses a fixed seed, so everyone gets the same 2,400 patients.

```python
from pathlib import Path
from urllib.request import urlretrieve

REPO_RAW = 'https://raw.githubusercontent.com/christopherseaman/datasci_217/main/07/demo'
if not Path('generate_data.py').exists():
    urlretrieve(f'{REPO_RAW}/generate_data.py', 'generate_data.py')
if not (Path('data/patients.parquet').exists() and Path('data/clinic_monthly.parquet').exists()):
    import generate_data
    generate_data.build()

patients = pd.read_parquet('data/patients.parquet')
monthly = pd.read_parquet('data/clinic_monthly.parquet')
print(patients.shape, monthly.shape)
display(patients.head(3))
```

Expect `(2400, 8) (12, 4)`: 2,400 patients with clinic, sex, age, BMI, systolic BP, steps per day, and fasting glucose, plus 12 months of clinic visits.

Altair embeds every row in the chart's specification, so take a random 300 patients from North and South (`isin()`, Lecture 04; `sample()`, Lecture 05). Each Altair field carries its data type: `:Q` quantitative, `:N` nominal (categorical).

```python
bp = patients[patients['clinic'].isin(['North', 'South'])].sample(n=300, random_state=7)
display(bp['clinic'].value_counts())
for clinic in ['North', 'South']:
    rows = bp[bp['clinic'] == clinic]
    print(f"{clinic} mean age {rows['age'].mean():.1f} mean BP {rows['systolic_bp'].mean():.1f}")
```

Expect counts of North 174 and South 126, then `North mean age 54.2 mean BP 123.9` and `South mean age 61.6 mean BP 130.5`. South's patients are about 7 years older.

### 7. Altair: data, mark, and typed encodings

Color and shape both encode the clinic, so the groups survive grayscale printing. Tooltips show the values behind a point on hover, and `.interactive()` adds pan and zoom, but the title, axes, and legend stay visible without them.

```python
import altair as alt

scatter = alt.Chart(bp).mark_point(filled=True, size=40, opacity=0.6).encode(
    x=alt.X('age:Q', title='Age (years)', scale=alt.Scale(zero=False)),
    y=alt.Y('systolic_bp:Q', title='Systolic BP (mmHg)', scale=alt.Scale(zero=False)),
    color=alt.Color('clinic:N', title='Clinic'),
    shape=alt.Shape('clinic:N', title='Clinic'),
    tooltip=[
        alt.Tooltip('patient_id:N', title='Patient'),
        alt.Tooltip('age:Q', title='Age (years)'),
        alt.Tooltip('systolic_bp:Q', title='Systolic BP (mmHg)'),
    ],
).properties(title='Systolic BP by age', width=320, height=260).interactive()

scatter
```

Expect 300 points spread from 90 to 197 mmHg and ages 19 to 90: blue circles for North and orange squares for South. BP drifts upward with age, but at any age readings spread over 40 mmHg or more, and a few points sit above 170. Hovering over a point shows its patient ID, age, and blood pressure.

A bar chart of `mean(systolic_bp):Q` changes the grain: one bar per clinic, each the mean of its patients. Altair starts bars at zero by default, as bars need. The clinic is already named on the x-axis, so the bars share one neutral color, and the scatter's color-and-shape legend stays the only one.

```python
means = alt.Chart(bp).mark_bar(color='gray').encode(  # one color for every bar
    x=alt.X('clinic:N', title='Clinic'),
    y=alt.Y('mean(systolic_bp):Q', title='Mean systolic BP (mmHg)'),
    tooltip=[alt.Tooltip('mean(systolic_bp):Q', title='Mean systolic BP (mmHg)', format='.1f')],
).properties(title='Mean systolic BP per clinic', width=160, height=260)

combined = alt.hconcat(scatter, means)
combined
```

Expect the scatter on the left and two gray bars from 0 on the right, North 123.9 and South 130.5 mmHg in their tooltips. The bars hide what the scatter shows: South's points sit further right, at older ages, so part of its 6.6 mmHg higher mean reflects age rather than the clinic.

### 8. Save the Altair chart and check the specification

`chart.save()` writes the Vega-Lite specification with the 300 rows embedded, as JSON or as a web page that opens in any browser. `chart.to_dict()` returns the same specification as a Python dictionary.

```python
combined.save('output/bp_charts.json')
combined.save('output/bp_charts.html')
spec = combined.to_dict()

print(spec['hconcat'][0]['mark'])
print(spec['hconcat'][0]['encoding']['x'])
print(spec['hconcat'][1]['encoding']['y'])
with open('output/bp_charts.json', encoding='utf-8') as file:
    print(json.load(file)['hconcat'][1]['encoding'] == spec['hconcat'][1]['encoding'])
```

Expect three dictionaries, then `True`: the saved file holds the same specification. Each `:Q` became `'type': 'quantitative'`, and `mean(...)` became `'aggregate': 'mean'`. Open `output/bp_charts.html` in a browser to hover and zoom.

```text
{'type': 'point', 'filled': True, 'opacity': 0.6, 'size': 40}
{'field': 'age', 'scale': {'zero': False}, 'title': 'Age (years)', 'type': 'quantitative'}
{'aggregate': 'mean', 'field': 'systolic_bp', 'title': 'Mean systolic BP (mmHg)', 'type': 'quantitative'}
True
```

A shared browser chart needs its own text alternative, naming both views, their units, the comparison, and the limitation.

```python
bp_alt = ('Two views of 300 synthetic patients from North and South clinics. Left: scatter plot of '
          'systolic BP (mmHg) against age (years), North as circles and South as squares; BP rises '
          'with age at both clinics, with wide spread at every age. Right: bar chart of mean systolic BP '
          'per clinic, 123.9 for North and 130.5 for South. South patients are about 7 years older, '
          'so the gap partly reflects age.')
print(bp_alt)
```

Expect the paragraph printed in full. Check its two means against the bars' tooltips.

## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run the cells above again first.

### 1. Calculate the lie factor

The **lie factor** compares the change the drawing shows with the change in the data. On the draft's y-axis, which starts at 50, North's bars are 6 and 13 units tall.

```python
before_pct = uptake.set_index('clinic').loc['North', '2024-25']
after_pct = uptake.set_index('clinic').loc['North', '2025-26']
baseline = 50
shown_change = ((after_pct - baseline) - (before_pct - baseline)) / (before_pct - baseline)  # bar length
data_change = (after_pct - before_pct) / before_pct                                          # the data
print(f'Bars grow by {shown_change:.0%}')
print(f'Uptake grows by (relative) {data_change:.1%}')
print(f'Lie factor: {shown_change / data_change:.1f}')
```

Expect `Bars grow by 117%`, `Uptake grows by (relative) 12.5%`, and `Lie factor: 9.3`: the drawing exaggerates North's change about ninefold.

### 2. Stack an Altair line chart above a bar chart

`melt()` (Lecture 06) turns `uptake` into long form, one row per clinic and season. The season is ordinal, `:O`: its values have an order but are not measurements. `sort=` fixes the clinic order in the legend.

```python
uptake_long = uptake.melt(id_vars='clinic', value_vars=['2024-25', '2025-26'],
                          var_name='season', value_name='vaccinated_pct')
display(uptake_long)

clinic_color = alt.Color('clinic:N', title='Clinic', sort=['North', 'South', 'East'])
lines = alt.Chart(uptake_long).mark_line(point=True).encode(
    x=alt.X('season:O', title='Season'),
    y=alt.Y('vaccinated_pct:Q', title='Adults vaccinated (%)', scale=alt.Scale(zero=False)),
    color=clinic_color,
).properties(title='Flu vaccination by season', width=280, height=180)

season_means = alt.Chart(uptake_long).mark_bar().encode(
    x=alt.X('clinic:N', title='Clinic', sort=['North', 'South', 'East']),
    y=alt.Y('mean(vaccinated_pct):Q', title='Two-season mean (%)'),
    color=clinic_color,
).properties(width=280, height=120)

alt.vconcat(lines, season_means)
```

Expect six rows, then three two-point lines above three bars, blue for North, orange for South, and red for East. North's line climbs most steeply, from 56 to 63%. The bars reuse `clinic_color`, so both charts share one Clinic legend; North's two-season mean is 59.5%, South's 58.5%, and East's 61.5%.

`json.dumps()` turns a dictionary into a JSON string, such as a value to store in one CSV cell, and `json.loads()` turns it back.

```python
text = json.dumps({'grain': 'one clinic in one season', 'seasons': 2})
print(text)
print(json.loads(text)['seasons'])
```

Expect `{"grain": "one clinic in one season", "seasons": 2}` and `2`.

### 3. Small multiples on a shared scale

`monthly` holds a year of visit counts for North, South, and East. `sharey=True` puts the three panels on one y-scale.

```python
axes = monthly.set_index('month').plot(subplots=True, sharey=True, figsize=(7, 6),
                                       ylabel='Visits', title='Monthly clinic visits')
plt.show()
display(monthly[['North', 'South', 'East']].min())
display(monthly[['North', 'South', 'East']].max())
```

Expect three stacked panels under the title Monthly clinic visits, all on one y-scale from about 14 to 262 visits. North runs from 201 visits in July to 251 in December; East's line stays flat near the bottom of its panel. Then the minimums, North 201, South 112, East 25, and the maximums, North 251, South 134, East 37. Without `sharey=True`, East's 25 to 37 visits would fill its panel and look as variable as North's.

For an Altair version, `melt()` turns `monthly` into long form, one row per clinic and month. `pd.to_datetime()` (Lecture 05) builds a date from each month number, and `:T` marks the field as **temporal**, so the x-axis shows month names in time order.

```python
m = monthly.melt(id_vars='month', var_name='clinic', value_name='visits')
m['date'] = pd.to_datetime('2025-' + m['month'].astype('string') + '-01', format='%Y-%m-%d')
display(m.head(3))

alt.Chart(m).mark_line(point=True).encode(
    x=alt.X('date:T', title='Month'),
    y=alt.Y('visits:Q', title='Visits'),
    color=alt.Color('clinic:N', title='Clinic', sort=['North', 'South', 'East']),
).properties(title='Monthly clinic visits', width=360, height=240)
```

Expect three rows of `m`: North in months 1 to 3, with 247, 249, and 240 visits and dates 2025-01-01 to 2025-03-01. Then one chart with three lines over January to December 2025: North near the top, dipping to 201 in July, South near 112 to 134, and East flat near the bottom. All three share one y-axis, as with `sharey=True`.

### 4. Write the line chart's text alternative

Name the chart type, both programs, the pattern, and the limitation.

```python
line_alt = ('Line chart of mean HbA1c (%) at four quarterly visits for standard care and group '
            'education. Both fall; group education drops from 8.5% to 7.5% and ends 0.5 points below '
            'standard care. Patients chose their program, so the gap is descriptive, not a program effect.')
print(line_alt)
```

Expect the paragraph printed in full.
