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

A draft dashboard chart about flu vaccination misleads its readers. You critique it with Tufte's checks, one problem and repair at a time, redesign it so it works without color, draw a line chart with redundant cues, write text alternatives, and save the chart record as JSON. Then you build an Altair chart of patient blood pressure and save its Vega-Lite specification. Everything here comes from Lecture 07 up to the last demo break, plus Lectures 01 to 06. All values are synthetic.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-24 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, matplotlib 3.11.1, and Altair 5.5.0; the whole notebook runs in a few seconds.

## Setup

Run this cell first. It installs pandas 3.0.5, the course version, into the notebook's environment: in Colab, which ships an older pandas, and in your local `.venv` alike. Colab already has Altair.

- pip may print a warning that other Colab packages expect a different pandas. That is expected; this demo does not use those packages.
- If Colab asks you to restart after the install, choose **Runtime → Restart session**, then run the notebook from the top.
- Locally, the `.venv` you made with `uv venv --seed` includes pip, so `%pip` installs into it too. When pandas 3.0.5 is already there, the cell only prints `Note: you may need to restart the kernel to use updated packages.`; nothing needs doing.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

```python
import json

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("pandas:", pd.__version__)
print("Altair:", alt.__version__)
```

Expect `pandas: 3.0.5` and an Altair version starting with `5.`. If Colab shows an older pandas, it was imported before the install finished: restart the session and run all cells again.

## 1. A chart that gets the numbers right and still misleads

The prepared table has one row per clinic: the percentage of adult patients vaccinated against flu in two seasons. North started sending reminder texts before the 2025-26 season.

```python
uptake = pd.DataFrame({
    'clinic': ['North', 'South', 'East'],
    '2024-25': [56, 57, 61],
    '2025-26': [63, 60, 62],
})
print(uptake)
```

Expect three rows. North rose from 56% to 63%, South from 57% to 60%, and East from 61% to 62%.

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

## 2. Critique it with Tufte's checks

The **lie factor** compares the change the drawing shows with the change in the data. On a y-axis that starts at 50, North's bars are 6 and 13 units tall.

```python
baseline = 50
shown_change = ((63 - baseline) - (56 - baseline)) / (56 - baseline)  # change in bar length
data_change = (63 - 56) / 56                                          # change in the data
print('Bars grow by', round(shown_change * 100), '%')
print('Uptake grows by', round(data_change * 100, 1), '%')
print('Lie factor:', round(shown_change / data_change, 1))
```

Expect `Bars grow by 117 %`, `Uptake grows by 12.5 %`, and `Lie factor: 9.3`: the drawing exaggerates North's change about ninefold.

Write the critique down: for each problem, what is wrong and how to repair it. The truncated baseline is the lie-factor check from the lecture's Tufte card, and the decoration is the data-ink check; the claim and the color-only encoding come from the contract and the accessibility section.

```python
critique = [
    {'category': 'unsupported claim',
     'problem': 'The title says reminder texts caused the rise, but the table only shows that uptake rose, and it rose at South and East too.',
     'repair': 'Use a descriptive title: uptake rose at all three clinics, most at North.'},
    {'category': 'truncated baseline',
     'problem': "The y-axis starts at 50%, so North's 12.5% rise is drawn as a 117% rise (lie factor 9.3).",
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
    print(f"{entry['category']}: {entry['problem']}")
    print(f"    repair: {entry['repair']}")
```

Expect five problems, each followed by its repair.

## 3. Redesign: zero baseline, redundant cues, labels on the bars

The contract for the redesign: the question is how uptake changed at each clinic; the audience is clinic managers; the claim is descriptive; one bar is one clinic in one season. Each repair from the critique appears in the code below, reusing `clinics`, `x`, and `width`.

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

fig.savefig('vaccination_redesign.png', dpi=150, bbox_inches='tight')
plt.show()
bottom, top = ax.get_ylim()                                # reads back the limits set above
print(bottom, top)
print([label.get_text() for label in after_labels])  # the text bar_label() wrote
```

Expect `0.0 100.0` and `['63%', '60%', '62%']`. Three pairs of bars rise from 0: orange dotted bars for 2024-25 and blue striped bars for 2025-26, each labeled with its value, and the legend just outside the right edge. Here the hatch is data ink: it marks the season, so the pairs stay distinguishable in grayscale.

## 4. A line chart with redundant cues and one annotation

A second prepared table: mean HbA1c (%) at four quarterly visits for patients in two diabetes programs. Lower is better. Each series gets its own color, marker, and line style, plus a direct label, so the chart needs no legend.

```python
visits = [1, 2, 3, 4]
standard = [8.4, 8.2, 8.1, 8.0]
education = [8.5, 8.0, 7.7, 7.5]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(visits, standard, color='#E69F00', marker='s', linestyle='--')
ax.plot(visits, education, color='#0072B2', marker='o', linestyle='-')
ax.text(4.08, standard[-1], 'Standard care', va='center')
ax.text(4.08, education[-1], 'Group education', va='center')
ax.annotate('0.5 points lower at visit 4', xy=(4, 7.5), xytext=(2.3, 7.4),
            arrowprops=dict(arrowstyle='->'))
ax.set(xlabel='Quarterly visit', ylabel='Mean HbA1c (%)', ylim=(7, 9), xlim=(0.8, 4.9),
       title='HbA1c fell further in the group-education program')
ax.set_xticks(visits)
ax.spines[['top', 'right']].set_visible(False)
plt.show()
print(round(standard[-1] - education[-1], 1))
```

Expect two falling lines, orange dashed squares for Standard care and blue solid circles for Group education, each labeled at its right end, and an arrow from the annotation to the last blue point. `0.5` prints. This y-axis starts at 7 rather than 0: a line chart need not start at zero because position, not length, encodes the value, and the axis labels make the range clear.

## 5. Write the text alternatives and save the chart record

A **text alternative** states the chart type, axes, main pattern, and a relevant limitation, so a reader who cannot see the chart still gets its comparison.

```python
bar_alt = ('Grouped bar chart of adults vaccinated against flu (%) at three clinics in the 2024-25 '
           'and 2025-26 seasons, on a 0 to 100% axis. All three rose: North from 56% to 63%, '
           'South from 57% to 60%, and East from 61% to 62%. These are clinic records only; they '
           "do not show that North's reminder texts caused its larger rise.")
line_alt = ('Line chart of mean HbA1c (%) at four quarterly visits for standard care and group '
            'education. Both fall; group education drops from 8.5% to 7.5% and ends 0.5 points below '
            'standard care. Patients chose their program, so the gap is descriptive, not a program effect.')
print(bar_alt)
print(line_alt)
```

Expect the two paragraphs printed in full.

The chart record keeps the contract, the critique, and the text alternative beside the saved PNG. `json.dump()` writes it; `json.load()` reads it back.

```python
chart_record = {
    'chart': 'vaccination_redesign.png',
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
with open('vaccination_record.json', 'w', encoding='utf-8') as file:
    json.dump(chart_record, file, indent=2, ensure_ascii=False)

with open('vaccination_record.json', encoding='utf-8') as file:
    saved = json.load(file)
print(list(saved))
print(saved['grain'])
print(len(saved['critique_of_draft']), 'critique entries')
```

Expect the seven keys from `chart` to `text_alternative`, then `one clinic in one season` and `5 critique entries`.

## 6. Altair: data, mark, and typed encodings

Twelve patients, one row each: age in years, systolic blood pressure in mmHg, and clinic. Each Altair field carries its data type: `:Q` quantitative, `:N` nominal (categorical).

```python
patients = pd.DataFrame({
    'patient_id': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06',
                   'P07', 'P08', 'P09', 'P10', 'P11', 'P12'],
    'age': [34, 45, 52, 61, 68, 74, 38, 47, 55, 63, 70, 77],
    'systolic_bp': [116, 123, 128, 134, 139, 146, 121, 130, 136, 141, 148, 155],
    'clinic': ['North'] * 6 + ['South'] * 6,
})
print(patients.shape)
```

Expect `(12, 4)`.

Color and shape both encode the clinic, so the groups survive grayscale printing. Tooltips show the values behind a point on hover, and `.interactive()` adds pan and zoom, but the title, axes, and legend stay visible without them.

```python
scatter = alt.Chart(patients).mark_point(filled=True, size=90).encode(
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

Expect twelve filled points rising from lower left to upper right: blue circles for North and orange squares for South, with South's points a few mmHg higher at similar ages. Hovering over a point shows its patient ID, age, and blood pressure.

A bar chart of `mean(systolic_bp):Q` changes the grain: one bar per clinic, each the mean of six patients. Altair starts bars at zero by default, as bars need. The clinic is already named on the x-axis, so the bars share one neutral color, and the scatter's color-and-shape legend stays the only one.

```python
means = alt.Chart(patients).mark_bar(color='gray').encode(  # one color for every bar
    x=alt.X('clinic:N', title='Clinic'),
    y=alt.Y('mean(systolic_bp):Q', title='Mean systolic BP (mmHg)'),
    tooltip=[alt.Tooltip('mean(systolic_bp):Q', title='Mean systolic BP (mmHg)', format='.1f')],
).properties(title='Mean systolic BP per clinic', width=160, height=260)

combined = alt.hconcat(scatter, means)
combined
```

Expect the scatter on the left and two gray bars from 0 on the right, North near 131 and South near 139 mmHg, with one legend of colored circles and squares. Hovering over a bar shows its mean to one decimal.

## 7. Save the Altair chart and check the specification

`chart.save()` writes the Vega-Lite specification with the twelve rows embedded, so anyone with the file can render the same chart.

```python
combined.save('bp_charts.json')

with open('bp_charts.json', encoding='utf-8') as file:
    spec = json.load(file)
print(spec['hconcat'][0]['mark'])
print(spec['hconcat'][0]['encoding']['x'])
print(spec['hconcat'][1]['encoding']['y'])
```

Expect three lines: the point mark, the scatter's x encoding, and the bar's y encoding. Each `:Q` became `'type': 'quantitative'`, and `mean(...)` became `'aggregate': 'mean'`.

```text
{'type': 'point', 'filled': True, 'size': 90}
{'field': 'age', 'scale': {'zero': False}, 'title': 'Age (years)', 'type': 'quantitative'}
{'aggregate': 'mean', 'field': 'systolic_bp', 'title': 'Mean systolic BP (mmHg)', 'type': 'quantitative'}
```

A shared browser chart needs its own text alternative, naming both views, their units, the comparison, and the limitation.

```python
bp_alt = ('Two views of twelve patients. Left: scatter plot of systolic BP (mmHg) against age (years), '
          'North as circles and South as squares; BP rises with age at both clinics, and South is a few '
          'mmHg higher at similar ages. Right: bar chart of mean systolic BP per clinic, 131.0 for North '
          'and 138.5 for South. Twelve synthetic patients describe these rows, not the clinics.')
print(bp_alt)
```

Expect the paragraph printed in full. Check its two means against the bars' tooltips.
