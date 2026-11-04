---
jupyter:
  jupytext:
    notebook_metadata_filter: language_info
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

# Demo 0: An EDA Pass on the 2025 Visit Extract

**Project brief.** Dr. Renee Alvarez, VP of Ambulatory Operations at a health system with eight clinics, presents a 2025 access review to the board on January 15, 2026. The board uses it to decide where to add provider hours in 2026 and how large to make a diabetes care-management pilot at Bayview and Excelsior. The work comes in four phases:

- **Demo 0**: an EDA pass on the visit extract before any counting.
- **Demo 1**: the year-end wait scorecard for Dr. Alvarez's staffing memo.
- **Demo 2**: the fourth-quarter packet for the Quality Committee.
- **Demo 3**: the pilot's patient list, rebuilt every night on the secure analytics server.

**This phase's ask.** Before counting anything, the analyst walks the 2025 visit extract through the steps of EDA: grain, structure, missing values, distributions, and relationships. The extract holds 1.2 million synthetic visits, one row per visit. Each finding here shapes a choice in Demo 1.

Run the cells from top to bottom.

```python
# Installs the course's pandas, pyarrow (for Parquet), dowhy (for the data generator), and altair in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0 dowhy==0.14 altair==5.5.0
```

This supplied cell downloads `generate_data.py` when it is missing, runs `generate_data.build()` to write the tables into `data/` if they are not there yet (about 30 seconds), then loads the visit extract.

```python
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/08/demo"
if not Path("generate_data.py").exists():
    urlretrieve(f"{REPO_RAW}/generate_data.py", "generate_data.py")
if not all(Path(f"data/{name}.parquet").exists() for name in ["clinics", "visits", "labs"]):
    import generate_data
    generate_data.build()

visits = pd.read_parquet("data/visits.parquet")
print(visits.shape)
```

Expect `(1200000, 10)`.

## Core walkthrough

### 1. Grain and structure

One row should be one visit; check the ID, the columns, and their ranges.

```python
display(visits.head(3))
print(f"Rows: {len(visits):,}, distinct visit IDs: {visits['visit_id'].nunique():,}")
print(f"Distinct patients: {visits['patient_id'].nunique():,}")
print("Months:", visits["visit_month"].min(), "to", visits["visit_month"].max())
visits.info()
display(visits.describe().round(1))
```

Expect three visits, one row each, then 1,200,000 rows and as many visit IDs, so the grain is one row per visit; a patient seen ten times has ten rows. Then 193,034 distinct patients and months `2025-01` to `2025-12`. `info()` lists ten columns: `clinic` has 1,197,017 non-null values, `wait_min` 1,171,036, and `satisfaction` 420,456 (surveys returned, 1 to 5). `describe()` shows ages from 0 to 94 and waits from 1 to 240 minutes, with a median wait of 17.0 and a mean of 20.5.

### 2. Missing values, overall and by clinic

```python
print(f"No clinic recorded: {visits['clinic'].isna().sum():,}")
print(f"No wait recorded:   {visits['wait_min'].isna().sum():,}")
display(visits["wait_min"].isna().groupby(visits["clinic"]).mean().round(3))
```

Expect 2,983 visits with no clinic and 28,964 with no wait (the check-in kiosk missed them). By clinic, 1.7% to 2.2% of waits are missing everywhere except Excelsior, at 7.0%. Two gaps matter for the scorecard: visits with no clinic cannot be assigned to a clinic, and visits with no wait cannot count for or against the target. Demo 1 handles both and looks closer at Excelsior's gap.

### 3. Distribution and outliers

A histogram shows the shape of the waits; the IQR rule from Lecture 05 marks the extreme ones, and a box plot by department shows where they sit. `ax.axvline(x)` draws a vertical reference line at `x`.

```python
import matplotlib.pyplot as plt
import seaborn as sns

q1, q3 = visits["wait_min"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print(f"Quartiles {q1:.0f} and {q3:.0f} minutes; IQR upper fence {upper:.1f}")
print(f"Waits above the fence: {(visits['wait_min'] > upper).sum():,}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=visits, x="wait_min", binwidth=5, ax=axes[0])
axes[0].axvline(upper, color="red", linestyle="--")
axes[0].set(title="Wait per visit, 2025 (red: IQR fence)", xlabel="Minutes")
sns.boxplot(data=visits, y="department", x="wait_min", ax=axes[1])
axes[1].set(title="Wait by department", xlabel="Minutes", ylabel="")
plt.tight_layout()
plt.show()
```

Expect quartiles of 11 and 26 minutes, a fence at 48.5, and 53,361 waits above it. The histogram peaks near 10 to 15 minutes and has a long right tail to 240, so a few long waits will pull each clinic's mean above its median. These are long waits, not entry errors, so they stay. In the box plot, Orthopedics sits furthest right and Behavioral Health furthest left: the department mix will matter in Demo 1.

### 4. Relationships

A pairplot of 2,000 sampled visits shows every pair of the numeric columns at once.

```python
cols = ["age", "wait_min", "satisfaction"]
sample = visits[cols].sample(2000, random_state=217)
display(sample.corr().round(2))
sns.pairplot(sample, corner=True, plot_kws={"s": 8, "alpha": 0.4})
plt.show()
```

Expect correlations of -0.5 between wait and satisfaction, 0.2 between age and wait, and -0.1 between age and satisfaction. The age histogram has two humps, children and adults over 50. Children cluster at short waits, which points to departments (Pediatrics) rather than age. Ratings fall as waits grow, a question the board returns to in Demo 1.

### 5. The same pairplot for the dashboard

The analytics team wants the same view on the operations dashboard, where a clinic manager can hover over a point to read its age, wait, and rating. Altair draws it as a repeated scatter and saves two files: a web page that opens in any browser, and the Vega-Lite specification the dashboard loads. A brush links the panels: `.add_params(brush)` attaches it to the chart, and `alt.condition(brush, ...)` picks one color for points inside the box and another for points outside.

```python
import altair as alt

Path("output").mkdir(exist_ok=True)
# alt.selection_interval(): drag a box; resolve="global" shares one box across every panel
brush = alt.selection_interval(resolve="global")
pairs = alt.Chart(sample).mark_point(size=10, opacity=0.4).encode(
    alt.X(alt.repeat("column"), type="quantitative"),
    alt.Y(alt.repeat("row"), type="quantitative"),
    color=alt.condition(brush, alt.value("steelblue"), alt.value("lightgray")),
    tooltip=cols,
).add_params(brush).properties(width=150, height=150).repeat(row=cols, column=cols)
pairs.save("output/visits_pairplot.html")
pairs.save("output/visits_pairplot.json")
pairs
```

Expect a 3 by 3 grid of the same 2,000 visits, with the same clouds as the seaborn pairplot. The diagonal panels plot each column against itself, so they are straight lines where seaborn drew histograms. The grid is live in Jupyter, VS Code, and Colab because `pairs` is the cell's last line. Hovering over a point shows its three values. Drag a box in any panel, for example long waits in the wait-by-satisfaction panel: the same visits stay blue in every other panel and the rest turn gray, so you can see where the long waits fall on age and rating. Double-click to clear the box. The saved HTML page behaves the same way. The sample stays at 2,000 rows because Altair refuses more than 5,000 by default (`MaxRowsError`).

**What the pass decides:** score clinics on the share of waits over 30 minutes rather than the mean, which the tail inflates; keep the missing waits out of that share; and break clinics down by department.

## Independent practice

Run the core walkthrough first. Each item is a follow-up request that came in after the pass.

### 6. The patient experience lead asks where surveys come back

A satisfaction mean rests only on the visits that returned a survey. Use the step 2 pattern on `satisfaction`, with `notna()` for the share returned.

```python
print(f"Surveys returned overall: {visits['satisfaction'].notna().mean():.3f}")
display(visits["satisfaction"].notna().groupby(visits["clinic"]).mean().round(3))
```

Expect 0.350 overall: about one visit in three returns a survey. By clinic, the share runs from 0.331 at Bayview and 0.332 at Excelsior to 0.380 at Marina. Every clinic's satisfaction mean rests on about a third of its visits, so Demo 1 reports the survey count beside each mean.

### 7. The data team asks whether any ages look like entry errors

Apply the IQR rule from step 3 to `age`, both fences this time.

```python
a1, a3 = visits["age"].quantile([0.25, 0.75])
low, high = a1 - 1.5 * (a3 - a1), a3 + 1.5 * (a3 - a1)
print(f"Age quartiles {a1:.0f} and {a3:.0f}; fences {low:.1f} and {high:.1f}")
print(f"Ages outside the fences: {((visits['age'] < low) | (visits['age'] > high)).sum():,}")
```

Expect quartiles of 38 and 76, fences at -19.0 and 133.0, and 0 ages outside them. No age is impossible, so none is dropped. The fences fall outside 0 to 94 because the two humps from step 4 spread the middle half widely: the IQR rule finds nothing here, and `describe()` in step 1 already showed the range.

### 8. The dashboard team asks what the saved files hold

Read the two files step 5 saved.

```python
import json

html = Path("output/visits_pairplot.html").read_text()
for line in html.splitlines():
    if "<script" in line or "vegaEmbed(" in line:
        print(line.strip())

spec = json.loads(Path("output/visits_pairplot.json").read_text())
print(sorted(spec))
print(spec["repeat"])
print(spec["params"][0]["select"])
print(len(next(iter(spec["datasets"].values()))), "rows embedded")
```

Expect three `<script>` tags that load `vega@5`, `vega-lite@5.20.1`, and `vega-embed@6` from `cdn.jsdelivr.net`, then the page's own `<script>` and its `vegaEmbed("#vis", spec, embedOpt)` call, which draws the chart in the `#vis` box. The page needs internet access to fetch those three libraries. The JSON's top-level keys are `$schema`, `config`, `datasets`, `params`, `repeat`, and `spec`: `repeat` lists the three columns for the rows and the columns, `spec` is the one scatter Vega-Lite repeats, `params` holds the brush (`param_1`, an interval selection shared by all nine panels), and `datasets` holds the 2,000 rows. The brush prints as `{'type': 'interval', 'resolve': 'global'}`. Each file is about 105 KB.
