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

**Project brief.** Dr. Renee Alvarez, VP of Ambulatory Operations at a health system with eight clinics, presents a 2025 access review to the board on January 15, 2026. The board uses it to decide where to add provider hours in 2026 and whether to fund a diabetes care-management pilot at Bayview and Excelsior. The work comes in four phases:

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

Expect `(1200000, 11)`.

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

Expect three visits, one row each, then 1,200,000 rows and as many visit IDs, so the grain is one row per visit; a patient seen ten times has ten rows. Then 193,034 distinct patients and months `2025-01` to `2025-12`. `info()` lists eleven columns: `clinic` has 1,197,017 non-null values, `wait_min` 1,164,941, `visit_length_min` all 1,200,000, and `satisfaction` 420,518 (surveys returned, 1 to 5). `describe()` shows ages from 0 to 94 and waits from 1 to 240 minutes, with a median wait of 17.0 and a mean of 20.2; visit lengths run from 10 to 374 minutes, median 44.

### 2. Missing values: how many, where, and why

The check-in kiosk records each wait; the EHR separately records `visit_length_min`, minutes from arrival at the front desk to checkout, so a visit length includes the wait. Why a wait is missing decides whether the recorded waits still describe all visits. Lecture 05's three mechanisms:

- **MCAR** (missing completely at random): unrelated to anything, like a kiosk that is switched off.
- **MAR** (missing at random): explained by a recorded column, like visit type.
- **MNAR** (missing not at random): related to the missing value itself, like a wait too long for the kiosk to record.

Count the gaps, then map the missing share by clinic and month. `pivot_table` (Lecture 06) takes the mean of a `True`/`False` column, which is the share of `True`s.

```python
import matplotlib.pyplot as plt
import seaborn as sns

print(f"No clinic recorded: {visits['clinic'].isna().sum():,}")
print(f"No wait recorded:   {visits['wait_min'].isna().sum():,}")

flagged = visits.assign(wait_missing=visits["wait_min"].isna())
missing_grid = flagged.pivot_table(index="clinic", columns="visit_month", values="wait_missing", aggfunc="mean")
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(missing_grid * 100, annot=True, fmt=".1f", cmap="Reds", ax=ax)
ax.set(title="Visits with no wait recorded (%)", xlabel="", ylabel="")
plt.show()
```

Expect 2,983 visits with no clinic and 35,059 (2.9%) with no wait. The heatmap is pale everywhere, 1.4% to 3.6%, except one cell: Excelsior in October, at 59.7%, when its kiosk was down. The outage hit every Excelsior visit that month whatever its wait, so those gaps are MCAR within the month: a recording artifact, not a pattern in the patients. Leave that cell out of the next two checks.

Compare new and follow-up visits outside the outage. `sns.barplot` (Lecture 07) draws the mean of a `True`/`False` column, the missing share.

```python
outage = (flagged["clinic"] == "Excelsior") & (flagged["visit_month"] == "2025-10")
usual = flagged[~outage]
ax = sns.barplot(data=usual, x="visit_type", y="wait_missing", errorbar=None)
ax.bar_label(ax.containers[0], fmt="%.3f")
ax.set(title="No wait recorded, outside Excelsior's October", xlabel="", ylabel="Share missing")
plt.show()
```

Expect two bars: 0.037 for new-patient visits and 0.020 for follow-up visits. New patients fill in paper forms at the desk, so the kiosk misses them nearly twice as often. That gap is MAR: `visit_type` is recorded, so a summary can be split by it.

MNAR cannot be proven from `wait_min` alone, because the waits that would show it are the ones missing. The evidence comes from a second source that records the same time span independently: the EHR's visit length. Band it with `pd.cut` (Lecture 05) and plot the missing share in each band, for follow-up visits only, so visit type cannot explain the pattern.

```python
follow_ups = usual[usual["visit_type"] == "Follow-up"].copy()
follow_ups["length_band"] = pd.cut(
    follow_ups["visit_length_min"], bins=[0, 30, 45, 60, 90, 120, 400],
    labels=["up to 30", "30 to 45", "45 to 60", "60 to 90", "90 to 120", "over 120"],
)
ax = sns.barplot(data=follow_ups, x="length_band", y="wait_missing", errorbar=None)
ax.bar_label(ax.containers[0], fmt="%.3f")
ax.set(title="No wait recorded, follow-up visits by EHR visit length", xlabel="Visit length (minutes)", ylabel="Share missing")
plt.show()
```

Expect the share flat at 0.015 for visits up to an hour and 0.026 for 60 to 90 minutes, then 0.160 for 90 to 120 minutes and 0.233 over 120. Long visits usually contain long waits, so the kiosk is losing the long waits: its sessions expire after an hour without rooming. That is MNAR, and it biases the scorecard: the recorded waits run shorter than the real ones, most at clinics with long waits. Three gaps matter for Demo 1: visits with no clinic cannot be assigned to a clinic, visits with no wait cannot count for or against the target, and the lost long waits make each clinic's share over 30 minutes a lower bound. The heatmap already hinted at this: Bayview's cells rise to 3.1% to 3.6% from September, when its Orthopedics waits grew.

### 3. Distribution and outliers

A histogram shows the shape of the waits; the IQR rule from Lecture 05 marks the extreme ones, and a box plot by department shows where they sit. `ax.axvline(x)` draws a vertical reference line at `x`.

```python
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

Expect quartiles of 11 and 26 minutes, a fence at 48.5, and 47,266 waits above it. The histogram peaks near 10 to 15 minutes and has a long right tail to 240, so a few long waits will pull each clinic's mean above its median. These are long waits, not entry errors, so they stay. In the box plot, Orthopedics sits furthest right and Behavioral Health furthest left: the department mix will matter in Demo 1.

### 4. Relationships

A pairplot of 2,000 sampled visits shows every pair of the numeric columns at once. `grid.map_lower(show_r)` calls `show_r` once per lower panel with that panel's two columns, and `x.corr(y)` writes their r in the corner.

```python
cols = ["age", "wait_min", "satisfaction"]
sample = visits[cols].sample(2000, random_state=217)
display(sample.corr().round(2))
grid = sns.pairplot(sample, corner=True, plot_kws={"s": 8, "alpha": 0.4})


def show_r(x, y, **kwargs):
    plt.gca().annotate(f"r = {x.corr(y):.2f}", xy=(0.95, 0.95), xycoords="axes fraction",
                       ha="right", va="top", backgroundcolor="white")


grid.map_lower(show_r)
plt.show()
```

Expect correlations of -0.48 between wait and satisfaction, 0.20 between age and wait, and -0.10 between age and satisfaction, printed in the table and in each lower panel's top right corner. Each r uses only the rows where both columns are recorded: 665 sampled visits have both a wait and a survey, and 1,936 have an age and a wait. The age histogram has two humps, children and adults over 50. Children cluster at short waits, which points to departments (Pediatrics) rather than age. Ratings fall as waits grow, a question the board returns to in Demo 1.

### 5. The same pairplot for the dashboard

The analytics team wants the same view on the operations dashboard, where a clinic manager can hover over a point to read its age, wait, and rating. Altair draws it as a repeated scatter and saves two files: a web page that opens in any browser, and the Vega-Lite specification the dashboard loads. A brush links the panels: `.add_params(brush)` attaches it to the chart, and `alt.condition(brush, ...)` picks one color for points inside the box and another for points outside. A repeated chart has no per-panel hook like `map_lower`, so a correlation heatmap beside it carries the r values: `stack()` turns the correlation table into one row per pair, `mark_rect()` colors each cell, `mark_text()` writes its r, and `|` places the two charts side by side.

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

r = sample.corr().rename_axis(index="row", columns="column").stack().reset_index(name="r")
base = alt.Chart(r).encode(alt.X("column:N", sort=cols), alt.Y("row:N", sort=cols))
heat = base.mark_rect().encode(alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])))
labels = base.mark_text(fontSize=14).encode(alt.Text("r:Q", format=".2f"))
dashboard = pairs | (heat + labels).properties(width=240, height=240)
dashboard.save("output/visits_pairplot.html")
dashboard.save("output/visits_pairplot.json")
dashboard
```

Expect a 3 by 3 grid of the same 2,000 visits, with the same clouds as the seaborn pairplot, and a 3 by 3 heatmap of r to its right. The diagonal panels plot each column against itself, so they are straight lines where seaborn drew histograms, and the heatmap's diagonal is dark blue at 1.00. The reddest cell is wait and satisfaction at -0.48; age and wait is pale blue at 0.20. The charts are live in Jupyter, VS Code, and Colab because `dashboard` is the cell's last line. Hovering over a point shows its three values. Drag a box in any panel, for example long waits in the wait-by-satisfaction panel: the same visits stay blue in every other panel and the rest turn gray, so you can see where the long waits fall on age and rating. Double-click to clear the box. The saved HTML page behaves the same way. The sample stays at 2,000 rows because Altair refuses more than 5,000 by default (`MaxRowsError`).

**What the pass decides:** score clinics on the share of waits over 30 minutes rather than the mean, which the tail inflates; keep the missing waits out of that share and read it as a lower bound, because the kiosk loses long waits; and break clinics down by department.

## Independent practice

Run the core walkthrough first. Each item is a follow-up request that came in after the pass.

### 6. The patient experience lead asks where surveys come back

A satisfaction mean rests only on the visits that returned a survey. Use the step 2 pattern on `satisfaction`: flag the visits with a rating, then draw the returned share by clinic.

```python
returns = visits.assign(returned=visits["satisfaction"].notna())
print(f"Surveys returned overall: {returns['returned'].mean():.3f}")
ax = sns.barplot(data=returns, y="clinic", x="returned", order=sorted(returns["clinic"].dropna().unique()), errorbar=None)
ax.bar_label(ax.containers[0], fmt="%.3f")
ax.set(title="Share of visits that returned a survey", xlabel="Share returned", ylabel="")
plt.show()
```

Expect 0.350 overall: about one visit in three returns a survey. The eight bars are nearly level, from 0.331 at Bayview and 0.332 at Excelsior to 0.380 at Marina. Every clinic's satisfaction mean rests on about a third of its visits, so Demo 1 reports the survey count beside each mean.

### 7. The data team asks whether any ages look like entry errors

Apply the IQR rule from step 3 to `age`, both fences this time, and draw them on the age histogram.

```python
a1, a3 = visits["age"].quantile([0.25, 0.75])
low, high = a1 - 1.5 * (a3 - a1), a3 + 1.5 * (a3 - a1)
print(f"Age quartiles {a1:.0f} and {a3:.0f}; fences {low:.1f} and {high:.1f}")
print(f"Ages outside the fences: {((visits['age'] < low) | (visits['age'] > high)).sum():,}")

ax = sns.histplot(data=visits, x="age", binwidth=5)
for fence in [low, high]:
    ax.axvline(fence, color="red", linestyle="--")
ax.set(title="Age per visit (red: IQR fences)", xlabel="Age (years)")
plt.show()
```

Expect quartiles of 38 and 76, fences at -19.0 and 133.0, and 0 ages outside them. The histogram shows the two humps from step 4, children and adults over 50, with both red lines well outside the bars. No age is impossible, so none is dropped. The humps spread the middle half widely, so the IQR rule finds nothing here; `describe()` in step 1 already showed the range of 0 to 94.

### 8. The dashboard team asks what the saved files hold

Read the two files step 5 saved, then chart how much of the JSON each top-level key takes.

```python
import json

html = Path("output/visits_pairplot.html").read_text()
for line in html.splitlines():
    if "<script" in line or "vegaEmbed(" in line:
        print(line.strip())

spec = json.loads(Path("output/visits_pairplot.json").read_text())
print(sorted(spec))
print(spec["hconcat"][0]["repeat"])
print(spec["params"][0]["select"])
print([len(rows) for rows in spec["datasets"].values()], "rows embedded")

sizes = pd.Series({key: len(json.dumps(value)) / 1000 for key, value in spec.items()}).sort_values()
ax = sizes.plot(kind="barh", title="Size of each top-level key in visits_pairplot.json")
ax.bar_label(ax.containers[0], fmt="%.1f")
ax.set(xlabel="KB")
plt.show()
```

Expect three `<script>` tags that load `vega@5`, `vega-lite@5.20.1`, and `vega-embed@6` from `cdn.jsdelivr.net`, then the page's own `<script>` and its `vegaEmbed("#vis", spec, embedOpt)` call, which draws the chart in the `#vis` box. The page needs internet access to fetch those three libraries. The JSON's top-level keys are `$schema`, `config`, `datasets`, `hconcat`, and `params`: `hconcat` holds the two charts side by side, the repeated scatter first, whose `repeat` lists the three columns for the rows and the columns; `params` holds the brush, an interval selection shared by all nine panels, printed as `{'type': 'interval', 'resolve': 'global'}`; and `datasets` holds two tables, `[2000, 9]`: the 2,000 sampled visits and the 9 r values. In the bar chart, `datasets` takes 105.3 KB and every other key under 2 KB, so the embedded rows are almost the whole file (about 107 KB).
