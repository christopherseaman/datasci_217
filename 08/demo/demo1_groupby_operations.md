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

# Demo 1: Where Patients Waited in 2025

**Project brief.** Dr. Renee Alvarez, VP of Ambulatory Operations at a health system with eight clinics, presents a 2025 access review to the board on January 15, 2026. The board uses it to decide where to add provider hours in 2026 and how large to make a diabetes care-management pilot at Bayview and Excelsior. The work comes in three phases:

- **Demo 1**: the year-end wait scorecard for Dr. Alvarez's staffing memo.
- **Demo 2**: the fourth-quarter packet for the Quality Committee.
- **Demo 3**: the pilot's patient list, rebuilt every night on the secure analytics server.

**This phase's ask.** By Friday, Dr. Alvarez wants three things: which clinics miss the system's access target (at least 85% of patients seen within 30 minutes), where inside those clinics the long waits are, and visit counts by department with totals for the board's appendix. The source is the 2025 visit extract: 1.2 million synthetic visits, one row per visit.

Run the cells from top to bottom.

```python
# Installs the course's pandas, pyarrow (for Parquet), dowhy (for the data generator), and altair in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0 dowhy==0.14 altair==5.5.0
```

This supplied cell downloads `generate_data.py` when it is missing, runs `generate_data.build()` to write the tables into `data/` if they are not there yet (about 30 seconds), then loads the visit extract.

```python
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
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

### 1. An EDA pass before counting

Before building the scorecard, walk the extract through the steps of EDA: grain, structure, missing values, distributions, and relationships. Each finding below shapes a later step.

#### Grain and structure

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

#### Missing values, overall and by clinic

```python
print(f"No clinic recorded: {visits['clinic'].isna().sum():,}")
print(f"No wait recorded:   {visits['wait_min'].isna().sum():,}")
display(visits["wait_min"].isna().groupby(visits["clinic"]).mean().round(3))
```

Expect 2,983 visits with no clinic and 28,964 with no wait (the check-in kiosk missed them). By clinic, 1.7% to 2.2% of waits are missing everywhere except Excelsior, at 7.0%. Two gaps matter for this ask: visits with no clinic cannot be assigned to a clinic, and visits with no wait cannot count for or against the target. Step 2 handles both, and Excelsior's gap gets a closer look there.

#### Distribution and outliers

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

Expect quartiles of 11 and 26 minutes, a fence at 48.5, and 53,361 waits above it. The histogram peaks near 10 to 15 minutes and has a long right tail to 240, so a few long waits will pull each clinic's mean above its median. These are long waits, not entry errors, so they stay. In the box plot, Orthopedics sits furthest right and Behavioral Health furthest left: the department mix will matter in step 3.

#### Relationships

A pairplot of 2,000 sampled visits shows every pair of the numeric columns at once.

```python
cols = ["age", "wait_min", "satisfaction"]
sample = visits[cols].sample(2000, random_state=217)
display(sample.corr().round(2))
sns.pairplot(sample, corner=True, plot_kws={"s": 8, "alpha": 0.4})
plt.show()
```

Expect correlations of -0.5 between wait and satisfaction, 0.2 between age and wait, and -0.1 between age and satisfaction. The age histogram has two humps, children and adults over 50. Children cluster at short waits, which points to departments (Pediatrics) rather than age. Ratings fall as waits grow, the question the board asks in step 12.

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

Look inside the two saved files.

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

**What the pass decides:** score clinics on the share of waits over 30 minutes rather than the mean, which the tail inflates; keep the missing waits out of that share; and break clinics down by department.

### 2. Which clinics miss the 30-minute target?

Mark each visit as over 30 minutes or not. A missing wait compares as `False`, so the second line puts the missing waits back; without it, the 28,964 visits the kiosk missed would count as on time.

```python
visits["over_30"] = (visits["wait_min"] > 30).astype(float)
visits.loc[visits["wait_min"].isna(), "over_30"] = np.nan
```

One row per clinic with `groupby` and named aggregation: each keyword names an output column, and its value is a `(source column, function)` pair. The mean of a 0-or-1 column is the share of 1s, and `mean` skips the `NaN`s.

```python
scorecard = visits.groupby("clinic", as_index=False).agg(
    visits=("visit_id", "size"),
    patients=("patient_id", "nunique"),
    mean_wait=("wait_min", "mean"),
    median_wait=("wait_min", "median"),
    share_over_30=("over_30", "mean"),
)
scorecard["visits_per_patient"] = scorecard["visits"] / scorecard["patients"]
display(scorecard.sort_values("share_over_30", ascending=False).round(3))
```

Expect eight rows, worst first. Mission has 24.0% of visits over 30 minutes, then Bayview (19.7%), Sunset (17.6%), Excelsior (17.5%), and Richmond (15.9%); Parkside (12.4%) and Marina (10.0%) pass, and Ridgecrest, the small rural clinic, is best (4.7%). Mission is also the busiest (264,185 visits from 62,757 patients). Every clinic's mean wait is above its median, because a few very long waits pull the mean up. Parkside has the fewest visits per patient (2.67) because it opened in July. Without the `NaN` fix, Excelsior would show 16.3%, because its kiosk lost most of October's waits. Count Excelsior's recorded waits by month to see the gap.

```python
display(visits[visits["clinic"] == "Excelsior"].groupby("visit_month")["wait_min"].count())
```

Expect 10,716 to 13,744 recorded waits in every month except October, which drops to 5,309.

The slide for the memo: each clinic's share over 30 minutes beside the 15% line.

```python
share = scorecard.set_index("clinic")["share_over_30"].sort_values()
ax = share.plot.barh(figsize=(7, 4))
ax.axvline(0.15, color="red", linestyle="--")
ax.set(title="Share of 2025 visits waiting over 30 minutes (target: 15% or less)", xlabel="Share of visits", ylabel="")
plt.show()
```

Expect five bars past the red line (Richmond, Excelsior, Sunset, Bayview, Mission at the top) and three short of it.

Before the table goes into a memo, check that it accounts for every visit.

```python
print(f"Visits in the scorecard: {scorecard['visits'].sum():,}")
print(f"Rows in the extract:     {len(visits):,}")
print(f"Gap:                     {len(visits) - scorecard['visits'].sum():,}")
print(f"Sum of clinic patients:  {scorecard['patients'].sum():,}")
print(f"Distinct patients:       {visits['patient_id'].nunique():,}")
```

Expect a gap of 2,983: exactly the visits with no clinic recorded, which `groupby` dropped without a warning. The clinic patient counts add up to 304,445, more than the 193,034 distinct patients, because a patient seen at two clinics counts once at each: visit counts add up across groups, distinct counts do not.

**What we tell Dr. Alvarez:** five clinics miss the target, Mission by the most (24.0% over 30 minutes). 2,983 visits (0.2%) have no clinic recorded and are footnoted, not assigned.

### 3. Where inside the clinics are the long waits?

A clinic-level share cannot say which department to staff. Group by two keys, one group per observed (department, clinic) pair, then `.unstack()` the clinics into columns. A heatmap colors each cell of the grid by its value.

```python
by_cell = visits.groupby(["department", "clinic"])
wait_grid = by_cell["wait_min"].median().unstack()

fig, ax = plt.subplots(figsize=(11, 4.5))
sns.heatmap(wait_grid, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
ax.set(title="Median wait (minutes) by department and clinic, 2025", xlabel="Clinic", ylabel="")
plt.show()
```

Expect a 7 by 8 grid of median waits. Orthopedics, the darkest row, is the slowest department at every clinic that runs it (25 to 38 minutes), and Bayview's Orthopedics cell, 38, is the darkest of all. Mission's column is a shade darker than its neighbors: it has the longest median in every other department (tied with Bayview in Dermatology), 1 to 4 minutes above Sunset. The blank cells are departments a clinic does not run, such as Orthopedics at Marina: no visits, so the median is `NaN`.

Bayview's Orthopedics cell is the outlier, so pull out that one group with `get_group` and plot its median wait month by month.

```python
bayview_ortho = by_cell.get_group(("Orthopedics", "Bayview"))
monthly = bayview_ortho.groupby("visit_month")["wait_min"].agg(["size", "median"])
print(f"Bayview Orthopedics: {len(bayview_ortho):,} visits, {monthly['size'].min():,} to {monthly['size'].max():,} a month")

ax = monthly["median"].plot(marker="o", figsize=(8, 3.5))
ax.set(title="Bayview Orthopedics: median wait by month, 2025", xlabel="", ylabel="Minutes", ylim=(0, None))
plt.show()
```

Expect 23,690 visits, 1,743 to 2,273 a month, then a flat line at 29 to 34 minutes from January to August that jumps to 55 to 60 from September on. Volume stayed level while the wait nearly doubled. The yearly cell of 38 hid that jump.

**What we tell Dr. Alvarez:** Bayview's Orthopedics waits nearly doubled in September with no rise in visits. She confirms Bayview lost an orthopedic surgeon that month, so a replacement goes first in the staffing memo.

### 4. The board appendix: visits by department, with totals

The appendix uses `pivot_table`, the board's usual layout. First confirm that `pivot_table` gives the same grid as the analysis in step 3, so the appendix and the heatmap cannot disagree.

```python
median_wait = pd.pivot_table(visits, values="wait_min", index="department",
                             columns="clinic", aggfunc="median")
print("Same as the groupby grid?", median_wait.equals(wait_grid))
```

Expect `True`.

Now the visit counts with totals. An absent department really has zero visits, so `fill_value=0` is correct here.

```python
volume = pd.pivot_table(visits, values="visit_id", index="department", columns="clinic",
                        aggfunc="count", fill_value=0, margins=True, margins_name="Total")
display(volume)
print(f"Corner cell: {volume.loc['Total', 'Total']:,}")
```

Expect seven department rows plus `Total`, eight clinic columns plus `Total`, and `0` where a clinic does not run a department (Orthopedics at Marina; Dermatology and Endocrinology at Parkside; Dermatology, Endocrinology, and Orthopedics at Ridgecrest). Ridgecrest's Cardiology count is 179, from its monthly outreach clinic (one more outreach visit has no clinic recorded). The corner cell is 1,197,017, the scorecard's total, so the appendix and the memo agree.

### 5. Did Parkside take load off Sunset?

Parkside opened in July to relieve Sunset. `crosstab` counts rows for each pair of values directly, here month by clinic.

```python
by_month = pd.crosstab(visits["visit_month"], visits["clinic"], margins=True, margins_name="Total")
display(by_month)
```

Expect 12 month rows plus `Total`. Parkside shows `0` through June, then 5,641 in July: it opened on July 7. December (114,209) and January (111,396) are the busiest months, the winter respiratory season. A crosstab cell with no rows is a real count of 0.

Plot the two clinics, leaving out the `Total` row.

```python
ax = by_month.loc[by_month.index != "Total", ["Sunset", "Parkside"]].plot(marker="o", figsize=(8, 3.5))
ax.set(title="Visits per month, 2025", xlabel="", ylabel="Visits", ylim=(0, None))
plt.show()
```

Expect Parkside flat at 0 through June, then 5,641 in July, rising to 8,114 in December. From June to July, Sunset drops from 21,353 to 16,379.

Fewer visits may not mean shorter waits. With `values=` and `aggfunc=`, `crosstab` summarizes a third column instead of counting rows, like `pivot_table`; here, each month's share of visits over 30 minutes.

```python
share_by_month = pd.crosstab(visits["visit_month"], visits["clinic"],
                             values=visits["over_30"], aggfunc="mean")
display(share_by_month[["Sunset", "Parkside"]].round(3))
```

Expect 12 month rows. Parkside is `NaN` through June: no visits, so nothing to average, where the count crosstab showed 0. Sunset sits at 0.163 in June and 0.162 in July, and stays between 0.155 and 0.163 through November. Its worst months are January (0.219), February (0.209), and December (0.187).

**What we tell Dr. Alvarez:** Sunset lost about 5,000 visits a month to Parkside, but its share over 30 minutes stayed near 16% (16.3% in June, 16.2% in July). It misses the 15% target in every month and by the most in winter (21.9% in January), so Sunset needs 2026 hours all year, with the most in winter.

## Independent practice

Run the core walkthrough first. Each item is a follow-up question that came back after the memo went out.

### 6. The CFO asks whether the Total column is the system average

In a pivot table of means, the `Total` column is the mean of all the department's visits, not the average of its cells. Check it on Orthopedics.

```python
mean_wait = pd.pivot_table(visits, values="wait_min", index="department", columns="clinic",
                           aggfunc="mean", margins=True, margins_name="Total")
ortho = mean_wait.loc["Orthopedics"]
print("Total column:          ", round(ortho["Total"], 2))
print("Average of clinic cells:", round(ortho.drop("Total").mean(), 2))
```

Expect two different numbers, 36.71 and 35.25. Mission, slow and busy, has the most Orthopedics visits, so it counts for more in the `Total`; that is the system average a patient experiences. The empty Marina and Ridgecrest cells are `NaN`, and `.mean()` skips them.

### 7. A reviewer asks how many waits stand behind each median

A median is only as good as the count behind it. `aggfunc=["count", "median"]` gives both grids from one pivot table.

```python
wait_stats = pd.pivot_table(visits, values="wait_min", index="department", columns="clinic",
                            aggfunc=["count", "median"])
display(wait_stats["count"])
display(wait_stats["median"])
```

Expect two 7 by 8 tables. Most cells rest on thousands of waits, but Ridgecrest's Cardiology median (14.0) rests on 179, from 10 outreach patients. The counts print as `179.0` because the absent cells are `NaN`, which forces the column to floats.

### 8. Dr. Alvarez asks whether new patients wait longer

Group by clinic and visit type, then `unstack` the visit types into columns to subtract them.

```python
by_type = visits.groupby(["clinic", "visit_type"])["wait_min"].mean().unstack()
by_type["extra_for_new"] = by_type["New"] - by_type["Follow-up"]
display(by_type.round(1))

long_again = by_type[["Follow-up", "New"]].stack()
print("Rows after stack():", len(long_again))
```

Expect eight clinic rows. New patients wait about 3 minutes longer at every clinic (1.9 to 3.9), so the gap is system-wide, not one clinic's problem. `stack()` turns the two columns back into an index level: 16 rows, one per clinic and visit type.

The data team's dashboard loads flat tables, one row per clinic and visit type with the keys as ordinary columns. `as_index=False` gives that shape directly.

```python
flat = visits.groupby(["clinic", "visit_type"], as_index=False)["wait_min"].mean()
display(flat.head(4).round(1))
print(flat.shape)
```

Expect `clinic`, `visit_type`, and `wait_min` columns with a `0..n-1` index, starting with Bayview Follow-up (21.4) and Bayview New (24.5), and a shape of `(16, 3)`: the same 16 pairs as `stack()`.

### 9. Parkside's manager disputes the ranking

The scorecard puts Parkside (12.4% over 30 minutes) behind Marina (10.0%). Parkside's manager replies that a clinic opened in July sees mostly new patients, and step 8 showed new patients wait longer everywhere. First compare the visit mix: `normalize="index"` turns each clinic's row of counts into shares that sum to 1.

```python
pair = visits[visits["clinic"].isin(["Marina", "Parkside"])]
display(pd.crosstab(pair["clinic"], pair["visit_type"], normalize="index").round(3))
```

Expect Marina at 0.825 follow-up and 0.175 new, and Parkside at 0.588 and 0.412: Parkside books more than twice Marina's share of new-patient visits.

Now compare the two clinics within each visit type, beside the overall share.

```python
share = pair.groupby(["clinic", "visit_type"])["over_30"].mean().unstack()
share["All visits"] = pair.groupby("clinic")["over_30"].mean()
display(share.round(3))

ax = share.T.plot.bar(rot=0, figsize=(7, 3.5))
ax.set(title="Share of 2025 visits over 30 minutes", xlabel="", ylabel="Share of visits")
plt.show()
```

Expect Parkside still higher within each visit type, but by less: 0.103 against 0.091 for follow-ups and 0.155 against 0.148 for new patients, against 0.124 and 0.100 for all visits. The chart shows Parkside's bar above Marina's in all three groups, with the gap smallest for new patients. Visit type explains part of the 2.4-point gap, not all of it.

Visit type is not the only difference in case mix. Step 4 showed that Marina runs no Orthopedics, the slowest department, while 12.5% of Parkside's visits are Orthopedics. Compare the two clinics within each department and visit type, on the four departments both clinics run.

```python
shared = pair[pair["department"].isin(["Behavioral Health", "Cardiology", "Family Medicine", "Pediatrics"])]
display(shared.groupby(["department", "visit_type", "clinic"])["over_30"].mean().unstack().round(3))
display(shared.groupby("clinic")["over_30"].mean().round(3))
```

Expect eight rows with a Marina and a Parkside column. Parkside is lower or tied in every row: Cardiology 0.112 against 0.176 for follow-ups and 0.198 against 0.300 for new patients, Family Medicine 0.115 against 0.120 and 0.175 against 0.185, Pediatrics 0.004 against 0.008 and 0.012 against 0.027, and Behavioral Health 0 for both. Then two rows for the four departments both clinics run combined: Marina 0.100 and Parkside 0.096. Parkside is lower there too, so nothing reverses and this is not Simpson's paradox. Parkside is higher overall (12.4% against 10.0%) only because 12.5% of its visits are Orthopedics, which Marina does not run.

**What we tell Dr. Alvarez:** Parkside is as fast as Marina or faster in every department and visit type both clinics run. It ranks behind because of its case mix: 12.5% of its visits are Orthopedics, which Marina does not run, and 41.2% are new patients (17.5% at Marina). The memo compares clinics within department and visit type before ranking them.

### 10. The board asks for patients, not visits

```python
patients = pd.pivot_table(visits, values="patient_id", index="department", columns="clinic",
                          aggfunc="nunique", fill_value=0)
display(patients)

per_patient = visits.groupby("department").agg(visits=("visit_id", "size"), patients=("patient_id", "nunique"))
per_patient["visits_per_patient"] = per_patient["visits"] / per_patient["patients"]
display(per_patient.sort_values("visits_per_patient", ascending=False).round(2))
```

Expect counts well below the visit counts in step 4, and Ridgecrest's Cardiology at 10 patients: ten regulars account for all 179 outreach visits with a clinic recorded. In the second table, Pediatrics has the most visits per patient (5.36), because every child's visits land in one department while an adult's spread across several; Dermatology has the fewest (1.96).

### 11. Each clinic manager wants one line about their own clinic

Dr. Alvarez sends every clinic manager a one-line note before the memo goes out. Looping over the GroupBy object hands over each clinic's name and its rows, one clinic at a time.

```python
for clinic, group in visits.groupby("clinic"):
    print(f"{clinic}: {len(group):,} visits, {group['over_30'].mean():.1%} over 30 minutes")
```

Expect eight lines in alphabetical order, from `Bayview: 185,625 visits, 19.7% over 30 minutes` to `Sunset: 245,192 visits, 17.6% over 30 minutes`. The numbers match the scorecard: the loop writes one message per group, and the `agg` table stays the report.

### 12. The board asks whether long waits cost satisfaction

Select two columns to summarize both with one function, then use a dictionary in `.agg()` to give each column its own list of functions.

```python
display(visits.groupby("clinic")[["wait_min", "satisfaction"]].mean().round(2))

wait_and_rating = visits.groupby("clinic").agg({"wait_min": ["mean", "median"],
                                                 "satisfaction": ["mean", "count"]})
display(wait_and_rating.round(2))
print(wait_and_rating.columns.tolist())
```

Expect Mission with the longest mean wait (23.42 minutes) and the lowest satisfaction (3.81), and Ridgecrest with the shortest wait (13.29) and the highest satisfaction (4.19). The dictionary result has two-level column labels, such as `('wait_min', 'median')`; `wait_and_rating["satisfaction"]` selects one column's pair. The `count` column shows how many surveys stand behind each mean, from 12,280 at Ridgecrest to 89,951 at Mission.
