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

# Demo 2: A Fourth-Quarter Quality Audit

The quality team reviews the last quarter of 2025 in three parts: a visit-volume table in the board's template that accounts for every visit, an audit of unusually long waits, and a public satisfaction table that hides groups too small to publish. The source is the same 1.2-million-visit extract as Demo 1, plus the health system's clinic list.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

This supplied cell downloads `generate_data.py` when it is missing, runs `generate_data.build()` to write the tables into `data/` if they are not there yet (about 30 seconds), then loads the clinic list and the visits.

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

clinics = pd.read_parquet("data/clinics.parquet")
visits = pd.read_parquet("data/visits.parquet")
print(clinics.shape, visits.shape)
```

Expect `(9, 3) (1200000, 10)`.

## Core walkthrough

### 1. Keep the fourth quarter

`visit_month` is text such as `2025-10`, so text comparison keeps October to December.

```python
display(clinics)
q4 = visits[visits["visit_month"] >= "2025-10"].copy()
print(f"Q4 visits: {len(q4):,}")
print(f"No clinic recorded: {q4['clinic'].isna().sum():,}")
```

Expect nine clinics in the board's reporting order, ending with Oceanview, which opens on 2026-01-12, then 316,158 Q4 visits, 744 of them with no clinic recorded.

### 2. A volume table that accounts for every visit

The board's template lists all nine clinics in its own order. Start with the default summary.

```python
default = q4.groupby("clinic")["visit_id"].count()
display(default)
print(f"Visits counted: {default.sum():,} of {len(q4):,}")
```

Expect eight clinics in alphabetical order, no Oceanview, and 315,414 of 316,158 visits counted: the 744 visits with no clinic vanished without a warning.

Declare the template's clinics and order with `pd.Categorical`, then ask for every category (`observed=False`) and the missing key (`dropna=False`).

```python
q4["clinic"] = pd.Categorical(q4["clinic"], categories=clinics["clinic"], ordered=True)
volume = q4.groupby("clinic", observed=False, dropna=False).agg(
    visits=("visit_id", "size"),
    patients=("patient_id", "nunique"),
    median_wait=("wait_min", "median"),
)
display(volume)
print(f"Visits counted: {volume['visits'].sum():,} of {len(q4):,}")
```

Expect ten rows: the nine clinics in the template's order, Oceanview with 0 visits, 0 patients, and a `NaN` median (no waits to take a median of), then a `NaN` row with the 744 unrecorded visits. Now all 316,158 visits are counted, and someone can go find the missing clinic names.

Draw the board's chart. A chart axis needs text labels, so the `NaN` key gets the label `No clinic` (`astype(object)` turns the categorical labels into plain values that `fillna` can replace), and `ax.invert_yaxis()` puts the first clinic at the top.

```python
import matplotlib.pyplot as plt
import seaborn as sns

labels = volume.index.astype(object).fillna("No clinic")
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(labels, volume["visits"])
ax.invert_yaxis()
ax.set(title="Q4 2025 visits by clinic, board order", xlabel="Visits")
plt.show()
```

Expect ten bars from Mission at the top to `No clinic` at the bottom, with an empty slot for Oceanview (0 visits) and a sliver for the 744 unrecorded visits. The default summary would have drawn eight bars and said nothing about the other two.

### 3. Flag waits that are long for their own clinic and department

A 45-minute wait is routine in Orthopedics and alarming in Behavioral Health, so the audit compares each visit with its own clinic and department.

_Intentional mistake:_ attach each department's mean wait as a column.

```python
dept_mean = q4.groupby("department")["wait_min"].mean()
wrong = q4.copy()
wrong["dept_mean"] = dept_mean
print("Missing values in the new column:", wrong["dept_mean"].isna().sum())

try:
    wrong["dept_mean"] = dept_mean.values
except ValueError as error:
    print("ValueError:", error)
```

Expect `316158` missing values with no error: the summary is labeled by department and the visits by row number, so no label matches. Then `ValueError: Length of values (7) does not match length of index (316158)`: seven department means do not fit 316,158 rows.

**The fix:** `transform` returns one value per visit, aligned to the visits.

```python
cell = q4.groupby(["clinic", "department"], observed=True)["wait_min"]
q4["cell_mean"] = cell.transform("mean")
q4["cell_sd"] = cell.transform("std")
q4["wait_z"] = (q4["wait_min"] - q4["cell_mean"]) / q4["cell_sd"]
q4["long_for_cell"] = q4["wait_z"] > 3
q4["over_60"] = q4["wait_min"] > 60
display(q4[["clinic", "department", "wait_min", "cell_mean", "wait_z"]].head().round(2))
```

Expect five visits, each with its own cell's mean wait and a z-score, the number of standard deviations above that mean. The first has no recorded wait, so its z-score is `NaN`, as it is for visits with no clinic. The second, a 9-minute Pediatrics wait at Bayview, has a z-score of -0.86, below its cell's mean of 14.58. The index keeps the visits' row numbers from the full extract (883842 onward).

Compare the z-score rule with a flat 60-minute rule, department by department: raw waits against the 60-minute line, and z-scores against the line at 3. `ax.axvline(x)` draws a vertical reference line at `x`.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
sns.boxplot(data=q4, x="wait_min", y="department", fliersize=1, ax=axes[0])
axes[0].axvline(60, color="red", linestyle="--")
axes[0].set(title="Raw wait: flat rule at 60 minutes", xlabel="Wait (minutes)", ylabel="")
sns.boxplot(data=q4, x="wait_z", y="department", fliersize=1, ax=axes[1])
axes[1].axvline(3, color="red", linestyle="--")
axes[1].set(title="Wait z-score within clinic and department: rule at 3", xlabel="z-score")
plt.show()
```

Expect the left boxes at very different positions: Orthopedics far right, Behavioral Health far left with no dot past the red line. Most dots past 60 belong to Orthopedics and Family Medicine. On the right every box is centered near 0 with a similar width, and every department, Behavioral Health included, has a tail of dots past 3.

```python
flags = q4.groupby("department").agg(
    visits=("visit_id", "size"),
    over_60=("over_60", "sum"),
    long_for_cell=("long_for_cell", "sum"),
)
display(flags)
```

Expect the flat rule to flag 7,203 visits, 6,708 of them in Orthopedics (4,721) and Family Medicine (1,987), and none in Behavioral Health. The z-score rule flags 4,497 visits spread across every department, including 438 in Behavioral Health, where a 20-minute wait is far from usual.

A cell's own mean can be the problem. Check Bayview's Orthopedics, where Demo 1 found waits nearly doubled after September.

```python
bayview = q4[(q4["clinic"] == "Bayview") & (q4["department"] == "Orthopedics")]
print(f"Visits: {len(bayview):,}  mean wait: {bayview['wait_min'].mean():.1f}")
print("Over 60 minutes:", bayview["over_60"].sum(), " long for its cell:", bayview["long_for_cell"].sum())
```

Expect 6,136 visits with a mean wait of 62.4 minutes: 2,727 are over 60, but only 76 are long for their cell. Compared with its own slow quarter, almost nothing at Bayview looks unusual. The z-score finds odd visits within a group; the group summary (Demo 1's grid) finds the odd group.

### 4. A public satisfaction table that hides small groups

Public reports of patient data hide any group with 10 or fewer patients. The table uses Q4 survey respondents, grouped by clinic and department.

```python
respondents = q4.dropna(subset=["satisfaction"])
cells = respondents.groupby(["clinic", "department"], observed=True).agg(
    surveys=("visit_id", "size"),
    patients=("patient_id", "nunique"),
)
display(cells[cells["patients"] < 11])
```

Expect one cell, Ridgecrest Cardiology: 21 surveys from only 10 patients, because its outreach clinic sees the same regulars every month.

Counting rows would publish that cell; counting distinct patients hides it.

```python
by_cell = respondents.groupby(["clinic", "department"], observed=True)
rows_rule = by_cell.filter(lambda g: len(g) >= 11)
patients_rule = by_cell.filter(lambda g: g["patient_id"].nunique() >= 11)
print(f"Respondents: {len(respondents):,}")
print(f"Kept by len(g) >= 11:     {len(rows_rule):,}")
print(f"Kept by nunique() >= 11:  {len(patients_rule):,}")

public = pd.pivot_table(patients_rule, values="satisfaction", index="department",
                        columns="clinic", aggfunc="mean", observed=True)
display(public.round(2))
```

Expect 113,820 respondents. The row rule keeps 113,580, every respondent with a clinic (the 240 with no clinic have no group), and the patient rule keeps 21 fewer, 113,559. In the public table, Ridgecrest's Cardiology cell is `NaN` (suppressed), beside the cells for departments Ridgecrest does not run.

```python
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.heatmap(public, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
ax.text(list(public.columns).index("Ridgecrest") + 0.5, list(public.index).index("Cardiology") + 0.5,
        "suppressed", ha="center", va="center", fontsize=8)
ax.set(title="Mean satisfaction (1 to 5), Q4 2025, groups of 11+ patients", xlabel="Clinic", ylabel="")
plt.show()
```

Expect Ridgecrest's Cardiology cell labeled `suppressed` and its three other blank cells, the departments it does not run, left empty. Scores follow waits: the Behavioral Health row, the shortest wait, is the greenest at every clinic (4.20 to 4.61), and Bayview's Orthopedics, the longest wait, is the reddest cell (2.95).

## Independent practice

Run the core walkthrough first.

### 5. Fill the October kiosk gap with each cell's median

```python
october = q4[q4["visit_month"] == "2025-10"]
display(october.groupby("clinic", observed=True)["wait_min"].agg(["size", "count"]))

cell_median = q4.groupby(["clinic", "department"], observed=True)["wait_min"].transform("median")
q4["wait_filled"] = q4["wait_min"].fillna(cell_median)
overall_fill = q4["wait_min"].fillna(q4["wait_min"].median())
excelsior_ortho = (q4["clinic"] == "Excelsior") & (q4["department"] == "Orthopedics") & q4["wait_min"].isna()
print("Excelsior Orthopedics, cell-median fill:", q4.loc[excelsior_ortho, "wait_filled"].unique())
print("Excelsior Orthopedics, overall fill:   ", overall_fill[excelsior_ortho].unique())
```

Expect Excelsior's October `count` (5,250) far below its `size` (13,363): its kiosk was down for most of the month. The cell-median fill gives Excelsior's missing Orthopedics waits `[32.]`, its own Orthopedics median; the overall fill gives them `[17.]`, the all-visit median. Waits with no clinic recorded stay missing, because their group has no key.

### 6. The volume table as a department-by-clinic pivot

```python
volume_grid = pd.pivot_table(q4, values="visit_id", index="clinic", columns="department",
                             aggfunc="count", observed=False, dropna=False, fill_value=0)
display(volume_grid)
print("Rows:", len(volume_grid), " total:", volume_grid.to_numpy().sum())
```

Expect 10 rows: a `NaN` row for the unrecorded visits (listed first here, so find it by its label), the nine clinics, and Oceanview all zeros. The total is 316,158. A count of 0 is a real value here, so `fill_value=0` is right.

### 7. Staffing review: cells by mean and by total wait

```python
by_cell_all = q4.groupby(["clinic", "department"], observed=True)
slow = by_cell_all.filter(lambda g: g["wait_min"].mean() > 40)
heavy = by_cell_all.filter(lambda g: g["wait_min"].sum() > 365_000)
print("Mean wait over 40 minutes:", slow.groupby(["clinic", "department"], observed=True).size().to_dict())
display(heavy.groupby(["clinic", "department"], observed=True)["wait_min"].agg(["size", "sum", "mean"]))
```

Expect one cell by mean, Bayview Orthopedics with 6,136 visits. By total minutes, Bayview Orthopedics (376,887 minutes) appears again, beside Family Medicine at Mission (578,743) and Sunset (372,992), with mean waits of about 24 and 20: not slow, but where most patient time goes.

A row mask with the same threshold answers a different question: it keeps single long waits from every cell.

```python
long_rows = q4[q4["wait_min"] > 40]
print(f"Row mask: {len(long_rows):,} visits from {len(long_rows.groupby(['clinic', 'department'], observed=True).size())} cells")
print(f"filter:   {len(slow):,} visits from 1 cell")
```

Expect the row mask to keep 24,842 visits from 39 cells, and `filter` all 6,136 visits, short waits included, from Bayview Orthopedics.

### 8. The same z-score with a lambda

```python
z_lambda = cell.transform(lambda x: (x - x.mean()) / x.std())
print("Same index as q4?", z_lambda.index.equals(q4.index))
print("Largest difference:", (z_lambda - q4["wait_z"]).abs().max())
```

Expect `True` and a difference near `2e-14`: the same z-scores, with the lambda running once per cell.

## Optional extension: `dropna` in a pivot table

Prerequisite: BONUS.md, Group Coverage Edge Cases.

```python
for drop in [True, False]:
    means = pd.pivot_table(q4, values="wait_min", index="clinic", columns="department",
                           aggfunc="mean", observed=False, dropna=drop)
    print(f"dropna={drop}: {len(means)} rows")
display(means.round(1))
```

Expect 8 rows with `dropna=True`: the `NaN` group goes because its key is missing, and Oceanview goes because every one of its cells is `NaN`. With `dropna=False`, 10 rows. The `NaN` row, listed first, holds real means, such as 39.2 minutes in Orthopedics, computed from the 744 visits the default report never mentions.
