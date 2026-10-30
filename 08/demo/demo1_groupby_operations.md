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
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
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

### 1. What is in the extract?

Before counting anything, check what one row is and which fields have gaps.

```python
display(visits.head())
print(f"Distinct patients: {visits['patient_id'].nunique():,}")
print("Months:", visits["visit_month"].min(), "to", visits["visit_month"].max())
print(f"No clinic recorded: {visits['clinic'].isna().sum():,}")
print(f"No wait recorded:   {visits['wait_min'].isna().sum():,}")
print(f"Surveys returned:   {visits['satisfaction'].notna().sum():,}")
```

Expect five visits with ten columns, then 193,873 distinct patients, months `2025-01` to `2025-12`, 2,862 visits with no clinic, 28,973 with no wait (the check-in kiosk missed them), and 431,984 returned satisfaction surveys (1 to 5). The grain is one row per visit, so a patient seen ten times has ten rows. Two gaps matter for this ask: visits with no clinic cannot be assigned to a clinic, and visits with no wait cannot count for or against the target.

### 2. Which clinics miss the 30-minute target?

Mark each visit as over 30 minutes or not. A missing wait compares as `False`, so the second line puts the missing waits back; without it, the 28,973 visits the kiosk missed would count as on time.

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

Expect eight rows, worst first. Mission has 25.3% of visits over 30 minutes, then Bayview (18.5%), Excelsior (18.4%), and Sunset (15.4%); Richmond (14.2%) just passes, and Ridgecrest, the small rural clinic, is best (5.1%). Mission is also the busiest (260,579 visits from 62,546 patients). Every clinic's mean wait is above its median, because a few very long waits pull the mean up. Parkside has the fewest visits per patient (2.65) because it opened in July. Without the `NaN` fix, Excelsior would show 17.1%, because its kiosk lost most of October's waits.

The slide for the memo: each clinic's share over 30 minutes beside the 15% line. `ax.axvline(x)` draws a vertical reference line at `x`.

```python
import matplotlib.pyplot as plt
import seaborn as sns

share = scorecard.set_index("clinic")["share_over_30"].sort_values()
ax = share.plot.barh(figsize=(7, 4))
ax.axvline(0.15, color="red", linestyle="--")
ax.set(title="Share of 2025 visits waiting over 30 minutes (target: 15% or less)", xlabel="Share of visits", ylabel="")
plt.show()
```

Expect four bars past the red line (Sunset, Excelsior, Bayview, Mission at the top) and four short of it.

Before the table goes into a memo, check that it accounts for every visit.

```python
print(f"Visits in the scorecard: {scorecard['visits'].sum():,}")
print(f"Rows in the extract:     {len(visits):,}")
print(f"Gap:                     {len(visits) - scorecard['visits'].sum():,}")
print(f"Sum of clinic patients:  {scorecard['patients'].sum():,}")
print(f"Distinct patients:       {visits['patient_id'].nunique():,}")
```

Expect a gap of 2,862: exactly the visits with no clinic recorded, which `groupby` dropped without a warning. The clinic patient counts add up to 304,914, more than the 193,873 distinct patients, because a patient seen at two clinics counts once at each: visit counts add up across groups, distinct counts do not.

**What we tell Dr. Alvarez:** four clinics miss the target, Mission by the most (25.3% over 30 minutes). 2,862 visits (0.2%) have no clinic recorded and are footnoted, not assigned.

### 3. Where inside the clinics are the long waits?

A clinic-level share cannot say which department to staff. Group by two keys, one group per observed (department, clinic) pair, then `.unstack()` the clinics into columns. A heatmap colors each cell of the grid by its value.

```python
by_cell = visits.groupby(["department", "clinic"])
wait_grid = by_cell["wait_min"].median().unstack()

fig, ax = plt.subplots(figsize=(10, 4.5))
sns.heatmap(wait_grid, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
ax.set(title="Median wait (minutes) by department and clinic, 2025", xlabel="Clinic", ylabel="")
plt.show()
```

Expect a 7 by 8 grid of median waits. Orthopedics, the darkest row, is the slowest department at every clinic that runs it (21 to 37 minutes), and Bayview's Orthopedics cell, 37, is the darkest of all. Mission's column is a shade darker than its neighbors: it has the longest median in every other department, 2 to 6 minutes above Sunset. The blank cells are departments a clinic does not run, such as Orthopedics at Marina: no visits, so the median is `NaN`.

Bayview's Orthopedics cell is the outlier, so pull out that one group with `get_group` and plot its median wait month by month.

```python
bayview_ortho = by_cell.get_group(("Orthopedics", "Bayview"))
monthly = bayview_ortho.groupby("visit_month")["wait_min"].agg(["size", "median"])
print(f"Bayview Orthopedics: {len(bayview_ortho):,} visits, {monthly['size'].min():,} to {monthly['size'].max():,} a month")

ax = monthly["median"].plot(marker="o", figsize=(8, 3.5))
ax.set(title="Bayview Orthopedics: median wait by month, 2025", xlabel="", ylabel="Minutes", ylim=(0, None))
plt.show()
```

Expect 23,425 visits, 1,722 to 2,197 a month, then a flat line at 30 to 34 minutes from January to August that jumps to 56 to 59 from September on. Volume stayed level while the wait nearly doubled. The yearly cell of 37 hid that jump.

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

Expect seven department rows plus `Total`, eight clinic columns plus `Total`, and `0` where a clinic does not run a department (Dermatology and Endocrinology at Parkside; Dermatology, Endocrinology, and Orthopedics at Ridgecrest). Ridgecrest's Cardiology count is 180, from its monthly outreach clinic. The corner cell is 1,197,138, the scorecard's total, so the appendix and the memo agree.

### 5. Did Parkside take load off Sunset?

Parkside opened in July to relieve Sunset. `crosstab` counts rows for each pair of values directly, here month by clinic.

```python
by_month = pd.crosstab(visits["visit_month"], visits["clinic"], margins=True, margins_name="Total")
display(by_month)
```

Expect 12 month rows plus `Total`. Parkside shows `0` through June, then 5,575 in July: it opened on July 7. December (114,171) and January (111,492) are the busiest months, the winter respiratory season. A crosstab cell with no rows is a real count of 0.

Plot the two clinics, leaving out the `Total` row.

```python
ax = by_month.loc[by_month.index != "Total", ["Sunset", "Parkside"]].plot(marker="o", figsize=(8, 3.5))
ax.set(title="Visits per month, 2025", xlabel="", ylabel="Visits", ylim=(0, None))
plt.show()
```

Expect Parkside flat at 0 through June, then 5,575 in July, rising to 7,836 in December. From June to July, Sunset drops from 21,075 to 16,386.

**What we tell Dr. Alvarez:** Sunset lost about 4,700 visits a month to Parkside, yet still misses the target at 15.4% for the year. Decide Sunset's 2026 hours after one full year of Parkside.

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

Expect two different numbers, 36.27 and 34.44. Mission, slow and busy, has the most Orthopedics visits, so it counts for more in the `Total`; that is the system average a patient experiences. The empty Marina and Ridgecrest cells are `NaN`, and `.mean()` skips them.

### 7. A reviewer asks how many waits stand behind each median

A median is only as good as the count behind it. `aggfunc=["count", "median"]` gives both grids from one pivot table.

```python
wait_stats = pd.pivot_table(visits, values="wait_min", index="department", columns="clinic",
                            aggfunc=["count", "median"])
display(wait_stats["count"])
display(wait_stats["median"])
```

Expect two 7 by 8 tables. Most cells rest on thousands of waits, but Ridgecrest's Cardiology median (14.0) rests on 178, because 2 of its 180 visits have no recorded wait. The counts print as `178.0` because the absent cells are `NaN`, which forces the column to floats.

### 8. Dr. Alvarez asks whether new patients wait longer

Group by clinic and visit type, then `unstack` the visit types into columns to subtract them.

```python
by_type = visits.groupby(["clinic", "visit_type"])["wait_min"].mean().unstack()
by_type["extra_for_new"] = by_type["New"] - by_type["Follow-up"]
display(by_type.round(1))

long_again = by_type[["Follow-up", "New"]].stack()
print("Rows after stack():", len(long_again))
```

Expect eight clinic rows. New patients wait about 3 minutes longer at every clinic (2.5 to 4.0), so the gap is system-wide, not one clinic's problem. `stack()` turns the two columns back into an index level: 16 rows, one per clinic and visit type.

### 9. The board asks for patients, not visits

```python
patients = pd.pivot_table(visits, values="patient_id", index="department", columns="clinic",
                          aggfunc="nunique", fill_value=0)
display(patients)

per_patient = visits.groupby("department").agg(visits=("visit_id", "size"), patients=("patient_id", "nunique"))
per_patient["visits_per_patient"] = per_patient["visits"] / per_patient["patients"]
display(per_patient.sort_values("visits_per_patient", ascending=False).round(2))
```

Expect counts well below the visit counts in step 4, and Ridgecrest's Cardiology at 10 patients: ten regulars account for all 180 outreach visits. In the second table, Pediatrics has the most visits per patient (5.37), because every child's visits land in one department while an adult's spread across several; Dermatology has the fewest (1.96).
