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

# Demo 1: The 2025 Clinic Operations Report

The operations director wants the year-end report for a health system's eight clinics: how busy each clinic was, how long patients waited, and where the waits were worst. The source is the 2025 visit extract, 1.2 million synthetic visits with one row per visit.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
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

### 1. Name the grain

```python
display(visits.head())
print(f"Distinct patients: {visits['patient_id'].nunique():,}")
print("Months:", visits["visit_month"].min(), "to", visits["visit_month"].max())
print(f"No clinic recorded: {visits['clinic'].isna().sum():,}")
print(f"No wait recorded:   {visits['wait_min'].isna().sum():,}")
print(f"Surveys returned:   {visits['satisfaction'].notna().sum():,}")
```

Expect five visits with ten columns, then 193,873 distinct patients, months `2025-01` to `2025-12`, 2,862 visits with no clinic, 28,973 with no wait (the check-in kiosk missed them), and 431,984 returned satisfaction surveys (1 to 5). The grain is one row per visit, so a patient seen ten times has ten rows.

### 2. One row per clinic: the scorecard

Each keyword names an output column; its value is a `(source column, function)` pair.

```python
scorecard = visits.groupby("clinic", as_index=False).agg(
    visits=("visit_id", "size"),
    patients=("patient_id", "nunique"),
    surveys=("satisfaction", "count"),
    mean_wait=("wait_min", "mean"),
    median_wait=("wait_min", "median"),
)
scorecard["visits_per_patient"] = scorecard["visits"] / scorecard["patients"]
scorecard["survey_rate"] = scorecard["surveys"] / scorecard["visits"]
display(scorecard.sort_values("visits", ascending=False).round(2))
```

Expect eight rows, busiest first. Mission is the busiest clinic (260,579 visits from 62,546 patients) and the slowest (mean wait 24.05 minutes, median 20.0). Ridgecrest, the rural clinic, is the smallest (36,367 visits) and the fastest (median 11.0). Every clinic's mean is above its median, because a few very long waits pull the mean up. Parkside has the fewest visits per patient (2.65) because it opened in July. Survey rates run from 0.32 at Bayview and 0.33 at Excelsior, in the most deprived neighborhoods, to 0.41 at Marina, in the least: the surveys underrepresent the patients those two clinics serve.

Before handing this over, check that the scorecard accounts for every visit.

```python
print(f"Visits in the scorecard: {scorecard['visits'].sum():,}")
print(f"Rows in the extract:     {len(visits):,}")
print(f"Gap:                     {len(visits) - scorecard['visits'].sum():,}")
print(f"Sum of clinic patients:  {scorecard['patients'].sum():,}")
print(f"Distinct patients:       {visits['patient_id'].nunique():,}")
```

Expect a gap of 2,862: exactly the visits with no clinic recorded, which `groupby` dropped without a warning. Report the gap beside the table. The clinic patient counts add up to 304,914, more than the 193,873 distinct patients, because a patient seen at two clinics counts once at each: visit counts add up across groups, distinct counts do not.

### 3. Where are waits worst? Department by clinic

One group per observed (department, clinic) pair; `.unstack()` moves the clinics into columns.

```python
wait_grid = visits.groupby(["department", "clinic"])["wait_min"].median().unstack()
display(wait_grid)
```

Expect a 7 by 8 grid of median waits. Orthopedics is the slowest department at every clinic that runs it (21 to 37 minutes), and Bayview's Orthopedics median, 37.0, is the highest cell in the table. Mission has the longest median in every other department, 2 to 6 minutes above Sunset. The `NaN` cells are departments a clinic does not run, such as Orthopedics at Marina: no visits, so no median.

Bayview stands out, so look at its Orthopedics waits month by month.

```python
bayview_ortho = visits[(visits["clinic"] == "Bayview") & (visits["department"] == "Orthopedics")]
display(bayview_ortho.groupby("visit_month")["wait_min"].agg(["size", "median"]))
```

Expect 1,722 to 2,197 visits a month throughout, with a median of 30 to 34 minutes from January to August, then 56 to 59 from September on. Volume stayed level while the wait nearly doubled (Bayview lost an orthopedic surgeon in September). The yearly cell of 37.0 hid that jump.

### 4. The same grid as a pivot table, and visit counts with totals

`pivot_table` builds the grid in one call.

```python
median_wait = pd.pivot_table(visits, values="wait_min", index="department",
                             columns="clinic", aggfunc="median")
print("Same as the groupby grid?", median_wait.equals(wait_grid))
```

Expect `True`.

The director also wants visit counts by department and clinic, with totals. Here an absent department really has zero visits, so `fill_value=0` is correct.

```python
volume = pd.pivot_table(visits, values="visit_id", index="department", columns="clinic",
                        aggfunc="count", fill_value=0, margins=True, margins_name="Total")
display(volume)
print(f"Corner cell: {volume.loc['Total', 'Total']:,}")
```

Expect seven department rows plus `Total`, eight clinic columns plus `Total`, and `0` where a clinic does not run a department (Dermatology and Endocrinology at Parkside, four specialties at Ridgecrest). Ridgecrest's Cardiology count is 180, from its monthly outreach clinic. The corner cell is 1,197,138, the scorecard's total.

### 5. Volume by month with `crosstab`

`crosstab` counts rows for each pair of values directly.

```python
by_month = pd.crosstab(visits["visit_month"], visits["clinic"], margins=True, margins_name="Total")
display(by_month)
```

Expect 12 month rows plus `Total`. Parkside shows `0` through June, then 5,575 in July: it opened on July 7. In the same month Sunset falls from 21,075 to 16,386 visits, because Parkside's patients used to go to Sunset. December (114,171) and January (111,492) are the busiest months, the winter respiratory season. A crosstab cell with no rows is a real count of 0.

## Independent practice

Run the core walkthrough first.

### 6. What a margin averages

In a table of means, the `Total` column is the mean of all the department's visits, not the average of its cells.

```python
mean_wait = pd.pivot_table(visits, values="wait_min", index="department", columns="clinic",
                           aggfunc="mean", margins=True, margins_name="Total")
ortho = mean_wait.loc["Orthopedics"]
print("Total column:          ", round(ortho["Total"], 2))
print("Average of clinic cells:", round(ortho.drop("Total").mean(), 2))
```

Expect two different numbers, 36.27 and 34.44. Mission, slow and busy, has the most Orthopedics visits, so it counts for more in the `Total`. The empty Marina and Ridgecrest cells are `NaN`, and `.mean()` skips them.

### 7. A median is only as good as the count behind it

```python
wait_stats = pd.pivot_table(visits, values="wait_min", index="department", columns="clinic",
                            aggfunc=["count", "median"])
display(wait_stats["count"])
display(wait_stats["median"])
```

Expect two 7 by 8 tables. Most cells rest on thousands of waits, but Ridgecrest's Cardiology median (14.0) rests on 178, because 2 of its 180 visits have no recorded wait. The counts print as `178.0` because the absent cells are `NaN`, which forces the column to floats.

### 8. New patients wait longer

```python
by_type = visits.groupby(["clinic", "visit_type"])["wait_min"].mean().unstack()
by_type["extra_for_new"] = by_type["New"] - by_type["Follow-up"]
display(by_type.round(1))

long_again = by_type[["Follow-up", "New"]].stack()
print("Rows after stack():", len(long_again))
```

Expect eight clinic rows. New patients wait about 3 minutes longer at every clinic (2.5 to 4.0). `stack()` turns the two columns back into an index level: 16 rows, one per clinic and visit type.

### 9. Draw the median-wait grid as a heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 4.5))
sns.heatmap(median_wait, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
ax.set(title="Median wait (minutes) by department and clinic, 2025", xlabel="Clinic", ylabel="")
plt.show()
```

Expect the Orthopedics row darkest, Bayview's Orthopedics cell the darkest of all, the Mission column a shade darker than its neighbors, and blank cells where a clinic does not run a department.

### 10. Patients, not visits, per department

```python
patients = pd.pivot_table(visits, values="patient_id", index="department", columns="clinic",
                          aggfunc="nunique", fill_value=0)
display(patients)

per_patient = visits.groupby("department").agg(visits=("visit_id", "size"), patients=("patient_id", "nunique"))
per_patient["visits_per_patient"] = per_patient["visits"] / per_patient["patients"]
display(per_patient.sort_values("visits_per_patient", ascending=False).round(2))
```

Expect counts well below the visit counts in step 4, and Ridgecrest's Cardiology at 10 patients: ten regulars account for all 180 outreach visits. In the second table, Pediatrics has the most visits per patient (5.37), because every child's visits land in one department while an adult's spread across several; Dermatology has the fewest (1.96).
