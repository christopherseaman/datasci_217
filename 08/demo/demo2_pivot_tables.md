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

# Demo 2: Group Coverage and Result Shapes

A clinic network's quarterly report groups 100,000 synthetic visits by clinic and by department. This demo checks which visits and which clinics the default report silently leaves out, then adds department context to every visit with `transform`, keeps or drops whole departments with `filter`, and runs custom per-department summaries with `apply`. Everything here comes from Lecture 08 up to the second demo break, plus Lectures 01 to 07.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-25 with Python 3.13, pandas 3.0.5, and NumPy 2.3.3; the whole notebook runs in under a minute. The patient IDs and values are synthetic.

## Setup

The first cell installs pandas 3.0.5, the course version. Colab ships an older pandas (2.2). A `.venv` made with `uv venv --seed` includes pip, so the same `%pip` cell works locally too.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.`, perhaps after a notice that a newer pip is available; neither needs any action. Locally, with the requirements already installed, the cell changes nothing and you can go on. In Colab, pip may also print a dependency conflict because some preinstalled packages expect pandas 2.2; that is expected, and this demo does not use them. If Colab asks you to restart the session (or says pandas was previously imported), choose **Runtime → Restart session**, then continue with the next cell. You do not need to rerun the install.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('pandas', pd.__version__)
```

**Expect:** `pandas 3.0.5`.

## Part 1: Which Groups Appear in a Summary

This demo builds the same visit log Demo 1 used, so it runs on its own. Rerun the next cell even if you still have Demo 1 open.

### Build the Visit Log

```python
rng = np.random.default_rng(42)
n_visits = 100_000
n_patients = 30_000

# One age per patient; about 15% of patients are children
patient_age = np.where(
    rng.random(n_patients) < 0.15,
    rng.integers(0, 18, n_patients),
    rng.integers(18, 96, n_patients),
)

# Each visit belongs to one patient; patients visit about three times each
who = rng.integers(0, n_patients, n_visits)
visits = pd.DataFrame({
    "visit_id": [f"V{i:06d}" for i in range(1, n_visits + 1)],
    "patient_id": [f"P{i:05d}" for i in who],
    "age": patient_age[who],   # each visit gets its patient's age (fancy indexing, Lecture 03)
    "clinic": rng.choice(["North", "South", "East", "West", "Central"], n_visits),
    "visit_type": rng.choice(["New", "Follow-up"], n_visits, p=[0.3, 0.7]),
})

# Children go to Pediatrics; adults to one of five adult departments
adult_departments = ["Family Medicine", "Cardiology", "Orthopedics", "Endocrinology", "Dermatology"]
visits["department"] = np.where(
    visits["age"] < 18,
    "Pediatrics",
    rng.choice(adult_departments, n_visits, p=[0.40, 0.20, 0.17, 0.13, 0.10]),
)

# Typical wait (minutes) and its spread differ by department; walk-in Family Medicine varies most
typical_wait = {"Orthopedics": 38, "Cardiology": 30, "Endocrinology": 26,
                "Family Medicine": 22, "Dermatology": 18, "Pediatrics": 15}
wait_spread = {"Orthopedics": 10, "Cardiology": 9, "Endocrinology": 8,
               "Family Medicine": 14, "Dermatology": 6, "Pediatrics": 5}
visits["wait_min"] = rng.normal(
    visits["department"].map(typical_wait), visits["department"].map(wait_spread)
).round()
visits["wait_min"] = visits["wait_min"].clip(lower=0).astype(int)

# Systolic blood pressure (mmHg) rises with age
visits["systolic_bp"] = rng.normal(100 + 0.5 * visits["age"], 14).round().astype(int)

# Satisfaction survey, 1 (poor) to 5 (excellent): longer waits, lower scores
visits["satisfaction"] = (5.5 - visits["wait_min"] / 15 + rng.normal(0, 0.8, n_visits)).round()
visits["satisfaction"] = visits["satisfaction"].clip(lower=1, upper=5)
# About 3% of surveys were never returned
visits.loc[rng.random(n_visits) < 0.03, "satisfaction"] = np.nan

print(visits.shape)
print(f"Visits with no satisfaction survey: {visits['satisfaction'].isna().sum():,}")
```

**Expect:** `(100000, 9)` and 2,984 visits with no survey, the same log as Demo 1.

### Two Realities the Default Summary Hides

```python
# Work on a copy so `visits` stays unchanged for Parts 2-4
reported = visits.copy()

# 1. Some visits were logged without a clinic (sample, Lecture 05, picks the same 250 every run)
unrecorded = reported.sample(n=250, random_state=8).index
reported.loc[unrecorded, "clinic"] = None

# 2. Bayview opened this quarter and has not seen a patient yet
reporting_order = ["North", "South", "East", "West", "Central", "Bayview"]

print(f"Rows: {len(reported):,}")
print(f"Visits with no clinic recorded: {reported['clinic'].isna().sum()}")
print(f"Clinics named in the report: {len(reporting_order)}")
print(f"Clinics present in the data: {reported['clinic'].nunique()}")
```

**Expect:** 100,000 rows, 250 visits with no clinic, 6 clinics in the report, and 5 in the data. A default `groupby` shows five groups and says nothing about the rest.

### Missing Keys: `dropna`

```python
# The default drops rows whose key is missing, before the split
default_counts = reported.groupby("clinic")["wait_min"].count()
kept_counts = reported.groupby("clinic", dropna=False)["wait_min"].count()

print("=== Default: dropna=True ===")
print(default_counts)
print(f"Visits counted: {default_counts.sum():,} of {len(reported):,}")

print("\n=== dropna=False ===")
print(kept_counts)
print(f"Visits counted: {kept_counts.sum():,} of {len(reported):,}")
```

**Expect:** the default report counts 99,750 of 100,000 visits across five clinics, with no warning that 250 are missing. `dropna=False` keeps them as a `NaN` group, listed last, so the counts add up to 100,000 and someone can go find the missing clinic names.

### Reporting Order and Empty Groups: `pd.Categorical` and `observed`

```python
# Declare the allowed values and the order the report should use
reported["clinic"] = pd.Categorical(reported["clinic"], categories=reporting_order, ordered=True)

present_only = reported.groupby("clinic", observed=True, dropna=False)["wait_min"].agg(["count", "mean"])
every_clinic = reported.groupby("clinic", observed=False, dropna=False)["wait_min"].agg(["count", "mean"])

print("=== observed=True: only clinics with visits ===")
print(present_only.round(1))
print("\n=== observed=False: every declared clinic ===")
print(every_clinic.round(1))
```

**Expect:** both tables follow the declared order (North, South, East, West, Central) instead of alphabetical order, and both keep the `NaN` group last because of `dropna=False`. Only the second shows Bayview, with a count of `0` and a mean of `NaN`: no waits to average is not a mean wait of zero.

### The Same Choices in a Pivot Table

```python
# Counts: an empty clinic gives 0, which is a real value
counts_kept = pd.pivot_table(reported, values="wait_min", index="clinic", columns="department",
                             aggfunc="count", observed=False, dropna=False, fill_value=0)
print("=== Visit counts, dropna=False ===")
print(counts_kept)
print("Rows:", len(counts_kept))

counts_default = pd.pivot_table(reported, values="wait_min", index="clinic", columns="department",
                                aggfunc="count", observed=False, dropna=True, fill_value=0)
print("\n=== Visit counts, dropna=True ===")
print(counts_default)
print("Rows:", len(counts_default))
```

**Expect:** 7 rows with `dropna=False`: the five open clinics, Bayview with six zeros, and the `NaN` group. The `NaN` row comes first here, not last: with a categorical key, `observed=False`, and a `columns=` key, `pivot_table` puts it at the top, so find that row by its label, not by its position. With `dropna=True`, 6 rows: the `NaN` group goes, but Bayview stays, because a count of `0` is a value, not a missing one.

```python
# Means: an empty clinic gives NaN in every cell, and dropna=True removes all-NaN rows
means_default = pd.pivot_table(reported, values="wait_min", index="clinic", columns="department",
                               aggfunc="mean", observed=False, dropna=True)
print("=== Mean wait, dropna=True ===")
print(means_default.round(1))
print("Rows:", len(means_default))

means_kept = pd.pivot_table(reported, values="wait_min", index="clinic", columns="department",
                            aggfunc="mean", observed=False, dropna=False)
print("\n=== Mean wait, dropna=False ===")
print(means_kept.round(1))
print("Rows:", len(means_kept))
```

**Expect:** 5 rows with `dropna=True`: the `NaN` group goes because its key is missing, and Bayview goes because all six of its cells are `NaN`. With `dropna=False`, 7 rows. The `NaN` row, again at the top, is not empty: six real means, from 16.5 minutes in Pediatrics to 38.6 in Orthopedics, computed from the 250 visits the default report never mentions.

### Show the Difference

```python
# Every group the data can produce, from the observed=False, dropna=False table above
full_counts = every_clinic["count"]
# One label and color per row, in the same order; red marks the two the default report hides
labels = ["North", "South", "East", "West", "Central", "Bayview", "(not recorded)"]
colors = ["steelblue", "steelblue", "steelblue", "steelblue", "steelblue", "red", "red"]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(labels, full_counts, color=colors)
ax.bar_label(bars)
ax.set_title("Visits per clinic; red bars are missing from the default report")
ax.set_ylabel("Visits counted")
plt.tight_layout()
plt.show()
```

**Expect:** five blue bars of about 20,000 visits each, then Bayview labelled `0` with no bar at all and a thin red bar labelled `250` for the visits with no clinic. Neither would appear in a default report, and neither is easy to see even here, which is why a report needs `observed=False` and `dropna=False` to list them.

## Part 2: Transform Keeps Every Row

`transform` computes a statistic per group and copies it back to every row of that group, so the result has the same length and index as the input and can become a new column. The rest of the demo uses `visits`, which still has every clinic recorded.

### Intentional Mistake: Attaching a Summary to Rows

```python
# A per-department summary has 6 rows, labeled by department
dept_mean = visits.groupby("department")["wait_min"].mean()

wrong = visits.copy()
wrong["dept_mean_wait"] = dept_mean
print(wrong[["department", "wait_min", "dept_mean_wait"]].head())
print("Missing values in the new column:", wrong["dept_mean_wait"].isna().sum())
```

**Expect:** every `dept_mean_wait` is `NaN` (100,000 missing), with no error. The summary's labels are department names and the visits are labeled 0 to 99,999, so nothing lines up. The fix is `transform`, next.

### Add Department Context to Every Visit

```python
by_dept = visits.groupby("department")["wait_min"]
visits["dept_mean_wait"] = by_dept.transform("mean")
visits["dept_sd_wait"] = by_dept.transform("std")
visits["wait_vs_dept"] = visits["wait_min"] - visits["dept_mean_wait"]
visits["wait_z"] = visits["wait_vs_dept"] / visits["dept_sd_wait"]

# The same z-score with a lambda: the function receives one department's waits at a time
z_lambda = by_dept.transform(lambda x: (x - x.mean()) / x.std())

print(visits[["department", "wait_min", "dept_mean_wait", "wait_vs_dept", "wait_z"]].head().round(2))
print("\nRows:", len(visits))
print("Same index as visits?", z_lambda.index.equals(visits.index))
print("Largest difference between the two z-scores:", (visits["wait_z"] - z_lambda).abs().max())
```

**Expect:** the first five visits with their department's mean, the difference, and the z-score; row 0 is a 28-minute Family Medicine wait, 5.70 minutes above that department's 22.30, so `wait_z` is 0.43. Then `Rows: 100000`, `True`, and a difference of about `6e-15`: the two z-scores agree to rounding error. `transform` added columns; it did not summarize.

```python
# The same 30-minute wait, measured against each department's own waits
thirty = visits[visits["wait_min"] == 30]
print(thirty.groupby("department")["wait_z"].mean().round(2))
```

**Expect:** the same 30-minute wait is ordinary in Cardiology (z = 0.01), about 3 standard deviations above the Pediatrics mean (2.99), and shorter than usual in Orthopedics (-0.8). Group context changes what one number means.

### Quartiles Within Each Department

```python
# qcut (Lecture 05) inside transform: quartiles of each department's own waits
visits["wait_quartile"] = by_dept.transform(
    lambda x: pd.qcut(x, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
)
print(pd.crosstab(visits["department"], visits["wait_quartile"], margins=True))
```

**Expect:** each department split into Q1 to Q4 by its own waits, with row totals equal to its visit counts and 100,000 in the corner. The quarters are not exactly equal (Pediatrics Q1 has 4,628 visits, Q3 has 3,153) because many visits share the same whole-minute wait, and tied waits always land in the same quartile.

## Part 3: Filter Keeps or Drops Whole Groups

`filter` tests each whole group and keeps every row of the groups that pass. Rows are never changed, only kept or dropped.

### Keep Only the Departments That Qualify

```python
by_department = visits.groupby("department")

# Departments with at least 15,000 visits
busy = by_department.filter(lambda g: len(g) >= 15_000)
print(f"At least 15,000 visits: {len(busy):,} rows, departments {sorted(busy['department'].unique())}")

# Departments whose mean wait is over 25 minutes
slow = by_department.filter(lambda g: g["wait_min"].mean() > 25)
print(f"Mean wait over 25 min:  {len(slow):,} rows, departments {sorted(slow['department'].unique())}")

# Departments whose waits vary the most (standard deviation over 9.5 minutes)
variable = by_department.filter(lambda g: g["wait_min"].std() > 9.5)
print(f"Wait SD over 9.5 min:   {len(variable):,} rows, departments {sorted(variable['department'].unique())}")

# Checkpoint: the kept rows are the original rows, unchanged
print("\nRows unchanged?", busy.equals(visits.loc[busy.index]))
```

**Expect:** only Family Medicine and Cardiology have 15,000 or more visits (51,310 rows); Pediatrics, at 14,672, just misses. Cardiology, Endocrinology, and Orthopedics average over 25 minutes (42,724 rows). Family Medicine and Orthopedics have the most variable waits (48,659 rows). Then `True`: `filter` returns the original rows of the passing groups, not one row per group.

## Part 4: Apply Runs Your Own Function per Group

`apply` hands each group to your function as a small DataFrame and stitches the results together. Use it when no built-in aggregation fits; it is the slowest of the four, and its output shape depends on what your function returns.

### A Custom Summary per Department

```python
def wait_summary(group):
    """Summarize one department's waits (minutes) as a Series."""
    q25 = group["wait_min"].quantile(0.25)
    q75 = group["wait_min"].quantile(0.75)
    return pd.Series({
        "visits": len(group),
        "median": group["wait_min"].median(),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "over_60_min": (group["wait_min"] > 60).sum(),
    })

dept_summary = visits.groupby("department").apply(wait_summary, include_groups=False)
print(dept_summary)

# Checkpoint: the medians match the built-in aggregation
builtin = visits.groupby("department")["wait_min"].median()
print("\nSame medians as agg('median')?", dept_summary["median"].equals(builtin))
```

**Expect:** one row per department, like `agg`, with the columns your function named. Every number prints with `.0` because a `Series` that holds a median stores all its values as floats. Family Medicine has the widest interquartile range (18 minutes) and Orthopedics the most waits over an hour (187). Then `True`.

### The Longest Waits in Each Department

```python
longest = visits.groupby("department").apply(
    lambda g: g.nlargest(2, "wait_min"), include_groups=False
)
print(longest[["patient_id", "clinic", "wait_min"]])
print("\nRows:", len(longest))
```

**Expect:** 12 rows: the two longest waits in each department, with the department as the outer index level and each visit's original row number as the inner one. Family Medicine's longest wait was 80 minutes. When the function returns whole rows, `apply` returns rows; when it returns one `Series` per group, it returns one row per group. When `agg` or `transform` can do the job, prefer them: they run as compiled code instead of once per group.
