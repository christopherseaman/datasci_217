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

# Demo 1: GroupBy Summaries and Pivot Tables

A health system's visit log has one row per clinic visit: 100,000 synthetic visits across five clinics and six departments. This demo answers the questions a clinic manager asks of that log (how long each department's patients wait, how many patients each one sees, how the answers differ by clinic) with `groupby`, named aggregation, two-key groups, pivot tables with totals, and cross-tabulations. Everything here comes from Lecture 08 up to the first demo break, plus Lectures 01 to 07.

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
import seaborn as sns

print('pandas', pd.__version__)
```

**Expect:** `pandas 3.0.5`.

## Part 1: One Row per Group

The visit log has one row per visit. Every question in this part has an answer with one row per department, so each answer is a split (by `department`), an apply (mean, count, ...), and a combine.

### Build the Visit Log

The next cell generates the log. You do not need to follow every line: it draws a fixed set of patients, gives each visit a clinic, a department, a wait, a blood pressure, and a satisfaction score, and leaves some surveys blank. The seed makes every run produce the same table.

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
print(f"Distinct patients: {visits['patient_id'].nunique():,}")
print(f"Visits with no satisfaction survey: {visits['satisfaction'].isna().sum():,}")
visits.head()
```

**Expect:** `(100000, 9)`, 28,929 distinct patients, and 2,984 visits with no survey. The grain is one row per visit, so a patient who came three times has three rows.

### Basic Aggregation

```python
# One mean per department: split by department, average wait_min, combine
dept_wait = visits.groupby("department")["wait_min"].mean()
print(dept_wait.round(1))

# Several summaries per column: a dictionary gives two-level column labels
dept_stats = visits.groupby("department").agg({
    "wait_min": ["mean", "median", "std", "min", "max"],
    "age": ["mean", "max"],
    "satisfaction": "mean",
})
dept_stats.round(1)
```

**Expect:** six rows in alphabetical order. Orthopedics has the longest mean wait (38.1 minutes) and Pediatrics the shortest (14.9). Pediatrics' mean age is 8.5 and its oldest patient is 17, because only children go there. Satisfaction runs opposite to wait: 4.4 in Pediatrics, 3.0 in Orthopedics. The columns have two levels (`wait_min` over `mean`, `median`, ...) because `.agg()` received a dictionary.

```python
# Two views of the same groups (Lecture 07): the means, and the spread behind them
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dept_wait.sort_values().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Mean wait by department")
axes[0].set_ylabel("Mean wait (minutes)")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=45)

sns.boxplot(data=visits, x="department", y="wait_min", ax=axes[1])
axes[1].set_title("Wait distribution by department")
axes[1].set_ylabel("Wait (minutes)")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
```

**Expect:** two panels. The bars climb from Pediatrics (shortest mean wait) to Orthopedics (longest). In the box plot, Family Medicine's box and whiskers are the widest: its mean is middling, but its waits are the least predictable.

### Named Aggregation: One Flat Row per Department

Each keyword below becomes an output column, and its value is a `(source column, function)` pair.

```python
dept_report = visits.groupby("department", as_index=False).agg(
    visits=("visit_id", "size"),
    patients=("patient_id", "nunique"),
    rated=("satisfaction", "count"),
    mean_wait=("wait_min", "mean"),
    median_wait=("wait_min", "median"),
    longest_wait=("wait_min", "max"),
)
print(dept_report.round(1))

# Checkpoint: name the grain before and after
print(f"\nInput grain:  one row per visit      -> {len(visits):,} rows")
print(f"Output grain: one row per department -> {len(dept_report)} rows")
print(f"Visits accounted for: {dept_report['visits'].sum():,}")
```

**Expect:** six rows with flat column names and `visits` adding up to 100,000. `patients` is smaller than `visits` in every row because patients come back: Pediatrics saw 4,285 children across 14,672 visits. `rated` is smaller than `visits` too, because `count` skips the blank surveys while `size` counts every row.

```python
# Checkpoint: the size/count gap is exactly the surveys that never came back
gap = dept_report["visits"].sum() - dept_report["rated"].sum()
print(f"Visits without a satisfaction score: {gap:,}")
print(f"Blank satisfaction values:           {visits['satisfaction'].isna().sum():,}")

# Checkpoint: nunique does not add up across groups
print(f"\nSum of per-department patients: {dept_report['patients'].sum():,}")
print(f"Distinct patients overall:      {visits['patient_id'].nunique():,}")
```

**Expect:** `2,984` twice: the whole size/count gap is the unreturned surveys. The per-department patient counts add up to 63,091, more than twice the 28,929 distinct patients, because an adult seen in both Cardiology and Endocrinology counts once in each. Row counts add up across groups; distinct counts do not.

## Part 2: Grouping by Two Keys

Two keys means one group per observed combination. The result carries a MultiIndex, one level per key; `.unstack()` moves the inner level into columns so the same numbers read as a grid.

### Summarize by Department and Clinic

```python
by_dept_clinic = visits.groupby(["department", "clinic"]).agg({
    "wait_min": ["mean", "count"],
    "satisfaction": "mean",
})
print(f"Groups: {len(by_dept_clinic)}")
by_dept_clinic.head(10).round(1)
```

**Expect:** `Groups: 30` (six departments × five clinics, all observed), then the first 10 rows of a two-level index: Cardiology's five clinics, then Dermatology's. Cardiology waits about 30 minutes at every clinic.

```python
# The same means as a grid: one row per department, one column per clinic
wait_grid = visits.groupby(["department", "clinic"])["wait_min"].mean().unstack()
wait_grid.round(1)
```

**Expect:** a 6 × 5 grid with clinics in alphabetical order (Central, East, North, South, West). Each row is nearly flat: clinic was assigned at random, so the department, not the clinic, sets the wait.

### A Small Table You Can Read Row by Row

The 100,000-row result is too tall to check by eye. Two days of visit counts from two clinics make the same structure easy to follow.

```python
daily = pd.DataFrame({
    "day": ["Mon", "Mon", "Mon", "Mon", "Tue", "Tue", "Tue", "Tue"],
    "clinic": ["North", "North", "South", "South", "North", "North", "South", "South"],
    "department": ["Cardiology", "Pediatrics", "Cardiology", "Pediatrics",
                   "Cardiology", "Pediatrics", "Cardiology", "Dermatology"],
    "visits": [42, 35, 38, 51, 40, 30, 36, 18],
    "no_shows": [3, 5, 2, 6, 4, 2, 1, 2],
})
print(daily)

# Select the numeric columns first: summing "day" would glue the text together
two_day = daily.groupby(["clinic", "department"])[["visits", "no_shows"]].sum()
print()
print(two_day)
print()
print(two_day["visits"].unstack())
```

**Expect:** 8 input rows become 5 groups, one per observed clinic and department pair. North Cardiology shows `82` visits and `7` no-shows (42 + 40 and 3 + 4). In the grid, North–Dermatology is `NaN`: North logged no Dermatology visits on either day, which is not the same as a count someone recorded as zero.

### Work with the MultiIndex

`two_day` has two index levels: `clinic` on the outside, `department` inside.

```python
# .loc on the outer level keeps the inner level as the index
print("North only:")
print(two_day.loc["North"])

# reset_index() (Lecture 06) turns both levels back into ordinary columns
print("\nFlattened:")
print(two_day.reset_index())
```

**Expect:** North's two departments, Cardiology and Pediatrics, with `department` as the index; then 5 rows with `clinic` and `department` as ordinary columns and a `0..4` index.

## Part 3: Pivot Tables

A pivot table is the two-key grouping from Part 2 in one call: `index` picks the row key, `columns` the column key, and `aggfunc` the summary in each cell.

### The Pivot Table and Its GroupBy Twin

```python
wait_pivot = pd.pivot_table(
    visits, values="wait_min", index="department", columns="clinic", aggfunc="mean",
)
print(wait_pivot.round(1))
print("\nSame table as groupby(...).mean().unstack()?", wait_pivot.equals(wait_grid))

# sort=False keeps whichever key turned up first in the rows
unsorted = pd.pivot_table(
    visits, values="wait_min", index="department", columns="clinic", aggfunc="mean",
    sort=False,
)
print("\nRow order with sort=True: ", list(wait_pivot.index))
print("Row order with sort=False:", list(unsorted.index))
```

**Expect:** the same 6 × 5 grid as `wait_grid`, then `True`. `sort=True` lists departments alphabetically; `sort=False` lists them in the order each first appears in the rows: Family Medicine, Dermatology, Pediatrics, Orthopedics, Cardiology, Endocrinology. Pass `sort=` when a reader will compare your table with another one.

### Several Summaries at Once

```python
# aggfunc as a list gives two-level column labels: result["count"], result["mean"]
wait_multi = pd.pivot_table(
    visits, values="wait_min", index="department", columns="clinic",
    aggfunc=["count", "mean"],
)
print("=== Visits behind each cell ===")
print(wait_multi["count"])
print("\n=== Mean wait in each cell ===")
print(wait_multi["mean"].round(1))
# .sum() adds up each clinic's column; the second .sum() adds those totals
print("\nTotal visits counted:", wait_multi["count"].sum().sum())
```

**Expect:** two tables with the same shape, and `Total visits counted: 100000`. The smallest cell is Dermatology at North with 1,647 visits, so every mean rests on well over a thousand visits. Read the two tables together: a mean is only as trustworthy as the count underneath it.

### Totals with `margins`

```python
# Total patient-minutes spent waiting, with a Total row and column
wait_totals = pd.pivot_table(
    visits, values="wait_min", index="department", columns="clinic",
    aggfunc="sum", margins=True, margins_name="Total",
)
print(wait_totals)

# Checkpoint: the corner cell is every minute of waiting in the log
print(f"\nCorner cell:     {wait_totals.loc['Total', 'Total']:,.0f}")
print(f"Sum of wait_min: {visits['wait_min'].sum():,.0f}")
```

**Expect:** a `Total` row and column, and `2,485,272` on both checkpoint lines. Totals come from the raw rows, not from adding up rounded cells.

### Absent Cells: `NaN` or Zero?

```python
# Nurses scheduled per department and clinic; Dermatology has no South clinic
nurses = pd.DataFrame({
    "department": ["Cardiology", "Cardiology", "Pediatrics", "Pediatrics", "Dermatology"],
    "clinic": ["North", "South", "North", "South", "North"],
    "nurses": [4, 6, 9, 5, 2],
})

raw = pd.pivot_table(nurses, values="nurses", index="department", columns="clinic", aggfunc="sum")
print("=== The absent combination shows as NaN ===")
print(raw)

filled = pd.pivot_table(nurses, values="nurses", index="department", columns="clinic",
                        aggfunc="sum", fill_value=0)
print("\n=== fill_value=0: Dermatology really has no nurses in the South ===")
print(filled)
```

**Expect:** Dermatology–South is `NaN` in the first table and `0` in the second, and the other cells are unchanged. For a count of nurses, zero is the true answer, so `fill_value=0` is right here. It would be wrong for a mean wait: filling an empty cell with 0 would report a zero-minute wait that never happened.

### Read a Pivot Table as a Heatmap

```python
# A pivot table is already the shape sns.heatmap() wants (Lecture 07)
fig, ax = plt.subplots(figsize=(9, 5))
# fmt=".1f" writes each value with one decimal, like an f-string's :.1f
sns.heatmap(wait_pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
ax.set_title("Mean wait (minutes) by department and clinic")
ax.set_xlabel("Clinic")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
```

**Expect:** horizontal bands of color: Orthopedics darkest, Pediatrics palest, and each row nearly one color across the clinics. A heatmap makes that pattern obvious in a way 30 printed numbers do not.

### Back to Long Form

```python
# stack() moves the columns back into the index, undoing unstack()
long_form = wait_pivot.stack()
print(long_form.head(6).round(1))
print("\nRows in long form:", len(long_form))
print("Round trip back to the grid?", long_form.unstack().equals(wait_pivot))
```

**Expect:** 30 rows (six departments × five clinics) and `True`: wide and long are two layouts of the same summary.

## Part 4: Cross-Tabulations

`pd.crosstab()` is the counting special case: hand it two columns and it reports how many rows fall in each combination.

### Count Every Combination

```python
visit_counts = pd.crosstab(visits["department"], visits["clinic"], margins=True)
print(visit_counts)
print("\nGrand total:", visit_counts.loc["All", "All"])
```

**Expect:** the same counts as the `count` table above, plus an `All` row and column; the grand total is `100000`. An absent combination would show `0` here, not `NaN`: crosstab counts rows, and no rows is a count of zero.

### Summarize a Third Column Instead of Counting

```python
# values= plus aggfunc= turns crosstab into a pivot table
mean_wait = pd.crosstab(visits["department"], visits["clinic"],
                        values=visits["wait_min"], aggfunc="mean")
print(mean_wait.round(1))
print("\nSame numbers as the pivot table?", mean_wait.round(6).equals(wait_pivot.round(6)))
```

**Expect:** the same grid as the pivot table, and `True`.

### Two-Level Rows

```python
# Bin age into bands (Lecture 05), then use two row keys
visits["age_band"] = pd.cut(
    visits["age"], bins=[-1, 17, 39, 64, 120], labels=["0-17", "18-39", "40-64", "65+"],
)
band_counts = pd.crosstab([visits["department"], visits["age_band"]], visits["clinic"])
print(band_counts)
print("\nRows in the table:", len(band_counts))
```

**Expect:** 16 rows, not 24 (six departments × four bands). Pediatrics has only a `0-17` row and the adult departments have none, because a combination with no visits never becomes a row.
