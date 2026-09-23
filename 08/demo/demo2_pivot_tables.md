# Demo 2: Group Coverage and Result Shapes

## Learning Objectives
- See which rows and which groups a default summary silently leaves out
- Keep unrecorded keys visible with `dropna=False`
- Set a reporting order with `pd.Categorical` and show empty groups with `observed=False`
- Add group context to every row with `transform`
- Keep or drop whole groups with `filter`
- Run a custom function per group with `apply`

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set inline plotting for Jupyter
%matplotlib inline

# Set random seed for reproducibility
np.random.seed(42)
```

## Part 1: Which Groups Appear in a Summary

This demo builds the same employee table Demo 1 used, so it runs on its own. Rerun the next cell even if you still have Demo 1 open.

### Create Sample Data

```python
# Create large-scale employee dataset (100,000 rows)
print("=== Creating Large-Scale Employee Dataset ===")
n_employees = 100000

# Generate realistic employee data
departments = [
    "Sales",
    "Engineering",
    "Marketing",
    "HR",
    "Finance",
    "Operations",
]
regions = ["North", "South", "East", "West", "Central"]
employee_names = [f"Emp_{i:05d}" for i in range(n_employees)]

# Create correlated data: Engineering has higher salaries, Sales varies more
dept_salary_base = {
    "Engineering": 85000,
    "Finance": 75000,
    "Marketing": 65000,
    "Sales": 60000,
    "HR": 55000,
    "Operations": 50000,
}

dept_salary_std = {
    "Engineering": 15000,
    "Finance": 12000,
    "Marketing": 10000,
    "Sales": 20000,  # Higher variance
    "HR": 8000,
    "Operations": 7000,
}

# Generate data
np.random.seed(42)
departments_list = np.random.choice(departments, n_employees)
regions_list = np.random.choice(regions, n_employees)

# Create correlated salaries based on department
salaries = []
for dept in departments_list:
    base = dept_salary_base[dept]
    std = dept_salary_std[dept]
    salary = np.random.normal(base, std)
    salaries.append(max(30000, salary))  # Minimum wage floor

# Experience correlates with salary (but with noise)
experience = []
for salary in salaries:
    # More experienced employees tend to earn more, but with variation
    exp_base = (salary - 40000) / 8000
    exp = max(0, int(np.random.normal(exp_base, 2)))
    experience.append(min(exp, 30))  # Cap at 30 years

# Create DataFrame
df = pd.DataFrame({
    "Employee": employee_names,
    "Department": departments_list,
    "Region": regions_list,
    "Salary": np.round(salaries, 2),
    "Experience": experience,
})

# Add some additional features
df["Years_At_Company"] = np.round(np.random.uniform(0.5, 14, n_employees), 1)
df["Performance_Score"] = np.random.uniform(1, 5, n_employees)
df["Bonus"] = df["Salary"] * df["Performance_Score"] * 0.1

# Bonuses are paid after a full year, so this year's hires have no bonus on file
df.loc[df["Years_At_Company"] < 1, "Bonus"] = np.nan

print(f"Dataset shape: {df.shape}")
print(f"Employees in their first year (no bonus yet): {df['Bonus'].isna().sum():,}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())
```

### Two Realities the Default Summary Hides

```python
# Work on a copy so the original table stays clean for Parts 2-4
reported = df.copy()

# 1. Some records arrived without a department
unrecorded = np.random.choice(reported.index, size=250, replace=False)
reported.loc[unrecorded, "Department"] = None

# 2. Legal opened this quarter and has not hired anyone yet
reporting_order = [
    "Engineering",
    "Finance",
    "Marketing",
    "Sales",
    "HR",
    "Operations",
    "Legal",
]

print("=== What the Table Contains ===")
print(f"Rows: {len(reported):,}")
print(f"Rows with no department recorded: {reported['Department'].isna().sum()}")
print(f"Departments named in the report: {len(reporting_order)}")
print(f"Departments present in the data: {reported['Department'].nunique()}")
```

Six departments have employees, a seventh exists on paper, and 250 rows name no department at all. A default `groupby` shows six groups and says nothing about the rest.

### Missing Keys: `dropna`

```python
# The default drops rows whose key is missing, before the split
default_counts = reported.groupby("Department")["Salary"].count()
kept_counts = reported.groupby("Department", dropna=False)["Salary"].count()

print("=== Default: dropna=True ===")
print(default_counts)
print(f"Employees counted: {default_counts.sum():,} of {len(reported):,}")

print("\n=== dropna=False ===")
print(kept_counts)
print(f"Employees counted: {kept_counts.sum():,} of {len(reported):,}")
```

The default report loses 250 people with no warning; `dropna=False` keeps them as a `NaN` group, listed last, so the gap is visible and someone can go find the missing departments.

### Reporting Order and Empty Groups: `pd.Categorical` and `observed`

```python
# Declare the allowed values and the order the report should use
reported["Department"] = pd.Categorical(
    reported["Department"], categories=reporting_order, ordered=True
)

present_only = reported.groupby("Department", observed=True, dropna=False)[
    "Salary"
].agg(["count", "mean"])
every_department = reported.groupby("Department", observed=False, dropna=False)[
    "Salary"
].agg(["count", "mean"])

print("=== observed=True: Only Departments With Employees ===")
print(present_only.round(0))
print("\n=== observed=False: Every Declared Department ===")
print(every_department.round(0))
```

Both tables follow the order you declared, not alphabetical order. Only the second one shows Legal, with a count of `0` and a mean of `NaN`: no salaries to average is not the same as a mean salary of zero.

### The Same Choices in a Pivot Table

```python
# pivot_table makes the same two choices, and dropna= has one extra job here
staffing = pd.pivot_table(
    reported,
    values="Salary",
    index="Department",
    columns="Region",
    aggfunc="count",
    observed=False,
    dropna=False,
    fill_value=0,
)
print("=== Headcount by Department × Region (dropna=False) ===")
print(staffing)
print("\nRows in the table:", len(staffing))

# The same counts with the default dropna=True
counted_default = pd.pivot_table(
    reported,
    values="Salary",
    index="Department",
    columns="Region",
    aggfunc="count",
    observed=False,
    dropna=True,
    fill_value=0,
)
print("\n=== Same Counts with dropna=True ===")
print(counted_default)
print("\nRows in the table:", len(counted_default))

# Mean salary instead of a count, with the default dropna=True
mean_default = pd.pivot_table(
    reported,
    values="Salary",
    index="Department",
    columns="Region",
    aggfunc="mean",
    observed=False,
    dropna=True,
)
print("\n=== Mean Salary with dropna=True ===")
print(mean_default.round(0))
print("\nRows in the table:", len(mean_default))

# The same means with dropna=False, to see which rows the default removed
mean_kept = pd.pivot_table(
    reported,
    values="Salary",
    index="Department",
    columns="Region",
    aggfunc="mean",
    observed=False,
    dropna=False,
)
print("\n=== Mean Salary with dropna=False ===")
print(mean_kept.round(0))
print("\nRows in the table:", len(mean_kept))
```

`observed=False` asks for every declared department, so Legal gets a row in three of the four tables; only the mean with `dropna=True` loses it. `dropna=False` adds the rows whose department was never recorded: eight rows, the six staffed departments, Legal, and the `NaN` group. Switching to `dropna=True` drops that `NaN` group but keeps Legal at seven rows, because counting an empty group gives `0`, and `0` is not missing. The extra job `dropna=` has in a pivot table is about the result, not the category: `dropna=True` also removes any row or column whose cells all came out `NaN`. Both rules cut a row from the table of means: the `NaN` group goes because its key is missing, and Legal goes because its five cells are all `NaN`, leaving six rows. The last table keeps both, and its `NaN` row is not empty at all: five real means, from `61288` in East to `67633` in Central, averaging the salaries of the 250 people the default report never mentions. Counting rows is exactly the case where `fill_value=0` is right.

One thing to read carefully: the `NaN` row is not last here. `groupby(dropna=False)` lists it after every named group, but `pivot_table` with a categorical key, `observed=False`, and a `columns=` key puts it at the top, above Engineering, in both tables that keep it. Find that row by its label, not by its position.

### Show the Difference

```python
# Every group the data can produce, marking the ones a default report hides
full_counts = reported.groupby("Department", observed=False, dropna=False)[
    "Salary"
].count()
default_groups = list(
    reported.groupby("Department", observed=True)["Salary"].count().index
)

labels = [
    name if pd.notna(name) else "(not recorded)" for name in full_counts.index
]
hidden = [
    pd.isna(name) or name not in default_groups for name in full_counts.index
]
colors = ["tab:red" if is_hidden else "tab:blue" for is_hidden in hidden]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(labels, full_counts.to_numpy(), color=colors, alpha=0.85)
for i, (value, is_hidden) in enumerate(zip(full_counts.to_numpy(), hidden)):
    if is_hidden:
        ax.text(
            i,
            value + 600,
            "missing from the default report",
            ha="center",
            va="bottom",
            rotation=90,
            color="tab:red",
            fontsize=9,
            fontweight="bold",
        )
ax.set_title(
    "What the Default Report Leaves Out", fontsize=14, fontweight="bold"
)
ax.set_ylabel("Employees Counted")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

Two bars are red: Legal, an empty group with a real headcount question behind it, and the 250 employees whose department was never recorded. A default `groupby` prints neither one.

## Part 2: Transform Keeps Every Row

`transform` computes a statistic per group and copies it back to every row of that group, so the result has the same length and index as the input and can become a new column.

### Add Department Statistics to Every Row

```python
# Transform: Add group statistics as new columns
print("=== Transform Operations ===")
print("Adding department-level statistics to each employee record...")

df["Dept_Salary_Mean"] = df.groupby("Department")["Salary"].transform("mean")
df["Dept_Salary_Std"] = df.groupby("Department")["Salary"].transform("std")
df["Dept_Salary_Median"] = df.groupby("Department")["Salary"].transform(
    "median"
)
df["Salary_Normalized"] = df.groupby("Department")["Salary"].transform(
    lambda x: (x - x.mean()) / x.std()
)
df["Salary_Percentile_Rank"] = df.groupby("Department")["Salary"].transform(
    lambda x: pd.qcut(
        x, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
)

# Calculate how many standard deviations each employee is from their department mean
df["Salary_Z_Score"] = (df["Salary"] - df["Dept_Salary_Mean"]) / df[
    "Dept_Salary_Std"
]

print("Sample of transformed data:")
sample_cols = [
    "Department",
    "Employee",
    "Salary",
    "Dept_Salary_Mean",
    "Dept_Salary_Std",
    "Salary_Normalized",
    "Salary_Z_Score",
    "Salary_Percentile_Rank",
]
print(df[sample_cols].head(10))

# Visualize transform results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Distribution of normalized salaries by department
for dept in df["Department"].unique():
    dept_data = df[df["Department"] == dept]["Salary_Normalized"]
    axes[0, 0].hist(dept_data, alpha=0.5, label=dept, bins=30)
axes[0, 0].set_title(
    "Normalized Salary Distribution by Department",
    fontsize=14,
    fontweight="bold",
)
axes[0, 0].set_xlabel("Normalized Salary (Z-score)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Z-score distribution
df["Salary_Z_Score"].hist(
    bins=50, ax=axes[0, 1], color="steelblue", edgecolor="black"
)
axes[0, 1].axvline(
    0, color="red", linestyle="--", linewidth=2, label="Department Mean"
)
axes[0, 1].set_title(
    "Salary Z-Score Distribution (All Employees)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 1].set_xlabel("Z-Score (Standard Deviations from Dept Mean)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Salary vs Department Mean (scatter)
for dept in df["Department"].unique():
    dept_data = df[df["Department"] == dept]
    axes[1, 0].scatter(
        dept_data["Dept_Salary_Mean"],
        dept_data["Salary"],
        alpha=0.3,
        label=dept,
        s=10,
    )
axes[1, 0].plot(
    [df["Salary"].min(), df["Salary"].max()],
    [df["Salary"].min(), df["Salary"].max()],
    "r--",
    linewidth=2,
    label="y=x (at mean)",
)
axes[1, 0].set_title(
    "Individual Salary vs Department Mean", fontsize=14, fontweight="bold"
)
axes[1, 0].set_xlabel("Department Mean Salary ($)")
axes[1, 0].set_ylabel("Individual Salary ($)")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Percentile rank distribution
percentile_counts = df["Salary_Percentile_Rank"].value_counts().sort_index()
percentile_counts.plot(kind="bar", ax=axes[1, 1], color="coral")
axes[1, 1].set_title(
    "Salary Percentile Rank Distribution", fontsize=14, fontweight="bold"
)
axes[1, 1].set_xlabel("Percentile Rank")
axes[1, 1].set_ylabel("Number of Employees")
axes[1, 1].tick_params(axis="x", rotation=0)
axes[1, 1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

`df` still has 100,000 rows in the same order: `transform` added columns, it did not summarize. `Salary_Normalized` and `Salary_Z_Score` agree to the last decimal, because both compare a salary with its own department's mean and spread. The z-score histogram is centred on 0 by construction, and the department histograms overlap once each one is measured against its own mean.

## Part 3: Filter Keeps or Drops Whole Groups

`filter` tests each whole group and keeps every row of the groups that pass. Rows are never changed, only kept or dropped.

### Keep Only the Groups That Qualify

```python
# Filter: Keep only departments with more than threshold employees
print("=== Filter Operations ===")
min_employees = 15000  # Filter departments with at least 15,000 employees
filtered_large_depts = df.groupby("Department").filter(
    lambda x: len(x) >= min_employees
)
print(f"Departments with at least {min_employees:,} employees:")
print(f"Filtered dataset shape: {filtered_large_depts.shape}")
print(f"Departments kept: {filtered_large_depts['Department'].unique()}")

# Filter: Keep only departments with average salary > threshold
salary_threshold = 65000
high_salary_depts = df.groupby("Department").filter(
    lambda x: x["Salary"].mean() > salary_threshold
)
print(f"\nDepartments with average salary > ${salary_threshold:,}:")
print(f"Filtered dataset shape: {high_salary_depts.shape}")
print(f"Departments kept: {high_salary_depts['Department'].unique()}")

# Filter: Keep departments with high variance (interesting for analysis)
high_variance_depts = df.groupby("Department").filter(
    lambda x: x["Salary"].std() > 12000
)
print(f"\nDepartments with salary std > $12,000:")
print(f"Filtered dataset shape: {high_variance_depts.shape}")
print(f"Departments kept: {high_variance_depts['Department'].unique()}")

# Visualize filtering effects
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Department sizes (original vs filtered)
dept_counts_original = df["Department"].value_counts().sort_index()
dept_counts_filtered = (
    filtered_large_depts["Department"].value_counts().sort_index()
)
x_pos = np.arange(len(dept_counts_original.index))
width = 0.35
axes[0].bar(
    x_pos - width / 2,
    dept_counts_original.values,
    width,
    label="Original",
    alpha=0.7,
)
axes[0].bar(
    x_pos + width / 2,
    dept_counts_filtered.values,
    width,
    label="Filtered",
    alpha=0.7,
)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(dept_counts_original.index, rotation=45)
axes[0].set_title(
    f"Department Size: Original vs Filtered (min {min_employees:,})",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_ylabel("Number of Employees")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# 2. Salary distributions: original vs high-salary departments
axes[1].hist(
    df["Salary"], bins=50, alpha=0.5, label="All Departments", color="blue"
)
axes[1].hist(
    high_salary_depts["Salary"],
    bins=50,
    alpha=0.5,
    label=f"Avg Salary > ${salary_threshold:,}",
    color="red",
)
axes[1].set_title(
    "Salary Distribution: Filtering Effect", fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Salary ($)")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].grid(alpha=0.3)

# 3. Department salary statistics comparison
dept_stats_all = df.groupby("Department")["Salary"].agg(["mean", "std"])
dept_stats_filtered = high_salary_depts.groupby("Department")["Salary"].agg([
    "mean",
    "std",
])
# Only compare departments that exist in both datasets
common_depts = dept_stats_all.index.intersection(dept_stats_filtered.index)
if len(common_depts) > 0:
    dept_stats_all_subset = dept_stats_all.loc[common_depts]
    dept_stats_filtered_subset = dept_stats_filtered.loc[common_depts]
    x_pos = np.arange(len(common_depts))
    axes[2].bar(
        x_pos - width / 2,
        dept_stats_all_subset["mean"],
        width,
        label="All",
        alpha=0.7,
        yerr=dept_stats_all_subset["std"],
        capsize=5,
    )
    axes[2].bar(
        x_pos + width / 2,
        dept_stats_filtered_subset["mean"],
        width,
        label="Filtered",
        alpha=0.7,
        yerr=dept_stats_filtered_subset["std"],
        capsize=5,
    )
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(common_depts, rotation=45)
else:
    axes[2].text(
        0.5,
        0.5,
        "No common departments\nbetween filtered datasets",
        ha="center",
        va="center",
        transform=axes[2].transAxes,
    )
axes[2].set_title(
    f"Mean Salary: Original vs Filtered (avg > ${salary_threshold:,})",
    fontsize=12,
    fontweight="bold",
)
axes[2].set_ylabel("Mean Salary ($)")
axes[2].legend()
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

`filter` returns rows, not groups. All six departments clear 15,000 employees, so the first result keeps all 100,000 rows. Only Engineering and Finance average more than $65,000 (33,609 rows) and only Engineering and Sales have a salary spread above $12,000 (33,391 rows). The kept rows are the originals, unchanged.

## Part 4: Apply Runs Your Own Function per Group

`apply` hands each group to your function as a small DataFrame and stitches the results together. Use it when no built-in aggregation fits; it is the slowest of the four, and its output shape depends on what your function returns.

### Custom Statistics and Top-N per Group

```python
# Apply: Custom function for comprehensive salary statistics
def comprehensive_salary_stats(group):
    """Calculate comprehensive statistics for a group"""
    return pd.Series({
        "count": len(group),
        "mean": group["Salary"].mean(),
        "median": group["Salary"].median(),
        "std": group["Salary"].std(),
        "min": group["Salary"].min(),
        "max": group["Salary"].max(),
        "range": group["Salary"].max() - group["Salary"].min(),
        "q25": group["Salary"].quantile(0.25),
        "q75": group["Salary"].quantile(0.75),
        "iqr": group["Salary"].quantile(0.75) - group["Salary"].quantile(0.25),
        "mean_experience": group["Experience"].mean(),
        "mean_performance": group["Performance_Score"].mean(),
    })

print("=== Apply Operations ===")
print("Comprehensive statistics by department:")
dept_stats_apply = df.groupby("Department").apply(
    comprehensive_salary_stats, include_groups=False
)
print(dept_stats_apply)

# Apply: Get top N earners in each department
top_n = 5
top_earners = df.groupby("Department").apply(
    lambda x: x.nlargest(top_n, "Salary"), include_groups=False
)
print(f"\nTop {top_n} earners per department:")
# Department is in the index, so we need to reset it or access it differently
top_earners_display = top_earners.reset_index(level=0, drop=False)
print(
    top_earners_display[
        ["Department", "Employee", "Salary", "Experience", "Performance_Score"]
    ]
)

# Apply: Calculate department-specific percentiles
def calculate_percentiles(group):
    """Calculate salary percentiles for a group"""
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    return pd.Series({
        f"p{p}": group["Salary"].quantile(p / 100) for p in percentiles
    })

dept_percentiles = df.groupby("Department").apply(
    calculate_percentiles, include_groups=False
)
print("\nSalary percentiles by department:")
print(dept_percentiles)

# Visualize apply results
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Top earners visualization
top_earners_plot = (
    top_earners_display.groupby("Department")["Salary"]
    .mean()
    .sort_values(ascending=False)
)
top_earners_plot.plot(kind="barh", ax=axes[0, 0], color="gold")
axes[0, 0].set_title(
    f"Mean Salary of Top {top_n} Earners by Department",
    fontsize=14,
    fontweight="bold",
)
axes[0, 0].set_xlabel("Mean Salary ($)")
axes[0, 0].grid(axis="x", alpha=0.3)

# 2. Percentile comparison across departments
dept_percentiles.T.plot(kind="bar", ax=axes[0, 1], width=0.8)
axes[0, 1].set_title(
    "Salary Percentiles by Department", fontsize=14, fontweight="bold"
)
axes[0, 1].set_ylabel("Salary ($)")
axes[0, 1].set_xlabel("Percentile")
axes[0, 1].legend(
    title="Department", bbox_to_anchor=(1.05, 1), loc="upper left"
)
axes[0, 1].tick_params(axis="x", rotation=0)
axes[0, 1].grid(axis="y", alpha=0.3)

# 3. IQR comparison (shows salary spread)
iqr_data = dept_stats_apply["iqr"].sort_values(ascending=False)
iqr_data.plot(kind="bar", ax=axes[1, 0], color="steelblue")
axes[1, 0].set_title(
    "Interquartile Range (IQR) by Department", fontsize=14, fontweight="bold"
)
axes[1, 0].set_ylabel("IQR ($)")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].grid(axis="y", alpha=0.3)

# 4. Mean vs Median comparison (shows skewness)
comparison_df = pd.DataFrame({
    "Mean": dept_stats_apply["mean"],
    "Median": dept_stats_apply["median"],
})
comparison_df.plot(kind="bar", ax=axes[1, 1], width=0.8)
axes[1, 1].set_title(
    "Mean vs Median Salary by Department", fontsize=14, fontweight="bold"
)
axes[1, 1].set_ylabel("Salary ($)")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

The shape of an `apply` result follows the function you pass. Returning a `Series` of statistics gives one row per department, just like `agg`; returning `nlargest(5, "Salary")` gives whole employee rows, five per department, with the department in an outer index level. When the function could have been `agg` or `transform`, prefer those: they run as compiled code instead of once per group.

## Key Takeaways

1. **Coverage is a choice**: `dropna=` decides whether rows with a missing key are counted, `observed=` whether empty categories appear
2. **Declare the categories**: `pd.Categorical(..., categories=[...], ordered=True)` fixes the reporting order and names the groups that should exist
3. **Pivot tables make the same choices**: plus `dropna=False` to keep rows with a missing key and any row or column whose cells all came out `NaN`
4. **Transform**: same rows, same index, one new column of group context
5. **Filter**: original rows from the groups that pass a whole-group test
6. **Apply**: any per-group function, at the cost of speed and a shape you have to check
7. **Check the totals**: every result in this demo can be checked by counting rows before and after

## Next Steps

- Rerun Part 1 with `dropna=True` and confirm exactly 250 employees disappear
- Rewrite one `apply` call in Part 4 as `agg` or `transform`, and compare the results

