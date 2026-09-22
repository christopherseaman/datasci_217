# Demo 1: GroupBy Summaries and Pivot Tables

## Learning Objectives
- Split rows into groups and combine one summary row per group
- State the grain of the input and of the result before every grouping
- Summarize several columns at once with `.agg()` and named aggregation
- Group by two keys and read the result as a MultiIndex or as a grid
- Build the same grid with `pd.pivot_table()`, add totals, and count combinations with `pd.crosstab()`

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set inline plotting for Jupyter
%matplotlib inline

# Set random seed for reproducibility
np.random.seed(42)
```

## Part 1: One Row per Group

The input table has one row per employee. Every question in this part has an answer with one row per department, so each answer is a split (by `Department`), an apply (mean, count, ...), and a combine.

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

### Basic Aggregation

```python
# Group by department and calculate comprehensive statistics
print("=== Basic Aggregation ===")
print("Mean salary by department:")
dept_salary_mean = df.groupby("Department")["Salary"].mean()
print(dept_salary_mean)

print("\n=== Comprehensive Department Statistics ===")
dept_stats = df.groupby("Department").agg({
    "Salary": ["mean", "median", "std", "min", "max", "count"],
    "Experience": ["mean", "max"],
    "Performance_Score": "mean",
    "Bonus": "sum",
})
print(dept_stats)

# Visualize department statistics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Mean salary by department
dept_salary_mean.plot(kind="bar", ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title(
    "Mean Salary by Department", fontsize=14, fontweight="bold"
)
axes[0, 0].set_ylabel("Salary ($)")
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].grid(axis="y", alpha=0.3)

# 2. Salary distribution by department (box plot)
df.boxplot(column="Salary", by="Department", ax=axes[0, 1])
axes[0, 1].set_title(
    "Salary Distribution by Department", fontsize=14, fontweight="bold"
)
axes[0, 1].set_xlabel("Department")
axes[0, 1].set_ylabel("Salary ($)")
axes[0, 1].tick_params(axis="x", rotation=45)

# 3. Employee count by department
dept_counts = df["Department"].value_counts().sort_index()
dept_counts.plot(kind="bar", ax=axes[1, 0], color="coral")
axes[1, 0].set_title(
    "Employee Count by Department", fontsize=14, fontweight="bold"
)
axes[1, 0].set_ylabel("Number of Employees")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].grid(axis="y", alpha=0.3)

# 4. Total bonus by department
dept_bonus = df.groupby("Department")["Bonus"].sum()
dept_bonus.plot(kind="bar", ax=axes[1, 1], color="green")
axes[1, 1].set_title(
    "Total Bonus by Department", fontsize=14, fontweight="bold"
)
axes[1, 1].set_ylabel("Total Bonus ($)")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

### Named Aggregation: One Flat Row per Department

```python
# Named aggregation: output name = (source column, function)
# as_index=False keeps Department as an ordinary column instead of the index
# size counts rows; count counts recorded Bonus values, so the two differ
dept_report = df.groupby("Department", as_index=False).agg(
    employees=("Employee", "size"),
    people_with_bonus=("Bonus", "count"),
    mean_salary=("Salary", "mean"),
    median_salary=("Salary", "median"),
    top_salary=("Salary", "max"),
    mean_experience=("Experience", "mean"),
)
print("=== One Flat Row per Department ===")
print(dept_report.round(1))

# Checkpoint: state the grain before and after
print(f"\nInput grain:  one row per employee  -> {len(df):,} rows")
print(f"Output grain: one row per department -> {len(dept_report)} rows")
print(f"Employees accounted for: {dept_report['employees'].sum():,}")

# Checkpoint: the size/count gap is exactly the first-year hires
gap = dept_report["employees"].sum() - dept_report["people_with_bonus"].sum()
print(f"\nEmployees without a recorded bonus: {gap:,}")
print(f"Employees in their first year:      {(df['Years_At_Company'] < 1).sum():,}")
print("\nFirst-year hires per department:")
print(df[df["Bonus"].isna()].groupby("Department").size())
```

Expect six departments, `employees` summing to 100,000, and flat column names (no two-level labels) because every aggregation was named. `employees` and `people_with_bonus` are different numbers in every row: `size` counts the rows in the group, while `count` counts the rows whose `Bonus` is recorded. The 3,337 employees hired within the last year have no bonus yet, and those are exactly the rows in the gap - 539 in Engineering, 573 in Finance, 575 in HR, 558 in Marketing, 575 in Operations, and 517 in Sales.

## Part 2: Grouping by Two Keys

Two keys means one group per observed combination. The result carries a MultiIndex, one level per key; `.unstack()` moves the inner level into columns so the same numbers read as a grid.

### Summarize by Department and Region

```python
# Group by multiple columns - Department and Region
print("=== Multi-column Grouping: Department × Region ===")
result = df.groupby(["Department", "Region"]).agg({
    "Salary": ["mean", "std", "count"],
    "Experience": "mean",
    "Performance_Score": "mean",
    "Bonus": "sum",
})
print(result.head(20))

# Visualize multi-dimensional grouping
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Heatmap of mean salary by Department × Region (Lecture 07)
salary_grid = df.groupby(["Department", "Region"])["Salary"].mean().unstack()
sns.heatmap(
    salary_grid, annot=True, fmt=",.0f", cmap="YlOrRd", ax=axes[0],
    cbar_kws={"label": "Mean Salary ($)"},
)
axes[0].set_title(
    "Mean Salary Heatmap: Department × Region", fontsize=14, fontweight="bold"
)
axes[0].set_xlabel("Region")
axes[0].set_ylabel("Department")
axes[0].tick_params(axis="y", rotation=0)

# 2. Grouped bar chart
dept_region_salary = (
    df.groupby(["Department", "Region"])["Salary"].mean().unstack()
)
dept_region_salary.plot(kind="bar", ax=axes[1], width=0.8)
axes[1].set_title(
    "Mean Salary by Department and Region", fontsize=14, fontweight="bold"
)
axes[1].set_ylabel("Mean Salary ($)")
axes[1].set_xlabel("Department")
axes[1].legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

Six departments × five regions gives 30 groups, all of them present in 100,000 rows, so `result` has 30 rows and a two-level index. The columns are two-level as well, because `.agg()` received a dictionary rather than named aggregations.

### A Small Table You Can Read Row by Row

The 100,000-row result is too tall to check by eye. Six rows make the same structure obvious: two keys in, one row per observed combination out.

```python
# Create hierarchical data
hierarchical_data = {
    "Region": ["North", "North", "South", "South", "North", "South"],
    "Department": [
        "Sales",
        "Engineering",
        "Sales",
        "Engineering",
        "Marketing",
        "Marketing",
    ],
    "Revenue": [100000, 150000, 120000, 180000, 80000, 90000],
    "Employees": [5, 8, 6, 10, 4, 5],
}

hierarchical_df = pd.DataFrame(hierarchical_data)
print("=== Hierarchical Grouping ===")
print("Original data:")
print(hierarchical_df)

# Hierarchical grouping
hierarchical_grouped = hierarchical_df.groupby(["Region", "Department"]).sum()
print("\nHierarchical grouping:")
print(hierarchical_grouped)

# Unstack to wide format
wide_format = hierarchical_grouped.unstack()
print("\nWide format:")
print(wide_format)
```

### Work with the MultiIndex

```python
# Work with MultiIndex
print("=== MultiIndex Operations ===")
print("Index levels:", hierarchical_grouped.index.names)
print("Index values:", hierarchical_grouped.index.values)

# Access specific groups
print("\nNorth region data:")
print(hierarchical_grouped.loc["North"])

# Reset index to flatten
flattened = hierarchical_grouped.reset_index()
print("\nFlattened data:")
print(flattened)
```

`hierarchical_grouped` has six rows and a two-level index (`Region`, `Department`). `.loc["North"]` selects the outer level and leaves the inner one as the index; `.unstack()` turns that inner level into columns; `.reset_index()` flattens both levels back into ordinary columns.

## Part 3: Pivot Tables

A pivot table is the two-key grouping from Part 2 in one call: `index` picks the row key, `columns` the column key, and `aggfunc` the summary in each cell.

### The Pivot Table and Its GroupBy Twin

```python
# Same numbers, one call instead of two steps
# sort=True is the default: order the rows and columns by key, not by luck
salary_pivot = pd.pivot_table(
    df, values="Salary", index="Department", columns="Region", aggfunc="mean",
    sort=True,
)
print("=== Mean Salary by Department × Region ===")
print(salary_pivot.round(0))

twin = df.groupby(["Department", "Region"])["Salary"].mean().unstack()
print("\nSame table as groupby(...).mean().unstack()?", salary_pivot.equals(twin))

# sort=False keeps whichever key turned up first in the rows
unsorted = pd.pivot_table(
    df, values="Salary", index="Department", columns="Region", aggfunc="mean",
    sort=False,
)
print("\nRow order with sort=True: ", list(salary_pivot.index))
print("Row order with sort=False:", list(unsorted.index))
```

The `equals()` check must print `True`: `pivot_table` and the two-key groupby answer the same question in the same layout. The two row orders differ: `sort=True` gives alphabetical Engineering, Finance, HR, Marketing, Operations, Sales, while `sort=False` gives HR, Finance, Marketing, Engineering, Operations, Sales - the order those departments happen to appear in the rows. Pass `sort=` when a reader will compare your table with another one.

### Several Summaries at Once

```python
# aggfunc as a list gives two-level column labels: result["count"], result["mean"]
salary_multi = pd.pivot_table(
    df,
    values="Salary",
    index="Department",
    columns="Region",
    aggfunc=["count", "mean"],
)
print("=== Employees Behind Each Cell ===")
print(salary_multi["count"])
print("\n=== Mean Salary in Each Cell ===")
print(salary_multi["mean"].round(0))
print("\nTotal employees counted:", salary_multi["count"].to_numpy().sum())
```

Read the two tables together: a mean is only as trustworthy as the count underneath it. The counts must still add up to 100,000.

### Totals with `margins`

```python
# margins adds a row and a column of totals computed from the underlying rows
bonus_totals = pd.pivot_table(
    df,
    values="Bonus",
    index="Department",
    columns="Region",
    aggfunc="sum",
    margins=True,
    margins_name="Total",
)
print("=== Total Bonus by Department × Region ===")
print(bonus_totals.round(0))

# Checkpoint: the corner cell is the total of every bonus paid
print(f"\nCorner cell:    {bonus_totals.loc['Total', 'Total']:,.0f}")
print(f"Sum of Bonus:   {df['Bonus'].sum():,.0f}")
```

The two printed numbers must match. Totals come from the raw rows, not from adding up rounded cells.

### Absent Cells: `NaN` or Zero?

```python
# A small table where one combination never happened
new_offices = pd.DataFrame({
    "Department": ["Sales", "Sales", "Engineering", "Engineering", "HR"],
    "Region": ["North", "South", "North", "South", "North"],
    "Headcount": [4, 6, 9, 5, 2],
})
print("=== Raw Rows ===")
print(new_offices)

raw = pd.pivot_table(
    new_offices, values="Headcount", index="Department", columns="Region", aggfunc="sum"
)
print("\n=== Absent Combination Shows as NaN ===")
print(raw)

filled = pd.pivot_table(
    new_offices,
    values="Headcount",
    index="Department",
    columns="Region",
    aggfunc="sum",
    fill_value=0,
)
print("\n=== fill_value=0: HR Really Has Nobody in the South ===")
print(filled)
```

HR–South is missing from the rows, so the first table prints `NaN`. For a headcount, zero is the true answer, so `fill_value=0` is right here. It would be wrong for a mean salary: filling an empty cell with 0 would report a salary nobody earns.

### Read a Pivot Table as a Heatmap

```python
# A pivot table is already the shape sns.heatmap() wants (Lecture 07)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    salary_pivot,
    annot=True,
    fmt=",.0f",
    cmap="YlOrRd",
    ax=ax,
    cbar_kws={"label": "Mean Salary ($)"},
)
ax.set_title("Mean Salary: Department × Region", fontsize=14, fontweight="bold")
ax.set_xlabel("Region")
ax.set_ylabel("Department")
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.show()
```

The rows separate clearly by department, Engineering highest and Operations lowest, while each row looks flat across the columns: region was assigned at random, so it carries no signal. A heatmap makes that pattern obvious in a way 30 printed numbers do not.

### Back to Long Form

```python
# stack() moves the columns back into the index, undoing unstack()
long_form = salary_pivot.stack()
print("=== One Row per Department × Region ===")
print(long_form.head(8).round(0))
print("\nRows in long form:", len(long_form))
print("Round trip back to the grid?", long_form.unstack().equals(salary_pivot))
```

Thirty rows (six departments × five regions) and a `True` round trip: wide and long are two layouts of the same summary.

## Part 4: Cross-Tabulations

`pd.crosstab()` is the counting special case: hand it two columns and it reports how many rows fall in each combination.

### Count Every Combination

```python
# Counts, plus row and column totals
headcount = pd.crosstab(df["Department"], df["Region"], margins=True)
print("=== Employees per Department × Region ===")
print(headcount)
print("\nGrand total:", headcount.loc["All", "All"])
```

The `All` corner must equal 100,000. An absent combination would show `0` here, not `NaN`: crosstab counts rows, and "no rows" is a count of zero.

### Summarize a Third Column Instead of Counting

```python
# values= plus aggfunc= turns crosstab into a pivot table
mean_salary = pd.crosstab(
    df["Department"], df["Region"], values=df["Salary"], aggfunc="mean"
)
print("=== Mean Salary by Department × Region (crosstab) ===")
print(mean_salary.round(0))
print("\nSame numbers as the pivot table?", mean_salary.round(6).equals(salary_pivot.round(6)))
```

### Two-Level Rows

```python
# Bin experience into bands (Lecture 05), then use two row keys
df["Experience_Band"] = pd.cut(
    df["Experience"],
    bins=[-1, 4, 9, 14, 100],
    labels=["0-4 yrs", "5-9 yrs", "10-14 yrs", "15+ yrs"],
)

band_counts = pd.crosstab(
    [df["Department"], df["Experience_Band"]], df["Region"]
)
print("=== Department × Experience Band, by Region ===")
print(band_counts.head(12))
print("\nRows in the table:", len(band_counts))
```

The index now has two levels, so the table is much taller than the crosstab above: 19 rows, not the 6 of the single-key version. It is not 24 either, because a band with no employees anywhere in a department never becomes a row. Engineering, the best-paid department, is the only one whose employees cluster in the 5-9 year band, while HR and Operations sit almost entirely in 0-4, since the dataset ties experience to salary.

## Key Takeaways

1. **Split-apply-combine**: `groupby` splits rows, the aggregation applies to each group, and pandas combines one row per group
2. **Name the grain**: one row per employee going in, one row per department coming out, and the counts must still add up
3. **Named aggregation**: `.agg(name=("column", "function"))` gives flat, self-describing column names, and `size` and `count` differ by exactly the rows whose value is missing
4. **Two keys**: one group per observed combination, with `.unstack()` and `.stack()` switching between a MultiIndex and a grid
5. **Pivot tables**: `pd.pivot_table()` is that two-key summary in one call, with `margins=True` for totals and `sort=` for the row and column order
6. **Absent is not zero**: `fill_value=0` is right for counts and sums, wrong for means
7. **Cross-tabulations**: `pd.crosstab()` counts combinations, and `values=` with `aggfunc=` summarizes a third column instead

## Next Steps

- Rebuild each summary in this demo with the other layout, and check the two agree
- Try the same groupings on a dataset of your own, predicting the number of output rows first

