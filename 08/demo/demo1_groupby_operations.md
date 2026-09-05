# Demo 1: GroupBy Operations

## Learning Objectives
- Master the split-apply-combine paradigm
- Apply aggregation functions to grouped data
- Use transform, filter, and apply operations
- Handle hierarchical grouping

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

## Part 1: Basic GroupBy Operations

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
# Use random dates within a reasonable range to avoid datetime overflow
start_date = pd.Timestamp("2010-01-01")
end_date = pd.Timestamp("2024-01-01")
date_range_days = (end_date - start_date).days
random_days = np.random.randint(0, date_range_days, n_employees)
join_dates = [start_date + pd.Timedelta(days=int(d)) for d in random_days]

df = pd.DataFrame({
    "Employee": employee_names,
    "Department": departments_list,
    "Region": regions_list,
    "Salary": np.round(salaries, 2),
    "Experience": experience,
    "Join_Date": join_dates,
})

# Add some additional features
df["Years_At_Company"] = (pd.Timestamp.now() - df["Join_Date"]).dt.days / 365.25
df["Performance_Score"] = np.random.uniform(1, 5, n_employees)
df["Bonus"] = df["Salary"] * df["Performance_Score"] * 0.1

print(f"Dataset shape: {df.shape}")
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

### GroupBy with Multiple Columns

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

# 1. Heatmap of mean salary by Department × Region
salary_pivot = df.groupby(["Department", "Region"])["Salary"].mean().unstack()
im = axes[0].imshow(salary_pivot.values, cmap="YlOrRd", aspect="auto")
axes[0].set_xticks(range(len(salary_pivot.columns)))
axes[0].set_yticks(range(len(salary_pivot.index)))
axes[0].set_xticklabels(salary_pivot.columns)
axes[0].set_yticklabels(salary_pivot.index)
axes[0].set_title(
    "Mean Salary Heatmap: Department × Region", fontsize=14, fontweight="bold"
)
axes[0].set_xlabel("Region")
axes[0].set_ylabel("Department")
plt.colorbar(im, ax=axes[0], label="Mean Salary ($)")

# Add text annotations
for i in range(len(salary_pivot.index)):
    for j in range(len(salary_pivot.columns)):
        text = axes[0].text(
            j,
            i,
            f"${salary_pivot.iloc[i, j]:,.0f}",
            ha="center",
            va="center",
            color="black",
            fontsize=9,
        )

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

## Part 2: Advanced GroupBy Operations

### Transform Operations

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

### Filter Operations

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

### Apply Operations

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

## Part 3: Hierarchical Grouping

### Multi-level Grouping

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

### MultiIndex Operations

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

## Part 4: Real-world Example

### Sales Analysis

```python
# Create large-scale time-series sales data (100,000 transactions)
print("=== Creating Large-Scale Sales Dataset ===")
np.random.seed(42)
n_sales = 100000

# Generate realistic sales data with trends and seasonality
start_date = pd.Timestamp("2020-01-01")
dates = pd.date_range(start_date, periods=n_sales, freq="h")[
    :n_sales
]  # Hourly data

products = ["Electronics", "Clothing", "Books", "Home", "Sports", "Toys"]
regions = ["North", "South", "East", "West", "Central"]
salespeople = [f"Sales_{i:02d}" for i in range(1, 21)]  # 20 salespeople

# Create seasonal patterns
day_of_year = dates.dayofyear
month = dates.month
day_of_week = dates.dayofweek

# Base prices with product-specific ranges
product_prices = {
    "Electronics": (100, 2000),
    "Clothing": (20, 300),
    "Books": (10, 50),
    "Home": (50, 800),
    "Sports": (30, 500),
    "Toys": (15, 200),
}

# Generate sales data with correlations
sales_data = {
    "Date": dates,
    "Product": np.random.choice(products, n_sales),
    "Region": np.random.choice(regions, n_sales),
    "Salesperson": np.random.choice(salespeople, n_sales),
}

sales_df = pd.DataFrame(sales_data)

# Generate correlated prices and quantities
unit_prices = []
quantities = []
for product in sales_df["Product"]:
    price_min, price_max = product_prices[product]
    price = np.random.uniform(price_min, price_max)
    unit_prices.append(price)
    # Higher prices tend to have lower quantities
    qty = max(
        1,
        int(
            np.random.exponential(3)
            * (1 - (price - price_min) / (price_max - price_min))
        ),
    )
    quantities.append(min(qty, 20))

sales_df["Unit_Price"] = unit_prices
sales_df["Quantity"] = quantities
sales_df["Total_Sales"] = sales_df["Quantity"] * sales_df["Unit_Price"]

# Add time-based features
sales_df["Year"] = sales_df["Date"].dt.year
sales_df["Month"] = sales_df["Date"].dt.month
sales_df["Quarter"] = sales_df["Date"].dt.quarter
sales_df["DayOfWeek"] = sales_df["Date"].dt.day_name()
sales_df["Hour"] = sales_df["Date"].dt.hour

# Add seasonal multiplier (higher sales in Q4, lower in Q1)
seasonal_mult = sales_df["Quarter"].map({1: 0.8, 2: 1.0, 3: 1.1, 4: 1.3})
sales_df["Total_Sales"] = sales_df["Total_Sales"] * seasonal_mult

print(f"Sales dataset shape: {sales_df.shape}")
print(f"Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
print(
    f"Memory usage: {sales_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
)
print("\nSample sales data:")
print(sales_df.head())

# Comprehensive time-series analysis
print("\n=== Time-Series GroupBy Analysis ===")

# Monthly sales by region
monthly_sales = sales_df.groupby([
    sales_df["Date"].dt.to_period("M"),
    "Region",
])["Total_Sales"].agg(["sum", "mean", "count"])
print("\nMonthly sales summary by region:")
print(monthly_sales.head(15))

# Top salesperson by region (using apply with include_groups=False)
top_salesperson = sales_df.groupby("Region").apply(
    lambda x: x.groupby("Salesperson")["Total_Sales"].sum().idxmax(),
    include_groups=False,
)
print("\nTop salesperson by region:")
print(top_salesperson)

# Quarterly product performance
quarterly_product = sales_df.groupby(["Quarter", "Product"])["Total_Sales"].agg([
    "sum",
    "mean",
])
print("\nQuarterly product performance:")
print(quarterly_product)

# Visualize time-series patterns
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

# 1. Monthly sales trend by region
monthly_trend = (
    sales_df.groupby([sales_df["Date"].dt.to_period("M"), "Region"])[
        "Total_Sales"
    ]
    .sum()
    .unstack()
)
monthly_trend.plot(ax=axes[0, 0], marker="o", linewidth=2, markersize=4)
axes[0, 0].set_title(
    "Monthly Sales Trend by Region", fontsize=14, fontweight="bold"
)
axes[0, 0].set_ylabel("Total Sales ($)")
axes[0, 0].set_xlabel("Month")
axes[0, 0].legend(title="Region")
axes[0, 0].grid(alpha=0.3)
axes[0, 0].tick_params(axis="x", rotation=45)

# 2. Sales by day of week
dow_sales = sales_df.groupby("DayOfWeek")["Total_Sales"].sum()
dow_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
dow_sales = dow_sales.reindex(dow_order)
dow_sales.plot(kind="bar", ax=axes[0, 1], color="steelblue")
axes[0, 1].set_title(
    "Total Sales by Day of Week", fontsize=14, fontweight="bold"
)
axes[0, 1].set_ylabel("Total Sales ($)")
axes[0, 1].set_xlabel("Day of Week")
axes[0, 1].tick_params(axis="x", rotation=45)
axes[0, 1].grid(axis="y", alpha=0.3)

# 3. Quarterly product performance heatmap
quarterly_pivot = (
    sales_df.groupby(["Quarter", "Product"])["Total_Sales"].sum().unstack()
)
im = axes[1, 0].imshow(quarterly_pivot.values, cmap="YlGnBu", aspect="auto")
axes[1, 0].set_xticks(range(len(quarterly_pivot.columns)))
axes[1, 0].set_yticks(range(len(quarterly_pivot.index)))
axes[1, 0].set_xticklabels(quarterly_pivot.columns, rotation=45)
axes[1, 0].set_yticklabels(quarterly_pivot.index)
axes[1, 0].set_title(
    "Quarterly Sales Heatmap by Product", fontsize=14, fontweight="bold"
)
axes[1, 0].set_xlabel("Product")
axes[1, 0].set_ylabel("Quarter")
plt.colorbar(im, ax=axes[1, 0], label="Total Sales ($)")

# 4. Top 10 salespeople performance
top_salespeople = (
    sales_df.groupby("Salesperson")["Total_Sales"].sum().nlargest(10)
)
top_salespeople.plot(kind="barh", ax=axes[1, 1], color="coral")
axes[1, 1].set_title(
    "Top 10 Salespeople by Total Sales", fontsize=14, fontweight="bold"
)
axes[1, 1].set_xlabel("Total Sales ($)")
axes[1, 1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Takeaways

1. **Split-Apply-Combine**: The fundamental pattern of data aggregation
2. **Aggregation Functions**: Use mean, sum, count, and custom functions
3. **Transform Operations**: Add group statistics to original data
4. **Filter Operations**: Remove groups based on conditions
5. **Apply Operations**: Use custom functions on groups
6. **Hierarchical Grouping**: Work with multi-level group structures

## Next Steps

- Practice with your own datasets
- Experiment with different aggregation functions
- Learn about pivot tables for multi-dimensional analysis
