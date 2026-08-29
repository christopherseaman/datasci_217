---
jupyter:
  jupytext:
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
    version: 3.12.13
---

# Demo 2: Survey Data Reshaping with pivot() and melt()

## Learning Objectives

- Understand the difference between wide and long data formats
- Transform wide data to long format using `melt()`
- Transform long data to wide format using `pivot()`
- Use `pivot_table()` for aggregation with duplicate keys
- Recognize when each format is better for analysis vs presentation
- Build complete reshape workflows combining multiple operations


## Setup

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
```

## Create Sample Data: Employee Satisfaction Survey

We'll create a realistic employee satisfaction survey with multiple questions stored in **wide format**.

```python
# Employee satisfaction survey (wide format)
survey_wide = pd.DataFrame({
    'employee_id': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006'],
    'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales', 'HR'],
    'Q1_workload': [4, 5, 3, 4, 5, 4],         # 1-5 scale
    'Q2_management': [5, 4, 4, 5, 3, 5],       # 1-5 scale
    'Q3_compensation': [3, 4, 3, 4, 4, 5],     # 1-5 scale
    'Q4_work_life': [4, 3, 5, 4, 3, 4],        # 1-5 scale
    'Q5_growth': [5, 4, 4, 5, 4, 5]            # 1-5 scale
})

print("Wide Format - Survey Data:")
print("Each row = one employee, each column = one question")
survey_wide
```

**Wide format characteristics:**
- ✅ Easy to read (one row per employee)
- ✅ Good for spreadsheets and human viewing
- ❌ Hard to analyze (can't easily do `groupby('question')`)
- ❌ Hard to plot (plotting libraries prefer long format)


## Converting Wide → Long with melt()

The `melt()` function "melts" wide data into long format by:
1. Keeping identifier columns fixed (`id_vars`)
2. Stacking value columns into rows (`value_vars`)
3. Creating a new column for variable names (`var_name`)
4. Creating a new column for values (`value_name`)

```python
# Melt wide to long format
survey_long = pd.melt(
    survey_wide,
    id_vars=['employee_id', 'department'],        # Keep these columns
    value_vars=['Q1_workload', 'Q2_management',   # Stack these columns
                'Q3_compensation', 'Q4_work_life', 'Q5_growth'],
    var_name='question',                          # Name for the question column
    value_name='rating'                           # Name for the rating column
)

print("Long Format - Survey Data:")
print("Each row = one response to one question")
survey_long.head(10)
```

**Long format characteristics:**
- ✅ Perfect for analysis (easy `groupby('question')`)
- ✅ Required for most plotting libraries
- ✅ Flexible for filtering and aggregation
- ❌ Harder to read (many more rows)

**What happened:**
- 6 employees × 5 questions = **30 rows**
- Each employee now appears 5 times (once per question)
- Question names moved from column headers to values in `question` column


## Analyzing Long Format Data

Long format shines for groupby operations and statistical analysis.

```python
# Calculate average rating per question (easy in long format!)
question_stats = survey_long.groupby('question')['rating'].agg([
    ('avg_rating', 'mean'),
    ('std_dev', 'std'),
    ('min_rating', 'min'),
    ('max_rating', 'max')
]).round(2)

print("Question Statistics:")
question_stats.sort_values('avg_rating', ascending=False)
```

**Insights from long format analysis:**
- Q2 (management) and Q5 (growth) score highest (avg 4.5)
- Q3 (compensation) scores lowest (avg 3.83)
- All questions show variation (std_dev > 0.5)

```python
# Calculate average rating by department (also easy!)
dept_stats = survey_long.groupby('department')['rating'].agg([
    ('avg_rating', 'mean'),
    ('response_count', 'count')
]).round(2)

print("Department Statistics:")
dept_stats.sort_values('avg_rating', ascending=False)
```

**Try doing this in wide format** - it's much harder! You'd need to manually select columns, stack them, etc.


## Converting Long → Wide with pivot()

The `pivot()` function converts long format back to wide format:
- **index**: Column to use for row labels
- **columns**: Column to use for column labels
- **values**: Column containing the data values

```python
# Pivot long back to wide
survey_wide_restored = survey_long.pivot(
    index=['employee_id', 'department'],  # Row labels
    columns='question',                   # Column labels
    values='rating'                       # Values to fill cells
)

print("Wide Format (Restored):")
survey_wide_restored
```

**Notice:**
- Back to original wide structure (6 rows × 5 question columns)
- Index is now hierarchical (employee_id + department)
- To make it look exactly like original, we need to reset the index

```python
# Reset index to match original format
survey_wide_clean = survey_wide_restored.reset_index()
survey_wide_clean.columns.name = None  # Remove 'question' label from columns

print("Wide Format (Clean):")
survey_wide_clean
```

**Perfect!** We've successfully transformed: wide → long → wide

**When to use each:**
- **Wide format:** Reports, presentations, spreadsheets
- **Long format:** Analysis, groupby operations, plotting


## Using pivot_table() for Aggregation

What if we have duplicate index/column combinations? Use `pivot_table()` to aggregate!

```python
# Create data with duplicates (multiple responses from same dept/question)
# This simulates multiple employees from same department
survey_duplicates = pd.DataFrame({
    'department': ['Engineering', 'Engineering', 'Sales', 'Sales',
                   'Engineering', 'Sales', 'Marketing', 'Marketing'],
    'question': ['Q1_workload', 'Q1_workload', 'Q1_workload', 'Q1_workload',
                 'Q2_management', 'Q2_management', 'Q1_workload', 'Q2_management'],
    'rating': [4, 3, 5, 5, 5, 4, 4, 5]
})

print("Data with Duplicate Dept/Question Combinations:")
survey_duplicates
```

**Problem:** Engineering has 2 responses for Q1_workload (4 and 3). Simple `pivot()` will fail!

```python
# This would fail with pivot():
# survey_duplicates.pivot(index='department', columns='question', values='rating')
# Error: Index contains duplicate entries, cannot reshape

# Use pivot_table() instead to aggregate
dept_question_avg = survey_duplicates.pivot_table(
    index='department',
    columns='question',
    values='rating',
    aggfunc='mean'  # Calculate mean for duplicates
)

print("Department Average Ratings (with aggregation):")
dept_question_avg
```

**What happened:**
- Engineering's Q1_workload: (4 + 3) / 2 = **3.5**
- Sales' Q1_workload: (5 + 5) / 2 = **5.0**
- `pivot_table()` automatically aggregated duplicates

**Other aggfunc options:** `'sum'`, `'count'`, `'median'`, `'std'`, `'min'`, `'max'`

```python
# Count how many responses per department/question
response_counts = survey_duplicates.pivot_table(
    index='department',
    columns='question',
    values='rating',
    aggfunc='count',
    fill_value=0  # Show 0 for missing combinations
)

print("Response Counts:")
response_counts
```

**Insight:** Not all departments answered all questions - some cells are 0.


## Real-World Workflow: Survey Analysis Report

Combining reshape operations to create a comprehensive survey report.

```python
# Step 1: Start with wide format data
print("Step 1: Original wide format survey")
print(f"Shape: {survey_wide.shape}")
display(survey_wide.head(3))
```

```python
# Step 2: Melt to long format for analysis
survey_analysis = pd.melt(
    survey_wide,
    id_vars=['employee_id', 'department'],
    value_vars=['Q1_workload', 'Q2_management', 'Q3_compensation',
                'Q4_work_life', 'Q5_growth'],
    var_name='question',
    value_name='rating'
)

print("\nStep 2: Melted to long format for analysis")
print(f"Shape: {survey_analysis.shape}")
display(survey_analysis.head(3))
```

```python
# Step 3: Calculate summary statistics by department and question
dept_question_summary = survey_analysis.pivot_table(
    index='department',
    columns='question',
    values='rating',
    aggfunc='mean'
).round(2)

print("\nStep 3: Department × Question Summary (pivot_table)")
display(dept_question_summary)
```

```python
# Step 4: Add department average column
dept_question_summary['Dept_Avg'] = dept_question_summary.mean(axis=1).round(2)

# Add overall question averages as a row
dept_question_summary.loc['Overall_Avg'] = dept_question_summary.mean(axis=0).round(2)

print("\nStep 4: Final Report with Averages")
dept_question_summary
```

**Report Insights:**
- **HR** has highest overall satisfaction (4.60)
- **Sales** has lowest overall satisfaction (3.80)
- **Q3_compensation** is weakest area company-wide (3.83 avg)
- **Q2_management** and **Q5_growth** are strongest (4.50 avg)

**Workflow summary:**
1. Wide format (input data)
2. Melt to long (for analysis)
3. Pivot_table to wide (for reporting)
4. Add summary stats (for insights)


## Cleaning Up Question Labels

Often you want to replace codes (Q1_workload) with readable names.

```python
# Create readable question labels
question_labels = {
    'Q1_workload': 'Workload Balance',
    'Q2_management': 'Management Support',
    'Q3_compensation': 'Compensation',
    'Q4_work_life': 'Work-Life Balance',
    'Q5_growth': 'Career Growth'
}

# Apply to our long format data
survey_readable = survey_long.copy()
survey_readable['question'] = survey_readable['question'].map(question_labels)

print("Survey with Readable Labels:")
survey_readable.head()
```

```python
# Create final presentation table with readable names
final_report = survey_readable.pivot_table(
    index='department',
    columns='question',
    values='rating',
    aggfunc='mean'
).round(2)

# Add department averages
final_report['Department Average'] = final_report.mean(axis=1).round(2)

print("\nFinal Survey Report (Presentation-Ready):")
final_report.sort_values('Department Average', ascending=False)
```

**Now the report is ready for executives!** Clear labels, sorted by performance.


## When to Use Each Format

**Use WIDE format when:**
- Presenting to humans (reports, dashboards)
- Each row represents one logical entity
- Comparing values across columns visually
- Working in spreadsheet software

**Use LONG format when:**
- Performing statistical analysis
- Using groupby operations
- Creating visualizations (seaborn, plotly prefer long)
- Filtering by variable type
- Database storage (more flexible schema)

**Common workflow:**
1. **Store** in long format (database)
2. **Analyze** in long format (groupby, stats)
3. **Present** in wide format (pivot tables, reports)


## Key Takeaways

1. **melt() transforms wide → long:**
   - Use `id_vars` for columns to keep
   - Use `value_vars` for columns to stack
   - Customize names with `var_name` and `value_name`

2. **pivot() transforms long → wide:**
   - Use `index` for row labels
   - Use `columns` for column labels
   - Use `values` for data to fill cells
   - **Requires unique index/column combinations**

3. **pivot_table() is pivot() with aggregation:**
   - Handles duplicate index/column pairs
   - Specify `aggfunc` (mean, sum, count, etc.)
   - Use `fill_value` for missing combinations

4. **Complete workflow pattern:**
   - Wide (input) → Long (analysis) → Wide (presentation)
   - Clean labels for readability
   - Add summary statistics

5. **Common mistakes to avoid:**
   - Using pivot() with duplicates (use pivot_table instead)
   - Forgetting to reset_index() after pivot
   - Not cleaning column names before pivoting

**Practice tip:** When stuck, ask: "What do I want as rows? What as columns? What as values?" This guides your reshape operation!
