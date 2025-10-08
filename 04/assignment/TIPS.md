# Assignment 4: Tips & Troubleshooting Guide

This guide provides detailed tips for completing the assignment and solutions to common problems.

## General Workflow Tips

### 1. Work Sequentially

Each question builds on the previous one:

- **Q1** creates and inspects the DataFrame
- **Q2** uses that DataFrame to create cleaned data
- **Q3** uses the cleaned data for analysis

**Don't skip ahead!** If Q2 fails, Q3 will likely fail too.

### 2. Test After Each Question

Run the tests after completing each question to catch errors early:

```bash
# Test just Question 1
pytest .github/test/test_assignment.py::test_q1_exploration -v

# Test just Question 2
pytest .github/test/test_assignment.py::test_q2_cleaning -v

# Test just Question 3
pytest .github/test/test_assignment.py::test_q3_analysis -v

# Test all at once
pytest .github/test/test_assignment.py -v
```

### 3. Inspect Your DataFrames Frequently

Use these commands liberally as you work:

```python
# See first few rows
df.head()

# Check structure and types
df.info()

# Check for missing values
df.isnull().sum()

# Check shape
df.shape

# See unique values in a column
df['column_name'].unique()

# Quick stats
df.describe()
```

### 4. Verify Output Files Exist

Before running tests, manually check your output files:

```python
import os

# Check all required files exist
for filename in ['exploration_summary.csv', 'cleaned_data.csv', 'analysis_results.csv']:
    filepath = f'output/{filename}'
    if os.path.exists(filepath):
        print(f'✓ {filepath} exists')
    else:
        print(f'✗ {filepath} MISSING!')
```

### 5. Clear and Restart Before Submission

Before final submission:

1. **Kernel** → **Restart & Clear Output**
2. **Cell** → **Run All**
3. Verify no errors
4. Check all three output files are created
5. Run `pytest .github/test/test_assignment.py -v`
6. Only then push to GitHub

## Question-Specific Tips

### Question 1: Data Loading & Exploration

**Loading the CSV:**

```python
df = pd.read_csv('data/customer_purchases.csv')
```

**Selecting numeric columns:**

```python
df_numeric = df.select_dtypes(include=['number'])
# This gets only 'quantity' and 'price_per_item'
```

**Generating summary statistics:**

```python
summary = df_numeric.describe()
# Returns a DataFrame with count, mean, std, min, max, etc.
```

**Saving the summary:**

```python
summary.to_csv('output/exploration_summary.csv')
# Include the index (row names like 'count', 'mean', etc.)
```

**Common mistake:** Forgetting to create the `output/` directory first:

```python
import os
os.makedirs('output', exist_ok=True)
```

### Question 2: Data Cleaning & Transformation

**Filling missing values:**

```python
# Fill missing quantity with 1
df['quantity'] = df['quantity'].fillna(1)
```

**Dropping rows with missing values:**

```python
# Drop rows where shipping_method is missing
df = df.dropna(subset=['shipping_method'])
```

**Converting data types:**

```python
# Convert purchase_date to datetime
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# Convert quantity to integer (do this AFTER filling NaN!)
df['quantity'] = df['quantity'].astype('int64')
```

**Filtering with multiple conditions:**

```python
# Method 1: Using .isin() and &
df_filtered = df[(df['customer_state'].isin(['CA', 'NY'])) & (df['quantity'] >= 2)]

# Method 2: Using OR conditions
df_filtered = df[((df['customer_state'] == 'CA') | (df['customer_state'] == 'NY')) & (df['quantity'] >= 2)]
```

**Important:** Use parentheses around each condition when using `&` or `|`!

**Saving cleaned data:**

```python
df_filtered.to_csv('output/cleaned_data.csv', index=False)
# index=False prevents writing row numbers as a column
```

### Question 3: Analysis & Aggregation

**Creating calculated columns:**

```python
df_filtered['total_price'] = df_filtered['quantity'] * df_filtered['price_per_item']
```

**Grouping and aggregating:**

```python
# Group by category and sum the total_price
revenue_by_category = df_filtered.groupby('product_category')['total_price'].sum()

# Sort by revenue (descending)
revenue_by_category = revenue_by_category.sort_values(ascending=False)
```

**Finding top N items:**

```python
# Method 1: Using .nlargest()
top_5 = df_filtered.groupby('product_name')['quantity'].sum().nlargest(5)

# Method 2: Using .sort_values() and .head()
top_5 = df_filtered.groupby('product_name')['quantity'].sum().sort_values(ascending=False).head(5)
```

**Saving analysis results:**

The tests are flexible about the exact structure, but make sure you include:

- Product categories with their total revenue
- The data should be in a DataFrame format

```python
# Convert Series to DataFrame for easier saving
analysis = pd.DataFrame({
    'product_category': revenue_by_category.index,
    'total_revenue': revenue_by_category.values
})

analysis.to_csv('output/analysis_results.csv', index=False)
```

## Common Error Messages

### FileNotFoundError: data/customer_purchases.csv

**Cause:** You haven't generated the dataset yet.

**Solution:** Run `data_generator.ipynb` first to create the dataset.

### FileNotFoundError: output/exploration_summary.csv

**Cause:** The output directory doesn't exist.

**Solution:**

```python
import os
os.makedirs('output', exist_ok=True)
```

Put this at the top of your notebook (or in the setup cell).

### AssertionError: Column X not found

**Cause:** Your output CSV has the wrong column names.

**Solution:** Check spelling and capitalization. Column names are case-sensitive!

```python
# Check what columns you actually saved
saved_df = pd.read_csv('output/cleaned_data.csv')
print(saved_df.columns)
```

### AssertionError: Expected X rows, got Y rows

**Cause:** Your filtering logic removed too many or too few rows.

**Solution:** Double-check your filtering conditions:

```python
# After filtering, check how many rows remain
print(f"Rows after filtering: {len(df_filtered)}")

# Check what states are left
print(df_filtered['customer_state'].unique())

# Check quantity range
print(f"Min quantity: {df_filtered['quantity'].min()}")
print(f"Max quantity: {df_filtered['quantity'].max()}")
```

### AssertionError: quantity should be integer type

**Cause:** You converted quantity to integer before filling NaN values, or didn't convert at all.

**Solution:** Correct order:

```python
# 1. Fill NaN FIRST
df['quantity'] = df['quantity'].fillna(1)

# 2. THEN convert to integer
df['quantity'] = df['quantity'].astype('int64')
```

### Tests pass locally but fail on GitHub

**Causes:**

1. You're using a different dataset locally than what's in `data/customer_purchases.csv`
2. Your code has hard-coded values instead of calculated results
3. Random seed issues (if you modified the data generator)

**Solution:**

1. Make sure you're using the dataset created by `data_generator.ipynb`
2. Delete `output/` directory and regenerate by running your notebook fresh
3. Use the exact code from `data_generator.ipynb` without modifications

### SettingWithCopyWarning

**Cause:** You're modifying a filtered DataFrame.

**Solution:** Use `.copy()` after filtering:

```python
df_filtered = df[(df['customer_state'].isin(['CA', 'NY'])) & (df['quantity'] >= 2)].copy()
```

Or use `.loc[]` for modifications:

```python
df.loc[df['quantity'].isnull(), 'quantity'] = 1
```

## Debugging Strategies

### 1. Print Everything

When stuck, print intermediate results:

```python
print("Shape after loading:", df.shape)
print("Missing values:", df.isnull().sum())
print("Data types:", df.dtypes)
print("Unique states:", df['customer_state'].unique())
```

### 2. Work in Small Steps

Don't chain too many operations at once:

```python
# ❌ Hard to debug
df = df.fillna(1).dropna().astype('int64')

# ✓ Easy to debug
df = df.fillna(1)
print("After fillna:", df.isnull().sum())

df = df.dropna()
print("After dropna:", df.shape)

df = df.astype('int64')
print("After astype:", df.dtypes)
```

### 3. Compare with Expected Output

Load your output file and inspect it:

```python
# Check Q1 output
summary = pd.read_csv('output/exploration_summary.csv', index_col=0)
print(summary)
print("Columns:", summary.columns.tolist())

# Check Q2 output
cleaned = pd.read_csv('output/cleaned_data.csv')
print(cleaned.info())
print(cleaned.head())

# Check Q3 output
analysis = pd.read_csv('output/analysis_results.csv')
print(analysis)
```

### 4. Use the Test File as a Guide

Read `.github/test/test_assignment.py` to see exactly what the tests expect:

```python
# The test checks for these things:
# Q1: 'count', 'mean', 'std' in index; 'quantity', 'price_per_item' in columns
# Q2: No missing quantity/shipping_method, only CA/NY states, quantity >= 2, integer type
# Q3: 'product_category' column, numeric revenue column, at least 3 categories
```

## Getting Help

### Before Asking for Help

1. ✓ Read the error message completely
2. ✓ Check this TIPS.md file
3. ✓ Print your DataFrames to see what's actually in them
4. ✓ Try running just the failing line in a new cell
5. ✓ Check the test file to see what's expected

### When Asking for Help

Provide:

1. The complete error message (copy-paste, don't screenshot)
2. The relevant code that's failing
3. What you've already tried
4. Output from `df.head()` and `df.info()` if relevant

### Resources

- **Pandas documentation:** https://pandas.pydata.org/docs/
- **Lecture 4 materials:** `04/README.md` and demo notebooks
- **Office hours:** Bring specific error messages
- **Ed Discussion:** Post code snippets (not full solutions)

## Final Checklist

Before submission:

- [ ] All cells run without errors (Restart & Run All)
- [ ] `output/` directory exists
- [ ] `output/exploration_summary.csv` created
- [ ] `output/cleaned_data.csv` created
- [ ] `output/analysis_results.csv` created
- [ ] Local tests pass: `pytest .github/test/test_assignment.py -v`
- [ ] Notebook has outputs visible (don't clear outputs before committing!)
- [ ] Code is readable with comments explaining complex parts
- [ ] Committed and pushed to GitHub
- [ ] Checked GitHub Actions for green checkmark ✓

Good luck! 🐼
