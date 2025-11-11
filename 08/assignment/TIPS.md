# Assignment 8 Tips and Hints

This document provides helpful hints and guidance for completing Assignment 8 without giving away solutions. Use these tips if you're stuck, but remember to think through the problems yourself first!

## General Approach

1. **Read the lecture notes** - Review Lecture 08 content on groupby operations, transform, filter, apply, and pivot tables
2. **Start with the data** - Make sure you understand the structure of your merged dataset before attempting operations
3. **Test incrementally** - Run each cell after making changes to see intermediate results
4. **Check the output format** - Look at the expected output file names and think about what DataFrame structure would produce those files

## Question 1: Basic GroupBy Operations

### Part 1.2: Basic Aggregation

**Hints:**
- Remember that `groupby()` returns a GroupBy object - you need to apply an aggregation function to get results
- Multiple aggregations can be done with `.agg()` - you can pass a dictionary or list of functions
- To save a DataFrame to CSV, use `.to_csv()` with the filepath
- Think about what columns you want to aggregate and what the output should look like

**Common mistakes:**
- Forgetting to merge the dataframes first
- Not specifying which columns to aggregate
- Saving without checking the DataFrame structure first

### Part 1.3: Transform Operations

**Hints:**
- `transform()` returns a Series/DataFrame with the same index as the original data
- Transform is different from aggregation - it adds new columns to your existing DataFrame
- You can use string names for common functions ('mean', 'std') or lambda functions for custom operations
- Z-score normalization is: (value - mean) / standard_deviation

**Common mistakes:**
- Expecting transform to return a grouped result (it returns original-length data)
- Not assigning the transform result to a new column
- Using the wrong formula for z-score normalization

## Question 2: Advanced GroupBy Operations

### Part 2.1: Filter Operations

**Hints:**
- `filter()` removes entire groups based on a condition
- The lambda function receives a DataFrame for each group
- You can check group size with `len(x)` or `x.shape[0]`
- Multiple filter conditions can be combined with `&` (and) or `|` (or)
- Think about what "filtered results" means - you're keeping only groups that meet certain criteria

**Common mistakes:**
- Using `filter()` when you meant `query()` or boolean indexing
- Not understanding that filter removes entire groups, not individual rows
- Forgetting to check the condition on the grouped data, not the original DataFrame

### Part 2.2: Apply Operations

**Hints:**
- `apply()` is very flexible - your function receives a DataFrame for each group
- Your function should return a Series or DataFrame
- For custom statistics, return a `pd.Series()` with named values
- For top N items, consider using `.nlargest()` or `.nsmallest()` on the group
- The result of `apply()` will be a DataFrame with the group keys as the index

**Common mistakes:**
- Not returning the right type from your custom function
- Forgetting to handle the index when using `apply()` with `nlargest()`
- Not understanding that `apply()` can return different structures than `agg()`

### Part 2.3: Hierarchical Grouping

**Hints:**
- Multiple columns in `groupby()` create a MultiIndex
- `.unstack()` moves one level of the index to columns (wide format)
- `.stack()` moves columns back to the index (long format)
- You can specify which level to unstack with the `level` parameter
- Think about the difference between "long" and "wide" data formats

**Common mistakes:**
- Not understanding MultiIndex structure
- Forgetting to specify which level to unstack
- Confusing stack/unstack with pivot operations

## Question 3: Pivot Tables and Cross-Tabulations

### Part 3.1: Basic Pivot Tables

**Hints:**
- `pd.pivot_table()` requires `values`, `index`, and `columns` parameters
- `aggfunc` defaults to 'mean' - specify 'sum', 'count', or a list for multiple aggregations
- `fill_value` replaces NaN values with a specified value
- `margins=True` adds row and column totals
- The result is a DataFrame with a potentially MultiIndex structure

**Common mistakes:**
- Not specifying all required parameters
- Forgetting that pivot_table creates a MultiIndex by default with multiple aggfuncs
- Not handling missing values appropriately

### Part 3.2: Cross-Tabulations

**Hints:**
- `pd.crosstab()` is simpler than `pivot_table()` for frequency counts
- It takes two Series or arrays as input
- `margins=True` adds totals
- For multi-dimensional crosstabs, you can pass multiple index/column arrays
- The result shows counts/frequencies by default

**Common mistakes:**
- Using crosstab when you need aggregation (use pivot_table instead)
- Not understanding the difference between crosstab and pivot_table
- Forgetting that crosstab counts occurrences, not sums values

### Part 3.3: Pivot Table Visualization

**Hints:**
- `seaborn.heatmap()` is great for visualizing pivot tables
- Make sure your pivot table is numeric for heatmaps
- Use `plt.figure(figsize=(width, height))` to control plot size
- `plt.title()` and axis labels improve readability
- `plt.savefig()` saves the plot - specify the path and format

**Common mistakes:**
- Not converting the pivot table to numeric format first
- Forgetting to save the figure before the cell completes
- Not setting appropriate figure size for readability

## Debugging Tips

1. **Print intermediate results** - Use `print()` to see what your operations produce
2. **Check DataFrame shape** - Use `.shape` to verify your data has the expected dimensions
3. **Inspect column names** - Use `.columns` to see what columns are available
4. **Look at the head** - Use `.head()` to see sample rows
5. **Check data types** - Use `.dtypes` to ensure columns are the right type
6. **Verify merges** - After merging, check that you have the expected number of rows

## Common Error Messages

- **KeyError**: You're trying to access a column that doesn't exist - check spelling and case
- **AttributeError**: You're calling a method on the wrong type - check if you have a DataFrame or GroupBy object
- **ValueError**: Often means wrong shape or incompatible types - check your data structure
- **FileNotFoundError**: Make sure you're in the right directory and the data files exist

## When to Use Each Operation

- **`.agg()`**: When you want to summarize groups (mean, sum, count, etc.)
- **`.transform()`**: When you want to add group statistics back to original rows
- **`.filter()`**: When you want to remove entire groups based on conditions
- **`.apply()`**: When you need custom logic or complex operations on groups
- **`pivot_table()`**: When you want to reshape and aggregate across multiple dimensions
- **`crosstab()`**: When you want to count frequencies of categorical combinations

## Final Checklist

Before submitting, verify:
- [ ] All output files exist in the `output/` directory
- [ ] CSV files can be read and have data
- [ ] Text files have meaningful content
- [ ] Visualization file is not empty
- [ ] Your code runs without errors
- [ ] Results make sense (e.g., filtered data has fewer rows, aggregations are numeric)

## Getting Help

If you're still stuck after trying these hints:
1. Review the lecture notes and examples
2. Check pandas documentation for the specific function
3. Try a simpler version of the problem first
4. Ask for help from your instructor or TA

Remember: The goal is to learn, not just to complete the assignment. Understanding why each operation works is more important than getting the exact code!

