---
notion:
  title_line: "# DLC: Jupyter Workflows and Advanced Pandas Operations"
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-8119-83f4-fb2c4b73430e"
  url: "https://app.notion.com/p/3d2d9fdd1a1a811983f4fb2c4b73430e"
---

# DLC: Jupyter Workflows and Advanced Pandas Operations

*This material builds on the lecture essentials in [README.md](README.md). Revisit the lecture for Series/DataFrame basics, selection and Boolean masks, column creation, and the core CSV workflow before tackling these extensions.*


# Running Notebooks Non-Interactively

Notebooks are interactive by default, but you can also run one from the command line, as a reproducibility check or as one step in a batch workflow.

```bash
# Execute every cell in order and write a separate output notebook.
jupyter nbconvert --execute --to notebook \
    --output executed_analysis.ipynb analysis.ipynb
```

By default, a cell error makes the command fail with a nonzero exit status. In a script, `set -euo pipefail` stops at that failure instead of running the next notebook:

```bash
#!/usr/bin/env bash
set -euo pipefail

jupyter nbconvert --execute --to notebook \
    --output executed_prepare.ipynb prepare.ipynb
jupyter nbconvert --execute --to notebook \
    --output executed_analyze.ipynb analyze.ipynb
```

Keep the source notebook unchanged by writing a distinct output file. Avoid `--inplace` unless overwriting the source is deliberate and recoverable. Avoid `--allow-errors` in validation or production workflows because it can produce an output notebook containing failed cells. The working directory, selected kernel, and installed packages all affect the result.


# More Ways to Filter Rows

The lecture filters rows with a named Boolean mask and `.loc`. These alternatives build the same kind of mask with less typing, or select by position instead of label.

## Reference Card: Filter shortcuts

- `df.query("temp_c >= 37 and age < 50")`: Filter with an expression string that names columns directly; `and`/`or` replace `&`/`|`. Returns a filtered `DataFrame`.
- `df["col"].between(left, right)`: Test an inclusive range; returns a Boolean `Series`.
- `df["col"].isin(["North", "East"])`: Test membership in a list; returns a Boolean `Series`. `df.isin([...])` tests every cell and returns a Boolean `DataFrame`.
- `df.iloc[mask.to_numpy()]`: `.iloc` accepts positions only, so convert a Boolean Series to a plain array first.

## Code Snippet: Filter shortcuts

```python
visits = pd.DataFrame(
    {"age": [34, 58, 41], "temp_c": [36.8, 38.1, 37.2], "clinic": ["North", "South", "East"]},
    index=pd.Index(["P001", "P002", "P003"], name="patient_id"),
)
print(visits.query("temp_c >= 37 and age < 50"))           # P003
print(visits.loc[visits["age"].between(40, 60)])            # P002, P003
print(visits.loc[visits["clinic"].isin(["North", "East"])])  # P001, P003

over_40 = visits["age"] > 40
# visits.iloc[over_40] raises ValueError: iLocation based boolean indexing cannot use an indexable as a mask
print(visits.iloc[over_40.to_numpy()])                      # P002, P003
```


# Data Alignment and Broadcasting

The lecture's derived columns line up by index label. This section shows what happens when labels do not match, how to reindex explicitly, and how DataFrame↔Series broadcasting works, so multi-source arithmetic remains predictable.

## Reference Card: Alignment and broadcasting

- Automatic index alignment during arithmetic operations
- Series align on their index; DataFrames align on both axes when mixed
- `df.add(series, axis='columns')` / `df.sub(series, axis='index')` / `df.mul(...)` / `df.div(...)`: combine after choosing the broadcast axis
- `.reindex()` and `.align(join='inner' | 'outer')`: enforce explicit label sets before combining

When you mix a DataFrame and a Series, pandas broadcasts along matching labels and introduces `NaN` wherever labels do not overlap, so make the intended axis explicit when you call arithmetic methods.

## Code Snippet: Align, broadcast, and reindex

```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])
print(s1 + s2)                      # a: NaN, b: 6, c: 8, d: NaN
print(s1.add(s2, fill_value=0))     # a: 1, b: 6, c: 8, d: 6

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
row = pd.Series([10, 20], index=['A', 'B'])
print(df.sub(row, axis='columns'))  # Broadcast Series across DataFrame columns

left, right = df.align(df.iloc[:2], join='outer', axis=0)
print(left)
print(right)

metrics = pd.DataFrame({
    'Salary': [120000, 95000, 88000],
    'Bonus': [6000, 4750, 4400]
}, index=['Avery', 'Bianca', 'Cheng'])

averages = metrics.mean()
targets = pd.Series({'Salary': 110000, 'Bonus': 5000, 'Equity': 2000})

print(metrics.sub(averages, axis='columns'))  # Broadcast Series down rows
# Reindex explicitly before arithmetic: this makes both the labels and the
# treatment of labels absent from the Series visible (and works in pandas 3).
targets_for_metrics = targets.reindex(metrics.columns, fill_value=0)
print(metrics.add(targets_for_metrics, axis='columns'))
```

## Reindex Before Combining

Adding a Series to a column matches rows by label, not by position. A label missing from either side produces `NaN`:

```python
scores = pd.DataFrame(
    {'score': [80, 90, 70]},
    index=['student_a', 'student_b', 'student_c']
)
bonus = pd.Series({'student_c': 5, 'student_a': 2})
print(scores['score'] + bonus)
```

```text
student_a    82.0
student_b     NaN
student_c    75.0
dtype: float64
```

`student_b` has no bonus, so its total is missing, and the whole column becomes floating point. `reindex()` makes the target labels, their order, and the missing-label policy explicit:

```python
scores['bonus'] = bonus.reindex(scores.index, fill_value=0)
scores['adjusted_score'] = scores['score'] + scores['bonus']
print(scores)
```

```text
           score  bonus  adjusted_score
student_a     80      2              82
student_b     90      0              90
student_c     70      5              75
```

## Assignment Under Copy-on-Write

With Copy-on-Write, a subset behaves independently: changing it never changes the DataFrame it came from, so chained assignment such as `scores[scores['score'] < 75]['status'] = 'review'` never updates `scores`. Update the owner in one statement with `.loc[row_mask, column] = value`, or `.iloc[row_positions, column_positions] = value` for positional assignment. For a separate result, transform the subset and assign the returned object to a name.

For the version-specific details behind these examples, see the official [pandas 3.0 release notes](https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v3.0.0.html), [string-dtype migration guide](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html), and [Copy-on-Write guide](https://pandas.pydata.org/docs/user_guide/copy_on_write.html).


# Function Application and Method Chaining

The lecture adds columns with bracket assignment and vectorized arithmetic; reach for the tools below when you need custom logic or a readable chain of steps. A `lambda` is a one-line function without a name: `lambda d: d['salary'] * 0.05` takes `d` and returns the expression. Lecture 05 teaches `apply()` and `map()` in the core path: [Applying Custom Functions](../05/README.md#applying-custom-functions).

## Reference Card: Apply, map, and column helpers

- `df.apply(func)`: column-wise by default; add `axis='columns'` for row-wise logic
- `series.map(func)`: element-level transformations with optional dict/Series mapping
- `df.map(func)`: element-wise DataFrame transform (use sparingly for performance)
- `df.assign(name=lambda d: ...)`: return a new DataFrame with added columns; each `lambda` receives the DataFrame built so far, so later columns can use earlier ones, and the original is unchanged
- `df.insert(loc, column, value)`: insert a column at integer position `loc`; changes `df` itself and returns `None`
- `df.eval("new = expression")`: compute a column from an expression string that names columns directly; returns a new DataFrame
- Chain helpers: `.assign()`, `.pipe()`, `.rename()` to build fluent pipelines

## Code Snippet: Apply, map, and chain

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.apply(lambda col: col.max() - col.min()))
print(df.apply(lambda row: row.sum(), axis='columns'))
print(df.map(lambda x: f"${x:.2f}"))

summary = (
    df.assign(total=lambda d: d.sum(axis=1))
      .pipe(lambda d: d / d['total'].max())
)
print(summary)
```

## Adding Columns with `assign()`, `insert()`, and `eval()`

These three alternatives to bracket assignment differ in what they return and whether they change the original DataFrame.

```python
salaries = pd.DataFrame({
    'name': ['Avery', 'Bianca', 'Cheng'],
    'salary': [120000, 95000, 88000],
})
augmented = salaries.assign(
    bonus=lambda d: d['salary'] * 0.05,
    total_comp=lambda d: d['salary'] + d['bonus'],
)
print(augmented)                     # adds bonus and total_comp; salaries is unchanged
salaries.insert(1, 'hourly', salaries['salary'] / 2080)
print(salaries.columns.tolist())     # ['name', 'hourly', 'salary']
print(salaries.eval('monthly = salary / 12'))
```


# Ranking Strategies

Go beyond the lecture's sorting by assigning ranks, controlling tie behavior, and ranking across rows or columns.

## Reference Card: Ranking

- `series.rank()`: mean rank for ties (default)
- `method='first' | 'min' | 'max' | 'dense'`: tie handling strategies
- `ascending=False`: reverse ranking
- `df.rank(axis='columns')`: rank across columns within each row

## Code Snippet: Rank with ties

```python
s = pd.Series([7, -5, 7, 4, 2, 0, 4])
print(s.rank())                 # Mean rank for ties
print(s.rank(method='first'))   # First occurrence gets the better rank
print(s.rank(ascending=False))  # Reverse order ranking
```


# Handling Duplicate Index Labels

The lecture counts repeated rows with `df.duplicated()`. This section covers repeated index labels: what selection returns when a label appears more than once, and how to check for or collapse the repeats.

## Reference Card: Duplicate index labels

- `index.is_unique`: quick sanity check
- Label-based selection returns Series/DataFrame when duplicates exist
- `duplicated()` and `drop_duplicates()` also operate on indexes
- `groupby(level=0)` or `.reset_index()` can normalize duplicates

## Code Snippet: Select repeated labels

```python
import numpy as np
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'a', 'b', 'b', 'c'])
print(s.index.is_unique)  # False
print(s['a'])             # Series with two values
print(s['c'])             # Scalar

rng = np.random.default_rng(42)
df = pd.DataFrame(rng.standard_normal((5, 3)), index=['a', 'a', 'b', 'b', 'c'])
print(df.loc['b'])        # DataFrame with the duplicate rows
```


# Extended I/O and Performance

The lecture covers core CSV reading and writing. Use this section when you need other formats, files too large for memory, or messier CSV input.

## Excel Integration

Ideal for business spreadsheets or multi-sheet workbooks. Excel files need the `openpyxl` package in the kernel's environment; without it pandas raises `ModuleNotFoundError: No module named 'openpyxl'`.

```python
# Read entire workbook
df = pd.read_excel('data.xlsx')
print(df.head())

# Target a specific sheet
df_sales = pd.read_excel('data.xlsx', sheet_name='Sales')
print(df_sales.head())

# Write results back out
df_sales.to_excel('sales_summary.xlsx', sheet_name='Summary', index=False)
```

`sheet_name=None` reads every sheet into a dictionary of DataFrames keyed by sheet name. Use Excel output when you need Excel-native formatting or your stakeholders expect `.xlsx` files.

## JSON and Semi-Structured Data

Designed for API payloads or nested records.

```python
df = pd.read_json('data.json')
df.to_json('output.json', orient='records', indent=2)
```

Switch the `orient` parameter (`'records'`, `'columns'`, `'table'`, etc.) based on the consumer.

## SQL Databases

Ideal when data already resides in transactional stores. Requires a SQLAlchemy engine or DB-API connection.

```python
# import sqlalchemy as sqla
# engine = sqla.create_engine('sqlite:///mydb.sqlite')
# query = "SELECT name, total, date FROM sales WHERE date >= '2024-01-01'"
# df = pd.read_sql(query, engine)
```

Once records are in a DataFrame, downstream cleaning and analysis mirrors the lecture workflow.

## Reading Large Files in Chunks

Break massive files into bite-sized pieces without exhausting RAM.

```python
chunk_iter = pd.read_csv('huge_file.csv', chunksize=10000)
results = []

for chunk in chunk_iter:
    processed = chunk[chunk['value'] > 0].groupby('category').sum()
    results.append(processed)

final = pd.concat(results, axis=0).groupby(level=0).sum()
```

Use chunking when files exceed memory or when you only need aggregated results. The loop uses `groupby()` (Lecture 08) and `pd.concat()` (Lecture 06); Lecture 08's bonus covers chunked summaries in more depth, including why chunk means cannot simply be averaged: [Scaling Past Memory](../08/BONUS.md#scaling-past-memory-chunks-and-processes).

## Advanced CSV Options

Tame messy inputs with other delimiters, extra rows, source-specific missing codes, or a quick preview. Suppose `survey.csv` uses semicolons and has a units row under its header:

```text
name;role;bonus
text;text;USD
Avery;analyst;500
Bianca;nurse;missing
Cheng;analyst;-999
```

```python
survey = pd.read_csv(
    'survey.csv',
    sep=';',                           # semicolon-delimited; use '\t' for tabs
    skiprows=[1],                      # skip line 1 (the units row); the header is line 0
    usecols=['name', 'bonus'],         # keep only these columns
    na_values=['missing', '-999'],     # codes beyond the defaults (blank, NA, NULL, ...)
)
print(survey)                          # bonus: 500.0, NaN, NaN (float64)

preview = pd.read_csv('large_file.csv', nrows=100)  # first 100 records only
```

The defaults sometimes misfire: a country column with `NA` for Namibia reads as missing. `keep_default_na=False` turns every default marker off, blanks included, so only your `na_values` count as missing.
