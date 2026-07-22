Bonus Content: Advanced Pandas Operations

*This material builds on the lecture essentials in [README.md](README.md). Revisit the lecture for Series/DataFrame basics, column creation, groupby introductions, and the core CSV workflow before tackling these extensions.*

---

## Data Alignment and Broadcasting

See the lecture for baseline Series/DataFrame comparisons. This section deepens alignment control for mismatched labels, explicit reindexing, and DataFrame↔Series broadcasting so multi-source arithmetic remains predictable.

**Reference:**

- Automatic index alignment during arithmetic operations
- Series align on their index; DataFrames align on both axes when mixed
- `df.add(series, axis='columns', fill_value=0)` / `df.sub(series, axis='index', fill_value=0)` / `df.mul(...)` / `df.div(...)` — combine while filling gaps
- `.reindex()` and `.align(join='inner' | 'outer')` — enforce explicit label sets before combining

Series align on their index, while DataFrames track both axes. When you mix them, pandas broadcasts along matching labels and introduces `NaN` wherever labels do not overlap, so make the intended axis explicit when you call arithmetic methods.

**Brief Example:**

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
print(metrics.add(targets, fill_value=0))     # Fill missing labels before combining
```

---

## Function Application and Method Chaining

The lecture covers vectorized operations; reach for the tools below when you need custom logic or pipeline readability. Combine `apply`/`map` with chaining helpers to keep transformations compact and transparent.

**Reference:**

- `df.apply(func)` — column-wise by default; add `axis='columns'` for row-wise logic
- `series.map(func)` — element-level transformations with optional dict/Series mapping
- `df.applymap(func)` — element-wise DataFrame transform (use sparingly for performance)
- Chain helpers: `.assign()`, `.pipe()`, `.rename()` to build fluent pipelines

**Brief Example:**

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

---

## Ranking Strategies

Go beyond simple sorting by assigning ranks, controlling tie behavior, and ranking across rows or columns. Pair these techniques with the lecture's descriptions of sorting and unique values when you need ordered analytics.

**Reference:**

- `series.rank()` — mean rank for ties (default)
- `method='first' | 'min' | 'max' | 'dense'` — tie handling strategies
- `ascending=False` — reverse ranking
- `df.rank(axis='columns')` — rank across columns within each row

**Brief Example:**

```python
s = pd.Series([7, -5, 7, 4, 2, 0, 4])
print(s.rank())                 # Mean rank for ties
print(s.rank(method='first'))   # First occurrence gets the better rank
print(s.rank(ascending=False))  # Reverse order ranking
```

---

## Handling Duplicate Index Labels

The lecture covers duplicate detection at the column level. This section focuses on index semantics when labels repeat, plus tactics for normalizing or exploiting duplicates in time-series and log pipelines.

**Reference:**

- `index.is_unique` — quick sanity check
- Label-based selection returns Series/DataFrame when duplicates exist
- `duplicated()` and `drop_duplicates()` also operate on indexes
- `groupby(level=0)` or `.reset_index()` can normalize duplicates

**Brief Example:**

```python
import numpy as np
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'a', 'b', 'b', 'c'])
print(s.index.is_unique)  # False
print(s['a'])             # Series with two values
print(s['c'])             # Scalar

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'a', 'b', 'b', 'c'])
print(df.loc['b'])        # DataFrame with the duplicate rows
```

---

## Extended I/O and Performance

Revisit the lecture for core CSV ingestion/export. Use this section when you need alternate formats, iterative processing, or performance tuning. Each technique notes the scenarios where it adds value. See the lecture for baseline syntax before layering these extensions.

### Excel Integration

Ideal for business spreadsheets or multi-sheet workbooks.

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

Use when you need Excel-native formatting or your stakeholders expect `.xlsx` outputs.

### JSON and Semi-Structured Data

Designed for API payloads or nested records.

```python
df = pd.read_json('data.json')
df.to_json('output.json', orient='records', indent=2)
```

Switch the `orient` parameter (`'records'`, `'columns'`, `'table'`, etc.) based on the consumer. JSON is great for web services and lightweight integrations.

### SQL Databases

Ideal when data already resides in transactional stores. Requires a SQLAlchemy engine or DB-API connection.

```python
# import sqlalchemy as sqla
# engine = sqla.create_engine('sqlite:///mydb.sqlite')
# query = "SELECT name, total, date FROM sales WHERE date >= '2024-01-01'"
# df = pd.read_sql(query, engine)
```

Once records are in a DataFrame, downstream cleaning and analysis mirrors the lecture workflow.

### Reading Large Files in Chunks

Break massive files into bite-sized pieces without exhausting RAM.

```python
chunk_iter = pd.read_csv('huge_file.csv', chunksize=10000)
results = []

for chunk in chunk_iter:
    processed = chunk[chunk['value'] > 0].groupby('category').sum()
    results.append(processed)

final = pd.concat(results, axis=0).groupby(level=0).sum()
```

Use chunking when files exceed memory or when you only need aggregated results. See lecture fundamentals for basic `read_csv`; layer this pattern when datasets push RAM limits.

### Advanced CSV Options

Tame messy inputs with custom NA markers, delimiters, or sampling.

```python
df = pd.read_csv(
    'data.csv',
    na_values=['NA', 'NULL', 'missing', '?'],
    keep_default_na=True
)

preview = pd.read_csv('large_file.csv', nrows=100)
print(preview.head())
```

Start with the lecture’s clean CSV example, then layer these options as you encounter real-world quirks.

---


