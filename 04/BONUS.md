# Bonus Content: Advanced Pandas Operations

*This content goes beyond the daily toolkit for data scientists. These are powerful features you'll appreciate when you need them, but they're not essential for getting started.*

## Data Alignment and Arithmetic Operations

One of pandas' most powerful features is automatic data alignment during arithmetic operations. When you combine objects with different indexes, pandas aligns them automatically.

*Think of this like Excel's VLOOKUP on steroids - pandas automatically matches up your data by labels, no formulas needed.*

**Reference:**

- Automatic index alignment in operations
- `add(other, fill_value=0)` - Add with fill value for missing labels
- `sub(other, fill_value=0)` - Subtract with fill value
- `mul(other, fill_value=1)` - Multiply with fill value
- `div(other, fill_value=1)` - Divide with fill value
- Broadcasting between DataFrame and Series

**Brief Example:**

```python
# Different indexes align automatically
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])
print(s1 + s2)  # a: NaN, b: 6, c: 8, d: NaN

# Use fill_value to handle missing
print(s1.add(s2, fill_value=0))  # a: 1, b: 6, c: 8, d: 6

# DataFrame and Series broadcasting
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
row = pd.Series([10, 20], index=['A', 'B'])
print(df - row)  # Subtract row from each row of df
```

## Function Application and Mapping

Apply custom functions across entire rows or columns of DataFrames using `apply`, or to every element using `applymap`.

*Warning: While `apply` is super flexible, it's often slower than vectorized operations. If you find yourself using `apply`, first ask: "Is there a built-in pandas function for this?" The answer is usually yes.*

**Reference:**

- `df.apply(func)` - Apply function to each column
- `df.apply(func, axis='columns')` - Apply function to each row
- `df.applymap(func)` - Apply function to every element (deprecated, use `df.map(func)` instead)
- `series.map(func)` - Apply function to each element of Series
- Functions can return scalars or Series

**Brief Example:**

```python
# Apply to columns
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.apply(lambda x: x.max() - x.min()))  # Range of each column

# Apply to rows
print(df.apply(lambda x: x.max() - x.min(), axis='columns'))

# Map to format elements
print(df.map(lambda x: f"${x}.00"))
```

## Ranking Data

Ranking assigns rank values (1 through n) to data, with various methods for handling ties.

**Reference:**

- `series.rank()` - Assign ranks, ties get mean rank
- `method='first'` - Ties ranked by order in data
- `method='min'` - Ties all get minimum rank
- `method='max'` - Ties all get maximum rank
- `method='dense'` - Like min, but ranks increase by 1
- `ascending=False` - Rank in descending order
- `df.rank(axis='columns')` - Rank across columns

**Brief Example:**

```python
# Ranking with ties
s = pd.Series([7, -5, 7, 4, 2, 0, 4])
print(s.rank())  # Ties get average rank: 6.5, 1.0, 6.5, 4.5, ...

# First occurrence wins
print(s.rank(method='first'))  # 6.0, 1.0, 7.0, 4.0, 3.0, 2.0, 5.0

# Descending order
print(s.rank(ascending=False))  # 1.5, 7.0, 1.5, 3.5, ...
```

## Handling Duplicate Index Labels

While unique index labels are recommended, pandas allows duplicates. Be aware that indexing behavior changes when duplicates exist.

*Pro tip: Duplicate index labels are like having two people with the same name in your contacts - it works, but it's confusing and you'll probably regret it later.*

**Reference:**

- `index.is_unique` - Check if index has duplicates
- Indexing returns Series for duplicate labels, scalar for unique
- `duplicated()` works on indexes too
- `drop_duplicates()` can operate on index

**Brief Example:**

```python
# Duplicate indexes change behavior
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'a', 'b', 'b', 'c'])
print(s.index.is_unique)  # False

# Returns Series for duplicates
print(s['a'])  # Series with 2 values

# Returns scalar for unique
print(s['c'])  # 5

# DataFrame with duplicate row labels
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'a', 'b', 'b', 'c'])
print(df.loc['b'])  # Returns DataFrame with 2 rows
```

## When to Use These Features

**Data Alignment:** When combining datasets from different sources with overlapping but not identical indexes.

**Apply/Map:** When you have complex custom logic that can't be expressed with built-in pandas functions (rare!).

**Ranking:** For statistical analysis, creating percentiles, or competition-style ranking systems.

**Duplicate Indexes:** Generally avoid, but occasionally useful for time series with multiple observations per timestamp.
