# Advanced Data Aggregation Topics

## Advanced GroupBy Operations

### Custom Aggregation Functions

**Reference:**

```python
import pandas as pd
import numpy as np

# Named/list aggregations keep each aggregation scalar, producing stable columns.
summary = df.groupby('category').agg(
    mean=('value', 'mean'),
    std=('value', 'std'),
    min=('value', 'min'),
    max=('value', 'max'),
    range=('value', lambda s: s.max() - s.min()),
    iqr=('value', lambda s: s.quantile(0.75) - s.quantile(0.25)),
)
```

### Lambda Functions in GroupBy

**Reference:**

```python
# Lambda functions for complex operations
df.groupby('category').agg({
    'value': lambda x: x.quantile(0.95),  # 95th percentile
    'other': lambda x: x.nunique(),       # Count unique values
    'score': lambda x: (x > x.mean()).sum()  # Count above mean
})

# Multiple lambda functions
df.groupby('category').agg({
    'value': [
        lambda x: x.mean(),
        lambda x: x.std(),
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ]
})
```

### GroupBy with Time Windows

**Reference:**

```python
# Time-based grouping
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Group by time periods
df.groupby(pd.Grouper(freq='ME'))[['value']].sum()  # Month-end groups
df.groupby(pd.Grouper(freq='QE'))[['value']].mean()  # Quarter-end groups
df.groupby(pd.Grouper(freq='YE'))[['value']].max()   # Year-end groups

# Custom time windows
df.groupby(pd.Grouper(freq='7D')).agg({
    'value': ['sum', 'mean', 'count']
})
```

## Advanced Pivot Table Operations

### Multi-Level Pivot Tables

**Reference:**

```python
# Multi-level pivot tables
pivot = pd.pivot_table(df,
                      values=['sales', 'profit'],
                      index=['region', 'product'],
                      columns=['quarter', 'year'],
                      aggfunc={'sales': 'sum', 'profit': 'mean'},
                      fill_value=0,
                      margins=True)

# Flatten multi-level columns
pivot.columns = ['_'.join(col).strip() for col in pivot.columns]
```

### Pivot Table with Custom Functions

**Reference:**

```python
# A pivot_table aggregator receives only the selected ``values`` series, not
# the full rows. Compute weighted components first, validate the denominators,
# then reshape their sums.
if (df['weight'] < 0).any():
    raise ValueError('weights must be nonnegative')

df = df.assign(weighted_value=df['value'] * df['weight'])
weighted = (df.groupby(['category', 'region'], observed=True)
              [['weighted_value', 'weight']].sum())
weight_totals = weighted['weight'].unstack('region')
if weight_totals.eq(0).any().any():
    raise ValueError('each observed group must have positive total weight')

pivot = weighted['weighted_value'].unstack('region').div(weight_totals)
```

### Pivot Table with Missing Data Handling

**Reference:**

In pandas 3, categorical groupers default to `observed=True`. Use `observed=False` only when a table must include every defined category or category combination.

```python
# Advanced missing data handling
pivot = pd.pivot_table(df,
                      values='value',
                      index='category',
                      columns='region',
                      aggfunc='mean',
                      fill_value=0,           # Fill missing with 0
                      dropna=False,           # Retain all-NaN result columns
                      observed=True)          # Show only observed categorical groups

# Handle missing data in different ways
pivot_filled = pivot.ffill()                 # Forward fill
pivot_interpolated = pivot.interpolate()     # Linear interpolation
pivot_dropped = pivot.dropna()               # Drop missing rows
```

## Hierarchical Grouping and MultiIndex

### MultiIndex Operations

**Reference:**

```python
# Create MultiIndex
df_multi = df.set_index(['level1', 'level2'])

# Operations on MultiIndex
df_multi.groupby(level=0, observed=True)[['value']].sum()  # First level
df_multi.groupby(level=1, observed=True)[['value']].mean()  # Second level
df_multi.groupby(level=[0, 1], observed=True)[['value']].max()  # Both levels

# Swap levels
df_multi.swaplevel(0, 1)

# Sort by index
df_multi.sort_index()

# Access specific levels
df_multi.loc[('A', 'X')]  # Access specific combination
df_multi.xs('A', level=0)  # Cross-section
```

### Advanced MultiIndex Grouping

**Reference:**

```python
# Complex MultiIndex operations
def hierarchical_analysis(df):
    """Perform hierarchical analysis"""
    
    # Group by multiple levels
    grouped = df.groupby(['level1', 'level2', 'level3'])
    
    # Apply different functions to different columns
    result = grouped.agg({
        'numeric_col': ['mean', 'std', 'count'],
        'categorical_col': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'date_col': ['min', 'max']
    })
    
    # Flatten column names
    result.columns = ['_'.join(col).strip() for col in result.columns]
    
    return result
```

## Performance Optimization

### Dtype-Aware GroupBy

**Reference:**

```python
# GroupBy with caller-validated dtype choices
def configured_groupby(df, group_cols, agg_cols, dtype_map=None):
    """Group after applying only caller-supplied, validated dtype conversions."""

    working = df.astype(dtype_map) if dtype_map else df
    
    result = working.groupby(group_cols, observed=True)[agg_cols].sum()
    
    return result
```

Supply `dtype_map` only after validating numeric range and precision requirements and, for categorical conversions, the intended category semantics. Omitting it preserves the input dtypes; the function never auto-downcasts.

### Chunked Processing

**Reference:**

```python
# Process large datasets in chunks
def chunked_groupby(file_path, group_cols, agg_cols, chunk_size=10000):
    """Process large file in chunks"""
    
    results = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if chunk.empty:
            continue

        # Process chunk
        chunk_result = chunk.groupby(group_cols, observed=True)[agg_cols].sum()
        results.append(chunk_result)

    if not results:
        raise ValueError("input file must contain at least one data row")

    # Combine results
    all_index_levels = list(range(results[0].index.nlevels))
    final_result = (pd.concat(results)
                    .groupby(level=all_index_levels, observed=True)[agg_cols]
                    .sum())
    
    return final_result
```

### Parallel Processing

**Reference:**

```python
from multiprocessing import Pool
import pandas as pd

def process_chunk(chunk_data):
    """Process a single chunk"""
    return chunk_data.groupby('category', observed=True)[['value']].sum()

def parallel_groupby(df, n_processes=4):
    """Parallel groupby processing"""

    if n_processes < 1:
        raise ValueError("n_processes must be at least 1")
    if df.empty:
        raise ValueError("df must contain at least one row")

    # Split data into chunks
    chunk_size = max(1, len(df) // n_processes)
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process in parallel
    with Pool(n_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combine results
    return pd.concat(results).groupby(level=0, observed=True)[['value']].sum()
```

## Advanced Statistical Aggregations

### Rolling Statistics

**Reference:**

```python
# Rolling statistics within groups
grouped_values = df.groupby('category')['value']
df['rolling_mean'] = grouped_values.transform(lambda s: s.rolling(window=5).mean())
df['rolling_std'] = grouped_values.transform(lambda s: s.rolling(window=5).std())

# Expanding statistics
df['expanding_sum'] = grouped_values.transform(lambda s: s.expanding().sum())
df['expanding_mean'] = grouped_values.transform(lambda s: s.expanding().mean())
```

### Percentile Aggregations

**Reference:**

```python
# Custom percentile functions
def percentile_agg(series):
    """Calculate multiple percentiles"""
    return pd.Series({
        'p25': series.quantile(0.25),
        'p50': series.quantile(0.50),
        'p75': series.quantile(0.75),
        'p90': series.quantile(0.90),
        'p95': series.quantile(0.95),
        'p99': series.quantile(0.99)
    })

# Apply to groups
df.groupby('category')['value'].apply(percentile_agg)
```

### Statistical Tests in Groups

This optional example requires SciPy, which is not part of Lecture 08's recorded core environment. Install it in the active notebook environment with `%pip install scipy` before running the example.

**Reference:**

```python
from scipy import stats

def statistical_tests(group):
    """Perform statistical tests on group"""
    values = group['value'].dropna()

    # scipy.stats.normaltest requires at least eight observations.
    if len(values) < 8:
        stat, p_value = np.nan, np.nan
    else:
        stat, p_value = stats.normaltest(values)
    
    return pd.Series({
        'normality_stat': stat,
        'normality_p': p_value,
        'mean': values.mean(),
        'std': values.std(),
        'skewness': values.skew(),
        'kurtosis': values.kurt()
    })

# Apply to groups
df.groupby('category').apply(statistical_tests)
```

## Advanced Pivot Table Features

### Pivot Table with Custom Index

**Reference:**

```python
# Custom index in pivot tables
pivot = pd.pivot_table(df,
                      values='value',
                      index=pd.cut(df['numeric_col'], bins=5),
                      columns='category',
                      aggfunc='mean')

# Multi-level index
pivot = pd.pivot_table(df,
                      values='value',
                      index=['level1', 'level2'],
                      columns='category',
                      aggfunc='sum')
```

### Pivot Table with Time Index

**Reference:**

```python
# Time-based pivot tables
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

pivot = pd.pivot_table(df,
                      values='value',
                      index=['year', 'month'],
                      columns='category',
                      aggfunc='sum',
                      fill_value=0)
```

### Pivot Table with Custom Aggregation

The grouped weighted-mean workflow in
[Pivot Table with Custom Functions](#pivot-table-with-custom-functions) is the
canonical example. It computes and validates the numerator and denominator
before reshaping because `pivot_table(values='value')` does not pass the
separate `weight` column to its aggregator.

## Advanced GroupBy Transformations

### Ranking Within Groups

**Reference:**

```python
# Ranking within groups
df['rank'] = df.groupby('category')['value'].rank(ascending=False)
df['percentile'] = df.groupby('category')['value'].rank(pct=True)

# Multiple ranking methods
df['rank_dense'] = df.groupby('category')['value'].rank(method='dense')
df['rank_min'] = df.groupby('category')['value'].rank(method='min')
df['rank_max'] = df.groupby('category')['value'].rank(method='max')
```

### Lag and Lead Operations

**Reference:**

```python
# Lag and lead operations within groups
df['value_lag1'] = df.groupby('category')['value'].shift(1)
df['value_lag2'] = df.groupby('category')['value'].shift(2)
df['value_lead1'] = df.groupby('category')['value'].shift(-1)

# Difference from previous value
df['value_diff'] = df.groupby('category')['value'].diff()

# Percentage change
df['value_pct_change'] = df.groupby('category')['value'].pct_change()
```

### Window Functions

**Reference:**

```python
# Window functions within groups
grouped_values = df.groupby('category')['value']
df['rolling_mean'] = grouped_values.transform(lambda s: s.rolling(window=3).mean())
df['rolling_std'] = grouped_values.transform(lambda s: s.rolling(window=3).std())
df['expanding_sum'] = grouped_values.transform(lambda s: s.expanding().sum())
df['expanding_mean'] = grouped_values.transform(lambda s: s.expanding().mean())
```

## Custom GroupBy Classes

### Custom GroupBy Aggregator

**Reference:**

```python
class CustomGroupBy:
    """Custom groupby aggregator"""
    
    def __init__(self, df, group_cols):
        self.df = df
        self.group_cols = group_cols
        self.grouped = df.groupby(group_cols)
    
    def custom_agg(self, agg_col, func):
        """Apply custom aggregation function"""
        return self.grouped[agg_col].apply(func)
    
    def multiple_aggs(self, agg_dict):
        """Apply multiple aggregations"""
        return self.grouped.agg(agg_dict)
    
    def filter_groups(self, condition):
        """Filter groups based on condition"""
        return self.grouped.filter(condition)
    
    def transform_groups(self, func):
        """Transform groups"""
        return self.grouped.transform(func)

# Usage
custom_gb = CustomGroupBy(df, ['category'])
result = custom_gb.custom_agg('value', lambda x: x.quantile(0.95))
```

## Advanced Remote Computing

### Distributed Computing

**Reference:**

```python
# Distributed computing with Dask
import dask.dataframe as dd

# Read large dataset with Dask
df = dd.read_csv('large_dataset.csv')

# Perform groupby operations
result = df.groupby('category').agg({
    'value': ['sum', 'mean', 'count']
}).compute()

# Save results
result.to_csv('distributed_results.csv')
```

### Cloud Computing

**Reference:**

```python
# Cloud computing with AWS/GCP
import boto3
import pandas as pd

# Read from S3
s3 = boto3.client('s3')
df = pd.read_csv('s3://bucket/data.csv')

# Process data
result = df.groupby('category', observed=True)[['value']].sum()

# Save back to S3
result.to_csv('s3://bucket/results.csv')
```

These advanced topics will help you handle complex aggregation scenarios and optimize performance for large datasets in your data science work.
