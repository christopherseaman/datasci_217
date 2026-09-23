---
notion:
  title_line: "# DLC: Advanced Data Aggregation Topics"
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-81b9-9e57-e4de7f2b74da"
  url: "https://app.notion.com/p/3d2d9fdd1a1a81b99e57e4de7f2b74da"
---

# DLC: Advanced Data Aggregation Topics

# Advanced GroupBy Operations

## Custom Aggregation Functions

### Reference Card: Named Aggregations with `.agg()`

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': rng.choice(['A', 'B', 'C'], size=12),
    'value': rng.normal(50, 10, size=12).round(1),
})

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

## Lambda Functions in GroupBy

### Reference Card: Lambda Aggregations

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': rng.choice(['A', 'B', 'C'], size=12),
    'value': rng.normal(50, 10, size=12).round(1),
    'other': rng.integers(1, 5, size=12),
    'score': rng.normal(0, 1, size=12).round(2),
})

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

## GroupBy with Time Windows

### Reference Card: Grouping by Time Windows

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=60, freq='7D'),
    'value': rng.normal(50, 10, size=60).round(1),
})

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

# Advanced Pivot Table Operations

## Multi-Level Pivot Tables

### Reference Card: Multi-Level Pivot Tables

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'region': rng.choice(['North', 'South'], size=16),
    'product': rng.choice(['Widget', 'Gadget'], size=16),
    'quarter': rng.choice(['Q1', 'Q2'], size=16),
    'year': rng.choice([2023, 2024], size=16),
    'sales': rng.integers(100, 1000, size=16),
    'profit': rng.normal(50, 20, size=16).round(1),
})

# Multi-level pivot tables. No fill_value: 'profit' is a mean, and an
# absent cell is not a profit of 0.
pivot = pd.pivot_table(df,
                      values=['sales', 'profit'],
                      index=['region', 'product'],
                      columns=['quarter', 'year'],
                      aggfunc={'sales': 'sum', 'profit': 'mean'},
                      margins=True)

# Flatten multi-level columns; str() handles integer years, and rstrip('_')
# tidies the margin columns, whose lower levels are empty
pivot.columns = ['_'.join(map(str, col)).rstrip('_') for col in pivot.columns]
```

## Pivot Table with Custom Functions

### Reference Card: Weighted-Mean Pivot Workaround

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': rng.choice(['A', 'B'], size=12),
    'region': rng.choice(['North', 'South'], size=12),
    'value': rng.normal(50, 10, size=12).round(1),
    'weight': rng.uniform(0.5, 2.0, size=12).round(2),
})

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

## Pivot Table with Missing Data Handling

### Reference Card: Missing Cells in Pivot Tables

A `NaN` cell means no rows had that combination. In a mean table, leave it `NaN`: nothing was measured, and 0 would report a measurement that never happened. Use `fill_value=0` only for counts and sums, where an absent combination really is zero rows. Print a count table beside the mean table so readers can see which cells are empty or rest on only a few rows.

In pandas 3, categorical groupers default to `observed=True`. Use `observed=False` only when a table must include every defined category or category combination. In a mean table, an unused category is an all-`NaN` row, which the default `dropna=True` removes, so also pass `dropna=False`. That also keeps rows with a missing key as a `NaN` row.

```python
import pandas as pd

df = pd.DataFrame({
    'category': pd.Categorical(['A', 'A', 'B', 'B'], categories=['A', 'B', 'C']),
    'region': ['North', 'South', 'North', 'North'],
    'value': [10.0, 12.0, 8.0, 6.0],
})

# Mean: absent cells stay NaN; C has no rows, so its whole row is NaN
means = pd.pivot_table(df, values='value', index='category', columns='region',
                       aggfunc='mean', observed=False, dropna=False)

# Count: an absent combination really is 0 rows
counts = pd.pivot_table(df, values='value', index='category', columns='region',
                        aggfunc='count', fill_value=0, observed=False)
print(means)
print(counts)
```

```text
region    North  South
category
A          10.0   12.0
B           7.0    NaN
C           NaN    NaN
region    North  South
category
A             1      1
B             2      0
C             0      0
```

Forward-filling or interpolating fabricates values the same way: `means.ffill()` copies A's South mean (12.0) into B and C, which have no South rows. To keep only complete rows, use `means.dropna()` (here, only A).

# Hierarchical Grouping and MultiIndex

## MultiIndex Operations

### Reference Card: MultiIndex Operations

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'level1': ['A', 'A', 'B', 'B'],
    'level2': ['X', 'Y', 'X', 'Y'],
    'value': rng.normal(50, 10, size=4).round(1),
})

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

## Advanced MultiIndex Grouping

### Reference Card: Advanced MultiIndex Grouping

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

# Advanced Statistical Aggregations

## Rolling Statistics

### Reference Card: Rolling and Expanding Statistics

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': ['A', 'B', 'C'] * 5,
    'value': rng.normal(50, 10, size=15).round(1),
})

# Rolling statistics within groups
grouped_values = df.groupby('category')['value']
df['rolling_mean'] = grouped_values.transform(lambda s: s.rolling(window=5).mean())
df['rolling_std'] = grouped_values.transform(lambda s: s.rolling(window=5).std())

# Expanding statistics
df['expanding_sum'] = grouped_values.transform(lambda s: s.expanding().sum())
df['expanding_mean'] = grouped_values.transform(lambda s: s.expanding().mean())
```

## Percentile Aggregations

### Reference Card: Percentile Aggregations

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': rng.choice(['A', 'B', 'C'], size=12),
    'value': rng.normal(50, 10, size=12).round(1),
})

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

## Statistical Tests in Groups

This optional example requires SciPy, which is not part of Lecture 08's recorded core environment. Install it in the active notebook environment with `%pip install scipy` before running the example.

### Reference Card: Statistical Tests in Groups

```python
import pandas as pd
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': ['A', 'B', 'C'] * 10,
    'value': rng.normal(50, 10, size=30).round(1),
})

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

# Advanced Pivot Table Features

## Pivot Table with Custom Index

### Reference Card: Pivot Tables with a Custom Index

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': rng.choice(['A', 'B'], size=12),
    'level1': rng.choice(['X', 'Y'], size=12),
    'level2': rng.choice(['P', 'Q'], size=12),
    'numeric_col': rng.normal(50, 10, size=12).round(1),
    'value': rng.normal(100, 20, size=12).round(1),
})

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

## Pivot Table with Time Index

### Reference Card: Pivot Tables with a Time Index

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
    'category': rng.choice(['A', 'B'], size=12),
    'value': rng.normal(100, 20, size=12).round(1),
})

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

## Pivot Table with Custom Aggregation

The grouped weighted-mean workflow in [Pivot Table with Custom Functions](#pivot-table-with-custom-functions) is the canonical example. It computes and validates the numerator and denominator before reshaping because `pivot_table(values='value')` does not pass the separate `weight` column to its aggregator.

# Advanced GroupBy Transformations

Lecture 09 teaches grouped lags and rolling windows for time-ordered rows, including past-only windows for prediction: [Entity-Aware Features and Past-Only Windows](../09/README.md#entity-aware-features-and-past-only-windows).

## Ranking Within Groups

### Reference Card: Ranking Within Groups

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': rng.choice(['A', 'B', 'C'], size=12),
    'value': rng.normal(50, 10, size=12).round(1),
})

# Ranking within groups
df['rank'] = df.groupby('category')['value'].rank(ascending=False)
df['percentile'] = df.groupby('category')['value'].rank(pct=True)

# Multiple ranking methods
df['rank_dense'] = df.groupby('category')['value'].rank(method='dense')
df['rank_min'] = df.groupby('category')['value'].rank(method='min')
df['rank_max'] = df.groupby('category')['value'].rank(method='max')
```

## Lag and Lead Operations

### Reference Card: Lag and Lead Within Groups

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': ['A', 'B', 'C'] * 5,
    'value': rng.normal(50, 10, size=15).round(1),
})

# Lag and lead operations within groups
df['value_lag1'] = df.groupby('category')['value'].shift(1)
df['value_lag2'] = df.groupby('category')['value'].shift(2)
df['value_lead1'] = df.groupby('category')['value'].shift(-1)

# Difference from previous value
df['value_diff'] = df.groupby('category')['value'].diff()

# Percentage change
df['value_pct_change'] = df.groupby('category')['value'].pct_change()
```

## Window Functions

### Reference Card: Window Functions Within Groups

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': ['A', 'B', 'C'] * 5,
    'value': rng.normal(50, 10, size=15).round(1),
})

# Window functions within groups
grouped_values = df.groupby('category')['value']
df['rolling_mean'] = grouped_values.transform(lambda s: s.rolling(window=3).mean())
df['rolling_std'] = grouped_values.transform(lambda s: s.rolling(window=3).std())
df['expanding_sum'] = grouped_values.transform(lambda s: s.expanding().sum())
df['expanding_mean'] = grouped_values.transform(lambda s: s.expanding().mean())
```

# Custom GroupBy Classes

## Custom GroupBy Aggregator

### Reference Card: Custom GroupBy Aggregator Class

```python
import pandas as pd
import numpy as np

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
rng = np.random.default_rng(42)
df = pd.DataFrame({
    'category': rng.choice(['A', 'B', 'C'], size=12),
    'value': rng.normal(50, 10, size=12).round(1),
})
custom_gb = CustomGroupBy(df, ['category'])
result = custom_gb.custom_agg('value', lambda x: x.quantile(0.95))
```

# Scaling Past Memory: Chunks and Processes

The lecture's first moves are to measure, compute several summaries in one `.agg()` call, prefer built-in aggregations, and store repeated text keys as `category`. When a file is still too large to load, or one CPU core is the bottleneck, two further options exist. Both add complexity, so time the complete operation before and after.

## Chunked Processing

`pd.read_csv(path, chunksize=n)` reads a file `n` rows at a time, so only one chunk is in memory at once. Summarize each chunk, then combine the partial summaries. The partial results must combine correctly: chunk sums add up to the total sum and chunk counts to the total count, but averaging chunk means gives the wrong mean unless you keep each chunk's sum and count. Chunking saves memory; it is not automatically faster.

```python
import pandas as pd

def chunked_groupby(file_path, group_cols, agg_cols, chunk_size=10000):
    """Sum groups from a CSV that is read in chunks."""
    results = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if not chunk.empty:
            results.append(chunk.groupby(group_cols)[agg_cols].sum())

    if not results:
        raise ValueError("input file must contain at least one data row")

    levels = list(range(results[0].index.nlevels))
    return pd.concat(results).groupby(level=levels)[agg_cols].sum()
```

## Parallel Processing

`multiprocessing.Pool` runs a function on several CPU cores at once, each in a separate Python process. Parallel work adds process startup, copying data between processes, and merge costs, so more processes do not guarantee a faster result.

Each worker process must be able to find the worker function (`process_chunk` below). In a notebook, that works only with the `fork` start method, the default on Linux (including Colab). macOS and Windows use `spawn`: a worker function defined in a notebook makes the workers fail and the cell can hang. There, put the worker function in a `.py` file and import it, and start the pool from a script under `if __name__ == "__main__":`.

```python
from multiprocessing import Pool

import pandas as pd

def process_chunk(chunk):
    return chunk.groupby('category')[['value']].sum()

def parallel_groupby(df, n_processes=4):
    if n_processes < 1:
        raise ValueError("n_processes must be at least 1")
    if df.empty:
        raise ValueError("df must contain at least one row")

    chunk_size = max(1, len(df) // n_processes)
    chunks = [df.iloc[i:i + chunk_size]
              for i in range(0, len(df), chunk_size)]

    with Pool(n_processes) as pool:
        results = pool.map(process_chunk, chunks)

    return pd.concat(results).groupby(level=0)[['value']].sum()
```

# Advanced Remote Computing

The core lecture introduces SSH, file transfer, Jupyter port forwarding, and persistent terminal sessions. The tools below extend that workflow to distributed and cloud-managed data processing.

## Distributed Computing

### Reference Card: Distributed Computing with Dask

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

## Cloud Computing

### Reference Card: Cloud Computing with S3

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




# Bonus: Advanced Data Analysis Debugging and Profiling

This bonus content covers advanced debugging techniques, performance profiling, and enterprise-level data analysis patterns for students ready to work with complex, large-scale datasets.

Under pandas 3, inferred text uses the `str` dtype and Copy-on-Write makes view-oriented mutation advice obsolete. Profile first, treat dtype changes as reviewed data-contract decisions, and apply them through explicit returned objects rather than automatic guesses.

## Memory Profiling and Optimization

### Understanding Memory Usage in pandas

```python
import pandas as pd
import numpy as np
import psutil
import os
from memory_profiler import profile

def analyze_memory_usage(df):
    """
    Comprehensive memory analysis of DataFrame
    """
    print("MEMORY USAGE ANALYSIS")
    print("=" * 30)

    # Overall memory usage
    total_memory = df.memory_usage(deep=True).sum()
    print(f"Total memory usage: {total_memory / 1024**2:.2f} MB")

    # Per-column memory usage
    memory_by_column = df.memory_usage(deep=True)
    print("\nMemory usage by column:")
    for col, mem in memory_by_column.sort_values(ascending=False).items():
        if col == 'Index':
            continue
        print(f"  {col}: {mem / 1024**2:.2f} MB ({mem/total_memory*100:.1f}%)")

    # Data type optimization recommendations
    print("\nOptimization recommendations:")

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_string_dtype(df[col]):
            # Flag repeated text as a candidate for a reviewed category contract.
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                print(f"  {col}: Repeated text ({unique_ratio:.1%} unique); measure a categorical representation if its finite-domain semantics fit")

        elif dtype in ['int64', 'float64']:
            # Check if can use smaller numeric types
            if dtype == 'int64':
                min_val, max_val = df[col].min(), df[col].max()
                if min_val >= -128 and max_val <= 127:
                    print(f"  {col}: Can use int8 (saves ~87.5% memory)")
                elif min_val >= -32768 and max_val <= 32767:
                    print(f"  {col}: Can use int16 (saves ~75% memory)")
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    print(f"  {col}: Can use int32 (saves ~50% memory)")

            elif dtype == 'float64':
                # Check if float32 precision is sufficient
                float32_version = df[col].astype('float32')
                if np.allclose(df[col], float32_version, equal_nan=True):
                    print(f"  {col}: Can use float32 (saves ~50% memory)")

def apply_reviewed_dtypes(df, dtype_map):
    """
    Return a DataFrame using caller-reviewed dtype conversions.

    Validate integer ranges, float precision, and category semantics before
    constructing dtype_map. This function deliberately does not infer them.
    """
    original_memory = df.memory_usage(deep=True).sum()
    optimized = df.astype(dtype_map)
    optimized_memory = optimized.memory_usage(deep=True).sum()
    reduction = (1 - optimized_memory/original_memory) * 100

    print(f"Memory optimization complete:")
    print(f"  Original: {original_memory / 1024**2:.2f} MB")
    print(f"  Optimized: {optimized_memory / 1024**2:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")

    return optimized

@profile  # Requires memory_profiler package
def memory_intensive_analysis(df):
    """
    Example function to profile memory usage
    """
    # Create several large intermediate objects
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()

    # Memory-intensive operations
    scaled_data = (numeric_df - numeric_df.mean()) / numeric_df.std()
    covariance_matrix = numeric_df.cov()

    # Large intermediate calculations
    result = scaled_data.dot(correlation_matrix)

    return result
```

## Advanced Debugging Patterns

### Debugging Complex Data Pipelines

```python
import functools
import traceback
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis_debug.log'),
        logging.StreamHandler()
    ]
)

def debug_pipeline_step(step_name, save_intermediate=True):
    """
    Decorator for debugging pipeline steps
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            logger.info(f"Starting step: {step_name}")

            # Log input information
            if len(args) > 0 and hasattr(args[0], 'shape'):
                logger.info(f"Input shape: {args[0].shape}")

            try:
                # Record start time and memory
                start_time = datetime.now()
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss / 1024**2

                # Execute function
                result = func(*args, **kwargs)

                # Record completion metrics
                end_time = datetime.now()
                end_memory = process.memory_info().rss / 1024**2
                execution_time = (end_time - start_time).total_seconds()

                logger.info(f"Step '{step_name}' completed successfully")
                logger.info(f"Execution time: {execution_time:.2f} seconds")
                logger.info(f"Memory change: {end_memory - start_memory:+.1f} MB")

                if hasattr(result, 'shape'):
                    logger.info(f"Output shape: {result.shape}")

                # Save intermediate result if requested
                if save_intermediate and hasattr(result, 'to_csv'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"debug_{step_name}_{timestamp}.csv"
                    result.to_csv(filename, index=False)
                    logger.info(f"Intermediate result saved: {filename}")

                return result

            except Exception as e:
                logger.error(f"Step '{step_name}' failed: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")

                # Log debugging information
                if len(args) > 0 and hasattr(args[0], 'dtypes'):
                    logger.error(f"Input data types: {dict(args[0].dtypes)}")
                    logger.error(f"Input null counts: {dict(args[0].isna().sum())}")

                raise

        return wrapper
    return decorator

class PipelineDebugger:
    """
    Advanced pipeline debugging class
    """

    def __init__(self, pipeline_name):
        self.pipeline_name = pipeline_name
        self.checkpoints = {}
        self.step_times = {}
        self.logger = logging.getLogger(f"Pipeline.{pipeline_name}")

    def checkpoint(self, step_name, data, metadata=None):
        """Save checkpoint with metadata"""
        timestamp = datetime.now()

        self.checkpoints[step_name] = {
            'timestamp': timestamp,
            'data_shape': data.shape if hasattr(data, 'shape') else str(type(data)),
            'metadata': metadata or {}
        }

        # Save data
        filename = f"checkpoint_{self.pipeline_name}_{step_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        if hasattr(data, 'to_csv'):
            data.to_csv(filename, index=False)

        self.logger.info(f"Checkpoint '{step_name}' saved: {filename}")

    def validate_step(self, step_name, data, validations):
        """Validate data at pipeline step"""
        self.logger.info(f"Validating step: {step_name}")

        validation_results = {}

        for validation_name, validation_func in validations.items():
            try:
                result = validation_func(data)
                # Validators may return a Boolean or raise. A false result is
                # a failure even when no exception was raised.
                passed = bool(result)
                validation_results[validation_name] = {
                    'passed': passed,
                    'result': result,
                }
                if passed:
                    self.logger.info(f"  ✓ {validation_name}: PASSED")
                else:
                    self.logger.error(f"  ✗ {validation_name}: FAILED")
            except Exception as e:
                validation_results[validation_name] = {'passed': False, 'error': str(e)}
                self.logger.error(f"  ✗ {validation_name}: FAILED - {str(e)}")

        # Overall validation
        all_passed = all(r['passed'] for r in validation_results.values())
        if not all_passed:
            raise ValueError(f"Validation failed for step '{step_name}'")

        return validation_results

    def summarize_pipeline(self):
        """Generate pipeline execution summary"""
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info(f"Pipeline: {self.pipeline_name}")
        self.logger.info(f"Checkpoints: {len(self.checkpoints)}")

        for step_name, checkpoint_info in self.checkpoints.items():
            self.logger.info(f"  {step_name}: {checkpoint_info['data_shape']} at {checkpoint_info['timestamp']}")

# Usage example
debugger = PipelineDebugger("customer_analysis")

@debug_pipeline_step("data_loading")
def load_and_validate_data(filename):
    df = pd.read_csv(filename)

    # Validation checks
    validations = {
        'has_data': lambda df: len(df) > 0,
        'has_required_columns': lambda df: all(col in df.columns for col in ['customer_id', 'revenue']),
        'no_all_null_columns': lambda df: not df.isna().all().any()
    }

    debugger.validate_step("data_loading", df, validations)
    debugger.checkpoint("raw_data", df, {'source': filename})

    return df

@debug_pipeline_step("data_cleaning")
def clean_data(df):
    # Cleaning operations with validation
    df_clean = df.dropna(subset=['customer_id', 'revenue'])

    validations = {
        'data_not_empty': lambda df: len(df) > 0,
        'revenue_positive': lambda df: (df['revenue'] >= 0).all(),
        'no_duplicate_customers': lambda df: df['customer_id'].nunique() == len(df)
    }

    debugger.validate_step("data_cleaning", df_clean, validations)
    debugger.checkpoint("cleaned_data", df_clean)

    return df_clean
```

## Performance Profiling

### Advanced Profiling Techniques

```python
import cProfile
import pstats
import time
from functools import wraps
import pandas as pd
import numpy as np

def profile_function(sort_by='cumulative', lines_to_show=20):
    """
    Decorator to profile function performance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()

            # Generate profile report
            stats = pstats.Stats(profiler)
            stats.sort_stats(sort_by)

            print(f"\nPROFILE REPORT: {func.__name__}")
            print("=" * 50)
            stats.print_stats(lines_to_show)

            return result

        return wrapper
    return decorator

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis tools
    """

    @staticmethod
    def compare_operations(operations, data, iterations=5):
        """
        Compare performance of different operations
        """
        results = {}

        print("PERFORMANCE COMPARISON")
        print("=" * 30)

        for name, operation in operations.items():
            times = []

            for i in range(iterations):
                start_time = time.perf_counter()
                try:
                    result = operation(data)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                except Exception as e:
                    print(f"  {name}: FAILED - {str(e)}")
                    times.append(float('inf'))
                    break

            if times and times[0] != float('inf'):
                avg_time = np.mean(times)
                std_time = np.std(times)
                results[name] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'times': times
                }
                print(f"  {name}: {avg_time:.4f}s (±{std_time:.4f}s)")
            else:
                results[name] = {'avg_time': float('inf'), 'failed': True}

        # Show relative performance
        valid_results = {k: v for k, v in results.items() if v['avg_time'] != float('inf')}
        if valid_results:
            fastest = min(valid_results.values(), key=lambda x: x['avg_time'])['avg_time']

            print("\nRelative performance:")
            for name, result in valid_results.items():
                speedup = result['avg_time'] / fastest
                print(f"  {name}: {speedup:.1f}x slower than fastest")

        return results

    @staticmethod
    def analyze_scaling_performance(operation, data_sizes, data_generator):
        """
        Analyze how operation scales with data size
        """
        results = []

        print("SCALING PERFORMANCE ANALYSIS")
        print("=" * 35)

        for size in data_sizes:
            data = data_generator(size)

            # Time the operation
            start_time = time.perf_counter()
            try:
                result = operation(data)
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                results.append({
                    'size': size,
                    'time': execution_time,
                    'time_per_row': execution_time / size
                })

                print(f"  Size {size:,}: {execution_time:.4f}s ({execution_time/size*1000:.2f}ms per 1000 rows)")

            except Exception as e:
                print(f"  Size {size:,}: FAILED - {str(e)}")
                results.append({'size': size, 'time': float('inf'), 'error': str(e)})

        return results

# Example usage
@profile_function()
def intensive_analysis(df):
    """Example intensive analysis function"""
    # Multiple operations that might be slow
    correlation_matrix = df.corr()
    grouped_stats = df.groupby(df.columns[0]).agg(['mean', 'std', 'count'])
    rolling_means = df.select_dtypes(include=[np.number]).rolling(window=10).mean()

    return correlation_matrix, grouped_stats, rolling_means

# Performance comparison example
def performance_comparison_example(df):
    operations = {
        'pandas_corr': lambda df: df.corr(),
        'numpy_corrcoef': lambda df: np.corrcoef(df.select_dtypes(include=[np.number]).T),
        'manual_correlation': lambda df: df.cov() / (df.std().values.reshape(-1, 1) @ df.std().values.reshape(1, -1))
    }

    analyzer = PerformanceAnalyzer()
    results = analyzer.compare_operations(operations, df.select_dtypes(include=[np.number]))

    return results
```

## Enterprise-Level Data Validation

### Comprehensive Data Validation Framework

```python
from typing import Dict, List, Callable, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    rule_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Dict[str, Any]
    affected_columns: List[str] = None
    affected_rows: int = 0

class DataValidator:
    """
    Enterprise-level data validation framework
    """

    def __init__(self):
        self.rules = {}
        self.results = []

    def add_rule(self, name: str, rule_func: Callable, severity: ValidationSeverity = ValidationSeverity.WARNING):
        """Add validation rule"""
        self.rules[name] = {
            'function': rule_func,
            'severity': severity
        }

    def validate(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Execute all validation rules"""
        self.results = []

        print("EXECUTING DATA VALIDATION RULES")
        print("=" * 40)

        for rule_name, rule_config in self.rules.items():
            try:
                result = rule_config['function'](df)

                if isinstance(result, ValidationResult):
                    validation_result = result
                else:
                    # Create result from boolean return
                    validation_result = ValidationResult(
                        rule_name=rule_name,
                        severity=rule_config['severity'],
                        passed=bool(result),
                        message=f"Rule '{rule_name}' {'passed' if result else 'failed'}",
                        details={}
                    )

                self.results.append(validation_result)

                # Print result
                status_symbol = "✓" if validation_result.passed else "✗"
                print(f"{status_symbol} {rule_name} ({validation_result.severity.value}): {validation_result.message}")

            except Exception as e:
                error_result = ValidationResult(
                    rule_name=rule_name,
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    message=f"Validation rule failed with error: {str(e)}",
                    details={'error': str(e)}
                )
                self.results.append(error_result)
                print(f"✗ {rule_name} (CRITICAL): Rule execution failed - {str(e)}")

        return self.results

    def get_summary(self) -> Dict:
        """Get validation summary"""
        summary = {
            'total_rules': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'failed': sum(1 for r in self.results if not r.passed),
            'by_severity': {}
        }

        for severity in ValidationSeverity:
            severity_results = [r for r in self.results if r.severity == severity]
            summary['by_severity'][severity.value] = {
                'total': len(severity_results),
                'passed': sum(1 for r in severity_results if r.passed),
                'failed': sum(1 for r in severity_results if not r.passed)
            }

        return summary

    def generate_report(self, output_file: str = 'validation_report.md'):
        """Generate comprehensive validation report"""
        summary = self.get_summary()

        with open(output_file, 'w') as f:
            f.write("# Data Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now()}\n")
            f.write(f"**Total Rules:** {summary['total_rules']}\n")
            f.write(f"**Passed:** {summary['passed']}\n")
            f.write(f"**Failed:** {summary['failed']}\n\n")

            f.write("## Results by Severity\n\n")
            for severity, counts in summary['by_severity'].items():
                if counts['total'] > 0:
                    f.write(f"### {severity.upper()}\n")
                    f.write(f"- Total: {counts['total']}\n")
                    f.write(f"- Passed: {counts['passed']}\n")
                    f.write(f"- Failed: {counts['failed']}\n\n")

            f.write("## Detailed Results\n\n")
            for result in self.results:
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                f.write(f"### {result.rule_name} - {status}\n")
                f.write(f"**Severity:** {result.severity.value.upper()}\n")
                f.write(f"**Message:** {result.message}\n")

                if result.details:
                    f.write("**Details:**\n")
                    for key, value in result.details.items():
                        f.write(f"- {key}: {value}\n")

                if result.affected_columns:
                    f.write(f"**Affected Columns:** {', '.join(result.affected_columns)}\n")

                if result.affected_rows > 0:
                    f.write(f"**Affected Rows:** {result.affected_rows}\n")

                f.write("\n")

# Predefined validation rules
def create_standard_validator() -> DataValidator:
    """Create validator with standard data quality rules"""
    validator = DataValidator()

    # Data completeness rules
    def check_missing_data_threshold(df, threshold=0.5):
        missing_cols = []
        for col in df.columns:
            missing_pct = df[col].isna().mean()
            if missing_pct > threshold:
                missing_cols.append(f"{col} ({missing_pct:.1%})")

        passed = len(missing_cols) == 0
        return ValidationResult(
            rule_name="missing_data_threshold",
            severity=ValidationSeverity.WARNING,
            passed=passed,
            message=f"No columns exceed {threshold:.0%} missing data" if passed else f"Columns with >{threshold:.0%} missing: {', '.join(missing_cols)}",
            details={'threshold': threshold, 'violating_columns': missing_cols},
            affected_columns=[col.split(' ')[0] for col in missing_cols]
        )

    def check_duplicate_rows(df):
        duplicate_count = df.duplicated().sum()
        passed = duplicate_count == 0

        return ValidationResult(
            rule_name="duplicate_rows",
            severity=ValidationSeverity.WARNING,
            passed=passed,
            message=f"No duplicate rows found" if passed else f"Found {duplicate_count} duplicate rows ({duplicate_count/len(df):.1%})",
            details={'duplicate_count': duplicate_count, 'duplicate_percentage': duplicate_count/len(df)},
            affected_rows=duplicate_count
        )

    def check_data_types(df):
        type_issues = []
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                # Check if numeric-looking data is stored as text.
                try:
                    numeric_version = pd.to_numeric(df[col], errors='coerce')
                    non_numeric_count = numeric_version.isna().sum() - df[col].isna().sum()
                    if non_numeric_count > 0 and non_numeric_count < len(df) * 0.1:
                        type_issues.append(f"{col} (appears mostly numeric but has {non_numeric_count} non-numeric values)")
                except (TypeError, ValueError):
                    pass

        passed = len(type_issues) == 0
        return ValidationResult(
            rule_name="data_type_consistency",
            severity=ValidationSeverity.INFO,
            passed=passed,
            message="No data type inconsistencies found" if passed else f"Potential type issues: {'; '.join(type_issues)}",
            details={'issues': type_issues},
            affected_columns=[issue.split(' ')[0] for issue in type_issues]
        )

    # Add rules to validator
    validator.add_rule("missing_data_threshold", check_missing_data_threshold, ValidationSeverity.WARNING)
    validator.add_rule("duplicate_rows", check_duplicate_rows, ValidationSeverity.WARNING)
    validator.add_rule("data_type_consistency", check_data_types, ValidationSeverity.INFO)

    return validator

# Example usage
def run_enterprise_validation(df):
    """Run enterprise-level validation on dataset"""
    validator = create_standard_validator()

    # Add custom rules
    validator.add_rule(
        "sufficient_data_size",
        lambda df: ValidationResult(
            rule_name="sufficient_data_size",
            severity=ValidationSeverity.ERROR,
            passed=len(df) >= 1000,
            message=f"Dataset has {len(df)} rows - {'sufficient' if len(df) >= 1000 else 'insufficient'} for analysis",
            details={'row_count': len(df), 'minimum_required': 1000}
        ),
        ValidationSeverity.ERROR
    )

    # Run validation
    results = validator.validate(df)

    # Generate report
    validator.generate_report('enterprise_validation_report.md')

    # Print summary
    summary = validator.get_summary()
    print(f"\nVALIDATION SUMMARY:")
    print(f"Total rules: {summary['total_rules']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")

    return results, summary
```

This bonus content provides enterprise-level debugging, profiling, and validation techniques that professional data scientists use in production environments. These advanced patterns help ensure robust, scalable data analysis workflows.
