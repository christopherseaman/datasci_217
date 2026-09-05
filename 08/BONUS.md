---
notion:
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-81b9-9e57-e4de7f2b74da"
  url: "https://app.notion.com/p/3d2d9fdd1a1a81b99e57e4de7f2b74da"
---

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

The core lecture introduces SSH, file transfer, Jupyter port forwarding, and
persistent terminal sessions. The tools below extend that workflow to distributed
and cloud-managed data processing.

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


---


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
