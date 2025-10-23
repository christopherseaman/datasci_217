# Demo 3: Remote Computing and Performance

## Learning Objectives
- Set up SSH connections for remote computing
- Use tmux for persistent sessions
- Optimize performance for large datasets
- Apply parallel processing techniques

## Setup

```python
import pandas as pd
import numpy as np
import time
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
```

## Part 1: Performance Optimization

### Create Large Dataset

```python
# Create large dataset for performance testing
print("=== Creating Large Dataset ===")
n_rows = 100000
n_groups = 1000

# Generate data
data = {
    'group': np.random.randint(0, n_groups, n_rows),
    'value1': np.random.randn(n_rows),
    'value2': np.random.randn(n_rows),
    'value3': np.random.randn(n_rows),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows)
}

df_large = pd.DataFrame(data)
print(f"Dataset shape: {df_large.shape}")
print(f"Memory usage: {df_large.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### Performance Comparison

```python
# Compare different aggregation methods
print("=== Performance Comparison ===")

# Method 1: Multiple groupby operations
start_time = time.time()
result1 = df_large.groupby('group')['value1'].sum()
result2 = df_large.groupby('group')['value2'].sum()
result3 = df_large.groupby('group')['value3'].sum()
method1_time = time.time() - start_time

# Method 2: Single groupby with multiple aggregations
start_time = time.time()
result4 = df_large.groupby('group').agg({
    'value1': 'sum',
    'value2': 'sum',
    'value3': 'sum'
})
method2_time = time.time() - start_time

print(f"Method 1 (multiple groupby): {method1_time:.4f} seconds")
print(f"Method 2 (single groupby): {method2_time:.4f} seconds")
print(f"Performance improvement: {method1_time/method2_time:.2f}x")
```

### Memory Optimization

```python
# Memory optimization techniques
print("=== Memory Optimization ===")

# Check data types
print("Original data types:")
print(df_large.dtypes)
print(f"Memory usage: {df_large.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize data types
df_optimized = df_large.copy()
df_optimized['group'] = df_optimized['group'].astype('category')
df_optimized['category'] = df_optimized['category'].astype('category')

print("\nOptimized data types:")
print(df_optimized.dtypes)
print(f"Memory usage: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Memory reduction: {(1 - df_optimized.memory_usage(deep=True).sum() / df_large.memory_usage(deep=True).sum()) * 100:.1f}%")
```

## Part 2: Parallel Processing

### Chunked Processing

```python
# Process data in chunks
def process_chunk(chunk):
    """Process a single chunk of data"""
    return chunk.groupby('group').agg({
        'value1': 'sum',
        'value2': 'sum',
        'value3': 'sum'
    })

# Sequential processing
print("=== Sequential Processing ===")
start_time = time.time()
chunk_size = 10000
chunks = [df_large.iloc[i:i+chunk_size] for i in range(0, len(df_large), chunk_size)]
sequential_results = []
for chunk in chunks:
    result = process_chunk(chunk)
    sequential_results.append(result)
sequential_time = time.time() - start_time

print(f"Sequential processing time: {sequential_time:.4f} seconds")
```

### Parallel Processing

```python
# Parallel processing
print("=== Parallel Processing ===")
start_time = time.time()
with Pool(processes=4) as pool:
    parallel_results = pool.map(process_chunk, chunks)
parallel_time = time.time() - start_time

print(f"Parallel processing time: {parallel_time:.4f} seconds")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")
```

## Part 3: Remote Computing Simulation

### SSH Connection Simulation

```python
# Simulate SSH connection setup
print("=== SSH Connection Simulation ===")
print("In a real scenario, you would:")
print("1. Generate SSH key pair: ssh-keygen -t rsa -b 4096")
print("2. Copy public key to server: ssh-copy-id username@server.com")
print("3. Connect to server: ssh username@server.com")
print("4. Set up environment on remote server")
```

### tmux Session Management

```python
# Simulate tmux session management
print("=== tmux Session Management ===")
print("tmux commands for persistent sessions:")
print("- tmux new-session -s analysis")
print("- tmux list-sessions")
print("- tmux attach-session -t analysis")
print("- Ctrl+b d (detach from session)")
print("- tmux kill-session -t analysis")
```

### Remote Data Analysis Workflow

```python
# Simulate remote data analysis workflow
def simulate_remote_analysis():
    """Simulate remote data analysis workflow"""
    print("=== Remote Data Analysis Workflow ===")
    
    # Simulate loading large dataset
    print("1. Loading large dataset on remote server...")
    time.sleep(0.1)  # Simulate loading time
    
    # Simulate analysis
    print("2. Performing aggregation analysis...")
    start_time = time.time()
    result = df_large.groupby('group').agg({
        'value1': ['sum', 'mean', 'std'],
        'value2': ['sum', 'mean', 'std'],
        'value3': ['sum', 'mean', 'std']
    })
    analysis_time = time.time() - start_time
    
    print(f"3. Analysis completed in {analysis_time:.4f} seconds")
    print(f"4. Results shape: {result.shape}")
    
    # Simulate saving results
    print("5. Saving results to remote server...")
    time.sleep(0.1)  # Simulate saving time
    
    # Simulate downloading results
    print("6. Downloading results to local machine...")
    time.sleep(0.1)  # Simulate download time
    
    print("7. Remote analysis workflow completed!")
    return result

# Run simulation
remote_result = simulate_remote_analysis()
```

## Part 4: Performance Monitoring

### Memory Usage Tracking

```python
# Monitor memory usage during operations
import psutil
import os

def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

print("=== Memory Usage Monitoring ===")
print(f"Initial memory usage: {monitor_memory():.2f} MB")

# Perform memory-intensive operation
start_memory = monitor_memory()
large_operation = df_large.groupby('group').agg({
    'value1': 'sum',
    'value2': 'sum',
    'value3': 'sum'
})
end_memory = monitor_memory()

print(f"Memory usage after operation: {end_memory:.2f} MB")
print(f"Memory increase: {end_memory - start_memory:.2f} MB")
```

### Performance Profiling

```python
# Performance profiling
def profile_operation(func, *args, **kwargs):
    """Profile a function's performance"""
    start_time = time.time()
    start_memory = monitor_memory()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    end_memory = monitor_memory()
    
    return {
        'result': result,
        'execution_time': end_time - start_time,
        'memory_usage': end_memory - start_memory
    }

# Profile different operations
print("=== Performance Profiling ===")

# Profile groupby operation
groupby_profile = profile_operation(
    lambda: df_large.groupby('group')['value1'].sum()
)
print(f"GroupBy operation: {groupby_profile['execution_time']:.4f}s, {groupby_profile['memory_usage']:.2f}MB")

# Profile pivot table operation
pivot_profile = profile_operation(
    lambda: pd.pivot_table(df_large, values='value1', index='group', columns='category', aggfunc='sum')
)
print(f"Pivot table operation: {pivot_profile['execution_time']:.4f}s, {pivot_profile['memory_usage']:.2f}MB")
```

## Part 5: Optimization Strategies

### Data Type Optimization

```python
# Optimize data types for better performance
print("=== Data Type Optimization ===")

# Check current data types
print("Current data types:")
print(df_large.dtypes)

# Optimize integer columns
df_optimized = df_large.copy()
df_optimized['group'] = pd.Categorical(df_optimized['group'])

# Measure performance improvement
start_time = time.time()
result_optimized = df_optimized.groupby('group')['value1'].sum()
optimized_time = time.time() - start_time

start_time = time.time()
result_original = df_large.groupby('group')['value1'].sum()
original_time = time.time() - start_time

print(f"Original performance: {original_time:.4f} seconds")
print(f"Optimized performance: {optimized_time:.4f} seconds")
print(f"Performance improvement: {original_time/optimized_time:.2f}x")
```

### Chunked Processing Strategy

```python
# Chunked processing for large datasets
def process_large_dataset(df, chunk_size=10000):
    """Process large dataset in chunks"""
    results = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk_result = chunk.groupby('group').agg({
            'value1': 'sum',
            'value2': 'sum',
            'value3': 'sum'
        })
        results.append(chunk_result)
    
    # Combine results
    final_result = pd.concat(results).groupby(level=0).sum()
    return final_result

print("=== Chunked Processing Strategy ===")
start_time = time.time()
chunked_result = process_large_dataset(df_large, chunk_size=5000)
chunked_time = time.time() - start_time

print(f"Chunked processing time: {chunked_time:.4f} seconds")
print(f"Result shape: {chunked_result.shape}")
```

## Key Takeaways

1. **Performance Optimization**: Use single groupby with multiple aggregations
2. **Memory Optimization**: Optimize data types and use categorical data
3. **Parallel Processing**: Use multiprocessing for CPU-intensive tasks
4. **Remote Computing**: Use SSH and tmux for large dataset analysis
5. **Chunked Processing**: Process large datasets in manageable chunks
6. **Performance Monitoring**: Track memory usage and execution time
7. **Data Type Optimization**: Use appropriate data types for better performance

## Next Steps

- Practice with your own large datasets
- Set up remote computing environment
- Learn about distributed computing frameworks
- Explore cloud computing options for big data analysis
