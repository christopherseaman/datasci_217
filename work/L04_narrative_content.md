# Lecture 04: NumPy Foundations for Data Science

**Duration**: 4 hours  
**Focus**: N-dimensional arrays, vectorized operations, performance optimization, mathematical computing

## Learning Objectives

By the end of this lecture, students will:
- Understand why NumPy is the foundation of Python's data science ecosystem
- Master N-dimensional array operations and vectorized computing
- Implement high-performance numerical algorithms with 10-100x speedups
- Apply broadcasting and advanced indexing for complex data manipulations
- Optimize memory usage and computational efficiency in data processing

---

## Part 1: Why NumPy Changes Everything (45 minutes)

### Opening Performance Demonstration: "The 100x Speedup"

```python
import numpy as np
import time
import matplotlib.pyplot as plt

# Generate test data
n = 1_000_000
python_list = list(range(n))
numpy_array = np.arange(n)

print(f"Processing {n:,} numbers...")

# Pure Python approach
start_time = time.time()
python_result = [x**2 + 2*x + 1 for x in python_list]
python_time = time.time() - start_time

# NumPy approach
start_time = time.time()
numpy_result = numpy_array**2 + 2*numpy_array + 1
numpy_time = time.time() - start_time

print(f"Pure Python: {python_time:.4f} seconds")
print(f"NumPy:       {numpy_time:.4f} seconds")
print(f"Speedup:     {python_time/numpy_time:.1f}x faster")

# Verify results are identical
print(f"Results match: {np.allclose(python_result, numpy_result)}")
```

**Typical Results**:
- Pure Python: 0.2847 seconds
- NumPy: 0.0029 seconds  
- Speedup: **98.2x faster**

### The Science Behind the Speed

```python
# Memory efficiency comparison
import sys

# Python list memory usage
python_list = [i for i in range(1000)]
python_memory = sys.getsizeof(python_list)
for item in python_list[:10]:  # Sample a few items
    python_memory += sys.getsizeof(item)

# NumPy array memory usage
numpy_array = np.arange(1000)
numpy_memory = numpy_array.nbytes

print(f"Python list memory: {python_memory:,} bytes")
print(f"NumPy array memory: {numpy_memory:,} bytes")
print(f"Memory efficiency: {python_memory/numpy_memory:.1f}x more efficient")

# Data type precision and memory
print("\nNumPy Data Type Efficiency:")
data = np.arange(1000)
for dtype in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
    typed_array = data.astype(dtype)
    print(f"{dtype:>7}: {typed_array.nbytes:,} bytes ({typed_array.itemsize} bytes per element)")
```

### CPU Optimization: SIMD and Vectorization

```python
# Demonstrate vectorized operations
def compare_operations(size=1_000_000):
    """Compare different approaches to mathematical operations."""
    
    # Generate test data
    x = np.random.random(size)
    y = np.random.random(size)
    
    results = {}
    
    # Method 1: Pure Python with explicit loops
    start = time.time()
    python_result = []
    for i in range(size):
        python_result.append(x[i] * y[i] + np.sin(x[i]))
    results['Pure Python'] = time.time() - start
    
    # Method 2: NumPy vectorized
    start = time.time()
    numpy_result = x * y + np.sin(x)
    results['NumPy Vectorized'] = time.time() - start
    
    # Method 3: NumPy with explicit indexing (slower)
    start = time.time()
    indexed_result = np.array([x[i] * y[i] + np.sin(x[i]) for i in range(size)])
    results['NumPy Indexed'] = time.time() - start
    
    # Display results
    fastest = min(results.values())
    for method, time_taken in results.items():
        speedup = fastest / time_taken if time_taken != fastest else 1.0
        print(f"{method:>15}: {time_taken:.4f}s (baseline)" if time_taken == fastest 
              else f"{method:>15}: {time_taken:.4f}s ({speedup:.1f}x slower)")
    
    return results

compare_operations()
```

---

## Part 2: Array Thinking - Beyond Lists (60 minutes)

### N-Dimensional Data Structures

```python
# 1D Array: Time series data
temperatures = np.array([20.1, 21.3, 19.8, 22.5, 18.9])
print(f"1D - Shape: {temperatures.shape}, Size: {temperatures.size}")

# 2D Array: Tabular data (like spreadsheet)
sales_data = np.array([
    [100, 150, 120],  # Product A sales by month
    [200, 180, 220],  # Product B sales by month
    [80,  95,  110]   # Product C sales by month
])
print(f"2D - Shape: {sales_data.shape} (rows × columns)")

# 3D Array: Time series of tabular data
# Shape: (time_periods, products, months)
quarterly_data = np.random.randint(50, 300, size=(4, 3, 3))
print(f"3D - Shape: {quarterly_data.shape} (quarters × products × months)")

# 4D Array: Multi-location time series
# Shape: (locations, quarters, products, months)
multi_location = np.random.randint(50, 300, size=(5, 4, 3, 3))
print(f"4D - Shape: {multi_location.shape} (locations × quarters × products × months)")
```

### Array Creation Patterns

```python
# Zeros and ones - common initialization patterns
zeros_2d = np.zeros((3, 4))           # 3×4 matrix of zeros
ones_3d = np.ones((2, 3, 4))          # 2×3×4 tensor of ones
identity = np.eye(5)                  # 5×5 identity matrix

# Ranges and sequences
linear_space = np.linspace(0, 10, 101)    # 101 points from 0 to 10
log_space = np.logspace(0, 3, 4)          # [1, 10, 100, 1000]
mesh_x, mesh_y = np.meshgrid(
    np.linspace(-5, 5, 11),
    np.linspace(-3, 3, 7)
)  # 2D coordinate grids

# Random data generation for testing
np.random.seed(42)  # Reproducible results
normal_data = np.random.normal(100, 15, size=(1000, 5))  # Mean=100, std=15
uniform_data = np.random.uniform(0, 1, size=(500, 3))
choice_data = np.random.choice(['A', 'B', 'C'], size=100, p=[0.5, 0.3, 0.2])

# Structured arrays for mixed data types
structured_dtype = np.dtype([
    ('name', 'U20'),      # Unicode string, max 20 chars
    ('age', 'i4'),        # 32-bit integer
    ('salary', 'f8')      # 64-bit float
])

employees = np.array([
    ('Alice Johnson', 32, 75000.0),
    ('Bob Smith', 28, 68000.0),
    ('Carol Davis', 35, 82000.0)
], dtype=structured_dtype)

print(f"Employee names: {employees['name']}")
print(f"Average salary: ${employees['salary'].mean():,.2f}")
```

### Data Type Optimization

```python
def optimize_data_types(data):
    """Demonstrate automatic data type optimization."""
    
    # Start with default (usually int64/float64)
    original_array = np.array(data)
    original_memory = original_array.nbytes
    
    print(f"Original: {original_array.dtype}, {original_memory} bytes")
    
    # Optimize integer data
    if np.issubdtype(original_array.dtype, np.integer):
        max_val = np.max(original_array)
        min_val = np.min(original_array)
        
        if min_val >= 0:  # Unsigned
            if max_val <= 255:
                optimized = original_array.astype(np.uint8)
            elif max_val <= 65535:
                optimized = original_array.astype(np.uint16)
            else:
                optimized = original_array.astype(np.uint32)
        else:  # Signed
            if min_val >= -128 and max_val <= 127:
                optimized = original_array.astype(np.int8)
            elif min_val >= -32768 and max_val <= 32767:
                optimized = original_array.astype(np.int16)
            else:
                optimized = original_array.astype(np.int32)
    
    # Optimize float data
    elif np.issubdtype(original_array.dtype, np.floating):
        # Try float32 if precision loss is acceptable
        float32_version = original_array.astype(np.float32)
        if np.allclose(original_array, float32_version):
            optimized = float32_version
        else:
            optimized = original_array
    
    optimized_memory = optimized.nbytes
    savings = (original_memory - optimized_memory) / original_memory * 100
    
    print(f"Optimized: {optimized.dtype}, {optimized_memory} bytes")
    print(f"Memory savings: {savings:.1f}%")
    
    return optimized

# Example usage
large_integers = np.random.randint(0, 100, size=10000)
optimize_data_types(large_integers)
```

---

## Part 3: Advanced Operations - The Power Tools (75 minutes)

### Indexing and Slicing Mastery

```python
# Create sample 3D dataset: (days, products, metrics)
data = np.random.randint(10, 100, size=(7, 5, 4))
print(f"Dataset shape: {data.shape} (days × products × metrics)")

# Basic indexing
print(f"Day 0, Product 0, All metrics: {data[0, 0, :]}")
print(f"All days, Product 2, Metric 1: {data[:, 2, 1]}")

# Advanced slicing
print(f"First 3 days, Last 2 products, All metrics:")
print(data[:3, -2:, :])

# Boolean indexing - the game changer
high_performance = data > 80
print(f"Number of high-performance instances: {np.sum(high_performance)}")

# Extract all high-performance values
high_values = data[high_performance]
print(f"High performance values: {high_values}")

# Conditional indexing
# Find products that had any day with all metrics > 70
all_metrics_high = np.all(data > 70, axis=2)  # Check across metrics
products_with_high_days = np.any(all_metrics_high, axis=0)  # Check across days
print(f"Products with at least one high-performance day: {np.where(products_with_high_days)[0]}")

# Fancy indexing - select specific combinations
selected_days = [0, 2, 4]
selected_products = [1, 3]
selected_metrics = [0, 2]

# This creates a subset: selected days × selected products × selected metrics
subset = data[np.ix_(selected_days, selected_products, selected_metrics)]
print(f"Subset shape: {subset.shape}")
```

### Broadcasting: The Silent Superpower

```python
# Broadcasting enables operations between arrays of different shapes
sales_by_month = np.array([[100, 120, 140],    # Product A
                          [200, 180, 220],    # Product B
                          [150, 160, 180]])   # Product C

# Scenario 1: Apply monthly growth rates
monthly_growth = np.array([1.05, 1.08, 1.12])  # 5%, 8%, 12%
projected_sales = sales_by_month * monthly_growth
print("Projected sales with growth:")
print(projected_sales)

# Scenario 2: Normalize by product totals
product_totals = np.sum(sales_by_month, axis=1, keepdims=True)
normalized_sales = sales_by_month / product_totals
print("\nSales as fraction of product total:")
print(normalized_sales)

# Scenario 3: Complex broadcasting
# Add fixed costs per product and variable costs per month
fixed_costs_per_product = np.array([[50], [75], [60]])    # Shape: (3, 1)
variable_costs_per_month = np.array([10, 15, 20])         # Shape: (3,)

total_costs = fixed_costs_per_product + variable_costs_per_month
profit = sales_by_month - total_costs
print("\nProfit after costs:")
print(profit)

# Broadcasting rules visualization
def show_broadcasting_rules():
    """Demonstrate NumPy broadcasting alignment rules."""
    examples = [
        ("(3, 4) + (4,)", "✓ Compatible - trailing dimensions match"),
        ("(3, 4) + (3, 1)", "✓ Compatible - one dimension is 1"),
        ("(3, 4) + (1, 4)", "✓ Compatible - one dimension is 1"),
        ("(3, 4) + (2, 4)", "✗ Incompatible - neither is 1, and 3 ≠ 2"),
        ("(3, 1, 4) + (2, 4)", "✓ Compatible - adds dimension"),
    ]
    
    for shapes, result in examples:
        print(f"{shapes:20} → {result}")

show_broadcasting_rules()
```

### Mathematical Operations Arsenal

```python
# Statistical operations across different axes
data = np.random.normal(100, 15, size=(30, 5, 4))  # 30 days, 5 products, 4 metrics

# Axis operations
daily_averages = np.mean(data, axis=0)      # Average across days: (5, 4)
product_totals = np.sum(data, axis=1)       # Sum across products: (30, 4)
metric_stdev = np.std(data, axis=2)         # StdDev across metrics: (30, 5)

print(f"Daily averages shape: {daily_averages.shape}")
print(f"Product totals shape: {product_totals.shape}")
print(f"Metric standard deviations shape: {metric_stdev.shape}")

# Advanced mathematical functions
angles = np.linspace(0, 2*np.pi, 100)
sine_wave = np.sin(angles)
cosine_wave = np.cos(angles)
combined_wave = sine_wave * cosine_wave

# Financial calculations
principal = 10000
rates = np.array([0.03, 0.04, 0.05, 0.06])  # 3%, 4%, 5%, 6%
years = np.arange(1, 11)  # 1 to 10 years

# Compound interest calculation using broadcasting
compound_growth = principal * (1 + rates.reshape(-1, 1)) ** years
print(f"Compound growth shape: {compound_growth.shape} (rates × years)")

# Linear algebra operations
A = np.random.random((3, 3))
B = np.random.random((3, 3))

# Matrix operations
matrix_product = A @ B                    # Matrix multiplication (preferred)
element_product = A * B                   # Element-wise multiplication
transpose = A.T                           # Transpose
determinant = np.linalg.det(A)           # Determinant
eigenvals, eigenvecs = np.linalg.eig(A)  # Eigenvalues and eigenvectors

print(f"Determinant: {determinant:.4f}")
print(f"Eigenvalues: {eigenvals}")
```

### Performance Optimization Techniques

```python
def performance_comparison():
    """Compare different approaches to common operations."""
    
    # Large dataset for timing
    data = np.random.random((10000, 100))
    
    times = {}
    
    # Method 1: Python loops (slow)
    start = time.time()
    result1 = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result1[i, j] = data[i, j] ** 2 + 2 * data[i, j]
    times['Python loops'] = time.time() - start
    
    # Method 2: NumPy vectorized (fast)
    start = time.time()
    result2 = data**2 + 2*data
    times['NumPy vectorized'] = time.time() - start
    
    # Method 3: NumPy with pre-allocation
    start = time.time()
    result3 = np.empty_like(data)
    np.square(data, out=result3)  # In-place square
    result3 += 2 * data
    times['NumPy in-place'] = time.time() - start
    
    # Method 4: Using numexpr for very large operations
    try:
        import numexpr as ne
        start = time.time()
        result4 = ne.evaluate("data**2 + 2*data")
        times['NumExpr'] = time.time() - start
    except ImportError:
        print("NumExpr not installed - skipping")
    
    # Display results
    fastest = min(times.values())
    for method, time_taken in times.items():
        speedup = time_taken / fastest
        print(f"{method:>18}: {time_taken:.4f}s ({speedup:.1f}x)")
    
    return times

performance_comparison()
```

---

## Part 4: Real-World Applications (60 minutes)

### Data Analysis Pipeline with NumPy

```python
class DataProcessor:
    """Professional data processing with NumPy."""
    
    def __init__(self):
        self.processing_stats = {}
    
    def load_and_validate(self, data_source):
        """Load data with validation and basic cleaning."""
        if isinstance(data_source, str):  # File path
            data = np.loadtxt(data_source, delimiter=',', skiprows=1)
        else:  # Direct array
            data = np.asarray(data_source)
        
        # Validation
        if data.size == 0:
            raise ValueError("Empty dataset")
        
        # Check for problematic values
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        self.processing_stats['nan_values'] = nan_count
        self.processing_stats['inf_values'] = inf_count
        self.processing_stats['shape'] = data.shape
        
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values detected")
        if inf_count > 0:
            print(f"Warning: {inf_count} infinite values detected")
        
        return data
    
    def outlier_detection(self, data, method='iqr', factor=1.5):
        """Detect outliers using various methods."""
        if method == 'iqr':
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
            outliers = z_scores > factor
        
        elif method == 'modified_zscore':
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            modified_z = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z) > factor
        
        outlier_count = np.sum(outliers)
        outlier_percentage = outlier_count / data.size * 100
        
        print(f"Outlier detection ({method}): {outlier_count} outliers ({outlier_percentage:.2f}%)")
        
        return outliers
    
    def normalize_data(self, data, method='minmax'):
        """Normalize data using different methods."""
        if method == 'minmax':
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            normalized = (data - data_min) / (data_max - data_min)
            
        elif method == 'zscore':
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            normalized = (data - data_mean) / data_std
            
        elif method == 'robust':
            data_median = np.median(data, axis=0)
            data_mad = np.median(np.abs(data - data_median), axis=0)
            normalized = (data - data_median) / data_mad
        
        return normalized
    
    def compute_statistics(self, data):
        """Compute comprehensive statistics."""
        stats = {
            'count': data.shape[0],
            'mean': np.mean(data, axis=0),
            'median': np.median(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0),
            'skewness': self._skewness(data),
            'kurtosis': self._kurtosis(data)
        }
        
        return stats
    
    def _skewness(self, data):
        """Calculate skewness (third moment)."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        skew = np.mean(((data - mean) / std) ** 3, axis=0)
        return skew
    
    def _kurtosis(self, data):
        """Calculate kurtosis (fourth moment)."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        kurt = np.mean(((data - mean) / std) ** 4, axis=0) - 3
        return kurt

# Usage example
processor = DataProcessor()

# Generate sample data
np.random.seed(42)
sample_data = np.random.multivariate_normal(
    mean=[100, 50, 25],
    cov=[[100, 20, 10],
         [20, 50, 15],
         [10, 15, 25]],
    size=1000
)

# Add some outliers
outlier_indices = np.random.choice(1000, size=50, replace=False)
sample_data[outlier_indices] += np.random.normal(0, 50, size=(50, 3))

# Process the data
validated_data = processor.load_and_validate(sample_data)
outliers = processor.outlier_detection(validated_data, method='iqr')
normalized_data = processor.normalize_data(validated_data, method='zscore')
statistics = processor.compute_statistics(validated_data)

print("\nDataset Statistics:")
for stat_name, values in statistics.items():
    if isinstance(values, np.ndarray):
        print(f"{stat_name:>10}: [{', '.join([f'{v:.3f}' for v in values])}]")
    else:
        print(f"{stat_name:>10}: {values}")
```

### Financial Time Series Analysis

```python
def financial_analysis_demo():
    """Demonstrate NumPy for financial data analysis."""
    
    # Simulate stock prices with random walk
    np.random.seed(42)
    days = 252  # One trading year
    initial_price = 100
    daily_returns = np.random.normal(0.0005, 0.02, days)  # Mean return, volatility
    
    # Calculate cumulative prices
    prices = initial_price * np.exp(np.cumsum(daily_returns))
    
    # Technical analysis calculations
    def moving_average(data, window):
        """Calculate moving average using convolution."""
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')
    
    def rsi(prices, period=14):
        """Relative Strength Index calculation."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / avg_losses
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values
    
    def bollinger_bands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        ma = moving_average(prices, period)
        rolling_std = np.array([
            np.std(prices[i:i+period]) 
            for i in range(len(prices) - period + 1)
        ])
        
        upper_band = ma + (std_dev * rolling_std)
        lower_band = ma - (std_dev * rolling_std)
        
        return upper_band, ma, lower_band
    
    # Calculate indicators
    ma_20 = moving_average(prices, 20)
    ma_50 = moving_average(prices, 50)
    rsi_values = rsi(prices)
    bb_upper, bb_middle, bb_lower = bollinger_bands(prices)
    
    # Portfolio analysis
    def portfolio_metrics(returns):
        """Calculate portfolio performance metrics."""
        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility
        
        # Maximum drawdown calculation
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    metrics = portfolio_metrics(daily_returns)
    
    print("Financial Analysis Results:")
    print(f"Final Price: ${prices[-1]:.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Annual Volatility: {metrics['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    
    return prices, metrics

prices, metrics = financial_analysis_demo()
```

---

## Practical Exercises

### Exercise 1: Performance Comparison Workshop (60 minutes)

Build a comprehensive performance testing framework that compares Pure Python, NumPy, and optimized NumPy approaches.

**Requirements**:
- Test multiple operation types (arithmetic, statistical, matrix operations)
- Measure execution time and memory usage
- Generate performance visualizations
- Test different data sizes to show scaling behavior
- Include real-world data processing scenarios

**Implementation Framework**:
```python
class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        self.data_sizes = [1000, 10000, 100000, 1000000]
    
    def benchmark_arithmetic_operations(self):
        """Benchmark basic arithmetic across implementations."""
        pass
    
    def benchmark_statistical_operations(self):
        """Benchmark statistical calculations."""
        pass
    
    def benchmark_matrix_operations(self):
        """Benchmark linear algebra operations."""
        pass
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        pass
```

### Exercise 2: Array Manipulation for Data Analysis (45 minutes)

Create a data analysis toolkit using advanced NumPy operations on a real dataset.

**Requirements**:
- Load multi-dimensional sales data (provided CSV)
- Implement complex indexing and slicing operations
- Use broadcasting for calculations across different dimensions
- Apply statistical operations with proper axis handling
- Create summary reports with insights

**Data Structure**: Sales data with dimensions (months, products, regions, metrics)

### Exercise 3: Statistical Computing with NumPy (45 minutes)

Implement statistical functions from scratch using NumPy's mathematical operations.

**Requirements**:
- Correlation and covariance matrices
- Principal Component Analysis (PCA)
- Linear regression with confidence intervals
- Statistical hypothesis testing
- Bootstrap resampling

**Expected Functions**:
```python
def correlation_matrix(data):
    """Calculate correlation matrix from scratch."""
    pass

def pca_analysis(data, n_components=None):
    """Perform PCA using NumPy linear algebra."""
    pass

def linear_regression(X, y):
    """Implement linear regression with statistics."""
    pass

def bootstrap_confidence_interval(data, statistic, confidence=0.95, n_bootstrap=1000):
    """Calculate bootstrap confidence intervals."""
    pass
```

---

## Wrap-up and Integration (30 minutes)

### Key Performance Principles

1. **Vectorization First**: Always think in terms of array operations, not loops
2. **Memory Awareness**: Choose appropriate data types and pre-allocate arrays
3. **Broadcasting Mastery**: Leverage NumPy's broadcasting for complex operations
4. **Axis Understanding**: Master axis-based operations for multi-dimensional data

### NumPy Best Practices Checklist

```python
# ✓ DO: Vectorized operations
result = data**2 + 2*data

# ✗ DON'T: Element-wise loops
result = np.array([x**2 + 2*x for x in data])

# ✓ DO: Proper memory pre-allocation
result = np.empty_like(data)
np.square(data, out=result)

# ✗ DON'T: Repeated array creation
result = np.array([])
for x in data:
    result = np.append(result, x**2)  # Very slow!

# ✓ DO: Appropriate data types
small_integers = np.array(data, dtype=np.int16)  # If data fits

# ✗ DON'T: Default to largest types
large_integers = np.array(data, dtype=np.int64)  # Wastes memory
```

### Integration with the Data Science Ecosystem

NumPy is the foundation for:
- **Pandas**: All DataFrame operations are built on NumPy arrays
- **Matplotlib**: Plotting functions expect NumPy arrays
- **Scikit-learn**: Machine learning algorithms use NumPy for computation
- **SciPy**: Advanced scientific computing built on NumPy
- **TensorFlow/PyTorch**: Deep learning frameworks use NumPy-compatible arrays

### Preparation for Next Lecture

Next lecture introduces Pandas, which adds labeled, structured data operations on top of NumPy's raw performance. You'll see how:
- NumPy arrays become Pandas Series and DataFrames
- Broadcasting principles apply to data alignment
- Performance optimizations carry forward
- Complex data operations become intuitive

The transition from NumPy to Pandas represents moving from mathematical computing to data analysis - keeping all the performance benefits while gaining usability for real-world data challenges.

### Extended Learning Resources

- **NumPy Documentation**: User guide and API reference
- **Performance**: "Guide to NumPy" by Travis Oliphant
- **Advanced Topics**: SciPy lectures and NumPy tutorials
- **Integration**: How other libraries build on NumPy foundations

NumPy mastery is the difference between writing slow, memory-intensive Python and building high-performance data science applications that can handle real-world scale data.