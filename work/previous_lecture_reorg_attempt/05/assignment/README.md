# Assignment 05: NumPy Environment Setup and Array Operations

**Due:** Before next class  
**Points:** 18 points total  
**Submit:** Via GitHub repository (link submitted to Canvas)

## Overview

Master Python environment management and NumPy fundamentals by setting up a professional data science environment and performing numerical analysis on real-world data. This assignment builds the foundation for all future data analysis work.

## Learning Objectives

By completing this assignment, you will:
- Create and manage virtual environments using conda
- Install and manage data science packages professionally
- Perform efficient numerical computations using NumPy arrays
- Compare NumPy performance against Python lists
- Document environment setup for reproducible analysis
- Apply statistical analysis to real datasets

## Part 1: Environment Setup (5 points)

### Task 1.1: Create Project Environment

1. **Create conda environment:**
   ```bash
   conda create -n datasci217-assignment05 python=3.11
   conda activate datasci217-assignment05
   ```

2. **Install required packages:**
   ```bash
   conda install numpy matplotlib jupyter pytest
   pip install tabulate  # For nice table formatting
   ```

3. **Document your environment:**
   ```bash
   conda env export > environment.yml
   pip freeze > requirements.txt
   ```

### Task 1.2: Environment Documentation

Create `ENVIRONMENT.md` with:
```markdown
# Environment Setup for Assignment 05

## System Information
- Operating System: [Your OS]
- Python Version: [Run `python --version`]
- Conda Version: [Run `conda --version`]

## Package Versions
[Copy output from `conda list numpy matplotlib jupyter pytest`]

## Setup Commands
```bash
conda create -n datasci217-assignment05 python=3.11
conda activate datasci217-assignment05
conda install numpy matplotlib jupyter pytest
pip install tabulate
```

## Verification
[Include output from running `python -c "import numpy; print(numpy.__version__)"`]
```

## Part 2: NumPy Fundamentals (8 points)

Create `src/numpy_analysis.py` with the following functions:

### Function 1: Array Creation and Properties (2 points)

```python
import numpy as np

def analyze_array_properties():
    """
    Create different types of NumPy arrays and analyze their properties.
    
    Returns:
        dict: Dictionary containing analysis results
    """
    results = {}
    
    # Create different arrays
    list_array = np.array([1, 2, 3, 4, 5])
    zeros_array = np.zeros(10)
    ones_array = np.ones((3, 4))
    range_array = np.arange(0, 20, 2)
    random_array = np.random.rand(5, 5)
    
    # Analyze each array
    arrays = {
        'list_array': list_array,
        'zeros_array': zeros_array,
        'ones_array': ones_array,
        'range_array': range_array,
        'random_array': random_array
    }
    
    for name, arr in arrays.items():
        results[name] = {
            'shape': arr.shape,
            'size': arr.size,
            'ndim': arr.ndim,
            'dtype': str(arr.dtype),
            'memory_bytes': arr.nbytes
        }
    
    return results

def print_array_analysis():
    """Print formatted analysis of array properties"""
    results = analyze_array_properties()
    
    print("Array Properties Analysis")
    print("=" * 50)
    for name, props in results.items():
        print(f"\n{name}:")
        for key, value in props.items():
            print(f"  {key}: {value}")
```

### Function 2: Statistical Analysis (3 points)

```python
def analyze_student_grades():
    """
    Perform comprehensive statistical analysis on student grade data.
    
    Returns:
        dict: Statistical analysis results
    """
    # Sample grade data (replace with provided dataset)
    grades = np.array([
        85, 92, 78, 96, 88, 91, 84, 87, 93, 79, 
        90, 86, 82, 95, 89, 77, 94, 83, 88, 92
    ])
    
    # Basic statistics
    stats = {
        'count': grades.size,
        'mean': np.mean(grades),
        'median': np.median(grades),
        'std_dev': np.std(grades),
        'min_grade': np.min(grades),
        'max_grade': np.max(grades),
        'range': np.ptp(grades)  # peak-to-peak (max - min)
    }
    
    # Grade distribution analysis
    stats['grades_above_90'] = np.sum(grades >= 90)
    stats['grades_below_80'] = np.sum(grades < 80)
    stats['passing_rate'] = np.mean(grades >= 70) * 100
    
    # Percentiles
    stats['25th_percentile'] = np.percentile(grades, 25)
    stats['75th_percentile'] = np.percentile(grades, 75)
    stats['90th_percentile'] = np.percentile(grades, 90)
    
    return stats, grades

def print_grade_analysis():
    """Print formatted grade analysis"""
    stats, grades = analyze_student_grades()
    
    print("Student Grade Analysis")
    print("=" * 40)
    print(f"Sample size: {stats['count']} students")
    print(f"Mean grade: {stats['mean']:.1f}")
    print(f"Median grade: {stats['median']:.1f}")
    print(f"Standard deviation: {stats['std_dev']:.1f}")
    print(f"Grade range: {stats['min_grade']:.0f} - {stats['max_grade']:.0f}")
    
    print(f"\nGrade Distribution:")
    print(f"Grades ≥ 90 (A): {stats['grades_above_90']} students")
    print(f"Grades < 80: {stats['grades_below_80']} students")
    print(f"Passing rate (≥ 70): {stats['passing_rate']:.1f}%")
    
    print(f"\nPercentiles:")
    print(f"25th percentile: {stats['25th_percentile']:.1f}")
    print(f"75th percentile: {stats['75th_percentile']:.1f}")
    print(f"90th percentile: {stats['90th_percentile']:.1f}")
```

### Function 3: Array Operations and Filtering (3 points)

```python
def demonstrate_array_operations():
    """
    Demonstrate various NumPy array operations and filtering.
    
    Returns:
        dict: Results of different operations
    """
    # Create sample data
    temperatures = np.array([22.5, 25.1, 19.8, 28.3, 21.7, 26.4, 30.2, 18.9, 23.6, 27.8])
    
    results = {}
    
    # Basic operations
    results['celsius'] = temperatures
    results['fahrenheit'] = temperatures * 9/5 + 32
    results['kelvin'] = temperatures + 273.15
    
    # Mathematical operations
    results['temp_squared'] = temperatures ** 2
    results['temp_sqrt'] = np.sqrt(temperatures)
    results['temp_rounded'] = np.round(temperatures, 1)
    
    # Filtering operations
    results['hot_days'] = temperatures[temperatures > 25]
    results['cold_days'] = temperatures[temperatures < 20]
    results['mild_days'] = temperatures[(temperatures >= 20) & (temperatures <= 25)]
    
    # Boolean arrays
    results['is_hot'] = temperatures > 25
    results['is_cold'] = temperatures < 20
    
    # Aggregations
    results['hot_day_count'] = np.sum(temperatures > 25)
    results['cold_day_count'] = np.sum(temperatures < 20)
    results['avg_hot_temp'] = np.mean(temperatures[temperatures > 25])
    
    return results

def print_temperature_analysis():
    """Print formatted temperature analysis"""
    results = demonstrate_array_operations()
    
    print("Temperature Analysis")
    print("=" * 30)
    print(f"Original temperatures (°C): {results['celsius']}")
    print(f"In Fahrenheit: {results['fahrenheit'].round(1)}")
    print(f"Hot days (>25°C): {results['hot_days']}")
    print(f"Cold days (<20°C): {results['cold_days']}")
    print(f"Number of hot days: {results['hot_day_count']}")
    print(f"Average temperature on hot days: {results['avg_hot_temp']:.1f}°C")
```

## Part 3: Performance Comparison (3 points)

### Function 4: Speed Comparison

```python
import time

def compare_numpy_vs_lists():
    """
    Compare performance of NumPy arrays vs Python lists for various operations.
    
    Returns:
        dict: Timing results for different operations
    """
    # Create large datasets
    size = 100000
    python_list = list(range(size))
    numpy_array = np.arange(size)
    
    results = {}
    
    # Test 1: Sum calculation
    start_time = time.time()
    list_sum = sum(python_list)
    list_sum_time = time.time() - start_time
    
    start_time = time.time()
    array_sum = np.sum(numpy_array)
    array_sum_time = time.time() - start_time
    
    results['sum_operation'] = {
        'list_time': list_sum_time,
        'array_time': array_sum_time,
        'speedup': list_sum_time / array_sum_time
    }
    
    # Test 2: Square calculation
    start_time = time.time()
    list_squares = [x**2 for x in python_list]
    list_square_time = time.time() - start_time
    
    start_time = time.time()
    array_squares = numpy_array ** 2
    array_square_time = time.time() - start_time
    
    results['square_operation'] = {
        'list_time': list_square_time,
        'array_time': array_square_time,
        'speedup': list_square_time / array_square_time
    }
    
    # Test 3: Filtering operation
    threshold = size // 2
    
    start_time = time.time()
    list_filtered = [x for x in python_list if x > threshold]
    list_filter_time = time.time() - start_time
    
    start_time = time.time()
    array_filtered = numpy_array[numpy_array > threshold]
    array_filter_time = time.time() - start_time
    
    results['filter_operation'] = {
        'list_time': list_filter_time,
        'array_time': array_filter_time,
        'speedup': list_filter_time / array_filter_time
    }
    
    return results

def print_performance_comparison():
    """Print formatted performance comparison"""
    results = compare_numpy_vs_lists()
    
    print("Performance Comparison: NumPy vs Python Lists")
    print("=" * 50)
    
    for operation, times in results.items():
        print(f"\n{operation.replace('_', ' ').title()}:")
        print(f"  Python list time: {times['list_time']:.6f} seconds")
        print(f"  NumPy array time: {times['array_time']:.6f} seconds")
        print(f"  NumPy speedup: {times['speedup']:.1f}x faster")
```

## Part 4: Main Script and Testing (2 points)

### Main Execution Block

```python
if __name__ == "__main__":
    print("DataSci 217 - Assignment 05: NumPy Analysis")
    print("=" * 60)
    
    # Run all analyses
    print("\n1. Array Properties Analysis")
    print_array_analysis()
    
    print("\n2. Student Grade Analysis")
    print_grade_analysis()
    
    print("\n3. Temperature Analysis")
    print_temperature_analysis()
    
    print("\n4. Performance Comparison")
    print_performance_comparison()
    
    print("\nAnalysis complete! Check output files for detailed results.")
```

### Testing File

Create `tests/test_numpy_analysis.py`:
```python
import pytest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from numpy_analysis import analyze_array_properties, analyze_student_grades, demonstrate_array_operations, compare_numpy_vs_lists

def test_array_properties():
    """Test array properties analysis"""
    results = analyze_array_properties()
    
    # Check that all expected arrays are analyzed
    expected_arrays = ['list_array', 'zeros_array', 'ones_array', 'range_array', 'random_array']
    for array_name in expected_arrays:
        assert array_name in results
        assert 'shape' in results[array_name]
        assert 'size' in results[array_name]
        assert 'dtype' in results[array_name]

def test_grade_analysis():
    """Test grade analysis functions"""
    stats, grades = analyze_student_grades()
    
    # Verify statistics make sense
    assert stats['mean'] >= 0 and stats['mean'] <= 100
    assert stats['min_grade'] <= stats['mean'] <= stats['max_grade']
    assert stats['count'] == len(grades)
    assert stats['passing_rate'] >= 0 and stats['passing_rate'] <= 100

def test_array_operations():
    """Test array operations"""
    results = demonstrate_array_operations()
    
    # Check required operations exist
    assert 'celsius' in results
    assert 'fahrenheit' in results
    assert 'hot_days' in results
    assert len(results['hot_days']) >= 0  # Could be empty

def test_performance_comparison():
    """Test performance comparison (basic check)"""
    results = compare_numpy_vs_lists()
    
    # Check that timing results exist
    assert 'sum_operation' in results
    assert 'square_operation' in results
    assert 'filter_operation' in results
    
    # Check that all operations have timing data
    for operation in results.values():
        assert 'list_time' in operation
        assert 'array_time' in operation
        assert 'speedup' in operation
```

## Submission Requirements

### File Structure
```
assignment05/
├── README.md
├── ENVIRONMENT.md
├── environment.yml
├── requirements.txt
├── src/
│   └── numpy_analysis.py
├── tests/
│   └── test_numpy_analysis.py
├── output/
│   └── analysis_results.txt
└── .gitignore
```

### .gitignore
```
# Virtual environments
datasci217-assignment05/
.venv/
__pycache__/

# Jupyter notebooks
.ipynb_checkpoints/

# Output files
*.pyc
.pytest_cache/
```

### Running Instructions

Include in README.md:
```markdown
## Running the Analysis

1. **Setup environment:**
   ```bash
   conda env create -f environment.yml
   conda activate datasci217-assignment05
   ```

2. **Run analysis:**
   ```bash
   python src/numpy_analysis.py > output/analysis_results.txt
   ```

3. **Run tests:**
   ```bash
   pytest tests/ -v
   ```
```

## Grading Rubric

- **Environment Setup (5 pts):** Correct conda environment creation, package installation, documentation
- **Array Properties (2 pts):** Correct array creation and property analysis
- **Statistical Analysis (3 pts):** Comprehensive grade analysis with all required statistics
- **Array Operations (3 pts):** Proper filtering, mathematical operations, and boolean indexing
- **Performance Comparison (3 pts):** Valid timing comparison showing NumPy advantages
- **Code Quality (2 pts):** Clean code, good documentation, passing tests

## Common Issues to Avoid

1. **Environment confusion:** Always activate your environment before running code
2. **Array vs scalar:** Remember that NumPy functions often return arrays, not scalars
3. **Boolean indexing:** Use parentheses in complex conditions: `(arr > 5) & (arr < 10)`
4. **Performance testing:** Don't time single operations - use larger datasets for meaningful comparisons
5. **Import statements:** Always import numpy as np at the top of your file

## Getting Help

- **Discord:** #assignment05-help channel
- **Office Hours:** [Schedule]
- **Documentation:** https://numpy.org/doc/stable/user/quickstart.html

Submit your repository link via Canvas by [due date] at 11:59 PM.