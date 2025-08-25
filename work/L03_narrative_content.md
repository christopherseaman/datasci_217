# Lecture 03: File Operations + Jupyter Interactive Development

**Duration**: 3.5 hours  
**Focus**: Professional file handling, command line tools, Jupyter mastery, data loading foundation

## Learning Objectives

By the end of this lecture, students will:
- Master professional file I/O patterns with context managers and error handling
- Build command line tools with flexible argument processing
- Leverage Jupyter's full interactive development ecosystem
- Implement robust data loading pipelines for various file formats
- Understand the workflow from script development to production deployment

---

## Part 1: Professional File Handling (45 minutes)

### Opening Performance Hook: "The Cost of Poor File Handling"

```python
# The WRONG way - fragile, resource-leaking code
def load_data_badly(filename):
    file = open(filename)  # No error handling
    data = file.read()     # No resource cleanup
    return data.split('\n')  # File handle stays open!

# The RIGHT way - robust, professional code
def load_data_professionally(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return []
    except PermissionError:
        print(f"Error: Permission denied for '{filename}'")
        return []
```

**Why This Matters**: In production data science, files can be gigabytes large, come from unreliable sources, or be accessed concurrently. Professional file handling prevents:
- Memory leaks from unclosed files
- Data corruption from encoding issues
- System crashes from unhandled exceptions
- Security vulnerabilities from path traversal

### Context Managers: The Python Way

```python
# Context manager ensures cleanup even if errors occur
with open('data.csv', 'r') as file:
    data = file.read()
    # File automatically closed here, even if exception occurs

# Multiple files with context managers
with open('input.txt', 'r') as infile, open('output.txt', 'w') as outfile:
    processed_data = process(infile.read())
    outfile.write(processed_data)

# Custom context manager for data processing
from contextlib import contextmanager

@contextmanager
def data_processor(filename):
    print(f"Starting processing of {filename}")
    start_time = time.time()
    try:
        with open(filename, 'r') as file:
            yield file
    finally:
        elapsed = time.time() - start_time
        print(f"Processing completed in {elapsed:.2f} seconds")

# Usage
with data_processor('large_dataset.csv') as file:
    # Your processing code here
    pass
```

### Advanced File Operations

```python
import os
import pathlib
from pathlib import Path

# Modern path handling with pathlib
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)  # Create directory if needed

# Robust file discovery
csv_files = list(data_dir.glob('*.csv'))
all_data_files = list(data_dir.rglob('*.{csv,txt,json}'))  # Recursive search

# File metadata and validation
def validate_data_file(filepath):
    path = Path(filepath)
    
    if not path.exists():
        return False, "File does not exist"
    
    if path.stat().st_size == 0:
        return False, "File is empty"
    
    if not path.suffix.lower() in ['.csv', '.txt', '.json']:
        return False, "Unsupported file format"
    
    return True, "File is valid"

# Batch file processing
def process_data_files(directory):
    results = {}
    data_dir = Path(directory)
    
    for filepath in data_dir.glob('*.csv'):
        try:
            with filepath.open('r', encoding='utf-8') as file:
                line_count = sum(1 for line in file)
                results[filepath.name] = line_count
        except Exception as e:
            results[filepath.name] = f"Error: {e}"
    
    return results
```

---

## Part 2: Command Line Integration (50 minutes)

### From Script to Tool: The Transformation

```python
# Basic script - hardcoded and inflexible
def analyze_data():
    with open('data.csv', 'r') as file:
        lines = file.readlines()
        print(f"Total lines: {len(lines)}")

# Professional tool - flexible and reusable
import argparse
import sys

def analyze_data_tool():
    parser = argparse.ArgumentParser(
        description='Analyze data files and generate reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py data.csv
  python analyze.py data.csv --output report.txt --verbose
  python analyze.py *.csv --format json
        """
    )
    
    parser.add_argument('files', nargs='+', help='Data files to analyze')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--format', choices=['text', 'json', 'csv'], 
                       default='text', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    results = {}
    for filename in args.files:
        try:
            results[filename] = analyze_file(filename, verbose=args.verbose)
        except Exception as e:
            if args.verbose:
                print(f"Error processing {filename}: {e}", file=sys.stderr)
    
    # Output results in requested format
    if args.output:
        with open(args.output, 'w') as outfile:
            write_results(results, args.format, outfile)
    else:
        write_results(results, args.format, sys.stdout)

def analyze_file(filename, verbose=False):
    """Analyze a single file and return statistics."""
    path = Path(filename)
    
    if verbose:
        print(f"Processing {filename}...")
    
    stats = {
        'size_bytes': path.stat().st_size,
        'line_count': 0,
        'word_count': 0,
        'encoding': 'utf-8'  # Could detect automatically
    }
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                stats['line_count'] += 1
                stats['word_count'] += len(line.split())
    except UnicodeDecodeError:
        # Try different encoding
        with open(filename, 'r', encoding='latin-1') as file:
            stats['encoding'] = 'latin-1'
            # Repeat counting logic
    
    return stats

if __name__ == '__main__':
    analyze_data_tool()
```

### Environment Integration

```python
import os
import sys
from pathlib import Path

# Environment-aware configuration
def get_data_directory():
    """Get data directory from environment or default."""
    return Path(os.getenv('DATA_DIR', './data'))

def get_config():
    """Load configuration from multiple sources."""
    config = {
        'data_dir': get_data_directory(),
        'output_dir': Path(os.getenv('OUTPUT_DIR', './results')),
        'max_file_size': int(os.getenv('MAX_FILE_SIZE', '100000000')),  # 100MB
        'supported_formats': ['csv', 'tsv', 'txt', 'json']
    }
    
    # Override with command line config file if provided
    config_file = Path('config.json')
    if config_file.exists():
        import json
        with config_file.open() as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config

# Cross-platform compatibility
def ensure_directory(path):
    """Create directory if it doesn't exist, cross-platform."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
```

---

## Part 3: Jupyter Interactive Development Mastery (90 minutes)

### Beyond Basic Notebooks: The Professional Jupyter Workflow

#### IPython Magic Commands - Your Secret Weapons

```python
# In Jupyter cell:

# Time your code execution
%timeit sum(range(1000))
%%timeit
# Multi-line timing
data = list(range(1000))
result = sum(data)

# Profile your code
%prun expensive_function()

# Debug interactively
%debug  # After an exception occurs

# Load external scripts
%load external_script.py

# Run external Python files
%run analysis_script.py

# System commands
!ls -la data/
!pip install pandas

# Environment variables
%env DATA_DIR=/path/to/data

# Auto-reload modules during development
%load_ext autoreload
%autoreload 2
```

#### Advanced Jupyter Features

```python
# Interactive widgets for parameter exploration
import ipywidgets as widgets
from IPython.display import display

@widgets.interact(
    n_samples=(100, 10000, 100),
    noise_level=(0.0, 1.0, 0.1)
)
def plot_synthetic_data(n_samples=1000, noise_level=0.2):
    # Your plotting code here
    pass

# Rich display capabilities
from IPython.display import HTML, Markdown, Image, display

display(HTML('<h3>Analysis Results</h3>'))
display(Markdown('## Summary Statistics'))

# Progress bars for long operations
from tqdm.notebook import tqdm
import time

for i in tqdm(range(100), desc="Processing files"):
    time.sleep(0.01)  # Simulate work

# Multi-output cells
from IPython.display import display
import matplotlib.pyplot as plt

def analyze_dataset(data):
    # Display summary statistics
    display(data.describe())
    
    # Display correlation heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(data.corr(), cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.show()
    
    # Display missing data report
    missing_data = data.isnull().sum()
    if missing_data.any():
        display(Markdown('### Missing Data Report'))
        display(missing_data[missing_data > 0])
```

#### Jupyter Project Structure

```
project/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   └── analyzer.py
├── tests/
│   └── test_data_processing.py
├── data/
│   ├── raw/
│   └── processed/
└── results/
    ├── figures/
    └── reports/
```

#### Development Workflow Integration

```python
# In notebook: Load and test your modules
import sys
sys.path.append('../src')

from data_loader import load_dataset
from preprocessor import clean_data

# Develop functions interactively in notebook
def explore_data(df):
    """Interactive data exploration function."""
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    return df.info()

# Test the function
data = load_dataset('data/sample.csv')
explore_data(data)

# When satisfied, move to module file
%%writefile ../src/explorer.py
def explore_data(df):
    """Interactive data exploration function."""
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    return df.info()
```

---

## Part 4: Data Loading Foundation (75 minutes)

### Professional Data Loading Patterns

```python
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

class DataLoader:
    """Professional data loading with validation and error handling."""
    
    def __init__(self, data_dir='data', encoding='utf-8'):
        self.data_dir = Path(data_dir)
        self.encoding = encoding
        self.load_stats = {}
    
    def load_csv(self, filename, **kwargs):
        """Load CSV with intelligent defaults and error handling."""
        filepath = self.data_dir / filename
        
        # Default parameters for robust loading
        default_params = {
            'encoding': self.encoding,
            'low_memory': False,  # Prevent mixed type warnings
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'missing'],
            'keep_default_na': True,
        }
        default_params.update(kwargs)
        
        try:
            # Try loading with defaults
            df = pd.read_csv(filepath, **default_params)
            
            # Store loading statistics
            self.load_stats[filename] = {
                'shape': df.shape,
                'memory_mb': df.memory_usage().sum() / 1024**2,
                'null_columns': df.isnull().any().sum(),
                'dtypes': dict(df.dtypes)
            }
            
            print(f"✓ Loaded {filename}: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return df
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    default_params['encoding'] = encoding
                    df = pd.read_csv(filepath, **default_params)
                    print(f"✓ Loaded {filename} with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"Could not decode {filename} with common encodings")
        
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return None
        
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
            return None
    
    def quick_explore(self, df, name="dataset"):
        """Quick exploration of loaded data."""
        if df is None:
            return
        
        print(f"\n=== Quick Exploration: {name} ===")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
        
        # Data types summary
        dtype_counts = df.dtypes.value_counts()
        print(f"Data types: {dtype_counts.to_dict()}")
        
        # Missing data summary
        missing = df.isnull().sum()
        if missing.any():
            print(f"Missing data in {missing.sum()} columns:")
            for col, count in missing[missing > 0].items():
                pct = count / len(df) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print("No missing data")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric columns ({len(numeric_cols)}):")
            print(df[numeric_cols].describe().round(2))
        
        return df

# Usage example
loader = DataLoader('data')
df = loader.load_csv('sales_data.csv')
loader.quick_explore(df, "sales_data")
```

### Handling Different File Formats

```python
class UniversalDataLoader:
    """Load data from multiple formats with consistent interface."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
    
    def load_file(self, filename, **kwargs):
        """Automatically detect format and load appropriately."""
        filepath = self.data_dir / filename
        suffix = filepath.suffix.lower()
        
        loaders = {
            '.csv': self._load_csv,
            '.tsv': self._load_tsv,
            '.txt': self._load_text,
            '.json': self._load_json,
            '.xlsx': self._load_excel,
            '.parquet': self._load_parquet
        }
        
        if suffix not in loaders:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        return loaders[suffix](filepath, **kwargs)
    
    def _load_csv(self, filepath, **kwargs):
        return pd.read_csv(filepath, **kwargs)
    
    def _load_tsv(self, filepath, **kwargs):
        kwargs.setdefault('sep', '\t')
        return pd.read_csv(filepath, **kwargs)
    
    def _load_text(self, filepath, **kwargs):
        """Load text file as single column DataFrame."""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({'text': lines})
    
    def _load_json(self, filepath, **kwargs):
        return pd.read_json(filepath, **kwargs)
    
    def _load_excel(self, filepath, **kwargs):
        return pd.read_excel(filepath, **kwargs)
    
    def _load_parquet(self, filepath, **kwargs):
        return pd.read_parquet(filepath, **kwargs)

# Data validation pipeline
def validate_dataset(df, required_columns=None, min_rows=1):
    """Validate dataset meets basic requirements."""
    issues = []
    
    if df is None or df.empty:
        issues.append("Dataset is empty")
        return issues
    
    if len(df) < min_rows:
        issues.append(f"Dataset has {len(df)} rows, minimum required: {min_rows}")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        issues.append(f"Completely empty columns: {empty_cols}")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows")
    
    return issues
```

---

## Practical Exercises

### Exercise 1: File Processing Pipeline Tool (45 minutes)

Build a command-line tool that processes multiple data files and generates a comprehensive report.

**Requirements**:
- Accept multiple file paths as arguments
- Support CSV, TSV, and text files
- Generate summary statistics for each file
- Output results in multiple formats (text, JSON, HTML)
- Handle errors gracefully and continue processing
- Include progress indicators for large batches

**Starter Code Structure**:
```python
# file_processor.py
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class FileStats:
    filename: str
    size_bytes: int
    line_count: int
    column_count: int
    encoding: str
    errors: List[str]

class FileProcessor:
    def __init__(self, output_format='text'):
        self.output_format = output_format
        self.results = []
    
    def process_file(self, filepath: Path) -> FileStats:
        # Your implementation here
        pass
    
    def process_batch(self, filepaths: List[Path]) -> List[FileStats]:
        # Your implementation here
        pass
    
    def generate_report(self, stats: List[FileStats]) -> str:
        # Your implementation here
        pass

def main():
    # Command line interface implementation
    pass

if __name__ == '__main__':
    main()
```

### Exercise 2: Jupyter Interactive Data Exploration (60 minutes)

Create a Jupyter notebook that demonstrates advanced interactive data exploration techniques.

**Requirements**:
- Load a real dataset (provided sample sales data)
- Use widgets for interactive parameter adjustment
- Implement custom magic commands for repetitive tasks
- Create reusable exploration functions
- Generate automated reports with rich output
- Handle missing data interactively

**Notebook Structure**:
1. **Setup and Configuration** - Import libraries, set display options
2. **Data Loading** - With error handling and validation
3. **Interactive Exploration** - Widgets for filtering and visualization
4. **Custom Analysis Functions** - Reusable code blocks
5. **Automated Reporting** - Generate summary reports
6. **Export Results** - Save processed data and figures

### Exercise 3: Data Inspector Command Line Utility (30 minutes)

Build a professional data inspection tool similar to `ls` but for data files.

**Requirements**:
- Display file metadata (size, format, encoding)
- Show data preview (first few rows)
- Calculate basic statistics
- Identify data quality issues
- Support multiple output formats
- Work with various file types

**Expected Output**:
```bash
$ python data_inspector.py data/*.csv

sales_data.csv          [CSV, 1.2MB, UTF-8]
├── Shape: 15,423 × 12
├── Memory: 1.2MB
├── Missing: 3 columns affected
├── Types: int64(5), float64(4), object(3)
└── Preview: customer_id,product_name,sale_date...

customers.csv           [CSV, 856KB, UTF-8]
├── Shape: 8,917 × 8
├── Memory: 856KB
├── Missing: No missing data
├── Types: int64(2), float64(2), object(4)
└── Preview: id,name,email,registration_date...
```

---

## Wrap-up and Next Steps (20 minutes)

### Key Takeaways

1. **Professional file handling** prevents production failures and security issues
2. **Command line tools** make your analysis scripts reusable and shareable
3. **Jupyter mastery** accelerates interactive development and exploration
4. **Robust data loading** handles real-world data quality issues automatically

### Preparation for Next Lecture

- Install NumPy and familiarize yourself with array concepts
- Review linear algebra basics (matrix operations, dot products)
- Think about performance: when might you need to process millions of data points?

### Extended Learning Resources

- **File Handling**: Python's `pathlib` documentation
- **Command Line**: `argparse` tutorial and `click` library
- **Jupyter**: JupyterLab user guide and widget documentation
- **Data Loading**: Pandas I/O documentation and `pyjanitor` library

The next lecture will demonstrate why NumPy is the foundation of the entire Python data science ecosystem, with dramatic performance improvements that will change how you think about data processing.