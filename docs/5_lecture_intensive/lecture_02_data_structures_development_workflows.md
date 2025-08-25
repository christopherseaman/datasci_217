# Lecture 02: Data Structures & Development Workflows

*Advanced Python Programming and Professional Development Practices*

## Learning Objectives

By the end of this lecture, you will be able to:
- Master Python's core data structures and their appropriate use cases
- Implement advanced programming patterns like list comprehensions
- Set up and manage remote development environments
- Create and manage Python modules and packages
- Work with command-line arguments and environment variables
- Apply professional development workflows and best practices

## Introduction: Building on the Foundation

In our first lecture, we established the fundamentals of command line navigation and basic Python programming. Today, we'll elevate your skills to a professional level by diving deep into Python's powerful data structures and establishing robust development workflows.

Modern data science requires more than just knowing how to write code - it demands understanding how to organize, structure, and deploy that code effectively. We'll explore advanced Python features that make your code more efficient and readable, and establish workflows that will serve you throughout your career.

## Part 1: Advanced Python Data Structures

### Lists: Beyond the Basics

**Advanced List Operations:**
```python
# Creating lists with various methods
numbers = list(range(1, 11))          # [1, 2, 3, ..., 10]
repeated = [0] * 5                    # [0, 0, 0, 0, 0]
combined = [1, 2] + [3, 4]            # [1, 2, 3, 4]

# Advanced slicing
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
every_second = data[::2]              # [0, 2, 4, 6, 8]
reversed_data = data[::-1]            # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
middle_part = data[3:7]               # [3, 4, 5, 6]

# List methods for data manipulation
scores = [85, 92, 78, 95, 88]
scores.sort()                         # Sort in place
sorted_scores = sorted(scores, reverse=True)  # Create new sorted list

# Finding elements
max_score = max(scores)
min_score = min(scores)
average = sum(scores) / len(scores)

# List operations with conditions
high_scores = [score for score in scores if score >= 90]
```

**Working with Nested Lists:**
```python
# 2D lists (matrices)
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Accessing elements
first_row = matrix[0]                 # [1, 2, 3]
center_element = matrix[1][1]         # 5

# Iterating through 2D lists
for row in matrix:
    for element in row:
        print(element, end=' ')
    print()  # New line after each row

# List comprehension for 2D lists
flattened = [element for row in matrix for element in row]
# Result: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### List Comprehensions: Pythonic Data Processing

List comprehensions provide a concise way to create lists and are fundamental to writing Pythonic code:

```python
# Basic syntax: [expression for item in iterable]
squares = [x**2 for x in range(10)]
# Equivalent to:
squares = []
for x in range(10):
    squares.append(x**2)

# With conditions: [expression for item in iterable if condition]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Processing strings
words = ['hello', 'world', 'python', 'data', 'science']
capitalized = [word.capitalize() for word in words]
long_words = [word for word in words if len(word) > 5]

# Nested list comprehensions
matrix = [[i + j for j in range(3)] for i in range(3)]
# Result: [[0, 1, 2], [1, 2, 3], [2, 3, 4]]

# Real-world example: processing data
raw_data = ['  Alice,25,Engineer  ', '  Bob,30,Designer  ', '  Charlie,35,Manager  ']
cleaned_data = [
    {
        'name': parts[0].strip(),
        'age': int(parts[1].strip()),
        'job': parts[2].strip()
    }
    for line in raw_data
    for parts in [line.strip().split(',')]
]
```

### Dictionaries: Advanced Patterns

**Dictionary Comprehensions:**
```python
# Basic dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
# Result: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# From two lists
keys = ['name', 'age', 'job']
values = ['Alice', 25, 'Engineer']
person = {k: v for k, v in zip(keys, values)}

# Filtering dictionaries
scores = {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'Diana': 95}
high_performers = {name: score for name, score in scores.items() if score >= 90}
```

**Advanced Dictionary Operations:**
```python
# Merging dictionaries (Python 3.9+)
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
merged = dict1 | dict2

# For older Python versions
merged = {**dict1, **dict2}

# Default values with get()
user_settings = {'theme': 'dark', 'language': 'en'}
theme = user_settings.get('theme', 'light')        # Returns 'dark'
font_size = user_settings.get('font_size', 12)     # Returns 12 (default)

# Using defaultdict for automatic default values
from collections import defaultdict

# Counting items
word_count = defaultdict(int)
text = "the quick brown fox jumps over the lazy dog"
for word in text.split():
    word_count[word] += 1

# Grouping data
from collections import defaultdict
data = [
    {'name': 'Alice', 'department': 'Engineering'},
    {'name': 'Bob', 'department': 'Marketing'},
    {'name': 'Charlie', 'department': 'Engineering'},
]

by_department = defaultdict(list)
for person in data:
    by_department[person['department']].append(person['name'])
```

### Sets: Efficient Collection Operations

Sets are perfect for membership testing, removing duplicates, and mathematical set operations:

```python
# Creating sets
unique_numbers = {1, 2, 3, 4, 5}
from_list = set([1, 1, 2, 2, 3, 3])     # {1, 2, 3}

# Set operations for data analysis
customers_2022 = {'Alice', 'Bob', 'Charlie', 'Diana'}
customers_2023 = {'Bob', 'Charlie', 'Eve', 'Frank'}

# Who are our returning customers?
returning = customers_2022 & customers_2023    # {'Bob', 'Charlie'}

# Who are all our customers?
all_customers = customers_2022 | customers_2023

# Who are new customers in 2023?
new_customers = customers_2023 - customers_2022  # {'Eve', 'Frank'}

# Who stopped being customers?
lost_customers = customers_2022 - customers_2023  # {'Alice', 'Diana'}

# Fast membership testing
def is_valid_user_id(user_id, valid_ids):
    """Check if user_id is valid - O(1) operation with set"""
    return user_id in valid_ids  # Much faster than list lookup for large datasets
```

### Advanced String Operations

**String Methods for Data Processing:**
```python
# Cleaning and formatting data
messy_data = "  ALICE SMITH  |  Data Scientist  |  alice@email.com  "

# Chain string methods
cleaned = messy_data.strip().lower().replace('|', ',')
parts = [part.strip() for part in cleaned.split(',')]

# String formatting techniques
name = "Alice"
age = 30
salary = 75000.50

# f-strings (recommended, Python 3.6+)
message = f"Hello, {name}! You are {age} years old and earn ${salary:,.2f}"

# Format method
message = "Hello, {}! You are {} years old and earn ${:,.2f}".format(name, age, salary)

# String validation
email = "alice@example.com"
is_email = '@' in email and '.' in email.split('@')[1]

# Regular expressions for complex patterns
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def extract_phone_numbers(text):
    pattern = r'\b\d{3}-\d{3}-\d{4}\b'
    return re.findall(pattern, text)
```

### Working with Files: Advanced Patterns

**Processing Large Files:**
```python
def process_large_file(filename):
    """Process large files line by line to avoid memory issues"""
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            # Process line without loading entire file into memory
            processed_line = line.strip().upper()
            if line_number % 1000 == 0:
                print(f"Processed {line_number} lines")

# Working with different file formats
import json
import csv

def load_config(filename):
    """Load configuration from JSON file"""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Configuration file {filename} not found")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in {filename}")
        return {}

def process_csv_data(filename):
    """Process CSV data with proper error handling"""
    data = []
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row_num, row in enumerate(reader, 1):
                try:
                    # Validate and clean data
                    if 'age' in row:
                        row['age'] = int(row['age'])
                    data.append(row)
                except ValueError as e:
                    print(f"Skipping row {row_num}: {e}")
        return data
    except FileNotFoundError:
        print(f"Data file {filename} not found")
        return []
```

## Part 2: Functions and Modules

### Advanced Function Patterns

**Function Documentation:**
```python
def calculate_statistics(data, exclude_outliers=True, outlier_threshold=2.0):
    """
    Calculate descriptive statistics for numerical data.
    
    Args:
        data (list): List of numerical values
        exclude_outliers (bool): Whether to exclude outliers from calculation
        outlier_threshold (float): Standard deviations from mean to consider outlier
        
    Returns:
        dict: Dictionary containing mean, median, std, min, max
        
    Raises:
        ValueError: If data is empty or contains non-numerical values
        
    Examples:
        >>> data = [1, 2, 3, 4, 5, 100]  # 100 is outlier
        >>> stats = calculate_statistics(data, exclude_outliers=True)
        >>> stats['mean']  # Will exclude the outlier
        3.0
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Convert to float and validate
    try:
        numeric_data = [float(x) for x in data]
    except (ValueError, TypeError):
        raise ValueError("All data must be numerical")
    
    if exclude_outliers and len(numeric_data) > 2:
        mean = sum(numeric_data) / len(numeric_data)
        std = (sum((x - mean) ** 2 for x in numeric_data) / len(numeric_data)) ** 0.5
        numeric_data = [x for x in numeric_data 
                       if abs(x - mean) <= outlier_threshold * std]
    
    if not numeric_data:
        raise ValueError("No valid data points after outlier removal")
    
    sorted_data = sorted(numeric_data)
    n = len(sorted_data)
    
    return {
        'count': n,
        'mean': sum(sorted_data) / n,
        'median': sorted_data[n // 2] if n % 2 == 1 else 
                 (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2,
        'min': min(sorted_data),
        'max': max(sorted_data),
        'std': (sum((x - sum(sorted_data) / n) ** 2 for x in sorted_data) / n) ** 0.5
    }
```

**Lambda Functions and Functional Programming:**
```python
# Lambda functions for simple operations
square = lambda x: x**2
add = lambda x, y: x + y

# Common use with built-in functions
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Transform data
squares = list(map(lambda x: x**2, numbers))

# Sort complex data
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78}
]
by_grade = sorted(students, key=lambda student: student['grade'], reverse=True)

# Real-world example: data processing pipeline
def process_sales_data(sales_data):
    """Process sales data using functional programming concepts"""
    # Filter valid sales (amount > 0)
    valid_sales = filter(lambda sale: sale.get('amount', 0) > 0, sales_data)
    
    # Convert to standardized format
    standardized = map(
        lambda sale: {
            'date': sale['date'],
            'amount': float(sale['amount']),
            'category': sale['category'].lower().strip()
        },
        valid_sales
    )
    
    return list(standardized)
```

### Creating and Using Modules

**Module Structure:**
```python
# utils/data_processing.py
"""
Data processing utilities module.

This module provides functions for common data processing tasks.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Constants
DEFAULT_CHUNK_SIZE = 1000
SUPPORTED_FORMATS = ['csv', 'json', 'txt']

def chunk_list(data, chunk_size=DEFAULT_CHUNK_SIZE):
    """Split list into smaller chunks."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def validate_data_format(filename):
    """Validate if file format is supported."""
    extension = filename.split('.')[-1].lower()
    return extension in SUPPORTED_FORMATS

class DataProcessor:
    """A class for processing data with configurable options."""
    
    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.processed_count = 0
    
    def process_file(self, filename):
        """Process a data file."""
        if not validate_data_format(filename):
            raise ValueError(f"Unsupported format. Supported: {SUPPORTED_FORMATS}")
        
        # Processing logic here
        self.processed_count += 1
        return f"Processed {filename}"

# If run as script
if __name__ == "__main__":
    # Test the module
    processor = DataProcessor()
    print(f"Data processor version {__version__}")
```

**Using the Module:**
```python
# In another file
from utils.data_processing import DataProcessor, chunk_list, validate_data_format
from utils import data_processing  # Import entire module

# Use the functions
processor = DataProcessor(chunk_size=500)
large_list = list(range(10000))
for chunk in chunk_list(large_list, 1000):
    print(f"Processing chunk of {len(chunk)} items")
```

**Package Structure:**
```
my_data_project/
├── main.py
├── requirements.txt
├── utils/
│   ├── __init__.py          # Makes utils a package
│   ├── data_processing.py
│   ├── visualization.py
│   └── statistics.py
└── tests/
    ├── __init__.py
    ├── test_data_processing.py
    └── test_statistics.py
```

**Package __init__.py:**
```python
# utils/__init__.py
"""
Data analysis utilities package.
"""

from .data_processing import DataProcessor, chunk_list
from .statistics import calculate_statistics
from .visualization import create_histogram

__version__ = "1.0.0"
__all__ = ['DataProcessor', 'chunk_list', 'calculate_statistics', 'create_histogram']
```

## Part 3: Command Line Arguments and Environment Variables

### Using argparse for Command Line Arguments

**Basic Argument Parsing:**
```python
#!/usr/bin/env python3
"""
Data analysis script with command line interface.
"""

import argparse
import sys
import os

def create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Analyze data files and generate reports',
        epilog='Example: python analyze.py data.csv --output report.txt --verbose'
    )
    
    # Positional argument (required)
    parser.add_argument(
        'input_file',
        help='Path to the input data file'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        default='output.txt',
        help='Output file path (default: output.txt)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'json', 'txt'],
        default='csv',
        help='Input file format (default: csv)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='Statistical threshold (default: 0.05)'
    )
    
    parser.add_argument(
        '--columns',
        nargs='+',
        help='Columns to analyze (space-separated)'
    )
    
    return parser

def main():
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Use arguments
    if args.verbose:
        print(f"Processing file: {args.input_file}")
        print(f"Output format: {args.format}")
        print(f"Threshold: {args.threshold}")
        if args.columns:
            print(f"Analyzing columns: {', '.join(args.columns)}")
    
    # Your analysis code here
    analyze_data(args.input_file, args.output, args.format, 
                args.threshold, args.columns, args.verbose)

def analyze_data(input_file, output_file, file_format, threshold, columns, verbose):
    """Perform the actual data analysis."""
    # Implementation here
    if verbose:
        print(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
```

**Advanced Argument Patterns:**
```python
import argparse
from pathlib import Path

def advanced_parser():
    """Advanced argument parsing examples."""
    parser = argparse.ArgumentParser()
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data processing command
    process_parser = subparsers.add_parser('process', help='Process data files')
    process_parser.add_argument('files', nargs='+', help='Files to process')
    process_parser.add_argument('--clean', action='store_true', help='Clean data')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze processed data')
    analyze_parser.add_argument('--model', choices=['linear', 'logistic'], required=True)
    analyze_parser.add_argument('--output-dir', type=Path, default=Path('results'))
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--template', default='default.html')
    
    return parser

# Usage: python script.py process file1.csv file2.csv --clean
# Usage: python script.py analyze --model linear --output-dir /path/to/results
```

### Environment Variables and Configuration

**Reading Environment Variables:**
```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration class
class Config:
    """Application configuration from environment variables."""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///default.db')
        self.api_key = os.getenv('API_KEY')
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.data_dir = Path(os.getenv('DATA_DIR', './data'))
        self.max_workers = int(os.getenv('MAX_WORKERS', '4'))
        
        # Validate required variables
        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")
    
    def __repr__(self):
        # Don't expose sensitive information
        return f"Config(debug={self.debug}, log_level={self.log_level})"

# Usage
config = Config()
print(f"Using configuration: {config}")
```

**Environment-Specific Configuration:**
```python
# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    pool_size: int = 5
    echo: bool = False

@dataclass
class AppConfig:
    """Application configuration."""
    environment: str
    debug: bool
    secret_key: str
    database: DatabaseConfig
    api_timeout: int = 30

def load_config() -> AppConfig:
    """Load configuration based on environment."""
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'development':
        return AppConfig(
            environment=env,
            debug=True,
            secret_key=os.getenv('SECRET_KEY', 'dev-secret-key'),
            database=DatabaseConfig(
                url=os.getenv('DATABASE_URL', 'sqlite:///dev.db'),
                echo=True
            )
        )
    elif env == 'production':
        # Require all critical settings in production
        secret_key = os.getenv('SECRET_KEY')
        if not secret_key:
            raise ValueError("SECRET_KEY is required in production")
        
        return AppConfig(
            environment=env,
            debug=False,
            secret_key=secret_key,
            database=DatabaseConfig(
                url=os.getenv('DATABASE_URL', 'postgresql://...'),
                pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
                echo=False
            )
        )
    else:
        raise ValueError(f"Unknown environment: {env}")
```

## Part 4: Remote Development Workflows

### SSH and Remote Access

**Setting up SSH:**
```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add public key to remote server
ssh-copy-id user@remote-server.com

# Connect to remote server
ssh user@remote-server.com

# SSH with port forwarding (for Jupyter notebooks)
ssh -L 8888:localhost:8888 user@remote-server.com

# SSH configuration file (~/.ssh/config)
Host myserver
    HostName remote-server.com
    User myusername
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    
Host gpu-server
    HostName gpu.university.edu
    User student
    ProxyJump myserver  # Use myserver as jump host
```

**Secure File Transfer:**
```bash
# Copy files to remote server
scp file.txt user@server:/path/to/destination/

# Copy directories
scp -r project_folder/ user@server:/path/to/destination/

# Copy from remote to local
scp user@server:/path/to/file.txt ./local_destination/

# Sync directories with rsync (more efficient)
rsync -avz --progress project_folder/ user@server:/path/to/destination/
```

### Remote Jupyter Notebooks

**Setting up Remote Jupyter:**
```bash
# On remote server: install and configure Jupyter
pip install jupyter notebook jupyterlab

# Generate Jupyter configuration
jupyter notebook --generate-config

# Set password (optional but recommended)
jupyter notebook password

# Run Jupyter on remote server
jupyter lab --no-browser --port=8888 --ip=0.0.0.0

# Or create a script for easy startup
# start_jupyter.sh
#!/bin/bash
export PYTHONPATH=/path/to/your/project:$PYTHONPATH
cd /path/to/your/project
jupyter lab --no-browser --port=8888 --ip=0.0.0.0
```

**Local Connection:**
```bash
# Connect with port forwarding
ssh -L 8888:localhost:8888 user@remote-server

# Then open browser to http://localhost:8888
```

### VS Code Remote Development

**Remote-SSH Extension:**
1. Install "Remote - SSH" extension in VS Code
2. Press `Ctrl+Shift+P` and run "Remote-SSH: Connect to Host"
3. Enter connection details or select from SSH config

**Development Workflow:**
```bash
# On remote server, set up development environment
python3 -m venv project_env
source project_env/bin/activate
pip install -r requirements.txt

# Install VS Code server (automatic via Remote-SSH extension)
# Work directly on remote files through VS Code interface
```

### Session Management with tmux

**Basic tmux workflow:**
```bash
# Start new tmux session
tmux new-session -s data_project

# Detach from session (keeps running)
# Press Ctrl+b, then d

# List sessions
tmux list-sessions

# Reattach to session
tmux attach-session -t data_project

# Create windows within session
# Ctrl+b, then c (new window)
# Ctrl+b, then n (next window)
# Ctrl+b, then p (previous window)

# Split panes
# Ctrl+b, then % (vertical split)
# Ctrl+b, then " (horizontal split)
```

**tmux Configuration (~/.tmux.conf):**
```bash
# Enable mouse support
set -g mouse on

# Start window numbering at 1
set -g base-index 1
set -g pane-base-index 1

# Reload config
bind r source-file ~/.tmux.conf \; display-message "Config reloaded!"

# Easy pane switching
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R
```

### Docker for Development Environments

**Basic Dockerfile for Data Science:**
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

**Docker Compose for Complete Environment:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  notebook:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./data:/app/data
      - ./notebooks:/app/notebooks
    environment:
      - PYTHONPATH=/app
    
  database:
    image: postgres:13
    environment:
      POSTGRES_DB: research
      POSTGRES_USER: scientist
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

## Part 5: Professional Development Practices

### Code Organization and Project Structure

**Standard Project Layout:**
```
data_science_project/
├── README.md                 # Project overview and setup instructions
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation script
├── .gitignore               # Git ignore patterns
├── .env.example             # Example environment variables
├── Makefile                 # Common tasks automation
├── docker-compose.yml       # Container orchestration
├── docs/                    # Documentation
│   ├── api.md
│   └── deployment.md
├── src/                     # Source code
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── data/               # Data processing modules
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   └── processors.py
│   ├── models/             # Model definitions
│   │   ├── __init__.py
│   │   └── predictors.py
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── helpers.py
├── tests/                  # Test files
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
├── notebooks/              # Jupyter notebooks
│   ├── 01_exploration.ipynb
│   └── 02_modeling.ipynb
├── data/                   # Data files
│   ├── raw/
│   ├── processed/
│   └── external/
└── results/               # Output files
    ├── figures/
    └── models/
```

**Makefile for Task Automation:**
```makefile
# Makefile
.PHONY: install test lint format clean docker-build docker-run

# Install dependencies
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	python -m pytest tests/ -v

# Lint code
lint:
	flake8 src/ tests/
	black --check src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/

# Docker commands
docker-build:
	docker build -t data-project .

docker-run:
	docker-compose up

# Setup development environment
setup-dev: install
	pre-commit install
```

### Error Handling and Logging

**Professional Error Handling:**
```python
import logging
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('application.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

class DataProcessor:
    """Professional data processor with comprehensive error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def process_file(self, filepath: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Process a data file with comprehensive error handling.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Processed data or None if processing failed
            
        Raises:
            DataProcessingError: If processing fails due to data issues
        """
        try:
            self.logger.info(f"Starting processing of {filepath}")
            
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            if filepath.stat().st_size == 0:
                self.logger.warning(f"File {filepath} is empty")
                return []
            
            # Process the file
            data = self._load_data(filepath)
            validated_data = self._validate_data(data)
            processed_data = self._transform_data(validated_data)
            
            self.logger.info(f"Successfully processed {len(processed_data)} records")
            return processed_data
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except PermissionError as e:
            self.logger.error(f"Permission denied: {e}")
            raise
        except DataProcessingError as e:
            self.logger.error(f"Data processing error: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error processing {filepath}")
            raise DataProcessingError(f"Failed to process {filepath}: {str(e)}") from e
    
    def _load_data(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load data from file."""
        # Implementation details...
        pass
    
    def _validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate data format and content."""
        validated = []
        for i, record in enumerate(data):
            try:
                # Validation logic
                if self._is_valid_record(record):
                    validated.append(record)
                else:
                    self.logger.warning(f"Skipping invalid record at line {i + 1}")
            except Exception as e:
                self.logger.warning(f"Error validating record {i + 1}: {e}")
        
        return validated
    
    def _is_valid_record(self, record: Dict[str, Any]) -> bool:
        """Check if record is valid."""
        # Validation logic
        return True
    
    def _transform_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform data according to business rules."""
        # Transformation logic
        return data
```

### Testing Strategies

**Unit Testing with pytest:**
```python
# tests/test_data_processor.py
import pytest
import tempfile
import json
from pathlib import Path
from src.data.processors import DataProcessor, DataProcessingError

class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create DataProcessor instance for testing."""
        config = {'chunk_size': 100, 'validate': True}
        return DataProcessor(config)
    
    @pytest.fixture
    def sample_data_file(self):
        """Create temporary file with sample data."""
        data = [
            {'id': 1, 'name': 'Alice', 'age': 25},
            {'id': 2, 'name': 'Bob', 'age': 30},
            {'id': 3, 'name': 'Charlie', 'age': 35}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return Path(f.name)
    
    def test_process_valid_file(self, processor, sample_data_file):
        """Test processing of valid data file."""
        result = processor.process_file(sample_data_file)
        assert result is not None
        assert len(result) == 3
        assert result[0]['name'] == 'Alice'
    
    def test_process_nonexistent_file(self, processor):
        """Test handling of non-existent file."""
        nonexistent_file = Path('/path/to/nonexistent/file.json')
        with pytest.raises(FileNotFoundError):
            processor.process_file(nonexistent_file)
    
    def test_process_empty_file(self, processor):
        """Test handling of empty file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            empty_file = Path(f.name)
        
        result = processor.process_file(empty_file)
        assert result == []
    
    @pytest.mark.parametrize("invalid_data", [
        [{'id': 'invalid', 'name': 'Alice'}],  # Invalid ID
        [{'name': 'Bob'}],                     # Missing ID
        [{'id': 1, 'name': '', 'age': -5}]    # Invalid age
    ])
    def test_validate_data_with_invalid_records(self, processor, invalid_data):
        """Test data validation with various invalid records."""
        # Create temporary file with invalid data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            invalid_file = Path(f.name)
        
        # Should handle gracefully, not crash
        result = processor.process_file(invalid_file)
        assert isinstance(result, list)
    
    def teardown_method(self):
        """Clean up temporary files after each test."""
        # Clean up any temporary files created during tests
        pass
```

**Integration Testing:**
```python
# tests/test_integration.py
import pytest
import tempfile
import shutil
from pathlib import Path
from src.pipeline import DataPipeline

class TestDataPipeline:
    """Integration tests for complete data pipeline."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        (temp_dir / 'data' / 'raw').mkdir(parents=True)
        (temp_dir / 'data' / 'processed').mkdir(parents=True)
        (temp_dir / 'results').mkdir(parents=True)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_full_pipeline(self, temp_workspace):
        """Test complete data processing pipeline."""
        # Setup test data
        raw_data_file = temp_workspace / 'data' / 'raw' / 'test_data.csv'
        raw_data_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n")
        
        # Run pipeline
        pipeline = DataPipeline(workspace=temp_workspace)
        results = pipeline.run()
        
        # Verify results
        assert results['status'] == 'success'
        assert (temp_workspace / 'data' / 'processed' / 'cleaned_data.csv').exists()
        assert (temp_workspace / 'results' / 'summary.json').exists()
```

## Summary and Next Steps

### What We've Accomplished

Today we've significantly advanced your Python programming skills and established professional development workflows:

1. **Advanced Data Structures**: Mastered lists, dictionaries, sets, and their advanced patterns including comprehensions

2. **Professional Functions**: Learned to write well-documented, error-handled functions and organize code into modules

3. **Command Line Proficiency**: Can create scripts with argument parsing and environment configuration

4. **Remote Development**: Set up and work with remote development environments using SSH, tmux, and VS Code

5. **Professional Practices**: Understand project organization, testing, logging, and error handling

### Key Takeaways

- **Pythonic Code**: Use list comprehensions, dictionary comprehensions, and appropriate data structures for each task
- **Modularity**: Organize code into reusable modules and packages
- **Configuration**: Use environment variables and command-line arguments for flexible applications
- **Remote Work**: Modern data science often involves remote servers and distributed computing
- **Professional Standards**: Follow established patterns for project structure, testing, and documentation

### Practice Exercises

1. **Data Structure Challenge**: Create a program that processes a directory of CSV files, extracting summary statistics using advanced Python data structures and comprehensions.

2. **Module Development**: Build a reusable module for data validation with comprehensive error handling and testing.

3. **Remote Setup**: Set up a remote development environment with Jupyter notebooks and practice the full workflow.

4. **Command Line Tool**: Create a command-line application that processes data files with configurable options.

### Preparation for Next Lecture

In our next lecture, we'll dive into the NumPy and Pandas foundations that power data science in Python. To prepare:

1. Practice the concepts covered today, especially data structures and list comprehensions
2. Set up your remote development environment
3. Review basic statistics concepts (mean, median, standard deviation)
4. Install NumPy and Pandas in your environment: `pip install numpy pandas`

The skills you've learned today form the backbone of professional Python development. These patterns and practices will serve you throughout your data science career, making your code more maintainable, reliable, and professional.