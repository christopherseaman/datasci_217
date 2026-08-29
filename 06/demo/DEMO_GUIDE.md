# Data Wrangling Demo Guide

## Overview
Hands-on practice with the three fundamental data wrangling operations: merging datasets, concatenating DataFrames, and reshaping data formats.

The paired Markdown files are authoritative; regenerate a notebook with
Jupytext after editing Markdown. The tested activity environment is CPython
3.12.13 with NumPy 2.0.2, pandas 3.0.3, JupyterLab 4.4.10, and Jupytext 1.18.1
as pinned in `requirements.txt`:

```bash
uv venv --python 3.12.13 .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
jupyter lab
```

## Demo Structure

### Demo 1: Merge Operations
**File**: `demo1_merge_operations.ipynb`
**Duration**: 20 minutes
**Focus**: Database-style joins with customer, product, and purchase data

**Key Activities**:
- Inner, left, right, and outer joins
- Multi-column merges
- Handling duplicate keys

### Demo 2: Pivot and Melt Operations
**File**: `demo2_pivot_melt.ipynb`
**Duration**: 20 minutes
**Focus**: Converting between wide and long data formats

**Key Activities**:
- Wide to long conversion with `melt()`
- Long to wide conversion with `pivot()`
- Analysis on long format data

### Demo 3: Concatenation and Index Management
**File**: `demo3_concat_timeseries.ipynb`
**Duration**: 20 minutes
**Focus**: Combining time series data and managing indexes

**Key Activities**:
- Vertical and horizontal concatenation
- Index management strategies
- Time series data handling

## Learning Objectives
- Master the three core data wrangling operations
- Choose appropriate tools for different scenarios
- Prepare data for analysis and visualization

## Required Materials
- Python environment with pandas
- Sample datasets (customers, products, purchases)
- Jupyter notebook interface

## Instructor Notes
- Each demo builds on the previous one
- Focus on practical application over theoretical mastery
- Encourage students to experiment with different parameters
- Emphasize real-world scenarios and common pitfalls
