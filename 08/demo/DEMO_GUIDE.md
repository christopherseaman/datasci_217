# Data Aggregation Demo Guide

## Overview
Hands-on practice with data aggregation operations: groupby operations, pivot tables, remote computing, and performance optimization.

Markdown is the authoritative source for each generated notebook. The tested
activity environment is CPython 3.12.13 with NumPy 2.0.2, pandas 3.0.5,
Matplotlib 3.11.1, Seaborn 0.13.2, JupyterLab 4.4.10, Jupytext 1.18.1, and
psutil 7.0.0 as pinned in `requirements.txt`:

```bash
uv venv --python 3.12.13 .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
jupyter lab
```

## Demo Structure

### Demo 1: GroupBy Operations
**File**: `demo1_groupby_operations.ipynb` (generated from `demo1_groupby_operations.md`)
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo1_groupby_operations.ipynb)
**Duration**: 25 minutes
**Focus**: Split-apply-combine paradigm and aggregation functions

**Key Activities**:
- Basic groupby operations and aggregation
- Transform, filter, and apply operations
- Hierarchical grouping and MultiIndex

### Demo 2: Pivot Tables and Cross-Tabulations
**File**: `demo2_pivot_tables.ipynb` (generated from `demo2_pivot_tables.md`)
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo2_pivot_tables.ipynb)
**Duration**: 25 minutes
**Focus**: Multi-dimensional data analysis

**Key Activities**:
- Pivot table creation and customization
- Cross-tabulation analysis
- Advanced pivot operations

### Demo 3: Remote Computing and Performance
**File**: `demo3_remote_performance.ipynb` (generated from `demo3_remote_performance.md`)
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo3_remote_performance.ipynb)
**Duration**: 25 minutes
**Focus**: Large dataset handling and optimization

**Key Activities**:
- SSH and remote computing setup
- Performance optimization techniques
- Parallel processing with large datasets

## Learning Objectives
- Master the split-apply-combine paradigm
- Create pivot tables for multi-dimensional analysis
- Use remote computing for large datasets
- Optimize performance for aggregation operations
- Apply advanced groupby techniques

## Required Materials
- Python environment with pandas, numpy
- Sample datasets (sales, customer, product data)
- Jupyter notebook interface
- Optional: Remote server access for SSH demo
- Jupytext; Markdown is the authoritative source for each generated notebook

## Instructor Notes
- Each demo builds aggregation skills progressively
- Focus on practical application over theoretical mastery
- Encourage students to experiment with different aggregation functions
- Emphasize real-world scenarios and performance considerations
