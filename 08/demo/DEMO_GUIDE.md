# Data Aggregation Demo Guide

## Overview
This guide supports executable demonstrations of core data aggregation, remote
computing, and performance. The lecture page explains the concepts; the generated
notebooks are their runnable companions.

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

**Illustrative topics**:
- Basic groupby operations and aggregation
- Transform, filter, and apply operations
- Hierarchical grouping and MultiIndex

### Demo 2: Pivot Tables and Cross-Tabulations
**File**: `demo2_pivot_tables.ipynb` (generated from `demo2_pivot_tables.md`)
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo2_pivot_tables.ipynb)
**Duration**: 25 minutes
**Focus**: Multi-dimensional data analysis

**Illustrative topics**:
- Pivot table creation and customization
- Cross-tabulation analysis
- Advanced pivot operations

### Demo 3: Remote Computing and Performance
**File**: `demo3_remote_performance.ipynb` (generated from `demo3_remote_performance.md`)
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo3_remote_performance.ipynb)
**Duration**: 25 minutes
**Focus**: Large dataset handling, optimization, and remote computing

**Illustrative topics**:
- SSH and remote computing setup
- Performance measurement and dtype/memory experiments
- Chunking plus a conceptual parallel-processing comparison

## Core Learning Objectives
- Master the split-apply-combine paradigm
- Create pivot tables for multi-dimensional analysis
- Apply advanced groupby techniques
- Measure and improve aggregation performance
- Explain how SSH and a persistent tmux session fit into a remote analysis workflow

## Required Materials
- Python environment with pandas, numpy
- Sample datasets (sales, customer, product data)
- Jupyter notebook interface
- No remote server is required; the notebook simulates the SSH workflow so its
  executable cells run with the recorded environment
- Jupytext; Markdown is the authoritative source for each generated notebook

## Instructor Notes
- Demos 1–3 form the lecture sequence.
- Focus on practical application over theoretical mastery
- Use the executable examples to demonstrate different aggregation functions.
- Use the executable examples to connect aggregation performance with the remote
  workflow presented in the lecture.
