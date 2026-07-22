# Demo 2: Pivot and Melt Operations

## Overview
Convert between wide and long data formats for analysis and visualization.

## Learning Objectives
- Convert between wide and long formats
- Choose appropriate format for analysis
- Use groupby operations on long data
- Create publication-ready tables

## Activities

### 1. Wide to Long Conversion
- Use `pd.melt()` to convert survey data
- Handle `id_vars` and `value_vars` parameters
- Customize variable and value column names

### 2. Long to Wide Conversion
- Use `pivot()` to create summary tables
- Handle unique index/column combinations
- Create readable reports

### 3. Analysis on Long Format
- Use groupby on long data for statistics
- Calculate summary metrics by category
- Prepare data for plotting

## Key Concepts
- **Wide Format**: Categories as columns (Q1, Q2, Q3)
- **Long Format**: Categories as rows (question, response)
- **Pivot**: Long → Wide (groupby + unstack)
- **Melt**: Wide → Long (gather columns)

## When to Use Each Format
- **Wide**: Good for reporting, human-readable
- **Long**: Good for analysis, groupby operations
