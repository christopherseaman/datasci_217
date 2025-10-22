# Demo 3: Concatenation and Index Management

## Overview
Combine time series data and manage DataFrame indexes for analysis.

## Learning Objectives
- Master `pd.concat()` for different scenarios
- Understand index management strategies
- Handle time series data concatenation
- Choose between merge and concat

## Activities

### 1. Vertical Concatenation
- Combine quarterly data files
- Use `ignore_index=True` vs preserving indexes
- Handle different column structures

### 2. Horizontal Concatenation
- Add related information side-by-side
- Understand index alignment requirements
- Handle misaligned indexes

### 3. Index Management
- Use `set_index()` for meaningful row labels
- Reset indexes when needed
- Create datetime indexes for time series

## Key Concepts
- **Vertical Concat**: Stack DataFrames vertically (rows)
- **Horizontal Concat**: Stack DataFrames horizontally (columns)
- **Index Alignment**: Indexes must match for horizontal concat
- **ignore_index**: Reset row numbers after concat

## When to Use Concat vs Merge
- **Concat**: Same structure, different data
- **Merge**: Different structures, related data
