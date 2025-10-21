# Data Wrangling: Join, Combine, and Reshape - Demo

## Demo Overview
Hands-on practice with the three fundamental data wrangling operations: merging datasets, concatenating DataFrames, and reshaping data formats.

## Demo 1: Customer Purchase Analysis (Merge Operations)

### Setup
- Load customer, product, and purchase datasets
- Explore data structure and relationships
- Understand join key requirements

### Activities
1. **Basic merge operations**
   - Inner join between purchases and customers
   - Left join to keep all purchases
   - Right join to keep all customers
   - Outer join to see everything

2. **Join validation and debugging**
   - Check row counts before and after merge
   - Identify relationship types (one-to-one, one-to-many)
   - Handle duplicate keys and validation

3. **Multi-column merges**
   - Merge on composite keys (store + date)
   - Handle overlapping column names with suffixes
   - Validate merge results

### Learning Objectives
- Master pd.merge() with different join types
- Understand when to use each join type
- Debug unexpected merge results
- Handle real-world merge challenges

## Demo 2: Survey Data Reshaping (Wide ↔ Long Format)

### Setup
- Load wide-format survey data (Q1, Q2, Q3 columns)
- Understand when you need long format
- Prepare for analysis and visualization

### Activities
1. **Wide to Long conversion**
   - Use pd.melt() to convert survey data
   - Handle id_vars and value_vars parameters
   - Customize variable and value column names

2. **Long to Wide conversion**
   - Use pivot() to create summary tables
   - Handle unique index/column combinations
   - Create readable reports

3. **Analysis on long format**
   - Use groupby on long data for statistics
   - Calculate summary metrics by category
   - Prepare data for plotting

### Learning Objectives
- Convert between wide and long formats
- Choose appropriate format for analysis
- Use groupby operations on long data
- Create publication-ready tables

## Demo 3: Time Series Concatenation (Index Management)

### Setup
- Split time series data into quarterly files
- Understand index alignment requirements
- Prepare for concatenation operations

### Activities
1. **Vertical concatenation**
   - Combine quarterly data files
   - Use ignore_index=True vs preserving indexes
   - Handle different column structures

2. **Horizontal concatenation**
   - Add related information side-by-side
   - Understand index alignment requirements
   - Handle misaligned indexes

3. **Index management**
   - Use set_index() for meaningful row labels
   - Reset indexes when needed
   - Create datetime indexes for time series

### Learning Objectives
- Master pd.concat() for different scenarios
- Understand index management strategies
- Handle time series data concatenation
- Choose between merge and concat

## Learning Objectives
- Master the three core data wrangling operations
- Choose appropriate tools for different scenarios
- Debug and validate data operations
- Prepare data for analysis and visualization

## Required Materials
- Python environment with pandas
- Sample datasets (customers, products, purchases)
- Jupyter notebook interface