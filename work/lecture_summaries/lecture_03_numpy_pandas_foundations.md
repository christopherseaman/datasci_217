# Lecture 3: NumPy + Pandas Foundations

## Learning Objectives
By the end of this lecture, students will be able to:
- Create and manipulate NumPy arrays for numerical computing
- Understand array operations, broadcasting, and vectorization concepts
- Load, explore, and manipulate data using Pandas DataFrames and Series
- Perform basic data cleaning, filtering, and transformation operations
- Apply NumPy and Pandas to solve real-world data science problems
- Understand when to use NumPy vs Pandas for different data tasks

## Content Consolidation Details

### Primary Sources (Current Lectures)
- **Lecture 05 (95%)**: NumPy fundamentals, arrays, operations, Pandas basics, DataFrames, Series
- **Lecture 06 (40%)**: Basic data manipulation, reading files, data types
- **Lecture 04 (15%)**: File operations in Python

### Secondary Integration
- **Previous lectures**: Python data structures provide foundation for understanding arrays and DataFrames

## Specific Topics Covered

### NumPy Fundamentals (45 minutes)
1. **Introduction to NumPy**
   - Why NumPy: performance benefits for numerical computing
   - Installation and importing: `import numpy as np`
   - NumPy vs Python lists: memory efficiency and speed
   - Scientific computing ecosystem overview

2. **Creating NumPy Arrays**
   - From Python lists: `np.array([1, 2, 3])`
   - Built-in creation functions: `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`
   - Array from ranges: `np.arange(start, stop, step)`
   - Multi-dimensional arrays: 2D and 3D array creation
   - Random number generation: `np.random.rand()`, `np.random.randint()`

3. **Array Properties and Attributes**
   - Shape, size, and dimensions: `.shape`, `.size`, `.ndim`
   - Data types: `.dtype`, type conversion with `.astype()`
   - Memory layout and efficiency considerations

4. **Array Indexing and Slicing**
   - 1D array indexing: positive and negative indices
   - Multi-dimensional indexing: `arr[row, col]`
   - Slicing: `arr[start:stop:step]`, multi-dimensional slicing
   - Boolean indexing: using conditions to filter arrays
   - Fancy indexing: using arrays of indices

5. **Array Operations**
   - Element-wise operations: arithmetic (`+`, `-`, `*`, `/`)
   - Universal functions (ufuncs): `np.sqrt()`, `np.exp()`, `np.sin()`
   - Array aggregations: `np.sum()`, `np.mean()`, `np.std()`, `np.min()`, `np.max()`
   - Axis-specific operations: operating along rows or columns

6. **Broadcasting**
   - Broadcasting rules and principles
   - Scalar with array operations
   - Array with different shapes operations
   - Practical examples and common use cases

7. **Array Manipulation**
   - Reshaping: `.reshape()`, `.flatten()`, `.ravel()`
   - Stacking: `np.vstack()`, `np.hstack()`, `np.concatenate()`
   - Splitting: `np.split()`, `np.hsplit()`, `np.vsplit()`

### Pandas Foundations (50 minutes)
1. **Introduction to Pandas**
   - What is Pandas: data analysis and manipulation library
   - Installation and importing: `import pandas as pd`
   - Relationship to NumPy: built on top of NumPy arrays
   - Main data structures: Series and DataFrame

2. **Pandas Series**
   - Creating Series: from lists, arrays, dictionaries
   - Series indexing: labels and positions
   - Series operations: mathematical and string operations
   - Series methods: `.value_counts()`, `.unique()`, `.sort_values()`

3. **Pandas DataFrames**
   - Creating DataFrames: from dictionaries, lists of dictionaries, CSV files
   - DataFrame structure: rows, columns, index
   - Basic DataFrame attributes: `.shape`, `.columns`, `.index`, `.dtypes`
   - DataFrame information: `.info()`, `.describe()`, `.head()`, `.tail()`

4. **Reading and Writing Data**
   - Reading CSV files: `pd.read_csv()` with various parameters
   - Reading Excel files: `pd.read_excel()`
   - Reading JSON files: `pd.read_json()`
   - Writing data: `.to_csv()`, `.to_excel()`, `.to_json()`
   - Handling different file formats and encoding issues

5. **Data Selection and Filtering**
   - Column selection: single columns `df['column']`, multiple columns `df[['col1', 'col2']]`
   - Row selection: `.loc[]` (label-based), `.iloc[]` (position-based)
   - Boolean filtering: `df[df['column'] > value]`
   - Combining conditions: `&` (and), `|` (or), `~` (not)
   - Query method: `df.query('condition')`

6. **Basic Data Manipulation**
   - Adding new columns: direct assignment and calculations
   - Modifying existing columns: data type conversions, transformations
   - Dropping columns and rows: `.drop()` method
   - Sorting data: `.sort_values()`, `.sort_index()`
   - Renaming columns: `.rename()` method

### Data Cleaning Essentials (25 minutes)
1. **Handling Missing Data**
   - Identifying missing values: `.isnull()`, `.isna()`, `.notnull()`
   - Counting missing values: `.isnull().sum()`
   - Dropping missing values: `.dropna()` with parameters
   - Filling missing values: `.fillna()` with different strategies

2. **Data Type Management**
   - Checking data types: `.dtypes`
   - Converting data types: `.astype()`, `pd.to_numeric()`, `pd.to_datetime()`
   - Categorical data: `pd.Categorical`, `.astype('category')`

3. **Basic String Operations**
   - String methods in Pandas: `.str.lower()`, `.str.upper()`, `.str.strip()`
   - String splitting and extraction: `.str.split()`, `.str.extract()`
   - String replacement: `.str.replace()`

### NumPy-Pandas Integration (15 minutes)
1. **When to Use Each Library**
   - NumPy for: numerical computing, homogeneous data, mathematical operations
   - Pandas for: data analysis, heterogeneous data, labeled data
   - Performance considerations and memory usage

2. **Converting Between NumPy and Pandas**
   - DataFrame to NumPy: `.values`, `.to_numpy()`
   - NumPy to DataFrame: `pd.DataFrame(array)`
   - Maintaining data types and structure

3. **Using NumPy Functions with Pandas**
   - Applying NumPy functions to DataFrames
   - Performance optimization techniques
   - Best practices for integration

## Content to Trim (25% reduction from source lectures)

### From Lecture 05
- **Remove (15 minutes)**: Advanced NumPy operations (complex mathematical functions, advanced broadcasting)
- **Reduce (10 minutes)**: Multiple file format examples - focus on CSV and basic Excel
- **Simplify (8 minutes)**: Complex array manipulation examples

### From Lecture 06
- **Remove (12 minutes)**: Advanced data cleaning techniques (move to Lecture 4)
- **Remove (10 minutes)**: Complex string processing and regex (save for advanced topics)
- **Reduce (8 minutes)**: Multiple data visualization examples - focus on basic exploration

## Practical Exercises and Hands-on Components

### NumPy Skills Development (25 minutes)
1. **Array Creation and Manipulation**
   - Create arrays of different shapes and types
   - Practice indexing and slicing operations
   - Implement mathematical computations using arrays

2. **Real Data Analysis with NumPy**
   - Load numerical data from CSV files
   - Perform statistical analysis on temperature or sales data
   - Create visualizations using matplotlib (basic plotting)

3. **Broadcasting Exercise**
   - Normalize data across different dimensions
   - Apply operations to multi-dimensional arrays
   - Understand and debug broadcasting errors

### Pandas Data Exploration (30 minutes)
1. **Data Loading and Exploration**
   - Load real-world dataset (e.g., Titanic, cars, weather data)
   - Explore data structure and basic statistics
   - Identify data quality issues

2. **Data Filtering and Selection**
   - Extract subsets of data based on conditions
   - Create new columns based on existing data
   - Practice different selection methods

3. **Data Cleaning Workshop**
   - Handle missing values in realistic scenarios
   - Convert data types appropriately
   - Clean and standardize text data

### Integration Project (20 minutes)
1. **Combined NumPy-Pandas Analysis**
   - Load data with Pandas, perform calculations with NumPy
   - Compare performance between approaches
   - Create summary statistics and basic visualizations

2. **Real-World Problem Solving**
   - Analyze sales data: calculate moving averages, growth rates
   - Process survey data: handle missing responses, categorize answers
   - Time series basics: date parsing and simple temporal analysis

## Prerequisites and Dependencies

### From Previous Lectures
- Python data structures (lists, dictionaries)
- Basic file operations and reading/writing
- Command line skills for package installation
- Git workflow for assignment submission

### Technical Requirements
- NumPy and Pandas installation: `pip install numpy pandas`
- Jupyter Notebook or IDE with Python support
- Sample datasets for practice exercises
- Matplotlib for basic plotting: `pip install matplotlib`

### Preparation Materials
- NumPy/Pandas installation guide
- Sample datasets with documentation
- Quick reference sheets for common operations

## Assessment Components

### Formative Assessment (During Class)
- Interactive coding exercises with immediate feedback
- Pair programming for data exploration tasks
- Live polling on concept understanding
- Code review and discussion sessions

### Summative Assessment (Assignment)
1. **NumPy Computing Project**
   - Implement statistical calculations using NumPy
   - Demonstrate array manipulation and broadcasting
   - Show performance comparisons with pure Python

2. **Pandas Data Analysis**
   - Load and clean real-world dataset
   - Perform exploratory data analysis
   - Generate summary report with findings

3. **Integration Challenge**
   - Combine NumPy and Pandas for complex analysis
   - Optimize code for performance
   - Document methodology and results

4. **Practical Application**
   - Solve domain-specific problem (business, health, environment)
   - Use appropriate tools for each task
   - Present findings with supporting visualizations

## Key Success Metrics
- Students can create and manipulate NumPy arrays confidently
- Students can load, explore, and clean data using Pandas
- Students understand when to use NumPy vs Pandas for different tasks
- Students can solve practical data analysis problems
- 85% of students complete hands-on exercises successfully

## Integration with Course Progression
This lecture enables students for:
- **Lecture 4**: Advanced data analysis and visualization
- **Lecture 5**: Statistical analysis and machine learning applications
- **Future projects**: All data science work relies on these foundational skills

## Resources and References

### Essential Resources
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [Pandas Getting Started Guide](https://pandas.pydata.org/docs/getting_started/index.html)
- [Python for Data Analysis (McKinney)](https://wesmckinney.com/book/) - Chapters 4-7
- [NumPy/Pandas Cheat Sheets](https://github.com/pandas-dev/pandas/tree/main/doc/cheatsheet)

### Interactive Learning
- [NumPy Exercises](https://github.com/rougier/numpy-100)
- [Pandas Exercises](https://github.com/guipsamora/pandas_exercises)
- [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas)

### Practice Datasets
- [Seaborn Built-in Datasets](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Government Open Data Portals](https://data.gov/)

### Performance and Best Practices
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/performance.html)
- [Pandas Performance and Optimization](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [Effective Python for Data Science](https://realpython.com/python-data-science/)

### Community Support
- [NumPy Community](https://numpy.org/community/)
- [Pandas Community](https://pandas.pydata.org/community/)
- Stack Overflow tags: numpy, pandas, python-data
- Course forum for peer assistance and discussion