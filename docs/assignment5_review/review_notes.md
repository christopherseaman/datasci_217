# Assignment 5 Review Notes

## Overview
Completing Assignment 5 (Clinical Trial Data Analysis) while documenting unclear instructions and coverage gaps relative to lectures 01-05.

## Assignment Intentionally Does NOT Use NumPy
- This is explicitly stated in the assignment
- All operations must use pandas and base Python only
- Students are expected to work with pandas DataFrames without numpy array operations

## Progress Tracking

### Completed
- ✅ Read assignment requirements
- ✅ Identified scaffold files

### In Progress
- 🔄 Reviewing lecture coverage

### Pending Questions
1. **Q3 `create_bins()` function**: Assignment requires `new_column` parameter, but example usage in assignment is unclear. Need to clarify if this is for creating a new column name or using existing column for values.

2. **Q3 `summarize_by_group()` function**: Assignment shows `agg_dict=None` parameter but doesn't specify default behavior when None. Should it compute all common aggregations (count, mean, etc.)?

3. **Q2 validation rules**: Assignment specifies "sample_data_rows must be an int and > 0" but doesn't clarify error handling strategy (raise exception vs return False in validation dict).

4. **Q8 Pipeline script**: Assignment mentions using `||` operator for error handling but doesn't specify whether to use `if [ $? -ne 0 ]` style or `||` style consistently.

## Coverage Analysis

### Lecture 01 (Command Line & Python Basics)
**Covered:**
- Shell scripting basics (#!/bin/bash, mkdir, echo)
- File operations (touch, cat)
- Python fundamentals (functions, if/elif/else, for loops)
- File I/O (open, read, write)
- String operations (strip, split, replace)
- List operations and basic statistics

**Used in Assignment:**
- Q1: Shell scripting, directory creation
- Q2: Python functions, conditionals, file I/O, string parsing
- Q8: Shell scripting, pipeline automation

### Lecture 02 (Git & Python Functions)
**Covered:**
- Git version control
- Functions and parameters
- Type checking
- F-string formatting
- Error handling basics

**Used in Assignment:**
- Q2: Function definitions with type hints
- Q3: Reusable function library
- All: Documentation and code organization

### Lecture 03 (NumPy & Virtual Environments)
**NOT APPLICABLE** - Assignment explicitly excludes NumPy

### Lecture 04 (Pandas Basics & Jupyter)
**Covered:**
- pd.read_csv(), pd.DataFrame()
- Basic indexing and selection (.loc, .iloc, boolean indexing)
- Missing data detection (.isnull(), .notnull())
- Basic operations (sort_values, unique, value_counts)
- Summary statistics (.describe(), .info())

**Used in Assignment:**
- Q3: load_data(), detect_missing()
- Q4: Data exploration
- Q5: Missing data analysis
- Q7: Aggregation

**Potential Gaps:**
- `pd.cut()` for binning (Q3, Q6) - covered in lecture 04 under "Creating Categories"
- `transform_types()` with datetime conversion - covered
- `drop_duplicates()` - covered

### Lecture 05 (Data Cleaning)
**Covered:**
- Missing data handling (fillna, dropna, forward fill, backward fill)
- Duplicate removal (duplicated(), drop_duplicates())
- Data type conversion (astype(), pd.to_datetime(), pd.to_numeric())
- String operations (.str methods)
- Categorical encoding (pd.get_dummies())
- Outlier detection and handling
- Data validation

**Used in Assignment:**
- Q3: clean_data(), fill_missing(), filter_data(), create_bins()
- Q5: Missing data imputation strategies
- Q6: Type conversions, feature engineering, dummy variables
- Q7: Groupby operations

**Potential Gaps:**
- None identified - all required methods covered

## Unclear Instructions

### Q1: Directory Structure Documentation
**Issue:** Assignment says "Save directory listing to `reports/directory_structure.txt` using `tree`"
**Unclear:** Should this be run after data generation or before? Tree output will differ.
**Assumption:** Run after all steps complete (seems most useful)

### Q2: Config Validation Return Format
**Issue:** Assignment says "Returns: {key: True/False} for each validation rule"
**Unclear:** Should return look like `{"sample_data_rows": True, "sample_data_min": True}` or `{"rows_positive": True, "min_valid": True}`?
**Assumption:** Using descriptive validation rule names like `{"rows_positive": True, "min_greater_than_zero": True}`

### Q2: Sample Data Generation
**Issue:** Assignment says "Generate file with random numbers using config parameters"
**Unclear:** Should numbers be integers or floats? One number per line or all on one line?
**Assumption:** Integers (age context), one per line (CSV format)

### Q3: filter_data() Filter Syntax
**Issue:** Assignment mentions "equals, greater_than, in_range, etc." but doesn't specify exact dictionary format
**Unclear:** Is it `[{"column": "age", "op": "greater_than", "value": 18}]` or different syntax?
**Assumption:** `[{"column": "age", "operator": ">=", "value": 18}]` style

### Q5: Missing Data Report Format
**Issue:** Assignment requires `output/q5_missing_report.txt` but doesn't specify format
**Unclear:** Should it include percentages, counts, recommendations?
**Assumption:** Include column names, counts, percentages, and chosen strategies

### Q6: Age Groups Binning
**Issue:** Assignment specifies bins `[0, 40, 55, 70, 100]` and labels `['<40', '40-54', '55-69', '70+']`
**Unclear:** Should bin edges be inclusive or exclusive? Default pd.cut behavior?
**Assumption:** Use pd.cut defaults (right edge inclusive)

### Q7: Analysis Report Content
**Issue:** Assignment says "Save key findings to `output/q7_analysis_report.txt`"
**Unclear:** What constitutes "key findings"? Should include statistical interpretation?
**Assumption:** Include summary statistics, notable patterns, group comparisons

## Methods Not Covered in Lectures

None identified - all required pandas methods were covered in lectures 04 and 05.

## Implementation Notes

### Design Decisions Made

1. **Q2 Validation**: Will return detailed validation results with descriptive keys
2. **Q3 Utilities**: All functions will follow pandas conventions and return new DataFrames (non-destructive)
3. **Q5 Imputation**: Will document strategy choices clearly in report
4. **Q6 Type Conversions**: Will use pd.to_datetime with error handling
5. **Q7 Aggregations**: Will use descriptive column names in output files

### Testing Strategy

1. Run Q1 first to set up directory structure
2. Test Q2 independently (standalone exercise)
3. Test Q3 functions with sample data before using in notebooks
4. Execute Q4-Q7 in sequence, checking outputs
5. Run Q8 pipeline to verify full integration
6. Validate all output files exist and contain expected data

## Completed Work

### Q1: Project Setup Script ✅
**Status:** Completed and tested successfully

**Implementation:**
- Created bash script with shebang `#!/bin/bash`
- Made executable with proper permissions
- Created directories: `data/`, `output/`, `reports/`
- Successfully generated dataset using `python3 generate_data.py`
- Saved directory structure to `reports/directory_structure.txt`
- Added fallback from `tree` to `ls -R` for compatibility

**Coverage Assessment:**
- ✅ All methods covered in Lecture 01 (shell scripting basics)
- ✅ Clear instructions, no gaps identified

**Output:**
- ✅ `data/clinical_trial_raw.csv` (10,000 patients, 18 variables)
- ✅ `reports/directory_structure.txt`
- ✅ All required directories created

### Q2: Python Data Processing ✅
**Status:** Completed and tested successfully

**Implementation:**
- ✅ `parse_config()`: Reads key=value config format, handles comments
- ✅ `validate_config()`: Uses if/elif/else for validation logic
- ✅ `generate_sample_data()`: Creates 100 random integers (18-75), one per line
- ✅ `calculate_statistics()`: Computes mean, median, sum, count without pandas

**Coverage Assessment:**
- ✅ All methods covered in Lectures 01-02 (file I/O, functions, conditionals)
- ✅ No numpy used (as required)
- ✅ Clear instructions

**Key Design Decision:**
- Validation returns descriptive keys (`rows_valid`, `min_valid`, `max_valid`)
- Used try/except for robust error handling
- Generated integers (appropriate for age data)
- One number per line (standard CSV format)

**Output:**
- ✅ `data/sample_data.csv` (100 random integers)
- ✅ `output/statistics.txt` (count, sum, mean, median)

**Test Results:**
```
Configuration loaded:
  sample_data_rows: 100
  sample_data_min: 18
  sample_data_max: 75

Validation results:
  rows_valid: True
  min_valid: True
  max_valid: True

Statistics Summary:
  count: 100
  sum: 4684
  mean: 46.84
  median: 47.00
```

### Q3: Data Utilities Library ✅
**Status:** Completed and tested successfully

**Implementation:**
All 8 required functions implemented with pandas (no numpy):

1. ✅ `load_data()`: Uses `pd.read_csv()`
2. ✅ `clean_data()`: Removes duplicates, replaces sentinel values with `pd.NA`
3. ✅ `detect_missing()`: Returns `df.isnull().sum()`
4. ✅ `fill_missing()`: Supports mean, median, and forward fill strategies
5. ✅ `filter_data()`: Handles equals, greater_than, less_than, in_range, in_list conditions
6. ✅ `transform_types()`: Converts to datetime, numeric, category, string types
7. ✅ `create_bins()`: Uses `pd.cut()` with optional `new_column` parameter
8. ✅ `summarize_by_group()`: Uses `.groupby()` with custom or default aggregations

**Coverage Assessment:**
- ✅ All methods covered in Lectures 04-05 (pandas operations, data cleaning)
- ✅ No numpy used (as required)
- ✅ All functions follow pandas conventions (non-destructive, return new DataFrames)

**Key Design Decisions:**
- `filter_data()` uses dict format: `{'column': str, 'condition': str, 'value': any}`
- `create_bins()` defaults new column name to `{column}_binned` if not specified
- `summarize_by_group()` uses `.describe()` when `agg_dict=None`
- All functions use `.copy()` to avoid modifying original DataFrames

**Test Results:**
```
Data utilities loaded successfully!
Available functions:
  - load_data()
  - clean_data()
  - detect_missing()
  - fill_missing()
  - filter_data()
  - transform_types()
  - create_bins()
  - summarize_by_group()

Test DataFrame: (5, 3)
Detect missing:
age    0
bmi    1
site   0

Clean data (replace -999):
age    1
bmi    1
site   0

Test passed!
```

### Q4-Q7: Analysis Notebooks
**Status:** Scaffold files exist, ready to be completed using Q3 utilities

### Q8: Pipeline Automation Script
**Status:** Scaffold exists, ready to be completed

## Final Coverage Analysis

### Methods Covered in Lectures vs Assignment Requirements

**Lecture 01 (Command Line & Python Basics):**
- ✅ Used in Q1, Q2, Q8
- All required methods covered (shell scripting, file I/O, functions, loops)

**Lecture 02 (Git & Functions):**
- ✅ Used in Q2, Q3 (function definitions, type hints)
- All required methods covered

**Lecture 03 (NumPy):**
- ❌ NOT APPLICABLE (assignment explicitly excludes numpy)

**Lecture 04 (Pandas Basics):**
- ✅ Used in Q3, Q4, Q5, Q6, Q7
- All required methods covered: read_csv, DataFrame operations, indexing, groupby, aggregations

**Lecture 05 (Data Cleaning):**
- ✅ Used in Q3, Q5, Q6
- All required methods covered: missing data, duplicates, type conversion, string operations, categorical encoding

### No Coverage Gaps Identified
All assignment requirements can be completed using methods taught in lectures 01-05.

## Observations and Recommendations

### Clear Instructions
1. Q1: Clear and straightforward
2. Q2: Clear, good standalone Python practice
3. Q3: Well-designed utility library approach
4. Q4-Q7: Scaffold structure is helpful
5. Q8: Good pipeline automation practice

### Ambiguities Resolved
1. **Q2 Validation Return Format**: Implemented with descriptive keys (`rows_valid`, `min_valid`, `max_valid`)
2. **Q3 filter_data() syntax**: Used dictionary format with `column`, `condition`, `value` keys
3. **Q3 create_bins() new_column parameter**: Defaults to `{column}_binned` if None
4. **Q3 summarize_by_group() default behavior**: Uses `.describe()` when `agg_dict=None`

### Student Learning Benefits
1. **Modular Design**: Q3 utilities promote code reuse (excellent professional practice)
2. **Configuration-Driven**: Q2 demonstrates parameterized workflows
3. **Testing Mindset**: Scaffold structure encourages incremental testing
4. **Real-World Workflow**: Mirrors professional data science project structure

### Potential Student Challenges

1. **Virtual Environment Setup**: Students may encounter externally-managed environment error
   - **Solution**: Instructions should include virtual environment creation

2. **Q3 filter_data() Complexity**: Filter dictionary format may be confusing initially
   - **Mitigation**: Examples in docstrings are clear and comprehensive

3. **Q5 Missing Data Strategies**: Choosing appropriate imputation methods requires judgment
   - **Mitigation**: Lecture 05 covers this well

4. **Q6 pd.cut() Usage**: Students may be unclear about inclusive/exclusive bin edges
   - **Mitigation**: Lecture 04 "Creating Categories" section covers this

## Recommendations for Future Iterations

1. **Add Virtual Environment Setup**: Include in Q1 or assignment README
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   ```

2. **Clarify Q3 filter_data() Format**: Add explicit example in assignment description

3. **Q5 Missing Report Format**: Provide example output format for consistency

4. **Q8 Error Handling**: Clarify whether to use `||` or `if [ $? -ne 0 ]` style

## Time Investment
- Q1: ~15 minutes (straightforward shell scripting)
- Q2: ~30 minutes (python fundamentals, testing)
- Q3: ~60 minutes (8 functions with proper testing)
- Q4-Q7: Estimated ~90 minutes total (using Q3 utilities)
- Q8: Estimated ~20 minutes (pipeline script)

**Total Estimated**: ~3.5 hours for complete assignment

---

*Completed: 2025-10-15*
*All core functions tested and working*
*Ready for Q4-Q7 notebook completion*
