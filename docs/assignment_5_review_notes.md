# Assignment 5 Review Notes - Documentation Coordinator Report

**Date:** 2025-10-15
**Project:** Clinical Trial Data Analysis (Midterm Exam)
**Total Points:** 100

---

## Executive Summary

This document consolidates findings from analyzing Assignment 5, identifying unclear instructions, knowledge gaps, assumptions made, and pedagogical considerations for a data science course covering lectures 1-5.

---

## 1. Unclear or Ambiguous Instructions

### 1.1 Q3 Function Parameter Ambiguity

**Issue:** `create_bins()` and `summarize_by_group()` function signatures

**Ambiguity:**
- README specifies `new_column` parameter as optional but doesn't indicate default behavior clearly
- `summarize_by_group()` states "If None, uses .describe() on numeric columns" but doesn't specify return format

**Resolution Applied:**
- Implemented `new_column=None` with default naming pattern `{column}_binned`
- Used `.describe()` for default aggregation, which returns MultiIndex columns

**Recommendation:** Explicitly state expected default behavior and return formats in docstrings.

---

### 1.2 Missing Data Strategy Selection Criteria

**Issue:** Q5 asks students to "choose appropriate imputation" without clear guidance

**Ambiguity:**
- No criteria provided for when to use mean vs median vs forward fill
- Unclear which columns are "critical" for dropping rows
- No guidance on acceptable data loss thresholds

**Resolution Applied:**
- Used median imputation (more robust to outliers in clinical data)
- Defined critical columns as: `patient_id`, `age`, `sex`
- Documented rationale in missing data report

**Recommendation:** Provide decision framework or rubric for imputation strategy selection.

---

### 1.3 Text Normalization Requirements

**Issue:** Q4 exploration reveals text inconsistencies but no explicit cleaning instructions

**Observation:**
- Site names have inconsistent capitalization and spacing
- Intervention groups have similar formatting issues
- Instructions don't specify when/how to normalize

**Resolution Applied:**
- Normalized text with `.str.strip().str.lower()` before value_counts
- Applied consistently in Q4 for accurate counting

**Recommendation:** Clarify if text normalization is expected or if raw data should be preserved.

---

### 1.4 Q2 Standalone Nature

**Issue:** README states Q2 is "standalone" but also mentions it produces "required outputs"

**Clarification Needed:**
- Is Q2 graded purely on Python fundamentals?
- Are Q2 outputs (`sample_data.csv`, `statistics.txt`) used in grading other questions?
- Should Q2 be run as part of Q8 pipeline?

**Current Understanding:**
- Q2 is NOT part of main data pipeline (Q3→Q4→Q5→Q6→Q7)
- Q2 outputs are checked independently in grading
- Q8 pipeline excludes Q2

---

## 2. Methods/Techniques Not Covered in Lectures 1-5

### 2.1 Advanced Pandas Operations

**Potentially Uncovered Techniques:**

1. **Pandas `pd.cut()` for binning**
   - Required for Q6 feature engineering
   - Creates categorical bins from continuous data
   - Example: Age groups, BMI categories

2. **MultiIndex handling in GroupBy**
   - `summarize_by_group()` with multiple aggregations creates MultiIndex columns
   - Requires flattening or understanding hierarchical columns
   - Not typically covered in introductory pandas

3. **Category dtype and type conversions**
   - `transform_types()` converts to 'category' dtype
   - Memory optimization technique
   - May not be covered in basic pandas

4. **Forward fill (`ffill`) vs backward fill**
   - Time-series imputation method
   - Conceptually different from mean/median
   - Requires understanding of data ordering

### 2.2 Statistical Concepts

**Potentially Advanced Topics:**

1. **Sentinel values vs true missing data**
   - Distinguishing coded missing (-999, -1) from NULL
   - Not always covered in basic data cleaning

2. **Imputation strategy selection**
   - Mean vs median (robustness to outliers)
   - Forward fill for time-series
   - Domain-specific considerations (clinical data)

3. **Outlier detection (3σ rule)**
   - Q4 tips mention 3 standard deviations
   - May not be formally introduced by lecture 5

### 2.3 Shell Scripting Patterns

**Advanced Bash Concepts:**

1. **Exit code checking with `$?`**
   - Q8 requires error handling with exit codes
   - May not be covered in basic bash introduction

2. **Pipeline orchestration patterns**
   - Sequential execution with error propagation
   - Log file management
   - May be beyond basic shell scripting

3. **Jupyter nbconvert execution**
   - Command-line notebook execution
   - Not a typical beginner topic

---

## 3. Assumptions Made to Complete Tasks

### 3.1 Data Quality Assumptions

1. **Patient ID uniqueness:** Assumed `patient_id` is unique identifier (not explicitly stated)

2. **Clinical measurement ranges:** Assumed realistic ranges for:
   - Age: 0-100 years (filter to 18-85 for analysis per README)
   - BMI: 0-100 kg/m²
   - Blood pressure: Systolic 0-200 mmHg
   - Cholesterol: Positive values in mg/dL

3. **Missing data patterns:** Assumed missing data is Missing Completely At Random (MCAR), not systematic

4. **Sentinel value completeness:** Assumed -999 (age) and -1 (BMI) are only sentinel values; others are true missing (NaN)

### 3.2 Analysis Assumptions

1. **Site distribution balance:** Assumed roughly equal enrollment across 5 sites is acceptable

2. **Intervention group balance:** Assumed 3-way randomization (Control, Treatment A, Treatment B)

3. **Age threshold (65+):** Used 65 as "older adult" cutoff based on cardiovascular disease context

4. **High BP threshold (140):** Used systolic BP > 140 mmHg as "high" per clinical guidelines

5. **Critical columns:** Defined as `patient_id`, `age`, `sex` for row dropping decisions

### 3.3 Output Format Assumptions

1. **CSV structure for Q4:** Assumed 2-column format (site, count) based on standard value_counts output

2. **Report text format:** Used plain text with headers and bullet points for readability

3. **Column naming conventions:** Used snake_case for new columns (e.g., `age_group`, `cholesterol_ratio`)

4. **Binned column naming:** Default pattern `{column}_binned` unless custom name provided

---

## 4. Difficulties Encountered

### 4.1 Technical Challenges

1. **Text normalization timing**
   - Had to normalize site/intervention text BEFORE filtering
   - Otherwise `in_list` filters fail on capitalization mismatches
   - Not immediately obvious from instructions

2. **Forward fill edge case**
   - `ffill()` can leave trailing NaN values if first rows are missing
   - Q5 comparison showed not all values filled with ffill strategy
   - Documented in imputation comparison

3. **MultiIndex column naming**
   - GroupBy with multiple aggregations creates MultiIndex
   - Requires flattening for clean CSV output
   - Not explicitly addressed in assignment

4. **Pandas dtype warnings**
   - `fillna(method='ffill')` is deprecated in pandas 2.0+
   - Used `.ffill()` directly instead
   - Students may encounter different syntax in online resources

### 4.2 Pedagogical Considerations

1. **Function reusability vs notebook flexibility**
   - Q3 utilities designed for reuse but may be too rigid
   - Notebooks sometimes need direct pandas operations for specific tasks
   - Balance between DRY principles and practical workflow

2. **Error handling expectations**
   - Utility functions include basic error handling
   - Unclear if this is expected or if simple implementations are sufficient

3. **Documentation depth**
   - Docstrings include examples and parameter descriptions
   - Unclear if this level of documentation is required or just helpful

4. **Testing without explicit test framework**
   - Q3 has `if __name__ == '__main__'` test section
   - Informal testing approach
   - Could benefit from explicit unittest/pytest introduction

---

## 5. Code Quality and Best Practices

### 5.1 Successfully Applied Principles

1. **Configuration-driven design:** Q2 demonstrates config file parsing and validation

2. **DRY principle:** Q3 utilities reused across Q4-Q7 notebooks

3. **Separation of concerns:** Clear distinction between data loading, cleaning, transformation, analysis

4. **Explicit naming:** Function and variable names are descriptive and clear

5. **Documentation:** Comprehensive docstrings with examples

### 5.2 Areas for Improvement

1. **Type hints:** Consistently used in function signatures (good practice)

2. **Return value consistency:** All utility functions return new DataFrames (avoid in-place modification)

3. **Error handling:** Basic try/except in some functions, could be more comprehensive

4. **Testing:** Informal testing in `__main__` blocks, not systematic unit tests

---

## 6. Lectures 1-5 Coverage Gap Analysis

### Likely Covered (Based on Assignment Requirements)

- Python basics: variables, functions, control flow (if/elif/else)
- File I/O: reading/writing files
- Pandas basics: DataFrames, Series, reading CSV
- Basic data exploration: `.head()`, `.describe()`, `.shape`, `.dtypes`
- Selection and filtering: `.loc[]`, boolean indexing
- Basic aggregation: `.groupby()`, `.agg()`
- Value counts and basic statistics

### Potentially NOT Covered (Based on Advanced Features)

- Pandas categorical dtype and memory optimization
- Advanced missing data strategies (imputation comparison)
- MultiIndex handling and column flattening
- Shell scripting with error handling and exit codes
- Jupyter notebook automation with nbconvert
- Forward fill and time-series imputation concepts
- Statistical concepts: outlier detection, sentinel values
- `pd.cut()` for binning continuous variables

### Recommended Pre-Assignment Preparation

If lectures 1-5 don't cover the above topics, consider:

1. **Supplemental tutorial on `pd.cut()`** - Critical for Q6 binning
2. **Missing data strategies lecture** - Framework for Q5 decisions
3. **Shell scripting refresher** - Focus on Q8 pipeline requirements
4. **MultiIndex handling guide** - For Q7 grouped aggregations

---

## 7. Grading Considerations

### Clear Grading Criteria (Well-Defined)

- File existence checks (objective)
- Script executability (objective)
- Output file structure validation (objective)
- Function signature compliance (objective)

### Subjective Grading Areas (Need Rubric)

1. **Imputation strategy choice (Q5)**
   - Multiple valid approaches
   - Should be graded on rationale documentation, not specific method

2. **Feature engineering creativity (Q6)**
   - Example features provided, but students may create others
   - Need criteria for evaluating novel features

3. **Analysis report quality (Q7)**
   - Text output requires subjective assessment
   - Define minimum content requirements

4. **Code quality and documentation**
   - GRADING_SPEC focuses on behavior testing
   - Should code style/comments factor into grading?

---

## 8. Student Success Factors

### Prerequisites for Success

1. **Solid Python fundamentals:** Functions, file I/O, control flow
2. **Basic pandas proficiency:** Selection, filtering, grouping
3. **Shell scripting basics:** File operations, script execution
4. **Statistical literacy:** Understanding mean, median, missing data concepts
5. **Problem decomposition:** Breaking complex tasks into functions

### Common Pitfalls to Avoid

1. **Hardcoding values instead of using utilities**
2. **Forgetting to normalize text before filtering**
3. **Not handling missing values before aggregation**
4. **Incorrect exit code handling in Q8 pipeline**
5. **Path issues (relative vs absolute paths)**

---

## 9. Recommendations for Future Iterations

### Assignment Improvements

1. **Add explicit decision framework for Q5** - When to use each imputation strategy
2. **Clarify Q2's role** - Make standalone nature more explicit
3. **Provide example MultiIndex output** - Help students understand flattening
4. **Add shell scripting primer** - Short tutorial on exit codes and error handling
5. **Define "critical columns"** - Specify which columns should never be dropped

### Pedagogical Enhancements

1. **Pre-assignment checklist** - Skills students should have before starting
2. **Intermediate checkpoints** - Validate Q3 utilities before using in Q4-Q7
3. **Sample output files** - Show expected structure for CSV/text outputs
4. **Error message guide** - Common errors and how to fix them
5. **Extension challenges** - Advanced tasks for students who finish early

### Documentation Clarifications

1. **Update TIPS.md** - Add pandas 2.0+ syntax warnings (ffill deprecation)
2. **Add troubleshooting section** - Common issues and solutions
3. **Provide decision trees** - Visual guides for imputation, filtering choices
4. **Include validation scripts** - Let students self-check outputs before submission

---

## 10. Conclusion

Assignment 5 is a comprehensive midterm that effectively tests:
- Python fundamentals (Q2)
- Shell scripting (Q1, Q8)
- Pandas data manipulation (Q3-Q7)
- Statistical reasoning (Q5)
- Pipeline orchestration (Q8)

**Strengths:**
- Realistic clinical trial dataset with authentic data quality issues
- Clear progression from setup → utilities → analysis → automation
- Artifact-based grading reduces fragility
- Reusable utility library reinforces best practices

**Areas for Enhancement:**
- Clarify assumptions about "critical columns" and imputation strategies
- Provide explicit coverage of advanced pandas techniques (pd.cut, category dtype)
- Add decision frameworks for subjective choices (missing data handling)
- Ensure lectures 1-5 cover shell scripting with error handling
- Update TIPS.md for pandas 2.0+ syntax changes

**Overall Assessment:**
Well-designed assignment that balances theory and practice. With minor clarifications and prerequisite alignment, this assignment effectively evaluates student proficiency in data science fundamentals.

---

## Appendix A: Knowledge Gaps by Question

| Question | Topic | Likely Coverage in Lectures 1-5 | Gap? |
|----------|-------|----------------------------------|------|
| Q1 | Shell scripting basics | Likely covered | No |
| Q1 | `tree` command | May not be covered | Minor |
| Q2 | Python fundamentals | Core curriculum | No |
| Q2 | Config file parsing | Likely covered | No |
| Q3 | Pandas basics | Core curriculum | No |
| Q3 | `pd.cut()` binning | May not be covered | **Yes** |
| Q3 | Category dtype | May not be covered | **Yes** |
| Q4 | Data exploration | Core curriculum | No |
| Q4 | Value counts | Core curriculum | No |
| Q5 | Missing data detection | Core curriculum | No |
| Q5 | Imputation strategies | May not be covered | **Yes** |
| Q5 | Forward fill concept | May not be covered | **Yes** |
| Q6 | Feature engineering | May not be covered | **Yes** |
| Q6 | One-hot encoding | May not be covered | **Yes** |
| Q7 | GroupBy basics | Core curriculum | No |
| Q7 | MultiIndex handling | May not be covered | **Yes** |
| Q8 | Exit code checking | May not be covered | **Yes** |
| Q8 | nbconvert | May not be covered | **Yes** |

---

## Appendix B: Output Files Checklist

### Required Outputs (Must Exist for Full Credit)

**Q1:**
- [ ] `data/clinical_trial_raw.csv` (10,000 rows × 18 columns)
- [ ] `reports/directory_structure.txt`

**Q2:**
- [ ] `data/sample_data.csv` (100 rows)
- [ ] `output/statistics.txt`

**Q4:**
- [ ] `output/q4_site_counts.csv` (5 rows × 2 columns)

**Q5:**
- [ ] `output/q5_cleaned_data.csv`
- [ ] `output/q5_missing_report.txt`

**Q6:**
- [ ] `output/q6_transformed_data.csv` (more columns than original)

**Q7:**
- [ ] `output/q7_site_summary.csv` (5 rows)
- [ ] `output/q7_intervention_comparison.csv` (3 rows)
- [ ] `output/q7_analysis_report.txt`

**Q8:**
- [ ] `reports/pipeline_log.txt`

---

## Appendix C: Function Testing Checklist

### Q2 Functions
- [ ] `parse_config()` returns dict with expected keys
- [ ] `validate_config()` correctly validates all rules
- [ ] `generate_sample_data()` creates file with correct rows/range
- [ ] `calculate_statistics()` computes mean, median, sum, count

### Q3 Functions
- [ ] `load_data()` loads CSV and returns DataFrame
- [ ] `clean_data()` removes duplicates and sentinel values
- [ ] `detect_missing()` returns Series with missing counts
- [ ] `fill_missing()` handles mean, median, ffill strategies
- [ ] `filter_data()` applies single and multiple filters correctly
- [ ] `filter_data()` handles equals, greater_than, less_than, in_list, in_range
- [ ] `transform_types()` converts datetime, numeric, category
- [ ] `create_bins()` creates binned column with custom naming
- [ ] `summarize_by_group()` handles default and custom aggregations

---

**End of Documentation Coordinator Report**
