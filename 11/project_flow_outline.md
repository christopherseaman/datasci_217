# Final Project Flow: Lecture 11 & Final Exam Outline

## Overview

This document outlines the common project flow that will be shared between **Lecture 11** (final lecture) and the **Final Exam**. Both will follow the same analytical path through a complete data science project, using different datasets from `data_for_final.md`. The lecture will provide detailed explanations and demonstrations, while the **take-home exam** will test the same core skills in a structured assessment format.

**Key Principle:** Both the lecture and exam follow a realistic data science project workflow, moving from raw data to insights and predictions. **Time series analysis is a core component** - all datasets support temporal analysis and students must demonstrate time series skills.

---

## Project Flow Structure

### Phase 1: Project Setup & Data Acquisition
**Lecture Topics:** 01 (Environment), 02 (Python basics)  
**Core Skills:**
- Environment setup verification
- Import necessary libraries (pandas, numpy, matplotlib, seaborn, etc.)
- Load data from files (CSV, Parquet, or specialized formats)
- Initial data inspection (`df.head()`, `df.info()`, `df.describe()`)
- Understand data structure and schema

**Lecture Focus:**
- Best practices for project organization
- Library import strategies
- Data source documentation review
- Initial data quality assessment

**Exam Focus:**
- Quick data loading
- Basic inspection commands
- Identify data types and structure

---

### Phase 2: Data Exploration & Understanding
**Lecture Topics:** 04 (Pandas basics), 07 (Visualization)  
**Core Skills:**
- Shape and basic statistics (`df.shape`, `df.describe()`)
- Data type inspection (`df.dtypes`, `df.info()`)
- Missing value overview (`df.isnull().sum()`)
- Initial visualizations:
  - Distribution plots (histograms, box plots)
  - Relationship plots (scatter plots, correlation heatmaps)
  - **Time series plots** (required - all datasets have temporal components)
- Identify key variables and potential relationships

**Lecture Focus:**
- Exploratory data analysis (EDA) best practices
- Visualization principles (Tufte's principles)
- Identifying interesting patterns early
- Documentation of findings

**Exam Focus:**
- Quick summary statistics
- Basic visualization creation
- Pattern identification

---

### Phase 3: Data Cleaning & Preprocessing
**Lecture Topics:** 05 (Data cleaning)  
**Core Skills:**
- **Missing data:**
  - Identify missing patterns (`df.isnull().sum()`, `df.isnull().mean()`)
  - Decide on handling strategy (drop, impute, forward-fill)
  - Implement missing data handling
- **Outliers:**
  - Detect outliers (IQR method, z-scores, domain knowledge)
  - Decide on handling (remove, cap, transform)
- **Data types:**
  - Convert data types (`astype()`, `pd.to_datetime()`)
  - Handle mixed types
- **Duplicates:**
  - Identify and handle duplicates (`df.duplicated()`, `df.drop_duplicates()`)
- **Data validation:**
  - Check for logical inconsistencies
  - Validate ranges and constraints

**Lecture Focus:**
- Systematic cleaning workflow
- Understanding missing data mechanisms (MCAR, MAR, MNAR)
- Domain-specific cleaning considerations
- Creating reusable cleaning functions

**Exam Focus:**
- Identify missing data patterns
- Apply appropriate cleaning techniques
- Handle common data quality issues

---

### Phase 4: Data Wrangling & Transformation
**Lecture Topics:** 06 (Joins, reshaping), 09 (Time series - **REQUIRED**)  
**Core Skills:**
- **Merging/Joining:**
  - Combine multiple data sources (`pd.merge()`)
  - Choose appropriate join type (inner, left, right, outer)
  - Handle key mismatches
- **Reshaping:**
  - Pivot tables (`df.pivot()`, `df.pivot_table()`)
  - Melt operations (`df.melt()`)
  - Wide to long / long to wide transformations
- **Time series (REQUIRED for all datasets):**
  - Parse datetime columns (`pd.to_datetime()`)
  - Set datetime index (`df.set_index()`)
  - Extract temporal features (hour, day, month, year, day_of_week)
  - Handle time zones if needed
  - Resample irregular time series to regular intervals (if needed)
- **Index management:**
  - Set and reset indexes
  - Work with MultiIndex if needed

**Lecture Focus:**
- Database-style joins and when to use each type
- Reshaping strategies for different analysis needs
- Time series feature engineering
- Performance considerations for large datasets

**Exam Focus:**
- Merge datasets correctly
- Reshape data for analysis
- **Extract temporal features (required)**
- Handle datetime parsing and indexing

---

### Phase 5: Feature Engineering & Aggregation
**Lecture Topics:** 08 (GroupBy, aggregation), 09 (Time series features)  
**Core Skills:**
- **GroupBy operations:**
  - Split-apply-combine pattern
  - Aggregate by groups (`df.groupby().agg()`)
  - Multiple aggregation functions
  - Group-specific transformations
- **Feature creation:**
  - Create derived variables
  - Binning/categorization
  - Interaction terms
  - Time-based features (if temporal)
- **Rolling windows (REQUIRED - time series):**
  - Moving averages (`df.rolling()`)
  - Exponentially weighted functions (`df.ewm()`)
  - Window-based feature extraction
- **Pivot tables:**
  - Cross-tabulation (`pd.crosstab()`)
  - Multi-dimensional aggregation

**Lecture Focus:**
- Feature engineering best practices
- Domain knowledge application
- Avoiding data leakage
- Creating interpretable features

**Exam Focus:**
- GroupBy aggregations
- Create key derived features
- **Rolling calculations (required - time series)**
- Time-based aggregations (by hour, day, month, etc.)

---

### Phase 6: Pattern Analysis & Advanced Visualization
**Lecture Topics:** 07 (Visualization), 08 (Aggregation)  
**Core Skills:**
- **Statistical summaries:**
  - Grouped statistics
  - Correlation analysis
  - Distribution comparisons
- **Advanced visualizations:**
  - Multi-panel plots (subplots)
  - Grouped visualizations (by category)
  - **Time series plots (required)**
  - Relationship visualizations
- **Pattern identification:**
  - **Trends over time (required)**
  - **Seasonal patterns (required)**
  - Relationships between variables
  - Anomalies and outliers
  - Temporal relationships

**Lecture Focus:**
- Tufte's principles in practice
- Storytelling with data
- Effective visualization choices
- Statistical interpretation
- Moving beyond initial exploration to deeper insights

**Exam Focus:**
- Create informative visualizations
- Identify key patterns
- Summarize findings
- **Note:** This phase builds on Phase 2's initial exploration with more sophisticated analysis

---

### Phase 7: Modeling Preparation
**Lecture Topics:** 10 (Modeling intro), 08 (Feature selection)  
**Core Skills:**
- **Train/test split:**
  - **Split data temporally (required for time series)**
  - Use time-based splitting (e.g., train on earlier data, test on later)
  - Avoid data leakage from future to past
- **Feature selection:**
  - Identify target variable
  - Select relevant features
  - Handle categorical variables (encoding if needed)
- **Data preparation:**
  - Scale features if needed
  - Create final modeling dataset
  - Handle class imbalance (if classification)

**Lecture Focus:**
- Proper train/test splitting strategies
- Feature selection techniques
- Data leakage prevention
- Model-appropriate preprocessing

**Exam Focus:**
- Create train/test split
- Prepare features for modeling
- Identify target and features

---

### Phase 8: Modeling
**Lecture Topics:** 10 (Modeling ecosystem)  
**Core Skills:**
- **Model selection:**
  - Choose appropriate model type (regression vs classification)
  - Start with simple models (linear regression)
  - Progress to more complex (Random Forest, XGBoost)
- **Model training:**
  - Fit models on training data
  - Use appropriate libraries (scikit-learn, XGBoost, statsmodels)
- **Model evaluation:**
  - Calculate performance metrics (R², accuracy, RMSE, etc.)
  - Compare train vs test performance
  - Identify overfitting
- **Model interpretation:**
  - Feature importance (if available)
  - Coefficient interpretation (for linear models)
  - Prediction analysis

**Lecture Focus:**
- Model selection rationale
- Understanding model assumptions
- Interpreting results
- When to use which model type
- Hyperparameter basics (time permitting)

**Exam Focus:**
- Train a model
- Evaluate performance
- Interpret basic results

---

### Phase 9: Results & Insights
**Lecture Topics:** 07 (Communication), All previous  
**Core Skills:**
- **Summarize findings:**
  - Key insights from EDA
  - Model performance summary
  - Important patterns discovered
- **Create final visualizations:**
  - Model performance plots
  - Key insight visualizations
  - Prediction visualizations
- **Documentation:**
  - Clear summary of process
  - Key decisions and rationale
  - Limitations and next steps

**Lecture Focus:**
- Effective communication of results
- Storytelling with data
- Acknowledging limitations
- Suggesting next steps

**Exam Focus:**
- Summarize key findings
- Create final summary visualizations
- Document process briefly

---

## Dataset-Specific Considerations

### NYC Taxi Trip Dataset
**Recommended for:** **Lecture 11** (comprehensive walkthrough)  
- **Time series focus:** Hourly/daily patterns, seasonal trends, day-of-week effects
- **Key wrangling:** Merge with zone lookup tables, combine multiple taxi types
- **Cleaning:** Outlier trips (24+ hour trips), invalid coordinates, missing timestamps
- **Modeling:** Predict fare amount or trip duration
- **Why lecture:** Large dataset allows demonstrating memory management, complex merging, rich temporal patterns

### California Wildfire Dataset
**Recommended for:** Not selected (has limitations)  
- **Time series focus:** Long-term trends (decades), seasonal patterns, year-over-year analysis
- **Key wrangling:** Aggregate by year/county, merge with population data (if available)
- **Cleaning:** Incomplete historical records, missing containment dates
- **Modeling:** Predict fire size or analyze trends
- **Why not selected:** Contains significant geospatial data (perimeter data, coordinates) which is not covered in this course's curriculum

### Chicago Beach Weather Sensors
**Recommended for:** **Final Exam** (focused, manageable)  
- **Time series focus:** Irregular sampling, sensor dropouts, resampling challenges
- **Key wrangling:** Merge multiple sensor streams, resample to regular intervals
- **Cleaning:** Handle sensor dropouts, validate sensor readings, data type issues
- **Modeling:** Predict water conditions or temperature
- **Why exam:** Smaller dataset, clear time series challenges (irregular → regular), focused scope

### MIT-BIH Arrhythmia Database
**Recommended for:** Not selected (requires specialized tools)  
- **Time series focus:** High-frequency signals (360 Hz), event detection, signal processing
- **Key wrangling:** Join signal data with annotations, handle multi-lead data
- **Cleaning:** Signal artifacts, missing annotations
- **Modeling:** Classification (arrhythmia detection) or feature extraction
- **Why not selected:** Requires specialized package (`wfdb`) for reading PhysioNet format, which adds unnecessary complexity and dependency beyond core pandas/numpy skills

---

## Lecture vs Exam Differences

### Lecture 11 (Detailed Walkthrough)
- **Time:** Full lecture period (90+ minutes)
- **Depth:** Detailed explanations at each step
- **Demonstrations:** Live coding with explanations
- **Best practices:** Emphasize proper workflow and documentation
- **Troubleshooting:** Show common issues and solutions
- **Extensions:** Mention advanced techniques and next steps

### Final Exam (Take-Home Assessment)
- **Format:** Take-home exam with structured tasks/questions
- **Depth:** Focus on core skills, less explanation
- **Scope:** Cover all phases with realistic time allocation
- **Assessment:** 
  - **70% Automated grading** (code execution, output validation)
  - **30% Writeup evaluation** (tables, plots, model results, interpretation)
- **Documentation:** Clear process documentation with rationale
- **Deliverables:** 
  - Code notebook/script with all analysis
  - Written report with key findings, visualizations, and model results

---

## Common Core Skills Checklist

Both lecture and exam should demonstrate:

- [ ] Data loading and initial inspection
- [ ] Missing data identification and handling
- [ ] Data type conversions
- [ ] Outlier detection and handling
- [ ] Merging/joining datasets
- [ ] Reshaping data (pivot/melt)
- [ ] GroupBy aggregations
- [ ] **Time series operations (REQUIRED)**
- [ ] Creating visualizations
- [ ] Feature engineering
- [ ] Train/test splitting
- [ ] Model training and evaluation
- [ ] Results interpretation and communication

---

## Implementation Notes

1. **Dataset Selection:**
   - **Lecture 11:** NYC Taxi Trip Dataset (comprehensive, large-scale, rich patterns, no special tools needed)
   - **Final Exam:** Chicago Beach Weather Sensors (focused, manageable, clear time series challenges, standard formats)
   - **Excluded:** CA Wildfires (geospatial data not covered), MIT-BIH (requires specialized `wfdb` package)
   - Both selected datasets allow demonstration of all core skills using only standard pandas/numpy/matplotlib tools

2. **Progressive Complexity:**
   - Start simple (basic operations)
   - Build complexity (combining operations)
   - End with modeling (synthesis of all skills)

3. **Real-World Authenticity:**
   - Include messy data (missing values, outliers, inconsistencies)
   - Require multiple data sources (merging)
   - Present realistic questions/problems to solve

4. **Skill Integration:**
   - Each phase builds on previous phases
   - Techniques from multiple lectures used together
   - Demonstrate how skills work in combination

5. **Documentation:**
   - Both should include clear documentation
   - Explain decisions and rationale
   - Show professional data science workflow

---

## Grading Rubric Structure

The final exam will be graded using a combination of **automated testing (70%)** and **writeup evaluation (30%)**. The rubric aligns with the project phases:

### Automated Grading (70% of total score)

**Phase 1: Setup & Data Acquisition (5%)**
- [ ] Correct library imports
- [ ] Data loaded successfully
- [ ] Initial inspection commands produce expected output

**Phase 2: Data Exploration (5%)**
- [ ] Summary statistics calculated correctly
- [ ] Missing value counts accurate
- [ ] Basic visualizations generated (code executes)

**Phase 3: Data Cleaning (10%)**
- [ ] Missing data handled appropriately
- [ ] Outliers identified and handled
- [ ] Data types converted correctly
- [ ] Duplicates removed (if applicable)
- [ ] Data validation checks pass

**Phase 4: Data Wrangling & Transformation (15%)**
- [ ] Merges/joins executed correctly
- [ ] Data reshaping operations work
- [ ] **Datetime parsing successful**
- [ ] **Datetime index set correctly**
- [ ] **Temporal features extracted** (hour, day, month, etc.)

**Phase 5: Feature Engineering & Aggregation (10%)**
- [ ] GroupBy operations produce correct results
- [ ] Aggregations calculated correctly
- [ ] **Rolling window calculations work**
- [ ] Derived features created successfully

**Phase 6: Exploratory Data Analysis (5%)**
- [ ] Statistical summaries calculated
- [ ] Visualizations generate without errors
- [ ] **Time series plots created**

**Phase 7: Modeling Preparation (5%)**
- [ ] **Temporal train/test split implemented correctly**
- [ ] Features prepared appropriately
- [ ] No data leakage detected

**Phase 8: Modeling (10%)**
- [ ] Model trained successfully
- [ ] Performance metrics calculated correctly
- [ ] Train/test performance compared
- [ ] Model outputs match expected format

**Phase 9: Results (5%)**
- [ ] Final visualizations generated
- [ ] Results tables formatted correctly
- [ ] Code executes end-to-end without errors

### Writeup Evaluation (30% of total score)

**Documentation Quality (10%)**
- Clear explanation of process and decisions
- Rationale for key choices (cleaning, modeling, etc.)
- Professional presentation

**Visualizations & Tables (10%)**
- **Time series visualizations** clearly show trends/patterns
- Tables formatted appropriately
- Visualizations follow best practices (labels, titles, clarity)
- Model results presented clearly

**Interpretation & Insights (10%)**
- Key findings from EDA summarized
- **Time series patterns identified and explained**
- Model performance interpreted correctly
- Limitations acknowledged
- Conclusions supported by evidence

### Automated Testing Strategy

**Test Structure:**
- Unit tests for individual functions/operations
- Integration tests for complete workflows
- Output validation (DataFrame shapes, column names, value ranges)
- Visualization existence checks (files created, no errors)
- Model output validation (metrics within expected ranges)

**Testable Elements:**
- Code execution (no syntax errors, imports work)
- Data transformations (shapes, dtypes, values)
- Statistical calculations (means, sums, counts)
- Model outputs (predictions, metrics)
- File outputs (CSVs, plots saved)

**Non-Testable Elements (Writeup):**
- Interpretation quality
- Visualization aesthetics
- Writing clarity
- Domain knowledge application
- Critical thinking

---

## Time Series Requirements

**All datasets support time series analysis and students must demonstrate:**

1. **Datetime handling:**
   - Parse datetime columns correctly
   - Set datetime index
   - Handle time zones (if applicable)

2. **Temporal feature extraction:**
   - Extract hour, day, month, year, day_of_week
   - Create time-based categorical features

3. **Time series operations:**
   - Resample irregular data (if needed)
   - Rolling window calculations
   - Time-based aggregations

4. **Time series visualization:**
   - Plot trends over time
   - Identify seasonal patterns
   - Show temporal relationships

5. **Temporal modeling:**
   - Use time-based train/test splits
   - Include temporal features in models
   - Consider temporal dependencies

---

## Lecture Format: Notebook-Based Walkthrough

**Structure:** Multiple Jupyter notebooks (4-5 notebooks) organized by workflow phase

**Proposed Notebook Structure:**

1. **Notebook 1: Setup, Exploration & Cleaning** (Phases 1-3)
   - Data loading and initial inspection
   - Basic exploration and visualization
   - Data cleaning workflow
   - *Focus: Getting data ready for analysis*

2. **Notebook 2: Wrangling & Feature Engineering** (Phases 4-5)
   - Merging/joining datasets
   - Time series datetime handling
   - Reshaping operations
   - Feature engineering and aggregations
   - *Focus: Transforming data for analysis*

3. **Notebook 3: Pattern Analysis & Modeling Prep** (Phases 6-7)
   - Advanced visualizations
   - Pattern identification (trends, seasonality)
   - Statistical summaries
   - Train/test splitting (temporal)
   - Feature preparation
   - *Focus: Understanding patterns and preparing for modeling*

4. **Notebook 4: Modeling & Results** (Phases 8-9)
   - Model training and evaluation
   - Model interpretation
   - Final visualizations
   - Results summary and insights
   - *Focus: Building models and communicating results*

**Alternative (5 notebooks):** Split Phase 3 (cleaning) into its own notebook if it's extensive.

**Format Benefits:**
- Interactive, executable examples
- Students can run code and see results
- Easy to follow along during lecture
- Can be converted to markdown for static viewing
- Supports both live demo and self-study

---

## Feasibility Assessment

**Overall:** ✅ **Feasible** - The outline represents a realistic data science workflow

**Strengths:**
- Logical progression from raw data to insights
- All phases build on previous work
- Time series operations included as required component
- Clear separation of automated vs. manual grading
- Realistic scope for take-home exam

**Considerations:**
- **Exam timing:** Take-home format allows for realistic time allocation
- **Dataset size:** Chicago Beach Sensors is appropriately sized for exam
- **Complexity:** NYC Taxi provides rich patterns for lecture without being overwhelming
- **Time series:** All datasets support temporal analysis as required

**Potential Challenges:**
- Automated grading setup requires careful test design
- Writeup evaluation needs clear rubric to ensure consistency
- Notebook format for lecture requires good organization

---

## Next Steps

1. ✅ **Dataset selection** - NYC Taxi (lecture), Chicago Beach Sensors (exam)
2. **Create lecture notebooks** (4-5 notebooks following workflow phases)
3. **Develop exam questions** aligned with this flow
4. **Build automated grading tests** for 70% of score
5. **Create writeup rubric** for 30% of score
6. **Prepare sample solutions** for both lecture and exam
7. **Test automated grading** on sample submissions

