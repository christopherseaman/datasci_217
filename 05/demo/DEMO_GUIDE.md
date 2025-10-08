Demo Guide - Lecture 5: Data Cleaning and Preparation

# Demo 1: Missing Data Detective Work

**File:** [demo1_missing_data.ipynb](https://github.com/christopherseaman/datasci_217/blob/main/05/demo/demo1_missing_data.ipynb)

**Objective**: Master missing data detection, analysis, and handling strategies.

**Key Concepts**: Missing data patterns, fillna strategies, dropna decisions

## Demo Flow

Students work through patient data with various missing values:

1. **Create messy dataset** - Patient data with missing ages, blood pressure, cholesterol, test dates
2. **Detect and visualize** - Use `.isnull().sum()`, calculate percentages, create heatmap visualization
3. **Strategic handling** - Apply different strategies:
   - Median fill for age (numerical, robust to outliers)
   - Forward fill for test dates (temporal/sequential data)
   - Drop rows missing both BP and cholesterol (critical data)

**Key Teaching Points**:
- Missing data is normal in real-world datasets
- Visualization reveals patterns (random vs systematic missingness)
- Different strategies for different data types and contexts
- Document why you chose each strategy

**Expected Outcomes**:
- Students can quantify missingness (count and percentage)
- Students understand when to fill vs drop
- Students recognize that one strategy doesn't fit all columns

---

# Demo 2: Data Transformation and Cleaning Pipeline

**File:** [demo2_transformations.ipynb](https://github.com/christopherseaman/datasci_217/blob/main/05/demo/demo2_transformations.ipynb)

**Objective**: Build a complete data cleaning pipeline with transformations, custom functions, and categorical encoding.

**Key Concepts**: Replace, rename, astype, apply/map, cut/qcut, categorical data, dummy variables

## Demo Flow

Students work through messy survey data with multiple quality issues:

1. **Load messy survey data** - Inconsistent formatting, sentinel values (-999), encoding issues
2. **Clean column names** - Lowercase, strip whitespace, replace spaces with underscores
3. **Handle sentinel values** - Replace -999 with NaN, use `errors='coerce'` for invalid data, fill strategically
4. **Standardize text** - Use `.str.strip()`, `.str.title()`, and mapping dictionaries
5. **Apply custom functions**:
   - `.apply()` with custom function to score satisfaction levels
   - `.map()` with dictionary to rank education levels
   - Lambda functions for quick calculations (income in thousands)
6. **Create categories** - Use `pd.cut()` for age groups, `pd.qcut()` for income levels
7. **Create dummy variables** - Use `pd.get_dummies()` for one-hot encoding regions
8. **Categorical dtype**:
   - Show memory usage before/after conversion to categorical
   - Access categories with `.cat.categories`
   - Access codes with `.cat.codes`

**Key Teaching Points**:
- `.apply()` enables custom transformation logic
- `.map()` is perfect for categorical mappings (like scoring or ranking)
- Lambda functions great for one-liners, named functions better for complex logic
- `cut()` = equal-width bins, `qcut()` = equal-frequency bins
- Dummy variables prepare categorical data for machine learning
- Categorical dtype saves memory when you have repeated string values
- The `.get()` method in dictionaries prevents KeyErrors with defaults

**Expected Outcomes**:
- Students can apply custom functions to transform data
- Students understand difference between .apply() and .map()
- Students can create categorical variables for analysis
- Students can encode categorical data for modeling
- Students recognize when categorical dtype is beneficial

---

# Demo 3: Complete Data Cleaning Workflow

**File:** [demo3_workflow.ipynb](https://github.com/christopherseaman/datasci_217/blob/main/05/demo/demo3_workflow.ipynb)

**Objective**: Put it all together - a realistic, end-to-end cleaning pipeline.

**Key Concepts**: Detect → Handle → Validate → Transform → Save

## Demo Flow

Students work through e-commerce order data with multiple simultaneous issues:

1. **Load dirty data** - E-commerce orders with inconsistent names, negative prices, missing values, invalid dates
2. **Detect issues** - Systematically audit: missing values, duplicates, negative prices, invalid dates
3. **Handle systematically** - Apply cleaning steps in sequence:
   - Standardize customer/product names
   - Replace negative prices with NaN, fill with median
   - Fill missing quantities with 1
   - Convert dates with `errors='coerce'`
   - Standardize status values
4. **Validate cleaning** - Verify each issue was resolved, check data types
5. **Transform for analysis** - Add calculated fields (total_price), extract time periods
6. **Detect outliers** - IQR method for finding unusual transactions
7. **Save results** - Cleaned data, summaries, and data quality report

**Key Teaching Points**:
- Always copy original data before modifying (`.copy()`)
- Systematic approach: detect before handling, validate after
- `.loc[]` for conditional replacement is powerful
- Document decisions in a data quality report
- Save intermediate results and final outputs
- The workflow is iterative: detect → handle → validate → repeat

**Expected Outcomes**:
- Students can build end-to-end cleaning pipeline
- Students validate that cleaning achieved its goals
- Students document their cleaning decisions
- Students save cleaned data and create audit trails

---

# Key Takeaways Across All Demos

**Demo 1 - Missing Data**:
- Quantify and visualize missing patterns first
- Different fill strategies: median (numerical), forward fill (temporal), drop (critical missing)
- Always understand WHY data is missing before deciding how to handle it

**Demo 2 - Transformations**:
- Clean column names first (lowercase, no spaces)
- Handle sentinel values (-999, "N/A") before analysis
- `.apply()` and `.map()` enable custom transformations
- Standardize text data (capitalization, spelling)
- Create categories with cut/qcut for analysis
- Dummy variables prepare data for modeling
- Categorical dtype saves memory for repeated values

**Demo 3 - Complete Workflow**:
- Systematic process: detect → handle → validate → transform → save
- Always validate cleaning worked
- Calculate derived fields after cleaning
- Detect outliers with IQR method
- Save cleaned data, summaries, and reports

**Best Practices Across All Demos**:
1. Never modify original data - always use `.copy()`
2. Document every cleaning decision
3. Validate at each step
4. Save intermediate results
5. Create audit trails (reports, logs)

**Common Student Mistakes to Watch For**:
- Forgetting to use `.copy()` and modifying original data
- Filling all missing values the same way (median for everything)
- Not validating that cleaning actually worked
- Using `.apply()` when vectorized operations would be faster
- Forgetting that `cut()` uses explicit bins while `qcut()` uses quantiles
- Not understanding categorical dtype vs dummy variables (when to use each)

**Next Steps for Students**:
- Practice with their own messy datasets
- Build reusable cleaning functions
- Create data quality checklists
- Develop validation test suites
