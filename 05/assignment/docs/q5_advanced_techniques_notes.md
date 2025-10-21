# Q5 Missing Data Analysis - Advanced Techniques Notes

## Analysis Summary

The Q5 missing data analysis was completed successfully using techniques covered in lectures 1-5:

### Techniques Implemented (Covered in Lectures)
1. **Missing value detection** - Using `df.isnull().sum()` and pandas utilities
2. **Mean imputation** - Filling missing values with column mean
3. **Median imputation** - Filling missing values with column median
4. **Forward fill** - Propagating last valid observation forward
5. **Dropping missing data** - Removing rows with missing values (selective and complete)

### Results
- **Original dataset**: 10,000 patients, 18 columns
- **Missing data**: 8 columns with missing values (3.7% - 14.7% missing per column)
- **Final clean dataset**: 10,000 patients retained (100% retention)
- **Strategy**: Hybrid approach - dropped rows with missing critical fields, imputed numeric measurements with median

---

## Advanced Techniques NOT Covered in Lectures 1-5

The following advanced missing data techniques were not covered in lectures 1-5 but are commonly used in production data science:

### 1. Multiple Imputation (MICE)
**What it is**: Multiple Imputation by Chained Equations
- Creates multiple imputed datasets (typically 5-10)
- Performs analysis on each dataset separately
- Combines results using Rubin's rules
- Accounts for uncertainty in imputation

**When to use**:
- When missing data patterns are complex
- When you need to account for imputation uncertainty
- For inference and hypothesis testing

**Python implementation**:
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

### 2. K-Nearest Neighbors (KNN) Imputation
**What it is**: Imputes missing values using the mean/median of K nearest neighbors
- Uses similarity between samples to impute values
- Considers relationships between features
- Better than simple mean/median for correlated features

**When to use**:
- When features are correlated
- When patterns in data suggest similar samples should have similar values
- For clinical data where patients with similar characteristics likely have similar measurements

**Python implementation**:
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

### 3. Model-Based Imputation
**What it is**: Use machine learning models to predict missing values
- Random Forest, Linear Regression, or other algorithms
- Learns patterns from complete data to predict missing values
- Can capture complex relationships

**When to use**:
- When missing data mechanism is MAR (Missing At Random)
- When you have sufficient complete data to train models
- For important features where accuracy matters

**Python implementation**:
```python
from sklearn.ensemble import RandomForestRegressor

# Train model on complete cases
complete_df = df.dropna(subset=['bmi'])
X = complete_df[['age', 'sex', 'height']]  # predictor features
y = complete_df['bmi']

model = RandomForestRegressor()
model.fit(X, y)

# Predict missing values
missing_df = df[df['bmi'].isnull()]
df.loc[df['bmi'].isnull(), 'bmi'] = model.predict(missing_df[['age', 'sex', 'height']])
```

### 4. Missing Indicator Variables
**What it is**: Create binary indicator columns showing which values were missing
- Allows models to learn patterns in missingness
- Preserves information about which values were imputed
- Enables sensitivity analyses

**When to use**:
- Always recommended for production ML pipelines
- When missing data patterns may be informative
- For auditing and quality control

**Python implementation**:
```python
from sklearn.impute import SimpleImputer, MissingIndicator

# Create missing indicators
indicator = MissingIndicator()
missing_mask = indicator.fit_transform(df)
missing_df = pd.DataFrame(missing_mask, columns=[f'{col}_missing' for col in df.columns])

# Impute and combine
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df_final = pd.concat([df_imputed, missing_df], axis=1)
```

### 5. Expectation-Maximization (EM) Algorithm
**What it is**: Iterative statistical method for finding maximum likelihood estimates
- E-step: Estimates missing values given current parameters
- M-step: Updates parameters given current estimates
- Converges to optimal imputation

**When to use**:
- When data follows known distributions (e.g., multivariate normal)
- For structural equation modeling
- When statistical rigor is important

### 6. Pattern-Based Imputation
**What it is**: Identifies patterns in missingness and uses different strategies per pattern
- Groups data by missing data patterns
- Applies tailored imputation to each pattern
- More sophisticated than blanket approach

**When to use**:
- When different missing patterns have different causes
- In complex datasets with multiple data collection procedures
- For clinical trials with protocol deviations

### 7. Time-Series Specific Methods
**What it is**: Methods designed for temporal data
- Linear interpolation
- Spline interpolation
- Seasonal decomposition
- ARIMA-based imputation

**When to use**:
- For longitudinal clinical trial data
- When measurements have temporal dependencies
- For sensor data or continuous monitoring

**Python implementation**:
```python
# Linear interpolation
df['measurement'] = df['measurement'].interpolate(method='linear')

# Time-based interpolation
df['measurement'] = df['measurement'].interpolate(method='time')
```

---

## Comparison to Implemented Approach

### What We Did (Lectures 1-5)
✓ Simple, interpretable methods
✓ Fast computation
✓ Good for initial analysis
✓ Appropriate for low missing data rates (<5% per column)
✓ Suitable for independent observations

### Advanced Techniques Add
✓ Account for uncertainty in imputation
✓ Model complex relationships between features
✓ Better handle MAR mechanisms
✓ Preserve information about missingness
✓ Improve prediction accuracy for ML models

---

## Recommendations for Production

For a production clinical trial analysis, consider:

1. **Start with simple methods** (as we did) for exploratory analysis
2. **Add missing indicators** to preserve information about imputation
3. **Use KNN or model-based imputation** for features with strong correlations
4. **Implement multiple imputation** if performing statistical inference
5. **Document and version** all imputation decisions
6. **Perform sensitivity analyses** comparing different imputation strategies
7. **Validate** imputation quality using holdout data

---

## Missing Data Mechanisms (Theory)

Understanding why data is missing is crucial:

1. **MCAR (Missing Completely At Random)**: Missingness is independent of observed and unobserved data
   - Simplest assumption
   - Any imputation method works
   - Least common in practice

2. **MAR (Missing At Random)**: Missingness depends on observed data but not unobserved data
   - Most common assumption
   - Advanced imputation methods recommended
   - Can be addressed with proper modeling

3. **MNAR (Missing Not At Random)**: Missingness depends on unobserved data
   - Most problematic
   - Requires domain knowledge
   - May need selection models or pattern mixture models

Our analysis assumes **MAR** - that missing clinical measurements depend on observed demographics but not on the unobserved measurement itself.

---

## References for Further Learning

- Little, R. J., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.)
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.)
- Scikit-learn documentation: [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)
- Pandas documentation: [Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)

---

*Note: This document supplements the Q5 assignment by identifying advanced techniques beyond the scope of lectures 1-5.*
