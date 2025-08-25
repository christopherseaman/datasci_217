# Lecture 05: Pandas Fundamentals + Data Analysis Workflow

**Duration**: 4.5 hours  
**Focus**: Series/DataFrame mastery, labeled data operations, complete analysis workflow, real-world data handling

## Learning Objectives

By the end of this lecture, students will:
- Master Pandas Series and DataFrame for labeled data manipulation
- Implement complete data analysis workflows from raw data to insights
- Handle real-world data quality issues systematically
- Perform complex data operations with intuitive, readable code
- Build reproducible data analysis pipelines for production use

---

## Part 1: The Labeled Data Revolution (50 minutes)

### Opening Hook: "Why Labels Change Everything"

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The NumPy way - positional, error-prone
sales_array = np.array([
    [150, 200, 175],  # What do these numbers mean?
    [220, 180, 190],  # Which is which?
    [300, 250, 280]   # Easy to make mistakes!
])

# Calculate Q2 growth - but which is Q2?
q1_total = sales_array[:, 0].sum()  # Assuming column 0 is Q1
q2_total = sales_array[:, 1].sum()  # Assuming column 1 is Q2
growth_rate = (q2_total - q1_total) / q1_total

print(f"NumPy approach - Q2 Growth: {growth_rate:.2%}")
print("But... are we sure column 1 is Q2? What if data changed?")

# The Pandas way - explicit, self-documenting
sales_df = pd.DataFrame({
    'Q1_2023': [150, 220, 300],
    'Q2_2023': [200, 180, 250], 
    'Q3_2023': [175, 190, 280]
}, index=['Product_A', 'Product_B', 'Product_C'])

print("\nPandas DataFrame:")
print(sales_df)

# Calculate growth - explicit and safe
q2_growth = (sales_df['Q2_2023'].sum() - sales_df['Q1_2023'].sum()) / sales_df['Q1_2023'].sum()
print(f"\nPandas approach - Q2 Growth: {q2_growth:.2%}")
print("Clear, explicit, and self-documenting!")

# Show the power of labels
print(f"\nProduct B Q1 performance: ${sales_df.loc['Product_B', 'Q1_2023']:,}")
print(f"Q3 performance across all products:")
print(sales_df['Q3_2023'])
```

### Series: The Building Block

```python
# Series creation and manipulation
revenue_series = pd.Series([1000, 1200, 800, 1500, 900], 
                          index=['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                          name='Monthly_Revenue')

print("Revenue Series:")
print(revenue_series)
print(f"\nSeries info:")
print(f"Name: {revenue_series.name}")
print(f"Index: {list(revenue_series.index)}")
print(f"Values: {list(revenue_series.values)}")
print(f"Data type: {revenue_series.dtype}")

# Series operations maintain alignment
costs_series = pd.Series([600, 700, 500, 900, 550],
                        index=['Feb', 'Mar', 'Jan', 'May', 'Apr'])  # Different order!

# Automatic alignment by index
profit_series = revenue_series - costs_series
print(f"\nProfit calculation (automatic alignment):")
print(profit_series)

# Series behaves like a dictionary
print(f"\nMarch profit: ${profit_series['Mar']:,}")
print(f"Months with profit > $400: {list(profit_series[profit_series > 400].index)}")

# Series mathematical operations
print(f"\nTotal profit: ${profit_series.sum():,}")
print(f"Average monthly profit: ${profit_series.mean():,.2f}")
print(f"Best performing month: {profit_series.idxmax()} (${profit_series.max():,})")

# Advanced Series operations
def analyze_series_performance(series, name="Series"):
    """Comprehensive series analysis."""
    stats = {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'range': series.max() - series.min(),
        'cv': series.std() / series.mean()  # Coefficient of variation
    }
    
    print(f"\n=== {name} Analysis ===")
    for metric, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"{metric:>8}: {value:8.2f}")
        else:
            print(f"{metric:>8}: {value}")
    
    return stats

revenue_stats = analyze_series_performance(revenue_series, "Monthly Revenue")
```

---

## Part 2: DataFrame Mastery (70 minutes)

### DataFrame Creation and Structure

```python
# Multiple ways to create DataFrames
# Method 1: From dictionary
company_data = {
    'Company': ['Apple', 'Google', 'Microsoft', 'Amazon', 'Meta'],
    'Revenue_B': [394.3, 282.8, 198.3, 513.9, 117.9],
    'Employees_K': [164, 174, 221, 1608, 87],
    'Founded': [1976, 1998, 1975, 1994, 2004],
    'Sector': ['Technology', 'Technology', 'Technology', 'E-commerce', 'Technology']
}

df = pd.DataFrame(company_data)
print("DataFrame from dictionary:")
print(df)

# Method 2: From lists with column names
data_lists = [
    ['Apple', 394.3, 164, 1976, 'Technology'],
    ['Google', 282.8, 174, 1998, 'Technology'],
    ['Microsoft', 198.3, 221, 1975, 'Technology'],
    ['Amazon', 513.9, 1608, 1994, 'E-commerce'],
    ['Meta', 117.9, 87, 2004, 'Technology']
]

df2 = pd.DataFrame(data_lists, 
                  columns=['Company', 'Revenue_B', 'Employees_K', 'Founded', 'Sector'])

# Method 3: From NumPy array with custom index
numeric_data = np.random.randn(5, 4)
df3 = pd.DataFrame(numeric_data,
                  index=['Company_A', 'Company_B', 'Company_C', 'Company_D', 'Company_E'],
                  columns=['Q1', 'Q2', 'Q3', 'Q4'])

print(f"\nDataFrame shapes:")
print(f"df: {df.shape}, df2: {df2.shape}, df3: {df3.shape}")

# DataFrame inspection
def inspect_dataframe(df, name="DataFrame"):
    """Comprehensive DataFrame inspection."""
    print(f"\n=== {name} Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {list(df.index)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Memory usage: {df.memory_usage().sum():,} bytes")
    
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    print(f"\nBasic statistics:")
    print(df.describe())

inspect_dataframe(df, "Company Data")
```

### Data Selection and Indexing Mastery

```python
# Single column selection (returns Series)
companies = df['Company']
revenues = df.Revenue_B  # Attribute access (if no spaces in name)

print("Company names (Series):")
print(companies)
print(f"Type: {type(companies)}")

# Multiple column selection (returns DataFrame)
financial_data = df[['Company', 'Revenue_B', 'Employees_K']]
print("\nFinancial subset (DataFrame):")
print(financial_data)

# Row selection by position (.iloc)
print(f"\nFirst row (by position):")
print(df.iloc[0])

print(f"\nFirst 3 companies, first 3 columns:")
print(df.iloc[0:3, 0:3])

# Row selection by label (.loc)
# First, let's set Company as index for label-based selection
df_indexed = df.set_index('Company')
print(f"\nDataFrame with Company as index:")
print(df_indexed)

print(f"\nApple's data (by label):")
print(df_indexed.loc['Apple'])

print(f"\nRevenue data for Apple and Google:")
print(df_indexed.loc[['Apple', 'Google'], 'Revenue_B'])

# Boolean indexing - the most powerful selection method
high_revenue_companies = df[df['Revenue_B'] > 250]
print(f"\nCompanies with revenue > $250B:")
print(high_revenue_companies)

# Complex boolean conditions
tech_giants = df[
    (df['Sector'] == 'Technology') & 
    (df['Revenue_B'] > 200) & 
    (df['Founded'] < 2000)
]
print(f"\nEstablished tech giants (Tech sector, >$200B revenue, founded before 2000):")
print(tech_giants)

# Query method for readable complex conditions
young_companies = df.query("Founded > 1990 and Revenue_B > 100")
print(f"\nYoung successful companies (founded after 1990, >$100B revenue):")
print(young_companies)

# Selection with string methods
tech_companies = df[df['Sector'].str.contains('Tech')]
print(f"\nTechnology companies:")
print(tech_companies[['Company', 'Sector']])
```

### Advanced DataFrame Operations

```python
# Adding calculated columns
df['Revenue_per_Employee'] = df['Revenue_B'] * 1000 / df['Employees_K']  # Million per employee
df['Company_Age'] = 2024 - df['Founded']
df['Size_Category'] = pd.cut(df['Employees_K'], 
                            bins=[0, 100, 500, 1000, float('inf')],
                            labels=['Small', 'Medium', 'Large', 'Massive'])

print("DataFrame with calculated columns:")
print(df[['Company', 'Revenue_per_Employee', 'Company_Age', 'Size_Category']])

# Sorting operations
revenue_sorted = df.sort_values('Revenue_B', ascending=False)
print(f"\nCompanies sorted by revenue (descending):")
print(revenue_sorted[['Company', 'Revenue_B']])

# Multi-level sorting
multi_sorted = df.sort_values(['Sector', 'Revenue_B'], ascending=[True, False])
print(f"\nSorted by sector, then revenue within sector:")
print(multi_sorted[['Company', 'Sector', 'Revenue_B']])

# Aggregation operations
sector_analysis = df.groupby('Sector').agg({
    'Revenue_B': ['mean', 'sum', 'count'],
    'Employees_K': ['mean', 'sum'],
    'Founded': 'min',
    'Company_Age': 'mean'
}).round(2)

print(f"\nSector analysis:")
print(sector_analysis)

# Apply custom functions
def efficiency_score(row):
    """Calculate a custom efficiency score."""
    revenue_score = row['Revenue_B'] / 500  # Normalize by max possible
    employee_efficiency = row['Revenue_per_Employee'] / 3.0  # Normalize
    age_factor = 1 + (50 - row['Company_Age']) / 50  # Bonus for longevity
    
    return (revenue_score + employee_efficiency) * age_factor

df['Efficiency_Score'] = df.apply(efficiency_score, axis=1)
print(f"\nEfficiency rankings:")
print(df[['Company', 'Efficiency_Score']].sort_values('Efficiency_Score', ascending=False))
```

---

## Part 3: Real-World Data Challenges (80 minutes)

### Missing Data Handling Strategies

```python
# Create realistic dataset with missing values
np.random.seed(42)

# Simulate customer data with realistic missing patterns
n_customers = 1000
customer_data = {
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(40, 12, n_customers),
    'income': np.random.lognormal(10.5, 0.5, n_customers),
    'purchase_amount': np.random.exponential(200, n_customers),
    'satisfaction_score': np.random.beta(3, 1, n_customers) * 100,
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers),
    'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], 
                                        n_customers, p=[0.6, 0.3, 0.1])
}

customers_df = pd.DataFrame(customer_data)

# Introduce realistic missing data patterns
# Age missing for privacy reasons (random)
age_missing_idx = np.random.choice(n_customers, size=int(0.05 * n_customers), replace=False)
customers_df.loc[age_missing_idx, 'age'] = np.nan

# Income missing more often for younger people (not random)
young_customers = customers_df['age'] < 30
income_missing_prob = np.where(young_customers, 0.15, 0.03)
income_missing = np.random.random(n_customers) < income_missing_prob
customers_df.loc[income_missing, 'income'] = np.nan

# Satisfaction scores missing when purchase amount is very low (systematic)
low_purchase = customers_df['purchase_amount'] < 50
customers_df.loc[low_purchase, 'satisfaction_score'] = np.nan

print("Dataset with missing values:")
print(customers_df.info())

# Analyze missing data patterns
def analyze_missing_data(df):
    """Comprehensive missing data analysis."""
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Data_Type': df.dtypes
    })
    
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    print("Missing Data Analysis:")
    print(missing_stats)
    
    # Check for patterns in missing data
    print(f"\nMissing data patterns:")
    missing_combinations = df.isnull().groupby(df.columns.tolist()).size().sort_values(ascending=False)
    print(missing_combinations.head(10))
    
    return missing_stats

missing_analysis = analyze_missing_data(customers_df)

# Missing data handling strategies
class MissingDataHandler:
    """Professional missing data handling toolkit."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.strategies_applied = []
    
    def drop_missing_threshold(self, threshold=0.5):
        """Drop columns with missing percentage above threshold."""
        missing_pct = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)
            self.strategies_applied.append(f"Dropped columns with >{threshold*100}% missing: {columns_to_drop}")
        
        return self
    
    def impute_numerical_median(self, columns=None):
        """Impute numerical columns with median values."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if self.df[col].isnull().any():
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)
                self.strategies_applied.append(f"Imputed {col} with median: {median_value:.2f}")
        
        return self
    
    def impute_categorical_mode(self, columns=None):
        """Impute categorical columns with mode values."""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        
        for col in columns:
            if self.df[col].isnull().any():
                mode_value = self.df[col].mode().iloc[0]
                self.df[col].fillna(mode_value, inplace=True)
                self.strategies_applied.append(f"Imputed {col} with mode: {mode_value}")
        
        return self
    
    def impute_with_groups(self, target_col, group_cols):
        """Impute using group-wise statistics."""
        group_medians = self.df.groupby(group_cols)[target_col].median()
        
        # Create a function to impute based on group
        def impute_by_group(row):
            if pd.isna(row[target_col]):
                group_key = tuple(row[col] for col in group_cols)
                if group_key in group_medians.index:
                    return group_medians[group_key]
                else:
                    return self.df[target_col].median()  # Fallback to overall median
            return row[target_col]
        
        self.df[target_col] = self.df.apply(impute_by_group, axis=1)
        self.strategies_applied.append(f"Imputed {target_col} using group statistics based on {group_cols}")
        
        return self
    
    def add_missing_indicators(self, columns=None):
        """Add binary indicators for missing values."""
        if columns is None:
            columns = self.df.columns[self.df.isnull().any()]
        
        for col in columns:
            if self.df[col].isnull().any():
                indicator_col = f"{col}_was_missing"
                self.df[indicator_col] = self.df[col].isnull().astype(int)
                self.strategies_applied.append(f"Added missing indicator: {indicator_col}")
        
        return self
    
    def summary(self):
        """Print summary of applied strategies."""
        print(f"\nMissing Data Handling Summary:")
        print(f"Original shape: {self.original_shape}")
        print(f"Final shape: {self.df.shape}")
        print(f"Strategies applied:")
        for i, strategy in enumerate(self.strategies_applied, 1):
            print(f"  {i}. {strategy}")
        
        return self.df

# Apply missing data handling
handler = MissingDataHandler(customers_df)
cleaned_df = (handler
              .add_missing_indicators(['income', 'satisfaction_score'])
              .impute_with_groups('income', ['region', 'subscription_type'])
              .impute_numerical_median(['age', 'satisfaction_score'])
              .impute_categorical_mode()
              .summary())

print(f"\nFinal dataset info:")
print(cleaned_df.info())
```

### Data Type Optimization and Memory Management

```python
# Demonstrate data type optimization for large datasets
def optimize_dataframe_memory(df, verbose=True):
    """Optimize DataFrame memory usage by adjusting data types."""
    original_memory = df.memory_usage().sum()
    
    optimized_df = df.copy()
    
    # Optimize integer columns
    for col in optimized_df.select_dtypes(include=['int64']).columns:
        col_min = optimized_df[col].min()
        col_max = optimized_df[col].max()
        
        if col_min >= 0:  # Unsigned integers
            if col_max <= np.iinfo(np.uint8).max:
                optimized_df[col] = optimized_df[col].astype(np.uint8)
            elif col_max <= np.iinfo(np.uint16).max:
                optimized_df[col] = optimized_df[col].astype(np.uint16)
            elif col_max <= np.iinfo(np.uint32).max:
                optimized_df[col] = optimized_df[col].astype(np.uint32)
        else:  # Signed integers
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
    
    # Optimize float columns
    for col in optimized_df.select_dtypes(include=['float64']).columns:
        # Check if conversion to float32 loses precision
        float32_version = optimized_df[col].astype(np.float32)
        if np.allclose(optimized_df[col].values, float32_version.values, equal_nan=True):
            optimized_df[col] = float32_version
    
    # Convert string columns to categories if beneficial
    for col in optimized_df.select_dtypes(include=['object']).columns:
        unique_count = optimized_df[col].nunique()
        total_count = len(optimized_df[col])
        
        # Convert to category if less than 50% unique values
        if unique_count / total_count < 0.5:
            optimized_df[col] = optimized_df[col].astype('category')
    
    optimized_memory = optimized_df.memory_usage().sum()
    memory_reduction = (original_memory - optimized_memory) / original_memory * 100
    
    if verbose:
        print(f"Memory Optimization Results:")
        print(f"Original memory usage: {original_memory:,} bytes")
        print(f"Optimized memory usage: {optimized_memory:,} bytes")
        print(f"Memory reduction: {memory_reduction:.1f}%")
        
        # Show data type changes
        dtype_changes = []
        for col in optimized_df.columns:
            if df[col].dtype != optimized_df[col].dtype:
                dtype_changes.append(f"  {col}: {df[col].dtype} → {optimized_df[col].dtype}")
        
        if dtype_changes:
            print("Data type changes:")
            for change in dtype_changes:
                print(change)
    
    return optimized_df

# Optimize the customer dataset
optimized_customers = optimize_dataframe_memory(cleaned_df)
```

### Data Quality Assessment Framework

```python
class DataQualityAssessment:
    """Comprehensive data quality assessment toolkit."""
    
    def __init__(self, df):
        self.df = df
        self.quality_report = {}
    
    def assess_completeness(self):
        """Assess data completeness."""
        missing_data = self.df.isnull().sum()
        completeness_pct = (1 - missing_data / len(self.df)) * 100
        
        self.quality_report['completeness'] = {
            'missing_values_per_column': missing_data.to_dict(),
            'completeness_percentage': completeness_pct.to_dict(),
            'overall_completeness': completeness_pct.mean()
        }
        
        return self
    
    def assess_uniqueness(self):
        """Assess data uniqueness."""
        uniqueness_stats = {}
        
        for col in self.df.columns:
            total_values = len(self.df[col].dropna())
            unique_values = self.df[col].nunique()
            uniqueness_pct = (unique_values / total_values * 100) if total_values > 0 else 0
            
            uniqueness_stats[col] = {
                'unique_count': unique_values,
                'total_count': total_values,
                'uniqueness_percentage': uniqueness_pct,
                'duplicate_count': total_values - unique_values
            }
        
        # Check for completely duplicate rows
        duplicate_rows = self.df.duplicated().sum()
        
        self.quality_report['uniqueness'] = {
            'column_stats': uniqueness_stats,
            'duplicate_rows': duplicate_rows,
            'duplicate_row_percentage': (duplicate_rows / len(self.df)) * 100
        }
        
        return self
    
    def assess_validity(self):
        """Assess data validity based on expected patterns."""
        validity_issues = {}
        
        # Check numeric ranges
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                issues = []
                
                # Check for negative values where they shouldn't be
                if col in ['age', 'income', 'purchase_amount'] and (col_data < 0).any():
                    negative_count = (col_data < 0).sum()
                    issues.append(f"{negative_count} negative values")
                
                # Check for extreme outliers
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_threshold = 3 * IQR
                outliers = ((col_data < (Q1 - outlier_threshold)) | 
                           (col_data > (Q3 + outlier_threshold))).sum()
                
                if outliers > 0:
                    issues.append(f"{outliers} extreme outliers")
                
                if issues:
                    validity_issues[col] = issues
        
        # Check string formats
        string_cols = self.df.select_dtypes(include=['object']).columns
        for col in string_cols:
            issues = []
            col_data = self.df[col].dropna()
            
            # Check for inconsistent formatting
            if len(col_data) > 0:
                # Example: check for mixed case in categorical data
                if col in ['region', 'subscription_type']:
                    mixed_case = col_data.str.lower().nunique() != col_data.nunique()
                    if mixed_case:
                        issues.append("Inconsistent case formatting")
                
                if issues:
                    validity_issues[col] = issues
        
        self.quality_report['validity'] = validity_issues
        
        return self
    
    def assess_consistency(self):
        """Assess data consistency across related fields."""
        consistency_issues = []
        
        # Example consistency checks
        if 'age' in self.df.columns and 'subscription_type' in self.df.columns:
            # Check if very young customers have enterprise subscriptions
            young_enterprise = self.df[
                (self.df['age'] < 18) & 
                (self.df['subscription_type'] == 'Enterprise')
            ]
            if len(young_enterprise) > 0:
                consistency_issues.append(
                    f"{len(young_enterprise)} customers under 18 with Enterprise subscriptions"
                )
        
        if 'purchase_amount' in self.df.columns and 'satisfaction_score' in self.df.columns:
            # Check for high satisfaction with zero purchases
            high_sat_no_purchase = self.df[
                (self.df['purchase_amount'] == 0) & 
                (self.df['satisfaction_score'] > 80)
            ]
            if len(high_sat_no_purchase) > 0:
                consistency_issues.append(
                    f"{len(high_sat_no_purchase)} high satisfaction scores with zero purchases"
                )
        
        self.quality_report['consistency'] = consistency_issues
        
        return self
    
    def generate_report(self):
        """Generate comprehensive data quality report."""
        print("=" * 60)
        print("DATA QUALITY ASSESSMENT REPORT")
        print("=" * 60)
        
        # Completeness report
        if 'completeness' in self.quality_report:
            print(f"\n📊 COMPLETENESS:")
            print(f"Overall completeness: {self.quality_report['completeness']['overall_completeness']:.1f}%")
            
            missing_cols = {k: v for k, v in self.quality_report['completeness']['missing_values_per_column'].items() if v > 0}
            if missing_cols:
                print("Columns with missing data:")
                for col, missing_count in missing_cols.items():
                    pct = (missing_count / len(self.df)) * 100
                    print(f"  {col}: {missing_count} ({pct:.1f}%)")
        
        # Uniqueness report
        if 'uniqueness' in self.quality_report:
            print(f"\n🔍 UNIQUENESS:")
            duplicate_rows = self.quality_report['uniqueness']['duplicate_rows']
            if duplicate_rows > 0:
                print(f"Duplicate rows: {duplicate_rows} ({self.quality_report['uniqueness']['duplicate_row_percentage']:.1f}%)")
            
            low_uniqueness = {k: v for k, v in self.quality_report['uniqueness']['column_stats'].items() 
                            if v['uniqueness_percentage'] < 10 and v['total_count'] > 100}
            if low_uniqueness:
                print("Columns with low uniqueness:")
                for col, stats in low_uniqueness.items():
                    print(f"  {col}: {stats['uniqueness_percentage']:.1f}% unique")
        
        # Validity report
        if 'validity' in self.quality_report:
            print(f"\n⚠️  VALIDITY ISSUES:")
            if self.quality_report['validity']:
                for col, issues in self.quality_report['validity'].items():
                    print(f"  {col}: {', '.join(issues)}")
            else:
                print("  No validity issues detected")
        
        # Consistency report
        if 'consistency' in self.quality_report:
            print(f"\n🔗 CONSISTENCY ISSUES:")
            if self.quality_report['consistency']:
                for issue in self.quality_report['consistency']:
                    print(f"  • {issue}")
            else:
                print("  No consistency issues detected")
        
        print("\n" + "=" * 60)
        
        return self.quality_report

# Run data quality assessment
quality_assessor = DataQualityAssessment(optimized_customers)
quality_report = (quality_assessor
                  .assess_completeness()
                  .assess_uniqueness()
                  .assess_validity()
                  .assess_consistency()
                  .generate_report())
```

---

## Part 4: Complete Analysis Workflow (85 minutes)

### End-to-End Data Analysis Pipeline

```python
class DataAnalysisPipeline:
    """Professional data analysis pipeline for reproducible workflows."""
    
    def __init__(self, data_source, config=None):
        self.data_source = data_source
        self.config = config or self.default_config()
        self.raw_data = None
        self.processed_data = None
        self.analysis_results = {}
        self.pipeline_log = []
    
    def default_config(self):
        """Default configuration for the pipeline."""
        return {
            'missing_data_threshold': 0.3,
            'outlier_method': 'iqr',
            'outlier_factor': 1.5,
            'correlation_threshold': 0.7,
            'significance_level': 0.05
        }
    
    def load_data(self):
        """Load and validate raw data."""
        try:
            if isinstance(self.data_source, str):
                if self.data_source.endswith('.csv'):
                    self.raw_data = pd.read_csv(self.data_source)
                elif self.data_source.endswith('.xlsx'):
                    self.raw_data = pd.read_excel(self.data_source)
                else:
                    raise ValueError(f"Unsupported file format: {self.data_source}")
            else:
                self.raw_data = self.data_source.copy()
            
            self.pipeline_log.append(f"✓ Loaded data: {self.raw_data.shape}")
            print(f"Data loaded successfully: {self.raw_data.shape}")
            
        except Exception as e:
            self.pipeline_log.append(f"✗ Data loading failed: {str(e)}")
            raise
        
        return self
    
    def explore_data(self):
        """Comprehensive exploratory data analysis."""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Run load_data() first.")
        
        exploration = {}
        
        # Basic information
        exploration['shape'] = self.raw_data.shape
        exploration['columns'] = list(self.raw_data.columns)
        exploration['dtypes'] = dict(self.raw_data.dtypes)
        exploration['memory_usage'] = self.raw_data.memory_usage().sum()
        
        # Missing data analysis
        missing_data = self.raw_data.isnull().sum()
        exploration['missing_data'] = missing_data[missing_data > 0].to_dict()
        
        # Descriptive statistics
        numeric_data = self.raw_data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            exploration['numeric_summary'] = numeric_data.describe().to_dict()
        
        categorical_data = self.raw_data.select_dtypes(include=['object', 'category'])
        if not categorical_data.empty:
            cat_summary = {}
            for col in categorical_data.columns:
                cat_summary[col] = {
                    'unique_count': self.raw_data[col].nunique(),
                    'top_values': self.raw_data[col].value_counts().head().to_dict()
                }
            exploration['categorical_summary'] = cat_summary
        
        self.analysis_results['exploration'] = exploration
        self.pipeline_log.append("✓ Data exploration completed")
        
        # Display key findings
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        print(f"Dataset shape: {exploration['shape']}")
        print(f"Memory usage: {exploration['memory_usage']:,} bytes")
        
        if exploration['missing_data']:
            print(f"\nMissing data found in {len(exploration['missing_data'])} columns:")
            for col, count in exploration['missing_data'].items():
                pct = (count / self.raw_data.shape[0]) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        
        return self
    
    def clean_data(self):
        """Clean and preprocess the data."""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Run load_data() first.")
        
        self.processed_data = self.raw_data.copy()
        cleaning_steps = []
        
        # Remove columns with too much missing data
        missing_pct = self.processed_data.isnull().sum() / len(self.processed_data)
        cols_to_drop = missing_pct[missing_pct > self.config['missing_data_threshold']].index.tolist()
        
        if cols_to_drop:
            self.processed_data = self.processed_data.drop(columns=cols_to_drop)
            cleaning_steps.append(f"Dropped {len(cols_to_drop)} columns with >{self.config['missing_data_threshold']*100}% missing data")
        
        # Handle remaining missing data
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.processed_data[col].isnull().any():
                median_val = self.processed_data[col].median()
                self.processed_data[col].fillna(median_val, inplace=True)
                cleaning_steps.append(f"Imputed {col} missing values with median")
        
        categorical_cols = self.processed_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.processed_data[col].isnull().any():
                mode_val = self.processed_data[col].mode().iloc[0] if not self.processed_data[col].mode().empty else 'Unknown'
                self.processed_data[col].fillna(mode_val, inplace=True)
                cleaning_steps.append(f"Imputed {col} missing values with mode")
        
        # Remove outliers for numeric columns
        if self.config['outlier_method'] == 'iqr':
            for col in numeric_cols:
                Q1 = self.processed_data[col].quantile(0.25)
                Q3 = self.processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config['outlier_factor'] * IQR
                upper_bound = Q3 + self.config['outlier_factor'] * IQR
                
                outliers_before = len(self.processed_data)
                self.processed_data = self.processed_data[
                    (self.processed_data[col] >= lower_bound) & 
                    (self.processed_data[col] <= upper_bound)
                ]
                outliers_removed = outliers_before - len(self.processed_data)
                
                if outliers_removed > 0:
                    cleaning_steps.append(f"Removed {outliers_removed} outliers from {col}")
        
        self.analysis_results['cleaning_steps'] = cleaning_steps
        self.pipeline_log.append(f"✓ Data cleaning completed: {len(cleaning_steps)} steps applied")
        
        print(f"\nData cleaning completed:")
        for step in cleaning_steps:
            print(f"  • {step}")
        print(f"Final dataset shape: {self.processed_data.shape}")
        
        return self
    
    def analyze_relationships(self):
        """Analyze relationships between variables."""
        if self.processed_data is None:
            raise ValueError("Data not processed. Run clean_data() first.")
        
        relationships = {}
        
        # Correlation analysis for numeric variables
        numeric_data = self.processed_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            correlation_matrix = numeric_data.corr()
            
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > self.config['correlation_threshold']:
                        high_corr_pairs.append({
                            'var1': correlation_matrix.columns[i],
                            'var2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            relationships['correlations'] = {
                'matrix': correlation_matrix.to_dict(),
                'high_correlations': high_corr_pairs
            }
        
        # Categorical vs Numeric relationships (simplified ANOVA)
        categorical_cols = self.processed_data.select_dtypes(include=['object', 'category']).columns
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_num_relationships = []
            
            for cat_col in categorical_cols:
                for num_col in numeric_cols:
                    groups = self.processed_data.groupby(cat_col)[num_col].mean()
                    variance_ratio = groups.var() / self.processed_data[num_col].var()
                    
                    if variance_ratio > 0.1:  # Simplified significance test
                        cat_num_relationships.append({
                            'categorical': cat_col,
                            'numeric': num_col,
                            'group_means': groups.to_dict(),
                            'variance_ratio': variance_ratio
                        })
            
            relationships['categorical_numeric'] = cat_num_relationships
        
        self.analysis_results['relationships'] = relationships
        self.pipeline_log.append("✓ Relationship analysis completed")
        
        # Display findings
        if 'high_correlations' in relationships.get('correlations', {}):
            print(f"\nHigh correlations found (|r| > {self.config['correlation_threshold']}):")
            for pair in relationships['correlations']['high_correlations']:
                print(f"  {pair['var1']} ↔ {pair['var2']}: r = {pair['correlation']:.3f}")
        
        return self
    
    def generate_insights(self):
        """Generate actionable insights from the analysis."""
        insights = []
        
        if 'exploration' in self.analysis_results:
            # Data quality insights
            exploration = self.analysis_results['exploration']
            
            if exploration.get('missing_data'):
                missing_cols = len(exploration['missing_data'])
                insights.append(f"Data quality concern: {missing_cols} columns have missing values")
            
            # Data distribution insights
            if 'numeric_summary' in exploration:
                for col, stats in exploration['numeric_summary'].items():
                    if stats['std'] > stats['mean']:
                        insights.append(f"High variability detected in {col} (CV = {stats['std']/stats['mean']:.2f})")
        
        if 'relationships' in self.analysis_results:
            relationships = self.analysis_results['relationships']
            
            # Correlation insights
            if 'high_correlations' in relationships.get('correlations', {}):
                high_corrs = relationships['correlations']['high_correlations']
                if high_corrs:
                    insights.append(f"Strong relationships found between {len(high_corrs)} variable pairs")
                    for pair in high_corrs[:3]:  # Top 3
                        direction = "positive" if pair['correlation'] > 0 else "negative"
                        insights.append(f"  {pair['var1']} and {pair['var2']} show strong {direction} correlation")
        
        self.analysis_results['insights'] = insights
        self.pipeline_log.append("✓ Insights generation completed")
        
        print(f"\n🎯 KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return self
    
    def export_results(self, output_dir='results'):
        """Export analysis results and processed data."""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export processed data
        processed_data_path = os.path.join(output_dir, 'processed_data.csv')
        self.processed_data.to_csv(processed_data_path, index=False)
        
        # Export analysis results
        results_path = os.path.join(output_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = self._make_serializable(self.analysis_results)
            json.dump(serializable_results, f, indent=2)
        
        # Export pipeline log
        log_path = os.path.join(output_dir, 'pipeline_log.txt')
        with open(log_path, 'w') as f:
            f.write("Data Analysis Pipeline Log\n")
            f.write("=" * 30 + "\n\n")
            for entry in self.pipeline_log:
                f.write(f"{entry}\n")
        
        self.pipeline_log.append(f"✓ Results exported to {output_dir}")
        print(f"\nResults exported to '{output_dir}' directory:")
        print(f"  • processed_data.csv")
        print(f"  • analysis_results.json")
        print(f"  • pipeline_log.txt")
        
        return self
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

# Run the complete pipeline
print("Starting Complete Data Analysis Pipeline...")
print("=" * 60)

pipeline = (DataAnalysisPipeline(optimized_customers)
           .load_data()
           .explore_data()
           .clean_data()
           .analyze_relationships()
           .generate_insights()
           .export_results())

print("\n🎉 Pipeline completed successfully!")
```

---

## Practical Exercises

### Exercise 1: Data Exploration Pipeline Development (60 minutes)

Build a comprehensive data exploration system that automatically analyzes any dataset and generates detailed reports.

**Requirements**:
- Accept various file formats (CSV, Excel, JSON)
- Generate automatic data profiling reports
- Identify data quality issues
- Create visualization summaries
- Export findings in multiple formats

**Starter Framework**:
```python
class AutoDataExplorer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.profile_results = {}
    
    def load_and_profile(self):
        """Load data and generate comprehensive profile."""
        pass
    
    def detect_data_types(self):
        """Intelligently detect and suggest data types."""
        pass
    
    def identify_patterns(self):
        """Identify patterns in the data."""
        pass
    
    def generate_visualizations(self):
        """Create automatic visualizations."""
        pass
    
    def export_report(self, format='html'):
        """Export comprehensive report."""
        pass
```

### Exercise 2: Data Cleaning Workshop (75 minutes)

Create a professional data cleaning toolkit that handles common real-world data issues.

**Requirements**:
- Handle multiple types of missing data patterns
- Detect and clean inconsistent formatting
- Identify and handle outliers appropriately
- Standardize categorical values
- Validate data integrity

**Real Dataset**: Messy customer transaction data (provided)

**Expected Capabilities**:
- Clean phone numbers, email addresses, and names
- Standardize date formats
- Handle currency and numeric formatting
- Resolve duplicate customer records
- Flag suspicious transactions

### Exercise 3: End-to-End Analysis Mini-Project (90 minutes)

Conduct a complete business analysis using real e-commerce data.

**Business Questions**:
1. What factors drive customer satisfaction?
2. Which customer segments are most profitable?
3. How do seasonal patterns affect sales?
4. What products should be recommended together?

**Requirements**:
- Complete data pipeline from raw to insights
- Statistical analysis with proper interpretation
- Business recommendations based on findings
- Professional report with visualizations
- Reproducible analysis workflow

**Dataset**: Multi-table e-commerce data including:
- Customer demographics and behavior
- Product catalog and sales
- Transaction history
- Customer service interactions

---

## Wrap-up and Professional Applications (30 minutes)

### Pandas in Production: Best Practices

```python
# Production-ready Pandas patterns
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional
import logging

class ProductionDataProcessor:
    """Production-ready data processing with error handling and logging."""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
    def safe_read_csv(self, filepath: str, **kwargs) -> Optional[pd.DataFrame]:
        """Safely read CSV with comprehensive error handling."""
        try:
            # Default safe parameters
            safe_defaults = {
                'low_memory': False,
                'encoding': 'utf-8',
                'na_values': ['', 'NA', 'N/A', 'null', 'NULL'],
                'keep_default_na': True
            }
            safe_defaults.update(kwargs)
            
            df = pd.read_csv(filepath, **safe_defaults)
            self.logger.info(f"Successfully loaded {filepath}: {df.shape}")
            return df
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            return None
        except pd.errors.EmptyDataError:
            self.logger.error(f"Empty file: {filepath}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {str(e)}")
            return None
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate DataFrame schema against expected structure."""
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            self.logger.error(f"Missing columns: {missing_cols}")
            return False
        return True
    
    def process_with_chunks(self, filepath: str, chunk_size: int = 10000, 
                          processor_func=None) -> pd.DataFrame:
        """Process large files in chunks to manage memory."""
        results = []
        
        try:
            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                if processor_func:
                    processed_chunk = processor_func(chunk)
                    results.append(processed_chunk)
                else:
                    results.append(chunk)
            
            return pd.concat(results, ignore_index=True)
            
        except Exception as e:
            self.logger.error(f"Error processing chunks: {str(e)}")
            return pd.DataFrame()

# Usage example
processor = ProductionDataProcessor()
data = processor.safe_read_csv('large_dataset.csv')

if data is not None:
    # Process the data
    pass
```

### Integration with Data Science Ecosystem

```python
# Pandas + Scikit-learn integration
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def pandas_to_ml_pipeline(df, target_column):
    """Convert Pandas DataFrame to ML-ready format."""
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'performance': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': dict(zip(X.columns, model.feature_importances_))
    }
```

### Performance Optimization Checklist

1. **Use vectorized operations instead of loops**
2. **Choose appropriate data types (especially categories)**
3. **Use `query()` method for complex filtering**
4. **Leverage `groupby` for aggregations**
5. **Process large files in chunks**
6. **Use `pd.eval()` for complex expressions**
7. **Avoid chained indexing (use `.loc[]` instead)**

### Key Takeaways

1. **Labels Transform Data Science**: Moving from positional to labeled data makes analysis intuitive and error-resistant
2. **Real-World Data is Messy**: Professional data cleaning is a systematic process, not ad-hoc fixes
3. **Workflow Automation**: Reproducible pipelines are essential for production data science
4. **Integration is Key**: Pandas integrates seamlessly with the entire Python data science ecosystem

### Preparation for Next Lectures

The foundation you've built with file handling, NumPy performance, and Pandas data manipulation will support:
- Advanced statistical analysis
- Machine learning workflows
- Data visualization and reporting
- Database integration and big data processing

### Extended Learning Resources

- **Pandas Documentation**: User guide and API reference
- **Performance**: "Python for Data Analysis" by Wes McKinney (Pandas creator)
- **Advanced Patterns**: "Effective Pandas" patterns and best practices
- **Integration**: How Pandas connects to databases, web APIs, and cloud storage

You now have the core skills to handle any data analysis challenge - from small datasets to production-scale data processing pipelines. The combination of professional file handling, NumPy's computational power, and Pandas' intuitive data manipulation creates a complete foundation for data science excellence.