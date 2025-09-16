Data Analysis and Debugging Techniques

Welcome to week 8! Now that you have solid foundations in data manipulation and visualization, it's time to develop systematic approaches for analyzing complex datasets and debugging data analysis code. You'll learn professional techniques for exploring data patterns, identifying issues, and writing robust analysis workflows.

By the end of today, you'll have a toolkit for approaching any data analysis problem methodically and debugging issues that inevitably arise in real-world data work.

![xkcd 1845: Data](media/xkcd_1845.png)

Don't worry - systematic analysis techniques make data insights much clearer!

Systematic Data Analysis Workflow

The Analysis Mindset

**Professional Approach:**
1. **Start with questions** - What are you trying to learn?
2. **Understand the data** - Where did it come from? What do columns represent?
3. **Check data quality** - Missing values, outliers, inconsistencies
4. **Explore systematically** - Use consistent patterns to examine different aspects
5. **Document findings** - Keep track of insights and decisions
6. **Validate results** - Double-check surprising findings

**Analysis Questions Framework:**
- **Descriptive:** What happened? (summary statistics, distributions)
- **Diagnostic:** Why did it happen? (correlations, comparisons)
- **Predictive:** What might happen? (trends, patterns)
- **Prescriptive:** What should we do? (recommendations based on data)

Building Analysis Habits

**Reference:**
```python
Standard analysis template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset(df, target_column=None):
    """
    Systematic dataset analysis template
    """
    print("=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes.value_counts())
    
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("No missing values found")
    
    print("\n=== NUMERIC COLUMNS SUMMARY ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    print("\n=== CATEGORICAL COLUMNS SUMMARY ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head())
```

Exploratory Data Analysis (EDA) Techniques

Understanding Distributions

**Reference:**
```python
def explore_distribution(series, name):
    """
    Comprehensive distribution analysis for a single column
    """
    print(f"=== {name.upper()} DISTRIBUTION ===")
    
    # Basic statistics
    print(f"Count: {series.count()}")
    print(f"Missing: {series.isnull().sum()}")
    print(f"Mean: {series.mean():.2f}")
    print(f"Median: {series.median():.2f}")
    print(f"Std Dev: {series.std():.2f}")
    print(f"Min: {series.min():.2f}")
    print(f"Max: {series.max():.2f}")
    
    # Distribution characteristics
    skewness = series.skew()
    print(f"Skewness: {skewness:.2f} {'(right-skewed)' if skewness > 0 else '(left-skewed)' if skewness < 0 else '(symmetric)'}")
    
    # Outlier detection using IQR method
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    print(f"Potential outliers: {len(outliers)} values outside [{lower_bound:.1f}, {upper_bound:.1f}]")
    
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'skewness': skewness,
        'outlier_count': len(outliers)
    }
```

Relationship Analysis

**Reference:**
```python
def analyze_relationships(df, target_col, max_categories=10):
    """
    Analyze relationships between variables and a target column
    """
    results = {}
    
    # Numeric columns - correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(target_col, errors='ignore')
    
    if len(numeric_cols) > 0:
        correlations = df[numeric_cols].corrwith(df[target_col]).sort_values(key=abs, ascending=False)
        print(f"=== CORRELATIONS WITH {target_col.upper()} ===")
        for col, corr in correlations.head(5).items():
            print(f"{col}: {corr:.3f}")
        results['correlations'] = correlations
    
    # Categorical columns - group analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= max_categories:
            print(f"\n=== {col.upper()} GROUP ANALYSIS ===")
            group_stats = df.groupby(col)[target_col].agg(['count', 'mean', 'median', 'std'])
            print(group_stats)
            results[f'{col}_groups'] = group_stats
    
    return results
```

**Brief Example:**
```python
Load sales data for analysis
sales_data = pd.read_csv('monthly_sales.csv')

Systematic exploration
print("Starting systematic analysis of sales data...")
analyze_dataset(sales_data, target_column='revenue')

Focus on key metric
revenue_analysis = explore_distribution(sales_data['revenue'], 'Monthly Revenue')

Understand what drives revenue
relationship_analysis = analyze_relationships(sales_data, 'revenue')
```

Data Quality Assessment

Identifying Data Issues

Common Data Problems

**Reference:**
```python
def comprehensive_data_quality_check(df):
    """
    Comprehensive data quality assessment
    """
    issues = {}
    
    print("=== DATA QUALITY ASSESSMENT ===")
    
    # 1. Missing values analysis
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    if missing_data.any():
        print("\nMISSING VALUES:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"  {col}: {count} ({missing_percent[col]:.1f}%)")
        issues['missing_values'] = missing_data[missing_data > 0]
    
    # 2. Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\nDUPLICATE ROWS: {duplicate_count}")
        issues['duplicates'] = duplicate_count
    
    # 3. Data type inconsistencies
    print("\nDATA TYPE ANALYSIS:")
    for col in df.columns:
        col_type = df[col].dtype
        print(f"  {col}: {col_type}")
        
        # Check for mixed types in object columns
        if col_type == 'object':
            try:
                # Try to convert to numeric
                numeric_version = pd.to_numeric(df[col], errors='coerce')
                if not numeric_version.isnull().all():
                    non_numeric_count = numeric_version.isnull().sum() - df[col].isnull().sum()
                    if non_numeric_count > 0:
                        print(f"    WARNING: {non_numeric_count} non-numeric values in potentially numeric column")
                        issues[f'{col}_mixed_types'] = non_numeric_count
            except:
                pass
    
    # 4. Outlier detection for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        print("\nOUTLIER ANALYSIS:")
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
                issues[f'{col}_outliers'] = len(outliers)
    
    # 5. Consistency checks
    print("\nCONSISTENCY CHECKS:")
    
    # Check for leading/trailing whitespace in string columns
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        whitespace_issues = df[col].str.strip() != df[col]
        whitespace_count = whitespace_issues.sum()
        if whitespace_count > 0:
            print(f"  {col}: {whitespace_count} values with leading/trailing whitespace")
            issues[f'{col}_whitespace'] = whitespace_count
    
    return issues

def create_quality_report(df, output_file='data_quality_report.txt'):
    """
    Generate a comprehensive data quality report
    """
    with open(output_file, 'w') as f:
        f.write("DATA QUALITY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        # Capture quality issues
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        issues = comprehensive_data_quality_check(df)
        
        sys.stdout = old_stdout
        f.write(captured_output.getvalue())
        
        f.write(f"\n\nSUMMARY: Found {len(issues)} types of data quality issues\n")
        for issue_type, count in issues.items():
            f.write(f"  - {issue_type}: {count}\n")
    
    print(f"Data quality report saved to {output_file}")
    return issues
```

Debugging Data Analysis Code

Systematic Debugging Approach

**Reference:**
```python
def debug_analysis_step(func, data, step_name, save_intermediate=True):
    """
    Wrapper for debugging analysis steps
    """
    print(f"\n=== DEBUGGING: {step_name} ===")
    
    # Check input data
    print("INPUT CHECK:")
    print(f"  Shape: {data.shape}")
    print(f"  Columns: {list(data.columns)}")
    print(f"  Memory: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    try:
        # Execute the analysis step
        result = func(data)
        
        print("EXECUTION: SUCCESS")
        print(f"  Result shape: {result.shape if hasattr(result, 'shape') else type(result)}")
        
        # Save intermediate results if requested
        if save_intermediate:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_{step_name}_{timestamp}.csv"
            if hasattr(result, 'to_csv'):
                result.to_csv(filename, index=False)
                print(f"  Saved intermediate result: {filename}")
        
        return result
        
    except Exception as e:
        print(f"EXECUTION: FAILED")
        print(f"  Error: {str(e)}")
        print(f"  Error type: {type(e).__name__}")
        
        # Additional debugging info
        if hasattr(data, 'dtypes'):
            print("  Data types at failure:")
            for col, dtype in data.dtypes.items():
                print(f"    {col}: {dtype}")
        
        raise  # Re-raise the exception after logging
```

Common Debugging Patterns

**Reference:**
```python
Pattern 1: Check data at each step
def safe_data_processing(df):
    """
    Process data with built-in checks
    """
    print(f"Starting with {len(df)} rows")
    
    # Step 1: Remove missing values
    df_clean = df.dropna()
    print(f"After removing NaN: {len(df_clean)} rows")
    
    # Step 2: Filter outliers
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        before_count = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        after_count = len(df_clean)
        print(f"After removing {col} outliers: {after_count} rows ({before_count - after_count} removed)")
    
    return df_clean

Pattern 2: Validation functions
def validate_analysis_assumptions(df, assumptions):
    """
    Validate key assumptions before analysis
    """
    validation_results = {}
    
    for assumption, check_func in assumptions.items():
        try:
            result = check_func(df)
            validation_results[assumption] = {'passed': True, 'result': result}
            print(f"✓ {assumption}: PASSED")
        except Exception as e:
            validation_results[assumption] = {'passed': False, 'error': str(e)}
            print(f"✗ {assumption}: FAILED - {str(e)}")
    
    return validation_results

Pattern 3: Checkpoints in long analysis
def analysis_with_checkpoints(df, checkpoint_dir='checkpoints'):
    """
    Long analysis with save points
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Checkpoint 1: After data cleaning
    df_clean = clean_data(df)
    df_clean.to_csv(f'{checkpoint_dir}/01_cleaned_data.csv', index=False)
    print("Checkpoint 1: Data cleaning complete")
    
    # Checkpoint 2: After feature engineering
    df_features = create_features(df_clean)
    df_features.to_csv(f'{checkpoint_dir}/02_with_features.csv', index=False)
    print("Checkpoint 2: Feature engineering complete")
    
    # Checkpoint 3: After analysis
    results = perform_analysis(df_features)
    results.to_csv(f'{checkpoint_dir}/03_analysis_results.csv', index=False)
    print("Checkpoint 3: Analysis complete")
    
    return results
```

**Brief Example:**
```python
Debug a problematic analysis
def analyze_customer_churn(customers_df):
    """Example analysis function with debugging built in"""
    
    # Validation assumptions
    assumptions = {
        'has_customer_id': lambda df: 'customer_id' in df.columns,
        'no_duplicate_customers': lambda df: df['customer_id'].nunique() == len(df),
        'has_required_columns': lambda df: all(col in df.columns for col in ['last_purchase', 'total_spent'])
    }
    
    # Validate before proceeding
    validation_results = validate_analysis_assumptions(customers_df, assumptions)
    
    if not all(result['passed'] for result in validation_results.values()):
        print("Cannot proceed - validation failed")
        return None
    
    # Process with checkpoints
    return debug_analysis_step(
        lambda df: df.groupby('customer_segment')['total_spent'].agg(['mean', 'median', 'count']),
        customers_df,
        'customer_segment_analysis'
    )
```

LIVE DEMO!
*Systematic analysis of a complex dataset, demonstrating debugging techniques, quality assessment, and relationship analysis*

Advanced Analysis Patterns

Time Series Analysis Basics

**Reference:**
```python
def analyze_time_series(df, date_col, value_col):
    """
    Basic time series analysis pattern
    """
    # Ensure proper date format
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    print(f"=== TIME SERIES ANALYSIS: {value_col} ===")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    
    # Basic trend analysis
    df['trend'] = df[value_col].rolling(window=7, center=True).mean()
    
    # Seasonal patterns (if monthly data)
    df['month'] = df[date_col].dt.month
    monthly_pattern = df.groupby('month')[value_col].mean()
    
    print("Monthly averages:")
    for month, avg in monthly_pattern.items():
        print(f"  Month {month}: {avg:.2f}")
    
    # Growth rate calculation
    df['pct_change'] = df[value_col].pct_change() * 100
    avg_growth = df['pct_change'].mean()
    print(f"Average period-over-period growth: {avg_growth:.1f}%")
    
    return df

def detect_anomalies(series, method='iqr'):
    """
    Detect anomalies in time series data
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = series[(series < lower_bound) | (series > upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        anomalies = series[z_scores > 3]
    
    return anomalies.index
```

Cohort Analysis Pattern

**Reference:**
```python
def cohort_analysis(df, customer_col, date_col, value_col):
    """
    Basic cohort analysis for customer behavior
    """
    # Create period columns
    df[date_col] = pd.to_datetime(df[date_col])
    df['order_period'] = df[date_col].dt.to_period('M')
    
    # Get customer's first order period
    df['cohort_group'] = df.groupby(customer_col)[date_col].transform('min').dt.to_period('M')
    
    # Calculate period number
    def get_period_number(df):
        return (df['order_period'] - df['cohort_group']).apply(attrgetter('n'))
    
    df['period_number'] = get_period_number(df)
    
    # Create cohort table
    cohort_data = df.groupby(['cohort_group', 'period_number'])[customer_col].nunique().reset_index()
    cohort_table = cohort_data.pivot(index='cohort_group', columns='period_number', values=customer_col)
    
    # Calculate retention rates
    cohort_sizes = df.groupby('cohort_group')[customer_col].nunique()
    retention_table = cohort_table.divide(cohort_sizes, axis=0)
    
    print("COHORT RETENTION ANALYSIS")
    print("=" * 40)
    print("Retention rates by cohort and period:")
    print(retention_table.round(3))
    
    return cohort_table, retention_table
```

Professional Analysis Documentation

Creating Analysis Reports

**Reference:**
```python
def generate_analysis_report(df, analysis_results, output_file='analysis_report.md'):
    """
    Generate a professional analysis report
    """
    with open(output_file, 'w') as f:
        f.write("# Data Analysis Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now()}\n")
        f.write(f"**Dataset:** {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("Key findings from the analysis:\n\n")
        
        # Data overview
        f.write("## Data Overview\n\n")
        f.write(f"- **Size:** {df.shape[0]:,} records with {df.shape[1]} variables\n")
        f.write(f"- **Memory usage:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")
        f.write(f"- **Missing values:** {df.isnull().sum().sum()} total\n\n")
        
        # Statistical summary
        f.write("## Statistical Summary\n\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            f.write("### Numeric Variables\n\n")
            summary = df[numeric_cols].describe()
            f.write(summary.round(2).to_string())
            f.write("\n\n")
        
        # Analysis results
        f.write("## Analysis Results\n\n")
        for analysis_name, results in analysis_results.items():
            f.write(f"### {analysis_name.title().replace('_', ' ')}\n\n")
            if isinstance(results, dict):
                for key, value in results.items():
                    f.write(f"- **{key}:** {value}\n")
            else:
                f.write(f"{results}\n")
            f.write("\n")
        
        f.write("## Methodology\n\n")
        f.write("This analysis was conducted using:\n")
        f.write("- Python pandas for data manipulation\n")
        f.write("- Systematic exploratory data analysis\n")
        f.write("- Statistical validation of assumptions\n")
        f.write("- Quality checks and outlier detection\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the analysis results:\n\n")
        f.write("1. [Add specific recommendations based on findings]\n")
        f.write("2. [Include actionable insights]\n")
        f.write("3. [Note any data quality concerns]\n\n")
    
    print(f"Analysis report generated: {output_file}")
```

**Brief Example:**
```python
Complete analysis workflow with documentation
def complete_analysis_workflow(data_file):
    """
    End-to-end analysis with documentation
    """
    # Load and validate data
    df = pd.read_csv(data_file)
    issues = create_quality_report(df)
    
    # Perform systematic analysis
    analysis_results = {}
    
    # Distribution analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:3]:  # Top 3 numeric columns
        analysis_results[f'{col}_distribution'] = explore_distribution(df[col], col)
    
    # Relationship analysis if target column exists
    if 'target' in df.columns:
        analysis_results['relationships'] = analyze_relationships(df, 'target')
    
    # Generate comprehensive report
    generate_analysis_report(df, analysis_results, 'final_analysis_report.md')
    
    return df, analysis_results
```

Key Takeaways

1. **Systematic approach** beats random exploration - follow consistent analysis patterns
2. **Quality assessment first** - understand your data before analyzing it
3. **Debug proactively** - build checks and validations into your analysis code
4. **Document everything** - your future self will thank you
5. **Save intermediate results** - checkpoints prevent losing work from errors
6. **Validate assumptions** - check that your data meets analysis requirements
7. **Professional reporting** - clear documentation makes analysis actionable

You now have professional techniques for approaching complex data analysis problems systematically. These debugging and quality assessment skills will serve you well in real-world data science projects.

Next week: We'll explore advanced data manipulation with pandas and prepare for our final projects!

Practice Challenge

Before next class:
1. **Analysis Practice:**
   - Find a dataset with at least 5 columns and 1000+ rows
   - Apply the systematic analysis workflow
   - Create a comprehensive quality report
   
2. **Debugging Skills:**
   - Intentionally break some analysis code and practice debugging
   - Use the checkpoint pattern on a multi-step analysis
   - Validate assumptions before proceeding with analysis
   
3. **Professional Documentation:**
   - Generate an analysis report following the template
   - Include visualizations and clear recommendations
   - Practice explaining technical findings clearly

Remember: Good analysis is systematic, documented, and validated - develop these professional habits now!