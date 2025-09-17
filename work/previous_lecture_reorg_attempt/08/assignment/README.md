# Assignment 08: Data Analysis and Debugging Techniques

**Due:** Before next class  
**Points:** 20 points total  
**Submit:** Via GitHub repository (link submitted to Canvas)

## Overview

Master systematic data analysis and debugging techniques by performing comprehensive analysis on a complex real-world dataset. You'll develop professional workflows for exploring data quality, identifying patterns, and creating robust analysis code that handles edge cases gracefully.

## Learning Objectives

By completing this assignment, you will:
- Apply systematic data analysis workflows to complex datasets
- Perform comprehensive data quality assessments
- Implement debugging techniques for data analysis code
- Create professional analysis reports with clear documentation
- Validate analysis assumptions before drawing conclusions
- Handle edge cases and data quality issues professionally

## Part 1: Dataset Selection and Quality Assessment (6 points)

### Task 1.1: Dataset Acquisition

Choose **one** of the following options:

**Option A - Public Dataset:**
Find a dataset from Kaggle, UCI ML Repository, or similar source with:
- At least 1000 rows and 8+ columns
- Mix of numeric and categorical variables
- Some data quality issues (missing values, outliers, inconsistencies)
- A clear analysis question (e.g., "What predicts customer satisfaction?")

**Option B - Provided Dataset:**
Use the course dataset: `complex_sales_data.csv` (if available)

Document your choice in `dataset_selection.md`:
```markdown
# Dataset Selection

## Dataset Information
- **Name:** [Dataset name]
- **Source:** [Where you found it]
- **Size:** [rows x columns]
- **Description:** [What the data represents]

## Analysis Question
**Primary Question:** [What you want to learn from this data]

**Secondary Questions:**
1. [Additional question 1]
2. [Additional question 2]
3. [Additional question 3]

## Expected Challenges
Based on initial inspection, I expect these data quality issues:
- [Issue 1]
- [Issue 2]
- [Issue 3]
```

### Task 1.2: Comprehensive Quality Assessment

Create `src/quality_assessment.py`:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def comprehensive_quality_assessment(df, output_dir='output'):
    """
    Perform comprehensive data quality assessment
    
    Args:
        df: pandas DataFrame to assess
        output_dir: directory to save quality reports
    
    Returns:
        dict: Summary of quality issues found
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    quality_issues = {}
    
    print("COMPREHENSIVE DATA QUALITY ASSESSMENT")
    print("=" * 50)
    
    # 1. Basic dataset information
    print(f"\nDATASET OVERVIEW:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Missing values analysis
    print(f"\nMISSING VALUES ANALYSIS:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    if missing_data.any():
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Count', ascending=False)
        
        missing_cols = missing_summary[missing_summary['Missing_Count'] > 0]
        print(missing_cols)
        quality_issues['missing_values'] = missing_cols.to_dict()
    else:
        print("No missing values found")
    
    # 3. Duplicate analysis
    print(f"\nDUPLICATE ANALYSIS:")
    total_duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates} ({total_duplicates/len(df)*100:.1f}%)")
    
    if total_duplicates > 0:
        quality_issues['duplicates'] = total_duplicates
        
        # Analyze duplicates by columns
        duplicate_analysis = {}
        for col in df.columns:
            col_duplicates = df[col].duplicated().sum()
            duplicate_analysis[col] = col_duplicates
        
        print("Duplicate values by column:")
        for col, count in sorted(duplicate_analysis.items(), key=lambda x: x[1], reverse=True)[:5]:
            if count > 0:
                print(f"  {col}: {count}")
    
    # 4. Data type analysis
    print(f"\nDATA TYPE ANALYSIS:")
    type_summary = df.dtypes.value_counts()
    print(type_summary)
    
    # Check for potential type issues
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Check if numeric data is stored as object
        try:
            numeric_version = pd.to_numeric(df[col], errors='coerce')
            non_numeric_mask = numeric_version.isnull() & df[col].notnull()
            non_numeric_count = non_numeric_mask.sum()
            
            if non_numeric_count > 0 and non_numeric_count < len(df) * 0.5:
                print(f"  WARNING: {col} appears mostly numeric but has {non_numeric_count} non-numeric values")
                quality_issues[f'{col}_type_inconsistency'] = non_numeric_count
        except:
            pass
    
    # 5. Outlier detection for numeric columns
    print(f"\nOUTLIER ANALYSIS:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers_mask.sum()
        outlier_percent = outlier_count / len(df) * 100
        
        outlier_summary[col] = {
            'count': outlier_count,
            'percent': outlier_percent,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        if outlier_count > 0:
            print(f"  {col}: {outlier_count} outliers ({outlier_percent:.1f}%) outside [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    quality_issues['outliers'] = outlier_summary
    
    # 6. Consistency checks
    print(f"\nCONSISTENCY ANALYSIS:")
    consistency_issues = {}
    
    for col in object_cols:
        if df[col].dtype == 'object':
            # Check for whitespace issues
            has_whitespace = df[col].astype(str).str.strip() != df[col].astype(str)
            whitespace_count = has_whitespace.sum()
            
            if whitespace_count > 0:
                print(f"  {col}: {whitespace_count} values with leading/trailing whitespace")
                consistency_issues[f'{col}_whitespace'] = whitespace_count
            
            # Check for case inconsistencies
            if df[col].nunique() < len(df) * 0.1:  # If relatively few unique values
                case_variations = df[col].astype(str).str.lower().nunique() < df[col].nunique()
                if case_variations:
                    case_difference = df[col].nunique() - df[col].astype(str).str.lower().nunique()
                    print(f"  {col}: {case_difference} potential case inconsistencies")
                    consistency_issues[f'{col}_case_issues'] = case_difference
    
    quality_issues['consistency'] = consistency_issues
    
    # 7. Generate summary report
    create_quality_report(df, quality_issues, f'{output_dir}/quality_assessment.txt')
    
    return quality_issues

def create_quality_report(df, quality_issues, output_file):
    """Generate detailed quality assessment report"""
    with open(output_file, 'w') as f:
        f.write("DATA QUALITY ASSESSMENT REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        # Summary of issues
        f.write("QUALITY ISSUES SUMMARY\n")
        f.write("-" * 30 + "\n")
        
        total_issue_types = len([k for k, v in quality_issues.items() if v])
        f.write(f"Total issue types found: {total_issue_types}\n\n")
        
        for issue_type, details in quality_issues.items():
            if details:
                f.write(f"{issue_type.upper()}:\n")
                if isinstance(details, dict):
                    for key, value in details.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  Count: {details}\n")
                f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        
        if 'missing_values' in quality_issues and quality_issues['missing_values']:
            f.write("1. Address missing values through imputation or removal\n")
        
        if 'duplicates' in quality_issues and quality_issues['duplicates'] > 0:
            f.write("2. Investigate and remove duplicate records\n")
        
        if 'outliers' in quality_issues:
            f.write("3. Examine outliers for data entry errors or valid extreme values\n")
        
        if 'consistency' in quality_issues and quality_issues['consistency']:
            f.write("4. Standardize text data formatting and case consistency\n")
        
        f.write("\n5. Validate any data transformations before analysis\n")
        f.write("6. Document all data cleaning decisions\n")
        f.write("7. Consider the impact of quality issues on analysis conclusions\n")
    
    print(f"\nQuality assessment report saved: {output_file}")

if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv('your_dataset.csv')  # Replace with actual filename
    
    # Perform quality assessment
    issues = comprehensive_quality_assessment(df)
    
    print(f"\nQuality assessment complete. Found {len(issues)} types of issues.")
```

## Part 2: Systematic Exploratory Data Analysis (8 points)

### Task 2.1: Distribution Analysis

Create `src/distribution_analysis.py`:

```python
def analyze_distributions(df, output_dir='output'):
    """
    Comprehensive distribution analysis for all variables
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("NUMERIC VARIABLE DISTRIBUTIONS")
    print("=" * 40)
    
    for col in numeric_cols:
        print(f"\n{col.upper()}:")
        
        # Basic statistics
        stats = {
            'count': df[col].count(),
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
        
        # Print key statistics
        print(f"  Mean: {stats['mean']:.2f} | Median: {stats['median']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f} | Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
        print(f"  Skewness: {stats['skewness']:.2f} | Kurtosis: {stats['kurtosis']:.2f}")
        
        # Percentiles
        percentiles = df[col].quantile([0.1, 0.25, 0.75, 0.9])
        print(f"  10th-90th percentile: [{percentiles[0.1]:.1f}, {percentiles[0.9]:.1f}]")
        
        # Distribution characterization
        if abs(stats['skewness']) < 0.5:
            distribution_shape = "approximately symmetric"
        elif stats['skewness'] > 0.5:
            distribution_shape = "right-skewed (long right tail)"
        else:
            distribution_shape = "left-skewed (long left tail)"
        
        print(f"  Distribution: {distribution_shape}")
        
        results[col] = stats
        results[col]['percentiles'] = percentiles.to_dict()
        results[col]['distribution_shape'] = distribution_shape
    
    # Categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\n\nCATEGORICAL VARIABLE DISTRIBUTIONS")
    print("=" * 40)
    
    for col in categorical_cols:
        print(f"\n{col.upper()}:")
        
        value_counts = df[col].value_counts()
        unique_count = df[col].nunique()
        most_common = value_counts.index[0]
        most_common_freq = value_counts.iloc[0]
        
        print(f"  Unique values: {unique_count}")
        print(f"  Most common: '{most_common}' ({most_common_freq} occurrences, {most_common_freq/len(df)*100:.1f}%)")
        
        # Show top categories
        print("  Top 5 categories:")
        for i, (value, count) in enumerate(value_counts.head().items()):
            print(f"    {i+1}. {value}: {count} ({count/len(df)*100:.1f}%)")
        
        results[f"{col}_categorical"] = {
            'unique_count': unique_count,
            'most_common': most_common,
            'most_common_freq': most_common_freq,
            'value_counts': value_counts.head(10).to_dict()
        }
    
    return results

def identify_interesting_patterns(df):
    """
    Identify potentially interesting patterns in the data
    """
    patterns = []
    
    # Check for highly correlated numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_value))
        
        if high_corr_pairs:
            patterns.append("HIGH CORRELATIONS FOUND:")
            for col1, col2, corr in high_corr_pairs:
                patterns.append(f"  {col1} ↔ {col2}: {corr:.3f}")
    
    # Check for potential categorical-numeric relationships
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            if df[cat_col].nunique() <= 10:  # Reasonable number of categories
                group_means = df.groupby(cat_col)[num_col].mean()
                group_std = df.groupby(cat_col)[num_col].std()
                
                # Check if there's substantial variation between groups
                overall_mean = df[num_col].mean()
                max_deviation = abs(group_means - overall_mean).max()
                
                if max_deviation > df[num_col].std():  # Substantial difference
                    patterns.append(f"POTENTIAL RELATIONSHIP: {cat_col} → {num_col}")
                    patterns.append(f"  Group means vary substantially (max deviation: {max_deviation:.2f})")
    
    return patterns
```

### Task 2.2: Relationship Analysis

```python
def comprehensive_relationship_analysis(df, target_column=None):
    """
    Analyze relationships between variables
    """
    results = {}
    
    print("RELATIONSHIP ANALYSIS")
    print("=" * 30)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Correlation analysis for numeric variables
    if len(numeric_cols) > 1:
        print("\nNUMERIC CORRELATIONS:")
        correlation_matrix = df[numeric_cols].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                corr_pairs.append((abs(corr_value), col1, col2, corr_value))
        
        # Sort by correlation strength
        corr_pairs.sort(reverse=True)
        
        print("Strongest correlations:")
        for abs_corr, col1, col2, corr in corr_pairs[:5]:
            if abs_corr > 0.1:  # Only show meaningful correlations
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs_corr > 0.7 else "moderate" if abs_corr > 0.4 else "weak"
                print(f"  {col1} ↔ {col2}: {corr:.3f} ({strength} {direction})")
        
        results['numeric_correlations'] = correlation_matrix
    
    # Categorical-numeric relationships
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        print("\nCATEGORICAL-NUMERIC RELATIONSHIPS:")
        
        for cat_col in categorical_cols:
            if df[cat_col].nunique() <= 10:  # Manageable number of categories
                print(f"\n{cat_col.upper()} groups:")
                
                for num_col in numeric_cols[:3]:  # Top 3 numeric columns
                    group_stats = df.groupby(cat_col)[num_col].agg(['count', 'mean', 'std'])
                    
                    # Check for significant differences
                    overall_mean = df[num_col].mean()
                    max_diff = abs(group_stats['mean'] - overall_mean).max()
                    
                    if max_diff > df[num_col].std() * 0.5:  # Notable difference
                        print(f"  {num_col} varies by {cat_col}:")
                        for category, stats in group_stats.iterrows():
                            deviation = stats['mean'] - overall_mean
                            print(f"    {category}: mean={stats['mean']:.2f} (n={stats['count']}, dev={deviation:+.2f})")
                        
                        results[f'{cat_col}_{num_col}_relationship'] = group_stats
    
    # Target variable analysis (if specified)
    if target_column and target_column in df.columns:
        print(f"\nTARGET VARIABLE ANALYSIS: {target_column}")
        
        if target_column in numeric_cols:
            # Correlations with target
            target_correlations = df[numeric_cols].corrwith(df[target_column]).sort_values(key=abs, ascending=False)
            print("\nCorrelations with target:")
            for col, corr in target_correlations.head(5).items():
                if col != target_column:
                    print(f"  {col}: {corr:.3f}")
            
            results['target_correlations'] = target_correlations
        
        # Categorical relationships with target
        for cat_col in categorical_cols:
            if df[cat_col].nunique() <= 10:
                group_analysis = df.groupby(cat_col)[target_column].agg(['count', 'mean', 'std'])
                print(f"\n{target_column} by {cat_col}:")
                print(group_analysis.round(2))
                results[f'{cat_col}_target_analysis'] = group_analysis
    
    return results
```

## Part 3: Analysis Debugging and Validation (4 points)

### Task 3.1: Robust Analysis Functions

Create `src/robust_analysis.py`:

```python
def validate_assumptions(df, assumptions):
    """
    Validate analysis assumptions before proceeding
    """
    validation_results = {}
    
    print("VALIDATING ANALYSIS ASSUMPTIONS")
    print("=" * 40)
    
    for assumption_name, check_function in assumptions.items():
        try:
            result = check_function(df)
            validation_results[assumption_name] = {'passed': True, 'result': result}
            print(f"✓ {assumption_name}: PASSED")
            if result is not True:  # If there's additional info
                print(f"    Result: {result}")
        except Exception as e:
            validation_results[assumption_name] = {'passed': False, 'error': str(e)}
            print(f"✗ {assumption_name}: FAILED")
            print(f"    Error: {str(e)}")
    
    # Overall validation status
    all_passed = all(result['passed'] for result in validation_results.values())
    print(f"\nValidation {'PASSED' if all_passed else 'FAILED'}: {sum(r['passed'] for r in validation_results.values())}/{len(validation_results)} checks passed")
    
    return validation_results, all_passed

def robust_analysis_pipeline(df, target_column=None):
    """
    Robust analysis pipeline with validation and error handling
    """
    results = {}
    
    # Define assumptions for your specific analysis
    assumptions = {
        'sufficient_data': lambda df: len(df) >= 100,
        'target_exists': lambda df: target_column is None or target_column in df.columns,
        'numeric_data_available': lambda df: len(df.select_dtypes(include=[np.number]).columns) > 0,
        'no_all_null_columns': lambda df: not df.isnull().all().any(),
        'reasonable_missing_data': lambda df: (df.isnull().sum() / len(df)).max() < 0.8
    }
    
    # Validate assumptions
    validation_results, assumptions_met = validate_assumptions(df, assumptions)
    results['validation'] = validation_results
    
    if not assumptions_met:
        print("⚠️  Cannot proceed with analysis - assumptions not met")
        return results
    
    # Proceed with analysis if assumptions are met
    try:
        print("\n🔍 PROCEEDING WITH ANALYSIS")
        
        # Basic statistics with error handling
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                results['basic_stats'] = df[numeric_cols].describe()
                print("✓ Basic statistics computed")
            else:
                print("⚠️  No numeric columns for basic statistics")
        except Exception as e:
            print(f"✗ Basic statistics failed: {e}")
        
        # Distribution analysis with error handling
        try:
            results['distributions'] = analyze_distributions(df)
            print("✓ Distribution analysis completed")
        except Exception as e:
            print(f"✗ Distribution analysis failed: {e}")
        
        # Relationship analysis with error handling
        try:
            results['relationships'] = comprehensive_relationship_analysis(df, target_column)
            print("✓ Relationship analysis completed")
        except Exception as e:
            print(f"✗ Relationship analysis failed: {e}")
        
        # Pattern identification
        try:
            results['patterns'] = identify_interesting_patterns(df)
            print("✓ Pattern identification completed")
        except Exception as e:
            print(f"✗ Pattern identification failed: {e}")
        
    except Exception as e:
        print(f"✗ Analysis pipeline failed: {e}")
        results['error'] = str(e)
    
    return results

def create_analysis_checkpoint(df, step_name, output_dir='checkpoints'):
    """
    Create checkpoint of analysis state
    """
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{output_dir}/{step_name}_{timestamp}.csv'
    
    df.to_csv(filename, index=False)
    print(f"📁 Checkpoint saved: {filename}")
    
    return filename

if __name__ == "__main__":
    # Load and analyze your dataset
    df = pd.read_csv('your_dataset.csv')  # Replace with your dataset
    
    # Run robust analysis pipeline
    analysis_results = robust_analysis_pipeline(df, target_column='your_target_column')
    
    print("\nAnalysis pipeline complete!")
```

### Task 3.2: Analysis Report Generation

Create `src/report_generator.py`:

```python
def generate_comprehensive_report(df, analysis_results, output_file='analysis_report.md'):
    """
    Generate comprehensive analysis report
    """
    with open(output_file, 'w') as f:
        f.write("# Comprehensive Data Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Dataset:** {df.shape[0]:,} rows × {df.shape[1]} columns\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("### Key Findings\n")
        f.write("- [Key finding 1 based on your analysis]\n")
        f.write("- [Key finding 2 based on your analysis]\n")
        f.write("- [Key finding 3 based on your analysis]\n\n")
        
        f.write("### Data Quality Assessment\n")
        quality_issues = analysis_results.get('quality_issues', {})
        if quality_issues:
            f.write(f"- Found {len(quality_issues)} types of data quality issues\n")
            f.write("- Primary concerns: [list main issues]\n")
        else:
            f.write("- No significant data quality issues identified\n")
        f.write("\n")
        
        f.write("## Dataset Overview\n\n")
        f.write(f"**Size:** {df.shape[0]:,} records with {df.shape[1]} variables\n")
        f.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")
        
        # Variable types
        f.write("\n**Variable Types:**\n")
        type_counts = df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            f.write(f"- {dtype}: {count} variables\n")
        
        f.write("\n## Analysis Results\n\n")
        
        # Add specific analysis results based on what's in analysis_results
        for section, data in analysis_results.items():
            if section not in ['validation', 'error']:
                f.write(f"### {section.title().replace('_', ' ')}\n\n")
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        f.write(f"**{key}:** {value}\n")
                elif hasattr(data, 'to_string'):
                    f.write("```\n")
                    f.write(data.to_string())
                    f.write("\n```\n")
                else:
                    f.write(f"{data}\n")
                f.write("\n")
        
        f.write("## Methodology\n\n")
        f.write("This analysis employed systematic data science techniques:\n\n")
        f.write("1. **Data Quality Assessment** - Comprehensive evaluation of missing values, duplicates, outliers, and consistency issues\n")
        f.write("2. **Exploratory Data Analysis** - Distribution analysis and relationship identification\n")
        f.write("3. **Statistical Analysis** - Correlation analysis and group comparisons\n")
        f.write("4. **Validation** - Assumption checking and robustness testing\n")
        f.write("5. **Documentation** - Clear reporting of methods and findings\n\n")
        
        f.write("## Limitations and Considerations\n\n")
        f.write("### Data Limitations\n")
        f.write("- [Limitation 1 - e.g., sample size, time period]\n")
        f.write("- [Limitation 2 - e.g., missing variables, data quality]\n")
        f.write("- [Limitation 3 - e.g., potential biases]\n\n")
        
        f.write("### Analysis Limitations\n")
        f.write("- Analysis is descriptive/exploratory in nature\n")
        f.write("- Correlation does not imply causation\n")
        f.write("- Results should be validated with additional data\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("### Data Collection\n")
        f.write("1. [Recommendation for improving data quality]\n")
        f.write("2. [Recommendation for additional variables]\n")
        f.write("3. [Recommendation for data collection process]\n\n")
        
        f.write("### Further Analysis\n")
        f.write("1. [Suggestion for deeper analysis]\n")
        f.write("2. [Suggestion for predictive modeling]\n")
        f.write("3. [Suggestion for validation studies]\n\n")
        
        f.write("### Business Actions\n")
        f.write("1. [Actionable recommendation based on findings]\n")
        f.write("2. [Another actionable recommendation]\n")
        f.write("3. [Process improvement suggestion]\n\n")
        
        f.write("---\n")
        f.write("*This report was generated using systematic data analysis techniques and professional data science workflows.*\n")
    
    print(f"📊 Comprehensive report generated: {output_file}")
```

## Part 4: Complete Analysis Workflow (2 points)

Create `src/main_analysis.py` that orchestrates the complete analysis:

```python
import pandas as pd
from quality_assessment import comprehensive_quality_assessment
from distribution_analysis import analyze_distributions, identify_interesting_patterns
from distribution_analysis import comprehensive_relationship_analysis
from robust_analysis import robust_analysis_pipeline, validate_assumptions
from report_generator import generate_comprehensive_report

def main():
    """
    Complete analysis workflow
    """
    print("🚀 STARTING COMPREHENSIVE DATA ANALYSIS")
    print("=" * 60)
    
    # Load dataset
    try:
        df = pd.read_csv('your_dataset.csv')  # Replace with your dataset filename
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Step 1: Quality Assessment
    print("\n📊 STEP 1: DATA QUALITY ASSESSMENT")
    try:
        quality_issues = comprehensive_quality_assessment(df)
        print("✓ Quality assessment complete")
    except Exception as e:
        print(f"✗ Quality assessment failed: {e}")
        quality_issues = {}
    
    # Step 2: Robust Analysis Pipeline
    print("\n🔍 STEP 2: SYSTEMATIC ANALYSIS")
    target_column = 'your_target_column'  # Replace with your target or None
    analysis_results = robust_analysis_pipeline(df, target_column)
    analysis_results['quality_issues'] = quality_issues
    
    # Step 3: Generate Report
    print("\n📋 STEP 3: REPORT GENERATION")
    try:
        generate_comprehensive_report(df, analysis_results, 'comprehensive_analysis_report.md')
        print("✓ Report generated successfully")
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("\nFiles generated:")
    print("- output/quality_assessment.txt")
    print("- comprehensive_analysis_report.md")
    print("- Any checkpoint files in checkpoints/")
    
    return analysis_results

if __name__ == "__main__":
    results = main()
```

## Submission Requirements

### File Structure
```
assignment08/
├── README.md
├── dataset_selection.md
├── src/
│   ├── quality_assessment.py
│   ├── distribution_analysis.py
│   ├── robust_analysis.py
│   ├── report_generator.py
│   └── main_analysis.py
├── output/
│   ├── quality_assessment.txt
│   └── comprehensive_analysis_report.md
├── checkpoints/
│   └── [checkpoint files]
├── tests/
│   └── test_analysis.py
├── your_dataset.csv (or link to dataset)
└── .gitignore
```

### Testing File

Create `tests/test_analysis.py`:

```python
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from quality_assessment import comprehensive_quality_assessment
from robust_analysis import validate_assumptions, robust_analysis_pipeline

# Create sample data for testing
@pytest.fixture
def sample_data():
    np.random.seed(42)
    return pd.DataFrame({
        'numeric1': np.random.normal(100, 15, 200),
        'numeric2': np.random.uniform(0, 1000, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200),
        'text': ['item_' + str(i) for i in range(200)],
        'target': np.random.normal(50, 10, 200)
    })

def test_quality_assessment(sample_data):
    """Test quality assessment functions"""
    issues = comprehensive_quality_assessment(sample_data, output_dir='test_output')
    
    assert isinstance(issues, dict)
    assert 'outliers' in issues
    # Clean up test files
    import shutil
    if os.path.exists('test_output'):
        shutil.rmtree('test_output')

def test_assumption_validation(sample_data):
    """Test assumption validation"""
    assumptions = {
        'has_data': lambda df: len(df) > 0,
        'has_numeric_columns': lambda df: len(df.select_dtypes(include=[np.number]).columns) > 0
    }
    
    validation_results, passed = validate_assumptions(sample_data, assumptions)
    
    assert isinstance(validation_results, dict)
    assert isinstance(passed, bool)
    assert passed is True

def test_robust_pipeline(sample_data):
    """Test robust analysis pipeline"""
    results = robust_analysis_pipeline(sample_data, target_column='target')
    
    assert isinstance(results, dict)
    assert 'validation' in results

def test_error_handling():
    """Test error handling with problematic data"""
    # Create problematic dataset
    bad_data = pd.DataFrame({
        'all_null': [None] * 10,
        'single_value': [1] * 10
    })
    
    # Should handle gracefully without crashing
    try:
        issues = comprehensive_quality_assessment(bad_data, output_dir='test_output')
        assert isinstance(issues, dict)
        # Clean up
        import shutil
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
    except Exception as e:
        pytest.fail(f"Function should handle bad data gracefully, but raised: {e}")
```

## Grading Rubric

- **Dataset Selection & Documentation (2 pts):** Clear dataset choice with analysis questions
- **Quality Assessment (4 pts):** Comprehensive quality evaluation with proper issue identification
- **Distribution Analysis (4 pts):** Thorough analysis of variable distributions and patterns
- **Relationship Analysis (4 pts):** Systematic examination of variable relationships
- **Debugging & Validation (3 pts):** Proper assumption validation and error handling
- **Report Generation (2 pts):** Professional, comprehensive analysis report
- **Code Quality (1 pt):** Clean, documented, tested code

## Common Issues to Avoid

1. **Assumption blindness:** Always validate assumptions before analysis
2. **Quality ignorance:** Don't proceed with dirty data without acknowledging issues
3. **Single-method dependence:** Use multiple approaches to verify findings
4. **Undocumented decisions:** Document all data cleaning and analysis choices
5. **Error-prone code:** Test your analysis functions with different datasets

## Getting Help

- **Discord:** #assignment08-help channel
- **Office Hours:** [Schedule]
- **Resources:** 
  - pandas documentation for advanced functions
  - Course debugging techniques reference

Submit your repository link via Canvas by [due date] at 11:59 PM.