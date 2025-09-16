# Live Demo Guide: Data Analysis and Debugging Techniques

**Duration:** 30-35 minutes  
**Format:** Interactive coding demonstration with student participation

## Demo Overview

Show students professional data analysis workflows through systematic exploration of a real dataset, demonstrating quality assessment, debugging techniques, and robust analysis patterns they'll use in industry.

### Learning Objectives
Students will observe:
- Systematic approach to unknown data exploration
- Professional data quality assessment workflow
- Debugging techniques for data analysis code
- Validation and assumption checking patterns
- Professional documentation and reporting practices

---

## Pre-Demo Setup (5 minutes before class)

### 1. Environment Preparation
```bash
# Activate environment and install any missing packages
conda activate datasci217
pip install tabulate memory-profiler

# Create demo directory
mkdir -p demo_analysis
cd demo_analysis

# Download demo dataset (or prepare sample data)
# Use a dataset with known quality issues for demonstration
```

### 2. Dataset Preparation
Create `messy_sales_data.csv` with intentional issues:
```python
import pandas as pd
import numpy as np

# Create sample dataset with quality issues
np.random.seed(42)

data = {
    'customer_id': ['CUST_' + str(i) for i in range(1, 501)] + [None] * 20 + ['CUST_' + str(i) for i in range(1, 21)],  # Duplicates + missing
    'product_name': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor', None, ' Tablet ', 'laptop'], 541),
    'revenue': np.concatenate([np.random.normal(1000, 300, 500), [None] * 20, np.random.normal(5000, 1000, 21)]),
    'order_date': pd.date_range('2023-01-01', periods=541, freq='D').astype(str).tolist()[:520] + [None] * 21,
    'region': np.random.choice(['North', 'South', 'East', 'West', 'north', None], 541)
}

df = pd.DataFrame(data)
# Add some extreme outliers
df.loc[df.index[-5:], 'revenue'] = [50000, 75000, 100000, -1000, 200000]

df.to_csv('messy_sales_data.csv', index=False)
print("Demo dataset created with intentional quality issues")
```

### 3. Code Templates
Prepare `demo_analysis.py` skeleton:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_dataset(df, target_column=None):
    """Template function for systematic analysis"""
    pass

def comprehensive_quality_assessment(df):
    """Template for quality assessment"""
    pass

def debug_analysis_step(func, data, step_name):
    """Template for debugging wrapper"""
    pass
```

---

## Live Demo Script

### Opening (2 minutes)

> **Instructor:** "Today we're going to analyze a real dataset the way you would in industry - we don't know what we'll find, but we'll use systematic approaches to understand the data and handle any issues professionally. 
>
> I'm going to load this sales dataset and we'll discover together what challenges it presents. This mirrors what you'll encounter in real data science work."

```python
# Load the mysterious dataset
import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv('messy_sales_data.csv')

print(f"Initial look: {df.shape}")
df.head()
```

**Ask students:** "What should we check first before doing any analysis?"

### Step 1: Systematic Data Exploration (8 minutes)

```python
def explore_dataset_systematically(df):
    """
    Systematic dataset exploration - live coding
    """
    print("=== SYSTEMATIC DATA EXPLORATION ===")
    
    # Basic information
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Data types
    print(f"\nData types:")
    print(df.dtypes)
    
    # First look at data
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Check for obvious issues
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df

# Run exploration
df_info = explore_dataset_systematically(df)
```

**Pause for student questions:** "What do you notice about this data? Any red flags?"

**Expected student observations:**
- Missing values in multiple columns
- Different case variations in text
- Some very high revenue values
- Duplicates in customer_id

### Step 2: Quality Assessment - Live Problem Solving (10 minutes)

```python
def live_quality_assessment(df):
    """
    Quality assessment with live debugging
    """
    print("=== LIVE QUALITY ASSESSMENT ===")
    
    quality_issues = {}
    
    # 1. Missing values analysis
    print("\n1. MISSING VALUES:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    for col, count in missing_data[missing_data > 0].items():
        print(f"  {col}: {count} missing ({missing_percent[col]:.1f}%)")
        quality_issues[f'{col}_missing'] = count
    
    # 2. Duplicate analysis - demonstrate debugging
    print("\n2. DUPLICATE ANALYSIS:")
    total_duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates}")
    
    # Let's investigate customer_id duplicates specifically
    customer_duplicates = df['customer_id'].duplicated().sum()
    print(f"Customer ID duplicates: {customer_duplicates}")
    
    # Show the duplicates - this is where students learn debugging
    if customer_duplicates > 0:
        print("Let's examine these duplicates:")
        duplicate_customers = df[df['customer_id'].duplicated(keep=False) & df['customer_id'].notna()]
        print(duplicate_customers[['customer_id', 'product_name', 'revenue']].head(10))
    
    # 3. Data consistency issues
    print("\n3. CONSISTENCY ISSUES:")
    
    # Product names - demonstrate string cleaning needs
    print("Product name variations:")
    print(df['product_name'].value_counts())
    
    # Revenue outliers - demonstrate statistical detection
    print("\n4. OUTLIER DETECTION:")
    Q1 = df['revenue'].quantile(0.25)
    Q3 = df['revenue'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['revenue'] < lower_bound) | (df['revenue'] > upper_bound)]
    print(f"Revenue outliers: {len(outliers)} values outside [{lower_bound:.0f}, {upper_bound:.0f}]")
    
    if len(outliers) > 0:
        print("Extreme values:")
        print(outliers[['customer_id', 'product_name', 'revenue']].head())
    
    return quality_issues

# Run assessment
issues = live_quality_assessment(df)
```

**Interactive moment:** "Let's vote - which issues should we prioritize fixing first?"

### Step 3: Debugging Analysis Code (8 minutes)

```python
def demonstrate_debugging_techniques(df):
    """
    Show professional debugging approaches
    """
    print("=== DEBUGGING TECHNIQUES DEMONSTRATION ===")
    
    # 1. Validation wrapper
    def validate_data_assumptions(df, step_name):
        print(f"\nValidating assumptions for: {step_name}")
        
        checks = {
            'has_data': len(df) > 0,
            'has_revenue': 'revenue' in df.columns,
            'revenue_is_numeric': df['revenue'].dtype in ['int64', 'float64'],
            'has_non_null_revenue': df['revenue'].notna().any()
        }
        
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
            
            if not result:
                raise ValueError(f"Assumption failed: {check}")
    
    # 2. Analysis with debugging
    def calculate_revenue_by_product_safe(df):
        """Example analysis with built-in debugging"""
        
        # Validate assumptions first
        validate_data_assumptions(df, "revenue_by_product")
        
        print("\nCalculating revenue by product...")
        
        # Debug: Check data before analysis
        print(f"  Input: {len(df)} rows")
        print(f"  Non-null revenue: {df['revenue'].notna().sum()}")
        print(f"  Non-null products: {df['product_name'].notna().sum()}")
        
        # Perform analysis with error handling
        try:
            # Clean data first
            clean_data = df[df['revenue'].notna() & df['product_name'].notna()].copy()
            print(f"  After cleaning: {len(clean_data)} rows")
            
            # Calculate results
            revenue_by_product = clean_data.groupby('product_name')['revenue'].agg(['count', 'mean', 'sum'])
            
            print("  Analysis successful!")
            return revenue_by_product
            
        except Exception as e:
            print(f"  ✗ Analysis failed: {str(e)}")
            # Return debugging info
            return None
    
    # 3. Demonstrate the debugging in action
    try:
        results = calculate_revenue_by_product_safe(df)
        if results is not None:
            print("\nResults:")
            print(results.round(2))
    except Exception as e:
        print(f"Caught error: {e}")
        print("This is how we handle analysis failures professionally")

# Run debugging demonstration
demonstrate_debugging_techniques(df)
```

**Ask students:** "What did we learn about handling data quality issues during analysis?"

### Step 4: Professional Analysis Pattern (5 minutes)

```python
def complete_professional_workflow(df):
    """
    Demonstrate complete professional workflow
    """
    print("=== PROFESSIONAL ANALYSIS WORKFLOW ===")
    
    # Step 1: Data cleaning decisions (document everything)
    print("\n1. CLEANING DECISIONS:")
    print("  - Remove rows with missing customer_id or revenue")
    print("  - Standardize product names (lowercase, strip whitespace)")
    print("  - Flag extreme revenue outliers for investigation")
    
    # Step 2: Implement cleaning with logging
    original_size = len(df)
    
    # Clean missing values
    df_clean = df[df['customer_id'].notna() & df['revenue'].notna()].copy()
    print(f"  After removing missing: {len(df_clean)} rows ({original_size - len(df_clean)} removed)")
    
    # Clean product names
    df_clean['product_name_clean'] = df_clean['product_name'].str.lower().str.strip()
    print(f"  Product names standardized")
    
    # Flag outliers but keep them (business decision)
    Q1, Q3 = df_clean['revenue'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_clean['is_outlier'] = (df_clean['revenue'] < Q1 - 1.5*IQR) | (df_clean['revenue'] > Q3 + 1.5*IQR)
    print(f"  Flagged {df_clean['is_outlier'].sum()} outliers")
    
    # Step 3: Analysis with validation
    print("\n2. ANALYSIS RESULTS:")
    
    # Revenue by product (cleaned)
    product_analysis = df_clean.groupby('product_name_clean')['revenue'].agg(['count', 'mean', 'sum'])
    print("Revenue by product (cleaned data):")
    print(product_analysis.round(2))
    
    # Outlier analysis separately
    if df_clean['is_outlier'].any():
        print("\nOutlier investigation:")
        outlier_summary = df_clean[df_clean['is_outlier']][['customer_id', 'product_name_clean', 'revenue']]
        print(outlier_summary)
    
    print("\n3. NEXT STEPS:")
    print("  - Investigate high-revenue outliers with business team")
    print("  - Set up automated quality monitoring")
    print("  - Document cleaning decisions for future reference")
    
    return df_clean

# Run complete workflow
final_data = complete_professional_workflow(df)
```

### Closing (2 minutes)

> **Instructor:** "Notice how we approached this systematically:
> 1. **Explored first** - no assumptions about the data
> 2. **Assessed quality** - found issues before analyzing  
> 3. **Debugged proactively** - built validation into our analysis
> 4. **Documented decisions** - others can understand our choices
> 5. **Planned next steps** - analysis is just the beginning
>
> This is exactly how you'll work with real data in industry. Always expect quality issues, always validate assumptions, and always document your decisions."

---

## Student Participation Moments

### Discussion Questions Throughout Demo

1. **After initial data load:** "What should we check first?"
2. **After seeing missing values:** "How would you prioritize these issues?"
3. **After finding outliers:** "Should we remove these or investigate them?"
4. **After cleaning decisions:** "What would happen if we made different choices?"
5. **During debugging:** "Why is validation important in analysis code?"

### Interactive Coding Moments

1. **Student prediction:** Before running quality assessment, have students guess what issues they'll find
2. **Student decision:** When finding duplicates, ask class to vote on handling approach
3. **Student debugging:** Present a "broken" analysis function and ask students to identify the issue
4. **Student design:** Ask how they would modify the validation for a different type of dataset

---

## Common Demo Issues & Solutions

### Technical Issues

1. **Package not installed:**
   ```bash
   pip install tabulate memory-profiler
   ```

2. **Dataset generation fails:**
   - Have backup CSV file ready
   - Use simpler synthetic data if needed

3. **Memory issues with large datasets:**
   - Reduce dataset size
   - Use `df.head(1000)` for demonstration

### Timing Issues

**Running ahead of schedule:**
- Add more detailed exploration of specific quality issues
- Show additional debugging techniques
- Demonstrate memory profiling

**Running behind schedule:**
- Skip detailed outlier investigation
- Focus on main concepts: systematic approach, validation, documentation
- Save advanced debugging for next class

### Student Engagement

**Low participation:**
- Ask more direct questions: "Sarah, what do you think this outlier represents?"
- Use polls/votes for decisions
- Break into small groups for 2-minute discussions

**Too many questions:**
- Acknowledge questions but defer complex ones: "Great question - let's address that in next week's deep dive"
- Keep focus on main workflow pattern

---

## Post-Demo Activities

### Immediate Follow-up (5 minutes)
> "Now you try - I want you to load your assignment dataset and run through the first two steps we just demonstrated. You have 5 minutes to explore and find at least 3 quality issues."

### Assignment Connection
> "Your assignment this week asks you to build exactly these systems we demonstrated. You'll create your own quality assessment function, build debugging into your analysis, and generate a professional report. Start with the patterns we just showed."

### Next Class Preview
> "Next week we'll build on this foundation with advanced data manipulation techniques. We'll see how to handle complex data transformations while maintaining the systematic, validated approach you learned today."