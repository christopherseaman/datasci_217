# L02: Data Structures + Development Environment Mastery
**Duration**: 3.5 hours  
**Learning Objectives**: Master Python data structures for data science applications, implement advanced Git workflows for professional collaboration, organize code with functions and modules, and establish documentation best practices

---

## Opening: The Data Structures That Power Discovery

Imagine you're analyzing customer behavior across a global e-commerce platform. You have millions of transactions, thousands of products, and complex relationships between users, purchases, and recommendations. How do you organize this complexity? How do you make it searchable, analyzable, and actionable?

The answer lies in choosing the right data structures—not just as containers for information, but as the fundamental building blocks that determine what's possible in your analysis. Today, we're not just learning lists and dictionaries; we're mastering the data organization patterns that separate inefficient analysts from data science professionals.

By the end of today's session, you'll understand why a pandas DataFrame is built on dictionaries, how to structure data for maximum analytical power, and how to collaborate on complex projects using professional Git workflows. These aren't academic exercises—they're the core competencies that enable every advanced technique we'll explore this semester.

## Chapter 1: Data Structures for Data Science - Beyond Basic Containers

### The Strategic View: Matching Structure to Analysis

Before diving into syntax, let's establish the strategic mindset. In data science, your choice of data structure directly impacts:
- **Performance**: How fast your analysis runs
- **Memory usage**: Whether you can handle large datasets
- **Code clarity**: How easily others understand your work
- **Analytical possibilities**: What questions you can ask of your data

### Lists: The Workhorses of Sequential Data

Lists aren't just arrays—they're the foundation for time series, ordered relationships, and iterative processing patterns.

**Time Series Data Patterns**:
```python
# Stock prices over time - order matters
stock_prices = [142.50, 145.20, 144.80, 147.10, 146.95]
daily_returns = []

# Calculate daily returns (classic data science pattern)
for i in range(1, len(stock_prices)):
    return_pct = (stock_prices[i] - stock_prices[i-1]) / stock_prices[i-1] * 100
    daily_returns.append(round(return_pct, 2))

print(f"Daily returns: {daily_returns}")
# Output: [1.89, -0.28, 1.59, -0.10]
```

**List Comprehensions for Data Transformation**:
```python
# Customer ages - need to categorize for marketing segments
customer_ages = [22, 35, 41, 28, 52, 33, 45, 29, 38, 44]

# Traditional approach (verbose)
age_categories = []
for age in customer_ages:
    if age < 30:
        age_categories.append("young_adult")
    elif age < 45:
        age_categories.append("middle_aged")
    else:
        age_categories.append("mature")

# Data science approach (concise and readable)
age_categories = [
    "young_adult" if age < 30 
    else "middle_aged" if age < 45 
    else "mature" 
    for age in customer_ages
]

print(dict(zip(customer_ages, age_categories)))
```

**Advanced List Operations for Analysis**:
```python
# Sales data with outliers - need robust analysis
monthly_sales = [120000, 135000, 142000, 950000, 148000, 155000, 162000]

# Identify outliers using interquartile range method
def detect_outliers(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Calculate quartiles
    q1_index = n // 4
    q3_index = 3 * n // 4
    q1 = sorted_data[q1_index]
    q3 = sorted_data[q3_index]
    iqr = q3 - q1
    
    # Outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Identify outliers
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    clean_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return outliers, clean_data

outliers, clean_sales = detect_outliers(monthly_sales)
print(f"Outliers detected: {outliers}")
print(f"Clean dataset: {clean_sales}")
```

### Dictionaries: The Swiss Army Knife of Data Organization

Dictionaries model real-world relationships and enable lightning-fast lookups—essential for data aggregation, grouping, and analysis.

**Data Aggregation Patterns**:
```python
# Customer transaction data - need to aggregate by region
transactions = [
    {"customer_id": "C001", "region": "North", "amount": 150.00},
    {"customer_id": "C002", "region": "South", "amount": 200.00},
    {"customer_id": "C003", "region": "North", "amount": 175.00},
    {"customer_id": "C004", "region": "West", "amount": 300.00},
    {"customer_id": "C005", "region": "South", "amount": 125.00}
]

# Regional sales aggregation
regional_sales = {}
regional_counts = {}

for transaction in transactions:
    region = transaction["region"]
    amount = transaction["amount"]
    
    # Initialize if first transaction for region
    if region not in regional_sales:
        regional_sales[region] = 0
        regional_counts[region] = 0
    
    # Accumulate totals
    regional_sales[region] += amount
    regional_counts[region] += 1

# Calculate averages
regional_averages = {
    region: regional_sales[region] / regional_counts[region]
    for region in regional_sales
}

print("Regional Analysis:")
for region in regional_sales:
    print(f"{region}: Total=${regional_sales[region]:.2f}, "
          f"Average=${regional_averages[region]:.2f}, "
          f"Count={regional_counts[region]}")
```

**Nested Data Structures for Complex Relationships**:
```python
# Product catalog with hierarchical categories
product_catalog = {
    "electronics": {
        "computers": {
            "laptops": ["MacBook Pro", "Dell XPS", "ThinkPad"],
            "desktops": ["iMac", "Dell OptiPlex", "HP Pavilion"]
        },
        "phones": {
            "smartphones": ["iPhone 14", "Samsung Galaxy", "Google Pixel"],
            "feature_phones": ["Nokia 3310", "Jitterbug"]
        }
    },
    "clothing": {
        "men": {
            "shirts": ["Cotton T-Shirt", "Dress Shirt", "Polo"],
            "pants": ["Jeans", "Chinos", "Dress Pants"]
        },
        "women": {
            "dresses": ["Summer Dress", "Evening Gown", "Casual Dress"],
            "tops": ["Blouse", "Tank Top", "Sweater"]
        }
    }
}

def find_products_in_category(catalog, category_path):
    """
    Navigate nested dictionary structure to find products.
    
    Args:
        catalog (dict): Product catalog
        category_path (list): Path to category (e.g., ["electronics", "computers", "laptops"])
    
    Returns:
        list: Products in specified category
    """
    current_level = catalog
    
    try:
        for category in category_path:
            current_level = current_level[category]
        return current_level if isinstance(current_level, list) else None
    except KeyError:
        return f"Category path {' -> '.join(category_path)} not found"

# Usage examples
laptops = find_products_in_category(product_catalog, ["electronics", "computers", "laptops"])
print(f"Available laptops: {laptops}")
```

### Sets: Uniqueness and Relationships

Sets excel at data deduplication, membership testing, and relationship analysis.

**Data Cleaning with Sets**:
```python
# Customer email list with duplicates - need to clean
raw_emails = [
    "john@email.com", "mary@email.com", "john@email.com", 
    "sarah@email.com", "mike@email.com", "mary@email.com"
]

# Remove duplicates while preserving insertion order (Python 3.7+)
unique_emails = list(dict.fromkeys(raw_emails))
print(f"Cleaned email list: {unique_emails}")

# Set operations for customer segmentation analysis
email_subscribers = {"john@email.com", "mary@email.com", "sarah@email.com", "alex@email.com"}
app_users = {"mary@email.com", "mike@email.com", "sarah@email.com", "lisa@email.com"}
premium_customers = {"sarah@email.com", "alex@email.com", "david@email.com"}

# Analyze customer segments
multi_channel_users = email_subscribers & app_users  # Intersection
email_only = email_subscribers - app_users           # Difference
all_customers = email_subscribers | app_users        # Union
premium_non_subscribers = premium_customers - email_subscribers

print("Customer Segmentation Analysis:")
print(f"Multi-channel users: {multi_channel_users}")
print(f"Email-only subscribers: {email_only}")
print(f"Total customer base: {len(all_customers)} customers")
print(f"Premium customers not on email list: {premium_non_subscribers}")
```

### Practical Exercise 1: Data Structure Olympics

**Objective**: Solve complex data analysis challenges using appropriate data structure choices.

**Challenge 1: Product Inventory Analysis**
```python
# Given: Product sales data for the last month
sales_data = [
    {"product": "laptop", "category": "electronics", "price": 899, "quantity": 15},
    {"product": "t-shirt", "category": "clothing", "price": 25, "quantity": 50},
    {"product": "smartphone", "category": "electronics", "price": 699, "quantity": 20},
    {"product": "jeans", "category": "clothing", "price": 79, "quantity": 30},
    {"product": "tablet", "category": "electronics", "price": 299, "quantity": 25}
]

# Tasks:
# 1. Calculate total revenue by category
# 2. Find the top 3 products by revenue
# 3. Identify which categories have average price > $100
# 4. Create a set of all unique categories
```

**Challenge 2: Customer Behavior Analysis**
```python
# Given: Customer browsing and purchase history
browsing_history = [
    {"customer": "C001", "pages": ["home", "products", "laptop", "cart"]},
    {"customer": "C002", "pages": ["home", "search", "smartphone", "reviews", "cart", "checkout"]},
    {"customer": "C003", "pages": ["home", "products", "tablet"]},
    {"customer": "C004", "pages": ["home", "search", "laptop", "compare", "cart", "checkout"]}
]

# Tasks:
# 1. Identify conversion rate (customers who reached checkout)
# 2. Find most common page sequences
# 3. Calculate average session length
# 4. Identify customers who abandoned carts
```

## Chapter 2: Advanced Git Workflows - Professional Collaboration Mastery

### Beyond Basic Commits: Git as a Collaboration Platform

Git isn't just version control—it's the infrastructure that enables distributed teams to work on complex projects simultaneously. Understanding advanced Git workflows separates individual contributors from team leaders who can coordinate large-scale development efforts.

### Branching Strategies for Data Science Projects

**Feature Branch Workflow**:
```bash
# Start new analysis feature
git checkout -b feature/customer-segmentation-analysis

# Work on your analysis...
# Create customer_segmentation.py
# Develop clustering algorithms
# Add visualization functions

# Commit incremental progress
git add customer_segmentation.py
git commit -m "Implement K-means clustering for customer segmentation

- Add data preprocessing pipeline
- Implement elbow method for optimal K selection
- Create visualization functions for cluster analysis"

# Continue development...
git add tests/test_segmentation.py
git commit -m "Add comprehensive tests for segmentation module

- Test data preprocessing edge cases
- Validate clustering parameter selection
- Ensure visualization functions handle empty datasets"

# Merge back to main branch
git checkout main
git merge feature/customer-segmentation-analysis
git branch -d feature/customer-segmentation-analysis
```

**Collaborative Workflow with Pull Requests**:
```bash
# Clone team repository
git clone https://github.com/team/datasci-project.git
cd datasci-project

# Create feature branch for your contribution
git checkout -b feature/add-predictive-modeling

# Make your changes...
# ... work on your feature ...

# Push feature branch to remote
git push -u origin feature/add-predictive-modeling

# Create pull request on GitHub
# Team reviews your code
# Incorporate feedback and push updates
# Merge pull request when approved
```

### Handling Merge Conflicts in Data Science Context

**Common Conflict Scenario**:
```python
# Your version (HEAD)
def calculate_conversion_rate(data):
    """Calculate customer conversion rate."""
    total_visitors = len(data['visitors'])
    total_conversions = len(data['conversions'])
    return total_conversions / total_visitors

# Teammate's version (incoming)
def calculate_conversion_rate(data):
    """Calculate customer conversion rate with error handling."""
    total_visitors = len(data.get('visitors', []))
    total_conversions = len(data.get('conversions', []))
    
    if total_visitors == 0:
        return 0.0
    
    return total_conversions / total_visitors * 100
```

**Resolving the Conflict**:
```python
def calculate_conversion_rate(data):
    """
    Calculate customer conversion rate with comprehensive error handling.
    
    Args:
        data (dict): Dictionary containing 'visitors' and 'conversions' lists
        
    Returns:
        float: Conversion rate as percentage (0-100)
    """
    total_visitors = len(data.get('visitors', []))
    total_conversions = len(data.get('conversions', []))
    
    if total_visitors == 0:
        return 0.0
    
    return (total_conversions / total_visitors) * 100
```

### Advanced Git Operations for Data Science

**Interactive Rebase for Clean History**:
```bash
# Clean up commit history before merging
git rebase -i HEAD~3

# In the interactive editor:
# pick abc1234 Initial data preprocessing
# squash def5678 Fix preprocessing bug
# squash ghi9012 Update preprocessing comments

# Result: Clean, logical commit history
```

**Git Stash for Context Switching**:
```bash
# You're working on feature A when urgent bug fix needed
git stash push -m "WIP: customer segmentation analysis"

# Switch to main branch and fix bug
git checkout main
git checkout -b hotfix/data-loading-error
# ... fix the bug ...
git commit -m "Fix data loading error for CSV files with BOM"

# Return to your original work
git checkout feature/customer-segmentation
git stash pop

# Continue where you left off
```

### Practical Exercise 2: Git Collaboration Simulation

**Objective**: Practice advanced Git workflows in a simulated team environment.

**Setup**:
1. Create a repository with initial data analysis project
2. Simulate multiple contributors working on different features
3. Practice conflict resolution and collaborative workflows

**Scenario Tasks**:
1. **Feature Development**: Each team member works on different analysis modules
2. **Conflict Resolution**: Intentionally create conflicts and resolve them professionally
3. **Code Review**: Practice reviewing pull requests and providing constructive feedback
4. **Release Management**: Tag versions and maintain stable main branch

## Chapter 3: Functions and Modules - Code Organization for Scale

### From Scripts to Systems: Building Reusable Code

As data science projects grow in complexity, script-based approaches become unmaintainable. Professional data scientists organize code into functions, modules, and packages that can be tested, documented, and reused across projects.

### Function Design Patterns for Data Science

**Pure Functions for Reliability**:
```python
def calculate_portfolio_return(weights, returns):
    """
    Calculate portfolio return given asset weights and returns.
    
    Pure function: same inputs always produce same outputs,
    no side effects, easy to test and reason about.
    
    Args:
        weights (list): Asset allocation weights (must sum to 1.0)
        returns (list): Individual asset returns
        
    Returns:
        float: Portfolio return
        
    Raises:
        ValueError: If weights don't sum to 1.0 or lengths don't match
    """
    if len(weights) != len(returns):
        raise ValueError("Weights and returns must have same length")
    
    if abs(sum(weights) - 1.0) > 1e-10:  # Account for floating point precision
        raise ValueError("Weights must sum to 1.0")
    
    portfolio_return = sum(w * r for w, r in zip(weights, returns))
    return portfolio_return

# Usage - predictable and testable
tech_weights = [0.4, 0.3, 0.3]
tech_returns = [0.12, 0.08, 0.15]
portfolio_return = calculate_portfolio_return(tech_weights, tech_returns)
```

**Higher-Order Functions for Analysis Pipelines**:
```python
def apply_transformation_pipeline(data, transformations):
    """
    Apply a series of transformations to data.
    
    Enables composition of data processing steps for complex pipelines.
    
    Args:
        data: Input data to transform
        transformations (list): List of transformation functions
        
    Returns:
        Transformed data after applying all functions
    """
    result = data
    for transform_func in transformations:
        result = transform_func(result)
    return result

# Define transformation functions
def remove_outliers(data, threshold=2.5):
    """Remove values beyond threshold standard deviations."""
    import statistics
    
    if len(data) < 2:
        return data
    
    mean = statistics.mean(data)
    stdev = statistics.stdev(data)
    
    return [x for x in data if abs(x - mean) <= threshold * stdev]

def normalize_data(data):
    """Normalize data to 0-1 range."""
    if not data:
        return data
    
    min_val = min(data)
    max_val = max(data)
    
    if min_val == max_val:
        return [0.5] * len(data)  # All values identical
    
    return [(x - min_val) / (max_val - min_val) for x in data]

def log_transform(data):
    """Apply log transformation (safe for values <= 0)."""
    import math
    return [math.log(x) if x > 0 else 0 for x in data]

# Use pipeline for data preprocessing
raw_data = [1, 2, 50, 3, 4, 5, 100, 6, 7, 8]
pipeline = [remove_outliers, normalize_data]
processed_data = apply_transformation_pipeline(raw_data, pipeline)

print(f"Original: {raw_data}")
print(f"Processed: {processed_data}")
```

### Module Organization for Professional Projects

**Creating Analysis Modules**:

```python
# File: analysis/statistics.py
"""
Statistical analysis utilities for data science projects.

This module provides commonly used statistical functions
with robust error handling and comprehensive documentation.
"""

import math
from typing import List, Optional, Tuple

def descriptive_stats(data: List[float]) -> dict:
    """
    Calculate comprehensive descriptive statistics.
    
    Args:
        data: List of numeric values
        
    Returns:
        Dictionary containing mean, median, mode, std dev, etc.
        
    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> stats = descriptive_stats(data)
        >>> print(stats['mean'])
        3.0
    """
    if not data:
        return {"error": "Empty dataset"}
    
    sorted_data = sorted(data)
    n = len(data)
    
    # Calculate statistics
    mean = sum(data) / n
    median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    # Standard deviation
    variance = sum((x - mean) ** 2 for x in data) / (n - 1) if n > 1 else 0
    std_dev = math.sqrt(variance)
    
    return {
        "count": n,
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "min": min(data),
        "max": max(data),
        "range": max(data) - min(data)
    }

def correlation_coefficient(x: List[float], y: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient between two variables.
    
    Args:
        x, y: Lists of numeric values (must be same length)
        
    Returns:
        Correlation coefficient (-1 to 1)
        
    Raises:
        ValueError: If inputs have different lengths or insufficient data
    """
    if len(x) != len(y):
        raise ValueError("Input lists must have same length")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 data points for correlation")
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate covariance and standard deviations
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if std_x == 0 or std_y == 0:
        return 0.0  # No correlation if either variable is constant
    
    correlation = covariance / (std_x * std_y)
    return correlation
```

**Using Modules in Analysis Scripts**:
```python
# File: scripts/customer_analysis.py
"""
Customer behavior analysis using reusable statistical modules.
"""

# Import our custom modules
from analysis.statistics import descriptive_stats, correlation_coefficient
from analysis.visualization import create_scatter_plot, create_histogram

def analyze_customer_spending():
    """Comprehensive customer spending analysis."""
    
    # Sample data (in real project, load from CSV/database)
    customer_ages = [25, 30, 35, 28, 45, 33, 29, 41, 38, 44]
    spending_amounts = [150, 200, 350, 175, 500, 225, 165, 400, 300, 450]
    
    # Use our statistical functions
    age_stats = descriptive_stats(customer_ages)
    spending_stats = descriptive_stats(spending_amounts)
    age_spending_correlation = correlation_coefficient(customer_ages, spending_amounts)
    
    # Generate report
    print("Customer Demographics Analysis")
    print("=" * 40)
    print(f"Average customer age: {age_stats['mean']:.1f} years")
    print(f"Age range: {age_stats['range']:.1f} years")
    print(f"Average spending: ${spending_stats['mean']:.2f}")
    print(f"Spending standard deviation: ${spending_stats['std_dev']:.2f}")
    print(f"Age-spending correlation: {age_spending_correlation:.3f}")
    
    # Interpretation
    if age_spending_correlation > 0.5:
        print("Strong positive correlation: Older customers tend to spend more")
    elif age_spending_correlation > 0.2:
        print("Moderate positive correlation: Some relationship between age and spending")
    else:
        print("Weak correlation: Age doesn't strongly predict spending")

if __name__ == "__main__":
    analyze_customer_spending()
```

### Practical Exercise 3: Building a Reusable Analysis Module

**Objective**: Create a comprehensive data analysis module that demonstrates professional code organization.

**Module Requirements**:
1. **Statistical Functions**: Implement descriptive statistics, correlation analysis, and hypothesis testing
2. **Data Cleaning Functions**: Handle missing values, outliers, and data validation
3. **Visualization Functions**: Create standardized plots with consistent styling
4. **Documentation**: Comprehensive docstrings and usage examples
5. **Error Handling**: Robust input validation and informative error messages

**Project Structure**:
```
analysis_toolkit/
├── __init__.py
├── statistics.py       # Statistical analysis functions
├── cleaning.py         # Data preprocessing functions
├── visualization.py    # Plotting utilities
└── utils.py           # Helper functions
```

## Chapter 4: Professional Documentation - Making Code Communicate

### Documentation as Data Science Infrastructure

Documentation isn't just comments—it's the infrastructure that makes your analysis reproducible, your code maintainable, and your insights communicable. Professional data scientists treat documentation as a first-class deliverable, not an afterthought.

### Markdown for Data Science Communication

**Project README Template**:
```markdown
# Customer Segmentation Analysis

## Overview
This project implements unsupervised machine learning techniques to identify distinct customer segments based on purchasing behavior, demographics, and engagement metrics.

## Business Problem
The marketing team needs to create targeted campaigns but lacks clear customer segments. Current one-size-fits-all approach results in low engagement rates and inefficient ad spend.

## Data Sources
- **Customer Database**: Demographics, registration dates, location data
- **Transaction History**: Purchase amounts, frequency, product categories
- **Engagement Metrics**: Email opens, website visits, app usage

## Methodology
1. **Data Preprocessing**: Handle missing values, outlier detection, feature scaling
2. **Exploratory Analysis**: Distribution analysis, correlation studies
3. **Clustering Analysis**: K-means clustering with optimal K selection
4. **Segment Validation**: Statistical testing and business sense validation
5. **Actionable Insights**: Segment profiles and marketing recommendations

## Key Findings
- Identified 4 distinct customer segments with different value propositions
- High-value segment (23% of customers) generates 67% of revenue
- Young professionals segment shows highest engagement but lowest spending
- Price-sensitive segment responds best to discount campaigns

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python run_analysis.py

# Generate customer segment report
python generate_report.py --output reports/segments.html
```

## Project Structure
```
customer_segmentation/
├── data/
│   ├── raw/                    # Original data files
│   └── processed/              # Cleaned datasets
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_clustering_analysis.ipynb
│   └── 03_validation_testing.ipynb
├── src/
│   ├── preprocessing.py        # Data cleaning functions
│   ├── clustering.py          # Clustering algorithms
│   ├── validation.py          # Model validation
│   └── visualization.py       # Plotting functions
├── tests/                     # Unit tests
├── reports/                   # Generated analysis reports
└── requirements.txt
```

## Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-analysis`)
3. Run tests (`python -m pytest tests/`)
4. Submit pull request with detailed description

## Contact
- **Analyst**: [Your Name] ([email])
- **Team Lead**: [Manager Name] ([email])
- **Stakeholder**: Marketing Team ([email])
```

### Python Docstring Standards

**Google Style Docstrings** (recommended for data science):
```python
def segment_customers(data, n_segments=4, random_state=42):
    """
    Perform customer segmentation using K-means clustering.
    
    This function implements a complete customer segmentation pipeline
    including data preprocessing, optimal cluster selection, and 
    segment interpretation.
    
    Args:
        data (pd.DataFrame): Customer data with features for clustering.
            Must contain columns: ['age', 'spending_score', 'annual_income']
        n_segments (int, optional): Number of customer segments to create.
            Defaults to 4. If None, will use elbow method for selection.
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.
    
    Returns:
        tuple: A tuple containing:
            - segment_labels (np.array): Cluster labels for each customer
            - cluster_centers (np.array): Centroid coordinates for each segment
            - segment_summary (dict): Statistical summary of each segment
    
    Raises:
        ValueError: If required columns are missing from data.
        TypeError: If data is not a pandas DataFrame.
    
    Example:
        >>> import pandas as pd
        >>> customer_data = pd.read_csv('customers.csv')
        >>> labels, centers, summary = segment_customers(customer_data)
        >>> print(f"Created {len(set(labels))} customer segments")
        Created 4 customer segments
    
    Note:
        This function assumes data has been preprocessed (missing values
        handled, outliers treated). For raw data, use preprocess_customer_data()
        first.
    
    See Also:
        preprocess_customer_data(): Data cleaning pipeline
        validate_segments(): Statistical validation of clustering results
        visualize_segments(): Create segment visualization plots
    """
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    required_columns = ['age', 'spending_score', 'annual_income']
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Implementation continues...
    pass
```

### Documentation-Driven Development

**Analysis Planning Document**:
```markdown
# Customer Lifetime Value Analysis - Technical Specification

## Analysis Objectives
1. **Primary**: Develop predictive model for customer lifetime value (CLV)
2. **Secondary**: Identify key drivers of high-value customers
3. **Tertiary**: Create actionable customer retention strategies

## Data Requirements
### Input Data
- Customer transaction history (2+ years)
- Customer demographic information
- Product interaction data
- Customer service interactions

### Data Quality Criteria
- Minimum 1000 customers with complete transaction history
- Missing value percentage < 15% for key features
- Transaction data covers full seasonal cycles

## Methodology
### Phase 1: Exploratory Data Analysis (Week 1)
- [ ] Data quality assessment and cleaning pipeline
- [ ] Univariate analysis of all variables
- [ ] Bivariate analysis of CLV relationships
- [ ] Correlation analysis and feature importance

### Phase 2: Feature Engineering (Week 2)
- [ ] RFM analysis (Recency, Frequency, Monetary)
- [ ] Customer lifecycle stage identification
- [ ] Behavioral feature extraction
- [ ] Time-based aggregation features

### Phase 3: Model Development (Week 3)
- [ ] Train-test split with temporal considerations
- [ ] Baseline model (simple linear regression)
- [ ] Advanced models (Random Forest, XGBoost)
- [ ] Model validation and selection

### Phase 4: Insights and Deployment (Week 4)
- [ ] Feature importance analysis
- [ ] Customer segment insights
- [ ] Model deployment pipeline
- [ ] Monitoring and maintenance plan

## Success Criteria
- **Model Performance**: R² > 0.75 on validation set
- **Business Impact**: Identify 80% of high-value customers
- **Actionability**: Provide specific retention strategies for each segment

## Deliverables
1. **Technical Report**: Model development and validation results
2. **Business Report**: Insights and recommendations for stakeholders
3. **Code Package**: Reusable CLV prediction pipeline
4. **Deployment Guide**: Instructions for production implementation
```

## Bringing It All Together: Professional Project Setup

### Complete Project Initialization

Let's create a professional data science project that demonstrates all the concepts we've covered:

```bash
# Project initialization script
mkdir customer_analytics_platform
cd customer_analytics_platform

# Create professional directory structure
mkdir -p {data/{raw,processed,external},notebooks,src/{preprocessing,modeling,visualization},tests,docs,reports,config}

# Initialize Git repository with proper .gitignore
git init
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.so
.venv/
venv/

# Data files (keep structure, not data)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Jupyter Notebooks
.ipynb_checkpoints/

# Environment variables
.env

# IDE files
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Reports (generated content)
reports/*.html
reports/*.pdf
EOF

# Create environment setup
python3 -m venv .venv
source .venv/bin/activate

# Create comprehensive requirements.txt
cat > requirements.txt << EOF
# Data manipulation and analysis
pandas>=1.5.0
numpy>=1.21.0

# Machine learning
scikit-learn>=1.1.0
scipy>=1.9.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Jupyter environment
jupyter>=1.0.0
ipykernel>=6.0.0

# Development tools
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
EOF

# Install dependencies
pip install -r requirements.txt

# Create project configuration
cat > config/analysis_config.py << EOF
"""
Configuration settings for customer analytics platform.
"""

# Data paths
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed" 
DATA_EXTERNAL_PATH = "data/external"

# Model parameters
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = "viridis"

# Analysis thresholds
OUTLIER_THRESHOLD = 2.5
MISSING_VALUE_THRESHOLD = 0.15
CORRELATION_THRESHOLD = 0.05
EOF

# Initial commit
git add .
git commit -m "Initial project setup

- Create professional directory structure
- Add comprehensive .gitignore for data science project
- Set up Python virtual environment with data science stack
- Create configuration management system
- Initialize Git repository with proper structure"
```

### Assignment Blueprint: Professional Development Portfolio

**Assignment Overview**: Students create a complete professional development environment that demonstrates mastery of all covered concepts and serves as a foundation for future projects.

**Core Deliverables**:

1. **Git Repository Structure** (25%)
   - Professional directory organization
   - Comprehensive .gitignore file
   - Meaningful commit history with feature branches
   - README with clear project description and setup instructions

2. **Python Environment Management** (25%)
   - Virtual environment setup and documentation
   - Comprehensive requirements.txt file
   - Configuration management system
   - Cross-platform compatibility verification

3. **Data Analysis Module** (25%)
   - Reusable statistical analysis functions
   - Robust error handling and input validation
   - Comprehensive docstring documentation
   - Unit tests for all major functions

4. **Collaborative Workflow Demonstration** (25%)
   - Multi-branch development workflow
   - Simulated merge conflict resolution
   - Pull request creation and review
   - Professional commit message standards

**Advanced Challenges** (Bonus Points):
- **Automation**: Create setup scripts for one-command project initialization
- **Testing**: Implement comprehensive test suite with coverage reporting
- **Documentation**: Generate professional documentation using Sphinx
- **CI/CD**: Set up GitHub Actions for automated testing and validation

**Assessment Rubric**:
- **Professional Standards**: Code follows PEP 8, commits are meaningful, documentation is comprehensive
- **Technical Competency**: All tools are used correctly and effectively
- **Collaboration Skills**: Git workflow demonstrates understanding of team development practices
- **Code Quality**: Functions are well-designed, reusable, and properly tested

**Real-World Application**: This assignment creates a professional development environment that students will use throughout the semester and can showcase to potential employers as evidence of professional development practices.

## Looking Forward: Building on Solid Foundations

Today's session established the core data structures and development practices that will support every advanced technique we'll explore this semester. When we dive into pandas DataFrames next week, you'll recognize them as sophisticated dictionaries. When we implement machine learning pipelines, you'll organize them using the modular patterns we practiced today.

The Git workflows we've mastered become essential when collaborating on complex analysis projects, and the documentation practices we've established will make your work reproducible and communicable to stakeholders.

Most importantly, you've started thinking like a professional data scientist—someone who builds systems, not just scripts; who collaborates effectively; and who communicates insights clearly through code, documentation, and analysis.

---

**Next Session Preview**: In L03, we'll explore file operations and Jupyter notebook workflows, building on today's foundations to create complete analysis pipelines that can handle real-world data complexity and scale.