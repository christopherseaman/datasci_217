#!/usr/bin/env python3
"""
Lecture 06 Live Demo: Pandas Data Manipulation and Jupyter Workflows
DataSci 217 - Hands-on demonstration of pandas for data analysis

This demo shows practical pandas operations and Jupyter notebook best practices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

def create_sample_datasets():
    """
    Create realistic sample datasets for pandas demonstration
    """
    print("Creating sample datasets for pandas demo...")

    # Dataset 1: Customer data
    np.random.seed(42)
    n_customers = 100

    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'name': [f"Customer_{i}" for i in range(1, n_customers + 1)],
        'age': np.random.randint(18, 80, n_customers),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
        'signup_date': pd.date_range('2020-01-01', periods=n_customers, freq='3D'),
        'premium_member': np.random.choice([True, False], n_customers, p=[0.3, 0.7])
    })

    # Dataset 2: Sales transactions
    n_transactions = 500
    transactions = pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'customer_id': np.random.randint(1, n_customers + 1, n_transactions),
        'product': np.random.choice(['Widget A', 'Widget B', 'Widget C', 'Gadget X', 'Gadget Y'], n_transactions),
        'amount': np.random.uniform(10, 500, n_transactions),
        'transaction_date': pd.date_range('2020-01-01', periods=n_transactions, freq='12H'),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer'], n_transactions)
    })

    # Dataset 3: Product inventory
    products = pd.DataFrame({
        'product': ['Widget A', 'Widget B', 'Widget C', 'Gadget X', 'Gadget Y'],
        'category': ['Widgets', 'Widgets', 'Widgets', 'Gadgets', 'Gadgets'],
        'price': [29.99, 39.99, 49.99, 79.99, 99.99],
        'cost': [15.00, 20.00, 25.00, 40.00, 50.00],
        'stock_quantity': [150, 200, 100, 75, 50]
    })

    # Save datasets
    customers.to_csv('customers.csv', index=False)
    transactions.to_csv('transactions.csv', index=False)
    products.to_csv('products.csv', index=False)

    print("✓ Sample datasets created:")
    print(f"  - customers.csv: {len(customers)} records")
    print(f"  - transactions.csv: {len(transactions)} records")
    print(f"  - products.csv: {len(products)} records")
    print()

    return customers, transactions, products

def demo_pandas_basics():
    """
    Demonstrate fundamental pandas operations
    """
    print("=== PANDAS FUNDAMENTALS DEMO ===")
    print()

    # Load data
    customers = pd.read_csv('customers.csv')
    print("1. Loading and Exploring Data")
    print("=" * 30)

    print("First look at customer data:")
    print(customers.head())
    print()

    print("Data info:")
    print(customers.info())
    print()

    print("Summary statistics:")
    print(customers.describe())
    print()

    print("2. Data Selection and Filtering")
    print("=" * 30)

    # Column selection
    print("Select specific columns:")
    customer_names = customers[['name', 'age', 'city']]
    print(customer_names.head())
    print()

    # Row filtering
    print("Filter customers over 50:")
    older_customers = customers[customers['age'] > 50]
    print(f"Found {len(older_customers)} customers over 50")
    print(older_customers[['name', 'age', 'city']].head())
    print()

    # Multiple conditions
    print("Premium members in New York:")
    ny_premium = customers[(customers['city'] == 'New York') & (customers['premium_member'] == True)]
    print(f"Found {len(ny_premium)} premium members in New York")
    print()

    print("3. Data Transformation")
    print("=" * 30)

    # Add new columns
    customers_copy = customers.copy()
    customers_copy['age_group'] = pd.cut(customers_copy['age'],
                                       bins=[0, 30, 50, 100],
                                       labels=['Young', 'Middle', 'Senior'])

    print("Age group distribution:")
    print(customers_copy['age_group'].value_counts())
    print()

    # Date operations
    customers_copy['signup_date'] = pd.to_datetime(customers_copy['signup_date'])
    customers_copy['days_since_signup'] = (datetime.now() - customers_copy['signup_date']).dt.days

    print("Average days since signup:")
    print(f"{customers_copy['days_since_signup'].mean():.0f} days")
    print()

def demo_pandas_grouping():
    """
    Demonstrate pandas grouping and aggregation operations
    """
    print("=== PANDAS GROUPING AND AGGREGATION ===")
    print()

    # Load transaction data
    transactions = pd.read_csv('transactions.csv')
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

    print("1. Basic Grouping Operations")
    print("=" * 30)

    # Group by single column
    print("Sales by product:")
    product_sales = transactions.groupby('product')['amount'].agg(['sum', 'mean', 'count'])
    print(product_sales.round(2))
    print()

    # Group by multiple columns
    print("Sales by product and payment method:")
    product_payment_sales = transactions.groupby(['product', 'payment_method'])['amount'].sum()
    print(product_payment_sales.head(10))
    print()

    print("2. Time-based Analysis")
    print("=" * 30)

    # Extract date components
    transactions['year'] = transactions['transaction_date'].dt.year
    transactions['month'] = transactions['transaction_date'].dt.month
    transactions['day_of_week'] = transactions['transaction_date'].dt.day_name()

    # Monthly sales trends
    print("Monthly sales totals:")
    monthly_sales = transactions.groupby(['year', 'month'])['amount'].sum()
    print(monthly_sales)
    print()

    # Day of week analysis
    print("Sales by day of week:")
    dow_sales = transactions.groupby('day_of_week')['amount'].agg(['sum', 'mean'])
    # Reorder by actual day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_sales = dow_sales.reindex(day_order)
    print(dow_sales.round(2))
    print()

def demo_pandas_merging():
    """
    Demonstrate pandas data merging and joining operations
    """
    print("=== PANDAS MERGING AND JOINING ===")
    print()

    # Load all datasets
    customers = pd.read_csv('customers.csv')
    transactions = pd.read_csv('transactions.csv')
    products = pd.read_csv('products.csv')

    print("1. Basic Merging")
    print("=" * 30)

    # Merge transactions with customer data
    customer_transactions = transactions.merge(customers, on='customer_id', how='left')
    print("Merged transaction and customer data:")
    print(customer_transactions[['transaction_id', 'name', 'product', 'amount', 'city']].head())
    print()

    # Merge with product data
    full_data = customer_transactions.merge(products, on='product', how='left')
    full_data['profit'] = full_data['amount'] - full_data['cost']

    print("2. Customer Analysis with Merged Data")
    print("=" * 30)

    # Customer profitability analysis
    customer_analysis = full_data.groupby(['customer_id', 'name']).agg({
        'amount': ['sum', 'count'],
        'profit': 'sum'
    }).round(2)

    # Flatten column names
    customer_analysis.columns = ['total_revenue', 'transaction_count', 'total_profit']
    customer_analysis = customer_analysis.reset_index()

    # Add profitability metrics
    customer_analysis['avg_transaction'] = (customer_analysis['total_revenue'] /
                                          customer_analysis['transaction_count']).round(2)

    print("Top 10 customers by total revenue:")
    top_customers = customer_analysis.nlargest(10, 'total_revenue')
    print(top_customers[['name', 'total_revenue', 'transaction_count', 'avg_transaction']])
    print()

    print("3. Product Performance Analysis")
    print("=" * 30)

    # Product analysis with categories
    product_performance = full_data.groupby(['category', 'product']).agg({
        'amount': ['sum', 'count'],
        'profit': 'sum'
    }).round(2)

    product_performance.columns = ['total_sales', 'units_sold', 'total_profit']
    product_performance = product_performance.reset_index()
    product_performance['profit_margin'] = (product_performance['total_profit'] /
                                           product_performance['total_sales'] * 100).round(1)

    print("Product performance by category:")
    print(product_performance.sort_values('total_sales', ascending=False))
    print()

def demo_pandas_data_cleaning():
    """
    Demonstrate pandas data cleaning techniques
    """
    print("=== PANDAS DATA CLEANING ===")
    print()

    print("1. Creating Messy Data")
    print("=" * 30)

    # Create dataset with common data quality issues
    np.random.seed(42)
    dirty_data = pd.DataFrame({
        'name': ['Alice Johnson', 'bob smith', 'CAROL DAVIS', '  David Wilson  ', None],
        'email': ['alice@email.com', 'BOB@EMAIL.COM', 'carol@email', 'david@email.com', 'invalid-email'],
        'age': [25, -5, 150, 35, None],
        'salary': ['50000', '60,000', '$75000', '45000.0', 'unknown'],
        'phone': ['123-456-7890', '(987) 654-3210', '555.123.4567', '1234567890', '123']
    })

    print("Original messy data:")
    print(dirty_data)
    print()

    print("2. Data Cleaning Operations")
    print("=" * 30)

    # Clean names
    dirty_data['name_clean'] = (dirty_data['name']
                               .str.strip()  # Remove whitespace
                               .str.title()  # Proper case
                               .fillna('Unknown'))

    print("Cleaned names:")
    print(dirty_data[['name', 'name_clean']])
    print()

    # Clean emails
    dirty_data['email_valid'] = dirty_data['email'].str.contains('@.*\.', na=False)
    print("Email validation:")
    print(dirty_data[['email', 'email_valid']])
    print()

    # Clean ages (handle outliers and missing values)
    dirty_data['age_clean'] = dirty_data['age']
    # Replace invalid ages with NaN
    dirty_data.loc[(dirty_data['age_clean'] < 0) | (dirty_data['age_clean'] > 120), 'age_clean'] = np.nan
    # Fill missing ages with median
    median_age = dirty_data['age_clean'].median()
    dirty_data['age_clean'] = dirty_data['age_clean'].fillna(median_age)

    print("Age cleaning:")
    print(dirty_data[['age', 'age_clean']])
    print()

    # Clean salary data
    def clean_salary(salary_str):
        if pd.isna(salary_str) or str(salary_str).lower() == 'unknown':
            return np.nan
        # Remove currency symbols, commas, and convert to float
        cleaned = str(salary_str).replace('$', '').replace(',', '')
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    dirty_data['salary_clean'] = dirty_data['salary'].apply(clean_salary)
    print("Salary cleaning:")
    print(dirty_data[['salary', 'salary_clean']])
    print()

def demo_jupyter_best_practices():
    """
    Demonstrate Jupyter notebook best practices
    """
    print("=== JUPYTER NOTEBOOK BEST PRACTICES ===")
    print()

    print("1. Notebook Organization")
    print("=" * 30)

    print("Best practices for Jupyter notebooks:")
    print("✓ Use clear, descriptive cell headers")
    print("✓ Add markdown cells to explain your analysis")
    print("✓ Keep cells focused and not too long")
    print("✓ Run cells in order from top to bottom")
    print("✓ Clear outputs before saving to version control")
    print()

    print("2. Code Structure Example")
    print("=" * 30)

    print("Good notebook structure:")
    print("""
# 1. Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2. Load Data
data = pd.read_csv('data.csv')

# 3. Data Exploration
print(data.head())
print(data.info())

# 4. Data Cleaning
# Document each cleaning step clearly

# 5. Analysis
# Break complex analysis into logical steps

# 6. Visualization
# Create clear, labeled plots

# 7. Conclusions
# Summarize findings in markdown
""")
    print()

    print("3. Documentation Tips")
    print("=" * 30)

    print("Use markdown cells for:")
    print("• Project overview and objectives")
    print("• Data source descriptions")
    print("• Analysis methodology explanations")
    print("• Interpretation of results")
    print("• Next steps and recommendations")
    print()

    print("Use code comments for:")
    print("• Complex data transformations")
    print("• Parameter explanations")
    print("• Algorithm choices")
    print("• Assumption documentation")
    print()

def create_analysis_report():
    """
    Create a comprehensive analysis report
    """
    print("=== COMPREHENSIVE ANALYSIS REPORT ===")
    print()

    # Load and merge all data
    customers = pd.read_csv('customers.csv')
    transactions = pd.read_csv('transactions.csv')
    products = pd.read_csv('products.csv')

    # Convert dates
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

    # Merge datasets
    full_data = (transactions
                .merge(customers, on='customer_id', how='left')
                .merge(products, on='product', how='left'))

    # Calculate key metrics
    total_revenue = full_data['amount'].sum()
    total_customers = customers['customer_id'].nunique()
    avg_transaction = full_data['amount'].mean()
    premium_customers = customers['premium_member'].sum()

    print("BUSINESS INTELLIGENCE DASHBOARD")
    print("=" * 40)
    print(f"Analysis Period: {transactions['transaction_date'].min().date()} to {transactions['transaction_date'].max().date()}")
    print()

    print("KEY METRICS:")
    print(f"• Total Revenue: ${total_revenue:,.2f}")
    print(f"• Total Customers: {total_customers:,}")
    print(f"• Average Transaction: ${avg_transaction:.2f}")
    print(f"• Premium Customers: {premium_customers} ({premium_customers/total_customers*100:.1f}%)")
    print()

    # Top performing segments
    print("TOP PERFORMING SEGMENTS:")
    city_performance = (full_data.groupby('city')['amount']
                       .agg(['sum', 'count', 'mean'])
                       .round(2)
                       .sort_values('sum', ascending=False))

    print("By City:")
    for city, row in city_performance.head(3).iterrows():
        print(f"• {city}: ${row['sum']:,.2f} revenue, {row['count']} transactions")
    print()

    # Product analysis
    product_analysis = (full_data.groupby('product')['amount']
                       .agg(['sum', 'count'])
                       .sort_values('sum', ascending=False))

    print("By Product:")
    for product, row in product_analysis.head(3).iterrows():
        print(f"• {product}: ${row['sum']:,.2f} revenue, {row['count']} sales")
    print()

    # Customer segmentation
    customer_segments = (full_data.groupby('customer_id')['amount']
                        .sum()
                        .describe()
                        .round(2))

    print("CUSTOMER ANALYSIS:")
    print(f"• Average customer value: ${customer_segments['mean']:,.2f}")
    print(f"• Top 25% customers spend: ${customer_segments['75%']:,.2f}+")
    print(f"• High-value customers (>$1000): {(full_data.groupby('customer_id')['amount'].sum() > 1000).sum()}")
    print()

    print("RECOMMENDATIONS:")
    print("• Focus marketing efforts on top-performing cities")
    print("• Develop retention programs for high-value customers")
    print("• Analyze seasonal trends for inventory planning")
    print("• Consider premium membership benefits to increase conversion")

def main():
    """
    Main demo execution function
    """
    print("Welcome to DataSci 217 - Lecture 06 Live Demo!")
    print("Pandas Data Manipulation and Jupyter Workflows")
    print("=" * 60)
    print()

    # Create sample data
    create_sample_datasets()

    # Run demos
    demo_pandas_basics()
    print("\n" + "="*60 + "\n")

    demo_pandas_grouping()
    print("\n" + "="*60 + "\n")

    demo_pandas_merging()
    print("\n" + "="*60 + "\n")

    demo_pandas_data_cleaning()
    print("\n" + "="*60 + "\n")

    demo_jupyter_best_practices()
    print("\n" + "="*60 + "\n")

    create_analysis_report()
    print("\n" + "="*60 + "\n")

    print("Demo complete!")
    print("\nKey takeaways:")
    print("1. Pandas DataFrames are perfect for structured data analysis")
    print("2. Always explore your data before analysis (head, info, describe)")
    print("3. Group operations enable powerful aggregation and analysis")
    print("4. Merging datasets reveals deeper insights")
    print("5. Data cleaning is essential - expect and plan for messy data")
    print("6. Jupyter notebooks excel at exploratory data analysis")
    print("7. Document your analysis process clearly")

if __name__ == "__main__":
    main()