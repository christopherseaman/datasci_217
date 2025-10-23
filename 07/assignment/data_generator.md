# Data Generator for Assignment 7

## Overview
This notebook generates sample datasets for the data visualization assignment.

## Setup

```python
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create data directory
os.makedirs('data', exist_ok=True)
```

## Generate Sales Data

```python
# Generate sales transactions
n_transactions = 1000
transaction_ids = [f'T{i:04d}' for i in range(1, n_transactions + 1)]
customer_ids = [f'C{i:04d}' for i in np.random.randint(1, 201, n_transactions)]
product_ids = [f'P{i:03d}' for i in np.random.randint(1, 101, n_transactions)]

# Generate transaction dates (last 6 months)
start_date = datetime.now() - timedelta(days=180)
dates = [start_date + timedelta(days=np.random.randint(0, 180)) for _ in range(n_transactions)]

# Generate quantities and prices
quantities = np.random.randint(1, 11, n_transactions)
unit_prices = np.random.uniform(10, 500, n_transactions)
total_amounts = quantities * unit_prices

# Generate store locations
store_locations = np.random.choice(['North', 'South', 'East', 'West'], n_transactions)

# Create sales DataFrame
sales_data = pd.DataFrame({
    'transaction_id': transaction_ids,
    'customer_id': customer_ids,
    'product_id': product_ids,
    'quantity': quantities,
    'unit_price': unit_prices,
    'total_amount': total_amounts,
    'transaction_date': [d.strftime('%Y-%m-%d') for d in dates],
    'store_location': store_locations
})

# Save to CSV
sales_data.to_csv('data/sales_data.csv', index=False)
print(f"Generated {len(sales_data)} sales transactions")
print(sales_data.head())
```

## Generate Customer Data

```python
# Generate customer information
n_customers = 200
customer_ids = [f'C{i:04d}' for i in range(1, n_customers + 1)]
names = [f'Customer {i}' for i in range(1, n_customers + 1)]

# Generate demographics
ages = np.random.randint(18, 80, n_customers)
genders = np.random.choice(['M', 'F'], n_customers)
cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                          'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], n_customers)
states = np.random.choice(['CA', 'NY', 'TX', 'FL', 'WA', 'IL', 'PA', 'OH', 'GA', 'NC'], n_customers)

# Generate registration dates
reg_dates = [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_customers)]

# Create customer DataFrame
customer_data = pd.DataFrame({
    'customer_id': customer_ids,
    'customer_name': names,
    'age': ages,
    'gender': genders,
    'city': cities,
    'state': states,
    'registration_date': [d.strftime('%Y-%m-%d') for d in reg_dates]
})

# Save to CSV
customer_data.to_csv('data/customer_data.csv', index=False)
print(f"Generated {len(customer_data)} customers")
print(customer_data.head())
```

## Generate Product Data

```python
# Generate product catalog
n_products = 100
product_ids = [f'P{i:03d}' for i in range(1, n_products + 1)]

# Generate product information
categories = np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports'], n_products)
brands = np.random.choice(['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E'], n_products)
product_names = [f'Product {i}' for i in range(1, n_products + 1)]

# Generate prices and stock
unit_prices = np.random.uniform(10, 500, n_products)
stock_quantities = np.random.randint(0, 100, n_products)

# Create product DataFrame
product_data = pd.DataFrame({
    'product_id': product_ids,
    'product_name': product_names,
    'category': categories,
    'brand': brands,
    'unit_price': unit_prices,
    'stock_quantity': stock_quantities
})

# Save to CSV
product_data.to_csv('data/product_data.csv', index=False)
print(f"Generated {len(product_data)} products")
print(product_data.head())
```

## Data Summary

```python
# Display data summary
print("=== DATA SUMMARY ===")
print(f"Sales Data: {len(sales_data)} transactions")
print(f"Customer Data: {len(customer_data)} customers")
print(f"Product Data: {len(product_data)} products")
print("\nSales Data Columns:", sales_data.columns.tolist())
print("Customer Data Columns:", customer_data.columns.tolist())
print("Product Data Columns:", product_data.columns.tolist())

# Display sample data
print("\n=== SAMPLE SALES DATA ===")
print(sales_data.head())
print("\n=== SAMPLE CUSTOMER DATA ===")
print(customer_data.head())
print("\n=== SAMPLE PRODUCT DATA ===")
print(product_data.head())
```

## Validation

```python
# Validate data integrity
print("=== DATA VALIDATION ===")

# Check for missing values
print("Missing values in sales_data:", sales_data.isnull().sum().sum())
print("Missing values in customer_data:", customer_data.isnull().sum().sum())
print("Missing values in product_data:", product_data.isnull().sum().sum())

# Check data types
print("\nSales data types:")
print(sales_data.dtypes)
print("\nCustomer data types:")
print(customer_data.dtypes)
print("\nProduct data types:")
print(product_data.dtypes)

print("\nData generation completed successfully!")
```
