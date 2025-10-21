#!/usr/bin/env python3
"""
Data Generator for Assignment 6
Generates all required CSV files in data/ directory
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
print("Created data/ directory")

# Generate 500 customers
print("\nGenerating customers dataset...")
n_customers = 500

first_names = ['John', 'Jane', 'Michael', 'Emily', 'David', 'Sarah', 'Robert', 'Lisa', 'James', 'Mary',
               'William', 'Patricia', 'Richard', 'Jennifer', 'Charles', 'Linda', 'Joseph', 'Elizabeth',
               'Thomas', 'Barbara', 'Christopher', 'Susan', 'Daniel', 'Jessica', 'Matthew', 'Karen']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez',
              'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor',
              'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez']

cities = ['San Francisco', 'Los Angeles', 'San Diego', 'New York', 'Brooklyn', 'Austin', 'Houston',
          'Miami', 'Orlando', 'Seattle', 'Portland']
states = ['CA', 'NY', 'TX', 'FL', 'WA']

customers = pd.DataFrame({
    'customer_id': [f'C{str(i).zfill(4)}' for i in range(1, n_customers + 1)],
    'name': [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_customers)],
    'email': [f"customer{i}@email.com" for i in range(1, n_customers + 1)],
    'city': np.random.choice(cities, n_customers),
    'state': np.random.choice(states, n_customers),
    'join_date': [(datetime(2022, 1, 1) + timedelta(days=int(x))).strftime('%Y-%m-%d')
                  for x in np.random.randint(0, 730, n_customers)]
})

customers.to_csv('data/customers.csv', index=False)
print(f"Created data/customers.csv with {len(customers)} customers")

# Generate 100 products
print("\nGenerating products dataset...")
n_products = 100

categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports']
product_names = {
    'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch', 'Camera', 'Speaker'],
    'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Shoes', 'Hat', 'Sweater'],
    'Home & Garden': ['Lamp', 'Chair', 'Table', 'Plant', 'Rug', 'Pillow', 'Vase'],
    'Books': ['Novel', 'Cookbook', 'Biography', 'Textbook', 'Comic', 'Magazine', 'Guide'],
    'Sports': ['Basketball', 'Tennis Racket', 'Yoga Mat', 'Dumbbells', 'Bike', 'Helmet', 'Sneakers']
}

products_list = []
product_id = 1
for _ in range(n_products):
    category = np.random.choice(categories)
    name = np.random.choice(product_names[category])
    products_list.append({
        'product_id': f'P{str(product_id).zfill(4)}',
        'product_name': f"{name} {np.random.choice(['Pro', 'Plus', 'Deluxe', 'Standard', 'Basic'])}",
        'category': category,
        'price': round(np.random.uniform(10, 500), 2),
        'stock': np.random.randint(0, 200)
    })
    product_id += 1

products = pd.DataFrame(products_list)
products.to_csv('data/products.csv', index=False)
print(f"Created data/products.csv with {len(products)} products")

# Generate 2000 orders
print("\nGenerating orders dataset...")
n_orders = 2000

# Most orders have valid customer_id and product_id, but some don't (for testing joins)
valid_customer_ids = customers['customer_id'].tolist()
valid_product_ids = products['product_id'].tolist()

# Add some invalid IDs for testing
invalid_customer_ids = [f'C{str(i).zfill(4)}' for i in range(n_customers + 1, n_customers + 50)]
invalid_product_ids = [f'P{str(i).zfill(4)}' for i in range(n_products + 1, n_products + 50)]

orders_list = []
for i in range(1, n_orders + 1):
    # 95% valid customer_ids, 5% invalid
    if np.random.random() < 0.95:
        customer_id = np.random.choice(valid_customer_ids)
    else:
        customer_id = np.random.choice(invalid_customer_ids)

    # 95% valid product_ids, 5% invalid
    if np.random.random() < 0.95:
        product_id = np.random.choice(valid_product_ids)
    else:
        product_id = np.random.choice(invalid_product_ids)

    quantity = np.random.randint(1, 10)
    price = products[products['product_id'] == product_id]['price'].values[0] if product_id in valid_product_ids else np.random.uniform(10, 500)

    orders_list.append({
        'order_id': f'ORD{str(i).zfill(5)}',
        'customer_id': customer_id,
        'product_id': product_id,
        'quantity': quantity,
        'order_date': (datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 730))).strftime('%Y-%m-%d'),
        'order_total': round(price * quantity, 2)
    })

orders = pd.DataFrame(orders_list)
orders.to_csv('data/orders.csv', index=False)
print(f"Created data/orders.csv with {len(orders)} orders")

# Generate monthly sales for 2023
print("\nGenerating monthly sales 2023...")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Select a subset of products for monthly sales
sales_products = products.sample(50, random_state=42)

sales_2023 = sales_products[['product_id', 'product_name']].copy()
for month in months:
    sales_2023[month] = np.random.uniform(1000, 10000, len(sales_products)).round(2)

sales_2023.to_csv('data/monthly_sales_2023.csv', index=False)
print(f"Created data/monthly_sales_2023.csv with {len(sales_2023)} products")

# Generate monthly sales for 2024
print("\nGenerating monthly sales 2024...")
sales_2024 = sales_products[['product_id', 'product_name']].copy()
for month in months:
    # 2024 sales are slightly higher (growth)
    sales_2024[month] = np.random.uniform(1200, 11000, len(sales_products)).round(2)

sales_2024.to_csv('data/monthly_sales_2024.csv', index=False)
print(f"Created data/monthly_sales_2024.csv with {len(sales_2024)} products")

# Verify all files were created
print("\n" + "="*60)
print("Verifying generated files:")
print("="*60)

required_files = [
    'data/customers.csv',
    'data/orders.csv',
    'data/products.csv',
    'data/monthly_sales_2023.csv',
    'data/monthly_sales_2024.csv'
]

all_good = True
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✓ {file} ({size:,} bytes)")
    else:
        print(f"✗ {file} MISSING")
        all_good = False

if all_good:
    print("\n✓ Data generation complete! You can now work on assignment.ipynb")
else:
    print("\n✗ Some files are missing. Please check for errors.")
