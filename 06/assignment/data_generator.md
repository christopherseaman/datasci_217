---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Assignment 6: Data Generator

This notebook generates the datasets you'll use for Assignment 6 (Data Wrangling).

**Run this notebook ONCE to create the data files**, then work on `assignment.ipynb`.

---

## Generated Files

This notebook creates:
- `data/customers.csv` - Customer information (100 customers)
- `data/purchases.csv` - Purchase transactions (2,000 purchases)
- `data/products.csv` - Product catalog (50 products)

---

## Setup

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
print("✓ Libraries imported")
```

---

## Generate Customer Data

```python
# Customer IDs
customer_ids = [f'C{i:03d}' for i in range(1, 101)]

# Names (realistic distribution)
first_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eric', 'Fiona', 'George', 'Hannah',
               'Ian', 'Julia', 'Kevin', 'Laura', 'Michael', 'Nina', 'Oscar', 'Patricia',
               'Quinn', 'Rachel', 'Steve', 'Teresa']
last_names = ['Chen', 'Martinez', 'Kim', 'Patel', 'Thompson', 'Garcia', 'Lee', 'Wilson',
              'Anderson', 'Jackson', 'Brown', 'Davis', 'Miller', 'Rodriguez', 'Singh']

# Cities weighted by population
cities = np.random.choice(
    ['Seattle', 'Portland', 'San Francisco', 'Los Angeles', 'San Diego', 'Sacramento'],
    size=100,
    p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
)

# Generate customer data
customers = pd.DataFrame({
    'customer_id': customer_ids,
    'name': [f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
             for _ in range(100)],
    'city': cities,
    'signup_date': pd.date_range('2023-01-01', periods=100, freq='3D')
})

print(f"✓ Generated {len(customers)} customers")
customers.head()
```

---

## Generate Product Catalog

```python
# Product categories
categories = {
    'Electronics': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Tablet', 'Smartphone',
                    'Headphones', 'Webcam', 'USB Cable', 'Power Bank'],
    'Clothing': ['T-Shirt', 'Jeans', 'Sweater', 'Jacket', 'Shoes', 'Hat', 'Socks',
                 'Dress', 'Shorts', 'Scarf'],
    'Home & Garden': ['Coffee Maker', 'Blender', 'Vacuum', 'Lamp', 'Plant Pot',
                      'Rug', 'Curtains', 'Pillow', 'Candle', 'Picture Frame'],
    'Books': ['Fiction Novel', 'Cookbook', 'Biography', 'Textbook', 'Magazine',
              'Comic Book', 'Travel Guide', 'Self-Help', 'Poetry', 'Reference Book'],
    'Sports': ['Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Soccer Ball', 'Running Shoes',
               'Water Bottle', 'Resistance Bands', 'Jump Rope', 'Bicycle', 'Skateboard']
}

# Build product catalog
product_list = []
product_id = 1

for category, items in categories.items():
    for item in items:
        # Price varies by category
        base_prices = {
            'Electronics': (50, 1500),
            'Clothing': (20, 150),
            'Home & Garden': (15, 300),
            'Books': (10, 50),
            'Sports': (15, 500)
        }

        min_price, max_price = base_prices[category]
        price = round(np.random.uniform(min_price, max_price), 2)

        product_list.append({
            'product_id': f'P{product_id:03d}',
            'product_name': item,
            'category': category,
            'price': price
        })
        product_id += 1

products = pd.DataFrame(product_list)

print(f"✓ Generated {len(products)} products across {len(categories)} categories")
products.head(10)
```

---

## Generate Purchase Transactions

```python
# Generate 2,000 purchases
num_purchases = 2000

# Weighted customer selection (some customers buy more)
customer_weights = np.exp(np.linspace(0, 2, 100))  # Exponential distribution
customer_weights = customer_weights / customer_weights.sum()

purchases = pd.DataFrame({
    'purchase_id': [f'T{i:04d}' for i in range(1, num_purchases + 1)],
    'customer_id': np.random.choice(customer_ids, size=num_purchases, p=customer_weights),
    'product_id': np.random.choice(products['product_id'], size=num_purchases),
    'quantity': np.random.choice([1, 2, 3, 4, 5], size=num_purchases, p=[0.5, 0.25, 0.15, 0.07, 0.03]),
    'purchase_date': pd.date_range('2023-01-01', periods=num_purchases, freq='4H'),
    'store': np.random.choice(['Store A', 'Store B', 'Store C'], size=num_purchases)
})

# Add total_price by merging with products
purchases = purchases.merge(products[['product_id', 'price']], on='product_id')
purchases['total_price'] = (purchases['quantity'] * purchases['price']).round(2)
purchases = purchases.drop('price', axis=1)  # Remove intermediate price column

print(f"✓ Generated {len(purchases)} purchase transactions")
purchases.head(10)
```

---

## Verify Data Relationships

```python
# Check for many-to-one relationships
print("Data Quality Checks:")
print(f"  Unique customers: {customers['customer_id'].nunique()}")
print(f"  Unique products: {products['product_id'].nunique()}")
print(f"  Total purchases: {len(purchases)}")
print()

# Customer purchase frequency
purchase_counts = purchases['customer_id'].value_counts()
print(f"  Customers with purchases: {len(purchase_counts)}")
print(f"  Max purchases by one customer: {purchase_counts.max()}")
print(f"  Customers with no purchases: {len(customers) - len(purchase_counts)}")
print()

# Product popularity
product_counts = purchases['product_id'].value_counts()
print(f"  Products sold: {len(product_counts)}")
print(f"  Products never sold: {len(products) - len(product_counts)}")
print()

print("✓ Data relationships look good for assignment!")
```

---

## Save to CSV Files

```python
# Create data directory if it doesn't exist
import os
os.makedirs('data', exist_ok=True)

# Save all datasets
customers.to_csv('data/customers.csv', index=False)
print("✓ Saved data/customers.csv")

products.to_csv('data/products.csv', index=False)
print("✓ Saved data/products.csv")

purchases.to_csv('data/purchases.csv', index=False)
print("✓ Saved data/purchases.csv")

print()
print("=" * 50)
print("✓ ALL DATA FILES GENERATED SUCCESSFULLY!")
print("=" * 50)
print()
print("Next step: Open assignment.ipynb and complete the questions.")
```

---

## Preview Generated Data

```python
print("Customers:")
display(customers.head())

print("\nProducts:")
display(products.head())

print("\nPurchases:")
display(purchases.head())
```
