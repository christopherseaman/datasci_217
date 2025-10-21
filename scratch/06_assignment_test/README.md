# Assignment 6: Data Wrangling - Join, Combine, and Reshape

**Deliverable:** Completed `assignment.ipynb` with output files in `output/`

## Generate the Dataset (Provided)

Run the data generator notebook to create your dataset:

```bash
jupyter notebook data_generator.ipynb
```

Run all cells to create multiple related CSV files in `data/`:
- `data/customers.csv` (customer information)
- `data/orders.csv` (order transactions)
- `data/products.csv` (product catalog)
- `data/monthly_sales_2023.csv` (wide-format sales by month)
- `data/monthly_sales_2024.csv` (wide-format sales by month)

## Complete the Three Questions

Open `assignment.ipynb` and work through the three questions.

### Question 1: Merging DataFrames (40 points)

**What you'll do:**

- Load customer, order, and product datasets
- Perform inner join between orders and customers using `pd.merge()`
- Perform left join to keep all orders (even without customer data)
- Perform outer join between orders and products
- Merge on multiple columns (customer_id + order_date)
- Handle duplicate keys and validate merge results
- Save merged output to `output/q1_merged_data.csv`

**Skills tested:** Database-style joins (inner, left, right, outer), merge validation, handling duplicate keys, multi-column merges

**Output:** `output/q1_merged_data.csv`

### Question 2: Concatenation & Index Management (30 points)

**What you'll do:**

- Load 2023 and 2024 monthly sales datasets
- Concatenate DataFrames vertically using `pd.concat()`
- Use `ignore_index=True` to reset row numbers
- Concatenate horizontally with index alignment
- Use `set_index()` to make customer_id the index
- Use `reset_index()` to convert index back to column
- Handle misaligned indexes during concatenation
- Save concatenated output to `output/q2_concatenated_data.csv`

**Skills tested:** Vertical/horizontal concatenation, index management, handling misaligned data

**Output:** `output/q2_concatenated_data.csv`

### Question 3: Reshaping & Analysis (30 points)

**What you'll do:**

- Transform wide-format sales data to long format using `pd.melt()`
- Transform long-format data back to wide using `pivot()`
- Create pivot table for aggregation with `pivot_table()`
- Combine reshape operations with merge/concat
- Group reshaped data by category and calculate totals
- Save reshaped output to `output/q3_reshaped_data.csv`

**Skills tested:** Wide ↔ long format conversion, pivot tables, combining reshape with merge operations

**Output:** `output/q3_reshaped_data.csv`

**Note:** Each question builds on the previous one. Complete in order!

## Assignment Structure

```
06/assignment/
├── README.md                      # This file - assignment instructions
├── assignment.md                  # Notebook source (for jupytext)
├── assignment.ipynb              # Completed notebook (you work here)
├── data_generator.ipynb          # Run once to create datasets
├── data/                         # Generated datasets
│   ├── customers.csv             # Customer information (500 customers)
│   ├── orders.csv                # Order transactions (2,000 orders)
│   ├── products.csv              # Product catalog (100 products)
│   ├── monthly_sales_2023.csv    # Wide-format sales (12 months)
│   └── monthly_sales_2024.csv    # Wide-format sales (12 months)
├── output/                       # Your saved results (created by your code)
│   ├── q1_merged_data.csv        # Q1 output
│   ├── q2_concatenated_data.csv  # Q2 output
│   └── q3_reshaped_data.csv      # Q3 output
└── .github/
    └── tests/
        ├── test_assignment.py    # Auto-grading tests
        └── requirements.txt      # Test dependencies
```

## Dataset Schemas

### `data/customers.csv`

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Unique customer ID (C0001, C0002, ...) |
| `name` | string | Customer full name |
| `email` | string | Customer email address |
| `city` | string | Customer city |
| `state` | string | Customer state (CA, NY, TX, FL, WA) |
| `join_date` | string | Customer registration date (YYYY-MM-DD) |

### `data/orders.csv`

| Column | Type | Description |
|--------|------|-------------|
| `order_id` | string | Unique order ID (ORD00001, ORD00002, ...) |
| `customer_id` | string | Customer ID (links to customers.csv) |
| `product_id` | string | Product ID (links to products.csv) |
| `quantity` | int | Number of items ordered |
| `order_date` | string | Order date (YYYY-MM-DD) |
| `order_total` | float | Total order amount in dollars |

**Note:** Some orders may have customer_id or product_id values that don't exist in the respective files (to test join types).

### `data/products.csv`

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | string | Unique product ID (P0001, P0002, ...) |
| `product_name` | string | Product name |
| `category` | string | Product category (Electronics, Clothing, Home & Garden, Books, Sports) |
| `price` | float | Product price in dollars |
| `stock` | int | Current inventory level |

### `data/monthly_sales_2023.csv` and `data/monthly_sales_2024.csv`

**Wide format:** One row per product, one column per month

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | string | Unique product ID |
| `product_name` | string | Product name |
| `Jan` | float | January sales |
| `Feb` | float | February sales |
| `Mar` | float | March sales |
| ... | ... | (continues for all 12 months) |
| `Dec` | float | December sales |

**Note:** This wide format needs to be converted to long format for analysis.

## Example Workflows

### Question 1 Workflow
```python
# Load data
customers = pd.read_csv('data/customers.csv')
orders = pd.read_csv('data/orders.csv')
products = pd.read_csv('data/products.csv')

# Inner join - only matching records
inner_merged = pd.merge(orders, customers, on='customer_id', how='inner')

# Left join - all orders, even without customer data
left_merged = pd.merge(orders, customers, on='customer_id', how='left')

# Outer join - everything from both sides
outer_merged = pd.merge(orders, products, on='product_id', how='outer')

# Multi-column merge
multi_merged = pd.merge(df1, df2, on=['customer_id', 'order_date'])
```

### Question 2 Workflow
```python
# Load sales data
sales_2023 = pd.read_csv('data/monthly_sales_2023.csv')
sales_2024 = pd.read_csv('data/monthly_sales_2024.csv')

# Vertical concatenation
combined = pd.concat([sales_2023, sales_2024], ignore_index=True)

# Set index
indexed = combined.set_index('product_id')

# Reset index
reset = indexed.reset_index()

# Horizontal concatenation (careful with alignment!)
horiz = pd.concat([df1, df2], axis=1)
```

### Question 3 Workflow
```python
# Wide to long format
long_data = pd.melt(sales_2023,
                    id_vars=['product_id', 'product_name'],
                    value_vars=['Jan', 'Feb', 'Mar', ...],
                    var_name='month',
                    value_name='sales')

# Long to wide format
wide_data = long_data.pivot(index='product_id',
                             columns='month',
                             values='sales')

# Pivot table with aggregation
summary = pd.pivot_table(long_data,
                         values='sales',
                         index='product_name',
                         columns='month',
                         aggfunc='sum')
```

## Common Pitfalls to Avoid

### Merge Issues
- **Exploding rows:** Many-to-many joins create Cartesian products. Always check row counts!
- **Lost data:** Using `how='inner'` when you meant `how='left'` loses rows
- **Wrong keys:** Merging on the wrong column(s) produces garbage results
- **Duplicate keys:** Check for duplicates in merge keys before merging

### Concatenation Issues
- **Repeated indexes:** Use `ignore_index=True` for vertical concat to avoid duplicate index values
- **Misalignment:** Horizontal concat uses indexes for alignment - if they don't match, you get NaN values
- **Wrong axis:** `axis=0` is vertical (rows), `axis=1` is horizontal (columns)

### Reshape Issues
- **Unique index/column combinations:** `pivot()` fails if index/columns aren't unique. Use `pivot_table()` instead
- **Wrong var_name:** In `melt()`, set `var_name` and `value_name` for clarity
- **Forgetting id_vars:** In `melt()`, specify which columns to keep as identifiers

## Testing Your Work

Before submission, verify:

1. **All output files exist:**
   ```bash
   ls output/q*.csv
   ```

2. **Files are not empty:**
   ```bash
   wc -l output/*.csv
   ```

3. **Data looks reasonable:**
   ```python
   pd.read_csv('output/q1_merged_data.csv').head()
   ```

4. **Row counts make sense:**
   - Q1: Inner join should have fewer rows than outer join
   - Q2: Concatenated data should have more rows than either input
   - Q3: Long format should have more rows than wide format

## Need Help?

**Common error messages and solutions:**

### "ValueError: You are trying to merge on object and int64 columns"
- **Problem:** Merge columns have different types
- **Solution:** Convert to same type before merging: `df['col'] = df['col'].astype(str)`

### "ValueError: Index contains duplicate entries, cannot reshape"
- **Problem:** Trying to pivot with non-unique index/column combinations
- **Solution:** Use `pivot_table()` instead of `pivot()`, or ensure combinations are unique

### "KeyError: 'column_name'"
- **Problem:** Column doesn't exist in DataFrame
- **Solution:** Check spelling and use `df.columns` to see available columns

### Getting NaN values everywhere after merge
- **Problem:** Merge keys don't match (wrong column or different values)
- **Solution:** Check unique values in both merge columns: `df['col'].unique()`

### Concatenation produces way too many columns
- **Problem:** Used `axis=1` (horizontal) when you meant `axis=0` (vertical)
- **Solution:** Use `axis=0` for stacking rows, `axis=1` for adding columns

## Grading Rubric

- **Question 1 (40 points):** All merge operations correct, output file valid
- **Question 2 (30 points):** Concatenation and index operations correct, output file valid
- **Question 3 (30 points):** Reshape operations correct, output file valid

**Auto-grading via pytest:**
- Tests verify output files exist and have correct structure
- Tests check for expected columns and row counts
- Tests validate data types and values

Good luck! Data wrangling is 80% of data science - master these operations and you'll save countless hours!
