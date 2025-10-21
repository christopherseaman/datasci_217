# Assignment 6: Data Wrangling - Join, Combine, and Reshape

**Deliverable:** Completed `assignment.ipynb` with output files in `output/`

## Environment Setup

### Create Virtual Environment

Create a virtual environment for this assignment:

```bash
# Create venv
python3 -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Install Requirements

You have two options to install the required packages:

**Option 1: Using pip in terminal**
```bash
pip install -r requirements.txt
```

**Option 2: Using %pip magic in Jupyter**

You can install packages directly from a Jupyter notebook cell using the `%pip` magic command:

```python
# Install single package
%pip install pandas

# Install from requirements.txt
%pip install -r requirements.txt
```

**Important:** Make sure your Jupyter notebook is using the same virtual environment as your kernel. Select the `.venv` kernel in Jupyter's kernel menu.

## Generate the Dataset (Provided)

Run the data generator notebook to create your dataset:

```bash
jupyter notebook data_generator.ipynb
```

Run all cells to create the CSV files in `data/`:
- `data/customers.csv` (customer information)
- `data/products.csv` (product catalog)
- `data/purchases.csv` (purchase transactions)

## Complete the Three Questions

Open `assignment.ipynb` and work through the three questions.

### Question 1: Merging DataFrames

**What you'll do:**
- Load customer, product, and purchase datasets
- Perform inner join between purchases and customers
- Perform left join to keep all purchases
- Perform outer join between purchases and products
- Merge on multiple columns (composite keys)
- Handle duplicate keys and validate merge results
- Save merged output to `output/q1_merged_data.csv`

**Skills:** Database-style joins (inner, left, right, outer), merge validation, handling duplicate keys, multi-column merges

**Output:** `output/q1_merged_data.csv`, `output/q1_validation.txt`

### Question 2: Concatenation & Index Management

**What you'll do:**
- Split purchases into quarterly datasets
- Concatenate DataFrames vertically using `pd.concat()`
- Use `ignore_index=True` to reset row numbers
- Use `keys` parameter for source tracking
- Concatenate horizontally with index alignment
- Handle misaligned indexes during concatenation
- Save concatenated output to `output/q2_combined_data.csv`

**Skills:** Vertical/horizontal concatenation, index management, handling misaligned data

**Output:** `output/q2_combined_data.csv`

### Question 3: Reshaping & Analysis

**What you'll do:**
- Merge purchases with products to get categories
- Create pivot table for sales by category and month
- Save pivoted data (wide format)
- Transform wide format back to long using `pd.melt()`
- Calculate summary statistics by category
- Save analysis report

**Skills:** Pivot tables, wide ↔ long format conversion, aggregation

**Output:** `output/q3_category_sales_wide.csv`, `output/q3_analysis_report.txt`

## Assignment Structure

```
06/assignment/
├── README.md                      # This file - assignment instructions
├── assignment.md                  # Notebook source (for jupytext)
├── assignment.ipynb              # Completed notebook (you work here)
├── data_generator.ipynb          # Run once to create datasets
├── data/                         # Generated datasets
│   ├── customers.csv             # Customer information (100 customers)
│   ├── products.csv              # Product catalog (50 products)
│   └── purchases.csv             # Purchase transactions (2,000 purchases)
├── output/                       # Your saved results (created by your code)
│   ├── q1_merged_data.csv        # Q1 output
│   ├── q1_validation.txt         # Q1 validation report
│   ├── q2_combined_data.csv      # Q2 output
│   ├── q3_category_sales_wide.csv  # Q3 output
│   └── q3_analysis_report.txt    # Q3 analysis report
└── .github/
    └── test/
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

### `data/products.csv`

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | string | Unique product ID (P001, P002, ...) |
| `product_name` | string | Product name |
| `category` | string | Product category (Electronics, Clothing, Home & Garden, Books, Sports) |
| `price` | float | Product price in dollars |
| `stock` | int | Current inventory level |

### `data/purchases.csv`

| Column | Type | Description |
|--------|------|-------------|
| `purchase_id` | string | Unique purchase ID (T0001, T0002, ...) |
| `customer_id` | string | Customer ID (links to customers.csv) |
| `product_id` | string | Product ID (links to products.csv) |
| `quantity` | int | Number of items purchased |
| `purchase_date` | string | Purchase date (YYYY-MM-DD) |
| `store` | string | Store location (Store A, B, or C) |

**Note:** You'll calculate `total_price` in Question 1 by merging with products and multiplying `quantity * price`.

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_merged_data.csv` - Merged customer/product/purchase data
- [ ] `output/q1_validation.txt` - Merge validation report
- [ ] `output/q2_combined_data.csv` - Concatenated data with metrics
- [ ] `output/q3_category_sales_wide.csv` - Pivoted category sales
- [ ] `output/q3_analysis_report.txt` - Sales analysis report
