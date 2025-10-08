# Assignment 4: Pandas Data Analysis

**Deliverable:** Completed `assignment.ipynb` with output files in `output/`

## Generate the Dataset (Provided)

Run the data generator notebook to create your dataset:

```bash
jupyter notebook data_generator.ipynb
```

Run all cells to create `data/customer_purchases.csv` (records of customer purchase data).

## Complete the Three Questions

Open `assignment.ipynb` and work through the three questions.

### Question 1: Data Loading & Exploration

**What you'll do:**

- Load `data/customer_purchases.csv` into a pandas DataFrame
- Select only numeric columns using `.select_dtypes()`
- Check for missing values with `.isnull().sum()`
- Generate summary statistics using `.describe()`
- Save summary to `output/exploration_summary.csv`

**Skills:** CSV reading, column selection, data inspection, summary statistics

**Output:** `output/exploration_summary.csv`

### Question 2: Data Cleaning & Transformation

**What you'll do:**

- Fill missing `quantity` values with 1 using `.fillna()`
- Drop rows with missing `shipping_method` using `.dropna()`
- Convert `purchase_date` to datetime with `pd.to_datetime()`
- Convert `quantity` to integer with `.astype()`
- Filter to CA/NY states with quantity >= 2 using boolean indexing
- Save cleaned data to `output/cleaned_data.csv`

**Skills:** Missing value handling, type conversion, boolean filtering

**Output:** `output/cleaned_data.csv`

### Question 3: Analysis & Aggregation

**What you'll do:**

- Create `total_price` column (`quantity` × `price_per_item`)
- Group by `product_category` and sum revenue with `.groupby()`
- Find top 5 products by quantity using `.nlargest()` or `.sort_values()`
- Save analysis to `output/analysis_results.csv`

**Skills:** Calculated columns, groupby aggregation, sorting

**Output:** `output/analysis_results.csv`

**Note:** Each question builds on the previous one. Complete in order!

## Assignment Structure

```
04/assignment/
├── README.md                      # This file - assignment instructions
├── assignment.md                  # Notebook source (for jupytext)
├── assignment.ipynb              # Completed notebook (you work here)
├── data_generator.ipynb          # Run once to create dataset
├── data/
│   └── customer_purchases.csv    # Generated dataset (15,000 records)
├── output/                       # Your saved results (created by your code)
│   ├── exploration_summary.csv   # Q1 output
│   ├── cleaned_data.csv          # Q2 output
│   └── analysis_results.csv      # Q3 output
└── .github/
    └── test/
        ├── test_assignment.py    # Auto-grading tests
        └── requirements.txt      # Test dependencies
```

## Dataset Schema

`data/customer_purchases.csv` contains:

| Column | Type | Description |
|--------|------|-------------|
| `purchase_id` | string | Unique purchase ID (P0001, P0002, ...) |
| `customer_id` | string | Customer ID (C001, C002, ...) |
| `product_category` | string | Electronics, Clothing, Home & Garden, Books, Sports |
| `product_name` | string | Specific product name |
| `quantity` | float | Number of items (has missing values) |
| `price_per_item` | float | Price per item in dollars |
| `purchase_date` | string | Date in YYYY-MM-DD format |
| `customer_state` | string | US state code (CA, NY, TX, FL, WA, IL) |
| `shipping_method` | string | Standard, Express, or Overnight (has missing values) |

**Note:** The dataset has intentional missing values in `quantity` and `shipping_method` columns.

## Need Help?

**See [TIPS.md](TIPS.md) for:**

- Step-by-step walkthrough of each question
- Code examples and common patterns
- Solutions to common error messages
- Debugging strategies
- Final submission checklist
