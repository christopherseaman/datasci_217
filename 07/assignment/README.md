# Assignment 7: Data Visualization

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
%pip install matplotlib

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
- `data/sales_data.csv` (sales transactions)
- `data/customer_data.csv` (customer information)
- `data/product_data.csv` (product catalog)

## Complete the Four Questions

Open `assignment.ipynb` and work through the four questions. The notebook provides:

- **Step-by-step instructions** with clear TODO items
- **Helpful hints** for each operation
- **Sample data** and examples to guide your work
- **Validation checks** to ensure your outputs are correct

**Prerequisites:** This assignment uses matplotlib, seaborn, and pandas plotting from Lecture 07.

**How to use the scaffold notebook:**
1. Read each cell carefully - they contain detailed instructions
2. Complete the TODO items by replacing `None` with your code
3. Run each cell to see your progress
4. Use the hints provided in comments
5. Check the submission checklist at the end

### Question 1: matplotlib Fundamentals

**What you'll do:**
- Create figures and subplots with matplotlib
- Customize plot appearance (colors, markers, styles)
- Generate different plot types (line, bar, scatter, histogram)
- Save plots in multiple formats
- Create a multi-panel visualization

**Skills:** matplotlib figures, subplots, customization, plot types, file export

**Output:** `output/q1_matplotlib_plots.png`, `output/q1_multi_panel.png`

### Question 2: seaborn Statistical Visualization

**What you'll do:**
- Create statistical plots with seaborn
- Visualize relationships between variables
- Analyze distributions and patterns
- Apply seaborn styling and themes
- Create correlation analysis

**Skills:** seaborn statistical plots, relationship visualization, distribution analysis, styling

**Output:** `output/q2_seaborn_plots.png`, `output/q2_correlation_heatmap.png`

### Question 3: pandas Plotting and Data Exploration

**What you'll do:**
- Use pandas plotting for quick data exploration
- Create time series visualizations
- Generate multiple plot types with pandas
- Apply visualization best practices
- Create a comprehensive data overview

**Skills:** pandas plotting, time series visualization, data exploration, best practices

**Output:** `output/q3_pandas_plots.png`, `output/q3_data_overview.png`

## Assignment Structure

```
07/assignment/
├── README.md                      # This file - assignment instructions
├── assignment.md                  # Notebook source (for jupytext)
├── assignment.ipynb              # Completed notebook (you work here)
├── data_generator.ipynb          # Run once to create datasets
├── data/                         # Generated datasets
│   ├── sales_data.csv            # Sales transactions (1,000 records)
│   ├── customer_data.csv         # Customer information (200 customers)
│   └── product_data.csv          # Product catalog (100 products)
├── output/                       # Your saved results (created by your code)
│   ├── q1_matplotlib_plots.png   # Q1 matplotlib output
│   ├── q1_multi_panel.png        # Q1 multi-panel plot
│   ├── q2_seaborn_plots.png      # Q2 seaborn output
│   ├── q2_correlation_heatmap.png # Q2 correlation analysis
│   ├── q3_pandas_plots.png       # Q3 pandas output
│   └── q3_data_overview.png      # Q3 data overview
└── .github/
    └── test/
        ├── test_assignment.py    # Auto-grading tests
        └── requirements.txt      # Test dependencies
```

## Dataset Schemas

### `data/sales_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Unique transaction ID (T0001, T0002, ...) |
| `customer_id` | string | Customer ID (links to customer_data.csv) |
| `product_id` | string | Product ID (links to product_data.csv) |
| `quantity` | int | Number of items purchased |
| `unit_price` | float | Price per unit |
| `total_amount` | float | Total transaction amount |
| `transaction_date` | string | Transaction date (YYYY-MM-DD) |
| `store_location` | string | Store location (North, South, East, West) |

### `data/customer_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Unique customer ID (C0001, C0002, ...) |
| `customer_name` | string | Customer full name |
| `age` | int | Customer age |
| `gender` | string | Customer gender (M, F) |
| `city` | string | Customer city |
| `state` | string | Customer state |
| `registration_date` | string | Customer registration date (YYYY-MM-DD) |

### `data/product_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | string | Unique product ID (P001, P002, ...) |
| `product_name` | string | Product name |
| `category` | string | Product category (Electronics, Clothing, Home & Garden, Books, Sports) |
| `brand` | string | Product brand |
| `unit_price` | float | Product price |
| `stock_quantity` | int | Current inventory level |

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_matplotlib_plots.png` - matplotlib fundamentals
- [ ] `output/q1_multi_panel.png` - multi-panel visualization
- [ ] `output/q2_seaborn_plots.png` - seaborn statistical plots
- [ ] `output/q2_correlation_heatmap.png` - correlation analysis
- [ ] `output/q3_pandas_plots.png` - pandas plotting
- [ ] `output/q3_data_overview.png` - data exploration