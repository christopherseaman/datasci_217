# Assignment 8: Data Aggregation and Group Operations

**Deliverable:**

- Pass all auto-grading tests by generating the required output files from `assignment.ipynb`/`assignment.md`

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
- `data/employee_data.csv` (employee information)
- `data/department_data.csv` (department information)
- `data/sales_data.csv` (sales transactions)

## Complete the Four Questions

Open `assignment.ipynb` and work through the four questions. The notebook provides:

- **Step-by-step instructions** with clear TODO items
- **Helpful hints** for each operation
- **Sample data** and examples to guide your work
- **Validation checks** to ensure your outputs are correct

**Prerequisites:** This assignment uses groupby operations, pivot tables, and aggregation functions from Lecture 08.

**How to use the scaffold notebook:**
1. Read each cell carefully - they contain detailed instructions
2. Complete the TODO items by replacing `None` with your code
3. Run each cell to see your progress
4. Use the hints provided in comments
5. Check the submission checklist at the end

### Auto-Grading (Required)

Run all required cells in `assignment.ipynb` so that the following files are created in `output/`:

- `q1_groupby_analysis.csv`, `q1_aggregation_report.txt`
- `q2_hierarchical_analysis.csv`, `q2_performance_report.txt`
- `q3_pivot_analysis.csv`, `q3_crosstab_analysis.csv`, `q3_pivot_visualization.png`

Run tests locally:

```bash
pytest -q 08/assignment/.github/test/test_assignment.py
```

GitHub Classroom will run the same tests on push.

### Question 1: Basic GroupBy Operations

**What you'll do:**
- Load and merge employee, department, and sales data
- Perform basic groupby operations with aggregation functions
- Use transform operations to add group statistics
- Apply filter operations to remove groups
- Create custom aggregation functions

**Skills:** groupby operations, aggregation functions, transform, filter, apply

**Output:** `output/q1_groupby_analysis.csv`, `output/q1_aggregation_report.txt`

### Question 2: Advanced GroupBy Operations

**What you'll do:**
- Perform hierarchical grouping with multiple columns
- Use apply operations with custom functions
- Handle MultiIndex structures
- Create group-level statistics and rankings
- Analyze performance differences between methods

**Skills:** hierarchical grouping, MultiIndex, custom functions, performance analysis

**Output:** `output/q2_hierarchical_analysis.csv`, `output/q2_performance_report.txt`

### Question 3: Pivot Tables and Cross-Tabulations

**What you'll do:**
- Create pivot tables for multi-dimensional analysis
- Use cross-tabulations for frequency analysis
- Apply advanced pivot operations with totals
- Handle missing values and custom aggregations
- Create visualizations from pivot tables

**Skills:** pivot tables, cross-tabulations, multi-dimensional analysis, visualization

**Output:** `output/q3_pivot_analysis.csv`, `output/q3_crosstab_analysis.csv`, `output/q3_pivot_visualization.png`

 

## Assignment Structure

```
08/assignment/
├── README.md                      # This file - assignment instructions
├── assignment.md                  # Notebook source (for jupytext)
├── assignment.ipynb              # Completed notebook (you work here)
├── data_generator.ipynb          # Run once to create datasets
├── data/                         # Generated datasets
│   ├── employee_data.csv         # Employee information (500 employees)
│   ├── department_data.csv       # Department information (20 departments)
│   └── sales_data.csv            # Sales transactions (5,000 transactions)
├── output/                       # Your saved results (created by your code)
│   ├── q1_groupby_analysis.csv   # Q1 groupby analysis
│   ├── q1_aggregation_report.txt # Q1 aggregation report
│   ├── q2_hierarchical_analysis.csv # Q2 hierarchical analysis
│   ├── q2_performance_report.txt # Q2 performance report
│   ├── q3_pivot_analysis.csv     # Q3 pivot table analysis
│   ├── q3_crosstab_analysis.csv  # Q3 cross-tabulation analysis
│   ├── q3_pivot_visualization.png # Q3 pivot visualization
 
└── .github/
    └── test/
        └── test_assignment.py    # Auto-grading tests
```

## Dataset Schemas

### `data/employee_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `employee_id` | string | Unique employee ID (E0001, E0002, ...) |
| `name` | string | Employee full name |
| `department_id` | string | Department ID (links to department_data.csv) |
| `position` | string | Employee position |
| `salary` | float | Employee salary |
| `hire_date` | string | Employee hire date (YYYY-MM-DD) |
| `performance_score` | float | Employee performance score (1-10) |

### `data/department_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `department_id` | string | Unique department ID (D001, D002, ...) |
| `department_name` | string | Department name |
| `manager_id` | string | Manager employee ID |
| `budget` | float | Department budget |
| `location` | string | Department location |

### `data/sales_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Unique transaction ID (T0001, T0002, ...) |
| `employee_id` | string | Employee ID (links to employee_data.csv) |
| `customer_id` | string | Customer ID |
| `product_id` | string | Product ID |
| `quantity` | int | Number of items sold |
| `unit_price` | float | Price per unit |
| `total_amount` | float | Total transaction amount |
| `transaction_date` | string | Transaction date (YYYY-MM-DD) |
| `region` | string | Sales region (North, South, East, West) |

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_groupby_analysis.csv` - Basic groupby analysis
- [ ] `output/q1_aggregation_report.txt` - Aggregation report
- [ ] `output/q2_hierarchical_analysis.csv` - Hierarchical analysis
- [ ] `output/q2_performance_report.txt` - Performance report
- [ ] `output/q3_pivot_analysis.csv` - Pivot table analysis
- [ ] `output/q3_crosstab_analysis.csv` - Cross-tabulation analysis
- [ ] `output/q3_pivot_visualization.png` - Pivot visualization
 