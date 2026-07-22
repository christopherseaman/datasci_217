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
- `data/provider_data.csv` (healthcare provider information)
- `data/facility_data.csv` (healthcare facility information)
- `data/encounter_data.csv` (patient encounters/insurance claims)

## Complete the Three Questions

Open `assignment.ipynb` and work through the three questions. The notebook provides:

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
- `q2_filter_analysis.csv`, `q2_hierarchical_analysis.csv`, `q2_performance_report.txt`
- `q3_pivot_analysis.csv`, `q3_crosstab_analysis.csv`, `q3_pivot_visualization.png`

Run tests locally:

```bash
pytest -q 08/assignment/.github/test/test_assignment.py
```

GitHub Classroom will run the same tests on push.

### Question 1: Basic GroupBy Operations

**What you'll do:**
- Load and merge provider, facility, and encounter data
- Perform basic groupby operations with aggregation functions
- Use transform operations to add group statistics

**Skills:** groupby operations, aggregation functions, transform

**Output:** `output/q1_groupby_analysis.csv`, `output/q1_aggregation_report.txt`

### Question 2: Advanced GroupBy Operations

**What you'll do:**
- Apply filter operations to remove groups based on conditions
- Use apply operations with custom functions
- Perform hierarchical grouping with multiple columns
- Handle MultiIndex structures

**Skills:** filter operations, apply operations, hierarchical grouping, MultiIndex

**Output:** `output/q2_filter_analysis.csv`, `output/q2_performance_report.txt`, `output/q2_hierarchical_analysis.csv`

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
│   ├── provider_data.csv         # Healthcare provider information (500 providers)
│   ├── facility_data.csv         # Healthcare facility information (10 facilities)
│   └── encounter_data.csv       # Patient encounters/claims (5,000 encounters)
├── output/                       # Your saved results (created by your code)
│   ├── q1_groupby_analysis.csv   # Q1 groupby analysis
│   ├── q1_aggregation_report.txt # Q1 aggregation report
│   ├── q2_filter_analysis.csv       # Q2 filter operations analysis
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

### `data/provider_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `provider_id` | string | Unique provider ID (PR0001, PR0002, ...) |
| `provider_name` | string | Provider full name |
| `provider_type` | string | Provider type (Physician, Nurse Practitioner, etc.) |
| `facility_id` | string | Facility ID (links to facility_data.csv) |
| `specialty` | string | Medical specialty (Cardiology, Oncology, etc.) |
| `years_experience` | int | Years of experience |
| `license_number` | string | License number |

### `data/facility_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `facility_id` | string | Unique facility ID (FAC001, FAC002, ...) |
| `facility_name` | string | Facility name |
| `facility_type` | string | Facility type (Hospital, Clinic, Urgent Care, etc.) |
| `region` | string | Geographic region (North, South, East, West) |
| `beds` | int | Number of beds (if applicable) |
| `established_date` | string | Date facility was established (YYYY-MM-DD) |

### `data/encounter_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `encounter_id` | string | Unique encounter ID (ENC00001, ENC00002, ...) |
| `patient_id` | string | Patient ID |
| `provider_id` | string | Provider ID (links to provider_data.csv) |
| `facility_id` | string | Facility ID (links to facility_data.csv) |
| `encounter_date` | string | Encounter date (YYYY-MM-DD) |
| `encounter_type` | string | Type of encounter (Office Visit, Emergency, Inpatient, etc.) |
| `diagnosis_code` | string | ICD-10 diagnosis code |
| `procedure_code` | string | CPT procedure code |
| `service_charge` | float | Total service charge |
| `insurance_paid` | float | Amount insurance covered |
| `patient_paid` | float | Amount patient paid |
| `region` | string | Geographic region (North, South, East, West) |

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_groupby_analysis.csv` - Basic groupby analysis
- [ ] `output/q1_aggregation_report.txt` - Aggregation report
- [ ] `output/q2_filter_analysis.csv` - Filter operations analysis
- [ ] `output/q2_hierarchical_analysis.csv` - Hierarchical analysis
- [ ] `output/q2_performance_report.txt` - Performance report
- [ ] `output/q3_pivot_analysis.csv` - Pivot table analysis
- [ ] `output/q3_crosstab_analysis.csv` - Cross-tabulation analysis
- [ ] `output/q3_pivot_visualization.png` - Pivot visualization
 