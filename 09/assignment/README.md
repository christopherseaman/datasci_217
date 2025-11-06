# Assignment 9: Time Series Analysis

**Deliverable:** Completed `assignment.ipynb` with output files in `output/`

## Overview

This assignment focuses on time series analysis using health and medical research data. You'll work with patient monitoring data, clinical trial results, and disease surveillance data to practice datetime handling, resampling, rolling windows, and time series visualization.

**Prerequisites:** This assignment uses concepts from:
- Lecture 05 (GroupBy) - for aggregations in resampling
- Lecture 07 (Visualization) - for time series plots
- Lecture 09 (Time Series) - all core concepts

## Environment Setup

### Using uv venv (Recommended)

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Using Standard venv

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Generate Notebooks from Markdown

This assignment uses **jupytext** to pair markdown files with notebooks. Edit the markdown files, then generate notebooks:

```bash
# Install jupytext if needed
uv pip install jupytext
# or
pip install jupytext

# Generate notebooks from markdown
jupytext --to notebook assignment.md
jupytext --to notebook data_generator.md
```

**Note:** You can edit either the `.md` or `.ipynb` files - changes sync automatically with jupytext.

## Generate the Dataset

Run the data generator to create your health/medical datasets:

```bash
# Option 1: Run the markdown file directly (if using jupytext)
jupytext --to notebook data_generator.md
jupyter notebook data_generator.ipynb

# Option 2: Run the notebook directly
jupyter notebook data_generator.ipynb
```

Run all cells to create the CSV files in `data/`:
- `data/patient_vitals.csv` - Daily patient vital signs (1 year)
- `data/icu_monitoring.csv` - Hourly ICU patient data (6 months)
- `data/disease_surveillance.csv` - Monthly disease case counts by site (3 years)

## Complete the Three Questions

Open `assignment.ipynb` (or `assignment.md`) and work through the three questions. The notebook provides:

- **Step-by-step instructions** with clear TODO items
- **Helpful hints** for each operation
- **Sample data** and examples to guide your work
- **Validation checks** to ensure your outputs are correct

**How to use the scaffold:**
1. Read each cell carefully - they contain detailed instructions
2. Complete the TODO items by replacing `None` with your code
3. Run each cell to see your progress
4. Use the hints provided in comments
5. Check the submission checklist at the end

### Question 1: datetime Fundamentals and Time Series Indexing

**What you'll do:**
- Load and parse patient vital signs data with datetime indexing
- Perform basic datetime operations (calculate patient age, visit intervals)
- Create date ranges for clinical monitoring schedules
- Perform time series indexing and selection (select by month, date range)
- Handle time zones for multi-site clinical trials

**Skills:** datetime module, DatetimeIndex, date range generation, time series indexing, time zones

**Output:** `output/q1_datetime_analysis.csv`, `output/q1_timezone_report.txt`

### Question 2: Resampling and Frequency Conversion

**What you'll do:**
- Resample hourly ICU data to daily summaries
- Resample daily patient data to weekly and monthly aggregations
- Handle missing data in time series (forward fill, interpolation)
- Apply shifting and lagging operations to detect changes
- Use multiple aggregation functions in resampling

**Skills:** resampling, frequency conversion, missing data handling, shifting/lagging, multiple aggregations

**Output:** `output/q2_resampling_analysis.csv`, `output/q2_missing_data_report.txt`

### Question 3: Rolling Windows and Visualization

**What you'll do:**
- Apply rolling window operations for trend detection (7-day, 30-day windows)
- Calculate exponentially weighted moving averages
- Create time series visualizations using matplotlib and seaborn
- Identify trends and patterns in patient data
- Generate publication-quality plots

**Skills:** rolling windows, exponentially weighted functions, time series visualization, trend analysis

**Output:** `output/q3_rolling_analysis.csv`, `output/q3_trend_analysis.png`, `output/q3_visualization.png`

## Assignment Structure

```
09/assignment/
├── README.md                      # This file - assignment instructions
├── assignment.md                  # Notebook source (edit this, generate .ipynb)
├── assignment.ipynb               # Generated notebook (or edit directly)
├── data_generator.md              # Data generator source (markdown)
├── data_generator.ipynb          # Generated data generator notebook
├── requirements.txt               # Python dependencies
├── data/                          # Generated datasets (created by data_generator)
│   ├── patient_vitals.csv        # Daily patient vital signs (365 days)
│   ├── icu_monitoring.csv        # Hourly ICU data (6 months)
│   └── disease_surveillance.csv  # Monthly disease cases by site (36 months)
├── output/                        # Your saved results (create this directory)
│   ├── q1_datetime_analysis.csv  # Q1 datetime analysis
│   ├── q1_timezone_report.txt   # Q1 timezone report
│   ├── q2_resampling_analysis.csv # Q2 resampling analysis
│   ├── q2_missing_data_report.txt # Q2 missing data report
│   ├── q3_rolling_analysis.csv   # Q3 rolling analysis
│   ├── q3_trend_analysis.png      # Q3 trend analysis plot
│   └── q3_visualization.png      # Q3 time series visualization
└── .github/
    └── tests/
        ├── test_assignment.py    # Auto-grading tests
        └── requirements.txt      # Test dependencies
```

## Dataset Schemas

### `data/patient_vitals.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | Measurement date (YYYY-MM-DD) |
| `patient_id` | string | Unique patient identifier |
| `temperature` | float | Body temperature in Fahrenheit |
| `heart_rate` | int | Heart rate in beats per minute |
| `blood_pressure_systolic` | int | Systolic blood pressure (mmHg) |
| `blood_pressure_diastolic` | int | Diastolic blood pressure (mmHg) |
| `weight` | float | Patient weight in kg |

**Use case:** Daily monitoring, resampling to weekly/monthly, trend analysis

### `data/icu_monitoring.csv`

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | string | Timestamp (YYYY-MM-DD HH:MM:SS) |
| `patient_id` | string | ICU patient identifier |
| `heart_rate` | int | Heart rate (bpm) |
| `blood_pressure_systolic` | int | Systolic BP (mmHg) |
| `blood_pressure_diastolic` | int | Diastolic BP (mmHg) |
| `oxygen_saturation` | int | Oxygen saturation (%) |
| `temperature` | float | Body temperature (°F) |

**Use case:** High-frequency data, resampling hourly to daily, rolling windows

### `data/disease_surveillance.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | Month-end date (YYYY-MM-DD) |
| `site` | string | Surveillance site (Site_A, Site_B, Site_C) |
| `cases` | int | Number of disease cases |
| `temperature` | float | Average monthly temperature (°F) |
| `humidity` | float | Average monthly humidity (%) |

**Use case:** Multi-site analysis, seasonal patterns, monthly aggregations

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_datetime_analysis.csv` - datetime analysis results
- [ ] `output/q1_timezone_report.txt` - timezone handling report
- [ ] `output/q2_resampling_analysis.csv` - resampling results
- [ ] `output/q2_missing_data_report.txt` - missing data handling report
- [ ] `output/q3_rolling_analysis.csv` - rolling window analysis
- [ ] `output/q3_trend_analysis.png` - trend analysis visualization
- [ ] `output/q3_visualization.png` - comprehensive time series plot

## Grading Criteria

This assignment focuses on **practical competence** - demonstrating that you can use the time series tools correctly, not excellence or optimization.

- **Question 1:** Correct datetime parsing, indexing, and time zone handling
- **Question 2:** Proper resampling operations and missing data handling
- **Question 3:** Accurate rolling window calculations and clear visualizations

Questions build progressively - each question uses concepts from earlier questions.

## Tips

- **Start with the data generator** - understand your datasets before starting
- **Use the hints** - each TODO has helpful comments
- **Test incrementally** - run cells as you complete them
- **Check outputs** - verify your CSV files and plots look correct
- **Read error messages** - they often point to the exact issue

## Getting Help

- Review the lecture README.md for examples
- Check the demos for hands-on practice
- Consult pandas documentation: https://pandas.pydata.org/docs/
- Ask questions early - don't wait until the deadline!
