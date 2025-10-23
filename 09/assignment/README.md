# Assignment 9: Time Series Analysis

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
- `data/stock_prices.csv` (daily stock price data)
- `data/weather_data.csv` (hourly weather data)
- `data/sales_data.csv` (monthly sales data)

## Complete the Four Questions

Open `assignment.ipynb` and work through the four questions. The notebook provides:

- **Step-by-step instructions** with clear TODO items
- **Helpful hints** for each operation
- **Sample data** and examples to guide your work
- **Validation checks** to ensure your outputs are correct

**Prerequisites:** This assignment uses datetime handling, resampling, rolling windows, and time series visualization from Lecture 09.

**How to use the scaffold notebook:**
1. Read each cell carefully - they contain detailed instructions
2. Complete the TODO items by replacing `None` with your code
3. Run each cell to see your progress
4. Use the hints provided in comments
5. Check the submission checklist at the end

### Question 1: datetime Fundamentals

**What you'll do:**
- Load and parse time series data with datetime indexing
- Perform basic datetime operations and arithmetic
- Handle time zones and business day calculations
- Create date ranges with different frequencies

**Skills:** datetime module, DatetimeIndex, time zones, business days

**Output:** `output/q1_datetime_analysis.csv`, `output/q1_timezone_report.txt`

### Question 2: Time Series Indexing and Resampling

**What you'll do:**
- Perform time series indexing and selection
- Use resampling for frequency conversion
- Handle missing data in time series
- Apply shifting and lagging operations

**Skills:** time series indexing, resampling, missing data handling, shifting

**Output:** `output/q2_resampling_analysis.csv`, `output/q2_missing_data_report.txt`

### Question 3: Rolling Window Operations

**What you'll do:**
- Apply rolling window operations for trend analysis
- Calculate exponentially weighted moving averages
- Perform expanding window operations
- Create custom rolling functions

**Skills:** rolling windows, moving averages, exponential smoothing, custom functions

**Output:** `output/q3_rolling_analysis.csv`, `output/q3_trend_analysis.png`

### Question 4: Time Series Visualization and Automation

**What you'll do:**
- Create comprehensive time series visualizations
- Perform seasonal decomposition analysis
- Set up automated analysis workflows
- Monitor time series performance metrics

**Skills:** time series visualization, seasonal decomposition, automation, monitoring

**Output:** `output/q4_visualization.png`, `output/q4_seasonal_analysis.csv`, `output/q4_automation_report.txt`

## Assignment Structure

```
09/assignment/
├── README.md                      # This file - assignment instructions
├── assignment.md                  # Notebook source (for jupytext)
├── assignment.ipynb              # Completed notebook (you work here)
├── data_generator.ipynb          # Run once to create datasets
├── data/                         # Generated datasets
│   ├── stock_prices.csv          # Daily stock price data (3 years)
│   ├── weather_data.csv          # Hourly weather data (1 year)
│   └── sales_data.csv            # Monthly sales data (5 years)
├── output/                       # Your saved results (created by your code)
│   ├── q1_datetime_analysis.csv  # Q1 datetime analysis
│   ├── q1_timezone_report.txt    # Q1 timezone report
│   ├── q2_resampling_analysis.csv # Q2 resampling analysis
│   ├── q2_missing_data_report.txt # Q2 missing data report
│   ├── q3_rolling_analysis.csv   # Q3 rolling analysis
│   ├── q3_trend_analysis.png     # Q3 trend analysis plot
│   ├── q4_visualization.png      # Q4 time series visualization
│   ├── q4_seasonal_analysis.csv  # Q4 seasonal analysis
│   └── q4_automation_report.txt  # Q4 automation report
└── .github/
    └── test/
        ├── test_assignment.py    # Auto-grading tests
        └── requirements.txt      # Test dependencies
```

## Dataset Schemas

### `data/stock_prices.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | Trading date (YYYY-MM-DD) |
| `open` | float | Opening price |
| `high` | float | Highest price |
| `low` | float | Lowest price |
| `close` | float | Closing price |
| `volume` | int | Trading volume |
| `adj_close` | float | Adjusted closing price |

### `data/weather_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | string | Timestamp (YYYY-MM-DD HH:MM:SS) |
| `temperature` | float | Temperature in Celsius |
| `humidity` | float | Humidity percentage |
| `pressure` | float | Atmospheric pressure |
| `wind_speed` | float | Wind speed in km/h |
| `precipitation` | float | Precipitation in mm |

### `data/sales_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | Month-end date (YYYY-MM-DD) |
| `product` | string | Product category |
| `region` | string | Sales region |
| `sales` | float | Monthly sales amount |
| `units` | int | Number of units sold |
| `revenue` | float | Total revenue |

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_datetime_analysis.csv` - datetime analysis
- [ ] `output/q1_timezone_report.txt` - timezone report
- [ ] `output/q2_resampling_analysis.csv` - resampling analysis
- [ ] `output/q2_missing_data_report.txt` - missing data report
- [ ] `output/q3_rolling_analysis.csv` - rolling analysis
- [ ] `output/q3_trend_analysis.png` - trend analysis plot
- [ ] `output/q4_visualization.png` - time series visualization
- [ ] `output/q4_seasonal_analysis.csv` - seasonal analysis
- [ ] `output/q4_automation_report.txt` - automation report