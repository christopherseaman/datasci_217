# Assignment 9: Time Series Analysis

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
- `data/patient_vitals.csv` - Daily patient vital signs (200 patients, 1 year, ~69,000 records)
- `data/icu_monitoring.csv` - Hourly ICU patient data (75 patients, 6 months, variable stay lengths)
- `data/disease_surveillance.csv` - Monthly disease case counts by site (6 sites, 5 years, 360 records)

## Complete the Three Questions

This assignment consists of three separate notebooks, each focusing on different datasets and concepts:

- **Question 1** (`q1_datetime.md`) - Uses `patient_vitals.csv` (daily data)
- **Question 2** (`q2_resampling.md`) - Uses `icu_monitoring.csv` (hourly data) and `patient_vitals.csv` (daily data)
- **Question 3** (`q3_rolling.md`) - Uses `disease_surveillance.csv` (monthly data) and `patient_vitals.csv` (daily data)

**How to use the notebooks:**
1. Generate notebooks from markdown files using jupytext:
   ```bash
   jupytext --to notebook q1_datetime.md
   jupytext --to notebook q2_resampling.md
   jupytext --to notebook q3_rolling.md
   ```
2. Open each notebook and work through it sequentially
3. Read each cell carefully - they contain detailed instructions
4. Complete the TODO items by replacing `None` with your code
5. Run each cell to see your progress
6. Use the hints provided in comments
7. Check the submission checklist at the end of each notebook

### Question 1: datetime Fundamentals and Time Series Indexing
**File:** `q1_datetime.md` / `q1_datetime.ipynb`  
**Dataset:** `patient_vitals.csv` (daily patient vital signs)

**What you'll do:**
- Load and parse patient vital signs data with datetime indexing
- Perform basic datetime operations (calculate patient age, visit intervals)
- Create date ranges for clinical monitoring schedules
- Handle time zones for multi-site clinical trials

**Skills:** datetime module, DatetimeIndex, date range generation, time zones

**Output:** `output/q1_datetime_analysis.csv`, `output/q1_timezone_report.txt`

### Question 2: Resampling and Frequency Conversion
**File:** `q2_resampling.md` / `q2_resampling.ipynb`  
**Dataset:** `icu_monitoring.csv` (hourly ICU data), `patient_vitals.csv` (daily data)

**What you'll do:**
- Resample hourly ICU data to daily summaries
- Resample daily patient data to weekly and monthly aggregations
- Handle missing data in time series (forward fill, interpolation)
- Use multiple aggregation functions in resampling

**Skills:** resampling, frequency conversion, missing data handling, multiple aggregations

**Output:** `output/q2_resampling_analysis.csv`, `output/q2_missing_data_report.txt`

### Question 3: Rolling Windows and Visualization
**File:** `q3_rolling.md` / `q3_rolling.ipynb`  
**Dataset:** `disease_surveillance.csv` (monthly data), `patient_vitals.csv` (daily data)

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
├── q1_datetime.md                 # Question 1 notebook source (markdown)
├── q1_datetime.ipynb              # Question 1 notebook (generated)
├── q2_resampling.md               # Question 2 notebook source (markdown)
├── q2_resampling.ipynb            # Question 2 notebook (generated)
├── q3_rolling.md                  # Question 3 notebook source (markdown)
├── q3_rolling.ipynb               # Question 3 notebook (generated)
├── data_generator.md              # Data generator source (markdown)
├── data_generator.ipynb          # Generated data generator notebook
├── requirements.txt               # Python dependencies
├── data/                          # Generated datasets (created by data_generator)
│   ├── patient_vitals.csv        # Daily patient vital signs (200 patients, ~69k records)
│   ├── icu_monitoring.csv        # Hourly ICU data (75 patients, variable stay lengths)
│   └── disease_surveillance.csv  # Monthly disease cases by site (6 sites, 5 years, 360 records)
├── output/                        # Your saved results (create this directory)
│   ├── q1_datetime_analysis.csv  # Q1 datetime analysis
│   ├── q1_timezone_report.txt   # Q1 timezone report
│   ├── q2_resampling_analysis.csv # Q2 resampling analysis
│   ├── q2_missing_data_report.txt # Q2 missing data report
│   ├── q3_rolling_analysis.csv   # Q3 rolling analysis
│   ├── q3_trend_analysis.png      # Q3 trend analysis plot
│   └── q3_visualization.png      # Q3 time series visualization
└── .github/
    └── test/
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

**Use case:** Daily monitoring, resampling to weekly/monthly, trend analysis, missing data patterns

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

**Use case:** High-frequency data, resampling hourly to daily, rolling windows, patient trajectories, recovery patterns

### `data/disease_surveillance.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | Month-end date (YYYY-MM-DD) |
| `site` | string | Surveillance site (Site_A through Site_F, 6 sites total) |
| `cases` | int | Number of disease cases |
| `temperature` | float | Average monthly temperature (°F) |
| `humidity` | float | Average monthly humidity (%) |

**Use case:** Multi-site analysis, seasonal patterns, monthly aggregations, climate effects, outbreak detection

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

## Missing Data Handling (Question 2)

When working with missing data in time series (Part 2.4), keep these guidelines in mind:

1. **Understanding Missing Data Sources:**
   - The `patient_vitals` dataset has naturally occurring missing data (~5% missing visits per patient)
   - You can also create missing data by upsampling (e.g., monthly to daily) to practice different imputation methods
   - For upsampling practice: aggregate `patient_vitals` by date first to create a single time series, then upsample monthly to daily

2. **Creating a Time Series with Missing Values:**
   - If using naturally occurring missing data: aggregate `patient_vitals` by date using `groupby('date')` and aggregate one column (e.g., `['temperature']`) with `.mean()` to create a single daily time series
   - If creating missing data via upsampling: use the monthly resampled data from Part 2.3 and upsample to daily using `.resample('D').asfreq()`
   - This creates a time series where missing values are clearly visible

3. **Imputation Methods:**
   - **Forward fill (`.ffill()`)**: Carries the last known value forward - good for stable measurements
   - **Backward fill (`.bfill()`)**: Carries the next known value backward - useful when you have recent data
   - **Interpolation (`.interpolate()`)**: Fills missing values using linear interpolation between known points - good for gradual changes
   - **Time-based interpolation (`.interpolate(method='time')`)**: Uses time-aware interpolation - best for time series data
   - **Rolling mean**: Fill missing values with rolling mean of surrounding values - smooths out noise

4. **Choosing an Imputation Method:**
   - Consider the nature of your data: temperature changes gradually (interpolation), while discrete counts might use forward fill
   - For healthcare data, linear or time-based interpolation often works well for continuous measurements
   - Document your choice and rationale in the missing data report

## Getting Help

- Review the lecture README.md for examples
- Check the demos for hands-on practice
- Consult pandas documentation: https://pandas.pydata.org/docs/
- Ask questions early - don't wait until the deadline!
