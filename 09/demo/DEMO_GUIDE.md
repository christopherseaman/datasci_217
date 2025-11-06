# Time Series Analysis Demo Guide

## Overview
Hands-on practice with time series analysis using health and medical research data. These demos correspond to the three LIVE DEMO sections in the lecture, providing practical application of datetime handling, resampling, rolling windows, and time series visualization.

## Demo Structure

### Demo 1: datetime Fundamentals and Time Series Indexing
**File**: `demo1_datetime_fundamentals.ipynb` (generated from `demo1_datetime_fundamentals.md`)  
**Placement**: After "Shifting and Lagging" section (~1/3 through lecture)  
**Duration**: 25 minutes  
**Focus**: Python datetime module, pandas DatetimeIndex, and time series indexing

**Key Activities**:
- Python datetime module basics with clinical timestamps
- pandas DatetimeIndex creation and manipulation
- Date range generation for patient monitoring schedules
- Time series indexing and selection with patient data
- Time zone handling for multi-site clinical trials

**Dataset**: Daily patient vital signs data (temperature, heart rate, blood pressure) over 1 year

### Demo 2: Resampling and Rolling Window Operations
**File**: `demo2_indexing_resampling.ipynb` (generated from `demo2_indexing_resampling.md`)  
**Placement**: After "Resampling" section (~2/3 through lecture)  
**Duration**: 25 minutes  
**Focus**: Resampling operations, rolling windows, and exponentially weighted functions

**Key Activities**:
- Resampling hourly ICU data to daily summaries
- Frequency conversion (daily to weekly/monthly)
- Rolling window operations for trend detection
- Exponentially weighted moving averages
- Combining resampling with visualization

**Dataset**: Hourly ICU patient monitoring data (heart rate, blood pressure, oxygen saturation) over 6 months

### Demo 3: Time Series Visualization and Integration
**File**: `demo3_visualization_automation.ipynb` (generated from `demo3_visualization_automation.md`)  
**Placement**: After "Time Zone Handling" section (end of lecture)  
**Duration**: 25 minutes  
**Focus**: Time series visualization, combining concepts from earlier lectures

**Key Activities**:
- Time series visualization with matplotlib and seaborn
- Combining pandas, matplotlib, and altair for interactive plots
- Seasonal pattern identification
- Multi-variable time series visualization
- Integration with concepts from Lecture 07 (visualization)

**Dataset**: Multi-year disease surveillance data (monthly case counts, temperature, humidity) from multiple sites

## Learning Objectives
- Master datetime data types and parsing with real health data
- Perform time series indexing and selection
- Use resampling and frequency conversion for clinical data
- Apply rolling window operations for trend analysis
- Create comprehensive time series visualizations
- Integrate time series concepts with visualization tools

## Required Materials
- Python environment with pandas, numpy, matplotlib, seaborn, altair
- Jupyter notebook interface
- All demo notebooks are generated from markdown files using jupytext

## Setup Instructions

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

# Generate notebooks from markdown (if using jupytext)
jupytext --to notebook demo1_datetime_fundamentals.md
jupytext --to notebook demo2_indexing_resampling.md
jupytext --to notebook demo3_visualization_automation.md
```

### Using Standard venv

```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Generate notebooks from markdown
jupytext --to notebook demo1_datetime_fundamentals.md
```

## Instructor Notes

### Demo 1: datetime Fundamentals
- Emphasize the transition from Python's datetime to pandas DatetimeIndex
- Show how datetime indexing makes time series selection intuitive
- Use real clinical scenarios (patient visit schedules, lab test dates)
- Highlight time zone issues that arise in multi-site studies

### Demo 2: Resampling and Rolling Windows
- Connect resampling to the `groupby()` concepts from Lecture 5
- Demonstrate how rolling windows smooth noisy medical data
- Show practical applications: detecting trends in patient outcomes
- Compare different window sizes and their effects

### Demo 3: Visualization and Integration
- Combine time series concepts with visualization from Lecture 07
- Show how different visualization libraries work with time series
- Demonstrate seasonal pattern identification
- Create publication-quality plots for medical research

## Common Pitfalls to Address
- **Timezone confusion**: Show how to properly localize and convert timezones
- **Frequency mismatches**: Demonstrate what happens when resampling irregular data
- **Rolling window edge effects**: Explain NaN values at the beginning of series
- **Visualization formatting**: Show how to properly format dates on axes

## Integration with Previous Lectures
- **Lecture 05 (GroupBy)**: Resampling is similar to groupby but for time intervals
- **Lecture 06 (Data Wrangling)**: Time series data often needs merging and combining
- **Lecture 07 (Visualization)**: Time series visualization uses matplotlib, seaborn, and altair
- **Lecture 08 (Data Cleaning)**: Time series data often has missing values and outliers
