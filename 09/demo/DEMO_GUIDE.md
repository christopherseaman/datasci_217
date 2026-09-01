# Time Series Analysis Demo Guide

## Overview
Hands-on practice with time series analysis using health and medical research data. These demos correspond to the three LIVE DEMO sections in the lecture, providing practical application of datetime handling, resampling, rolling windows, and time series visualization.

**Note**: All demos include comprehensive pedagogical context with Introduction sections and explanatory paragraphs throughout, making them suitable for independent student study. Each demo explains concepts before code, includes "Why this matters" sections, and provides clinical/analytical context for understanding.

## Demo Structure

### Demo 1: datetime Fundamentals and Time Series Indexing
**File**: `demo1_datetime_fundamentals.ipynb` (generated from `demo1_datetime_fundamentals.md`)  
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo1_datetime_fundamentals.ipynb)
**Placement**: After "Shifting and Lagging" section (~1/3 through lecture)  
**Duration**: 25 minutes  
**Focus**: Python datetime module, pandas DatetimeIndex, and time series indexing

**Key Activities**:
- Python datetime module basics with clinical timestamps
- pandas DatetimeIndex creation and manipulation
- Date range generation for patient monitoring schedules
- Time series indexing and selection with patient data
- Using diff() and pct_change() to analyze changes over time

**Dataset**: Daily patient vital signs data (temperature, heart rate, blood pressure) over 1 year

### Demo 2: Resampling and Rolling Window Operations
**File**: `demo2_indexing_resampling.ipynb` (generated from `demo2_indexing_resampling.md`)  
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo2_indexing_resampling.ipynb)
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
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/09/demo/demo3_visualization_automation.ipynb)
**Placement**: After "Time Zone Handling" section (end of lecture)  
**Duration**: 25 minutes  
**Focus**: Time series visualization, combining concepts from earlier lectures

**Key Activities**:
- Time series visualization with matplotlib and seaborn
- Combining pandas, matplotlib, and altair for interactive plots
- Time-zone localization and conversion for multi-site reporting
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
- Python environment with pandas, numpy, matplotlib, seaborn
- Jupyter notebook interface
- All demo notebooks are generated from Markdown files using Jupytext; Markdown is authoritative
- **Note**: altair is optional (commented out in Demo 3) - uncomment if you want interactive visualizations

## Setup Instructions

### Using uv venv (Recommended)

```bash
# Create the tested virtual environment
uv venv --python 3.12.13 .venv

# Activate environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -r requirements.txt

# Generate notebooks from Markdown (Markdown is authoritative)
jupytext --to notebook --output demo1_datetime_fundamentals.ipynb demo1_datetime_fundamentals.md
jupytext --to notebook --output demo2_indexing_resampling.ipynb demo2_indexing_resampling.md
jupytext --to notebook --output demo3_visualization_automation.ipynb demo3_visualization_automation.md
```

### Using Standard venv

```bash
# Create the tested virtual environment with Python 3.12.13
python3.12 -m venv .venv

# Activate environment
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Generate notebooks from Markdown
jupytext --to notebook --output demo1_datetime_fundamentals.ipynb demo1_datetime_fundamentals.md
jupytext --to notebook --output demo2_indexing_resampling.ipynb demo2_indexing_resampling.md
jupytext --to notebook --output demo3_visualization_automation.ipynb demo3_visualization_automation.md
```

## Instructor Notes

### Demo 1: datetime Fundamentals
- Emphasize the transition from Python's datetime to pandas DatetimeIndex
- Show how datetime indexing makes time series selection intuitive
- Use real clinical scenarios (patient visit schedules, lab test dates)
- Demonstrate diff() and pct_change() for analyzing temporal changes
- **Note**: This demo includes comprehensive pedagogical context (Introduction sections and explanatory paragraphs) to help students understand concepts independently

### Demo 2: Resampling and Rolling Windows
- Connect resampling to the `groupby()` concepts from Lecture 08
- Demonstrate how rolling windows smooth noisy medical data
- Show practical applications: detecting trends in patient outcomes
- Compare different window sizes and their effects
- Explain EWM parameters (span, alpha, halflife) and when to use them
- **Note**: This demo includes comprehensive pedagogical context (Introduction sections and explanatory paragraphs) to help students understand concepts independently

### Demo 3: Visualization and Integration
- Combine time series concepts with visualization from Lecture 07
- Show how different visualization libraries work with time series
- Demonstrate seasonal pattern identification
- Create publication-quality plots for medical research
- Integrate resampling, rolling windows, and visualization techniques
- Teach time-zone localization and conversion after the lecture's time-zone section
- **Note**: This demo includes comprehensive pedagogical context (Introduction sections and explanatory paragraphs) to help students understand concepts independently. Altair is optional (commented out) to avoid dependency issues.

## Common Pitfalls to Address
- **Timezone confusion**: Address localization and conversion in Demo 3, after the lecture introduces time zones
- **Frequency mismatches**: Demonstrate what happens when resampling irregular data
- **Rolling window edge effects**: Explain NaN values at the beginning of series
- **Visualization formatting**: Show how to properly format dates on axes

## Integration with Previous Lectures
- **Lecture 08 (GroupBy)**: Resampling is similar to groupby but for time intervals
- **Lecture 06 (Data Wrangling)**: Time series data often needs merging and combining
- **Lecture 07 (Visualization)**: Time series visualization uses matplotlib, seaborn, and altair
- **Lecture 05 (Data Cleaning)**: Time series data often has missing values and outliers
