# Final Exam: Chicago Beach Weather Sensors Analysis

**Total: 100 points**  
**Format: Take-home exam**  
**Time: Recommended 4-6 hours**

## Overview

This final exam tests your ability to complete a full data science project workflow using the **Chicago Beach Weather Sensors Dataset**. You'll follow the same 9-phase workflow demonstrated in Lecture 11, but with a different dataset and focused questions.

**Assessment Breakdown:**
- **60% Automated Grading** - Code execution and output validation
- **40% Writeup Evaluation** - Written report with findings, visualizations, and model results

## Dataset: Chicago Beach Weather Sensors

**Source:** [Beach Weather Stations - Automated Sensors](https://data.cityofchicago.org/Parks-Recreation/Beach-Weather-Stations-Automated-Sensors/k7hf-8y75/about_data)

Real-time weather sensor readings from Lake Michigan beaches including:
- Temperature measurements (air and water)
- Wind speed and direction
- Water conditions
- Air quality measurements
- Sensor metadata

**Dataset Size:** ~195,000+ records (varies by download date)

**Key Challenges:**
- Irregular sampling intervals (sensors report at different frequencies)
- Sensor dropouts (missing data periods)
- Multiple sensor streams to merge
- Time series analysis required throughout

### Downloading the Dataset

**Use the provided download script to ensure you have the exact same dataset as the auto-grading system:**

```bash
chmod +x download_data.sh
./download_data.sh
```

This will download the Chicago Beach Weather Sensors dataset and place it in `data/beach_sensors.csv`.

**Alternative:** If the script doesn't work, you can manually download from:
- [Beach Weather Stations - Automated Sensors](https://data.cityofchicago.org/Parks-Recreation/Beach-Weather-Stations-Automated-Sensors/k7hf-8y75/about_data)
- Click "Export" → "CSV"
- Save as `data/beach_sensors.csv`

**Important:** Using the download script ensures data consistency for auto-grading. If you use a different dataset, some tests may fail.

## Deliverables

### 1. Code Notebook (`assignment.ipynb`)
Complete all questions in the provided notebook. Your notebook should:
- Execute without errors
- Generate all required output files
- Include clear comments explaining your approach
- Follow the 9-phase workflow structure

### 2. Written Report (`report.md` or `report.pdf`)
A 3-5 page report including:
- **Executive Summary** (1 paragraph)
- **Key Findings** from each phase
- **Visualizations** (at least 5 figures with captions)
- **Model Results** (performance metrics, feature importance)
- **Time Series Patterns** identified
- **Limitations and Next Steps**

## File Structure

```
11/assignment/
├── README.md                    # This file
├── assignment.md                # Main assignment overview
├── q1_setup_exploration.md      # Q1: Setup & Exploration
├── q2_data_cleaning.md          # Q2: Data Cleaning
├── q3_data_wrangling.md         # Q3: Data Wrangling
├── q4_feature_engineering.md     # Q4: Feature Engineering
├── q5_pattern_analysis.md       # Q5: Pattern Analysis
├── q6_modeling_preparation.md   # Q6: Modeling Preparation
├── q7_modeling.md               # Q7: Modeling
├── q8_results.md                # Q8: Results
├── q9_writeup.md                # Q9: Writeup
├── download_data.sh             # Dataset download script
├── requirements.txt             # Python dependencies
├── data/                        # Input data (created by download script)
│   └── beach_sensors.csv       # Main sensor data
├── output/                      # Your saved results (create this)
│   ├── q1_data_info.txt        # Q1 outputs
│   ├── q1_exploration.csv       # Q1 outputs
│   ├── q1_visualizations.png    # Q1 outputs
│   ├── q2_cleaned_data.csv      # Q2 outputs
│   ├── q2_cleaning_report.txt  # Q2 outputs
│   ├── q2_rows_cleaned.txt      # Q2 outputs
│   └── ... (see Required Milestone Artifacts for full list)
├── report.md                    # Your written report (Q9)
└── .github/
    └── test/
        └── test_assignment.py   # Auto-grading tests
```

## Setup

1. **Create virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Download dataset:**
   ```bash
   chmod +x download_data.sh
   ./download_data.sh
   ```
   This will create `data/beach_sensors.csv` automatically.

4. **Start working:**
   ```bash
   # Convert first assignment file to notebook
   jupytext --to notebook q1_setup_exploration.md
   jupyter notebook q1_setup_exploration.ipynb
   ```
   
   **Note:** The assignment is split into 9 files, one per question:
   - `q1_setup_exploration.md` - Setup & Exploration (Phase 1-2)
   - `q2_data_cleaning.md` - Data Cleaning (Phase 3)
   - `q3_data_wrangling.md` - Data Wrangling (Phase 4)
   - `q4_feature_engineering.md` - Feature Engineering (Phase 5)
   - `q5_pattern_analysis.md` - Pattern Analysis (Phase 6)
   - `q6_modeling_preparation.md` - Modeling Preparation (Phase 7)
   - `q7_modeling.md` - Modeling (Phase 8)
   - `q8_results.md` - Results (Phase 9)
   - `q9_writeup.md` - Writeup instructions
   
   Work through the files in order (q1 → q2 → ... → q9)

## Assignment Structure

This assignment uses a **looser scaffold** than previous assignments:
- You'll write your own code from scratch (no TODO placeholders)
- Follow the 9-phase workflow structure demonstrated in Lecture 11
- Produce required milestone artifacts (see below)
- Make your own decisions about data cleaning, feature engineering, and modeling approaches
- Reference Lecture 11 notebooks for examples, but implement your own solution

## Using Lecture 11 as Reference

When working on each assignment file, reference the corresponding lecture notebook:

| Assignment File | Lecture Notebook | Phase(s) Covered |
|----------------|------------------|------------------|
| `q1_setup_exploration.md` | Lecture 11 Notebook 1 | Phases 1-2 (Setup, Exploration) |
| `q2_data_cleaning.md` | Lecture 11 Notebook 1 | Phase 3 (Cleaning) |
| `q3_data_wrangling.md` | Lecture 11 Notebook 2 | Phase 4 (Wrangling) |
| `q4_feature_engineering.md` | Lecture 11 Notebook 2 | Phase 5 (Feature Engineering) |
| `q5_pattern_analysis.md` | Lecture 11 Notebook 3 | Phase 6 (Pattern Analysis) |
| `q6_modeling_preparation.md` | Lecture 11 Notebook 3 | Phase 7 (Modeling Prep) |
| `q7_modeling.md` | Lecture 11 Notebook 4 | Phase 8 (Modeling) |
| `q8_results.md` | Lecture 11 Notebook 4 | Phase 9 (Results) |
| `q9_writeup.md` | Lecture 11 Notebook 4 | Writeup |

**Key Difference:** The lecture shows **HOW** to do things with the NYC Taxi dataset (event-based data). Your assignment requires you to **DO** them with the Chicago Beach Weather Sensors dataset (time-series data). Use the lecture as a guide, but adapt the techniques to your dataset.

**Time Series Note:** This dataset is time-series data (continuous sensor readings), unlike the lecture's event-based taxi data. See Lecture 09 for time series operations, and note the helpful nudges in each question file about adapting techniques for time-series data.

## Questions Overview

The exam follows the 9-phase data science workflow:

**Phase 1-2: Setup & Exploration (Q1)** - 6 points
- Load data and perform initial inspection
- Create basic visualizations
- Identify data quality issues

**Phase 3: Data Cleaning (Q2)** - 9 points
- Handle missing data
- Detect and handle outliers
- Validate data types
- Remove duplicates

**Phase 4: Data Wrangling (Q3)** - 9 points
- Merge multiple sensor streams (if applicable)
- Parse datetime columns
- Set datetime index
- Extract time-based features

**Phase 5: Feature Engineering (Q4)** - 9 points
- Create derived features
- Perform time-based aggregations
- Calculate rolling windows
- Create categorical features

**Phase 6: Pattern Analysis (Q5)** - 6 points
- Identify trends over time
- Analyze seasonal patterns
- Create correlation analysis
- Advanced visualizations

**Phase 7: Modeling Preparation (Q6)** - 3 points
- Temporal train/test split
- Feature selection
- Handle categorical variables

**Phase 8: Modeling (Q7)** - 9 points
- Train multiple models
- Evaluate performance
- Compare models
- Extract feature importance

**Phase 9: Results (Q8)** - 3 points
- Generate final visualizations
- Create summary tables
- Document key findings

**Code Quality & Execution** - 6 points
- Code executes without errors
- All required artifacts generated

**Writeup (Q9)** - 40 points
- Written report with all findings (see Writeup section below)

## Required Milestone Artifacts

Each phase must produce specific output files for auto-grading. These are **milestone artifacts** that demonstrate completion of each phase:

### Q1 (Exploration):
- `output/q1_data_info.txt` - Dataset information: shape, column names, data types, date range (if temporal), missing value counts
- `output/q1_exploration.csv` - Summary statistics with columns: `column_name`, `mean`, `std`, `min`, `max`, `missing_count` (one row per numeric column)
- `output/q1_visualizations.png` - At least 2 plots (distribution, time series if applicable) with clear labels

### Q2 (Cleaning):
- `output/q2_cleaned_data.csv` - Cleaned dataset (same structure as original, with cleaned values)
- `output/q2_cleaning_report.txt` - What was cleaned: missing data handling method, outliers removed/capped, duplicates removed, rows before/after
- `output/q2_rows_cleaned.txt` - Single number: total rows after cleaning

### Q3 (Wrangling):
- `output/q3_wrangled_data.csv` - Merged/wrangled dataset with datetime index set
- `output/q3_temporal_features.csv` - Must include columns: `hour`, `day_of_week`, `month` (and original datetime column)
- `output/q3_datetime_info.txt` - Date range of dataset after datetime parsing

### Q4 (Feature Engineering):
- `output/q4_features.csv` - Dataset with derived features added
- `output/q4_rolling_features.csv` - Must include at least one rolling window calculation (e.g., rolling mean, rolling median)
- `output/q4_feature_list.txt` - List of new features created (one per line)

### Q5 (Pattern Analysis):
- `output/q5_correlations.csv` - Correlation matrix (can be subset of key variables)
- `output/q5_patterns.png` - Advanced visualizations showing trends/seasonality
- `output/q5_trend_summary.txt` - Brief text summary of key patterns identified

### Q6 (Modeling Prep):
- `output/q6_X_train.csv` - Training features
- `output/q6_X_test.csv` - Test features
- `output/q6_y_train.csv` - Training target
- `output/q6_y_test.csv` - Test target
- `output/q6_train_test_info.txt` - Train/test sizes, date ranges, feature count

### Q7 (Modeling):
- `output/q7_predictions.csv` - Model predictions (columns: `actual`, `predicted_model1`, `predicted_model2`, etc.)
- `output/q7_model_metrics.txt` - Performance metrics: R², RMSE, MAE for each model (clearly labeled)
- `output/q7_feature_importance.csv` - Feature importance (if applicable): `feature`, `importance` columns

### Q8 (Results):
- `output/q8_final_visualizations.png` - Final summary visualizations
- `output/q8_summary.csv` - Key findings summary table
- `output/q8_key_findings.txt` - Text summary of main insights

**Note:** Exact column names and formats may vary slightly, but the content requirements above must be met for auto-grading to work.

## Auto-Grading (60 points)

**Run tests locally before submitting:**
```bash
cd 11/assignment
pytest -q .github/test/test_assignment.py -v
```

GitHub Classroom will run the same tests on push.

**What's tested:**
- Code execution (no errors)
- Output file existence (all 24 required artifacts)
- Data format validation (columns, dtypes, value ranges)
- Statistical calculations (means, counts, aggregations)
- Model outputs (predictions, metrics)
- Visualization file creation
- Temporal features (hour, day_of_week, month ranges)
- Train/test split (temporal, not random)
- Feature importance format

**Test File Location:** `.github/test/test_assignment.py` - You can review this file to understand exactly what's being tested.

## Writeup Evaluation (40 points)

Your written report (`report.md`) will be evaluated on:

**Documentation Quality (12 points)**
- **Process Explanation (4 points):** Clear, step-by-step description of entire workflow
- **Decision Rationale (4 points):** All major decisions explained with reasoning (why you chose specific cleaning methods, features, models, etc.)
- **Professional Presentation (4 points):** Well-formatted markdown, error-free, professional appearance

**Visualizations & Tables (14 points)**
- **Time Series Visualizations (5 points):** At least 2 time series plots showing trends/patterns clearly
- **Other Visualizations (5 points):** At least 3 additional plots (distributions, correlations, model performance, etc.) with appropriate choices and clear labels
- **Tables (2 points):** Model results and key findings presented in well-formatted tables
- **Best Practices (2 points):** All visualizations have titles, axis labels, legends, and captions

**Interpretation & Insights (14 points)**
- **EDA Findings (5 points):** Key patterns from exploration phase clearly summarized
- **Time Series Patterns (5 points):** Trends, seasonality, and temporal relationships identified and explained
- **Model Interpretation (2 points):** Model performance metrics interpreted correctly (what do R², RMSE mean in context?)
- **Limitations & Conclusions (2 points):** Honest assessment of limitations and conclusions supported by evidence

### Writeup Format Requirements

**File Format:** Markdown (`.md`) with embedded images

**Image Embedding:**
- Save visualizations to `output/` directory
- Embed using: `![Figure 1: Description](output/figure1.png)`
- All images must have captions (either in alt text or as separate text)

**Tables:**
- Use markdown table format (recommended for model results)
- Example:
  ```markdown
  | Model | R² | RMSE | MAE |
  |-------|----|----|----|
  | Linear Regression | 0.XX | X.XX | X.XX |
  | Random Forest | 0.XX | X.XX | X.XX |
  ```

**Length:** 3-5 pages when rendered (adjust formatting as needed)

**Structure:** See "Writeup Template" section below

## Submission Checklist

Before submitting, ensure:

- [ ] All code cells execute without errors
- [ ] All required output files exist in `output/` directory
- [ ] Written report is complete (3-5 pages)
- [ ] Report includes at least 5 visualizations with captions
- [ ] All time series operations are demonstrated
- [ ] Model results are clearly presented
- [ ] Code is commented and readable

## Time Management Tips

- **Phases 1-3 (Q1-Q2):** ~1 hour - Get data loaded and cleaned
- **Phases 4-5 (Q3-Q4):** ~1.5 hours - Wrangling and feature engineering
- **Phases 6-7 (Q5-Q6):** ~1 hour - Analysis and prep
- **Phase 8 (Q7):** ~1 hour - Modeling
- **Phase 9 + Writeup (Q8-Q9):** ~1.5 hours - Results and documentation

**Total: ~6 hours** (adjust based on your pace)

## Academic Integrity

This is a take-home exam. You may:
- Use course materials and notes
- Use pandas/numpy/matplotlib documentation
- Use online resources for syntax help

You may NOT:
- Share code or solutions with others
- Use AI tools to generate complete solutions
- Copy code from other students

All work must be your own.

## Writeup Template

Use this structure for your `report.md`:

```markdown
# Chicago Beach Weather Sensors Analysis

## Executive Summary
[1 paragraph: What dataset was analyzed, main goal/question, key finding in one sentence]

## Phase-by-Phase Findings

### Phase 1-2: Exploration
[What you found, key insights, data quality issues identified]

### Phase 3: Data Cleaning
[What was cleaned, how you handled missing data/outliers, impact on dataset size]

### Phase 4: Data Wrangling
[Datetime parsing, temporal features extracted, any merging/reshaping done]

### Phase 5: Feature Engineering
[Derived features created, rolling windows calculated, aggregations performed]

### Phase 6: Pattern Analysis
[Trends identified, seasonal patterns, correlations found]

### Phase 7: Modeling Preparation
[Train/test split approach, features selected, preprocessing steps]

### Phase 8: Modeling
[Models trained, performance metrics, feature importance]

### Phase 9: Results
[Final visualizations, summary of key findings]

## Visualizations

![Figure 1: Time Series of Temperature](output/q1_temperature_timeseries.png)
*Figure 1: Daily temperature readings show clear seasonal patterns with higher values in summer months...*

![Figure 2: Model Performance Comparison](output/q7_model_comparison.png)
*Figure 2: Random Forest outperforms Linear Regression with R² of 0.XX...*

[Include at least 5 figures total with captions]

## Model Results

| Model | R² | RMSE | MAE |
|-------|----|----|----|
| Linear Regression | 0.XX | X.XX | X.XX |
| Random Forest | 0.XX | X.XX | X.XX |

[Interpretation of results: what do these metrics mean? Which model performs best and why?]

## Time Series Patterns

[Detailed discussion of:
- Trends over time (increasing/decreasing/stable)
- Seasonal patterns (daily, weekly, monthly cycles)
- Temporal relationships between variables
- Any anomalies or interesting temporal features]

## Limitations & Next Steps

[What could be improved:
- Data quality issues that couldn't be fully addressed
- Model limitations
- Additional features that could be created
- Additional analysis that would be valuable
- How results could be validated or extended]
```

## Getting Help

If you encounter technical issues:
1. Check `HINTS.md` for common problems and solutions
2. Review Lecture 11 notebooks for similar operations
3. Check error messages carefully - they often tell you what's wrong
4. Consult the textbook or online documentation
5. Contact instructor and EAs for assistance (clarification and nudges)

**See `HINTS.md` for detailed troubleshooting guide covering:**
- Data loading issues
- Datetime parsing problems
- Missing data handling
- Feature engineering challenges
- Train/test split issues
- Modeling problems
- Visualization issues
- Output file format problems

Good luck!

