# Final Exam: Chicago Beach Weather Sensors Analysis

**Total: 100 points**  
**Format: Take-home exam**  
**Time: Recommended 4-6 hours**

## Overview

This final exam tests your ability to complete a full data science project workflow using the **Chicago Beach Weather Sensors Dataset**. You'll follow the same 9-phase workflow demonstrated in Lecture 11, but with a different dataset and focused questions.

**Assessment Breakdown:**
- **70% Automated Grading** - Code execution and output validation
- **30% Writeup Evaluation** - Written report with findings, visualizations, and model results

## Dataset: Chicago Beach Weather Sensors

**Source:** [data.gov - Chicago Beach Weather Sensors](https://catalog.data.gov/dataset?tags=sensors)

Real-time weather sensor readings from Lake Michigan beaches including:
- Temperature measurements (air and water)
- Wind speed and direction
- Water conditions
- Air quality measurements
- Sensor metadata

**Key Challenges:**
- Irregular sampling intervals (sensors report at different frequencies)
- Sensor dropouts (missing data periods)
- Multiple sensor streams to merge
- Time series analysis required throughout

### Downloading the Dataset

The Chicago Beach Weather Sensors dataset is available from data.gov. You can:

1. **Search data.gov:** Go to https://catalog.data.gov/dataset?tags=sensors and search for "Chicago Beach Weather"
2. **Direct download:** The dataset is typically available as CSV or JSON format
3. **API access:** Some versions may be available via API

**Note:** If you have trouble accessing the exact dataset, you may use any publicly available beach/weather sensor dataset with similar characteristics (multiple sensors, time series data, irregular sampling). The key requirements are:
- Multiple sensor readings over time
- Datetime timestamps
- Some missing data (sensor dropouts)
- Multiple measurement types (temperature, wind, etc.)

Place the downloaded dataset in the `data/` directory as `beach_sensors.csv`.

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
├── assignment.md                # Notebook source (jupytext)
├── assignment.ipynb             # Your work goes here
├── requirements.txt             # Python dependencies
├── data/                        # Input data (download from data.gov)
│   └── beach_sensors.csv       # Main sensor data (you download this)
├── output/                      # Your saved results (create this)
│   ├── q1_exploration.csv      # Q1 outputs
│   ├── q2_cleaned_data.csv     # Q2 outputs
│   ├── q3_features.csv         # Q3 outputs
│   ├── q4_patterns.csv         # Q4 outputs
│   ├── q5_model_results.csv    # Q5 outputs
│   └── ... (see questions for full list)
└── .github/
    └── test/
        └── test_assignment.py   # Auto-grading tests
```

## Setup

1. **Create virtual environment:**
   ```bash
   cd 11/assignment
   uv venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Download dataset:**
   - Download the Chicago Beach Weather Sensors dataset from data.gov
   - Place it in the `data/` directory as `beach_sensors.csv`
   - See "Downloading the Dataset" section above for details

4. **Start working:**
   ```bash
   jupytext --to notebook assignment.md  # If starting from markdown
   jupyter notebook assignment.ipynb
   ```

## Questions Overview

The exam follows the 9-phase data science workflow:

**Phase 1-2: Setup & Exploration (Q1)** - 10 points
- Load data and perform initial inspection
- Create basic visualizations
- Identify data quality issues

**Phase 3: Data Cleaning (Q2)** - 15 points
- Handle missing data
- Detect and handle outliers
- Validate data types
- Remove duplicates

**Phase 4: Data Wrangling (Q3)** - 15 points
- Merge multiple sensor streams
- Parse datetime columns
- Set datetime index
- Extract time-based features

**Phase 5: Feature Engineering (Q4)** - 15 points
- Create derived features
- Perform time-based aggregations
- Calculate rolling windows
- Create categorical features

**Phase 6: Pattern Analysis (Q5)** - 10 points
- Identify trends over time
- Analyze seasonal patterns
- Create correlation analysis
- Advanced visualizations

**Phase 7: Modeling Preparation (Q6)** - 5 points
- Temporal train/test split
- Feature selection
- Handle categorical variables

**Phase 8: Modeling (Q7)** - 15 points
- Train multiple models
- Evaluate performance
- Compare models
- Extract feature importance

**Phase 9: Results (Q8)** - 5 points
- Generate final visualizations
- Create summary tables
- Document key findings

**Writeup (Q9)** - 30 points
- Written report with all findings

## Auto-Grading (70 points)

Run tests locally:
```bash
pytest -q .github/test/test_assignment.py -v
```

GitHub Classroom will run the same tests on push.

**What's tested:**
- Code execution (no errors)
- Output file existence
- Data format validation (columns, dtypes, value ranges)
- Statistical calculations (means, counts, aggregations)
- Model outputs (predictions, metrics)
- Visualization file creation

## Writeup Evaluation (30 points)

Your written report will be evaluated on:

**Documentation Quality (10 points)**
- Clear explanation of process
- Rationale for key decisions
- Professional presentation

**Visualizations & Tables (10 points)**
- Time series visualizations show trends/patterns
- Tables formatted appropriately
- Visualizations follow best practices
- Model results presented clearly

**Interpretation & Insights (10 points)**
- Key findings from EDA summarized
- Time series patterns identified and explained
- Model performance interpreted correctly
- Limitations acknowledged
- Conclusions supported by evidence

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

## Getting Help

If you encounter technical issues:
1. Check error messages carefully
2. Review Lecture 11 notebooks for similar operations
3. Consult pandas documentation
4. Contact instructor for clarification (not solutions)

Good luck!

