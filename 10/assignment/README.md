# Assignment 10: Modeling Fundamentals

**Deliverable:**

- Pass all auto-grading tests by completing the required tasks in `assignment.ipynb`/`assignment.md`

## Environment Setup

### Create Virtual Environment

Create a virtual environment for this assignment:

```bash
# Using uv (recommended)
uv venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

**Important:** Make sure your Jupyter notebook is using the same virtual environment as your kernel. Select the `.venv` kernel in Jupyter's kernel menu.

## Generate the Dataset (Provided)

Run the data generator notebook to create your dataset:

```bash
# Convert markdown to notebook (if using jupytext)
jupytext --to notebook data_generator.md
jupyter notebook data_generator.ipynb
```

Run all cells to create the CSV files in `data/`:
- `data/patient_data.csv` (patient characteristics and outcomes)

## Complete the Three Questions

Open `assignment.ipynb` and work through the three questions. The notebook provides:

- **Step-by-step instructions** with clear TODO items
- **Helpful hints** for each operation
- **Sample code** to guide your work
- **Validation checks** to ensure your outputs are correct

**Prerequisites:** This assignment uses statistical modeling, machine learning, and gradient boosting from Lecture 10.

**How to use the scaffold notebook:**
1. Read each cell carefully - they contain detailed instructions
2. Complete the TODO items by replacing `None` with your code
3. Run each cell to see your progress
4. Use the hints provided in comments
5. Check the submission checklist at the end

### Auto-Grading (Required)

Run all required cells in `assignment.ipynb` so that the following files are created in `output/`:

- `q1_statistical_model.csv`, `q1_model_summary.txt`
- `q2_ml_predictions.csv`, `q2_model_comparison.txt`
- `q3_xgboost_model.csv`, `q3_feature_importance.txt`

Run tests locally:

```bash
pytest -q 10/assignment/.github/test/test_assignment.py
```

GitHub Classroom will run the same tests on push.

### Question 1: Statistical Modeling with statsmodels

**What you'll do:**
- Load patient data
- Fit a linear regression model using `statsmodels`
- Extract and interpret model coefficients and p-values
- Make predictions with confidence intervals

**Skills:** `statsmodels`, statistical inference, model interpretation

**Output:** `output/q1_statistical_model.csv`, `output/q1_model_summary.txt`

### Question 2: Machine Learning with scikit-learn

**What you'll do:**
- Split data into training and test sets
- Fit linear regression and random forest models
- Compare model performance
- Make predictions on test data

**Skills:** `scikit-learn`, train/test split, model evaluation, Random Forest

**Output:** `output/q2_ml_predictions.csv`, `output/q2_model_comparison.txt`

### Question 3: Gradient Boosting with XGBoost

**What you'll do:**
- Fit an XGBoost model
- Extract feature importance
- Make predictions
- Compare with previous models

**Skills:** `XGBoost`, gradient boosting, feature importance

**Output:** `output/q3_xgboost_model.csv`, `output/q3_feature_importance.txt`

## Assignment Structure

```
10/assignment/
├── README.md                      # This file - assignment instructions
├── assignment.md                  # Notebook source (for jupytext)
├── assignment.ipynb              # Completed notebook (you work here)
├── data_generator.md              # Data generator source (markdown)
├── data_generator.ipynb          # Generated data generator notebook
├── requirements.txt               # Python dependencies
├── data/                          # Generated datasets
│   └── patient_data.csv          # Patient characteristics and outcomes
├── output/                        # Your saved results (create this directory)
│   ├── q1_statistical_model.csv  # Q1 model predictions
│   ├── q1_model_summary.txt      # Q1 model summary
│   ├── q2_ml_predictions.csv     # Q2 ML predictions
│   ├── q2_model_comparison.txt   # Q2 model comparison
│   ├── q3_xgboost_model.csv      # Q3 XGBoost predictions
│   └── q3_feature_importance.txt # Q3 feature importance
└── .github/
    └── test/
        └── test_assignment.py    # Auto-grading tests
```

## Dataset Schema

### `data/patient_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Unique patient identifier |
| `age` | int | Patient age in years |
| `bmi` | float | Body mass index |
| `chronic_conditions` | int | Number of chronic conditions |
| `medication_count` | int | Number of medications |
| `hospital_stay_days` | int | Length of hospital stay in days |
| `readmission_risk` | float | Target variable: readmission risk score (0-100) |

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_statistical_model.csv` - Statistical model predictions
- [ ] `output/q1_model_summary.txt` - Model summary statistics
- [ ] `output/q2_ml_predictions.csv` - ML model predictions
- [ ] `output/q2_model_comparison.txt` - Model comparison results
- [ ] `output/q3_xgboost_model.csv` - XGBoost predictions
- [ ] `output/q3_feature_importance.txt` - Feature importance results

## Tips

- **Question 1**: Use `statsmodels.formula.api.ols()` for the formula API. Remember to call `.fit()` and use `.summary()` for the full output.
- **Question 2**: Use `train_test_split()` from `sklearn.model_selection`. Set `random_state=42` for reproducibility.
- **Question 3**: XGBoost models have `.feature_importances_` attribute similar to Random Forest.
- **All questions**: Make sure to save your outputs to the `output/` directory with the exact filenames specified.

