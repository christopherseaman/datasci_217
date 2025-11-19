# Assignment 10: Modeling Fundamentals

## Load the Dataset

The assignment uses the **California Housing** dataset from scikit-learn. This is a real-world dataset from the 1990 US Census containing information about housing prices in California districts.

The dataset is automatically loaded in the assignment notebook using:
```python
from sklearn.datasets import fetch_california_housing
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame
```

**Note:** The dataset is loaded directly from scikit-learn in the assignment notebook. No separate data generation step is required.

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

### Question 1: Statistical Modeling with statsmodels

**What you'll do:**
- Load California Housing dataset
- Fit a linear regression model using `statsmodels` (predicting from `MedInc`, `AveBedrms`, `Population`)
- Extract and save model statistics (R-squared, observations, AIC)
- Make predictions and save to CSV

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
├── requirements.txt               # Python dependencies
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

### California Housing Dataset

| Column | Type | Description |
|--------|------|-------------|
| `MedInc` | float | Median income in block group |
| `HouseAge` | float | Median house age in block group |
| `AveRooms` | float | Average number of rooms per household |
| `AveBedrms` | float | Average number of bedrooms per household |
| `Population` | float | Block group population |
| `AveOccup` | float | Average number of household members |
| `Latitude` | float | Block group latitude |
| `Longitude` | float | Block group longitude |
| `house_value` | float | Target variable: median house value (in hundreds of thousands of dollars) |

## Submission Checklist

Before submitting, verify you've created:

- [ ] `output/q1_statistical_model.csv` - Statistical model predictions
- [ ] `output/q1_model_summary.txt` - Model summary statistics
- [ ] `output/q2_ml_predictions.csv` - ML model predictions
- [ ] `output/q2_model_comparison.txt` - Model comparison results
- [ ] `output/q3_xgboost_model.csv` - XGBoost predictions
- [ ] `output/q3_feature_importance.txt` - Feature importance results

