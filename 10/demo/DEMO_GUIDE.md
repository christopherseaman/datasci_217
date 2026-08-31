# Demo Guide for Lecture 10: Modeling

This guide provides an overview of the three demos for Lecture 10.

## Demo 1: Statistical Modeling with statsmodels

**File:** `demo1_statistical_modeling.md`

**Learning Objectives:**
- Fit and interpret linear regression models using `statsmodels`
- Understand statistical inference (p-values, confidence intervals)
- Compare formula API vs array API
- Generate and analyze realistic datasets
- Visualize model results with Altair

**Key Topics:**
- Formula API (`smf.ols()`) vs Array API (`sm.OLS()`)
- Model summary and interpretation
- Coefficient significance testing
- Making predictions with mean confidence intervals and individual prediction intervals
- Model comparison

**Estimated Time:** 30-40 minutes

## Demo 2: Machine Learning with scikit-learn and XGBoost

**File:** `demo2_ml_boosting.md`

**Learning Objectives:**
- Master the scikit-learn fit/predict pattern
- Build and evaluate linear regression and random forest models
- Use XGBoost for gradient boosting
- Interpret model-specific and permutation feature importance cautiously
- Compare model performance
- Visualize results with Altair

**Key Topics:**
- Train/validation/test split with one final test report
- Linear regression with regularization (Ridge, Lasso)
- Random Forest for non-linear relationships
- XGBoost gradient boosting
- Model-specific importance and validation-set permutation importance with MAE
- Early stopping
- Model comparison

**Estimated Time:** 40-50 minutes

## Demo 3: Deep Learning with TensorFlow/Keras

**File:** `demo3_deep_learning.md`

**Learning Objectives:**
- Build neural networks using TensorFlow/Keras
- Understand the Sequential API
- Train models and monitor progress
- Evaluate model performance
- Visualize training history
- Compare deep learning with traditional ML

**Key Topics:**
- Data preprocessing and scaling
- Building neural networks with Sequential API
- Compiling models (optimizer, loss, metrics)
- Training and monitoring
- Regularization (dropout, L2)
- Architecture experimentation
- When to use deep learning vs traditional ML

**Estimated Time:** 40-50 minutes

## Running the Demos

### Prerequisites

1. Create a virtual environment using the tested course candidate, Python 3.12.13:
```bash
# Use the course's tested Python and create the local environment
uv venv --python 3.12.13 .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

**Important:** The exact pins in `requirements.txt` are the supported demo matrix.
They were tested with Python 3.12.13; use a separate, explicitly tested matrix
before substituting another Python or package version.

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

This will install all packages including TensorFlow for Demo 3.

### Converting Markdown to Notebooks

The demos are written in markdown format (jupytext). Convert them to notebooks:

```bash
# Convert all demos
jupytext --to notebook demo1_statistical_modeling.md
jupytext --to notebook demo2_ml_boosting.md
jupytext --to notebook demo3_deep_learning.md
```

Or use jupytext in Jupyter to open them directly - they'll be treated as notebooks.

### Running the Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open the converted `.ipynb` files

3. Run all cells sequentially

## Tips for Instructors

- **Demo 1**: Emphasize the difference between statistical inference (statsmodels) and prediction (scikit-learn)
- **Demo 2**: Highlight the consistent scikit-learn API and when to use each model type
- **Demo 3**: Stress that deep learning isn't always better - show the comparison with traditional ML
- Demos 1–2 download California Housing (20,640 California census block groups from the 1990 census); Demo 3 uses scikit-learn's bundled Wine recognition dataset (178 samples)
- The demos build complexity gradually - don't skip ahead
- Encourage students to experiment with hyperparameters and see how results change

## Common Issues

- **Import errors**: Make sure all packages are installed in the correct environment
- **Jupytext not working**: Install with `pip install jupytext` or use `jupyter notebook` which should recognize `.md` files automatically
- **Altair plots not showing**: Make sure you're running in Jupyter, not plain Python
- **Dataset download failure**: Demos 1–2 fetch California Housing on first run; cache it or arrange network access before class
