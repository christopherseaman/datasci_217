# Statistical & Machine Learning Methods Assignment

## Submission

Submit a Jupyter notebook named `modeling.ipynb` containing:
1. Clear markdown cells explaining your approach
2. Well-commented code cells showing your analysis
3. Visualizations of results where appropriate
4. Interpretations of your findings

Your notebook should demonstrate proper data handling, statistical analysis, and machine learning practices as covered in the lectures.

## Data Source

The data is generated using `generate_health_data.py`, which creates three tab-separated (TSV) datasets:

1. `patient_baseline.csv`: Patient demographics and initial measurements
   - `patient_id`: Unique identifier
   - `age`: Age in years
   - `sex`: Binary (0=male, 1=female)
   - `bmi`: Body mass index
   - `smoking`: Binary smoking status
   - `diabetes`: Binary diabetes status
   - `bp_systolic`: Systolic blood pressure
   - `cholesterol`: Total cholesterol
   - `heart_rate`: Heart rate

2. `patient_longitudinal.csv`: Blood pressure and heart rate measurements over time
   - `patient_id`: Unique identifier
   - `visit_date`: Date of measurement (YYYY-MM-DD)
   - `bp_systolic`: Systolic blood pressure
   - `heart_rate`: Heart rate
   - `adverse_event`: Binary indicator of cardiovascular event
   - Plus baseline characteristics

3. `patient_treatment.csv`: Treatment assignments and outcomes
   - All baseline characteristics
   - `treatment`: Binary indicator of receiving treatment
   - `adherence`: Percentage of prescribed medication taken
   - `outcome`: Binary indicator of reaching treatment goal

## Time Series Tasks

1. Convert the longitudinal blood pressure measurements into a proper time series:
   - Create a pandas Series with DatetimeIndex using `visit_date`
   - Handle any missing or duplicate measurements appropriately
   - Tips: 
     - Use `pd.to_datetime()` with format='%Y-%m-%d' for reliable parsing
     - Handle duplicates with `duplicated()` and appropriate aggregation
     - Consider timezone handling with `tz_localize()` if needed
     - Use `interpolate()` or `fillna()` for missing values

2. Analyze blood pressure trends:
   - Resample the data to monthly frequency using mean aggregation
   - Calculate 3-month moving averages to smooth out short-term fluctuations
   - Visualize both the original and smoothed trends
   - Tips:
     - Use `resample('ME').mean()` for month-end frequency
     - Add `min_periods` to `rolling()` to handle edge cases
     - Consider `interpolate()` method for gaps
     - Use `ewm()` for exponential weighted alternatives

## Statistical Modeling Tasks

1. Analyze factors affecting baseline blood pressure:
   - Use statsmodels OLS to predict `bp_systolic`
   - Include `age`, `bmi`, `smoking`, and `diabetes` as predictors
   - Interpret the coefficients and their p-values
   - Assess model fit using R-squared and diagnostic plots
   - Tips:
     - Create feature matrix `X` with predictors and add constant term using `sm.add_constant()`
     - Use `sm.OLS(y, X).fit()` to fit the model
     - Use `summary()` to examine p-values and confidence intervals
     - Plot residuals vs fitted values and Q-Q plot
     - Consider robust standard errors with `HC3` covariance type

2. Model treatment effectiveness:
   - Fit a GLM with binomial family to predict treatment success
   - Use baseline characteristics and `adherence` as predictors
   - Report odds ratios and their confidence intervals
   - Assess model fit using deviance and diagnostic plots
   - Tips:
     - Create feature matrix `X` with predictors and add constant term
     - Use `sm.GLM(y, X, family=sm.families.Binomial()).fit()`
     - Get odds ratios with `np.exp(params)`
     - Check residual deviance vs null deviance
     - Use `influence()` to detect influential observations

## Machine Learning Tasks (stretch goal)

1. Build a prediction pipeline:
   - Create features from baseline characteristics
   - Standardize numeric features using `StandardScaler`
   - Train a logistic regression model to predict treatment outcomes
   - Include regularization to prevent overfitting
   - Tips:
     ```python
     from sklearn.pipeline import make_pipeline
     from sklearn.preprocessing import StandardScaler
     from sklearn.linear_model import LogisticRegression
     
     # Create pipeline with specific components
     pipeline = make_pipeline(
         StandardScaler(),
         LogisticRegression(
             penalty='l2',
             solver='lbfgs',
             max_iter=1000,
             class_weight='balanced'
         )
     )
     ```
     - Use `ColumnTransformer` for mixed numeric/categorical features
     - Consider `SelectKBest` or `RFE` for feature selection
     - Try different regularization strengths with `C` parameter
     - Use `Pipeline` to prevent data leakage

2. Validate model performance:
   - Split data into 70% training and 30% test sets
   - Implement 5-fold cross-validation on the training set
   - Report accuracy, precision, recall, and ROC AUC
   - Generate confusion matrix and ROC curve
   - Tips:
     ```python
     from sklearn.model_selection import (
         train_test_split, 
         StratifiedKFold,
         cross_validate
     )
     from sklearn.metrics import (
         classification_report,
         RocCurveDisplay,
         confusion_matrix
     )
     
     # Stratified split for imbalanced data
     X_train, X_test, y_train, y_test = train_test_split(
         X, y, 
         test_size=0.3, 
         stratify=y,
         random_state=42
     )
     
     # Cross-validation with multiple metrics
     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
     scores = cross_validate(
         pipeline,
         X_train, y_train,
         cv=cv,
         scoring=['accuracy', 'precision', 'recall', 'roc_auc']
     )
     ```
     - Use `StratifiedKFold` for imbalanced datasets
     - Consider precision-recall curve for imbalanced data
     - Plot learning curves to diagnose bias/variance
     - Use `cross_validate` for multiple metrics at once
