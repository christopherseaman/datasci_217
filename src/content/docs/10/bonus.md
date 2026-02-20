---
title: "Bonus: Advanced Modeling"
---


## Hyperparameter Tuning Strategies

### Grid Search and Random Search

**Reference:**

- `from sklearn.model_selection import GridSearchCV` - Exhaustive grid search
- `from sklearn.model_selection import RandomizedSearchCV` - Random search
- `from sklearn.model_selection import cross_val_score` - Cross-validation scoring

**Example:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search
model = RandomForestClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

### Bayesian Optimization

**Reference:**

- `from skopt import gp_minimize` - Gaussian process optimization
- `from skopt.space import Real, Integer, Categorical` - Parameter spaces

**Example:**

```python
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Define search space
space = [
    Integer(50, 300, name='n_estimators'),
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 10, name='max_depth')
]

# Objective function
@use_named_args(space=space)
def objective(**params):
    model = XGBClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return -scores.mean()  # Minimize negative score

# Optimize
result = gp_minimize(objective, space, n_calls=20, random_state=42)
print(f"Best parameters: {result.x}")
```

## Model Interpretability and Explainability

### SHAP Values

**Reference:**

- `import shap` - SHAP library
- `shap.Explainer(model)` - Create explainer
- `explainer.shap_values(X)` - Calculate SHAP values
- `shap.summary_plot(shap_values, X)` - Summary plot

**Example:**

```python
import shap
import xgboost as xgb

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Feature importance plot
shap.plots.bar(shap_values)
```

### Partial Dependence Plots

**Reference:**

- `from sklearn.inspection import PartialDependenceDisplay` - Partial dependence
- `PartialDependenceDisplay.from_estimator(model, X, features)` - Create plots

**Example:**

```python
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Partial dependence plots
features = [0, 1, (0, 1)]  # Individual and interaction
PartialDependenceDisplay.from_estimator(
    model, X_train, features, 
    grid_resolution=20
)
```

## Advanced Statistical Modeling

### Mixed Effects Models

**Reference:**

- `from statsmodels.regression.mixed_linear_model import MixedLM` - Mixed linear models
- `MixedLM.from_formula(formula, data, groups)` - Create model

**Example:**

```python
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.formula.api as smf

# Mixed effects model
model = MixedLM.from_formula('y ~ x1 + x2', data=df, groups=df['group'])
result = model.fit()
print(result.summary())
```

### Generalized Additive Models (GAMs)

**Reference:**

- `from pygam import LinearGAM` - Generalized additive models
- `gam = LinearGAM().fit(X, y)` - Fit GAM

**Example:**

```python
from pygam import LinearGAM
import numpy as np

# Create GAM
gam = LinearGAM().fit(X_train, y_train)

# Predictions
predictions = gam.predict(X_test)

# Plot partial dependence
for i in range(X_train.shape[1]):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    plt.plot(XX[:, i], pdep)
    plt.fill_between(XX[:, i], confi[:, 0], confi[:, 1], alpha=0.3)
```

## Advanced Deep Learning

### Transfer Learning

**Reference:**

- `from tensorflow.keras.applications import VGG16` - Pre-trained models
- `model = VGG16(weights='imagenet', include_top=False)` - Load pre-trained
- `model.trainable = False` - Freeze layers

**Example:**

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Add custom classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

### Custom Layers and Models

**Reference:**

- `from tensorflow.keras import layers, Model` - Custom model building
- `class CustomLayer(layers.Layer)` - Custom layer class
- `class CustomModel(Model)` - Custom model class

**Example:**

```python
from tensorflow.keras import layers, Model

# Custom layer
class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
    
    def call(self, inputs):
        attention_weights = tf.nn.softmax(tf.matmul(inputs, self.W))
        return tf.matmul(attention_weights, inputs)

# Custom model
class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.attention = AttentionLayer(64)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.attention(inputs)
        x = self.dense1(x)
        return self.dense2(x)
```

## Model Ensembling

### Stacking

**Reference:**

- `from sklearn.ensemble import StackingClassifier` - Stacking ensemble
- `StackingClassifier(estimators, final_estimator)` - Create stacker

**Example:**

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svm', SVC(probability=True))
]

# Meta-learner
meta_learner = LogisticRegression()

# Stacking ensemble
stacker = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)

stacker.fit(X_train, y_train)
predictions = stacker.predict(X_test)
```

### Blending

**Reference:**

- Manual blending by training models separately and combining predictions

**Example:**

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Train multiple models
models = {
    'rf': RandomForestClassifier().fit(X_train, y_train),
    'gb': GradientBoostingClassifier().fit(X_train, y_train),
    'lr': LogisticRegression().fit(X_train, y_train)
}

# Get predictions
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict_proba(X_test)

# Blend (weighted average)
weights = {'rf': 0.4, 'gb': 0.4, 'lr': 0.2}
blended = sum(weights[name] * predictions[name] for name in weights.keys())
final_predictions = np.argmax(blended, axis=1)
```

## Time Series Modeling

### ARIMA Models

**Reference:**

- `from statsmodels.tsa.arima.model import ARIMA` - ARIMA models
- `model = ARIMA(data, order=(p, d, q))` - Create ARIMA
- `result = model.fit()` - Fit model
- `result.forecast(steps)` - Forecast

**Example:**

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Create ARIMA model
model = ARIMA(data, order=(1, 1, 1))  # AR(1), I(1), MA(1)
result = model.fit()

# Summary
print(result.summary())

# Forecast
forecast = result.forecast(steps=10)
conf_int = result.get_forecast(steps=10).conf_int()
```

### Prophet for Time Series

**Reference:**

- `from prophet import Prophet` - Facebook Prophet
- `model = Prophet()` - Create model
- `model.fit(df)` - Fit model
- `model.predict(future)` - Make predictions

**Example:**

```python
from prophet import Prophet
import pandas as pd

# Prepare data (columns: ds, y)
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365),
    'y': time_series_data
})

# Create and fit model
model = Prophet()
model.fit(df)

# Create future dataframe
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
model.plot(forecast)
```

## Production Deployment Considerations

### Model Serialization

**Reference:**

- `import joblib` - Joblib for scikit-learn models
- `joblib.dump(model, 'model.pkl')` - Save model
- `model = joblib.load('model.pkl')` - Load model
- `tf.keras.models.save_model(model, 'path')` - Save TensorFlow model

**Example:**

```python
import joblib
import pickle

# Save scikit-learn model
joblib.dump(model, 'model.pkl')

# Save with metadata
model_package = {
    'model': model,
    'version': '1.0',
    'features': feature_names,
    'preprocessor': scaler
}
joblib.dump(model_package, 'model_package.pkl')

# Load
loaded = joblib.load('model_package.pkl')
model = loaded['model']
```

### Model Versioning

**Reference:**

- Use MLflow or similar tools for model versioning
- Track model metadata, parameters, and performance

**Example:**

```python
import mlflow
import mlflow.sklearn

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

## Advanced Feature Engineering

### Automated Feature Engineering

**Reference:**

- `from featuretools import dfs` - Automated feature engineering
- `feature_matrix, feature_defs = dfs(entities, relationships)` - Generate features

**Example:**

```python
import featuretools as ft

# Create entity set
es = ft.EntitySet(id='data')

# Add entities
es = es.entity_from_dataframe(
    entity_id='customers',
    dataframe=customer_df,
    index='customer_id'
)

es = es.entity_from_dataframe(
    entity_id='transactions',
    dataframe=transaction_df,
    index='transaction_id',
    time_index='transaction_date'
)

# Define relationships
es = es.add_relationship(
    ft.Relationship(es['customers']['customer_id'],
                   es['transactions']['customer_id'])
)

# Generate features
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='customers',
    max_depth=2
)
```

### Polynomial and Interaction Features

**Reference:**

- `from sklearn.preprocessing import PolynomialFeatures` - Polynomial features
- `poly = PolynomialFeatures(degree=2, interaction_only=True)` - Create transformer

**Example:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Use in pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)
```

## Model Monitoring and Maintenance

### Drift Detection

**Reference:**

- Monitor model performance over time
- Detect data drift and concept drift

**Example:**

```python
import numpy as np
from scipy import stats

def detect_drift(reference_data, new_data, threshold=0.05):
    """Detect statistical drift between reference and new data"""
    drift_detected = {}
    
    for col in reference_data.columns:
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(
            reference_data[col], 
            new_data[col]
        )
        
        drift_detected[col] = {
            'statistic': statistic,
            'p_value': p_value,
            'drift': p_value < threshold
        }
    
    return drift_detected

# Monitor over time
for batch in data_batches:
    drift = detect_drift(reference_data, batch)
    if any(d['drift'] for d in drift.values()):
        print("Drift detected! Retrain model.")
```

### A/B Testing for Models

**Reference:**

- Compare model performance in production
- Statistical significance testing

**Example:**

```python
from scipy import stats

def compare_models(model_a_predictions, model_b_predictions, true_labels):
    """Compare two models using statistical tests"""
    
    # Calculate accuracies
    accuracy_a = (model_a_predictions == true_labels).mean()
    accuracy_b = (model_b_predictions == true_labels).mean()
    
    # McNemar's test for paired comparisons
    from statsmodels.stats.contingency_tables import mcnemar
    
    # Create contingency table
    both_correct = ((model_a_predictions == true_labels) & 
                   (model_b_predictions == true_labels)).sum()
    a_correct_b_wrong = ((model_a_predictions == true_labels) & 
                         (model_b_predictions != true_labels)).sum()
    a_wrong_b_correct = ((model_a_predictions != true_labels) & 
                         (model_b_predictions == true_labels)).sum()
    both_wrong = ((model_a_predictions != true_labels) & 
                 (model_b_predictions != true_labels)).sum()
    
    table = [[both_correct, a_correct_b_wrong],
             [a_wrong_b_correct, both_wrong]]
    
    result = mcnemar(table, exact=False, correction=True)
    
    return {
        'accuracy_a': accuracy_a,
        'accuracy_b': accuracy_b,
        'p_value': result.pvalue,
        'significant': result.pvalue < 0.05
    }
```

These advanced topics will help you build production-ready models, understand model behavior, and maintain models over time in real-world applications.

