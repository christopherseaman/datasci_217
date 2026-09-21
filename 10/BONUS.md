---
notion:
  title_line: "# DLC: Advanced Modeling Topics"
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-8135-b60c-cf029cc707a7"
  url: "https://app.notion.com/p/3d2d9fdd1a1a8135b60ccf029cc707a7"
---

# DLC: Advanced Modeling Topics

# More scikit-learn Tools

The lecture's main path uses linear models, random forests, and gradient boosting. Once that workflow feels routine, benchmark other candidates whose assumptions fit the task, always against the same baseline, split, and metric:

- Classification: `SVC`, a support vector classifier that looks for the widest possible boundary between classes, beside the lecture's `LogisticRegression`.
- Unsupervised work: `KMeans` for clustering and `PCA` for dimensionality reduction. Neither uses a target column.
- Selection: `cross_val_score` for cross-validation and `GridSearchCV` for hyperparameter tuning within the training data (examples under Hyperparameter Tuning Strategies below).

**Cross-validation** splits the training rows into k parts (folds), fits on k - 1 of them, scores on the fold left out, and repeats until every fold has been scored once. It stands in for a single validation set when rows are scarce, and it never touches the test set. For time-ordered rows, `TimeSeriesSplit` keeps every validation fold later than the rows it trains on.

*Let validation evidence—not a favorite algorithm—decide. Blue steel is a style, not a model-selection rule.*

# Other Boosting Libraries

Beyond `XGBoost`, two other gradient-boosting libraries are common. Neither is part of Lecture 10's recorded environment; install them in the active notebook environment with `%pip install lightgbm catboost` before trying them.

## `LightGBM`

- Designed for efficient training and memory use
- A candidate when scale or training speed is an important constraint

## `CatBoost`

- Provides native mechanisms for categorical features
- A candidate when the table contains important categorical variables

## The Boosting Family Tree

```
Gradient Boosting
├── XGBoost (widely used general implementation)
├── LightGBM (efficiency-oriented implementation)
└── CatBoost (native categorical-feature support)
```

*Benchmark them under the same split, measure, and budget. Blue steel, magnum, and le tigre are all amazing, just slightly different—so test them on your data.*

# Hyperparameter Tuning Strategies

## Grid Search and Random Search

### Reference Card: Grid and Random Search Tools

- `from sklearn.model_selection import GridSearchCV` - Exhaustive grid search
- `from sklearn.model_selection import RandomizedSearchCV` - Random search
- `from sklearn.model_selection import cross_val_score` - Cross-validation scoring

### Code Snippet: Tuning a Random Forest with Grid Search

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

## Bayesian Optimization

This optional example requires `scikit-optimize`, which is not part of Lecture 10's recorded core environment. Install it in the active notebook environment with `%pip install scikit-optimize` before running the example.

### Reference Card: Bayesian Optimization Tools

- `from skopt import gp_minimize` - Gaussian process optimization
- `from skopt.space import Real, Integer, Categorical` - Parameter spaces

### Code Snippet: Bayesian Optimization for XGBoost Hyperparameters

```python
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Model Interpretability and Explainability

## SHAP Values

This optional example requires SHAP, which is not part of Lecture 10's recorded core environment. Install it in the active notebook environment with `%pip install shap` before running the example.

### Reference Card: SHAP Tools

- `import shap` - SHAP library
- `shap.TreeExplainer(model)` - Create an explainer for a tree-based model
- `explainer(X)` - Return SHAP values with feature and observation metadata
- `shap.plots.beeswarm(shap_values)` - Show feature effects across observations
- `shap.plots.bar(shap_values)` - Summarize global feature importance

### Code Snippet: Explaining an XGBoost Model with SHAP

```python
import shap
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Summary plot
shap.plots.beeswarm(shap_values)

# Feature importance plot
shap.plots.bar(shap_values)
```

## Partial Dependence Plots

### Reference Card: Partial Dependence Tools

- `from sklearn.inspection import PartialDependenceDisplay` - Partial dependence
- `PartialDependenceDisplay.from_estimator(model, X, features)` - Create plots

### Code Snippet: Plotting Partial Dependence

```python
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
import numpy as np

rng = np.random.default_rng(42)
X_train = rng.normal(size=(200, 2))
y_train = X_train[:, 0] ** 2 + X_train[:, 1] + rng.normal(scale=0.5, size=200)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Partial dependence plots
features = [0, 1, (0, 1)]  # Individual and interaction
PartialDependenceDisplay.from_estimator(
    model, X_train, features, 
    grid_resolution=20
)
```

# Advanced Statistical Modeling

Choose an inferential model when the question requires interpretable parameters, uncertainty, or hypothesis tests and its design and model assumptions are defensible. Inference quantifies associations under assumptions; prediction estimates performance on new data. Neither alone establishes causation.

## Generalized Linear Models (GLMs)

Linear regression suits a numeric outcome that scatters evenly around the fitted line. **Generalized linear models** keep the same formula interface but change how the outcome is modeled:

- Logistic regression for binary outcomes
- Poisson regression for count data
- Other exponential family distributions
- Use when: You need statistical inference for non-normal data

## Mixed Effects Models

### Reference Card: Mixed Effects Model Tools

- `from statsmodels.regression.mixed_linear_model import MixedLM` - Mixed linear models
- `MixedLM.from_formula(formula, data, groups)` - Create model

### Code Snippet: Fitting a Mixed Effects Model

```python
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# 10 groups (for example, clinics) with 8 rows each
rng = np.random.default_rng(42)
n_groups, n_per_group = 10, 8
group = np.repeat(np.arange(n_groups), n_per_group)
x1 = rng.normal(size=n_groups * n_per_group)
x2 = rng.normal(size=n_groups * n_per_group)
group_effect = rng.normal(scale=2.0, size=n_groups)[group]
y = 5 + 1.5 * x1 - 0.5 * x2 + group_effect + rng.normal(scale=1.0, size=n_groups * n_per_group)
df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'group': group})

# Mixed effects model
model = MixedLM.from_formula('y ~ x1 + x2', data=df, groups=df['group'])
result = model.fit()
print(result.summary())
```

## Generalized Additive Models (GAMs)

This optional example requires `pygam`, which is not part of Lecture 10's recorded core environment. Install it in the active notebook environment with `%pip install pygam` before running the example.

### Reference Card: GAM Tools

- `from pygam import LinearGAM` - Generalized additive models
- `gam = LinearGAM().fit(X, y)` - Fit GAM

### Code Snippet: Fitting and Plotting a GAM

```python
from pygam import LinearGAM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 2))
y = X[:, 0] ** 2 + X[:, 1] + rng.normal(scale=0.5, size=200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Advanced Deep Learning

## Transfer Learning

### Reference Card: Transfer Learning Tools

- `from tensorflow.keras.applications import VGG16` - Pre-trained models
- `model = VGG16(weights='imagenet', include_top=False)` - Load pre-trained
- `model.trainable = False` - Freeze layers

### Code Snippet: Fine-Tuning a Pretrained VGG16 Model

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import numpy as np

# Small stand-in image batches (real projects load actual images)
rng = np.random.default_rng(42)
X_train = rng.random(size=(20, 224, 224, 3)).astype('float32')
y_train = np.eye(10)[rng.integers(0, 10, size=20)]
X_val = rng.random(size=(5, 224, 224, 3)).astype('float32')
y_val = np.eye(10)[rng.integers(0, 10, size=5)]

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

## Sequence attention with a documented Keras layer

### Reference Card: Keras Attention Layer

- `keras.layers.MultiHeadAttention` - Current built-in self/cross-attention layer

### Code Snippet: Self-Attention with MultiHeadAttention

```python
import tensorflow as tf
from tensorflow import keras

# A batch of sequences: (batch, timesteps, embedding_dim)
inputs = keras.Input(shape=(20, 64))
attention = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16, dropout=0.1
)(inputs, inputs)  # self-attention; output shape is (batch, 20, 64)
x = keras.layers.LayerNormalization()(inputs + attention)
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
```

# Other Deep-Learning Frameworks

The lecture uses TensorFlow/Keras for its worked example—a teaching choice, not a universal ranking.

## `PyTorch`

`PyTorch` provides an eager, Python-oriented interface used in research and production. It is not part of Lecture 10's recorded environment; install it in the active notebook environment with `%pip install torch` before running the example.

- **PyTorch:** Eager execution and a Python-oriented modeling ecosystem
- **TensorFlow/Keras:** High-level Keras APIs within TensorFlow's broader modeling and deployment ecosystem
- **Both:** Used for research and production; neither role belongs exclusively to one framework
- **Choice:** Depends on required libraries, deployment target, team expertise, maintenance, and measured performance

### Reference Card: PyTorch Essentials

| Function / method | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `nn.Sequential(...)` | Build a simple ordered neural-network stack. | Module |
| `torch.optim.Adam(model.parameters(), lr=...)` | Update model parameters using Adam. | Optimizer |
| `nn.BCELoss()` | Calculate binary cross-entropy for probability outputs. | Loss function |
| `model.train()` / `model.eval()` | Switch behavior for training or evaluation (for example, Dropout). | Module state change |

### Code Snippet: PyTorch Evaluation Mode

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 1), nn.Sigmoid(),
)
X_test = torch.randn(5, 10)  # 5 rows, 10 features

model.eval()                 # inference behavior for layers such as Dropout
with torch.no_grad():        # skip gradient tracking while predicting
    predictions = model(X_test)
print(predictions.shape)     # torch.Size([5, 1])
```

## Other Modern Frameworks

Other frameworks serve different computational styles. `JAX` combines NumPy-like arrays with automatic differentiation and JIT compilation; choose it or another specialized tool when its capabilities and dependency cost fit the task.

```
Deep Learning Frameworks
├── TensorFlow/Keras (high-level modeling and deployment ecosystem)
├── PyTorch (eager, Python-oriented modeling ecosystem)
└── JAX (array programming with transformations and JIT compilation)
```

# Model Ensembling

## Stacking

### Reference Card: Stacking Tools

- `from sklearn.ensemble import StackingClassifier` - Stacking ensemble
- `StackingClassifier(estimators, final_estimator)` - Create stacker

### Code Snippet: Stacking a Random Forest and SVM

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
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

## Blending

### Reference Card: Blending Approach

- Manual blending by training models separately and combining predictions

### Code Snippet: Blending Model Predictions

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'rf': RandomForestClassifier(random_state=42).fit(X_train, y_train),
    'gb': GradientBoostingClassifier(random_state=42).fit(X_train, y_train),
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

# Time Series Modeling

When each observation depends on the ones just before it, `statsmodels` offers dedicated time-series tools:

- ARIMA models for time series forecasting
- Seasonal decomposition
- Use when: You have temporal dependencies in your data

Lecture 09's bonus has worked examples of [decomposition](../09/BONUS.md#advanced-time-series-decomposition) and [ARIMA and exponential smoothing](../09/BONUS.md#time-series-forecasting).

## ARIMA Models

### Reference Card: ARIMA Tools

- `from statsmodels.tsa.arima.model import ARIMA` - ARIMA models
- `model = ARIMA(data, order=(p, d, q))` - Create ARIMA
- `result = model.fit()` - Fit model
- `result.forecast(steps)` - Forecast

### Code Snippet: Fitting and Forecasting an ARIMA Model

```python
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

# A random-walk-like series (for example, daily readings)
rng = np.random.default_rng(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
data = pd.Series(np.cumsum(rng.normal(0, 1, size=100)) + 50, index=dates)

# Create ARIMA model
model = ARIMA(data, order=(1, 1, 1))  # AR(1), I(1), MA(1)
result = model.fit()

# Summary
print(result.summary())

# Forecast
forecast = result.forecast(steps=10)
conf_int = result.get_forecast(steps=10).conf_int()
```

## Prophet for Time Series

This optional example requires `prophet`, which is not part of Lecture 10's recorded core environment. Install it in the active notebook environment with `%pip install prophet` before running the example.

### Reference Card: Prophet Tools

- `from prophet import Prophet` - Facebook Prophet
- `model = Prophet()` - Create model
- `model.fit(df)` - Fit model
- `model.predict(future)` - Make predictions

### Code Snippet: Forecasting with Prophet

```python
from prophet import Prophet
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
time_series_data = np.cumsum(rng.normal(0, 1, size=365)) + 50

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

# Production Deployment Considerations

## Model Serialization

### Reference Card: Model Serialization Tools

- `import joblib` - Joblib for scikit-learn models
- `joblib.dump(model, 'model.pkl')` - Save model
- `model = joblib.load('model.pkl')` - Load model
- `model.save('model.keras')` - Save a Keras model in the native `.keras` format
- `model.export('saved_model')` - Export a TensorFlow SavedModel for serving (Keras 3)

### Code Snippet: Saving and Loading a Model with joblib

```python
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X_train = rng.normal(size=(100, 3))
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
feature_names = ['x1', 'x2', 'x3']
scaler = StandardScaler().fit(X_train)
model = LogisticRegression().fit(scaler.transform(X_train), y_train)

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

Pickle/joblib files can execute arbitrary code while loading. Load them only from a trusted, integrity-checked source in a compatible environment; never treat an uploaded or untrusted pickle as data. For Keras, use `keras.models.load_model('model.keras')` for the native format; use the exported SavedModel with a serving/runtime tool rather than passing it to `load_model`.

## Model Versioning

This optional example requires `mlflow`, which is not part of Lecture 10's recorded core environment. Install it in the active notebook environment with `%pip install mlflow` before running the example.

### Reference Card: Model Versioning Approach

- Use MLflow or similar tools for model versioning
- Track model metadata, parameters, and performance

### Code Snippet: Logging a Model Run with MLflow

```python
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

# Advanced Feature Engineering

## Automated Feature Engineering

This optional example requires `featuretools`, which is not part of Lecture 10's recorded core environment. Install it in the active notebook environment with `%pip install featuretools` before running the example.

### Reference Card: Featuretools Functions

- `EntitySet.add_dataframe` - Register related tables
- `featuretools.dfs` - Generate features from an EntitySet

### Code Snippet: Generating Features with Deep Feature Synthesis

```python
import featuretools as ft
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
customer_df = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'signup_date': pd.date_range('2024-01-01', periods=3),
})
transaction_df = pd.DataFrame({
    'transaction_id': range(6),
    'customer_id': [1, 1, 2, 2, 3, 3],
    'transaction_date': pd.date_range('2024-02-01', periods=6),
    'amount': rng.normal(50, 10, size=6).round(2),
})

# Create entity set
es = ft.EntitySet(id='data')

# Add entities
es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customer_df,
    index='customer_id'
)

es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transaction_df,
    index='transaction_id',
    time_index='transaction_date'
)

# Define relationships
es = es.add_relationship(
    parent_dataframe_name='customers',
    parent_column_name='customer_id',
    child_dataframe_name='transactions',
    child_column_name='customer_id',
)

# Generate features
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2
)
```

## Polynomial and Interaction Features

### Reference Card: Polynomial Feature Tools

- `from sklearn.preprocessing import PolynomialFeatures` - Polynomial features
- `poly = PolynomialFeatures(degree=2, interaction_only=True)` - Create transformer

### Code Snippet: Adding Polynomial Features to a Pipeline

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 2))
y = 3 + 2 * X[:, 0] * X[:, 1] + rng.normal(scale=0.5, size=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Model Monitoring and Maintenance

## Drift Detection

### Reference Card: Drift Detection Approach

- Monitor model performance over time
- Detect data drift and concept drift

### Code Snippet: Detecting Drift with a KS Test

```python
import numpy as np
import pandas as pd
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

# Reference window and two later batches (the second shifted, to simulate drift)
rng = np.random.default_rng(42)
reference_data = pd.DataFrame({
    'age': rng.normal(50, 10, size=200),
    'bmi': rng.normal(27, 4, size=200),
})
data_batches = [
    pd.DataFrame({
        'age': rng.normal(50, 10, size=200),
        'bmi': rng.normal(27, 4, size=200),
    }),
    pd.DataFrame({
        'age': rng.normal(58, 10, size=200),  # shifted: simulates drift
        'bmi': rng.normal(27, 4, size=200),
    }),
]

# Monitor over time
for batch in data_batches:
    drift = detect_drift(reference_data, batch)
    if any(d['drift'] for d in drift.values()):
        print("Drift signal: investigate data quality, affected segments, and live performance.")
        # Retraining requires explicit review, leakage checks, and validation
        # against the current model before any deployment decision.
```

## A/B Testing for Models

### Reference Card: Model Comparison Approach

- Compare model performance in production
- Statistical significance testing

### Code Snippet: Comparing Two Models with McNemar's Test

```python
import numpy as np
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

rng = np.random.default_rng(42)
true_labels = rng.integers(0, 2, size=100)
model_a_predictions = np.where(rng.random(100) < 0.85, true_labels, 1 - true_labels)
model_b_predictions = np.where(rng.random(100) < 0.70, true_labels, 1 - true_labels)

print(compare_models(model_a_predictions, model_b_predictions, true_labels))
```

These advanced topics will help you build production-ready models, understand model behavior, and maintain models over time in real-world applications.
