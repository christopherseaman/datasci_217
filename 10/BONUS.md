---
notion:
  title_line: "# DLC: Advanced Modeling Topics"
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-8135-b60c-cf029cc707a7"
  url: "https://app.notion.com/p/3d2d9fdd1a1a8135b60ccf029cc707a7"
---

# DLC: Advanced Modeling Topics

# A Shared Example Cohort

Most snippets on this page model one synthetic cohort: 3,000 discharged patients from five clinics, with a 30-day readmission flag. The columns follow a causal chain, so the models have real structure to find:

- **Age, BMI, and weekly activity minutes** set systolic blood pressure (SBP), with a curved age effect, an age-by-BMI interaction, and a clinic offset.
- **Age and BMI** set hemoglobin A1c.
- **SBP, A1c, and age** set readmission risk, with a jump above an A1c of 7%.

Run this cell once; later snippets reuse `cohort`, `features`, and the train/test split.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(217)
n = 3000
clinics = ['Mission', 'Sunset', 'Richmond', 'Bayview', 'Marina']
clinic = rng.choice(clinics, n)
age = rng.integers(25, 90, n)
active_min = rng.gamma(2.0, 60, n).round()  # weekly minutes of exercise
bmi = (rng.normal(29, 4, n) - 0.01 * active_min).round(1)
clinic_offset = pd.Series([4, 0, -2, 6, -3], index=clinics)[clinic].to_numpy()
sbp = (118 + 0.5 * (age - 50) + 0.008 * (age - 50) ** 2 + 0.8 * (bmi - 28)
       + 0.03 * (age - 50) * (bmi - 28) - 0.03 * active_min
       + clinic_offset + rng.normal(0, 10, n)).round()
a1c = (5.2 + 0.015 * (age - 50) + 0.08 * (bmi - 28) + rng.gamma(1.5, 0.4, n)).round(1)
risk = -2.0 + 0.03 * (sbp - 130) + 0.3 * (a1c - 6) + 1.0 * (a1c > 7) + 0.02 * (age - 60)
readmit_30d = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)

cohort = pd.DataFrame({'clinic': clinic, 'age': age, 'active_min': active_min,
                       'bmi': bmi, 'sbp': sbp, 'a1c': a1c, 'readmit_30d': readmit_30d})
features = ['age', 'active_min', 'bmi', 'sbp', 'a1c']
X_train, X_test, y_train, y_test = train_test_split(
    cohort[features], cohort['readmit_30d'], test_size=0.25,
    stratify=cohort['readmit_30d'], random_state=42)
print(round(cohort['readmit_30d'].mean(), 3))  # 0.133
```

# More scikit-learn Tools

The lecture's main path uses linear models, random forests, and gradient boosting. Once that workflow feels routine, benchmark other candidates whose assumptions fit the task, always against the same baseline, split, and metric:

- Classification: `SVC`, a support vector classifier that looks for the widest possible boundary between classes, beside the lecture's `LogisticRegression`.
- Unsupervised work: `KMeans` for clustering patients into care-need groups and `PCA` for compressing dozens of correlated labs into a few components. Neither uses a target column.
- Selection: `GridSearchCV` for hyperparameter tuning within the training data (examples under Hyperparameter Tuning Strategies below).
- Independent rows: `cv=5` uses `StratifiedKFold` for classifiers with binary or multiclass targets, and `KFold` otherwise. Each fold takes a turn as the scored block; neither protects time order like `TimeSeriesSplit`.
- Patient groups: `GroupKFold(n_splits=5)` and `GroupShuffleSplit` keep each patient's rows in one part; pass `groups=df['patient_id']` to `split(X, y, groups=...)` or `cross_val_score(..., groups=...)`. Use them to evaluate on entirely new patients.
- Delayed targets: `TimeSeriesSplit(gap=n)` leaves `n` rows out between each training block and the block it scores. Choose a gap large enough that every retained label was available at that fold's first prediction cutoff; a row count is not a fixed time delay with irregular visits.

_Let validation evidence decide, not a favorite algorithm. Blue steel is a style, not a model-selection rule._

# Other Boosting Libraries

Beyond `XGBoost`, two other gradient-boosting libraries are common. Neither is part of Lecture 10's recorded environment; add them to the demo folder's environment with `uv add lightgbm catboost` (Lecture 03), or run `%pip install lightgbm catboost` in Colab, before trying them.

## `LightGBM`

- Designed for efficient training and memory use
- A candidate when scale or training speed is an important constraint, such as millions of claims lines

## `CatBoost`

- Provides native mechanisms for categorical features
- A candidate when the table contains important categorical variables, such as clinic, payer, or discharge disposition

## The Boosting Family Tree

```
Gradient Boosting
├── XGBoost (widely used general implementation)
├── LightGBM (efficiency-oriented implementation)
└── CatBoost (native categorical-feature support)
```

_Benchmark them under the same split, measure, and budget. Blue steel, magnum, and le tigre are all amazing, just slightly different, so test them on your data._

# Hyperparameter Tuning Strategies

## Grid Search and Random Search

**Grid search** tries every listed combination of settings; **random search** samples combinations from specified ranges. Each compares cross-validation scores within training data, keeping the final test out of tuning. A larger search costs more fits and gives more chances to select a lucky validation result.

### Reference Card: Grid and Random Search Tools

- `GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')`: Fit every combination in `param_grid`; read `best_params_` and `best_score_`.
- `RandomizedSearchCV(model, param_distributions, n_iter=20, random_state=...)`: Fit `n_iter` sampled combinations; distributions such as `scipy.stats.randint(2, 12)` sample integers.
- `cross_val_score(model, X, y, cv=5, scoring=...)`: Score one fixed setting across folds.

### Code Snippet: Tuning a Readmission Forest

```python
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5, None],
    'min_samples_leaf': [1, 10, 50],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                    cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)  # {'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 100}
print(round(grid.best_score_, 3))  # 0.74

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    {'max_depth': randint(2, 12), 'min_samples_leaf': randint(1, 100)},
    n_iter=20, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
print(search.best_params_, round(search.best_score_, 3))  # {'max_depth': 3, 'min_samples_leaf': 4} 0.74
```

The grid fit 18 settings and the random search 20; both land at a cross-validated AUC of 0.74. Shallow trees win because the cohort's risk is a smooth function of three columns.

## Bayesian Optimization

**Bayesian optimization** uses scores from earlier trials to choose promising settings for the next trial. Here it minimizes negative cross-validation AUC, so a smaller objective means a better score; the test rows stay unused.

This optional example requires `scikit-optimize`, which is not part of Lecture 10's recorded core environment. Add it to the demo folder's environment with `uv add scikit-optimize` (Lecture 03), or run `%pip install scikit-optimize` in Colab, before running the example.

### Reference Card: Bayesian Optimization Tools

- `gp_minimize(objective, space, n_calls=20, random_state=...)`: Minimize `objective` with a Gaussian process; `result.x` holds the best settings and `result.fun` the best objective.
- `Integer(low, high, name=...)`, `Real(low, high, prior='log-uniform', name=...)`, `Categorical([...])`: Define the search space.
- `@use_named_args(space)`: Pass each trial's settings to the objective as keyword arguments.

### Code Snippet: Bayesian Search for XGBoost Settings

```python
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

space = [
    Integer(50, 300, name='n_estimators'),
    Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
    Integer(2, 6, name='max_depth'),
]

@use_named_args(space)
def objective(**params):
    model = XGBClassifier(**params, random_state=42)
    return -cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

result = gp_minimize(objective, space, n_calls=20, random_state=42)
print(result.x[0], round(result.x[1], 3), result.x[2])  # 199 0.013 2
print(round(-result.fun, 3))  # 0.74
```

Twenty guided trials reach the same 0.74 as the forest grid. The ceiling comes from the signal in five columns, not from the search method.

# Model Interpretability and Explainability

## SHAP Values

**SHAP values** allocate a model's prediction among its input features relative to a reference prediction. A positive value raises the explained output and a negative one lowers it; the output scale depends on the model and explainer. These explain fitted predictions, not causal effects, and correlated features complicate the allocation.

This optional example requires SHAP, which is not part of Lecture 10's recorded core environment. Add it to the demo folder's environment with `uv add shap` (Lecture 03), or run `%pip install shap` in Colab, before running the example.

### Reference Card: SHAP Tools

- `shap.TreeExplainer(model)`: Create an explainer for a tree-based model.
- `explainer(X)`: Return SHAP values with `.values`, `.base_values`, and feature names.
- `shap.plots.beeswarm(shap_values)`: Show each feature's effect across patients, colored by the feature's value.
- `shap.plots.bar(shap_values)`: Rank features by mean absolute SHAP value.

### Code Snippet: Explaining a Readmission Model

```python
import shap
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)
print(pd.Series(np.abs(shap_values.values).mean(axis=0), index=features)
      .sort_values(ascending=False).round(3))
```

```text
sbp           0.697
a1c           0.166
active_min    0.165
age           0.148
bmi           0.128
dtype: float32
```

SBP ranks first because it carries both its own effect and much of age's (correlation 0.69). The generator gives activity no direct path to readmission: its 0.165 is borrowed through SBP and BMI, a reminder that SHAP explains the model, not the cause.

## Partial Dependence Plots

**Partial dependence** varies one or two features while averaging predictions over the other columns. It shows the fitted response shape, not what changing a patient's feature would cause. Strongly correlated columns can create unrealistic combinations, such as a BMI of 40 paired with a weight of 50 kg when both are inputs.

### Reference Card: Partial Dependence Tools

- `PartialDependenceDisplay.from_estimator(model, X, features, grid_resolution=20)`: Plot one curve per feature name and one heatmap per `(feature, feature)` pair.
- Partial dependence needs float columns; scikit-learn 1.9 raises `ValueError` for integer columns such as `age`, so cast them with `.astype(float)` first.

### Code Snippet: Plotting Readmission Risk Against A1c and SBP

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

model = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
model.fit(X_train, y_train)
PartialDependenceDisplay.from_estimator(
    model, X_train, ['a1c', 'sbp', ('bmi', 'sbp')], grid_resolution=20)
```

Expected output: two line plots and one heatmap. Average predicted risk stays near 0.12 for A1c up to 6.3% and climbs to 0.18 by 7%, where the generator adds its jump. The SBP curve rises from 0.08 to 0.27 between 100 and 145 mmHg. The default grid spans the 5th to 95th percentiles, so the A1c axis stops at 7%.

# Advanced Statistical Modeling

Choose an inferential model when the question requires interpretable parameters, uncertainty, or hypothesis tests and its design and model assumptions are defensible. Inference quantifies associations under assumptions; prediction estimates performance on new data. Neither alone establishes causation.

## Generalized Linear Models (GLMs)

Linear regression suits a numeric outcome that scatters evenly around the fitted line. **Generalized linear models** keep the same formula interface but change how the outcome is modeled:

- Logistic regression for binary outcomes, such as 30-day readmission
- Poisson or negative binomial regression for counts, such as ED visits per patient-year, with an `offset` for follow-up time
- Gamma regression for positive, right-skewed values, such as length of stay or claim cost
- Use when: You need statistical inference for outcomes that are not normal

## Mixed Effects Models

**Mixed effects** combine shared slopes with group-specific variation, such as different baseline SBP across clinics. The example gives each clinic a **random intercept**, allowing rows in the same clinic to share an offset instead of assuming all rows are independent.

### Reference Card: Mixed Effects Model Tools

- `smf.mixedlm(formula, data, groups=data['clinic'])`: Build a linear mixed model with a random intercept per group.
- `result.fe_params`: Fixed-effect slopes shared by every clinic.
- `result.random_effects`: Each group's estimated offset from the overall intercept.

### Code Snippet: Clinic Offsets in SBP

```python
import statsmodels.formula.api as smf

result = smf.mixedlm('sbp ~ age + bmi + active_min', data=cohort,
                     groups=cohort['clinic']).fit()
print(result.fe_params.round(2))
print(pd.Series({clinic: effect.iloc[0]
                 for clinic, effect in result.random_effects.items()}).round(1))
```

```text
Intercept     62.71
age            0.61
bmi            1.02
active_min    -0.03
dtype: float64
Bayview     5.1
Marina     -4.2
Mission     3.0
Richmond   -2.9
Sunset     -1.0
dtype: float64
```

The generator's clinic offsets, centered, are 5, -4, 3, -3, and -1 mmHg; the model recovers each within 0.3.

## Generalized Additive Models (GAMs)

A **generalized additive model** adds smooth curves for individual features instead of requiring every relationship to be a straight line. The example lets SBP bend with age and with activity; interactions require additional terms such as `te(0, 1)`.

This optional example requires `pygam`, which is not part of Lecture 10's recorded core environment. Add it to the demo folder's environment with `uv add pygam` (Lecture 03), or run `%pip install pygam` in Colab, before running the example.

### Reference Card: GAM Tools

- `LinearGAM(s(0) + s(1)).fit(X, y)`: Fit one smooth term per column of a numeric outcome; `LogisticGAM` fits a binary one.
- `gam.generate_X_grid(term=i)`: Build a grid that varies feature `i`.
- `gam.partial_dependence(term=i, X=XX, width=0.95)`: Return the term's curve and its 95% interval.

### Code Snippet: Smooth Age and Activity Effects on SBP

```python
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

gam = LinearGAM(s(0) + s(1)).fit(cohort[['age', 'active_min']].to_numpy(), cohort['sbp'])

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for i, (ax, name) in enumerate(zip(axes, ['age', 'active_min'])):
    XX = gam.generate_X_grid(term=i)
    curve, interval = gam.partial_dependence(term=i, X=XX, width=0.95)
    ax.plot(XX[:, i], curve)
    ax.fill_between(XX[:, i], interval[:, 0], interval[:, 1], alpha=0.3)
    ax.set_xlabel(name)
axes[0].set_ylabel('SBP contribution (mmHg)')
plt.show()
```

Expected output: the age curve rises about 40 mmHg from age 25 to 89 and steepens after 50, matching the generator's squared age term. The activity curve falls about 20 mmHg from 0 to 450 minutes a week, and its interval widens where few patients exercise that much.

# Advanced Deep Learning

## Transfer Learning

**Transfer learning** reuses features learned on a large source dataset. Here VGG16 supplies frozen image features, and a small new classifier learns to separate normal chest X-rays from pneumonia; ImageNet features are a starting point, not evidence of medical-image accuracy.

### Reference Card: Transfer Learning Tools

- `VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`: Load the pretrained convolutional layers without the ImageNet classifier.
- `base_model.trainable = False`: Freeze the pretrained layers so only new layers learn.
- `preprocess_input(images)`: Convert 0 to 255 RGB pixels to VGG16's expected channel order and centering.

### Code Snippet: A Pneumonia Classifier on Frozen VGG16 Features

Only the added classifier learns; **fine-tuning** would also update pretrained weights. The random arrays below stand in for X-ray batches, so the snippet demonstrates the API, not useful classification. VGG16 expects RGB pixels on a 0 to 255 scale passed through its own `preprocess_input`, which changes channel order and subtracts the ImageNet channel means; a grayscale X-ray is repeated across three channels.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models

rng = np.random.default_rng(42)
X_xray = preprocess_input(rng.uniform(0, 255, size=(20, 224, 224, 3)).astype('float32'))
y_pneumonia = rng.integers(0, 2, size=20)
X_val = preprocess_input(rng.uniform(0, 255, size=(6, 224, 224, 3)).astype('float32'))
y_val = rng.integers(0, 2, size=6)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_xray, y_pneumonia, epochs=10, validation_data=(X_val, y_val))
print(sum(w.numpy().size for w in model.trainable_weights))  # 32897
```

The model holds 14,747,585 weights; only the 32,897 in the new head learn.

## Sequence Attention with a Documented Keras Layer

**Self-attention** lets each position in a sequence combine information from other positions. Several **heads** learn different combinations. The example reads 24 hourly rows of 8 vitals and labs per ICU stay and averages the attended hours into one deterioration probability; ordered tasks also need information about position, and forecasting needs a mask that blocks future hours.

### Reference Card: Keras Attention Layer

- `keras.layers.MultiHeadAttention(num_heads, key_dim, dropout)`: Built-in self- or cross-attention; call it as `layer(query, value)`, and pass the same tensor twice for self-attention.

### Code Snippet: Attention over 24 Hours of Vitals

```python
from tensorflow import keras

inputs = keras.Input(shape=(24, 8))  # (stays, hours, vitals and labs)
x = keras.layers.Dense(32)(inputs)
attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.1)(x, x)
x = keras.layers.LayerNormalization()(x + attention)
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')
print(model.output_shape)  # (None, 1)
```

# Other Deep-Learning Frameworks

The lecture uses TensorFlow/Keras for its worked example. That is a teaching choice, not a universal ranking.

## `PyTorch`

`PyTorch` provides an eager, Python-oriented interface used in research and production. It is not part of Lecture 10's recorded environment; add it to the demo folder's environment with `uv add torch` (Lecture 03), or run `%pip install torch` in Colab, before running the example.

On Linux and WSL, plain `uv add torch` also installs the CUDA GPU packages, a multi-GB download. For the CPU-only build, add these lines to `pyproject.toml` before running `uv add torch`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

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

### Code Snippet: Scoring New Patients in Evaluation Mode

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(5, 16), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(16, 1), nn.Sigmoid(),
)
X_new = torch.tensor(X_test.head(4).to_numpy(), dtype=torch.float32)  # 4 patients, 5 features

model.eval()                 # turn Dropout off for scoring
with torch.no_grad():        # skip gradient tracking while predicting
    risk = model(X_new)
print(risk.shape)            # torch.Size([4, 1])
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

**Stacking** trains a final model on predictions from other models. The final model must learn from predictions made when each row was held out of the base model's fit; `StackingClassifier(cv=5)` constructs these cross-validation predictions within training data.

### Reference Card: Stacking Tools

- `StackingClassifier(estimators=[(name, model), ...], final_estimator=..., cv=5)`: Fit base models, then fit the final model on their out-of-fold predictions.
- `stacker.final_estimator_.coef_`: The weight a logistic final model gives each base model.

### Code Snippet: Stacking a Forest and an SVM

```python
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

stacker = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(min_samples_leaf=20, random_state=42)),
        ('svm', make_pipeline(StandardScaler(), SVC())),
    ],
    final_estimator=LogisticRegression(),
    cv=5,
)
stacker.fit(X_train, y_train)
print(stacker.final_estimator_.coef_.round(2))  # [[4.95 0.72]]
print(round(roc_auc_score(y_test, stacker.predict_proba(X_test)[:, 1]), 3))  # 0.78
```

The final model leans on the forest (4.95) and gives the SVM's scores little weight (0.72).

## Blending

**Blending** averages class probabilities with weights fixed in advance. The models' probability columns must represent the same classes in the same order. If choosing the weights from scores, use validation rows and reserve the test for the final combination.

### Reference Card: Blending Approach

- Fit each model separately, take `predict_proba(X)[:, 1]` from each, and combine with weights that sum to 1.

### Code Snippet: Blending Three Readmission Models

```python
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

models = {
    'rf': RandomForestClassifier(min_samples_leaf=20, random_state=42),
    'gb': HistGradientBoostingClassifier(max_depth=3, random_state=42),
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
}
weights = {'rf': 0.4, 'gb': 0.4, 'lr': 0.2}
risk = {name: m.fit(X_train, y_train).predict_proba(X_test)[:, 1] for name, m in models.items()}
risk['blend'] = sum(weights[name] * risk[name] for name in weights)
print(pd.Series({name: roc_auc_score(y_test, p) for name, p in risk.items()}).round(3))
```

```text
rf       0.783
gb       0.749
lr       0.788
blend    0.782
dtype: float64
```

The blend (0.782) trails plain logistic regression (0.788), because the cohort's risk is close to logistic. An ensemble earns its extra parts only when it beats the best single model on held-out rows.

# Time Series Modeling

When each observation depends on the ones just before it, `statsmodels` offers dedicated time-series tools:

- ARIMA models for forecasting, such as daily ED arrivals or weekly flu-like illness visits
- Seasonal decomposition into trend, weekly cycle, and remainder
- Use when: You have temporal dependencies in your data

## Weekly Persistence on an Hourly Grid

On a complete elapsed-hour grid, `shift(168)` gives the value 168 hours earlier; across a daylight-saving change this can differ from the same local clock hour last week.

## Prophet for Time Series

**Prophet** combines a trend with seasonal and optional holiday effects. Its `ds` column holds times and `y` the observations. Judge any forecast against persistence on later, held-out dates.

This optional example requires `prophet`, which is not part of Lecture 10's recorded core environment. Add it to the demo folder's environment with `uv add prophet` (Lecture 03), or run `%pip install prophet` in Colab, before running the example.

### Reference Card: Prophet Tools

- `Prophet(daily_seasonality=False)`: Create a model; weekly and yearly seasonality turn on when the data spans enough time.
- `model.fit(df)`: Fit on a table with `ds` and `y` columns.
- `model.make_future_dataframe(periods=28)`: Extend `ds` by 28 days; `model.predict(future)` returns `yhat` with `yhat_lower` and `yhat_upper`.
- `model.plot(forecast)`: Plot history, forecast, and interval.

### Code Snippet: Forecasting Daily ED Arrivals

```python
from prophet import Prophet

rng = np.random.default_rng(42)
ds = pd.date_range('2023-01-01', '2024-12-31', freq='D')
weekday_effect = np.array([22, 12, 6, 4, 2, -8, -10])[ds.dayofweek]  # Monday busiest
winter_effect = 15 * np.cos(2 * np.pi * (ds.dayofyear - 15) / 365.25)
arrivals = pd.DataFrame({'ds': ds, 'y': rng.poisson(
    180 + 0.02 * np.arange(len(ds)) + weekday_effect + winter_effect)})

train, test = arrivals.iloc[:-28], arrivals.iloc[-28:]
model = Prophet(daily_seasonality=False).fit(train)
forecast = model.predict(test[['ds']])

persistence = np.tile(train['y'].iloc[-7:].to_numpy(), 4)  # repeat the last observed week
print(f"Prophet MAE: {np.mean(np.abs(forecast['yhat'].to_numpy() - test['y'].to_numpy())):.1f}")
print(f"Persistence MAE: {np.mean(np.abs(persistence - test['y'].to_numpy())):.1f}")
```

```text
Prophet MAE: 12.3
Persistence MAE: 13.8
```

Over the last 28 days, Prophet misses by 12.3 arrivals a day and repeating last week misses by 13.8. A 1.5-arrival gain may not justify a new dependency for a staffing forecast.

# Production Deployment Considerations

## Model Serialization

**Serialization** saves a fitted model so another session can use it. Keep its preprocessing, ordered feature names, and environment versions with it: the same weights with differently scaled or reordered inputs give different predictions.

### Reference Card: Model Serialization Tools

- `joblib.dump(obj, 'readmit_model.joblib')`: Save a fitted scikit-learn object, or a dict holding one with its metadata.
- `joblib.load('readmit_model.joblib')`: Load it back.
- `model.save('model.keras')`: Save a Keras model in the native `.keras` format.
- `model.export('saved_model')`: Export a TensorFlow SavedModel for serving (Keras 3).

### Code Snippet: Packaging a Readmission Pipeline

```python
import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
joblib.dump({
    'pipeline': pipeline,
    'features': features,
    'trained_through': '2025-06-30',
    'sklearn_version': sklearn.__version__,
}, 'readmit_model.joblib')

package = joblib.load('readmit_model.joblib')
new_patient = pd.DataFrame([{'a1c': 7.8, 'sbp': 152, 'bmi': 31.0, 'active_min': 30, 'age': 71}])
print(package['pipeline'].predict_proba(new_patient[package['features']])[:, 1].round(2))  # [0.53]
```

The new patient's columns arrive in a different order; indexing with the saved `features` list restores the training order before scoring.

Pickle/joblib files can execute arbitrary code while loading. Load them only from a trusted, integrity-checked source in a compatible environment; never treat an uploaded or untrusted pickle as data. For Keras, use `keras.models.load_model('model.keras')` for the native format; use the exported SavedModel with a serving/runtime tool rather than passing it to `load_model`.

## Model Versioning

**Versioning** records which data, settings, environment, and fitted model produced a result. An experiment log makes comparisons reproducible; a logged model is not automatically approved for deployment.

This optional example requires `mlflow`, which is not part of Lecture 10's recorded core environment. Add it to the demo folder's environment with `uv add mlflow` (Lecture 03), or run `%pip install mlflow` in Colab, before running the example.

### Reference Card: MLflow Tracking

- `mlflow.set_experiment('readmit_30d')`: Group runs under one named experiment.
- `mlflow.start_run(run_name=...)`: Open a run; `log_params(dict)` and `log_metric(name, value)` record settings and scores.
- `mlflow.sklearn.log_model(model, name='model', input_example=...)`: Store the fitted model with an example of its input columns.
- `mlflow.search_runs()`: Return every run as a `DataFrame` for comparison.

### Code Snippet: Logging a Readmission Run with MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

params = {'n_estimators': 300, 'min_samples_leaf': 20}
mlflow.set_experiment('readmit_30d')
with mlflow.start_run(run_name='rf_leaf20'):
    mlflow.log_params(params)
    model = RandomForestClassifier(**params, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    mlflow.log_metric('cv_auc', cv_auc)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, name='model', input_example=X_train.head(3),
                             skops_trusted_types=['sklearn.tree._tree.Tree'])

print(mlflow.search_runs()[['tags.mlflow.runName', 'params.min_samples_leaf', 'metrics.cv_auc']].round(3))
```

```text
  tags.mlflow.runName params.min_samples_leaf  metrics.cv_auc
0           rf_leaf20                      20           0.729
```

The run logs a cross-validated AUC, so the test rows stay untouched for the final comparison.

# Advanced Feature Engineering

## Automated Feature Engineering

**Feature engineering** turns raw records into model inputs. A patient-level model needs one row per patient, so encounter-level rows are aggregated: counts, means, maxima, and distinct types. Each aggregate must use only encounters before the **prediction cutoff**; a later encounter in a feature leaks the future.

### Code Snippet: Encounter Features Before a Cutoff in pandas

```python
patients = pd.DataFrame({'patient_id': ['P01', 'P02', 'P03'], 'birth_year': [1948, 1971, 1990]})
encounters = pd.DataFrame({
    'encounter_id': range(7),
    'patient_id': ['P01', 'P01', 'P01', 'P02', 'P02', 'P03', 'P03'],
    'admit_time': pd.to_datetime(['2025-01-04', '2025-02-11', '2025-03-20', '2025-01-15',
                                  '2025-03-02', '2025-02-08', '2025-04-01']),
    'encounter_type': ['ED', 'inpatient', 'ED', 'clinic', 'ED', 'clinic', 'clinic'],
    'los_days': [0.3, 4.0, 0.4, 0.1, 0.5, 0.1, 0.1],
})

history = encounters[encounters['admit_time'] < pd.Timestamp('2025-03-01')]
print(history.groupby('patient_id').agg(
    n_encounters=('encounter_id', 'count'),
    mean_los=('los_days', 'mean'),
    max_los=('los_days', 'max'),
    n_types=('encounter_type', 'nunique'),
))
```

```text
            n_encounters  mean_los  max_los  n_types
patient_id                                          
P01                    2      2.15      4.0        2
P02                    1      0.10      0.1        1
P03                    1      0.10      0.1        1
```

The three encounters after March 1 stay out of every feature.

### Deep Feature Synthesis with Featuretools

**Featuretools** builds these aggregates automatically from declared tables and relationships, a method called **Deep Feature Synthesis**, and applies a per-patient `cutoff_time`. Featuretools 1.31.0 fails under pandas 3.0.5 (Woodwork cannot initialize its table types), so the snippet below runs only in a separate environment with `pandas==2.3.3`, `featuretools==1.31.0`, and `setuptools<81`.

#### Reference Card: Featuretools Functions

- `EntitySet.add_dataframe(dataframe_name, dataframe, index, time_index=...)`: Register a table; `time_index` marks when each row became known.
- `EntitySet.add_relationship(parent, parent_column, child, child_column)`: Link patients to their encounters.
- `ft.dfs(entityset, target_dataframe_name, cutoff_time=..., agg_primitives=[...])`: Return a feature matrix with one row per target row.

#### Code Snippet: The Same Features with `ft.dfs`

```python
import featuretools as ft

es = ft.EntitySet(id='ehr')
es = es.add_dataframe(dataframe_name='patients', dataframe=patients, index='patient_id')
es = es.add_dataframe(dataframe_name='encounters', dataframe=encounters,
                      index='encounter_id', time_index='admit_time',
                      logical_types={'encounter_type': 'Categorical'})
es = es.add_relationship('patients', 'patient_id', 'encounters', 'patient_id')

cutoffs = pd.DataFrame({'patient_id': ['P01', 'P02', 'P03'],
                        'time': pd.to_datetime(['2025-03-01'] * 3)})
feature_matrix, feature_defs = ft.dfs(
    entityset=es, target_dataframe_name='patients', cutoff_time=cutoffs,
    agg_primitives=['count', 'mean', 'max', 'num_unique'], trans_primitives=[])
print(feature_matrix.to_string())
```

```text
            birth_year  COUNT(encounters)  MAX(encounters.los_days)  MEAN(encounters.los_days)  NUM_UNIQUE(encounters.encounter_type)
patient_id                                                                                                                           
P01               1948                  2                       4.0                       2.15                                      2
P02               1971                  1                       0.1                       0.10                                      1
P03               1990                  1                       0.1                       0.10                                      1
```

The counts match the pandas version, because `cutoff_time` applies the same March 1 boundary.

## Polynomial and Interaction Features

**Polynomial features** add powers such as `age²`; **interaction features** add products such as `age * bmi`, letting BMI's contribution to SBP depend on age. Validate this extra flexibility against the plain linear model.

### Reference Card: Polynomial Feature Tools

- `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)`: Add pairwise products only.
- `PolynomialFeatures(degree=2, include_bias=False)`: Add products and squares.
- `poly.get_feature_names_out()`: Name the generated columns.

### Code Snippet: Does SBP Need an Age-by-BMI Term?

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

X_vitals = cohort[['age', 'bmi']]
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit(X_vitals)
print(poly.get_feature_names_out())  # ['age' 'bmi' 'age bmi']

candidates = {
    'linear': LinearRegression(),
    'interaction': make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), LinearRegression()),
    'degree 2': make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
}
for name, model in candidates.items():
    print(name, cross_val_score(model, X_vitals, cohort['sbp'], cv=5, scoring='r2').mean().round(3))
```

```text
linear 0.551
interaction 0.571
degree 2 0.59
```

Each added term raises cross-validated R² because the generator contains both the interaction and the squared age term.

# Model Monitoring and Maintenance

## Drift Detection

**Data drift** is a change in input distributions; **concept drift** is a change in the relationship between inputs and outcomes. A monthly feed monitor compares each new batch of model inputs against the training data, one column at a time, with the two-sample **Kolmogorov-Smirnov (KS) test**. It cannot establish concept drift or a loss of predictive performance; small repeated samples and testing many columns also produce false alarms, so the example uses a strict `alpha=0.01`.

### Reference Card: Drift Detection Approach

- `scipy.stats.ks_2samp(reference, batch)`: Return the KS statistic (largest gap between the two cumulative distributions) and a p-value.
- Monitor input distributions, label rates, and live performance together; a flag starts an investigation.

### Code Snippet: Catching a Lab Assay Change

```python
from scipy import stats

def ks_drift(reference, batch, columns, alpha=0.01):
    rows = [(col, *stats.ks_2samp(reference[col], batch[col])) for col in columns]
    report = pd.DataFrame(rows, columns=['column', 'ks_stat', 'p_value']).set_index('column')
    report['flag'] = report['p_value'] < alpha
    return report.round(3)

march = cohort.sample(400, random_state=1)
april = cohort.sample(400, random_state=2).assign(a1c=lambda d: (d['a1c'] + 0.4).round(1))  # new assay reads 0.4 high

for month, batch in {'March': march, 'April': april}.items():
    print(month)
    print(ks_drift(X_train, batch, ['age', 'bmi', 'sbp', 'a1c']))
```

```text
March
        ks_stat  p_value   flag
column                         
age       0.023    0.991  False
bmi       0.046    0.467  False
sbp       0.024    0.984  False
a1c       0.038    0.679  False
April
        ks_stat  p_value   flag
column                         
age       0.048    0.412  False
bmi       0.045    0.490  False
sbp       0.037    0.715  False
a1c       0.283    0.000   True
```

April flags only A1c (KS 0.283). The next step is a call to the lab about the assay, not an automatic retrain: retraining requires explicit review, leakage checks, and validation against the current model.

## Paired Model Comparison

**McNemar's test** compares two classifiers' mistakes on the same cases. It uses only the discordant cells, where one model is right and the other wrong. It needs both predictions for each case; a randomized A/B deployment instead assigns different cases to each model.

### Reference Card: Model Comparison Approach

- `pd.crosstab(a_correct, b_correct)`: Build the 2 by 2 table of agreement on the same cases.
- `statsmodels.stats.contingency_tables.mcnemar(table, exact=False, correction=True)`: Test whether the discordant cells differ; `exact=True` suits small counts.

### Code Snippet: Old Sepsis Alert Versus a New Model

```python
from statsmodels.stats.contingency_tables import mcnemar

rng = np.random.default_rng(42)
sepsis = rng.random(1000) < 0.08
old_alert = np.where(rng.random(1000) < 0.82, sepsis, ~sepsis)
new_model = np.where(rng.random(1000) < 0.87, sepsis, ~sepsis)

table = pd.crosstab(old_alert == sepsis, new_model == sepsis,
                    rownames=['old correct'], colnames=['new correct'])
print(table)
result = mcnemar(table, exact=False, correction=True)
print(f"Accuracy old {np.mean(old_alert == sepsis):.3f}, new {np.mean(new_model == sepsis):.3f}, p = {result.pvalue:.1e}")
```

```text
new correct  False  True 
old correct              
False           20    179
True           109    692
Accuracy old 0.801, new 0.871, p = 4.8e-05
```

The new model fixes 179 of the old alert's misses and adds 109 of its own. With 8% prevalence, also compare sensitivity on the 73 sepsis cases before switching, because accuracy mostly counts the negatives.
