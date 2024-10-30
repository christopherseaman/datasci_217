---
marp: true
theme: default
paginate: true
---

# Statistical & Machine Learning Methods

- Time series
- `statsmodels`
- `scikit-learn`
- ML packages
  - Keras/TensorFlow
  - PyTorch

<!--

this is a speaker note

-->

---

# Time Series Analysis (1/8)

## Introduction to Time Series
- Definition and key components
  ```python
  # Time series data structure
  ts = pd.Series(data, index=pd.date_range('2023', periods=12, freq='M'))
  ```
- Types:
  - Fixed frequency (regular intervals)
  - Irregular frequency (variable intervals)
- Common applications: finance, economics, sensor data

---

# Time Series Analysis (2/8)
## pandas DateTime Objects
```python
# DatetimeIndex
dates = pd.date_range('2023-01-01', periods=5)

# Period vs Timestamp
ts = pd.Timestamp('2023-01-01')
period = pd.Period('2023-01', freq='M')

# Time zones
ts_ny = ts.tz_localize('America/New_York')
ts_utc = ts_ny.tz_convert('UTC')
```

---

# Time Series Analysis (3/8)

## Time Series Operations
```python
# Resampling
df.resample('M').mean()    # Downsample to monthly
df.resample('H').ffill()   # Upsample to hourly

# Rolling windows
df.rolling(window=7).mean()

# Shifting
df.shift(periods=1)        # Forward shift
df.shift(periods=-1)       # Backward shift
```

---

# Time Series Analysis (4/8)

## Frequency and Date Ranges
```python
# Common frequencies
pd.date_range('2023', freq='D', periods=365)   # Daily
pd.date_range('2023', freq='B', periods=252)   # Business days
pd.date_range('2023', freq='M', periods=12)    # Month end
pd.date_range('2023', freq='Q', periods=4)     # Quarter end

# Working with periods
pd.period_range('2023', freq='M', periods=12)
```

---

# Time Series Analysis (5/8)

## Moving Window Functions
```python
# Rolling statistics
df.rolling(window=30, min_periods=25).mean()
df.rolling(window=30, min_periods=25).std()

# Exponential weighted functions
df.ewm(span=30, min_periods=25).mean()

# Custom window functions
def custom_stat(x): return x.max() - x.min()
df.rolling(window=30).apply(custom_stat)
```

---

# Time Series Analysis (6/8)

## Time Series Decomposition
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose series into components
result = seasonal_decompose(
    df['value'],
    model='additive',  # or 'multiplicative'
    period=12
)

trend = result.trend
seasonal = result.seasonal
residual = result.resid
```

---

# Time Series Analysis (7/8)

## Time Series Visualization
```python
# Line plots with dates
df.plot(figsize=(12,6))

# Seasonal plots
df.groupby(df.index.month).mean().plot(kind='bar')

# Autocorrelation
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df)
```

---

# Time Series Analysis (8/8)

## Advanced Time Series Models
```python
from statsmodels.tsa.arima.model import ARIMA

# ARIMA model
model = ARIMA(data, order=(1,1,1))
results = model.fit()

# Forecasting
forecast = results.forecast(steps=5)
conf_int = results.get_forecast(steps=5).conf_int()
```

---

# statsmodels (1/9)

#FIXME add note about X as independent vars

## Linear Regression Basics
```python
import statsmodels.api as sm

# OLS implementation
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X)
results = model.fit()

# Key outputs
print(results.params)        # Coefficients
print(results.rsquared)     # R-squared
print(results.pvalues)      # P-values
```

---

# statsmodels (2/9)

## Model Diagnostics
```python
# R-squared and adjusted R-squared
print(results.rsquared)
print(results.rsquared_adj)

# P-values
print(results.pvalues)

# Residual analysis
residuals = results.resid
sm.stats.diagnostic.het_breuschpagan(residuals, X)
```

---

# statsmodels (3/9)

## Generalized Linear Models

```python
import statsmodels.api as sm

# GLM specification
model = sm.GLM(y, X, family=sm.families.Binomial())
results = model.fit()

# Link functions
sm.families.Gaussian(link=sm.families.links.log)
sm.families.Poisson(link=sm.families.links.log)
```

---

# statsmodels (4/9)
****
## Statistical Tests
```python
# T-test
from scipy import stats
t_stat, p_val = stats.ttest_ind(group1, group2)

# ANOVA
from statsmodels.stats.anova import anova_lm
anova_table = anova_lm(fitted_model, typ=2)

# Chi-square
chi2, p_val = stats.chi2_contingency(contingency_table)
```

---

# statsmodels (5/9)

## Regression Diagnostics
```python
# Influence statistics
influence = results.get_influence()
leverage = influence.hat_matrix_diag
cooks = influence.cooks_distance[0]

# Outlier detection
student_resid = results.outlier_test()

# Leverage plots
sm.graphics.influence_plot(results)
```

---

# statsmodels (6/9)

## Panel Data Analysis
```python
from statsmodels.regression.linear_model import PanelOLS

# Fixed effects
model_fe = PanelOLS(y, X, entity_effects=True)
fe_results = model_fe.fit()

# Random effects
model_re = PanelOLS(y, X, random_effects=True)
re_results = model_re.fit()
```

---

# statsmodels (7/9)

## Robust Regression
```python
# Robust linear model
model = sm.RLM(y, X)
results = model.fit()

# Robust standard errors
results = model.fit(cov_type='HC3')

# M-estimators
results = model.fit(method='huber')
```

---

# statsmodels (8/9)

## Model Output & Visualization
```python
# Summary tables
print(results.summary())

# Diagnostic plots
fig = sm.graphics.plot_regress_exog(results, 'x1')
fig = sm.graphics.plot_fit(results, 'x1')

# Results extraction
conf_int = results.conf_int()
predicted = results.predict(X_new)
```

---

# statsmodels (9/9)

## Advanced Applications
```python
# Model comparison
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Common pitfalls check
def check_assumptions(results):
    print("Normality:", stats.jarque_bera(results.resid)[1])
    print("Heteroskedasticity:", stats.het_breuschpagan(results.resid, X)[1])
    print("Autocorrelation:", stats.durbin_watson(results.resid))
```

---

# scikit-learn (1/9)

## Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encoding
enc = OneHotEncoder(sparse=False)
X_encoded = enc.fit_transform(X_cat)
```

---

# scikit-learn (2/9)

## Feature Selection
```python
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier

# SelectKBest
selector = SelectKBest(k=5)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
rfe = RFE(RandomForestClassifier(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```

---

# scikit-learn (3/9)

## Supervised Learning: Classification
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Models
lr = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC(kernel='rbf')

# Common interface
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

# scikit-learn (4/9)

## Supervised Learning: Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Models
lr = LinearRegression()
rf = RandomForestRegressor()
svr = SVR(kernel='rbf')

# Common interface
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

# scikit-learn (5/9)

## Unsupervised Learning
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Clustering
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

---

# scikit-learn (6/9)

## Model Selection
```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Grid search
params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(), params, cv=5)
grid.fit(X, y)
```

---

# scikit-learn (7/9)

## Model Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Classification metrics
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# Regression metrics
print(f"MSE: {mean_squared_error(y_true, y_pred)}")
print(f"R2: {r2_score(y_true, y_pred)}")
```

---

# scikit-learn (8/9)

## Pipeline Construction
```python
from sklearn.pipeline import Pipeline, FeatureUnion

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=5)),
    ('classifier', LogisticRegression())
])

# Feature union
union = FeatureUnion([
    ('numeric', numeric_pipeline),
    ('text', text_pipeline)
])
```

---

# scikit-learn (9/9)

## Model Persistence
```python
import joblib

# Save model
joblib.dump(model, 'model.joblib')

# Load model
model = joblib.load('model.joblib')

# Update model
model.partial_fit(X_new, y_new)
```

---

# Patsy (1/3)

## Formula Specification
```python
import patsy

# Basic syntax
y, X = patsy.dmatrices('y ~ x1 + x2', data)

# Transformations
'y ~ standardize(x1) + center(x2)'

# Interactions
'y ~ x1 + x2 + x1:x2'
```

---

# Patsy (2/3)

## Design Matrices
```python
# Creating matrices
y, X = patsy.dmatrices('y ~ x1 + C(category)', data)

# Categorical data
'y ~ C(category, Treatment)'
'y ~ C(category, Sum)'

# Missing data
y, X = patsy.dmatrices('y ~ x1', data, NA_action='drop')
```

---

# Patsy (3/3)

## Advanced Features
```python
# Custom transformations
def log_plus_1(x): return np.log(x + 1)
'y ~ log_plus_1(x1)'

# Splines
'y ~ bs(x, df=3)'

# Integration with statsmodels
model = sm.OLS.from_formula('y ~ x1 + x2', data)
```

---

# Deep Learning Frameworks Introduction (1/2)

## Framework Overview
```python
# Keras/TensorFlow
import tensorflow as tf
from tensorflow import keras

# PyTorch
import torch
import torch.nn as nn

# Core concepts comparison
keras_model = keras.Sequential()  # Layer-based
torch_model = nn.Module()        # Class-based
```

---

# Deep Learning Frameworks Introduction (2/2)

Hand-wavy explanation (many caveats apply)
- When to use Keras:
  - Quick prototyping
  - High-level APIs needed
  - Production with TensorFlow Serving

- When to use PyTorch:
  - Research/experimentation
  - Custom architectures
  - Dynamic computation graphs

Note: _most_ neural nets perform better with normalized data regardless of platform

---

# Keras Implementation (1/8)

## Layers & Activation Functions
```python
from tensorflow import keras

# Dense layers
model.add(keras.layers.Dense(64))

# Convolutional layers
model.add(keras.layers.Conv2D(32, (3, 3)))

# RNN layers
model.add(keras.layers.LSTM(64))

# Activation options
keras.activations.relu
keras.activations.sigmoid
keras.activations.tanh
```

---

# Keras Implementation (2/8)

## Building Models - Part 1
```python
# Sequential API
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Model lifecycle
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
```

---

# Keras Implementation (3/8)

## Building Models - Part 2
```python
# Functional API
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(64, activation='relu')(inputs)
outputs = keras.layers.Dense(10)(x)
model = keras.Model(inputs, outputs)

# Model subclassing
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(64)
        self.dense2 = keras.layers.Dense(10)
```

---

# Keras Implementation (4/8)

## Data Preprocessing
```python
# Image data generators
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Sequence generators
seq = keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=100
)

# Text preprocessing
vectorizer = keras.layers.TextVectorization(max_tokens=20000)
```

---

# Keras Implementation (5/8)

## Loss Functions & Optimizers
```python
# Common loss functions
keras.losses.BinaryCrossentropy()
keras.losses.CategoricalCrossentropy()
keras.losses.MeanSquaredError()

# Optimizer selection
keras.optimizers.Adam(learning_rate=0.001)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Learning rate scheduling
keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.9
)
```

---

# Keras Implementation (6/8)

## Model Training
```python
# Fit method
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=3),
    keras.callbacks.ModelCheckpoint('best_model.h5'),
    keras.callbacks.TensorBoard(log_dir='./logs')
]
```

---

# Keras Implementation (7/8)

## Transfer Learning
```python
# Pre-trained models
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False
)

# Feature extraction
base_model.trainable = False

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False
```

---

# Keras Implementation (8/8)

## Model Evaluation
```python
# Metrics
model.evaluate(x_test, y_test)

# Custom metrics
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()

# Visualization
keras.utils.plot_model(model, show_shapes=True)
```

---

# PyTorch Implementation (1/6)

## Neural Network Fundamentals
```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.fc1 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        return self.fc1(x)
```

---

# PyTorch Implementation (2/6)

## Data Management
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

# PyTorch Implementation (3/6)

## Model Architecture
```python
# Creating networks
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

# Loss functions
criterion = nn.CrossEntropyLoss()
```

---

# PyTorch Implementation (4/6)

## Training Implementation
```python
# Training loop
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Validation
model.eval()
with torch.no_grad():
    val_loss = criterion(model(val_data), val_target)
```

---

# PyTorch Implementation (5/6)

## Advanced Features
```python
# Hooks
def hook_fn(module, input, output):
    print(output.shape)
model.register_forward_hook(hook_fn)

# Distributed training
model = nn.DataParallel(model)

# Mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
```

---

# PyTorch Implementation (6/6)

## Advanced Training Concepts
```python
# Memory management
torch.cuda.empty_cache()

# GPU utilization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

# Framework Integration (1/5)

## Best Practices
```python
# Memory management
@torch.no_grad()
def evaluate():
    model.eval()
    return model(test_data)

# Production deployment
model.eval()
torch.jit.script(model)
```

---

# Framework Integration (2/5)

## Training Monitoring
```python
# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Loss/train', loss, epoch)

# Progress tracking
from tqdm import tqdm
for batch in tqdm(dataloader):
    # training step
```

---

# Framework Integration (3/5)

## Debugging Tools
```python
# Gradient checking
torch.autograd.gradcheck(func, inputs)

# Memory profiling
with torch.profiler.profile() as prof:
    model(input)
print(prof.key_averages().table())

# Performance optimization
torch.backends.cudnn.benchmark = True
```

---

# Framework Integration (4/5)

## Ecosystem Integration
```python
# Extensions
import torchvision
import tensorflow_addons

# Libraries
from transformers import AutoModel
from pytorch_lightning import LightningModule

# Community tools
import wandb
wandb.init(project="my-project")
```

---

# Framework Integration (5/5)

## Production Considerations
```python
# Model serving
model.save('model.h5')  # Keras
torch.save(model.state_dict(), 'model.pt')  # PyTorch

# Deployment options
import onnx
import torch.onnx
torch.onnx.export(model, dummy_input, "model.onnx")

# Hugging Face deployment
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
