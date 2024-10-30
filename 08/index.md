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
Key points to discuss:
- Time series analysis is crucial for analyzing temporal data patterns
- statsmodels provides comprehensive statistical modeling capabilities
- scikit-learn offers a unified API for machine learning tasks
- Deep learning frameworks (Keras/TensorFlow, PyTorch) enable complex neural network architectures
-->

---

# Time Series Analysis (1/8)

## Introduction to Time Series
- Definition and key components
  ```python
  # Input: data (array-like of numeric values), index (datetime-like array of same length as data)
  # Output: pandas Series with datetime index
  ts = pd.Series(data, index=pd.date_range('2023', periods=12, freq='M'))
  ```
- Types:
  - Fixed frequency (regular intervals)
  - Irregular frequency (variable intervals)
- Common applications: finance, economics, sensor data

<!--
Key points:
- Time series data represents observations collected over time
- Regular vs irregular intervals affect analysis methods
- Real-world applications span multiple domains
-->

---

# Time Series Analysis (2/8)
## pandas DateTime Objects
```python
# Input: start_date (str in YYYY-MM-DD format), periods (int)
# Output: DatetimeIndex object
dates = pd.date_range('2023-01-01', periods=5)

# Input: date_str (str in YYYY-MM-DD format)
# Output: Timestamp and Period objects
ts = pd.Timestamp('2023-01-01')
period = pd.Period('2023-01', freq='M')

# Input: timezone_str (str representing timezone)
# Output: timezone-aware Timestamp objects
ts_ny = ts.tz_localize('America/New_York')
ts_utc = ts_ny.tz_convert('UTC')
```

<!--
Key points:
- DateTime handling is crucial for time series analysis
- pandas provides multiple datetime object types
- Timezone awareness is important for global data
-->

---

# Time Series Analysis (3/8)

## Time Series Operations
```python
# Input: freq (str representing frequency e.g., 'M', 'H'), aggregation function
# Output: resampled DataFrame with new frequency
df.resample('M').mean()    # Downsample to monthly
df.resample('H').ffill()   # Upsample to hourly

# Input: window (int) size of rolling window
# Output: DataFrame with rolling statistics
df.rolling(window=7).mean()

# Input: periods (int) number of periods to shift
# Output: shifted DataFrame
df.shift(periods=1)        # Forward shift
df.shift(periods=-1)       # Backward shift
```

<!--
Key points:
- Resampling allows changing time series frequency
- Rolling windows enable moving calculations
- Shifting helps analyze lagged relationships
-->

---

# Time Series Analysis (4/8)

## Frequency and Date Ranges
```python
# Input: start (str in YYYY format), freq (str), periods (int)
# Output: DatetimeIndex with specified frequency
pd.date_range('2023', freq='D', periods=365)   # Daily
pd.date_range('2023', freq='B', periods=252)   # Business days
pd.date_range('2023', freq='M', periods=12)    # Month end
pd.date_range('2023', freq='Q', periods=4)     # Quarter end

# Input: start (str), freq (str), periods (int)
# Output: PeriodIndex with specified frequency
pd.period_range('2023', freq='M', periods=12)
```

<!--
Key points:
- Different frequency options for various analysis needs
- Business days vs calendar days handling
- Period ranges for fiscal/accounting periods
-->

---

# Time Series Analysis (5/8)

## Moving Window Functions
```python
# Input: window (int), min_periods (int)
# Output: DataFrame with rolling statistics
df.rolling(window=30, min_periods=25).mean()
df.rolling(window=30, min_periods=25).std()

# Input: span (int), min_periods (int)
# Output: DataFrame with exponential weighted statistics
df.ewm(span=30, min_periods=25).mean()

# Input: window (int), custom function
# Output: DataFrame with custom rolling statistics
def custom_stat(x): return x.max() - x.min()
df.rolling(window=30).apply(custom_stat)
```

<!--
Key points:
- Moving windows capture local patterns
- Exponential weighting gives more weight to recent data
- Custom functions enable flexible analysis
-->

---

# Time Series Analysis (6/8)

## Time Series Decomposition
```python
# Input: time series (array-like), model type (str), period (int)
# Output: decomposition object with trend, seasonal, and residual components
result = seasonal_decompose(
    df['value'],
    model='additive',  # or 'multiplicative'
    period=12
)

trend = result.trend
seasonal = result.seasonal
residual = result.resid
```

<!--
Key points:
- Decomposition separates time series into components
- Additive vs multiplicative models for different patterns
- Understanding trend, seasonality, and noise
-->

---

# Time Series Analysis (7/8)

## Time Series Visualization
```python
# Input: DataFrame with datetime index
# Output: matplotlib figure
df.plot(figsize=(12,6))

# Input: DataFrame grouped by time component
# Output: bar plot of seasonal patterns
df.groupby(df.index.month).mean().plot(kind='bar')

# Input: time series data
# Output: autocorrelation plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df)
```

<!--
Key points:
- Visualization reveals patterns and anomalies
- Seasonal plots show recurring patterns
- Autocorrelation helps identify time dependencies
-->

---

# Time Series Analysis (8/8)

## Advanced Time Series Models
```python
# Input: time series data, ARIMA order parameters (p,d,q)
# Output: fitted ARIMA model
model = ARIMA(data, order=(1,1,1))
results = model.fit()

# Input: steps (int) for forecast horizon
# Output: point forecasts and confidence intervals
forecast = results.forecast(steps=5)
conf_int = results.get_forecast(steps=5).conf_int()
```

<!--
Key points:
- ARIMA models capture complex time series patterns
- Model parameters affect forecasting accuracy
- Confidence intervals quantify prediction uncertainty
-->

---
# statsmodels (1/9)

## Linear Regression Basics
```python
import statsmodels.api as sm

# Input: X (array-like, independent variables), y (array-like, dependent variable)
# Note: X should be a matrix of independent variables (features)
#       Each column represents a different predictor variable
#       Rows represent observations
X = sm.add_constant(X)  # Add intercept column to X
model = sm.OLS(y, X)
results = model.fit()

# Key outputs
print(results.params)        # Coefficients
print(results.rsquared)     # R-squared
print(results.pvalues)      # P-values
```

<!--
- statsmodels provides comprehensive statistical modeling tools
- OLS is the foundation of linear regression analysis
- Adding a constant term is crucial for proper model fitting
- Model results provide multiple statistical measures
-->

---

# statsmodels (2/9)

## Model Diagnostics
```python
# Input: model results object
# Output: various diagnostic statistics
print(results.rsquared)
print(results.rsquared_adj)

# Input: model results object
# Output: p-values for each coefficient
print(results.pvalues)

# Input: residuals (array-like), X matrix
# Output: heteroskedasticity test statistics
residuals = results.resid
sm.stats.diagnostic.het_breuschpagan(residuals, X)
```

<!--
- Model diagnostics help validate assumptions
- R-squared vs adjusted R-squared tradeoffs
- P-values indicate statistical significance
- Heteroskedasticity testing is crucial for valid inference
-->

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

<!--
- GLMs extend linear regression to non-normal distributions
- Different families handle different response types
- Link functions connect linear predictor to response
- Common applications in binary and count data
-->

---

# statsmodels (4/9)

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

<!--
- Statistical tests help validate hypotheses
- T-tests compare means between groups
- ANOVA examines variance between groups
- Chi-square tests analyze categorical relationships
-->

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

<!--
- Influence measures identify impactful observations
- Leverage indicates potential outliers in predictors
- Cook's distance combines leverage and residuals
- Visual diagnostics help spot problematic points
-->

---

# statsmodels (6/9)

## Panel Data Analysis
```pythoncode 
from statsmodels.regression.linear_model import PanelOLS

# Fixed effects
model_fe = PanelOLS(y, X, entity_effects=True)
fe_results = model_fe.fit()

# Random effects
model_re = PanelOLS(y, X, random_effects=True)
re_results = model_re.fit()
```

<!--
- Panel data combines cross-sectional and time series
- Fixed effects control for time-invariant factors
- Random effects assume random individual differences
- Choice between models depends on Hausman test
-->

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

<!--
- Robust methods handle outliers and violations
- Different types of robust standard errors
- M-estimators provide resistance to outliers
- Trade-off between efficiency and robustness
-->

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

<!--
- Summary tables provide comprehensive results
- Visual diagnostics help assess assumptions
- Confidence intervals quantify uncertainty
- Prediction helps validate model performance
-->

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

<!--
- VIF helps detect multicollinearity
- Assumption checking is crucial
- Multiple tests needed for thorough validation
- Automated checks streamline analysis
-->

---
# scikit-learn (1/9)

## Data Preprocessing
```python
# Input: X (array-like of shape (n_samples, n_features))
# Output: standardized features with zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Input: X_cat (array-like of categorical variables)
# Output: one-hot encoded matrix
enc = OneHotEncoder(sparse=False)
X_encoded = enc.fit_transform(X_cat)
```

<!--
- Preprocessing is crucial for model performance
- Scaling ensures features are comparable
- One-hot encoding handles categorical variables
- Always fit preprocessors on training data only
-->

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

<!--
- Feature selection reduces dimensionality
- SelectKBest uses univariate statistical tests
- RFE iteratively removes least important features
- Different methods suit different problems
-->

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

<!--
- Classification predicts categorical outcomes
- Different algorithms suit different problems
- Consistent API across all models
- Model selection depends on data characteristics
-->

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

<!--
- Regression predicts continuous outcomes
- Similar interface to classification
- Different algorithms handle nonlinearity differently
- Consider interpretability vs performance
-->

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

<!--
- Unsupervised learning finds patterns without labels
- Clustering groups similar data points
- PCA reduces dimensions while preserving variance
- Different methods reveal different patterns
-->

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

<!--
- Cross-validation assesses model generalization
- Grid search optimizes hyperparameters
- Multiple metrics available for evaluation
- Balance between computation and thoroughness
-->

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

<!--
- Different metrics suit different problems
- Classification report provides comprehensive view
- Confusion matrix shows error types
- R-squared and MSE common for regression
-->

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

<!--
- Pipelines chain operations together
- Ensures consistent preprocessing
- Feature union combines different features
- Helps prevent data leakage
-->

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

<!--
- Model persistence saves trained models
- Joblib efficient for large NumPy arrays
- Partial fitting allows incremental updates
- Consider version compatibility
-->

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

<!--
- Patsy provides R-like formula syntax
- Easy specification of model terms
- Built-in transformations available
- Interaction terms easily specified
-->

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

<!--
- Automatic creation of design matrices
- Special handling for categorical data
- Multiple coding schemes available
- Missing data handling options
-->

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

<!--
- Custom transformations possible
- Spline basis functions supported
- Seamless statsmodels integration
- Flexible formula specification
-->

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

<!--
- Multiple frameworks available
- Different programming paradigms
- Keras focuses on simplicity
- PyTorch emphasizes flexibility
-->

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

<!--
- Framework choice depends on use case
- Keras excels in production deployment
- PyTorch popular in research
- Data preprocessing important for all
-->

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

<!--
- Keras provides high-level neural network APIs
- Common layer types available out of box
- Built-in activation functions
- Easy layer stacking with add()
-->

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

<!--
- Sequential API for linear layer stacks
- Easy model definition and compilation
- Common optimizers and losses built-in
- Multiple metrics can be tracked
-->

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

<!--
- Functional API for complex architectures
- Multiple inputs/outputs possible
- Subclassing for custom behavior
- Different APIs for different needs
-->

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

<!--
- Built-in data augmentation tools
- Sequence handling utilities
- Text preprocessing capabilities
- Real-time data generation
-->

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

<!--
- Comprehensive loss function library
- Multiple optimizer options
- Learning rate scheduling
- Easy parameter tuning
-->

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

<!--
- Simple fit API for training
- Built-in validation support
- Rich callback system
- Training monitoring tools
-->

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

<!--
- Pre-trained models available
- Easy feature extraction
- Fine-tuning capabilities
- Layer freezing control
-->

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

<!--
- Built-in evaluation methods
- Custom metric definition
- Model visualization tools
- Comprehensive evaluation options
-->

---
# PyTorch Implementation (1/6)

## Neural Network Fundamentals
```python
# Input: input tensor of shape (batch_size, channels, height, width)
# Output: tensor transformed by convolutional and linear layers
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel, 32 output channels, 3x3 kernel
        self.fc1 = nn.Linear(32, 10)      # 32 input features, 10 output classes
    
    def forward(self, x):
        x = self.conv1(x)
        return self.fc1(x)
```

<!--
- PyTorch uses object-oriented model definition
- Models inherit from nn.Module base class
- Forward method defines computation flow
- Explicit control over network architecture
-->

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

<!--
- Custom datasets inherit from Dataset class
- DataLoader handles batching and shuffling
- Efficient memory management for large datasets
- Easy integration with custom data formats
-->

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

<!--
- Sequential containers simplify architecture
- Built-in activation functions and layers
- Common loss functions provided
- Flexible architecture customization
-->

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

<!--
- Explicit training loop provides control
- Gradient computation with backward()
- Optimizer handles parameter updates
- Context managers for evaluation mode
-->

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

<!--
- Hooks enable intermediate layer inspection
- Distributed training scales to multiple GPUs
- Mixed precision speeds up training
- Advanced features for research needs
-->

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

<!--
- Explicit memory management available
- Seamless CPU/GPU switching
- Multi-GPU support built-in
- Performance optimization options
-->

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

<!--
- Memory optimization crucial for large models
- Evaluation mode prevents gradient computation
- TorchScript for production deployment
- Best practices improve performance
-->

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

<!--
- TensorBoard works with multiple frameworks
- Real-time training visualization
- Progress bars aid monitoring
- Logging helps track experiments
-->

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

<!--
- Gradient checking validates implementation
- Profiling identifies bottlenecks
- Memory tracking prevents leaks
- Performance optimization options
-->

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

<!--
- Rich ecosystem of extensions
- Cross-framework compatibility
- Community tools enhance workflow
- Experiment tracking solutions
-->

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
```

<!--
- Multiple model saving formats
- ONNX for framework interoperability
- Deployment considerations important
- Hugging Face simplifies deployment
-->

---
