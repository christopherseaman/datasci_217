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
