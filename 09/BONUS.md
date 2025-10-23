# Advanced Time Series Analysis Topics

## Advanced Time Series Decomposition

### Seasonal Decomposition

**Reference:**

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np

# Create seasonal time series
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
trend = np.linspace(100, 200, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 5, len(dates))
values = trend + seasonal + noise

ts = pd.Series(values, index=dates)

# Decompose time series
decomposition = seasonal_decompose(ts, model='additive', period=365)
decomposition_multiplicative = seasonal_decompose(ts, model='multiplicative', period=365)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 12))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()
```

### STL Decomposition

**Reference:**

```python
from statsmodels.tsa.seasonal import STL

# STL decomposition (more robust)
stl = STL(ts, seasonal=365)
result = stl.fit()

# Plot STL decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 12))
result.observed.plot(ax=axes[0], title='Original')
result.trend.plot(ax=axes[1], title='Trend')
result.seasonal.plot(ax=axes[2], title='Seasonal')
result.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()
```

## Time Series Forecasting

### ARIMA Models

**Reference:**

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Check stationarity
def check_stationarity(series):
    """Check if time series is stationary"""
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: {result[4]}')
    
    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")

# Make series stationary
def make_stationary(series):
    """Make time series stationary"""
    # First difference
    diff1 = series.diff().dropna()
    
    # Check stationarity
    check_stationarity(diff1)
    
    return diff1

# Fit ARIMA model
def fit_arima(series, order=(1, 1, 1)):
    """Fit ARIMA model"""
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    
    # Print summary
    print(fitted_model.summary())
    
    # Forecast
    forecast = fitted_model.forecast(steps=30)
    
    return fitted_model, forecast

# Example usage
stationary_ts = make_stationary(ts)
model, forecast = fit_arima(stationary_ts)
```

### Exponential Smoothing

**Reference:**

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Simple exponential smoothing
def simple_exponential_smoothing(series, alpha=0.3):
    """Simple exponential smoothing"""
    model = ExponentialSmoothing(series, trend=None, seasonal=None)
    fitted_model = model.fit(smoothing_level=alpha)
    forecast = fitted_model.forecast(steps=30)
    
    return fitted_model, forecast

# Holt's method (trend)
def holts_method(series, alpha=0.3, beta=0.3):
    """Holt's exponential smoothing with trend"""
    model = ExponentialSmoothing(series, trend='add', seasonal=None)
    fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=beta)
    forecast = fitted_model.forecast(steps=30)
    
    return fitted_model, forecast

# Holt-Winters (trend + seasonal)
def holt_winters(series, alpha=0.3, beta=0.3, gamma=0.3):
    """Holt-Winters exponential smoothing"""
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
    fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
    forecast = fitted_model.forecast(steps=30)
    
    return fitted_model, forecast
```

## Advanced Time Series Operations

### Time Series Alignment

**Reference:**

```python
# Align multiple time series
def align_time_series(ts1, ts2, method='inner'):
    """Align two time series"""
    if method == 'inner':
        return ts1.align(ts2, join='inner')
    elif method == 'outer':
        return ts1.align(ts2, join='outer')
    elif method == 'left':
        return ts1.align(ts2, join='left')
    elif method == 'right':
        return ts1.align(ts2, join='right')

# Interpolate missing values
def interpolate_missing(series, method='linear'):
    """Interpolate missing values in time series"""
    return series.interpolate(method=method)

# Forward fill and backward fill
def fill_missing_values(series, method='ffill'):
    """Fill missing values"""
    if method == 'ffill':
        return series.fillna(method='ffill')
    elif method == 'bfill':
        return series.fillna(method='bfill')
    elif method == 'interpolate':
        return series.interpolate()
```

### Time Series Clustering

**Reference:**

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_time_series(df, n_clusters=3):
    """Cluster time series data"""
    # Prepare data for clustering
    data = df.values
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    
    # Add cluster labels to DataFrame
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    return df_clustered, kmeans

# Example usage
# Create multiple time series
ts1 = pd.Series(np.random.randn(100), index=pd.date_range('2023-01-01', periods=100))
ts2 = pd.Series(np.random.randn(100), index=pd.date_range('2023-01-01', periods=100))
ts3 = pd.Series(np.random.randn(100), index=pd.date_range('2023-01-01', periods=100))

# Combine into DataFrame
df_combined = pd.DataFrame({'ts1': ts1, 'ts2': ts2, 'ts3': ts3})

# Cluster time series
df_clustered, model = cluster_time_series(df_combined)
print("Clustered time series:")
print(df_clustered.head())
```

## High-Frequency Data Analysis

### Tick Data Processing

**Reference:**

```python
# Process high-frequency tick data
def process_tick_data(df, freq='1T'):
    """Process tick data into regular intervals"""
    # Resample to regular frequency
    resampled = df.resample(freq).agg({
        'price': 'last',  # Last price in interval
        'volume': 'sum',  # Total volume in interval
        'trades': 'count'  # Number of trades in interval
    })
    
    return resampled

# Calculate high-frequency statistics
def calculate_hf_stats(df):
    """Calculate high-frequency statistics"""
    stats = {
        'tick_count': len(df),
        'price_range': df['price'].max() - df['price'].min(),
        'volume_weighted_price': (df['price'] * df['volume']).sum() / df['volume'].sum(),
        'trade_intensity': len(df) / (df.index[-1] - df.index[0]).total_seconds() * 60,  # trades per minute
        'price_volatility': df['price'].std()
    }
    
    return stats
```

### Microstructure Analysis

**Reference:**

```python
# Calculate bid-ask spread
def calculate_spread(df):
    """Calculate bid-ask spread"""
    df['spread'] = df['ask'] - df['bid']
    df['spread_pct'] = (df['spread'] / df['mid_price']) * 100
    
    return df

# Calculate order flow imbalance
def calculate_order_flow(df):
    """Calculate order flow imbalance"""
    df['buy_volume'] = df['volume'] * (df['side'] == 'buy')
    df['sell_volume'] = df['volume'] * (df['side'] == 'sell')
    df['order_flow'] = df['buy_volume'] - df['sell_volume']
    
    return df

# Calculate price impact
def calculate_price_impact(df):
    """Calculate price impact of trades"""
    df['price_change'] = df['price'].diff()
    df['volume_change'] = df['volume'].diff()
    df['price_impact'] = df['price_change'] / df['volume_change']
    
    return df
```

## Time Series Visualization

### Interactive Time Series Plots

**Reference:**

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_plot(ts, title="Interactive Time Series"):
    """Create interactive time series plot"""
    fig = go.Figure()
    
    # Add main time series
    fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values,
        mode='lines',
        name='Time Series',
        line=dict(color='blue', width=2)
    ))
    
    # Add rolling mean
    rolling_mean = ts.rolling(window=7).mean()
    fig.add_trace(go.Scatter(
        x=rolling_mean.index,
        y=rolling_mean.values,
        mode='lines',
        name='Rolling Mean (7 days)',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

# Create subplot with multiple time series
def create_multi_plot(ts1, ts2, ts3):
    """Create subplot with multiple time series"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Time Series 1', 'Time Series 2', 'Time Series 3'],
        vertical_spacing=0.1
    )
    
    # Add each time series
    fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, name='TS1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, name='TS2'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts3.index, y=ts3.values, name='TS3'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    return fig
```

### Advanced Statistical Plots

**Reference:**

```python
# Autocorrelation and partial autocorrelation
def plot_acf_pacf(series, lags=40):
    """Plot ACF and PACF"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function')
    
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()

# Seasonal plot
def plot_seasonal(series, period=12):
    """Plot seasonal patterns"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Seasonal subseries plot
    seasonal_data = series.groupby(series.index.month).mean()
    seasonal_data.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Seasonal Pattern by Month')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Value')
    
    # Seasonal box plot
    series.groupby(series.index.month).plot(kind='box', ax=axes[1])
    axes[1].set_title('Seasonal Distribution by Month')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()
```

## Time Series Anomaly Detection

### Statistical Methods

**Reference:**

```python
# Z-score anomaly detection
def detect_anomalies_zscore(series, threshold=3):
    """Detect anomalies using Z-score"""
    z_scores = np.abs((series - series.mean()) / series.std())
    anomalies = series[z_scores > threshold]
    
    return anomalies, z_scores

# Modified Z-score (more robust)
def detect_anomalies_modified_zscore(series, threshold=3.5):
    """Detect anomalies using modified Z-score"""
    median = series.median()
    mad = np.median(np.abs(series - median))
    modified_z_scores = 0.6745 * (series - median) / mad
    anomalies = series[np.abs(modified_z_scores) > threshold]
    
    return anomalies, modified_z_scores

# Isolation Forest
from sklearn.ensemble import IsolationForest

def detect_anomalies_isolation_forest(series, contamination=0.1):
    """Detect anomalies using Isolation Forest"""
    # Reshape data for sklearn
    data = series.values.reshape(-1, 1)
    
    # Fit isolation forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(data)
    
    # Get anomalies
    anomalies = series[anomaly_labels == -1]
    
    return anomalies, anomaly_labels
```

### Time Series Specific Methods

**Reference:**

```python
# Seasonal anomaly detection
def detect_seasonal_anomalies(series, period=12, threshold=2):
    """Detect anomalies considering seasonality"""
    # Calculate seasonal mean and std
    seasonal_mean = series.groupby(series.index.month).mean()
    seasonal_std = series.groupby(series.index.month).std()
    
    # Calculate seasonal z-scores
    seasonal_z_scores = []
    for i, (date, value) in enumerate(series.items()):
        month = date.month
        mean_val = seasonal_mean[month]
        std_val = seasonal_std[month]
        z_score = (value - mean_val) / std_val
        seasonal_z_scores.append(z_score)
    
    seasonal_z_scores = pd.Series(seasonal_z_scores, index=series.index)
    anomalies = series[np.abs(seasonal_z_scores) > threshold]
    
    return anomalies, seasonal_z_scores

# Change point detection
def detect_change_points(series, window=10, threshold=2):
    """Detect change points in time series"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Calculate z-score for change points
    z_scores = (series - rolling_mean) / rolling_std
    change_points = series[np.abs(z_scores) > threshold]
    
    return change_points, z_scores
```

## Time Series Machine Learning

### Feature Engineering

**Reference:**

```python
# Create time series features
def create_ts_features(series, lags=5):
    """Create features for time series machine learning"""
    features = pd.DataFrame(index=series.index)
    
    # Original series
    features['value'] = series
    
    # Lag features
    for lag in range(1, lags + 1):
        features[f'lag_{lag}'] = series.shift(lag)
    
    # Rolling statistics
    features['rolling_mean_7'] = series.rolling(window=7).mean()
    features['rolling_std_7'] = series.rolling(window=7).std()
    features['rolling_min_7'] = series.rolling(window=7).min()
    features['rolling_max_7'] = series.rolling(window=7).max()
    
    # Difference features
    features['diff_1'] = series.diff()
    features['diff_2'] = series.diff().diff()
    
    # Percentage change
    features['pct_change'] = series.pct_change()
    
    # Time-based features
    features['year'] = series.index.year
    features['month'] = series.index.month
    features['day'] = series.index.day
    features['dayofweek'] = series.index.dayofweek
    features['dayofyear'] = series.index.dayofyear
    
    # Cyclical encoding
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    features['dayofweek_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
    features['dayofweek_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
    
    return features.dropna()
```

### Time Series Cross-Validation

**Reference:**

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def time_series_cv(series, features, model, n_splits=5):
    """Perform time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    for train_idx, test_idx in tscv.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = series.iloc[train_idx], series.iloc[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        scores.append({'mse': mse, 'mae': mae})
    
    return scores

# Example usage
features = create_ts_features(ts)
model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = time_series_cv(ts, features, model)
print("Cross-validation scores:")
for i, score in enumerate(scores):
    print(f"Fold {i+1}: MSE={score['mse']:.4f}, MAE={score['mae']:.4f}")
```

These advanced topics will help you handle complex time series analysis scenarios and build sophisticated temporal models in your data science work.
