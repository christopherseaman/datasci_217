"""Brief examples of time series operations"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365, freq='D')
trend = np.linspace(0, 10, 365)
seasonal = 5 * np.sin(2 * np.pi * np.arange(365) / 30)  # 30-day seasonality
noise = np.random.normal(0, 1, 365)
data = trend + seasonal + noise

# Create time series
ts = pd.Series(data, index=dates)

# DateTime operations
print("\n=== DateTime Operations ===")
ts_ny = pd.Timestamp('2023-01-01').tz_localize('America/New_York')
print(f"NY Time: {ts_ny}")
print(f"UTC Time: {ts_ny.tz_convert('UTC')}")

# Resampling
print("\n=== Resampling ===")
monthly_mean = ts.resample('M').mean()
print("Monthly averages:", monthly_mean.head())

# Rolling windows
print("\n=== Rolling Windows ===")
rolling_mean = ts.rolling(window=7).mean()
print("7-day rolling average:", rolling_mean.head())

# Time series decomposition
print("\n=== Decomposition ===")
decomposition = seasonal_decompose(ts, period=30)
components = {
    'Trend': decomposition.trend[0],
    'Seasonal': decomposition.seasonal[0],
    'Residual': decomposition.resid[0]
}
print("Components:", components)

if __name__ == '__main__':
    # Optional: Plot components
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))
    ts.plot(ax=ax1, title='Original')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
    decomposition.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    plt.show()
