# Demo 3: Time Series Visualization and Automation

## Learning Objectives
- Create effective time series visualizations
- Perform seasonal decomposition
- Set up automated analysis with cron jobs
- Monitor time series performance

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Set style
plt.style.use('default')
sns.set_style('whitegrid')

# Set random seed for reproducibility
np.random.seed(42)
```

## Part 1: Time Series Visualization

### Create Sample Time Series

```python
# Create realistic time series data
print("=== Creating Sample Time Series ===")

# Generate time series with trend, seasonality, and noise
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
n = len(dates)

# Trend component
trend = np.linspace(100, 150, n)

# Seasonal component (annual cycle)
seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)

# Noise component
noise = np.random.normal(0, 5, n)

# Combine components
values = trend + seasonal + noise
ts = pd.Series(values, index=dates)
ts.name = 'Time Series'

print(f"Time series shape: {ts.shape}")
print(f"Date range: {ts.index.min()} to {ts.index.max()}")
print(f"Value range: {ts.min():.2f} to {ts.max():.2f}")
```

### Basic Time Series Plots

```python
# Create basic time series plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Line plot
axes[0, 0].plot(ts.index, ts.values, linewidth=1)
axes[0, 0].set_title('Time Series Line Plot')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

# Rolling mean
rolling_mean = ts.rolling(window=30).mean()
axes[0, 1].plot(ts.index, ts.values, alpha=0.7, label='Original')
axes[0, 1].plot(rolling_mean.index, rolling_mean.values, linewidth=2, label='30-day MA')
axes[0, 1].set_title('Time Series with Rolling Mean')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Histogram
axes[1, 0].hist(ts.values, bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Value Distribution')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Box plot by month
monthly_data = [ts[ts.index.month == i].values for i in range(1, 13)]
axes[1, 1].boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axes[1, 1].set_title('Monthly Distribution')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Value')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Advanced Time Series Visualization

```python
# Create advanced visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Multiple time series
ts_ma5 = ts.rolling(window=5).mean()
ts_ma20 = ts.rolling(window=20).mean()
ts_ma50 = ts.rolling(window=50).mean()

axes[0, 0].plot(ts.index, ts.values, alpha=0.7, label='Original')
axes[0, 0].plot(ts_ma5.index, ts_ma5.values, label='5-day MA')
axes[0, 0].plot(ts_ma20.index, ts_ma20.values, label='20-day MA')
axes[0, 0].plot(ts_ma50.index, ts_ma50.values, label='50-day MA')
axes[0, 0].set_title('Multiple Moving Averages')
axes[0, 0].set_ylabel('Value')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Volatility analysis
returns = ts.pct_change()
volatility = returns.rolling(window=20).std()

axes[0, 1].plot(ts.index, ts.values, alpha=0.7, label='Price')
axes[0, 1].set_ylabel('Price', color='blue')
axes[0, 1].tick_params(axis='y', labelcolor='blue')

ax2 = axes[0, 1].twinx()
ax2.plot(volatility.index, volatility.values, color='red', alpha=0.7, label='Volatility')
ax2.set_ylabel('Volatility', color='red')
ax2.tick_params(axis='y', labelcolor='red')

axes[0, 1].set_title('Price and Volatility')
axes[0, 1].grid(True, alpha=0.3)

# Seasonal analysis
monthly_means = ts.groupby(ts.index.month).mean()
axes[1, 0].plot(monthly_means.index, monthly_means.values, marker='o', linewidth=2)
axes[1, 0].set_title('Seasonal Pattern (Monthly Means)')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Average Value')
axes[1, 0].set_xticks(range(1, 13))
axes[1, 0].grid(True, alpha=0.3)

# Correlation analysis
ts_df = ts.to_frame()
ts_df['lag_1'] = ts.shift(1)
ts_df['lag_7'] = ts.shift(7)
ts_df['lag_30'] = ts.shift(30)

correlation_matrix = ts_df.corr()
im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
axes[1, 1].set_title('Autocorrelation Matrix')
axes[1, 1].set_xticks(range(len(correlation_matrix.columns)))
axes[1, 1].set_yticks(range(len(correlation_matrix.index)))
axes[1, 1].set_xticklabels(correlation_matrix.columns, rotation=45)
axes[1, 1].set_yticklabels(correlation_matrix.index)

# Add colorbar
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

## Part 2: Seasonal Decomposition

### Basic Seasonal Decomposition

```python
# Perform seasonal decomposition
print("=== Seasonal Decomposition ===")

# Use statsmodels for decomposition
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Decompose time series
    decomposition = seasonal_decompose(ts, model='additive', period=365)
    
    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    decomposition.observed.plot(ax=axes[0], title='Original')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    plt.tight_layout()
    plt.show()
    
    # Print decomposition statistics
    print("Decomposition Statistics:")
    print(f"Trend range: {decomposition.trend.min():.2f} to {decomposition.trend.max():.2f}")
    print(f"Seasonal range: {decomposition.seasonal.min():.2f} to {decomposition.seasonal.max():.2f}")
    print(f"Residual range: {decomposition.resid.min():.2f} to {decomposition.resid.max():.2f}")
    
except ImportError:
    print("statsmodels not available. Install with: pip install statsmodels")
```

### Manual Seasonal Analysis

```python
# Manual seasonal analysis
print("=== Manual Seasonal Analysis ===")

# Calculate seasonal components
ts_df = ts.to_frame()
ts_df['year'] = ts_df.index.year
ts_df['month'] = ts_df.index.month
ts_df['day_of_year'] = ts_df.index.dayofyear

# Monthly seasonal pattern
monthly_pattern = ts_df.groupby('month')['Time Series'].mean()
print("Monthly seasonal pattern:")
print(monthly_pattern)

# Plot seasonal patterns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Monthly pattern
monthly_pattern.plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Monthly Seasonal Pattern')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Average Value')
axes[0, 0].tick_params(axis='x', rotation=45)

# Day of year pattern
daily_pattern = ts_df.groupby('day_of_year')['Time Series'].mean()
axes[0, 1].plot(daily_pattern.index, daily_pattern.values)
axes[0, 1].set_title('Daily Seasonal Pattern')
axes[0, 1].set_xlabel('Day of Year')
axes[0, 1].set_ylabel('Average Value')
axes[0, 1].grid(True, alpha=0.3)

# Yearly comparison
yearly_data = ts_df.groupby('year')['Time Series'].mean()
axes[1, 0].plot(yearly_data.index, yearly_data.values, marker='o')
axes[1, 0].set_title('Yearly Averages')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Average Value')
axes[1, 0].grid(True, alpha=0.3)

# Heatmap of seasonal patterns
pivot_data = ts_df.pivot_table(values='Time Series', index='month', columns='year', aggfunc='mean')
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 1])
axes[1, 1].set_title('Seasonal Heatmap (Month vs Year)')

plt.tight_layout()
plt.show()
```

## Part 3: Automation with Cron Jobs

### Cron Job Setup

```python
# Simulate cron job setup
print("=== Cron Job Setup ===")
print("In a real scenario, you would:")
print("1. Create a Python script for time series analysis")
print("2. Set up cron job to run the script automatically")
print("3. Configure logging and error handling")
print("4. Set up monitoring and alerts")
```

### Time Series Analysis Script

```python
# Create time series analysis script
def create_analysis_script():
    """Create a time series analysis script for cron job"""
    
    script_content = '''#!/usr/bin/env python3
"""
Time series analysis script for cron job
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(
    filename='/path/to/logs/time_series_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_time_series_analysis():
    """Run daily time series analysis"""
    try:
        # Load data
        df = pd.read_csv('/path/to/data/time_series_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Perform analysis
        daily_stats = df.resample('D').agg({
            'value': ['mean', 'std', 'min', 'max'],
            'volume': 'sum'
        })
        
        # Calculate moving averages
        df['ma_7'] = df['value'].rolling(window=7).mean()
        df['ma_30'] = df['value'].rolling(window=30).mean()
        
        # Detect anomalies
        df['anomaly'] = abs(df['value'] - df['ma_7']) > 2 * df['value'].rolling(window=30).std()
        
        # Save results
        daily_stats.to_csv('/path/to/results/daily_stats.csv')
        df.to_csv('/path/to/results/analysis_results.csv')
        
        # Log completion
        logging.info("Time series analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in time series analysis: {e}")

if __name__ == "__main__":
    run_time_series_analysis()
'''
    
    return script_content

# Display the script
print("=== Time Series Analysis Script ===")
script = create_analysis_script()
print(script)
```

### Cron Job Configuration

```python
# Cron job configuration
def create_cron_config():
    """Create cron job configuration"""
    
    cron_config = '''
# Time series analysis cron jobs
# Run every day at 2 AM
0 2 * * * /path/to/scripts/time_series_analysis.sh

# Run every Monday at 9 AM
0 9 * * 1 /path/to/scripts/weekly_analysis.sh

# Run every 15 minutes during business hours
*/15 9-17 * * 1-5 /path/to/scripts/realtime_analysis.sh

# Run at specific times
0 9,17 * * * /path/to/scripts/business_hours_analysis.sh
'''
    
    return cron_config

print("=== Cron Job Configuration ===")
config = create_cron_config()
print(config)
```

## Part 4: Performance Monitoring

### Performance Metrics

```python
# Create performance monitoring
def monitor_performance():
    """Monitor time series analysis performance"""
    
    print("=== Performance Monitoring ===")
    
    # Simulate performance metrics
    metrics = {
        'data_size': '1M records',
        'processing_time': '2.5 seconds',
        'memory_usage': '512 MB',
        'cpu_usage': '45%',
        'disk_usage': '2.1 GB',
        'error_rate': '0.1%'
    }
    
    print("Performance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Create performance report
    report = f"""
Time Series Analysis Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Data Processing:
- Data size: {metrics['data_size']}
- Processing time: {metrics['processing_time']}
- Memory usage: {metrics['memory_usage']}
- CPU usage: {metrics['cpu_usage']}

Storage:
- Disk usage: {metrics['disk_usage']}

Quality:
- Error rate: {metrics['error_rate']}

Status: All systems operational
"""
    
    return report

# Generate performance report
performance_report = monitor_performance()
print(performance_report)
```

### Automated Monitoring

```python
# Automated monitoring system
def create_monitoring_system():
    """Create automated monitoring system"""
    
    monitoring_script = '''#!/usr/bin/env python3
"""
Automated monitoring system for time series analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import smtplib
from email.mime.text import MIMEText

def check_data_quality(df):
    """Check data quality metrics"""
    quality_metrics = {
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_range': (df.max() - df.min()).mean(),
        'outlier_count': len(df[(df - df.mean()).abs() > 3 * df.std()])
    }
    return quality_metrics

def check_performance_metrics():
    """Check performance metrics"""
    # Simulate performance monitoring
    return {
        'processing_time': 2.5,
        'memory_usage': 512,
        'cpu_usage': 45,
        'error_count': 0
    }

def send_alert(message):
    """Send alert notification"""
    # Simulate alert sending
    print(f"ALERT: {message}")
    logging.warning(f"Alert sent: {message}")

def run_monitoring():
    """Run automated monitoring"""
    try:
        # Load data
        df = pd.read_csv('/path/to/data/time_series_data.csv')
        
        # Check data quality
        quality = check_data_quality(df)
        if quality['missing_values'] > 100:
            send_alert("High number of missing values detected")
        
        # Check performance
        performance = check_performance_metrics()
        if performance['processing_time'] > 10:
            send_alert("Processing time exceeded threshold")
        
        if performance['memory_usage'] > 1000:
            send_alert("Memory usage exceeded threshold")
        
        # Log monitoring results
        logging.info(f"Monitoring completed - Quality: {quality}, Performance: {performance}")
        
    except Exception as e:
        logging.error(f"Monitoring error: {e}")
        send_alert(f"Monitoring system error: {e}")

if __name__ == "__main__":
    run_monitoring()
'''
    
    return monitoring_script

print("=== Automated Monitoring System ===")
monitoring_script = create_monitoring_system()
print(monitoring_script)
```

## Key Takeaways

1. **Time Series Visualization**: Create effective plots for trend, seasonality, and patterns
2. **Seasonal Decomposition**: Separate trend, seasonal, and residual components
3. **Automation**: Set up cron jobs for automated time series analysis
4. **Monitoring**: Track performance and data quality metrics
5. **Real-world Application**: Apply techniques to production time series systems
6. **Best Practices**: Use logging, error handling, and alerting for robust systems

## Next Steps

- Practice with your own time series data
- Set up automated analysis pipelines
- Learn about time series forecasting
- Explore advanced time series techniques
