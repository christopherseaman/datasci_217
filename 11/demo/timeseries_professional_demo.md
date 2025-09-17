# Time Series Analysis Hands-On Demo

This demo accompanies Lecture 11 on Time Series Analysis, covering the essential concepts from McKinney Chapter 11. We'll explore pandas' powerful time series capabilities through practical examples.

## Demo Structure

The hands-on demo is implemented in [`timeseries_hands_on_demo.py`](11/demo/timeseries_hands_on_demo.py) and covers:

### 1. Date and Time Fundamentals
- Python's `datetime` module basics
- String to datetime conversion with `pd.to_datetime()`
- Working with time deltas

### 2. Creating Time Series
- Basic time series with `DatetimeIndex`
- Business day frequencies
- Understanding index types

### 3. Time Series Indexing and Selection
- Year-based and month-based selection
- Date range slicing
- Flexible time-based filtering

### 4. Frequency Codes and Date Ranges
- Common frequency codes (D, B, W, M, Q, A, H)
- Custom frequencies and intervals
- Special cases like "third Friday of month"

### 5. Shifting and Lag Operations
- Data shifting vs. timestamp shifting
- Calculating returns and changes
- Lead and lag analysis

### 6. Time Zone Handling
- Timezone localization with `tz_localize()`
- Timezone conversion with `tz_convert()`
- Working with timezone-aware data

### 7. Resampling Operations
- Downsampling: aggregating high to low frequency
- Upsampling: converting low to high frequency
- OHLC (Open-High-Low-Close) aggregations
- Forward and backward filling

### 8. Moving Window Functions
- Rolling windows: `rolling()`
- Expanding windows: `expanding()`
- Exponentially weighted: `ewm()`
- Custom window functions

### 9. Practical Financial Analysis
- Stock price simulation and analysis
- Return calculations and risk metrics
- Rolling correlations and volatility
- Performance statistics

### 10. Period-Based Analysis
- Working with `Period` objects
- Quarterly and annual aggregations
- Period arithmetic and conversions
- Business calendar operations

## Running the Demo

Execute the demo script to see all concepts in action:

```bash
python 11/demo/timeseries_hands_on_demo.py
```

The script includes realistic examples and sample data generation, so you can run it immediately without external data files.

## Key Learning Outcomes

After completing this demo, you should be able to:

- Create and manipulate time series data with pandas
- Understand the relationship between timestamps, periods, and frequencies
- Perform resampling operations for different analytical needs
- Apply moving window functions for trend and volatility analysis
- Handle time zones and calendar arithmetic
- Analyze real-world temporal datasets effectively

## Extension Activities

1. **Real Data Practice**: Download stock price data and reproduce the financial analysis examples
2. **Custom Frequencies**: Experiment with business calendar frequencies for your domain
3. **Seasonal Analysis**: Add seasonal decomposition to the financial time series
4. **Irregular Data**: Practice with unevenly spaced time series data
5. **Visualization**: Create time series plots to visualize trends and patterns

## Connection to Course Materials

This demo directly implements concepts from:
- **McKinney Chapter 11**: Time Series (comprehensive coverage)
- **Course Lecture 11**: Applied time series analysis
- **Previous lectures**: Building on pandas DataFrame and Series operations

The time series skills developed here are fundamental for:
- Financial data analysis
- Sensor and IoT data processing
- Economic and business analytics
- Scientific time series research
- Web analytics and user behavior analysis

Time series analysis is one of the most practical and widely-used data science skills across industries!