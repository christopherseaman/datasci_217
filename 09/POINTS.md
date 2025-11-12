# Time Series Analysis: Temporal Data and Trends

## Understanding Time Series Data

Time series data is characterized by observations collected over time, where the order and timing of observations matter. Unlike cross-sectional data, time series data has a natural temporal structure that we can exploit for analysis. Understanding this temporal structure is crucial for effective time series analysis.

- Time series data has temporal ordering that matters - the sequence of observations contains important information
- Observations are often dependent on previous values - today's stock price depends on yesterday's price
- Time series data often contains patterns like trends, seasonality, and cycles that can be identified and analyzed
- The key insight: time series analysis is about understanding how things change over time, not just what they are
- Real-world time series data is messy - missing values, irregular intervals, and outliers are common

## Types of Time Series

Different types of time series require different analytical approaches. Understanding the characteristics of your time series data helps you choose appropriate analysis methods and identify potential issues.

- Regular time series have fixed intervals (daily, hourly, monthly) - these are easiest to analyze and most common in practice
- Irregular time series have variable intervals (event-based) - these require special handling and interpolation techniques
- Seasonal time series have patterns that repeat over time - these require seasonal decomposition and adjustment
- Trending time series have long-term directional movement - these require trend analysis and detrending
- Stationary time series have statistical properties that don't change over time - these are ideal for many analytical methods

# Date and Time Data Types

Working with dates and times in Python requires understanding the datetime module and pandas' powerful time series capabilities. The key is knowing when to use Python's datetime objects versus pandas' time series features.

- Python's datetime module provides basic date and time functionality - use it for simple date arithmetic and formatting
- pandas' DatetimeIndex is optimized for time series analysis - it provides powerful indexing, resampling, and time-based operations
- pd.to_datetime() converts strings to datetime objects with intelligent parsing - it handles many common date formats automatically
- pd.date_range() creates sequences of dates with specified frequencies - essential for creating time series data
- Always use pandas for time series analysis - it's much more powerful than basic datetime operations

## Python datetime Module

Python's datetime module provides the foundation for working with dates and times. Understanding these basics is essential before moving to pandas' more advanced time series features.

- datetime.now() gets the current date and time - useful for timestamps and logging
- datetime(year, month, day) creates specific dates - useful for date arithmetic and comparisons
- strptime() parses strings into datetime objects - you specify the format string to match your data
- strftime() formats datetime objects into strings - essential for creating human-readable dates
- timedelta objects represent time differences - use them for date arithmetic and time-based calculations

## pandas DatetimeIndex

pandas' DatetimeIndex is optimized for time series operations and provides powerful indexing, resampling, and time-based operations.

- pd.to_datetime() converts strings, lists, or Series to datetime objects with intelligent parsing
- pd.date_range() creates sequences of dates with specified frequencies
- pd.DatetimeIndex() creates a datetime index
- df.set_index('date') converts a date column to the index, enabling all pandas time series features
- When setting datetime index for DataFrames with multiple rows per date, sort the index first: df.sort_index() to enable reliable date range selection

## Date Range Generation

pandas provides flexible date range generation for creating regular time series.

- pd.date_range() creates sequences of dates with specified frequencies - 'D' (daily), 'W' (weekly), 'ME' (month end), etc.
- pd.bdate_range() creates business day sequences, excluding weekends
- Frequency codes: 'D' (daily), 'B' (business days), 'W-MON' (weekly on Monday), 'MS' (month start), 'QS' (quarter start), 'h' (hourly)
- Note: Both 'H' and 'h' work for hourly frequency, but 'h' is the recommended form (uppercase 'H' is deprecated in pandas 2.0+)

## Frequency Inference

You can infer the frequency of a time series and convert between frequencies.

- pd.infer_freq(ts.index) infers frequency from a time series index
- ts.asfreq(freq) converts a time series to a specific frequency
- ts.resample(freq).asfreq() combines resampling with frequency conversion

## Shifting and Lagging

Shifting allows you to create lagged or leading versions of your time series, essential for analyzing changes over time.

- ts.shift(1) shifts by 1 period (lag) - creates a version with previous values
- ts.shift(-1) shifts by -1 period (lead) - creates a version with future values
- ts.diff() calculates first difference - day-to-day changes
- ts.pct_change() calculates percentage change
- ts.shift(1, freq='D') shifts by 1 day with timestamp adjustment

## Time Zone Handling

pandas provides time zone localization and conversion for timezone-aware datetime objects.

- Best Practice: Use UTC as your base timezone. UTC has no daylight saving time, avoiding ambiguity. Store data in UTC, convert to local timezones only when needed for display or analysis
- ts.index.tz_localize('UTC') adds timezone to naive datetime
- ts.index.tz_convert('US/Eastern') converts timezone
- pd.Timestamp.now(tz='UTC') gets current time in timezone
- pd.date_range(..., tz='UTC') creates timezone-aware date range

# LIVE DEMO!

# Time Series Indexing and Selection

## Basic Time Series Selection

pandas provides intuitive ways to select data from time series using string-based indexing. You can write "2023" and pandas knows you mean "all of 2023".

- ts['2023-01-01'] selects a specific date
- ts['2023-01-01':'2023-01-31'] selects a date range
- ts['2023'] selects the entire year
- ts['2023-01'] selects a specific month
- ts.loc['2023-01-01'] for label-based selection
- ts.iloc[0:10] for position-based selection

## Advanced Time Series Selection

For time series with time components, you can select based on time of day. This is useful for selecting data from business hours or specific times of day.

- ts.between_time('09:00', '17:00') selects time range
- ts.at_time('12:00') selects specific time
- ts.loc[:start_date + pd.Timedelta(days=9)] selects first 10 days
- ts.loc[end_date - pd.Timedelta(days=9):] selects last 10 days
- ts.truncate(before='2023-06-01') truncates before date (requires sorted index)
- ts.truncate(after='2023-06-30') truncates after date (requires sorted index)

# Resampling and Frequency Conversion

Resampling converts time series from one frequency to another. **Downsampling** aggregates higher frequency data to lower frequency (e.g., daily to monthly). **Upsampling** converts lower frequency to higher frequency (e.g., monthly to daily), often introducing missing values.

## Basic Resampling

The `resample()` method is the workhorse for frequency conversion, similar to `groupby()` but for time intervals.

- Frequency codes: 'D' (daily), 'W' (weekly), 'ME' (month end), 'Q' (quarterly), 'A' (annual), 'h' (hourly)
- ts.resample('D').mean() converts to daily averages
- ts.resample('ME').mean() converts to monthly averages (Month End)
- ts.resample('W').mean() converts to weekly averages

## Resampling with Different Aggregations

You can apply various aggregation functions when resampling, just like with `groupby()`. The syntax is the same, but instead of grouping by categories, you're grouping by time intervals.

- Important Note: When resampling DataFrames with non-numeric columns, use `df.select_dtypes(include=[np.number])` or specify columns in `.agg()`
- ts.resample('D').mean() - mean aggregation
- ts.resample('D').sum() - sum aggregation
- ts.resample('D').max() - maximum aggregation
- ts.resample('D').min() - minimum aggregation
- ts.resample('D').std() - standard deviation
- ts.resample('D').agg(['mean', 'std', 'min', 'max']) - multiple aggregations

# LIVE DEMO!

# Rolling Window Operations

Rolling window functions compute statistics over a fixed-size window that moves through the time series. This is useful for smoothing noisy data and identifying trends.

## Basic Rolling Operations

The `rolling()` method creates a rolling window object that can be used with various aggregation functions.

- ts.rolling(window=5) creates a 5-period rolling window
- ts.rolling(window=5).mean() - rolling mean
- ts.rolling(window=5).std() - rolling standard deviation
- ts.rolling(window=5).sum() - rolling sum
- ts.rolling(window=5).min() - rolling minimum
- ts.rolling(window=5).max() - rolling maximum

## Advanced Rolling Operations

Rolling windows can be centered, have minimum periods, and use custom functions. Centered windows look both backward and forward from each point. Minimum periods allow calculations even before you have a full window.

- ts.rolling(window=5, center=True) - centered rolling window
- ts.rolling(window=5, min_periods=3) - minimum periods required
- ts.rolling(window=5).quantile(0.5) - rolling median
- ts.rolling(window=5).apply(custom_func) - custom rolling function
- ts.expanding() - expanding window (from start to current)
- ts.ewm(span=5) - exponentially weighted moving average

## Exponentially Weighted Functions

Exponentially weighted functions give more weight to recent observations, making them more responsive to recent changes.

- ts.ewm(span=5).mean() - exponentially weighted moving average
- ts.ewm(alpha=0.3).mean() - EWM with alpha parameter
- ts.ewm(halflife=2).mean() - EWM with half-life
- ts.ewm(span=5).std() - exponentially weighted standard deviation

# Time Series Visualization

Visualization is essential for understanding time series data. A good plot can reveal patterns, trends, and anomalies that summary statistics miss.

## Basic Time Series Plots

Creating effective time series visualizations helps identify patterns, trends, and anomalies. The most common visualization is a simple line plot showing values over time.

- ts.plot() - basic line plot of time series
- ts.plot(figsize=(12, 6)) - plot with custom figure size
- ts.plot(title='Title') - plot with title
- ts.plot(style='-', marker='o') - plot with custom style and markers
- ax = ts.plot() - get axes for further customization
- Overlay rolling means on raw data to smooth noise and reveal underlying trends

## Visualizing Time Series Components

Real-world time series data often contains multiple components: trend, seasonality, and noise. Visualizing these components separately helps understand the underlying patterns.

- Create separate plots for each component (trend, seasonal, noise) to see how they contribute to the overall pattern
- Trend component shows long-term direction
- Seasonal component shows repeating patterns
- Noise component shows random variation
- For advanced seasonal decomposition techniques (like STL decomposition), see [BONUS.md](BONUS.md)

# LIVE DEMO!
