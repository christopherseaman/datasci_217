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

pandas' DatetimeIndex is the foundation of time series analysis in Python. It provides powerful indexing, resampling, and time-based operations that make time series analysis much easier than working with basic datetime objects.

- pd.to_datetime() converts strings, lists, or Series to datetime objects with intelligent parsing
- pd.date_range() creates sequences of dates with specified frequencies - much more powerful than manual date creation
- DatetimeIndex provides time-based indexing - you can select data by date ranges, specific dates, or time periods
- df.set_index('date') converts a date column to the index - this enables all pandas time series features
- DatetimeIndex automatically handles time zones, business days, and other time-related complexities

## Date Range Generation

Creating sequences of dates is a common task in time series analysis. pandas provides powerful tools for generating date ranges with various frequencies and business rules.

- pd.date_range() creates sequences of dates with specified frequencies - daily, weekly, monthly, etc.
- pd.bdate_range() creates business day sequences - automatically excludes weekends and holidays
- Frequency strings like 'D' (daily), 'W' (weekly), 'M' (monthly) control the interval between dates
- Business frequency 'B' excludes weekends - essential for financial and business analysis
- Custom frequencies like 'W-MON' (weekly on Monday) provide fine-grained control over date generation

## Frequency Inference

Understanding the frequency of your time series data is crucial for proper analysis. pandas can automatically detect frequencies and convert between different frequencies as needed.

- pd.infer_freq() automatically detects the frequency of a time series - useful for understanding your data structure
- ts.asfreq() converts a time series to a specific frequency - this may require interpolation or aggregation
- ts.resample().asfreq() combines resampling with frequency conversion - more flexible than asfreq() alone
- Frequency inference helps identify irregular data and choose appropriate analysis methods
- Always check the inferred frequency before proceeding with time series analysis

## Shifting and Lagging

Shifting and lagging operations are fundamental to time series analysis. They allow you to compare current values with past values, calculate changes over time, and create features for machine learning models.

- ts.shift(1) creates a lagged version of your time series - essential for comparing current values with past values
- ts.shift(-1) creates a lead version - useful for forecasting and forward-looking analysis
- ts.diff() calculates first differences - this removes trends and makes data more stationary
- ts.pct_change() calculates percentage changes - essential for financial analysis and growth rates
- Shifting operations are fundamental to time series analysis - they enable trend analysis and forecasting

## Exponentially Weighted Functions

Exponentially weighted functions give more weight to recent observations, making them ideal for smoothing time series data and creating adaptive indicators.

- ts.ewm(span=5).mean() creates exponentially weighted moving averages - more responsive to recent changes than simple moving averages
- The span parameter controls the decay rate - smaller values make the function more responsive to recent changes
- ts.ewm(alpha=0.3).mean() uses the alpha parameter for direct control over the decay rate
- Exponentially weighted functions are perfect for smoothing noisy data while preserving recent trends
- Use these functions for creating adaptive indicators that respond to changing conditions

## Time Zone Handling

Time zone handling is crucial for global data analysis. Understanding how to work with time zones prevents common errors and ensures accurate temporal analysis.

- pd.to_datetime().dt.tz_localize() adds timezone information to naive datetime objects
- pd.to_datetime().dt.tz_convert() converts between timezones - essential for global data analysis
- Always be explicit about timezones - naive datetime objects can lead to serious errors
- Use UTC for storage and analysis, convert to local timezones only for display
- Time zone handling is particularly important for financial data and global business analysis

## Business Day Handling

Business day handling is essential for financial and business analysis. It automatically excludes weekends and holidays, providing more accurate business metrics.

- pd.bdate_range() creates business day sequences - automatically excludes weekends and holidays
- ts.resample('B') resamples data to business days - essential for financial analysis
- Business day handling is crucial for financial data where weekends and holidays don't exist
- Use business day frequencies for revenue analysis, trading data, and business performance metrics
- Always consider whether your analysis should use calendar days or business days

# LIVE DEMO!

# Time Series Indexing and Selection

Time series indexing allows you to select data based on time periods, making it easy to analyze specific time ranges or compare different periods.

- ts['2023-01-01'] selects data for a specific date - much more intuitive than positional indexing
- ts['2023-01-01':'2023-01-31'] selects data for a date range - essential for period analysis
- ts['2023'] selects data for an entire year - useful for annual analysis and comparisons
- ts['2023-01'] selects data for a specific month - perfect for monthly analysis
- Time-based indexing is one of the most powerful features of pandas time series

## Basic Time Series Selection

Basic time series selection uses date strings and ranges to select specific time periods. This is much more intuitive than positional indexing and makes your code more readable.

- Use date strings like '2023-01-01' to select specific dates - pandas automatically parses these strings
- Use date ranges like '2023-01-01':'2023-01-31' to select time periods - the colon creates inclusive ranges
- Use year strings like '2023' to select entire years - perfect for annual analysis
- Use month strings like '2023-01' to select specific months - useful for monthly comparisons
- Time-based selection is much more intuitive than positional indexing for time series data

## Advanced Time Series Selection

Advanced time series selection includes time-based filtering, truncation, and selection based on time of day. These features make it easy to analyze specific time periods or business hours.

- ts.between_time('09:00', '17:00') selects data within specific hours - perfect for business hour analysis
- ts.at_time('12:00') selects data at specific times - useful for daily snapshots
- ts.first('10D') and ts.last('10D') select the first or last N periods - useful for recent analysis
- ts.truncate() removes data before or after specific dates - useful for data cleaning
- Advanced selection features make it easy to analyze specific time periods or business conditions

# Resampling and Frequency Conversion

Resampling changes the frequency of your time series data, allowing you to analyze data at different time scales. This is essential for comparing data collected at different frequencies or creating summary statistics.

- Resampling is like changing the lens on your camera - you can zoom in for detail or zoom out for the big picture
- ts.resample('D') resamples to daily frequency - use this to convert higher frequency data to daily summaries
- ts.resample('M') resamples to monthly frequency - perfect for creating monthly reports and analysis
- ts.resample('W') resamples to weekly frequency - useful for weekly business analysis
- Resampling is essential for comparing data collected at different frequencies

## Basic Resampling

Basic resampling changes the frequency of your time series data using simple aggregation functions. This is the most common use case for resampling operations.

- Use ts.resample('D').mean() to convert hourly data to daily averages - this preserves the overall trend while reducing noise
- Use ts.resample('M').sum() to convert daily data to monthly totals - perfect for revenue and sales analysis
- Use ts.resample('W').max() to convert daily data to weekly maximums - useful for peak analysis
- Resampling automatically handles missing values and irregular intervals - much more robust than manual aggregation
- Always choose the appropriate aggregation function based on your analysis goals

## Resampling with Different Aggregations

Resampling with different aggregation functions allows you to create various summary statistics at different time scales. This is essential for comprehensive time series analysis.

- Use multiple aggregation functions like ['mean', 'std', 'min', 'max'] to create comprehensive summaries
- Custom aggregation functions let you implement business-specific logic for resampling
- Different aggregation functions answer different business questions - sum for totals, mean for averages, max for peaks
- Resampling with multiple functions creates hierarchical column structure - use this for comprehensive analysis
- Always consider which aggregation function makes sense for your specific data and analysis goals

# LIVE DEMO!

# Rolling Window Operations

Rolling window operations analyze data within sliding windows, revealing trends and patterns that might be hidden in the raw data. These operations are essential for smoothing noisy data and identifying trends.

- Rolling windows are like looking through a moving frame - you can see how things change over time by examining sliding windows
- ts.rolling(window=7).mean() creates a 7-day moving average - this smooths daily fluctuations to reveal weekly trends
- ts.rolling(window=30).std() creates a 30-day rolling standard deviation - this shows how volatility changes over time
- Rolling operations are perfect for smoothing noisy data while preserving important trends
- Use rolling windows to identify trends, seasonality, and anomalies in your time series data

## Basic Rolling Operations

Basic rolling operations calculate statistics within sliding windows, providing smoothed views of your time series data.

- ts.rolling(window=5).mean() creates a 5-period moving average - this smooths short-term fluctuations
- ts.rolling(window=5).std() creates a 5-period rolling standard deviation - this shows how volatility changes
- ts.rolling(window=5).sum() creates a 5-period rolling sum - useful for cumulative analysis
- Rolling operations automatically handle missing values and irregular intervals - much more robust than manual calculations
- Use rolling windows to smooth noisy data and identify underlying trends

## Advanced Rolling Operations

Advanced rolling operations include centered windows, minimum periods, and custom functions. These features provide fine-grained control over rolling calculations.

- ts.rolling(window=5, center=True) creates centered rolling windows - this provides more accurate trend analysis
- ts.rolling(window=5, min_periods=3) requires minimum periods for calculation - this handles missing data gracefully
- ts.rolling(window=5).apply(custom_func) applies custom functions to rolling windows - this enables complex analysis
- ts.expanding() creates expanding windows that grow over time - this shows cumulative effects
- Advanced rolling operations provide the flexibility needed for sophisticated time series analysis

# Command Line: Automation with Cron Jobs

Automation is essential for time series analysis that needs to run regularly. Cron jobs provide a reliable way to schedule and run time series analysis automatically.

- Cron jobs run commands at specified times - perfect for daily, weekly, or monthly time series analysis
- Use crontab -e to edit your cron schedule - this is the standard way to manage scheduled tasks
- Cron schedule format: minute hour day month dayofweek command - learn this format for scheduling
- Always test your cron jobs manually before scheduling them - debugging scheduled tasks is much harder
- Use logging and error handling in your cron jobs - this helps identify and fix problems automatically

## Cron Job Basics

Cron jobs provide a reliable way to schedule and run time series analysis automatically. Understanding the basics of cron scheduling is essential for automated analysis.

- crontab -e edits your personal cron schedule - this is where you add scheduled tasks
- crontab -l lists your current cron jobs - use this to verify your schedule
- Cron schedule format: * * * * * command (minute hour day month dayofweek)
- Use specific times like 0 2 * * * for daily at 2 AM, or */15 * * * * for every 15 minutes
- Always test your commands manually before scheduling them - cron jobs run in a minimal environment

## Python Scripts for Cron Jobs

Python scripts for cron jobs require careful setup to ensure they run reliably in the cron environment. This includes proper environment setup, logging, and error handling.

- Use absolute paths for all file references - cron jobs run in a minimal environment without your usual PATH
- Set up logging to track script execution and identify problems - this is crucial for debugging scheduled tasks
- Use try-except blocks to handle errors gracefully - cron jobs that crash silently are hard to debug
- Activate your conda environment explicitly in the script - cron jobs don't have access to your shell environment
- Test your scripts in the cron environment before scheduling them - the environment is different from your interactive shell

## Advanced Cron Job Management

Advanced cron job management includes environment setup, logging, and monitoring. These features ensure your automated analysis runs reliably and can be debugged when problems occur.

- Create shell scripts that set up the environment before running Python - this ensures consistent execution
- Use logging to track script execution and identify problems - this is essential for debugging
- Set up monitoring to alert you when cron jobs fail - this prevents silent failures
- Use version control for your cron job scripts - this makes it easy to update and rollback changes
- Document your cron jobs and their purposes - this helps others understand and maintain your automation

# Time Series Visualization

Time series visualization is essential for understanding temporal patterns in your data. Different visualization techniques reveal different aspects of your time series data.

## Basic Time Series Plots

Basic time series plots show how your data changes over time, revealing trends, seasonality, and anomalies. These plots are the foundation of time series analysis.

- `ts.plot()` creates basic line plots of time series data - this is the most common visualization
- Customize plots with `figsize`, `title`, `xlabel`, `ylabel` parameters - essential for publication-quality figures
- Overlay rolling means on raw data to smooth noise and reveal underlying trends - this helps identify long-term patterns
- Use `ax = ts.plot()` to get axes for further customization - this enables advanced plot modifications
- Basic plots reveal patterns that summary statistics miss - visualization is essential for understanding time series data

## Visualizing Time Series Components

Real-world time series data often contains multiple components: trend, seasonality, and noise. Visualizing these components separately helps understand the underlying patterns.

- Real-world data combines trend, seasonal, and noise components - understanding each component is crucial for analysis
- Create separate plots for each component to see how they contribute to the overall pattern - this reveals the underlying structure
- Trend component shows long-term direction - use `np.linspace()` or rolling means to extract trends
- Seasonal component shows repeating patterns - use sine/cosine functions or seasonal decomposition to identify
- Noise component shows random variation - this is what remains after removing trend and seasonal patterns
- Visualizing components separately helps identify which patterns are meaningful and which are just noise
- For advanced seasonal decomposition techniques (STL, ARIMA), see BONUS.md - the main lecture covers basic component visualization

# LIVE DEMO!
