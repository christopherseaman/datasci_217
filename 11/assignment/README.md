# Assignment 11: Time Series Analysis with pandas

## Overview

This assignment focuses on applying time series analysis techniques using pandas, covering the core concepts from McKinney Chapter 11. You'll work with real-world temporal data to demonstrate mastery of date/time operations, resampling, and moving window functions.

## Learning Objectives

By completing this assignment, you will:
- Manipulate date and time data using pandas datetime functionality
- Create and analyze time series with appropriate frequency settings
- Perform resampling operations for different analytical purposes
- Apply moving window functions to identify trends and patterns
- Handle time zones and period-based calculations
- Conduct practical time series analysis on realistic datasets

## Assignment Structure

### Part 1: Time Series Fundamentals (25 points)
- Create time series data with various frequencies
- Demonstrate indexing and selection operations
- Handle missing dates and irregular data

### Part 2: Resampling and Aggregation (25 points)
- Downsample high-frequency data with different aggregation methods
- Upsample low-frequency data with appropriate interpolation
- Compare different resampling strategies

### Part 3: Moving Window Analysis (25 points)
- Calculate rolling statistics (mean, std, min, max)
- Implement exponentially weighted moving averages
- Analyze rolling correlations between time series

### Part 4: Applied Analysis (25 points)
- Perform comprehensive analysis on provided dataset
- Generate insights using time series techniques
- Present findings with appropriate visualizations

## Data Files

Your assignment will use the following datasets:
- `stock_prices.csv`: Daily stock price data for multiple companies
- `weather_data.csv`: Hourly weather measurements over one year
- `sales_data.csv`: Monthly sales figures with seasonal patterns

## Submission Requirements

1. **Code File**: `main.py` - Your complete implementation
2. **Documentation**: Comments explaining your approach for each part
3. **Results**: Clear output demonstrating your analysis
4. **Testing**: Your code should pass all provided test cases

## Getting Started

1. Review the lecture materials on time series analysis
2. Run the hands-on demo to understand the concepts
3. Examine the provided data files to understand their structure
4. Implement each part step by step
5. Test your implementation thoroughly

## Key Concepts to Demonstrate

### DateTime Operations
- Converting strings to datetime objects
- Creating date ranges with specific frequencies
- Handling time zones and period arithmetic

### Time Series Indexing
- Selecting data by date ranges
- Filtering based on temporal conditions
- Working with partial date strings

### Resampling Techniques
- Downsampling with aggregation (mean, sum, OHLC)
- Upsampling with interpolation methods
- Custom resampling rules

### Moving Window Functions
- Rolling calculations for trend analysis
- Exponentially weighted operations
- Custom window functions for specific metrics

### Practical Applications
- Financial time series analysis
- Seasonal pattern identification
- Performance metric calculations

## Grading Criteria

### Code Quality (20%)
- Clean, readable, and well-documented code
- Proper use of pandas time series functionality
- Efficient implementation of operations

### Correctness (40%)
- Accurate implementation of time series operations
- Correct handling of edge cases
- Proper data type usage (datetime64, Period, etc.)

### Analysis Quality (25%)
- Appropriate choice of methods for each task
- Meaningful interpretation of results
- Effective use of time series techniques

### Presentation (15%)
- Clear output and formatting
- Logical organization of results
- Professional presentation of findings

## Tips for Success

1. **Understand the Data**: Examine the structure and patterns in your time series before analysis
2. **Choose Appropriate Frequencies**: Match your resampling frequency to your analytical goals
3. **Handle Missing Data**: Use appropriate methods for dealing with gaps in time series
4. **Validate Results**: Check that your calculations make sense in context
5. **Document Your Work**: Explain your reasoning for method choices

## Common Pitfalls to Avoid

- Forgetting to handle time zones when combining datasets
- Using inappropriate aggregation methods for resampling
- Ignoring the impact of missing data on moving window calculations
- Not considering the business meaning of your time series operations
- Mixing up data shifting vs. timestamp shifting operations

## Resources

- **Lecture 11**: Time Series Analysis concepts and examples
- **McKinney Chapter 11**: Comprehensive time series reference
- **pandas Documentation**: Official time series functionality guide
- **Demo Files**: Hands-on examples of all major operations

## Academic Integrity

- This is an individual assignment
- You may discuss concepts with classmates but code must be your own
- Properly cite any external resources used
- Follow course policies on collaboration and original work

Good luck with your time series analysis! This assignment will give you practical experience with one of the most important and widely-used areas of data science.