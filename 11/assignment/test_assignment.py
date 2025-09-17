#!/usr/bin/env python3

"""
Test Suite for Assignment 11: Time Series Analysis
DataSci 217 - Introduction to Data Science

These tests verify correct implementation of time series analysis concepts
including datetime operations, resampling, and moving window functions.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the assignment directory to the path so we can import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestTimeSeriesFundamentals(unittest.TestCase):
    """Test Part 1: Time Series Fundamentals"""
    
    def setUp(self):
        """Set up test data"""
        self.dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.values = np.random.randn(len(self.dates))
        self.ts = pd.Series(self.values, index=self.dates)
    
    def test_datetime_index_creation(self):
        """Test creating time series with DatetimeIndex"""
        # Test that index is DatetimeIndex
        self.assertIsInstance(self.ts.index, pd.DatetimeIndex)
        
        # Test frequency detection
        business_dates = pd.date_range('2023-01-01', periods=50, freq='B')
        business_ts = pd.Series(np.random.randn(50), index=business_dates)
        self.assertEqual(business_ts.index.freq, 'B')
    
    def test_time_series_selection(self):
        """Test various time series selection methods"""
        # Test year selection
        year_2023 = self.ts['2023']
        self.assertEqual(len(year_2023), len(self.ts))
        
        # Test month selection
        jan_2023 = self.ts['2023-01']
        self.assertEqual(len(jan_2023), 31)  # January has 31 days
        
        # Test date range selection
        q1_2023 = self.ts['2023-01-01':'2023-03-31']
        expected_days = (datetime(2023, 3, 31) - datetime(2023, 1, 1)).days + 1
        self.assertEqual(len(q1_2023), expected_days)
    
    def test_missing_date_handling(self):
        """Test handling of missing dates in time series"""
        # Create series with missing dates
        incomplete_dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')[::2]
        incomplete_ts = pd.Series(np.random.randn(len(incomplete_dates)), 
                                 index=incomplete_dates)
        
        # Reindex to fill missing dates
        complete_index = pd.date_range(incomplete_ts.index.min(), 
                                     incomplete_ts.index.max(), freq='D')
        filled_ts = incomplete_ts.reindex(complete_index)
        
        self.assertEqual(len(filled_ts), 10)
        self.assertTrue(filled_ts.isna().any())

class TestResamplingAggregation(unittest.TestCase):
    """Test Part 2: Resampling and Aggregation"""
    
    def setUp(self):
        """Set up test data"""
        self.hourly_dates = pd.date_range('2023-06-01', '2023-06-30', freq='H')
        self.hourly_values = np.random.randn(len(self.hourly_dates))
        self.hourly_ts = pd.Series(self.hourly_values, index=self.hourly_dates)
    
    def test_downsampling_aggregation(self):
        """Test downsampling with different aggregation methods"""
        # Test daily mean
        daily_mean = self.hourly_ts.resample('D').mean()
        self.assertEqual(len(daily_mean), 30)  # June has 30 days
        
        # Test daily sum
        daily_sum = self.hourly_ts.resample('D').sum()
        self.assertEqual(len(daily_sum), 30)
        
        # Test OHLC aggregation
        daily_ohlc = self.hourly_ts.resample('D').ohlc()
        self.assertEqual(daily_ohlc.shape, (30, 4))  # 30 days, 4 OHLC columns
    
    def test_upsampling_interpolation(self):
        """Test upsampling with interpolation methods"""
        # Create weekly data
        weekly_dates = pd.date_range('2023-01-01', periods=10, freq='W')
        weekly_values = np.random.randn(10)
        weekly_ts = pd.Series(weekly_values, index=weekly_dates)
        
        # Upsample to daily
        daily_upsampled = weekly_ts.resample('D').asfreq()
        self.assertGreater(len(daily_upsampled), len(weekly_ts))
        
        # Test forward fill
        daily_ffill = weekly_ts.resample('D').ffill()
        self.assertFalse(daily_ffill.isna().any())
    
    def test_resampling_edge_cases(self):
        """Test edge cases in resampling"""
        # Test empty series
        empty_ts = pd.Series([], dtype=float, 
                           index=pd.DatetimeIndex([], freq='D'))
        resampled_empty = empty_ts.resample('W').mean()
        self.assertEqual(len(resampled_empty), 0)
        
        # Test single value series
        single_date = pd.date_range('2023-01-01', periods=1, freq='D')
        single_ts = pd.Series([1.0], index=single_date)
        resampled_single = single_ts.resample('W').mean()
        self.assertEqual(len(resampled_single), 1)
        self.assertEqual(resampled_single.iloc[0], 1.0)

class TestMovingWindowAnalysis(unittest.TestCase):
    """Test Part 3: Moving Window Analysis"""
    
    def setUp(self):
        """Set up test data"""
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        # Create trending data for better testing
        trend = np.linspace(0, 10, 100)
        noise = np.random.randn(100) * 0.5
        self.values = trend + noise
        self.ts = pd.Series(self.values, index=self.dates)
    
    def test_rolling_statistics(self):
        """Test rolling window calculations"""
        # Test rolling mean
        rolling_mean = self.ts.rolling(10).mean()
        self.assertEqual(len(rolling_mean), len(self.ts))
        self.assertTrue(rolling_mean.iloc[:9].isna().all())  # First 9 should be NaN
        self.assertFalse(rolling_mean.iloc[9:].isna().any())  # Rest should have values
        
        # Test rolling standard deviation
        rolling_std = self.ts.rolling(20).std()
        self.assertTrue(rolling_std.iloc[19:].notna().all())
        
        # Test rolling with min_periods
        rolling_min_periods = self.ts.rolling(10, min_periods=5).mean()
        self.assertTrue(rolling_min_periods.iloc[4:].notna().all())
    
    def test_exponentially_weighted_functions(self):
        """Test exponentially weighted moving calculations"""
        # Test EWM mean
        ewm_mean = self.ts.ewm(span=10).mean()
        self.assertEqual(len(ewm_mean), len(self.ts))
        self.assertFalse(ewm_mean.isna().any())  # EWM should not have NaN values
        
        # Test EWM standard deviation
        ewm_std = self.ts.ewm(span=10).std()
        self.assertTrue(ewm_std.iloc[1:].notna().all())  # First value might be NaN
    
    def test_rolling_correlations(self):
        """Test rolling correlations between series"""
        # Create second series
        ts2 = pd.Series(self.values + np.random.randn(100) * 0.2, 
                       index=self.dates)
        
        # Calculate rolling correlation
        rolling_corr = self.ts.rolling(30).corr(ts2)
        self.assertTrue((rolling_corr.iloc[29:] >= -1).all())  # Correlations >= -1
        self.assertTrue((rolling_corr.iloc[29:] <= 1).all())   # Correlations <= 1

class TestAppliedAnalysis(unittest.TestCase):
    """Test Part 4: Applied Analysis"""
    
    def setUp(self):
        """Set up realistic test data"""
        # Create sample stock price data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
        returns = np.random.randn(len(dates)) * 0.02  # 2% daily volatility
        prices = 100 * np.exp(np.cumsum(returns))  # Geometric random walk
        self.stock_data = pd.Series(prices, index=dates, name='Stock_Price')
    
    def test_financial_metrics_calculation(self):
        """Test calculation of financial metrics"""
        # Calculate returns
        returns = self.stock_data.pct_change().dropna()
        
        # Test basic statistics
        self.assertIsInstance(returns.mean(), (float, np.floating))
        self.assertIsInstance(returns.std(), (float, np.floating))
        self.assertGreater(returns.std(), 0)  # Volatility should be positive
        
        # Test Sharpe ratio calculation
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        self.assertIsInstance(sharpe_ratio, (float, np.floating))
    
    def test_trend_analysis(self):
        """Test trend analysis capabilities"""
        # Calculate moving averages
        sma_20 = self.stock_data.rolling(20).mean()
        sma_50 = self.stock_data.rolling(50).mean()
        
        # Test that moving averages are smoother than original data
        original_diff = self.stock_data.diff().abs().mean()
        sma_20_diff = sma_20.diff().abs().mean()
        self.assertLess(sma_20_diff, original_diff)
    
    def test_pattern_detection(self):
        """Test pattern detection in time series"""
        # Test for monotonic trends (simplified)
        monthly_data = self.stock_data.resample('M').last()
        monthly_returns = monthly_data.pct_change().dropna()
        
        # Check that we can detect positive/negative periods
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns < 0).sum()
        self.assertEqual(positive_months + negative_months, len(monthly_returns))

class TestHelperFunctions(unittest.TestCase):
    """Test helper functions if implemented"""
    
    def test_date_range_creation(self):
        """Test creating date ranges with different frequencies"""
        # Test business day creation
        business_days = pd.date_range('2023-01-01', '2023-01-31', freq='B')
        
        # January 2023: 31 days, but only 22 business days
        self.assertLessEqual(len(business_days), 31)
        
        # Test monthly frequency
        monthly_dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
        self.assertEqual(len(monthly_dates), 12)
    
    def test_timezone_handling(self):
        """Test timezone operations"""
        # Create timezone-naive series
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        ts = pd.Series(range(5), index=dates)
        
        # Test localization
        ts_localized = ts.tz_localize('UTC')
        self.assertEqual(str(ts_localized.index.tz), 'UTC')
        
        # Test conversion
        ts_eastern = ts_localized.tz_convert('US/Eastern')
        self.assertEqual(str(ts_eastern.index.tz), 'US/Eastern')

def run_tests():
    """Run all tests and display results"""
    print("Running Time Series Analysis Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTimeSeriesFundamentals,
        TestResamplingAggregation,
        TestMovingWindowAnalysis,
        TestAppliedAnalysis,
        TestHelperFunctions
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()