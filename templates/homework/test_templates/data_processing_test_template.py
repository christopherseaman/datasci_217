"""
Data Processing Assignment Test Template
=======================================

Template for testing data processing assignments involving pandas, numpy,
file handling, and data analysis tasks.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
from pathlib import Path
from base_test_template import (
    TestHomeworkBase, TestConfig, TestUtils,
    assert_function_exists, assert_file_contains
)


class TestDataProcessingAssignment(TestHomeworkBase):
    """Test suite for data processing assignments"""
    
    # Define expected functions
    REQUIRED_FUNCTIONS = [
        'load_data',
        'clean_data',
        'analyze_data',
        'save_results'
    ]
    
    def setup_method(self):
        """Setup test data and verify required functions"""
        super().setup_method()
        
        # Verify all required functions exist
        for func_name in self.REQUIRED_FUNCTIONS:
            assert_function_exists(self.student_module, func_name)
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, None, 28, 35],
            'salary': [50000, 60000, 55000, None, 70000],
            'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
        })
        
        # Create test CSV file
        self.test_csv_path = 'test_data.csv'
        self.sample_data.to_csv(self.test_csv_path, index=False)
    
    def teardown_method(self):
        """Cleanup test files"""
        super().teardown_method()
        # Clean up test files
        test_files = [self.test_csv_path, 'output.csv', 'results.json']
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
    
    @pytest.mark.function_test
    def test_load_data_function(self):
        """Test data loading functionality"""
        load_data = self.student_module.load_data
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 4
        
        try:
            # Test loading CSV file
            result = TestUtils.timeout_test(
                lambda: load_data(self.test_csv_path),
                timeout=TestConfig.TIMEOUT['file_operation']
            )
            
            # Verify it's a DataFrame
            assert isinstance(result, pd.DataFrame), \
                "load_data should return a pandas DataFrame"
            points_earned += total_points // 3
            
            # Verify correct shape
            assert result.shape == self.sample_data.shape, \
                f"Expected shape {self.sample_data.shape}, got {result.shape}"
            points_earned += total_points // 3
            
            # Verify column names
            assert list(result.columns) == list(self.sample_data.columns), \
                "Column names don't match expected"
            points_earned += total_points // 3
            
        except Exception as e:
            pytest.fail(f"Data loading failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="Data loading functionality")
    
    @pytest.mark.function_test
    def test_clean_data_function(self):
        """Test data cleaning functionality"""
        clean_data = self.student_module.clean_data
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 4
        
        try:
            # Test cleaning with missing values
            dirty_data = self.sample_data.copy()
            result = TestUtils.timeout_test(
                lambda: clean_data(dirty_data),
                timeout=TestConfig.TIMEOUT['function_call']
            )
            
            # Verify it's a DataFrame
            assert isinstance(result, pd.DataFrame), \
                "clean_data should return a pandas DataFrame"
            points_earned += total_points // 4
            
            # Check that missing values are handled
            # (Either filled or rows removed)
            original_nulls = dirty_data.isnull().sum().sum()
            result_nulls = result.isnull().sum().sum()
            
            if original_nulls > 0:
                assert result_nulls <= original_nulls, \
                    "Cleaning should reduce or eliminate null values"
                points_earned += total_points // 2
            else:
                points_earned += total_points // 2
            
            # Verify data integrity
            assert len(result.columns) >= len(dirty_data.columns) * 0.8, \
                "Too many columns removed during cleaning"
            points_earned += total_points // 4
            
        except Exception as e:
            pytest.fail(f"Data cleaning failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="Data cleaning functionality")
    
    @pytest.mark.function_test
    def test_analyze_data_function(self):
        """Test data analysis functionality"""
        analyze_data = self.student_module.analyze_data
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 4
        
        try:
            # Test analysis with sample data
            clean_sample = self.sample_data.dropna()
            result = TestUtils.timeout_test(
                lambda: analyze_data(clean_sample),
                timeout=TestConfig.TIMEOUT['function_call']
            )
            
            # Result should be a dictionary with analysis results
            assert isinstance(result, dict), \
                "analyze_data should return a dictionary"
            points_earned += total_points // 4
            
            # Check for common analysis results
            expected_keys = ['mean_age', 'mean_salary', 'department_counts']
            found_keys = 0
            for key in expected_keys:
                if key in result:
                    found_keys += 1
            
            points_earned += (total_points * 3 // 4) * (found_keys / len(expected_keys))
            
        except Exception as e:
            pytest.fail(f"Data analysis failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="Data analysis functionality")
    
    @pytest.mark.function_test
    def test_save_results_function(self):
        """Test results saving functionality"""
        save_results = self.student_module.save_results
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 4
        
        try:
            # Test saving analysis results
            test_results = {
                'mean_age': 29.5,
                'mean_salary': 58750.0,
                'department_counts': {'IT': 3, 'HR': 1, 'Finance': 1}
            }
            
            output_file = 'test_output.json'
            TestUtils.timeout_test(
                lambda: save_results(test_results, output_file),
                timeout=TestConfig.TIMEOUT['file_operation']
            )
            
            # Verify file was created
            assert os.path.exists(output_file), \
                f"Output file {output_file} was not created"
            points_earned += total_points // 2
            
            # Verify file contents
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data == test_results, \
                "Saved data doesn't match expected results"
            points_earned += total_points // 2
            
            # Cleanup
            os.remove(output_file)
            
        except Exception as e:
            pytest.fail(f"Results saving failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="Results saving functionality")
    
    @pytest.mark.edge_case
    def test_edge_cases(self):
        """Test edge cases for data processing"""
        points_earned = 0
        total_points = TestConfig.POINTS['edge_cases']
        
        # Test with empty DataFrame
        try:
            empty_df = pd.DataFrame()
            clean_data = self.student_module.clean_data
            
            result = clean_data(empty_df)
            assert isinstance(result, pd.DataFrame), \
                "Should handle empty DataFrame gracefully"
            points_earned += total_points // 4
            
        except Exception as e:
            # Partial credit for attempting to handle empty data
            points_earned += total_points // 8
        
        # Test with all missing values
        try:
            all_na_df = pd.DataFrame({
                'col1': [None, None, None],
                'col2': [np.nan, np.nan, np.nan]
            })
            
            result = clean_data(all_na_df)
            # Should either return empty DataFrame or handle gracefully
            assert isinstance(result, pd.DataFrame), \
                "Should handle all-missing DataFrame"
            points_earned += total_points // 4
            
        except Exception:
            points_earned += total_points // 8
        
        # Test with very large dataset
        try:
            large_df = pd.DataFrame({
                'id': range(10000),
                'value': np.random.randn(10000)
            })
            
            result = TestUtils.timeout_test(
                lambda: clean_data(large_df),
                timeout=10.0
            )
            points_earned += total_points // 4
            
        except Exception:
            points_earned += total_points // 8
        
        # Test with unusual data types
        try:
            mixed_df = pd.DataFrame({
                'numbers': [1, 2, 3],
                'strings': ['a', 'b', 'c'],
                'dates': pd.date_range('2021-01-01', periods=3),
                'booleans': [True, False, True]
            })
            
            result = clean_data(mixed_df)
            points_earned += total_points // 4
            
        except Exception:
            points_earned += total_points // 8
        
        self.award_points('edge_cases', points_earned,
                         description="Edge cases handling")
    
    @pytest.mark.error_handling
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        points_earned = 0
        total_points = TestConfig.POINTS['error_handling']
        
        load_data = self.student_module.load_data
        clean_data = self.student_module.clean_data
        analyze_data = self.student_module.analyze_data
        save_results = self.student_module.save_results
        
        # Test loading non-existent file
        try:
            with pytest.raises((FileNotFoundError, IOError)):
                load_data('non_existent_file.csv')
            points_earned += total_points // 4
        except Exception:
            pass  # Partial credit for attempting error handling
        
        # Test cleaning invalid input
        try:
            with pytest.raises((TypeError, ValueError)):
                clean_data("not a dataframe")
            points_earned += total_points // 4
        except Exception:
            pass
        
        # Test analyzing invalid input
        try:
            with pytest.raises((TypeError, ValueError, AttributeError)):
                analyze_data("not a dataframe")
            points_earned += total_points // 4
        except Exception:
            pass
        
        # Test saving to invalid path
        try:
            with pytest.raises((IOError, OSError, PermissionError)):
                save_results({}, '/invalid/path/file.json')
            points_earned += total_points // 4
        except Exception:
            pass
        
        self.award_points('error_handling', points_earned,
                         description="Error handling for invalid inputs")
    
    def test_data_integrity(self):
        """Test that data processing maintains data integrity"""
        points_earned = 0
        quality_points = TestConfig.POINTS['code_quality']
        
        try:
            # Load and clean data
            load_data = self.student_module.load_data
            clean_data = self.student_module.clean_data
            
            original_data = load_data(self.test_csv_path)
            cleaned_data = clean_data(original_data.copy())
            
            # Check that essential data relationships are preserved
            if not cleaned_data.empty and not original_data.empty:
                # Check that no duplicate IDs were introduced
                if 'id' in cleaned_data.columns:
                    original_ids = set(original_data['id'].dropna())
                    cleaned_ids = set(cleaned_data['id'].dropna())
                    assert cleaned_ids.issubset(original_ids), \
                        "Data cleaning introduced new IDs"
                    points_earned += quality_points // 2
                
                # Check that data types are reasonable
                for col in cleaned_data.columns:
                    if col in original_data.columns:
                        orig_dtype = original_data[col].dtype
                        clean_dtype = cleaned_data[col].dtype
                        
                        # Allow conversion from object to numeric, but not the reverse
                        # unless it's intentional categorization
                        if orig_dtype in ['int64', 'float64'] and clean_dtype == 'object':
                            pytest.fail(f"Column {col} changed from numeric to object")
                
                points_earned += quality_points // 2
            
        except Exception as e:
            pytest.fail(f"Data integrity check failed: {e}")
        
        self.award_points('code_quality', points_earned,
                         description="Data integrity preservation")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])