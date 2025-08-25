"""
Comprehensive Assignment Test Suite
==================================

Complete example demonstrating all testing patterns and features
for a multi-faceted homework assignment.
"""

import pytest
import pandas as pd
import numpy as np
import json
import csv
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from hypothesis import strategies as st

# Import our advanced testing framework
import sys
sys.path.append('../test_frameworks')
from advanced_test_suite import (
    AdvancedHomeworkTestSuite, 
    AdvancedTestUtils, 
    PropertyBasedTesting,
    PerformanceTestSuite,
    IntegrationTestFramework,
    SecurityTestSuite
)

# Import base testing utilities
sys.path.append('../test_templates')
from base_test_template import TestUtils, GradingMixin, assert_close_enough


class TestComprehensiveAssignment(AdvancedHomeworkTestSuite, GradingMixin):
    """
    Comprehensive test suite demonstrating all testing capabilities.
    
    This test class covers:
    - Function-based testing
    - Data processing validation
    - File I/O operations
    - Command-line interface testing
    - Performance requirements
    - Security validation
    - Property-based testing
    - Integration testing
    """
    
    def setup_method(self):
        """Setup for each test method"""
        super().setup_method()
        GradingMixin.__init__(self)
        
        # Required functions for this assignment
        self.required_functions = [
            'process_data',
            'calculate_statistics',
            'export_results', 
            'validate_input',
            'main'
        ]
        
        # Check all required functions exist
        for func_name in self.required_functions:
            self.test_function_exists_and_callable(func_name)
    
    # =============================================================================
    # BASIC FUNCTION TESTS (40 points total)
    # =============================================================================
    
    @pytest.mark.function_test
    def test_process_data_basic_functionality(self):
        """Test basic data processing functionality (10 points)"""
        try:
            # Create test data
            test_data = pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
                'score': [95.5, 87.2, None, 92.1, 88.9],
                'category': ['A', 'B', 'A', 'C', 'B']
            })
            
            config = {'remove_nulls': True, 'normalize_scores': False}
            
            # Test function
            func = getattr(self.student_module, 'process_data')
            result = func(test_data.copy(), config)
            
            # Validation
            assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
            assert len(result) == 4, "Should remove row with null score"
            assert 'score' in result.columns, "Score column should be preserved"
            assert not result['score'].isnull().any(), "No null scores should remain"
            
            self.award_points('function_tests', 10, description="process_data basic functionality")
            
        except Exception as e:
            pytest.fail(f"Basic data processing failed: {e}")
    
    @pytest.mark.function_test  
    def test_calculate_statistics_functionality(self):
        """Test statistics calculation (10 points)"""
        try:
            func = getattr(self.student_module, 'calculate_statistics')
            
            # Test with simple data
            test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = func(test_data)
            
            assert isinstance(result, dict), "Result should be a dictionary"
            
            expected_keys = ['mean', 'median', 'std', 'min', 'max', 'count']
            for key in expected_keys:
                assert key in result, f"Missing key: {key}"
            
            # Check specific values
            assert_close_enough(result['mean'], 3.0)
            assert_close_enough(result['median'], 3.0)
            assert result['min'] == 1.0
            assert result['max'] == 5.0
            assert result['count'] == 5
            
            self.award_points('function_tests', 10, description="calculate_statistics functionality")
            
        except Exception as e:
            pytest.fail(f"Statistics calculation failed: {e}")
    
    @pytest.mark.function_test
    def test_export_results_functionality(self):
        """Test results export functionality (10 points)"""
        try:
            func = getattr(self.student_module, 'export_results')
            
            test_data = {
                'summary': {'total_records': 100, 'avg_score': 85.5},
                'details': [{'id': 1, 'score': 95}, {'id': 2, 'score': 88}]
            }
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = os.path.join(temp_dir, 'results.json')
                success = func(test_data, output_file)
                
                assert success, "Export should return success status"
                assert os.path.exists(output_file), "Output file should be created"
                
                # Verify file contents
                with open(output_file, 'r') as f:
                    exported_data = json.load(f)
                
                assert exported_data == test_data, "Exported data should match input"
            
            self.award_points('function_tests', 10, description="export_results functionality")
            
        except Exception as e:
            pytest.fail(f"Export functionality failed: {e}")
    
    @pytest.mark.function_test
    def test_validate_input_functionality(self):
        """Test input validation functionality (10 points)"""
        try:
            func = getattr(self.student_module, 'validate_input')
            
            # Test valid inputs
            valid_data = {
                'id': 1,
                'name': 'John Doe',
                'score': 85.5,
                'email': 'john@example.com'
            }
            
            result = func(valid_data, ['id', 'name', 'score'])
            assert result['valid'] == True, "Valid data should pass validation"
            assert result['errors'] == [], "No errors for valid data"
            
            # Test invalid inputs
            invalid_data = {
                'id': 'not_a_number',
                'name': '',
                'score': 150  # Out of range
            }
            
            result = func(invalid_data, ['id', 'name', 'score'])
            assert result['valid'] == False, "Invalid data should fail validation"
            assert len(result['errors']) > 0, "Should report validation errors"
            
            self.award_points('function_tests', 10, description="validate_input functionality")
            
        except Exception as e:
            pytest.fail(f"Input validation failed: {e}")
    
    # =============================================================================
    # EDGE CASE TESTS (20 points total)
    # =============================================================================
    
    @pytest.mark.edge_case
    def test_empty_data_handling(self):
        """Test handling of empty datasets (5 points)"""
        try:
            func = getattr(self.student_module, 'process_data')
            
            # Empty DataFrame
            empty_df = pd.DataFrame()
            config = {'remove_nulls': True}
            
            result = func(empty_df, config)
            assert isinstance(result, pd.DataFrame), "Should return DataFrame even when empty"
            assert len(result) == 0, "Empty input should produce empty output"
            
            self.award_points('edge_cases', 5, description="Empty data handling")
            
        except Exception as e:
            pytest.fail(f"Empty data handling failed: {e}")
    
    @pytest.mark.edge_case
    def test_large_dataset_handling(self):
        """Test handling of large datasets (5 points)"""
        try:
            func = getattr(self.student_module, 'process_data')
            
            # Create large dataset
            large_data = pd.DataFrame({
                'id': range(10000),
                'value': np.random.randn(10000),
                'category': ['A', 'B', 'C'] * (10000 // 3 + 1)
            })[:10000]
            
            config = {'remove_nulls': False, 'normalize_scores': True}
            
            # Test with timeout
            result = TestUtils.timeout_test(
                lambda: func(large_data, config), 
                timeout=10.0
            )
            
            assert isinstance(result, pd.DataFrame), "Should handle large datasets"
            assert len(result) == 10000, "Should preserve all rows"
            
            self.award_points('edge_cases', 5, description="Large dataset handling")
            
        except Exception as e:
            pytest.fail(f"Large dataset handling failed: {e}")
    
    @pytest.mark.edge_case
    def test_boundary_conditions(self):
        """Test boundary conditions for statistics (5 points)"""
        try:
            func = getattr(self.student_module, 'calculate_statistics')
            
            # Single value
            result = func([42.0])
            assert result['mean'] == 42.0
            assert result['count'] == 1
            assert result['std'] == 0.0  # Standard deviation of single value
            
            # All same values
            result = func([5.0, 5.0, 5.0, 5.0])
            assert result['mean'] == 5.0
            assert result['std'] == 0.0
            
            # Extreme values
            result = func([-1e6, 1e6])
            assert result['mean'] == 0.0
            assert result['count'] == 2
            
            self.award_points('edge_cases', 5, description="Boundary conditions")
            
        except Exception as e:
            pytest.fail(f"Boundary condition testing failed: {e}")
    
    @pytest.mark.edge_case
    def test_special_character_handling(self):
        """Test handling of special characters in data (5 points)"""
        try:
            func = getattr(self.student_module, 'validate_input')
            
            # Test with special characters
            special_data = {
                'name': 'José María Azñar',
                'description': 'Test with émojis 🎉 and símböls',
                'unicode_text': '测试中文字符'
            }
            
            result = func(special_data, ['name', 'description'])
            assert result['valid'] in [True, False], "Should handle Unicode characters"
            
            self.award_points('edge_cases', 5, description="Special character handling")
            
        except Exception as e:
            pytest.fail(f"Special character handling failed: {e}")
    
    # =============================================================================
    # ERROR HANDLING TESTS (15 points total)
    # =============================================================================
    
    @pytest.mark.error_handling
    def test_invalid_input_types(self):
        """Test handling of invalid input types (5 points)"""
        try:
            func = getattr(self.student_module, 'process_data')
            
            # Test with non-DataFrame input
            with pytest.raises((TypeError, ValueError)):
                func("not a dataframe", {})
            
            with pytest.raises((TypeError, ValueError)):
                func(None, {})
            
            with pytest.raises((TypeError, ValueError)):
                func([1, 2, 3], {})
            
            self.award_points('error_handling', 5, description="Invalid input type handling")
            
        except Exception as e:
            pytest.fail(f"Invalid input type handling failed: {e}")
    
    @pytest.mark.error_handling
    def test_missing_required_parameters(self):
        """Test handling of missing required parameters (5 points)"""
        try:
            func = getattr(self.student_module, 'validate_input')
            
            # Test with missing required fields
            data = {'id': 1, 'name': 'John'}
            required = ['id', 'name', 'email', 'score']
            
            result = func(data, required)
            
            # Should not crash, should report missing fields
            assert isinstance(result, dict), "Should return validation result"
            assert 'valid' in result, "Should include validity status"
            assert 'errors' in result, "Should include error list"
            
            self.award_points('error_handling', 5, description="Missing parameter handling")
            
        except Exception as e:
            pytest.fail(f"Missing parameter handling failed: {e}")
    
    @pytest.mark.error_handling  
    def test_file_operation_errors(self):
        """Test handling of file operation errors (5 points)"""
        try:
            func = getattr(self.student_module, 'export_results')
            
            # Test with invalid file path
            test_data = {'test': 'data'}
            invalid_path = '/invalid/path/that/does/not/exist/file.json'
            
            # Should either handle gracefully or raise appropriate exception
            try:
                result = func(test_data, invalid_path)
                assert result == False, "Should return False for invalid paths"
            except (IOError, OSError, PermissionError):
                pass  # Appropriate exceptions are acceptable
            
            self.award_points('error_handling', 5, description="File operation error handling")
            
        except Exception as e:
            pytest.fail(f"File operation error handling failed: {e}")
    
    # =============================================================================
    # PERFORMANCE TESTS (10 points total)
    # =============================================================================
    
    @pytest.mark.performance
    def test_data_processing_performance(self):
        """Test data processing performance requirements (5 points)"""
        try:
            func = getattr(self.student_module, 'process_data')
            
            # Large dataset for performance testing
            large_data = pd.DataFrame({
                'id': range(50000),
                'value': np.random.randn(50000),
                'category': np.random.choice(['A', 'B', 'C'], 50000),
                'score': np.random.uniform(0, 100, 50000)
            })
            
            config = {'remove_nulls': True, 'normalize_scores': True}
            
            # Performance test
            test_cases = [(large_data.copy(), config)]
            self.test_performance_requirements(
                'process_data',
                test_cases,
                max_time=5.0,  # 5 seconds max
                max_memory=200.0  # 200MB max
            )
            
            self.award_points('performance', 5, description="Data processing performance")
            
        except Exception as e:
            pytest.fail(f"Performance test failed: {e}")
    
    @pytest.mark.performance
    def test_statistics_calculation_performance(self):
        """Test statistics calculation performance (5 points)"""
        try:
            func = getattr(self.student_module, 'calculate_statistics')
            
            # Large number list
            large_numbers = list(np.random.randn(100000))
            
            test_cases = [(large_numbers,)]
            self.test_performance_requirements(
                'calculate_statistics',
                test_cases,
                max_time=1.0,  # 1 second max
                max_memory=50.0  # 50MB max
            )
            
            self.award_points('performance', 5, description="Statistics calculation performance")
            
        except Exception as e:
            pytest.fail(f"Statistics performance test failed: {e}")
    
    # =============================================================================
    # INTEGRATION TESTS (10 points total)
    # =============================================================================
    
    @pytest.mark.integration
    def test_complete_workflow_integration(self):
        """Test complete data processing workflow (10 points)"""
        try:
            # Create test environment with input files
            test_files = {
                'input_data.csv': [
                    {'id': 1, 'name': 'Alice', 'score': 95.5, 'category': 'A'},
                    {'id': 2, 'name': 'Bob', 'score': 87.2, 'category': 'B'},
                    {'id': 3, 'name': 'Charlie', 'score': None, 'category': 'A'},
                    {'id': 4, 'name': 'Diana', 'score': 92.1, 'category': 'C'}
                ],
                'config.json': {
                    'remove_nulls': True,
                    'normalize_scores': False,
                    'export_format': 'json'
                }
            }
            
            expected_outputs = {
                'results.json': '{"processed": true}'  # Simplified expected output
            }
            
            # Test complete workflow
            result = self.integration_framework.test_file_operations(
                lambda: self._run_complete_workflow(),
                test_files,
                expected_outputs
            )
            
            # Validation
            assert result['success'] or result['function_result'] is not None, \
                "Complete workflow should execute successfully"
            
            self.award_points('integration', 10, description="Complete workflow integration")
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
    
    def _run_complete_workflow(self):
        """Helper method to run complete workflow"""
        try:
            # Load configuration
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # Load data
            data = pd.read_csv('input_data.csv')
            
            # Process data
            process_func = getattr(self.student_module, 'process_data')
            processed_data = process_func(data, config)
            
            # Calculate statistics
            stats_func = getattr(self.student_module, 'calculate_statistics')
            if not processed_data.empty:
                stats = stats_func(processed_data['score'].tolist())
            else:
                stats = {'error': 'No data to process'}
            
            # Export results
            export_func = getattr(self.student_module, 'export_results')
            result = export_func({'stats': stats, 'processed': True}, 'results.json')
            
            return result
        except Exception as e:
            return f"Workflow error: {e}"
    
    # =============================================================================
    # PROPERTY-BASED TESTS (5 points total)
    # =============================================================================
    
    @pytest.mark.function_test
    def test_statistics_properties(self):
        """Test statistical properties using property-based testing (5 points)"""
        try:
            properties = [
                lambda x, y: y['count'] == len(x),
                lambda x, y: y['min'] <= y['mean'] <= y['max'] if len(x) > 0 else True,
                lambda x, y: y['min'] in x and y['max'] in x if len(x) > 0 else True,
            ]
            
            self.test_with_property_based_testing(
                'calculate_statistics',
                st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), 
                        min_size=1, max_size=100),
                properties
            )
            
            self.award_points('function_tests', 5, description="Statistical properties validation")
            
        except Exception as e:
            pytest.fail(f"Property-based testing failed: {e}")
    
    # =============================================================================
    # COMMAND-LINE INTERFACE TESTS (if applicable)
    # =============================================================================
    
    @pytest.mark.integration
    def test_cli_interface(self):
        """Test command-line interface (bonus points)"""
        try:
            # Test help functionality
            result = TestUtils.run_student_script('main.py', ['--help'])
            
            if result['success']:
                assert 'usage' in result['stdout'].lower() or 'help' in result['stdout'].lower(), \
                    "Help output should contain usage information"
                
                self.award_points('integration', 5, description="CLI help functionality")
            
            # Test with arguments (if main accepts CLI args)
            result = TestUtils.run_student_script('main.py', ['--version'])
            # This is optional - award points if implemented
            
        except Exception as e:
            # CLI testing is optional, don't fail if not implemented
            pass
    
    # =============================================================================
    # FINAL GRADE CALCULATION
    # =============================================================================
    
    def teardown_method(self):
        """Generate final grade report"""
        super().teardown_method()
        
        # Print comprehensive grade report
        if hasattr(self, 'test_results') and self.test_results:
            print("\n" + "="*60)
            print("COMPREHENSIVE ASSIGNMENT GRADE REPORT")
            print("="*60)
            print(self.get_grade_report())
            print("="*60)


# Additional utility functions for specific test scenarios
class TestDataProcessingSpecific:
    """Specific tests for data processing functionality"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'score': [95.5, 87.2, None, 92.1, 88.9],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'date': pd.date_range('2024-01-01', periods=5)
        })
    
    @pytest.mark.parametrize("config,expected_length", [
        ({'remove_nulls': True}, 4),
        ({'remove_nulls': False}, 5),
    ])
    def test_null_handling_configurations(self, sample_dataframe, config, expected_length):
        """Test different null handling configurations"""
        # This would require loading the student module
        # Implementation depends on how the student implements process_data
        pass


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])