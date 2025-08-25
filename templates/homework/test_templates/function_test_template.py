"""
Function-Based Assignment Test Template
======================================

Template for testing homework assignments focused on implementing specific functions.
Includes patterns for testing function behavior, edge cases, and error handling.
"""

import pytest
import math
from base_test_template import (
    TestHomeworkBase, TestConfig, TestUtils,
    assert_function_exists, assert_close_enough
)


class TestFunctionAssignment(TestHomeworkBase):
    """Test suite for function-based assignments"""
    
    # Define expected functions for this assignment
    REQUIRED_FUNCTIONS = [
        'calculate_average',
        'find_maximum',
        'count_occurrences',
        'validate_input'
    ]
    
    def setup_method(self):
        """Setup for each test method"""
        super().setup_method()
        # Verify all required functions exist
        for func_name in self.REQUIRED_FUNCTIONS:
            assert_function_exists(self.student_module, func_name)
    
    @pytest.mark.function_test
    def test_calculate_average_basic(self):
        """Test basic average calculation"""
        func = self.student_module.calculate_average
        
        # Test with simple cases
        test_cases = [
            ([1, 2, 3, 4, 5], 3.0),
            ([10, 20, 30], 20.0),
            ([1], 1.0),
            ([0, 0, 0], 0.0)
        ]
        
        points_per_case = TestConfig.POINTS['function_tests'] // (len(test_cases) * 2)
        total_earned = 0
        
        for inputs, expected in test_cases:
            try:
                result = TestUtils.timeout_test(lambda: func(inputs))
                assert_close_enough(result, expected)
                total_earned += points_per_case
            except Exception as e:
                pytest.fail(f"Failed for input {inputs}: {e}")
        
        self.award_points('function_tests', total_earned, 
                         description="Basic average calculation")
    
    @pytest.mark.function_test
    def test_find_maximum_basic(self):
        """Test basic maximum finding"""
        func = self.student_module.find_maximum
        
        test_cases = [
            ([1, 5, 3, 9, 2], 9),
            ([-5, -2, -10, -1], -1),
            ([42], 42),
            ([1.5, 2.7, 1.2], 2.7)
        ]
        
        points_per_case = TestConfig.POINTS['function_tests'] // (len(test_cases) * 2)
        total_earned = 0
        
        for inputs, expected in test_cases:
            try:
                result = TestUtils.timeout_test(lambda: func(inputs))
                assert_close_enough(result, expected)
                total_earned += points_per_case
            except Exception as e:
                pytest.fail(f"Failed for input {inputs}: {e}")
        
        self.award_points('function_tests', total_earned,
                         description="Basic maximum finding")
    
    @pytest.mark.edge_case
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        total_earned = 0
        edge_points = TestConfig.POINTS['edge_cases']
        
        # Test empty list handling
        try:
            avg_func = self.student_module.calculate_average
            max_func = self.student_module.find_maximum
            
            # These should handle empty lists gracefully
            with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
                avg_func([])
            total_earned += edge_points // 4
            
            with pytest.raises((ValueError, IndexError)):
                max_func([])
            total_earned += edge_points // 4
            
        except Exception as e:
            pytest.fail(f"Edge case handling failed: {e}")
        
        # Test very large numbers
        try:
            large_numbers = [1e10, 2e10, 3e10]
            result = avg_func(large_numbers)
            assert_close_enough(result, 2e10)
            total_earned += edge_points // 4
        except Exception as e:
            pytest.fail(f"Large number handling failed: {e}")
        
        # Test very small differences
        try:
            small_numbers = [0.0000001, 0.0000002, 0.0000003]
            result = avg_func(small_numbers)
            assert_close_enough(result, 0.0000002, tolerance=1e-10)
            total_earned += edge_points // 4
        except Exception as e:
            pytest.fail(f"Small number precision failed: {e}")
        
        self.award_points('edge_cases', total_earned,
                         description="Edge cases and boundary conditions")
    
    @pytest.mark.error_handling
    def test_error_handling(self):
        """Test proper error handling"""
        total_earned = 0
        error_points = TestConfig.POINTS['error_handling']
        
        avg_func = self.student_module.calculate_average
        max_func = self.student_module.find_maximum
        
        # Test invalid input types
        invalid_inputs = [
            "not a list",
            123,
            None,
            [1, 2, "three", 4]  # mixed types
        ]
        
        for invalid_input in invalid_inputs:
            try:
                with pytest.raises((TypeError, ValueError)):
                    avg_func(invalid_input)
                total_earned += error_points // (len(invalid_inputs) * 2)
            except Exception:
                pass  # Partial credit for attempting error handling
            
            try:
                with pytest.raises((TypeError, ValueError)):
                    max_func(invalid_input)
                total_earned += error_points // (len(invalid_inputs) * 2)
            except Exception:
                pass
        
        self.award_points('error_handling', total_earned,
                         description="Proper error handling")
    
    @pytest.mark.function_test
    def test_count_occurrences(self):
        """Test counting occurrences function"""
        func = self.student_module.count_occurrences
        
        test_cases = [
            (["apple", "banana", "apple", "cherry", "apple"], "apple", 3),
            ([1, 2, 3, 2, 2, 4], 2, 3),
            ([], "anything", 0),
            (["test"], "test", 1),
            (["test"], "missing", 0)
        ]
        
        points_per_case = TestConfig.POINTS['function_tests'] // 4  # Remaining points
        total_earned = 0
        
        for lst, item, expected in test_cases:
            try:
                result = TestUtils.timeout_test(lambda: func(lst, item))
                assert result == expected, f"Expected {expected}, got {result}"
                total_earned += points_per_case // len(test_cases)
            except Exception as e:
                pytest.fail(f"Failed for input {lst}, {item}: {e}")
        
        self.award_points('function_tests', total_earned,
                         description="Count occurrences function")
    
    def test_code_quality(self):
        """Test code quality aspects"""
        import ast
        import inspect
        
        total_earned = 0
        quality_points = TestConfig.POINTS['code_quality']
        
        # Check if functions have docstrings
        docstring_points = 0
        for func_name in self.REQUIRED_FUNCTIONS:
            func = getattr(self.student_module, func_name)
            if func.__doc__ and func.__doc__.strip():
                docstring_points += quality_points // (len(self.REQUIRED_FUNCTIONS) * 2)
        
        total_earned += docstring_points
        
        # Check for reasonable function length (not too long)
        length_points = 0
        for func_name in self.REQUIRED_FUNCTIONS:
            func = getattr(self.student_module, func_name)
            source = inspect.getsource(func)
            line_count = len(source.split('\n'))
            if line_count <= 20:  # Reasonable function length
                length_points += quality_points // (len(self.REQUIRED_FUNCTIONS) * 2)
        
        total_earned += length_points
        
        self.award_points('code_quality', total_earned,
                         description="Code quality (docstrings, function length)")
    
    def test_documentation_exists(self):
        """Test for existence of documentation"""
        total_earned = 0
        doc_points = TestConfig.POINTS['documentation']
        
        # Check for README file
        import os
        if os.path.exists('README.md'):
            total_earned += doc_points // 2
        
        # Check main module docstring
        if hasattr(self.student_module, '__doc__') and self.student_module.__doc__:
            total_earned += doc_points // 2
        
        self.award_points('documentation', total_earned,
                         description="Documentation (README, module docstring)")


# Performance tests (optional)
class TestPerformance:
    """Performance tests for function implementations"""
    
    @pytest.mark.performance
    def test_performance_large_datasets(self):
        """Test performance with large datasets"""
        import time
        from base_test_template import TestUtils
        
        # Create large dataset
        large_list = list(range(100000))
        
        # Test average calculation performance
        student_module = TestUtils.safe_import('main')
        avg_func = student_module.calculate_average
        
        start_time = time.time()
        TestUtils.timeout_test(lambda: avg_func(large_list), timeout=2.0)
        end_time = time.time()
        
        assert end_time - start_time < 1.0, \
            f"Function took too long: {end_time - start_time:.2f} seconds"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])