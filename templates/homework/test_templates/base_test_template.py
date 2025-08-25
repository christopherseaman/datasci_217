"""
Base Test Template for Homework Assignments
==========================================

This template provides a foundation for creating pytest-based auto-grading tests.
It includes common patterns, utilities, and best practices for educational testing.
"""

import pytest
import sys
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
import json
import time


class TestConfig:
    """Configuration for test execution and grading"""
    
    # Point values for different test categories
    POINTS = {
        'function_tests': 40,
        'edge_cases': 20,
        'error_handling': 15,
        'code_quality': 15,
        'documentation': 10
    }
    
    # Timeout settings (in seconds)
    TIMEOUT = {
        'function_call': 5,
        'file_operation': 10,
        'subprocess': 30
    }
    
    # Required files for submission
    REQUIRED_FILES = ['main.py']
    
    # Optional files
    OPTIONAL_FILES = ['README.md', 'requirements.txt']


class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def safe_import(module_name: str, required_functions: List[str] = None):
        """Safely import student module and check for required functions"""
        try:
            # Add current directory to path for student code
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            module = __import__(module_name)
            
            # Check for required functions
            if required_functions:
                missing_functions = []
                for func_name in required_functions:
                    if not hasattr(module, func_name):
                        missing_functions.append(func_name)
                
                if missing_functions:
                    pytest.fail(f"Missing required functions: {', '.join(missing_functions)}")
            
            return module
        except ImportError as e:
            pytest.fail(f"Could not import {module_name}: {e}")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {module_name}: {e}")
    
    @staticmethod
    def timeout_test(func: Callable, timeout: float = 5.0):
        """Run function with timeout protection"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function exceeded {timeout} second timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func()
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            raise
        finally:
            signal.alarm(0)  # Ensure alarm is cancelled
    
    @staticmethod
    @contextmanager
    def temp_directory():
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def create_test_file(path: str, content: str) -> None:
        """Create a test file with given content"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
    
    @staticmethod
    def run_student_script(script_path: str, args: List[str] = None, 
                          input_text: str = None, timeout: int = 30) -> Dict[str, Any]:
        """Run student script and capture output"""
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        try:
            result = subprocess.run(
                cmd,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout
            )
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': f'Process timed out after {timeout} seconds',
                'success': False,
                'timeout': True
            }


class GradingMixin:
    """Mixin class for grading functionality"""
    
    def __init__(self):
        self.points_earned = 0
        self.total_points = sum(TestConfig.POINTS.values())
        self.test_results = []
    
    def award_points(self, category: str, points: int = None, 
                    partial: float = 1.0, description: str = ""):
        """Award points for a test category"""
        if points is None:
            points = TestConfig.POINTS.get(category, 0)
        
        earned = int(points * partial)
        self.points_earned += earned
        
        self.test_results.append({
            'category': category,
            'points_possible': points,
            'points_earned': earned,
            'partial_credit': partial,
            'description': description
        })
        
        return earned
    
    def get_grade_report(self) -> str:
        """Generate a detailed grade report"""
        report = [
            "="*50,
            "GRADE REPORT",
            "="*50,
            f"Total Score: {self.points_earned}/{self.total_points} "
            f"({self.points_earned/self.total_points*100:.1f}%)",
            "",
            "Breakdown by Category:",
            "-"*30
        ]
        
        for result in self.test_results:
            report.append(
                f"{result['category']:20} "
                f"{result['points_earned']:3}/{result['points_possible']:3} "
                f"({result['partial_credit']*100:5.1f}%) "
                f"{result['description']}"
            )
        
        return "\n".join(report)


# Custom assertions for educational testing
def assert_function_exists(module, func_name: str):
    """Assert that a function exists in the module"""
    assert hasattr(module, func_name), f"Function '{func_name}' not found"
    assert callable(getattr(module, func_name)), f"'{func_name}' is not callable"


def assert_close_enough(actual, expected, tolerance=1e-6):
    """Assert that numeric values are close enough (handles floating point precision)"""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        assert abs(actual - expected) <= tolerance, \
            f"Expected {expected}, got {actual} (tolerance: {tolerance})"
    else:
        assert actual == expected, f"Expected {expected}, got {actual}"


def assert_file_contains(file_path: str, expected_content: str):
    """Assert that a file contains expected content"""
    assert os.path.exists(file_path), f"File {file_path} does not exist"
    with open(file_path, 'r') as f:
        content = f.read()
    assert expected_content in content, \
        f"File {file_path} does not contain expected content: {expected_content}"


# Test fixtures
@pytest.fixture
def student_module():
    """Import the main student module"""
    return TestUtils.safe_import('main')


@pytest.fixture
def temp_workspace():
    """Provide a temporary workspace for file operations"""
    with TestUtils.temp_directory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            yield temp_dir
        finally:
            os.chdir(original_cwd)


@pytest.fixture
def grader():
    """Provide grading functionality"""
    return GradingMixin()


# Example test class structure
class TestHomeworkBase(GradingMixin):
    """Base class for homework tests"""
    
    def setup_method(self):
        """Setup for each test method"""
        super().__init__()
        self.student_module = TestUtils.safe_import('main')
    
    def teardown_method(self):
        """Cleanup after each test method"""
        # Print grade report after all tests
        if hasattr(self, 'test_results'):
            print(self.get_grade_report())


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for educational testing"""
    # Add custom markers
    config.addinivalue_line("markers", "function_test: test basic function implementation")
    config.addinivalue_line("markers", "edge_case: test edge cases and boundary conditions")
    config.addinivalue_line("markers", "error_handling: test error handling and exceptions")
    config.addinivalue_line("markers", "integration: test integration with other components")
    config.addinivalue_line("markers", "performance: test performance requirements")


def pytest_runtest_makereport(item, call):
    """Customize test reporting for grading"""
    if call.when == "call":
        # Extract grading information if available
        if hasattr(item.instance, 'points_earned'):
            item.user_properties.append(("points_earned", item.instance.points_earned))
            item.user_properties.append(("total_points", item.instance.total_points))