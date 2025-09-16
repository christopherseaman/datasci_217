#!/usr/bin/env python3
"""
DataSci 217 - Lecture 11 Assignment Tests
Reproducible Research and Professional Workflows

Test cases for validating assignment completion.
"""

import pytest
import sys
from pathlib import Path

# Add the assignment directory to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import main
except ImportError:
    print("Error: Could not import main.py")
    sys.exit(1)

class TestLecture11:
    """Test cases for Lecture 11 assignment"""

    def test_main_function_exists(self):
        """Test that main function exists"""
        assert hasattr(main, 'main'), "main() function not found in main.py"

    def test_main_function_callable(self):
        """Test that main function is callable"""
        assert callable(main.main), "main() function is not callable"

    def test_main_runs_without_error(self):
        """Test that main function runs without errors"""
        try:
            main.main()
        except Exception as e:
            pytest.fail(f"main() function raised an exception: {e}")

# Additional test cases will be added based on specific assignment requirements

if __name__ == "__main__":
    pytest.main([__file__])
