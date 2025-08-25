"""
File I/O Operations Test Template
================================

Template for testing file input/output operations, including reading/writing
various formats (CSV, JSON, text files), file handling, and path management.
"""

import pytest
import os
import json
import csv
import tempfile
import shutil
from pathlib import Path
from base_test_template import (
    TestHomeworkBase, TestConfig, TestUtils,
    assert_function_exists, assert_file_contains
)


class TestFileIOAssignment(TestHomeworkBase):
    """Test suite for file I/O assignments"""
    
    REQUIRED_FUNCTIONS = [
        'read_csv_file',
        'write_csv_file',
        'read_json_file',
        'write_json_file',
        'process_text_file'
    ]
    
    def setup_method(self):
        """Setup test files and verify required functions"""
        super().setup_method()
        
        # Verify all required functions exist
        for func_name in self.REQUIRED_FUNCTIONS:
            assert_function_exists(self.student_module, func_name)
        
        # Create test directory
        self.test_dir = Path('test_files')
        self.test_dir.mkdir(exist_ok=True)
        
        # Create sample CSV data
        self.csv_data = [
            ['Name', 'Age', 'City'],
            ['Alice', '25', 'New York'],
            ['Bob', '30', 'Los Angeles'],
            ['Charlie', '35', 'Chicago']
        ]
        
        self.csv_file = self.test_dir / 'test.csv'
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.csv_data)
        
        # Create sample JSON data
        self.json_data = {
            'users': [
                {'name': 'Alice', 'age': 25, 'city': 'New York'},
                {'name': 'Bob', 'age': 30, 'city': 'Los Angeles'}
            ],
            'metadata': {
                'version': '1.0',
                'created': '2024-01-01'
            }
        }
        
        self.json_file = self.test_dir / 'test.json'
        with open(self.json_file, 'w') as f:
            json.dump(self.json_data, f)
        
        # Create sample text file
        self.text_content = """This is a sample text file.
It contains multiple lines.
Some lines have numbers: 123, 456, 789.
And some have special characters: @#$%!
"""
        
        self.text_file = self.test_dir / 'test.txt'
        with open(self.text_file, 'w') as f:
            f.write(self.text_content)
    
    def teardown_method(self):
        """Cleanup test files"""
        super().teardown_method()
        # Remove test directory and all files
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        # Clean up any output files in current directory
        output_files = ['output.csv', 'output.json', 'output.txt', 'processed.txt']
        for file in output_files:
            if os.path.exists(file):
                os.remove(file)
    
    @pytest.mark.function_test
    def test_read_csv_file(self):
        """Test CSV file reading functionality"""
        read_csv = self.student_module.read_csv_file
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        try:
            # Test reading CSV file
            result = TestUtils.timeout_test(
                lambda: read_csv(str(self.csv_file)),
                timeout=TestConfig.TIMEOUT['file_operation']
            )
            
            # Should return list of lists or similar structure
            assert isinstance(result, (list, tuple)), \
                "CSV reader should return a list or tuple"
            points_earned += total_points // 3
            
            # Check correct number of rows
            expected_rows = len(self.csv_data)
            assert len(result) == expected_rows, \
                f"Expected {expected_rows} rows, got {len(result)}"
            points_earned += total_points // 3
            
            # Check header row
            if isinstance(result[0], (list, tuple)):
                assert result[0] == self.csv_data[0], \
                    f"Header row mismatch: expected {self.csv_data[0]}, got {result[0]}"
                points_earned += total_points // 3
            
        except Exception as e:
            pytest.fail(f"CSV reading failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="CSV file reading")
    
    @pytest.mark.function_test
    def test_write_csv_file(self):
        """Test CSV file writing functionality"""
        write_csv = self.student_module.write_csv_file
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        try:
            # Test writing CSV file
            output_file = self.test_dir / 'output.csv'
            test_data = [
                ['Product', 'Price', 'Quantity'],
                ['Apple', '1.50', '10'],
                ['Banana', '0.75', '20']
            ]
            
            TestUtils.timeout_test(
                lambda: write_csv(str(output_file), test_data),
                timeout=TestConfig.TIMEOUT['file_operation']
            )
            
            # Verify file was created
            assert output_file.exists(), \
                f"Output file {output_file} was not created"
            points_earned += total_points // 3
            
            # Verify file contents
            with open(output_file, 'r') as f:
                reader = csv.reader(f)
                written_data = list(reader)
            
            assert written_data == test_data, \
                "Written data doesn't match expected data"
            points_earned += total_points * 2 // 3
            
        except Exception as e:
            pytest.fail(f"CSV writing failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="CSV file writing")
    
    @pytest.mark.function_test
    def test_read_json_file(self):
        """Test JSON file reading functionality"""
        read_json = self.student_module.read_json_file
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        try:
            # Test reading JSON file
            result = TestUtils.timeout_test(
                lambda: read_json(str(self.json_file)),
                timeout=TestConfig.TIMEOUT['file_operation']
            )
            
            # Should return dictionary
            assert isinstance(result, dict), \
                "JSON reader should return a dictionary"
            points_earned += total_points // 3
            
            # Check for expected keys
            assert 'users' in result, "Missing 'users' key in JSON data"
            assert 'metadata' in result, "Missing 'metadata' key in JSON data"
            points_earned += total_points // 3
            
            # Check data integrity
            assert result == self.json_data, \
                "Read JSON data doesn't match original"
            points_earned += total_points // 3
            
        except Exception as e:
            pytest.fail(f"JSON reading failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="JSON file reading")
    
    @pytest.mark.function_test
    def test_write_json_file(self):
        """Test JSON file writing functionality"""
        write_json = self.student_module.write_json_file
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        try:
            # Test writing JSON file
            output_file = self.test_dir / 'output.json'
            test_data = {
                'items': ['item1', 'item2', 'item3'],
                'count': 3,
                'active': True
            }
            
            TestUtils.timeout_test(
                lambda: write_json(str(output_file), test_data),
                timeout=TestConfig.TIMEOUT['file_operation']
            )
            
            # Verify file was created
            assert output_file.exists(), \
                f"Output file {output_file} was not created"
            points_earned += total_points // 3
            
            # Verify file contents
            with open(output_file, 'r') as f:
                written_data = json.load(f)
            
            assert written_data == test_data, \
                "Written JSON data doesn't match expected data"
            points_earned += total_points * 2 // 3
            
        except Exception as e:
            pytest.fail(f"JSON writing failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="JSON file writing")
    
    @pytest.mark.function_test
    def test_process_text_file(self):
        """Test text file processing functionality"""
        process_text = self.student_module.process_text_file
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        try:
            # Test processing text file
            result = TestUtils.timeout_test(
                lambda: process_text(str(self.text_file)),
                timeout=TestConfig.TIMEOUT['file_operation']
            )
            
            # Result should be meaningful (dict with stats, list of lines, etc.)
            assert result is not None, \
                "Text processor should return meaningful result"
            points_earned += total_points // 3
            
            # If result is a dict, check for common text processing metrics
            if isinstance(result, dict):
                common_keys = ['line_count', 'word_count', 'char_count']
                found_keys = sum(1 for key in common_keys if key in result)
                if found_keys > 0:
                    points_earned += total_points * 2 // 3
            elif isinstance(result, (list, tuple)):
                # If it's a list, should have reasonable length
                if len(result) > 0:
                    points_earned += total_points * 2 // 3
            else:
                # Any other reasonable processing result
                points_earned += total_points // 3
            
        except Exception as e:
            pytest.fail(f"Text processing failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="Text file processing")
    
    @pytest.mark.edge_case
    def test_edge_cases(self):
        """Test edge cases for file operations"""
        points_earned = 0
        total_points = TestConfig.POINTS['edge_cases']
        
        # Test with empty files
        try:
            empty_csv = self.test_dir / 'empty.csv'
            empty_csv.touch()
            
            read_csv = self.student_module.read_csv_file
            result = read_csv(str(empty_csv))
            
            # Should handle empty file gracefully
            assert isinstance(result, (list, tuple)), \
                "Should return list/tuple for empty CSV"
            points_earned += total_points // 4
            
        except Exception:
            points_earned += total_points // 8  # Partial credit
        
        # Test with malformed JSON
        try:
            malformed_json = self.test_dir / 'malformed.json'
            with open(malformed_json, 'w') as f:
                f.write('{"incomplete": json')
            
            read_json = self.student_module.read_json_file
            
            with pytest.raises((json.JSONDecodeError, ValueError)):
                read_json(str(malformed_json))
            points_earned += total_points // 4
            
        except Exception:
            points_earned += total_points // 8
        
        # Test with very large files
        try:
            large_text = self.test_dir / 'large.txt'
            with open(large_text, 'w') as f:
                for i in range(1000):
                    f.write(f"Line {i}: " + "x" * 100 + "\n")
            
            process_text = self.student_module.process_text_file
            result = TestUtils.timeout_test(
                lambda: process_text(str(large_text)),
                timeout=10.0
            )
            
            points_earned += total_points // 4
            
        except Exception:
            points_earned += total_points // 8
        
        # Test with special characters and encoding
        try:
            unicode_file = self.test_dir / 'unicode.txt'
            unicode_content = "Hello 世界! Café résumé naïve 🚀 émojis"
            
            with open(unicode_file, 'w', encoding='utf-8') as f:
                f.write(unicode_content)
            
            process_text = self.student_module.process_text_file
            result = process_text(str(unicode_file))
            
            points_earned += total_points // 4
            
        except Exception:
            points_earned += total_points // 8
        
        self.award_points('edge_cases', points_earned,
                         description="Edge cases handling")
    
    @pytest.mark.error_handling
    def test_error_handling(self):
        """Test error handling for file operations"""
        points_earned = 0
        total_points = TestConfig.POINTS['error_handling']
        
        read_csv = self.student_module.read_csv_file
        write_csv = self.student_module.write_csv_file
        read_json = self.student_module.read_json_file
        write_json = self.student_module.write_json_file
        
        # Test reading non-existent files
        try:
            with pytest.raises((FileNotFoundError, IOError)):
                read_csv('non_existent.csv')
            points_earned += total_points // 5
        except Exception:
            pass
        
        try:
            with pytest.raises((FileNotFoundError, IOError)):
                read_json('non_existent.json')
            points_earned += total_points // 5
        except Exception:
            pass
        
        # Test writing to invalid paths
        try:
            with pytest.raises((IOError, OSError, PermissionError)):
                write_csv('/invalid/path/file.csv', [['data']])
            points_earned += total_points // 5
        except Exception:
            pass
        
        try:
            with pytest.raises((IOError, OSError, PermissionError)):
                write_json('/invalid/path/file.json', {})
            points_earned += total_points // 5
        except Exception:
            pass
        
        # Test with invalid data types
        try:
            with pytest.raises((TypeError, ValueError)):
                write_json('test.json', lambda x: x)  # Functions can't be serialized
            points_earned += total_points // 5
        except Exception:
            pass
        
        self.award_points('error_handling', points_earned,
                         description="Error handling for file operations")
    
    def test_file_handling_quality(self):
        """Test file handling best practices"""
        points_earned = 0
        quality_points = TestConfig.POINTS['code_quality']
        
        # Check if functions properly close files (no resource leaks)
        try:
            # Create many small operations to test resource management
            for i in range(100):
                temp_file = self.test_dir / f'temp_{i}.csv'
                self.student_module.write_csv_file(
                    str(temp_file), 
                    [['test', 'data'], ['1', '2']]
                )
                result = self.student_module.read_csv_file(str(temp_file))
                temp_file.unlink()  # Clean up immediately
            
            points_earned += quality_points // 2
            
        except Exception as e:
            pytest.fail(f"Resource management test failed: {e}")
        
        # Check for proper path handling
        try:
            # Test with different path formats
            path_formats = [
                str(self.csv_file),
                Path(self.csv_file),
                str(self.csv_file.absolute())
            ]
            
            read_csv = self.student_module.read_csv_file
            
            for path_format in path_formats:
                try:
                    result = read_csv(path_format)
                    assert result is not None
                except Exception:
                    # If it fails, it should fail gracefully
                    pass
            
            points_earned += quality_points // 2
            
        except Exception as e:
            pytest.fail(f"Path handling test failed: {e}")
        
        self.award_points('code_quality', points_earned,
                         description="File handling best practices")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])