"""
Command Line Interface Test Template
===================================

Template for testing command-line tools, scripts with arguments,
subprocess execution, and CLI behavior verification.
"""

import pytest
import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
from base_test_template import (
    TestHomeworkBase, TestConfig, TestUtils,
    assert_function_exists, assert_file_contains
)


class TestCLIAssignment(TestHomeworkBase):
    """Test suite for command-line interface assignments"""
    
    def setup_method(self):
        """Setup test environment for CLI testing"""
        super().setup_method()
        
        # Create test directory with sample files
        self.test_dir = Path('cli_test_files')
        self.test_dir.mkdir(exist_ok=True)
        
        # Create sample input files
        self.sample_text_file = self.test_dir / 'input.txt'
        with open(self.sample_text_file, 'w') as f:
            f.write("Hello World\nThis is a test file\nWith multiple lines\n")
        
        self.sample_data_file = self.test_dir / 'data.csv'
        with open(self.sample_data_file, 'w') as f:
            f.write("name,age,city\nAlice,25,NYC\nBob,30,LA\nCharlie,35,Chicago\n")
        
        # Expected CLI script path
        self.cli_script = 'main.py'  # Adjust based on assignment requirements
        
        # Verify CLI script exists
        if not os.path.exists(self.cli_script):
            pytest.skip(f"CLI script {self.cli_script} not found")
    
    def teardown_method(self):
        """Cleanup test files"""
        super().teardown_method()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        # Clean up any output files
        output_files = ['output.txt', 'result.csv', 'processed.json']
        for file in output_files:
            if os.path.exists(file):
                os.remove(file)
    
    @pytest.mark.function_test
    def test_cli_help_option(self):
        """Test that CLI script provides help information"""
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        help_options = ['-h', '--help']
        
        for help_option in help_options:
            try:
                result = TestUtils.run_student_script(
                    self.cli_script, 
                    args=[help_option],
                    timeout=TestConfig.TIMEOUT['subprocess']
                )
                
                # Help should exit with code 0 and provide output
                if result['returncode'] == 0 and (result['stdout'] or result['stderr']):
                    # Check for common help indicators
                    help_text = result['stdout'] + result['stderr']
                    help_indicators = ['usage', 'options', 'arguments', 'help']
                    
                    if any(indicator in help_text.lower() for indicator in help_indicators):
                        points_earned += total_points // len(help_options)
                        break
                
            except Exception as e:
                # Continue trying other help options
                continue
        
        self.award_points('function_tests', points_earned,
                         description="CLI help option")
    
    @pytest.mark.function_test
    def test_cli_basic_execution(self):
        """Test basic CLI script execution"""
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        try:
            # Test running script with no arguments
            result = TestUtils.run_student_script(
                self.cli_script,
                timeout=TestConfig.TIMEOUT['subprocess']
            )
            
            # Script should run without crashing
            if result['returncode'] in [0, 1]:  # 0 = success, 1 = expected error
                points_earned += total_points // 3
            
            # Should provide some output (stdout or stderr)
            if result['stdout'] or result['stderr']:
                points_earned += total_points // 3
            
            # If it fails, error message should be informative
            if result['returncode'] != 0:
                error_output = result['stderr'] or result['stdout']
                if len(error_output.strip()) > 10:  # Non-trivial error message
                    points_earned += total_points // 3
            else:
                points_earned += total_points // 3
                
        except Exception as e:
            pytest.fail(f"Basic CLI execution failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="Basic CLI execution")
    
    @pytest.mark.function_test
    def test_cli_file_processing(self):
        """Test CLI script with file input/output"""
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        try:
            # Test processing input file
            output_file = 'processed_output.txt'
            
            # Common CLI patterns to test
            test_commands = [
                [str(self.sample_text_file)],  # Single file argument
                [str(self.sample_text_file), output_file],  # Input and output files
                ['-i', str(self.sample_text_file)],  # -i flag for input
                ['--input', str(self.sample_text_file)],  # --input flag
                ['-o', output_file, str(self.sample_text_file)],  # -o flag for output
            ]
            
            success_count = 0
            for args in test_commands:
                try:
                    result = TestUtils.run_student_script(
                        self.cli_script,
                        args=args,
                        timeout=TestConfig.TIMEOUT['subprocess']
                    )
                    
                    if result['returncode'] == 0:
                        success_count += 1
                        
                        # If output file was specified and created
                        if output_file in args and os.path.exists(output_file):
                            success_count += 1
                            os.remove(output_file)  # Clean up
                            
                        break  # Found working command pattern
                        
                except Exception:
                    continue
            
            if success_count > 0:
                points_earned = total_points * min(success_count, 2) // 2
                
        except Exception as e:
            pytest.fail(f"File processing test failed: {e}")
        
        self.award_points('function_tests', points_earned,
                         description="CLI file processing")
    
    @pytest.mark.function_test
    def test_cli_with_options(self):
        """Test CLI script with various options/flags"""
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        # Common CLI options to test
        option_tests = [
            ['-v'],  # Verbose
            ['--verbose'],
            ['-q'],  # Quiet
            ['--quiet'],
            ['-c', '10'],  # Count/number option
            ['--count', '5'],
            ['-f', 'json'],  # Format option
            ['--format', 'csv'],
        ]
        
        working_options = 0
        
        for options in option_tests:
            try:
                result = TestUtils.run_student_script(
                    self.cli_script,
                    args=options,
                    timeout=TestConfig.TIMEOUT['subprocess']
                )
                
                # Accept both success (0) and graceful failure (1)
                if result['returncode'] in [0, 1]:
                    working_options += 1
                    
                    # If verbose option, should produce more output
                    if any(opt in ['-v', '--verbose'] for opt in options):
                        if len(result['stdout']) > 50:  # Verbose output
                            working_options += 1
                    
                    # If quiet option, should produce less output
                    elif any(opt in ['-q', '--quiet'] for opt in options):
                        if len(result['stdout']) < 20:  # Quiet output
                            working_options += 1
                
            except Exception:
                continue
        
        # Award points based on number of working options
        if working_options > 0:
            points_earned = total_points * min(working_options, 4) // 4
        
        self.award_points('function_tests', points_earned,
                         description="CLI options handling")
    
    @pytest.mark.function_test
    def test_cli_interactive_input(self):
        """Test CLI script with interactive input"""
        points_earned = 0
        total_points = TestConfig.POINTS['function_tests'] // 5
        
        try:
            # Test providing input via stdin
            test_input = "test input\n5\ny\n"
            
            result = TestUtils.run_student_script(
                self.cli_script,
                input_text=test_input,
                timeout=TestConfig.TIMEOUT['subprocess']
            )
            
            # Script should handle input without hanging
            if 'timeout' not in result:
                points_earned += total_points // 2
            
            # Should produce some output
            if result['stdout'] or result['stderr']:
                points_earned += total_points // 2
                
        except Exception as e:
            # Interactive input might not be implemented
            points_earned += total_points // 4  # Partial credit
        
        self.award_points('function_tests', points_earned,
                         description="Interactive input handling")
    
    @pytest.mark.edge_case
    def test_cli_edge_cases(self):
        """Test CLI edge cases"""
        points_earned = 0
        total_points = TestConfig.POINTS['edge_cases']
        
        edge_case_tests = [
            # Test with non-existent file
            (['non_existent_file.txt'], "non-existent file"),
            
            # Test with empty file
            ([], "empty arguments"),
            
            # Test with too many arguments
            (['arg1', 'arg2', 'arg3', 'arg4', 'arg5'] * 10, "too many arguments"),
            
            # Test with special characters in arguments
            (['file with spaces.txt'], "spaces in filename"),
            (['special@#$chars.txt'], "special characters"),
        ]
        
        for args, description in edge_case_tests:
            try:
                result = TestUtils.run_student_script(
                    self.cli_script,
                    args=args,
                    timeout=TestConfig.TIMEOUT['subprocess']
                )
                
                # Should not crash (return code should be reasonable)
                if result['returncode'] in range(-15, 128):  # Reasonable exit codes
                    points_earned += total_points // len(edge_case_tests)
                
                # If error, should provide meaningful error message
                if result['returncode'] != 0:
                    error_msg = result['stderr'] or result['stdout']
                    if len(error_msg.strip()) > 5:  # Has error message
                        points_earned += total_points // (len(edge_case_tests) * 2)
                
            except Exception:
                # Partial credit for not crashing the test
                points_earned += total_points // (len(edge_case_tests) * 2)
        
        self.award_points('edge_cases', points_earned,
                         description="CLI edge cases")
    
    @pytest.mark.error_handling
    def test_cli_error_handling(self):
        """Test CLI error handling"""
        points_earned = 0
        total_points = TestConfig.POINTS['error_handling']
        
        error_tests = [
            # Invalid options
            (['-invalid'], "invalid option"),
            (['--unknown-flag'], "unknown flag"),
            
            # Permission errors (if applicable)
            (['/root/protected_file.txt'], "permission denied"),
            
            # Invalid file formats
            ([self.sample_text_file, '--format', 'invalid_format'], "invalid format"),
        ]
        
        for args, error_type in error_tests:
            try:
                result = TestUtils.run_student_script(
                    self.cli_script,
                    args=args,
                    timeout=TestConfig.TIMEOUT['subprocess']
                )
                
                # Should exit with non-zero code for errors
                if result['returncode'] != 0:
                    points_earned += total_points // (len(error_tests) * 2)
                
                # Should provide error message
                error_output = result['stderr'] or result['stdout']
                if error_output and len(error_output.strip()) > 10:
                    points_earned += total_points // (len(error_tests) * 2)
                
            except Exception:
                # At least it tried to handle the error
                points_earned += total_points // (len(error_tests) * 4)
        
        self.award_points('error_handling', points_earned,
                         description="CLI error handling")
    
    def test_cli_code_quality(self):
        """Test CLI code quality aspects"""
        points_earned = 0
        quality_points = TestConfig.POINTS['code_quality']
        
        try:
            # Check if script is executable
            import stat
            if os.path.exists(self.cli_script):
                file_stat = os.stat(self.cli_script)
                if file_stat.st_mode & stat.S_IEXEC:
                    points_earned += quality_points // 4
            
            # Check for proper argument parsing (look for argparse usage)
            with open(self.cli_script, 'r') as f:
                script_content = f.read()
                
                # Look for argument parsing libraries
                if 'argparse' in script_content or 'click' in script_content:
                    points_earned += quality_points // 2
                elif 'sys.argv' in script_content:
                    points_earned += quality_points // 4  # Basic argument handling
            
            # Check for proper exit codes
            if 'sys.exit' in script_content or 'return' in script_content:
                points_earned += quality_points // 4
                
        except Exception:
            pass
        
        self.award_points('code_quality', points_earned,
                         description="CLI code quality")
    
    def test_cli_documentation(self):
        """Test CLI documentation"""
        points_earned = 0
        doc_points = TestConfig.POINTS['documentation']
        
        try:
            # Test if help output is informative
            result = TestUtils.run_student_script(
                self.cli_script,
                args=['-h'],
                timeout=TestConfig.TIMEOUT['subprocess']
            )
            
            if result['returncode'] == 0:
                help_output = result['stdout'] or result['stderr']
                
                # Check for comprehensive help
                help_elements = ['usage:', 'options:', 'description', 'example']
                found_elements = sum(1 for elem in help_elements 
                                   if elem in help_output.lower())
                
                points_earned += doc_points * found_elements // len(help_elements)
            
        except Exception:
            pass
        
        # Check for README or documentation file
        doc_files = ['README.md', 'README.txt', 'USAGE.md', 'docs/']
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                points_earned += doc_points // 4
                break
        
        self.award_points('documentation', points_earned,
                         description="CLI documentation")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])