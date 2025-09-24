#!/usr/bin/env python3
"""
DataSci 217 - Lecture 02 Assignment Tests
Git Workflow, CLI Automation, and Python Data Processing

Test cases for validating assignment completion with progressive difficulty.
"""

import pytest
import sys
import os
import subprocess
import json
from pathlib import Path

# Add the assignment directory to the path
sys.path.insert(0, str(Path(__file__).parent))

class TestPart1GitWorkflow:
    """Test cases for Part 1: Git Workflow Mastery (7 points)"""

    def test_git_repository_exists(self):
        """Test that Git repository is properly initialized"""
        assert Path('.git').exists(), "Git repository not found (.git directory missing)"
    
    def test_gitignore_exists(self):
        """Test that .gitignore file exists"""
        assert Path('.gitignore').exists(), ".gitignore file not found"
    
    def test_readme_exists(self):
        """Test that README.md exists and has content"""
        readme_path = Path('README.md')
        assert readme_path.exists(), "README.md not found"
        
        readme_content = readme_path.read_text()
        assert len(readme_content) > 100, "README.md appears to be too short or empty"
        assert "DataSci" in readme_content, "README.md should mention DataSci"
    
    def test_requirements_exists(self):
        """Test that requirements.txt exists"""
        assert Path('requirements.txt').exists(), "requirements.txt not found"

class TestPart2CLIAutomation:
    """Test cases for Part 2: CLI Project Scaffold Script (6 points)"""

    def test_setup_script_exists(self):
        """Test that setup_project.sh exists"""
        assert Path('setup_project.sh').exists(), "setup_project.sh not found"
    
    def test_setup_script_executable(self):
        """Test that setup_project.sh is executable"""
        setup_script = Path('setup_project.sh')
        assert setup_script.exists(), "setup_project.sh not found"
        
        # Check if script is executable
        assert os.access(setup_script, os.X_OK), "setup_project.sh is not executable"
    
    def test_setup_script_content(self):
        """Test that setup_project.sh has required functionality"""
        setup_script = Path('setup_project.sh')
        assert setup_script.exists(), "setup_project.sh not found"

        script_content = setup_script.read_text()

        # Check for required commands (not functions)
        assert 'mkdir -p' in script_content, "mkdir -p command not found"
        assert 'echo' in script_content, "echo command not found"
        assert 'cat >' in script_content, "cat > command for file creation not found"
        assert 'EOF' in script_content, "Here-document (EOF) not found"

        # Check for shebang
        assert script_content.startswith('#!/bin/bash'), "Script should start with shebang"
    
    def test_directory_structure_created(self):
        """Test that required directory structure exists"""
        required_dirs = ['src', 'data', 'output']
        for dir_name in required_dirs:
            assert Path(dir_name).exists(), f"Required directory '{dir_name}' not found"
    
    def test_sample_data_files(self):
        """Test that sample data files exist"""
        students_file = Path('data/students.csv')
        assert students_file.exists(), "Sample data file 'data/students.csv' not found"
    
    def test_python_templates(self):
        """Test that Python template files exist"""
        python_files = ['src/data_analysis.py', 'src/data_analysis_functions.py']
        for file_path in python_files:
            assert Path(file_path).exists(), f"Python template '{file_path}' not found"

class TestPart3PythonProgramming:
    """Test cases for Part 3: Python Data Processing (7 points)"""

    def test_basic_analysis_script(self):
        """Test that basic analysis script exists and is functional"""
        script_path = Path('src/data_analysis.py')
        assert script_path.exists(), "src/data_analysis.py not found"
        
        # Check for required functions
        script_content = script_path.read_text()
        assert 'def main():' in script_content, "main() function not found in basic script"
        assert 'def load_students(' in script_content, "load_students() function not found"
        assert 'def calculate_average_grade(' in script_content, "calculate_average_grade() function not found"
    
    def test_advanced_analysis_script(self):
        """Test that advanced analysis script exists and is functional"""
        script_path = Path('src/data_analysis_functions.py')
        assert script_path.exists(), "src/data_analysis_functions.py not found"
        
        # Check for required functions
        script_content = script_path.read_text()
        assert 'def main():' in script_content, "main() function not found in advanced script"
        assert 'def load_data(' in script_content, "load_data() function not found"
        assert 'def analyze_data(' in script_content, "analyze_data() function not found"
        assert 'def save_results(' in script_content, "save_results() function not found"
    
    def test_scripts_run_without_error(self):
        """Test that both Python scripts run without errors"""
        scripts = ['src/data_analysis.py', 'src/data_analysis_functions.py']
        
        for script_path in scripts:
            if Path(script_path).exists():
                try:
                    # Run the script and check for errors
                    result = subprocess.run([sys.executable, script_path], 
                                          capture_output=True, text=True, timeout=30)
                    assert result.returncode == 0, f"Script {script_path} failed with error: {result.stderr}"
                except subprocess.TimeoutExpired:
                    pytest.fail(f"Script {script_path} timed out")
                except Exception as e:
                    pytest.fail(f"Error running {script_path}: {e}")
    
    def test_output_file_generated(self):
        """Test that analysis output file is generated"""
        output_file = Path('output/analysis_report.txt')
        assert output_file.exists(), "output/analysis_report.txt not found"
        
        # Check that output file has content
        output_content = output_file.read_text()
        assert len(output_content) > 50, "Output file appears to be empty or too short"
        assert "Analysis" in output_content, "Output file should contain analysis results"
    
    def test_data_processing_functionality(self):
        """Test that data processing works correctly"""
        # Check if students.csv exists and has expected content
        students_file = Path('data/students.csv')
        if students_file.exists():
            students_content = students_file.read_text()
            assert 'name,age,grade,subject' in students_content, "Students CSV missing header"
            assert 'Alice' in students_content, "Students CSV missing sample data"

class TestIntegration:
    """Integration tests for the complete assignment"""

    def test_complete_workflow(self):
        """Test that the complete workflow can be executed"""
        # Test that all required files exist
        required_files = [
            'README.md',
            '.gitignore', 
            'requirements.txt',
            'setup_project.sh',
            'src/data_analysis.py',
            'src/data_analysis_functions.py'
        ]
        
        for file_path in required_files:
            assert Path(file_path).exists(), f"Required file '{file_path}' not found"
    
    def test_git_workflow_demonstration(self):
        """Test that Git workflow is properly demonstrated"""
        # Check for Git repository
        assert Path('.git').exists(), "Git repository not initialized"
        
        # Check for meaningful commit history (this is a basic check)
        try:
            result = subprocess.run(['git', 'log', '--oneline'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                assert len(commits) >= 1, "No Git commits found"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Git might not be available in test environment
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
