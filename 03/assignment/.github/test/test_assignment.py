#!/usr/bin/env python3
"""
DataSci 217 - Lecture 03 Assignment Tests
NumPy Arrays & Virtual Environments - Health Sensor Data Analysis

Test cases for validating assignment completion.
"""

import pytest
import sys
import os
import subprocess
from pathlib import Path

# Add the assignment directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# NOTE: Part 2 (Virtual Environment Setup) is not directly tested because
# students should not commit .venv directories. Part 3 completion implicitly
# validates that Part 2 was done correctly.


class TestPart1CLI:
    """Test cases for Part 1: CLI Data Tools (7 points)"""

    def test_patient_count_file_exists(self):
        """Test that output/part1_patient_count.txt exists"""
        assert Path('output/part1_patient_count.txt').exists(), "output/part1_patient_count.txt not found"

    def test_patient_count_reasonable(self):
        """Test that patient count is reasonable (should be ~10000)"""
        patient_file = Path('output/part1_patient_count.txt')
        if patient_file.exists():
            content = patient_file.read_text().strip()
            try:
                count = int(content)
                assert 8000 <= count <= 10000, f"Patient count {count} seems unreasonable (expected ~10000)"
            except ValueError:
                pytest.fail("part1_patient_count.txt should contain a single number")

    def test_high_bp_count_file_exists(self):
        """Test that output/part1_high_bp_count.txt exists"""
        assert Path('output/part1_high_bp_count.txt').exists(), "output/part1_high_bp_count.txt not found"

    def test_high_bp_count_reasonable(self):
        """Test that high BP count is reasonable"""
        bp_file = Path('output/part1_high_bp_count.txt')
        if bp_file.exists():
            content = bp_file.read_text().strip()
            try:
                count = int(content)
                assert 0 <= count <= 50000, f"High BP count {count} seems unreasonable"
                assert count > 0, "High BP count should be greater than 0"
            except ValueError:
                pytest.fail("part1_high_bp_count.txt should contain a single number")

    def test_avg_temp_file_exists(self):
        """Test that output/part1_avg_temp.txt exists"""
        assert Path('output/part1_avg_temp.txt').exists(), "output/part1_avg_temp.txt not found"

    def test_avg_temp_format(self):
        """Test that average temperature is properly formatted"""
        temp_file = Path('output/part1_avg_temp.txt')
        if temp_file.exists():
            content = temp_file.read_text().strip()
            try:
                temp = float(content)
                assert 97.0 <= temp <= 99.5, f"Average temperature {temp} seems unreasonable"
                # Check for decimal places (should have 2)
                assert '.' in content, "Temperature should have decimal places"
            except ValueError:
                pytest.fail("part1_avg_temp.txt should contain a decimal number")

    def test_glucose_stats_file_exists(self):
        """Test that output/part1_glucose_stats.txt exists"""
        assert Path('output/part1_glucose_stats.txt').exists(), "output/part1_glucose_stats.txt not found"

    def test_glucose_stats_format(self):
        """Test that glucose stats has 5 values"""
        glucose_file = Path('output/part1_glucose_stats.txt')
        if glucose_file.exists():
            content = glucose_file.read_text().strip()
            lines = content.split('\n')
            assert len(lines) == 5, f"Expected 5 glucose values, found {len(lines)}"
            
            # Verify all are numbers
            for i, line in enumerate(lines):
                try:
                    value = int(line.strip())
                    assert 70 <= value <= 130, f"Glucose value {value} on line {i+1} seems unreasonable"
                except ValueError:
                    pytest.fail(f"Line {i+1} in part1_glucose_stats.txt should contain a number")


class TestPart3NumPyAnalysis:
    """Test cases for Part 3: NumPy Data Analysis (8 points)"""

    def test_analyze_script_runs(self):
        """Test that analyze_health_data.py runs without errors"""
        script_path = Path('analyze_health_data.py')
        if script_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                assert result.returncode == 0, f"Script failed with error: {result.stderr}"
            except subprocess.TimeoutExpired:
                pytest.fail("Script timed out")
            except Exception as e:
                pytest.fail(f"Error running script: {e}")

    def test_analysis_report_generated(self):
        """Test that output/analysis_report.txt is created"""
        report_file = Path('output/analysis_report.txt')
        assert report_file.exists(), "output/analysis_report.txt not found - script may not have completed"

    def test_report_has_content(self):
        """Test that analysis report has meaningful content"""
        report_file = Path('output/analysis_report.txt')
        if report_file.exists():
            content = report_file.read_text()
            assert len(content) > 100, "Report appears to be too short or empty"
            
            # Check for expected sections
            assert 'Health Sensor' in content or 'Analysis' in content, "Report should have a title"
            assert 'Average' in content or 'Mean' in content, "Report should include averages"

    def test_report_contains_statistics(self):
        """Test that report contains expected statistics"""
        report_file = Path('output/analysis_report.txt')
        if report_file.exists():
            content = report_file.read_text()
            
            # Check for key metrics
            metrics = ['heart rate', 'blood pressure', 'glucose']
            found_metrics = sum(1 for metric in metrics if metric.lower() in content.lower())
            assert found_metrics >= 2, "Report should mention at least 2 of: heart rate, blood pressure, glucose"

    def test_report_has_abnormal_counts(self):
        """Test that report includes abnormal reading counts"""
        report_file = Path('output/analysis_report.txt')
        if report_file.exists():
            content = report_file.read_text().lower()
            
            # Check for abnormal/high readings mentioned
            abnormal_keywords = ['abnormal', 'high', 'elevated']
            found_abnormal = any(keyword in content for keyword in abnormal_keywords)
            assert found_abnormal, "Report should mention abnormal or high readings"

    def test_numeric_values_reasonable(self):
        """Test that numeric values in report are reasonable"""
        report_file = Path('output/analysis_report.txt')
        if report_file.exists():
            content = report_file.read_text()
            
            # Look for numbers that might be statistics
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            
            # Should have multiple numeric values
            assert len(numbers) >= 5, "Report should contain multiple numeric values"




if __name__ == "__main__":
    pytest.main([__file__, "-v"])