#!/usr/bin/env python3
"""
Assignment 04: Test Suite
Automated tests for validating student solutions.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import json
import csv

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from starter_code import (
    validate_email,
    clean_text,
    extract_numbers,
    process_csv_file,
    create_frequency_table,
    filter_data_by_criteria,
    write_results_to_file
)


class TestEmailValidation:
    """Test email validation function."""

    def test_valid_emails(self):
        """Test with valid email addresses."""
        valid_emails = [
            "user@example.com",
            "student@university.edu",
            "name.surname@domain.org",
            "test123@test-domain.co.uk"
        ]
        for email in valid_emails:
            assert validate_email(email), f"Failed to validate: {email}"

    def test_invalid_emails(self):
        """Test with invalid email addresses."""
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user..name@domain.com",
            "user@domain",
            ""
        ]
        for email in invalid_emails:
            assert not validate_email(email), f"Incorrectly validated: {email}"


class TestTextCleaning:
    """Test text cleaning function."""

    def test_basic_cleaning(self):
        """Test basic text cleaning operations."""
        assert clean_text("  Hello, World!  ") == "hello world"
        assert clean_text("UPPERCASE TEXT") == "uppercase text"
        assert clean_text("   mixed   spacing   ") == "mixed spacing"

    def test_punctuation_removal(self):
        """Test punctuation removal."""
        text_with_punct = "Hello, world! How are you?"
        expected = "hello world how are you"
        assert clean_text(text_with_punct, remove_punctuation=True) == expected

    def test_punctuation_retention(self):
        """Test keeping punctuation when specified."""
        text_with_punct = "Hello, world!"
        result = clean_text(text_with_punct, remove_punctuation=False)
        assert "," in result and "!" in result


class TestNumberExtraction:
    """Test number extraction function."""

    def test_simple_numbers(self):
        """Test extraction of simple numbers."""
        text = "The temperature is 23.5 degrees and humidity is 67%"
        numbers = extract_numbers(text)
        assert 23.5 in numbers
        assert 67.0 in numbers

    def test_multiple_number_formats(self):
        """Test various number formats."""
        text = "Values: 100, -15.7, 0.001, 1e5"
        numbers = extract_numbers(text)
        assert len(numbers) >= 3  # At least basic numbers should be found

    def test_no_numbers(self):
        """Test text with no numbers."""
        text = "No numbers in this text"
        numbers = extract_numbers(text)
        assert len(numbers) == 0


class TestCSVProcessing:
    """Test CSV file processing function."""

    def test_valid_csv_processing(self):
        """Test processing a valid CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age', 'city'])
            writer.writerow(['Alice', '25', 'New York'])
            writer.writerow(['Bob', '30', 'San Francisco'])
            temp_filepath = f.name

        try:
            result = process_csv_file(temp_filepath, ['name', 'age'])
            assert result['row_count'] == 2  # Excluding header
            assert result['column_count'] == 3
            assert len(result['missing_columns']) == 0
        finally:
            Path(temp_filepath).unlink()

    def test_missing_columns(self):
        """Test with missing required columns."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age'])
            writer.writerow(['Alice', '25'])
            temp_filepath = f.name

        try:
            result = process_csv_file(temp_filepath, ['name', 'age', 'city'])
            assert 'city' in result['missing_columns']
        finally:
            Path(temp_filepath).unlink()


class TestFrequencyTable:
    """Test frequency table creation."""

    def test_basic_frequency_count(self):
        """Test basic frequency counting."""
        data = ['a', 'b', 'a', 'c', 'b', 'a']
        result = create_frequency_table(data)
        assert result['a'] == 3
        assert result['b'] == 2
        assert result['c'] == 1

    def test_empty_list(self):
        """Test with empty list."""
        result = create_frequency_table([])
        assert len(result) == 0

    def test_single_item(self):
        """Test with single item."""
        result = create_frequency_table(['item'])
        assert result['item'] == 1


class TestDataFiltering:
    """Test data filtering function."""

    def test_simple_filtering(self):
        """Test simple filtering by criteria."""
        data = [
            {'name': 'Alice', 'age': 25, 'city': 'New York'},
            {'name': 'Bob', 'age': 30, 'city': 'San Francisco'},
            {'name': 'Charlie', 'age': 25, 'city': 'Chicago'}
        ]
        result = filter_data_by_criteria(data, {'age': 25})
        assert len(result) == 2
        assert all(record['age'] == 25 for record in result)

    def test_multiple_criteria(self):
        """Test filtering with multiple criteria."""
        data = [
            {'name': 'Alice', 'age': 25, 'city': 'New York'},
            {'name': 'Bob', 'age': 30, 'city': 'New York'},
            {'name': 'Charlie', 'age': 25, 'city': 'Chicago'}
        ]
        result = filter_data_by_criteria(data, {'age': 25, 'city': 'New York'})
        assert len(result) == 1
        assert result[0]['name'] == 'Alice'


class TestFileWriting:
    """Test file writing function."""

    def test_json_writing(self):
        """Test writing JSON format."""
        data = {'test': 'value', 'number': 42}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name

        try:
            success = write_results_to_file(data, temp_filepath, 'json')
            assert success

            # Verify the file was written correctly
            with open(temp_filepath, 'r') as f:
                loaded_data = json.load(f)
                assert loaded_data == data
        finally:
            Path(temp_filepath).unlink()

    def test_invalid_format(self):
        """Test with invalid format type."""
        data = {'test': 'value'}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_filepath = f.name

        try:
            success = write_results_to_file(data, temp_filepath, 'invalid_format')
            # Should handle gracefully, either return False or use default format
            assert isinstance(success, bool)
        finally:
            if Path(temp_filepath).exists():
                Path(temp_filepath).unlink()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])