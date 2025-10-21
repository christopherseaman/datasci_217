#!/usr/bin/env python3
"""
Test suite for q2_process_metadata.py
Tests all functions with various inputs and edge cases.
"""

import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q2_process_metadata import (
    parse_config,
    validate_config,
    generate_sample_data,
    calculate_statistics
)


def test_parse_config():
    """Test configuration file parsing."""
    print("\n=== Testing parse_config ===")

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("sample_data_rows=100\n")
        f.write("sample_data_min=18\n")
        f.write("sample_data_max=75\n")
        f.write("# This is a comment\n")
        f.write("\n")  # Empty line
        temp_config = f.name

    try:
        config = parse_config(temp_config)

        # Verify all keys are present
        assert 'sample_data_rows' in config, "Missing sample_data_rows"
        assert 'sample_data_min' in config, "Missing sample_data_min"
        assert 'sample_data_max' in config, "Missing sample_data_max"

        # Verify values are correct
        assert config['sample_data_rows'] == '100', "Incorrect rows value"
        assert config['sample_data_min'] == '18', "Incorrect min value"
        assert config['sample_data_max'] == '75', "Incorrect max value"

        print("✓ parse_config: All tests passed")
        print(f"  Loaded config: {config}")

    finally:
        os.unlink(temp_config)


def test_validate_config():
    """Test configuration validation."""
    print("\n=== Testing validate_config ===")

    # Test valid configuration
    valid_config = {
        'sample_data_rows': '100',
        'sample_data_min': '18',
        'sample_data_max': '75'
    }
    results = validate_config(valid_config)
    assert results['rows_valid'] == True, "Valid rows should pass"
    assert results['min_valid'] == True, "Valid min should pass"
    assert results['max_valid'] == True, "Valid max should pass"
    print("✓ Valid configuration passed all checks")

    # Test invalid rows (zero)
    invalid_rows = {
        'sample_data_rows': '0',
        'sample_data_min': '18',
        'sample_data_max': '75'
    }
    results = validate_config(invalid_rows)
    assert results['rows_valid'] == False, "Zero rows should fail"
    print("✓ Zero rows correctly rejected")

    # Test invalid rows (negative)
    invalid_rows = {
        'sample_data_rows': '-10',
        'sample_data_min': '18',
        'sample_data_max': '75'
    }
    results = validate_config(invalid_rows)
    assert results['rows_valid'] == False, "Negative rows should fail"
    print("✓ Negative rows correctly rejected")

    # Test invalid min (zero)
    invalid_min = {
        'sample_data_rows': '100',
        'sample_data_min': '0',
        'sample_data_max': '75'
    }
    results = validate_config(invalid_min)
    assert results['min_valid'] == False, "Zero min should fail"
    print("✓ Min value of 0 correctly rejected")

    # Test invalid max (less than min)
    invalid_max = {
        'sample_data_rows': '100',
        'sample_data_min': '75',
        'sample_data_max': '18'
    }
    results = validate_config(invalid_max)
    assert results['max_valid'] == False, "Max < min should fail"
    print("✓ Max < min correctly rejected")

    # Test invalid max (equal to min)
    invalid_max = {
        'sample_data_rows': '100',
        'sample_data_min': '50',
        'sample_data_max': '50'
    }
    results = validate_config(invalid_max)
    assert results['max_valid'] == False, "Max == min should fail"
    print("✓ Max == min correctly rejected")

    # Test non-numeric values
    invalid_format = {
        'sample_data_rows': 'abc',
        'sample_data_min': '18',
        'sample_data_max': '75'
    }
    results = validate_config(invalid_format)
    assert results['rows_valid'] == False, "Non-numeric rows should fail"
    print("✓ Non-numeric values correctly rejected")

    # Test missing keys
    missing_keys = {'sample_data_rows': '100'}
    results = validate_config(missing_keys)
    assert results['min_valid'] == False, "Missing min should fail"
    assert results['max_valid'] == False, "Missing max should fail"
    print("✓ Missing keys correctly handled")

    print("✓ validate_config: All tests passed")


def test_generate_sample_data():
    """Test sample data generation."""
    print("\n=== Testing generate_sample_data ===")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test_sample.csv')

    try:
        config = {
            'sample_data_rows': '50',
            'sample_data_min': '10',
            'sample_data_max': '100'
        }

        generate_sample_data(temp_file, config)

        # Verify file was created
        assert os.path.exists(temp_file), "File was not created"
        print("✓ File created successfully")

        # Read and verify contents
        with open(temp_file, 'r') as f:
            lines = f.readlines()

        # Check number of rows
        assert len(lines) == 50, f"Expected 50 rows, got {len(lines)}"
        print("✓ Correct number of rows generated")

        # Verify each line is a valid integer in range
        for i, line in enumerate(lines):
            value = int(line.strip())
            assert 10 <= value <= 100, f"Row {i+1}: value {value} out of range [10, 100]"

        print("✓ All values are integers in the correct range")

        # Verify no header
        first_value = int(lines[0].strip())
        assert isinstance(first_value, int), "First line should be a number, not a header"
        print("✓ No header present (data starts immediately)")

        print("✓ generate_sample_data: All tests passed")

    finally:
        shutil.rmtree(temp_dir)


def test_calculate_statistics():
    """Test statistics calculation."""
    print("\n=== Testing calculate_statistics ===")

    # Test with simple dataset
    data = [10, 20, 30, 40, 50]
    stats = calculate_statistics(data)

    assert stats['count'] == 5, "Incorrect count"
    assert stats['sum'] == 150, "Incorrect sum"
    assert stats['mean'] == 30.0, "Incorrect mean"
    assert stats['median'] == 30, "Incorrect median (odd count)"
    print("✓ Odd-length dataset: correct statistics")

    # Test with even-length dataset
    data = [10, 20, 30, 40]
    stats = calculate_statistics(data)

    assert stats['count'] == 4, "Incorrect count"
    assert stats['sum'] == 100, "Incorrect sum"
    assert stats['mean'] == 25.0, "Incorrect mean"
    assert stats['median'] == 25.0, "Incorrect median (even count)"
    print("✓ Even-length dataset: correct statistics")

    # Test with single value
    data = [42]
    stats = calculate_statistics(data)

    assert stats['count'] == 1, "Incorrect count"
    assert stats['sum'] == 42, "Incorrect sum"
    assert stats['mean'] == 42.0, "Incorrect mean"
    assert stats['median'] == 42, "Incorrect median"
    print("✓ Single value: correct statistics")

    # Test with unsorted data
    data = [50, 10, 40, 20, 30]
    stats = calculate_statistics(data)

    assert stats['median'] == 30, "Median should work with unsorted data"
    print("✓ Unsorted data: correct median calculation")

    # Test with duplicates
    data = [10, 20, 20, 30, 30, 30]
    stats = calculate_statistics(data)

    assert stats['count'] == 6, "Incorrect count with duplicates"
    assert stats['sum'] == 140, "Incorrect sum with duplicates"
    assert stats['mean'] == 140/6, "Incorrect mean with duplicates"
    assert stats['median'] == 25.0, "Incorrect median with duplicates"
    print("✓ Dataset with duplicates: correct statistics")

    # Test with realistic range (matching config)
    data = list(range(18, 76))  # 18 to 75 inclusive
    stats = calculate_statistics(data)

    assert stats['count'] == 58, "Incorrect count for realistic range"
    assert stats['median'] == 46.5, "Incorrect median for realistic range"
    print("✓ Realistic range (18-75): correct statistics")

    print("✓ calculate_statistics: All tests passed")


def test_integration():
    """Test full workflow integration."""
    print("\n=== Testing Full Integration ===")

    # Create temporary directory structure
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, 'data')
    output_dir = os.path.join(temp_dir, 'output')
    os.makedirs(data_dir)
    os.makedirs(output_dir)

    try:
        # Create config file
        config_file = os.path.join(temp_dir, 'test_config.txt')
        with open(config_file, 'w') as f:
            f.write("sample_data_rows=100\n")
            f.write("sample_data_min=18\n")
            f.write("sample_data_max=75\n")

        # Parse config
        config = parse_config(config_file)
        print("✓ Config parsed")

        # Validate config
        validation = validate_config(config)
        assert all(validation.values()), "Config validation failed"
        print("✓ Config validated")

        # Generate sample data
        sample_file = os.path.join(data_dir, 'sample_data.csv')
        generate_sample_data(sample_file, config)
        assert os.path.exists(sample_file), "Sample file not created"
        print("✓ Sample data generated")

        # Read and calculate statistics
        with open(sample_file, 'r') as f:
            data = [int(line.strip()) for line in f]
        stats = calculate_statistics(data)
        print("✓ Statistics calculated")

        # Verify statistics make sense
        assert stats['count'] == 100, "Expected 100 data points"
        assert 18 <= stats['mean'] <= 75, "Mean out of expected range"
        assert 18 <= stats['median'] <= 75, "Median out of expected range"
        assert stats['sum'] == sum(data), "Sum mismatch"

        print("✓ Integration test: All steps completed successfully")
        print(f"  Generated {stats['count']} values")
        print(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")

    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all test suites."""
    print("=" * 50)
    print("Running Q2 Metadata Processing Tests")
    print("=" * 50)

    try:
        test_parse_config()
        test_validate_config()
        test_generate_sample_data()
        test_calculate_statistics()
        test_integration()

        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED")
        print("=" * 50)
        return True

    except AssertionError as e:
        print("\n" + "=" * 50)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 50)
        return False
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"✗ ERROR: {e}")
        print("=" * 50)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
