#!/usr/bin/env python3
"""
Assignment 5, Question 2: Python Data Processing
Process clinical trial metadata and configuration files.
"""


def parse_config(filepath: str) -> dict:
    """
    Parse config file (key=value format) into dictionary.

    Args:
        filepath: Path to config.txt

    Returns:
        dict: Configuration as key-value pairs

    Example:
        >>> config = parse_config('config.txt')
        >>> config['study_name']
        'CardioHealth Trial 2023'
    """
    # TODO: Read file, split on '=', create dict
    pass


def validate_config(config: dict) -> dict:
    """
    Validate configuration values using if/elif/else logic.

    Rules:
    - min_age must be >= 18
    - max_age must be <= 100
    - target_enrollment must be > 0
    - sites must be >= 1

    Args:
        config: Configuration dictionary

    Returns:
        dict: Validation results {key: True/False}

    Example:
        >>> config = {'min_age': '18', 'max_age': '85'}
        >>> results = validate_config(config)
        >>> results['min_age']
        True
    """
    # TODO: Implement with if/elif/else
    pass


def process_files(file_list: list) -> list:
    """
    Filter file list to only .csv files.

    Args:
        file_list: List of filenames

    Returns:
        list: Filtered list of .csv files only

    Example:
        >>> files = ['data.csv', 'script.py', 'output.csv']
        >>> process_files(files)
        ['data.csv', 'output.csv']
    """
    # TODO: Filter to .csv files (any valid approach)
    pass


def calculate_statistics(data: list) -> dict:
    """
    Calculate basic statistics.

    Args:
        data: List of numbers

    Returns:
        dict: {mean, median, sum, count}

    Example:
        >>> stats = calculate_statistics([10, 20, 30, 40, 50])
        >>> stats['mean']
        30.0
    """
    # TODO: Calculate stats
    pass


if __name__ == '__main__':
    # TODO: Load and process config.txt
    # TODO: Save outputs to output/
    #   - output/config_summary.txt
    #   - output/validation_report.txt
    #   - output/file_manifest.txt
    #   - output/statistics.txt
    pass
