#!/usr/bin/env python3
"""
Example: Function-Based Assignment - Math Operations

This example demonstrates a complete function-based homework assignment
with proper error handling, documentation, and testing structure.

Student: Example Student
Course: Data Science 217
Assignment: Functions and Error Handling
"""

from typing import List, Union, Optional
import math


def calculate_average(numbers: List[Union[int, float]]) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Args:
        numbers: List of numeric values
    
    Returns:
        The arithmetic mean as a float
    
    Raises:
        ValueError: If the list is empty
        TypeError: If the list contains non-numeric values
    
    Examples:
        >>> calculate_average([1, 2, 3, 4, 5])
        3.0
        >>> calculate_average([2.5, 3.5, 4.5])
        3.5
    """
    if not isinstance(numbers, list):
        raise TypeError("Input must be a list")
    
    if len(numbers) == 0:
        raise ValueError("Cannot calculate average of empty list")
    
    # Validate all elements are numeric
    for i, num in enumerate(numbers):
        if not isinstance(num, (int, float)):
            raise TypeError(f"Element at index {i} is not numeric: {num}")
        if math.isnan(num) or math.isinf(num):
            raise ValueError(f"Element at index {i} is not a finite number: {num}")
    
    return sum(numbers) / len(numbers)


def find_maximum(numbers: List[Union[int, float]]) -> Union[int, float]:
    """
    Find the maximum value in a list of numbers.
    
    Args:
        numbers: List of numeric values
    
    Returns:
        The largest number in the list
    
    Raises:
        ValueError: If the list is empty
        TypeError: If the list contains non-numeric values
    
    Examples:
        >>> find_maximum([3, 1, 4, 1, 5, 9])
        9
        >>> find_maximum([2.5, 3.7, 1.2])
        3.7
    """
    if not isinstance(numbers, list):
        raise TypeError("Input must be a list")
    
    if len(numbers) == 0:
        raise ValueError("Cannot find maximum of empty list")
    
    # Validate all elements are numeric
    for i, num in enumerate(numbers):
        if not isinstance(num, (int, float)):
            raise TypeError(f"Element at index {i} is not numeric: {num}")
        if math.isnan(num) or math.isinf(num):
            raise ValueError(f"Element at index {i} is not a finite number: {num}")
    
    return max(numbers)


def count_occurrences(items: List, target) -> int:
    """
    Count how many times a target item appears in a list.
    
    Args:
        items: List of items to search through
        target: Item to count occurrences of
    
    Returns:
        Number of times target appears in items
    
    Examples:
        >>> count_occurrences(['a', 'b', 'a', 'c', 'a'], 'a')
        3
        >>> count_occurrences([1, 2, 3, 2, 2], 2)
        3
        >>> count_occurrences([], 'anything')
        0
    """
    if not isinstance(items, list):
        raise TypeError("Input must be a list")
    
    return items.count(target)


def validate_input(data: dict, required_keys: List[str]) -> bool:
    """
    Validate that a dictionary contains all required keys.
    
    Args:
        data: Dictionary to validate
        required_keys: List of keys that must be present
    
    Returns:
        True if all required keys are present, False otherwise
    
    Raises:
        TypeError: If data is not a dictionary or required_keys is not a list
    
    Examples:
        >>> validate_input({'name': 'Alice', 'age': 25}, ['name', 'age'])
        True
        >>> validate_input({'name': 'Bob'}, ['name', 'age'])
        False
    """
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary")
    
    if not isinstance(required_keys, list):
        raise TypeError("Required keys must be a list")
    
    return all(key in data for key in required_keys)


def calculate_statistics(numbers: List[Union[int, float]]) -> dict:
    """
    Calculate comprehensive statistics for a list of numbers.
    
    Args:
        numbers: List of numeric values
    
    Returns:
        Dictionary containing mean, max, min, count, and sum
    
    Raises:
        ValueError: If the list is empty
        TypeError: If the list contains non-numeric values
    
    Examples:
        >>> stats = calculate_statistics([1, 2, 3, 4, 5])
        >>> stats['mean']
        3.0
        >>> stats['count']
        5
    """
    if not isinstance(numbers, list):
        raise TypeError("Input must be a list")
    
    if len(numbers) == 0:
        raise ValueError("Cannot calculate statistics for empty list")
    
    # Validate all elements are numeric
    for i, num in enumerate(numbers):
        if not isinstance(num, (int, float)):
            raise TypeError(f"Element at index {i} is not numeric: {num}")
        if math.isnan(num) or math.isinf(num):
            raise ValueError(f"Element at index {i} is not a finite number: {num}")
    
    return {
        'mean': calculate_average(numbers),
        'max': find_maximum(numbers),
        'min': min(numbers),
        'count': len(numbers),
        'sum': sum(numbers),
        'range': max(numbers) - min(numbers)
    }


def main():
    """
    Main function demonstrating usage of all implemented functions.
    """
    print("🧮 Math Operations Assignment Demo")
    print("=" * 40)
    
    # Test data
    test_numbers = [3.14, 2.71, 1.41, 1.73, 2.23]
    test_dict = {'name': 'Alice', 'age': 25, 'city': 'New York'}
    test_list = ['apple', 'banana', 'apple', 'cherry', 'apple']
    
    try:
        # Test calculate_average
        print(f"\n📊 Average of {test_numbers}:")
        avg = calculate_average(test_numbers)
        print(f"Result: {avg:.2f}")
        
        # Test find_maximum
        print(f"\n📈 Maximum of {test_numbers}:")
        maximum = find_maximum(test_numbers)
        print(f"Result: {maximum}")
        
        # Test count_occurrences
        print(f"\n🔍 Occurrences of 'apple' in {test_list}:")
        count = count_occurrences(test_list, 'apple')
        print(f"Result: {count}")
        
        # Test validate_input
        print(f"\n✅ Validating {test_dict} for keys ['name', 'age']:")
        is_valid = validate_input(test_dict, ['name', 'age'])
        print(f"Result: {is_valid}")
        
        print(f"\n✅ Validating {test_dict} for keys ['name', 'age', 'salary']:")
        is_valid = validate_input(test_dict, ['name', 'age', 'salary'])
        print(f"Result: {is_valid}")
        
        # Test calculate_statistics
        print(f"\n📈 Complete statistics for {test_numbers}:")
        stats = calculate_statistics(test_numbers)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n🎉 All functions executed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())