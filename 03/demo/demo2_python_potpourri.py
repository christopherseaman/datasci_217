#!/usr/bin/env python3
"""
Python Potpourri: Essential Skills for Data Science
Demonstrates type checking, f-string formatting, and professional practices
"""

import numpy as np

def demo_type_checking():
    """Demonstrate type checking for debugging."""
    print("=== Type Checking ===")

    # Common type confusion
    user_input = "42"
    print(f"user_input = '{user_input}'")
    print(f"type(user_input) = {type(user_input)}")

    # Convert and check
    number = int(user_input)
    print(f"number = {number}")
    print(f"type(number) = {type(number)}")

    # Type checking with isinstance
    data = [1, 2, "3", 4, 5]
    print(f"\nData with mixed types: {data}")
    for item in data:
        if isinstance(item, str):
            print(f"  '{item}' is a string - converting to int")
        else:
            print(f"  {item} is already an int")

    print()

def demo_f_strings():
    """Demonstrate modern f-string formatting."""
    print("=== F-String Formatting ===")

    # Basic f-strings
    name = "Alice"
    grade = 87.5
    message = f"Student {name} earned {grade:.1f}%"
    print(message)

    # Formatting numbers
    value = 3.14159
    print(f"Default: {value}")
    print(f"2 decimals: {value:.2f}")
    print(f"Right-aligned (width 10): {value:>10.2f}")
    print(f"Left-aligned (width 10): {value:<10.2f}")
    print(f"Center-aligned (width 10): {value:^10.2f}")

    # Expressions in f-strings
    arr = np.array([85, 92, 78, 88, 95])
    print(f"\nGrades: {arr}")
    print(f"Mean: {arr.mean():.1f}")
    print(f"Max: {arr.max()}")
    print(f"Students above 85: {(arr > 85).sum()}")

    # Multi-line f-strings
    report = f"""
Student Analysis Report
{'=' * 30}
Student: {name}
Grade: {grade:.1f}
Class Average: {arr.mean():.1f}
Status: {'Above Average' if grade > arr.mean() else 'Below Average'}
"""
    print(report)

def demo_practical_patterns():
    """Demonstrate practical patterns for data science."""
    print("=== Practical Patterns ===")

    # Pattern 1: Data validation with type checking
    def safe_convert(value):
        """Safely convert value to int with type checking."""
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                print(f"Warning: Cannot convert '{value}' to int")
                return None
        else:
            print(f"Warning: Unexpected type {type(value)}")
            return None

    test_values = [42, "123", "invalid", 3.14]
    print("Safe conversion results:")
    for val in test_values:
        result = safe_convert(val)
        print(f"  {val!r:12} -> {result}")

    # Pattern 2: Professional logging with f-strings
    def analyze_grades(grades, student_name):
        """Analyze grades with professional output."""
        mean = np.mean(grades)
        std = np.std(grades)

        # Professional formatted output
        print(f"\nAnalysis for {student_name}:")
        print(f"  Samples: {len(grades)}")
        print(f"  Mean: {mean:.2f} ± {std:.2f}")
        print(f"  Range: [{min(grades)}, {max(grades)}]")

        # Conditional formatting
        status = "Excellent" if mean >= 90 else "Good" if mean >= 80 else "Fair"
        print(f"  Overall: {status}")

    grades = np.array([85, 92, 78, 88, 95, 87, 91])
    analyze_grades(grades, "Alice")

    print()

def main():
    """Run all demos."""
    print("Python Potpourri: Essential Skills Demo")
    print("=" * 50)
    print()

    demo_type_checking()
    demo_f_strings()
    demo_practical_patterns()

    print("✅ All demos complete!")

if __name__ == "__main__":
    main()