#!/usr/bin/env python3
"""
Exercise: Hello Data Science Script - Comprehensive Skills Demonstration

This script demonstrates fundamental Python concepts and command line integration:
- Variables and data types (strings, numbers, booleans)
- Functions with parameters and return values
- Control structures (if/else statements, loops)
- File operations and script execution
- User input handling and error management
"""

import sys
import os

def greet_data_scientist(name):
    """
    Generate a personalized greeting for a data scientist.
    Demonstrates string variables and function return values.
    
    Args:
        name (str): The name of the person to greet
        
    Returns:
        str: A personalized greeting message
    """
    # String variables and formatting
    greeting_prefix = "Hello"
    subject_area = "data science"
    return f"{greeting_prefix}, {name}! Welcome to the world of {subject_area}!"

def analyze_number(num):
    """
    Perform basic analysis on a number.
    Demonstrates variables, control structures, and data organization.
    
    Args:
        num (float): The number to analyze
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Numeric variables and calculations
    squared_value = num ** 2
    is_even_number = num % 2 == 0
    is_positive_number = num > 0
    
    # Control structures - conditional logic
    if num > 100:
        magnitude = "large"
    elif num > 10:
        magnitude = "medium"  
    elif num > 0:
        magnitude = "small"
    else:
        magnitude = "non-positive"
    
    # Data organization in dictionary
    analysis = {
        "original": num,
        "squared": squared_value,
        "is_even": is_even_number,
        "is_positive": is_positive_number,
        "magnitude": magnitude
    }
    return analysis

def demonstrate_file_operations():
    """
    Demonstrate file operations and Python script capabilities.
    Shows how Python scripts can work with files and directories.
    """
    print("\n=== File Operations Demonstration ===")
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # List files in current directory
    files_in_dir = os.listdir(current_dir)
    python_files = [f for f in files_in_dir if f.endswith('.py')]
    
    print(f"Python scripts found: {len(python_files)}")
    for script in python_files:
        print(f"  - {script}")

def demonstrate_command_line_integration():
    """
    Show how Python scripts integrate with command line.
    Demonstrates sys.argv and script execution concepts.
    """
    print("\n=== Command Line Integration ===")
    print(f"Script name: {sys.argv[0]}")
    print(f"Number of arguments: {len(sys.argv)}")
    
    if len(sys.argv) > 1:
        print("Command line arguments provided:")
        for i, arg in enumerate(sys.argv[1:], 1):
            print(f"  Argument {i}: {arg}")
    else:
        print("No command line arguments provided")

def practice_control_structures():
    """
    Demonstrate various control structures with practical examples.
    Shows loops, conditionals, and iteration patterns.
    """
    print("\n=== Control Structures Practice ===")
    
    # List of sample data for processing
    sample_numbers = [1, 4, 7, 12, 15, 23, 31, 42]
    
    print("Processing sample data with loops:")
    
    # For loop demonstration
    even_count = 0
    odd_count = 0
    
    for number in sample_numbers:
        if number % 2 == 0:
            even_count += 1
            print(f"  {number} is even")
        else:
            odd_count += 1
            print(f"  {number} is odd")
    
    print(f"Summary: {even_count} even numbers, {odd_count} odd numbers")
    
    # While loop demonstration
    print("\nCountdown demonstration:")
    countdown = 5
    while countdown > 0:
        print(f"  {countdown}...")
        countdown -= 1
    print("  Launch!")

def main():
    """
    Main function that runs comprehensive Python skills demonstration.
    
    This function orchestrates multiple demonstrations showing:
    - Variables and functions
    - Control structures and loops  
    - File operations and script integration
    - Command line argument handling
    """
    print("=== Comprehensive Data Science Skills Demo ===")
    
    # Basic interaction and variables demonstration
    name = input("What's your name? ")
    greeting = greet_data_scientist(name)
    print(greeting)
    
    # Number analysis with variables, functions, and control structures
    try:
        number = float(input("Enter a number to analyze: "))
        results = analyze_number(number)
        
        print(f"\nAnalysis of {results['original']}:")
        print(f"- Squared: {results['squared']}")
        print(f"- Even number: {results['is_even']}")
        print(f"- Positive: {results['is_positive']}")
        print(f"- Magnitude: {results['magnitude']}")
    except ValueError:
        print("That wasn't a valid number, but that's okay!")
    
    # Demonstrate additional Python skills
    demonstrate_file_operations()
    demonstrate_command_line_integration()
    practice_control_structures()
    
    print("\n=== Skills Demonstrated ===")
    print("✓ Variables (strings, numbers, booleans)")
    print("✓ Functions with parameters and returns")
    print("✓ Control structures (if/elif/else, for/while loops)")
    print("✓ File operations and directory navigation")
    print("✓ Command line integration with scripts")
    print("✓ Python script execution and organization")
    
    print("\nKeep learning and exploring data science!")

if __name__ == "__main__":
    main()