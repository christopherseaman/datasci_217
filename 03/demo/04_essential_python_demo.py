#!/usr/bin/env python3
"""
Demo 4: Essential Python Skills
Demonstrates virtual environments, type checking, debugging, and f-string formatting
This demonstrates the Professional Python Development section starting at line 553 of Lecture 3
"""

import numpy as np
import logging
from datetime import datetime
from pathlib import Path

def demonstrate_virtual_environments():
    """Demonstrate virtual environments for project isolation"""
    print("VIRTUAL ENVIRONMENTS FOR PROJECT ISOLATION")
    print("=" * 50)
    
    print("Virtual environments create isolated Python installations for each project.")
    print("This prevents package conflicts and ensures reproducible environments.")
    print("")
    
    print("Why Virtual Environments Matter:")
    print("The Problem: Different projects need different package versions")
    print("- Project A needs pandas 1.3.0")
    print("- Project B needs pandas 2.0.0")
    print("- Installing one breaks the other!")
    print("")
    
    print("The Solution: Each project gets its own Python environment")
    print("")
    
    print("Creating Virtual Environments:")
    print("Method 1: Using conda (recommended)")
    print("  conda create -n datasci-demo python=3.11")
    print("  conda activate datasci-demo")
    print("  conda install numpy pandas matplotlib")
    print("")
    
    print("Method 2: Using venv")
    print("  python -m venv datasci-demo")
    print("  source datasci-demo/bin/activate  # Mac/Linux")
    print("  datasci-demo\\Scripts\\activate  # Windows")
    print("  pip install numpy pandas matplotlib")
    print("")
    
    print("Managing Dependencies:")
    print("  conda env export > environment.yml")
    print("  pip freeze > requirements.txt")
    print("")
    
    # Show current environment info
    print("Current Environment Info:")
    print(f"Python version: {__import__('sys').version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Current working directory: {Path.cwd()}")

def demonstrate_type_checking():
    """Demonstrate type checking and debugging techniques"""
    print("\nTYPE CHECKING AND DEBUGGING")
    print("=" * 35)
    
    print("Understanding data types is crucial for debugging.")
    print("Python's dynamic typing means variables can change type.")
    print("")
    
    # Type checking for debugging (as mentioned in lecture)
    print("1. Type Checking for Debugging:")
    user_input = "42"  # This is a string, not a number!
    print(f"Input: {user_input}")
    print(f"Type: {type(user_input)}")
    
    # Convert and verify
    try:
        number = int(user_input)
        print(f"Converted: {number}")
        print(f"New type: {type(number)}")
    except ValueError as e:
        print(f"Conversion error: {e}")
    
    print("")
    print("2. Debugging Data Processing:")
    data = [1, 2, "3", 4, 5]  # Mixed types!
    print(f"Original data: {data}")
    
    for i, item in enumerate(data):
        print(f"Item {i}: {item}, Type: {type(item)}")
        if isinstance(item, str):
            print(f"  Converting string '{item}' to int")
            try:
                data[i] = int(item)
                print(f"  Successfully converted to: {data[i]}")
            except ValueError:
                print(f"  Could not convert '{item}' to int")
    
    print(f"Final data: {data}")

def demonstrate_error_handling():
    """Demonstrate error handling techniques"""
    print("\nERROR HANDLING")
    print("=" * 20)
    
    print("Basic error handling (as mentioned in lecture):")
    print("")
    
    # Basic error handling
    print("1. Handling ValueError:")
    try:
        number = int("not_a_number")
    except ValueError:
        print("Could not convert to number")
    
    print("")
    print("2. Handling ZeroDivisionError:")
    try:
        result = 10 / 0
    except ZeroDivisionError:
        print("Cannot divide by zero")
    
    print("")
    print("3. Comprehensive Error Handling:")
    
    def safe_data_processing(data):
        """Demonstrate safe data processing with error handling"""
        processed_data = []
        
        for i, record in enumerate(data):
            print(f"Processing record {i+1}: {record}")
            
            try:
                # Validate and convert data
                name = str(record.get('name', '')).strip()
                age = int(record.get('age', 0))
                grade = float(record.get('grade', 0))
                
                processed_record = {
                    'name': name,
                    'age': age,
                    'grade': grade
                }
                processed_data.append(processed_record)
                print(f"  ✓ Successfully processed: {processed_record}")
                
            except ValueError as e:
                print(f"  ✗ Error processing record: {e}")
                print(f"  Skipping invalid record")
            except KeyError as e:
                print(f"  ✗ Missing required field: {e}")
            except Exception as e:
                print(f"  ✗ Unexpected error: {e}")
        
        return processed_data
    
    # Test with sample data
    sample_data = [
        {"name": "Alice", "age": "25", "grade": "85.5"},
        {"name": "Bob", "age": "invalid", "grade": "92"},
        {"name": "Charlie", "age": "22", "grade": "not_a_number"},
        {"name": "Diana", "age": "20", "grade": "88.0"}
    ]
    
    processed_data = safe_data_processing(sample_data)
    print(f"\nProcessed {len(processed_data)} valid records out of {len(sample_data)} total")

def demonstrate_fstring_formatting():
    """Demonstrate f-string formatting"""
    print("\nF-STRING FORMATTING")
    print("=" * 25)
    
    print("F-strings are the modern, readable way to format strings in Python.")
    print("They're like having a conversation with your data!")
    print("")
    
    # F-string basics (as mentioned in lecture)
    print("1. F-String Basics:")
    name = "Alice"
    grade = 87.5
    year = 2024
    
    # F-strings (preferred, Python 3.6+)
    message = f"Student {name} is {grade} years old and earned {grade}%"
    print(f"Basic interpolation: {message}")
    
    # Expressions in f-strings
    print(f"Age next year: {2024 - 2000 + 1}")
    print(f"Grade category: {'A' if grade >= 90 else 'B' if grade >= 80 else 'C' if grade >= 70 else 'F'}")
    
    print("")
    print("2. F-String Formatting Options:")
    price = 1234.5678
    print(f"Price: ${price:.2f}")        # 2 decimal places
    print(f"Price: ${price:>10.2f}")      # Right-aligned, 10 chars
    print(f"Price: ${price:<10.2f}")      # Left-aligned, 10 chars
    print(f"Price: ${price:^10.2f}")      # Center-aligned, 10 chars
    print(f"Price: ${price:010.2f}")      # Zero-padded, 10 chars
    
    # Percentage formatting
    percentage = 0.875
    print(f"Percentage: {percentage:.1%}")  # 87.5%
    
    # Scientific notation
    large_number = 1234567890
    print(f"Large number: {large_number:.2e}")  # 1.23e+09
    
    # Comma separator
    print(f"Large number: {large_number:,}")    # 1,234,567,890
    
    print("")
    print("3. F-Strings with Data Structures:")
    grades = [85, 92, 78, 88, 91]
    print(f"Grades: {grades}")
    print(f"Average: {sum(grades) / len(grades):.1f}")
    print(f"Highest: {max(grades)}")
    print(f"Lowest: {min(grades)}")
    
    # NumPy arrays
    arr = np.array([1, 2, 3, 4, 5])
    print(f"NumPy array: {arr}")
    print(f"Array sum: {arr.sum()}")
    print(f"Array mean: {arr.mean():.2f}")
    print(f"Array shape: {arr.shape}")
    
    print("")
    print("4. Multi-line F-Strings:")
    name = "Charlie"
    grades = [85, 92, 78, 88, 91]
    average = sum(grades) / len(grades)
    
    # Multi-line f-string
    report = f"""
Student Report
==============
Name: {name}
Grades: {grades}
Average: {average:.1f}
Performance: {'Excellent' if average >= 90 else 'Good' if average >= 80 else 'Needs Improvement'}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    print(report)

def demonstrate_professional_development():
    """Demonstrate professional development practices"""
    print("\nPROFESSIONAL DEVELOPMENT PRACTICES")
    print("=" * 40)
    
    print("Professional Python development practices that separate")
    print("hobbyist Python from production-ready data science:")
    print("")
    
    # Configure logging
    print("1. Logging Instead of Print Statements:")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("This is a professional log message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("")
    print("2. Type Hints and Documentation:")
    
    def calculate_student_average(grades: list[float]) -> float:
        """
        Calculate the average of a list of grades.
        
        Args:
            grades: List of numerical grades
            
        Returns:
            Average grade as a float
            
        Raises:
            ValueError: If grades list is empty
        """
        if not grades:
            raise ValueError("Grades list cannot be empty")
        return sum(grades) / len(grades)
    
    # Test the function
    sample_grades = [85.5, 92.0, 78.5, 88.0, 91.5]
    average = calculate_student_average(sample_grades)
    print(f"Student average: {average:.1f}")
    
    print("")
    print("3. Data Validation:")
    
    def validate_student_data(student: dict) -> bool:
        """Validate student data structure"""
        required_fields = ['name', 'age', 'grade']
        
        for field in required_fields:
            if field not in student:
                print(f"Missing required field: {field}")
                return False
        
        # Check data types
        if not isinstance(student['age'], int):
            print(f"Age should be integer, got {type(student['age'])}")
            return False
        
        if not isinstance(student['grade'], (int, float)):
            print(f"Grade should be numeric, got {type(student['grade'])}")
            return False
        
        # Check value ranges
        if student['age'] < 0 or student['age'] > 150:
            print(f"Age {student['age']} is unrealistic")
            return False
        
        if student['grade'] < 0 or student['grade'] > 100:
            print(f"Grade {student['grade']} is out of range")
            return False
        
        return True
    
    # Test validation
    test_student = {"name": "Alice", "age": 20, "grade": 85.5}
    is_valid = validate_student_data(test_student)
    print(f"Student data validation: {'✓ Valid' if is_valid else '✗ Invalid'}")

def demonstrate_performance_comparison():
    """Demonstrate f-string performance"""
    print("\nF-STRING PERFORMANCE")
    print("=" * 25)
    
    import time
    
    name = "Alice"
    age = 25
    grade = 87.5
    
    # Performance comparison
    print("Comparing string formatting methods:")
    
    # Method 1: String concatenation
    start_time = time.time()
    for i in range(10000):
        message = "Student " + name + " is " + str(age) + " years old and earned " + str(grade) + "%"
    concat_time = time.time() - start_time
    
    # Method 2: .format()
    start_time = time.time()
    for i in range(10000):
        message = "Student {} is {} years old and earned {}%".format(name, age, grade)
    format_time = time.time() - start_time
    
    # Method 3: f-strings
    start_time = time.time()
    for i in range(10000):
        message = f"Student {name} is {age} years old and earned {grade}%"
    fstring_time = time.time() - start_time
    
    print(f"String concatenation: {concat_time:.4f} seconds")
    print(f".format() method: {format_time:.4f} seconds")
    print(f"f-strings: {fstring_time:.4f} seconds")
    print(f"f-strings are {concat_time/fstring_time:.1f}x faster than concatenation")

def main():
    """Main function demonstrating essential Python skills"""
    print("ESSENTIAL PYTHON SKILLS DEMONSTRATION")
    print("=" * 50)
    print("This demo directly demonstrates the Professional Python")
    print("Development section from Lecture 3, showing the skills")
    print("that separate hobbyist Python from professional data science.")
    print("=" * 50)
    
    # Run all demonstrations
    demonstrate_virtual_environments()
    demonstrate_type_checking()
    demonstrate_error_handling()
    demonstrate_fstring_formatting()
    demonstrate_professional_development()
    demonstrate_performance_comparison()
    
    print(f"\nESSENTIAL PYTHON SKILLS DEMO COMPLETE!")
    print("=" * 45)
    print("This demonstration showed:")
    print("✓ Virtual environments for project isolation")
    print("✓ Type checking and debugging techniques")
    print("✓ Error handling and validation")
    print("✓ F-string formatting and performance")
    print("✓ Professional development practices")
    print("✓ Logging and documentation")
    print("")
    print("These skills form the foundation for professional")
    print("Python development in data science!")

if __name__ == "__main__":
    main()