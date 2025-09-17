#!/usr/bin/env python3
"""
Demo 3: Python Basics with Debugging Practice
Lecture 01 - Command Line + Python

This demo covers Python fundamentals with intentional errors
to practice debugging, corresponding to the "Python Basics" 
and "Debugging and Error Handling" sections of the lecture.

Usage: python 03_python_basics_demo.py

Author: Data Science 217 Course Materials
"""

def demo_header():
    """Display demo introduction"""
    print("=" * 60)
    print("DEMO 3: PYTHON BASICS WITH DEBUGGING")
    print("=" * 60)
    print("Goal: Learn Python fundamentals and practice debugging")
    print("Watch for intentional errors - they're learning opportunities!")
    print()

def demonstrate_variables_and_types():
    """Show variables, data types, and common type errors"""
    print("STEP 1: Variables and Data Types")
    print("-" * 40)
    
    # Good practice - clear variable names
    print("Creating variables with meaningful names:")
    student_name = "Alice Johnson"
    student_age = 22
    gpa = 3.85
    is_enrolled = True
    
    print(f"  Name: {student_name} (type: {type(student_name).__name__})")
    print(f"  Age: {student_age} (type: {type(student_age).__name__})")
    print(f"  GPA: {gpa} (type: {type(gpa).__name__})")
    print(f"  Enrolled: {is_enrolled} (type: {type(is_enrolled).__name__})")
    print()
    
    # INTENTIONAL ERROR 1: Variable name typo
    print("Let's make a common mistake - typo in variable name:")
    try:
        print(f"Student: {student_naem}")  # Typo: naem instead of name
    except NameError as e:
        print(f"❌ ERROR: {e}")
        print("💡 FIX: Check spelling! Should be 'student_name'")
        print(f"✓ Correct: Student: {student_name}")
    print()
    
    # INTENTIONAL ERROR 2: Type confusion
    print("Another common error - mixing types:")
    age_as_text = "25"
    print(f"age_as_text = '{age_as_text}' (type: {type(age_as_text).__name__})")
    
    try:
        next_year_age = age_as_text + 1  # Can't add number to string!
    except TypeError as e:
        print(f"❌ ERROR: {e}")
        print("💡 FIX: Convert string to integer first")
        age_as_number = int(age_as_text)
        next_year_age = age_as_number + 1
        print(f"✓ Correct: {age_as_number} + 1 = {next_year_age}")
    print()
    
    # Good debugging practice
    print("Debugging Tip: Always check data types when unsure!")
    mystery_value = "42"
    print(f"mystery_value = '{mystery_value}'")
    print(f"  Is it a string? {isinstance(mystery_value, str)}")
    print(f"  Is it an int? {isinstance(mystery_value, int)}")
    print(f"  Can we convert it? {mystery_value.isdigit()}")
    print()

def demonstrate_f_strings_and_formatting():
    """Show f-string formatting with common mistakes"""
    print("STEP 2: F-Strings and Output Formatting")
    print("-" * 40)
    
    # Good f-string examples
    print("Proper f-string formatting:")
    name = "Bob"
    score = 87.3456
    
    print(f"  Basic: {name} scored {score}")
    print(f"  Rounded: {name} scored {score:.1f}")
    print(f"  Percentage: {name} scored {score:.0f}%")
    print()
    
    # INTENTIONAL ERROR 3: Forgetting the 'f' prefix
    print("Common mistake - forgetting the 'f' in f-string:")
    # This won't format correctly
    bad_output = "Student: {name}, Score: {score}"  # Missing 'f' prefix!
    print(f"  Without 'f': {bad_output}")
    
    # Correct version
    good_output = f"Student: {name}, Score: {score:.1f}"
    print(f"  With 'f': {good_output}")
    print()
    
    # Advanced formatting for data science
    print("Data Science Formatting Examples:")
    
    # Financial data
    revenue = 15432.50
    print(f"  Revenue: ${revenue:,.2f}")
    
    # Scientific notation
    big_number = 1400000000
    print(f"  Population: {big_number:.2e}")
    
    # Percentage
    success_rate = 0.847
    print(f"  Success Rate: {success_rate:.1%}")
    
    # Table formatting
    print()
    print("  Formatted Table:")
    print(f"  {'Name':<10} {'Score':>8} {'Grade':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8}")
    print(f"  {'Alice':<10} {92.5:>8.1f} {'A':>8}")
    print(f"  {'Bob':<10} {87.3:>8.1f} {'B+':>8}")
    print()

def demonstrate_indentation_errors():
    """Show Python's indentation requirements"""
    print("STEP 3: Indentation - Python's Unique Feature")
    print("-" * 40)
    
    print("Python uses indentation to group code blocks.")
    print("Let's see what happens with wrong indentation:")
    print()
    
    # INTENTIONAL ERROR 4: Indentation error
    print("Attempting code with wrong indentation...")
    
    # This string contains intentionally bad Python code
    bad_code = """
score = 85
if score >= 80:
print("Good job!")  # Missing indentation!
    print("You got a B or better")
"""
    
    print("Bad code example:")
    print("```python")
    print(bad_code.strip())
    print("```")
    print()
    print("❌ This would cause: IndentationError")
    print("💡 FIX: Always indent code blocks with 4 spaces")
    print()
    
    # Correct version
    print("✓ Correct indentation:")
    score = 85
    if score >= 80:
        print("    Good job!")  # Properly indented
        print("    You got a B or better")
    print()

def demonstrate_list_operations():
    """Show list operations and common indexing errors"""
    print("STEP 4: Working with Lists")
    print("-" * 40)
    
    # Create a list
    grades = [85, 92, 78, 96, 88]
    print(f"Grades: {grades}")
    print(f"Number of grades: {len(grades)}")
    print()
    
    # Accessing elements
    print("Accessing list elements:")
    print(f"  First grade (index 0): {grades[0]}")
    print(f"  Last grade (index -1): {grades[-1]}")
    print()
    
    # INTENTIONAL ERROR 5: Index out of range
    print("Common mistake - index out of range:")
    try:
        # Lists are 0-indexed, so index 5 doesn't exist for 5 items
        sixth_grade = grades[5]  # We only have 5 grades (indices 0-4)
    except IndexError as e:
        print(f"❌ ERROR: {e}")
        print(f"💡 FIX: Valid indices are 0 to {len(grades)-1}")
        print(f"✓ Last grade using index {len(grades)-1}: {grades[len(grades)-1]}")
        print(f"✓ Or simply use -1: {grades[-1]}")
    print()
    
    # List operations
    print("Basic list operations:")
    grades.append(90)
    print(f"  After append(90): {grades}")
    
    average = sum(grades) / len(grades)
    print(f"  Average: {average:.1f}")
    print()

def demonstrate_common_calculation_errors():
    """Show common mathematical and logical errors"""
    print("STEP 5: Calculations and Common Pitfalls")
    print("-" * 40)
    
    # INTENTIONAL ERROR 6: Division by zero
    print("Division by zero error:")
    students_present = 0
    total_score = 450
    
    try:
        average_score = total_score / students_present
    except ZeroDivisionError as e:
        print(f"❌ ERROR: {e}")
        print("💡 FIX: Always check for zero before dividing")
        if students_present > 0:
            average_score = total_score / students_present
        else:
            average_score = 0
            print("✓ No students present, average set to 0")
    print()
    
    # INTENTIONAL ERROR 7: Off-by-one error
    print("Off-by-one error in loops:")
    numbers = [1, 2, 3, 4, 5]
    
    # Wrong way - trying to use 1-based indexing
    print("  Attempting to print with wrong range:")
    print("  for i in range(1, len(numbers)): # Misses first element!")
    for i in range(1, len(numbers)):
        print(f"    Position {i}: {numbers[i]}")
    
    print()
    print("💡 Notice we missed the first element (1)!")
    print("✓ Correct way - using 0-based indexing:")
    for i in range(len(numbers)):
        print(f"    Position {i}: {numbers[i]}")
    print()
    
    # Float precision issues
    print("Float precision gotcha:")
    result = 0.1 + 0.1 + 0.1
    print(f"  0.1 + 0.1 + 0.1 = {result}")
    print(f"  Is it exactly 0.3? {result == 0.3}")
    print(f"  Actual value: {result:.20f}")
    print("💡 Use rounding for display: {:.2f}".format(result))
    print()

def demonstrate_debugging_workflow():
    """Show a systematic debugging approach"""
    print("STEP 6: Debugging Workflow")
    print("-" * 40)
    
    print("Let's debug a broken function step by step:")
    print()
    
    def calculate_grade_broken(scores):
        """This function has multiple bugs - let's fix them!"""
        # Bug 1: Not checking for empty list
        # Bug 2: Using wrong variable name
        # Bug 3: Integer division instead of float
        
        total = sum(scores)
        average = total / len(scores)  # What if scores is empty?
        
        # Determine letter grade
        if average >= 90:
            grade = "A"
        elif averge >= 80:  # Typo here!
            grade = "B"
        else:
            grade = "C"
        
        return grade
    
    print("Testing the broken function:")
    print()
    
    # Test 1: Empty list
    print("Test 1: Empty list")
    try:
        result = calculate_grade_broken([])
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("💡 Need to check for empty list!")
    print()
    
    # Test 2: Fix and try again
    def calculate_grade_fixed(scores):
        """Fixed version with proper error handling"""
        # Fix 1: Check for empty list
        if not scores:
            return "No scores"
        
        total = sum(scores)
        average = total / len(scores)
        
        # Fix 2: Correct variable name
        if average >= 90:
            grade = "A"
        elif average >= 80:  # Fixed typo
            grade = "B"
        elif average >= 70:
            grade = "C"
        elif average >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return f"{grade} (avg: {average:.1f})"
    
    print("Testing the fixed function:")
    test_cases = [
        [],
        [95, 92, 98],
        [85, 82, 88],
        [75, 72, 78],
    ]
    
    for scores in test_cases:
        result = calculate_grade_fixed(scores)
        print(f"  Scores {scores}: {result}")
    print()

def demonstrate_real_world_debugging():
    """Show debugging in a real data science context"""
    print("STEP 7: Real-World Data Science Debugging")
    print("-" * 40)
    
    print("Let's debug a data analysis pipeline:")
    print()
    
    # Simulate loading data with potential issues
    raw_data = [
        "Alice,25,87.5",
        "Bob,30,92.1",
        "Charlie,28,",  # Missing score!
        "Diana,26,95.2",
        "Eve,unknown,88.3"  # Invalid age!
    ]
    
    print("Raw data (with problems):")
    for line in raw_data:
        print(f"  {line}")
    print()
    
    print("Processing data with error handling:")
    students = []
    
    for i, line in enumerate(raw_data, 1):
        parts = line.split(',')
        
        # Debug output
        print(f"Line {i}: {line}")
        
        try:
            name = parts[0]
            
            # Handle age conversion
            try:
                age = int(parts[1])
            except ValueError:
                print(f"  ⚠️ Invalid age '{parts[1]}' - using None")
                age = None
            
            # Handle score conversion
            try:
                score = float(parts[2]) if parts[2] else None
            except (ValueError, IndexError):
                print(f"  ⚠️ Invalid/missing score - using None")
                score = None
            
            students.append({
                'name': name,
                'age': age,
                'score': score
            })
            
            if age and score:
                print(f"  ✓ Processed: {name}, age {age}, score {score}")
            
        except Exception as e:
            print(f"  ❌ Failed to process: {e}")
    
    print()
    print("Final processed data:")
    for student in students:
        print(f"  {student}")
    
    # Calculate statistics with None handling
    valid_scores = [s['score'] for s in students if s['score'] is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\nAverage score (excluding missing): {avg_score:.1f}")
    else:
        print("\nNo valid scores to calculate average")
    
    print()

def main():
    """Run all Python basics demos"""
    demo_header()
    
    # Run all demonstration steps
    demonstrate_variables_and_types()
    demonstrate_f_strings_and_formatting()
    demonstrate_indentation_errors()
    demonstrate_list_operations()
    demonstrate_common_calculation_errors()
    demonstrate_debugging_workflow()
    demonstrate_real_world_debugging()
    
    # Final summary
    print("=" * 60)
    print("PYTHON BASICS & DEBUGGING DEMO COMPLETE!")
    print("=" * 60)
    print()
    print("Key Debugging Strategies:")
    print("1. Read error messages carefully - they tell you what's wrong")
    print("2. Use print() and type() to understand your data")
    print("3. Test with edge cases (empty lists, zero values, invalid input)")
    print("4. Handle errors gracefully with try/except")
    print("5. Break complex problems into smaller, testable pieces")
    print()
    print("Remember: Everyone makes these mistakes - even professionals!")
    print("The key is learning to recognize and fix them quickly.")
    print()
    print("Next: Control structures and more complex operations!")

if __name__ == "__main__":
    main()