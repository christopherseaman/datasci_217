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

print("=" * 60)
print("DEMO 3: PYTHON BASICS WITH DEBUGGING")
print("=" * 60)
print("Goal: Learn Python fundamentals and practice debugging")
print("Watch for intentional errors - they're learning opportunities!")
print()

# STEP 1: Variables and Data Types
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
# print(f"Student: {student_naem}")  # Typo: naem instead of name
# ❌ ERROR: NameError: name 'student_naem' is not defined
# 💡 FIX: Check spelling! Should be 'student_name'
print(f"✓ Correct: Student: {student_name}")
print()

# INTENTIONAL ERROR 2: Type confusion
print("Another common error - mixing types:")
age_as_text = "25"
print(f"age_as_text = '{age_as_text}' (type: {type(age_as_text).__name__})")

# next_year_age = age_as_text + 1  # Can't add number to string!
# ❌ ERROR: TypeError: can only concatenate str (not "int") to str
# 💡 FIX: Convert string to integer first
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

# STEP 2: F-Strings and Output Formatting
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

# STEP 3: Indentation - Python's Unique Feature
print("STEP 3: Indentation - Python's Unique Feature")
print("-" * 40)

print("Python uses indentation to group code blocks.")
print("Let's see what happens with wrong indentation:")
print()

# INTENTIONAL ERROR 4: Indentation error
print("Attempting code with wrong indentation...")

# This would cause an IndentationError:
# score = 85
# if score >= 80:
# print("Good job!")  # Missing indentation!
#     print("You got a B or better")

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

# STEP 4: Working with Lists
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
# sixth_grade = grades[5]  # We only have 5 grades (indices 0-4)
# ❌ ERROR: IndexError: list index out of range
# 💡 FIX: Valid indices are 0 to 4
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

# STEP 5: Calculations and Common Pitfalls
print("STEP 5: Calculations and Common Pitfalls")
print("-" * 40)

# INTENTIONAL ERROR 6: Division by zero
print("Division by zero error:")
students_present = 0
total_score = 450

# average_score = total_score / students_present
# ❌ ERROR: ZeroDivisionError: division by zero
# 💡 FIX: Always check for zero before dividing
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

# STEP 6: Basic Operations
print("STEP 6: Basic Operations")
print("-" * 40)

# Math operations
print("Math operations:")
result = 10 + 5         # Addition: 15
print(f"  10 + 5 = {result}")
result = 10 - 3         # Subtraction: 7
print(f"  10 - 3 = {result}")
result = 4 * 6          # Multiplication: 24
print(f"  4 * 6 = {result}")
result = 15 / 4         # Division: 3.75
print(f"  15 / 4 = {result}")
result = 15 // 4        # Integer division: 3
print(f"  15 // 4 = {result}")
result = 15 % 4         # Remainder: 3
print(f"  15 % 4 = {result}")
result = 2 ** 3         # Power: 8
print(f"  2 ** 3 = {result}")
print()

# String operations
print("String operations:")
first = "Alice"
last = "Smith"
full_name = first + " " + last        # Concatenation
print(f"  Concatenation: {full_name}")
message = f"Hello {first}!"            # f-string formatting (preferred)
print(f"  F-string: {message}")
print()

# Calculate BMI example
print("Practical example - Calculate BMI:")
weight_kg = 70
height_m = 1.75
bmi = weight_kg / (height_m ** 2)
print(f"  Weight: {weight_kg} kg")
print(f"  Height: {height_m} m")
print(f"  BMI: {bmi:.1f}")
print()

# STEP 7: Reading Python Error Messages
print("STEP 7: Reading Python Error Messages")
print("-" * 40)

print("When Python encounters a problem, it tells you exactly what went wrong.")
print("Learning to read these messages will save you hours of frustration.")
print()

# Common error: trying to use an undefined variable
print("Example error - undefined variable:")
# print(student_naem)  # Typo in variable name
print("❌ NameError: name 'student_naem' is not defined")
print("💡 How to read this error:")
print("   1. Error Type: NameError - Python doesn't recognize the variable name")
print("   2. Error Message: tells you exactly what's wrong")
print("   3. Your Action: Check spelling, make sure you defined the variable first")
print()

# Type errors - mixing incompatible data types
print("Example error - type confusion:")
age = "25"                    # This is text, not a number
# next_year = age + 1          # Can't add number to text
print("❌ TypeError: can only concatenate str (not \"int\") to str")
print("💡 How to fix it:")
age_number = int(age)         # Convert to number
next_year = age_number + 1    # Now this works!
print(f"✓ Converted '{age}' to {age_number}, next year: {next_year}")
print()

# Value Errors - wrong type of value
print("Example error - invalid conversion:")
# bad_number = int("hello")     # Can't convert "hello" to a number
print("❌ ValueError: invalid literal for int() with base 10: 'hello'")
print("💡 This means you tried to convert text that isn't a number")
print()

# Debugging Strategy for Beginners
print("Debugging Strategy for Beginners:")
print("1. Read the error message carefully - Python is usually very specific")
print("2. Check variable names for typos - most common beginner mistake")
print("3. Use print() to check variable values and types")
print("4. Check your data types with type(variable_name)")
print()

# Defensive Programming Example
print("Defensive Programming Example:")
user_input = "42"
print(f"Input: {user_input}")
print(f"Type: {type(user_input)}")       # Shows: <class 'str'>

# Convert and verify
number = int(user_input)
print(f"Converted: {number}")
print(f"New type: {type(number)}")       # Shows: <class 'int'>

# Now you can safely do math
result = number * 2
print(f"Result: {result}")
print()

# Error Prevention Tips
print("Error Prevention Tips:")
print("- Use descriptive variable names - reduces typos")
print("- Check types when debugging - use type() function")
print("- Test with small examples first - don't write 50 lines then run")
print("- One step at a time - add complexity gradually")
print()

# Final summary
print("=" * 60)
print("PYTHON BASICS & DEBUGGING DEMO COMPLETE!")
print("=" * 60)
print()
print("Key Debugging Strategies:")
print("1. Read error messages carefully - they tell you what's wrong")
print("2. Use print() and type() to understand your data")
print("3. Test with edge cases (empty lists, zero values, invalid input)")
print("4. Handle errors gracefully with if/else checks")
print("5. Break complex problems into smaller, testable pieces")
print()
print("Remember: Everyone makes these mistakes - even professionals!")
print("The key is learning to recognize and fix them quickly.")
print()
print("Next: Control structures and more complex operations!")