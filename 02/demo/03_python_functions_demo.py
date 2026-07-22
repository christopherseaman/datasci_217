#!/usr/bin/env python3
"""
Python Functions Demo - Building from inline to functions to modules
This script demonstrates the evolution from simple inline code to reusable functions
"""

# Step 1: Start with inline script (simple data processing)
print("=== Step 1: Inline Script ===")

# Create sample data with some messy strings for cleaning
students = [
    {"name": "  Alice  ", "age": 20, "grade": 85, "subject": "MATH"},
    {"name": "bob", "age": 19, "grade": 92, "subject": "science"},
    {"name": "CHARLIE", "age": 21, "grade": 78, "subject": "english"},
    {"name": "Diana", "age": 20, "grade": 88, "subject": "Math"},
    {"name": "Eve", "age": 22, "grade": 95, "subject": "Science"}
]

# String Operations for Data Cleaning
print("=== String Operations for Data Cleaning ===")
for student in students:
    # Clean name: strip whitespace and title case
    clean_name = student['name'].strip().title()
    # Clean subject: convert to title case
    clean_subject = student['subject'].lower().title()
    
    print(f"Original: '{student['name']}' -> Clean: '{clean_name}'")
    print(f"Subject: '{student['subject']}' -> Clean: '{clean_subject}'")

# Control Flow Structures
print("\n=== Control Flow Structures ===")
print("Using if/elif/else for grade classification:")
for student in students:
    grade = student['grade']
    if grade >= 90:
        classification = "A"
    elif grade >= 80:
        classification = "B"
    elif grade >= 70:
        classification = "C"
    else:
        classification = "F"
    
    clean_name = student['name'].strip().title()
    print(f"{clean_name}: {grade} -> {classification}")

# List Comprehensions
print("\n=== List Comprehensions ===")
# Extract grades using list comprehension
grades = [student['grade'] for student in students]
print(f"All grades: {grades}")

# Filter high grades using list comprehension
high_grades = [grade for grade in grades if grade >= 85]
print(f"High grades (85+): {high_grades}")

# Sequence Functions
print("\n=== Sequence Functions ===")
# Using enumerate to get index and value
print("Students with index:")
for i, student in enumerate(students):
    clean_name = student['name'].strip().title()
    print(f"{i}: {clean_name} - {student['grade']}")

# Using zip to combine lists
names = [student['name'].strip().title() for student in students]
scores = [student['grade'] for student in students]
print("\nNames and scores paired:")
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# Using sorted and reversed
print(f"\nSorted grades: {sorted(grades)}")
print(f"Reversed sorted: {list(reversed(sorted(grades)))}")

# Basic processing with while loop
print("\n=== While Loop Example ===")
count = 0
while count < len(students):
    student = students[count]
    clean_name = student['name'].strip().title()
    print(f"Processing {count + 1}: {clean_name}")
    count += 1

# Inline processing
print("\n=== Basic Analysis ===")
total_grade = 0
for student in students:
    clean_name = student['name'].strip().title()
    print(f"{clean_name}: {student['grade']}")
    total_grade += student['grade']

average_grade = total_grade / len(students)
print(f"Average grade: {average_grade:.1f}")

# Find highest grade
highest_grade = max(student['grade'] for student in students)
print(f"Highest grade: {highest_grade}")

print("\n" + "="*50)

# Step 2: Refactor into functions
print("=== Step 2: Refactoring into Functions ===")

def calculate_average(grades):
    """Calculate the average of a list of grades."""
    if not grades:
        return 0
    return sum(grades) / len(grades)

def find_highest_grade(grades):
    """Find the highest grade in a list."""
    if not grades:
        return 0
    return max(grades)

def print_student_grades(students):
    """Print all student grades in a formatted way."""
    for student in students:
        print(f"{student['name']}: {student['grade']}")

def get_grades_list(students):
    """Extract grades from student list."""
    return [student['grade'] for student in students]

# Use the functions
print("Student grades:")
print_student_grades(students)

grades = get_grades_list(students)
average = calculate_average(grades)
highest = find_highest_grade(grades)

print(f"Average grade: {average:.1f}")
print(f"Highest grade: {highest}")

print("\n" + "="*50)

# Step 3: Add error handling and validation
print("=== Step 3: Adding Error Handling ===")

def safe_calculate_average(grades):
    """Calculate average with error handling."""
    try:
        if not grades:
            print("Warning: No grades provided")
            return 0
        
        # Check if all grades are numeric
        for grade in grades:
            if not isinstance(grade, (int, float)):
                raise ValueError(f"Invalid grade type: {type(grade)}")
        
        return sum(grades) / len(grades)
    except Exception as e:
        print(f"Error calculating average: {e}")
        return 0

def validate_student_data(students):
    """Validate student data structure."""
    required_fields = ['name', 'age', 'grade', 'subject']
    
    for i, student in enumerate(students):
        for field in required_fields:
            if field not in student:
                print(f"Warning: Student {i+1} missing field '{field}'")
                return False
    return True

# Test with valid data
print("Testing with valid data:")
if validate_student_data(students):
    grades = get_grades_list(students)
    average = safe_calculate_average(grades)
    print(f"Average grade: {average:.1f}")

# Test with invalid data
print("\nTesting with invalid data:")
invalid_students = [
    {"name": "Alice", "age": 20, "grade": "A"},  # String grade
    {"name": "Bob", "age": 19, "grade": 92, "subject": "Science"}
]
if validate_student_data(invalid_students):
    grades = get_grades_list(invalid_students)
    average = safe_calculate_average(grades)
    print(f"Average grade: {average:.1f}")

print("\n" + "="*50)

# Step 4: Add file I/O operations
print("=== Step 4: Adding File I/O ===")

def save_results_to_file(filename, students, average, highest):
    """Save analysis results to a file."""
    try:
        with open(filename, 'w') as file:
            file.write("Student Grade Analysis\n")
            file.write("=" * 30 + "\n\n")
            
            file.write("Individual Grades:\n")
            for student in students:
                file.write(f"{student['name']}: {student['grade']}\n")
            
            file.write(f"\nSummary:\n")
            file.write(f"Average grade: {average:.1f}\n")
            file.write(f"Highest grade: {highest}\n")
            file.write(f"Total students: {len(students)}\n")
        
        print(f"Results saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def load_students_from_csv(filename):
    """Load student data from CSV file."""
    students = []
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            if len(lines) < 2:
                print("Error: CSV file must have header and at least one data row")
                return []
            
            # Skip header line
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    students.append({
                        'name': parts[0],
                        'age': int(parts[1]),
                        'grade': int(parts[2]),
                        'subject': parts[3]
                    })
        
        print(f"Loaded {len(students)} students from {filename}")
        return students
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return []
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

# Create sample CSV file
csv_content = """name,age,grade,subject
Alice,20,85,Math
Bob,19,92,Science
Charlie,21,78,English
Diana,20,88,Math
Eve,22,95,Science"""

with open('sample_students.csv', 'w') as file:
    file.write(csv_content)

print("Sample CSV file created: sample_students.csv")

# Load data from CSV
loaded_students = load_students_from_csv('sample_students.csv')

if loaded_students:
    grades = get_grades_list(loaded_students)
    average = safe_calculate_average(grades)
    highest = find_highest_grade(grades)
    
    # Save results
    save_results_to_file('analysis_results.txt', loaded_students, average, highest)

print("\n" + "="*50)

# Step 5: Add __main__ execution control
print("=== Step 5: Adding __main__ Execution Control ===")

def main():
    """Main function to run the analysis."""
    print("Running student grade analysis...")
    
    # Load data
    students = load_students_from_csv('sample_students.csv')
    if not students:
        print("No data to analyze")
        return
    
    # Perform analysis
    grades = get_grades_list(students)
    average = safe_calculate_average(grades)
    highest = find_highest_grade(grades)
    
    # Display results
    print("\nAnalysis Results:")
    print("-" * 20)
    print_student_grades(students)
    print(f"\nAverage grade: {average:.1f}")
    print(f"Highest grade: {highest}")
    
    # Save results
    save_results_to_file('analysis_results.txt', students, average, highest)

# This code only runs when the script is executed directly
# It won't run when the script is imported as a module
if __name__ == "__main__":
    main()

print("\n" + "="*50)
print("=== Demo Complete ===")
print("Key concepts demonstrated:")
print("1. Inline script development")
print("2. Function extraction and refactoring")
print("3. Error handling and validation")
print("4. File I/O operations")
print("5. __main__ execution control")
print("\nNext: See how these functions can be imported and used in other scripts!")
