# Python Data Structures and File Operations

Welcome to week 3! Now that you've mastered the basics of Python and Git, it's time to learn Python's powerful built-in data structures and how to work with files. These are the fundamental building blocks of data science.

By the end of today, you'll know how to organize data using lists and dictionaries, manipulate text files, and handle CSV data - the core skills for any data analysis project.

*[xkcd 353: "Python" - "I wrote 20 short programs in Python yesterday. It was wonderful. Perl, I'm leaving you." Shows a person floating away with "import antigravity"]*

Python's data structures really are that elegant and powerful!

# Why Data Structures Matter

## The Problem with Basic Variables

Imagine tracking student grades with individual variables:
```python
grade_1 = 85
grade_2 = 92  
grade_3 = 78
grade_4 = 95
# ... what about 200 students?
```

Or student information:
```python
student_1_name = "Alice"
student_1_email = "alice@ucsf.edu"  
student_1_grade = 85
student_2_name = "Bob"
# ... this gets unwieldy fast!
```

## The Data Structure Solution

Python's built-in data structures let you organize related data efficiently:

**Lists** handle sequences: `grades = [85, 92, 78, 95]`  
**Dictionaries** handle relationships: `student = {"name": "Alice", "email": "alice@ucsf.edu", "grade": 85}`

Combined, they can represent complex, real-world data structures that scale to thousands of records.

# Essential Data Structures

## Lists - Ordered Collections

Lists store multiple items in a specific order. Think of them as "smart arrays" with powerful built-in methods.

### Creating and Accessing Lists

**Reference:**
```python
# Create lists
grades = [85, 92, 78, 95]
names = ["Alice", "Bob", "Charlie"]
mixed = [85, "Alice", 3.14, True]

# Access items (zero-indexed)
first_grade = grades[0]      # 85
last_grade = grades[-1]      # 95 (negative indexing)

# Slicing
first_two = grades[0:2]      # [85, 92]
last_two = grades[-2:]       # [78, 95]
```

### Essential List Methods

**Reference:**
```python
# Adding items
grades.append(88)            # Add to end: [85, 92, 78, 95, 88]
grades.insert(0, 90)         # Insert at position: [90, 85, 92, 78, 95, 88]
grades.extend([77, 83])      # Add multiple: [90, 85, 92, 78, 95, 88, 77, 83]

# Removing items
grades.remove(78)            # Remove by value: [90, 85, 92, 95, 88, 77, 83]
last_item = grades.pop()     # Remove and return last: 83
second_item = grades.pop(1)  # Remove by index: 85

# Finding items
position = grades.index(92)  # Find index of value: 2
count = grades.count(90)     # Count occurrences: 1

# Sorting
grades.sort()                # Sort in place: [77, 88, 90, 92, 95]
grades.reverse()             # Reverse: [95, 92, 90, 88, 77]
```

**Brief Example:**
```python
# Track student assignments
assignments = []
assignments.append("Assignment 1")
assignments.append("Assignment 2") 
assignments.extend(["Midterm", "Final"])

print(f"Total assignments: {len(assignments)}")
print(f"Next assignment: {assignments[len(assignments)//2]}")
```

## Dictionaries - Key-Value Relationships

Dictionaries store data as key-value pairs. Perfect for representing real-world entities with named properties.

### Creating and Accessing Dictionaries

**Reference:**
```python
# Create dictionaries
student = {"name": "Alice", "email": "alice@ucsf.edu", "grade": 85}
course = {"code": "DATASCI217", "title": "Intro to Data Science", "units": 2}

# Access values
student_name = student["name"]           # "Alice"
student_email = student.get("email")     # "alice@ucsf.edu" 
student_phone = student.get("phone", "Not provided")  # Default value

# Modify dictionaries
student["grade"] = 90                    # Update existing
student["year"] = "Graduate"             # Add new key-value pair
```

### Essential Dictionary Methods

**Reference:**
```python
# Get information about dictionary
keys = student.keys()                    # dict_keys(['name', 'email', 'grade', 'year'])
values = student.values()                # dict_values(['Alice', 'alice@ucsf.edu', 90, 'Graduate'])
items = student.items()                  # Key-value pairs for iteration

# Check existence
if "email" in student:                   # True
    print("Email found!")

# Remove items
removed_grade = student.pop("grade")     # Remove and return value: 90
student.pop("phone", "Not found")       # Remove with default if key missing
```

### Dictionary Iteration Patterns

**Reference:**
```python
# Iterate over keys
for key in student:
    print(f"{key}: {student[key]}")

# Iterate over key-value pairs (preferred)
for key, value in student.items():
    print(f"{key}: {value}")

# Iterate over values only
for value in student.values():
    print(value)
```

**Brief Example:**
```python
# Student record system
students = {
    "alice": {"email": "alice@ucsf.edu", "grade": 85},
    "bob": {"email": "bob@ucsf.edu", "grade": 92}
}

# Look up student
if "alice" in students:
    print(f"Alice's grade: {students['alice']['grade']}")
```

## Sets and Tuples - Quick Overview

**Sets** - Unordered collections of unique items:
```python
unique_grades = {85, 92, 78, 92, 85}     # {78, 85, 92} - duplicates removed
unique_grades.add(95)                    # Add item
unique_grades.remove(78)                 # Remove item
```

**Tuples** - Immutable ordered collections:
```python
coordinates = (37.7749, -122.4194)       # San Francisco lat/lon
x, y = coordinates                       # Unpack values
```

*Use sets for uniqueness, tuples for fixed data that won't change.*

# LIVE DEMO!
*Working with nested data structures: lists of dictionaries representing a class roster*

# File Operations

## Why File I/O Matters for Data Science

Data rarely exists only in your Python script. You'll constantly read data from files (CSV, JSON, text) and write results back to files. File operations are fundamental to any data pipeline.

### Opening and Closing Files

**Reference:**
```python
# Basic file operations
file = open("data.txt", "r")             # Open for reading
content = file.read()                    # Read entire file
file.close()                            # Always close!

# Better: using 'with' statement (automatically closes)
with open("data.txt", "r") as file:
    content = file.read()
# File automatically closed when leaving 'with' block
```

**File modes:**
- `"r"` - Read (default)
- `"w"` - Write (overwrites existing file)  
- `"a"` - Append to end of file
- `"r+"` - Read and write

### Reading Files

**Reference:**
```python
# Different ways to read files
with open("data.txt", "r") as file:
    entire_file = file.read()            # Read all content as string
    
with open("data.txt", "r") as file:
    lines = file.readlines()             # Read all lines as list
    
with open("data.txt", "r") as file:
    for line in file:                    # Iterate line by line (memory efficient)
        print(line.strip())              # .strip() removes newlines
```

### Writing Files

**Reference:**
```python
# Write text to file
with open("output.txt", "w") as file:
    file.write("Hello, DataSci 217!\n")
    file.write("This is line 2\n")

# Write multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("output.txt", "w") as file:
    file.writelines(lines)

# Append to existing file
with open("output.txt", "a") as file:
    file.write("This gets added to the end\n")
```

### File Paths and Safety

**Reference:**
```python
import os

# Check if file exists before reading
if os.path.exists("data.txt"):
    with open("data.txt", "r") as file:
        content = file.read()
else:
    print("File not found!")

# Build cross-platform file paths
data_path = os.path.join("data", "students.csv")    # data/students.csv or data\students.csv
results_path = os.path.join("output", "results.txt")
```

**Brief Example:**
```python
# Save analysis results
results = ["Mean: 87.5", "Median: 89.0", "Count: 25"]

with open("analysis_results.txt", "w") as file:
    file.write("Statistical Analysis Results\n")
    file.write("=" * 30 + "\n")
    for result in results:
        file.write(f"{result}\n")
        
print("Results saved to analysis_results.txt")
```

## CSV File Handling

CSV (Comma-Separated Values) files are the most common data format in data science. Python's `csv` module makes them easy to work with.

### Reading CSV Files

**Reference:**
```python
import csv

# Read CSV as list of lists
with open("students.csv", "r") as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)            # First row (column names)
    
    for row in csv_reader:
        print(row)                       # Each row is a list

# Read CSV as list of dictionaries (preferred for data science)
with open("students.csv", "r") as file:
    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
        print(row)                       # Each row is a dictionary
        print(f"Name: {row['name']}, Grade: {row['grade']}")
```

### Writing CSV Files

**Reference:**
```python
import csv

# Write list of lists
data = [
    ["name", "email", "grade"],
    ["Alice", "alice@ucsf.edu", 85],
    ["Bob", "bob@ucsf.edu", 92]
]

with open("output.csv", "w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data)

# Write list of dictionaries (preferred)
students = [
    {"name": "Alice", "email": "alice@ucsf.edu", "grade": 85},
    {"name": "Bob", "email": "bob@ucsf.edu", "grade": 92}
]

with open("students_output.csv", "w", newline="") as file:
    fieldnames = ["name", "email", "grade"]
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    csv_writer.writeheader()             # Write column names
    csv_writer.writerows(students)       # Write all rows
```

**Brief Example:**
```python
# Process student data from CSV
import csv

students = []
with open("class_roster.csv", "r") as file:
    reader = csv.DictReader(file)
    for student in reader:
        students.append({
            "name": student["name"],
            "grade": int(student["grade"]),
            "status": "Pass" if int(student["grade"]) >= 70 else "Needs Help"
        })

# Write processed results
with open("grade_report.csv", "w", newline="") as file:
    fieldnames = ["name", "grade", "status"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(students)
```

# String Operations and Methods

Strings are everywhere in data science - column names, data cleaning, text processing. Python's string methods are incredibly powerful.

### Essential String Methods

**Reference:**
```python
text = "  DataSci 217: Introduction to Data Science  "

# Cleaning and formatting
clean_text = text.strip()                # Remove whitespace: "DataSci 217: Introduction to Data Science"
upper_text = text.upper()                # "  DATASCI 217: INTRODUCTION TO DATA SCIENCE  "
lower_text = text.lower()                # "  datasci 217: introduction to data science  "
title_text = text.title()                # "  Datasci 217: Introduction To Data Science  "

# Searching and checking
contains_data = "Data" in text           # True
starts_with = text.startswith("  Data")  # True  
ends_with = text.endswith("Science  ")   # True
position = text.find("217")              # 10 (index where "217" starts)

# Splitting and joining
words = text.strip().split(" ")          # ["DataSci", "217:", "Introduction", "to", "Data", "Science"]
parts = text.split(":")                  # ["  DataSci 217", " Introduction to Data Science  "]
rejoined = " | ".join(words)             # "DataSci | 217: | Introduction | to | Data | Science"

# Replacing
updated = text.replace("DataSci", "Data Science")
cleaned = text.replace("  ", " ")        # Replace double spaces with single
```

### String Formatting

**Reference:**
```python
name = "Alice"
grade = 87.5
year = 2024

# f-strings (preferred, Python 3.6+)
message = f"Student {name} earned {grade:.1f}% in {year}"

# .format() method  
message = "Student {} earned {:.1f}% in {}".format(name, grade, year)

# % formatting (older style)
message = "Student %s earned %.1f%% in %d" % (name, grade, year)
```

**Brief Example:**
```python
# Clean student names from messy data
messy_names = ["  alice SMITH  ", "BOB jones", "  Charlie Brown "]

clean_names = []
for name in messy_names:
    clean_name = name.strip().title()
    clean_names.append(clean_name)

print(clean_names)  # ["Alice Smith", "Bob Jones", "Charlie Brown"]
```

# LIVE DEMO!
*Reading a CSV file into a list of dictionaries, processing the data, and writing results to a new file*

# Combining Data Structures and Files

This is where the magic happens - using Python's data structures to organize and process data from files.

## Real-World Example: Student Grade Analysis

**Reference Pattern:**
```python
import csv

# Read data from CSV into list of dictionaries
students = []
with open("class_grades.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        students.append({
            "name": row["name"],
            "assignment1": int(row["assignment1"]),
            "assignment2": int(row["assignment2"]),
            "midterm": int(row["midterm"]),
            "final": int(row["final"])
        })

# Process data using dictionaries and lists
for student in students:
    # Calculate total grade
    grades = [student["assignment1"], student["assignment2"], 
              student["midterm"], student["final"]]
    student["total"] = sum(grades) / len(grades)
    
    # Determine letter grade
    if student["total"] >= 90:
        student["letter_grade"] = "A"
    elif student["total"] >= 80:
        student["letter_grade"] = "B"
    elif student["total"] >= 70:
        student["letter_grade"] = "C"
    else:
        student["letter_grade"] = "F"

# Write results back to CSV
with open("final_grades.csv", "w", newline="") as file:
    fieldnames = ["name", "assignment1", "assignment2", "midterm", "final", "total", "letter_grade"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(students)

# Generate summary statistics
total_students = len(students)
passing_students = len([s for s in students if s["letter_grade"] != "F"])
average_grade = sum(s["total"] for s in students) / total_students

print(f"Class Summary:")
print(f"Total students: {total_students}")
print(f"Passing students: {passing_students} ({passing_students/total_students:.1%})")
print(f"Class average: {average_grade:.1f}")
```

This example demonstrates the power of combining data structures with file I/O for real data analysis tasks.

# Key Takeaways

1. **Lists** organize ordered data and provide powerful methods for manipulation
2. **Dictionaries** represent structured data with meaningful names for properties
3. **File operations** with the `with` statement ensure safe and clean file handling
4. **CSV module** makes working with structured data files straightforward
5. **String methods** are essential for cleaning and processing text data
6. **Combining these tools** enables powerful data processing workflows

You now have the core tools for organizing and processing data in Python. These patterns - reading CSV files into lists of dictionaries, processing the data, and writing results - form the foundation of most data analysis scripts.

Next week: We'll learn command line text processing tools and dive deeper into Python functions to make your code more organized and reusable!

# Practice Challenge

Before next class:
1. Create a CSV file with fictional student data (name, age, major, GPA)
2. Write a Python script that:
   - Reads the CSV into a list of dictionaries
   - Calculates average GPA by major
   - Finds students with GPA above 3.5
   - Writes the results to a new CSV file
3. Use meaningful variable names and add comments
4. Test your script with different data to make sure it works

Remember: The best way to learn these concepts is by practicing with real data!