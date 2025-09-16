Command Line Text Processing and Python Functions

Welcome to week 4! Today we're combining two powerful skillsets: command line text processing tools and Python functions. You'll learn how to quickly explore and clean data using CLI tools, then organize your Python code into reusable functions.

By the end of today, you'll be able to perform rapid data exploration in the terminal and write clean, modular Python code that you can use across multiple projects.

![xkcd 1205: Is It Worth the Time?](media/xkcd_1205.png)

Don't worry - these command line skills save time almost immediately!

Why Command Line Text Processing?

The Problem with GUI-Only Data Exploration

Imagine you have a large CSV file and need to answer these questions:
- How many records are in the file?
- What are the unique values in a particular column?
- Are there any obviously bad data points?
- What's the general shape and quality of the data?

Opening in Excel or a text editor for large files is slow and often crashes. Python scripts work but require writing code for simple questions.

The Command Line Solution

CLI tools let you answer these questions in seconds:
```bash
wc -l data.csv                    # Count lines (records)
cut -d',' -f3 data.csv | sort | uniq    # Unique values in column 3
head -10 data.csv                 # Quick preview
grep "error\|null" data.csv       # Find problematic entries
```

These tools are fast, memory-efficient, and perfect for initial data exploration.

Essential CLI Text Processing Tools

File Content Viewing

head and tail - Quick Data Previews

**Reference:**
```bash
View first 10 lines (default)
head data.csv

View first 5 lines
head -5 data.csv
head -n 5 data.csv

View last 10 lines (default)
tail data.csv

View last 20 lines  
tail -20 data.csv

Combine to see structure
echo "First 5 lines:"
head -5 data.csv
echo "Last 5 lines:"
tail -5 data.csv
```

**Brief Example:**
```bash
Quick data exploration workflow
head -3 student_grades.csv     # See headers and first records
tail -3 student_grades.csv     # See last records  
wc -l student_grades.csv       # Count total records
```

cat - File Display and Combination

**Reference:**
```bash
Display entire file
cat small_data.csv

Display with line numbers
cat -n data.csv

Combine multiple files
cat file1.csv file2.csv > combined.csv

Display non-printing characters (debugging)
cat -A data.csv                # Shows tabs, line endings
```

Text Searching with grep

grep searches for patterns in text files. Essential for finding specific records or data quality issues.

Basic grep Usage

**Reference:**
```bash
Find lines containing specific text
grep "Biology" students.csv

Case-insensitive search
grep -i "biology" students.csv

Count matches instead of showing lines
grep -c "Biology" students.csv

Show line numbers with matches
grep -n "Biology" students.csv

Search multiple patterns
grep "Biology\|Chemistry" students.csv
grep -E "Biology|Chemistry" students.csv    # Extended regex
```

Advanced grep Patterns

**Reference:**
```bash
Find empty lines or missing values
grep "^$" data.csv                # Empty lines
grep ",," data.csv                # Empty CSV fields
grep "null\|NULL\|na\|NA" data.csv    # Common missing value indicators

Find lines that DON'T match pattern
grep -v "header" data.csv         # Exclude header lines

Search in specific columns (combine with cut)
cut -d',' -f3 students.csv | grep "90"    # Find 90 in column 3
```

**Brief Example:**
```bash
Data quality check workflow
grep -c "^$" data.csv              # Count empty lines
grep -c ",," data.csv              # Count records with missing fields  
grep -i "error\|null\|na" data.csv    # Find problematic entries
```

Column Extraction with cut

cut extracts specific columns from delimited files - perfect for CSV analysis.

Basic cut Usage

**Reference:**
```bash
Extract specific columns (1-indexed)
cut -d',' -f1 students.csv         # First column only
cut -d',' -f1,3 students.csv       # Columns 1 and 3
cut -d',' -f1-3 students.csv       # Columns 1 through 3
cut -d',' -f3- students.csv        # Column 3 to end

Different delimiters
cut -d';' -f2 semicolon_data.csv   # Semicolon-separated
cut -d$'\t' -f1,2 tab_data.tsv     # Tab-separated

Extract character positions
cut -c1-10 data.txt                # Characters 1-10
```

Practical cut Applications

**Reference:**
```bash
Extract student names (assuming column 1)
cut -d',' -f1 students.csv

Extract grades only (assuming column 4)
cut -d',' -f4 students.csv

Get just headers
head -1 students.csv | cut -d',' -f1-

Remove first column (useful for IDs)
cut -d',' -f2- students.csv
```

**Brief Example:**
```bash
Quick grade analysis
cut -d',' -f1,4 students.csv      # Names and grades only
cut -d',' -f4 students.csv | head -1    # Grade column header
cut -d',' -f4 students.csv | tail -n +2 # All grades without header
```

Data Organization with sort and uniq

Sorting Data

**Reference:**
```bash
Basic alphabetical sort
sort data.txt

Numerical sort (important for numbers!)
sort -n grades.txt

Reverse sort
sort -r data.txt
sort -nr grades.txt               # Reverse numerical sort

Sort by specific field
sort -t',' -k3 students.csv       # Sort by 3rd field
sort -t',' -k3 -n students.csv    # Numerical sort by 3rd field

Sort by multiple fields
sort -t',' -k2,2 -k3,3n students.csv    # By field 2, then field 3 numerically
```

Finding Unique Values

**Reference:**
```bash
Remove duplicate lines (requires sorted input)
sort data.txt | uniq

Count occurrences of each line
sort data.txt | uniq -c

Show only duplicated lines
sort data.txt | uniq -d

Show only unique lines (no duplicates)
sort data.txt | uniq -u
```

**Brief Example:**
```bash
Find unique majors in student data
cut -d',' -f2 students.csv | sort | uniq

Count students per major
cut -d',' -f2 students.csv | sort | uniq -c

Find duplicate student names
cut -d',' -f1 students.csv | sort | uniq -d
```

LIVE DEMO!
*Exploring a real CSV dataset using command line tools: counting records, finding unique values, checking data quality*

Pipes and Redirection - Combining Tools

Understanding Pipes (|)

Pipes connect the output of one command to the input of another, creating powerful data processing workflows.

Basic Pipe Patterns

**Reference:**
```bash
Basic pipe structure
command1 | command2 | command3

Common patterns
cat file.csv | head -5                    # Show first 5 lines
cut -d',' -f2 students.csv | sort         # Extract and sort column 2
cut -d',' -f3 data.csv | sort | uniq -c   # Count unique values in column 3
```

Complex Analysis Workflows

**Reference:**
```bash
Find most common grade (assuming grades in column 4)
cut -d',' -f4 students.csv | tail -n +2 | sort | uniq -c | sort -nr

Find students with highest grade  
sort -t',' -k4 -nr students.csv | head -5

Count records by major, sorted by frequency
cut -d',' -f2 students.csv | tail -n +2 | sort | uniq -c | sort -nr

Find all unique email domains
cut -d',' -f3 students.csv | cut -d'@' -f2 | sort | uniq -c
```

**Breaking down the complex command:**
```bash
cut -d',' -f4 students.csv    # Extract grades column
| tail -n +2                  # Skip header row  
| sort                        # Sort numerically
| uniq -c                     # Count occurrences
| sort -nr                    # Sort by count (descending)
```

Redirection - Saving Results

Output Redirection

**Reference:**
```bash
Save output to file (overwrites)
cut -d',' -f1 students.csv > student_names.txt

Append to file  
cut -d',' -f2 students.csv >> all_majors.txt

Save both output and errors
grep "pattern" data.csv > results.txt 2> errors.txt

Discard output (useful for commands that produce unwanted output)
command > /dev/null           # Linux/Mac
command > NUL                 # Windows
```

Input Redirection

**Reference:**
```bash
Read from file instead of typing
sort < unsorted_data.txt

Here documents (multi-line input)
cat << EOF > sample_data.csv
name,grade
Alice,85
Bob,92
EOF
```

**Brief Example:**
```bash
Create analysis pipeline with saved intermediate steps
cut -d',' -f1,4 students.csv > names_and_grades.csv
sort -t',' -k2 -nr names_and_grades.csv > sorted_by_grade.csv
head -10 sorted_by_grade.csv > top_students.csv
```

Python Functions - Organizing Your Code

Why Functions Matter

As your Python scripts grow, you'll notice repeated code patterns. Functions help you:
- **Reuse code** without copying and pasting
- **Organize logic** into meaningful chunks  
- **Test individual pieces** of your program
- **Collaborate** by sharing reusable components

From Repeated Code to Functions

**Before (repetitive):**
```python
Calculate grade average for Alice
alice_grades = [85, 90, 87]
alice_average = sum(alice_grades) / len(alice_grades)
print(f"Alice's average: {alice_average}")

Calculate grade average for Bob  
bob_grades = [92, 88, 95]
bob_average = sum(bob_grades) / len(bob_grades)
print(f"Bob's average: {bob_average}")

... repeat for every student
```

**After (with functions):**
```python
def calculate_average(grades):
    """Calculate the average of a list of grades"""
    return sum(grades) / len(grades)

Now reusable for any student
alice_average = calculate_average([85, 90, 87])
bob_average = calculate_average([92, 88, 95])
print(f"Alice's average: {alice_average}")
print(f"Bob's average: {bob_average}")
```

Function Basics

Function Definition and Structure

**Reference:**
```python
def function_name(parameters):
    """
    Optional docstring explaining what the function does
    
    Args:
        parameters: Description of what parameters do
        
    Returns:
        Description of what the function returns
    """
    # Function body
    result = some_calculation()
    return result
```

Simple Function Examples

**Reference:**
```python
Function without parameters
def greet():
    """Print a greeting message"""
    print("Hello from DataSci 217!")

Function with parameters
def greet_student(name):
    """Greet a specific student"""
    print(f"Hello, {name}! Welcome to DataSci 217!")

Function with return value
def calculate_grade_points(grade):
    """Convert letter grade to grade points"""
    if grade >= 90:
        return 4.0
    elif grade >= 80:
        return 3.0
    elif grade >= 70:
        return 2.0
    else:
        return 0.0

Using the functions
greet()
greet_student("Alice")
points = calculate_grade_points(85)
print(f"Grade points: {points}")
```

Function Parameters and Arguments

Different Parameter Types

**Reference:**
```python
Required parameters
def calculate_average(grades):
    return sum(grades) / len(grades)

Default parameters
def calculate_weighted_average(grades, weights=None):
    """Calculate average with optional weights"""
    if weights is None:
        return sum(grades) / len(grades)
    else:
        return sum(g * w for g, w in zip(grades, weights)) / sum(weights)

Multiple parameters
def format_grade_report(name, grades, major="Unknown"):
    """Format a student's grade report"""
    average = calculate_average(grades)
    return f"Student: {name} ({major})\nAverage: {average:.1f}"

Using the functions
avg1 = calculate_average([85, 90, 87])
avg2 = calculate_weighted_average([85, 90, 87], [0.2, 0.3, 0.5])
report = format_grade_report("Alice", [85, 90, 87], "Biology")
```

Parameter Best Practices

**Reference:**
```python
Good: Clear parameter names
def calculate_course_statistics(student_grades, course_name):
    pass

Avoid: Unclear names  
def calculate_stats(data, name):
    pass

Good: Default values for optional parameters
def load_student_data(filename, has_header=True, delimiter=','):
    pass

Good: Docstrings explain parameters
def process_grades(grades, scale_factor=1.0):
    """
    Process a list of grades with optional scaling
    
    Args:
        grades (list): List of numeric grades
        scale_factor (float): Multiplier to apply to grades (default: 1.0)
        
    Returns:
        list: Processed grades
    """
    return [grade * scale_factor for grade in grades]
```

Return Values and Scope

Understanding Return Values

**Reference:**
```python
Function that returns a value
def get_letter_grade(numeric_grade):
    """Convert numeric grade to letter grade"""
    if numeric_grade >= 90:
        return 'A'
    elif numeric_grade >= 80:
        return 'B'
    elif numeric_grade >= 70:
        return 'C'
    else:
        return 'F'

Function that returns multiple values (tuple)
def analyze_grades(grades):
    """Return min, max, and average of grades"""
    return min(grades), max(grades), sum(grades) / len(grades)

Using return values
letter = get_letter_grade(85)
min_grade, max_grade, avg_grade = analyze_grades([85, 90, 78, 92])

Function that doesn't return a value (returns None)
def print_grade_report(name, grade):
    """Print formatted grade report"""
    letter = get_letter_grade(grade)
    print(f"{name}: {grade} ({letter})")
    # No return statement = returns None
```

Variable Scope

**Reference:**
```python
Global variable
course_name = "DataSci 217"

def process_student_grade(name, grade):
    # Local variables
    letter_grade = get_letter_grade(grade)
    status = "Pass" if grade >= 70 else "Fail"
    
    # Can access global variables
    print(f"{course_name}: {name} - {grade} ({letter_grade}) - {status}")
    
    return letter_grade

Global variables accessible everywhere
print(course_name)

Local variables NOT accessible outside function
process_student_grade("Alice", 85)
print(letter_grade)  # This would cause an error!
```

**Brief Example:**
```python
def calculate_final_grade(assignments, midterm, final):
    """Calculate weighted final grade"""
    assignment_avg = sum(assignments) / len(assignments)
    final_grade = (assignment_avg * 0.4) + (midterm * 0.3) + (final * 0.3)
    return final_grade

Usage
alice_final = calculate_final_grade([85, 90, 87], 88, 92)
print(f"Alice's final grade: {alice_final:.1f}")
```

LIVE DEMO!
*Building a complete data analysis script using CLI tools for exploration and Python functions for processing*

Modules and Imports

Understanding Python Modules

Modules are Python files containing functions and variables that you can use in other programs. They help organize code and enable reuse across projects.

Built-in Modules

**Reference:**
```python
Import entire modules
import csv
import os
import math

Use module functions
with open('data.csv', 'r') as file:
    reader = csv.DictReader(file)

file_size = os.path.getsize('data.csv')
square_root = math.sqrt(25)

Import specific functions
from csv import DictReader, DictWriter
from os import path
from math import sqrt, pi

Use without module prefix
with open('data.csv', 'r') as file:
    reader = DictReader(file)

if path.exists('data.csv'):
    print("File exists!")

area = pi * (sqrt(25) ** 2)
```

Creating Your Own Modules

**Create file: `grade_utils.py`**
```python
"""
Utility functions for grade processing
"""

def calculate_average(grades):
    """Calculate average of numeric grades"""
    return sum(grades) / len(grades)

def get_letter_grade(numeric_grade):
    """Convert numeric grade to letter grade"""
    if numeric_grade >= 90:
        return 'A'
    elif numeric_grade >= 80:
        return 'B'
    elif numeric_grade >= 70:
        return 'C'
    else:
        return 'F'

def grade_statistics(grades):
    """Calculate comprehensive grade statistics"""
    return {
        'count': len(grades),
        'average': calculate_average(grades),
        'min': min(grades),
        'max': max(grades),
        'letter': get_letter_grade(calculate_average(grades))
    }

Module-level constants
PASSING_GRADE = 70
GRADE_SCALE = {
    'A': (90, 100),
    'B': (80, 89),
    'C': (70, 79),
    'F': (0, 69)
}
```

**Using your module: `main_analysis.py`**
```python
Import your custom module
import grade_utils

Or import specific functions
from grade_utils import calculate_average, get_letter_grade, PASSING_GRADE

Use the functions
student_grades = [85, 90, 78, 92]
average = grade_utils.calculate_average(student_grades)
letter = grade_utils.get_letter_grade(average)
stats = grade_utils.grade_statistics(student_grades)

print(f"Average: {average:.1f} ({letter})")
print(f"Statistics: {stats}")
print(f"Passing grade threshold: {grade_utils.PASSING_GRADE}")
```

**Brief Example:**
```python
Create reusable data processing functions
File: data_processing.py

def clean_csv_data(filename):
    """Load and clean CSV data"""
    import csv
    students = []
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            clean_row = {
                'name': row['name'].strip().title(),
                'grade': int(row['grade']) if row['grade'] else 0
            }
            students.append(clean_row)
    
    return students

def save_results(data, filename):
    """Save processed data to CSV"""
    import csv
    with open(filename, 'w', newline='') as file:
        if data:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

Usage in main script
from data_processing import clean_csv_data, save_results

students = clean_csv_data('raw_data.csv')
... process students ...
save_results(students, 'processed_data.csv')
```

Combining CLI and Python

Workflow Integration

The most powerful approach combines command line exploration with Python processing:

1. **CLI for rapid exploration** - understand data shape, quality, patterns
2. **Python for complex processing** - clean data, perform calculations, generate reports
3. **CLI for final verification** - check outputs, validate results

Example Workflow

**Step 1: CLI exploration**
```bash
Quick data exploration
head -5 student_data.csv              # See structure
wc -l student_data.csv                # Count records
cut -d',' -f2 student_data.csv | sort | uniq -c    # Major distribution
grep -c ",," student_data.csv         # Missing values check
```

**Step 2: Python processing**
```python
Based on CLI insights, write targeted Python script
import csv

def process_student_data():
    # Load data (we know structure from CLI exploration)
    students = []
    with open('student_data.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Handle missing values found during CLI exploration
            student = {
                'name': row['name'].strip().title(),
                'major': row['major'].strip().title(),
                'grade': int(row['grade']) if row['grade'] else 0
            }
            students.append(student)
    
    # Process based on CLI insights
    # ... analysis code ...
    
    return students

if __name__ == "__main__":
    processed_students = process_student_data()
    print(f"Processed {len(processed_students)} student records")
```

**Step 3: CLI verification**
```bash
Verify Python output
head -5 processed_data.csv            # Check format
wc -l processed_data.csv              # Verify record count
cut -d',' -f3 processed_data.csv | sort -n | tail -5    # Check grade ranges
```

Key Takeaways

1. **CLI tools** provide fast, memory-efficient data exploration and quality checks
2. **grep, cut, sort, uniq** are essential tools for CSV analysis and data exploration
3. **Pipes** combine simple tools into powerful data processing workflows
4. **Functions** organize Python code into reusable, testable components  
5. **Modules** enable code sharing across projects and team collaboration
6. **Combined workflows** leverage the strengths of both CLI and Python approaches

You now have a complete toolkit for data exploration and processing. The command line gives you speed and efficiency for initial analysis, while Python functions provide the structure and power for complex processing tasks.

Next week: We'll dive into Python's package ecosystem and introduce NumPy for numerical computing!

Practice Challenge

Before next class:
1. **CLI Practice:**
   - Download a CSV dataset (or create one)
   - Use head, tail, wc, cut, grep, sort, and uniq to explore the data
   - Create a "data profile" text file with your findings
   
2. **Python Practice:**
   - Create a module with at least 3 functions for data processing
   - Write a main script that imports and uses your module functions
   - Process the same dataset you explored with CLI tools
   
3. **Integration:**
   - Use CLI tools to verify your Python script outputs
   - Document the complete workflow from exploration to final results

Remember: The best data scientists combine multiple tools to work efficiently and effectively!