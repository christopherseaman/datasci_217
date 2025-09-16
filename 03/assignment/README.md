# Assignment 03: Data Processing with Python Structures

**Due:** Before next class  
**Points:** 20 points total  
**Submit:** Via GitHub repository (link submitted to Canvas)

## Overview

Practice Python data structures and file operations by processing a realistic dataset. You'll work with CSV files, organize data using lists and dictionaries, and generate meaningful analysis results.

This assignment simulates real data science workflows where you receive messy data and need to clean, analyze, and report results.

## Learning Objectives

By completing this assignment, you will:
- Read and write CSV files using Python's csv module
- Organize complex data using lists and dictionaries  
- Clean and process text data using string methods
- Generate summary statistics from datasets
- Create professional output files with results
- Practice defensive programming with file operations

## Dataset Description

You'll work with a fictional "Student Course Evaluations" dataset containing:
- Student information (ID, name, major, year)
- Course ratings (difficulty, usefulness, instructor_rating)
- Comments (free text responses)

The data contains realistic inconsistencies you'll need to handle:
- Mixed case names and majors
- Extra whitespace in text fields
- Some missing ratings (empty strings)
- Inconsistent year formats

## Setup Instructions

1. **Create Repository**
   - Name: `datasci217-assignment03`
   - Clone to your computer using VS Code
   - Create project structure:

```
datasci217-assignment03/
├── README.md
├── requirements.txt
├── src/
│   ├── data_processor.py
│   └── analysis_report.py
├── data/
│   ├── student_evaluations.csv (provided)
│   └── README.md
└── output/
    └── README.md
```

2. **Download Dataset**
   - Canvas → Assignment 03 → download `student_evaluations.csv`
   - Place in `data/` directory
   - **DO NOT** commit this file to Git (it's in .gitignore)

## Requirements

### Part 1: Data Loading and Cleaning (8 points)

Create `src/data_processor.py` with the following functions:

**Function 1: `load_student_data(filename)`**
```python
def load_student_data(filename):
    """
    Load student evaluation data from CSV file.
    
    Args:
        filename (str): Path to CSV file
        
    Returns:
        list: List of dictionaries, one per student record
        
    Example return:
    [
        {
            'student_id': 'S001',
            'name': 'Alice Smith',
            'major': 'Biology',
            'year': 2024,
            'difficulty': 4,
            'usefulness': 5,
            'instructor_rating': 4,
            'comments': 'Great course overall!'
        },
        ...
    ]
    """
    # Your implementation here
    pass
```

**Requirements:**
- Use `csv.DictReader` to read the file
- Handle missing files gracefully (print error, return empty list)
- Convert numeric fields to integers (difficulty, usefulness, instructor_rating, year)
- Handle empty/invalid numeric values by setting them to 0
- Clean text fields by stripping whitespace and applying title case to names/majors

**Function 2: `clean_student_record(record)`**
```python
def clean_student_record(record):
    """
    Clean a single student record dictionary.
    
    Args:
        record (dict): Raw student record from CSV
        
    Returns:
        dict: Cleaned student record
    """
    # Your implementation here
    pass
```

**Requirements:**
- Strip whitespace from all text fields
- Apply title case to name and major fields
- Convert year to integer, handle invalid years by using current year
- Convert rating fields to integers, use 0 for invalid/empty values
- Ensure comments field is a string (empty string if missing)

### Part 2: Data Analysis (6 points)

Create analysis functions in `src/data_processor.py`:

**Function 3: `calculate_major_statistics(students)`**
```python
def calculate_major_statistics(students):
    """
    Calculate statistics grouped by major.
    
    Args:
        students (list): List of student dictionaries
        
    Returns:
        dict: Statistics by major
        
    Example return:
    {
        'Biology': {
            'count': 15,
            'avg_difficulty': 3.8,
            'avg_usefulness': 4.2,
            'avg_instructor': 4.0
        },
        'Chemistry': {
            'count': 12,
            'avg_difficulty': 4.1,
            'avg_usefulness': 3.9,
            'avg_instructor': 3.8
        }
    }
    """
    # Your implementation here
    pass
```

**Function 4: `find_top_students(students, metric='usefulness', top_n=5)`**
```python
def find_top_students(students, metric='usefulness', top_n=5):
    """
    Find students with highest ratings for a given metric.
    
    Args:
        students (list): List of student dictionaries
        metric (str): Rating field to sort by ('difficulty', 'usefulness', 'instructor_rating')
        top_n (int): Number of top students to return
        
    Returns:
        list: List of student dictionaries, sorted by metric (highest first)
    """
    # Your implementation here
    pass
```

### Part 3: Report Generation (4 points)

Create `src/analysis_report.py` with reporting functions:

**Function 5: `generate_summary_report(students, output_file)`**
```python
def generate_summary_report(students, output_file):
    """
    Generate a comprehensive summary report.
    
    Args:
        students (list): List of student dictionaries
        output_file (str): Path for output text file
    """
    # Your implementation here
    pass
```

**Report should include:**
- Total number of evaluations
- Overall average ratings (difficulty, usefulness, instructor)
- Statistics by major (count, averages)
- Top 5 students by usefulness rating
- Bottom 5 students by difficulty rating (easiest experience)

**Function 6: `export_cleaned_data(students, output_file)`**
```python
def export_cleaned_data(students, output_file):
    """
    Export cleaned data to CSV file.
    
    Args:
        students (list): List of cleaned student dictionaries
        output_file (str): Path for output CSV file
    """
    # Your implementation here
    pass
```

### Part 4: Main Script (2 points)

Create a main execution block in `data_processor.py`:

```python
if __name__ == "__main__":
    # Load and clean data
    print("Loading student evaluation data...")
    students = load_student_data("data/student_evaluations.csv")
    print(f"Loaded {len(students)} student records")
    
    # Generate reports
    print("Generating analysis report...")
    generate_summary_report(students, "output/analysis_summary.txt")
    
    print("Exporting cleaned data...")
    export_cleaned_data(students, "output/cleaned_evaluations.csv")
    
    print("Analysis complete! Check output/ directory for results.")
```

## File Structure Requirements

**requirements.txt:**
```
# No external packages needed - using only Python standard library
```

**data/README.md:**
```markdown
# Data Directory

## Files
- `student_evaluations.csv` - Raw evaluation data (not tracked by Git)

## Data Dictionary
- student_id: Unique student identifier
- name: Student full name
- major: Academic major
- year: Graduation year
- difficulty: Course difficulty rating (1-5 scale)
- usefulness: Course usefulness rating (1-5 scale) 
- instructor_rating: Instructor effectiveness rating (1-5 scale)
- comments: Free text feedback
```

**output/README.md:**
```markdown
# Output Directory

Generated files from analysis:
- `analysis_summary.txt` - Summary report with key statistics
- `cleaned_evaluations.csv` - Cleaned version of original data

Files in this directory are generated by scripts and not tracked by Git.
```

**.gitignore** (add these lines):
```
# Data files (often contain sensitive information)
data/*.csv
data/*.xlsx

# Generated output files
output/*.txt
output/*.csv
output/*.json
```

## Sample Expected Output

**analysis_summary.txt** (partial example):
```
Student Course Evaluation Analysis
==================================
Generated: 2024-03-15 14:30:22

OVERALL STATISTICS
Total Evaluations: 47
Average Difficulty: 3.8 (out of 5)
Average Usefulness: 4.1 (out of 5) 
Average Instructor Rating: 4.0 (out of 5)

STATISTICS BY MAJOR
Biology (n=15):
  Avg Difficulty: 3.9
  Avg Usefulness: 4.2
  Avg Instructor: 4.1

Chemistry (n=12):
  Avg Difficulty: 4.0
  Avg Usefulness: 3.8
  Avg Instructor: 3.9

[... continues with more majors ...]

TOP 5 STUDENTS BY USEFULNESS RATING
1. Alice Johnson (Biology) - 5.0
2. Bob Chen (Chemistry) - 5.0  
3. Carol Davis (Physics) - 4.0
4. David Wilson (Biology) - 4.0
5. Emma Thompson (Math) - 4.0

[... more sections ...]
```

## Testing Your Code

Before submitting:

1. **Test with provided data:**
   ```bash
   python src/data_processor.py
   ```

2. **Verify output files:**
   - `output/analysis_summary.txt` should contain readable report
   - `output/cleaned_evaluations.csv` should be valid CSV

3. **Test error handling:**
   - Rename data file temporarily, run script (should handle gracefully)
   - Create CSV with some invalid/missing values

4. **Code quality check:**
   - All functions have docstrings
   - Code uses meaningful variable names
   - No hardcoded values (use parameters/constants)

## Grading Rubric

- **Data Loading & Cleaning (8 pts):** Functions load CSV correctly, handle errors, clean data appropriately
- **Data Analysis (6 pts):** Statistics calculations are correct, functions work with different inputs
- **Report Generation (4 pts):** Reports are readable, comprehensive, and properly formatted
- **Main Script (2 pts):** Script runs without errors, produces expected output files

**Bonus Points (up to 2 pts):**
- Handle edge cases elegantly (empty datasets, all missing values)
- Add additional meaningful statistics or analysis
- Create particularly well-structured, readable code

## Common Pitfalls to Avoid

1. **File paths:** Use `os.path.join()` for cross-platform compatibility
2. **Error handling:** Don't let your script crash on missing/invalid data
3. **Data types:** Convert strings to numbers where appropriate
4. **CSV headers:** Use `DictReader`/`DictWriter` for maintainable code
5. **Hardcoding:** Don't hardcode filenames or values that should be parameters

## Getting Help

- **Office Hours:** [Your office hours]
- **Discord:** #assignment03-help channel
- **Debugging Tips:** Add print statements to see what your data looks like at each step

## Submission Instructions

1. **Commit your code** using good commit messages:
   ```
   git add .
   git commit -m "Implement data loading and cleaning functions"
   git commit -m "Add analysis and reporting functionality"
   git commit -m "Complete main script and documentation"
   ```

2. **Test everything works** by running your script from scratch

3. **Push to GitHub**:
   ```
   git push origin main
   ```

4. **Submit repository link** via Canvas

**Due:** [Insert due date] at 11:59 PM

Late submissions: -10% per day, up to 3 days maximum.