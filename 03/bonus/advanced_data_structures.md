# Bonus Content: Advanced Data Structures and Techniques

*This content is optional and not required for assignments. It's here for students who want to dive deeper into Python's data manipulation capabilities.*

## Advanced List Operations

### List Comprehensions

List comprehensions provide a concise way to create lists based on existing sequences:

```python
# Basic list comprehension
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]  # [1, 4, 9, 16, 25]

# With conditional filtering
even_squares = [x**2 for x in numbers if x % 2 == 0]  # [4, 16]

# Processing strings
names = ["alice", "BOB", "Charlie"]
clean_names = [name.title() for name in names]  # ["Alice", "Bob", "Charlie"]

# Multiple conditions
grades = [85, 92, 78, 95, 67, 88]
letter_grades = ["A" if g >= 90 else "B" if g >= 80 else "C" if g >= 70 else "F" 
                 for g in grades]
```

### Advanced List Methods and Techniques

```python
# Sorting with custom keys
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]

# Sort by grade (descending)
students_by_grade = sorted(students, key=lambda x: x["grade"], reverse=True)

# Sort by multiple criteria (grade, then name)
students.sort(key=lambda x: (-x["grade"], x["name"]))

# Finding with conditions
high_performers = [s for s in students if s["grade"] >= 90]
top_student = max(students, key=lambda x: x["grade"])
```

### Working with Nested Lists

```python
# Matrix operations
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Transpose matrix
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
# Result: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

# Flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flattened = [item for sublist in nested for item in sublist]  # [1, 2, 3, 4, 5, 6]

# Process 2D data
grade_matrix = [
    ["Alice", 85, 90, 87],
    ["Bob", 92, 88, 95],
    ["Charlie", 78, 82, 80]
]

# Calculate averages for each student
student_averages = [
    [row[0], sum(row[1:]) / len(row[1:])] 
    for row in grade_matrix
]
```

## Advanced Dictionary Operations

### Dictionary Comprehensions

```python
# Basic dictionary comprehension
numbers = [1, 2, 3, 4, 5]
squares_dict = {x: x**2 for x in numbers}  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# From lists to dictionary
names = ["Alice", "Bob", "Charlie"]
grades = [85, 92, 78]
grade_dict = {name: grade for name, grade in zip(names, grades)}

# Filter and transform
student_data = {"alice": 85, "bob": 92, "charlie": 67, "diana": 95}
passing_students = {name.title(): grade for name, grade in student_data.items() if grade >= 70}
```

### Merging and Updating Dictionaries

```python
# Python 3.9+ dictionary merge operator
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
merged = dict1 | dict2  # {"a": 1, "b": 2, "c": 3, "d": 4}

# Update with conflict resolution
dict1 |= dict2  # In-place update

# Traditional methods (all Python versions)
merged = {**dict1, **dict2}
merged = dict(dict1, **dict2)

# Update with custom logic
def merge_student_records(old_record, new_record):
    """Merge student records, keeping highest grade"""
    merged = old_record.copy()
    for key, value in new_record.items():
        if key == "grade":
            merged[key] = max(old_record.get(key, 0), value)
        else:
            merged[key] = value
    return merged
```

### defaultdict for Cleaner Code

```python
from collections import defaultdict

# Group students by major
students = [
    {"name": "Alice", "major": "Biology"},
    {"name": "Bob", "major": "Chemistry"},
    {"name": "Charlie", "major": "Biology"}
]

# Without defaultdict (more verbose)
majors_dict = {}
for student in students:
    major = student["major"]
    if major not in majors_dict:
        majors_dict[major] = []
    majors_dict[major].append(student["name"])

# With defaultdict (cleaner)
majors_dict = defaultdict(list)
for student in students:
    majors_dict[student["major"]].append(student["name"])

# Result: {"Biology": ["Alice", "Charlie"], "Chemistry": ["Bob"]}
```

### Counter for Frequency Analysis

```python
from collections import Counter

# Count grades
grades = ["A", "B", "A", "C", "B", "A", "D", "B"]
grade_counts = Counter(grades)  # Counter({'A': 3, 'B': 3, 'C': 1, 'D': 1})

# Most common grades
top_grades = grade_counts.most_common(2)  # [('A', 3), ('B', 3)]

# Count words in comments
comments = ["great course", "very difficult", "great instructor", "course material great"]
all_words = " ".join(comments).split()
word_counts = Counter(all_words)
```

## Advanced String Processing

### Regular Expressions

```python
import re

# Clean student IDs
student_ids = ["S001", "s-002", "S_003", "student004"]
clean_ids = []
for sid in student_ids:
    # Extract numbers and ensure S prefix
    match = re.search(r'\d+', sid)
    if match:
        number = match.group()
        clean_ids.append(f"S{number.zfill(3)}")

# Validate email addresses
emails = ["alice@ucsf.edu", "bob@invalid", "charlie@ucsf.edu", "diana@stanford.edu"]
ucsf_emails = [email for email in emails if re.match(r'^[\w.-]+@ucsf\.edu$', email)]

# Extract course codes from text
text = "Students in DATASCI217 and BIOSTATS200 should register for MATH101"
course_codes = re.findall(r'[A-Z]+\d+', text)  # ['DATASCI217', 'BIOSTATS200', 'MATH101']
```

### Advanced Text Processing

```python
import string

def clean_comment_text(comment):
    """Clean and normalize comment text"""
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    clean_text = comment.translate(translator).lower()
    
    # Remove extra whitespace
    clean_text = ' '.join(clean_text.split())
    
    return clean_text

def extract_sentiment_words(comment):
    """Extract positive and negative sentiment words"""
    positive_words = {"good", "great", "excellent", "amazing", "wonderful", "helpful", "clear"}
    negative_words = {"bad", "terrible", "awful", "confusing", "difficult", "boring", "unclear"}
    
    words = comment.lower().split()
    sentiment = {
        "positive": [word for word in words if word in positive_words],
        "negative": [word for word in words if word in negative_words]
    }
    
    return sentiment

# Process student comments
comments = [
    "Great course! Very clear explanations.",
    "Difficult material but excellent instructor.",
    "Boring lectures, very confusing content."
]

for comment in comments:
    cleaned = clean_comment_text(comment)
    sentiment = extract_sentiment_words(comment)
    print(f"Comment: {comment}")
    print(f"Cleaned: {cleaned}")
    print(f"Sentiment: {sentiment}")
    print("-" * 40)
```

## Advanced File Operations

### Working with Different File Formats

#### JSON Files

```python
import json

# Save complex data structures
student_data = {
    "course": "DATASCI217",
    "students": [
        {"name": "Alice", "grades": [85, 90, 87], "major": "Biology"},
        {"name": "Bob", "grades": [92, 88, 95], "major": "Chemistry"}
    ],
    "metadata": {
        "semester": "Fall 2024",
        "instructor": "Dr. Smith"
    }
}

# Write to JSON
with open("student_data.json", "w") as file:
    json.dump(student_data, file, indent=2)

# Read from JSON
with open("student_data.json", "r") as file:
    loaded_data = json.load(file)
```

#### Processing Large Files

```python
def process_large_csv(filename, chunk_size=1000):
    """Process large CSV files in chunks to manage memory"""
    import csv
    
    chunk = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader):
            chunk.append(row)
            
            # Process chunk when it reaches desired size
            if len(chunk) >= chunk_size:
                yield process_chunk(chunk)
                chunk = []
        
        # Process remaining rows
        if chunk:
            yield process_chunk(chunk)

def process_chunk(chunk):
    """Process a chunk of data"""
    # Calculate statistics for this chunk
    grades = [int(row['grade']) for row in chunk if row['grade'].isdigit()]
    return {
        "count": len(chunk),
        "avg_grade": sum(grades) / len(grades) if grades else 0,
        "students": [row['name'] for row in chunk]
    }
```

### File System Operations

```python
import os
import glob
from pathlib import Path

# Modern path handling with pathlib
data_dir = Path("data")
output_dir = Path("output")

# Create directories if they don't exist
output_dir.mkdir(exist_ok=True)

# Find all CSV files
csv_files = list(data_dir.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files")

# Process all files in directory
for csv_file in csv_files:
    print(f"Processing: {csv_file.name}")
    output_file = output_dir / f"processed_{csv_file.name}"
    
    # Process file (your processing code here)
    process_csv_file(csv_file, output_file)

# File information
for file_path in csv_files:
    stat = file_path.stat()
    print(f"{file_path.name}: {stat.st_size} bytes, modified {stat.st_mtime}")
```

## Data Analysis Patterns

### Grouping and Aggregation

```python
def group_by_field(records, field):
    """Group records by a specific field"""
    groups = {}
    for record in records:
        key = record.get(field)
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    return groups

def calculate_group_statistics(groups, numeric_field):
    """Calculate statistics for each group"""
    stats = {}
    for group_name, records in groups.items():
        values = [float(r[numeric_field]) for r in records if r[numeric_field]]
        if values:
            stats[group_name] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "median": sorted(values)[len(values)//2]
            }
    return stats

# Example usage
students = [
    {"name": "Alice", "major": "Biology", "grade": 85},
    {"name": "Bob", "major": "Chemistry", "grade": 92},
    {"name": "Charlie", "major": "Biology", "grade": 78},
    {"name": "Diana", "major": "Chemistry", "grade": 95}
]

# Group by major and calculate grade statistics
groups = group_by_field(students, "major")
grade_stats = calculate_group_statistics(groups, "grade")

for major, stats in grade_stats.items():
    print(f"{major}: avg={stats['mean']:.1f}, count={stats['count']}")
```

### Data Validation and Quality Checks

```python
def validate_student_record(record):
    """Validate a student record and return list of issues"""
    issues = []
    
    # Check required fields
    required_fields = ["name", "major", "grade"]
    for field in required_fields:
        if field not in record or not record[field]:
            issues.append(f"Missing required field: {field}")
    
    # Validate data types and ranges
    if "grade" in record:
        try:
            grade = float(record["grade"])
            if grade < 0 or grade > 100:
                issues.append(f"Grade out of range (0-100): {grade}")
        except ValueError:
            issues.append(f"Invalid grade format: {record['grade']}")
    
    # Validate email format
    if "email" in record and record["email"]:
        email = record["email"]
        if "@" not in email or "." not in email:
            issues.append(f"Invalid email format: {email}")
    
    return issues

def quality_report(records):
    """Generate data quality report"""
    total_records = len(records)
    valid_records = 0
    all_issues = []
    
    for i, record in enumerate(records):
        issues = validate_student_record(record)
        if not issues:
            valid_records += 1
        else:
            all_issues.extend([(i, issue) for issue in issues])
    
    print(f"Data Quality Report")
    print(f"==================")
    print(f"Total Records: {total_records}")
    print(f"Valid Records: {valid_records} ({valid_records/total_records:.1%})")
    print(f"Records with Issues: {total_records - valid_records}")
    print()
    
    if all_issues:
        print("Issues Found:")
        for record_idx, issue in all_issues[:10]:  # Show first 10 issues
            print(f"  Record {record_idx}: {issue}")
        if len(all_issues) > 10:
            print(f"  ... and {len(all_issues) - 10} more issues")
```

## Performance Optimization

### Efficient Data Processing

```python
# Inefficient: Creating new lists repeatedly
def slow_processing(data):
    result = []
    for item in data:
        if item["grade"] >= 80:
            processed = {
                "name": item["name"].title(),
                "grade": item["grade"],
                "status": "Pass"
            }
            result.append(processed)
    return result

# More efficient: List comprehension
def fast_processing(data):
    return [
        {
            "name": item["name"].title(),
            "grade": item["grade"],
            "status": "Pass"
        }
        for item in data if item["grade"] >= 80
    ]

# Memory efficient: Generator for large datasets
def memory_efficient_processing(data):
    """Generator that yields processed items one at a time"""
    for item in data:
        if item["grade"] >= 80:
            yield {
                "name": item["name"].title(),
                "grade": item["grade"],
                "status": "Pass"
            }
```

### Benchmarking Your Code

```python
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def process_large_dataset(data):
    # Your processing code here
    return [process_record(record) for record in data]

# Compare different approaches
def compare_methods(data):
    """Compare different processing methods"""
    
    # Method 1: Traditional loop
    start = time.time()
    result1 = slow_processing(data)
    time1 = time.time() - start
    
    # Method 2: List comprehension
    start = time.time()
    result2 = fast_processing(data)
    time2 = time.time() - start
    
    print(f"Traditional loop: {time1:.4f} seconds")
    print(f"List comprehension: {time2:.4f} seconds")
    print(f"Speedup: {time1/time2:.2f}x faster")
```

## Error Handling and Robust Code

### Comprehensive Error Handling

```python
def robust_csv_processor(filename):
    """Robust CSV processing with comprehensive error handling"""
    try:
        # Check file existence
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
        
        # Check file size
        file_size = os.path.getsize(filename)
        if file_size == 0:
            raise ValueError(f"Data file is empty: {filename}")
        
        records = []
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (after header)
                try:
                    # Validate and clean each record
                    cleaned_record = clean_and_validate_record(row)
                    records.append(cleaned_record)
                except ValueError as e:
                    print(f"Warning: Skipping row {row_num}: {e}")
                    continue
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
    except PermissionError:
        print(f"Error: Permission denied accessing {filename}")
        return []
    except UnicodeDecodeError:
        print(f"Error: Unable to decode {filename}. Check file encoding.")
        return []
    except Exception as e:
        print(f"Unexpected error processing {filename}: {e}")
        return []
    
    return records

def clean_and_validate_record(record):
    """Clean and validate a single record"""
    cleaned = {}
    
    # Clean name
    if "name" in record and record["name"]:
        cleaned["name"] = record["name"].strip().title()
    else:
        raise ValueError("Missing or empty name field")
    
    # Clean and validate grade
    if "grade" in record and record["grade"]:
        try:
            grade = float(record["grade"])
            if 0 <= grade <= 100:
                cleaned["grade"] = grade
            else:
                raise ValueError(f"Grade out of range: {grade}")
        except ValueError:
            raise ValueError(f"Invalid grade format: {record['grade']}")
    else:
        cleaned["grade"] = 0  # Default for missing grades
    
    return cleaned
```

## When to Use These Advanced Techniques

### Good Use Cases

- **List/Dict Comprehensions:** When you need to transform or filter data in a single line
- **defaultdict/Counter:** When building frequency counts or grouping data
- **Regular expressions:** When dealing with complex text patterns
- **Generators:** When processing large datasets that don't fit in memory
- **Error handling:** Always! But especially when dealing with user input or external files

### When to Keep It Simple

- **Small datasets:** Basic loops might be more readable than complex comprehensions
- **One-time scripts:** Don't over-engineer simple tasks
- **Learning phase:** Master the basics before using advanced features
- **Team projects:** Use techniques that all team members understand

Remember: "Simple is better than complex" - from the Zen of Python. Use these advanced techniques when they make your code cleaner and more efficient, not just because they exist!