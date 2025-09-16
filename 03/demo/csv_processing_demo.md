# LIVE DEMO: CSV Processing with Python Data Structures

*This guide is for the instructor to follow during the live demonstration. Students can refer to this afterward.*

## Demo Setup (Pre-class)

1. **Prepare demo data** - simple CSV file with realistic student data
2. **VS Code** ready with Python file open
3. **Sample CSV file** visible in file explorer
4. **Terminal/output panel** ready for showing results

## Part 1: Understanding the Data (5 minutes)

### Show the CSV File

**Narration:** "Let's start by looking at some real data - this is what you'll often encounter in data science..."

**Create demo file: `student_data.csv`**
```csv
name,major,assignment1,assignment2,final_exam
Alice Smith,Biology,85,90,87
Bob Johnson,Chemistry,92,88,95
Charlie Brown,Biology,78,82,80
Diana Wilson,Chemistry,95,93,91
Eve Chen,Physics,88,85,92
```

**Key Points to Emphasize:**
- "This is typical CSV format - comma-separated values"
- "First row contains column headers"  
- "Each row represents one student record"
- "Notice the mixed data types - strings and numbers"

## Part 2: Reading CSV Files (8 minutes)

### Basic File Reading Approach

**Narration:** "Let's start with basic file reading to see what we get..."

```python
# Demo 1: Basic file reading (show the problem)
with open('student_data.csv', 'r') as file:
    content = file.read()
    print(content)
```

**Point out:** "This gives us the raw text, but it's not very useful for analysis."

### CSV Module Approach

**Narration:** "Now let's use Python's CSV module to parse this properly..."

```python
import csv

# Demo 2: Using csv.reader
with open('student_data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        print(row)
```

**Show the output:** Each row becomes a list of strings.

**Key Points:**
- "Now we have lists, which is much better!"
- "But we still need to remember which index is which column"
- "There's a better way..."

### DictReader Approach (The Best Way)

**Narration:** "DictReader gives us dictionaries with meaningful column names..."

```python
import csv

# Demo 3: Using csv.DictReader (the preferred method)
with open('student_data.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        print(row)
        print(f"Student: {row['name']}, Major: {row['major']}")
        print("-" * 30)
```

**Wow Moment:** "Look! Now we can access data by column name, not just position!"

## Part 3: Building Data Structures (10 minutes)

### Loading into a List of Dictionaries

**Narration:** "Let's load all the data into memory so we can work with it..."

```python
import csv

def load_student_data(filename):
    """Load student data from CSV into list of dictionaries"""
    students = []
    
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            # Convert grade strings to integers
            student = {
                'name': row['name'],
                'major': row['major'],
                'assignment1': int(row['assignment1']),
                'assignment2': int(row['assignment2']),
                'final_exam': int(row['final_exam'])
            }
            students.append(student)
    
    return students

# Load the data
students = load_student_data('student_data.csv')
print(f"Loaded {len(students)} students")

# Show the first student
print("First student:", students[0])
```

**Key Points:**
- "We're converting grade strings to integers for calculations"
- "Each student is now a dictionary with meaningful keys"
- "All students are stored in a list for easy processing"

### Data Cleaning Example

**Narration:** "Real data is often messy. Let's see how to handle that..."

**Create messier CSV file:**
```csv
name,major,assignment1,assignment2,final_exam
  Alice Smith  ,biology,85,90,87
BOB JOHNSON,Chemistry,92,,95
charlie brown,BIOLOGY,78,82,80
Diana Wilson,chemistry,95,93,
Eve Chen,Physics,88,85,92
```

```python
def clean_student_data(filename):
    """Load and clean messy student data"""
    students = []
    
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            # Clean the data
            student = {
                'name': row['name'].strip().title(),  # Remove whitespace, proper case
                'major': row['major'].strip().title(),
                'assignment1': int(row['assignment1']) if row['assignment1'] else 0,
                'assignment2': int(row['assignment2']) if row['assignment2'] else 0,
                'final_exam': int(row['final_exam']) if row['final_exam'] else 0
            }
            students.append(student)
    
    return students

# Clean and load the messy data
students = clean_student_data('messy_student_data.csv')
for student in students:
    print(student)
```

**Point out the cleaning steps:**
- "`.strip()` removes extra whitespace"
- "`.title()` fixes capitalization"
- "We handle missing values by defaulting to 0"

## Part 4: Data Analysis with Dictionaries and Lists (12 minutes)

### Calculate Individual Student Statistics

**Narration:** "Now let's analyze the data - first, individual student performance..."

```python
def calculate_student_average(student):
    """Calculate a student's average grade"""
    grades = [student['assignment1'], student['assignment2'], student['final_exam']]
    return sum(grades) / len(grades)

# Add average to each student record
for student in students:
    student['average'] = calculate_student_average(student)
    print(f"{student['name']}: {student['average']:.1f}")
```

### Group Analysis by Major

**Narration:** "Data science often involves grouping data - let's analyze by major..."

```python
def analyze_by_major(students):
    """Group students by major and calculate statistics"""
    majors = {}
    
    # Group students by major
    for student in students:
        major = student['major']
        if major not in majors:
            majors[major] = []
        majors[major].append(student)
    
    # Calculate statistics for each major
    major_stats = {}
    for major, major_students in majors.items():
        averages = [student['average'] for student in major_students]
        major_stats[major] = {
            'count': len(major_students),
            'avg_grade': sum(averages) / len(averages),
            'min_grade': min(averages),
            'max_grade': max(averages)
        }
    
    return major_stats

# Analyze by major
major_analysis = analyze_by_major(students)
for major, stats in major_analysis.items():
    print(f"{major}:")
    print(f"  Students: {stats['count']}")
    print(f"  Average: {stats['avg_grade']:.1f}")
    print(f"  Range: {stats['min_grade']:.1f} - {stats['max_grade']:.1f}")
    print()
```

### Find Top Performers

**Narration:** "Let's find our top-performing students..."

```python
def find_top_students(students, n=3):
    """Find top N students by average grade"""
    # Sort students by average grade (highest first)
    sorted_students = sorted(students, key=lambda x: x['average'], reverse=True)
    return sorted_students[:n]

# Find top 3 students
top_students = find_top_students(students, 3)
print("Top 3 Students:")
for i, student in enumerate(top_students, 1):
    print(f"{i}. {student['name']} ({student['major']}): {student['average']:.1f}")
```

## Part 5: Writing Results to Files (8 minutes)

### Save Processed Data to CSV

**Narration:** "After analysis, we usually want to save our results..."

```python
def save_student_results(students, filename):
    """Save processed student data to CSV"""
    with open(filename, 'w', newline='') as file:
        # Define the columns we want to save
        fieldnames = ['name', 'major', 'assignment1', 'assignment2', 'final_exam', 'average']
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header row
        csv_writer.writeheader()
        
        # Write all student records
        csv_writer.writerows(students)

# Save the results
save_student_results(students, 'student_results.csv')
print("Results saved to student_results.csv")
```

### Generate a Summary Report

**Narration:** "Let's also create a human-readable summary report..."

```python
def generate_summary_report(students, major_stats, filename):
    """Generate a text summary report"""
    with open(filename, 'w') as file:
        file.write("Student Performance Analysis\n")
        file.write("=" * 30 + "\n\n")
        
        # Overall statistics
        all_averages = [student['average'] for student in students]
        file.write(f"Total Students: {len(students)}\n")
        file.write(f"Overall Average: {sum(all_averages) / len(all_averages):.1f}\n")
        file.write(f"Highest Grade: {max(all_averages):.1f}\n")
        file.write(f"Lowest Grade: {min(all_averages):.1f}\n\n")
        
        # Major breakdown
        file.write("Performance by Major:\n")
        file.write("-" * 20 + "\n")
        for major, stats in major_stats.items():
            file.write(f"{major}:\n")
            file.write(f"  Students: {stats['count']}\n")
            file.write(f"  Average: {stats['avg_grade']:.1f}\n")
            file.write(f"  Range: {stats['min_grade']:.1f} - {stats['max_grade']:.1f}\n\n")
        
        # Top performers
        top_students = find_top_students(students, 3)
        file.write("Top Performers:\n")
        file.write("-" * 15 + "\n")
        for i, student in enumerate(top_students, 1):
            file.write(f"{i}. {student['name']} ({student['major']}): {student['average']:.1f}\n")

# Generate the report
generate_summary_report(students, major_analysis, 'analysis_report.txt')
print("Summary report saved to analysis_report.txt")
```

## Part 6: Putting It All Together (7 minutes)

### Complete Analysis Script

**Narration:** "Let's put everything together into a complete analysis pipeline..."

```python
import csv

def complete_student_analysis():
    """Complete student data analysis pipeline"""
    print("Student Grade Analysis Pipeline")
    print("=" * 35)
    
    # Step 1: Load data
    print("1. Loading student data...")
    students = clean_student_data('student_data.csv')
    print(f"   Loaded {len(students)} students")
    
    # Step 2: Calculate averages
    print("2. Calculating student averages...")
    for student in students:
        student['average'] = calculate_student_average(student)
    
    # Step 3: Analyze by major
    print("3. Analyzing performance by major...")
    major_stats = analyze_by_major(students)
    
    # Step 4: Find top performers
    print("4. Identifying top performers...")
    top_students = find_top_students(students, 3)
    
    # Step 5: Save results
    print("5. Saving results...")
    save_student_results(students, 'student_results.csv')
    generate_summary_report(students, major_stats, 'analysis_report.txt')
    
    # Step 6: Display summary
    print("\nAnalysis Summary:")
    print("-" * 16)
    for major, stats in major_stats.items():
        print(f"{major}: {stats['count']} students, avg {stats['avg_grade']:.1f}")
    
    print(f"\nTop performer: {top_students[0]['name']} ({top_students[0]['average']:.1f})")
    print("\n✅ Analysis complete! Check output files for detailed results.")

# Run the complete analysis
if __name__ == "__main__":
    complete_student_analysis()
```

### Show the Generated Files

**Narration:** "Let's see what our analysis produced..."

1. **Open student_results.csv** in VS Code - show the clean, processed data
2. **Open analysis_report.txt** - show the human-readable summary
3. **Highlight the workflow** - raw data → processing → analysis → output

## Part 7: Q&A and Common Issues (5 minutes)

### Address Common Questions

**"What if the CSV has different delimiters?"**
```python
# For semicolon-separated files
csv_reader = csv.DictReader(file, delimiter=';')

# For tab-separated files  
csv_reader = csv.DictReader(file, delimiter='\t')
```

**"What if there are missing values?"**
```python
# Handle missing numeric values
grade = int(row['grade']) if row['grade'] else 0

# Handle missing text values
name = row['name'].strip() if row['name'] else "Unknown"
```

**"How do I handle different data types?"**
```python
# Convert strings to appropriate types
student = {
    'name': str(row['name']).strip().title(),
    'age': int(row['age']) if row['age'] else 0,
    'gpa': float(row['gpa']) if row['gpa'] else 0.0,
    'active': row['active'].lower() == 'true'
}
```

### Error Handling Preview

**Narration:** "Always handle errors when working with files..."

```python
def safe_load_data(filename):
    """Safely load data with error handling"""
    try:
        with open(filename, 'r') as file:
            # ... processing code ...
            pass
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        return []
    except PermissionError:
        print(f"Error: Permission denied accessing {filename}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
```

## Wrap-up and Key Takeaways (5 minutes)

### Emphasize the Power of This Approach

**"What we just built is a complete data processing pipeline:"**
1. **Read** structured data from CSV files
2. **Clean** and standardize the data
3. **Analyze** using Python data structures  
4. **Generate** insights and summaries
5. **Save** results in multiple formats

### Real-World Applications

**"This same pattern applies to:"**
- Customer survey analysis
- Scientific experiment data
- Financial transaction processing
- Social media data analysis
- Healthcare outcome studies

### Next Steps

**"Building on this foundation:"**
- We'll learn command-line tools for data exploration
- Python functions will make our code more organized
- Later: pandas will automate many of these operations
- But understanding these fundamentals is crucial!

## Demo Repository Cleanup (After Class)

1. **Save demo files** for student reference
2. **Clean up temporary files**
3. **Note any questions** for next year's improvements
4. **Update assignment** if any issues were discovered

## Technical Notes for Instructor

### Backup Plans

- **File issues:** Have data hardcoded as backup
- **Import issues:** Show manual dictionary creation as fallback
- **Time constraints:** Skip the complete analysis section, focus on core concepts

### Visual Aids

- **Use print statements liberally** to show intermediate results
- **Display data structures** before and after processing
- **Show file contents** in VS Code to connect concepts to output

### Common Demo Problems

- **File paths:** Use simple filenames in same directory
- **Data types:** Emphasize string → number conversion
- **CSV formatting:** Show common issues (extra commas, quotes, etc.)

This demo connects abstract concepts (lists, dictionaries) to practical data science workflows!