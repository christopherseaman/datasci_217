Demo Guide - Lecture 3: NumPy Arrays and Data Science Tools

# Demo 1: Assignment 02 Walkthrough

## Part 1: Git Workflow Setup

2. Set up the following branch structure:
   - `main` branch (initial commit)
   - `feature/project-scaffold` branch
   - `feature/data-processing` branch
3. Develop features on separate branches
4. Merge branches back to main
5. Create comprehensive documentation

**Deliverables**:

- Repository with proper branch structure
- At least 3 commits per branch with meaningful messages
- Successful merge of both feature branches to main
- Professional README.md with project overview

### 1a. Repository Setup

```bash
# Create new repository (separate from assignment folder)
mkdir datasci-week02-integration
cd datasci-week02-integration
git init

# Create initial README
# (Students can follow README template from assignment)
git add README.md
git commit -m "Initial commit: Add project README"

# Verify first commit
git log --oneline
```

### 1b. Feature Branch for Project Scaffold

```bash
# Create branch for Part 2 - CLI automation script
git checkout -b feature/project-scaffold
```

## Part 2: CLI Project Scaffold Script

**Objective**: Create a shell script that automates project setup.

**Tasks**:

1. Create `setup_project.sh` from scratch with the following functionality:
   - Create directory structure (src, data, output)
   - Generate initial files (.gitignore, requirements.txt)
   - Create sample data files (students.csv with at least 8 records)
   - Set up Python template files with TODO placeholders
2. Make the script executable using chmod +x
3. Test the script and verify all files are created
4. Commit the script to the `feature/project-scaffold` branch

**Script Requirements**:

- Must start with `#!/bin/bash`
- Use `echo` to provide user feedback
- Use `mkdir -p` to create directories
- Use here-documents (`cat > filename << 'EOF'`) to create files
- Create a CSV file with student data (name,age,grade,subject)
- Create Python templates with function stubs and TODO comments
- Make the script executable: `chmod +x setup_project.sh`

**Expected Output Structure**:
After running your script, the following files and directories should exist:

```
├── src/
│   ├── data_analysis.py
│   └── data_analysis_functions.py
├── data/
│   └── students.csv
├── output/
├── .gitignore
└── requirements.txt
```

### _Create directory structure (src, data, output)_

Start with shebang and user feedback:

```bash
#!/bin/bash
echo "Creating project structure..."
```

Add directory creation (`mkdir -p` creates parent directories if needed):

```bash
mkdir -p src data output
echo "✓ Created directories: src, data, output"
```

### _Generate initial files (.gitignore, requirements.txt)_

Add .gitignore (heredoc with `'EOF'` prevents variable expansion):

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
.venv/

# OS files
.DS_Store
EOF
echo "✓ Created .gitignore"
```

### _Create sample data files (students.csv with at least 8 records)_

Add sample CSV (must have header row and at least 8 data rows):

```bash
cat > data/students.csv << 'EOF'
name,age,grade,subject
Alice,20,92,Math
Bob,21,85,Science
Charlie,19,78,Math
Diana,22,95,Science
Eve,20,88,Math
Frank,21,76,Science
Grace,19,91,Math
Henry,22,82,Science
EOF
echo "✓ Created data/students.csv with 8 records"
```

### _Set up Python template files with TODO placeholders_

Add Python templates with function stubs (students fill in TODOs):

```bash
cat > src/data_analysis.py << 'EOF'
"""Basic student data analysis script."""

def load_students(filename):
    """Load student data from CSV file."""
    # TODO: Implement CSV loading
    pass

def calculate_average_grade(students):
    """Calculate average grade from student data."""
    # TODO: Implement average calculation
    pass

def count_math_students(students):
    """Count students in Math."""
    # TODO: Implement counting
    pass

def generate_report(students):
    """Generate formatted report."""
    # TODO: Implement report generation
    pass

def save_report(report, filename):
    """Save report to file."""
    # TODO: Implement file saving
    pass

def main():
    """Main execution function."""
    # TODO: Orchestrate the analysis
    pass

if __name__ == "__main__":
    main()
EOF
echo "✓ Created src/data_analysis.py with function stubs"
```

Add requirements.txt:

```bash
cat > requirements.txt << 'EOF'
# No external packages required for basic functionality
EOF
echo "✓ Created requirements.txt"
```

### Test and Commit

```bash
# Make executable
chmod +x setup_project.sh

# Test it
./setup_project.sh

# Verify structure
ls -R

# Commit to feature branch
git add setup_project.sh
git commit -m "Add project setup automation script"
```

### Merge to Main

```bash
git checkout main
git merge feature/project-scaffold
git log --oneline --graph --all
```

## Part 3: Python Data Processing

**Objective**: Implement Python scripts that process data and output results to files.

**Tasks**:

1. Create and complete `src/data_analysis.py` with basic functionality
2. Create and complete `src/data_analysis_functions.py` with modular design
3. Ensure both scripts output results to `output/analysis_report.txt`
4. Test your implementation thoroughly

**Python Requirements**:

**Basic Analysis Script** (`src/data_analysis.py`):

Your script should:

- Read the CSV file line by line using `open()` and `readlines()`
- Split each line by commas to extract fields
- Calculate basic statistics (total students, average grade)
- Count students by subject
- Write results to `output/analysis_report.txt`
- Use f-strings with `.1f` formatting for decimal numbers

**Required Functions**:

- `load_students(filename)`: Read CSV and return list of student data
- `calculate_average_grade(students)`: Calculate and return average
- `count_math_students(students)`: Count students in Math
- `generate_report()`: Create formatted report string
- `save_report(report, filename)`: Write report to file
- `main()`: Orchestrate the analysis

### Create Feature Branch

```bash
# New feature = new branch
git checkout -b feature/data-processing
```

**Note**: Part 3 focuses on the required `src/data_analysis.py` script. Part 3b (advanced analysis) is OPTIONAL.

Edit `src/data_analysis.py` to add these implementations:

### _Read CSV file line by line using `open()` and `readlines()`_

Implement `load_students()` (`lines[1:]` skips header, `strip()` removes newlines, `split(',')` parses CSV):

```python
def load_students(filename):
    """Load student data from CSV file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    students = []
    for line in lines[1:]:  # Skip header
        line = line.strip()
        if line:
            name, age, grade, subject = line.split(',')
            students.append({
                'name': name,
                'age': int(age),
                'grade': int(grade),
                'subject': subject
            })

    return students
```

### _Calculate basic statistics (total students, average grade)_

Implement `calculate_average_grade()` (list comprehension with `sum()` is concise):

```python
def calculate_average_grade(students):
    """Calculate average grade from student data."""
    if not students:
        return 0.0

    total = sum(student['grade'] for student in students)
    return total / len(students)
```

### _Count students by subject_

Implement `count_math_students()` (generator expression for counting):

```python
def count_math_students(students):
    """Count students enrolled in Math."""
    return sum(1 for student in students if student['subject'] == 'Math')
```

### _Write results to `output/analysis_report.txt`_

Implement `generate_report()` (f-strings with `.1f` for one decimal, triple-quotes for multi-line):

```python
def generate_report(students):
    """Generate formatted analysis report."""
    total = len(students)
    avg = calculate_average_grade(students)
    math_count = count_math_students(students)

    report = f"""Student Analysis Report
{'=' * 40}

Total Students: {total}
Average Grade: {avg:.1f}

Subject Distribution:
  Math: {math_count}
  Science: {total - math_count}
"""
    return report
```

Implement `save_report()` and `main()` (`os.makedirs(..., exist_ok=True)` creates output directory if needed):

```python
import os

def save_report(report, filename):
    """Save report to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(report)

def main():
    """Main execution function."""
    students = load_students('data/students.csv')
    report = generate_report(students)
    save_report(report, 'output/analysis_report.txt')
    print(report)
```

### Complete Basic Script

For the complete implementation, see [demo1a_data_analysis.py](https://github.com/christopherseaman/datasci_217/blob/main/03/demo/demo1a_data_analysis.py).

### Test and Commit

```bash
# Test and verify output
python src/data_analysis.py
cat output/analysis_report.txt

# Commit
git add src/data_analysis.py
git commit -m "Implement basic student data analysis"

# Additional commits to meet "3 commits per branch" requirement
# Make small improvements or add documentation
git add README.md
git commit -m "Update README with data analysis documentation"

git add .gitignore
git commit -m "Ensure output files are tracked appropriately"
```

## Part 3b: Advanced Analysis (OPTIONAL)

**Note**: This section is OPTIONAL for students. It demonstrates more advanced modular design patterns.

**Advanced Analysis Script** (`src/data_analysis_functions.py`):

Your modular script should:

- Separate data loading, processing, and saving into different functions
- Load CSV data using the same technique as the basic script
- Provide more detailed analysis (highest/lowest grades, grade distribution)
- Generate a more comprehensive report
- Demonstrate function reusability and modular design

**Required Functions**:

- `load_data(filename)`: Generic loader that checks file extension
- `load_csv(filename)`: Load CSV data (same technique as basic script)
- `analyze_data(students)`: Return dictionary with multiple statistics
- `analyze_grade_distribution(grades)`: Count grades by letter grade ranges
- `save_results(results, filename)`: Save detailed report
- `main()`: Orchestrate the analysis using all functions

**Additional Requirements**:

- Calculate highest and lowest grades using `max()` and `min()`
- Count students by multiple subjects (Math, Science, etc.)
- Create grade distribution (A: 90-100, B: 80-89, C: 70-79, D: 60-69, F: 0-59)
- Calculate and display percentages with `.1f` formatting
- Use dictionaries to store analysis results

**Expected Output Format**:
Both scripts should create `output/analysis_report.txt` with:

- Total number of students
- Average grade (formatted to 1 decimal place)
- Subject counts
- For advanced script: grade distribution with percentages

Edit `src/data_analysis_functions.py` to add these implementations:

### _Modular design with separate functions_

First, implement `load_data()` and `load_csv()` for flexible data loading:

```python
import os

def load_data(filename):
    """Generic loader that checks file extension."""
    if filename.endswith('.csv'):
        return load_csv(filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def load_csv(filename):
    """Load CSV data using manual parsing."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    students = []
    for line in lines[1:]:  # Skip header
        line = line.strip()
        if line:
            name, age, grade, subject = line.split(',')
            students.append({
                'name': name,
                'age': int(age),
                'grade': int(grade),
                'subject': subject
            })

    return students
```

Implement `count_by_subject()` for multi-subject counting:

```python
def count_by_subject(students):
    """Count students by all subjects."""
    subjects = {}
    for student in students:
        subject = student['subject']
        subjects[subject] = subjects.get(subject, 0) + 1
    return subjects
```

Implement `analyze_data()` (return dictionary with all results for easy access):

```python
def analyze_data(students):
    """Perform comprehensive analysis on student data."""
    grades = [s['grade'] for s in students]

    return {
        'total_students': len(students),
        'average_grade': sum(grades) / len(grades),
        'highest_grade': max(grades),
        'lowest_grade': min(grades),
        'subjects': count_by_subject(students),
        'distribution': analyze_grade_distribution(grades)
    }
```

### _More detailed analysis (highest/lowest grades, grade distribution)_

Implement `analyze_grade_distribution()` (dictionary comprehension for percentages):

```python
def analyze_grade_distribution(grades):
    """Analyze grade distribution by letter grade."""
    total = len(grades)
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}

    for grade in grades:
        if grade >= 90:
            counts['A'] += 1
        elif grade >= 80:
            counts['B'] += 1
        elif grade >= 70:
            counts['C'] += 1
        elif grade >= 60:
            counts['D'] += 1
        else:
            counts['F'] += 1

    percentages = {
        letter: (count / total * 100)
        for letter, count in counts.items()
    }

    return {'counts': counts, 'percentages': percentages}
```

### _Grade distribution by letter grade with percentages_

Implement `save_results()` (accessing nested dictionaries, building string with `+=`):

```python
def save_results(results, filename):
    """Save analysis results to file."""
    report = f"""Advanced Student Analysis Report
{'=' * 50}

Total Students: {results['total_students']}
Average Grade: {results['average_grade']:.1f}
Highest: {results['highest_grade']} | Lowest: {results['lowest_grade']}

Subject Distribution:
"""

    for subject, count in results['subjects'].items():
        report += f"  {subject}: {count}\n"

    report += "\nGrade Distribution:\n"

    for letter in ['A', 'B', 'C', 'D', 'F']:
        count = results['distribution']['counts'][letter]
        pct = results['distribution']['percentages'][letter]
        report += f"  {letter}: {count} students ({pct:.1f}%)\n"

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(report)

def main():
    """Main execution function."""
    # Load data using modular functions
    students = load_data('data/students.csv')

    # Perform analysis
    results = analyze_data(students)

    # Save results
    save_results(results, 'output/analysis_report.txt')

    # Also print to console
    with open('output/analysis_report.txt', 'r') as f:
        print(f.read())

if __name__ == "__main__":
    main()
```

### Complete Advanced Script

For the complete implementation, see [demo1b_data_analysis_functions.py](https://github.com/christopherseaman/datasci_217/blob/main/03/demo/demo1b_data_analysis_functions.py).

### Test and Commit

```bash
# Test and verify enhanced output
python src/data_analysis_functions.py
cat output/analysis_report.txt

# Commit
git add src/data_analysis_functions.py
git commit -m "Add advanced student data analysis with modular design (optional)"
```

### Merge and Tag

```bash
# Merge to main
git checkout main
git merge feature/data-processing

# Verify merge was successful
git log --oneline --graph --all

# Create release tag
git tag -a v1.0 -m "Release 1.0: Complete integration project"

# View all tags
git tag -l
```

**Note**: If you completed the optional Part 3b (advanced analysis), you should have additional commits on the `feature/data-processing` branch before merging.

## Common Questions

**Q: "What if I mess up a commit?"**
A: `git commit --amend` for last commit, or `git revert` for older commits.

**Q: "Do I need to use `csv` module?"**
A: No - assignment requires manual parsing with `split()` to understand fundamentals.

**Q: "What if the output directory doesn't exist?"**
A: That's why we use `os.makedirs(..., exist_ok=True)` before writing files.

**Q: "Is Part 3b (advanced analysis) required?"**
A: No - Part 3b is OPTIONAL. The required script is `data_analysis.py` in Part 3. Part 3b demonstrates advanced modular design patterns for students who want additional practice.

**Q: "How do I get 3 commits per branch?"**
A: Break your work into logical steps. For example: (1) implement basic functions, (2) add report generation, (3) add documentation or error handling. Each meaningful change deserves its own commit.

**Q: "Should both scripts write to the same file?"**
A: Yes - both scripts write to `output/analysis_report.txt`. The basic script creates a simple report, while the advanced script (if completed) creates a more detailed report. They overwrite each other, which is expected.

# Demo 2: Virtual Environments and Python Potpourri

**Script** - [`demo2_python_potpourri.py`](https://github.com/christopherseaman/datasci_217/blob/main/03/demo/demo2_python_potpourri.py)

## Demo Flow

### 2a. Virtual Environment Setup

Show creating and activating a virtual environment:

**Method 1: Using venv (built-in)**
```bash
# Create virtual environment in .venv (standard location)
python -m venv .venv

# Activate it
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install required packages (from requirements.txt)
pip install -r requirements.txt

# Or install manually
pip install numpy
```

**Method 2: Using uv (faster, optional)**
```bash
# Install uv first (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate environment
uv venv .venv
source .venv/bin/activate  # Mac/Linux

# Install packages (much faster than pip!)
uv pip install -r requirements.txt

# Or install manually
uv pip install numpy
```

**Key Points:**
- Both methods create the same `.venv` directory
- `uv` is 10-100x faster for package installation
- Use whichever you prefer - results are identical

### 2b. Run Python Potpourri Demo

Execute the demo script:

```bash
source .venv/bin/activate  # Activate the virtual environment first
python demo2_python_potpourri.py
```

**Key Demonstrations**

1. **Type Checking** - Show how to check types with `type()`
2. **Type Conversions** - String → int, int → float, etc.
3. **F-String Formatting** - Modern Python string formatting
4. **Formatting Options** - Decimal places, alignment, padding

## Common Questions

**Q: "Can I have multiple virtual environments?"**
A: Yes - one per project is standard practice. Each environment is isolated.

**Q: "What's the difference between f-strings and .format()?"**
A: F-strings (Python 3.6+) are faster and more concise than `.format()`.

# Demo 3: NumPy Arrays and Operations

**Scripts**

- [`demo3_numpy_performance.py`](https://github.com/christopherseaman/datasci_217/blob/main/03/demo/demo3_numpy_performance.py)
- [`demo3_student_analysis.py`](https://github.com/christopherseaman/datasci_217/blob/main/03/demo/demo3_student_analysis.py)

## 3a. Performance Comparison

Run the performance demo:

```bash
source .venv/bin/activate  # Activate the virtual environment first
python demo3_numpy_performance.py
```

**Expected Output**

- Python list operations: tens of milliseconds
- NumPy array operations: single-digit milliseconds
- **10-100x speedup** (exact speedup varies by system)

## 3b. Student Grade Analysis

Run the practical demo:

```bash
source .venv/bin/activate  # Activate the virtual environment first
python demo3_student_analysis.py
```

**Key Demonstrations**

1. **Array Creation**
   - From lists - `np.array([[...]])`
   - Properties - `shape`, `dtype`, `size`

2. **Indexing and Slicing**
   - Single elements - `grades[0, 1]`
   - Rows/columns - `grades[0, :]`, `grades[:, 1]`
   - Slices - `grades[1:3, :]`

3. **Boolean Indexing** (most important!)
   - Create mask - `grades > 85`
   - Filter data - `grades[grades > 85]`
   - Multiple conditions - `(grades > 80) & (grades < 90)`

4. **Statistical Operations**
   - Basic - `.mean()`, `.std()`, `.max()`, `.min()`
   - Axis operations - `grades.mean(axis=0)` vs `grades.mean(axis=1)`
   - "Axis 0 = down the rows (column stats), Axis 1 = across columns (row stats)"

5. **Array Reshaping**
   - `.reshape()` - Change dimensions
   - `.flatten()` - 2D → 1D
   - `.T` - Transpose

**Points**

- After boolean indexing - "How would you find students with grades between 80-90?"
- After axis operations - "What's the difference between axis=0 and axis=1?"
- After reshaping - "Why would you need to reshape an array?"

## Common Questions

**Q: "Why does slicing give a view, not a copy?"**
A: NumPy uses views for memory efficiency. Use `.copy()` when you need independence.

**Q: "What's the difference between `grades[grades > 85]` and `np.where(grades > 85)`?"**
A: Boolean indexing returns values, `np.where()` returns indices.

**Q: "Why do I get a 1D array when I slice a single row?"**
A: NumPy reduces dimensions when possible. Use `grades[0:1, :]` to keep 2D.

# Demo 4: Command Line Data Processing

**Data file** - `students.csv` (1,500 students with name, age, grade, subject across 8 subjects)

## Demo Setup

First, navigate to the demo directory:
```bash
cd /path/to/datasci_217/03/demo
```

Verify the data file exists and preview it:
```bash
wc -l students.csv        # Should show 1501 lines (header + 1500 students)
head -10 students.csv     # Preview first 10 lines
tail -5 students.csv      # Preview last 5 lines
```

## Demo Flow

Run these commands manually to demonstrate CLI data processing:

**4a. `cut`** - Extract columns

```bash
cut -d',' -f1,3 students.csv | head -5
```

- `-d','` - Set delimiter to comma
- `-f1,3` - Select fields 1 and 3
- Use case - Quick column extraction from CSV

**4b. `sort`** - Sort data

```bash
sort -t',' -k3 -n students.csv | head -5
```

- `-n` - Numerical sort (not alphabetical)
- `-t','` - Set delimiter
- `-k3` - Sort by field 3
- Common mistake - Forgetting `-n` for numbers!

**4c. `uniq`** - Count occurrences

```bash
cut -d',' -f4 students.csv | sort | uniq -c
```

- **Must sort first!** `uniq` only removes adjacent duplicates
- `-c` - Count occurrences
- Pattern - `sort | uniq -c` is very common

**4d. `grep`** - Search and filter

```bash
grep "Math" students.csv
```

- Basic search - `grep "Math"`
- `-v` - Inverse match (NOT)
- `-i` - Case-insensitive
- Use case - Quick filtering

**4e. `tr`** - Transform characters
```bash
# Convert subject names to uppercase
cut -d',' -f4 students.csv | head -5 | tr 'a-z' 'A-Z'
```
- Use case - Case conversion, character cleanup
- Simple transformations only (use `sed` for complex patterns)

**4f. `sed`** - Stream editor

- Replace - `sed 's/old/new/g'`
- Delete lines - `sed '1d'` (removes header!)
- Use case - More powerful than `tr`

**4g. `awk`** - Pattern processing

```bash
awk -F',' 'NR>1 {sum+=$3; count++} END {print "Average grade:", sum/count}' students.csv
```

- Print columns - `awk '{print $1, $3}'`
- Filter rows - `awk '$3 > 85'`
- Calculate - `awk '{sum+=$3} END {print sum/NR}'`
- **Most powerful tool for structured data**

**4h. Complex pipelines**
How to chain commands:

```bash
grep "Math" students.csv | \
  cut -d',' -f1,3 | \
  sort -t',' -k2 -nr | \
  head -n 3
```

Break down the pipeline step by step:

1. Filter - What rows do we keep?
2. Extract - What columns do we need?
3. Sort - What order?
4. Limit - How many results?

**4i. Sparklines** - Inline graphs

Install (in your venv):

```bash
source .venv/bin/activate
pip install sparklines
```

Basic usage:

```bash
cut -d',' -f3 students.csv | tail -n +2 | sparklines
```

With statistics:

```bash
cut -d',' -f3 students.csv | tail -n +2 | sparklines --stat-min --stat-max --stat-mean
```

Note:

- `tail -n +2` means "start at line 2" (skip header)
- Shows inline graphs perfect for SSH sessions and remote work
- Lightweight visualization without leaving the terminal
- Great for quick data exploration

# Common Mistakes and How to Address Them

## Git Issues

**Mistake** - Committing too much at once

- **Fix** - Break into logical chunks
- **Thought** - "If you can't write a concise commit message, it's too much"

**Mistake** - Forgetting to add files

- **Fix** - Use `git status` before committing
- **Thought** - Show `git status` output interpretation

## Python Issues

**Mistake** - Mixing string and numeric operations

- **Fix** - Use `type()` to check, convert with `int()`, `float()`, `str()`
- **Thought** - "Python won't guess - you need to convert explicitly"

**Mistake** - F-string syntax errors

- **Fix** - Remember the `f` prefix - `f"..."` not `"..."`
- **Thought** - Show the error message, explain how to read it

## NumPy Issues

**Mistake** - Modifying a view thinking it's a copy

- **Fix** - Use `.copy()` when you need independence
- **Thought** - Show how changes propagate through views

**Mistake** - Wrong axis specification

- **Fix** - Remember "axis is what you're collapsing"
- **Thought** - Show with small 2D array, visualize what each axis does

**Mistake** - Boolean indexing without parentheses

- **Fix** - `(grades > 80) & (grades < 90)` not `grades > 80 & grades < 90`
- **Thought** - Show the error, explain operator precedence

## CLI Issues

**Mistake** - Forgetting to sort before `uniq`

- **Fix** - Always `sort | uniq -c`
- **Thought** - Show what happens without sort

**Mistake** - Using alphabetical sort on numbers

- **Fix** - Add `-n` flag for numerical sort
- **Thought** - Show "10" sorting before "2" without `-n`

**Mistake** - Not skipping CSV headers

- **Fix** - Use `tail -n +2`, `sed '1d'`, or `awk 'NR>1'`
- **Thought** - Show error when trying to do math on header text

**Mistake** - Wrong field delimiter

- **Fix** - Remember `-d','` for CSVs, `-t','` for sort
- **Thought** - Show what happens with wrong delimiter (gets wrong columns)
