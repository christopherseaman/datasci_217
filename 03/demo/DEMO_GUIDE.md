# Demo Guide - Lecture 3: NumPy Arrays and Data Science Tools

This guide provides hands-on demonstrations for Lecture 3, organized to match the lecture flow.

## Demo Structure

Each demo corresponds to a **LIVE DEMO!** callout in the lecture:

1. **Demo 1**: Assignment 02 Walkthrough - Git, CLI, and Python integration
2. **Demo 2**: Virtual Environments and Python Potpourri - Project setup and essential Python skills
3. **Demo 3**: NumPy Arrays and Operations - Array fundamentals and vectorized computing
4. **Demo 4**: Command Line Data Processing - Shell tools for data science

---

# Demo 1: Assignment 02 Walkthrough

**Location in Lecture**: First LIVE DEMO callout (after title)

**Purpose**: Walk through the previous week's assignment, demonstrating the integration of Git workflows, CLI automation, and Python data processing.

**Time**: 20-25 minutes

## Overview

This demo shows a complete solution to Assignment 02, illustrating how Git, shell scripting, and Python work together in a real data science project.

## Demo Script

### Part 1: Repository Setup (5 min)

```bash
# Create new repository for the integration project
mkdir datasci-week02-integration
cd datasci-week02-integration

# Initialize Git
git init
git branch -M main

# Create initial README
cat > README.md << 'EOF'
# DataSci Week 02 Integration Project

Integration of Git workflows, CLI automation, and Python data processing.
EOF

# Initial commit
git add README.md
git commit -m "Initial commit: Add project README"

# Show status
git log --oneline
```

**Key Points**:
- Git repository initialization
- Meaningful commit messages
- Clean starting point

### Part 2: Feature Branch - Project Scaffold (7 min)

```bash
# Create feature branch for project scaffold
git checkout -b feature/project-scaffold

# Create the automated setup script
cat > setup_project.sh << 'EOF'
#!/bin/bash
# Automated project setup script

echo "🚀 Setting up DataSci Week 02 Integration Project..."

# Create directory structure
mkdir -p src data output

# Create .gitignore
cat > .gitignore << 'GITIGNORE'
__pycache__/
*.pyc
.venv/
output/*.txt
GITIGNORE

# Create sample student data
cat > data/students.csv << 'CSV'
name,age,grade,subject
Alice,20,85,Math
Bob,19,92,Science
Charlie,21,78,English
Diana,20,88,Math
Eve,22,95,Science
Frank,19,82,History
Grace,21,91,Math
Henry,20,76,Science
CSV

# Create Python script template
cat > src/data_analysis.py << 'PYTHON'
#!/usr/bin/env python3
"""Basic Student Data Analysis"""

def load_students(filename):
    """Load student data from CSV."""
    # TODO: Implement
    pass

def calculate_average_grade(students):
    """Calculate average grade."""
    # TODO: Implement
    pass

def main():
    print("Student Data Analysis")
    # TODO: Implement analysis

if __name__ == "__main__":
    main()
PYTHON

echo "✅ Project setup complete!"
EOF

# Make executable and run
chmod +x setup_project.sh
./setup_project.sh

# Show what was created
ls -la
tree || find . -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g'

# Commit the scaffold
git add .
git commit -m "Add project scaffold with automated setup script

- Create directory structure (src, data, output)
- Add .gitignore for Python projects
- Generate sample student CSV data
- Create Python script template"

git log --oneline --graph
```

**Key Points**:
- Feature branch workflow
- Shell script automation with heredocs
- chmod for executability
- Comprehensive commit message

### Part 3: Feature Branch - Python Implementation (8 min)

```bash
# Create feature branch for implementation
git checkout -b feature/data-processing

# Implement the data analysis script
cat > src/data_analysis.py << 'EOF'
#!/usr/bin/env python3
"""Basic Student Data Analysis"""

def load_students(filename):
    """Load student data from CSV."""
    students = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.strip().split(',')
                students.append({
                    'name': parts[0],
                    'age': int(parts[1]),
                    'grade': int(parts[2]),
                    'subject': parts[3]
                })
    return students

def calculate_average_grade(students):
    """Calculate average grade."""
    if not students:
        return 0
    total = sum(s['grade'] for s in students)
    return total / len(students)

def count_by_subject(students, subject):
    """Count students in a subject."""
    return sum(1 for s in students if s['subject'] == subject)

def generate_report(students):
    """Generate analysis report."""
    avg = calculate_average_grade(students)
    math_count = count_by_subject(students, 'Math')

    report = "Student Data Analysis Report\n"
    report += "=" * 40 + "\n\n"
    report += f"Total Students: {len(students)}\n"
    report += f"Average Grade: {avg:.1f}\n"
    report += f"Math Students: {math_count}\n\n"

    report += "Individual Grades:\n"
    report += "-" * 40 + "\n"
    for s in students:
        report += f"{s['name']:12} | {s['grade']:3d} | {s['subject']}\n"

    return report

def main():
    print("Student Data Analysis")
    print("=" * 30)

    # Load and analyze
    students = load_students('data/students.csv')
    print(f"Loaded {len(students)} students")

    # Generate report
    report = generate_report(students)

    # Save to file
    with open('output/analysis_report.txt', 'w') as f:
        f.write(report)

    print("Report saved to output/analysis_report.txt")
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
EOF

# Test the implementation
python3 src/data_analysis.py

# Show the output
echo "\n--- Generated Report ---"
cat output/analysis_report.txt

# Commit the implementation
git add src/data_analysis.py
git commit -m "Implement student data analysis

- Complete CSV loading with error handling
- Add statistical calculations (average, counts)
- Generate formatted report with f-strings
- Save results to output file"

git log --oneline --graph --all
```

**Key Points**:
- Separate feature branch for implementation
- Functions for modularity
- File I/O operations
- F-string formatting

### Part 4: Merge and Documentation (5 min)

```bash
# Switch to main and merge both branches
git checkout main

# Merge project scaffold
git merge feature/project-scaffold -m "Merge feature/project-scaffold into main

Add automated project setup infrastructure"

# Merge implementation
git merge feature/data-processing -m "Merge feature/data-processing into main

Add complete data analysis implementation"

# View the final history
git log --oneline --graph --all

# Update README with comprehensive documentation
cat > README.md << 'EOF'
# DataSci Week 02 Integration Project

## Overview
Integration of Git workflows, CLI automation, and Python data processing.

## Features
- **Automated Setup**: `setup_project.sh` creates project structure
- **Data Analysis**: Python script analyzes student grades
- **Git Workflow**: Feature branch development and merging

## Usage
```bash
# Setup project
./setup_project.sh

# Run analysis
python src/data_analysis.py

# View results
cat output/analysis_report.txt
```

## Git Workflow
| Branch | Purpose | Status |
|--------|---------|--------|
| main | Production code | Active |
| feature/project-scaffold | Setup automation | Merged |
| feature/data-processing | Analysis implementation | Merged |

Created for DataSci 217 - Week 02 Assignment
EOF

git add README.md
git commit -m "Update README with comprehensive documentation"

# Final state
echo "\n🎉 Complete! Final repository state:"
git log --oneline --graph --all
ls -la
```

**Key Points**:
- Merging feature branches to main
- Professional documentation
- Complete project history

## Discussion Questions

1. **Why use feature branches instead of working directly on main?**
   - Keeps main stable
   - Allows parallel development
   - Makes code review easier

2. **What are the benefits of the shell script approach?**
   - Eliminates repetitive setup
   - Ensures consistency
   - Documents project structure

3. **How does this demonstrate modular Python design?**
   - Functions with single responsibilities
   - Reusable components
   - Clear separation of concerns

## Common Mistakes to Highlight

- Forgetting to `chmod +x` the shell script
- Not testing scripts before committing
- Poor commit messages ("updated file" vs descriptive messages)
- Working on main instead of feature branches

---

# Demo 2: Virtual Environments and Python Potpourri

**Location in Lecture**: Second LIVE DEMO callout (after Python Essentials section)

**Purpose**: Demonstrate virtual environment setup with different tools and essential Python skills for data science.

**Time**: 15-20 minutes

## Overview

This demo shows how to create isolated Python environments and use essential Python features for professional development.

## Demo Script

### Part 1: Virtual Environment Comparison (10 min)

```bash
# Create a demo directory
mkdir venv-demo
cd venv-demo

# OPTION 1: Using venv (built-in)
echo "=== Using venv (Python built-in) ==="
python3 -m venv myproject-venv

# Activate
source myproject-venv/bin/activate  # Mac/Linux
# myproject-venv\Scripts\activate   # Windows

# Check we're in the environment
which python
python --version

# Install a package
pip install numpy

# Show installed packages
pip list

# Save requirements
pip freeze > requirements.txt
cat requirements.txt

# Deactivate
deactivate

echo "✅ venv demo complete"
echo ""

# OPTION 2: Using uv (modern, fast)
echo "=== Using uv (Fast & Modern) ==="

# Create environment
uv venv myproject-uv

# Activate
source myproject-uv/bin/activate

# Install packages (much faster!)
uv pip install numpy pandas

# Show installed packages
uv pip list

# Save requirements
uv pip freeze > requirements-uv.txt
cat requirements-uv.txt

# Deactivate
deactivate

echo "✅ uv demo complete"
echo ""

# OPTION 3: Using conda (if available)
echo "=== Using conda (Full environment management) ==="

# Create environment with specific Python version
conda create -n myproject-conda python=3.11 -y

# Activate
conda activate myproject-conda

# Install packages from conda
conda install numpy pandas matplotlib -y

# List packages
conda list

# Export environment
conda env export > environment.yml
cat environment.yml

# Deactivate
conda deactivate

echo "✅ conda demo complete"
```

**Key Points**:
- Three different tools for virtual environments
- Each has different strengths (built-in, speed, full management)
- All achieve the same goal: project isolation

### Part 2: Python Potpourri - Essential Skills (10 min)

Create a demo script:

```python
#!/usr/bin/env python3
"""
Python Potpourri: Essential Skills for Data Science
Demonstrates type checking, f-string formatting, and professional practices
"""

import numpy as np

def demo_type_checking():
    """Demonstrate type checking for debugging."""
    print("=== Type Checking ===")

    # Common type confusion
    user_input = "42"
    print(f"user_input = '{user_input}'")
    print(f"type(user_input) = {type(user_input)}")

    # Convert and check
    number = int(user_input)
    print(f"number = {number}")
    print(f"type(number) = {type(number)}")

    # Type checking with isinstance
    data = [1, 2, "3", 4, 5]
    print(f"\nData with mixed types: {data}")
    for item in data:
        if isinstance(item, str):
            print(f"  '{item}' is a string - converting to int")
        else:
            print(f"  {item} is already an int")

    print()

def demo_f_strings():
    """Demonstrate modern f-string formatting."""
    print("=== F-String Formatting ===")

    # Basic f-strings
    name = "Alice"
    grade = 87.5
    message = f"Student {name} earned {grade:.1f}%"
    print(message)

    # Formatting numbers
    value = 3.14159
    print(f"Default: {value}")
    print(f"2 decimals: {value:.2f}")
    print(f"Right-aligned (width 10): {value:>10.2f}")
    print(f"Left-aligned (width 10): {value:<10.2f}")
    print(f"Center-aligned (width 10): {value:^10.2f}")

    # Expressions in f-strings
    arr = np.array([85, 92, 78, 88, 95])
    print(f"\nGrades: {arr}")
    print(f"Mean: {arr.mean():.1f}")
    print(f"Max: {arr.max()}")
    print(f"Students above 85: {(arr > 85).sum()}")

    # Multi-line f-strings
    report = f"""
Student Analysis Report
{'=' * 30}
Student: {name}
Grade: {grade:.1f}
Class Average: {arr.mean():.1f}
Status: {'Above Average' if grade > arr.mean() else 'Below Average'}
"""
    print(report)

def demo_practical_patterns():
    """Demonstrate practical patterns for data science."""
    print("=== Practical Patterns ===")

    # Pattern 1: Data validation with type checking
    def safe_convert(value):
        """Safely convert value to int with type checking."""
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                print(f"Warning: Cannot convert '{value}' to int")
                return None
        else:
            print(f"Warning: Unexpected type {type(value)}")
            return None

    test_values = [42, "123", "invalid", 3.14]
    print("Safe conversion results:")
    for val in test_values:
        result = safe_convert(val)
        print(f"  {val!r:12} -> {result}")

    # Pattern 2: Professional logging with f-strings
    def analyze_grades(grades, student_name):
        """Analyze grades with professional output."""
        mean = np.mean(grades)
        std = np.std(grades)

        # Professional formatted output
        print(f"\nAnalysis for {student_name}:")
        print(f"  Samples: {len(grades)}")
        print(f"  Mean: {mean:.2f} ± {std:.2f}")
        print(f"  Range: [{min(grades)}, {max(grades)}]")

        # Conditional formatting
        status = "Excellent" if mean >= 90 else "Good" if mean >= 80 else "Fair"
        print(f"  Overall: {status}")

    grades = np.array([85, 92, 78, 88, 95, 87, 91])
    analyze_grades(grades, "Alice")

    print()

def main():
    """Run all demos."""
    print("Python Potpourri: Essential Skills Demo")
    print("=" * 50)
    print()

    demo_type_checking()
    demo_f_strings()
    demo_practical_patterns()

    print("✅ All demos complete!")

if __name__ == "__main__":
    main()
```

Run the demo:

```bash
python3 python_potpourri_demo.py
```

**Key Points**:
- Type checking prevents bugs
- F-strings are modern and readable
- Practical patterns for data science

## Discussion Questions

1. **When should you use each virtual environment tool?**
   - venv: Quick projects, no extra install
   - uv: Speed-critical or modern workflows
   - conda: Complex dependencies or non-Python packages

2. **Why is type checking important for data science?**
   - Data comes from various sources with inconsistent types
   - Debugging is easier when you understand types
   - Prevents runtime errors

3. **What makes f-strings better than older formatting methods?**
   - More readable
   - Faster performance
   - Can include expressions directly

---

# Demo 3: NumPy Arrays and Operations

**Location in Lecture**: Third LIVE DEMO callout (after NumPy Operations section)

**Purpose**: Demonstrate NumPy array operations, vectorization, and practical data analysis.

**Time**: 20-25 minutes

## Overview

This demo shows NumPy's power through performance comparisons and real student data analysis.

## Demo Script

### Part 1: Why NumPy Matters - Performance (5 min)

Create `numpy_performance.py`:

```python
#!/usr/bin/env python3
"""
NumPy Performance Demonstration
Shows why NumPy is essential for data science
"""

import numpy as np
import time

def measure_python_list():
    """Measure Python list performance."""
    print("=== Python List Approach ===")

    # Create data
    data = list(range(1_000_000))

    # Time the operation
    start = time.time()
    result = [x * 2 for x in data]
    end = time.time()

    elapsed_ms = (end - start) * 1000
    print(f"Time: {elapsed_ms:.2f} ms")
    print(f"Result sample: {result[:5]}")

    return elapsed_ms

def measure_numpy_array():
    """Measure NumPy array performance."""
    print("\n=== NumPy Array Approach ===")

    # Create data
    data = np.arange(1_000_000)

    # Time the operation
    start = time.time()
    result = data * 2
    end = time.time()

    elapsed_ms = (end - start) * 1000
    print(f"Time: {elapsed_ms:.2f} ms")
    print(f"Result sample: {result[:5]}")

    return elapsed_ms

def main():
    """Run performance comparison."""
    print("NumPy Performance Comparison")
    print("=" * 40)
    print("Operation: Multiply 1 million numbers by 2\n")

    # Run tests
    python_time = measure_python_list()
    numpy_time = measure_numpy_array()

    # Show comparison
    print("\n" + "=" * 40)
    print(f"Speedup: {python_time / numpy_time:.1f}x faster!")
    print(f"Time saved: {python_time - numpy_time:.2f} ms")
    print("\n✅ NumPy is 10-100x faster for numerical operations")

if __name__ == "__main__":
    main()
```

Run:
```bash
python3 numpy_performance.py
```

**Key Points**:
- NumPy is dramatically faster
- Vectorized operations eliminate Python loops
- Critical for real data science workloads

### Part 2: NumPy in Action - Student Grade Analysis (15 min)

Create `student_analysis.py`:

```python
#!/usr/bin/env python3
"""
Student Grade Analysis with NumPy
Demonstrates practical NumPy operations
"""

import numpy as np

def create_sample_data():
    """Create realistic student grade data."""
    np.random.seed(42)  # Reproducible results

    n_students = 100
    n_assignments = 5

    # Generate grades (70-100 range)
    grades = np.random.randint(70, 101, size=(n_students, n_assignments))

    print(f"Created data: {n_students} students, {n_assignments} assignments")
    print(f"Array shape: {grades.shape}")
    print(f"Data type: {grades.dtype}\n")

    return grades

def demo_basic_operations(grades):
    """Demonstrate basic NumPy operations."""
    print("=== Basic Operations ===")

    # Arithmetic operations
    print(f"First student's grades: {grades[0]}")
    print(f"Doubled: {grades[0] * 2}")
    print(f"Curved (+5): {grades[0] + 5}")
    print()

def demo_statistical_operations(grades):
    """Demonstrate statistical operations."""
    print("=== Statistical Operations ===")

    # Overall statistics
    print(f"Overall average: {grades.mean():.1f}")
    print(f"Overall std dev: {grades.std():.1f}")
    print(f"Highest grade: {grades.max()}")
    print(f"Lowest grade: {grades.min()}")
    print()

    # Axis-specific operations
    student_averages = grades.mean(axis=1)  # Average per student
    assignment_averages = grades.mean(axis=0)  # Average per assignment

    print("Student averages (first 5):")
    print(student_averages[:5])
    print(f"\nAssignment averages:")
    for i, avg in enumerate(assignment_averages, 1):
        print(f"  Assignment {i}: {avg:.1f}")
    print()

def demo_boolean_indexing(grades):
    """Demonstrate boolean indexing."""
    print("=== Boolean Indexing ===")

    # Calculate student averages
    student_averages = grades.mean(axis=1)

    # Find high performers
    high_performers = student_averages > 90
    print(f"Students with average > 90: {high_performers.sum()}")
    print(f"High performer averages: {student_averages[high_performers][:5]}")

    # Multiple conditions
    excellent = (student_averages >= 90) & (student_averages <= 100)
    good = (student_averages >= 80) & (student_averages < 90)
    fair = (student_averages >= 70) & (student_averages < 80)

    print(f"\nGrade distribution:")
    print(f"  Excellent (90-100): {excellent.sum()} students")
    print(f"  Good (80-89): {good.sum()} students")
    print(f"  Fair (70-79): {fair.sum()} students")
    print()

def demo_array_reshaping(grades):
    """Demonstrate array reshaping."""
    print("=== Array Reshaping ===")

    # Get first 12 grades
    sample = grades.flat[:12]
    print(f"Flattened sample (12 grades): {sample}")

    # Reshape to different dimensions
    reshaped = sample.reshape(3, 4)
    print(f"\nReshaped to 3x4:")
    print(reshaped)

    # Transpose
    print(f"\nTransposed (4x3):")
    print(reshaped.T)
    print()

def demo_practical_analysis(grades):
    """Demonstrate practical analysis workflow."""
    print("=== Practical Analysis Workflow ===")

    # Find the hardest assignment
    assignment_averages = grades.mean(axis=0)
    hardest_idx = assignment_averages.argmin()
    easiest_idx = assignment_averages.argmax()

    print(f"Hardest assignment: #{hardest_idx + 1} (avg: {assignment_averages[hardest_idx]:.1f})")
    print(f"Easiest assignment: #{easiest_idx + 1} (avg: {assignment_averages[easiest_idx]:.1f})")

    # Find top 5 students
    student_averages = grades.mean(axis=1)
    top_5_indices = np.argsort(student_averages)[-5:][::-1]

    print(f"\nTop 5 students:")
    for rank, idx in enumerate(top_5_indices, 1):
        print(f"  #{rank}: Student {idx:3d} - Average: {student_averages[idx]:.1f}")

    # Calculate improvement (first vs last assignment)
    improvement = grades[:, -1] - grades[:, 0]
    improved_students = (improvement > 0).sum()
    avg_improvement = improvement[improvement > 0].mean()

    print(f"\nImprovement analysis:")
    print(f"  Students who improved: {improved_students}")
    print(f"  Average improvement: {avg_improvement:.1f} points")
    print()

def main():
    """Run all NumPy demos."""
    print("Student Grade Analysis with NumPy")
    print("=" * 50)
    print()

    # Create data
    grades = create_sample_data()

    # Run demos
    demo_basic_operations(grades)
    demo_statistical_operations(grades)
    demo_boolean_indexing(grades)
    demo_array_reshaping(grades)
    demo_practical_analysis(grades)

    print("✅ NumPy analysis complete!")

if __name__ == "__main__":
    main()
```

Run:
```bash
python3 student_analysis.py
```

**Key Points**:
- NumPy arrays enable vectorized operations
- Axis parameter for different aggregations
- Boolean indexing for filtering
- Practical data analysis patterns

## Discussion Questions

1. **Why is NumPy so much faster than Python lists?**
   - Contiguous memory storage
   - Vectorized C/Fortran operations
   - No Python overhead for loops

2. **When do you use axis=0 vs axis=1?**
   - axis=0: Operations across rows (column-wise)
   - axis=1: Operations across columns (row-wise)
   - Think: "axis to collapse"

3. **How does boolean indexing simplify data filtering?**
   - No explicit loops needed
   - Expressive and readable
   - Fast C-level implementation

---

# Demo 4: Command Line Data Processing

**Location in Lecture**: Fourth LIVE DEMO callout (after Command Line Data Processing section)

**Purpose**: Demonstrate essential shell tools for data exploration and preprocessing.

**Time**: 15-20 minutes

## Overview

This demo shows how to use command line tools to quickly explore and process data files.

## Demo Script

Create `cli_demo.sh`:

```bash
#!/bin/bash
# Command Line Data Processing Demo
# Demonstrates cut, sort, grep, tr, sed, awk, and pipelines

echo "=== Command Line Data Processing Demo ==="
echo ""

# Setup: Create sample data files
echo "Creating sample data files..."

cat > students.csv << 'EOF'
name,age,grade,subject
Alice,20,85,Math
Bob,19,92,Science
Charlie,21,78,English
Diana,20,88,Math
Eve,22,95,Science
Frank,19,82,History
Grace,21,91,Math
Henry,20,76,Science
Isabel,19,89,English
Jack,22,84,Math
EOF

cat > sales.csv << 'EOF'
date,product,quantity,price
2024-01-01,Widget,10,25.50
2024-01-02,Gadget,5,45.00
2024-01-03,Widget,8,25.50
2024-01-04,Doohickey,12,15.75
2024-01-05,Gadget,7,45.00
2024-01-06,Widget,15,25.50
EOF

echo "✅ Sample data created"
echo ""

# Demo 1: cut - Extract columns
echo "=== Demo 1: cut - Extract Columns ==="
echo "Extract student names and grades:"
cut -d',' -f1,3 students.csv | head -5
echo ""

echo "Extract first 10 characters of each line:"
cut -c1-10 students.csv | head -3
echo ""

# Demo 2: sort - Sort data
echo "=== Demo 2: sort - Sort Data ==="
echo "Sort students by name (alphabetically):"
sort students.csv | head -5
echo ""

echo "Sort by grade (numerically, 3rd field):"
sort -t',' -k3 -n students.csv | head -5
echo ""

# Demo 3: uniq - Remove duplicates
echo "=== Demo 3: uniq - Remove Duplicates ==="
echo "Count students per subject:"
cut -d',' -f4 students.csv | sort | uniq -c
echo ""

# Demo 4: grep - Search and filter
echo "=== Demo 4: grep - Search and Filter ==="
echo "Find all Math students:"
grep "Math" students.csv
echo ""

echo "Find students NOT in Science:"
grep -v "Science" students.csv | head -3
echo ""

echo "Case-insensitive search for 'alice':"
grep -i "alice" students.csv
echo ""

# Demo 5: tr - Transform characters
echo "=== Demo 5: tr - Transform Characters ==="
echo "Convert to uppercase:"
echo "Alice,20,85,Math" | tr 'a-z' 'A-Z'
echo ""

echo "Replace commas with tabs:"
head -3 students.csv | tr ',' '\t'
echo ""

# Demo 6: sed - Stream editor
echo "=== Demo 6: sed - Stream Editor ==="
echo "Replace 'Math' with 'Mathematics':"
sed 's/Math/Mathematics/g' students.csv | grep Mathematics | head -2
echo ""

echo "Delete the header line:"
sed '1d' students.csv | head -3
echo ""

# Demo 7: awk - Pattern processing
echo "=== Demo 7: awk - Pattern Processing ==="
echo "Print names and subjects:"
awk -F',' '{print $1, $4}' students.csv | head -5
echo ""

echo "Filter students with grade > 85:"
awk -F',' '$3 > 85 {print $1, $3}' students.csv
echo ""

echo "Calculate average grade:"
awk -F',' 'NR>1 {sum+=$3; count++} END {print "Average:", sum/count}' students.csv
echo ""

# Demo 8: Complex pipelines
echo "=== Demo 8: Complex Data Pipelines ==="

echo "Pipeline 1: Find top 3 Math students"
echo "Process: filter Math -> extract name,grade -> sort -> top 3"
grep "Math" students.csv | \
  cut -d',' -f1,3 | \
  sort -t',' -k2 -nr | \
  head -n 3
echo ""

echo "Pipeline 2: Subject distribution (sorted)"
echo "Process: extract subjects -> sort -> count -> sort by count"
cut -d',' -f4 students.csv | \
  tail -n +2 | \
  sort | \
  uniq -c | \
  sort -nr
echo ""

echo "Pipeline 3: High performers analysis"
echo "Process: filter grade>85 -> extract -> uppercase -> format"
awk -F',' '$3 > 85 {print $1","$4}' students.csv | \
  tr 'a-z' 'A-Z' | \
  sed 's/,/ - /g'
echo ""

# Demo 9: Sales data analysis
echo "=== Demo 9: Sales Data Analysis ==="

echo "Total revenue by product:"
awk -F',' 'NR>1 {revenue[$2] += $3 * $4} END {for (p in revenue) print p, revenue[p]}' sales.csv | \
  sort -k2 -nr
echo ""

echo "Total units sold:"
awk -F',' 'NR>1 {sum += $3} END {print "Total units:", sum}' sales.csv
echo ""

# Demo 10: Practical use case
echo "=== Demo 10: Generate Summary Report ==="

cat > summary_report.txt << 'REPORT'
Student Performance Summary
===========================

Top Performers (Grade > 85):
REPORT

awk -F',' '$3 > 85 {print $1, "-", $3}' students.csv >> summary_report.txt

echo "" >> summary_report.txt
echo "Subject Distribution:" >> summary_report.txt

cut -d',' -f4 students.csv | tail -n +2 | sort | uniq -c | sort -nr >> summary_report.txt

echo ""
echo "Generated summary_report.txt:"
cat summary_report.txt
echo ""

# Cleanup
echo "=== Cleanup ==="
read -p "Remove sample files? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm students.csv sales.csv summary_report.txt
    echo "✅ Sample files removed"
else
    echo "Sample files kept for exploration"
fi

echo ""
echo "✅ Demo complete!"
```

Make executable and run:

```bash
chmod +x cli_demo.sh
./cli_demo.sh
```

**Key Points**:
- Command line tools are powerful for quick data exploration
- Pipelines chain commands together
- Each tool does one thing well
- Much faster than loading into Python for simple operations

## Discussion Questions

1. **When should you use command line tools vs Python?**
   - CLI: Quick exploration, simple filtering, large files
   - Python: Complex analysis, visualizations, ML

2. **What does the backslash `\` do in pipelines?**
   - Line continuation character
   - Makes long commands more readable
   - Purely cosmetic (for humans)

3. **How do pipes work?**
   - Output of one command becomes input to next
   - Data flows through the pipeline
   - No intermediate files needed

## Common Mistakes to Highlight

- Forgetting to skip header lines (use `tail -n +2` or `NR>1` in awk)
- Wrong delimiter in `cut` or `awk`
- Not sorting before using `uniq`
- Forgetting to make scripts executable

---

# Summary

These four demos cover the complete lecture content:

1. **Demo 1** - Integration of previous week's material (Git + CLI + Python)
2. **Demo 2** - Virtual environments and Python essentials for professional development
3. **Demo 3** - NumPy arrays and vectorized operations for data science
4. **Demo 4** - Command line tools for data exploration and preprocessing

Each demo is self-contained but builds on previous concepts, following the lecture structure exactly.

## Running All Demos

```bash
# Demo 1: Assignment walkthrough (interactive, follow script)
# Covered in instructor-led walkthrough

# Demo 2: Virtual environments
cd 03/demo
python3 python_potpourri_demo.py

# Demo 3: NumPy operations
python3 numpy_performance.py
python3 student_analysis.py

# Demo 4: Command line processing
./cli_demo.sh
```

## Next Steps

After these demos, students should be able to:
- Set up proper Git workflows for data science projects
- Create and manage virtual environments
- Use NumPy for efficient numerical computing
- Process data with command line tools
- Apply professional Python development practices

Ready for **Lecture 4: Pandas DataFrames**!