# Assignment 02: Git Workflow, CLI Automation, and Python Data Processing

**Due Date**: [To be set by instructor]  
**Points**: 20 total  
**Estimated Time**: 3-4 hours

## Learning Objectives

By completing this assignment, you will demonstrate competence in:

1. **Git Version Control**: Branching, committing, merging, and collaboration workflows
2. **Command Line Interface**: Shell scripting, file operations, and automation
3. **Python Programming**: Functions, file I/O, data processing, and modular design

## Requirements

This assignment has **three progressive parts** that build upon each other. Each part focuses on one of the three main lecture topics while integrating with the others.

### Part 1: Git Workflow Mastery (7 points)

**Objective**: Demonstrate Git branching, committing, and merging workflows.

**Tasks**:
1. Create a new repository named `datasci-week02-integration`
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

**Git Workflow Requirements**:
```bash
# Your repository should have this structure:
main
├── feature/project-scaffold (merged)
└── feature/data-processing (merged)
```

**Documentation Requirements** (README.md):
```markdown
# DataSci Week 02 Integration Project

## Project Overview
This project demonstrates integration of Git workflows, CLI automation, and Python data processing.

## Project Structure
```
datasci-week02-integration/
├── README.md
├── .gitignore
├── requirements.txt
├── setup_project.sh
├── src/
│   ├── data_analysis.py
│   └── data_analysis_functions.py
├── data/
│   ├── students.csv
│   └── courses.json
└── output/
    └── analysis_report.txt
```

## Features
- **Project Scaffold**: Automated project setup with `setup_project.sh`
- **Data Processing**: Python scripts for student grade analysis
- **Git Workflow**: Feature branch development and merging

## Usage
1. Run `./setup_project.sh` to create project structure
2. Execute `python src/data_analysis.py` for basic analysis
3. Run `python src/data_analysis_functions.py` for advanced analysis

## Git Workflow
| Branch | Purpose | Status |
|--------|---------|--------|
| main | Production code | Active |
| feature/project-scaffold | CLI automation | Merged |
| feature/data-processing | Python analysis | Merged |
```

### Part 2: CLI Project Scaffold Script (6 points)

**Objective**: Create a shell script that automates project setup.

**Tasks**:
1. Create `setup_project.sh` with the following functionality:
   - Create directory structure
   - Generate initial files (.gitignore, requirements.txt)
   - Create sample data files
   - Set up Python template files
2. Make the script executable and test it
3. Commit the script to the `feature/project-scaffold` branch

**Script Requirements**:
```bash
#!/bin/bash
# setup_project.sh - Automated project setup script

# Function to create directory structure
create_directories() {
    echo "Creating directory structure..."
    mkdir -p src data output
    echo "✓ Directories created"
}

# Function to create initial files
create_initial_files() {
    echo "Creating initial files..."
    
    # Create .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# Data files
data/raw/*.csv
data/raw/*.json
output/*.txt

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

    # Create requirements.txt
    cat > requirements.txt << 'EOF'
# DataSci 217 - Week 02 Requirements
# No external dependencies required for this assignment
EOF

    echo "✓ Initial files created"
}

# Function to generate sample data
create_sample_data() {
    echo "Generating sample data..."
    
    # Create students.csv
    cat > data/students.csv << 'EOF'
name,age,grade,subject
Alice,20,85,Math
Bob,19,92,Science
Charlie,21,78,English
Diana,20,88,Math
Eve,22,95,Science
Frank,19,82,History
Grace,21,91,Math
Henry,20,76,Science
EOF

    # Create courses.json
    cat > data/courses.json << 'EOF'
{
    "courses": [
        {"id": "MATH101", "name": "Calculus I", "credits": 4},
        {"id": "SCI201", "name": "Physics I", "credits": 3},
        {"id": "ENG101", "name": "Composition", "credits": 3},
        {"id": "HIST101", "name": "World History", "credits": 3}
    ]
}
EOF

    echo "✓ Sample data generated"
}

# Function to create Python templates
create_python_templates() {
    echo "Creating Python templates..."
    
    # Create basic analysis script
    cat > src/data_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Basic Data Analysis Script
Demonstrates file I/O and data processing
"""

def main():
    print("Data Analysis Script")
    print("=" * 30)
    
    # TODO: Implement data analysis
    print("Analysis complete!")

if __name__ == "__main__":
    main()
EOF

    # Create functions template
    cat > src/data_analysis_functions.py << 'EOF'
#!/usr/bin/env python3
"""
Advanced Data Analysis with Functions
Demonstrates modular design and function usage
"""

def load_data(filename):
    """Load data from file."""
    # TODO: Implement data loading
    pass

def analyze_data(data):
    """Analyze the data."""
    # TODO: Implement analysis
    pass

def save_results(results, filename):
    """Save results to file."""
    # TODO: Implement result saving
    pass

def main():
    """Main function."""
    print("Advanced Data Analysis")
    print("=" * 30)
    
    # TODO: Implement main workflow
    print("Analysis complete!")

if __name__ == "__main__":
    main()
EOF

    echo "✓ Python templates created"
}

# Main execution
main() {
    echo "Setting up DataSci Week 02 Integration Project..."
    echo "=" * 50
    
    create_directories
    create_initial_files
    create_sample_data
    create_python_templates
    
    echo ""
    echo "✅ Project setup complete!"
    echo "Next steps:"
    echo "1. Review generated files"
    echo "2. Implement Python analysis scripts"
    echo "3. Test your implementation"
}

# Run main function
main "$@"
```

### Part 3: Python Data Processing (7 points)

**Objective**: Implement Python scripts that process data and output results to files.

**Tasks**:
1. Complete `src/data_analysis.py` with basic functionality
2. Complete `src/data_analysis_functions.py` with modular design
3. Ensure both scripts output results to `output/analysis_report.txt`
4. Test your implementation thoroughly

**Python Requirements**:

**Basic Analysis Script** (`src/data_analysis.py`):
```python
#!/usr/bin/env python3
"""
Basic Data Analysis Script
Demonstrates file I/O and data processing
"""

import csv
import json
from pathlib import Path

def main():
    print("Data Analysis Script")
    print("=" * 30)
    
    # Load student data
    students = load_students('data/students.csv')
    courses = load_courses('data/courses.json')
    
    # Perform basic analysis
    total_students = len(students)
    average_grade = calculate_average_grade(students)
    math_students = count_math_students(students)
    
    # Generate report
    report = generate_report(total_students, average_grade, math_students)
    
    # Save results
    save_report(report, 'output/analysis_report.txt')
    
    print(f"Analysis complete! Results saved to output/analysis_report.txt")

def load_students(filename):
    """Load student data from CSV file."""
    students = []
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                students.append({
                    'name': row['name'],
                    'age': int(row['age']),
                    'grade': int(row['grade']),
                    'subject': row['subject']
                })
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
    except Exception as e:
        print(f"Error loading students: {e}")
    
    return students

def load_courses(filename):
    """Load course data from JSON file."""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return {}
    except Exception as e:
        print(f"Error loading courses: {e}")
        return {}

def calculate_average_grade(students):
    """Calculate average grade across all students."""
    if not students:
        return 0
    total = sum(student['grade'] for student in students)
    return total / len(students)

def count_math_students(students):
    """Count students taking Math courses."""
    return len([s for s in students if s['subject'].lower() == 'math'])

def generate_report(total_students, average_grade, math_students):
    """Generate analysis report."""
    report = f"""Student Grade Analysis Report
{'=' * 40}

Summary Statistics:
- Total Students: {total_students}
- Average Grade: {average_grade:.1f}
- Math Students: {math_students}

Analysis completed successfully!
"""
    return report

def save_report(report, filename):
    """Save report to file."""
    try:
        Path('output').mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(report)
        print(f"Report saved to {filename}")
    except Exception as e:
        print(f"Error saving report: {e}")

if __name__ == "__main__":
    main()
```

**Advanced Analysis Script** (`src/data_analysis_functions.py`):
```python
#!/usr/bin/env python3
"""
Advanced Data Analysis with Functions
Demonstrates modular design and function usage
"""

import csv
import json
from pathlib import Path

def load_data(filename):
    """Load data from CSV or JSON file."""
    if filename.endswith('.csv'):
        return load_csv(filename)
    elif filename.endswith('.json'):
        return load_json(filename)
    else:
        print(f"Unsupported file format: {filename}")
        return None

def load_csv(filename):
    """Load data from CSV file."""
    data = []
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append({
                    'name': row['name'],
                    'age': int(row['age']),
                    'grade': int(row['grade']),
                    'subject': row['subject']
                })
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
    except Exception as e:
        print(f"Error loading CSV: {e}")
    
    return data

def load_json(filename):
    """Load data from JSON file."""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return {}
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}

def analyze_data(students):
    """Analyze student data and return statistics."""
    if not students:
        return {}
    
    grades = [student['grade'] for student in students]
    
    analysis = {
        'total_students': len(students),
        'average_grade': sum(grades) / len(grades),
        'highest_grade': max(grades),
        'lowest_grade': min(grades),
        'math_students': len([s for s in students if s['subject'].lower() == 'math']),
        'science_students': len([s for s in students if s['subject'].lower() == 'science']),
        'grade_distribution': analyze_grade_distribution(grades)
    }
    
    return analysis

def analyze_grade_distribution(grades):
    """Analyze grade distribution."""
    distribution = {
        'A (90-100)': 0,
        'B (80-89)': 0,
        'C (70-79)': 0,
        'D (60-69)': 0,
        'F (0-59)': 0
    }
    
    for grade in grades:
        if grade >= 90:
            distribution['A (90-100)'] += 1
        elif grade >= 80:
            distribution['B (80-89)'] += 1
        elif grade >= 70:
            distribution['C (70-79)'] += 1
        elif grade >= 60:
            distribution['D (60-69)'] += 1
        else:
            distribution['F (0-59)'] += 1
    
    return distribution

def save_results(results, filename):
    """Save analysis results to file."""
    try:
        Path('output').mkdir(exist_ok=True)
        
        with open(filename, 'w') as file:
            file.write("Advanced Student Analysis Report\n")
            file.write("=" * 40 + "\n\n")
            
            file.write("Basic Statistics:\n")
            file.write(f"- Total Students: {results['total_students']}\n")
            file.write(f"- Average Grade: {results['average_grade']:.1f}\n")
            file.write(f"- Highest Grade: {results['highest_grade']}\n")
            file.write(f"- Lowest Grade: {results['lowest_grade']}\n\n")
            
            file.write("Subject Distribution:\n")
            file.write(f"- Math Students: {results['math_students']}\n")
            file.write(f"- Science Students: {results['science_students']}\n\n")
            
            file.write("Grade Distribution:\n")
            for grade_range, count in results['grade_distribution'].items():
                percentage = (count / results['total_students']) * 100
                file.write(f"- {grade_range}: {count} students ({percentage:.1f}%)\n")
            
            file.write("\nAnalysis completed successfully!\n")
        
        print(f"Results saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def main():
    """Main function demonstrating module usage."""
    print("Advanced Data Analysis")
    print("=" * 30)
    
    # Load data
    students = load_data('data/students.csv')
    courses = load_data('data/courses.json')
    
    if not students:
        print("No student data to analyze")
        return
    
    # Analyze data
    results = analyze_data(students)
    
    # Save results
    if save_results(results, 'output/analysis_report.txt'):
        print("✅ Advanced analysis complete!")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()
```

## Submission Requirements

1. **Repository Structure**: Your repository must have the exact structure shown in the documentation
2. **Git History**: At least 3 commits per branch with meaningful commit messages
3. **Working Scripts**: Both Python scripts must run without errors
4. **Output Files**: `output/analysis_report.txt` must be generated by both scripts
5. **Documentation**: Professional README.md with project overview and usage instructions

## Grading Rubric

### Part 1: Git Workflow (7 points)
- **Repository Setup** (2 points): Proper repository creation and initial commit
- **Branch Management** (2 points): Correct branch creation and switching
- **Merge Workflow** (2 points): Successful merging of feature branches
- **Documentation** (1 point): Professional README.md with project overview

### Part 2: CLI Automation (6 points)
- **Script Functionality** (3 points): `setup_project.sh` creates all required files and directories
- **Script Quality** (2 points): Proper error handling and user feedback
- **Integration** (1 point): Script works with Git workflow

### Part 3: Python Programming (7 points)
- **Basic Script** (3 points): `data_analysis.py` processes data and outputs results
- **Advanced Script** (3 points): `data_analysis_functions.py` demonstrates modular design
- **File I/O** (1 point): Both scripts successfully write to `output/analysis_report.txt`

## Testing Your Assignment

Before submitting, test your implementation:

```bash
# Test the setup script
./setup_project.sh

# Test basic analysis
python src/data_analysis.py

# Test advanced analysis
python src/data_analysis_functions.py

# Verify output files
ls -la output/
cat output/analysis_report.txt
```

## Common Issues to Avoid

1. **Git Issues**: Don't forget to commit and push your changes
2. **File Paths**: Ensure all file paths are correct and relative
3. **Python Errors**: Test your scripts before submitting
4. **Documentation**: Make sure your README.md is comprehensive and professional

## Getting Help

- Review the demo guide for detailed examples
- Check the lecture content for theoretical background
- Test your code incrementally as you develop it
- Use Git to track your progress and revert if needed

## Due Date

**Assignment due**: [To be set by instructor]  
**Late submissions**: [Policy to be set by instructor]

---

**Remember**: This assignment focuses on **competence** rather than expertise. The goal is to demonstrate that you can use Git, CLI, and Python effectively for data science workflows.