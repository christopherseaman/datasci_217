# Assignment 02: Git Workflow, CLI Automation, and Python Data Processing

**Due Date**: [To be set by instructor]
**Points**: 20 total
**Estimated Time**: 3-4 hours

## Provided Files

- `requirements.txt` - Python requirements file (no external packages needed)
- `TIPS.md` - Troubleshooting guide for common issues
- `.github/test/test_assignment.py` - Automated tests for grading

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
1. Create a new repository named `datasci-week02-integration` (separate from this assignment folder)
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

### Part 3: Python Data Processing (7 points)

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

**Hints**:
- Skip the header line when reading CSV: `lines[1:]`
- Split CSV lines: `line.strip().split(',')`
- Convert strings to integers: `int(value)`
- Format decimals: `f"{average:.1f}"`
- Create output directory if needed before writing file

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