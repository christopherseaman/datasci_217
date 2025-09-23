# Demo Guide - Lecture 2

# Demo 1: Git & GitHub Workflow

## Step 1: Initialize Repository - Three Ways

### Initialize Repository: Command Line

```bash
# Create project directory
mkdir data_analysis_project
cd data_analysis_project

# Initialize Git repository
git init
git status

# Configure Git (one-time setup)
git config --global user.name "Your Name"
git config --global user.email "your.email@ucsf.edu"
```

### Initialize Repository: VS Code

1. Open VS Code
2. File → "Open Folder" → Create new folder: `data_analysis_project`
3. Open integrated terminal: `Ctrl+`` (backtick)
4. Run: `git init`
5. Configure Git (one-time setup):
   - `git config --global user.name "Your Name"`
   - `git config --global user.email "your.email@ucsf.edu"`

### Initialize Repository: GitHub

1. Go to github.com and click "+" → "New repository"
2. Name: `data_analysis_project`
3. Description: "DataSci 217 - Git workflow demonstration"
4. Make it public
5. **DO NOT** initialize with README, .gitignore, or license
6. Click "Create repository"
7. Click "Code" → "Clone" → Copy the repository URL

## Step 2: Create Initial Files - Three Ways

### Create Initial Files: Command Line

```bash
# Create directory structure
mkdir data scripts results

# Create initial files
touch README.md scripts/analysis.py data/sample_data.csv

# Add some content to README.md
echo "# Data Analysis Project" > README.md
echo "This project demonstrates Git workflow." >> README.md
```

### Create Initial Files: VS Code

1. Create folders: `data`, `scripts`, `results`
2. Create files: `README.md`, `scripts/analysis.py`, `data/sample_data.csv`
3. Open `README.md` and add content:

```markdown
# Data Analysis Project
This project demonstrates Git workflow.
```

### Create Initial Files: GitHub

1. Go to your repository on GitHub
2. Click "Add file" → "Create new file"
3. Name: `README.md`
4. Add content:

```markdown
# Data Analysis Project
This project demonstrates Git workflow.
```

5. Commit message: "Add initial README"
6. Click "Commit new file"

## Step 3: Initial Commit - Three Ways

### Initial Commit: Command Line

```bash
# Stage all files
git add .

# Commit with message
git commit -m "Initial project setup"

# Check status
git status
```

### Initial Commit: VS Code

1. Open VS Code in your project directory: `code .`
2. Open Source Control panel: `Ctrl+Shift+G` (Windows/Linux) or `Cmd+Shift+G` (Mac)
3. You should see your files listed under "Changes"
4. Stage files by clicking the `+` button next to each file
5. Type commit message: "Initial project setup"
6. Press `Ctrl+Enter` (Windows/Linux) or `Cmd+Enter` (Mac) to commit
7. Make a change to README.md and watch the diff appear in VS Code

### Initial Commit: GitHub

1. Go to your repository on GitHub
2. Click "Add file" → "Create new file"
3. Name: `scripts/analysis.py`
4. Add content: `print("Hello from GitHub!")`
5. Commit message: "Add analysis script via GitHub"
6. Click "Commit new file"

## Step 4: Publish to GitHub - Three Ways

### Publish to GitHub: Command Line

```bash
# Add GitHub as remote origin
git remote add origin https://github.com/YOUR_USERNAME/data-analysis-project.git

# Push your local commits to GitHub
git push -u origin main

# Verify connection
git remote -v
```

### Publish to GitHub: VS Code

1. Open Source Control panel: `Ctrl+Shift+G` (Windows/Linux) or `Cmd+Shift+G` (Mac)
2. Click "Publish to GitHub" button (if you see it)
3. Or click "..." menu → "Publish to GitHub"
4. Choose repository name: `data-analysis-project`
5. Choose visibility: Public or Private
6. Click "Publish to GitHub"
7. VS Code will automatically push your commits

### Publish to GitHub: GitHub

1. Go to your existing repository on GitHub
2. The repository already exists from Step 1
3. Your local changes are already published
4. No additional steps needed - the repo is live on GitHub

## Step 5: View Git History - Three Ways

### View Git History: Command Line

```bash
# Check status
git status

# View commit history
git log --oneline

# View what changed in last commit
git show

# View differences between working directory and last commit
git diff
```

### View Git History: VS Code

1. Open Source Control panel: `Ctrl+Shift+G`
2. Click on "..." menu → "View History"
3. See commit history in timeline view
4. Click on any commit to see changes
5. Use "Compare with..." to see differences

### View Git History: GitHub

1. Go to your repository on GitHub
2. Click on "commits" link (shows number of commits)
3. Click on any commit to see changes
4. Use "Browse files" to see repository state at any commit
5. Click on file names to see individual file changes

## Step 6: Create a Branch - Three Ways

### Create a Branch: Command Line

```bash
# Create and switch to feature branch
git checkout -b feature-data-processing

# Make changes to scripts/analysis.py
echo 'print("Hello from feature branch!")' > scripts/analysis.py

# Stage and commit changes
git add scripts/analysis.py
git commit -m "Add basic analysis script"

# Push feature branch to GitHub
git push origin feature-data-processing
```

### Create a Branch: VS Code

1. Open Command Palette: `Ctrl+Shift+P`
2. Type "Git: Create Branch"
3. Branch name: `feature-data-processing`
4. Make changes to `scripts/analysis.py`
5. Stage and commit in Source Control panel
6. Click sync button to push branch

### Create a Branch: GitHub

1. Go to your repository on GitHub
2. Click "main" branch dropdown
3. Type new branch name: `feature-data-processing`
4. Click "Create branch: feature-data-processing"
5. Edit files directly in GitHub
6. Commit changes with message

## Step 7: Merging Branches - Three Ways

### Merging Branches: Command Line

```bash
# Switch to the branch you'd like to merge into (in our case 'main')
git checkout main

# Merge feature branch
git merge feature-data-processing

# Push merged changes
git push origin main

# Delete feature branch (optional cleanup)
git branch -d feature-data-processing
git push origin --delete feature-data-processing
```

### Merging Branches: VS Code

1. Switch to the target branch: Click branch name in status bar (in our case 'main')
2. Open Command Palette: `Ctrl+Shift+P`
3. Type "Git: Merge Branch"
4. Select `feature-data-processing`
5. Push changes using sync button
6. Delete feature branch: Command Palette → "Git: Delete Branch"

### Merging Branches: GitHub

1. Go to your repository on GitHub
2. Click "Compare & pull request" (if you see the banner)
3. Or go to "Pull requests" tab → "New pull request"
4. Select branches: `feature-data-processing` → `main` (or your target branch)
5. Add title and description
6. Click "Create pull request"
7. Click "Merge pull request" → "Confirm merge"
8. Delete feature branch after merge

## Step 8: Undo Uncommitted Changes - Three Ways

### Undo Uncommitted Changes: Command Line

```bash
# Make a MISTAKEN change to README.md
echo "This is a mistake" >> README.md

# Check what changed
git diff

# Revert the file to last committed version
git checkout -- README.md

# Verify the change is gone
git status
```

### Undo Uncommitted Changes: VS Code

1. Make a change to README.md in VS Code
2. Open Source Control panel
3. Right-click on README.md in "Changes"
4. Select "Discard Changes"
5. Confirm the revert
6. File returns to last committed state

### Undo Uncommitted Changes: GitHub

1. Go to your repository on GitHub
2. Click on README.md to view it
3. Click "Edit" (pencil icon)
4. Make changes and commit
5. To revert: Go to file history, find previous version
6. Click "Revert" on the commit you want to go back to

## Step 9: Pull Changes (Simulate Collaboration) - Three Ways

### Pull Changes: Command Line

```bash
# Fetch latest changes from GitHub
git fetch origin

# Pull any new commits
git pull origin main

# Check for updates
git status
```

### Pull Changes: VS Code

1. Open Source Control panel
2. Click sync button (circular arrows)
3. Or use Command Palette: `Ctrl+Shift+P` → "Git: Pull"
4. VS Code will show if there are conflicts
5. Resolve any conflicts in the editor

### Pull Changes: GitHub

1. Go to your repository on GitHub
2. Make changes directly in GitHub
3. Commit changes
4. These changes will be available for local pull
5. Use local tools (CLI or VS Code) to pull changes

## Step 10: Rolling Back Commits - Three Ways

### Rolling Back Commits: Command Line

```bash
# ⚠️ WARNING: Only roll back commits that haven't been pushed to GitHub!
# Check if you've pushed: git log --oneline origin/main..HEAD

# Roll back the last commit (keeps changes in working directory)
git reset --soft HEAD~1

# Roll back the last commit (discards changes completely)
git reset --hard HEAD~1

# Roll back to a specific commit
git reset --hard <commit-hash>

# If you've already pushed, create a new commit that undoes changes
git revert <commit-hash>
```

### Rolling Back Commits: VS Code

1. Open Source Control panel: `Ctrl+Shift+G`
2. Click on "..." menu → "View History"
3. Find the commit you want to roll back to
4. Right-click on that commit → "Reset to this commit"
5. Choose reset type:
   - **Soft**: Keeps changes staged
   - **Mixed**: Keeps changes in working directory
   - **Hard**: Discards all changes
6. ⚠️ **WARNING**: Only do this if you haven't pushed to GitHub!

### Rolling Back Commits: GitHub

⚠️ **GitHub cannot roll back commits directly!** If you've already pushed:

1. Go to your repository on GitHub
2. Click on the commit you want to undo
3. Click "Revert" button
4. This creates a new commit that undoes the changes
5. Or use local tools (CLI or VS Code) to roll back, then force push:
   - `git push --force-with-lease origin main` # Safer than `--force` - checks if remote was updated by others
   - ⚠️ **DANGER**: This rewrites history and can break other people's work!

## Step 11: Resolving Merge Conflicts - Three Ways

### Resolving Merge Conflicts: Command Line

```bash
# Create a conflict scenario
git checkout -b feature-conflict
echo "Version A" > conflict_file.txt
git add conflict_file.txt
git commit -m "Add conflict file version A"

git checkout main
echo "Version B" > conflict_file.txt
git add conflict_file.txt
git commit -m "Add conflict file version B"

# Try to merge (this will create a conflict)
git merge feature-conflict

# Resolve conflict manually
# Edit conflict_file.txt to remove markers and keep desired content
git add conflict_file.txt
git commit
```

### Resolving Merge Conflicts: VS Code

1. Create conflict scenario:
   - Open integrated terminal: `Ctrl+`` (backtick)
   - Run: `git checkout -b feature-conflict`
   - Create file: `echo "Version A" > conflict_file.txt`
   - Stage and commit: `git add conflict_file.txt && git commit -m "Add conflict file version A"`
   - Switch to main: `git checkout main`
   - Create conflicting version: `echo "Version B" > conflict_file.txt`
   - Stage and commit: `git add conflict_file.txt && git commit -m "Add conflict file version B"`
   - Try to merge: `git merge feature-conflict`
2. Open conflict_file.txt in VS Code
3. VS Code shows conflict markers with "Accept Current Change" / "Accept Incoming Change" buttons
   - **Accept Current Change**: Keeps the version from your local/active branch (main)
   - **Accept Incoming Change**: Keeps the version from the branch being merged in (feature-conflict)
4. Click the buttons or manually edit to resolve
5. Save the changes once you've reviewed and accepted a version of each change
6. Stage the resolved file in Source Control panel
7. Commit the merge

### Resolving Merge Conflicts: GitHub

1. Create conflict scenario:
   - Go to your repository on GitHub
   - Click "main" branch dropdown → "Create branch: feature-conflict"
   - Click "Add file" → "Create new file"
   - Name: `conflict_file.txt`, Content: `Version A`
   - Commit message: "Add conflict file version A"
   - Click "Commit new file"
   - Switch back to main branch
   - Edit `conflict_file.txt` → Change content to `Version B`
   - Commit message: "Add conflict file version B"
   - Click "Commit changes"
2. Create a pull request:
   - Click "Compare & pull request" (if you see the banner)
   - Or go to "Pull requests" tab → "New pull request"
   - Select branches: `feature-conflict` → `main`
   - Add title: "Test merge conflict"
   - Click "Create pull request"
3. GitHub will show "This branch has conflicts that must be resolved"
4. Click "Resolve conflicts" button
5. Edit files in GitHub's conflict resolution interface
6. Mark as resolved and merge the pull request

## Step 12: Markdown Documentation

Create a comprehensive README.md:

```markdown
# DataSci 217 - Week 02 Git Workflow Demo

This repository demonstrates **Git version control** and **GitHub collaboration**.

## Project Structure
```

data_analysis_project/
├── README.md
├── scripts/
│   └── analysis.py
└── data/
    └── sample_data.csv

```

## Git Commands Demonstrated

| Command | Purpose | Example |
|---------|---------|---------|
| `git init` | Initialize repository | `git init` |
| `git add` | Stage changes | `git add README.md` |
| `git commit` | Save changes | `git commit -m "message"` |
| `git push` | Upload to GitHub | `git push origin main` |
| `git pull` | Download from GitHub | `git pull origin main` |
| `git clone` | Copy repository | `git clone URL` |
| `git branch` | List branches | `git branch` |
| `git checkout` | Switch branch | `git checkout -b feature` |
| `git merge` | Combine branches | `git merge feature` |

## Workflow Steps
1. **Initialize**: `git init`
2. **Add files**: `git add .`
3. **Commit**: `git commit -m "message"`
4. **Push**: `git push origin main`
5. **Branch**: `git checkout -b feature`
6. **Merge**: `git merge feature`
```

Stage and commit the documentation:

```bash
git add README.md
git commit -m "Add comprehensive Git workflow documentation"
git push origin main
```

# Demo 2: Command Line

## Step 1: Create Project Structure

```bash
# Create a new project directory
mkdir cli_demo_project
cd cli_demo_project

# Create complex directory structure
mkdir -p data/{raw,processed} scripts results logs

# Verify structure was created
tree
# If tree is not available: ls -la data/
```

## Step 2: Generate Sample Data

```bash
# Create sample CSV data
cat > data/raw/students.csv << 'EOF'
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

# Create sample log data
cat > logs/processing.log << 'EOF'
2023-12-01 10:00:00 - Processing started
2023-12-01 10:01:00 - Data validation complete
2023-12-01 10:02:00 - Analysis finished
2023-12-01 10:03:00 - Results saved
EOF

# Verify files were created
ls -la data/raw/
ls -la logs/
```

## Step 3: Essential Navigation Commands

```bash
# Show current location
pwd

# List files with details
ls -la

# Navigate to subdirectory
cd data/raw
pwd

# Go up two levels
cd ../..
pwd

# Go to home directory
cd ~
pwd

# Return to previous directory
cd -
pwd

# Navigate back to project
cd cli_demo_project
```

## Step 4: File Viewing Commands

```bash
# View entire file
cat data/raw/students.csv

# View first 3 lines
head -3 data/raw/students.csv

# View last 2 lines
tail -2 data/raw/students.csv

# View file with line numbers
cat -n logs/processing.log
```

## Step 5: Text Processing and Search

```bash
# Search for specific text (case insensitive)
grep -i "math" data/raw/students.csv

# Search for text in multiple files
grep -i "science" data/raw/*.csv

# Count lines in files
wc -l data/raw/students.csv

# Count words in files
wc -w logs/processing.log

# Count characters in files
wc -c data/raw/students.csv
```

## Step 6: Data Processing

```bash
# Create timestamp for file operations
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "Timestamp: $timestamp"

# Filter data with grep
grep "Math" data/raw/students.csv

# Recursive search in directories
grep -r "processing" logs/

# Search with line numbers
grep -n "Math" data/raw/students.csv
```

## Step 7: Command Chaining and Automation

```bash
# Pipe operations - count error lines
grep "error" logs/processing.log | wc -l

# Pipe operations - find and count
grep -i "math" data/raw/students.csv | wc -l

# Output redirection - save first 3 lines to file
head -3 data/raw/students.csv > results/sample_data.txt

# Output redirection - append to file
echo "Analysis completed at $(date)" >> results/analysis_log.txt

# Chain multiple commands
ls data/raw/*.csv | head -2 > results/file_list.txt
```

## Step 8: Create a Simple Shell Script

```bash
# Create a data processing script (you can also use VS Code, GitHub, `nano`, or whatever editor you like instead of `cat`)
cat > scripts/process_data.sh << 'EOF'
#!/bin/bash
# Simple data processing script

echo "Starting data processing..."
echo "Processing files in data/raw/"

# Count total lines
total_lines=$(wc -l < data/raw/students.csv)
echo "Total lines in students.csv: $total_lines"

# Count Math students
math_count=$(grep -i "math" data/raw/students.csv | wc -l)
echo "Math students: $math_count"

# Create summary report
echo "Data Processing Summary" > results/summary.txt
echo "Generated: $(date)" >> results/summary.txt
echo "Total students: $total_lines" >> results/summary.txt
echo "Math students: $math_count" >> results/summary.txt

echo "Processing complete. Check results/summary.txt"
EOF

# Make script executable
chmod +x scripts/process_data.sh

# Run the script
./scripts/process_data.sh
```

## Step 9: History Navigation and Shortcuts

```bash
# View command history
history

# Execute command by number (replace <number> with actual number)
# !<number>

# Repeat last command
!!

# Practice keyboard shortcuts:
# Up arrow - Previous command
# Down arrow - Next command
# Ctrl+A - Move to beginning of line
# Ctrl+E - Move to end of line
# Ctrl+K - Delete from cursor to end
# Ctrl+U - Delete from cursor to beginning
# Tab - Auto-complete commands/files
# Ctrl+R - Reverse search through history
# Ctrl+C - Cancel current command
```

## Step 10: File Operations with Wildcards

```bash
# List all CSV files
ls *.csv

# Copy all CSV files to processed directory
cp data/raw/*.csv data/processed/

# Create backup with timestamp
mkdir -p backups
cp data/raw/*.csv backups/backup_$(date +"%Y%m%d_%H%M%S")/

# List files by pattern
ls data/raw/*.csv
ls logs/*.log
```

## Step 11: Create Final Report

```bash
# Generate comprehensive report
# Note: using cat and $(command) in the shell will evaluate $(command)'s output 
# instead of writing "$(command)" as a literal string in the file output 
cat > results/final_report.txt << 'EOF'
CLI Demo Report
===============

Generated: $(date)

Directory Structure:
$(tree 2>/dev/null || find . -type d | head -10)

Files Created:
$(ls -la data/raw/)
$(ls -la logs/)

Data Analysis:
- Total students: $(wc -l < data/raw/students.csv)
- Math students: $(grep -i "math" data/raw/students.csv | wc -l)
- Science students: $(grep -i "science" data/raw/students.csv | wc -l)

Commands Used:
- Navigation: pwd, ls, cd
- File viewing: cat, head, tail
- Text processing: grep, wc
- File operations: cp, mkdir
- Scripting: chmod, variables, redirection
EOF

# View the final report
cat results/final_report.txt
```

# Demo 3: Python Functions and Modules

## Step 1: Create Project Structure

```bash
# Create Python demo directory
mkdir python_demo_project
cd python_demo_project

# Create directory structure
mkdir src data output

# Create sample data file
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
```

## Step 2: Start with Inline Script

Create `src/student_analysis.py`:

```python
#!/usr/bin/env python3
"""
Student Grade Analysis - Inline Script Version
This demonstrates basic Python concepts without functions
"""

# Create sample student data
students = [
    {"name": "Alice", "age": 20, "grade": 85, "subject": "Math"},
    {"name": "Bob", "age": 19, "grade": 92, "subject": "Science"},
    {"name": "Charlie", "age": 21, "grade": 78, "subject": "English"},
    {"name": "Diana", "age": 20, "grade": 88, "subject": "Math"},
    {"name": "Eve", "age": 22, "grade": 95, "subject": "Science"}
]

print("Student Grade Analysis")
print("=" * 30)

# Basic data processing with loops
total_grade = 0
for student in students:
    print(f"{student['name']}: {student['grade']}")
    total_grade += student['grade']

# Simple calculations
average_grade = total_grade / len(students)
print(f"\nAverage grade: {average_grade:.1f}")

# Find highest grade
highest_grade = max(student['grade'] for student in students)
print(f"Highest grade: {highest_grade}")

# File I/O operations
with open('output/analysis_results.txt', 'w') as file:
    file.write("Student Grade Analysis\n")
    file.write("=" * 30 + "\n\n")
    for student in students:
        file.write(f"{student['name']}: {student['grade']}\n")
    file.write(f"\nAverage grade: {average_grade:.1f}\n")
    file.write(f"Highest grade: {highest_grade}\n")

print("\nResults saved to output/analysis_results.txt")
```

Run the script:

```bash
python3 src/student_analysis.py
```

## Step 3: Refactor into Functions

Create `src/student_analysis_functions.py`:

```python
#!/usr/bin/env python3
"""
Student Grade Analysis - Functions Version
This demonstrates function extraction and organization
"""

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

# Sample data
students = [
    {"name": "Alice", "age": 20, "grade": 85, "subject": "Math"},
    {"name": "Bob", "age": 19, "grade": 92, "subject": "Science"},
    {"name": "Charlie", "age": 21, "grade": 78, "subject": "English"},
    {"name": "Diana", "age": 20, "grade": 88, "subject": "Math"},
    {"name": "Eve", "age": 22, "grade": 95, "subject": "Science"}
]

# Use the functions
print("Student grades:")
print_student_grades(students)

grades = get_grades_list(students)
average = safe_calculate_average(grades)
highest = find_highest_grade(grades)

print(f"\nAverage grade: {average:.1f}")
print(f"Highest grade: {highest}")

# Validate data
if validate_student_data(students):
    print("✓ Student data is valid")
else:
    print("✗ Student data has issues")
```

Run the functions version:

```bash
python3 src/student_analysis_functions.py
```

## Step 4: Add Modular Structure

Create `src/student_analysis_modular.py`:

```python
#!/usr/bin/env python3
"""
Student Grade Analysis - Modular Version
This demonstrates __main__ execution control and modular design
"""

import csv
from pathlib import Path

def load_student_data(csv_file):
    """Load student data from CSV file."""
    students = []
    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                students.append({
                    'name': row['name'],
                    'age': int(row['age']),
                    'grade': int(row['grade']),
                    'subject': row['subject']
                })
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found")
    except Exception as e:
        print(f"Error loading data: {e}")
    
    return students

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

def main():
    """Main function to run the analysis."""
    print("Student Grade Analysis - Modular Version")
    print("=" * 40)
    
    # Load data from CSV
    students = load_student_data('data/students.csv')
    
    if not students:
        print("No student data to analyze")
        return
    
    # Calculate statistics
    grades = [student['grade'] for student in students]
    average = calculate_average(grades)
    highest = find_highest_grade(grades)
    
    # Display results
    print(f"Analyzed {len(students)} students")
    print(f"Average grade: {average:.1f}")
    print(f"Highest grade: {highest}")
    
    # Save results
    output_file = 'output/modular_analysis.txt'
    Path('output').mkdir(exist_ok=True)
    
    if save_results_to_file(output_file, students, average, highest):
        print("✅ Analysis complete!")
    else:
        print("❌ Analysis failed")

# This code only runs when the script is executed directly
# It won't run when the script is imported as a module
if __name__ == "__main__":
    main()
```

Run the modular version:

```bash
python3 src/student_analysis_modular.py
```

## Step 5: Create Module and Import

Create `src/advanced_analysis.py`:

```python
#!/usr/bin/env python3
"""
Advanced Student Analysis - Module Usage
This demonstrates importing functions from other scripts
"""

# Import functions from the modular script
from student_analysis_modular import (
    load_student_data,
    calculate_average,
    find_highest_grade,
    save_results_to_file
)

def analyze_grade_distribution(grades):
    """Analyze the distribution of grades."""
    if not grades:
        return {}
    
    # Count grades by ranges
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

def find_top_performers(students, threshold=90):
    """Find students who scored above a threshold."""
    top_performers = []
    for student in students:
        if student['grade'] >= threshold:
            top_performers.append(student)
    return top_performers

def generate_detailed_report(students, filename):
    """Generate a comprehensive analysis report."""
    if not students:
        print("No data to analyze")
        return False
    
    # Get grades and basic statistics
    grades = [student['grade'] for student in students]
    average = calculate_average(grades)
    highest = find_highest_grade(grades)
    lowest = min(grades) if grades else 0
    
    # Analyze grade distribution
    distribution = analyze_grade_distribution(grades)
    
    # Find top performers
    top_performers = find_top_performers(students, 90)
    
    # Generate report
    try:
        with open(filename, 'w') as file:
            file.write("COMPREHENSIVE STUDENT ANALYSIS REPORT\n")
            file.write("=" * 50 + "\n\n")
            
            file.write(f"Report generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            file.write("BASIC STATISTICS\n")
            file.write("-" * 20 + "\n")
            file.write(f"Total students: {len(students)}\n")
            file.write(f"Average grade: {average:.1f}\n")
            file.write(f"Highest grade: {highest}\n")
            file.write(f"Lowest grade: {lowest}\n")
            file.write(f"Grade range: {highest - lowest}\n\n")
            
            file.write("GRADE DISTRIBUTION\n")
            file.write("-" * 20 + "\n")
            for grade_range, count in distribution.items():
                percentage = (count / len(students)) * 100 if students else 0
                file.write(f"{grade_range}: {count} students ({percentage:.1f}%)\n")
            file.write("\n")
            
            file.write("TOP PERFORMERS (90+)\n")
            file.write("-" * 20 + "\n")
            if top_performers:
                for student in top_performers:
                    file.write(f"{student['name']}: {student['grade']} ({student['subject']})\n")
            else:
                file.write("No students scored 90 or above\n")
            file.write("\n")
            
            file.write("INDIVIDUAL STUDENT RECORDS\n")
            file.write("-" * 30 + "\n")
            for student in students:
                file.write(f"Name: {student['name']}\n")
                file.write(f"  Age: {student['age']}\n")
                file.write(f"  Grade: {student['grade']}\n")
                file.write(f"  Subject: {student['subject']}\n")
                file.write("\n")
        
        print(f"Detailed report saved to {filename}")
        return True
    except Exception as e:
        print(f"Error generating report: {e}")
        return False

def main():
    """Main function demonstrating module usage."""
    print("Advanced Student Analysis - Module Usage")
    print("=" * 45)
    
    # Load data using imported function
    students = load_student_data('data/students.csv')
    
    if not students:
        print("No data loaded. Please check data/students.csv")
        return
    
    print(f"Loaded {len(students)} students")
    
    # Use imported functions
    grades = [student['grade'] for student in students]
    average = calculate_average(grades)
    highest = find_highest_grade(grades)
    
    print(f"Average grade: {average:.1f}")
    print(f"Highest grade: {highest}")
    
    # Advanced analysis using new functions
    distribution = analyze_grade_distribution(grades)
    print("\nGrade Distribution:")
    for grade_range, count in distribution.items():
        percentage = (count / len(students)) * 100
        print(f"{grade_range}: {count} students ({percentage:.1f}%)")
    
    # Top performers
    top_performers = find_top_performers(students, 90)
    print(f"\nTop Performers (90+): {len(top_performers)} students")
    for student in top_performers:
        print(f"  {student['name']}: {student['grade']} ({student['subject']})")
    
    # Generate comprehensive report
    generate_detailed_report(students, 'output/advanced_analysis.txt')
    
    # Save basic results using imported function
    save_results_to_file('output/module_analysis.txt', students, average, highest)
    
    print("\n✅ Advanced analysis complete!")

if __name__ == "__main__":
    main()
```

Run the advanced analysis:

```bash
python3 src/advanced_analysis.py
```

## Step 6: Verify All Output Files

```bash
# Check what files were created
ls -la output/

# View the different analysis results
echo "=== Basic Analysis ==="
cat output/analysis_results.txt

echo -e "\n=== Modular Analysis ==="
cat output/modular_analysis.txt

echo -e "\n=== Advanced Analysis ==="
cat output/advanced_analysis.txt
```
