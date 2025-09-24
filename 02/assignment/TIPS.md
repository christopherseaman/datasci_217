# Assignment 02: Tips and Troubleshooting Guide

## Quick Start Checklist

1. Create a new repository named `datasci-week02-integration`
2. Create your branches BEFORE adding files
3. Develop features on separate branches
4. Test your scripts before committing
5. Merge branches back to main when complete

## Common Issues and Solutions

### Git Workflow Issues

**Issue: "fatal: not a git repository"**
- Solution: Run `git init` in your project directory first

**Issue: "fatal: branch already exists"**
- Solution: Switch to the branch with `git checkout branch-name` instead of creating it

**Issue: Merge conflicts**
- Solution: Edit conflicted files manually, then `git add` and `git commit`

**Issue: Forgot to create branches before starting work**
- Solution: Create branch now with `git checkout -b feature/branch-name`, your changes will move with you

### Shell Script Issues

**Issue: "Permission denied" when running setup_project.sh**
- Solution: Make it executable first: `chmod +x setup_project.sh`

**Issue: Here-documents not working correctly**
- Solution: Use single quotes around EOF to prevent variable expansion:
  ```bash
  cat > file.txt << 'EOF'
  content here
  EOF
  ```

**Issue: Script creates files in wrong location**
- Solution: Run the script from your repository root directory

**Issue: Directories already exist error**
- Solution: Use `mkdir -p` which won't error if directory exists

### Python Issues

**Issue: CSV not parsing correctly**
- Solution: Remember to strip newlines: `line.strip().split(',')`

**Issue: Header line included in data**
- Solution: Skip first line with `lines[1:]` after `readlines()`

**Issue: "FileNotFoundError"**
- Solution: Check you're in the right directory and file paths are correct

**Issue: Output directory doesn't exist**
- Solution: Create it first:
  ```python
  import os
  if not os.path.exists('output'):
      os.makedirs('output')
  ```

**Issue: Decimal formatting not working**
- Solution: Use f-strings with format specifier: `f"{value:.1f}"`

## Testing Your Work

### Test Git Workflow
```bash
# Check your branch structure
git branch -a

# Check your commit history
git log --oneline --graph --all

# Verify merges completed
git log --merges
```

### Test Shell Script
```bash
# Make executable
chmod +x setup_project.sh

# Run it
./setup_project.sh

# Verify structure created
ls -la src/ data/ output/

# Check file contents
cat data/students.csv
```

### Test Python Scripts
```bash
# Test basic analysis
python src/data_analysis.py

# Test advanced analysis
python src/data_analysis_functions.py

# Check output was created
cat output/analysis_report.txt
```

### Run the Tests
```bash
# Run the automated tests
pytest .github/test/test_assignment.py -v
```
This will check if your implementation meets all requirements.

## File Path Reference

Your final structure should look like:
```
datasci-week02-integration/
├── .git/                     (Git repository)
├── .gitignore               (Created by script)
├── README.md                (Your documentation)
├── requirements.txt         (Created by script)
├── setup_project.sh         (Your shell script)
├── main.py                  (Validation script - provided)
├── src/
│   ├── data_analysis.py    (Your Python script)
│   └── data_analysis_functions.py (Your Python script)
├── data/
│   └── students.csv         (Created by script)
└── output/
    └── analysis_report.txt  (Created by Python scripts)
```

## Hints Without Spoilers

### For Shell Script
- Start simple: create directories first, then files
- Test each part as you add it
- Use `echo` statements to show progress
- Remember the shebang: `#!/bin/bash`

### For Python Scripts
- Read the CSV file completely before processing
- Store data in a list of dictionaries or lists
- Calculate statistics step by step
- Test with print statements before writing to file

### For Git Workflow
- Commit frequently with clear messages
- Use `git status` to check what will be committed
- Create branches from main, not from other branches
- Pull changes to main before creating new branches

## When Stuck

1. **Read the error message carefully** - It usually tells you what's wrong
2. **Check you're in the right directory** - Use `pwd` to verify
3. **Verify file paths** - Use `ls` to check files exist
4. **Test incrementally** - Don't write everything at once
5. **Use print debugging** - Add print statements to see what's happening

## Remember

- The assignment tests **competence**, not expertise
- Start with Part 1 (Git), it creates the foundation
- The shell script in Part 2 creates files for Part 3
- Each part builds on the previous one
- Use the techniques from the lecture - they're all you need!

## Emergency Scaffold (Only if Really Stuck)

If you're completely stuck and need a starting point, here are minimal scaffolds. Try to solve it yourself first!

### Shell Script Starter (setup_project.sh)
```bash
#!/bin/bash
# Project setup script

echo "Setting up project..."

# TODO: Create directories (hint: mkdir -p)

# TODO: Create .gitignore file (hint: use cat > with EOF)

# TODO: Create students.csv with data (hint: here-document)

# TODO: Create Python template files with TODO comments

echo "Setup complete!"
```

### Python Basic Script Starter (src/data_analysis.py)
```python
#!/usr/bin/env python3
"""Basic Data Analysis Script"""

def load_students(filename):
    """Load student data from CSV."""
    # TODO: Open file, read lines, skip header
    # TODO: Split each line by comma
    # TODO: Return list of student data
    pass

def calculate_average_grade(students):
    """Calculate average grade."""
    # TODO: Sum all grades
    # TODO: Divide by number of students
    pass

def count_math_students(students):
    """Count students in Math."""
    # TODO: Count students where subject is Math
    pass

def generate_report(total, average, math_count):
    """Generate report string."""
    # TODO: Create formatted string with results
    # TODO: Use f-strings with .1f for decimals
    pass

def save_report(report, filename):
    """Save report to file."""
    # TODO: Create output directory if needed
    # TODO: Write report to file
    pass

def main():
    # TODO: Load data
    # TODO: Calculate statistics
    # TODO: Generate and save report
    pass

if __name__ == "__main__":
    main()
```

### Python Advanced Script Starter (src/data_analysis_functions.py)
```python
#!/usr/bin/env python3
"""Advanced Data Analysis with Functions"""

def load_data(filename):
    """Load data from CSV file."""
    # TODO: Check file extension
    # TODO: Call appropriate loader
    pass

def load_csv(filename):
    """Load CSV data."""
    # TODO: Same technique as basic script
    pass

def analyze_data(students):
    """Analyze student data."""
    # TODO: Calculate multiple statistics
    # TODO: Return dictionary of results
    pass

def analyze_grade_distribution(grades):
    """Count grades by letter grade."""
    # TODO: Count A (90-100), B (80-89), etc.
    pass

def save_results(results, filename):
    """Save detailed results."""
    # TODO: Format and write comprehensive report
    pass

def main():
    # TODO: Orchestrate the analysis
    pass

if __name__ == "__main__":
    main()
```

**Note**: These scaffolds are intentionally incomplete. Your job is to implement the TODOs using techniques from the lecture. Don't just copy these - understand what each part needs to do!