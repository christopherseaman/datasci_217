# Assignment 02: Git Workflow and Project Setup

**Due:** Before next class  
**Points:** 15 points total  
**Submit:** Via GitHub repository (link submitted to Canvas)

## Overview

Practice the complete Git workflow using VS Code and GitHub. You'll create a personal data science project repository, practice version control operations, and set up a proper Python project structure.

## Learning Objectives

By completing this assignment, you will:
- Create and manage repositories on GitHub
- Use VS Code's Git integration effectively  
- Practice staging, committing, and pushing changes
- Set up proper project structure with .gitignore
- Write descriptive commit messages
- Create professional documentation

## Requirements

### Part 1: Repository Creation (3 points)

1. **Create GitHub Repository**
   - Name: `datasci-week02-practice`
   - Description: "Practice repository for DataSci 217 - Week 02 Git workflow"
   - Public repository (showcase your work!)
   - Initialize with README

2. **Clone Using VS Code**
   - Use VS Code's "Clone Repository" command
   - Choose appropriate location on your computer
   - Open the cloned repository in VS Code

### Part 2: Project Structure (4 points)

Create the following files and directories in your repository:

```
datasci-week02-practice/
├── README.md (already exists, you'll edit this)
├── .gitignore
├── requirements.txt
├── src/
│   └── data_analysis.py
└── data/
    └── README.md
```

**File Contents:**

1. **Update README.md** (replace default content):
```markdown
# DataSci 217 - Week 02 Practice

This repository demonstrates Git workflow and Python project setup for DataSci 217.

## Project Structure
- `src/` - Python source code
- `data/` - Data files (not tracked by Git)
- `requirements.txt` - Python dependencies

## Setup
1. Clone this repository
2. Create virtual environment: `conda create -n week02 python=3.11`
3. Activate environment: `conda activate week02`
4. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the analysis: `python src/data_analysis.py`

## Author
[Your Name] - [Your UCSF Email]
```

2. **Create .gitignore**:
```
# Virtual environments
venv/
env/
week02/

# Data files (often too large for Git)
*.csv
*.xlsx
data/*.csv
data/*.xlsx

# Python generated files
__pycache__/
*.pyc
*.pyo

# Jupyter notebook outputs
.ipynb_checkpoints/
*.ipynb

# Operating system files
.DS_Store
Thumbs.db
.vscode/settings.json

# IDE files
.idea/
*.swp
*.swo
```

3. **Create requirements.txt**:
```
# Basic data science packages
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0

# For future assignments
jupyter>=1.0.0
seaborn>=0.12.0
```

4. **Create src/data_analysis.py**:
```python
#!/usr/bin/env python3
"""
Basic data analysis script for DataSci 217 - Week 02
Demonstrates Python basics and Git workflow
"""

def calculate_statistics(numbers):
    """Calculate basic statistics for a list of numbers"""
    if not numbers:
        return None
    
    total = sum(numbers)
    count = len(numbers)
    mean = total / count
    
    sorted_nums = sorted(numbers)
    if count % 2 == 0:
        median = (sorted_nums[count//2 - 1] + sorted_nums[count//2]) / 2
    else:
        median = sorted_nums[count//2]
    
    return {
        'count': count,
        'sum': total,
        'mean': mean,
        'median': median,
        'min': min(numbers),
        'max': max(numbers)
    }

def main():
    """Main function to demonstrate the analysis"""
    print("DataSci 217 - Week 02 Practice")
    print("=" * 35)
    
    # Sample data for analysis
    sample_data = [23, 45, 67, 12, 89, 34, 56, 78, 90, 45, 67, 23]
    
    print(f"Sample data: {sample_data}")
    print()
    
    # Calculate statistics
    stats = calculate_statistics(sample_data)
    
    if stats:
        print("Statistical Summary:")
        print(f"Count: {stats['count']}")
        print(f"Sum: {stats['sum']}")
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Median: {stats['median']:.2f}")
        print(f"Min: {stats['min']}")
        print(f"Max: {stats['max']}")
    
    print("\n✅ Analysis complete! Git workflow practice successful.")

if __name__ == "__main__":
    main()
```

5. **Create data/README.md**:
```markdown
# Data Directory

This directory is for storing data files used in analysis.

## Important Notes
- Data files (*.csv, *.xlsx) are ignored by Git (see .gitignore)
- Always document your data sources
- Include metadata about datasets when possible

## Current Datasets
(None yet - add descriptions when you add data files)
```

### Part 3: Git Workflow Practice (5 points)

Using **only VS Code's Git interface** (no command line), complete these steps:

1. **Initial Commit Setup**
   - Create all files listed above
   - Stage all files using VS Code Source Control
   - Commit with message: "Initial project structure and documentation"
   - Push to GitHub

2. **Feature Development Workflow**
   - Modify `src/data_analysis.py` to add a new function:
   ```python
   def analyze_range(numbers):
       """Calculate the range of numbers"""
       if not numbers:
           return 0
       return max(numbers) - min(numbers)
   ```
   - Update the `main()` function to use this new function
   - Stage and commit with message: "Add range calculation function"
   - Push to GitHub

3. **Documentation Update**
   - Update README.md to include a "Features" section listing what the script does
   - Stage and commit with message: "Update documentation with features list"
   - Push to GitHub

4. **Project Refinement**
   - Add comments to your Python code (at least 2 additional comments)
   - Update requirements.txt to add `scipy>=1.10.0`
   - Stage and commit with message: "Improve code documentation and dependencies"
   - Push to GitHub

### Part 4: Environment Setup (3 points)

1. **Create Virtual Environment**
   - Create conda environment: `conda create -n week02 python=3.11`
   - Document the process in a file called `SETUP.md`

2. **Test Installation**
   - Activate your environment
   - Install packages from requirements.txt
   - Run your Python script to verify it works
   - Take a screenshot of successful execution

3. **Create SETUP.md**:
```markdown
# Environment Setup Instructions

## Creating the Environment
```bash
# Create virtual environment
conda create -n week02 python=3.11

# Activate environment
conda activate week02

# Install dependencies
pip install -r requirements.txt
```

## Testing the Setup
```bash
# Run the analysis script
python src/data_analysis.py
```

## Expected Output
The script should display sample data analysis including count, sum, mean, median, min, max, and range.

## Troubleshooting
- If packages fail to install, try updating conda: `conda update conda`
- If Python script fails, check that you're in the correct directory
- Ensure virtual environment is activated before running
```

## Submission Requirements

1. **Repository Link**
   - Submit your GitHub repository URL via Canvas
   - Repository must be public so we can review it

2. **Commit History**
   - At least 4 commits with descriptive messages
   - All commits should be pushed to GitHub
   - Commits should follow the workflow described above

3. **File Structure**
   - All required files present and properly structured
   - .gitignore working correctly (no unnecessary files tracked)
   - Python script runs without errors

4. **Documentation Quality**
   - README.md is professional and informative
   - Code includes appropriate comments
   - Setup instructions are clear

## Grading Rubric

- **Repository Setup (3 pts):** Proper GitHub repo creation and cloning
- **File Structure (4 pts):** All required files with correct content
- **Git Workflow (5 pts):** Proper use of VS Code Git interface, good commit messages
- **Environment Setup (3 pts):** Virtual environment documented and tested

## Common Issues to Avoid

1. **Poor Commit Messages:** "update" or "fix" - be descriptive!
2. **Missing .gitignore:** Don't track unnecessary files
3. **Broken Code:** Test your Python script before submitting
4. **Empty Directories:** Include README files in empty directories
5. **Private Repository:** Make sure repository is public for grading

## Getting Help

- **Office Hours:** [Your office hours]
- **Discord:** #week02-git-help channel
- **Classmates:** Collaboration on concepts is encouraged, but submit your own work

## Due Date

Submit repository link via Canvas by [date] at 11:59 PM.

**Late Policy:** -10% per day late, up to 3 days. After 3 days, assignment cannot be accepted.