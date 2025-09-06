# L01: Command Line Foundations + Python Setup
**Duration**: 3 hours  
**Learning Objectives**: Master command line navigation, establish professional Python development environment, implement basic scripting workflows, and integrate version control practices

---

## Opening: Why the Command Line Changes Everything

Picture this: You're working late on a critical data analysis project. Your GUI-based file manager crashes, your IDE becomes unresponsive, and you need to process 50 CSV files scattered across different directories. What separates a junior analyst from a data science professional in this moment? **Command line mastery.**

The command line isn't just a relic from computing's past—it's the power tool that transforms how you interact with data, automate workflows, and collaborate with teams. While others click through endless menus, you'll execute complex operations with single commands. While others manually copy files one by one, you'll process entire datasets with elegant scripts.

Today, we're not just learning commands—we're building the foundational skills that will make you 10x more efficient throughout your data science career. Every professional data scientist, regardless of their preferred tools, relies on command line competency for deployment, server management, batch processing, and automation.

## Chapter 1: Navigation Mastery - Your Digital Compass

### The Mental Model: Files as a Tree Structure

Before diving into commands, let's establish the fundamental mental model. Your computer's file system isn't a collection of folders—it's a hierarchical tree structure where every file has an exact address. Understanding this concept transforms navigation from memorizing commands to logical pathfinding.

```
/
├── home/
│   ├── username/
│   │   ├── Documents/
│   │   ├── Downloads/
│   │   └── projects/
│   │       └── datasci_217/
└── usr/
    ├── bin/
    └── lib/
```

### Essential Navigation Commands in Action

**pwd (Print Working Directory)**: Your GPS coordinate system
```bash
pwd
# Output: /home/username/projects/datasci_217
```

This isn't just showing you where you are—it's showing you your exact position in the file system tree. Every command you execute happens relative to this location unless you specify otherwise.

**ls (List)**: Your directory scanner
```bash
ls -la
```

The `-l` flag gives detailed information (permissions, size, date modified), while `-a` reveals hidden files (those starting with dots). This combination provides the complete picture of your current environment—essential for understanding file structures and debugging permissions issues.

**cd (Change Directory)**: Your teleportation device
```bash
# Absolute path (starts from root)
cd /home/username/projects/datasci_217

# Relative path (from current location)
cd ../parent_directory
cd ./subdirectory

# Special shortcuts
cd ~        # Home directory
cd -        # Previous directory
cd          # Also goes to home
```

**File Management Power Moves**:
```bash
# Create directory structure in one command
mkdir -p projects/datasci_217/assignments/week1

# Copy with verification and verbose output
cp -v source_file.py backup_file.py

# Move (rename) with confirmation for overwrites
mv -i old_name.txt new_name.txt

# Remove with interactive confirmation (safety first)
rm -i unwanted_file.txt
```

### Practical Exercise 1: Command Line Treasure Hunt

**Objective**: Navigate the file system and locate specific files using only command line tools.

**Setup Phase**:
1. Create a complex directory structure with hidden files and nested folders
2. Plant "treasure" files containing clues to the next location
3. Use commands like `find`, `grep`, and `cat` to solve puzzles

**Challenge Structure**:
- Start in your home directory
- Follow clues through different system locations
- Use `find . -name "*.txt" -type f` to locate files
- Use `grep -r "treasure_keyword" .` to search file contents
- Combine commands with pipes: `ls -la | grep "Feb"`

This exercise builds real-world skills—the same techniques you'll use to locate datasets, find configuration files, and debug issues in complex project structures.

## Chapter 2: Python Environment Setup - Building Your Professional Toolkit

### The Environment Problem Every Data Scientist Faces

You've probably experienced this nightmare: A Python script works perfectly on your machine but fails spectacularly on a colleague's computer. Different Python versions, conflicting package versions, missing dependencies—environment issues are the #1 cause of "it works on my machine" problems in data science.

Professional data scientists solve this with **virtual environments**—isolated Python installations for each project. This isn't optional nice-to-have knowledge; it's essential professional practice.

### Python Installation: Setting the Foundation

**For macOS**:
```bash
# Install Homebrew (the missing package manager)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python through Homebrew for better management
brew install python@3.11
```

**For Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**For Windows**:
Download from python.org, but ensure you check "Add Python to PATH" during installation.

### Virtual Environment Mastery

**Creating and Managing Virtual Environments**:
```bash
# Create a new virtual environment
python3 -m venv datasci_env

# Activate the environment (Linux/macOS)
source datasci_env/bin/activate

# Activate the environment (Windows)
datasci_env\Scripts\activate

# Verify activation (your prompt should change)
which python
python --version

# Install packages in isolated environment
pip install pandas numpy matplotlib jupyter

# Generate requirements file (reproducible environments)
pip freeze > requirements.txt

# Deactivate when done
deactivate
```

### Package Management Best Practices

**The Requirements File Workflow**:
```bash
# Create comprehensive requirements.txt
echo "pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
jupyter>=1.0.0
seaborn>=0.11.0
scikit-learn>=1.1.0" > requirements.txt

# Install all dependencies at once
pip install -r requirements.txt

# Update requirements after adding packages
pip freeze > requirements.txt
```

### Practical Exercise 2: Environment Setup Validation

**Objective**: Create a professional Python development environment and validate it works correctly.

**Tasks**:
1. Create a new virtual environment named `datasci_217`
2. Install the data science stack (pandas, numpy, matplotlib, jupyter)
3. Create a validation script that imports all packages and prints version numbers
4. Generate a requirements.txt file
5. Test environment reproducibility by creating a second environment from requirements.txt

**Validation Script Example**:
```python
import sys
import pandas as pd
import numpy as np
import matplotlib
import jupyter

print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print("Environment setup successful!")
```

## Chapter 3: Basic Python Programming - Building Your Data Science Foundation

### Python for Data Science: Beyond Basic Syntax

While you might know Python basics, data science requires specific patterns and practices. We're not just writing code—we're building analytical workflows that need to be readable, maintainable, and scalable.

### Data Types in Data Science Context

**Strings for Text Processing**:
```python
# Real-world example: cleaning column names
dirty_column = "  Customer Name (Primary)  "
clean_column = dirty_column.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
print(clean_column)  # customer_name_primary
```

**Numbers for Calculations**:
```python
# Financial calculations require precision
from decimal import Decimal

# Avoid floating point errors in financial data
price = Decimal('19.99')
quantity = Decimal('3')
total = price * quantity
print(f"Total: ${total}")  # Total: $59.97
```

**Booleans for Filtering**:
```python
# Data filtering patterns you'll use constantly
ages = [25, 30, 35, 40, 45]
is_millennial = [age >= 25 and age <= 40 for age in ages]
millennial_ages = [age for age, is_mill in zip(ages, is_millennial) if is_mill]
print(millennial_ages)  # [25, 30, 35, 40]
```

### Control Flow for Data Processing

**Conditional Logic for Data Cleaning**:
```python
def categorize_temperature(temp):
    """Categorize temperature readings for analysis."""
    if temp < 0:
        return "freezing"
    elif temp < 32:
        return "cold"
    elif temp < 70:
        return "moderate"
    elif temp < 90:
        return "warm"
    else:
        return "hot"

# Process a list of temperature readings
temperatures = [-5, 25, 65, 85, 95]
categories = [categorize_temperature(temp) for temp in temperatures]
print(dict(zip(temperatures, categories)))
```

**Loops for Data Processing**:
```python
# Reading multiple CSV files (common data science pattern)
import glob

data_files = glob.glob("data/*.csv")
datasets = {}

for file_path in data_files:
    # Extract filename without extension for dictionary key
    filename = file_path.split("/")[-1].replace(".csv", "")
    
    # Store for later processing
    datasets[filename] = file_path
    print(f"Found dataset: {filename}")
```

### Functions: The Building Blocks of Analytical Workflows

**Writing Reusable Analysis Functions**:
```python
def calculate_summary_stats(data_list):
    """
    Calculate comprehensive summary statistics.
    
    Args:
        data_list (list): Numeric data for analysis
        
    Returns:
        dict: Summary statistics including mean, median, std dev
    """
    import statistics
    
    if not data_list:
        return {"error": "Empty dataset"}
    
    try:
        stats = {
            "count": len(data_list),
            "mean": statistics.mean(data_list),
            "median": statistics.median(data_list),
            "std_dev": statistics.stdev(data_list) if len(data_list) > 1 else 0,
            "min": min(data_list),
            "max": max(data_list)
        }
        return stats
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}

# Usage example
sales_data = [120, 135, 142, 156, 143, 167, 178, 165, 189, 201]
results = calculate_summary_stats(sales_data)
print(f"Sales Analysis: {results}")
```

### Practical Exercise 3: First Python Script with Professional Structure

**Objective**: Create a complete Python script that demonstrates professional coding practices and data analysis workflow.

**Script Requirements**:
```python
#!/usr/bin/env python3
"""
Data Analysis Starter Script
Author: [Your Name]
Date: [Current Date]
Purpose: Demonstrate basic data analysis workflow with professional Python practices
"""

def main():
    """Main execution function."""
    print("Starting data analysis workflow...")
    
    # Your implementation here
    pass

if __name__ == "__main__":
    main()
```

**Tasks**:
1. Create a script that reads data from a CSV file (create sample data if needed)
2. Implement functions for data cleaning, analysis, and reporting
3. Include proper error handling and user feedback
4. Add comprehensive docstrings and comments
5. Follow PEP 8 style guidelines

## Chapter 4: Version Control Workflow - Professional Collaboration Foundation

### Git: The Time Machine for Code

Version control isn't just about backing up code—it's about enabling collaboration, tracking changes, experimenting safely, and maintaining project history. In data science, where experiments can take hours to run and results need to be reproducible, version control becomes absolutely critical.

### Git Fundamentals in Data Science Context

**Repository Initialization**:
```bash
# Initialize a new repository
git init datasci_project

# Configure your identity (critical for collaboration)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Check configuration
git config --list
```

**The Basic Workflow Pattern**:
```bash
# Check status (use this constantly)
git status

# Add specific files (selective staging)
git add analysis_script.py
git add data_cleaning.py

# Add all changes (use carefully)
git add .

# Commit with meaningful message
git commit -m "Implement initial data cleaning pipeline

- Add function to handle missing values
- Implement outlier detection algorithm
- Create data validation checks"

# View history
git log --oneline
```

### Professional Commit Message Practices

**Bad Commit Messages**:
- "fix bug"
- "update code"
- "working version"

**Good Commit Messages**:
- "Fix pandas groupby aggregation error in sales analysis"
- "Add automated data validation for customer dataset"
- "Implement feature engineering pipeline for ML model"

### Remote Repository Integration

**GitHub Workflow**:
```bash
# Connect local repository to GitHub
git remote add origin https://github.com/username/datasci_project.git

# Push to remote repository
git push -u origin main

# Clone existing repository
git clone https://github.com/username/datasci_project.git

# Sync with remote changes
git pull origin main
```

### Branching for Experimentation

```bash
# Create and switch to new branch for feature development
git checkout -b feature/add-visualization

# Work on your feature...
# ... make changes and commits ...

# Switch back to main branch
git checkout main

# Merge feature branch
git merge feature/add-visualization

# Delete merged branch
git branch -d feature/add-visualization
```

## Practical Integration: Bringing It All Together

### Real-World Scenario: Setting Up a Data Science Project

Let's walk through setting up a complete data science project that demonstrates all the skills we've covered:

```bash
# 1. Create project structure using command line
mkdir -p datasci_project/{data,scripts,notebooks,results,tests}
cd datasci_project

# 2. Initialize Git repository
git init
echo "*.pyc
__pycache__/
.DS_Store
data/*.csv
.env" > .gitignore

# 3. Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install data science packages
pip install pandas numpy matplotlib jupyter seaborn
pip freeze > requirements.txt

# 5. Create initial project files
echo "# Data Science Project
This project analyzes [dataset description].

## Setup
1. Create virtual environment: \`python3 -m venv .venv\`
2. Activate environment: \`source .venv/bin/activate\`
3. Install dependencies: \`pip install -r requirements.txt\`

## Usage
Run analysis with: \`python scripts/analyze_data.py\`
" > README.md

# 6. Initial commit
git add .
git commit -m "Initial project setup

- Add project structure with standard directories
- Configure Python virtual environment
- Add requirements.txt with data science stack
- Create README with setup instructions"
```

### Assignment Sketch: Professional Development Environment

**Assignment Overview**: Students create a complete professional development environment and demonstrate proficiency with all covered tools.

**Deliverables**:
1. **Command Line Proficiency**: Navigate complex directory structures, use advanced file operations, and demonstrate efficiency with keyboard shortcuts
2. **Python Environment**: Create isolated environments, manage dependencies, and validate installations
3. **Git Workflow**: Initialize repositories, make meaningful commits, and sync with remote repositories
4. **Integration Script**: Write a Python script that combines all learned concepts

**Assessment Criteria**:
- Command line efficiency and proper usage
- Virtual environment setup and dependency management
- Git workflow with meaningful commit messages
- Code quality and professional practices
- Documentation and project structure

**Bonus Challenges**:
- Automate environment setup with shell scripts
- Implement pre-commit hooks for code quality
- Create GitHub Actions workflow for continuous integration

## Looking Forward: Skills That Scale

Today's foundations—command line navigation, Python environments, version control—aren't just class requirements. They're the bedrock skills that separate data science hobbyists from professionals. Every advanced topic we'll cover this semester builds on these fundamentals.

When you're deploying machine learning models to cloud servers, you'll use command line skills. When you're collaborating on complex analysis projects, you'll rely on Git workflows. When you're reproducing research results, you'll need proper environment management.

Master these foundations now, and every future challenge becomes more manageable. Skip them, and you'll constantly fight your tools instead of focusing on insights and analysis.

---

**Next Session Preview**: In L02, we'll build on these foundations with advanced data structures, collaborative Git workflows, and professional code organization patterns that will serve you throughout your data science career.