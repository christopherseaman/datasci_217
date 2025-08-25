# Lecture 01: Python Fundamentals & Command Line Mastery

*Building the Foundation for Data Science*

## Learning Objectives

By the end of this lecture, you will be able to:
- Navigate and manipulate files using command line interfaces
- Set up and configure Python development environments
- Write and execute Python programs with proper syntax
- Use version control with Git for project management
- Understand fundamental programming concepts and data types

## Introduction: Why These Skills Matter

Data science isn't just about algorithms and models - it's about having the right foundation to work efficiently and effectively. Today we'll build that foundation by mastering two critical tools: the command line and Python fundamentals.

The command line is your gateway to powerful, efficient computing. While graphical interfaces are intuitive, the command line offers precision, automation, and access to the full power of your operating system. Python, meanwhile, has become the lingua franca of data science due to its readability, extensive libraries, and versatility.

## Part 1: Command Line Mastery

### Getting to the Command Line

Different operating systems provide different paths to command line access:

**Windows Users:**
- **PowerShell**: Built into Windows, offers modern command line features
- **Windows Subsystem for Linux (WSL)**: Run a Linux environment within Windows
  ```bash
  wsl --install
  ```
- **Command Prompt**: Traditional Windows command line

**Mac Users:**
- **Terminal**: Built-in application providing Unix command line access
- **iTerm2**: Popular third-party terminal with enhanced features

**Cloud Options:**
- **GitHub Codespaces**: Browser-based development environment
- **Google Cloud Shell**: Free cloud-based terminal

### Essential Navigation Commands

The command line organizes everything as files and directories in a hierarchical structure. Master these fundamental navigation commands:

```bash
# Show current location
pwd  # Print Working Directory

# List contents of current directory
ls       # Basic listing (Unix/Mac)
ls -la   # Detailed listing with hidden files
dir      # Windows equivalent

# Change directories
cd /path/to/directory    # Navigate to specific path
cd ~                     # Go to home directory
cd ..                    # Go up one level
cd -                     # Go to previous directory
```

**Understanding Paths:**
- **Absolute paths**: Start from root (`/home/user/documents`)
- **Relative paths**: Start from current location (`../documents`)
- **Special directories**: `.` (current), `..` (parent), `~` (home)

### File and Directory Operations

Create, manipulate, and organize your file system:

```bash
# Create directories
mkdir new_folder
mkdir -p path/to/nested/folders  # Create parent directories

# Create files
touch filename.txt          # Unix/Mac
New-Item filename.txt       # PowerShell

# Copy files and directories
cp source.txt destination.txt     # Copy file
cp -r folder/ backup_folder/      # Copy directory recursively

# Move and rename
mv old_name.txt new_name.txt      # Rename file
mv file.txt /new/location/        # Move file

# Remove files and directories
rm file.txt                       # Remove file
rm -r folder/                     # Remove directory recursively
# BE CAREFUL: No recycle bin in command line!
```

### Viewing and Searching File Contents

Efficiently examine file contents without opening editors:

```bash
# Display file contents
cat file.txt              # Show entire file
head file.txt             # Show first 10 lines
head -n 20 file.txt       # Show first 20 lines
tail file.txt             # Show last 10 lines
tail -f logfile.txt       # Follow file updates (logs)

# Search within files
grep "pattern" file.txt               # Find lines containing pattern
grep -i "pattern" file.txt            # Case-insensitive search
grep -r "pattern" directory/          # Recursive search
grep -n "pattern" file.txt            # Show line numbers
```

### Pipes and Redirection: Combining Commands

The power of the command line comes from combining simple tools:

```bash
# Pipes: Send output of one command to another
cat file.txt | grep "important"
ls -la | grep ".txt"

# Redirection: Save output to files
echo "Hello, Data Science!" > greeting.txt    # Overwrite
echo "More text" >> greeting.txt              # Append
ls -la > directory_listing.txt               # Save listing

# Combining multiple operations
cat data.csv | grep "2023" | head -20 > recent_data.txt
```

### Environment Variables and Configuration

Customize your command line environment:

```bash
# View environment variables
echo $HOME          # Your home directory
echo $PATH          # Command search paths
env                 # Show all environment variables

# Set environment variables
export EDITOR=nano                    # Set default editor
export PATH=$PATH:/new/directory      # Add to PATH

# Make changes permanent
echo 'export EDITOR=nano' >> ~/.bashrc    # Bash
echo 'export EDITOR=nano' >> ~/.zshrc     # Zsh
```

**Using .env files for projects:**
```bash
# Create .env file for project secrets
echo "API_KEY=your_secret_key" > .env
echo "DATABASE_URL=connection_string" >> .env

# NEVER commit .env files to version control!
echo ".env" >> .gitignore
```

### Shell Scripts and Automation

Create reusable command sequences:

```bash
#!/bin/bash
# data_backup.sh - Simple backup script

echo "Starting backup..."
DATE=$(date +%Y%m%d)
mkdir -p backups/backup_$DATE
cp -r data/ backups/backup_$DATE/
echo "Backup completed: backup_$DATE"
```

Make scripts executable and run them:
```bash
chmod +x data_backup.sh
./data_backup.sh
```

### File Permissions and Security

Understand and control file access:

```bash
# View permissions
ls -la file.txt
# Output: -rw-r--r-- 1 user group 1234 Date file.txt
#         │││ │││ │││
#         ││└ │└┘ └── Others (read)
#         │└─ └──── Group (read)
#         └─────── Owner (read, write)

# Change permissions
chmod +x script.sh          # Make executable
chmod 644 file.txt          # Owner: rw, Group/Others: r
chmod 755 directory/        # Owner: rwx, Group/Others: rx
```

### Task Scheduling with Cron

Automate recurring tasks:

```bash
# Edit cron jobs
crontab -e

# Cron format: minute hour day_of_month month day_of_week command
# Examples:
0 2 * * * /path/to/backup_script.sh           # Daily at 2 AM
*/15 * * * * python /path/to/monitor.py       # Every 15 minutes
0 9 * * 1 python /path/to/weekly_report.py    # Mondays at 9 AM

# Check scheduled jobs
crontab -l
```

## Part 2: Python Fundamentals

### Setting Up Python Environment

**Installation:**
```bash
# Check if Python is installed
python3 --version

# Install Python (various methods)
# Mac with Homebrew:
brew install python3

# Windows with winget:
winget install Python.Python.3.12

# Ubuntu/Debian:
sudo apt update && sudo apt install python3 python3-pip
```

**Virtual Environments** (Critical for project isolation):
```bash
# Create virtual environment
python3 -m venv my_project_env

# Activate virtual environment
source my_project_env/bin/activate    # Unix/Mac
my_project_env\Scripts\activate       # Windows

# Install packages in virtual environment
pip install pandas matplotlib seaborn

# Save project dependencies
pip freeze > requirements.txt

# Recreate environment from requirements
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Python Syntax and Basic Concepts

**Variables and Assignment:**
```python
# Variables are dynamically typed (duck typing)
name = "Data Scientist"          # String
age = 30                         # Integer
height = 5.9                     # Float
is_programmer = True             # Boolean

# Multiple assignment
x, y, z = 1, 2, 3
```

**Basic Data Types:**
```python
# Numbers
integer_value = 42
float_value = 3.14159
complex_value = 3 + 4j

# Strings
single_quoted = 'Hello'
double_quoted = "World"
multiline = """This is a
multiline string"""

# String operations
greeting = "Hello, " + "World!"        # Concatenation
formatted = f"Hello, {name}!"          # f-string formatting
repeated = "Python! " * 3              # Repetition
```

**Boolean Logic and Comparisons:**
```python
# Comparison operators
x > y        # Greater than
x >= y       # Greater than or equal
x < y        # Less than
x <= y       # Less than or equal
x == y       # Equal to
x != y       # Not equal to

# Logical operators
True and False    # False
True or False     # True
not True         # False

# Membership testing
'a' in 'apple'      # True
'z' not in 'hello'  # True
```

### Control Structures

**Conditional Statements:**
```python
# Basic if statement
if temperature > 80:
    print("It's hot outside!")
elif temperature > 60:
    print("Nice weather!")
else:
    print("It's cold!")

# Compound conditions
if (age >= 18) and (has_license == True):
    print("Can drive legally")

# Ternary operator (one-liner)
status = "adult" if age >= 18 else "minor"
```

**Loops:**
```python
# For loops with range
for i in range(5):           # 0, 1, 2, 3, 4
    print(f"Iteration {i}")

for i in range(2, 8, 2):     # 2, 4, 6 (start, stop, step)
    print(i)

# For loops with collections
fruits = ['apple', 'banana', 'orange']
for fruit in fruits:
    print(f"I like {fruit}")

# Enumerate for index and value
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# While loops
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1  # Increment
```

### Data Structures

**Lists** (ordered, mutable collections):
```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# Accessing elements
first = numbers[0]           # First element
last = numbers[-1]           # Last element
subset = numbers[1:4]        # Slicing: [2, 3, 4]

# Modifying lists
numbers.append(6)            # Add to end
numbers.insert(0, 0)         # Insert at position
numbers.remove(3)            # Remove first occurrence of 3
popped = numbers.pop()       # Remove and return last element

# List comprehensions (advanced but common)
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

**Dictionaries** (key-value pairs):
```python
# Creating dictionaries
person = {
    'name': 'Alice',
    'age': 30,
    'job': 'Data Scientist'
}

# Accessing values
name = person['name']              # Direct access
age = person.get('age', 0)         # Safe access with default

# Modifying dictionaries
person['email'] = 'alice@email.com'  # Add new key-value
person['age'] = 31                   # Update existing value
del person['job']                    # Remove key-value pair

# Dictionary methods
keys = person.keys()          # All keys
values = person.values()      # All values
items = person.items()        # Key-value pairs

# Checking for keys
if 'name' in person:
    print(f"Hello, {person['name']}")
```

**Tuples** (ordered, immutable collections):
```python
# Creating tuples
coordinates = (10, 20)
rgb_color = (255, 128, 0)

# Unpacking tuples
x, y = coordinates
red, green, blue = rgb_color

# Tuples are immutable
# coordinates[0] = 15  # This would cause an error!
```

**Sets** (unordered collections of unique elements):
```python
# Creating sets
unique_numbers = {1, 2, 3, 4, 5}
unique_from_list = set([1, 1, 2, 2, 3])  # {1, 2, 3}

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union = set1 | set2              # {1, 2, 3, 4, 5, 6}
intersection = set1 & set2        # {3, 4}
difference = set1 - set2          # {1, 2}
```

### Functions: Organizing Code

**Basic Function Definition:**
```python
def greet(name):
    """
    Greet a person by name.
    
    Args:
        name (str): The person's name
        
    Returns:
        str: A greeting message
    """
    return f"Hello, {name}!"

# Call the function
message = greet("Data Scientist")
print(message)
```

**Function Parameters:**
```python
# Default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Keyword arguments
def create_user(name, age, email="", is_admin=False):
    return {
        'name': name,
        'age': age,
        'email': email,
        'is_admin': is_admin
    }

# Call with keyword arguments
user = create_user(name="Alice", age=30, is_admin=True)

# Variable arguments
def calculate_average(*numbers):
    """Calculate average of any number of arguments"""
    return sum(numbers) / len(numbers) if numbers else 0

avg = calculate_average(1, 2, 3, 4, 5)
```

### File Operations

**Reading Files:**
```python
# Best practice: use 'with' statement for automatic file closing
with open('data.txt', 'r') as file:
    content = file.read()          # Read entire file
    print(content)

# Read line by line
with open('data.txt', 'r') as file:
    for line in file:
        print(line.strip())        # Remove trailing newline

# Read specific number of lines
with open('data.txt', 'r') as file:
    first_line = file.readline()
    all_lines = file.readlines()   # List of all lines
```

**Writing Files:**
```python
# Write to file (overwrites existing content)
with open('output.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("This is a new file.\n")

# Append to file
with open('output.txt', 'a') as file:
    file.write("This line is appended.\n")

# Write multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open('output.txt', 'w') as file:
    file.writelines(lines)
```

**Working with CSV Files:**
```python
import csv

# Read CSV
with open('data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)      # Get header row
    for row in csv_reader:
        print(row)

# Write CSV
data = [
    ['Name', 'Age', 'City'],
    ['Alice', '30', 'New York'],
    ['Bob', '25', 'San Francisco']
]

with open('people.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data)
```

### Error Handling

**Try-Except Blocks:**
```python
def safe_divide(x, y):
    """Safely divide two numbers with error handling"""
    try:
        result = x / y
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError:
        print("Error: Both arguments must be numbers!")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        print("Division operation completed")

# Examples
print(safe_divide(10, 2))      # 5.0
print(safe_divide(10, 0))      # Error message, returns None
print(safe_divide("10", 2))    # Error message, returns None
```

**File Operation Error Handling:**
```python
def read_file_safely(filename):
    """Read file with comprehensive error handling"""
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except PermissionError:
        print(f"Error: Permission denied for '{filename}'")
        return None
    except Exception as e:
        print(f"Unexpected error reading '{filename}': {e}")
        return None
```

## Part 3: Version Control with Git

### Why Version Control Matters

Version control is essential for:
- **Tracking changes**: See what changed, when, and why
- **Collaboration**: Multiple people working on same project
- **Backup**: Distributed copies of your work
- **Experimentation**: Try new features without breaking working code
- **History**: Ability to revert to previous versions

### Git Configuration

**Initial Setup:**
```bash
# Set your identity (use your real name and email)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default editor
git config --global core.editor "nano"

# View configuration
git config --list
```

**IMPORTANT**: Use GitHub's anonymous email to protect privacy:
1. Go to GitHub Settings → Emails
2. Find your anonymous email (format: `username@users.noreply.github.com`)
3. Use this for git configuration

### Basic Git Workflow

**Initialize Repository:**
```bash
# Create new repository
mkdir my_project
cd my_project
git init

# Or clone existing repository
git clone https://github.com/username/repository-name.git
```

**The Three States of Git:**
1. **Working Directory**: Your current files
2. **Staging Area**: Files prepared for commit
3. **Repository**: Committed file history

**Basic Commands:**
```bash
# Check status
git status

# Add files to staging area
git add filename.py           # Add specific file
git add *.py                 # Add all Python files
git add .                    # Add all files (be careful!)

# Commit changes
git commit -m "Add data processing function"

# View commit history
git log
git log --oneline            # Condensed view
```

### Working with Remote Repositories

**Connecting to GitHub:**
```bash
# Add remote repository
git remote add origin https://github.com/username/repository.git

# Push changes to remote
git push -u origin main      # First push (sets upstream)
git push                     # Subsequent pushes

# Pull changes from remote
git pull origin main
```

**Branching and Merging:**
```bash
# Create and switch to new branch
git checkout -b feature-branch
# Or with newer syntax:
git switch -c feature-branch

# List branches
git branch

# Switch branches
git checkout main
git switch main

# Merge branch into main
git checkout main
git merge feature-branch

# Delete branch after merging
git branch -d feature-branch
```

### Git Best Practices

**Commit Messages:**
```bash
# Good commit messages are:
# - Concise but descriptive
# - Start with verb in present tense
# - Explain what and why, not how

git commit -m "Add data validation for user input"
git commit -m "Fix memory leak in data processing loop"
git commit -m "Update README with installation instructions"
```

**What to Track and What to Ignore:**

Create `.gitignore` file:
```bash
# .gitignore file contents
__pycache__/              # Python bytecode
*.pyc                     # Compiled Python files
.env                      # Environment variables
.DS_Store                 # Mac system files
*.log                     # Log files
data/raw/                 # Raw data files (usually too large)
venv/                     # Virtual environment
.vscode/                  # Editor settings
```

**Handling Conflicts:**
```bash
# When merge conflicts occur:
# 1. Git marks conflicts in files like this:
<<<<<<< HEAD
Your current change
=======
Incoming change
>>>>>>> branch-name

# 2. Edit files to resolve conflicts
# 3. Add resolved files
git add resolved_file.py

# 4. Complete the merge
git commit -m "Resolve merge conflict in data processing"
```

## Part 4: Practical Project Setup

### Creating a Data Science Project Structure

```bash
# Create project structure
mkdir data_science_project
cd data_science_project

# Create directory structure
mkdir -p {data/{raw,processed,external},notebooks,src,tests,docs,results}

# Create essential files
touch README.md requirements.txt .gitignore .env

# Initialize git repository
git init
```

**Project Structure:**
```
data_science_project/
├── README.md              # Project description and instructions
├── requirements.txt       # Python dependencies
├── .gitignore            # Files to ignore in version control
├── .env                  # Environment variables (never commit!)
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned and transformed data
│   └── external/         # External data sources
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code modules
├── tests/                # Unit tests
├── docs/                 # Documentation
└── results/              # Output files, figures, models
```

### Sample Python Script

Create `src/data_processor.py`:
```python
#!/usr/bin/env python3
"""
Data processing utilities for the project.

This module contains functions for loading, cleaning, and transforming data.
"""

import csv
import os
from typing import List, Dict, Any


def load_csv_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries representing rows
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        csv.Error: If there's an error reading the CSV
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    except csv.Error as e:
        raise csv.Error(f"Error reading CSV file: {e}")


def clean_numeric_column(data: List[Dict[str, Any]], column_name: str) -> List[Dict[str, Any]]:
    """
    Clean numeric column by removing non-numeric values.
    
    Args:
        data: List of dictionaries representing rows
        column_name: Name of the column to clean
        
    Returns:
        List of dictionaries with cleaned numeric column
    """
    cleaned_data = []
    for row in data:
        if column_name in row:
            try:
                # Try to convert to float
                row[column_name] = float(row[column_name])
                cleaned_data.append(row)
            except (ValueError, TypeError):
                # Skip rows with invalid numeric values
                print(f"Skipping row with invalid {column_name}: {row.get(column_name)}")
                continue
        else:
            cleaned_data.append(row)
    
    return cleaned_data


def calculate_summary_stats(data: List[Dict[str, Any]], column_name: str) -> Dict[str, float]:
    """
    Calculate basic summary statistics for a numeric column.
    
    Args:
        data: List of dictionaries representing rows
        column_name: Name of the numeric column
        
    Returns:
        Dictionary with summary statistics
    """
    values = []
    for row in data:
        if column_name in row and isinstance(row[column_name], (int, float)):
            values.append(row[column_name])
    
    if not values:
        return {"count": 0, "mean": 0, "min": 0, "max": 0}
    
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values)
    }


if __name__ == "__main__":
    # Example usage
    print("Data Processing Module")
    print("This module provides utilities for data processing.")
    print("Import this module to use its functions in other scripts.")
```

### Sample Test File

Create `tests/test_data_processor.py`:
```python
"""
Unit tests for data_processor module.
"""

import unittest
import os
import tempfile
import csv
from src.data_processor import load_csv_data, clean_numeric_column, calculate_summary_stats


class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary CSV file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data = [
            ['name', 'age', 'score'],
            ['Alice', '25', '95.5'],
            ['Bob', '30', '87.2'],
            ['Charlie', 'invalid', '92.0'],
            ['Diana', '22', 'not_a_number']
        ]
        
        writer = csv.writer(self.temp_file)
        writer.writerows(test_data)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after each test method."""
        os.unlink(self.temp_file.name)
    
    def test_load_csv_data(self):
        """Test loading CSV data."""
        data = load_csv_data(self.temp_file.name)
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0]['name'], 'Alice')
        self.assertEqual(data[0]['age'], '25')
    
    def test_load_csv_data_file_not_found(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_csv_data('non_existent_file.csv')
    
    def test_clean_numeric_column(self):
        """Test cleaning numeric column."""
        data = [
            {'name': 'Alice', 'score': '95.5'},
            {'name': 'Bob', 'score': '87.2'},
            {'name': 'Charlie', 'score': 'invalid'}
        ]
        cleaned = clean_numeric_column(data, 'score')
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]['score'], 95.5)
        self.assertEqual(cleaned[1]['score'], 87.2)
    
    def test_calculate_summary_stats(self):
        """Test summary statistics calculation."""
        data = [
            {'score': 95.5},
            {'score': 87.2},
            {'score': 92.0}
        ]
        stats = calculate_summary_stats(data, 'score')
        self.assertEqual(stats['count'], 3)
        self.assertAlmostEqual(stats['mean'], 91.57, places=2)
        self.assertEqual(stats['min'], 87.2)
        self.assertEqual(stats['max'], 95.5)


if __name__ == '__main__':
    unittest.main()
```

### Running the Project

**Set up the environment:**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Or run specific test
python tests/test_data_processor.py
```

**Sample requirements.txt:**
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
pytest>=6.2.0
```

## Part 5: Development Environment Setup

### Choosing a Text Editor/IDE

**VS Code (Recommended for beginners):**
- Free, lightweight, and powerful
- Excellent Python support with extensions
- Integrated terminal and Git support
- Available on all platforms

**Essential VS Code Extensions:**
- Python (Microsoft): Python language support
- GitLens: Enhanced Git integration
- Jupyter: Notebook support in VS Code
- Python Docstring Generator: Auto-generate docstrings

**Other Options:**
- **PyCharm**: Full-featured Python IDE
- **Sublime Text**: Fast, customizable editor
- **Vim/Emacs**: For those who prefer command-line editors

### Package Management Best Practices

**Always Use Virtual Environments:**
```bash
# For each project:
python3 -m venv project_env
source project_env/bin/activate
pip install required_packages
pip freeze > requirements.txt
```

**Understanding pip:**
```bash
# Install packages
pip install package_name
pip install package_name==1.2.3      # Specific version
pip install -r requirements.txt       # From requirements file

# Upgrade packages
pip install --upgrade package_name

# List installed packages
pip list
pip freeze                            # Format for requirements.txt

# Uninstall packages
pip uninstall package_name
```

## Summary and Next Steps

### What We've Accomplished

Today we've built a solid foundation in two critical areas:

1. **Command Line Proficiency**: You can now navigate file systems, manipulate files, use pipes and redirection, create shell scripts, and automate tasks.

2. **Python Fundamentals**: You understand variables, data types, control structures, functions, file operations, and error handling.

3. **Version Control**: You can track changes, collaborate with others, and manage project history using Git.

4. **Development Workflow**: You know how to set up projects, manage dependencies, and organize code professionally.

### Practice Exercises

1. **Command Line Challenge**: Create a shell script that:
   - Creates a project directory structure
   - Downloads a sample dataset (using curl or wget)
   - Processes the data using command-line tools
   - Generates a summary report

2. **Python Project**: Build a data processing pipeline that:
   - Reads CSV data
   - Cleans and validates the data
   - Calculates summary statistics
   - Outputs results to a new file
   - Includes error handling and tests

3. **Git Workflow**: Practice the full Git workflow:
   - Initialize a repository
   - Create branches for features
   - Make commits with good messages
   - Merge branches
   - Handle conflicts

### Preparation for Next Lecture

In our next lecture, we'll dive deeper into Python data structures and development workflows. To prepare:

1. Practice the concepts covered today
2. Set up your development environment
3. Complete the practice exercises
4. Read about list comprehensions and lambda functions
5. Familiarize yourself with the concept of remote development

The skills you've learned today form the foundation for everything we'll do in data science. Master these fundamentals, and you'll be well-prepared for the more advanced topics ahead.

Remember: becoming proficient with these tools takes practice. Don't expect to remember everything immediately - focus on understanding the concepts and knowing where to find information when you need it. The command line and Python documentation are your friends!