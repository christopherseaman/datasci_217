Command Line & Python Fundamentals, plus Version Control

![xkcd 1597: Git](media/xkcd_1597.png)

# Git Version Control

Don't worry - we're taking a different approach than that xkcd suggests!

## Why Version Control Matters

### The Problem Without Version Control

Picture this: You're working on a data analysis. You create these files:

- `analysis_v1.py`
- `analysis_v2.py`
- `analysis_v2_final.py`
- `analysis_v2_final_ACTUALLY_FINAL.py`
- `analysis_fixed_broken_computer_recovery.py`

Sound familiar? Now imagine collaborating with teammates doing the same thing. Chaos!

### The Git Solution

Git tracks every change to every file in your project. You can:

- See exactly what changed and when
- Go back to any previous version
- Work on features in parallel without conflicts
- Collaborate with teammates seamlessly
- Never lose work (it's all backed up on GitHub)

It's like having infinite "undo" for your entire project, plus collaboration superpowers.

## Git Concepts - The Mental Model

### Repository (Repo)

Your project folder that Git tracks. Contains your files plus a hidden `.git` folder with all the version history.

Think: "This entire folder is under Git management."

### Commit

A saved snapshot of your project at a specific point in time. Like saving a game - you can always come back to this exact state.

Think: "I'm saving my progress with a description of what I accomplished."

### Remote

The version of your repository stored on GitHub (or similar service). Your local computer has a copy, GitHub has a copy, your teammates have copies.

Think: "The shared version everyone can access."

### Branch

A parallel timeline for your project. The main branch contains your official version, feature branches contain experimental work.

Think: "I'm trying something new without risking the working version."

*We'll focus on the main branch today - branches come later!*

**Reference:**

- **Repository**: Collection of objects and references
- **Commit**: Snapshot with metadata (author, message, parents)
- **Blob**: File content
- **Tree**: Directory structure
- **Reference**: Human-readable name pointing to commit
- **HEAD**: Current commit reference
- **Branch**: Movable reference to commit
- **Remote**: Reference to repository on another machine

![Git Branches](media/git_branches.png)

![Git Clone](media/git_clone.png)

<!-- FIXME: Add Git workflow diagram showing working directory → staging area → repository flow -->
![Git Workflow Diagram](media/git_workflow_diagram.png)

<!-- FIXME: Add VS Code Git interface screenshot showing Source Control panel -->
![VS Code Git Interface](media/vscode_git_interface.png)

## GUI-First Git with VS Code

### Why Start with GUI?

Command line Git is powerful, but the visual interface helps you understand what's happening. VS Code's Git integration shows you:

- Which files changed (visual diff)
- What you're about to commit
- The status of everything at a glance

Once you understand Git concepts, you can choose GUI or command line based on the task.

### Setting Up Git in VS Code

**Reference:**

1. Install VS Code (if not already done)
2. Open VS Code → View → Source Control (or Ctrl+Shift+G)
3. If first time: VS Code will prompt to configure Git username/email

**Brief Example:**

```
Git configuration (one-time setup):
- Full Name: Alice Smith
- Email: alice.smith@ucsf.edu (use your actual UCSF email)
```

![GitHub Email Setup](media/github_email.png)

## Essential Git Commands

Basic Git commands manipulate the object database and references. The three-stage workflow (working directory, staging area, repository) provides precise control over what changes are committed.

**Reference:**

Essential:

- `git init` - Initialize repository
- `git clone [url]` - Copy remote repository
- `git status` - Show working directory status
- `git add [file]` - Stage changes
- `git commit -m "message"` - Create commit
- `git push [remote] [branch]` - Send commits to remote
- `git pull [remote] [branch]` - Fetch and merge from remote

Helpful but less essential:

- `git remote add [name] [url]` - Add remote
- `git fetch [remote]` - Download commits without merging
- `git remote -v` - List remotes
- `git log` - Show commit history
- `git diff` - Show changes
- `git checkout [commit/branch]` - Switch to commit or branch
- `git branch [name]` - Create branch
- `git merge [branch]` - Merge branch

**Brief Example:**

```bash
git init                      # Start new repository
git add analysis.py           # Stage file
git commit -m "Add analysis script"  # Create commit
git log --oneline            # View history
git branch feature-analysis  # Create branch
git checkout feature-analysis # Switch to branch
```

**Another Example:**

```bash
git clone https://github.com/user/repo.git
git remote add origin https://github.com/user/repo.git
git push origin main
git pull origin main
```

**Good vs. Bad Commit Messages**

```bash
# Good commit message
git commit -m "Add data validation to analysis script

- Validate input file exists before processing
- Check data format matches expected schema
- Add error handling for malformed data

Fixes issue #123"

# Bad commit message
git commit -m "minor changes"
```

![xkcd 1296: Git Commit](media/xkcd_1296.png)
## VS Code Git Integration

VS Code provides a visual Git workflow through its Source Control panel, making version control accessible without memorizing command-line syntax. This integration streamlines the daily workflow of staging, committing, and managing changes.

**Reference:**

- **Source Control Panel**: `Ctrl+Shift+G` (Windows/Linux) or `Cmd+Shift+G` (Mac)
- **Stage Changes**: Click `+` next to files in "Changes" section
- **Commit**: Type message in text box, press `Ctrl+Enter` (Windows/Linux) or `Cmd+Enter` (Mac)
- **View Differences**: Click on modified files to see changes
- **Branch Management**: Click branch name in status bar to switch/create branches
- **Push/Pull**: Use sync button or command palette (`Ctrl+Shift+P`)

**Step-by-Step Workflow:**

```
1. Open VS Code in your project directory
2. Make changes to files (edit, create, delete)
3. Open Source Control panel (Ctrl+Shift+G)
4. Review changes in "Changes" section
5. Stage files by clicking + next to each file
6. Type commit message in text box
7. Press Ctrl+Enter to commit
8. Click sync button to push to remote
```

**Brief Example:**

```
Day-to-day VS Code Git workflow:
1. Edit analysis.py file
2. Ctrl+Shift+G → See "analysis.py" in Changes
3. Click + to stage file
4. Type: "Add data validation to analysis script"
5. Ctrl+Enter to commit
6. Click sync button to push to GitHub
```

## GitHub Web Interface

GitHub's web interface provides comprehensive repository management, collaboration features, and project organization tools. It serves as the central hub for code sharing, issue tracking, and team collaboration.

**Reference:**

- **Repository Creation**: "New repository" button, choose name and settings
- **File Management**: "Add file" → "Create new file" or "Upload files"
- **Commit via Web**: Edit files directly, add commit message, commit
- **Pull Requests**: "Pull requests" tab → "New pull request"
- **Issues**: "Issues" tab → "New issue" for bug reports and feature requests
- **Project Settings**: Settings tab for permissions, branches, and integrations
- **Code Review**: Comment on specific lines, approve/request changes

**Collaboration Features:**

**Gitignore Files:**
A `.gitignore` file specifies which files and directories Git should ignore when tracking changes. This is crucial for data science projects to avoid committing sensitive data, large datasets, or generated files.

**Reference:**

- `.gitignore` patterns use glob patterns
- `#` for comments
- `*` matches any characters
- `?` matches single character
- `[abc]` matches any character in brackets
- `!` negates pattern
- `**/` matches directories recursively

**Brief Example:**

```
# Ignore Python cache files
__pycache__/
*.pyc

# Ignore data files
data/raw/*.csv
results/**/*.txt

# Ignore secrets
.env
*.key

# Ignore IDE files
.vscode/
.idea/

# But track important data files
!data/processed/important_results.csv
```

This prevents accidentally committing sensitive information, large files, or generated content while preserving important project files.

```
Repository Management:
- Create repositories with README, .gitignore, license
- Manage branch protection rules
- Configure repository settings and permissions

File Operations:
- Create, edit, and delete files directly in browser
- Upload multiple files via drag-and-drop
- View file history and blame annotations

Team Collaboration:
- Create and manage pull requests
- Review code with inline comments
- Track issues and project milestones
- Use GitHub Actions for automation
```

**Brief Example:**

```
Creating a new repository on GitHub:
1. Go to github.com → Click "+" → "New repository"
2. Name: "data-analysis-project"
3. Add description: "Analysis of sales data"
4. Check "Add a README file"
5. Choose license (MIT recommended)
6. Click "Create repository"

Adding files via web interface:
1. Click "Add file" → "Create new file"
2. Name: "analysis.py"
3. Add Python code
4. Commit message: "Add initial analysis script"
5. Click "Commit new file"
```


# Markdown Documentation

Markdown is a lightweight markup language for creating formatted text. It's essential for documentation, README files, and project communication. Markdown files are human-readable in plain text but render beautifully when displayed on platforms like GitHub.

**Reference:**

- Headers: `# H1`, `## H2`, `### H3`
- Bold: `**bold text**`
- Italic: `*italic text*`
- Code: `` `inline code` ``
- Code blocks: ```language
- Lists: `- item` or `1. item`
- Links: `[text](url)`
- Images: `![alt](url)`
- Tables: `| col1 | col2 |`

**Brief Example:**

```
# Data Analysis Report

## Overview
This report analyzes the relationship between **study time** and *academic performance*.

### Key Findings
1. Students who study **more than 20 hours per week** achieve higher grades
2. *Regular study habits* correlate with better outcomes
3. **Group study sessions** are particularly effective

## Data Summary
| Study Hours | Average Grade | Sample Size |
|-------------|---------------|-------------|
| 0-10 | 2.8 | 45 |
| 11-20 | 3.2 | 67 |
| 21+ | 3.7 | 38 |

> **Note**: All grades are on a 4.0 scale

### Code Example
```python
import pandas as pd

# Load and analyze study data
df = pd.read_csv('study_data.csv')
correlation = df['hours'].corr(df['grade'])
print(f"Correlation: {correlation:.2f}")
```

[View raw data](data/study_data.csv) | [Download report](report.pdf)

```

# Python Fundamentals (McKinney Ch2+3)

Python's design emphasizes readability and simplicity, making it ideal for data analysis. The language's object model treats everything as an object, providing consistent behavior across different data types. McKinney's approach focuses on practical data manipulation skills essential for scientific computing.

![Python Import](media/python_import.webp)

## Language Semantics and Object Model


Python uses indentation for code structure instead of braces, creating visually clean code. Every value is an object with associated type information, enabling dynamic behavior and introspection capabilities.

**Reference:**
- Indentation defines code blocks (4 spaces recommended)
- `#` for comments
- `type(object)` - Get object type
- `isinstance(object, type)` - Type checking
- `id(object)` - Get object identity
- `dir(object)` - List object attributes

**Brief Example:**
```python
# Indentation matters
if x > 0:
    print("Positive")
    y = x * 2

# Everything is an object
print(type(42))        # <class 'int'>
print(isinstance("hello", str))  # True
```


## Object Introspection and Dynamic Type Checking

Object introspection allows you to examine objects at runtime, revealing their type, attributes, and methods. This is particularly valuable in data science for understanding unknown datasets, debugging complex objects, and writing flexible code that adapts to different data structures.

**Reference:**

- `type(object)` - Returns the object's type as a class
- `dir(object)` - Returns a list of valid attributes and methods
- `help(object)` - Displays comprehensive documentation for the object

**Brief Example:**

```python
# Understanding data structures
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
numbers = np.array([1, 2, 3, 4, 5])

# Basic object introspection
print(f"Data type: {type(data)}")           # <class 'pandas.core.frame.DataFrame'>
print(f"Array type: {type(numbers)}")       # <class 'numpy.ndarray'>

# Examine available methods
print("DataFrame methods:", [m for m in dir(data) if not m.startswith('_')][:10])

# Get help on specific methods
help(data.head)
```

## Scalar Types and Operations

Scalar types represent single values in Python. The language provides rich support for numeric operations, string manipulation, and boolean logic essential for data analysis.

**Reference:**

- `int` - Arbitrary precision integers
- `float` - Double-precision floating-point
- `str` - Unicode strings
- `bool` - True/False values
- `None` - Null value
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `and`, `or`, `not`

**Brief Example:**

```python
# Numeric operations
count = 150
average = 87.3
population = 1.4e9  # Scientific notation

# String operations
name = "Alice Johnson"
clean_name = "  Bob Smith  ".strip()

# Boolean logic
has_data = True
analysis_ready = has_data and count > 0
```

## String Operations for Data Cleaning

String operations are fundamental for data cleaning and preprocessing. Python provides powerful built-in methods for transforming, cleaning, and validating text data, essential for working with real-world datasets.

**Reference:**

- `str.strip()` - Remove leading/trailing whitespace
- `str.lower()`, `str.upper()` - Case conversion
- `str.split(separator)` - Split into list
- `str.replace(old, new)` - Replace substrings
- `str.startswith()`, `str.endswith()` - Check prefixes/suffixes
- `str.find()`, `str.index()` - Find substring positions
- `str.isdigit()`, `str.isalpha()`, `str.isalnum()` - Type validation

**Brief Example:**

```python
# Data cleaning operations
messy_data = "  Alice Johnson  "
clean_name = messy_data.strip().title()

# Text processing
filename = "data_2023_report.csv"
if filename.endswith(".csv"):
    print("Processing CSV file")

# Data validation
user_input = "123abc"
if user_input.isalnum():
    print("Input contains only letters and numbers")
```

## Print Statements and Output Formatting

Print statements are your primary tool for communicating results and debugging code. Understanding different formatting options enables clear, professional output essential for data analysis.

**Reference:**

- `print(value)` - Basic printing
- `print(value1, value2, value3)` - Multiple values
- `f"text {variable}"` - F-string formatting (preferred)
- `"text {}".format(variable)` - Format method
- `print(f"Debug: {var} = {value}")` - Debugging output
- `print(f"Result: {result:.2f}")` - Number formatting

**Brief Example:**

```python
# Basic printing
print("Analysis complete")
print("Value:", 42)

# F-string formatting (preferred)
name = "Alice"
score = 87.3
print(f"Student: {name}")
print(f"Score: {score:.1f}%")

# Debugging with print
data = [1, 2, 3, 4, 5]
print(f"Debug: data = {data}, length = {len(data)}")
print(f"Debug: type = {type(data)}")
```

## Basic File I/O Operations

File input/output operations are essential for data science workflows. Python provides simple yet powerful tools for reading from and writing to files, enabling data import, export, and persistent storage.

**Reference:**

- `open(file, mode)` - Open file with specified mode
- `'r'` - Read mode (default)
- `'w'` - Write mode (overwrites existing files)
- `'a'` - Append mode (adds to existing files)
- `'x'` - Create mode (fails if file exists)
- `file.read()` - Read entire file content
- `file.readline()` - Read single line
- `file.readlines()` - Read all lines into list
- `file.write(string)` - Write string to file
- `file.close()` - Close file handle

**Brief Example:**

```python
# Reading from a file
with open('data.txt', 'r') as file:
    content = file.read()
    print(f"File content: {content}")

# Writing to a file
results = ["Alice: 95", "Bob: 87", "Charlie: 92"]
with open('grades.txt', 'w') as file:
    for result in results:
        file.write(f"{result}\n")

# Appending to a file
with open('log.txt', 'a') as file:
    file.write("2023-12-01: Analysis completed\n")
```

## Type Checking and Debugging

Understanding data types is crucial for debugging and data analysis. Python's dynamic typing means variables can change type, making type checking essential for reliable code.

**Reference:**

- `type(variable)` - Get variable type
- `isinstance(variable, type)` - Check if variable is specific type
- `print(f"Type: {type(var)}")` - Debug type information
- `print(f"Value: {var}")` - Debug variable values
- `print(f"Debug: {var} = {value}, type = {type(value)}")` - Complete debugging

**Brief Example:**

```python
# Type checking for debugging
user_input = "42"  # This is a string, not a number!
print(f"Input: {user_input}")
print(f"Type: {type(user_input)}")  # <class 'str'>

# Convert and verify
number = int(user_input)
print(f"Converted: {number}")
print(f"New type: {type(number)}")  # <class 'int'>

# Debugging data processing
data = [1, 2, "3", 4, 5]  # Mixed types!
for item in data:
    print(f"Item: {item}, Type: {type(item)}")
    if isinstance(item, str):
        print(f"  Converting string '{item}' to int")
        item = int(item)
```

## Error Handling Basics

Error handling prevents your programs from crashing when unexpected things happen. Python's try/except statements let you catch errors and handle them gracefully.

**Reference:**

- `try: ... except: ...` - Basic exception handling
- `except ValueError:` - Catch specific exception types
- Common exceptions: `ValueError`, `TypeError`, `FileNotFoundError`

**Brief Example:**

```python
# Basic error handling
try:
    number = int("not_a_number")
except ValueError:
    print("Could not convert to number")

try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
```

## Control Flow Structures

Control flow determines program execution order through conditional statements and loops. Python's syntax emphasizes readability while providing powerful iteration capabilities.

**Reference:**

- `if condition: ... elif condition: ... else: ...`
- `for variable in iterable: ...`
- `while condition: ...`
- `break` - Exit loop
- `continue` - Skip to next iteration
- `range(start, stop, step)` - Generate number sequences

**Brief Example:**

```python
# Conditional logic
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# Iteration
for i in range(5):
    print(f"Count: {i}")

# List iteration
grades = [85, 92, 78, 96]
for grade in grades:
    if grade >= 90:
        print(f"Excellent: {grade}")
```

## Data Structures: Lists and Tuples

Lists provide mutable sequences for storing and manipulating data collections. Tuples offer immutable sequences useful for fixed data records and function returns.

**Reference:**

- `list()` - Create list
- `[item1, item2, ...]` - List literal
- `list.append(item)` - Add to end
- `list.insert(index, item)` - Insert at position
- `list.remove(item)` - Remove first occurrence
- `list.pop(index)` - Remove and return item
- `tuple()` - Create tuple
- `(item1, item2, ...)` - Tuple literal

**Brief Example:**

```python
# Lists - mutable sequences
grades = [85, 92, 78, 96, 88]
grades.append(90)
grades.insert(1, 87)
total = sum(grades)

# Tuples - immutable sequences
coordinates = (40.7128, -74.0060)
name, age, gpa = ("Alice", 22, 3.8)  # Unpacking
```


## Data Structures: Dictionaries and Sets

Dictionaries provide key-value storage for structured data, while sets offer unique collections with mathematical operations. Both are essential for data organization and lookup operations.

**Reference:**

- `dict()` - Create dictionary
- `{key: value, ...}` - Dictionary literal
- `dict[key]` - Access value
- `dict.get(key, default)` - Safe access
- `dict.keys()`, `dict.values()`, `dict.items()` - Iteration
- `set()` - Create set
- `{item1, item2, ...}` - Set literal
- `set.union()`, `set.intersection()`, `set.difference()` - Set operations

**Brief Example:**

```python
# Dictionaries - key-value storage
student = {"name": "Alice", "grade": 85, "major": "Data Science"}
print(student["name"])  # "Alice"
print(student.get("gpa", 0.0))  # Safe access

# Sets - unique collections
math_students = {"Alice", "Bob", "Charlie"}
cs_students = {"Alice", "Diana", "Eve"}
both_subjects = math_students & cs_students  # Intersection
```

## List Comprehensions and Sequence Functions

List comprehensions provide concise syntax for creating lists through transformation and filtering. Built-in sequence functions offer efficient operations on data collections.

**Reference:**

- `[expr for item in iterable if condition]` - List comprehension
- `enumerate(iterable)` - Get index and value pairs
- `zip(iterable1, iterable2)` - Combine sequences
- `sorted(iterable)` - Create sorted list
- `reversed(iterable)` - Reverse sequence
- `sum()`, `min()`, `max()`, `len()` - Aggregation functions

**Brief Example:**

```python
# List comprehensions
grades = [85, 92, 78, 96, 88]
passing_grades = [g for g in grades if g >= 80]

# Sequence functions
for index, grade in enumerate(grades):
    print(f"Student {index + 1}: {grade}")

names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
for name, score in zip(names, scores):
    print(f"{name}: {score}")
```

## Functions

Functions organize code into reusable units with clear interfaces. They enable code reuse, testing, and modular design essential for maintainable data science projects.

**Reference:**

- `def function_name(parameters): ...` - Function definition
- `return value` - Return value
- Function calls: `result = function_name(arguments)`
- Default parameters: `def func(param=default_value):`

**Brief Example:**

```python
# Function definition
def calculate_average(grades):
    """Calculate the average of a list of grades."""
    if not grades:
        return 0
    return sum(grades) / len(grades)

### Function usage
grades = [85, 92, 78, 96, 88]
average = calculate_average(grades)
print(f"Average grade: {average:.1f}")
```

**Library Usage:**

```python
# Other scripts can import and use these functions, e.g., analysis.py
from analysis import average
result = average([90, 95, 87])
```

### `__main__` for script execution

if __name__ == "__main__":
    # This code runs when script is executed directly
    grades = [85, 92, 78, 96, 88]
    average = calculate_average(grades)
    print(f"Average grade: {average:.1f}")



# Command Line Mastery (review)

### Essential Navigation Commands

Navigation commands form the foundation of command line usage, allowing you to orient yourself within the file system and move between directories efficiently.

**Reference:**

- `pwd` - Print working directory (shows current location)
- `ls` - List directory contents
- `ls -la` - List with detailed information (permissions, size, date)
- `cd [path]` - Change directory
- `cd ..` - Move up one directory level
- `cd ~` - Navigate to home directory
- `cd -` - Return to previous directory

**Brief Example:**

```bash
pwd                    # /Users/username/Documents
ls -la                 # Show all files with details
cd projects/data_science
pwd                    # /Users/username/Documents/projects/data_science
```

<!-- FIXME: Add directory structure tree diagram showing typical project organization -->
![Directory Structure Tree](media/directory_structure_tree.png)

<!-- FIXME: Add command flow visualization showing navigation command relationships -->
![Command Flow Visualization](media/command_flow_diagram.png)

### File and Directory Operations

File operations enable creation, modification, and organization of your project structure. These commands form the building blocks of data science project management.

**Reference:**

- `mkdir [name]` - Create directory
- `mkdir -p [path/to/nested]` - Create nested directories
- `touch [filename]` - Create empty file
- `cp [source] [destination]` - Copy files or directories
- `mv [source] [destination]` - Move or rename files
- `rm [filename]` - Remove file
- `rm -r [directory]` - Remove directory recursively
- `rm -rf [directory]` - Force remove directory (use with caution)

**Brief Example:**

```bash
mkdir -p data/raw data/processed scripts
touch scripts/analysis.py
cp data/raw/dataset.csv data/processed/
```

### Text Processing and Search

Text processing commands enable efficient data exploration and manipulation. These tools become essential when working with log files, configuration files, or any text-based data.

**Reference:**

- `cat [filename]` - Display entire file
- `head [filename]` - Show first 10 lines
- `tail [filename]` - Show last 10 lines
- `grep [pattern] [filename]` - Search for text patterns
- `grep -r [pattern] [directory]` - Recursive search
- `wc -l [filename]` - Count lines in file

**Brief Example:**

```bash
head -20 data.csv              # Preview first 20 lines
grep "error" logfile.txt       # Find error messages
```

### Visual Directory Structure

Sometimes you need to see the overall structure of a project or directory tree. The `tree` command provides a clean, hierarchical view that's invaluable for understanding project organization.

**Reference:**

- `tree` - Show directory structure
- `tree -L 2` - Limit depth to 2 levels
- `tree -a` - Show hidden files
- `tree -d` - Show directories only

**Brief Example:**

```bash
tree                    # Show full directory structure
tree -L 2              # Show only 2 levels deep
tree -d                # Show only directories
```

### History Navigation and Shortcuts

Command line efficiency comes from mastering shortcuts and history navigation. These tools save time and reduce errors in daily work.

**Reference:**

- `Up arrow` - Previous command
- `Down arrow` - Next command
- `Ctrl+A` - Move to beginning of line
- `Ctrl+E` - Move to end of line
- `Ctrl+K` - Delete from cursor to end of line
- `Ctrl+U` - Delete from cursor to beginning of line
- `Tab` - Auto-complete commands, files, directories
- `Ctrl+R` - Reverse search through history
- `Ctrl+C` - Cancel current command
- `Ctrl+D` - Exit shell

**Brief Example:**

```bash
# Use up arrow to recall previous commands
# Use Tab to complete: cd pro<Tab> → cd projects/
# Use Ctrl+R to search: Ctrl+R then type "git" to find git commands
```

### Shell Scripting Fundamentals

Shell scripting automates repetitive tasks and creates reusable command sequences. Scripts combine multiple commands with control flow to handle complex data processing workflows.

**Reference:**

- `#!/bin/bash` - Shebang line for bash scripts
- `$1, $2, $3...` - Command line arguments
- `$@` - All arguments
- `$#` - Number of arguments
- `$?` - Exit code of last command
- `if [ condition ]; then ... fi` - Conditional execution
- `for variable in list; do ... done` - Loop execution

**Brief Example:**

```bash
#!/bin/bash
# Process multiple data files
for file in data/*.csv; do
    echo "Processing $file"
    head -1 "$file" > "processed/$(basename "$file")"
done
```

### Command Chaining and Redirection

Command chaining allows complex data processing pipelines by connecting multiple commands. Redirection controls where input comes from and where output goes, enabling powerful data transformations.

**Reference:**

- `command1 | command2` - Pipe output to next command
- `command1 && command2` - Run command2 only if command1 succeeds
- `command1 || command2` - Run command2 only if command1 fails
- `command > file` - Redirect output to file
- `command >> file` - Append output to file
- `command < file` - Use file as input

**Brief Example:**

```bash
grep "error" logfile.txt | wc -l    # Count error lines
ls *.csv | head -5 > filelist.txt   # Save first 5 CSV files to list
```
