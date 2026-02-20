---
title: "02: Python + Git"
---



# VS Code Basics (GUI-first)

We'll start in the editor so Git makes visual sense later. No JSON needed—just the VS Code interface.

## Palette Cleanse: Command Palette & Quick Open

- Open Command Palette: View → Command Palette… (Cmd+Shift+P)
- Quick Open files: (Cmd+P)
- Search across files: View → Search (Cmd+Shift+F)

## Themes and Schemes: Make it Py‑pretty

- Change Color Theme: Code → Settings → Theme → Color Theme (or Cmd+K, Cmd+T). I am a fan of:
    - "Tomorrow Night Bright"
    - "GitHub Dark High Contrast"
- Toggle icons: Code → Settings → Theme → File Icon Theme

## Meet the Main Bars

- Activity Bar (left): Explorer, Search, Source Control, Run & Debug, Extensions
- Side Bar: Toggle via View → Appearance → Show Side Bar
- Panel (bottom): Problems, Output, Debug Console, Terminal (toggle: View → Appearance → Panel Position)
- Secondary Side Bar (right): View → Appearance → Show Secondary Side Bar
- Breadcrumbs: View → Appearance → Show Breadcrumbs
- Zen Mode: View → Appearance → Zen Mode (Esc Esc to exit)

## Core Panes You’ll Use

- Explorer: View → Explorer (Cmd+Shift+E)
- Source Control: View → Source Control (Cmd+Shift+G)
- Run & Debug: Run → Start Debugging (F5) or View → Run (Ctrl+Shift+D)
- Extensions: View → Extensions (Cmd+Shift+X)
- Terminal: View → Terminal (Ctrl+`)
- Split Editor: View → Editor Layout → Split Right (or Cmd+\)

## Settings (GUI) you’ll toggle today

- Format on Save: Code → Settings → Settings → Search “Format on Save” → check
- Python Interpreter: Click bottom‑right “Python” status or Cmd+Shift+P → “Python: Select Interpreter”
- Default Formatter (optional): Settings → Search “Default Formatter” → choose “Black” or “Ruff” if installed
- Markdown Preview: Right‑click a .md → “Open Preview to the Side” (Cmd+K V)

## Recommended Extensions (install via View → Extensions)

- Python
- Pylance (can help with debugging later, I prefer ruff)
- Jupyter (we'll use this a lot later)
- Markdown All in One
- markdownlint
- Markdown Checkboxes
- GitHub Markdown Preview
- Bonus mentions: Error Lens, YAML, indent‑rainbow, GitLens

## Break(points) the Ice: 5‑minute hands‑on

1) Change the Color Theme (Preferences: Color Theme)
2) Install “Python” and “Markdown All in One”
3) Turn on “Format on Save” in Settings (GUI)
4) Open a `.py` file → add a breakpoint (click gutter) → Run → Start Debugging
5) Open a `.md` file → right‑click → Open Preview to the Side
6) Make a small edit → View → Source Control → stage, commit (GUI)

# Git Version Control

![xkcd 1597: Git](/ds217/media/02/xkcd_1597.png)

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

Git tracks every change, letting you see what changed, restore versions, work in parallel, collaborate, and avoid losing work. Infinite undo plus collaboration.

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

![Git Branches](/ds217/media/02/git_branches.png)

## Essential Git Commands

Basic Git commands let you control what changes are committed using a three-stage workflow: working directory, staging area, repository.

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
# Local repository workflow
git init                      # Start new repository
git add analysis.py           # Stage file
git commit -m "Add analysis script"  # Create commit
git branch feature-analysis  # Create branch
git checkout feature-analysis # Switch to branch

# Remote repository workflow
git clone https://github.com/user/repo.git  # Clone existing repo
git push origin main          # Push changes
git pull origin main          # Pull updates
```

![Git Clone](/ds217/media/02/git_clone.png)

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

![xkcd 1296: Git Commit](/ds217/media/02/xkcd_1296.png)

## VS Code Git Integration

### Setting Up Git in VS Code

**Reference:**

1. Install VS Code (if not already done)
2. Open VS Code → View → Source Control (or Ctrl+Shift+G)
3. If first time: VS Code will prompt to configure Git username/email

VS Code's Source Control panel makes version control accessible without memorizing command-line syntax. This integration streamlines daily staging, committing, and managing changes.

**Reference:**

- **Source Control Panel**: `Ctrl+Shift+G` (Windows/Linux) or `Cmd+Shift+G` (Mac)
- **Stage Changes**: Click `+` next to files in "Changes" section
- **Commit**: Type message in text box, press `Ctrl+Enter` (Windows/Linux) or `Cmd+Enter` (Mac)
- **View Differences**: Click on modified files to see changes
- **Branch Management**: Click branch name in status bar to switch/create branches
- **Push/Pull**: Use sync button or command palette (`Ctrl+Shift+P`)

**VS Code Git Workflow:**

```
1. Edit files (e.g., analysis.py)
2. Ctrl+Shift+G → Open Source Control panel
3. Click + next to changed files to stage
4. Type commit message: "Add data validation to analysis script"
5. Ctrl+Enter to commit
6. Click sync button to push to GitHub
```

## Git Workflow: Branching and Merging

Git branching develops features in isolation before merging to main, enabling parallel development and safe experimentation.

**Reference:**

- `git branch [name]` - Create new branch
- `git checkout [branch]` - Switch to branch
- `git checkout -b [name]` - Create and switch to new branch
- `git merge [branch]` - Merge branch into current branch
- `git branch -d [name]` - Delete branch
- `git push origin [branch]` - Push branch to remote

**Branching Workflow:**

```bash
# Create feature branch
git checkout -b feature/data-analysis
# Make changes, commit
git add .
git commit -m "Add data analysis functionality"
git push origin feature/data-analysis

# Switch back to main and merge
git checkout main
git merge feature/data-analysis
git push origin main

# Clean up feature branch
git branch -d feature/data-analysis
```

**Merge Conflict Resolution:**
When Git cannot automatically merge changes, it creates merge conflicts that must be resolved manually:

1. Open conflicted files in VS Code
2. Choose which changes to keep
3. Remove conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
4. Stage resolved files: `git add [file]`
5. Complete merge: `git commit`

## GitHub Web Interface

GitHub's web interface manages repositories, enables collaboration, and organizes projects.

**Reference:**

- **Repository Creation**: "New repository" button, choose name and settings
- **File Management**: "Add file" → "Create new file" or "Upload files"
- **Commit via Web**: Edit files directly, add commit message, commit
- **Pull Requests**: "Pull requests" tab → "New pull request"
- **Issues**: "Issues" tab → "New issue" for bug reports and feature requests
- **Project Settings**: Settings tab for permissions, branches, and integrations
- **Code Review**: Comment on specific lines, approve/request changes

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
# Python cache files
__pycache__/
*.pyc

# Data and secrets
data/raw/*.csv
.env
*.key

# IDE files
.vscode/
.idea/

# Track important files
!data/processed/important_results.csv
```

This prevents accidentally committing sensitive information, large files, or generated content while preserving important project files.

**Brief Example:**

Create repository: github.com → "+" → "New repository" → Name, description, add README → Create.

Add files: "Add file" → "Create new file" → Name, add code, commit message → Commit.

# Markdown Documentation

Markdown is a lightweight markup language for formatted text, essential for documentation and project communication. Files are human-readable and render beautifully on GitHub.

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
Analyzes study time vs. performance.

## Key Findings
- More hours → higher grades
- Regular habits help

## Code Example
```python
import pandas as pd
df = pd.read_csv('study_data.csv')
print(f"Correlation: {df['hours'].corr(df['grade']):.2f}")
```

[Raw data](data/study_data.csv)
```

# Python Fundamentals (McKinney Ch2+3)

Python emphasizes readability for data analysis. Everything is an object, enabling consistent behavior. Focus is on practical data manipulation.

![Python Import](/ds217/media/02/python_import.webp)

## Language Semantics and Object Model


Python uses indentation for code structure, creating clean code. Every value is an object with type information, enabling dynamic behavior.

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

print(type(42))        # <class 'int'>
print(isinstance("hello", str))  # True
```

## Object Introspection and Dynamic Type Checking

Object introspection examines objects at runtime—their type, attributes, and methods. Valuable for unknown datasets and flexible code.

Python uses **duck typing**: "If it walks like a duck and quacks like a duck, then it must be a duck." If an object supports the needed methods, you can use it—regardless of its actual type.

![Duck Typing](/ds217/media/02/duck_typing.jpg)

This means functions work with any object that behaves as expected, not just those of a specific type.

**Reference:**

- `type(object)` - Returns the object's type
- `dir(object)` - Lists attributes and methods
- `help(object)` - Shows documentation

```python
# Duck typing: treat the same object as different types
big_number = 12345
print(f"As a number: {big_number} (type: {type(big_number).__name__})")

# Convert to string - now we can iterate through digits
number_as_string = str(big_number)
print(f"As a string: '{number_as_string}' (type: {type(number_as_string).__name__})")

# Duck typing: if it acts like an iterable, treat it like one
digit_sum = 0
for digit_char in number_as_string:  # Treating string like a list
    digit_sum += int(digit_char)     # Converting back to int
    
print(f"Sum of digits: {digit_sum}")
```

## Scalar Types and Operations

Scalar types represent single values. Python provides rich support for numeric operations, string manipulation, and boolean logic.

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

String operations are fundamental for data cleaning. Python provides built-in methods for transforming, cleaning, and validating text data.

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

Print statements communicate results and debug code. Understanding formatting options enables clear output.

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
score = 87.3456
print(f"Student: {name}")
print(f"Score: {score:.1f}%")

# Debugging with print
data = [1, 2, 3, 4, 5]
print(f"Debug: data = {data}, length = {len(data)}")
```

## Basic File I/O Operations

File I/O operations are essential for data science. Python provides simple tools for reading and writing files.

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

# Print to file examples
with open('results.txt', 'w') as file:
    print("Analysis Results", file=file)
    print(f"Average score: {score:.1f}", file=file)

# One-liner file output
print("Debug info", file=open('debug.log', 'a'))

# Multiple outputs to same file
with open('report.txt', 'w') as report:
    print("Data Science Report", file=report)
    print("=" * 20, file=report)
    print(f"Total samples: {len(data)}", file=report)
```

## Control Flow Structures

Control flow determines execution order through conditionals and loops. Python's syntax emphasizes readability and iteration.

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

Lists provide mutable sequences for data. Tuples offer immutable sequences useful for fixed records.

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

Dictionaries provide key-value storage for structured data. Sets offer unique collections with mathematical operations.

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

List comprehensions provide concise syntax for creating lists through transformation and filtering. Sequence functions offer efficient operations.

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

Functions organize code into reusable units with clear interfaces. They enable reuse, testing, and modular design.

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

### `__main__` for script execution

if **name** == "**main**":
    # This code runs when script is executed directly
    grades = [85, 92, 78, 96, 88]
    average = calculate_average(grades)
    print(f"Average grade: {average:.1f}")

# Command Line Mastery (review)

### Essential Navigation Commands

Navigation commands orient you within the file system.

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

### File and Directory Operations

File operations create and organize project structures.

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

Text processing commands explore and manipulate text data.

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

The `tree` command shows directory structure hierarchically.

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

Shortcuts and history navigation improve command line efficiency.

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

Shell scripting automates tasks and creates reusable command sequences.

**Reference:**

- `#!/bin/bash` - Shebang line for bash scripts
- `echo "text"` - Print text to terminal
- `mkdir -p dirname` - Create directory (and parents if needed)
- `chmod +x script.sh` - Make script executable
- `$1, $2, $3...` - Command line arguments
- `$@` - All arguments
- `$#` - Number of arguments
- `$?` - Exit code of last command

**Brief Example:**

```bash
#!/bin/bash
# Create project structure
echo "Setting up project..."
mkdir -p src data output
echo "Directories created"

# Make script executable
chmod +x setup.sh

# Create files using here-documents
cat > data/sample.csv << 'EOF'
name,age,grade
Alice,20,85
Bob,19,92
EOF

echo "Setup complete!"
```

### Command Chaining and Redirection

Command chaining creates data processing pipelines. Redirection controls input and output.

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
