---
notion:
  title_line: "# `git gud` with Version Control"
  role: lecture
  status: mapped
  page_id: "271d9fdd-1a1a-8036-9c2b-c4a66ae97d9d"
  url: "https://app.notion.com/p/271d9fdd1a1a80369c2bc4a66ae97d9d"
---

# `git gud` with Version Control

See [BONUS.md](BONUS.md) for the optional extensions.

[Live Demo Guide](demo/DEMO_GUIDE.md)

# VS Code Basics (GUI-first)

We'll start in the editor so Git makes visual sense later.

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
- Terminal: View → Terminal (Ctrl + grave accent key)
- Split Editor: View → Editor Layout → Split Right (or Cmd+\)

![VS Code's integrated terminal with command history](media/vscode-integrated-terminal.png)

The terminal accepts the same shell commands as your standalone terminal. Screenshot: [VS Code terminal documentation](https://code.visualstudio.com/docs/terminal/basics).

## Settings (GUI) you’ll toggle today

- Format on Save: Code → Settings → Settings → Search “Format on Save” → check
- Python Interpreter: Click bottom‑right “Python” status or Cmd+Shift+P → “Python: Select Interpreter”
- Default Formatter (optional): Settings → Search “Default Formatter” → choose “Black” or “Ruff” if installed
- Markdown Preview: Right‑click a .md → “Open Preview to the Side” (Cmd+K V)

![The Python Select Interpreter menu in VS Code](media/vscode-selected-interpreter.png)

Choose the Python 3.13 interpreter installed for the course; this documentation screenshot shows example versions and paths. Source: [VS Code Python environments](https://code.visualstudio.com/docs/python/environments).

## Recommended Extensions (install via View → Extensions)

- Python
- Pylance (can help with debugging later, I prefer ruff)
- Jupyter (we'll use this a lot later)
- Markdown All in One
- markdownlint
- Markdown Checkboxes
- GitHub Markdown Preview
- Bonus mentions: Error Lens, YAML, indent‑rainbow, GitLens

## Break(points) the Ice

1) Change the Color Theme (Preferences: Color Theme)
2) Install “Python” and “Markdown All in One”
3) Turn on “Format on Save” in Settings (GUI)
4) Open a `.py` file → add a breakpoint (click gutter) → Run → Start Debugging
5) Open a `.md` file → right‑click → Open Preview to the Side
6) Make a small edit → View → Source Control → stage, commit (GUI)

![Python paused at a breakpoint, with the variable's value visible at left](media/vscode-python-debug-paused.png)

The highlighted line runs next. Inspect **Variables**, then step forward to see what changes. Screenshot: [VS Code Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial); the status bar shows the example's older interpreter.

# Git Version Control

![xkcd 1597: Git](media/xkcd_1597.png)

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

### Commit

A saved snapshot of your project at a specific point in time. Like saving a game - you can always come back to this exact state.

### Remote

The version of your repository stored on GitHub (or similar service). Your local computer has a copy, GitHub has a copy, your teammates have copies.

### Branch

A parallel timeline for your project. The main branch contains your official version, feature branches contain experimental work.

*We'll focus on the main branch today—branches come later!*

### Reference Card: Git Vocabulary

- **Repository**: Collection of objects and references
- **Commit**: Snapshot with metadata (author, message, parents)
- **Blob**: File content
- **Tree**: Directory structure
- **Reference**: Human-readable name pointing to commit
- **HEAD**: Current commit reference
- **Branch**: Movable reference to commit
- **Remote**: Reference to repository on another machine

![Git Branches](media/git_branches.png)

## Essential Git Commands

Basic Git commands let you control what changes are committed using a three-stage workflow: working directory, staging area, repository.

### Reference Card: Essential Git Commands

| Task | Command | Result |
| --- | --- | --- |
| Start locally | `git init` | A local repository |
| Copy a remote | `git clone URL` | A working folder linked to the remote |
| Inspect work | `git status`, `git diff` | Changed-file list and line differences |
| Stage a file | `git add FILE` | File changes included in the next commit |
| Save a snapshot | `git commit -m "message"` | A local commit |
| Send commits | `git push REMOTE BRANCH` | Remote branch updated |
| Retrieve commits | `git fetch REMOTE` | Remote-tracking references updated |
| Retrieve and integrate | `git pull REMOTE BRANCH` | Incoming work integrated into the current branch |
| Connect a remote | `git remote add NAME URL` | Named remote; inspect with `git remote -v` |
| Inspect history | `git log` | Commits with messages and authors |
| Create a branch | `git branch NAME` | New branch at the current commit |
| Switch versions | `git checkout BRANCH_OR_COMMIT` | Working copy changes to that version |
| Integrate a branch | `git merge BRANCH` | Branch changes combined with the current branch |

### Code Snippet: Essential Git Commands

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

![Git Clone](media/git_clone.png)

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

Review a change, stage the files to include, commit the snapshot, then sync it to GitHub.

![Stage a changed file using the plus button in VS Code](media/vscode-stage.png)

![Enter a message and commit the staged files](media/vscode-commit.png)

![Sync committed changes with the GitHub copy](media/vscode-sync.png)

Screenshots: [VS Code source control documentation](https://code.visualstudio.com/docs/sourcecontrol/quickstart).

### Setting Up Git in VS Code

1. Install VS Code (if not already done)
2. Open VS Code → View → Source Control (or Ctrl+Shift+G)
3. If first time: VS Code will prompt to configure Git username/email

#### Reference Card: VS Code Git Actions

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

### Reference Card: Git Workflow: Branching and Merging

- `git branch [name]`: Create new branch
- `git checkout [branch]`: Switch to branch
- `git checkout -b [name]`: Create and switch to new branch
- `git merge [branch]`: Merge branch into current branch
- `git branch -d [name]`: Delete branch
- `git push origin [branch]`: Push branch to remote

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

**Merge Conflict Resolution:** When Git cannot automatically merge changes, it creates merge conflicts that must be resolved manually:

1. Open conflicted files in VS Code
2. Choose which changes to keep
3. Remove conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
4. Stage resolved files: `git add [file]`
5. Complete merge: `git commit`

## GitHub Web Interface

GitHub's web interface manages repositories, enables collaboration, and organizes projects.

### Reference Card: GitHub Web Interface

- **Repository Creation**: "New repository" button, choose name and settings
- **File Management**: "Add file" → "Create new file" or "Upload files"
- **Commit via Web**: Edit files directly, add commit message, commit
- **Pull Requests**: "Pull requests" tab → "New pull request"
- **Issues**: "Issues" tab → "New issue" for bug reports and feature requests
- **Project Settings**: Settings tab for permissions, branches, and integrations
- **Code Review**: Comment on specific lines, approve/request changes

### Gitignore Files

A `.gitignore` file lists patterns for untracked files Git should ignore. Adding a pattern does not untrack files already committed.

#### Reference Card: Ignore Patterns

- `# comment`: Explain a pattern
- `*.csv`: Match CSV filenames
- `file?.txt`: Match one character, such as `file1.txt`
- `[abc].txt`: Match `a.txt`, `b.txt`, or `c.txt`
- `!keep.csv`: Re-include a file matched by an earlier pattern
- `**/cache/`: Match cache directories at any depth

#### Code Snippet: A Project's `.gitignore`

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

### Create and Edit Files on GitHub

Create repository: github.com → "+" → "New repository" → Name, description, add README → Create.

Add files: "Add file" → "Create new file" → Name, add code, commit message → Commit.

# Markdown Documentation

Markdown is a lightweight markup language for formatted text, essential for documentation and project communication. Files are human-readable and render beautifully on GitHub.

## Reference Card: Markdown Documentation

- `# H1`, `## H2`, `### H3`: Headings at decreasing levels
- `**bold text**`: Strong emphasis
- `*italic text*`: Emphasis
- `` `inline code` ``: Literal code within a sentence
- Three backticks, then a language name: Start a fenced code block; close with three backticks
- `- item` or `1. item`: Unordered or ordered list
- `[text](url)`: Clickable link
- `![alt](url)`: Image with descriptive alternative text
- `\| col1 \| col2 \|`: Table row; follow the header with a separator row

## Code Snippet: Markdown Documentation

```markdown
# Data Analysis Report

## Overview
Analyzes study time vs. performance.

## Key Findings
- More hours → higher grades
- Regular habits help

## Code Example
~~~python
grades = [85, 92, 78]
average = sum(grades) / len(grades)
print(f"Average grade: {average:.1f}")
~~~
```

# Python Fundamentals (McKinney Ch2+3)

![xkcd 1429, “Data”: a grammar joke contrasting polling data with the Star Trek character Data.](media/xkcd_1429.png)

*Data* by xkcd — in Python, everything is an object. In Star Trek, Data is too.

Python emphasizes readable, practical data manipulation. Its values are objects with consistent behavior.

## What is new in Lecture 02?

Lecture 01 established the command line, Python installation, variables, basic expressions, and introductory scripts. This lecture uses that foundation in VS Code and Git, then moves into Python’s object model, imports, collections and mutability, functions, file I/O, targeted exceptions, and the `__main__` entry-point pattern.

![Python Import](media/python_import.webp)

## Printing and Basic Input

An f-string starts with `f` and inserts the value of each expression inside `{}`. Formatting controls how those values appear; `input()` reads typed text for interactive scripts.

### Reference Card: Printing and Formatting

| Syntax | Purpose | Example output |
| --- | --- | --- |
| `print("Score:", score)` | Print separate values with spaces | `Score: 87.3` |
| `f"{score}"` | Insert a value into text | `87.3` |
| `f"{score:.1f}"` | One decimal place | `87.3` |
| `f"{score:.0f}"` | No decimal places | `87` |
| `f"{revenue:,.2f}"` | Thousands separator and two decimals | `15,432.50` |
| `f"{success_rate:.1%}"` | Display a fraction as a percentage | `84.7%` |
| `f"{population:.2e}"` | Scientific notation | `1.40e+09` |
| `f"{name:<15}"` / `f"{score:>8}"` | Left/right alignment | Padded text |
| `input("Name: ")` | Read typed input as a string | User's text |
| `int(text)` / `float(text)` | Convert numeric text | A number |

### Code Snippet: Printing and F-Strings

F-strings put values, labels, and units together: `87.3` is a number; `Score: 87.3%` tells the reader what it means. Choose precision that helps interpretation rather than printing every available digit.

```python
# Basic printing - your daily communication tool
print("Hello world")                    # Basic printing
print("Value:", 42)                     # Multiple values
print("Processing complete!")           # Status updates

# F-string formatting - the data scientist's best friend
student_name = "Alice"
test_score = 87.3
class_average = 82.1

print(f"Student: {student_name}")                    # Basic variable insertion
print(f"Score: {test_score}")                        # Number display
print(f"Score: {test_score:.1f}")                    # One decimal place: 87.3
print(f"Score: {test_score:.0f}%")                   # No decimals: 87%
print(f"Above average by {test_score - class_average:.1f} points")  # Calculations inside f-strings
```

### Formatting Patterns for Data Analysis

```python
# Currency formatting (useful for business data)
revenue = 15432.50
print(f"Revenue: ${revenue:,.2f}")                   # $15,432.50

# Percentage formatting
success_rate = 0.847
print(f"Success rate: {success_rate:.1%}")           # 84.7%

# Scientific notation for very large/small numbers
population = 1400000000
print(f"Population: {population:.2e}")               # 1.40e+09

# Padding and alignment for clean output tables
print(f"{'Name':<15} {'Score':>8} {'Grade':>8}")    # Column headers
print(f"{'Alice':<15} {87.3:>8.1f} {'B+':>8}")      # Left/right aligned data
print(f"{'Bob':<15} {92.1:>8.1f} {'A-':>8}")
```

### Interactive Input

```python
# Interactive input - mainly for testing and debugging
name = input("Enter your name: ")                    # Gets text from user
age_str = input("Enter your age: ")                  # Always returns string!
age = int(age_str)                                   # Convert to number
print(f"Hello {name}, you are {age} years old")

# Be careful: input() always returns strings
user_number = input("Enter a number: ")              # This is text: "42"
print(type(user_number))                             # <class 'str'>
actual_number = float(user_number)                   # Convert to number: 42.0
print(type(actual_number))                           # <class 'float'>

```

## Brief review: scalars, strings, output, and control flow

Lecture 01 introduced these building blocks; focus here on how they support the new topics.

```python
count, average = 150, 87.3
name = "  Alice Johnson  ".strip()
analysis_ready = count > 0 and average > 0

if name.endswith("son") and analysis_ready:
    print(f"{name}: {average:.1f}%")

for number in range(3):
    print(number)
```

Useful reminders: `int`, `float`, `str`, `bool`, and `None` are common scalar types; use arithmetic, comparisons, and `and`/`or`/`not` as needed; strings provide methods such as `.strip()`, `.lower()`, `.split()`, `.replace()`, and `.isalpha()`; and `print()` with f-strings makes results readable. Indentation, `if`/`elif`/`else`, `for`, `while`, `break`, and `continue` control execution.

## Language Semantics and Object Model


Python uses indentation for code structure, creating clean code. Every value is an object with type information, enabling dynamic behavior.

### Reference Card: Language Semantics and Object Model

- Indentation defines code blocks (4 spaces recommended): Group statements inside functions, loops, and conditionals
- `#` for comments: Explain code; Python ignores the rest of that line
- `type(object)`: Get object type
- `isinstance(object, type)`: Type checking
- `id(object)`: Get object identity
- `dir(object)`: List object attributes

### Code Snippet: Language Semantics and Object Model
```python
# Indentation matters
x = 3
if x > 0:
    print("Positive")
    y = x * 2

print(type(42))        # <class 'int'>
print(isinstance("hello", str))  # True
```

## Object Introspection and Dynamic Type Checking

Object introspection examines objects at runtime—their type, attributes, and methods. Valuable for unknown datasets and flexible code.

### Reference Card: Object Introspection and Dynamic Type Checking

- `type(object)`: Returns the object's type
- `dir(object)`: Lists attributes and methods
- `help(object)`: Shows documentation

## Imports and Modules

A **module** is a Python file that provides reusable names. An `import` loads a module and binds a name for it in the current program. Modules in the standard library ship with Python; third-party modules must be installed in the active environment first.

### Reference Card: Imports and Modules

- `import module`: Import a module and use `module.name`
- `import module as alias`: Bind a shorter local name; this does not copy the module
- `from module import name`: Import one specific name

```python
import math
import statistics as stats
from math import pi

print(math.sqrt(16))
print(stats.mean([85, 92, 78]))
print(pi)
```

For a quick import check from a Bash terminal, `-c` runs the Python code supplied as a string:

```bash
python3 -c "import statistics; print(statistics.mean([1, 2, 3]))"
```

## Data Structures: Lists and Tuples

Lists provide mutable sequences for data. Tuples offer immutable sequences useful for fixed records.

### Reference Card: Data Structures: Lists and Tuples

- `list()`: Create list
- `[item1, item2, ...]`: List literal
- `list[index]`: Access one item using a zero-based index
- `list[start:stop]`: Slice from `start` up to, but not including, `stop`
- `list.append(item)`: Add to end
- `list.insert(index, item)`: Insert at position
- `list.remove(item)`: Remove first occurrence
- `list.pop(index)`: Remove and return item
- `tuple()`: Create tuple
- `(item1, item2, ...)`: Tuple literal

### Code Snippet: Data Structures: Lists and Tuples

```python
# Lists - mutable sequences
grades = [85, 92, 78, 96, 88]
grades.append(90)
grades.insert(1, 87)
total = sum(grades)

# Indexing and slicing
first_grade = grades[0]
last_grade = grades[-1]
middle_grades = grades[1:4]  # Indices 1, 2, and 3
every_other_grade = grades[::2]

# Tuples - immutable sequences
coordinates = (40.7128, -74.0060)
name, age, gpa = ("Alice", 22, 3.8)  # Unpacking
```

## Names, Aliasing, and Mutability

Assignment binds a name to an object; it does not automatically copy the object. Lists, dictionaries, and sets are **mutable**, so they can change in place. Numbers, strings, and tuples are **immutable**, so an operation produces a new value instead of changing the existing object.

Two names are **aliases** when they refer to the same object. A mutation through either alias is visible through the other:

```python
grades = [85, 92, 78]
same_grades = grades
same_grades[0] = 90

print(grades)                 # [90, 92, 78]
print(same_grades is grades)  # True: same object

copied_grades = grades.copy()
copied_grades[0] = 75
print(grades)                 # Still [90, 92, 78]
print(copied_grades)          # [75, 92, 78]
```

Use `==` to compare values. Use `is` for object identity, most commonly in a check such as `value is None`. `list.copy()` makes a new outer list; mutable objects nested inside it are still shared.

## Data Structures: Dictionaries and Sets

Dictionaries provide key-value storage for structured data. Sets offer unique collections with mathematical operations.

### Reference Card: Data Structures: Dictionaries and Sets

- `dict()`: Create dictionary
- `{key: value, ...}`: Dictionary literal
- `dict[key]`: Access value
- `dict.get(key, default)`: Safe access
- `dict.keys()`, `dict.values()`, `dict.items()`: Iteration
- `set()`: Create set
- `{item1, item2, ...}`: Set literal
- `set.union()`, `set.intersection()`, `set.difference()`: Set operations

### Code Snippet: Data Structures: Dictionaries and Sets

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

## Sequence Functions

Sequence functions combine, order, and summarize collections. Lecture 03 introduces comprehensions for constructing new lists.

### Reference Card: Sequence Functions

- `enumerate(iterable)`: Get index and value pairs
- `zip(iterable1, iterable2)`: Combine sequences
- `sorted(iterable)`: Create sorted list
- `reversed(iterable)`: Reverse sequence
- `sum()`, `min()`, `max()`, `len()`: Aggregation functions

### Code Snippet: Sequence Functions

```python
# Sequence functions
grades = [85, 92, 78, 96, 88]
for index, grade in enumerate(grades):
    print(f"Student {index + 1}: {grade}")

names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
for name, score in zip(names, scores):
    print(f"{name}: {score}")
```

## Functions

Functions organize code into reusable units with clear interfaces. They enable reuse, testing, and modular design.

### Reference Card: Functions

- `def function_name(parameters): ...`: Function definition
- `return value`: Return value
- Function calls: `result = function_name(arguments)`: Run a function with arguments and store its return value
- Default parameters: `def func(param=default_value):`: Use a fallback value when the caller omits an argument

### Code Snippet: Functions

```python
# Function definition
def calculate_average(grades):
    """Calculate the average of a list of grades."""
    if not grades:
        return 0
    return sum(grades) / len(grades)

# Function usage
grades = [85, 92, 78, 96, 88]
average = calculate_average(grades)
print(f"Average grade: {average:.1f}")
```

## Basic File I/O Operations

File I/O operations are essential for data science. Python provides simple tools for reading and writing files.

### Reference Card: Basic File I/O Operations

- `open(file, mode)`: Open file with specified mode
- `'r'`: Read mode (default)
- `'w'`: Write mode (overwrites existing files)
- `'a'`: Append mode (adds to existing files)
- `'x'`: Create mode (fails if file exists)
- `file.read()`: Read entire file content
- `file.readline()`: Read single line
- `file.readlines()`: Read all lines into list
- `file.write(string)`: Write string to file
- `file.close()`: Close file handle
- `with open(...) as file:`: Close the handle automatically when the block ends

### Code Snippet: Basic File I/O Operations

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
score = 87.3
with open('results.txt', 'w') as file:
    print("Analysis Results", file=file)
    print(f"Average score: {score:.1f}", file=file)

# Avoid print(..., file=open(...)): that pattern leaves closing the file
# handle implicit. Use a with block so the handle is always closed.
with open('debug.log', 'a') as log_file:
    print("Debug info", file=log_file)

# Multiple outputs to same file
data = [85, 92, 78]
with open('report.txt', 'w') as report:
    print("Data Science Report", file=report)
    print("=" * 20, file=report)
    print(f"Total samples: {len(data)}", file=report)
```

## Minimal Exception Handling

An **exception** reports a problem that interrupts normal execution. Use `try`/`except` around the operation that can fail, and catch the specific exception you expect rather than hiding every error.

```python
raw_score = "not available"

try:
    score = float(raw_score)
except ValueError as error:
    print(f"Could not parse score: {error}")
else:
    print(f"Parsed score: {score:.1f}")
```

Opening a missing path raises `FileNotFoundError`; converting invalid numeric text raises `ValueError`. The optional `else` block runs only when the `try` block succeeds.

## `__main__` for script execution

When Python runs a file directly, its special `__name__` variable is set to `"__main__"`. When another file imports it as a module, `__name__` is the module's name. A guard keeps script-only work from running during import:

```python
def main():
    grades = [85, 92, 78, 96, 88]
    average = sum(grades) / len(grades)
    print(f"Average grade: {average:.1f}")


if __name__ == "__main__":
    main()
```

If this is saved as `analysis.py`, the first command runs `main()` and the second only checks that importing the module has no script-only side effects:

```bash
python3 analysis.py
python3 -c "import analysis"
```

# Command-Line Catalog

These are names to recognize from command-line work. The [command-line bonus](BONUS.md#command-line-essentials) has short explanations and examples.

| Area | Commands | Purpose |
| --- | --- | --- |
| Navigation | `pwd`, `ls`, `cd` | Show where you are, list contents, and move between directories. |
| Files and directories | `mkdir`, `touch`, `cp`, `mv` | Create directories or empty files, copy items, and rename or move them. |
| Removal | `rm` | Remove a file; destructive, so check the path first. |
| Inspect and search text | `cat`, `head`, `tail`, `grep`, `wc` | Read, preview, search, and count text. |
| Directory overview | `tree` | Display a directory hierarchy when the command is available. |
| Recall and shortcuts | `history`, ↑/↓, `Tab`, `Ctrl+R` | Reuse earlier commands and complete or search command text. |

For this lecture's project work, use the VS Code terminal and focus on the Git commands introduced above:

```bash
git status
git add path/to/file.py
git commit -m "Describe the change"
git diff
git push
```

Lecture 03 owns the next shell pipeline activity; revisit Lecture 01 for the foundational shell workflow.

# LIVE DEMO!
