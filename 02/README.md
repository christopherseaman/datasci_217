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

# VS Code Basics

## Palette Cleanse: Command Palette & Quick Open

- Open Command Palette: **View → Command Palette…**, **Ctrl+Shift+P** (Windows/Linux), **Cmd+Shift+P** (macOS).
- Quick Open files: **Go → Go to File…**, **Ctrl+P** (Windows/Linux), **Cmd+P** (macOS).
- Search across files: **View → Search**, **Ctrl+Shift+F** (Windows/Linux), **Cmd+Shift+F** (macOS).

## Themes and Schemes: Make it Py‑pretty

- Change Color Theme: Command Palette → **Preferences: Color Theme** (**Ctrl+K** then **Ctrl+T** on Windows/Linux; **Cmd+K** then **Cmd+T** on macOS). I am a fan of:
    - "Tomorrow Night Bright"
    - "GitHub Dark High Contrast"
- Toggle icons: Command Palette → **Preferences: File Icon Theme**.

## Reference Card: Finding Your Way Around

- **Activity Bar** (left): Explorer, Search, Source Control, Run & Debug, and Extensions.
- **Panel** (bottom): Terminal, Problems, Output, and Debug Console; toggle via **View → Appearance → Panel**.
- Split an editor: **View → Editor Layout → Split Right**, **Ctrl+backslash** (**Cmd+backslash** on Mac).
- Start debugging: **Run → Start Debugging**, **F5**.
- Hide distractions: **View → Appearance → Zen Mode**; press **Esc** twice to leave.

Full shortcut list: **Help → Keyboard Shortcuts Reference** ([VS Code reference](https://code.visualstudio.com/docs/reference/default-keybindings)).

## Less Typing, More Doing

Edit and reuse commands in **VS Code's terminal** instead of retyping them.

![VS Code's integrated terminal with command history](media/vscode-integrated-terminal.png)

The terminal accepts the same shell commands as your standalone terminal. Screenshot: [VS Code terminal documentation](https://code.visualstudio.com/docs/terminal/basics).

### Reference Card: Shell Shortcuts

- **Tab:** Complete a command or path; if ambiguous, press again to see choices.
- **↑ / ↓:** Recall previous/next commands.
- **← / →:** Move one character.
- **Ctrl+A / Ctrl+E:** Move to the beginning/end of the line.
- **Ctrl+← / Ctrl+→** (Windows/Linux): Move by word. On Mac, use **Esc**, then **B** or **F**; **Option+← / Option+→** also work when VS Code's **Terminal › Integrated: Mac Option Is Meta** setting is enabled.
- **Ctrl+R:** Search command history; type part of a command, then press again for older matches.
- **Ctrl+W:** Delete the preceding word.
- **Ctrl+K:** Delete from the cursor to the end of the line.
- **Ctrl+L:** Clear the view without deleting command history.

These are Bash/Zsh's usual editing bindings, with the terminal focused. **Ctrl** means Control even on Mac. If word-arrow keys are intercepted, press **Esc**, then **B** or **F**, for backward/forward word movement. Press **Enter** to run the edited command; **Ctrl+C** cancels it. [Shell editing reference](https://www.gnu.org/software/bash/manual/html_node/Readline-Movement-Commands.html).

## Settings

- Settings: Command Palette → **Preferences: Open Settings (UI)**, or **Ctrl+,** (Windows/Linux), **Cmd+,** (macOS). Search **Format on Save** to enable it.
- Python Interpreter: Command Palette → **Python: Select Interpreter**.
- Default Formatter: In Settings, search **Default Formatter** and select an installed formatter such as Black or Ruff.
- Markdown Preview: Right-click an open `.md` editor tab → **Open Preview to the Side**, **Ctrl+K** then **V** (Windows/Linux), **Cmd+K** then **V** (macOS).

![The Python Select Interpreter menu in VS Code](media/vscode-selected-interpreter.png)

Choose the Python 3.13 interpreter installed for the course; this documentation screenshot shows example versions and paths. Source: [VS Code Python environments](https://code.visualstudio.com/docs/python/environments).

## Recommended Extensions (install via View → Extensions)

- Python
- Pylance (Python type information and completion); Ruff (linting and formatting)
- Jupyter (we'll use this a lot later)
- Markdown All in One
- markdownlint
- Markdown Checkboxes
- GitHub Markdown Preview
- Bonus mentions: Error Lens, YAML, indent‑rainbow, GitLens

## Break(points) the Ice

![Python paused at a breakpoint, with the variable's value visible at left](media/vscode-python-debug-paused.png)

The highlighted line runs next. Inspect **Variables**, then step forward to see what changes. Screenshot: [VS Code Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial); the status bar shows the example's older interpreter.

1. Open a `.py` file and click beside a line number to add a breakpoint.
2. Select **Run → Start Debugging** (**F5**).
3. Inspect **Variables**, then **Run → Step Over** (**F10**) to execute one line. Use **Run → Continue** (**F5**) to reach the next breakpoint.

## Command-Line Catalog

These are names to recognize from command-line work. The [command-line bonus](BONUS.md#command-line-essentials) has short explanations and examples.

| Area | Commands | Purpose |
| --- | --- | --- |
| Navigation | `pwd`, `ls`, `cd` | Show where you are, list contents, and move between directories. |
| Files and directories | `mkdir`, `touch`, `cp`, `mv` | Create directories or empty files, copy items, and rename or move them. |
| Removal | `rm` | Remove a file; destructive, so check the path first. |
| Inspect and search text | `cat`, `head`, `tail`, `grep`, `wc` | Read, preview, search, and count text. |
| Directory overview | `tree` | Display a directory hierarchy when the command is available. |
| Recall and shortcuts | `history`, ↑/↓, `Tab`, `Ctrl+R` | Reuse earlier commands and complete or search command text. |

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

Create an experimental branch, commit a change there, then merge it into `main`.

### Reference Card: Git Vocabulary

- **Working tree**: Your current files, including edits not yet committed
- **Diff**: Line-by-line comparison of two versions
- **Staging area**: Selected content for the next commit; later edits need staging again
- **Repository**: Files and their recorded history
- **Commit**: Snapshot with metadata (author, message, parents)
- **Blob**: File content
- **Tree**: Directory structure
- **Reference**: Human-readable name pointing to commit
- **HEAD**: Current commit reference
- **Branch**: Movable reference to commit
- **Remote**: Reference to repository on another machine
- **Synchronize**: VS Code's Sync Changes action pulls incoming commits and pushes outgoing commits

![Git branches split into parallel histories and merge back together](media/git_branches.png)

Each dot is a commit. This illustration calls its main branch `master`; our repositories use `main`.

## VS Code Git Integration

Review a change, stage the files to include, commit the snapshot, then sync it to GitHub.

![Stage a changed file using the plus button in VS Code](media/vscode-stage.png)

![Enter a message and commit the staged files](media/vscode-commit.png)

![Sync committed changes with the GitHub copy](media/vscode-sync.png)

Screenshots: [VS Code source control documentation](https://code.visualstudio.com/docs/sourcecontrol/quickstart).

### Setting Up Git in VS Code

1. Install VS Code (if not already done)
2. Open VS Code → View → Source Control (or **Ctrl+Shift+G**)
3. If Git reports a missing name/email, use the Git identity setup from Lecture 01.

### Reference Card: VS Code Git Actions

- **Source Control Panel**: **View → Source Control**, **Ctrl+Shift+G** (including Control on macOS)
- **Stage Changes**: Click `+` next to files in "Changes" section
- **Commit**: Type a message and select **Commit**
- **View Differences**: Click on modified files to see changes
- **Branch Management**: Click branch name in status bar to switch/create branches
- **Push/Pull**: Select **Sync Changes**, or Command Palette → **Git: Push** / **Git: Pull**

### From Edit to GitHub

```text
Working files → Staged changes → Local commit → GitHub
                 Stage (+)        Commit         Push
```

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

![Git Clone](media/git_clone.png)

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

### Good vs. Bad Commit Messages

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

## Git Workflow: Branching and Merging

Git branching develops features in isolation before merging to main, enabling parallel development and safe experimentation.

In VS Code, open the Command Palette and select **Git: Create Branch**, then name it. After editing and committing, select the branch name in the status bar to switch to `main`. Run **Git: Merge Branch…** and choose the feature branch. **Publish Branch** sends a new branch to GitHub; **Sync Changes** synchronizes an already published branch.

### Reference Card: Git Workflow: Branching and Merging

- `git branch [name]`: Create new branch
- `git checkout [branch]`: Switch to branch
- `git checkout -b [name]`: Create and switch to new branch
- `git merge [branch]`: Merge branch into current branch
- `git branch -d [name]`: Delete branch
- `git push origin [branch]`: Push branch to remote

### Code Snippet: Branching Workflow

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

### Merge Conflict Resolution

When Git cannot automatically combine overlapping changes, it creates a **conflict**:

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

### Create and Edit Files on GitHub

Create repository: github.com → "+" → "New repository" → Name, description, add README → Create.

Add files: "Add file" → "Create new file" → Name, add code, commit message → Commit.

## Gitignore Files

A `.gitignore` file lists patterns for untracked files Git should ignore. Adding a pattern does not untrack files already committed.

### Reference Card: Ignore Patterns

- `# comment`: Explain a pattern
- `*.csv`: Match CSV filenames
- `file?.txt`: Match one character, such as `file1.txt`
- `[abc].txt`: Match `a.txt`, `b.txt`, or `c.txt`
- `!keep.csv`: Re-include a file matched by an earlier pattern
- `**/cache/`: Match cache directories at any depth

### Code Snippet: A Project's `.gitignore`

```gitignore
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

## Example Output
~~~python
print("Analysis complete")
~~~
```

# LIVE DEMO!

# Python Fundamentals (McKinney Ch2+3)

![xkcd 1429, “Data”: a grammar joke contrasting polling data with the Star Trek character Data.](media/xkcd_1429.png)

*Data* by xkcd — in Python, everything is an object. In Star Trek, Data is too.

## Printing and Basic Input

An f-string starts with `f` and inserts the value of each expression inside `{}`. Formatting controls how those values appear; `input()` reads typed text for interactive scripts.

### Reference Card: Printing and Formatting

| Syntax | Purpose | Example output |
| --- | --- | --- |
| `print("Score:", score)` | Print separate values with spaces | `Score: 87.3` |
| `print(text, end="")` | Print without adding a newline, useful when `text` already ends with one | Text unchanged |
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
# F-string formatting - the data scientist's best friend
student_name = "Alice"
test_score = 87.3
class_average = 82.1

print(f"Student: {student_name}")                    # Basic variable insertion
print(f"Score: {test_score:.1f}")                    # One decimal place: 87.3
print(f"Above average by {test_score - class_average:.1f} points")  # Calculations inside f-strings
```

### Code Snippet: Text In, Number Out

```python
raw_score = input("Score: ")       # Typing 87.3 produces the string "87.3"
score = float(raw_score)           # Convert to the number 87.3
print(type(score))                 # <class 'float'>
print(f"Score: {score:.1f}")        # Score: 87.3
```

## More String Operations

### Reference Card: More String Operations

- `text.split(",")`: Split text at commas into a list: `"a,b"` → `["a", "b"]`.
- `"\n".join(lines)`: Combine a list of strings with newlines between them; add `+ "\n"` for a final newline.
- `text.replace("old", "new")`: Return text with matching parts replaced.
- `text.endswith("son")`: Test whether text ends with a suffix, returning `True` or `False`.
- `text.isalpha()`: Test whether nonempty text contains only letters; spaces are not letters.

## Imports and Modules

A **module** is a Python file that provides reusable names. An `import` loads a module and binds a name for it in the current program. Modules in the standard library ship with Python; third-party modules must be installed in the active environment first.

![Python Import](media/python_import.webp)

### Reference Card: Imports and Modules

- `import module`: Import a module and use `module.name`
- `import module as alias`: Bind a shorter local name; this does not copy the module
- `from module import name`: Import one specific name

### Code Snippet: Use a Module

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

Lists are **mutable**: their contents can change. Tuples are **immutable**: their entries cannot be replaced, making them useful for fixed records.

| Position | First | Second | Third | Fourth |
| --- | --- | --- | --- | --- |
| Value | 85 | 92 | 78 | 96 |
| Index | 0 | 1 | 2 | 3 |
| Negative index | -4 | -3 | -2 | -1 |

For this list, `[1:3]` selects `[92, 78]`: start included, stop excluded.

### Reference Card: Data Structures: Lists and Tuples

- `list()`: Create list
- `[item1, item2, ...]`: List literal
- `list[index]`: Access one item using a zero-based index
- `list[start:stop:step]`: Slice up to, but not including, `stop`; omitted bounds use the ends, and `step` defaults to 1
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

## Data Structures: Dictionaries and Sets

Dictionaries provide key-value storage for structured data. Sets offer unique collections with mathematical operations.

```text
Dictionary: "name"  → "Alice"     lookup by key
            "grade" → 85
Set:        {"Math", "Science"}   distinct values, no duplicates
```

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

## Reference Card: Summarize a Collection

For `grades = [85, 92, 78]`:

- `sum(grades)`: Total, `255`.
- `len(grades)`: Number of items, `3`.
- `min(grades)` / `max(grades)`: Smallest/largest value, `78` / `92`.
- `sorted(grades)`: New ordered list, `[78, 85, 92]`; leaves `grades` unchanged.

## Functions

Functions give reusable work a name: pass in **arguments**, receive a **return value**. In the definition, the argument names are called **parameters**.

```text
[85, 92, 78] → calculate_average(grades) → 85.0
  argument         parameter             return value
```

### Reference Card: Functions

- `def function_name(parameters): ...`: Function definition
- `return value`: Send a result back to the caller; without a value, the result is `None`
- Function calls: `result = function_name(arguments)`: Run a function with arguments and store its return value
- Default parameters: `def func(param=default_value):`: Use a fallback value when the caller omits an argument
- `"""Description."""` as the first line inside a function: A docstring describing its purpose and return value
- `if not values:`: An empty collection is false; handle it before dividing by its length
- `value is None`: Test for “no result,” distinct from a numeric zero

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

# LIVE DEMO!

# Files and Reusable Scripts

## Basic File I/O Operations

File **I/O** means input/output: read saved text into Python or write results for later use.

```text
Python text → write → grades.txt → read → saved text
     └──────────── compare with == ───────────┘
```

### Reference Card: Basic File I/O Operations

- `open(file, mode)`: Open file with specified mode
- `encoding="utf-8"`: Use UTF-8 text encoding explicitly when reading or writing
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
with open('grades.txt', 'w', encoding='utf-8') as file:
    for result in results:
        file.write(f"{result}\n")

# Read back and compare
with open('grades.txt', 'r', encoding='utf-8') as file:
    saved_text = file.read()
expected_text = "\n".join(results) + "\n"
print("Saved text matches:", saved_text == expected_text)

# Append with print instead of write; print adds a newline
with open('log.txt', 'a', encoding='utf-8') as file:
    print("Analysis completed", file=file)
```

## Minimal Exception Handling

An **exception** reports a problem that interrupts normal execution. Use `try`/`except` around the operation that can fail, and catch the specific exception you expect rather than hiding every error.

### Code Snippet: Handle Invalid Numeric Text

```python
raw_score = "not available"

try:
    score = float(raw_score)
except ValueError as error:
    print(f"Could not parse score: {error}")
else:
    print(f"Parsed score: {score:.1f}")
```

Opening a missing path raises `FileNotFoundError`; `OSError` covers file-system errors including missing paths and denied permissions. Converting invalid numeric text raises `ValueError`. The optional `else` block runs only when the `try` block succeeds.

## `__main__` for script execution

When Python runs a file directly, its special `__name__` variable is set to `"__main__"`. When another file imports it as a module, `__name__` is the module's name. A guard keeps script-only work from running during import:

### Code Snippet: Run Directly or Import

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

# LIVE DEMO!
