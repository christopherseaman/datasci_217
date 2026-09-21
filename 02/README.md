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

[Lecture 01 catch-up](LECTURE_01_CATCHUP.md) covers material added to Lecture 01 after it was delivered.

# VS Code Basics

In Lecture 01 you opened a folder in VS Code, ran commands in its terminal, and committed with Source Control. Today's work (Git branches, Python scripts that import each other) means more files and more commands, so this section is about moving faster: find any action by name, reuse commands instead of retyping them, and make sure the editor runs the right Python.

The **Command Palette** is the one to remember. Every VS Code action is listed there by name; type part of the name instead of hunting through menus.

## Palette Cleanse: Command Palette & Quick Open

- Open Command Palette: **View → Command Palette…**, **Ctrl+Shift+P** (Windows/Linux), **Cmd+Shift+P** (macOS).
- Quick Open files: **Go → Go to File…**, **Ctrl+P** (Windows/Linux), **Cmd+P** (macOS).
- Search across files: **View → Search**, **Ctrl+Shift+F** (Windows/Linux), **Cmd+Shift+F** (macOS).

### Reference Card: Finding Your Way Around

- **Activity Bar** (left): Explorer, Search, Source Control, Run & Debug, and Extensions.
- **Panel** (bottom): Terminal, Problems, Output, and Debug Console; toggle via **View → Appearance → Panel**.
- Split an editor: **View → Editor Layout → Split Right**, **Ctrl+backslash** (**Cmd+backslash** on Mac).
- Start debugging: **Run → Start Debugging**, **F5**.
- Hide distractions: **View → Appearance → Zen Mode**; press **Esc** twice to leave.

Full shortcut list: **Help → Keyboard Shortcuts Reference** ([VS Code reference](https://code.visualstudio.com/docs/reference/default-keybindings)).

## Themes and Schemes: Make it Py‑pretty

- Change Color Theme: Command Palette → **Preferences: Color Theme** (**Ctrl+K** then **Ctrl+T** on Windows/Linux; **Cmd+K** then **Cmd+T** on macOS). I am a fan of:
    - "Tomorrow Night Bright"
    - "GitHub Dark High Contrast"
- Toggle icons: Command Palette → **Preferences: File Icon Theme**.

## Less Typing, More Doing

Edit and reuse commands in **VS Code's terminal** instead of retyping them.

![VS Code's integrated terminal with command history](media/vscode-integrated-terminal.png)

The terminal accepts the same shell commands as your standalone terminal. Screenshot: [VS Code terminal documentation](https://code.visualstudio.com/docs/terminal/basics).

### Reference Card: Shell Shortcuts

- **Tab:** Complete a command or path; if ambiguous, press again to see choices.
- **↑ / ↓:** Recall previous/next commands.
- **← / →:** Move one character.
- **Ctrl+A:** Move to the beginning of the line.
- **Ctrl+E:** Move to the end of the line. On Windows/Linux, VS Code claims **Ctrl+E** for Quick Open, so press **End** there.
- **Ctrl+← / Ctrl+→** (Windows/Linux): Move by word. On Mac, use **Esc**, then **B** or **F**; **Option+← / Option+→** also work when VS Code's **Terminal › Integrated: Mac Option Is Meta** setting is enabled.
- **Ctrl+R:** Search command history; type part of a command, then press again for older matches.
- **Ctrl+W:** Delete the preceding word.
- **Ctrl+L:** Clear the view without deleting command history.

These are Bash/Zsh's usual editing bindings, with the terminal focused. **Ctrl** means Control even on Mac. If word-arrow keys are intercepted, press **Esc**, then **B** or **F**, for backward/forward word movement. Press **Enter** to run the edited command. Typed something that isn't right? Press **Ctrl+C**: the line is abandoned without running, and you get a fresh prompt to start over. [Shell editing reference](https://www.gnu.org/software/bash/manual/html_node/Readline-Movement-Commands.html).

## Recommended Extensions (install via View → Extensions)

- Python
- Pylance (Python type information and completion); Ruff (linting and formatting)
- Jupyter (we'll use this a lot later)
- Markdown All in One
- markdownlint
- Markdown Checkboxes
- GitHub Markdown Preview
- Bonus mentions: Error Lens, YAML, indent‑rainbow, GitLens

## Settings

- Settings: Command Palette → **Preferences: Open Settings (UI)**, or **Ctrl+,** (Windows/Linux), **Cmd+,** (macOS). Search **Format on Save** to enable it.
- Python Interpreter: Command Palette → **Python: Select Interpreter**.
- Default Formatter: In Settings, search **Default Formatter** and select an installed formatter such as Ruff.

![The Python Select Interpreter menu in VS Code](media/vscode-selected-interpreter.png)

Choose the Python 3.13 interpreter installed for the course; this documentation screenshot shows example versions and paths. Source: [VS Code Python environments](https://code.visualstudio.com/docs/python/environments).

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

In Lecture 01 you already saved a version of your assignment: you staged files with **+**, selected **Commit**, then **Sync Changes**. This section explains what each of those buttons did.

Git records your project as a series of **snapshots**. Each **commit** is one snapshot of every tracked file, plus who made it, when, and a message saying why. Like saving a game: you can always come back to this exact state.

Git does not snapshot whatever happens to be on disk. You choose what goes into each snapshot:

- The **working tree** is your files as they are right now, including edits Git has not recorded.
- The **staging area** holds the changes you selected for the next commit.
- The **repository** is the recorded history of commits, kept in the hidden `.git` folder.

Why the extra step? Suppose you finished fixing a bug in `bp_cleaning.py` and, in the same sitting, started an unfinished draft of `bp_plot.py` that isn't ready to share. Stage `bp_cleaning.py` and commit it; leave `bp_plot.py` unstaged until it earns its own commit.

```text
edit files        select changes       record snapshot       share
working tree  →   staging area     →   local commit      →   GitHub (remote)
               Stage (+)            Commit                 Sync Changes
```

### Reference Card: Git Vocabulary

- **Working tree**: Your current files, including edits not yet committed.
- **Diff**: Line-by-line comparison of two versions; click a changed file in Source Control to see it.
- **Staging area**: Changes selected for the next commit; edit the file again and the new edit needs staging too.
- **Commit**: A snapshot with author, time, and message.
- **Repository (repo)**: Your files plus their recorded history in `.git`.
- **Branch**: A named line of commits; `main` holds the official version.
- **Local branch**: The branch in the repository on your computer; it can be ahead of or behind GitHub.
- **Remote**: The copy of the repository on GitHub, usually named `origin`.
- **Synchronize**: VS Code's **Sync Changes** pulls incoming commits and pushes outgoing ones.
- **Merge**: Combine another branch's commits into the current branch.
- **Conflict**: Both branches changed the same lines, so Git asks you to choose.

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
- **Initialize Repository**: The Source Control button shown in a folder that is not a repository yet; the same as `git init`
- **Stage Changes**: Click `+` next to files in "Changes" section
- **Commit**: Type a message and select **Commit**
- **View Differences**: Click on modified files to see changes
- **Push/Pull**: Select **Sync Changes**, or Command Palette → **Git: Push** / **Git: Pull**

### Good vs. Bad Commit Messages

Type the message in the Source Control box. Start with a short summary that finishes the sentence "This commit will…": `Add blood pressure range check`, not `minor changes`. If it needs more, leave a blank line, then explain why.

![xkcd 1296: Git Commit](media/xkcd_1296.png)

## Git Workflow: Branching and Merging

A **branch** is a separate line of commits. Try a new way to flag abnormal lab values on a branch while `main` keeps the version your team trusts. If the experiment works, **merge** it into `main`; if not, switch back and leave the branch alone.

```text
main        A ─── B ─── E ─────────── M
                   \                 /
feature/…           C ─── D ─────────
```

Each letter is a commit. `C` and `D` exist only on the feature branch; the merge commit `M` brings them into `main`.

### Reference Card: Branches in VS Code

- **Git: Create Branch…** (Command Palette): Name the branch, such as `feature/measurement-summary`; VS Code switches to it.
- Branch name in the status bar (lower left): Shows the current branch; click it to switch branches or to select **Create new branch…**.
- **Publish Branch** (Source Control): Send a new branch to GitHub; afterwards **Sync Changes** keeps it up to date.
- **Git: Merge Branch…** (Command Palette): Combine the chosen branch's commits into the branch you are on.

### VS Code: Branch, Commit, Merge

1. **Git: Create Branch…** → `feature/measurement-summary`. The status bar shows the new name.
2. Edit and save a file, stage it with **+**, and commit.
3. Click the branch name → `main`. Your edit disappears from the editor: it exists only on the feature branch.
4. **Git: Merge Branch…** → `feature/measurement-summary`. The edit is back, now on `main`.
5. **Sync Changes** to send `main` to GitHub.

`main` never gained a commit of its own during these steps, so step 4 is a **fast-forward**: Git just moves the `main` pointer forward to the feature branch's latest commit. No merge commit like `M` in the diagram above appears; those only show up when both branches gained commits before merging.

### Merge Conflicts

A **conflict** happens when both branches changed the same lines, so Git cannot tell which version is right. Git stops the merge, keeps both versions in the file, and marks them:

```text
# Practice notes
<<<<<<< HEAD
Experiment: compare median grades.
=======
Experiment: compare three grade summaries.
>>>>>>> experiment
```

- Between `<<<<<<< HEAD` and `=======`: the branch you are on (**Current Change**).
- Between `=======` and `>>>>>>> experiment`: the branch you are merging in (**Incoming Change**).

1. Open the file listed under **Merge Changes** in Source Control. Above the block, choose **Accept Current Change**, **Accept Incoming Change**, or **Accept Both Changes**, or edit the lines yourself.
2. Check that no `<<<<<<<`, `=======`, or `>>>>>>>` lines remain, then save.
3. Stage the file with **+** and select **Commit** to finish the merge.

## Alternative: Git in the Terminal

Every Source Control button runs a Git command. Demo 1's terminal path uses these; the [bonus page](BONUS.md) covers the rest.

### Reference Card: Git in the Terminal

| Task | Command | Result |
| --- | --- | --- |
| Start a repository | `git init` | A hidden `.git` folder |
| Check state | `git status` | Modified, staged, or "working tree clean" |
| Inspect edits | `git diff` | Unstaged line changes: `+` added, `-` removed |
| Stage | `git add FILE` | FILE's changes go into the next commit |
| Commit | `git commit -m "message"` | A new local commit |
| Create and switch | `git checkout -b NAME` | A new branch at the current commit |
| Switch | `git checkout NAME` | Files change to that branch's version |
| Merge | `git merge NAME` | NAME's commits added to the current branch |
| History | `git log --oneline` | One line per commit; press `q` if the list fills the screen |
| Share | `git push` / `git pull` | Send / receive commits on a branch already linked to GitHub |

## GitHub Web Interface

GitHub's website shows the remote copy. Use it to check what actually arrived after **Sync Changes**, to make a quick edit without cloning, or to upload files as in Lecture 01. An edit made on the website is a commit on the remote, so select **Sync Changes** in VS Code before you keep working locally.

### Reference Card: GitHub Web Interface

- **Code** tab and branch menu: See the files on `main` or any published branch.
- **Add file → Create new file / Upload files**: Commit new files directly on GitHub.
- Pencil icon on a file: Edit it in the browser, then **Commit changes**.
- **Actions** tab: Results of the automatic assignment checks.
- **+ → New repository**: Create an empty remote with a README.

## Gitignore Files

Never commit protected health information (**PHI**), personally identifiable information (**PII**), passwords, or keys. Not once. A commit is permanent: deleting the file later leaves it in every earlier snapshot and every clone, and anyone can read a public repository. PHI work belongs in your institution's approved storage (UCSF has an internal GitHub for it), never on public GitHub.

A **`.gitignore`** file lists **patterns** for files and folders Git should not track. Matching files stay on your computer but never appear under **Changes**, so you cannot stage them by accident. It also hides clutter: running a script that imports your own module creates a `__pycache__/` folder of compiled `.pyc` files.

Adding a pattern does not untrack files already committed.

`git status --short` marks untracked files with `??`. Before adding a `.gitignore`:

```text
?? __pycache__/
?? analysis_utils.py
?? data/
?? main.py
```

After a `.gitignore` containing `__pycache__/`, `*.pyc`, and `data/raw/*.csv`:

```text
?? .gitignore
?? analysis_utils.py
?? main.py
```

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

# LIVE DEMO!

# Python Fundamentals (McKinney Ch2+3)

![xkcd 1429, “Data”: a grammar joke contrasting polling data with the Star Trek character Data.](media/xkcd_1429.png)

*Data* by xkcd — in Python, everything is an object. In Star Trek, Data is too.

Lecture 01 stored one value per variable and looped over a short list of grades. Health data needs more structure: a patient has a list of blood-pressure readings, a visit record has an ID and a date, and a study has a set of clinics that sent data. Python's containers hold these (lists, tuples, dictionaries, sets); f-strings print results people can read; functions name a job you repeat.

## Printing and Basic Input

In Lecture 01, `print("BMI is", bmi)` printed every digit: `BMI is 22.857142857142858`. An **f-string**, a string with `f` before the opening quote, puts values inside the text and controls how they look: `print(f"BMI: {bmi:.1f}")` prints `BMI: 22.9`. The part after the colon is the **format spec**; `.1f` means one digit after the decimal point. `input()` works the other way: it reads what someone types, always as text.

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

F-strings put values, labels, and units together: `87.3` is a number; `Above average by 5.2 points` tells the reader what it means. Choose precision that helps interpretation rather than printing every available digit.

```python
# F-string formatting - the data scientist's best friend
student_name = "Alice"
test_score = 87.3
class_average = 82.1

print(f"Student: {student_name}")                    # Basic variable insertion
print(f"Score: {test_score:.1f}")                    # One decimal place: 87.3
print(f"Above average by {test_score - class_average:.1f} points")  # Calculations inside f-strings
```

```text
Student: Alice
Score: 87.3
Above average by 5.2 points
```

### Code Snippet: Text In, Number Out

```python
raw_score = input("Score: ")       # Typing 87.3 produces the string "87.3"
score = float(raw_score)           # Convert to the number 87.3
print(type(score))                 # <class 'float'>
print(f"Score: {score:.1f}")        # Score: 87.3
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

### Reference Card: Summarize a Collection

For `grades = [85, 92, 78]`:

- `sum(grades)`: Total, `255`.
- `len(grades)`: Number of items, `3`.
- `min(grades)` / `max(grades)`: Smallest/largest value, `78` / `92`.
- `sorted(grades)`: New ordered list, `[78, 85, 92]`; leaves `grades` unchanged.

### Code Snippet: Data Structures: Lists and Tuples

```python
grades = [85, 92, 78, 96]
print(grades[0], grades[-1])   # 85 96
print(grades[1:3])             # [92, 78]
grades.append(88)
print(grades)                  # [85, 92, 78, 96, 88]

visit = ("P001", "2026-09-18")  # a fixed record: patient ID and visit date
patient_id, visit_date = visit  # unpacking
print(patient_id)               # P001
```

## More String Operations

Data often arrives as one line of text per record, such as a row of a CSV file. String methods take the line apart and put it back together.

### Reference Card: More String Operations

- `text.split(",")`: Split text at commas into a list: `"a,b"` → `["a", "b"]`.
- `"\n".join(lines)`: Combine a list of strings with newlines between them; add `+ "\n"` for a final newline.
- `text.replace("old", "new")`: Return text with matching parts replaced.
- `text.endswith("son")`: Test whether text ends with a suffix, returning `True` or `False`.

### Code Snippet: Split a CSV Row

```python
line = "Alice,22,85,Math"
fields = line.split(",")
print(fields)                          # ['Alice', '22', '85', 'Math']
name, age, grade, subject = fields     # unpack the four fields
print(name, int(grade) + 5)            # Alice 90
print("grades.csv".endswith(".csv"))   # True
```

The fields are still text; convert with `int()` before doing arithmetic.

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
- `a & b`, `a | b`, `a - b`: Items in both / either / only the first set; same as `.intersection()`, `.union()`, `.difference()`

### Code Snippet: Data Structures: Dictionaries and Sets

```python
student = {"name": "Alice", "grade": 85}
print(student["name"])                # Alice
print(student.get("gpa", "missing"))  # missing

math_students = {"Alice", "Bob", "Charlie"}
cs_students = {"Alice", "Diana", "Eve"}
print(math_students & cs_students)    # {'Alice'}
```

A set has no order, so print a larger result with `sorted()` when you need the same display every time.

## Functions

Every analysis repeats small jobs: average a patient's readings, find the highest value, format a line for a report. Copy the loop into every script and you will have to fix every copy when you find a bug. A **function** gives the job a name so you write it once and call it everywhere. Pass values in as **arguments** (inside the definition they are called **parameters**) and get back a **return value**. A function that reaches its end without `return` gives back `None`, Python's value for "nothing here."

```text
[118, 124, 130] → mean_reading(readings) → 124.0
    argument          parameter           return value
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
def mean_reading(readings):
    """Return the average reading, or None when there are no readings."""
    if not readings:
        return None
    return sum(readings) / len(readings)

print(mean_reading([118, 124, 130]))  # 124.0
print(mean_reading([]))               # None
print(mean_reading([0, 0]))           # 0.0
```

Why `None` instead of `0`? A mean of 0 can be real (zero steps recorded); `None` says there was nothing to average. `if not result:` treats `0.0` as missing too, so test for missing values with `result is None`.

## Imports and Modules

A **module** is a Python file that provides reusable names. An `import` loads a module and binds a name for it in the current program. Modules in the standard library ship with Python; third-party modules must be installed in the active environment first.

![Meme: Java insists you write your own code; Python replies from python.goes import brrrrr](media/python_import.webp)

### Reference Card: Imports and Modules

- `import module`: Import a module and use `module.name`
- `import module as alias`: Bind a shorter local name; this does not copy the module
- `from module import name`: Import one specific name

### Code Snippet: Use a Module

```python
import math
import statistics as stats
from math import pi

print(math.sqrt(16))              # 4.0
print(stats.mean([85, 92, 78]))   # 85
print(pi)                         # 3.141592653589793
```

For a quick import check from a Bash terminal, `-c` runs the Python code supplied as a string:

```bash
python3 -c "import statistics; print(statistics.mean([1, 2, 3]))"   # 2
```

### Code Snippet: Import Your Own Module

Any `.py` file is a module. Save a helper in `student_tools.py`:

```python
# student_tools.py
def calculate_average(grades):
    return sum(grades) / len(grades)
```

Import it from another script in the same folder:

```python
# report.py
from student_tools import calculate_average

print(calculate_average([85, 92, 78]))  # 85.0
```

Run `python3 report.py` from that folder. Python finds `student_tools.py` because it sits beside the script; the module name is the filename without `.py`. The first import also creates a `__pycache__/` folder of compiled files, which your `.gitignore` keeps out of Git.

## Break(points) the Ice

![Python paused at a breakpoint, with the variable's value visible at left](media/vscode-python-debug-paused.png)

The highlighted line runs next. Inspect **Variables**, then step forward to see what changes. Screenshot: [VS Code Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial); the status bar shows the example's older interpreter.

1. Open a `.py` file that calls a function and click left of a line number inside that function to add a breakpoint (a red dot).
2. Select **Run → Start Debugging** (**F5**).
3. Inspect **Variables**, then **Run → Step Over** (**F10**) to execute one line. Use **Run → Continue** (**F5**) to reach the next breakpoint.

# LIVE DEMO!

# Files and Reusable Scripts

Variables disappear when a script ends. To keep a result (a summary for your PI, a log of what ran), write it to a file and read it back to confirm what was saved. To reuse a script's functions elsewhere, make it safe to import, and document how to run it in a README. This block builds on Demo 2's helpers: save results to a file, check what was saved, handle bad input values without crashing, keep imports quiet, and write the command that runs the script.

## Basic File I/O Operations

File **I/O** means input/output: read saved text into Python or write results for later use. `open()` returns a **file handle**, Python's connection to the file; a `with` block closes it for you. Mode `"w"` replaces the whole file, so double-check the name.

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

### Code Snippet: Write, Read Back, Append

```python
results = ["Alice: 95", "Bob: 87", "Charlie: 92"]

# Write: "w" creates grades.txt, or replaces it if it exists
with open("grades.txt", "w", encoding="utf-8") as file:
    for result in results:
        file.write(f"{result}\n")

# Read back and compare
with open("grades.txt", "r", encoding="utf-8") as file:
    saved_text = file.read()
print(saved_text, end="")
expected_text = "\n".join(results) + "\n"
print("Saved text matches:", saved_text == expected_text)

# Append: "a" adds to the end; print(..., file=file) adds the newline
with open("log.txt", "a", encoding="utf-8") as file:
    print("Analysis completed", file=file)
```

```text
Alice: 95
Bob: 87
Charlie: 92
Saved text matches: True
```

`log.txt` gains one line every time you run the script: `"a"` never removes what is already there.

## Minimal Exception Handling

In Lecture 01, `int("hello")` stopped the script with a `ValueError` traceback. Real data has entries like `"not available"` in a numeric column. An **exception** is Python's report of that kind of problem; `try`/`except` lets your script respond instead of stopping. Catch only the exception you expect, so real bugs still show up.

### Reference Card: Exceptions You Will Meet

- `ValueError`: Right type, unusable value, such as `float("not available")`.
- `FileNotFoundError`: `open()` on a path that does not exist.
- `OSError`: Any file-system failure, including missing paths and denied permissions.
- `try:` / `except ValueError as error:`: Run the risky line; on that error only, run the handler with the message in `error`.
- `else:`: Runs only when the `try` block succeeded.

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

```text
Could not parse score: could not convert string to float: 'not available'
```

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

The first prints `Average grade: 87.8`. The second prints nothing: importing ran the `def` but skipped `main()`.

## Document How to Run It

Every repository needs a note that says what it is and how to run it. On GitHub that note is `README.md`, shown below the file list on the repository's front page. The `.md` means **Markdown**: plain text with a few symbols that mark formatting. The raw file stays readable in any editor, and GitHub, VS Code's preview, Notion, and the course site all render it as formatted text.

### Reference Card: Markdown Documentation

- `# Title`, `## Section`, `### Subsection`: Headings, from largest to smallest.
- `**bold text**`: Strong emphasis, shown as **bold text**.
- `*italic text*`: Emphasis, shown as *italic text*.
- Single backticks around text, such as `mean()`: Code within a sentence, shown in code font.
- Three backticks on their own line, optionally followed by a language name such as `bash`: Start a code block; close it with three backticks.
- `~~~` on its own line: Also opens and closes a code block. The snippet below uses it because the whole example already sits inside a backtick code block; in your own files either works.
- `- item` or `1. item`: Bulleted or numbered list.
- `[text](url)`: Clickable link.
- `![alt](url)`: Image with descriptive alternative text.
- `| col1 | col2 |`: Table row; follow the header row with a separator row such as `| --- | --- |`.
- Right-click an open `.md` editor tab → **Open Preview to the Side**, or **Ctrl+K** then **V** (**Cmd+K** then **V** on Mac): Open the rendered preview beside the file.

### Code Snippet: Markdown Documentation

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

In the preview this becomes a large title, three section headings, a short paragraph, a two-item bulleted list, and a shaded code block.

# LIVE DEMO!
