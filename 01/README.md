---
notion:
  role: lecture
  status: mapped
  page_id: "271d9fdd-1a1a-8057-84e1-fe68dc985696"
  url: "https://app.notion.com/p/271d9fdd1a1a805784e1fe68dc985696"
---

**Quick references**

[DLC](BONUS.md)

[Live Demo!](demo/DEMO_GUIDE.md)

- [Command-line (Bash) cheat sheet](https://cheatsheets.zip/bash)
- [Python cheat sheet](https://cheatsheets.zip/python)
- [futurecoder](https://futurecoder.io/) — Python basics with in-browser exercises and feedback
- [Official Python tutorial](https://docs.python.org/3/tutorial/) — tutorials straight from the source

<callout icon="🌉" color="green_bg">
	#### *San Francisco is a walkable city and I will literally die on this hill*
</callout>

# Class Structure

This course started as a Python introduction plus as much of the practical stuff I learned on the job but never in a course as I could fit. Halfway through preparing the first version, I found [The Missing Semester](https://missing.csail.mit.edu/). Apparently, I wasn't the only one who noticed the gap.

- **Lectures** cover new material
- **Assignments** after each lecture (caveats apply)
- **Lab** for hands-on help completing the practical assignment
- **Assignments (60%)** are always due the following week unless otherwise noted
- **Two exams (40%)** or just one for 1-unit course at weeks 5 and 11

# Getting Started: Your First Steps

## Getting to the Command Line

![learning to code is kind of like this](media/rocket_packs.png)

The shell examples in this lecture use POSIX commands in Bash (or a compatible shell). On Windows, WSL gives you that environment; native PowerShell uses different commands and syntax in several places.

Use [VS Code](https://code.visualstudio.com/)'s **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac) for course commands. Open your project folder first; the terminal starts there. On Windows, use the terminal dropdown's **Select Default Profile → Ubuntu (WSL)**, then open a new terminal. If WSL is not installed yet, use the setup step below.

The native terminal apps below are alternatives and can also handle initial installation before VS Code is ready. Use `cd` to enter your project folder when working in a separate terminal.

### Windows Users

**WSL:**

- **Windows Subsystem for Linux (WSL)** (recommended): Run `wsl --install` in PowerShell as Administrator

Native Windows:

- **PowerShell** (built-in): Press `Win + X`, then select "Terminal" or "Windows PowerShell." Use it to install WSL; use the Ubuntu terminal for the course's Bash commands.
- **GitHub Codespaces** (cloud option): No installation needed

### Mac Users

- **Terminal** (built-in): Press `Cmd + Space`, type "Terminal", press Enter
- **GitHub Codespaces** (cloud option): No installation needed

### Cloud Options

- **GitHub Codespaces**: Free tier available, works on any device with internet

## Installing Python

### Windows WSL (Ubuntu)

Use uv's installer to get the course Python version. The first command installs uv; the next commands install Python 3.13 and put it on your shell's PATH. We'll use uv to manage project environments in Lecture 03.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.13 --default
uv python update-shell
```

Open a new terminal before verifying Python below.

### Windows Native

```powershell
# Option 1: Official installer from python.org
# Download Python 3.13.x from <https://python.org>

# Option 2: Using winget (Windows Package Manager)
winget install -e --id Python.Python.3.13
```

### Mac

After installing Homebrew, run the PATH setup commands printed under **Next steps** so your shell can find `brew`. Use the commands shown for your machine, not someone else's home-directory path.

```bash
# Option 1: Using Homebrew (recommended)
# First install Homebrew from <https://brew.sh>
brew install python@3.13
export PATH="$(brew --prefix python@3.13)/libexec/bin:$PATH"

# Option 2: Official installer from python.org
# Download Python 3.13.x from <https://python.org>
```

For Homebrew, add the `export PATH=...` line to your shell startup file (`~/.zshrc` on a default macOS setup) so new terminals also use Python 3.13.

### Verify Installation

```bash
# WSL, macOS, or Codespaces
python3 --version
# Should show: Python 3.13.x
```

In native Windows PowerShell, use `py -3.13 --version`. Until we activate a virtual environment later in the course, Bash examples use `python3`; native PowerShell users should substitute `py -3.13`. Inside an activated environment, `python` will refer to that environment's interpreter.

## Text Editor Options

### Visual Studio Code (Recommended)

- Free, powerful, and perfect for data science
- Available on all platforms
- Python support through Microsoft's Python extension
- Can open files from command line with `code filename.py`

### Other Editors

- **Sublime Text**: Fast and lightweight
- **PyCharm**: Full-featured Python IDE
- **nano**: Simple command-line editor for quick fixes
- **Vim / Neovim**: Modal terminal editors; use `vim filename.py` or `nvim filename.py`. Press `i` to insert text, then `Esc` and `:wq` to save and quit.

### Why VS Code?

![IDE Choice Guidance](media/IDE_choice.png)

We'll use VS Code for its editor, integrated terminal, debugger, and Git interface.

### VS Code Basics

Open the assignment folder with **File → Open Folder** so the editor, terminal, and Source Control all use the same project.

![VS Code's labeled interface showing the Activity Bar, Primary Side Bar, editor, Panel, and Status Bar.](media/vscode-workspace.png)

- Open or create a file: **Explorer** at left; click a filename or the **New File** icon. Keep `.py` on Python filenames.
- Edit and save: Type in the editor; **File → Save** or Ctrl+S (Cmd+S on Mac). Save before running.
- Enable Python support: **Extensions** at left; install **Python** by Microsoft.
- Choose Python: Command Palette → **Python: Select Interpreter**; select the installed Python 3.13.
- Run a command: **Terminal → New Terminal**, or Ctrl+Shift+backtick (also Control on Mac); type the command at the prompt and press Enter.
- Find an editor action: **View → Command Palette**, or Ctrl+Shift+P (Cmd+Shift+P on Mac); type its name, such as `Git: Clone`.
- Review changed files: **View → Source Control**, or Ctrl+Shift+G (also Control on Mac); click a file to see its changes.

The editor changes files; the terminal runs commands. Saving a file does not run it or upload it to GitHub. Screenshot: [VS Code interface](https://code.visualstudio.com/docs/editing/getting-started/userinterface). **Help → Keyboard Shortcuts Reference** lists your platform's [default shortcuts](https://code.visualstudio.com/docs/reference/default-keybindings).

## Starting with GitHub

### Creating Your GitHub Account

#### Account Setup

1. Go to [github.com](http://github.com/)
2. Sign up with your UCSF email (or personal email)
    - Use your actual UCSF email so I can find you, or not
    - You can always add/remove email addresses later
3. Choose a professional username (you'll use this for years!)
4. Verify your email address

**Username Tips:**

- Use your name or initials: `alice-smith`, `asmith-the-best-one-ever`
- Avoid hard-to-remember numbers: `alice_smith_9847`
- Keep it professional? - future employers will see this
- You can change it later, but links might break

GitHub Student Pack (Optional Bonus) With your .edu email, you can get free premium features. We don't need them for class, but they're nice to have!

### DON'T USE YOUR REAL EMAIL IN GIT CONFIG

You don't want to put your email all over the public internet, so GitHub provides a proxy service. You can see the proxy email address in your [GitHub email settings https://github.com/settings/emails](https://github.com/settings/emails).

![GitHub Email Setup](media/github_email.png)

### Setting Up Git in VS Code

Install [Git](https://git-scm.com/downloads) if VS Code reports it missing, then restart VS Code.

Sign in to GitHub through VS Code when **Clone from GitHub** or **Sync Changes** prompts you: choose **Sign in with GitHub**, authorize in your browser, then return to VS Code.

If a commit reports a missing name or email, open **Terminal → New Terminal** in your cloned folder and run these once, using your GitHub `noreply` email:

```bash
git config user.name "Your Name"
git config user.email "YOUR GITHUB NOREPLY EMAIL"
```

GitHub login authorizes access to your repositories; these settings identify the author of your commits.

## Getting the First Assignment

Lecture 02 explains Git concepts but here’s what you’ll need to complete the first assignment

A **fork** is your copy on GitHub; a **clone** is the working copy on your computer. Follow these steps to start Assignment 01.

### Fork on GitHub

1. Open the assignment repository linked for this term, sign in, and select **Fork**.

![GitHub's Fork button](assignment/media/github-fork.png)

2. Choose **your account** as Owner, keep the repository name and **Copy the main branch only** checked, then select **Create fork**.

![GitHub's Create a new fork form: select your account as Owner, keep the repository name, and click Create fork.](assignment/media/github-create-fork.png)

### Clone Your Fork in VS Code

1. On **your fork**, select **Code → HTTPS** and copy the URL. The owner in the URL should be your GitHub username.

![Copy your fork's HTTPS URL from the Code menu](assignment/media/github-clone-url.png)

2. In VS Code, open **View → Command Palette** (Ctrl+Shift+P; Cmd+Shift+P on Mac), choose **Git: Clone**, paste that URL, choose a folder on your computer, and open the cloned repository. Sign in if prompted.

![VS Code's Clone from URL prompt](assignment/media/vscode-clone.png)

The screenshots use example repositories; paste your own fork's URL. Keep your assignment work in this folder.

## Submit Your Assignment Files

A **commit** saves a version; **push** sends local commits to GitHub. To submit the first assignment, use **VS Code to commit and push**, or **the GitHub website to upload and commit directly**, as shown below.

### VS Code: Commit and Sync

1. Save your files. Open **Source Control** and click each changed file to review it. Stage the completed scripts and checkpoint artifacts with **+**; for Assignment 01, include both files in `terminal-practice/` and both in `output/`.

![VS Code Source Control with the plus button highlighted to stage a file.](assignment/media/vscode-stage.png)

2. Enter a message such as `Complete Assignment 01`, then select **Commit**.

![VS Code's message field and Commit button above the staged changes.](assignment/media/vscode-commit.png)

3. Select **Sync Changes** to send your commit to your fork. Sign in to GitHub if prompted.

![VS Code's Sync Changes button highlighted.](assignment/media/vscode-sync.png)

### GitHub Website: Upload Files

1. Open **your fork** on GitHub, on `main`. Select **Add file → Upload files**.

![GitHub's Add file menu with Upload files highlighted.](assignment/media/github-upload-files.png)

2. Drag in the completed scripts and the `terminal-practice` and `output` folders. Keep the folders intact so paths such as `output/readiness.txt` stay correct. Upload the assignment files, not the whole project folder.
3. Enter `Complete Assignment 01` or another description, choose **Commit directly to the main branch**, and click **Commit changes**. Web upload commits directly on GitHub; no separate push is needed.

### Verify on GitHub

Open your fork's `output/readiness.txt` and `output/student_identity.txt` and check their contents. Open **Actions** to see the automatic checks; enable workflows once if prompted in a new fork. Your fork is the submission—no pull request to the course repository.

# LIVE DEMO!

# Why Both Python and Command Line?

Professional data scientists switch constantly between Python scripts and command-line operations: Python analyzes data; the command line organizes files, runs scripts, and manages projects.

It's like being bilingual in the data world. Python speaks to your data, command line speaks to your computer.

**Reality check:** Organizing files, inspecting data, and explaining results are part of the analysis—not chores you finish before the “real” work starts.

# Command Line Essentials

## What is the Command Line?

The **command line (CLI)** is a text-based interface.

Think of it as texting your computer instead of playing charades with icons.

- **Terminal:** The app displaying the session—Windows Terminal, macOS Terminal, or VS Code's terminal.
- **Shell:** The command interpreter running inside it—Bash, Zsh, or PowerShell.
- **Directories** are folders; **paths** locate files and folders.
- **Working directory (cwd):** Where your shell is now. `pwd` shows it; `cd` changes it.
- **Paths:** `/home/alice/data.csv` is absolute; `data/data.csv` is relative to cwd.
    - `.` = current directory
    - `..` = parent directory
    - `~` = home directory
- **Command + parameters:** `ls -l data` = list, detailed option, target folder. Separate parts with spaces; quote paths containing spaces.

![Unix System Reference](media/its-a-unix-system.jpeg)

## Navigation Commands

### Reference Card: Navigation Commands

- `pwd`: Print working directory (where am I?)
- `ls`: List contents (what's here?)
- `ls -la`: List with details (show me everything)
- `cd [path]`: Change directory (go somewhere)
- `cd ..`: Go up one level
- `cd ~`: Go to home directory

### Code Snippet: Navigation Commands

```bash
pwd                    # Shows: /Users/yourname
ls                     # Shows files in current directory
cd Documents           # Move to Documents folder
pwd                    # Shows: /Users/yourname/Documents

```

## File and Directory Operations

### Reference Card: File and Directory Operations

- `mkdir [name]`: Make directory
- `mkdir -p [path/to/nested]`: Make nested directories
- `touch [filename]`: Create empty file
- `cp [source] [destination]`: Copy file
- `mv [source] [destination]`: Move/rename file
- `rm [filename]`: Remove file (careful!)
- `rm -r [directory]`: Remove directory and contents (very careful!)

### Code Snippet: Extended Examples for Data Science Workflows

```bash
# Create a typical data science project structure
mkdir data-science-project
cd data-science-project
mkdir data scripts results docs

# Create placeholder files for our project
touch data/raw_data.csv
touch scripts/analysis.py
touch docs/project_notes.md

# View our project structure
ls -la
# You'll see: data/ scripts/ results/ docs/ and our files

# Copy important files to backup location
cp data/raw_data.csv data/raw_data_backup.csv

# Rename a file to be more descriptive
mv scripts/analysis.py scripts/customer_analysis.py

```

### Code Snippet: File and Directory Operations

```bash
mkdir my_data_project     # Create project folder
cd my_data_project        # Enter the folder
touch analysis.py         # Create Python file
mkdir data                # Create data subfolder

```

### Reference Card: Shell Output

- `echo "text"`: Print a line of text.
- `>`: Send command output to a file, replacing its contents.
- `>>`: Append command output to a file.

```bash
echo "# Practice notes" > notes.md
echo "Experiment: compare two grade summaries." >> notes.md
cat notes.md
```

## Viewing Files

### Reference Card: Viewing Files

- `cat [filename]`: Show entire file contents
- `head [filename]`: Show first 10 lines
- `head -n 5 [filename]`: Show first 5 lines
- `tail [filename]`: Show last 10 lines
- `tail -n 20 [filename]`: Show last 20 lines

### Code Snippet: Viewing Files

```bash
head data.csv           # Quick peek at data file
tail -n 5 results.txt   # See the last few results

```

## Create a Script by Pasting

1. Run `cat > file.sh` in your shell. `>` replaces that file if it exists.
2. Paste the script, press **Enter** if needed to reach a new line, then **Ctrl+D** to finish input (EOF, end of file).
3. Inspect with `cat file.sh`, then run with `bash file.sh`.

Enter ends a line; Ctrl+D finishes input. Ctrl+C interrupts `cat`; text already written remains.

## Getting Help

### Reference Card: Getting Help

- `man [command]`: Manual page for command
- `[command] --help`: Quick help for command
- `which [command]`: Find where command is located

Other help: books, your favorite LLM, a buddy, or the course EAs and instructor.

## Ctrl+C: Make it Stop!

- **Build the reflex: Ctrl+C to cancel** a running command or unfinished input. On Mac, Control—not Command.
- **Windows habit to unlearn:** Ctrl+C interrupts here, not copies; terminal copy is often Ctrl+Shift+C.

# LIVE DEMO!

# Python Basics

![xkcd_353.png](media/xkcd_353.png)

## Running Python

In VS Code, save a `.py` file and click the triangle at its top right to run it in the integrated terminal.

![Run a Python file with VS Code's triangle button](media/vscode-run-python-file.png)

For Lectures 01–03, we will use two Python modes:

1. **Interactive mode** (REPL): Type `python3` and start experimenting
2. **Script mode**: Write code in a file, run with `python3 filename.py`

**Jupyter notebooks** are another way to run Python, but we'll meet them later.

### Code Snippet: Running Python

```bash
python3                 # Start interactive Python
python3 script.py       # Run a Python script
```

These are Bash commands. In native Windows PowerShell, substitute `py` for `python3`. At the Python `>>>` prompt, enter `exit()` to leave the REPL.

**Interactive Mode Example:**

```console
$ python3
>>> print("Hello, World!")
Hello, World!
>>> exit()
```

**Script Mode Example:**

```bash
python3 my_script.py
```

## Python Syntax Overview

### Indentation Matters!

<callout icon="⚠️" color="green_bg">
	Python uses indentation to group code together.
	**Recommendation: Use four spaces per indentation level.**
</callout>

This is a preview of an `if` conditional; the Control Structures section below explains how the condition works.

```python
# Correct indentation
x = 1
if x > 0:
    print("Positive")    # This line is indented
    print("Still positive")  # This line is also indented
```

```text
# Wrong indentation (will cause an error)
if x > 0:
print("This will cause an IndentationError")
```

To fix the second example, indent the `print()` line four spaces beneath `if`, as in the first example.

### Comments Use `#`

```python
# This is a comment - Python ignores this line
print("This is code")  # Comments can also go at the end of lines
```

### Reference Card: Python Syntax

- Use 4 spaces for indentation (not tabs)
- No semicolons needed at the end of lines (but you can have them if you REALLY want them)
- Case-sensitive: `Name` and `name` are different variables
- Use quotes for strings: `"Hello"` or `'Hello'`

## Variables and Data Types

Python stores information in variables - think of them as labeled boxes that you can put different types of information in.

### Reference Card: Values and Types

| Value / operation | Meaning | Example |
| --- | --- | --- |
| `int` | Whole number | `22` |
| `float` | Number with a decimal part | `87.5` |
| `str` | Text in quotes | `"Alice"` |
| `bool` | True or false | `True` |
| `type(value)` | Inspect a value's type | `type(22)` → `<class 'int'>` |
| `name = value` | Assign a value to a name | `age = 22` |

### Numbers - The Foundation of Data Science

```python
# Integers (whole numbers)
student_count = 150
year = 2024
temperature_celsius = -5

# Floats (decimal numbers)
average_grade = 87.3
height_meters = 1.75
pi_approximation = 3.14159

# Scientific notation for very large/small numbers
population = 1.4e9          # 1.4 billion
atom_mass = 1.67e-27        # Very small number
```

### Text - Essential for Data Labels and Categories

#### Reference Card: Strings

- `text.upper()` / `text.lower()`: return uppercase / lowercase text.
- `text.title()`: return text with each word capitalized.
- `text.strip()`: remove leading and trailing whitespace.
- `len(text)`: count characters, including spaces.
- String methods return new text; assign the result to keep it.

```python
# Strings for text data
student_name = "Alice Johnson"
department = "Data Science"
file_path = "/Users/alice/projects/analysis.py"

# String methods you'll use constantly
name_upper = student_name.upper()        # "ALICE JOHNSON"
name_lower = student_name.lower()        # "alice johnson"
name_title = student_name.title()        # "Alice Johnson"
clean_name = "  Bob Smith  ".strip()     # Removes whitespace: "Bob Smith"
```

### Boolean - Essential for Data Filtering

#### Reference Card: Boolean Logic

- `True` / `False`: the two Boolean values.
- `a and b`: true when both are true.
- `a or b`: true when at least one is true.
- `not a`: reverse a Boolean value.

```python
# True/False values for logical operations
has_complete_data = True
missing_values = False
analysis_ready = True and has_complete_data    # True
needs_cleaning = missing_values or not analysis_ready  # False
```

### Variable Naming Best Practices

```python
# Good variable names (descriptive and clear)
student_age = 22
average_test_score = 85.7
data_file_path = "student_grades.csv"

# Avoid these (unclear or confusing)
a = 22                  # What does 'a' represent?
x1 = 85.7              # Meaningless variable name
temp = "grades.csv"     # 'temp' usually means temporary
```

### Understanding Variable Types (Debugging Foundation)

```python
# Check what type a variable is (essential for debugging!)
student_name = "Alice"
student_age = 22
grade_average = 87.5

print(type(student_name))    # <class 'str'>
print(type(student_age))     # <class 'int'>
print(type(grade_average))   # <class 'float'>

# This is crucial when data doesn't behave as expected!
mysterious_data = "22"       # Looks like a number, but it's text
print(type(mysterious_data)) # <class 'str'> - Aha! That's the problem
```

![Duck Typing](media/duck_typing.jpg)

### Duck Typing: Behavior Over Labels

Python is dynamically typed: a variable can refer to values of different types, and code often cares more about what an object can do than what type it is. If it walks like a duck and quacks like a duck, Python lets us treat it like a duck.

```python
label = "dataset"
grades = [85, 92, 78]

print(len(label))   # 7
print(len(grades))  # 3
```

Both objects support `len()`. Python checks the operation when it runs; unsupported operations raise `TypeError`.

![Duck typing animation](media/duck_typing_animation.gif)

## Basic Operations

### Reference Card: Plain Output

| Syntax | Purpose | Output |
| --- | --- | --- |
| `print("Hello")` | Display text | `Hello` |
| `print(2 + 3)` | Display a calculation | `5` |
| `print("Score:", 85)` | Display a label and value | `Score: 85` |

Use `print()` to display a value or several values separated by commas. Python puts spaces between them.

### Reference Card: Arithmetic

- `+`, `-`, `*`: add, subtract, multiply. For strings, `+` joins text.
- `/`: divide; `15 / 4` gives `3.75`.
- `//`: floor division; `15 // 4` gives `3`.
- `%`: remainder; `15 % 4` gives `3`.
- `**`: power; `2 ** 3` gives `8`.
- `(...)`: group an expression to control calculation order.

### Code Snippet: Arithmetic and Strings

```python
# Math operations
result = 10 + 5         # Addition: 15
result = 10 - 3         # Subtraction: 7
result = 4 * 6          # Multiplication: 24
result = 15 / 4         # Division: 3.75
result = 15 // 4        # Integer division: 3
result = 15 % 4         # Remainder: 3
result = 2 ** 3         # Power: 8

# String operations
first = "Ada"
last = "Lovelace"
name = first
full_name = first + " " + last        # Concatenation
print("Hello", name)                 # Print text and a value
```

### Code Snippet: Calculate BMI

```python
# Calculate BMI
weight_kg = 70
height_m = 1.75
bmi = weight_kg / (height_m ** 2)
print("BMI is", bmi)
```

![xkcd 1654: Universal Install Script](media/xkcd_1654.png)

# LIVE DEMO!

## Control Structures

Control structures let your programs make decisions and repeat actions - essential for data analysis!

### Reference Card: Decisions and Repetition

- `if` / `elif` / `else`: Choose which block runs based on a condition
- `for value in values:`: Visit each item in order
- `range(5)`: Supply integers 0 through 4
- `while condition:`: Repeat while the condition stays true
- `enumerate(values, start=1)`: Supply each position and value
- `break` / `continue`: Stop a loop / skip to its next iteration

### Comparison Operators

#### Reference Card: Comparisons

- `==` / `!=`: equal / not equal. Unlike `=`, these compare rather than assign.
- `<`, `<=`, `>`, `>=`: less than, at most, greater than, at least.
- `in` / `not in`: test whether a value belongs to a collection.
- Each comparison returns `True` or `False`.

#### Code Snippet: Comparison Operators

```python
# Equality and inequality
x = 2
y = 3
x == y          # Equal to
x != y          # Not equal to
x < y           # Less than
x > y           # Greater than
x <= y          # Less than or equal
x >= y          # Greater than or equal

# Membership testing
x in [1, 2, 3]  # Is x in the list?
x not in [1, 2, 3]  # Is x NOT in the list?
```

Square brackets create a **list**, an ordered collection. The loop examples below introduce the list operations needed today; Lecture 02 develops indexing and slicing.

### If Statements

#### Code Snippet: Basic If Statements

```python
# Simple decision making
score = 85

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
else:
    print("Grade: F")
```

#### Code Snippet: Compound Conditions

```python
# Multiple conditions with and/or
age = 25
has_license = True

if age >= 18 and has_license:
    print("Can drive")
elif age >= 16 and not has_license:
    print("Can learn to drive")
else:
    print("Cannot drive")
```

### For Loops

`range(5)` supplies the integers from 0 through 4. A list supplies its items in order; Lecture 02 covers lists in more depth.

#### Code Snippet: Basic For Loops

```python
# Count from 0 to 4
for i in range(5):
    print("Count:", i)

# Loop through a list
grades = [85, 92, 78, 96, 88]
for grade in grades:
    print("Grade:", grade)
```

#### Code Snippet: Practical Data Science Example

```python
# Calculate average grade
grades = [85, 92, 78, 96, 88]
total = 0
count = 0

for grade in grades:
    total += grade
    count += 1

average = total / count
print("Average grade:", average)
```

### While Loops and Loop Control

A `while` loop repeats as long as its condition is `True`. Update the loop variable inside the loop so it can eventually finish:

```python
count = 1
while count <= 3:
    print("Count:", count)
    count += 1
```

When a loop needs both a position and a value, `enumerate()` supplies them:

```python
grades = [85, 92, 78]
for position, grade in enumerate(grades, start=1):
    print("Assignment", position, "grade:", grade)
```

Use `break` to stop a loop early, and `continue` to skip the rest of the current iteration and move to the next item:

```python
for grade in grades:
    if grade < 80:
        continue
    print("Processing", grade)
    if grade >= 90:
        break
```

## Debugging and Error Handling Basics

![Programming is doing something wrong over and over until you do something right](media/it_works.png)

### Reading a Traceback

An error reports where execution stopped and what operation failed—not necessarily the underlying cause.

1. Read the final line for the error type and message.
2. Find the referenced line in your script.
3. Inspect the relevant values with `print()` and their types with `type()`.
4. Make one correction, save, and rerun. The next error may only become visible after this one is fixed.

### Reference Card: Inspecting and Converting Values

- `print(value)`: inspect the actual value.
- `type(value)`: inspect its type.
- `int("25")` / `float("25.5")`: convert numeric text to an integer / decimal number.
- `NameError`: a name is undefined; check spelling and execution order.
- `TypeError`: an operation does not support these types.
- `ValueError`: the type is accepted, but the value cannot be used as requested.

### NameError: Check the Name and Its Definition

```python
print(student_naem)
```

A traceback for this one-line script looks like:

```text
Traceback (most recent call last):
  File "analysis.py", line 1, in <module>
    print(student_naem)
NameError: name 'student_naem' is not defined
```

**Diagnosis:** Python cannot find that name. Check its spelling and whether the assignment ran before this line.

**Correction:**

```python
student_name = "Alice"
print(student_name)
```

### TypeError: Check the Operation and Types

```python
age = "25"
next_year = age + 1
```

```text
TypeError: can only concatenate str (not "int") to str
```

**Diagnosis:** `print(age, type(age))` reveals text. We want arithmetic, so convert numeric text before adding:

```python
age = "25"
age_number = int(age)
next_year = age_number + 1
print("Next year you'll be", next_year)
```

### ValueError: Check the Actual Value

```python
raw_age = "hello"
age = int(raw_age)
```

```text
ValueError: invalid literal for int() with base 10: 'hello'
```

**Diagnosis:** `int()` accepts numeric text such as `"25"`, but `"hello"` is not an integer representation. Checking only `type(raw_age)` would miss the difference; inspect its value and where it came from.

**Correction:** If the wrong field was selected, select the age field. If the source value is wrong, correct it only when you know the intended value. For this example, suppose the source confirms an age of 25:

```python
raw_age = "25"
age = int(raw_age)
print("Age:", age)
```

Do not replace unknown ages with invented numbers just to make the error disappear. Lecture 02 introduces `try`/`except` for responding to expected failures.

# LIVE DEMO!
