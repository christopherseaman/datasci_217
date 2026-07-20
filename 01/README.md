# Getting Started: The Terminal, Python Scripts, and VS Code

## Learning objectives

By the end of this lecture, you should be able to:

1. Use `pwd`, `ls`, `cd`, `mkdir`, `touch`, `cp`, and `mv` to navigate and manage a named scratch project, and safely remove a named scratch file after checking its location.
2. Open a project folder in VS Code, edit a `.py` file, and run it from the integrated terminal.
3. Assign scalar values, convert a numeric string, create and index a list, perform calculations, and print deterministic output.
4. Write one `if`/`elif`/`else` decision and one simple `for` loop over a list or `range`.
5. Use the final line and referenced source line in a traceback to correct a beginner Python error.

## Starting point and execution boundary

No programming or terminal experience is assumed.

Lectures 01–03 use Python scripts and a terminal. Jupyter notebooks and Google Colab begin in Lecture 04. In this lecture, every Python program runs from top to bottom in a `.py` file.

The required examples use a **POSIX-style shell**: Bash on Linux, WSL, and the supported cloud environment, or the default zsh shell on macOS. The required command subset behaves the same in Bash and zsh. Native PowerShell can be used to install or open WSL, but it is not a second assessed command interface.

## Tool readiness

Before beginning the exercises, confirm that you have:

- a POSIX-style terminal using Bash or macOS zsh;
- Python 3.12;
- VS Code with the Python extension;
- Git installed for Lecture 02;
- a GitHub account that can access the course organization.

Windows users should install and open WSL before running the Bash examples. macOS and Linux users can use their normal terminal application. A supported browser-based environment can be used when local installation is not possible.

Check Python in the terminal:

```bash
python --version
```

Some systems name the Python command `python3` instead:

```bash
python3 --version
```

Use whichever command reports the course Python version consistently for Lectures 01–02. Lecture 03 standardizes the command inside each course project.

Open VS Code and its integrated terminal with **Terminal → New Terminal**. The terminal's working directory should be the folder currently open in VS Code.

### GitHub privacy readiness

Lecture 02 teaches repositories, commits, branches, and synchronization. For now, only complete the account-readiness steps:

1. Sign in to GitHub and choose a professional username.
2. Verify an email address.
3. If you prefer not to publish a personal address in commit metadata, copy your GitHub-provided `noreply` address from [GitHub email settings](https://github.com/settings/emails).
4. Sign in to GitHub from VS Code when prompted.
5. Open the instructor-provided Classroom 50 readiness link and confirm that the provisioned repository page loads. Stop after confirming access.

Do not initialize a repository or practice Git commands yet. Those actions make more sense after Lecture 02 defines the Git model.

# The terminal model

![A learner improvising a rocket pack while learning a new technical skill](media/rocket_packs.png)

A **terminal** is the application window where you type. A **shell** is the program inside that terminal that reads commands. A **command** asks the shell to do one operation.

The shell displays a **prompt** when it is ready for the next command. Examples in this course show only what you type, not the prompt itself.

## Working directory and paths

The shell always has a **working directory**: the folder used as the starting point for relative paths.

A **path** identifies a file or directory.

- An **absolute path** starts from the filesystem root, such as `/home/alex/projects`.
- A **relative path** starts from the current working directory, such as `data/grades.csv`.
- `.` means the current directory.
- `..` means the parent directory.

Check the working directory before operating on files:

```bash
pwd
ls
```

`pwd` prints the working directory. `ls` lists its visible contents.

Move into a directory with `cd`:

```bash
cd Documents
pwd
ls
```

Move to the parent directory:

```bash
cd ..
```

Return to your home directory:

```bash
cd
```

If a relative path fails, ask two questions before changing the code:

1. What is my working directory?
2. From that directory, does the relative path point to the intended file?

## Safe file operations

Practice only inside a disposable directory created for this lecture:

```bash
mkdir ds217-lecture-01
cd ds217-lecture-01
pwd
```

Create a directory and an empty file:

```bash
mkdir scripts
touch notes.txt
ls
```

Copy and rename the file:

```bash
cp notes.txt notes-copy.txt
mv notes-copy.txt practice-notes.txt
ls
```

Inspect a small text file with `cat`, or inspect the beginning or end of a larger one with `head` and `tail`:

```bash
cat notes.txt
head notes.txt
tail notes.txt
```

Removal does not use a recycle bin. Confirm the directory and exact filename first:

```bash
pwd
ls practice-notes.txt
rm practice-notes.txt
ls
```

Do not use recursive removal in required work. Advanced shell expansion, recursive operations, search commands, and automation are in the bonus material.

Use `Control-C` to stop a command that is still running. Use a manual page such as `man ls` when it is available. Some commands also accept `--help`, but that option is not universal across macOS and Linux tools.

# LIVE DEMO!

**Project folder, paths, and the first script:** create a scratch project, open it in VS Code, compare the editor folder with `pwd`, and correct one deliberately wrong relative path.

# Create and run a Python script

Python can run interactively or execute a saved script. Scripts are the required workflow until Lecture 04 because their execution order is visible: Python starts at the first line and proceeds downward.

In VS Code, create `hello.py` inside the scratch project. Python provides `print()` to write a line of output:

```python
print("Hello from DataSci 217!")
```

Run it from the integrated terminal:

```bash
python hello.py
```

Expected output:

```text
Hello from DataSci 217!
```

If your setup uses `python3`, run `python3 hello.py` instead.

## Statements and comments

A **statement** is an instruction Python executes. Python normally uses one statement per line.

A comment begins with `#`. Python ignores the comment, so it can document intent:

```python
# Store the number of complete records.
complete_records = 18
print(complete_records)
```

Python is case-sensitive: `record_count` and `Record_Count` are different names.

# Names, values, and scalar types

A **variable name** refers to a **value**. A value has a **type**, which determines the operations that make sense for it.

```python
participant_count = 24       # int: a whole number
mean_age = 42.5              # float: a decimal number
study_name = "Pilot study"   # str: text
is_complete = False          # bool: True or False
```

Integers, floats, strings, and booleans are **scalar values**: each represents one value rather than a collection of values.

Use the built-in `type()` function to inspect a value:

```python
print(type(participant_count))
print(type(mean_age))
print(type(study_name))
print(type(is_complete))
```

`print`, `type`, `int`, `float`, `len`, and `range` are functions already provided by Python. Lecture 02 teaches how to define your own functions.

## Conversion

Text that looks numeric is still a string:

```python
age_text = "42"
print(type(age_text))
```

Convert it before numeric calculation:

```python
age = int(age_text)
age_next_year = age + 1
print(age_next_year)
```

Use `float()` when decimal values are allowed:

```python
temperature_text = "37.2"
temperature = float(temperature_text)
print(temperature)
```

Conversion can fail when the text is not a valid number. Reading that failure is part of the traceback section below; handling it automatically comes later.

# Lists and zero-based indexing

A **list** is an ordered collection of values. Each value in the list is an **element**.

```python
temperatures = [36.8, 37.1, 36.9, 37.4]
```

`len()` returns the number of elements:

```python
print(len(temperatures))  # 4
```

Python uses **zero-based indexing**. Index `0` selects the first element:

```python
first_temperature = temperatures[0]
second_temperature = temperatures[1]

print(first_temperature)
print(second_temperature)
```

For a four-element list, valid nonnegative indices are `0`, `1`, `2`, and `3`. Asking for `temperatures[4]` produces an `IndexError`.

# Arithmetic, comparisons, and boolean expressions

Common numeric operators are:

```python
total = 10 + 5
difference = 10 - 3
product = 4 * 6
quotient = 15 / 4
whole_quotient = 15 // 4
remainder = 15 % 4
square = 2 ** 2
```

A **comparison** produces the boolean value `True` or `False`:

```python
temperature = 37.4

print(temperature == 37.4)
print(temperature != 37.4)
print(temperature < 38.0)
print(temperature >= 37.0)
```

The comparison operators are `==`, `!=`, `<`, `>`, `<=`, and `>=`. A single `=` assigns a value; `==` compares two values.

A **condition** is a boolean expression used to decide which code should run.

## Decisions with `if`, `elif`, and `else`

Indentation groups the statements that belong to each branch. Use four spaces for each indentation level:

```python
score = 85

if score >= 90:
    label = "high"
elif score >= 70:
    label = "within expected range"
else:
    label = "review"

print(label)
```

Python evaluates the conditions from top to bottom and runs the first matching branch. `else` runs only if no earlier condition is true.

Combine small conditions with `and`, `or`, and `not` when the meaning remains clear:

```python
age = 42
has_measurement = True

if age >= 18 and has_measurement:
    print("Record is ready for the adult summary")
```

# Repetition with `for`

A **loop** repeats an indented block. A `for` loop takes one element at a time from a collection.

Direct list iteration is usually clearer than manually looking up every index:

```python
temperatures = [36.8, 37.1, 36.9, 37.4]

for temperature in temperatures:
    print(temperature)
```

`range(4)` produces the integers `0`, `1`, `2`, and `3`:

```python
for index in range(4):
    print(index)
```

Use direct iteration when you need each value. Use `range` when the sequence of integers itself is meaningful.

# Clear deterministic output

`print()` writes output to the terminal. An **f-string** inserts a value into labeled text. In `{mean_age:.1f}`, the format specifier `.1f` displays a floating-point value with one digit after the decimal:

```python
study_name = "Pilot study"
participant_count = 24
mean_age = 42.456

print(f"Study: {study_name}")
print(f"Participants: {participant_count}")
print(f"Mean age: {mean_age:.1f}")
```

Expected output:

```text
Study: Pilot study
Participants: 24
Mean age: 42.5
```

Labeled, deterministic output is easier for people to read and for assignment tests to check.

# LIVE DEMO!

**Values, lists, decisions, and loops:** build a short script one step at a time, checking each new value and its type before adding the next operation.

# Read a traceback, fix the source, and rerun

![Programming often involves iterating through mistakes until the program works](media/it_works.png)

When Python cannot complete a statement, it reports a **traceback**. A traceback shows where execution failed and ends with an **exception type** and message.

Suppose `traceback_practice.py` contains:

```python
temperatures = [36.8, 37.1]
print(temperatures[2])
```

Running it produces output similar to:

```text
Traceback (most recent call last):
  File "traceback_practice.py", line 2, in <module>
    print(temperatures[2])
          ~~~~~~~~~~~~^^^
IndexError: list index out of range
```

Read this short traceback from the bottom upward:

1. `IndexError` names the kind of failure.
2. `list index out of range` explains that the requested position does not exist.
3. The file and line number identify the failing source line.
4. The source excerpt points to the expression Python could not evaluate.

The list has two elements, so its valid indices are `0` and `1`. Correct the source and rerun:

```python
temperatures = [36.8, 37.1]
print(temperatures[1])
```

Common beginner exceptions include:

- `NameError`: a name is misspelled or has not been assigned;
- `TypeError`: an operation does not make sense for the value types;
- `ValueError`: a conversion received an unsuitable value;
- `IndexError`: a list position does not exist;
- `IndentationError`: indentation does not form a valid block.

The workflow is always the same: read the exception, find the referenced source line, make one small correction, save, and rerun.

# LIVE DEMO!

**Read, fix, rerun:** execute prepared scripts containing real beginner errors, identify the exception and source line, and finish with a clean top-to-bottom run.

# Integrated mini-script

This script combines the required Python concepts without functions, file input/output, or notebooks:

```python
measurements = [18, 21, 24, 19]
review_threshold = 20
total = 0

for measurement in measurements:
    total = total + measurement

    if measurement >= review_threshold:
        status = "review"
    else:
        status = "within range"

    print(f"Measurement {measurement}: {status}")

average = total / len(measurements)
print(f"Average: {average:.1f}")
```

Expected output:

```text
Measurement 18: within range
Measurement 21: review
Measurement 24: review
Measurement 19: within range
Average: 20.5
```

Run the script from the project directory:

```bash
python measurements.py
```

If the command fails, first check `pwd`, then confirm the filename with `ls`, then read the final line of any Python traceback.

# Key takeaways

- The working directory determines how relative paths are resolved.
- A Python script executes from top to bottom.
- Variable names refer to typed values; lists contain ordered elements with zero-based indices.
- Comparisons produce booleans; decisions and loops control which statements execute and how often.
- A traceback is diagnostic evidence: read its final line and referenced source line before guessing.
- Git repositories and reusable Python functions begin in Lecture 02. Jupyter and Colab begin in Lecture 04.
