# Version Control, Functions, and Reusable Python

## Learning objectives

By the end of this lecture, you should be able to:

1. Distinguish a repository, working tree, diff, staging area, commit, local branch, and remote.
2. Use VS Code Source Control to inspect a diff, stage selected changes, create a focused commit, and synchronize with GitHub.
3. Create and merge one feature branch through the GUI, then resolve a prepared text conflict without losing either intended change.
4. Define and call a function using parameters, arguments, a return value, local variables, a short docstring, and an empty-input guard.
5. Build a two-file Python program whose reusable module can be imported without running the program or writing files.

## Starting point and execution boundary

Lecture 01 introduced working directories, relative paths, top-to-bottom Python scripts, scalar values, lists, conditions, loops, and tracebacks. Before continuing, make sure you can:

- open a project folder in VS Code;
- compare that folder with `pwd` in the integrated terminal;
- run a `.py` file from the terminal;
- use a list, one condition, and one `for` loop; and
- read the final line and source location in a traceback.

For example, you should be able to save and run this short script:

```python
measurements = [18, 21, 24]

for measurement in measurements:
    if measurement >= 20:
        print(f"Review: {measurement}")
```

Lectures 01–03 use Python scripts and the terminal. Jupyter notebooks and Google Colab begin in Lecture 04.

The required Git interface in this lecture is **VS Code Source Control**. GitHub Desktop represents the same states and is an acceptable supported alternative, but we will not learn two sets of buttons. Command-line Git is optional bonus material and is not required here.

# Why version control?

Without version control, an evolving analysis often becomes a collection of ambiguous copies:

```text
analysis.py
analysis_v2.py
analysis_final.py
analysis_final_revised.py
```

Those filenames do not reliably answer important questions:

- What changed between two versions?
- Why was the change made?
- Which version should another person use?
- Can one change be reviewed without accepting every other change?

**Version control** records selected changes as an ordered project history. **Git** is the version-control system used in this course. **GitHub** stores a shared copy of a Git project and provides services around it.

![A comic about the confusion Git can cause without a clear mental model](media/xkcd_1597.png)

# The Git state model

Learn the states before learning the interface. The same model applies in VS Code, GitHub Desktop, and the optional command line.

## Repository

A **repository** is a project whose files and version history are managed by Git. A Classroom 50 assignment folder is already a repository; do not create a second repository inside it.

## Working tree

The **working tree** is the set of project files from the current branch that you can currently edit. Changing `README.md` changes the working tree.

The working tree is not the same as the shell's **working directory**:

- the working tree is a Git concept describing the editable project files;
- the working directory is the shell location from which a relative path is resolved.

## Change and diff

A **change** is a difference between a working-tree file and a recorded version. A **diff** is a line-by-line view of that change. In a typical diff, removed lines and added lines are shown separately.

Inspect a diff before staging. It can reveal an accidental edit, generated output, a secret, or an unrelated change that does not belong in the next commit.

## Staging area

The **staging area** is the proposed content of the next commit. Staging a file selects its current change for that commit; it does not yet create a commit or send anything to GitHub.

Selective staging lets one commit contain one coherent change even when the working tree contains other unfinished work.

## Commit

A **commit** is a recorded snapshot of the staged project changes, together with a message and other metadata. It belongs first to the local repository on your computer.

A useful commit answers one question: “What coherent change did this commit make?”

## Local branch

A **local branch** is a named line of development in your local repository. `main` is the course's primary branch. A short-lived **feature branch** holds a focused change until it is ready to merge into `main`.

A **merge** combines the changes from one branch into another. A **conflict** occurs when Git cannot safely decide how two changes should be combined.

## Remote

A **remote** is another copy of the repository, normally the course repository on GitHub. The local and remote repositories can each contain commits that the other does not yet have.

- **Push** sends local commits to the remote.
- **Pull** brings remote commits into the local repository and integrates them.
- **Synchronize** is a GUI action that coordinates the needed pull and push.

Synchronization moves commits. It is not a replacement for saving a file, staging a change, or creating a commit.

The required daily cycle is:

```text
edit in working tree
    → inspect diff
    → stage selected change
    → commit locally
    → synchronize with remote
```

# One VS Code Source Control workflow

Use an instructor-provided disposable repository while learning this sequence.

## 1. Open and orient

1. Open the repository folder in VS Code.
2. Open the integrated terminal and check `pwd`.
3. Select **Source Control** in the Activity Bar.
4. Confirm that the branch indicator in the status bar says `main`.
5. Synchronize before editing so the local repository starts from the current remote state.

## 2. Edit and inspect

1. Make one small change to `README.md` and save it.
2. In Source Control, find the file under **Changes**.
3. Select the filename to open its diff.
4. Read every changed line and confirm that it belongs to the intended task.

The file is changed but not staged. No commit has been created.

## 3. Stage deliberately

1. Select the `+` beside `README.md`.
2. Confirm that the file moves from **Changes** to **Staged Changes**.
3. Inspect the staged diff once more.

If a file should not be included, use the `−` beside it to unstage it. Unstaging preserves the working-tree edit; it only removes the change from the proposed commit.

## 4. Commit locally

1. Enter a focused message such as `Clarify project purpose`.
2. Select **Commit**.
3. Confirm that the staged change disappears from Source Control.
4. Open the Source Control graph or history view and locate the new commit.

Avoid messages such as `updates`, `stuff`, or `final`. They do not explain the change.

## 5. Synchronize

1. Select **Sync Changes**.
2. If VS Code reports incoming changes, review and integrate them before sending your local work.
3. Confirm on GitHub that the remote repository now shows the commit.

Do not assume that a local commit has reached GitHub until synchronization succeeds.

# Keep commits and repositories focused

## A focused commit

A focused commit contains one logical change. Before committing, ask:

- Does every staged line support the commit message?
- Did I accidentally include data, credentials, generated output, or editor files?
- Could another person understand or reverse this change without also reversing unrelated work?

Commit after a coherent improvement, not after an arbitrary number of edits.

## A minimal `.gitignore`

An **untracked file** is a working-tree file that Git has not yet staged or recorded in a commit. A `.gitignore` file names untracked files that Git should normally leave out of version control. A small Python project might begin with:

In a `.gitignore` pattern, `*` matches a sequence of characters and a trailing `/` names a directory.

```gitignore
__pycache__/
*.pyc
```

This example excludes cache files that Python may generate while it runs code. Lecture 03 adds an ignore pattern for its isolated environment after defining that concept. Add a pattern only when you understand what it matches. A `.gitignore` file does not remove a file that has already been committed, and it is not a privacy guarantee.

## A useful `README.md`

A **README** is the project's entry-point documentation. For a small course script, it needs only enough information to orient the next reader:

In the small Markdown example below, `#` begins a top-level heading, `##` begins a second-level heading, and backticks mark code within a sentence.

```markdown
# Measurement summary

Summarizes two supplied groups of measurements.

## Run

Open this repository in VS Code and run `python main.py` in the terminal.
```

Keep run instructions consistent with the files that actually exist.

# One feature branch and one prepared conflict

Use a feature branch for a focused change that should be reviewed before it joins `main`.

## Create and merge a feature branch in VS Code

1. Synchronize `main` and confirm that the working tree has no unfinished changes.
2. Select the branch name in the VS Code status bar.
3. Choose **Create new branch** and name it `feature/summary-label`.
4. Edit the supplied summary label, inspect the diff, stage it, and commit it.
5. Select the branch name again and switch to `main`.
6. Synchronize `main`.
7. Open the Command Palette, run **Git: Merge...**, and then select `feature/summary-label`.
8. Inspect the resulting diff or history and synchronize `main`.

Always notice which branch is active before editing or merging.

## Resolve a prepared text conflict

For the prepared exercise, `main` and the feature branch change the same line differently. Git pauses the merge because choosing automatically could discard intended work.

1. Open the conflicted file from Source Control.
2. Read the current and incoming versions in context.
3. Decide what the final combined text should say. Do not choose a side merely because a button calls it “current” or “incoming.”
4. Use the merge editor or edit the text manually so both intended ideas remain.
5. Save the file and verify that no **conflict markers** remain. Conflict markers are the special separator lines Git inserts around unresolved versions of the text.
6. Inspect the final diff, stage the resolved file, and complete the merge commit in Source Control.
7. Run any relevant script, then synchronize.

The goal is a correct final file, not winning one side of the conflict.

# LIVE DEMO!

**One GUI Git state cycle:** in a disposable repository, edit one line, inspect its diff, stage only that file, commit, synchronize, create and merge a feature branch, and resolve a prepared simple conflict while naming each Git state.

# From duplicated code to reusable code

Lecture 01 scripts placed all instructions from top to bottom. That is appropriate for a small first program, but repetition makes changes harder to apply consistently.

This script calculates the same kind of result twice:

```python
morning_values = [18, 21, 24]
morning_total = 0

for value in morning_values:
    morning_total = morning_total + value

morning_mean = morning_total / len(morning_values)
print(f"Morning mean: {morning_mean:.1f}")

evening_values = [20, 22, 26]
evening_total = 0

for value in evening_values:
    evening_total = evening_total + value

evening_mean = evening_total / len(evening_values)
print(f"Evening mean: {evening_mean:.1f}")
```

If the calculation changes, both copies must change. A function gives the repeated calculation one name and one reusable definition.

# A minimal dictionary

A **dictionary** stores named associations. Each **key** identifies a corresponding **value**.

```python
morning_record = {
    "label": "Morning",
    "values": [18, 21, 24],
}
```

Here, `"label"` and `"values"` are keys. Their corresponding values are the string `"Morning"` and the list `[18, 21, 24]`.

Select a value by its key:

```python
print(morning_record["label"])
print(morning_record["values"])
```

Expected output:

```text
Morning
[18, 21, 24]
```

Use a dictionary when names make a small record or result clearer. This lecture does not require nested dictionaries, dictionary comprehensions, or a survey of every dictionary method.

# Functions: inputs, work, and results

A **function** is a named block of code that performs one task. Defining a function does not run its body. A **call** runs the function.

A function's **interface** is how a caller supplies inputs and receives a result. Its **implementation** is the code inside the function that produces that result.

## Definition, parameter, argument, and return value

The empty-list case needs an explicit contract. `None` is Python's value for “no value here.” An empty list is false in a condition, so `if not values` detects that case before division.

```python
def mean(values):
    """Return the arithmetic mean, or None for empty input."""
    if not values:
        return None

    total = 0
    for value in values:
        total = total + value

    return total / len(values)
```

Read the definition from the outside inward:

- `def` begins a function definition.
- `mean` is the function name.
- `values` is a **parameter**: a local name in the function definition.
- The indented statements are the function body.
- The first statement is a **docstring**: a short description placed first in the function body.
- `total` and `value` are **local variables**. They exist for this call and are not used as names outside the function.
- `return` sends a result back to the caller and ends the call.

Call the function with a list:

```python
measurements = [18, 21, 24]
result = mean(measurements)
print(result)
```

In `mean(measurements)`, `measurements` is the **argument** supplied by the caller. During this call, its list value is assigned to the parameter `values`. The returned number is then assigned to `result`.

## Return is not print

`return` gives a value to the caller. `print` writes text to the terminal and returns no useful analysis result.

```python
def doubled(value):
    return value * 2

result = doubled(6)
print(result)
```

Expected output:

```text
12
```

Returning values lets another function, script, or test use the result before deciding how to display it.

## Guard empty input explicitly

The original duplicated script divides by `len(values)`. An empty list has length zero, so the division would fail.

```python
empty_result = mean([])
print(empty_result)
```

The function's docstring states that empty input returns `None`, making the edge-case contract visible to its callers.

Test both the ordinary and edge cases:

```python
print(mean([18, 21, 24]))
print(mean([]))
```

Expected output:

```text
21.0
None
```

## Build a report line from a dictionary

The identity check `is None` asks specifically whether a value is the no-value object. It does not treat zero as missing.

```python
def format_summary(record):
    """Return a one-line summary for a measurement record."""
    average = mean(record["values"])

    if average is None:
        return f'{record["label"]} mean: no measurements'

    return f'{record["label"]} mean: {average:.1f}'
```

The function returns text rather than printing it, so the caller can send that text to the terminal, a file, or a test.

```python
morning_record = {
    "label": "Morning",
    "values": [18, 21, 24],
}

summary_line = format_summary(morning_record)
print(summary_line)
```

Expected output:

```text
Morning mean: 21.0
```

# LIVE DEMO!

**From duplication to functions:** refactor two repeated list calculations into `mean()` and `format_summary()`, map an argument to its parameter, trace a local variable and return value, add one-sentence docstrings, and verify the empty-list behavior.

# Write one small text file

A **driver script** is the file that coordinates a program's steps. File output is a **side effect**: it changes something outside a function's returned value. Keep side effects in the driver script rather than in a reusable calculation function.

`open()` opens the file at a path. Mode `"w"` creates the file or replaces its existing contents. `encoding="utf-8"` selects a standard text encoding so the same characters are interpreted consistently across systems. The `with` statement closes the file automatically when its indented block finishes.

```python
summary_line = "Morning mean: 21.0"

with open("report.txt", "w", encoding="utf-8") as report_file:
    report_file.write(summary_line + "\n")
```

`write()` expects a string and does not add a newline automatically. The relative path `report.txt` is resolved from the terminal's working directory, just as in Lecture 01.

Open the same file in read mode (`"r"`) to verify what was saved. `read()` returns the file's text as one string:

```python
with open("report.txt", "r", encoding="utf-8") as report_file:
    saved_summary = report_file.read()

print(saved_summary == summary_line + "\n")
```

Expected output:

```text
True
```

Use these small text patterns only; structured tabular input begins later.

# Modules and imports

A **module** is a `.py` file that can be loaded with `import`. A module is useful when it contains reusable definitions that another file needs.

Suppose one project contains two files:

```text
measurement-summary/
├── analysis_utils.py
└── main.py
```

The **top level** of a Python file consists of statements that are not indented inside a function or another block. The reusable module `analysis_utils.py` contains definitions but does not print a report or write a file at top level:

```python
def mean(values):
    """Return the arithmetic mean, or None for empty input."""
    if not values:
        return None

    total = 0
    for value in values:
        total = total + value

    return total / len(values)


def format_summary(record):
    """Return a one-line summary for a measurement record."""
    average = mean(record["values"])

    if average is None:
        return f'{record["label"]} mean: no measurements'

    return f'{record["label"]} mean: {average:.1f}'
```

An **import statement** loads a module. `from analysis_utils import format_summary` loads `analysis_utils.py` and makes its `format_summary` function available in the importing file.

Python executes a module's top-level statements when it first imports that module. A top-level function definition creates the function but does not call it. A top-level `print()` or file-writing statement would run immediately and would be an unwanted import side effect.

Python assigns a special module name to `__name__`:

- when Python runs a file directly, that file's `__name__` is `"__main__"`;
- when Python imports a file, `__name__` is the module's name, such as `"main"` or `"analysis_utils"`.

`main()` is a conventional function that coordinates a program. The **main guard** is the condition `if __name__ == "__main__":`; its indented body calls `main()` only during direct script execution.

The driver script `main.py` uses that exact pattern:

```python
from analysis_utils import format_summary


def main():
    """Create the measurement report."""
    records = [
        {"label": "Morning", "values": [18, 21, 24]},
        {"label": "Evening", "values": [20, 22, 26]},
        {"label": "Overnight", "values": []},
    ]

    with open("report.txt", "w", encoding="utf-8") as report_file:
        for record in records:
            summary_line = format_summary(record)
            print(summary_line)
            report_file.write(summary_line + "\n")


if __name__ == "__main__":
    main()
```

# Why the main guard matters

The guard in `main.py` keeps orchestration and file output behind an explicit function call. Importing `main` loads its definitions without creating `report.txt` or printing the report.

# Run the two-file program from the terminal

First, navigate to the project directory and confirm both files are present:

```bash
pwd
ls
```

The `-c` option asks Python to execute the short code string that follows it. Check import safety before running the program:

```bash
python -c "import main"
```

A safe import produces no terminal output and does not create `report.txt`.

Run the driver script:

```bash
python main.py
```

Expected terminal output:

```text
Morning mean: 21.0
Evening mean: 22.7
Overnight mean: no measurements
```

The same three lines should be written to `report.txt`. If the module cannot be found, check `pwd` and `ls` before changing the import statement. Both `.py` files should be in the same project directory; no import-path modification is needed.

# LIVE DEMO!

**From functions to an import-safe module:** move the reusable functions into `analysis_utils.py`, import them into `main.py`, verify that importing `main` produces no output or file, then run `python main.py` to create one deterministic report.

# Handoff to Lecture 03

You should now be able to:

- use VS Code Source Control to inspect, stage, commit, synchronize, branch, merge, and resolve a prepared conflict;
- explain the Git states involved in that workflow;
- use a minimal dictionary for a named record;
- define and call a function with an explicit edge-case contract;
- write and read back one small text file at a resolved path; and
- import a local module without triggering its driver workflow.

Lecture 03 adds isolated environments, direct dependencies, and NumPy arrays while preserving this terminal-and-script workflow.

# Key takeaways

- A working-tree edit becomes part of history only after you inspect it, stage it, and commit it.
- Local commits reach GitHub only after successful synchronization.
- A feature branch isolates a focused change; conflict resolution preserves the intended final content.
- Parameters receive argument values, local variables support a call, and `return` gives a result back to the caller.
- Reusable modules contain definitions; driver scripts contain orchestration and side effects.
- A correct main guard keeps imports quiet and safe.

# Optional bonus material

These topics are not required for assignments or assumed by Lecture 03:

- [Command-line Git, Git internals, collaboration, and recovery](bonus/advanced_git.md)
- [Advanced Python function and collection patterns](bonus/bonus_python_concepts.md)
- [Optional shell automation](bonus/shell_automation.md)
