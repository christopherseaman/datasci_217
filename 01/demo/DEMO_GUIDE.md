---
notion:
  title_line: "# Lecture 01 Live Demo Guide"
  role: demo
  status: mapped
  page_id: "3a4d9fdd-1a1a-8132-b57d-dd4cf6b9a2fa"
  url: "https://app.notion.com/p/3a4d9fdd1a1a8132b57ddd4cf6b9a2fa"
---

# Lecture 01 Live Demo Guide

There are four live demos:

1. Git/GitHub and VS Code setup (`01_github_vscode_setup_guide.md`)
2. Shell navigation (`02_cli_navigation_demo.sh`)
3. Python basics (`03_python_basics_demo.py`)
4. Control structures and debugging (`04_control_structures_demo.py`, then `05_integration_workflow_demo.py`)

All files are in the [Lecture 01 demo folder on GitHub](https://github.com/christopherseaman/datasci_217/tree/main/01/demo). Follow the setup below, then create a practice folder in VS Code with **File → Open Folder**. For each script, copy the code below into a new file with the shown filename and save it (**File → Save**, Ctrl+S; Cmd+S on Mac). You can also download the file using **Download raw file** on its GitHub page.

Run the commands in VS Code's **Terminal → New Terminal** (`Ctrl+Shift+backtick` on all platforms). A separate Terminal or WSL Ubuntu window also works; use `cd` to enter your working folder first.

# Demo 1: Git setup

[Setup source on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/01_github_vscode_setup_guide.md)

Use this checklist to prepare the tools used in Lecture 01. The commands are safe to repeat, except where GitHub asks you to choose an account name.

## 1. GitHub account and email privacy

1. Open [github.com](https://github.com/) and create or open your account.
2. Choose a professional username; it will be part of your public portfolio.
3. Open [GitHub email settings](https://github.com/settings/emails).
4. Enable **Keep my email addresses private** and copy your GitHub `noreply` address. Use that address in Git configuration instead of your personal email.
5. The [GitHub Student Developer Pack](https://education.github.com/students) is optional.

## 2. Install and open VS Code

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Install the **Python** extension by Microsoft.
3. Optional extensions: GitLens and Rainbow CSV.
4. Open a course or practice folder in VS Code.
5. Open the integrated terminal with **Terminal → New Terminal** (`Ctrl+Shift+backtick` on all platforms).

Useful interface areas are Explorer, Search, Source Control, Run and Debug, and Extensions. Open the Command Palette with **View → Command Palette** (`Ctrl+Shift+P` on Windows/Linux; `Cmd+Shift+P` on macOS). Open Source Control with **View → Source Control** (`Ctrl+Shift+G` on all platforms, including macOS).

## 3. Choose a shell

The course shell examples use Bash or a compatible POSIX shell.

- In VS Code, open your project folder, then **Terminal → New Terminal**.
- Windows: install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) from Administrator PowerShell with `wsl --install`, restart if prompted, and choose **Ubuntu (WSL)** as VS Code's default terminal profile.
- Native-terminal alternative: open Terminal on macOS/Linux or Ubuntu on Windows, then `cd` to your project folder.
- Any platform: GitHub Codespaces is an optional browser-based alternative.

Install Python 3.13 and Git using [Lecture 01's setup instructions](../README.md). In the shell, check that the basic commands are available:

```bash
pwd
ls
python3 --version
git --version
```

The Python version should start with `3.13`; Git should print its version. If a command is missing, finish its installation before continuing.

Native Windows PowerShell uses different command names and syntax. Use WSL for the Bash examples in this course, or translate each command deliberately.

## 4. Configure Git in the VS Code terminal

Replace the placeholders with your own name and GitHub `noreply` address:

```bash
git config --global user.name "Your Name"
git config --global user.email "12345+yourusername@users.noreply.github.com"
git config --list --global
```

Confirm that the displayed email is the privacy-preserving GitHub address.

## 5. Fork, clone, and save a change

1. Open the assignment repository on GitHub. Select **Fork**, choose your account, and select **Create fork**.

    ![GitHub's Fork button](../assignment/media/github-fork.png)

2. From your fork, copy **Code → HTTPS**. Confirm that the URL contains your username as the owner.

    ![Copy the HTTPS URL from your fork](../assignment/media/github-clone-url.png)

3. Open **View → Command Palette** (`Ctrl+Shift+P` on Windows/Linux; `Cmd+Shift+P` on macOS), choose **Git: Clone**, paste the URL, choose a local folder, and open it.

    ![VS Code's Clone from URL prompt](../assignment/media/vscode-clone.png)

4. In Explorer, create `practice.txt` and write a sentence about what you want to learn.
5. Open **View → Source Control** (`Ctrl+Shift+G` on all platforms, including macOS), review the change, stage it with **+**, enter a commit message, and select **Commit**.
6. Select **Sync Changes**, then check your fork on GitHub to see the file there.

You have a copy on GitHub and a working copy on your computer. Lecture 02 develops the Git concepts behind this workflow.

Screenshots show example repositories; paste your own fork's URL. Sources: [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and [VS Code documentation](https://code.visualstudio.com/docs/sourcecontrol/quickstart).

# Demo 2: Shell navigation

Save the script below as `02_cli_navigation_demo.sh` in a fresh practice folder. Open a terminal in that folder and run:

```bash
bash 02_cli_navigation_demo.sh
```

[02_cli_navigation_demo.sh on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/02_cli_navigation_demo.sh)

```bash
#!/bin/bash

echo "=========================================="
echo "DEMO 2: COMMAND-LINE NAVIGATION"
echo "=========================================="
echo "The shell organizes files; Python will analyze data later."
echo
echo "Where am I?"
pwd
echo
echo "What is here?"
ls
echo
echo "Make a small project and enter it:"
mkdir cli_practice
cd cli_practice
pwd
echo "Create folders and an empty note:"
mkdir data scripts results
touch README.txt
ls
echo "Inspect the empty note file:"
cat README.txt
echo "Copy and rename the note:"
cp README.txt results/notes.txt
mv results/notes.txt results/lecture_notes.txt
ls results
echo
echo "Move back to the starting directory:"
cd ..
pwd
echo "The practice folder is still here:"
ls cli_practice
echo
echo "Key commands: pwd, ls, cd, mkdir, touch, cp, mv, cat"
echo "Tip: use pwd whenever you are unsure where a relative path starts."
```

After running, Explorer should show `cli_practice/data`, `cli_practice/scripts`, `cli_practice/results/lecture_notes.txt`, and `cli_practice/README.txt`. The two text files are empty.

Watch `pwd` and `ls` as the script creates folders, makes an empty file with `touch`, copies it, renames it, and returns to the starting directory. No Python is needed yet.

# Demo 3: Python basics

```bash
python3 03_python_basics_demo.py
```

[03_python_basics_demo.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/03_python_basics_demo.py)

```python
#!/usr/bin/env python3
"""Demo 3: the smallest useful Python building blocks."""

print("=" * 50)
print("DEMO 3: PYTHON BASICS")
print("=" * 50)
print("Variables, types, arithmetic, and strings.")
print()
print("1. Variables and types")
student_name = "Alice"
student_age = 22
average_score = 87.5
is_enrolled = True
print("name:", student_name, "type:", type(student_name))
print("age:", student_age, "type:", type(student_age))
print("score:", average_score, "type:", type(average_score))
print("enrolled:", is_enrolled, "type:", type(is_enrolled))
print()
print("2. Arithmetic")
hours_studied = 3
score_per_hour = 10
points_earned = hours_studied * score_per_hour
print("hours:", hours_studied)
print("points per hour:", score_per_hour)
print("points earned:", points_earned)
print("next score:", average_score + 2)
print()
print("3. Strings and length")
course = "Data Science 217"
welcome = "Welcome to " + course
print(welcome)
print("course length:", len(course))
print("upper case:", course.upper())
print("trimmed text:", "  ready  ".strip())
print()
print("4. A tiny calculation")
weight_kg = 70
height_m = 1.75
bmi = weight_kg / (height_m * height_m)
print("weight:", weight_kg, "kg")
print("height:", height_m, "m")
print("BMI:", bmi)
print()
print("PYTHON BASICS COMPLETE")
print("Next: control structures, loops, and debugging.")
```

Expected values include `points earned: 30`, `next score: 89.5`, and `course length: 16`.

This introduces variables, scalar values, arithmetic, strings, `type()`, and `len()`. Change a value and rerun to see how the output changes.

# Demo 4: controls and debugging

```bash
python3 04_control_structures_demo.py
```

[04_control_structures_demo.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04_control_structures_demo.py)

```python
#!/usr/bin/env python3
"""Demo 4: decisions, loops, and a small debugging workflow."""

print("=" * 50)
print("DEMO 4: CONTROL STRUCTURES AND DEBUGGING")
print("=" * 50)
print("1. Comparisons and decisions")
score = 85
print("score:", score)
if score >= 90:
    print("grade: A")
elif score >= 80:
    print("grade: B")
else:
    print("grade: keep practicing")
age = 25
has_experience = True
if age >= 21 and has_experience:
    print("candidate meets both requirements")
else:
    print("candidate needs another requirement")
print()
print("2. For loops over a list")
scores = [87, 92, 78, 95, 88]
total = 0
count = 0
for score in scores:
    print("score:", score)
    total = total + score
    count = count + 1
print("total:", total, "count:", count, "average:", total / count)
print()
print("3. enumerate() when a position helps")
for position, score in enumerate(scores, start=1):
    print("assignment", position, "score", score)
print()
print("4. while, break, and continue")
counter = 1
while counter <= 3:
    print("counter:", counter)
    counter = counter + 1
print("Stop at the first score above 90:")
for score in scores:
    if score > 90:
        print("found:", score)
        break
print("Skip scores below 80:")
for score in scores:
    if score < 80:
        continue
    print("processing:", score)
print()
print("5. Debugging: read the traceback, fix, save, rerun")
print("NameError: check the spelling and definition of a name.")
# Uncomment to see the NameError, then comment it again before rerunning.
# print(total_socre)
total_score = total
print("Corrected version:", total_score)
print("TypeError: check whether values are text or numbers.")
age_text = "25"
# Uncomment to see the TypeError, then comment it again before rerunning.
# print(age_text + 1)
age_number = int(age_text)
print("Corrected version:", age_number + 1)
print("ValueError: the value cannot be converted to the requested type.")
# Uncomment to see the ValueError, then comment it again before rerunning.
# invalid_number = int("hello")
valid_number = int("42")
print("Corrected version:", valid_number)
print("Inspect values with print() and type() before guessing.")
print()
print("CONTROL STRUCTURES COMPLETE")
print("Next: run the small end-to-end workflow.")
```

Practice comparisons, `if`/`elif`/`else`, direct list iteration, `enumerate`, `while`, `break`, and `continue`. The total/count example uses a loop rather than `sum()`, so the accumulator is visible. The debugging examples include expected `NameError`, `TypeError`, and `ValueError` lines, with corrected versions and explicit save/rerun instructions.

Expected values include `total: 440 count: 5 average: 88.0` and `found: 92`.

## Debugging practice

1. In section 5 of the control-structures script, uncomment one line immediately below an “Uncomment” comment.
2. Save and run the script. Read the final traceback line and the source line it names.
3. Compare the failing line with the corrected version below it. Comment the failing line again, save, and rerun to see the correction work.
4. Repeat for the other two errors, one at a time.

## Small integration workflow

```bash
python3 05_integration_workflow_demo.py
```

[05_integration_workflow_demo.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/05_integration_workflow_demo.py)

```python
#!/usr/bin/env python3
"""A compact end-to-end workflow for Lecture 01."""

print("=" * 50)
print("DEMO 4: COMPLETE WORKFLOW")
print("=" * 50)
print("Run this script from the shell after the controls demo.")
print()
scores = [92, 76, 88, 64]
passing_score = 70
total = 0
count = 0
passing = 0
for position, score in enumerate(scores, start=1):
    total = total + score
    count = count + 1
    if score >= passing_score:
        status = "PASS"
        passing = passing + 1
    else:
        status = "REVIEW"
    print("Student", position, "score:", score, status)
average = total / count
print()
print("total:", total)
print("count:", count)
print("average:", average)
print("passing:", passing)
print("Workflow complete.")
```

Expected results: total `320`, count `4`, average `80.0`, and passing `3`.

This final script stays small: it combines scores, a loop, a decision, and a total/count/average summary. Change one score, save, and rerun to see the result change.
