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

1. Git/GitHub and VS Code setup
2. Shell navigation
3. Python basics
4. Control structures and debugging

All files are in the [Lecture 01 demo folder on GitHub](https://github.com/christopherseaman/datasci_217/tree/main/01/demo). Follow the setup below, then create a practice folder and open it in VS Code with **File → Open Folder**. Each code block has a source link; use **Download raw file** on GitHub or paste the code into a new file with the shown filename. Save with **File → Save** (Ctrl+S; Cmd+S on Mac).

Run commands in **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac). A separate Terminal or WSL Ubuntu window also works; use `cd` to enter your working folder first.


# Demo 1: Git setup

[Setup source on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/01_github_vscode_setup_guide.md)

## 1.1 GitHub account and email privacy

1. Open [github.com](https://github.com/) and create or open your account.
2. Choose a professional username; it will be part of your public portfolio.
3. Open [GitHub email settings](https://github.com/settings/emails).
4. Enable **Keep my email addresses private** and copy your GitHub `noreply` address for your Git configuration.
5. The [GitHub Student Developer Pack](https://education.github.com/students) is optional.

## 1.2 Install and open VS Code

For browser-only setup, use the **Alternative: Codespaces** subsection in [Lecture 01](../README.md). Then run the Python setup in 1.4, skip 1.5 (your fork is already open), and continue at 1.6.

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Open **View → Extensions** (Ctrl+Shift+X; Cmd+Shift+X on Mac) and install **Python** by Microsoft. GitLens and Rainbow CSV are optional.
3. Open a course or practice folder with **File → Open Folder**.
4. Open **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac).

Open the Command Palette with **View → Command Palette** (Ctrl+Shift+P; Cmd+Shift+P on Mac). Explorer edits files; the terminal runs commands; Source Control saves versions.

## 1.3 Choose a terminal and shell

A **terminal** displays the session; a **shell** interprets your commands. Use VS Code's terminal with Bash or Zsh for these demos.

- **Windows:** For initial setup, run `wsl --install` in Administrator PowerShell, restart if prompted, and finish Ubuntu's username/password setup. Then return to VS Code:
    1. Open **View → Extensions** (**Ctrl+Shift+X**) and install **WSL** by Microsoft.
    2. Open **View → Command Palette** (**Ctrl+Shift+P**) → **WSL: Connect to WSL**. Expect **WSL: Ubuntu** in the lower-left corner.
    3. Install Microsoft's **Python** extension in WSL when prompted. Choose **Terminal → New Terminal** in this window; use it for the commands below and clone/open your project in this same window.
- **Mac:** VS Code's default terminal usually runs Zsh, which supports these commands.
- **Separate app fallback:** macOS Terminal or Windows Terminal's Ubuntu profile also works for commands; use `cd` to enter your working folder. Keep VS Code connected to WSL on Windows.

[WSL installation help](https://learn.microsoft.com/en-us/windows/wsl/install)

## 1.4 Install Python and Git

### Python: macOS, WSL Ubuntu, and Codespaces

Paste these commands into **VS Code's integrated terminal** one at a time (in the WSL-connected window on Windows). [uv](https://docs.astral.sh/uv/guides/install-python/) installs Python **3.13**; Lecture 03 explains environment management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.13 --default
uv python update-shell
```

Open a new terminal, then run `python3 --version`. Expect `Python 3.13.x`.

### Git

Run `git --version`. If Git is missing:

- **WSL Ubuntu:** Run `sudo apt update`, then `sudo apt install git`.
- **Mac:** Install [Homebrew](https://brew.sh/), follow its **Next steps** to put `brew` on your PATH, then run `brew install git`. Homebrew is recommended for other command-line tools; Python comes from uv.

### Select Python in VS Code

Open **View → Command Palette** (**Ctrl+Shift+P**; **Cmd+Shift+P** on Mac), choose **Python: Select Interpreter**, and select Python 3.13. Run the demos in the terminal you checked above.

## 1.5 Fork and clone

1. Open the assignment repository linked for this term on GitHub and select **Fork**.

    ![GitHub's Fork button](../assignment/media/github-fork.png)

2. Choose your account as Owner, keep the repository name, and select **Create fork**.

    ![Create a new fork under your account](../assignment/media/github-create-fork.png)

3. From your fork, copy **Code → HTTPS**. Confirm that the URL contains your username as the owner.

    ![Copy the HTTPS URL from your fork](../assignment/media/github-clone-url.png)

4. Open **View → Command Palette** (Ctrl+Shift+P; Cmd+Shift+P on Mac), choose **Git: Clone**, paste the URL, choose a folder, and open it. On Windows, do this in the **WSL: Ubuntu** window and choose a folder in your Linux home directory.

    ![VS Code's Clone from URL prompt](../assignment/media/vscode-clone.png)

Screenshots show example repositories; paste your own fork's URL. Sources: [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and [VS Code documentation](https://code.visualstudio.com/docs/sourcecontrol/quickstart).

## 1.6 Sign in to GitHub in VS Code

Sign in to GitHub through VS Code when **Clone from GitHub** or **Sync Changes** prompts you: choose **Sign in with GitHub**, authorize in your browser, then return to VS Code.

If a commit reports a missing name or email, open **Terminal → New Terminal** in your cloned folder and run these once, using your GitHub `noreply` email:

```bash
git config user.name "Your Name"
git config user.email "YOUR GITHUB NOREPLY EMAIL"
```

GitHub login authorizes access to your repositories; these settings identify the author of your commits.

## 1.7 Save a change on GitHub

1. In Explorer, create `practice.txt`, write a sentence about what you want to learn, and save it (**File → Save**, Ctrl+S; Cmd+S on Mac).
2. Open **View → Source Control** (Ctrl+Shift+G, including Control on Mac), review the change, stage it with **+**, enter a commit message, and select **Commit**.
3. Select **Sync Changes**, sign in if prompted, then check your fork on GitHub. It should now contain `practice.txt` with your sentence.

You have a copy on GitHub and a working copy on your computer. Lecture 02 develops the Git concepts behind this workflow.

# Demo 2: Shell navigation

## 2.1 Enter commands at the shell prompt

Start in your practice folder. Enter each line separately and inspect the result before continuing. The shell reads a command, runs it, and returns to its prompt.

```bash
pwd
ls
mkdir cli_manual
cd cli_manual
pwd
```

The second `pwd` ends in `cli_manual`. Create a small project:

```bash
mkdir data scripts results
touch README.txt
ls
cat README.txt
```

`ls` shows three folders and `README.txt`. `cat` prints nothing because the file is empty.

```bash
cp README.txt results/notes.txt
mv results/notes.txt results/lecture_notes.txt
ls results
cd ..
pwd
```

`ls results` shows `lecture_notes.txt`; the final `pwd` is back in your practice folder.

## 2.2 Create a script by pasting

From that same practice folder, run:

```bash
cat > 02_cli_navigation_demo.sh
```

Paste the script below. Press **Enter** if needed to reach a new line, then **Ctrl+D** to finish input. Enter ends a line; Ctrl+D signals **EOF** (end of file). Ctrl+C interrupts `cat` instead; it leaves text already written in the file. `>` replaces any previous contents.

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

You can also paste into a new file in VS Code and save it. Inspect your saved file:

```bash
cat 02_cli_navigation_demo.sh
```

## 2.3 Run the saved script

You can run scripts by calling the shell as a command and pointing it at the file. We’re using `bash` in this case but `sh` is common for scripts.

```bash
bash 02_cli_navigation_demo.sh
```

The script creates `cli_practice`, separate from your interactive `cli_manual` folder. Explorer should show `cli_practice/data`, `cli_practice/scripts`, `cli_practice/results/lecture_notes.txt`, and `cli_practice/README.txt`. The two text files are empty.

Watch `pwd` and `ls` as the script creates, copies, and renames files. Repeat in a fresh practice folder.

# Demo 3: Python basics

## 3.1 Start interactive Python

In your terminal, run `python3`. The `>>>` prompt means Python is ready. Enter the following examples one line at a time; do not type the prompt itself. Inspect each result before continuing.

To run a saved file instead, enter `exit()` to return to your shell, then use the command below that example. Restart `python3` whenever you want to return to interactive work.

## 3.2 Values and types

[03a_values.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/03a_values.py)

```python
student_name = "Alice"
student_age = 22
average_score = 87.5
is_enrolled = True
print("name:", student_name, "type:", type(student_name))
print("age:", student_age, "type:", type(student_age))
print("score:", average_score, "type:", type(average_score))
print("enrolled:", is_enrolled, "type:", type(is_enrolled))
print("next score:", average_score + 2)
```

From your shell:

```bash
python3 03a_values.py
```

Expect Alice (`str`), 22 (`int`), 87.5 (`float`), True (`bool`), and next score 89.5.

## 3.3 Strings

[03b_strings.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/03b_strings.py)

```python
course = "Data Science 217"
welcome = "Welcome to " + course
print(welcome)
print("course length:", len(course))
print("upper case:", course.upper())
print("trimmed text:", "  ready  ".strip())
```

From your shell:

```bash
python3 03b_strings.py
```

Expect `Welcome to Data Science 217`, length 16, `DATA SCIENCE 217`, and `ready` without surrounding spaces.

## 3.4 Calculations

[03c_calculations.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/03c_calculations.py)

```python
hours_studied = 3
score_per_hour = 10
points_earned = hours_studied * score_per_hour
print("hours:", hours_studied)
print("points per hour:", score_per_hour)
print("points earned:", points_earned)
weight_kg = 70
height_m = 1.75
bmi = weight_kg / (height_m * height_m)
print("BMI:", bmi)
```

From your shell:

```bash
python3 03c_calculations.py
```

Expect 30 points and a BMI of about 22.86. Change the hours or weight and observe which outputs change.

# Demo 4: Control structures and debugging

Use `python3` for interactive entry or run each saved file from your shell. At the `...` prompt, indent the block exactly as shown. Enter a blank line after each complete `if`/`elif`/`else`, `for`, or `while` block before starting the next top-level statement. Every file includes its own starting values.

## 4.1 Decisions

[04a_decisions.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04a_decisions.py)

```python
score = 85
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
```

From your shell:

```bash
python3 04a_decisions.py
```

Expect grade B and `candidate meets both requirements`. Try a score of 95, then 70.

## 4.2 For loops and running totals

[04b_for_loops.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04b_for_loops.py)

```python
scores = [87, 92, 78, 95, 88]
total = 0
count = 0
for score in scores:
    print("score:", score)
    total = total + score
    count = count + 1

print("total:", total)
print("count:", count)
print("average:", total / count)
for position, score in enumerate(scores, start=1):
    print("assignment", position, "score", score)
```

From your shell:

```bash
python3 04b_for_loops.py
```

Expect total 440, count 5, average 88.0, then assignment numbers 1–5 paired with scores.

## 4.3 While, break, and continue

[04c_loop_control.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04c_loop_control.py)

```python
counter = 1
while counter <= 3:
    print("counter:", counter)
    counter = counter + 1

scores = [87, 92, 78, 95, 88]
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
```

From your shell:

```bash
python3 04c_loop_control.py
```

Expect counters 1–3, a stop at 92, then processing of 87, 92, 95, and 88 (78 is skipped).

## 4.4 Debugging

[04d_debugging.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04d_debugging.py)

```python
# Uncomment to see the NameError, then comment it again before rerunning.
# print(total_socre)
total_score = 440
print("Corrected version:", total_score)

age_text = "25"
# Uncomment to see the TypeError, then comment it again before rerunning.
# print(age_text + 1)
age_number = int(age_text)
print("Corrected version:", age_number + 1)

# Uncomment to see the ValueError, then comment it again before rerunning.
# invalid_number = int("hello")
valid_number = int("42")
print("Corrected version:", valid_number)
```

From your shell:

```bash
python3 04d_debugging.py
```

Run the corrected version first: expect 440, 26, and 42. Then, one at a time, remove the `#` from an error line in `04d_debugging.py`, save it, and run the script again. Read the error, use the corrected line below it to identify the fix, then restore the `#` before trying the next error: fix the misspelled name, convert text with `int()`, or supply numeric text.

## 4.5 Combine the pieces

[04e_measurement_workflow.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04e_measurement_workflow.py)

```python
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
print("total:", total)
print("count:", count)
print("average:", average)
print("passing:", passing)
```

From your shell:

```bash
python3 04e_measurement_workflow.py
```

Expect PASS for scores 92, 76, and 88; REVIEW for 64; total 320, count 4, average 80.0, and 3 passing students. Change the threshold and inspect the status and count changes.
