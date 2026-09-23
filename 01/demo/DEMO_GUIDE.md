---
notion:
  title_line: "# Lecture 01 Demo Guide: Setup, the Shell, and Python Basics"
  role: demo
  status: mapped
  page_id: "3a4d9fdd-1a1a-8132-b57d-dd4cf6b9a2fa"
  url: "https://app.notion.com/p/3a4d9fdd1a1a8132b57ddd4cf6b9a2fa"
---

# Lecture 01 Demo Guide: Setup, the Shell, and Python Basics

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

Screenshots show example repositories; paste your own fork's URL.

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
mkdir clinic_manual
cd clinic_manual
pwd
```

The second `pwd` ends in `clinic_manual`. Create a small clinic project:

```bash
mkdir data scripts results
touch README.txt
ls
```

`ls` shows three folders and `README.txt`. Write a visits file: `>` creates it with a header line, and each `>>` appends one patient's reading.

```bash
echo "patient_id,systolic_mmHg" > data/visits.csv
echo "P001,118" >> data/visits.csv
echo "P002,142" >> data/visits.csv
echo "P003,131" >> data/visits.csv
cat data/visits.csv
```

```text
patient_id,systolic_mmHg
P001,118
P002,142
P003,131
```

Check only the start or the end of the file, as you would for a large export:

```bash
head -n 2 data/visits.csv
tail -n 1 data/visits.csv
```

```text
patient_id,systolic_mmHg
P001,118
P003,131
```

Copy the file, rename the copy, and go back up:

```bash
cp data/visits.csv results/visits_backup.csv
mv results/visits_backup.csv results/visits_raw.csv
ls results
cd ..
pwd
```

`ls results` shows `visits_raw.csv`; the final `pwd` is back in your practice folder.

## 2.2 Create a script by pasting

From that same practice folder, run:

```bash
cat > 02_cli_navigation_demo.sh
```

Paste the script below. Press **Enter** to end the last line, then **Ctrl+C** to stop `cat`. Every line you ended with Enter stays in the file; a line not yet ended with Enter is dropped. `>` replaces any previous contents.

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
echo "Make a clinic project and enter it:"
mkdir clinic_practice
cd clinic_practice
pwd
echo "Create folders and an empty README:"
mkdir data scripts results
touch README.txt
ls
echo "Write a visits file, one line at a time:"
echo "patient_id,systolic_mmHg" > data/visits.csv
echo "P001,118" >> data/visits.csv
echo "P002,142" >> data/visits.csv
echo "P003,131" >> data/visits.csv
cat data/visits.csv
echo "The header and first row, then the last row:"
head -n 2 data/visits.csv
tail -n 1 data/visits.csv
echo "Copy and rename the visits file:"
cp data/visits.csv results/visits_backup.csv
mv results/visits_backup.csv results/visits_raw.csv
ls results
echo
echo "Move back to the starting directory:"
cd ..
pwd
echo "The practice folder is still here:"
ls clinic_practice
echo
echo "Key commands: pwd, ls, cd, mkdir, touch, echo, cat, head, tail, cp, mv"
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

The script creates `clinic_practice`, separate from your interactive `clinic_manual` folder. Explorer should show `clinic_practice/data/visits.csv`, `clinic_practice/scripts`, `clinic_practice/results/visits_raw.csv`, and an empty `clinic_practice/README.txt`. Both CSV files hold the same four lines.

Watch `pwd`, `ls`, `head`, and `tail` as the script writes, views, copies, and renames files. The `cat`, `head`, and `tail` output matches 2.1. Repeat in a fresh practice folder.

# Demo 3: Python basics

## 3.1 Start interactive Python

In your terminal, run `python3`. The `>>>` prompt means Python is ready. Enter the following examples one line at a time; do not type the prompt itself. Inspect each result before continuing.

To run a saved file instead, enter `exit()` to return to your shell, then use the command below that example. Restart `python3` whenever you want to return to interactive work.

## 3.2 Values and types

[03a_values.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/03a_values.py)

```python
patient_id = "P001"
age_years = 67
temperature_c = 37.8
has_consent = True
print("patient:", patient_id, "type:", type(patient_id))
print("age:", age_years, "type:", type(age_years))
print("temperature:", temperature_c, "type:", type(temperature_c))
print("consent:", has_consent, "type:", type(has_consent))
print("age next year:", age_years + 1)
print("fever:", temperature_c >= 38.0)
print("fall-risk screen:", age_years >= 65 and has_consent)
systolic_readings = [118, 142, 131]
print("readings:", systolic_readings, "count:", len(systolic_readings))
```

From your shell:

```bash
python3 03a_values.py
```

```text
patient: P001 type: <class 'str'>
age: 67 type: <class 'int'>
temperature: 37.8 type: <class 'float'>
consent: True type: <class 'bool'>
age next year: 68
fever: False
fall-risk screen: True
readings: [118, 142, 131] count: 3
```

A comparison gives `True` or `False`: 37.8 is below the 38.0 °C fever cutoff, and 67 is at least 65 with consent recorded, so `and` gives `True`. `len()` counts the three readings in the list.

## 3.3 Strings

[03b_strings.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/03b_strings.py)

```python
clinic = "Mission Bay Clinic"
welcome = "Welcome to " + clinic
print(welcome)
print("name length:", len(clinic))
print("upper case:", clinic.upper())
print("trimmed ID:", "  P002  ".strip())
```

From your shell:

```bash
python3 03b_strings.py
```

```text
Welcome to Mission Bay Clinic
name length: 18
upper case: MISSION BAY CLINIC
trimmed ID: P002
```

`len()` counts the spaces inside the clinic name; `strip()` removes the spaces around `P002`.

## 3.4 Calculations

[03c_calculations.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/03c_calculations.py)

```python
doses_per_day = 3
dose_mg = 500
daily_mg = doses_per_day * dose_mg
print("doses per day:", doses_per_day)
print("dose (mg):", dose_mg)
print("daily total (mg):", daily_mg)
weight_kg = 70
height_m = 1.75
bmi = weight_kg / (height_m * height_m)
print("BMI:", bmi)
```

From your shell:

```bash
python3 03c_calculations.py
```

```text
doses per day: 3
dose (mg): 500
daily total (mg): 1500
BMI: 22.857142857142858
```

Change the doses or the weight and watch which outputs change.

# Demo 4: Control structures and debugging

Run each saved file from your shell, for example `python3 04a_decisions.py`. Every file includes its own starting values.

To type a block at the `python3` prompt instead, leave out the indentation shown here: after a line ending in `:`, Python 3.13 starts the next `...` line four spaces in for you. Press **Backspace** once for each level you move back out (before `elif` or `else`), and press **Enter** on an empty `...` line to finish the block. Typing the spaces yourself doubles the indentation, and the second line of a block then raises `IndentationError: unexpected indent`.

## 4.1 Decisions

[04a_decisions.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04a_decisions.py)

```python
systolic = 135
if systolic >= 140:
    print("systolic category: stage 2 hypertension")
elif systolic >= 130:
    print("systolic category: stage 1 hypertension")
elif systolic >= 120:
    print("systolic category: elevated")
else:
    print("systolic category: normal")

age_years = 67
has_consent = True
if age_years >= 65 and has_consent:
    print("eligible for fall-risk screening")
else:
    print("not eligible for fall-risk screening")
```

From your shell:

```bash
python3 04a_decisions.py
```

```text
systolic category: stage 1 hypertension
eligible for fall-risk screening
```

Python checks the conditions top to bottom: 135 fails `systolic >= 140` and passes `systolic >= 130`, so only the stage 1 block runs. These are systolic cutoffs only; a full blood-pressure category also uses diastolic pressure. Set `systolic` to 145, then 118, and rerun (stage 2 hypertension, then normal).

## 4.2 For loops and running totals

[04b_for_loops.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04b_for_loops.py)

```python
systolic_readings = [128, 142, 118, 135, 151]
total = 0
count = 0
for systolic in systolic_readings:
    print("systolic:", systolic)
    total = total + systolic
    count = count + 1

print("total:", total)
print("count:", count)
print("average:", total / count)
for visit, systolic in enumerate(systolic_readings, start=1):
    print("visit", visit, "systolic", systolic)
```

From your shell:

```bash
python3 04b_for_loops.py
```

```text
systolic: 128
systolic: 142
systolic: 118
systolic: 135
systolic: 151
total: 674
count: 5
average: 134.8
visit 1 systolic 128
visit 2 systolic 142
visit 3 systolic 118
visit 4 systolic 135
visit 5 systolic 151
```

`total = total + systolic` is the long form of `total += systolic`; both do the same thing, and 4.3 and 4.5 use `+=`.

## 4.3 While, break, and continue

[04c_loop_control.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04c_loop_control.py)

```python
check = 1
while check <= 3:
    print("blood pressure check:", check)
    check += 1

systolic_readings = [128, 142, 118, 135, 151]
print("Stop at the first reading of 140 or above:")
for systolic in systolic_readings:
    if systolic >= 140:
        print("found:", systolic)
        break

print("Skip readings below 130:")
for systolic in systolic_readings:
    if systolic < 130:
        continue
    print("review:", systolic)
```

From your shell:

```bash
python3 04c_loop_control.py
```

```text
blood pressure check: 1
blood pressure check: 2
blood pressure check: 3
Stop at the first reading of 140 or above:
found: 142
Skip readings below 130:
review: 142
review: 135
review: 151
```

`break` stops at 142, the first reading of 140 or above, so the loop never reaches 118, 135, or 151. `continue` skips 128 and 118.

## 4.4 Debugging

[04d_debugging.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04d_debugging.py)

```python
# Each line starting with # and no space is a broken version of the code just below it.
# Delete that # to see the error, then put it back before trying the next one.
heart_rate = 72
#print("heart rate:", heart_rat)
print("heart rate:", heart_rate)

age_text = "67"
#print("age next year:", age_text + 1)
age_years = int(age_text)
print("age next year:", age_years + 1)

#weight_kg = int("70 kg")
weight_kg = int("70")
print("weight (kg):", weight_kg)

if heart_rate > 100:
    print("heart rate: above 100 bpm")
else:
    print("heart rate: 100 bpm or below")
```

From your shell:

```bash
python3 04d_debugging.py
```

```text
heart rate: 72
age next year: 68
weight (kg): 70
heart rate: 100 bpm or below
```

Delete the `#` from one broken line, save, run, and compare the last line of the error with the table; the code just below it shows the fix. Put the `#` back before trying the next one.

| Delete the `#` from | Last line of the error |
| --- | --- |
| `#print("heart rate:", heart_rat)` | `NameError: name 'heart_rat' is not defined. Did you mean: 'heart_rate'?` |
| `#print("age next year:", age_text + 1)` | `TypeError: can only concatenate str (not "int") to str` |
| `#weight_kg = int("70 kg")` | `ValueError: invalid literal for int() with base 10: '70 kg'` |

These errors appear when Python reaches the bad line, so the lines before it have already printed: the `TypeError` run starts with `heart rate: 72`. **Ctrl+/** (**Cmd+/** on Mac) also removes or adds a `#`. If Python reports `IndentationError: unexpected indent` instead, a space is left at the start of the line; delete it.

Last, delete the four spaces before `print("heart rate: above 100 bpm")` on line 17, save, and run:

```text
  File "/home/alice/practice/04d_debugging.py", line 17
    print("heart rate: above 100 bpm")
    ^^^^^
IndentationError: expected an indented block after 'if' statement on line 16
```

Your own folder replaces `/home/alice/practice`. There is no `Traceback` header, and nothing prints, not even `heart rate: 72`: Python checks the whole file's structure before it runs any line. Put the four spaces back and rerun to see all four lines again.

## 4.5 Combine the pieces

[04e_measurement_workflow.py on GitHub](https://github.com/christopherseaman/datasci_217/blob/main/01/demo/04e_measurement_workflow.py)

```python
heart_rates = [72, 104, 88, 112]
review_above = 100
total = 0
count = 0
review_count = 0

for visit, heart_rate in enumerate(heart_rates, start=1):
    total += heart_rate
    count += 1
    if heart_rate > review_above:
        status = "REVIEW"
        review_count += 1
    else:
        status = "OK"
    print("Visit", visit, "heart rate:", heart_rate, "bpm", status)

average = total / count
print("total:", total)
print("count:", count)
print("average:", average)
print("readings to review:", review_count)
```

From your shell:

```bash
python3 04e_measurement_workflow.py
```

```text
Visit 1 heart rate: 72 bpm OK
Visit 2 heart rate: 104 bpm REVIEW
Visit 3 heart rate: 88 bpm OK
Visit 4 heart rate: 112 bpm REVIEW
total: 376
count: 4
average: 94.0
readings to review: 2
```

Lower `review_above` to 80 and rerun: 88 is now flagged too, and `readings to review` becomes 3.
