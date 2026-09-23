---
title: Lecture 01 Notes from Lab
permalink: /02/lecture-01-catchup/
notion:
  title_line: "# Lecture 01 Notes from Lab"
  role: catchup
  status: mapped
  page_id: "3e4d9fdd-1a1a-818f-ba42-dd95c3d30d78"
  url: "https://app.notion.com/p/3e4d9fdd1a1a818fba42dd95c3d30d78"
---

# Lecture 01 Notes from Lab

These are the notes from lab, covering material added to [Lecture 01](../01/README.md) after it was delivered in class, so read it once and use the lecture itself from now on.

Nothing here is new coursework: it fills in the explanations behind commands and code you have already run, and it corrects one instruction that did not work in our terminals.

# Staging and Sync Changes

In Source Control you clicked **+**, typed a message, selected **Commit**, and then **Sync Changes**. Here is what each of those words means, because Lecture 02 builds on them.

A **commit** saves a version of your files. **Staging** a file (the **+** button in Source Control) chooses which changes go into the next commit; changes you leave unstaged stay on your computer. **Push** sends local commits to GitHub; VS Code's **Sync Changes** pushes your commits and pulls any new ones from GitHub in one click.

# How the Shell Reads a Command

The shell shows a **prompt**, such as `alice@laptop:~/datasci217$`, and waits. Type a command and press Enter; the shell runs it, prints any output, and shows a new prompt. It splits your line at spaces: the first word is the program to run, words starting with `-` are **options** that change how it behaves, and the rest are **arguments**, usually the files or folders to act on.

```text
ls  -l  data
│   │   └── argument: which folder to list
│   └────── option: long, detailed format
└────────── command: list directory contents
```

Quote an argument that contains spaces: `cd "My Documents"`.

# Where You Are: Paths and the Working Directory

A folder is called a **directory** at the shell, and the shell is always "in" one of them: its **working directory**, the *You are here* dot on a map. `pwd` prints it, `cd` changes it, and commands look for files there unless you say otherwise.

A **path** names a file or folder. An **absolute path** starts at the top of the file system, `/`, like a full street address; it works from anywhere. A **relative path** starts from your working directory, like directions from where you are standing.

```text
/home/alice/                ← ~ (your home directory)
└── datasci217/             ← working directory: pwd prints /home/alice/datasci217
    ├── data/
    │   └── visits.csv      ← relative path from here: data/visits.csv
    └── scripts/
        └── clean.py
```

- `.` = current directory
- `..` = parent directory, one level up
- `~` = home directory (`/home/alice` on Linux and WSL; `/Users/alice` on macOS)

From `datasci217`, `data/visits.csv` and `/home/alice/datasci217/data/visits.csv` name the same file. After `cd scripts`, the relative path becomes `../data/visits.csv`.

## Code Snippet: Moving Around

```bash
cd ~        # go to your home directory
pwd
cd ..       # up one level
pwd
cd ~        # back home
pwd
```

```text
/home/alice
/home
/home/alice
```

Your username replaces `alice`; macOS prints `/Users/alice` and `/Users`.

# Shell Scripts and the `cat` Correction

A **shell script** is a text file of shell commands that Bash runs top to bottom. Its first line, `#!/bin/bash`, records which shell the script expects; when you run `bash file.sh`, Bash treats it like any other `#` comment. That is the line at the top of the demo script you pasted.

The demo told you to finish `cat > file.sh` with **Ctrl+D**, which did not work reliably in our terminals. Use this instead:

1. Run `cat > file.sh` in your shell. `>` replaces that file if it exists.
2. Paste the script, press **Enter** to reach a new line, then **Ctrl+C** to finish.
3. Inspect with `cat file.sh`, then run with `bash file.sh`.

Enter ends a line; Ctrl+C stops `cat`, and the text already written stays in the file.

# What Python Is Doing

Python is a program called an **interpreter**: it reads Python code and runs it one **statement** (one instruction, usually one line) at a time. The shell manages files. Python computes with what is inside them: one patient's BMI, then the same calculation for every row of a clinic export.

You can give Python code in two ways:

- **Interactive mode**, or the **REPL** (read–evaluate–print loop): run `python3`, then type one line at the `>>>` prompt. Python evaluates it and shows the result right away, which is good for quick experiments.
- **Script mode**: save code in a `.py` file and run the whole file with `python3 file.py` from the folder that contains it (check with `pwd` and `ls`). A script is a record you can rerun, fix, and commit to GitHub. It shows output only where you call `print()`.

Two kinds of calls appear throughout Lecture 01:

- A **function** does a job when you call it with parentheses: `print("Hello")`, `len("Alice")`, `type(22)`. The values inside the parentheses are its **arguments**, like a shell command's arguments.
- A **method** is a function that belongs to a value and is called with a dot: `"alice".upper()` returns `"ALICE"`.

## The `>>>` Prompt Indents for You

At the `>>>` prompt, Python 3.13 indents for you: after a line ending in `:`, the next `...` line already starts four spaces in, and later lines keep that indentation. Type the block without adding spaces yourself, press **Backspace** once for each level you want to move back out (before an `elif` or `else`, for example), and press **Enter** on an empty `...` line to finish the block. In a `.py` file you type the four spaces yourself.

```console
>>> score = 85
>>> if score >= 90:
...     print("Grade: A")
... else:
...     print("Grade: B")
...
Grade: B
```

Typing the spaces as well doubles the indentation. A one-line block still runs, but the next line of the block starts from that deeper indentation and your extra spaces push it deeper still, so Python reports `IndentationError: unexpected indent`.

# Lists: Ordered Collections

A **list** holds several values in order inside square brackets, such as one patient's systolic readings across three visits: `[118, 142, 131]`. For now, create a list, count its items with `len()`, and visit each item with a `for` loop. Lecture 02 adds indexing and slicing.

## Reference Card: Lists So Far

- `[value1, value2, ...]`: Create a list; `[]` is an empty list.
- `len(items)`: Count the items; `len([118, 142, 131])` gives `3`.
- `for item in items:`: Visit each item in order.

# Running Totals with `+=`

The lecture's loops use `total += grade` where the demos write the same step out in full as `total = total + score`. Both forms do the same thing:

- `total += 5`: Shorthand for `total = total + 5`; `-=` and `*=` work the same way.

# Making Choices and Repeating Work

So far, every script runs each line once, top to bottom. Data work needs two more moves: *choose* (flag a blood-pressure reading only if it is high) and *repeat* (apply the same check to every reading, whether there are 4 or 4,000). **Control flow** statements change that top-to-bottom order.

- A **condition** is an expression that is either `True` or `False`, such as `systolic >= 140`. Comparison operators build conditions; `and`, `or`, and `not` combine them.
- An `if` statement runs its indented **block** only when its condition is `True`. With `elif` and `else`, Python checks the conditions from top to bottom and runs only the first block whose condition is `True`.
- A `for` loop runs its block once for each item in a list, giving the current item a name: `for grade in grades:`. A `while` loop repeats as long as its condition stays `True`.

Indentation is how Python knows which lines belong to the `if` or the loop.

## Which Branch Runs?

| `score` | First condition that is `True` | Output |
| --- | --- | --- |
| 95 | `score >= 90` | `Grade: A` |
| 85 | `score >= 80` | `Grade: B` |
| 72 | `score >= 70` | `Grade: C` |
| 50 | none, so `else` runs | `Grade: F` |

## Tracing a Loop

Each pass through a running-total loop updates two values:

| Pass | `grade` | `total` after | `count` after |
| --- | --- | --- | --- |
| 1 | 85 | 85 | 1 |
| 2 | 92 | 177 | 2 |
| 3 | 78 | 255 | 3 |
| 4 | 96 | 351 | 4 |
| 5 | 88 | 439 | 5 |

To check a loop you wrote, add a temporary `print("grade:", grade, "total:", total)` inside it and compare with a table like this.

```python
grades = [85, 92, 78, 96, 88]
total = 0
count = 0

for grade in grades:
    total += grade
    count += 1

average = total / count
print("Average grade:", average)    # Average grade: 87.8
```

# IndentationError: Check the Block Under the Colon

This is the first error in Assignment 01's debugging task, and it looks different from the others.

```python
score = 85
if score >= 80:
print("Grade: B")
```

```text
  File "/home/alice/datasci217/analysis.py", line 3
    print("Grade: B")
    ^^^^^
IndentationError: expected an indented block after 'if' statement on line 2
```

**Diagnosis:** A line ending in `:` must be followed by at least one indented line. Python checks the whole file's structure before it runs anything, so this error has no `Traceback (most recent call last)` header and nothing in the script runs, not even the lines above the mistake. A `NameError` or `TypeError` appears only when Python reaches the bad line, after earlier lines have already printed.

**Correction:** Indent the block four spaces:

```python
score = 85
if score >= 80:
    print("Grade: B")
```

The `^^^^` markers in an error point at the part of the line that failed. Lecture 01's `NameError` traceback now shows them too, along with the hint Python 3.13 adds when a similar name exists: with `student_name = "Alice"` defined, `print(student_naem)` ends with `NameError: name 'student_naem' is not defined. Did you mean: 'student_name'?`
