# Assignment 02: Reusable Measurement Summary

## Project description

TODO: Replace this line with a 30–300 character description of what this measurement-summary project does.

## Run

TODO: Replace this line with the exact terminal command that runs the completed program.

## Files

```text
assignment/
├── README.md                  # complete description and Run answers
├── GIT_STATE_CHECK.md          # complete state answers
├── .gitignore                 # complete cache exclusions
├── analysis_utils.py, main.py  # Python scaffolds to complete
├── check_assignment.py        # supplied checker; keep unchanged
└── report.txt                 # generated report to commit
```

## Setup

Open **Terminal → New Terminal** in VS Code at the assignment directory. If you use a native terminal or WSL Ubuntu instead, first `cd` into the assignment directory.

Use Python 3.13 and terminal-executed scripts in `02/assignment` or its standalone assignment repository. Open the repository in VS Code, switch to `main`, select **Sync Changes**, and finish any outstanding changes. Use **Git: Create Branch** to create `feature/measurement-summary` from `main`.

## Part 1: Repository state and documentation

### 1.1 Complete the state snapshots

Open `GIT_STATE_CHECK.md`. For each scenario, replace only the `TODO` terms in the answer block. Use each defined Lecture 02 term where it describes the snapshot: working tree, diff, staging area, commit, local branch, remote, synchronize, merge, and conflict.

> **Checkpoint — `GIT_STATE_CHECK.md`**
> Save the four numbered answers inside the supplied answer block.

### 1.2 Complete the README

Replace the two TODO lines near the top of this file:

- Write a project description containing 30–300 characters after trimming surrounding whitespace and the word `measurement`.
- Put the exact command `python main.py` in the Run section.

> **Checkpoint — `README.md`**
> Save the description and exact Run command under their existing headings.

### 1.3 Complete `.gitignore`

Replace its TODO comments so its complete contents are exactly:

```gitignore
__pycache__/
*.pyc
```

Inspect the diffs in VS Code Source Control, stage `README.md`, `.gitignore`, and `GIT_STATE_CHECK.md`, and commit with `Complete repository documentation`.

## Part 2: Reusable calculations

Complete the two functions in `analysis_utils.py`, documenting what each returns.

### 2.1 Calculate `mean(values)`

Return `None` for an empty list. Otherwise, accumulate the values with a `for` loop and return the total divided by `len(values)`. Leave the input unchanged.

### 2.2 Format `format_summary(record)`

Call `mean(record["values"])`. If the result is `None`, return `<label> mean: no measurements`. Otherwise return `<label> mean: <value>` with one decimal place. Use the record's `"label"` in the returned string.

Required examples:

```text
mean([18, 21, 24]) -> 21.0
mean([]) -> None
format_summary({"label": "Zero", "values": [0, 0]}) -> "Zero mean: 0.0"
format_summary({"label": "Empty", "values": []}) -> "Empty mean: no measurements"
```

Importing `analysis_utils` must be silent and must not create or change files.

## Part 3: Import-safe driver and report

### 3.1 Complete the driver

Complete `main.py` using the supplied `format_summary` import, records, and main guard.

1. In `main()`, call `format_summary()` for each supplied record in order.
2. Join the resulting lines into `report_text`, with a newline after every line.
3. Open `report.txt` in text write mode with UTF-8 encoding and write `report_text`.
4. Read the saved text back into another variable.
5. Print the saved report and whether the read-back text equals `report_text`.

Importing `main` must print nothing and must not create `report.txt`. Running `python main.py` must overwrite a stale report and print exactly:

```text
Morning mean: 21.0
Evening mean: 22.7
Overnight mean: no measurements
Saved report matches: True
```

The exact bytes in `report.txt` are:

```text
Morning mean: 21.0
Evening mean: 22.7
Overnight mean: no measurements
```

There is one newline after the final report line. The `Saved report matches` status belongs only in terminal output, not in `report.txt`.

> **Checkpoint — `report.txt`**
> Run `python main.py` and save the three report lines above, including the final newline. Keep the read-back status in terminal output only.

## Check your work

Run each student script from the assignment directory:

```bash
python main.py
python check_assignment.py
```

A complete submission passes every check. If a check fails, revise the named artifact, regenerate `report.txt` if needed, and check again.

### Completion contract

Commit these files at the assignment repository root. Grading totals 100 points and reads their saved contents.

| Artifact | Format and completion criteria | Points |
|---|---|---:|
| `README.md` | Markdown with the existing Project description and Run headings; a 30–300 character description after trimming, containing `measurement`, and the exact Run line `python main.py`. | 30 |
| `GIT_STATE_CHECK.md` | Four numbered, semicolon-separated answers inside the supplied answer markers, using the terms for each snapshot. | 30 |
| `report.txt` | UTF-8 text containing the three report lines in Part 3, with a final newline. | 40 |

## Submit

Inspect the Python files and report in VS Code Source Control. Stage `analysis_utils.py`, `main.py`, and `report.txt`, then commit with `Implement reusable measurement summary`. Select **Publish Branch** or **Sync Changes**. With no unfinished changes, switch to `main`, run **Git: Merge...**, and select `feature/measurement-summary`. Resolve any unexpected conflict, inspect the resolution, and sync. Confirm in the repository browser that `main` contains the completed documentation, both Python files, and `report.txt`.

GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If a required VS Code control is unavailable, record its message and contact the instructor.
