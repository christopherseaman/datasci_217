---
notion:
  title_line: "# Lecture 01 Live Demo Guide"
  role: demo
  status: mapped
  page_id: "3a4d9fdd-1a1a-8132-b57d-dd4cf6b9a2fa"
  url: "https://app.notion.com/p/3a4d9fdd1a1a8132b57ddd4cf6b9a2fa"
---

# Lecture 01 Live Demo Guide

There are four live breaks:

1. Git/GitHub and VS Code setup (`01_github_vscode_setup_guide.md`)
2. Shell navigation (`02_cli_navigation_demo.sh`)
3. Python basics (`03_python_basics_demo.py`)
4. Control structures and debugging (`04_control_structures_demo.py`, then `05_integration_workflow_demo.py`)

Run the commands in VS Code's **Terminal → New Terminal**. A separate Terminal or WSL Ubuntu window also works; use `cd` to enter your working folder first.

## Break 1: Git setup

Use `01_github_vscode_setup_guide.md` to configure Git, fork the assignment, clone it in VS Code, make one practice file, and commit it. Git is the first checkpoint: save your work before the coding demos.

## Break 2: Shell navigation

Run from a disposable directory:

```bash
mkdir -p scratch/lecture01-cli
cd scratch/lecture01-cli
bash /path/to/datasci_217/01/demo/02_cli_navigation_demo.sh
```

Watch `pwd` and `ls` as the script creates folders, makes an empty file with `touch`, copies it, renames it, and returns to the starting directory. No Python is needed yet.

## Break 3: Python basics

```bash
python3 /path/to/datasci_217/01/demo/03_python_basics_demo.py
```

This introduces variables, scalar values, arithmetic, strings, `type()`, and `len()`. Change a value and rerun to see how the output changes.

## Break 4: controls and debugging

```bash
python3 /path/to/datasci_217/01/demo/04_control_structures_demo.py
```

Practice comparisons, `if`/`elif`/`else`, direct list iteration, `enumerate`, `while`, `break`, and `continue`. The total/count example uses a loop rather than `sum()`, so the accumulator is visible. The debugging examples include expected `NameError`, `TypeError`, and `ValueError` lines, with corrected versions and explicit save/rerun instructions.

### Debugging practice

1. In section 5 of the control-structures script, uncomment one line immediately below an “Uncomment” comment.
2. Save and run the script. Read the final traceback line and the source line it names.
3. Compare the failing line with the corrected version below it. Comment the failing line again, save, and rerun to see the correction work.
4. Repeat for the other two errors, one at a time.

### Small integration workflow

```bash
python3 /path/to/datasci_217/01/demo/05_integration_workflow_demo.py
```

This final script stays small: it combines scores, a loop, a decision, and a total/count/average summary. Change one score, save, and rerun to see the result change.
