# Lecture 02 Demo Guide: Git, CLI, Functions, and Modules

Run these scripts from this directory. They create their example files in the
current directory, so use a disposable copy when you want to repeat a demo.

```bash
cd 02/demo
```

## 1. Git workflow discussion

Use a small practice repository to demonstrate `git status`, `git add`,
`git commit`, branches, merges, and `git log --oneline --graph --all`.
For an already shared commit, prefer `git revert <commit>`: it records an undo
without rewriting history. This guide deliberately does not teach `git reset
--hard` or force-pushing. Those recovery operations require a verified
disposable repository, an identified backup, and agreement from every affected
collaborator.

## 2. CLI data-processing pipeline

```bash
bash 02_cli_advanced_demo.sh
```

The script creates a small project tree, processes CSV files with `head`,
`tail`, `grep`, `cut`, and `awk`, and writes a timestamped backup under
`backups/`. It stops on a failed command or failed copy, and falls back to
`find` for directory displays when `tree` is unavailable.

## 3. Functions and modules

```bash
python3 03_python_functions_demo.py
python3 03_module_usage_demo.py
```

The functions demo creates `sample_students.csv` and introduces reusable
functions. The module demo then loads `03_python_functions_demo.py` by its
actual numbered filename and reuses its functions to create reports. Run both
commands in a disposable directory if you do not want their report files.
