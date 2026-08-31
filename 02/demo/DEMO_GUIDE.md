# Lecture 02 Demo Guide: Git, Functions, and Modules

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

## 2. Functions: refactor a script into reusable helpers

```bash
python3 functions_demo.py
```

The functions demo creates `sample_students.csv` and introduces reusable
functions implemented in the import-safe `student_tools.py` module.

## 3. Modules: import the helpers in a second script

```bash
python3 module_usage_demo.py
```

The module demo imports `student_tools.py` as an ordinary module and reuses
its functions to create reports. Run both
commands in a disposable directory if you do not want their report files.
