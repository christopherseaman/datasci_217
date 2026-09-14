# Lecture 02 Demo Guide: Git, Functions, and Modules

This is three demonstrations, not one script: a Git workflow, a functions
script, and a module-reuse script. Run the Python scripts from a disposable
copy because they create example files and reports.

```bash
cd 02/demo
```

## 1. Git workflow (command-line or GUI)

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

`functions_demo.py` creates `sample_students.csv` and introduces reusable
functions implemented in the import-safe `student_tools.py` module.

## 3. Modules: import the helpers in a second script

```bash
python3 module_usage_demo.py
```

`module_usage_demo.py` imports `student_tools.py` as an ordinary module and
reuses its functions to create reports. Run Demo 2 first so
`sample_students.csv` exists.
