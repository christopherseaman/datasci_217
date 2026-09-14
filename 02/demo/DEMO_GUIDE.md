---
notion:
  title_line: "# Lecture 02 Demo Guide: Git, Functions, and Modules"
  role: demo
  status: mapped
  page_id: "3dbd9fdd-1a1a-813d-936e-e7011563ebde"
  url: "https://app.notion.com/p/3dbd9fdd1a1a813d936ee7011563ebde"
---

# Lecture 02 Demo Guide: Git, Functions, and Modules

This is three demonstrations, not one script: a Git workflow, a functions script, and a module-reuse script. Run the Python scripts from a disposable copy because they create example files and reports.

```bash
cd 02/demo
```

# 1. Git workflow (GUI first, CLI alongside)

Use a small practice repository in VS Code. Start in **View → Source Control**: open or initialize the repository, edit a file, review the diff, stage the change, enter a commit message, and commit. Use the branch menu in the status bar to create or switch branches, then use the Source Control menu to merge and sync. The GUI is the primary path; the equivalent commands make each action visible:

```bash
git status
git add path/to/file.py
git commit -m "Describe the change"
git switch -c experiment
git merge experiment
git log --oneline --graph --all
git push
```

For an already shared commit, prefer `git revert <commit>`: it records an undo without rewriting history. This guide deliberately does not teach `git reset --hard` or force-pushing. Those recovery operations require a verified disposable repository, an identified backup, and agreement from every affected collaborator.

# 2. Functions: refactor a script into reusable helpers

```bash
python3 functions_demo.py
```

`functions_demo.py` creates `sample_students.csv` and introduces reusable functions implemented in the import-safe `student_tools.py` module.

# 3. Modules: import the helpers in a second script

```bash
python3 module_usage_demo.py
```

`module_usage_demo.py` imports `student_tools.py` as an ordinary module and reuses its functions to create reports. Run Demo 2 first so `sample_students.csv` exists.
