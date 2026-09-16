---
notion:
  title_line: "# Lecture 02 Demo Guide: Git, Functions, and Modules"
  role: demo
  status: mapped
  page_id: "3dbd9fdd-1a1a-813d-936e-e7011563ebde"
  url: "https://app.notion.com/p/3dbd9fdd1a1a813d936ee7011563ebde"
---

# Lecture 02 Demo Guide: Git, Functions, and Modules

# 1. Git workflow

From the repository root, create `scratch/git-practice`, then use **File → Open Folder…** to open it in VS Code. Open **View → Source Control** (Ctrl+Shift+G, including Control on macOS) and select **Initialize Repository**. If the initial branch is not `main`, open **View → Command Palette…** (Ctrl+Shift+P on Windows/Linux, Cmd+Shift+P on macOS), select **Git: Create Branch**, and name it `main`.

Create `notes.md` with `# Practice notes`. In Source Control, stage it with the `+` button, enter `Start practice notes`, and select the visible **Commit** button. The changes list is empty after the commit: the working tree is clean.

Click the branch name in the status bar → **Create new branch…** → `experiment`. Add `Experiment: compare two grade summaries.` to `notes.md`. The Source Control view now shows a working (unstaged) change; select the file to inspect its diff. Stage it: the file moves to **Staged Changes**. Enter `Add experiment note` and select **Commit**; the lists are empty again.

Click the branch name → select `main`. Open the Command Palette, select **Git: Merge Branch…**, and choose `experiment`. The experiment change is now committed on `main`.

Alternatively, start at the repository root (with a `scratch` directory) and create the practice repository using terminal commands:

```bash
mkdir scratch/git-practice
cd scratch/git-practice
git init
git checkout -b main
echo "# Practice notes" > notes.md
git add notes.md
git commit -m "Start practice notes"
git checkout -b experiment
echo "Experiment: compare two grade summaries." >> notes.md
git status                    # working: notes.md is modified, not staged
git diff                      # working: shows the new line
git add notes.md
git status                    # staged: notes.md is ready to commit
git commit -m "Add experiment note"
git status                    # committed: working tree clean
git checkout main
git merge experiment
git log --oneline             # shows both commits; press q if needed
cd ../../02/demo
```

## Less typing: recall and edit

Open `scratch/git-practice` with **File → Open Folder**, then **Terminal → New Terminal**:

1. Type `cat no`, press **Tab** to complete `notes.md`, then **Enter**. Expect the practice heading and experiment note.
2. Press **↑** to recall it, then **Ctrl+A** to move to the start. Press **Delete** three times to remove `cat` (on Mac, **Fn+Delete**), type `git diff --`, and press **Ctrl+E**. The line should read `git diff -- notes.md`.
3. Press **Enter**. Expect no output: you already committed and merged those changes.
4. Press **Ctrl+R**, type `cat no`, and check that `cat notes.md` appears. Press **Esc** to accept the match, then **Enter** to run it again.

For the Python demos, open the course's `02/demo` folder in VS Code and use **Terminal → New Terminal** (Ctrl+Shift+backtick).

# 2. Functions: refactor repeated work into helpers

```bash
python3 functions_demo.py
```

The complete scripts are [functions_demo.py](functions_demo.py) and [student_tools.py](student_tools.py). The demo starts with student records in memory, then calls helpers to get grades, calculate an average, and find the highest grade.

```python
grades = get_grades(students)
print(f"Average grade: {calculate_average(grades):.1f}")
print(f"Highest grade: {find_highest_grade(grades)}")
```

`get_grades()` uses an ordinary `for` loop and `.append()`, so the same extraction work has one name instead of being copied into every analysis.

Expected checkpoints:

```text
Before loop extracted: [85, 92, 78]
After get_grades() extracted: [85, 92, 78]
Average grade: 85.0
Highest grade: 92
```

# 3. Modules: reuse helpers in an import-safe script

```bash
python3 -c "import module_usage_demo"
python3 module_usage_demo.py
```

The first command is intentionally silent: importing does not run `main()` or create a report. The complete script is [module_usage_demo.py](module_usage_demo.py). It reuses those helpers, writes `grade_report.txt`, reads it back, and confirms that the saved text matches.

```python
if __name__ == "__main__":
    main()
```

Expected output:

```text
Read back from grade_report.txt:
Alice: 85
Bob: 92
Charlie: 78
Average grade: 85.0
Highest grade: 92
Saved report matches: True
```

Open `grade_report.txt` in the Explorer: it contains the five report lines, without the terminal's heading or verification message.
