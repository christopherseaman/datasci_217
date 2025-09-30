# Demo Guide - Lecture 3: NumPy Arrays and Data Science Tools

This guide provides hands-on demonstrations for Lecture 3, organized to match the lecture flow.

## Demo Structure

Each demo corresponds to a **LIVE DEMO!** callout in the lecture:

1. **Demo 1**: Assignment 02 Walkthrough (no script)
2. **Demo 2**: Virtual Environments and Python Potpourri → `demo2_python_potpourri.py`
3. **Demo 3**: NumPy Arrays and Operations → `demo3_numpy_performance.py` + `demo3_student_analysis.py`
4. **Demo 4**: Command Line Data Processing → `demo4_cli_processing.sh`

---

# Demo 1: Assignment 02 Walkthrough

## Key Teaching Points

### 1a. Repository Setup
- Initialize Git repository
- Create meaningful README
- First commit with good commit message
- Show `git log --oneline`

### 1b. Feature Branch Workflow
- Create feature branch: `git checkout -b feature/project-scaffold`
- Build project structure with automation script
- Create `.gitignore` for Python projects
- Commit changes to feature branch

### 1c. Python Integration
- Write Python script for data analysis
- Use CLI tools to prepare data
- Integrate Python with shell scripts
- Test the complete workflow

### 1d. Merge and Tag
- Merge feature branch to main
- Create annotated tag: `git tag -a v1.0 -m "Release 1.0"`
- Show complete history: `git log --oneline --graph --all`

## Common Questions

**Q: "Why not just work on main?"**
A: Feature branches let you experiment without breaking working code. Multiple people can work simultaneously.

**Q: "What if I mess up a commit?"**
A: `git commit --amend` for last commit, or `git revert` for older commits. Show examples.

**Q: "How do I know what to commit together?"**
A: One logical change per commit. If you can't write a concise commit message, it's probably too much.

---

# Demo 2: Virtual Environments and Python Potpourri

**Script** - `demo2_python_potpourri.py`

## Demo Flow

### 2a. Virtual Environment Setup

Show creating and activating a virtual environment:

```bash
# Create virtual environment in .venv (standard location)
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install required packages
pip install numpy pandas matplotlib

# Save dependencies
pip freeze > requirements.txt
```

### 2b. Run Python Potpourri Demo

Execute the demo script:

```bash
source .venv/bin/activate  # Activate the virtual environment first
python demo2_python_potpourri.py
```

**Key Demonstrations**
1. **Type Checking** - Show how to check types with `type()`
2. **Type Conversions** - String → int, int → float, etc.
3. **F-String Formatting** - Modern Python string formatting
4. **Formatting Options** - Decimal places, alignment, padding

## Common Questions

**Q: "Can I have multiple virtual environments?"**
A: Yes! One per project is common. They're independent.

**Q: "What's the difference between f-strings and .format()?"**
A: F-strings are faster, more readable, and the modern Python way (3.6+).

**Q: "Why do I need type checking if Python is dynamically typed?"**
A: For debugging! `type()` helps you understand what you're actually working with.

---

# Demo 3: NumPy Arrays and Operations

**Scripts**
- `demo3_numpy_performance.py` (performance comparison)
- `demo3_student_analysis.py` (practical operations)

## 3a. Performance Comparison

Run the performance demo:

```bash
source .venv/bin/activate  # Activate the virtual environment first
python demo3_numpy_performance.py
```

**Expected Output**
- Python list operations: tens of milliseconds
- NumPy array operations: single-digit milliseconds
- **10-100x speedup** (exact speedup varies by system)

## 3b. Student Grade Analysis

Run the practical demo:

```bash
source .venv/bin/activate  # Activate the virtual environment first
python demo3_student_analysis.py
```

**Key Demonstrations**

1. **Array Creation**
   - From lists - `np.array([[...]])`
   - Properties - `shape`, `dtype`, `size`

2. **Indexing and Slicing**
   - Single elements - `grades[0, 1]`
   - Rows/columns - `grades[0, :]`, `grades[:, 1]`
   - Slices - `grades[1:3, :]`

3. **Boolean Indexing** (most important!)
   - Create mask - `grades > 85`
   - Filter data - `grades[grades > 85]`
   - Multiple conditions - `(grades > 80) & (grades < 90)`

4. **Statistical Operations**
   - Basic - `.mean()`, `.std()`, `.max()`, `.min()`
   - Axis operations - `grades.mean(axis=0)` vs `grades.mean(axis=1)`
   - "Axis 0 = down the rows (column stats), Axis 1 = across columns (row stats)"

5. **Array Reshaping**
   - `.reshape()` - Change dimensions
   - `.flatten()` - 2D → 1D
   - `.T` - Transpose

**Pause Points**
- After boolean indexing - "How would you find students with grades between 80-90?"
- After axis operations - "What's the difference between axis=0 and axis=1?"
- After reshaping - "Why would you need to reshape an array?"

## Common Questions

**Q: "Why does slicing give a view, not a copy?"**
A: Performance! Views save memory. Use `.copy()` when you need independence.

**Q: "What's the difference between `grades[grades > 85]` and `np.where(grades > 85)`?"**
A: Boolean indexing returns values, `np.where()` returns indices. Both are useful.

**Q: "Why do I get a 1D array when I slice a single row?"**
A: NumPy reduces dimensions. Use `grades[0:1, :]` to keep 2D.

---

# Demo 4: Command Line Data Processing

**Data file** - `students.csv` (already exists in repo)

## Demo Flow

Run these commands manually to demonstrate CLI data processing:

**4a. `cut`** - Extract columns
```bash
cut -d',' -f1,3 students.csv | head -5
```
- `-d','` - Set delimiter to comma
- `-f1,3` - Select fields 1 and 3
- Use case - Quick column extraction from CSV

**4b. `sort`** - Sort data
```bash
sort -t',' -k3 -n students.csv | head -5
```
- `-n` - Numerical sort (not alphabetical)
- `-t','` - Set delimiter
- `-k3` - Sort by field 3
- Common mistake - Forgetting `-n` for numbers!

**4c. `uniq`** - Count occurrences
```bash
cut -d',' -f4 students.csv | sort | uniq -c
```
- **Must sort first!** `uniq` only removes adjacent duplicates
- `-c` - Count occurrences
- Pattern - `sort | uniq -c` is very common

**4d. `grep`** - Search and filter
```bash
grep "Math" students.csv
```
- Basic search - `grep "Math"`
- `-v` - Inverse match (NOT)
- `-i` - Case-insensitive
- Use case - Quick filtering

**4e. `tr`** - Transform characters
- Character replacement - `tr 'a-z' 'A-Z'`
- Delete characters - `tr -d ' '`
- Use case - Case conversion, cleanup

**4f. `sed`** - Stream editor
- Replace - `sed 's/old/new/g'`
- Delete lines - `sed '1d'` (removes header!)
- Use case - More powerful than `tr`

**4g. `awk`** - Pattern processing
```bash
awk -F',' 'NR>1 {sum+=$3; count++} END {print "Average grade:", sum/count}' students.csv
```
- Print columns - `awk '{print $1, $3}'`
- Filter rows - `awk '$3 > 85'`
- Calculate - `awk '{sum+=$3} END {print sum/NR}'`
- **Most powerful tool for structured data**

**4h. Complex pipelines**
How to chain commands:
```bash
grep "Math" students.csv | \
  cut -d',' -f1,3 | \
  sort -t',' -k2 -nr | \
  head -n 3
```

**Points** - Break down the pipeline step by step
1. Filter - What rows do we keep?
2. Extract - What columns do we need?
3. Sort - What order?
4. Limit - How many results?

**4i. Sparklines** - Inline graphs

Install (in your venv):
```bash
source .venv/bin/activate
pip install sparklines
```

Basic usage:
```bash
cut -d',' -f3 students.csv | tail -n +2 | sparklines
```

With statistics:
```bash
cut -d',' -f3 students.csv | tail -n +2 | sparklines --stat-min --stat-max --stat-mean
```

**Teaching Points:**
- `tail -n +2` means "start at line 2" (skip header)
- Shows inline graphs perfect for SSH sessions and remote work
- Lightweight visualization without leaving the terminal
- Great for quick data exploration

# Common Mistakes and How to Address Them

## Git Issues

**Mistake** - Committing too much at once
- **Fix** - Break into logical chunks
- **Thought** - "If you can't write a concise commit message, it's too much"

**Mistake** - Forgetting to add files
- **Fix** - Use `git status` before committing
- **Thought** - Show `git status` output interpretation

## Python Issues

**Mistake** - Mixing string and numeric operations
- **Fix** - Use `type()` to check, convert with `int()`, `float()`, `str()`
- **Thought** - "Python won't guess - you need to convert explicitly"

**Mistake** - F-string syntax errors
- **Fix** - Remember the `f` prefix - `f"..."` not `"..."`
- **Thought** - Show the error message, explain how to read it

## NumPy Issues

**Mistake** - Modifying a view thinking it's a copy
- **Fix** - Use `.copy()` when you need independence
- **Thought** - Show how changes propagate through views

**Mistake** - Wrong axis specification
- **Fix** - Remember "axis is what you're collapsing"
- **Thought** - Show with small 2D array, visualize what each axis does

**Mistake** - Boolean indexing without parentheses
- **Fix** - `(grades > 80) & (grades < 90)` not `grades > 80 & grades < 90`
- **Thought** - Show the error, explain operator precedence

## CLI Issues

**Mistake** - Forgetting to sort before `uniq`
- **Fix** - Always `sort | uniq -c`
- **Thought** - Show what happens without sort

**Mistake** - Using alphabetical sort on numbers
- **Fix** - Add `-n` flag for numerical sort
- **Thought** - Show "10" sorting before "2" without `-n`

**Mistake** - Not skipping CSV headers
- **Fix** - Use `tail -n +2`, `sed '1d'`, or `awk 'NR>1'`
- **Thought** - Show error when trying to do math on header text

**Mistake** - Wrong field delimiter
- **Fix** - Remember `-d','` for CSVs, `-t','` for sort
- **Thought** - Show what happens with wrong delimiter (gets wrong columns)
