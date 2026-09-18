---
notion:
  title_line: "# Lecture 03 Demo Guide: Environments and NumPy"
  role: demo
  status: mapped
  page_id: "3a4d9fdd-1a1a-812d-9372-edd7cbeb8303"
  url: "https://app.notion.com/p/3a4d9fdd1a1a812d9372edd7cbeb8303"
---

# Lecture 03 Demo Guide: Environments and NumPy

Open `03/demo` in VS Code and select **Terminal → New Terminal**. Run the three demos below in order. Sources: [Lecture 03 demo files](https://github.com/christopherseaman/datasci_217/tree/main/03/demo).

# Setup: Create the tested environment

Lecture 03 uses CPython 3.13 and NumPy 2.3.3. With `uv`:

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
python -c "import numpy as np; print(np.__version__)"
```

Expect Python `3.13.x` and NumPy `2.3.3`. Use the course's Bash/Zsh terminal; in native PowerShell, activation is `.\.venv\Scripts\Activate.ps1`. The lecture also shows standard-library `venv` as an alternative setup.

# 1. Shell Pipeline

Run this from a disposable project-local directory because it creates `data/`, `logs/`, and `results/` below the current directory:

```bash
mkdir -p scratch/lecture03-cli
cd scratch/lecture03-cli
bash ../../01_cli_pipeline_demo.sh
cat results/summary_*.txt
cat logs/processing.log
cd ../..
```

The activity uses a bounded `tail | cut | sort | uniq | head` pipeline, plus `wc` for record counts, shell assignment, one captured timestamp reused in output names and log lines, and simple `echo >>` append logging. The longer shell-processing examples are optional reference material in the [bonus page](../BONUS.md).

Source: [01_cli_pipeline_demo.sh](01_cli_pipeline_demo.sh). Expect six records and these subject counts (padding may vary):

```text
1 English
3 Math
2 Science
```

Open the saved summary and log. Both use the same timestamp for this run; another run creates a new summary and appends log lines.

# 2. Python Collections and NumPy Performance

## 2.1 Inspect and Pair Python Values

```bash
python demo2_python_potpourri.py
```

Source: [demo2_python_potpourri.py](demo2_python_potpourri.py). Inspect the type before and after conversion, then pair names with grades. Expected checkpoints:

```text
Original value: 42 Type: <class 'str'>
Converted value: 42 Type: <class 'int'>
Strings have split: True
Reverse order: ['Charlie', 'Bob', 'Alice']
Sorted grades: [78, 85, 92]
```

## 2.2 Compare a List Loop with Array Arithmetic

```bash
python demo3_numpy_performance.py
```

Source: [demo3_numpy_performance.py](demo3_numpy_performance.py). Both approaches double the same values, so their result samples must match. Ignoring the timing lines, look for:

```text
=== Python List Approach ===
Result sample: [0, 2, 4, 6, 8]
=== NumPy Array Approach ===
Result sample: [0 2 4 6 8]
```

The timing wrapper uses `time.perf_counter()` to read a clock before and after each calculation; subtracting gives elapsed seconds. Timings and speedup vary by machine and run.

# 3. NumPy Student-Grade Analysis

```bash
python demo3_student_analysis.py
```

Source: [demo3_student_analysis.py](demo3_student_analysis.py). The seeded generator creates a `(100, 5)` grade array: 100 students, five assignments each. Follow the printed checkpoints:

- Compare the first student's original, doubled, and curved grades.
- Compare row averages (students) with column averages (assignments).
- Check that the three grade-band counts total 100.
- Follow the first 12 grades into a `(3, 4)` grid and its `(4, 3)` transpose.
- Use sorted indices to identify the top five students, then compare first and last assignments.

## Optional Python script practice

The two plain-Python analysis scripts use the bundled fixture and aggregate all subjects they find:

```bash
python demo1a_data_analysis.py
python demo1b_data_analysis_functions.py
```

For an optional terminal-data-processing discussion, inspect the fixture with commands such as:

```bash
head students.csv
cut -d',' -f4 students.csv | tail -n +2 | sort | uniq -c
awk -F',' 'NR > 1 {sum += $3; count += 1} END {print sum / count}' students.csv
```

`sparklines` is not part of this lecture's tested environment; install and use it only as an optional, separate terminal tool.
