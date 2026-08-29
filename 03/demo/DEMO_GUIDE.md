# Lecture 03 Demo Guide: Environments and NumPy

Run the scripts from `03/demo`. The two student-analysis scripts locate their
bundled `students.csv` relative to the script, so they also work when launched
from another directory. Their reports are written to `03/demo/output/`.

## 1. Create the tested environment

Lecture 03 uses CPython 3.12.13 and NumPy 2.0.2. With `uv`:

```bash
cd 03/demo
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import sys, numpy as np; print(sys.version.split()[0], np.__version__)"
```

The final command should report `3.12.13 2.0.2`. If `uv` is unavailable, use a
Python 3.12.13 interpreter to create `.venv`, then install the same pinned
`requirements.txt` with `python -m pip install -r requirements.txt`.

## 2. Python refresher and NumPy performance

```bash
python demo2_python_potpourri.py
python demo3_numpy_performance.py
```

The first script refreshes type checking and f-string formatting before the
NumPy examples. The second contrasts a Python loop with vectorized array
arithmetic. Performance ratios vary by machine; focus on the vectorized
operations rather than a fixed speedup.

## 3. NumPy student-grade analysis

```bash
python demo3_student_analysis.py
```

This end-of-lecture script applies array creation, indexing, masks, arithmetic,
and reductions to a reproducible grade table. It is the practical bridge to
the later pandas table workflow.

## Optional script and CLI practice

The two plain-Python analysis scripts use the bundled fixture and aggregate all
subjects they find:

```bash
python demo1a_data_analysis.py
python demo1b_data_analysis_functions.py
```

For an optional terminal-data-processing discussion, inspect the fixture with
commands such as:

```bash
head students.csv
cut -d',' -f4 students.csv | tail -n +2 | sort | uniq -c
awk -F',' 'NR > 1 {sum += $3; count += 1} END {print sum / count}' students.csv
```

`sparklines` is not part of this lecture's tested environment; install and use
it only as an optional, separate terminal tool.
