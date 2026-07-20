# Assignment 06 local Jupyter platform check

Complete this check before editing `assignment.ipynb`. The Python program that
launches Jupyter and the Python kernel selected inside a notebook can be
different programs; both must point to your Assignment 06 environment.

## 1. Check the terminal interpreter

Activate the environment created in `06/assignment`, then run:

```bash
python --version
python -c "import sys, numpy, pandas; print(sys.executable); print(numpy.__version__); print(pandas.__version__)"
```

Expected versions:

```text
Python 3.12.13
NumPy 2.0.2
pandas 3.0.3
```

The printed interpreter path should be inside the environment you activated.

## 2. Launch Jupyter from that environment

Launch Jupyter or open the notebook through VS Code only after activating the
environment. Select a portable Python 3 kernel backed by the same interpreter.
This assignment is supported in local Jupyter; it has no Colab workflow or
badge.

## 3. Check the notebook kernel

Temporarily run this in a notebook cell, then remove the temporary cell:

```python
import sys
import numpy as np
import pandas as pd

print(sys.version)
print(sys.executable)
print(np.__version__)
print(pd.__version__)
```

The kernel must report CPython 3.12.13, NumPy 2.0.2, and pandas 3.0.3, and its
interpreter path must match the intended environment. If not, stop and change
the notebook kernel before doing assignment work.

## 4. Check the portable data root

Run the supplied setup cell without editing it. It must print an assignment root
and verify fixture set `a06-structural-wrangling-v1`. It works from either:

- a standalone Classroom50 repository containing `data/fixture.json`; or
- the full course repository containing `06/assignment/data/fixture.json`.

A missing or checksum-mismatched fixture is a stop condition. Restore the
supplied files; do not add a fallback, upload prompt, absolute path, or download.

## 5. Final local check

Restart the kernel, run all 25 cells in order, confirm the five CSVs appear in
the Git GUI, and run:

```bash
python check_assignment.py
```

Submit only after the public checker reports that all checks passed.
