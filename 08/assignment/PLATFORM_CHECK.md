# Assignment 08 local Jupyter platform check

Complete this check before editing `assignment.ipynb`. The Python program that
launches Jupyter and the Python kernel selected inside a notebook can be
different programs; both must point to your Assignment 08 environment.

## 1. Check the terminal interpreter

Activate the environment created in `08/assignment`, then run:

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

The interpreter path should be inside the environment you activated.

## 2. Launch local Jupyter

Launch Jupyter or open the notebook through VS Code only after activating the
environment. Select a portable Python 3 kernel backed by the same interpreter.
This assignment has no Colab workflow or badge.

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
interpreter must match the intended environment. If not, stop and change the
notebook kernel before doing assignment work.

## 4. Check the portable assignment root

Run the supplied setup cell without editing it. It searches upward from the
kernel working directory and must print an assignment root after verifying
fixture `a08-support-requests-v1`. Supported checkouts include:

- a standalone Classroom50 repository containing `data/fixture.json`; or
- the full course repository containing `08/assignment/data/fixture.json`.

Launching from a nested directory inside the assignment is also supported. A
missing or checksum-mismatched fixture is a stop condition. Restore the supplied
files; do not add a fallback, upload prompt, absolute path, or download.

## 5. Run, inspect, and submit

Restart the kernel and run all 25 cells in order. Confirm exactly these five
generated CSVs appear in the Git GUI:

- `output/center_count_summary.csv`
- `output/center_summary.csv`
- `output/requests_with_context.csv`
- `output/center_channel_summary.csv`
- `output/mean_resolution_pivot.csv`

Then run:

```bash
python check_assignment.py
```

Commit and push the notebook and all five CSVs with VS Code Source Control or
GitHub Desktop. Submit through Classroom50, inspect its feedback and review
link, correct your notebook if needed, rerun from fresh state, commit and push
the corrected files, and resubmit according to the published course policy.
