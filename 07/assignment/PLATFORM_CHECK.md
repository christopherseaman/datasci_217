# Assignment 07 local Jupyter and repository check

Complete this check before editing `assignment.ipynb`. The Python program that
launches Jupyter and the Python kernel selected inside a notebook can be
different programs; both must point to your Assignment 07 environment.

## 1. Check the terminal interpreter

Activate the environment created in `07/assignment`, then run:

```bash
python --version
python -c "import sys, numpy, pandas, matplotlib, seaborn; print(sys.executable); print(numpy.__version__, pandas.__version__, matplotlib.__version__, seaborn.__version__)"
```

Expected versions:

```text
Python 3.12.13
NumPy 2.0.2
pandas 3.0.5
Matplotlib 3.11.1
seaborn 0.13.2
```

The printed interpreter path should be inside the environment you activated.

## 2. Launch local Jupyter from that environment

Launch Jupyter or open the notebook through VS Code only after activating the
environment. Select a portable Python 3 kernel backed by the same interpreter.
This initial assignment release is supported in clean local Jupyter; it has no
Assignment Colab workflow or badge.

## 3. Check the notebook kernel

Temporarily run this in a notebook cell, then remove the temporary cell:

```python
import sys
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

print(sys.version)
print(sys.executable)
print(np.__version__)
print(pd.__version__)
print(matplotlib.__version__)
print(sns.__version__)
```

The kernel must report the exact versions above, and its interpreter path must
match the intended environment. If not, stop and change the notebook kernel.

## 4. Check the portable fixture root

Run the supplied setup cell without editing it. It must print an assignment
root and verify fixture set `a07-visualization-v1`. It works from either:

- a standalone Assignment 07 repository containing `data/fixture.json`; or
- the full course repository containing `07/assignment/data/fixture.json`.

A missing, unexpected, or checksum-mismatched fixture is a stop condition.
Restore the supplied files; do not add a fallback, upload prompt, absolute
path, or download.

## 5. Restart, run, and inspect

Restart the kernel and run all 23 cells in order. Confirm that the exploratory
chart and the two saved teaching figures are visible in the notebook. Inspect
the charts yourself: automated checks cannot certify honesty, clarity,
accessibility, or visual quality. Then run:

```bash
python check_assignment.py
```

## 6. Commit and push with a Git GUI

In VS Code Source Control or GitHub Desktop, confirm that the completed
`assignment.ipynb` and all five files under `output/` are visible changes.
Review the diff, commit them, and push the commit used for submission. The
outputs are deliberately not ignored.

## 7. Review optional Actions feedback

The repository's optional Actions workflow runs the public pytest contract. Read
the per-test feedback there if you enable it. If a test fails, correct the source notebook,
restart and run all again, rerun `python check_assignment.py`, inspect the six
deliverables in the Git GUI, commit, push, and resubmit according to the course
policy. Automated results do not replace the separate human
visual and communication review.
