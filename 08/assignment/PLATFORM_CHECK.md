# Assignment 08 artifact and platform check

Complete this check before preparing artifacts. The notebook remains a required
coursework deliverable, although automated grading does not execute it.
local QA; the committed CSV artifacts are the assignment contract.

## 1. Check the terminal interpreter

Activate the environment created in `08/assignment`, then run:

```bash
python --version
python -c "import sys, numpy, pandas; print(sys.executable); print(numpy.__version__); print(pandas.__version__)"
```

Expected versions:

```text
Python 3.14
NumPy 2.3.3
pandas 3.0.5
```

The interpreter path should be inside the environment you activated.

## 2. Optional notebook workflow

If you use the notebook, launch Jupyter or open it through VS Code after
activating the environment. Select a portable Python 3 kernel backed by the
same interpreter. This assignment has no Colab workflow or badge.

## 3. Optional kernel check

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

If you run the notebook, the kernel must report CPython 3.14, NumPy 2.3.3,
and pandas 3.0.5, and its interpreter must match the intended environment.

## 4. Check the portable assignment root

Confirm the supplied fixture files and `data/fixture.json` are unchanged. If you
use the notebook, run its supplied setup cell without editing it; it searches
upward from the kernel working directory and verifies fixture
`a08-support-requests-v1`. Supported checkouts include:

- a standalone Assignment 08 repository containing `data/fixture.json`; or
- the full course repository containing `08/assignment/data/fixture.json`.

Launching from a nested directory inside the assignment is also supported. A
missing or checksum-mismatched fixture is a stop condition. Restore the supplied
files; do not add a fallback, upload prompt, absolute path, or download.

## 5. Create, inspect, and submit

Create exactly these five CSV artifacts and confirm they appear in the Git GUI:

- `output/center_count_summary.csv`
- `output/center_summary.csv`
- `output/requests_with_context.csv`
- `output/center_channel_summary.csv`
- `output/mean_resolution_pivot.csv`

Then run:

```bash
python check_assignment.py
```

Commit and push the five CSVs with VS Code Source Control or GitHub Desktop.
Notebook execution is optional local QA. Automated results do not replace the
separate human review.

Public grading uses the same grading.py ruleset for students, pytest, and
graders. Run python check_assignment.py [submission_dir] or add --json for a
machine-readable datasci217/grading-result/v1 result; it reads artifacts only.
