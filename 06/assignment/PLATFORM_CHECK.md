# Assignment 06 artifact and platform check

Complete this check before preparing artifacts. Notebook execution is optional
local QA; the committed CSV artifacts are the assignment contract.

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
pandas 3.0.5
```

The printed interpreter path should be inside the environment you activated.

## 2. Notebook workflow

In the supplied notebook, launch Jupyter or open it through VS Code after
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

When working in the notebook, the kernel must report CPython 3.12.13, NumPy 2.0.2,
and pandas 3.0.5, and its interpreter path must match the intended environment.

## 4. Check the portable data root

Confirm the supplied fixture files and `data/fixture.json` are unchanged. If you
use the notebook, run its supplied setup cell without editing it; it verifies
fixture set `a06-structural-wrangling-v1` from either:

- a standalone Assignment 06 repository containing `data/fixture.json`; or
- the full course repository containing `06/assignment/data/fixture.json`.

A missing or checksum-mismatched fixture is a stop condition. Restore the
supplied files; do not add a fallback, upload prompt, absolute path, or download.

## 5. Final local check

Create the five CSV artifacts, confirm they appear in the Git GUI, and run:

```bash
python check_assignment.py
```

Submit only after the public checker reports that all checks passed.
