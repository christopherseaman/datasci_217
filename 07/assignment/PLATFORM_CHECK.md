# Assignment 07 artifact and platform check

Complete this check before preparing artifacts. Notebook execution is optional
local QA; the committed artifacts are the assignment contract.

## 1. Check the terminal interpreter

Activate the environment created in `07/assignment`, then run:

```bash
python --version
python -c "import sys, numpy, pandas, matplotlib, seaborn, altair; print(sys.executable); print(numpy.__version__, pandas.__version__, matplotlib.__version__, seaborn.__version__, altair.__version__)"
```

Expected versions:

```text
Python 3.13
NumPy 2.3.3
pandas 3.0.5
Matplotlib 3.11.1
seaborn 0.13.2
Altair 5.5.0
```

The printed interpreter path should be inside the environment you activated.

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
import matplotlib
import seaborn as sns
import altair as alt

print(sys.version)
print(sys.executable)
print(np.__version__)
print(pd.__version__)
print(matplotlib.__version__)
print(sns.__version__)
print(alt.__version__)
```

If you run the notebook, the kernel must report the exact versions above and its
interpreter path must match the intended environment.

## 4. Check the portable fixture root

Confirm the supplied fixture files and `data/fixture.json` are unchanged. If you
use the notebook, run its supplied setup cell without editing it; it verifies
fixture set `a07-visualization-v1` from either:

- a standalone Assignment 07 repository containing `data/fixture.json`; or
- the full course repository containing `07/assignment/data/fixture.json`.

A missing, unexpected, or checksum-mismatched fixture is a stop condition.
Restore the supplied files; do not add a fallback, upload prompt, absolute
path, or download.

## 5. Create and inspect artifacts

Create the exported exploratory specification and two saved teaching figures. Inspect the
charts yourself: automated checks cannot certify honesty, clarity, accessibility,
or visual quality. Then run:

```bash
python check_assignment.py
```

## 6. Commit and push with a Git GUI

In VS Code Source Control or GitHub Desktop, confirm that the completed
notebook source and all six files under `output/` are visible changes.
Review the diff, commit them, and push the commit used for submission. The
outputs are deliberately not ignored.

## 7. Review optional Actions feedback

The repository's optional Actions workflow runs the public pytest contract. Read
the per-test feedback there if you enable it. If a test fails, regenerate the
artifacts, rerun `python check_assignment.py`, inspect the deliverables in the
Git GUI, commit, push, and resubmit according to the course
policy. Automated results do not replace the separate human
visual and communication review.

Public grading uses the same grading.py ruleset for students, pytest, and
graders. Run python check_assignment.py [submission_dir] or add --json for a
machine-readable datasci217/grading-result/v1 result; it reads artifacts only.
