# Lecture 04 demo guide

These three demos introduce notebooks, bridge the NumPy ideas from Lecture 03 to labeled pandas objects, and finish with a portable CSV round trip. Colab is the default launch experience; the same notebooks must also run top-to-bottom in local Jupyter.

## Launch the demos

The development badges point to the `eleventy` branch. Work opened from GitHub in Colab is not automatically saved back to GitHub.

| Demo | Colab | Local notebook | Purpose |
|---|---|---|---|
| 1. Notebook runtime and state | [![Open Demo 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/04/demo/demo1_jupyter_basics.ipynb) | `demo1_jupyter_basics.ipynb` | Expose stale state, a real restart failure, and the repair |
| 2. NumPy to labeled pandas | [![Open Demo 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/04/demo/demo2_pandas_basics.ipynb) | `demo2_pandas_basics.ipynb` | Construct, inspect, select, filter, derive, and sort |
| 3. Portable CSV round trip | [![Open Demo 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/04/demo/demo3_data_io.ipynb) | `demo3_data_io.ipynb` | Read a pinned input and reproduce one verified output |

Before publication, replace `eleventy` in all three badge targets with one immutable release tag. Open and fresh-run every resulting URL before calling the demos certified.

## Environment candidate

The compatibility candidate is Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. It is not the final course lock until both local-Jupyter and fresh-Colab certification are complete. Do not install pandas 3.0.4.

Each notebook begins with one supplied setup cell. It conditionally installs pandas 3.0.3 before the first pandas import and then prints the actual Python, NumPy, and pandas versions. It does not reinstall the complete Colab package collection.

For local use, start from this directory, create a Python 3.12.13 environment, and install the two direct course dependencies from `requirements.txt`. Open the notebook with the course Jupyter or VS Code host and select that environment as the Python 3 kernel. Jupyter host and kernel-support packages are platform tooling rather than lecture imports; record their versions during certification.

The portable kernelspec is intentionally named `Python 3`, not after a local `.venv`. Local Jupyter and Colab should therefore differ only in launch controls and runtime-local paths, not in teaching code.

## Demo 1: live state failure and repair

Use a disposable Colab session or a working copy. Do not save the temporary edits over the canonical notebook.

1. Insert a Markdown cell containing a prediction and a code cell containing a harmless print statement. Run the code cell, observe its stored output, and remove both scratch cells.
2. In the producer cell, temporarily change `rate` from `3` to `2`.
3. Run the producer and dependent cells. The displayed total is `24`.
4. Edit the visible producer source back to `rate = 3` without running it. The displayed total remains `24` because editing source does not update kernel state or stored output.
5. Run only the producer, then run the observer. The kernel now reports `rate = 3`, while the previously computed `total` remains stale at `24`.
6. Restart the runtime or kernel and run the dependent cell alone. This must produce a real `NameError` because the producer names do not exist in fresh state.
7. Run all cells from the canonical top-to-bottom order. The final total is `36`, the runtime-local file is recreated, and the final verification message appears.

Colab controls are **Runtime → Restart session** and **Runtime → Run all**. Use the equivalent restart-kernel and run-all controls in local Jupyter.

The committed notebook contains no intentionally failing cell: its canonical source always has `rate = 3`, so automated fresh execution succeeds. Reload the canonical source after the live mutation sequence.

## Demo 2: expected checkpoints

The first object is a three-value Series named `temperature_c`, labeled `north`, `south`, and `west`. The second begins as a four-row, two-column DataFrame whose index is named `record_id`.

Check these observable results in order:

- `head(3)`, direct `info()`, and numeric `describe()` inspect a bounded table without making a cleaning decision;
- bracket selection distinguishes a Series from a one-column DataFrame;
- matching `.loc` and `.iloc` requests return the same values;
- the one mask selects `obs-002` and `obs-003`;
- the one derived column contains four values of `10`; and
- the unique `record_id` tie-breaker orders the rows `obs-001`, `obs-002`, `obs-003`, `obs-004`.

The last cell asserts these results and prints the fresh-run verification message.

## Demo 3: path and round-trip checkpoints

Inside the course repository, the notebook must find the committed `data/anscombe.csv` while launched from either the repository root or a nested directory. Outside the repository, it must fetch the same bytes from the immutable upstream commit into a runtime-local `data` directory. Both paths verify the same SHA-256 checksum before pandas reads the file.

Repository executions write `04/demo/output/selected_anscombe.csv`. Non-repository executions write `output/selected_anscombe.csv` under the current working directory. Code creates either output directory when needed.

The input has shape `(44, 3)` and columns `dataset`, `x`, and `y`. Selecting `x >= 13` produces seven rows, preserves those three columns, and writes no DataFrame index column. Reading the file back must reach the final verification message without a manual upload, Drive mount, or previously defined name.

For portability testing, use disposable repository copies. Confirm the committed-fixture case, remove the fixture only in a disposable copy to confirm the fallback, corrupt it only in another disposable copy to confirm the checksum failure, and repeat a clean run to confirm deterministic replacement.

## Output and privacy policy

- Runtime files are not durable course storage. Colab may discard them when its runtime is deleted; local temporary files may outlive a kernel but are still recreated by code.
- Never put credentials, tokens, protected records, or identifying data in notebook source or output.
- Clear sensitive output immediately. These demos use only non-sensitive teaching values.
- Stored output is never proof that code runs. Certification and grading execute a fresh copy.
- Canonical demo notebooks are committed with cleared outputs and null execution counts. Executed certification copies are disposable.
- GitHub source opened in Colab is not automatically updated by edits made in the Colab tab.

## Certification record

Do not mark a row as passing without independent evidence from that environment.

| Notebook | Paired Markdown | Local candidate | Fresh Colab | Badge release ref |
|---|---|---|---|---|
| `demo1_jupyter_basics.ipynb` | none — canonical notebook policy | pending | pending | development: `eleventy` |
| `demo2_pandas_basics.ipynb` | none — canonical notebook policy | pending | pending | development: `eleventy` |
| `demo3_data_io.ipynb` | none — canonical notebook policy | pending | pending | development: `eleventy` |

For each certification run, record the notebook path, environment, Python/NumPy/pandas versions, launch working directory, fixture source, final verification result, tester, date, and immutable release ref. Do not treat this guide or stored notebook output as independent certification.
