# Lecture 04 Demo Guide: Jupyter and pandas

Lecture 04 has three executable demonstrations. The Markdown files are the
authoritative teaching sources; the checked-in notebooks are generated from
them with Jupytext. The lecture itself explains the concepts, while these
demos are the top-to-bottom execution contract.

## Tested environment

The demos were exercised with CPython 3.12.13, NumPy 2.0.2, pandas 3.0.5,
JupyterLab 4.4.10, and Jupytext 1.18.1. From this directory, create an
environment and install the recorded requirements before launching Jupyter:

```bash
uv venv --python 3.12.13 .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
jupyter lab
```

Do not install packages from inside a demo notebook. If you edit a Markdown
source, regenerate its notebook explicitly and execute a fresh copy:

```bash
jupytext --to notebook --output demo2_pandas_basics.generated.ipynb demo2_pandas_basics.md
jupyter nbconvert --to notebook --execute \
  --output demo2_pandas_basics.executed.ipynb \
  demo2_pandas_basics.generated.ipynb
```

The generated and executed files are disposable artifacts; do not overwrite a
source notebook with execution output. A kernel with the documented packages
must be selected before running the notebooks.

## Demo 1 — Notebook runtime, state, and fresh execution

**Sources:** `demo1_jupyter_basics.md` → `demo1_jupyter_basics.ipynb`
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/04/demo/demo1_jupyter_basics.ipynb)

This demonstration introduces the notebook document/kernel distinction,
producer and dependent cells, a real stale-state failure, restart-and-run-all,
runtime-local files, and the output/privacy policy. It belongs immediately
after the lecture introduces Jupyter and before pandas structures. The live
mutation is performed in a disposable copy so the committed source remains
clean.

## Demo 2 — From NumPy arrays to labeled pandas

**Sources:** `demo2_pandas_basics.md` → `demo2_pandas_basics.ipynb`
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/04/demo/demo2_pandas_basics.ipynb)

This demonstration starts with the already-taught NumPy array mental model,
then introduces `Series`, `DataFrame`, bounded inspection, bracket selection,
`.loc`/`.iloc`, one boolean mask, a derived column, and deterministic sorting.
It is placed after those pandas structures and selection operations are
defined in the lecture; it does not depend on later cleaning, grouping, or
visualization material.

## Demo 3 — Portable CSV round trip

**Sources:** `demo3_data_io.md` → `demo3_data_io.ipynb`
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/04/demo/demo3_data_io.ipynb)

This demonstration resolves a path relative to the notebook, verifies a
small committed CSV, reads it with pandas, selects a result, writes it to a
separate output path, reads it back, and checks the round trip. It belongs at
the end of Lecture 04 after file I/O, dtype inspection, and quality checks have
been introduced. The fixture is intentionally local and small, so execution
does not require a network connection or an untracked input file.

## Instructor checklist

- Demonstrate each notebook from a fresh kernel and run cells top-to-bottom.
- Keep the three demonstrations in lecture order: notebook state → labeled
  data structures → portable file I/O.
- Clear outputs before committing any notebook; the repository copies are kept
  output-free.
- Treat the exact pins in `requirements.txt` as the activity contract. Update
  the Markdown source and regenerate its notebook together when the contract
  changes.
