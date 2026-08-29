---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    name: python
    version: 3.12.13
---

# Demo 1 — Notebook runtime, state, and fresh execution

**Learning objectives**

- Distinguish a notebook document from its running kernel state.
- Explain why visible cell order and execution order can disagree.
- Produce a real fresh-state failure and repair it with restart-and-run-all.
- Identify runtime-local files and apply the notebook output/privacy policy.

A **notebook** is a document made of Markdown cells and code cells. A **kernel** is the Python process that executes code. The kernel's **state** is the collection of names and values currently held in memory. A Colab **runtime** includes that kernel and its runtime-local files. **Stored output** is text or another result saved beneath a cell; it can remain visible even when it no longer describes current state.

Colab is the default launch experience; local Jupyter uses the equivalent restart-kernel and run-all controls. See `DEMO_GUIDE.md` for the live mutation protocol. GitHub source opened in Colab is not automatically updated by edits in the Colab tab.

The later pandas demonstrations use the pinned activity environment documented
in `DEMO_GUIDE.md`. This notebook deliberately stays with notebook mechanics
and core Python; it does not import pandas before the lecture introduces it.
Never place credentials, tokens, protected records, or identifying data in
notebook source or output.

```python
import sys

print("Python:", sys.version.split()[0])
```

## Cell types and execution

Markdown cells explain, predict, and interpret. Code cells send Python to the kernel. Running a code cell can change state and create stored output; merely editing its visible source does neither.

For the live demonstration, use a disposable copy to insert one prediction Markdown cell and one harmless code cell. Remove those scratch cells before the stale-state sequence described in the guide.


## Producer and dependent cells

The producer cell defines names. The dependent cell requires those names and computes another value. Their canonical order is reproducible; running them out of order in a fresh kernel is not.

```python
units = 12
rate = 3

print("units:", units)
print("rate:", rate)
```

```python
total = units * rate
print("total:", total)
```

```python
print("rate in kernel:", rate)
print("total in kernel:", total)
```

## Repair the hidden dependency

The live sequence temporarily creates a stale total of `24`, restarts into a real `NameError`, and then restores the canonical source. **Restart-and-run-all** means starting with empty kernel state and executing every cell from top to bottom. The canonical result below must be `36`; stored output alone is never evidence that this happened.


## Runtime-local files

A **runtime-local file** belongs to the current execution environment rather than the course source. Colab may discard it when the runtime is deleted. A local temporary file may last longer, but reliable code still recreates it. The supplied path below does not depend on the directory from which Jupyter was launched.

```python
from pathlib import Path
from tempfile import gettempdir

runtime_dir = Path(gettempdir()) / "datasci_217_lecture04_demo1"
runtime_dir.mkdir(parents=True, exist_ok=True)
runtime_note = runtime_dir / "runtime_note.txt"
runtime_note.write_text(
    "runtime-local; safe demo content\n",
    encoding="utf-8",
)

print("runtime-local file:", runtime_note)
print(runtime_note.read_text(encoding="utf-8"), end="")
```

## Output and privacy policy

Clear sensitive output immediately and never put secrets or identifying records in a notebook. Ordinary non-sensitive output may support a human explanation, but it is not proof of execution: validation and grading run a fresh copy. Canonical demo notebooks are committed with outputs cleared and execution counts removed.

```python
assert units == 12
assert rate == 3
assert total == 36
assert runtime_note.read_text(encoding="utf-8") == (
    "runtime-local; safe demo content\n"
)

print("Demo 1 fresh-run verification passed: total = 36")
```
