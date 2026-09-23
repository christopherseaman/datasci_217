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
    version: 3.13
---

# Demo 1: Notebook runtime, state, and fresh execution

**Learning objectives**

- Distinguish a notebook document from its running kernel state.
- Explain why visible cell order and execution order can disagree.
- Produce a stale value and a `NameError` by hand, then repair both with restart-and-run-all.
- Check where the kernel is running with the `%pwd` and `%ls` magic commands.
- Identify runtime-local files and apply the notebook output/privacy policy.

A **notebook** is a document made of Markdown cells and code cells. A **kernel** is the Python process that executes code. The kernel's **state** is the collection of names and values currently held in memory. A Colab **runtime** includes that kernel and its runtime-local files. **Stored output** is text or another result saved beneath a cell; it can remain visible even when it no longer describes current state.

Colab is the default launch experience; local Jupyter uses the equivalent restart-kernel and run-all controls. "Repair the hidden dependency" below walks you through editing and rerunning cells out of order. GitHub source opened in Colab is not automatically updated by edits in the Colab tab.

The later pandas demonstrations use the pinned activity environment documented
in this notebook. It deliberately stays with notebook mechanics
and core Python; it does not import pandas before the lecture introduces it.
Never place credentials, tokens, protected records, or identifying data in
notebook source or output.

```python
import sys

print("Python:", sys.version.split()[0])
```

## Cell types and execution

Markdown cells explain, predict, and interpret. Code cells send Python to the kernel. Running a code cell can change state and create stored output; merely editing its visible source does neither.

Before you change anything, write down in a Markdown cell what you expect the next code cell to print. Predicting first is what turns a surprise into information.


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

The cell above printed `36` because the producer cell ran first. Do this by hand now, in this notebook, to see the two failures for yourself:

1. Change `rate = 3` to `rate = 2` in the producer cell and run **only** that cell. Then run the last cell again. It prints `total in kernel: 36`, a stale value: `total` still holds the old product, because nothing recomputed it.
2. Run the dependent cell (`total = units * rate`) and the last cell again. Now it prints `24`. The notebook's visible source never said `36` or `24` was correct; execution order decided.
3. Change `rate` back to `3`, then restart the kernel and run the last cell on its own. It raises `NameError: name 'rate' is not defined`, because a fresh kernel holds nothing at all.

**Restart-and-run-all** means starting with empty kernel state and executing every cell from top to bottom. Do that now: the last cell must print `36` again. Stored output alone is never evidence that this happened.


## Where the kernel is running

A **magic command** is a notebook-only shortcut that starts with `%`. `%pwd` reports the kernel's current working directory and `%ls` lists the files it can see from there. Check both first whenever a notebook cannot find a file: in Colab the directory is usually `/content`, and locally it is usually the notebook's own folder. A cell shows the value of its last line only, so `%pwd` gets a cell to itself.

```python
%pwd
```

```python
%ls
```

## Runtime-local files

A **runtime-local file** belongs to the current execution environment rather than the course source. Colab may discard it when the runtime is deleted. A local temporary file may last longer, but reliable code still recreates it. The supplied path below does not depend on the directory from which Jupyter was launched.

```python
from pathlib import Path
from tempfile import gettempdir

runtime_dir = Path(gettempdir()) / "datasci_217_lecture04_demo1"
runtime_dir.mkdir(parents=True, exist_ok=True)
runtime_note = runtime_dir / "runtime_note.txt"

# Write and read it back with open(), from Lecture 02.
with open(runtime_note, "w", encoding="utf-8") as file:
    file.write("runtime-local; safe demo content\n")

with open(runtime_note, "r", encoding="utf-8") as file:
    saved_text = file.read()

print("runtime-local file:", runtime_note)
print(saved_text, end="")
```

## Output and privacy policy

Clear sensitive output immediately and never put secrets or identifying records in a notebook. Ordinary non-sensitive output may support a human explanation, but it is not proof of execution: validation and grading run a fresh copy. Canonical demo notebooks are committed with outputs cleared and execution counts removed.

```python
assert units == 12
assert rate == 3
assert total == 36
assert saved_text == "runtime-local; safe demo content\n"

print("Demo 1 fresh-run verification passed: total = 36")
```
