# Lecture 01 demo guide

Run these three required demos in a POSIX-style terminal with Python 3.12. Use `python3` in place of `python` only when that is the working command established during onboarding.

## 1. Project folder, paths, and first script

From the course repository root:

```bash
mkdir lecture01-scratch
cd lecture01-scratch
pwd
```

In VS Code, select **File → Open Folder**, choose `lecture01-scratch`, and then select **Terminal → New Terminal**. Confirm that `pwd` ends in `lecture01-scratch`. In the Explorer, create `hello.py` and enter:

```python
print("Hello from DataSci 217!")
```

Save and run it in the integrated terminal:

```bash
python hello.py
```

Expected output:

```text
Hello from DataSci 217!
```

Demonstrate that a relative path starts at the working directory:

```bash
cd ..
pwd
python hello.py
python lecture01-scratch/hello.py
```

The first Python command should report that `hello.py` cannot be opened. The second should succeed because its relative path starts from the current directory. Keep the scratch folder for later Lecture 01 practice.

In VS Code, select **File → Open Folder** and reopen the course repository root. Then select **Terminal → New Terminal**, confirm that `pwd` ends at the course repository, and enter the demo directory:

```bash
cd 01/demo
```

The prepared copy of the script is `01_project_paths_first_script.py`.

## 2. Values, conversion, lists, a decision, and a loop

Run the script:

```bash
python 02_values_lists_decisions_loops.py
```

Before each section prints, identify the scalar values, the conversion from text to an integer, the first list element at index `0`, the calculation updated by the direct loop, and the boolean condition that selects the summary.

Expected final lines:

```text
Mean: 7.1 hours
Summary: The study mean met the seven-hour threshold.
```

Change the list to `[5.0, 6.0, 6.5, 6.5]`, predict the new final lines, save, and rerun. Restore the original list before continuing.

## 3. Read a traceback, fix the source, and rerun

Open `03_traceback_fix_rerun.py` in VS Code. Run it from `01/demo`:

```bash
python 03_traceback_fix_rerun.py
```

For each run, read the final traceback line, locate the referenced source line, make only the listed correction, save, and rerun:

1. `IndentationError`: indent `print("Participant target reached")` by four spaces.
2. `NameError`: change `participant_cout` to `participant_count`.
3. `TypeError`: convert `age_text` with `int(age_text)` before adding `1`.
4. `ValueError`: change `int("forty-two")` to `int("42")`.
5. `IndexError`: change the requested list index from `3` to `2`, and change the label from `Fourth` to `Third`.

The final rerun should exit successfully with:

```text
Participant target reached
Participants: 24
Age next year: 43
Baseline age: 42
Third measurement: 24
```

Restore `03_traceback_fix_rerun.py` to its original intentionally broken state after the demo so the next run begins with the `IndentationError`.
