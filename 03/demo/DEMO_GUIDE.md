---
notion:
  title_line: "# Lecture 03 Demo Guide: Environments and NumPy"
  role: demo
  status: mapped
  page_id: "3a4d9fdd-1a1a-812d-9372-edd7cbeb8303"
  url: "https://app.notion.com/p/3a4d9fdd1a1a812d9372edd7cbeb8303"
---

# Lecture 03 Demo Guide: Environments and NumPy

All three demos run in one folder, `~/03-demo`, which the first command of Demo 1 creates and fills. Run every command in VS Code's **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac); on Windows, use the **WSL: Ubuntu** window from Lecture 01. Apart from `setup_demo.sh` and the data file `encounters.csv`, each file name starts with the demo that runs it: `demo1_` for the shell pipeline, `demo2_` for types, lists, and array basics, `demo3_` for the analysis.

# Demo 1: Virtual Environments, Shell Pipelines, and Scripts

## 1.1 Download the Demo Files

This one command downloads the demo files, the same `curl ... | sh` pattern you used to install uv in Lecture 01. Its source is [setup_demo.sh](setup_demo.sh), and 1.5 reads it line by line.

```bash
curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/03/demo/setup_demo.sh | sh
```

```text
Made ~/03-demo with the Lecture 03 demo scripts and encounters.csv.
Next: cd ~/03-demo
```

```bash
cd ~/03-demo
ls
```

`ls` lists eight files: `setup_demo.sh`, the six demo scripts from `demo1_cli_pipeline.sh` to `demo3_csv_summary.py`, and `encounters.csv`. The script never overwrites earlier work: run it a second time and `mkdir` reports that `~/03-demo` already exists (`File exists`), and nothing else happens. To start over, rename the old folder first with `mv ~/03-demo ~/03-demo-old`, then run the `curl` line again. If a download fails partway, do the same. To browse the files in VS Code, use **File → Open Folder…** and choose `03-demo` in your home folder; new terminals then start there.

## 1.2 Create the Environment with `pyproject.toml`

Pin Python, start the project, create and activate the environment, then add NumPy, as in the lecture's "Create and Verify an Environment" snippet:

```bash
uv python pin 3.13
uv init --bare
uv venv --seed
source .venv/bin/activate
uv add numpy==2.3.3
python --version
python -c "import numpy as np; print(np.__version__)"
python -c "import sys; print(sys.executable)"
cat pyproject.toml
```

From `source .venv/bin/activate` on, the prompt starts with `(03-demo)`. uv also prints lines that differ from run to run, such as how long each step took; these are the lines that show each step worked, in order, where `3.13.x` is whichever 3.13 release you have, such as `3.13.14`:

```text
Pinned `.python-version` to `3.13`
Initialized project `03-demo`
Using CPython 3.13.x
Creating virtual environment with seed packages at: .venv
 + numpy==2.3.3
Python 3.13.x
2.3.3
```

`uv add` recorded NumPy in the project file that `uv init --bare` started:

```toml
[project]
name = "03-demo"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "numpy==2.3.3",
]
```

The `sys.executable` line shows which interpreter `python` runs; it should sit inside this folder's `.venv`, such as `/Users/alice/03-demo/.venv/bin/python`. A path without `.venv` in it means the environment is not active in this terminal. `ls -a` now also lists `.python-version`, `pyproject.toml`, `uv.lock`, and `.venv`.

## 1.3 Recreate It from the Records

First leave the environment, and check which Python `python` runs now:

```bash
deactivate
python -c "import sys; print(sys.executable)"
```

The prompt no longer starts with `(03-demo)`, and the path has no `.venv` in it, such as `/Users/alice/.local/bin/python`. That is the Python Lecture 01 installed, and it has no NumPy, so importing NumPy fails. This error is expected:

```bash
python -c "import numpy as np"
```

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

This is the lecture's "When `import numpy` Fails" pitfall, and the `sys.executable` path, with no `.venv` in it, names the cause: the environment is not active. `source .venv/bin/activate` fixes it, and so does `uv run`, which runs a command in the project's `.venv` without activating it, as the next block does. If the import printed nothing instead, the Python you reached already has NumPy, as Anaconda's does, but it is still not the project's Python.

`.python-version`, `pyproject.toml`, and `uv.lock` are the records another person needs; `.venv/` is not shared. With the environment still off, rebuild it from those three files in a new folder, as in the lecture's "Recreate from the Records" snippet:

```bash
mkdir recreation-check
cp .python-version pyproject.toml uv.lock recreation-check/
cd recreation-check
uv venv --seed
uv sync
uv run python -c "import numpy as np; print(np.__version__)"
cd ..
```

Among the lines these commands print, these show the rebuild worked:

```text
Using CPython 3.13.x
 + numpy==2.3.3
2.3.3
```

The environment stays off because `uv sync` ignores an active environment from another folder and warns about it. `uv sync` installed exactly the NumPy that `uv.lock` records, and `uv run` ran Python in the new `.venv` without activating it.

## 1.4 Share It as `requirements.txt`

Tools such as pip and Colab read `requirements.txt` instead. Write one for the same environment from `uv.lock`, and look at it:

```bash
uv export --no-hashes > requirements.txt
cat requirements.txt
```

```text
# This file was autogenerated by uv via the following command:
#    uv export --no-hashes
numpy==2.3.3
    # via 03-demo
```

`uv export` also prints `Resolved 2 packages` in the terminal; only the lines above go into the file. The `# via` comment says which project needs NumPy. Now install from that file into a fresh environment, the way a `requirements.txt` project is set up:

```bash
mkdir pip-check
cp .python-version requirements.txt pip-check/
cd pip-check
uv venv --seed
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import numpy as np; print(np.__version__)"
deactivate
cd ..
source .venv/bin/activate
```

Among the lines these commands print, these show the install worked:

```text
 + numpy==2.3.3
2.3.3
```

Both routes installed NumPy 2.3.3. The last line matters: Demos 2 and 3 run in the `03-demo` environment, so check that the prompt starts with `(03-demo)` again.

## 1.5 Read a Shell Script

The command in 1.1 ran a shell script, and it saved a copy of itself. Open it:

```bash
cat setup_demo.sh
```

```bash
#!/bin/sh
# Download the Lecture 03 demo files into a new folder, ~/03-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/03/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from: the course repository's 03/demo folder on GitHub.
base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/03/demo"

# mkdir without -p stops the script here if ~/03-demo already exists, so earlier work is never overwritten.
mkdir ~/03-demo
cd ~/03-demo

# One download per file; -o saves it under the name that follows.
curl -fsSL "$base_url/setup_demo.sh" -o setup_demo.sh
curl -fsSL "$base_url/demo1_cli_pipeline.sh" -o demo1_cli_pipeline.sh
curl -fsSL "$base_url/demo2_types_and_lists.py" -o demo2_types_and_lists.py
curl -fsSL "$base_url/demo2_numpy_performance.py" -o demo2_numpy_performance.py
curl -fsSL "$base_url/demo2_numpy_arrays.py" -o demo2_numpy_arrays.py
curl -fsSL "$base_url/demo3_bp_analysis.py" -o demo3_bp_analysis.py
curl -fsSL "$base_url/demo3_csv_summary.py" -o demo3_csv_summary.py
curl -fsSL "$base_url/encounters.csv" -o encounters.csv

echo "Made ~/03-demo with the Lecture 03 demo scripts and encounters.csv."
echo "Next: cd ~/03-demo"
```

Read it top to bottom, the order the shell ran it:

- `#!/bin/sh` names the shell the script expects: `sh`, a smaller relative of Bash that runs the same basic commands.
- Lines starting with `#` are comments; the shell skips them when it runs a script.
- `set -eu` is plumbing you do not need to write yourself: it stops the script at the first failing command, so a failed download cannot leave you with a half-made folder and no warning.
- `base_url=...` stores the download address in a variable once, and every `curl` line uses it as `"$base_url/..."`, as the lecture's "Shell Variables and Timestamps" card does with `timestamp`.
- `mkdir ~/03-demo` and `cd ~/03-demo` make the folder and move into it. Without `-p`, `mkdir` fails when the folder exists, and `set -e` then stops the script before any download.
- Each `curl -fsSL URL -o FILE` line downloads one file: `-f` fails on a missing file instead of saving an error page, `-s` hides the progress bar, `-S` still shows errors, `-L` follows redirects, and `-o` saves to `FILE` instead of printing it.
- The two `echo` lines printed the output you saw in 1.1.

`curl ... | sh` ran it like this: `curl` printed the script's text, and the pipe sent that text into `sh`, which ran the commands in order, just as `bash FILE.sh` runs a saved file. That `sh` was a separate shell, so its `cd ~/03-demo` changed only its own folder, which is why the script tells you to run `cd ~/03-demo` yourself.

## 1.6 Run a Pipeline Script

[demo1_cli_pipeline.sh](demo1_cli_pipeline.sh) writes a six-record CSV of clinic encounters to `data/raw/encounters.csv`, counts the records with `wc -l`, and counts encounters per clinic with a `tail | cut | sort | uniq | head` pipeline, whose `head -n 5` caps what it prints at five lines. It creates `data/`, `logs/`, and `results/` in the current folder, so run it from `~/03-demo`:

```bash
bash demo1_cli_pipeline.sh
```

Two pieces of it are script plumbing, explained in its comments, that you do not need to write yourself: `set -euo pipefail` stops the script at the first failing command, and the lines between `<<'EOF'` and `EOF` are written into the CSV unchanged. Its pipelines span several lines with a trailing `\`, the line continuation from the lecture's "Shell Scripts" card. Your timestamp will differ from the one below:

```text
=== Lecture 03: CLI pipeline ===
Encounter records: 6
Clinics (with counts):
      3 Cardiology
      2 Nephrology
      1 Primary Care
Summary written to results/summary_20260922_184147.txt
=== Demo complete ===
```

On macOS, `wc -l` and `uniq -c` pad their counts with a different number of spaces than this output shows, so `Encounter records:` can be followed by several spaces before the `6`. Only the spacing differs; the counts are the same.

Open the saved summary and log:

```bash
cat results/summary_*.txt
cat logs/processing.log
```

The `*` in `summary_*.txt` is a wildcard from the lecture's "Wildcards and Searching", like the `*.csv` patterns in Lecture 02's `.gitignore`: the shell replaces the pattern with the name of every summary file before `cat` runs, so you never type the timestamp. The summary repeats the counts under the timestamp that named the file:

```text
run timestamp: 20260922_184147
encounters: 6
clinic counts:
      3 Cardiology
      2 Nephrology
      1 Primary Care
```

The log records the same run:

```text
20260922_184147 pipeline started
20260922_184147 wrote results/summary_20260922_184147.txt
```

One captured timestamp names the result file and labels both log lines. Run the script again a second or more later and you get a second summary file (so `cat results/summary_*.txt` then prints both) and two more log lines, which is how a timestamped run keeps every result instead of overwriting it. The lecture's pipeline snippets read the same `data/raw/encounters.csv`, so they now run in this folder too.

## 1.7 Save a Pipeline as a Script

Now write a script of your own, the way the lecture's "Save a Pipeline as a Script" snippet does, but for `encounters.csv`, the 1,500-row file Demo 3 analyzes. Run `cat > count_clinics.sh`, paste these lines, press **Enter**, then **Ctrl+C**:

```bash
#!/bin/bash
# Count encounters per clinic in the 1,500-row file; save the counts under this run's timestamp.
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p results
tail -n +2 encounters.csv \
  | cut -d',' -f4 | sort | uniq -c > "results/clinic_counts_${timestamp}.txt"
echo "Saved results/clinic_counts_${timestamp}.txt"
```

Check it with `cat count_clinics.sh`, then run it and open what it saved:

```bash
bash count_clinics.sh
cat results/clinic_counts_*.txt
```

If you ran `count_clinics.sh` before in this folder, the lecture's version included, `cat` prints those earlier files' counts too; the file this run saved is the one its `Saved` line names.

```text
Saved results/clinic_counts_20260922_184530.txt
    260 Cardiology
    145 Dermatology
    190 Endocrinology
    125 Nephrology
    150 Neurology
    120 Obstetrics
    130 Oncology
    380 Primary Care
```

Eight clinics, 1,500 encounters in all. Apart from its comment, only the input file differs from the lecture's script, which is the point of saving a pipeline: the same commands rerun on new data with one line. Demo 3.4 gets the same counts from Python.

## 1.8 Search with Wildcards and `grep`

Practice the lecture's "Wildcards and Searching" in the same folder: see what a wildcard expands to, count each file's Cardiology encounters a second way, and check the files' contents with patterns.

```bash
echo demo2_*.py
grep -c ',Cardiology$' data/raw/encounters.csv
grep -c ',Cardiology$' encounters.csv
grep -v '^P' encounters.csv
grep ',17.,' encounters.csv
```

```text
demo2_numpy_arrays.py demo2_numpy_performance.py demo2_types_and_lists.py
3
260
patient_id,age,systolic_bp,clinic
P0280,48,174,Neurology
```

- `echo` prints the arguments it receives, so the first line shows the three names the shell put in place of `demo2_*.py`.
- The two counts match 1.6's `3 Cardiology` and 1.7's `260 Cardiology`. `$` ties `,Cardiology` to the end of the line, the clinic column, and the single quotes pass the `$` to `grep` unchanged.
- `-v '^P'` prints the lines that do _not_ start with `P`. Only the header is left, so every record starts with a patient ID.
- In `,17.,` the `.` matches any one character, so the pattern finds a field from 170 to 179: one reading, 174 mmHg, the highest in the file, as Demo 3.4 reports.

# Demo 2: Types, Lists, and NumPy Basics

Run these three from `~/03-demo` with the `(03-demo)` environment active. In a new terminal, start with these two lines, after which the prompt starts with `(03-demo)`:

```bash
cd ~/03-demo
source .venv/bin/activate
```

## 2.1 Check Types and Loop over Lists

```bash
python demo2_types_and_lists.py
```

Source: [demo2_types_and_lists.py](demo2_types_and_lists.py). Heart rates exported as text arrive as `"88"`, not `88`. The script checks each value with `isinstance()` and converts the text ones, pairs patient IDs with the cleaned rates, then builds four lists in one line each:

```text
Checking Types and Looping over Lists
==================================================

=== Checking Types ===
Raw heart rates: ['88', 104, '112']
Types of the first two: <class 'str'> <class 'int'>
Cleaned: [88, 104, 112]
Average: 101.3 bpm
Strings have split: True

=== Sequence Functions ===
Numbered patients:
  Patient 1: P001
  Patient 2: P002
  Patient 3: P003
Paired records:
  P001: 88 bpm
  P002: 104 bpm
  P003: 112 bpm
Reverse order: ['P003', 'P002', 'P001']
Sorted heart rates: [88, 104, 112]

=== List Comprehensions ===
Fevers (100.4 °F or above): [101.2, 103.1]
Doses in grams: [0.25, 0.5, 0.125]
Calibrated heart rates: [90, 106, 114]
Tachycardic (100 bpm or above): ['P002', 'P003']
```

`sum(raw_rates)` on the raw list would raise `TypeError`, which is why the loop converts first. `"split" in dir("88")` asks the string what it can do. The comprehensions filter (fevers at or above 100.4 °F), transform (milligrams to grams, and the wrist monitor's 2 bpm calibration offset), and combine `zip()` with a condition (which patient IDs are tachycardic).

## 2.2 Compare a List Loop with Array Arithmetic

```bash
python demo2_numpy_performance.py
```

Source: [demo2_numpy_performance.py](demo2_numpy_performance.py). A wrist monitor that records one heart rate per second produces a million readings in under 12 days. This script applies 2.1's 2 bpm calibration offset to one million readings two ways: with a list comprehension and with array arithmetic. Both calibrate the same readings, so their result samples must match. The two timings, the speedup, and the time saved come from one machine and yours will differ; every other line should match exactly:

```text
NumPy Performance Comparison
========================================
Heart-rate readings: 1000000
First five (bpm): [72, 88, 104, 65, 91]
Operation: add the monitor's 2 bpm calibration offset to every reading

=== Python List Approach ===
Time: 23.64 ms
Result sample: [74, 90, 106, 67, 93]

=== NumPy Array Approach ===
Time: 0.86 ms
Result sample: [ 74  90 106  67  93]

========================================
Speedup: 27.5x faster!
Time saved: 22.78 ms

Timing is machine-dependent; vectorized arithmetic does the work in array operations.
```

The list prints with commas and the array without. The script builds the readings as `[72, 88, 104, 65, 91] * 200_000`, the list repetition from the lecture's "Why NumPy"; Python ignores the underscores in `200_000`, which group the digits for reading. The timing wrapper uses `time.perf_counter()` to read a clock before and after each calculation; subtracting gives elapsed seconds. Each approach runs five times in a `for` loop and the script reports the fastest run with `min()`, because a background task can slow any single run.

## 2.3 Data Types, Arrays, and Indexing

```bash
python demo2_numpy_arrays.py
```

Source: [demo2_numpy_arrays.py](demo2_numpy_arrays.py). This follows the rest of the block in the lecture's order: data types, creating arrays, their properties, random arrays, arithmetic, ufuncs, and indexing in one, two, and three dimensions. It starts with numeric text, as a file delivers it, and six patients' body temperatures in °F:

```text
NumPy Basics: Types, Arrays, and Indexing
==================================================

=== NumPy Data Types ===
As text:    ['98.6' '101.2' '99.5']  dtype: <U5
As floats:  [ 98.6 101.2  99.5]  dtype: float64
As ints:    [ 98 101  99]  decimals dropped, not rounded
As a list:  [98.6, 101.2, 99.5]  plain Python floats

=== Creating Arrays ===
Temperatures (°F): [ 98.6 101.2  99.5 103.1  97.9 100.8]
np.arange(6):      [0 1 2 3 4 5]
np.zeros(6):       [0. 0. 0. 0. 0. 0.]

=== Array Properties ===
shape: (6,)
ndim:  1
size:  6
dtype: float64
```

`<U5` is text of up to five characters until `astype(float)` converts it, and `tolist()` turns the array back into a Python list, commas and all. `np.zeros` prints `0.` with a trailing dot because it makes floats.

Next, a seeded generator simulates a week of heart rates, and arithmetic runs on whole arrays:

```text
=== Random Arrays ===
week shape: (2, 7, 3), ndim: 3, size: 42
Patient 0, days 1-3 (one row per day, three readings each):
[[63 91 86]
 [77 77 95]
 [63 88 68]]

=== Vectorized Arithmetic ===
Above 98.6 °F:      [ 0.   2.6  0.9  4.5 -0.7  2.2]
Evening (°F):       [ 99.1 100.4  99.  102.   98.2 101.5]
Evening - morning:  [ 0.5 -0.8 -0.5 -1.1  0.3  0.7]
```

`rng.integers(60, 101, size=(2, 7, 3))` holds 2 patients × 7 days × 3 readings in bpm, and the seed `42` makes it print the same numbers on every machine. `temps_f - 98.6` applies one number to every temperature; `evening_f - temps_f` pairs the two arrays position by position, so each patient's evening reading is compared with that patient's morning one.

Two ufuncs follow. `np.maximum(temps_f, evening_f)` pairs the arrays the same way and keeps each patient's higher temperature of the day. `np.sqrt()` finishes the Mosteller formula for body surface area, which drug dosing uses: the square root of height in cm × weight in kg / 3600, for three patients at once:

```text
=== Universal Functions ===
Higher of the two (°F): [ 99.1 101.2  99.5 103.1  98.2 101.5]
Height (cm):            [170 158 182]
Weight (kg):            [72 55 90]
Body surface area (m²): [1.84390889 1.55366949 2.1330729 ]
```

NumPy prints floats to eight decimal places and drops trailing zeros, so `2.13307290` shows as `2.1330729` with a space in place of the zero.

The rest of the run selects parts of the temperatures, of a 3×3 table of systolic blood-pressure readings in mmHg, and of the simulated week:

```text
=== Indexing and Slicing: 1D ===
temps_f (°F):  [ 98.6 101.2  99.5 103.1  97.9 100.8]
temps_f[0]:    98.6
temps_f[-1]:   100.8
temps_f[2:5]:  [ 99.5 103.1  97.9]
temps_f[::2]:  [98.6 99.5 97.9]

=== Indexing and Slicing: 2D ===
bp shape: (3, 3), dtype: int64
[[128 131 126]
 [142 145 139]
 [118 121 119]]
bp[1, 2] (patient 1, visit 3): 139
bp[1]    (every visit for patient 1): [142 145 139]
bp[:, 0] (visit 1 for every patient): [128 142 118]
bp[:2, 1:] (patients 0-1, visits 2-3):
[[131 126]
 [145 139]]
bp - 120 (mmHg above 120), every cell at once:
[[ 8 11  6]
 [22 25 19]
 [-2  1 -1]]

=== Indexing: 3D ===
week[0, 6]    (patient 0, day 7): [94 78 80]
week[1, :, 0] (patient 1, first reading each day): [75 92 93 78 82 95 85]
week[:, :, 0].shape (every patient's first reading each day): (2, 7)
```

`temps_f[2:5]` stops before position 5, and `bp[:, 0]` reads down a column. In the 3-D array, each single-number index removes one dimension: `week[0, 6]` leaves the three readings of one day, and `week[:, :, 0]` keeps patients and days but only the first reading, so its shape is `(2, 7)`.

# Demo 3: Selecting, Reshaping, and Analyzing Arrays

Run these from `~/03-demo` with the `(03-demo)` environment active. In a new terminal, start with these two lines, after which the prompt starts with `(03-demo)`:

```bash
cd ~/03-demo
source .venv/bin/activate
```

```bash
python demo3_bp_analysis.py
```

Source: [demo3_bp_analysis.py](demo3_bp_analysis.py). The generator is seeded with `42`, so every number below is what you should see. It creates a `(100, 5)` array of diastolic blood-pressure readings in mmHg: 100 patients, five visits each. The blocks in 3.1 to 3.3 are the whole run, in order, following the lecture's two topics.

```text
Blood Pressure Analysis with NumPy
==================================================

Created data: 100 patients, 5 visits
Array shape: (100, 5)
Data type: int64

=== Basic Operations ===
First patient's readings: [72 93 90 83 83]
In kPa (x0.133): [ 9.576 12.369 11.97  11.039 11.039]
Calibrated (+3 mmHg): [75 96 93 86 86]
```

Both arithmetic lines reach all five readings at once: multiplying converts mmHg to kilopascals, the SI pressure unit, and adding applies the correction for a cuff that reads 3 mmHg low. Multiplying by a decimal turns the whole row into floats, which is why that line prints `9.576` where the other two print whole numbers.

## 3.1 Aliases, Views, and Copies

Basic Operations applied the cuff correction with `readings[0] + 3`. A first draft of it as a function uses `+=`:

```python
def calibrate_in_place(values):
    values += 3
```

`values` is another name for the caller's array, as in the lecture's "Functions Share the Caller's Array" snippet. The script therefore calls it on a copy of patient 0's readings, so the rest of the run still works from the original data, and prints that copy before and after. The fix returns a new array and leaves its input alone:

```python
def calibrate(values):
    return values + 3
```

```text
=== Functions and the Caller's Array ===
calibrate_in_place(row) on a copy of patient 0's readings:
  row before: [72 93 90 83 83]
  row after:  [75 96 93 86 86]
calibrated = calibrate(row) on a fresh copy:
  row after:  [72 93 90 83 83]
  calibrated: [75 96 93 86 86]
```

The first draft returned nothing, yet `row` changed. With the fix, `row` keeps the measured values and the corrected ones arrive in `calibrated`.

A second name and a slice share data the same way. The script copies a 2×3 block out of the readings and names it twice, `same = practice`. It then writes one value through `same`, one through the slice `view = practice[0, :]`, and one into `independent`, a `.copy()` of that row:

```text
=== Aliases, Views, and Copies ===
Practice block (2 patients, 3 visits):
[[72 93 90]
 [96 72 91]]
After same[1, 0] = 0, practice row 1: [ 0 72 91]
same is practice: True
After view[0] = 0, practice row 0: [ 0 93 90]
After independent[1] = 0, the copy: [72  0 90]
View shares memory with practice: True
Copy shares memory with practice: False
Original readings row 0, untouched: [72 93 90 83 83]
```

`same is practice` is `True` because both names point to one array, so the write through `same` reached `practice`. The write through the view reached it too; the write into the copy did not. That is the difference to remember when you name or slice an array you still need unchanged.

## 3.2 Masks, Positions, and Shapes

A comparison on the whole `(100, 5)` array gives one `True` or `False` per reading. `readings[mask]` keeps the matching readings as a 1-D array, and `.sum()` counts them. Assigning through a mask, `capped[capped > 95] = 95`, changes the array it indexes, so the script caps a `.copy()` of the readings and leaves the raw ones alone:

```text
=== Boolean Indexing ===
Readings of 90 mmHg or above: 180 of 500
First five of them: [ 93  90  96  91 100]
Readings from 80 to 89 mmHg: 155
Capped at 95 mmHg, patient 2: [86 95 92 93 92]
Raw readings, patient 2:      [ 86 100  92  93  92]
```

The 80 to 89 count combines two comparisons with `&`, each in its own parentheses. Patient 2's 100 became 95 in `capped` only.

A mask with one `True` or `False` per patient keeps whole rows, so the result stays 2-D. It can come from one column, `readings[:, 0] >= 98`, or from all five visits at once: `.any(axis=1)` collapses each patient's row into one answer, so `(readings >= 100).any(axis=1)` marks the patients with any visit at 100 mmHg. `readings[reached_100, -1]` then keeps those rows and one column, visit 5. A mask with one value per visit keeps whole columns instead: `readings.mean(axis=0)` gives each visit's average (3.3 prints all five), and comparing it with 85 gives the column mask `high_visits`:

```text
Patients whose visit 1 was 98 mmHg or above: 14
Their first three rows:
[[100  83  97  91  94]
 [ 98  93  81  99  82]
 [ 99  83  74  95  89]]
Patients with any visit at 100 mmHg or above: 13
Their visit 5 readings: [ 92  94 100  82  94 100 100  75 100  93  85  85  85]
Visits averaging 85 mmHg or above: [ True False False False  True]
readings[:, high_visits] shape: (100, 2)
```

Five of the 13 patients who reached 100 mmHg were below 90 by visit 5. Visits 1 and 5 average 85 or above, so `readings[:, high_visits]` keeps those two columns for all 100 patients. A list of positions picks columns too, in the order given; `[0, -1]` takes the same two visits, each patient's first and last:

```text
=== Fancy Indexing ===
readings[:, [0, -1]] shape: (100, 2)
Visit 1 and visit 5, first three patients:
[[72 83]
 [96 72]
 [86 92]]
```

Reshaping puts the first 12 readings into a `(3, 4)` grid; its transpose is `(4, 3)`:

```text
=== Array Reshaping ===
Flattened sample (12 readings): [ 72  93  90  83  83  96  72  91  76  72  86 100]

Reshaped to 3x4:
[[ 72  93  90  83]
 [ 83  96  72  91]
 [ 76  72  86 100]]

Transposed (4x3):
[[ 72  83  76]
 [ 93  96  72]
 [ 90  72  86]
 [ 83  91 100]]
```

Reshaping fills the grid row by row, and the transpose turns each of those rows into a column: row `[72 93 90 83]` becomes the first column `72, 93, 90, 83` reading down.

## 3.3 Summaries, Labels, and Rankings

Row averages summarize patients and column averages summarize visits. `axis=1` averages across a row, giving one number per patient; `axis=0` averages down a column, giving one number per visit:

```text
=== Summary Statistics ===
Overall average: 85.0 mmHg
Overall median:  85.0 mmHg
Overall std dev: 8.9 mmHg
Highest reading: 100
Lowest reading: 70
25th, 50th, 75th percentiles: [78. 85. 93.]

Patient averages (first 5):
[84.2 81.4 92.6 86.2 86.6]

Visit averages:
  Visit 1: 85.8
  Visit 2: 84.6
  Visit 3: 84.1
  Visit 4: 84.8
  Visit 5: 85.5
```

About a quarter of the readings fall below the 25th percentile, 78 mmHg, and about a quarter above the 75th, 93 mmHg; the 50th percentile is the median. `readings.std()` is the square root of the average squared distance from the mean. The script rebuilds it from that definition with the ufunc `np.sqrt()`:

```python
by_hand = np.sqrt(((readings - readings.mean()) ** 2).mean())
```

Read it from the inside out: `readings - readings.mean()` broadcasts the overall mean across all 500 readings, `** 2` squares each distance, `.mean()` averages the squares, and `np.sqrt()` turns the result back into mmHg.

```text
=== Standard Deviation by Hand ===
readings.std(): 8.8560 mmHg
By hand:        8.8560 mmHg
```

The two agree, and the `Overall std dev` of 8.9 above is the same number rounded to one decimal.

`np.where()` turns one comparison of the patient averages into two labels, or into substituted values, and `np.select()` gives three. Its conditions follow the usual diastolic thresholds, highest band first: 90 or above is stage 2 hypertension, 80-89 is stage 1, and below 80 is normal.

```text
=== Conditional Labels (np.where and np.select) ===
First five averages: [84.2 81.4 92.6 86.2 86.6]
First five labels:   ['monitor' 'monitor' 'refer' 'monitor' 'monitor']
Patients to refer: 9

Patient 0 readings:    [72 93 90 83 83]
Stage 2 visits only:   [ 0 93 90  0  0]

First five stages: ['stage 1' 'stage 1' 'stage 2' 'stage 1' 'stage 1']
  Stage 2 (90 mmHg or above): 9 patients
  Stage 1 (80-89): 85 patients
  Normal (below 80): 6 patients
```

The second `np.where` keeps a reading where it is 90 or above and substitutes `0` everywhere else, which picks out the visits that were in stage 2; the zeros mark positions that failed the test, not measured pressures. `np.select` checks `>= 90` before `>= 80`, so an average of 92.6 gets `stage 2`, and `default="normal"` fills every position where neither is true. The 9 stage-2 patients are the same 9 marked `refer`.

The last section ranks visits and patients:

```text
=== Sorting and Ranking ===
Lowest-average visit: #3 (avg: 84.1 mmHg)
Highest-average visit: #1 (avg: 85.8 mmHg)

Highest 4 patient averages:
  #1: Patient  63, average 93.2 mmHg
  #2: Patient   9, average 93.0 mmHg
  #3: Patient   2, average 92.6 mmHg
  #4: Patient  80, average 91.6 mmHg
Their readings, readings[top_4]:
[[ 90  92  92  99  93]
 [100  83  97  91  94]
 [ 86 100  92  93  92]
 [100  94  79 100  85]]

Each row sorted, np.sort(readings[:3], axis=1):
[[ 72  83  83  90  93]
 [ 72  72  76  91  96]
 [ 86  92  92  93 100]]

Change from visit 1 to visit 5:
  Patients whose reading rose: 49
  Average rise: 9.8 mmHg

NumPy analysis complete.
```

`argmin()` and `argmax()` give the position of the lowest and highest visit average rather than the value, which is what names visit #3 as the lowest. The ranking sorts the patient averages with `np.argsort()`, keeps the last four positions, and reverses them, so the four patients a clinic would follow up first come out in order; `readings[top_4]` then pulls their whole rows with fancy indexing. It stops at four because three patients tie for fifth at 90.6 mmHg, and any one of them could have been listed fifth. `np.sort(..., axis=1)` orders each patient's readings on its own, which loses which visit each came from. The change count compares each patient's fifth visit with their first.

## 3.4 Summarize the Bundled CSV by Clinic

`encounters.csv` is a 1,500-row synthetic data file: one clinic visit per row, with a patient ID, an age, a systolic reading in mmHg, and the clinic that saw the patient. This script reads it with Lecture 02's `open()` and `split()`, then answers the same questions with arrays. It only prints; it writes no files.

```bash
python demo3_csv_summary.py
```

Source: [demo3_csv_summary.py](demo3_csv_summary.py). This column is systolic pressure, so its bands use the systolic thresholds rather than the diastolic ones above: below 120 is normal, 120-129 is elevated, 130-139 is stage 1, and 140 or above is stage 2.

```text
Clinic Encounter Summary
==================================================
Encounters: 1500
Systolic: shape (1500,), dtype int64

=== Systolic pressure (mmHg) ===
Average: 129.6
Lowest:  89
Highest: 174

=== Blood pressure stages (boolean masks) ===
Stage 2 (140+):     341
Stage 1 (130-139):  424
Elevated (120-129): 396
Normal (below 120): 339
Stages total: 1500

=== Follow-up labels (np.where) ===
First five readings: [123 141 140 131 104]
First five labels:   ['routine' 'refer' 'refer' 'routine' 'routine']
Needs referral: 341

=== Clinics (the counting a cut | sort | uniq -c pipeline does) ===
Cardiology: 260 encounters, average 138.7 mmHg
Dermatology: 145 encounters, average 122.6 mmHg
Endocrinology: 190 encounters, average 131.5 mmHg
Nephrology: 125 encounters, average 140.3 mmHg
Neurology: 150 encounters, average 129.6 mmHg
Obstetrics: 120 encounters, average 115.3 mmHg
Oncology: 130 encounters, average 126.3 mmHg
Primary Care: 380 encounters, average 127.3 mmHg
Highest-average clinic: Nephrology (140.3 mmHg)
```

The clinic section groups the readings the way the lecture's "Select One Group by a Label" snippet does: it loops over `sorted(set(clinics))`, stores the mask `clinics == clinic` as `in_clinic`, counts that clinic's encounters with `in_clinic.sum()`, and averages its readings with `systolic[in_clinic].mean()`. Each average is also appended to a list in the same order as the sorted names, so `np.array(averages).argmax()` is the position of the highest average and the name at that position is its clinic, the way `ids[avg_glucose.argmax()]` names a patient in the lecture. The clinic averages differ the way a real case mix does: nephrology and cardiology see more uncontrolled hypertension than obstetrics or dermatology.

The last section compares Nephrology with every other clinic. It rebuilds Nephrology's mask from its name, `in_highest = clinics == names[highest]`, because after the loop `in_clinic` holds the last clinic's mask, Primary Care's. `~` flips a mask, so `others = ~in_highest` is `True` for every encounter outside Nephrology; `clinics != names[highest]` gives the same mask.

```text
=== Nephrology compared with every other clinic ===
Every other clinic: 1375 encounters, average 128.6 mmHg
Difference: 11.6 mmHg
Stage 2 readings outside Nephrology: 269
```

`systolic[others].mean()` pools all 1,375 of those encounters, which is not the same as averaging the seven other clinic averages, because the clinics differ in size. `stage_2 & others` keeps the stage 2 readings outside Nephrology, 269 of the 341.

### Check the counts against the shell

The encounter counts match the ones your `count_clinics.sh` saved in Demo 1.7. Straight from the shell, the same file gives them again; on macOS the counts sit at a different indent, as in Demo 1:

```bash
head -n 3 encounters.csv
cut -d',' -f4 encounters.csv | tail -n +2 | sort | uniq -c
```

```text
patient_id,age,systolic_bp,clinic
P0001,85,123,Primary Care
P0002,25,141,Neurology
```

```text
    260 Cardiology
    145 Dermatology
    190 Endocrinology
    125 Nephrology
    150 Neurology
    120 Obstetrics
    130 Oncology
    380 Primary Care
```

Two tools, one answer: the pipeline counts lines of text, the script counts array positions. The [bonus page](../BONUS.md) shows shell tools such as `awk` and `sparklines`, which are beyond this lecture; `uv add sparklines` adds the second to the demo project if you want to try it.
