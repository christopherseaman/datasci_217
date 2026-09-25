---
notion:
  title_line: "# Lecture 03 Demo Guide: Environments and NumPy"
  role: demo
  status: mapped
  page_id: "3a4d9fdd-1a1a-812d-9372-edd7cbeb8303"
  url: "https://app.notion.com/p/3a4d9fdd1a1a812d9372edd7cbeb8303"
---

# Lecture 03 Demo Guide: Environments and NumPy

Clone the course repository the way Lecture 01 cloned your fork (Command Palette → **Git: Clone**, paste `https://github.com/christopherseaman/datasci_217.git`, pick a folder), and the files are in its `03/demo` folder. Without cloning, open the [Lecture 03 demo folder on GitHub](https://github.com/christopherseaman/datasci_217/tree/main/03/demo), use **Download raw file** for `requirements.txt`, `encounters.csv`, and the six demo scripts, and save them together in one folder. Open that folder in VS Code and select **Terminal → New Terminal**. Run the three demos below in order. Every file name starts with the demo that runs it: `demo1_` for the shell pipeline, `demo2_` for Python collections and array basics, `demo3_` for the analysis workflow.

# Setup: Create the tested environment

Lecture 03 uses CPython 3.13 and NumPy 2.3.3. From the demo folder, with `uv`:

```bash
uv python pin 3.13                                      # Pinned `.python-version` to `3.13`
uv venv --seed --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt                      # + numpy==2.3.3
python --version                                        # Python 3.13.x
python -c "import numpy as np; print(np.__version__)"   # 2.3.3
python -c "import sys; print(sys.executable)"           # a path ending in .venv/bin/python
```

The last line prints the interpreter that `python` now runs, and it should sit inside this folder's `.venv`, such as `/Users/alice/datasci_217/03/demo/.venv/bin/python`. A path without `.venv` in it means the environment is not active in this terminal. Use the course's Bash/Zsh terminal; in native PowerShell, activation is `.\.venv\Scripts\Activate.ps1` and the path ends in `.venv\Scripts\python.exe`. The lecture also shows standard-library `venv` as an alternative setup.

## Recreate the environment from its records

`.python-version` and `requirements.txt` are the records another person needs; `.venv/` is not shared, so rebuild it from those records in a throwaway folder and confirm the same NumPy arrives. These are the lecture's "Recreate from the Records" steps:

```bash
mkdir -p scratch/recreation-check
cp .python-version requirements.txt scratch/recreation-check/
cd scratch/recreation-check
uv venv --seed .venv                                    # Using CPython 3.13.x
source .venv/bin/activate
uv pip install -r requirements.txt                      # + numpy==2.3.3
python -c "import numpy as np; print(np.__version__)"   # 2.3.3
deactivate
cd ../..
source .venv/bin/activate                               # back in the demo environment
```

`uv venv` reads the copied `.python-version`, so it builds a Python 3.13 environment without `--python 3.13`. The last line matters: activating one environment inside another saves the PATH from before _both_, so a single `deactivate` drops you out of the demo environment too. Re-activating brings the `(.venv)` prompt back, and the rest of this guide needs it.

Assignment 03 records its own environment the same way, with a `.python-version` and a `requirements.txt`.

# 1. Shell Pipeline

Run this from a disposable project-local directory because it creates `data/`, `logs/`, and `results/` below the current directory:

```bash
mkdir -p scratch/lecture03-cli
cd scratch/lecture03-cli
bash ../../demo1_cli_pipeline.sh
```

Source: [demo1_cli_pipeline.sh](demo1_cli_pipeline.sh). The script writes a six-record CSV of clinic encounters, counts the records with `wc -l`, and counts encounters per clinic with a bounded `tail | cut | sort | uniq | head` pipeline. Your timestamp will differ from the one below:

```text
=== Lecture 03: bounded CLI pipeline ===
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
cd ../..
```

The summary repeats the counts under the timestamp that named the file:

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

One captured timestamp names the result file and labels both log lines. Run the script again a second or more later and you get a second summary file (so `cat results/summary_*.txt` then prints both) and two more log lines, which is how a timestamped run keeps every result instead of overwriting it. The longer shell-processing examples are optional reference material in the [bonus page](../BONUS.md).

# 2. Python Collections and NumPy Arrays

Run these three from the demo folder.

## 2.1 Inspect Values, Pair Sequences, Build Lists

```bash
python demo2_python_collections.py
```

Source: [demo2_python_collections.py](demo2_python_collections.py). A heart rate exported as text arrives as `"88"`, not `88`. Check the type before and after conversion, pair patient IDs with their heart rates, then build four lists in one line each:

```text
Python Tools for Collections
==================================================

=== Introspection ===
Original value: 88 Type: <class 'str'>
Converted value: 88 Type: <class 'int'>
Strings have split: True

=== Sequence functions ===
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

=== List comprehensions ===
Fevers (100.4 or above): [101.2, 103.1]
Doses in grams: [0.25, 0.5, 0.125]
Calibrated heart rates: [90, 106, 114]
Tachycardic (100 bpm or above): ['P002', 'P003']
```

`"split" in dir("88")` asks the string what it can do; the comprehensions filter (fevers at or above 100.4 °F), transform (milligrams to grams, and the wrist monitor's 2 bpm calibration offset), and combine `zip()` with a condition (which patient IDs are tachycardic).

## 2.2 Compare a List Loop with Array Arithmetic

```bash
python demo2_numpy_performance.py
```

Source: [demo2_numpy_performance.py](demo2_numpy_performance.py). Both approaches double the same million values, so their result samples must match. The two timings, the speedup, and the time saved come from one machine and yours will differ; every other line should match exactly:

```text
NumPy Performance Comparison
========================================
Operation: Multiply 1 million numbers by 2

=== Python List Approach ===
Time: 42.43 ms
Result sample: [0, 2, 4, 6, 8]

=== NumPy Array Approach ===
Time: 1.35 ms
Result sample: [0 2 4 6 8]

========================================
Speedup: 31.4x faster!
Time saved: 41.08 ms

Timing is machine-dependent; vectorized arithmetic does the work in array operations.
```

The list prints with commas and the array without. The script writes the count as `1_000_000`; Python ignores the underscores, so that is the number `1000000` with its digits grouped for reading. The timing wrapper uses `time.perf_counter()` to read a clock before and after each calculation; subtracting gives elapsed seconds.

## 2.3 Create Arrays, Check Properties, Select Parts

```bash
python demo2_numpy_arrays.py
```

Source: [demo2_numpy_arrays.py](demo2_numpy_arrays.py). This is the array half of the block: creation functions, the four properties, `astype()` on numeric text, and selection in one and two dimensions.

```text
NumPy Arrays: Creation, Properties, and Selection
==================================================

=== Creating Arrays ===
From a list:   [ 98.6 101.2  99.5 103.1  97.9 100.8]
np.arange(6):  [0 1 2 3 4 5]
np.zeros(6):   [0. 0. 0. 0. 0. 0.]

=== Array Properties ===
shape: (6,)
ndim:  1
size:  6
dtype: float64

=== Data Types ===
As text:    ['98.6' '101.2' '99.5']  dtype: <U5
As floats:  [ 98.6 101.2  99.5]  dtype: float64
As ints:    [ 98 101  99]  decimals dropped, not rounded
```

`np.zeros` prints `0.` with a trailing dot because it makes floats, and `<U5` is text of up to five characters until `astype(float)` converts it.

The rest of the run selects parts of the six readings and of a 3×3 table of blood-pressure readings:

```text
=== Indexing and Slicing: 1D ===
readings:       [ 98.6 101.2  99.5 103.1  97.9 100.8]
readings[0]:    98.6
readings[-1]:   100.8
readings[2:5]:  [ 99.5 103.1  97.9]
readings[::2]:  [98.6 99.5 97.9]

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

=== Vectorized Arithmetic ===
bp - 120 (mmHg above 120), every cell at once:
[[ 8 11  6]
 [22 25 19]
 [-2  1 -1]]
```

`readings[2:5]` stops before position 5, and `bp[:, 0]` reads down a column. Subtracting 120 broadcasts one number across all nine cells without a loop.

# 3. NumPy Blood-Pressure Analysis

```bash
python demo3_bp_analysis.py
```

Source: [demo3_bp_analysis.py](demo3_bp_analysis.py). The generator is seeded with `42`, so every number below is what you should see. It creates a `(100, 5)` array of diastolic blood-pressure readings in mmHg: 100 patients, five visits each. The blocks in this section are the whole run, in order.

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

## 3.1 Views, Copies, and Functions

The script copies a 2×3 block out of the readings, then writes one value through a slice and one value into a `.copy()`:

```text
=== Views vs Copies ===
Practice block (2 patients, 3 visits):
[[72 93 90]
 [96 72 91]]
After view[0] = 0, practice row 0: [ 0 93 90]
After independent[1] = 0, the copy: [72  0 90]
View shares memory with practice: True
Copy shares memory with practice: False
Original readings row 0, untouched: [72 93 90 83 83]
```

The write through the view reached `practice`; the write into the copy did not. That is the difference to remember when you slice an array you still need unchanged.

A function can change an array the same way. Basic Operations applied the cuff correction with `readings[0] + 3`; a first draft of it as a function uses `+=`:

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

## 3.2 Statistics, Masks, and Labels

Row averages summarize patients and column averages summarize visits. `axis=1` averages across a row, giving one number per patient; `axis=0` averages down a column, giving one number per visit:

```text
=== Statistical Operations ===
Overall average: 85.0 mmHg
Overall std dev: 8.9 mmHg
Highest reading: 100
Lowest reading: 70

Patient averages (first 5):
[84.2 81.4 92.6 86.2 86.6]

Visit averages:
  Visit 1: 85.8
  Visit 2: 84.6
  Visit 3: 84.1
  Visit 4: 84.8
  Visit 5: 85.5
```

`readings.std()` is the square root of the average squared distance from the mean. The script rebuilds it from that definition:

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

A comparison such as `patient_averages > 90` gives one True or False per patient, and `.sum()` counts the Trues. The three stage counts split the same 100 patients at the usual diastolic thresholds: below 80 is normal, 80-89 is stage 1 hypertension, and 90 or above is stage 2.

```text
=== Boolean Indexing ===
Patients averaging above 90 mmHg: 9
First five of their averages: [92.6 93.  90.6 90.4 90.6]

Diastolic stages:
  Stage 2 (90+): 9 patients
  Stage 1 (80-89): 85 patients
  Normal (below 80): 6 patients
```

`np.where()` then turns the same comparison into labels, and into substituted values:

```text
=== Conditional Labels (np.where) ===
First five averages: [84.2 81.4 92.6 86.2 86.6]
First five labels:   ['monitor' 'monitor' 'refer' 'monitor' 'monitor']
Patients to refer: 9

Patient 0 readings:    [72 93 90 83 83]
Stage 2 visits only:   [ 0 93 90  0  0]
```

The second call keeps a reading where it is 90 or above and substitutes `0` everywhere else, which picks out the visits that were in stage 2. The zeros mark positions that failed the test; they are not measured pressures.

## 3.3 Reshaping and Ranking

The last two sections follow the first 12 readings into a `(3, 4)` grid and its `(4, 3)` transpose, then use sorted indices to rank patients:

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

```text
=== Practical Analysis Workflow ===
Lowest-average visit: #3 (avg: 84.1 mmHg)
Highest-average visit: #1 (avg: 85.8 mmHg)

Highest 4 patient averages:
  #1: Patient  63, average 93.2 mmHg
  #2: Patient   9, average 93.0 mmHg
  #3: Patient   2, average 92.6 mmHg
  #4: Patient  80, average 91.6 mmHg

Change from visit 1 to visit 5:
  Patients whose reading rose: 49
  Average rise: 9.8 mmHg

NumPy analysis complete.
```

`argmin()` and `argmax()` give the position of the lowest and highest visit average rather than the value, which is what names visit #3 as the lowest. The ranking sorts the patient averages, keeps the last four positions, and reverses them, so the four patients a clinic would follow up first come out in order. It stops at four because three patients tie for fifth at 90.6 mmHg, and any one of them could have been listed fifth. The change count compares each patient's fifth visit with their first.

## 3.4 Summarize the Bundled CSV by Clinic

`encounters.csv` is a 1,500-row synthetic fixture that ships with the demo folder: one clinic visit per row, with a patient ID, an age, a systolic reading in mmHg, and the clinic that saw the patient. This script reads it with Lecture 02's `open()` and `split()`, then answers the same questions with arrays. It only prints; it writes no files.

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

### Check the counts against the shell

The same file through Lecture 03's pipeline should agree with the script. On macOS the counts sit at a different indent, as in Demo 1:

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

Two tools, one answer: the pipeline counts lines of text, the script counts array positions. The [bonus page](../BONUS.md) shows shell tools such as `awk` and `sparklines`, which are beyond this lecture and are not installed by `requirements.txt`.
