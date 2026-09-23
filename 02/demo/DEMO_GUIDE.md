---
notion:
  title_line: "# Lecture 02 Demo Guide: Git, Functions, and Modules"
  role: demo
  status: mapped
  page_id: "3dbd9fdd-1a1a-813d-936e-e7011563ebde"
  url: "https://app.notion.com/p/3dbd9fdd1a1a813d936ee7011563ebde"
---

# Lecture 02 Demo Guide: Git, Functions, and Modules

Demos 2 and 3 run four files from the [Lecture 02 demo folder on GitHub](https://github.com/christopherseaman/datasci_217/tree/main/02/demo): [functions_demo.py](functions_demo.py), [vitals_tools.py](vitals_tools.py), [module_usage_demo.py](module_usage_demo.py), and [clinic_vitals.csv](clinic_vitals.csv). Get the whole course repository the way Lecture 01 cloned your fork: open the Command Palette, choose **Git: Clone**, paste `https://github.com/christopherseaman/datasci_217.git`, pick a folder, and open the clone. The demo files are in its `02/demo` folder. Demo 1 does not use the clone at all: it builds a practice repository of its own, outside it.

Without cloning, use **Download raw file** on GitHub for each of the four files and save them together in one folder; Demo 1 needs nothing downloaded. Run every command in **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac).

# 1. Git workflow

Make a practice folder **outside** your cloned course repository, such as `ds217-practice` in your home folder, and use **File → Open Folder…** to open it in VS Code. It has to sit outside the clone: VS Code hides **Initialize Repository** for any folder that is already inside a repository, so a folder made under the clone never offers the button. Open **View → Source Control** (Ctrl+Shift+G, including Control on macOS) and select **Initialize Repository**. If the initial branch is not `main`, open **View → Command Palette…** (Ctrl+Shift+P on Windows/Linux, Cmd+Shift+P on macOS), select **Git: Create Branch**, and name it `main`.

Create `notes.md` with `# Practice notes`. In Source Control, stage it with the `+` button, enter `Start practice notes`, and select the visible **Commit** button. The changes list is empty after the commit: the working tree is clean.

Click the branch name in the status bar → **Create new branch…** → `experiment`. Add `Experiment: compare two systolic summaries.` to `notes.md`. The Source Control view now shows a working (unstaged) change; select the file to inspect its diff. Stage it: the file moves to **Staged Changes**. Enter `Add experiment note` and select **Commit**; the lists are empty again.

Click the branch name → select `main`. Open the Command Palette, select **Git: Merge Branch…**, and choose `experiment`. The experiment change is now committed on `main`.

Alternatively, create the same practice repository from the terminal. Start in your home folder, again outside the clone:

```bash
cd ~
mkdir -p ds217-practice
cd ds217-practice
git init
git checkout -b main
echo "# Practice notes" > notes.md
git add notes.md
git commit -m "Start practice notes"
git checkout -b experiment
echo "Experiment: compare two systolic summaries." >> notes.md
git status                    # working: notes.md is modified, not staged
git diff                      # working: shows the new line
git add notes.md
git status                    # staged: notes.md is ready to commit
git commit -m "Add experiment note"
git status                    # committed: working tree clean
git checkout main
git merge experiment
git log --oneline             # shows both commits; press q if needed
cd -                          # back to where you started
```

## Less typing: recall and edit

Open `ds217-practice` with **File → Open Folder**, then **Terminal → New Terminal**:

1. Type `cat no`, press **Tab** to complete `notes.md`, then **Enter**. Expect the practice heading and experiment note.
2. Press **↑** to recall it, then **Ctrl+A** to move to the start. Press **Delete** three times to remove `cat` (on Mac, **Fn+Delete**), type `git diff --`, and press **Ctrl+E** (**End** on Windows/Linux, where VS Code claims Ctrl+E for Quick Open). The line should read `git diff -- notes.md`.
3. Press **Enter**. Expect no output: you already committed and merged those changes.
4. Press **Ctrl+R**, type `cat no`, and check that `cat notes.md` appears. Press **Enter** to run it straight away, or **→** to put it on the prompt first so you can edit it. (**Esc** also accepts the match in Bash, but in Zsh, the macOS default, it leaves you in the search.)

For the Python demos, open the course's `02/demo` folder in VS Code and use **Terminal → New Terminal** (Ctrl+Shift+backtick).

# 2. Containers, functions, and imports

```bash
python3 functions_demo.py
```

The complete scripts are [functions_demo.py](functions_demo.py) and [vitals_tools.py](vitals_tools.py). One morning's clinic log carries the whole demo: a tuple holds the clinic and the date of the visits, a list holds one dictionary per encounter, the helpers in `vitals_tools` do the repeated work, and a set compares two groups of patient IDs.

The log starts as three encounters, each one a patient ID and the systolic blood pressure recorded at that visit, in mmHg. The demo pulls those readings out of the log twice: first with the work you would otherwise copy into every script, then with an imported helper:

```python
encounters = [
    {"patient_id": "P001", "systolic": 128},
    {"patient_id": "P002", "systolic": 142},
    {"patient_id": "P003", "systolic": 118},
]

readings = []
for encounter in encounters:
    readings.append(encounter["systolic"])
print(f"Before loop extracted: {readings}")

readings = get_systolic(encounters)
print(f"After get_systolic() extracted: {readings}")
```

A walk-in arriving after the log was built shows why the encounters are a list and the clinic and date are a tuple. Appending P004 changes `encounters`, but `readings` still holds the three numbers extracted before that patient arrived, so the script calls `get_systolic()` again before it summarizes anything. Then `statistics` checks our own average:

```python
encounters.append({"patient_id": "P004", "systolic": 136})   # a list can grow; a tuple cannot
readings = get_systolic(encounters)     # read the log again: the old readings predate P004
ranked = sorted(readings)
print(f"P004 arrived late, so the log now holds {len(encounters)} encounters.")
print(f"Readings in order: {ranked}")
print(f"Two highest readings: {ranked[-2:]}")
print(f"Average systolic: {mean_reading(readings):.1f} mmHg")
print(f"Highest systolic: {highest_reading(readings)} mmHg")
print(f"statistics.mean agrees: {stats.mean(readings) == mean_reading(readings)}")
```

Each encounter is itself a dictionary, so `.items()` walks its fields, `record["systolic"]` reads one of them, and `.get()` answers for a field nobody filled in instead of raising `KeyError`. Only some visits schedule a follow-up, so `follow_up` is exactly that kind of field:

```python
record = encounters[0]                  # each encounter is a dictionary: field name to value
for field, value in record.items():
    print(f"  {field}: {value}")
print(f"P001's systolic: {record['systolic']} mmHg")
print(f"P001's follow-up: {record.get('follow_up', 'none scheduled')}")
```

The script ends by asking which readings to flag. A systolic reading of 130 mmHg or higher is the usual hypertension threshold, so that is the default; type `120` and press **Enter** to match the transcript below and flag the elevated readings too.

```python
typed_cutoff = input("Flag systolic at or above (press Enter for 130): ")
if not typed_cutoff:                    # an empty answer means Enter alone
    typed_cutoff = "130"
cutoff = int(typed_cutoff)
```

Expected checkpoints, with `120` typed at the prompt:

```text
=== Bayview Clinic 2026-09-18: systolic summary ===
Before: every script would repeat this loop.
Before loop extracted: [128, 142, 118]
After: reuse helpers from vitals_tools.
After get_systolic() extracted: [128, 142, 118]
P004 arrived late, so the log now holds 4 encounters.
Readings in order: [118, 128, 136, 142]
Two highest readings: [136, 142]
Average systolic: 131.0 mmHg
Highest systolic: 142 mmHg
statistics.mean agrees: True
Average with no readings: nothing to average
Average of two zero pain scores: 0.0
One encounter, field by field:
  patient_id: P001
  systolic: 128
P001's systolic: 128 mmHg
P001's follow-up: none scheduled
Flag systolic at or above (press Enter for 130): 120
Flagged (120 mmHg and above): ['P001', 'P002', 'P004']
Flagged patients in the morning session: ['P001', 'P004']
```

The two `Average with no readings` and `Average of two zero pain scores` lines are the empty-list case from the lecture. `mean_reading([])` has nothing to average, so it returns `None`, and the demo asks `is None` before it formats a number:

```python
empty_average = mean_reading([])
if empty_average is None:               # `is None`, because `if not empty_average` also catches 0.0
    print("Average with no readings: nothing to average")
else:
    print(f"Average with no readings: {empty_average:.1f}")
print(f"Average of two zero pain scores: {mean_reading([0, 0])}")
```

The last line of that snippet is why the test has to be `is None`. A pain score of 0 is a patient answering "no pain," a real measurement worth reporting, and `if not empty_average` would have thrown that answer away along with the empty list.

## Change the cutoff

Run the script again and press **Enter** alone. Only the last three lines change, because 130 leaves P001's 128 mmHg out:

```text
Flag systolic at or above (press Enter for 130): 
Flagged (130 mmHg and above): ['P002', 'P004']
Flagged patients in the morning session: ['P004']
```

A loop collects the IDs at or above the cutoff into `flagged_ids`. That list keeps the log's order, but a set has no order, so the group shared by both is wrapped in `sorted()` for a stable display:

```python
flagged_ids = []
for encounter in encounters:
    if encounter["systolic"] >= cutoff:
        flagged_ids.append(encounter["patient_id"])

morning_session = {"P001", "P003", "P004"}   # a set: distinct IDs, no order
flagged = set(flagged_ids)                   # the same IDs as a set, so & can compare groups
print(f"Flagged ({cutoff} mmHg and above): {flagged_ids}")
print(f"Flagged patients in the morning session: {sorted(flagged & morning_session)}")
```

# 3. Files, exceptions, and a checkpoint

```bash
python3 -c "import module_usage_demo"
python3 module_usage_demo.py
```

The first command is intentionally silent: importing runs the `def` lines but skips `main()`, so nothing is printed and no report is written.

```python
if __name__ == "__main__":
    main()
```

The complete script is [module_usage_demo.py](module_usage_demo.py). It reads [clinic_vitals.csv](clinic_vitals.csv), the kind of file an export from the clinic's records hands you, with one reading nobody wrote down:

```text
patient_id,systolic
P001,128
P002,142
P003,not recorded
P004,136
```

`int("not recorded")` raises `ValueError`, so the parsing loop catches that one exception, reports the row it skipped, and keeps going. Two other shapes of bad row are reported before the unpacking rather than after it, because unpacking them would crash instead of naming the problem. A blank line has no comma, so `patient_id, raw_systolic = ...` would fail with `ValueError: not enough values to unpack`; a row with an extra comma, such as `P005,134,extra`, splits into three pieces and would fail with `ValueError: too many values to unpack`. Splitting into a `fields` list first lets the loop count the pieces and report either one.

```python
for row in rows[1:]:                      # rows[0] is the header line
    if not row.strip():                   # an export often ends with a blank line
        print("Skipping a blank row.")
        continue
    fields = row.strip().split(",")
    if len(fields) != 2:                  # an extra comma leaves too many pieces to unpack
        print(f"Skipping a row with {len(fields)} fields: {row.strip()}")
        continue
    patient_id, raw_systolic = fields
    try:
        systolic = int(raw_systolic)
    except ValueError as error:
        print(f"Skipping {patient_id}: {error}")
    else:
        encounters.append({"patient_id": patient_id, "systolic": systolic})
```

If every row were unusable, there would be nothing to average, so the script states that expectation before it formats anything:

```python
assert encounters, f"no usable readings in {data_path}"
```

`Path` builds the output location, one `with` block writes the report, and a second one opens the same path again to read back what landed on disk. The two strings are then compared twice: once as a printed status, and once as an `assert` that stops the script instead of letting a wrong report look fine.

```python
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)           # no error when output/ already exists
report_path = output_dir / "vitals_report.txt"
with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write(report_text)

with open(report_path, "r", encoding="utf-8") as report_file:
    saved_text = report_file.read()

print(f"Read back from {report_path}:")
print(saved_text, end="")
print(f"Saved report matches: {saved_text == report_text}")
assert saved_text == report_text, "the saved report does not match the text we built"
```

Expected output:

```text
Skipping P003: invalid literal for int() with base 10: 'not recorded'
Read back from output/vitals_report.txt:
P001: 128 mmHg
P002: 142 mmHg
P004: 136 mmHg
Average systolic: 135.3 mmHg
Highest systolic: 142 mmHg
Saved report matches: True
Checkpoint passed: 5 lines saved to output/vitals_report.txt
Report on one line: P001: 128 mmHg | P002: 142 mmHg | P004: 136 mmHg | Average systolic: 135.3 mmHg | Highest systolic: 142 mmHg
```

Open `output/vitals_report.txt` in the Explorer: it holds the five report lines only, without the skip notice or the checkpoint lines. Run the script a second time and both the terminal output and the file are identical, because mode `"w"` replaces the file rather than adding to it.

## Watch the checkpoint fire

An `assert` is worth having only if you know what it looks like when it fails. Save different text than the script built: find the first `with` block in `main()` and add `.upper()` to what it writes, so the two lines read

```python
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text.upper())
```

Both lines are already in the file; the only edit is `.upper()`.

Save, run `python3 module_usage_demo.py` again, and the report reads back in capitals. The printed comparison answers `False`:

```text
Saved report matches: False
```

The `assert` on the next line then stops the script with a traceback whose last line is:

```text
AssertionError: the saved report does not match the text we built
```

Remove `.upper()` (**Ctrl+Z** undoes the edit), save, and run once more to get the expected output back.

## A blank row in the export

Exports often end with a blank line, and the loop reports it rather than failing on it. Open `clinic_vitals.csv`, press **Enter** at the end of the `P004,136` line to add one, save, and run `python3 module_usage_demo.py` again. The report itself is unchanged; one extra line appears before the read-back:

```text
Skipping P003: invalid literal for int() with base 10: 'not recorded'
Skipping a blank row.
Read back from output/vitals_report.txt:
```

Undo the edit (**Ctrl+Z**) and save.

## A row with an extra field

A hand-edited export can also hand you a row with one comma too many, and `patient_id, raw_systolic = fields` needs exactly two pieces. Open `clinic_vitals.csv`, add `P005,134,extra` on a new line after `P004,136`, save, and run `python3 module_usage_demo.py` again. P005 is reported and left out, the report itself is unchanged, and one extra line appears before the read-back:

```text
Skipping P003: invalid literal for int() with base 10: 'not recorded'
Skipping a row with 3 fields: P005,134,extra
Read back from output/vitals_report.txt:
```

Undo the edit (**Ctrl+Z**) and save.

## If the script cannot find the data

`Path("clinic_vitals.csv")` is a relative path, so the script looks in the folder you ran it from. `path.exists()` checks first, so running from somewhere else prints one line and stops instead of raising `FileNotFoundError`:

```text
Cannot find clinic_vitals.csv: run this script from the 02/demo folder.
```

`cd` into `02/demo` and run it again.
