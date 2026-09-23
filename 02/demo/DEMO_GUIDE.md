---
notion:
  title_line: "# Lecture 02 Demo Guide: Git, Functions, and Modules"
  role: demo
  status: mapped
  page_id: "3dbd9fdd-1a1a-813d-936e-e7011563ebde"
  url: "https://app.notion.com/p/3dbd9fdd1a1a813d936ee7011563ebde"
---

# Lecture 02 Demo Guide: Git, Functions, and Modules

Demos 2 and 3 run four files from the [Lecture 02 demo folder on GitHub](https://github.com/christopherseaman/datasci_217/tree/main/02/demo): [functions_demo.py](functions_demo.py), [vitals_tools.py](vitals_tools.py), [module_usage_demo.py](module_usage_demo.py), and [clinic_vitals.csv](clinic_vitals.csv). Clone the course repository the way Lecture 01 cloned your fork (Command Palette → **Git: Clone**, paste `https://github.com/christopherseaman/datasci_217.git`, pick a folder), and the files are in its `02/demo` folder. Without cloning, use **Download raw file** on GitHub for each of the four and save them together in one folder. Demo 1 needs neither: it builds a practice repository of its own. Run every command in **Terminal → New Terminal** (Ctrl+Shift+backtick, also Control on Mac).

# 1. Git workflow

Make a practice folder **outside** your cloned course repository, such as `ds217-practice` in your home folder, and open it with **File → Open Folder…**. It must sit outside the clone because VS Code hides **Initialize Repository** in any folder already inside a repository. Open **View → Source Control** (Ctrl+Shift+G, including Control on macOS) and select **Initialize Repository**. Or from the terminal, starting in your home folder:

```bash
cd ~
mkdir -p ds217-practice
cd ds217-practice
git init
git checkout -b main
```

Lecture 01 had you set your Git name and email, if a commit asked, only in your Assignment 01 clone, so this repository needs them too: without them, its first commit either fails with `Author identity unknown` or goes in under an address made from your computer's name. In a terminal in `ds217-practice` (**Terminal → New Terminal** opens there), type these two lines from Lecture 01 with your name and GitHub noreply email between the quotes:

```bash
git config user.name ""
git config user.email ""
```

Create `notes.md` with `# Practice notes`. In Source Control, stage it with the `+` button, enter `Start practice notes`, and select **Commit**. The changes list is empty after the commit: the working tree is clean.

Click the branch name in the status bar → **Create new branch…** → `experiment`. Add `Experiment: compare two systolic summaries.` to `notes.md`. Source Control now shows a working (unstaged) change; select the file to see its diff. Stage it: the file moves to **Staged Changes**. Enter `Add experiment note` and select **Commit**; the lists are empty again.

Click the branch name → select `main`. Open the Command Palette, select **Git: Merge...**, and choose `experiment`. The experiment change is now committed on `main`. Or the same steps from the terminal:

```bash
echo "# Practice notes" > notes.md
git add notes.md
git commit -m "Start practice notes"
git checkout -b experiment
echo "Experiment: compare two systolic summaries." >> notes.md
git status --short
git diff
git add notes.md
git status --short
git commit -m "Add experiment note"
git status --short
git checkout main
git merge experiment
git log --oneline
```

The three `git status --short` calls are the point: the same file reports a different state each time, and the space before `M` moves. `git diff` between the first two shows the line you added.

```text
 M notes.md      <- working tree: edited, not staged
M  notes.md      <- staging area: ready for the next commit
                 <- committed: nothing to report, the tree is clean
```

`git log --oneline` lists both commits, newest first (your hashes differ). Both branch names label the newest commit: `main` gained nothing meanwhile, so the merge was a fast-forward:

```text
<hash> (HEAD -> main, experiment) Add experiment note
<hash> Start practice notes
```

## A merge conflict

Now give each branch a different line 2 and merge again. Whichever path you took, paste this into the terminal; each `>` rewrites `notes.md` from its heading:

```bash
git checkout experiment
echo "# Practice notes" > notes.md
echo "Experiment: compare three systolic summaries." >> notes.md
git add notes.md
git commit -m "Compare three systolic summaries"
git checkout main
echo "# Practice notes" > notes.md
echo "Experiment: compare median systolic." >> notes.md
git add notes.md
git commit -m "Compare median systolic"
git merge experiment
git status --short
cat notes.md
```

`git merge` stops with `CONFLICT (content): Merge conflict in notes.md`, and `git status --short` prints `UU notes.md`: changed on both branches, not yet merged. `notes.md` holds both versions, marked exactly as in the lecture:

```text
# Practice notes
<<<<<<< HEAD
Experiment: compare median systolic.
=======
Experiment: compare three systolic summaries.
>>>>>>> experiment
```

In VS Code, open `notes.md` from **Merge Changes** in Source Control and select **Accept Incoming Change** above the block, since three summaries include the median. Save, stage the file with **+**, and select **Commit**; VS Code fills in the message `Merge branch 'experiment'`. Or in the terminal, write the version you want, then stage and commit it:

```bash
echo "# Practice notes" > notes.md
echo "Experiment: compare three systolic summaries." >> notes.md
git add notes.md
git commit -m "Merge branch 'experiment'"
```

Either way, run `git status --short` (it prints nothing) and `git log --oneline`, which puts the merge commit on top:

```text
<hash> (HEAD -> main) Merge branch 'experiment'
<hash> Compare median systolic
<hash> (experiment) Compare three systolic summaries
<hash> Add experiment note
<hash> Start practice notes
```

## Ignore a file

`raw_vitals.csv` stands in for a raw export of patient data, which must never be committed:

```bash
echo "patient_id,systolic" > raw_vitals.csv
git status --short
echo "*.csv" > .gitignore
git status --short
git add .gitignore
git commit -m "Ignore CSV exports"
git status --short
```

Once `.gitignore` covers the export, it leaves `git status --short` and Source Control's **Changes** list, though it stays on disk:

```text
?? raw_vitals.csv    <- ?? marks an untracked file
?? .gitignore        <- *.csv hides the export
                     <- committed: nothing to report
```

## Publish to GitHub

With `main` as the current branch, select **Publish Branch** in Source Control; sign in to GitHub if VS Code asks. Keep the name `ds217-practice` and select **Publish to GitHub private repository**. When VS Code reports `Successfully published the "<your-username>/ds217-practice" repository to GitHub.`, select **Open on GitHub**.

On GitHub, the **Code** tab lists `.gitignore` and `notes.md` but not `raw_vitals.csv`: the ignored export never left your computer. Select the commit count (**6 Commits**) for the same six commits `git log --oneline` lists, newest first: `Ignore CSV exports` on top of the five above.

## Less typing: recall and edit

1. In the `ds217-practice` terminal, type `cat no`, press **Tab** to complete `notes.md`, then **Enter**. Expect the heading and the three-summaries line.
2. Press **↑** to recall it and **Ctrl+A** to jump to the start. Press **Delete** three times to remove `cat` (on Mac, **Fn+Delete**) and type `git diff`, so the line reads `git diff notes.md`. Press **Enter**: no output, because every change to `notes.md` is committed.
3. Press **Ctrl+R**, type `cat no`, and check that `cat notes.md` appears. Press **Enter** to run it straight away, or **→** to put it on the prompt first so you can edit it. (**Esc** also accepts the match in Bash, but in Zsh, the macOS default, it leaves you in the search.)

# 2. Containers, functions, and imports

Open the course clone's `02/demo` folder with **File → Open Folder…** (or the folder where you saved the four downloads), then **Terminal → New Terminal**:

```bash
python3 functions_demo.py
```

The complete scripts are [functions_demo.py](functions_demo.py) and [vitals_tools.py](vitals_tools.py). One morning's clinic log carries the whole demo: a tuple holds the clinic and the date, a list holds one dictionary per encounter, the helpers in `vitals_tools` do the repeated work, and a set compares two groups of patient IDs. The log starts as three encounters, each a patient ID and a systolic blood pressure in mmHg, and the demo pulls out the readings twice: first with a loop you would otherwise copy into every script, then with an imported helper:

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

`get_systolic()` in `vitals_tools.py` is that loop with a name, a docstring, and a `return`:

```python
def get_systolic(encounters):
    """Return the systolic readings stored in encounter records."""
    readings = []
    for encounter in encounters:
        readings.append(encounter["systolic"])
    return readings
```

A walk-in shows why the encounters are a list and the clinic and date a tuple. Appending P004 changes `encounters`, but `readings` still holds the three numbers extracted earlier, so the script calls `get_systolic()` again before summarizing. Then `statistics` checks our own average:

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

Each encounter is a dictionary: `.items()` walks its fields, `record["systolic"]` reads one, and `.get()` answers for a field nobody filled in, such as a `follow_up` only some visits schedule, instead of raising `KeyError`:

```python
record = encounters[0]                  # each encounter is a dictionary: field name to value
for field, value in record.items():
    print(f"  {field}: {value}")
print(f"P001's systolic: {record['systolic']} mmHg")
print(f"P001's follow-up: {record.get('follow_up', 'none scheduled')}")
```

The script ends by asking which readings to flag. The default is 130 mmHg, the usual hypertension threshold; type `120` and press **Enter** to match the transcript below and flag elevated readings too.

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

The `Average with no readings` and `Average of two zero pain scores` lines are the lecture's empty-list case. `mean_reading([])` returns `None`, and the demo asks `is None`, because a pain score of 0 is a patient answering "no pain," a real measurement that `if not empty_average` would throw away along with the empty list:

```python
empty_average = mean_reading([])
if empty_average is None:               # `is None`, because `if not empty_average` also catches 0.0
    print("Average with no readings: nothing to average")
else:
    print(f"Average with no readings: {empty_average:.1f}")
print(f"Average of two zero pain scores: {mean_reading([0, 0])}")
```

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

The first command is intentionally silent: importing runs the `def` lines, but the `if __name__ == "__main__":` guard skips `main()`, so nothing is printed and no report is written. The complete script is [module_usage_demo.py](module_usage_demo.py). It reads [clinic_vitals.csv](clinic_vitals.csv), a clinic export with one reading nobody wrote down:

```text
patient_id,systolic
P001,128
P002,142
P003,not recorded
P004,136
```

`int("not recorded")` raises `ValueError`, so the parsing loop catches that one exception, reports the skipped row, and keeps going. Two other bad rows are reported before the unpacking, which would crash on them instead of naming the problem: a blank line has no comma (`ValueError: not enough values to unpack`), and `P005,134,extra` splits into three pieces (`ValueError: too many values to unpack`). Splitting into a `fields` list first lets the loop count the pieces and report either one.

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

If every row were unusable there would be nothing to average, so `assert encounters, f"no usable readings in {data_path}"` states that expectation before the script formats anything. Then, as in the lecture's write-and-read-back snippet, `Path` builds `output/vitals_report.txt`, one `with` block writes the report, and a second reads back what landed on disk. The two strings are compared twice: as a printed status, and as an `assert` that stops the script rather than let a wrong report look fine:

```python
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

Open `output/vitals_report.txt` in the Explorer: it holds only the five report lines, not the skip notice or checkpoint lines. A second run prints the same output and leaves the same file, because mode `"w"` replaces the file rather than adding to it.

## Watch the checkpoint fire

An `assert` is worth having only if you know what it looks like when it fails. Save different text than the script built: find the first `with` block in `main()` and add `.upper()` to what it writes, so the two lines read

```python
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text.upper())
```

Save, run `python3 module_usage_demo.py` again, and the report reads back in capitals. The printed comparison says `Saved report matches: False`, and the `assert` on the next line stops the script with a traceback whose last line is:

```text
AssertionError: the saved report does not match the text we built
```

Remove `.upper()` (**Ctrl+Z** undoes the edit, **Cmd+Z** on Mac), save, and run once more to get the expected output back.

## A blank row in the export

Exports often end with a blank line, and the loop reports it rather than failing on it. Open `clinic_vitals.csv`, press **Enter** at the end of the `P004,136` line to add one, save, and run `python3 module_usage_demo.py` again. The report itself is unchanged; one extra line appears before the read-back:

```text
Skipping P003: invalid literal for int() with base 10: 'not recorded'
Skipping a blank row.
Read back from output/vitals_report.txt:
```

Undo the edit (**Ctrl+Z**, **Cmd+Z** on Mac) and save.

## A row with an extra field

A hand-edited export can also hand you a row with one comma too many, and `patient_id, raw_systolic = fields` needs exactly two pieces. Open `clinic_vitals.csv`, add `P005,134,extra` on a new line after `P004,136`, save, and run `python3 module_usage_demo.py` again. P005 is reported and left out, the report itself is unchanged, and one extra line appears before the read-back:

```text
Skipping P003: invalid literal for int() with base 10: 'not recorded'
Skipping a row with 3 fields: P005,134,extra
Read back from output/vitals_report.txt:
```

Undo the edit (**Ctrl+Z**, **Cmd+Z** on Mac) and save.

## If the script cannot find the data

`Path("clinic_vitals.csv")` is a relative path, so the script looks in the folder you ran it from. `path.exists()` checks first, so running from somewhere else prints one line and stops instead of raising `FileNotFoundError`:

```text
Cannot find clinic_vitals.csv: run this script from the 02/demo folder.
```

`cd` into `02/demo` and run it again.

## Document how to run it

Create `README.md` in the same folder:

```markdown
# Clinic Vitals Report

Reads `clinic_vitals.csv` and saves each reading, the **average**, and the highest systolic to `output/vitals_report.txt`.

## Run

From this folder: `python3 module_usage_demo.py`

> Blank rows, rows with an extra field, and readings that are not numbers are _skipped_ and reported.
```

Save it and press **Ctrl+K** then **V** (**Cmd+K** then **V** on Mac). The preview opens beside it: the title in large type, `Run` as a smaller heading, **average** in bold, _skipped_ in italics, the file names and the command in code font, and the last line set off as a quote.
