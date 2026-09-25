# Assignment 05: Documented Cleaning Pipeline (Midterm)

## Files

```text
assignment/
├── assignment.ipynb           # notebook scaffold: complete every TODO, then run it top to bottom
├── data/people_raw.csv        # supplied raw export; never edit it
├── data/fixture.json          # supplied description of the raw file, including its sha256 checksum
├── requirements.txt           # supplied: numpy, pandas, and ipykernel
├── .python-version            # supplied: Python 3.13
├── .gitattributes, .gitignore # supplied: keep the data file byte for byte, keep .venv/ out of Git
└── output/
    ├── raw_preview.txt        # you make in Task 1.1, in the terminal
    ├── pipeline_summary.txt   # you make in Task 1.2
    ├── numpy_age_summary.csv  # you make in Task 1.3
    ├── pandas_selection.csv   # you make in Task 1.4
    ├── issue_audit.csv        # you make in Task 2.2
    ├── cleaned_people.csv     # you make in Task 4.2
    └── decision_log.csv       # you make in Task 4.2
```

## The data

`data/people_raw.csv` is a synthetic intake export of person records from three clinic sites; no real people are in it. One row is one submitted person record, and `record_id` should identify it.

| Column | Holds |
| --- | --- |
| `record_id` | The record's identifier, such as `R001` |
| `full_name` | The person's name; optional, so it may be empty |
| `site` | The clinic site that submitted the record: `north`, `south`, or `west` |
| `status` | The person's enrollment status: `active`, `pending`, or `complete`; `NA` means not recorded |
| `age_text` | Age in years as typed; `unknown` and `-9` mean not recorded |
| `visit_date` | Clinic visit date as typed, meant to be `YYYY-MM-DD` |

The values arrive as someone typed them, so expect stray spaces, mixed letter case, and entries that break these rules. Leave the file exactly as it ships: the notebook's first code cell compares its checksum with `data/fixture.json` and stops if the file has changed.

## Setup

Fork the assignment repository on GitHub and clone your fork the way Lecture 01 did: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open it. Then open **Terminal → New Terminal** in VS Code at the assignment directory (Ctrl+Shift+backtick, also Control on Mac). If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first. Run `ls data` and expect `fixture.json  people_raw.csv`. This clone is a new repository, so before your first commit run Lecture 02's two `git config user.name "..."` and `git config user.email "..."` lines in this terminal, with your name and GitHub noreply email.

> **Windows:** work in the **WSL: Ubuntu** window from Lecture 01's setup. Task 1.1 uses `echo`, `head`, and `tail`, which native PowerShell does not have. Git Bash also provides them; there the environment activates with `source .venv/Scripts/activate` instead.

Create the project environment, activate it, and install the supplied requirements, as in Lecture 03:

```bash
uv venv --seed --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` includes **ipykernel**, the package that lets a notebook run on this environment's Python (Lecture 04), so this one install is all the notebook needs. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`. If VS Code offers to install the **Jupyter** extension, accept. Run the first code cell. It prints the Python, NumPy, and pandas versions and `Verified data/people_raw.csv`. If it stops with a message about `data/people_raw.csv` instead, discard your changes to that file in Source Control and run the cell again.

Work in the notebook from top to bottom. Its "Cumulative midterm checkpoint" cell, just below the first code cell, asks for an evidence map that human review reads; fill it in whenever you like. The first code cell supplies `DATA_PATH`, `OUTPUT_DIR`, a path for each output file, and `manifest`, the contents of `data/fixture.json`. The function names in the scaffold are suggestions: grading reads the files in `output/` and the notebook cells that the Completion contract names, not how your code is organized.

## Task 1: Look at the raw data

Four quick looks at the raw file, each with a tool from Lectures 01 to 04. Nothing is cleaned yet.

### 1.1 Preview the file from the terminal

In the terminal, from the assignment directory, save `output/raw_preview.txt`. It holds a label line naming each command, followed by exactly what that command prints:

```text
$ head -n 4 data/people_raw.csv
<the 4 lines that head -n 4 prints>
$ tail -n 2 data/people_raw.csv
<the 2 lines that tail -n 2 prints>
```

Write the label lines with `echo` and add each command's output with `>>` (Lecture 01's "Writing and Viewing Files" card and Lecture 03's "Pipeline Building Blocks" card). Put a label in single quotes so the shell prints the `$` as written, as in `echo '$ head -n 4 data/people_raw.csv' > output/raw_preview.txt`. Then paste the commands you ran into the notebook's Task 1.1 cell.

> **Checkpoint: `output/raw_preview.txt`**
> 8 lines: the `head` label, the four lines `head -n 4` prints (the header and the first three records), the `tail` label, and the two lines `tail -n 2` prints. The records keep their stray spaces exactly as the file has them.

### 1.2 Count rows with a Python function

The notebook's Task 1 cell loads the raw file into `raw`, with every column as text. Write a small function that takes a table and returns the five counts below, and save them to `output/pipeline_summary.txt` as one `key=value` line each, in this order, with `open(path, "w")` and f-strings (Lecture 02):

```text
raw_rows=<number>
raw_columns=<number>
exact_duplicate_rows=<number>
candidate_id_duplicate_rows=<number>
clean_rows=<number>
```

| Key | What it counts in `raw` |
| --- | --- |
| `raw_rows` | Data rows; the header is not a row |
| `raw_columns` | Columns |
| `exact_duplicate_rows` | Rows identical in every column to an earlier row: `raw.duplicated().sum()` |
| `candidate_id_duplicate_rows` | Rows whose `record_id` repeats an earlier row's: `raw.duplicated(subset=["record_id"]).sum()` |
| `clean_rows` | Rows left after removing exact duplicates: `len(raw.drop_duplicates())` |

> **Checkpoint: `output/pipeline_summary.txt`**
> 5 lines, the keys in the order above, each followed by `=` and a whole number.

### 1.3 Summarize the valid ages with NumPy

A **valid age** is an `age_text` value that is a whole number from 0 through 120. `unknown`, `-9`, words, fractions such as `40.5`, and anything above 120 are not valid ages. Collect the valid ages in a NumPy array, with either a loop that tries `int()` inside `try`/`except` (Lecture 02) or `pd.to_numeric(..., errors="coerce")` and masks (Lecture 05), then `np.array(...)`. Save the array's size, minimum, maximum, sum, and mean (Lecture 03's array properties and summaries) to `output/numpy_age_summary.csv`:

```text
metric,value
count,<number>
min,<number>
max,<number>
sum,<number>
mean,<number>
```

A small DataFrame saved with `to_csv(path, index=False)` (Lecture 04) or lines written with f-strings both work.

> **Checkpoint: `output/numpy_age_summary.csv`**
> 6 lines: the header `metric,value` and one row for each metric, in the order above.

### 1.4 Select three records with pandas

From `raw`, select the rows for `R001`, `R003`, and `R010` and the columns `record_id`, `site`, and `status`. Build a mask that is `True` for those three `record_id` values (comparisons joined with `|` as in Lecture 03, or Lecture 05's `isin`) and select with `raw.loc[mask, [...]]` (Lecture 04). Keep the values exactly as the raw file has them, and save with `to_csv(path, index=False)`.

> **Checkpoint: `output/pandas_selection.csv`**
> 4 lines: the header `record_id,site,status` and the three records in the order `R001`, `R003`, `R010`.

## Task 2: Define the contract and audit

### 2.1 Write the data contract

In the notebook's Task 2.1 Markdown cell, state the row meaning and the candidate identifier, say how the raw table differs from the cleaned table, and define **schema**, **sentinel**, **duplicate**, **missing value**, **validation invariant**, and **provenance** in your own words (Lecture 05's "What Clean Means" and "Data Cleaning Pipeline"). Human review reads this cell.

### 2.2 Audit the raw table

Count each issue below in `raw`: every row, before any duplicate is removed. Save the counts to `output/issue_audit.csv` with these labels, in this order:

```text
issue,count
schema mismatch,<count>
empty full-name tokens,<count>
empty date tokens,<count>
age sentinel tokens,<count>
status sentinel tokens,<count>
age parse failures,<count>
numeric but noninteger age values,<count>
age values outside 0 through 120,<count>
date parse failures,<count>
rows in exact duplicate sets,<count>
rows with repeated candidate IDs,<count>
site values needing format normalization,<count>
status values needing format normalization,<count>
unexpected site values,<count>
unexpected non-sentinel status values,<count>
```

Compare values after `.str.strip()` unless the definition says otherwise.

| Issue | What it counts |
| --- | --- |
| `schema mismatch` | Columns of the six in "The data" that `raw` lacks, plus any columns `raw` has beyond them |
| `empty full-name tokens` | Rows whose `full_name` is empty |
| `empty date tokens` | Rows whose `visit_date` is empty |
| `age sentinel tokens` | Rows whose `age_text` is `unknown` or `-9` |
| `status sentinel tokens` | Rows whose `status` is `NA` |
| `age parse failures` | Rows whose `age_text` is not empty, not a sentinel, and not a number: `pd.to_numeric(..., errors="coerce")` makes it missing |
| `numeric but noninteger age values` | Rows whose `age_text` is a number with a fractional part |
| `age values outside 0 through 120` | Rows whose `age_text` is a whole number, not a sentinel, below 0 or above 120 |
| `date parse failures` | Rows whose `visit_date` is not empty but is not exact `YYYY-MM-DD` text for a date on the calendar (Lecture 05's "Accept only exact dates") |
| `rows in exact duplicate sets` | Every row in a set of identical rows, first copy included: `duplicated(keep=False)` |
| `rows with repeated candidate IDs` | Every row whose `record_id` also appears on another row: `duplicated(subset=["record_id"], keep=False)` |
| `site values needing format normalization` | Rows whose raw `site` changes under `.str.strip().str.lower()` |
| `status values needing format normalization` | Rows whose raw `status` is not the `NA` sentinel and changes under `.str.strip().str.lower()` |
| `unexpected site values` | Rows whose `site`, stripped and lowercased, is not `north`, `south`, or `west` |
| `unexpected non-sentinel status values` | Rows whose `status` is not `NA` and, stripped and lowercased, is not `active`, `pending`, or `complete` |

A count of 0 is a finding too; keep its row. Auditing must not change `raw`: the notebook's `raw.equals(raw_snapshot)` stays `True`.

> **Checkpoint: `output/issue_audit.csv`**
> 16 lines: the header `issue,count` and the 15 issues in the order above, each with a whole-number count.

## Task 3: Decide and clean

### 3.1 Record the decisions

In the notebook, build `decision_table`, a DataFrame with the columns `field`, `issue`, `action`, and `reason`: one row for each decision below, in this order, with `field`, `issue`, and `action` copied exactly. Write each `reason` yourself: one sentence, on one line, about this file.

| `field` | `issue` | `action` |
|---|---|---|
| `full_name` | `empty optional name` | `retain as missing` |
| `full_name, site, status` | `surrounding whitespace and case variants` | `strip surrounding whitespace and normalize bounded field case` |
| `status` | `NA sentinel` | `convert the documented sentinel to missing` |
| `age_text` | `unknown and -9 sentinels` | `convert the documented sentinels to missing` |
| `age_text` | `nonnumeric, fractional, or out-of-range values` | `coerce invalid values to missing without rounding` |
| `visit_date` | `empty, lexically invalid, or calendar-invalid values` | `coerce invalid values to missing after an exact-format check` |
| `all raw columns` | `exact duplicate submissions` | `keep the first exact raw row only` |
| `all fields` | `adjacent-row filling` | `do not forward-fill or backward-fill` |

### 3.2 Explain what you will not do

In the notebook's Task 3.2 Markdown cell, explain why forward or backward filling is wrong across these person records, why the values you flag stay missing for review, and why "clean" here means meeting the contract rather than being perfect. Human review reads this cell.

### 3.3 Clean a copy

Build `cleaned` from `raw.copy(deep=True)` with these rules, and never change `raw`:

- Remove exact duplicate rows, keeping the first, and decide that from the untouched raw rows. Keep every other record, however many problems it has. Then renumber the rows 0, 1, 2, ... with `reset_index(drop=True)` (Lecture 05's "Validation checks" card), so the table matches the file you read back in Task 4.3.
- `full_name`: strip surrounding spaces and title-case; an empty name becomes missing.
- `site`: strip and lowercase.
- `status`: strip and lowercase; the `NA` sentinel becomes missing.
- `age`, a new column from `age_text`: keep whole numbers from 0 through 120. Sentinels, words, fractions, and out-of-range values become missing; never round.
- `visit_date`: keep a value only when its text is exact `YYYY-MM-DD` and the date exists on the calendar; anything else becomes missing.
- `needs_review`: `True` when `age` or `visit_date` is missing, otherwise `False`.
- Never fill a missing value in the table: no `fillna`, `ffill`, or `bfill` on its columns.

Keep exactly these columns, in this order, with these dtypes (Lecture 05's "Data Type Conversion"):

| Column | dtype |
|---|---|
| `record_id` | `string` |
| `full_name` | `string` |
| `site` | `string` |
| `status` | `string` |
| `age` | `Int64` |
| `visit_date` | `datetime64[us]` |
| `needs_review` | `boolean` |

## Task 4: Validate, save, and read back

### 4.1 Validate before saving

In the Task 4.1 cell, collect named `True`/`False` checks in a Series, as Lecture 05's "Stop before saving a bad table" does, and print it. Check at least that:

- every `record_id` is present and unique;
- the row count equals the raw row count minus the exact duplicates;
- every `site` is `north`, `south`, or `west`;
- every `status` that is present is `active`, `pending`, or `complete`;
- every `age` that is present is from 0 through 120;
- `needs_review` is `True` exactly where `age` or `visit_date` is missing;
- `raw.equals(raw_snapshot)` is still `True`.

Then `assert checks.all(), checks[~checks]` directly before the cells that save, so a failed check means no file is written.

### 4.2 Save the cleaned table and the decision log

Save `cleaned` to `output/cleaned_people.csv` with `to_csv(path, index=False)`. pandas then writes dates as `YYYY-MM-DD`, missing values as empty fields, and `needs_review` as `True` or `False`.

Build `decision_log` from `decision_table` by adding four columns that hold the same value on every row, and save it with `index=False`:

| Column | Value |
| --- | --- |
| `source` | `data/people_raw.csv` |
| `source_sha256` | The `sha256` value in `data/fixture.json`, which the first code cell loads as `manifest["sha256"]` |
| `rows_before` | The number of rows in `raw` |
| `rows_after` | The number of rows in `cleaned` |

> **Checkpoint: `output/cleaned_people.csv`**
> 12 lines: the header `record_id,full_name,site,status,age,visit_date,needs_review` and one line per cleaned record.

> **Checkpoint: `output/decision_log.csv`**
> 9 lines: the header `field,issue,action,reason,source,source_sha256,rows_before,rows_after` and the eight decisions in the order of Task 3.1.

### 4.3 Read the files back

Read `output/cleaned_people.csv` back with its dtypes (`dtype={...}` for the text, `Int64`, and `boolean` columns and `parse_dates=["visit_date"]`) and print `round_trip.equals(cleaned)`, which should be `True` (Lecture 05's "Read the saved file back"). `equals` also compares row labels, which is why Task 3.3 renumbers the rows. Read `output/issue_audit.csv` and `output/decision_log.csv` back too, and show that each matches the table you saved.

## Check your work

Restart the kernel and **Run All**. Every cell should finish without an error, the validation cell should print only `True`, and each read-back should print `True`. Then open each file in `output/` in VS Code and check it against this list. The line count is the number beside the last line that has text. When a file ends with a newline, VS Code also numbers the empty line after it; that one does not count.

| File | Made in | First line | Lines |
| --- | --- | --- | --- |
| `output/raw_preview.txt` | Task 1.1 | `$ head -n 4 data/people_raw.csv` | 8 |
| `output/pipeline_summary.txt` | Task 1.2 | `raw_rows=<number>` | 5 |
| `output/numpy_age_summary.csv` | Task 1.3 | `metric,value` | 6 |
| `output/pandas_selection.csv` | Task 1.4 | `record_id,site,status` | 4 |
| `output/issue_audit.csv` | Task 2.2 | `issue,count` | 16 |
| `output/cleaned_people.csv` | Task 4.2 | `record_id,full_name,site,status,age,visit_date,needs_review` | 12 |
| `output/decision_log.csv` | Task 4.2 | `field,issue,action,reason,source,source_sha256,rows_before,rows_after` | 9 |

- [ ] All seven files exist in `output/`, each with the first line and line count above.
- [ ] No CSV starts with an extra column of row numbers; each was saved with `index=False`.
- [ ] `cleaned_people.csv` shows dates as `YYYY-MM-DD`, missing values as empty fields, and `needs_review` as `True` or `False`.
- [ ] Every row of `decision_log.csv` has a reason and the same `source`, `source_sha256`, `rows_before`, and `rows_after`.
- [ ] `data/people_raw.csv` is unchanged: Source Control lists no change to it.

### Completion contract

The midterm totals 100 points: 85 points graded from your committed files after the deadline, 15 by human review. Each file is graded on its own, so a wrong value costs only its own points.

| File | What earns the points | Points |
| --- | --- | ---: |
| `output/raw_preview.txt` | 2 for the `head` label with the four lines it prints, 2 for the `tail` label with the two lines it prints | 4 |
| `output/pipeline_summary.txt` | 1 for each key's count | 5 |
| `output/numpy_age_summary.csv` | 1 for each metric's value | 5 |
| `output/pandas_selection.csv` | 1 for each requested record with its raw `site` and `status`, less 1 for each other row; 1 for exactly the three columns | 4 |
| `output/issue_audit.csv` | 1 for each issue's count | 15 |
| `output/cleaned_people.csv` | 5 for each of the seven columns, in proportion to the records whose value in that column follows the cleaning rules (rounded down) | 35 |
| `output/decision_log.csv` | 1 for each decision with its `field`, `issue`, and `action`; 2 for a reason on every row; 1 for `source`, 2 for `source_sha256`, 2 for `rows_before`, and 2 for `rows_after` on every row | 17 |

How the files are read:

- Line endings, blank lines, spaces at the end of a line or around a header name, a comma, semicolon, or tab between cells, spaces that pad every separator in the header line and the rows alike, column order, a leading column of row numbers, and number format (`6`, `6.0`) do not matter, and neither do the letter case and spacing of labels: keys, metric names, issue names, decision text, and record IDs. Cleaned values must be exactly what the cleaning rules produce, such as `north` rather than `North` or ` north `.
- `True`/`False`, `true`/`false`, `1`/`0`, and `yes`/`no` all read as booleans; an empty field, `NaN`, and `<NA>` all read as missing; a date may carry a `00:00:00` time after it.
- `needs_review` is also right when it follows the rule from your own `age` and `visit_date` columns, and `rows_after` is also right when it equals the rows in your own `cleaned_people.csv`, so one mistake is not charged twice.

Human review reads the notebook:

| Category | What it reads | Full credit (5) | Partial credit |
| --- | --- | --- | --- |
| Lecture 01 to 05 evidence map | The "Cumulative midterm checkpoint" cell | One entry for each of Lectures 01 to 05, each naming a concrete technique or file from that lecture and saying in one sentence where this notebook uses it | 2 to 4 for three or four lectures, or for entries that name a tool without saying where it is used here |
| Cleaning decisions and rationale | The `reason` column of `output/decision_log.csv` and the Task 3.2 cell | Every reason names the problem in this file and why its action fits; Task 3.2 covers filling, flagged values, and what clean means here | 2 to 4 for generic reasons, such as "to clean the data", or a Task 3.2 answer missing one of its three points |
| Validation, provenance, organization, and privacy | The Task 2.1, 4.1, and 4.3 cells, the Task 1.1 commands, and the notebook as a whole | The contract defines all six terms; the checks are named and stop the save; each read-back prints `True`; the notebook runs top to bottom with its outputs saved; no credentials or personal information appear | 2 to 4 when a part is missing or unclear |

## Submit

Save the notebook after **Run All**, so its outputs are part of the commit. In VS Code Source Control, stage `assignment.ipynb` and the seven files in `output/`, commit with a message such as `Complete the midterm cleaning pipeline`, and sync. Keep `.venv/` out of the commit; `.gitignore` already lists it. The course grades the files on your fork's `main` branch after the deadline, so open your fork on GitHub and confirm that `output/` shows all seven files and that `assignment.ipynb` shows your outputs.
