# Assignment 02 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/5  README project description
         README.md still has the TODO line under `## Project description`. Replace it with 30-300 characters of your own saying what this project reads and what it produces (Task 1.1).
```

A clean run ends with:

```text
[PASS] 15/15 follow-up patient list

Score: 100/100
All checks passed.
```

How the files are read:

- Checks look only at your artifacts: `README.md`, `.gitignore`, and the two files in `output/`. They recompute the answers from their own copy of `data/clinic_encounters.csv` and never run or read your Python code.
- Each check is scored on its own, so one mistake costs only the checks it gets wrong.
- Letter case, spacing, the `mmHg` unit, and extra lines never cost points; a mean within 0.1 mmHg, or rounded to the decimals you wrote, passes.
- The patient list is checked against your declared numeric `Cutoff`, even when the cutoff is outside 120 to 180, which loses only the cutoff points.
- Extra files are ignored.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Points |
| --- | --- | ---: |
| `README.md` | `## Project description` holds 30-300 characters of your own text. | 5 |
| `README.md` | `## Run` holds a Python command that runs a `.py` script. | 5 |
| `.gitignore` | It lists a standard pattern for Python's bytecode cache, such as `__pycache__/` or `*.py[cod]`. | 5 |
| `output/vitals_report.txt` | UTF-8 text with at least one readable labelled numeric answer from Task 2.3; each missing answer is checked separately. | 10 |
| `output/vitals_report.txt` | `Usable encounters` matches the supplied encounters. | 8 |
| `output/vitals_report.txt` | `Skipped rows` matches the supplied encounters. | 7 |
| `output/vitals_report.txt` | `Patients seen` matches the distinct patient IDs among the usable encounters. | 10 |
| `output/vitals_report.txt` | `Mean systolic` matches the mean of the usable readings. | 15 |
| `output/vitals_report.txt` | `Highest systolic` matches the largest usable reading. | 5 |
| `output/vitals_report.txt` | `Lowest systolic` matches the smallest usable reading. | 5 |
| `output/followup_list.txt` | `Cutoff` is a number from 120 to 180 mmHg. | 5 |
| `output/followup_list.txt` | `Reason` is 20-300 characters on one line. | 5 |
| `output/followup_list.txt` | The listed patient IDs are exactly the patients with a usable reading at or above your cutoff. | 15 |
