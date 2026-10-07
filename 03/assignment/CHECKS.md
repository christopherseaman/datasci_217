# Assignment 03 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the first two checks report:

```text
[FIX ]   0/7   environment probe: numpy
         output/environment.txt is missing. Save its three labelled lines with the Task 1.2 commands, then commit it.
[FIX ]   0/6   environment probe: interpreter  (same fix as above)
```

A clean run ends with:

```text
[PASS]   3/3   answer: stage2_other_monitors

Score: 100/100
All checks passed.
```

How the answers are read:

- Each answer is scored on its own, so a wrong value costs only its own points. `highest_patient_mean` and `peak_hour_mean` are also accepted when they are right for the patient or hour your earlier label names. `monitor_offset` and `stage2_other_monitors` are accepted when they are right for your `high_monitor`. A wrong selection costs only its own points; an independently wrong calculation still costs its own points.
- A number answer is read from the first number after the colon, so put the answer first and any note after it: `readings: 3600 (300 x 12)` reads 3600, but `readings: 300 x 12 = 3600` reads 300.
- A number answer is one number, not an array. A line holding a printed array, such as `mean_sbp: [130.86666667 133.98333333 ...`, fails whatever its first number is: `readings.mean(axis=0)` gives one mean per hour column, and `mean_sbp` is the mean of every reading, which `readings.mean()` gives.
- mmHg values are accepted within 0.6 of the value recomputed from the data, so one decimal or every digit NumPy prints passes, and so does a whole number, whether rounded with `:.0f` or cut short with `int()`. A trailing unit such as `mmHg` is ignored. Counts and whole-number readings must match exactly.
- A NumPy scalar printed as `np.float64(121.5)` or `np.int64(96)` reads as the number inside it.
- Either the population or the sample standard deviation is accepted; at this many readings they agree far inside the tolerance.
- "140 mmHg or higher" includes a mean of exactly 140.
- A patient id, column name, or monitor id may sit in quotes or brackets, as a one-item list prints it (`['M02']`), and may carry a note before or after it, as in `monitor M02` or `M02 (128.4 mmHg)`, as long as the line names no other id of the same kind.
- Keys may appear in any order, spacing is free, a Markdown table row such as `| patients | 300 |` reads like `patients: 300`, and extra lines are ignored. When a key appears on more than one line, the first is read, so open the file with `"w"`, which replaces it on each run, rather than `"a"`.
- A write without `"\n"` runs the next answer onto the same line, as in `patients: 300readings: 3600`. Each answer on such a line is read up to the next answer's key and still scored on its own, but end each write with `"\n"` so every answer gets its own line; a wrong answer on a joined line says so in its feedback.

Checks read only the files in `output/` and recompute every answer from their own copy of `data/bp_readings.csv`; they never run or read your Python code.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact                                | Complete when                                                                                                                            | Check                                                       | Points |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | -----: |
| `output/environment.txt`                | Its `numpy` line holds a version number. Which version is not graded, and neither is the `python` line.                                  | environment probe: numpy                                    |      7 |
| `output/environment.txt`                | Its `interpreter` line is not empty. Which interpreter is not graded.                                                                    | environment probe: interpreter                              |      6 |
| `output/record_count.txt`               | It holds the number of patient records in the supplied CSV.                                                                              | record count artifact                                       |     10 |
| `output/monitor_counts_<timestamp>.txt` | A counts file in `output/` has the run timestamp in its name.                                                                            | monitor counts: timestamped name                            |      3 |
| `output/monitor_counts_<timestamp>.txt` | A counts file in `output/` holds that monitor's patient count.                                                                           | one check per monitor, named `monitor counts: M01` to `M06` |     12 |
| `output/vitals_summary.txt`             | It has a readable `key: value` line for at least one key in Task 3's table. A missing or unreadable key costs only its own answer check. | summary artifact format                                     |     12 |
| `output/vitals_summary.txt`             | Each of the 14 answers matches the supplied readings.                                                                                    | one check per key, named `answer: <key>`                    |     50 |

Extra files and extra lines are ignored.
