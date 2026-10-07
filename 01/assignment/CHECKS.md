# Assignment 01 checks

The report prints one `PASS` or `FIX` line per check with its points. A line that differs from the expected one shows what it should read, what yours reads, and which script prints it. A clean local run ends with:

```text
[PASS] 15/15 identity hash on the roster

Score: 100/100
All checks passed.
```

How the files are read:

- Checks read only the files in `terminal-practice/` and `output/`; they never run or read your Python code.
- Letter case, equivalent number formats, whitespace, blank lines, and extra lines are ignored.
- Below the score, `Left to fix` lists the failing checks and their points, as in `Left to fix (5 points): output/readiness.txt: Total.`
- When the next checks need the same fix, such as a missing file, they say `(same fix as above)`.

## Completion contract

Grading totals 100 points.

| Artifact | Complete when | Points |
|---|---|---:|
| `terminal-practice/source.txt` and `terminal-practice/path-check.txt` | Each exists as a regular file in a regular `terminal-practice` directory. Their contents are not checked. | 20 (10 each) |
| `output/readiness.txt` | A UTF-8 text file in a regular `output` directory holding the 13 report lines from Tasks 1.2, 2.2, and 3.1 after the Python version, in order. | 65 (5 each) |
| `output/student_identity.txt` | A regular file in a regular `output` directory holding one hash from the course roster; surrounding whitespace and letter case are ignored. | 15 |

A wrong or missing report line costs only its own 5 points. `Mean` also earns credit when it is calculated correctly from your declared `Total` and `Count`; `Review count` earns credit when it counts the four measurement labels you saved. The identity hash is checked separately from the report. Extra files are ignored, but keep the supplied ones, because `capture_identity.py` needs `process_email.py` and the checks need their own files.
