# Assignment 02: Reusable Measurement Summary

This assignment combines the Lecture 02 Git state model with a small function-to-module refactor. Work in the `02/assignment` directory, or in the standalone assignment repository created from this subtree. Do not create a second project repository.

Complete the work with terminal-executed Python scripts. Use VS Code Source Control or GitHub Desktop for all required Git actions. The delivery sequence is in [`PLATFORM_CHECK.md`](PLATFORM_CHECK.md); it is required but graded separately from the files below.

## Project description

TODO: Replace this line with a 30–300 character description of what this measurement-summary project does.

## Run

TODO: Replace this line with the exact terminal command that runs the completed program.

## Starter files

- `GIT_STATE_CHECK.md`: Part 1 state-model answers; edit this file.
- `.gitignore`: Part 1 cache patterns; edit this file.
- `analysis_utils.py`: Part 2 reusable functions; edit this file.
- `main.py`: Part 3 driver program; edit this file.
- `PLATFORM_CHECK.md`: supplied GUI delivery checklist; do not edit it.
- `check_assignment.py` and `_public_checks.py`: supplied dependency-free checker; do not edit them.
- `test_assignment.py`: public managed-pytest contract; do not edit it. You do not need pytest locally.

The supplied checking files use later Python features internally. Run them, but do not treat their implementation as a model for your Lecture 02 code.

## Part 1: Repository state and documentation

### Complete the state snapshots

Open `GIT_STATE_CHECK.md`. For each scenario, replace only the `TODO` terms in the answer block. Use each defined Lecture 02 term where it describes the snapshot: working tree, diff, staging area, commit, local branch, remote, synchronize, merge, and conflict.

### Complete the README

Replace the two TODO lines near the top of this file:

- Write a project description containing 30–300 non-whitespace characters and the word `measurement`.
- Put the exact command `python main.py` in the Run section.

Do not rewrite the rest of the assignment contract.

### Complete `.gitignore`

Replace its TODO comments so its complete contents are exactly:

```gitignore
__pycache__/
*.pyc
```

No environment pattern is required in this assignment.

The repository actions in `PLATFORM_CHECK.md` assess practical delivery separately. The Python checker does not infer GUI competence from commit counts, branch references, or history shape.

## Part 2: Reusable calculations

Complete exactly two functions in `analysis_utils.py`.

### `mean(values)`

Its interface and behavior are:

- Signature: `mean(values)` with no default value or annotation.
- First statement: a one-line docstring without `TODO`.
- Begin with `if not values:` and return `None` in that branch.
- Next initialize the local accumulator with `total = 0`.
- Use exactly one direct `for` loop over `values`. Its body must directly update `total` with the current loop value; do not put the required update in a nested or dead branch.
- After that loop, return exactly `total / len(values)`.
- Return the result; do not print it.
- Do not mutate `values` or retain accumulated state between calls.

### `format_summary(record)`

Its interface and behavior are:

- Signature: `format_summary(record)` with no default value or annotation.
- First statement: a one-line docstring without `TODO`.
- Use only the dictionary keys `"label"` and `"values"`.
- Assign one direct `mean(record["values"])` call to a local result. Do not loop, call `sum()`, or repeat arithmetic in this function.
- Test that local result with `is None`; if true, return `<label> mean: no measurements`.
- Otherwise build `<label> mean: <value>` with exactly one decimal place from that same local result.
- Return the string; do not print it or mutate the record.

Required examples:

```text
mean([18, 21, 24]) -> 21.0
mean([]) -> None
format_summary({"label": "Zero", "values": [0, 0]}) -> "Zero mean: 0.0"
format_summary({"label": "Empty", "values": []}) -> "Empty mean: no measurements"
```

Importing `analysis_utils` must be silent and must not create or change files.

## Part 3: Import-safe driver and report

Complete `main.py` without changing the supplied import, three records, or main guard.

The completed file must:

1. import `format_summary` with `from analysis_utils import format_summary`;
2. define exactly `main()` with no parameters, defaults, or annotations and a one-line docstring without `TODO`;
3. keep these supplied records in this order:

   ```python
   records = [
       {"label": "Morning", "values": [18, 21, 24]},
       {"label": "Evening", "values": [20, 22, 26]},
       {"label": "Overnight", "values": []},
   ]
   ```

4. call the imported `format_summary()` once per record, in order;
5. build the three-line report in a local name `report_text`, with a newline after every line;
6. use the exact context-manager form `with open("report.txt", "w", encoding="utf-8") as report_file:`, then write it with exactly `report_file.write(report_text)`—not `writelines()`;
7. use a second block with the exact form `with open("report.txt", "r", encoding="utf-8") as report_file:`, then assign one no-argument `report_file.read()` result to a different local name;
8. print the saved report, then print whether that read-back text equals `report_text`. Either print the equality comparison inline or first assign it to a local name such as `matches` and print that name; and
9. keep the exact guard:

   ```python
   if __name__ == "__main__":
       main()
   ```

Importing `main` must print nothing and must not create `report.txt`. Running `python main.py` must overwrite a stale report and print exactly:

```text
Morning mean: 21.0
Evening mean: 22.7
Overnight mean: no measurements
Saved report matches: True
```

The exact bytes in `report.txt` are:

```text
Morning mean: 21.0
Evening mean: 22.7
Overnight mean: no measurements
```

There is one newline after the final report line. The `Saved report matches` status belongs only in terminal output, not in `report.txt`.

## Check your work

Run each student script from the assignment directory:

```bash
python main.py
python check_assignment.py
```

A complete submission ends with:

```text
All public checks passed.
```

If a check fails, use its message to revise the named student file, rerun `python main.py`, and run the checker again. The checker executes fresh temporary copies and does not trust a stored report as proof that the current program works.

The optional GitHub Actions workflow may run the public `test_assignment.py` contract on pushes and pull requests. Instructor or TA grading may run the same written contract from a trusted checkout; the workflow is feedback, not a submission requirement, and its implementation is not a model for the student code.

## Scope boundaries

This is a competence-focused pass/fail assignment. Do not add a shell script, notebook, dependency file, third-party package, CSV/JSON input, second repository, command-line Git workflow, or forced merge conflict.

In the two student Python files, do not add:

- comprehensions or generator expressions;
- `lambda`, classes, async functions, decorators, type annotations, default parameters, keyword-only parameters, `*args`, or `**kwargs`;
- `try`/`except`, `global`, or `nonlocal`;
- imports other than the supplied `from analysis_utils import format_summary` line in `main.py`;
- `sum()` in place of the required local total and plain `for` loop;
- printing or file I/O in `analysis_utils.py`;
- dictionary keys other than `"label"` and `"values"` in `format_summary()`; or
- driver statements at module top level outside the exact main guard.

The direct-call boundary is also part of the assignment:

- `mean()` may directly call only `len()`;
- `format_summary()` may directly call only `mean()`;
- `main()` may directly call only the supplied `format_summary()`, `open()`, and `print()`, plus `write()` and `read()` on the matching report-file handles;
- do not replace those call names, select a call indirectly or dynamically, or call through an assigned alias, an attribute lookup, a subscript, or `__builtins__[...]`;
- do not use `exec()`, `eval()`, `compile()`, `__import__()`, or any other unapproved call; and
- do not add an extra file open, use append mode, call `writelines()`, or perform file I/O through an indirect call. `main()` must contain exactly the two ordered `report.txt` opens described in items 6–7.

These restrictions are explicit course boundaries, not hidden style rules. Every public structure check corresponds to an item above.
