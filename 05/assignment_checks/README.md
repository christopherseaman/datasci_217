# Assignment 05 checks (course-owned)

These are the checks that decide the Assignment 05 grade. They read the eleven
small summaries a submission saves in `output/` and compare them with values
recomputed from the lab export: `generate_data.py` rebuilds the export's rows in
memory, and `_value_checks.py` runs the README's pipeline on them without pandas,
once as written and once with each mistake a student's earlier files show.
`05/assignment/.github/workflows/tests.yml` downloads them on every student
push from `christopherseaman/datasci_217@main:05/assignment_checks/` (its
`CHECKS_REPO`, `CHECKS_REF`, and `CHECKS_PATH`), checks that they run, and
grades with them.

The handout ships a byte-identical copy of every file the workflow lists in
`CHECKS_FILES`, so `python3 check_assignment.py` in a student's repository
prints each check's result, what to fix, and the score, exactly as GitHub will. The notebook's last cell calls
`run_checks()` from the same file, so it prints that report too.
The listed files:

```text
.github/test/test_assignment.py
.github/test/requirements.txt
test_assignment.py
_value_checks.py
check_assignment.py
grading.py
generate_data.py
```

## Publishing

Pushing this directory to `main` publishes it; there is no separate checks
repository. The workflow downloads the files named in `CHECKS_FILES` over the
committed copies. A fetch that misses any one of them, or a set that does not
run together, is rolled back, and the run grades with the copy committed in the
student's repository and says so in the log.

Correct a check here and copy the changed file over its twin in
`05/assignment/`; the self-test fails while the two differ. Every fork gets the
correction on its next push. The local `check_assignment.py` downloads the
current `_value_checks.py`, `grading.py`, and `generate_data.py` from main, falling back to the
bundled copy offline or with `--local-checks` / `DS217_LOCAL_CHECKS=1`, so local
and GitHub runs agree.

The data is generated, never committed. `generate_data.py` (seed 217, stdlib
only) writes `data/lab_visits.csv` in about a second, and the handout's
`data_manifest.json` holds its checksum, which the notebook's first cell
verifies. A change to the generator changes the data: regenerate, update
`data_manifest.json`, and update the Expect values in `05/assignment/README.md`;
the self-test fails while the manifest and the generator disagree. Change the
generator only before handout, since the checks download it on every run.

A change to the check list or to `POINTS`
belongs in `_value_checks.py` and `grading.py` at once, in the same order:
`grading.py` pairs the checks with `POINTS` and raises when their counts
differ. It compares the counts itself rather than using `zip(strict=True)`, so
a student's Python 3.9 still runs the checks. Keep CHECKS.md's completion
contract in agreement with `POINTS`.

## Checking the checks

```bash
uv run --python 3.13 --with pandas==3.0.5 --with pyarrow==25.0.0 python 05/assignment_checks/_grader_selftest/run.py
```

It grades every kind of submission and confirms the handout's copy matches this
one byte for byte.
