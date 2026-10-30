# Assignment 08 checks (course-owned)

These are the checks that decide the Assignment 08 grade. They read the six
CSV files a submission saves in `output/` and compare them with values frozen
from the supplied exam log, `data/imaging_exams.parquet`.
`08/assignment/.github/workflows/tests.yml` downloads them on every student
push from `christopherseaman/datasci_217@main:08/assignment_checks/` (its
`CHECKS_REPO`, `CHECKS_REF`, and `CHECKS_PATH`), checks that they run, and
grades with them.

The handout ships a byte-identical copy of every file the workflow lists in
`CHECKS_FILES`, so `python3 check_assignment.py` in a student's repository
prints each check's result, what to fix, and the score, exactly as GitHub will.
The listed files:

```text
.github/test/test_assignment.py
.github/test/requirements.txt
test_assignment.py
_value_checks.py
check_assignment.py
grading.py
```

## Publishing

Pushing this directory to `main` publishes it; there is no separate checks
repository. The workflow downloads the files named in `CHECKS_FILES` over the
committed copies. A fetch that misses any one of them, or a set that does not
run together, is rolled back, and the run grades with the copy committed in the
student's repository and says so in the log.

Correct a check here and copy the changed file over its twin in
`08/assignment/`; the self-test fails while the two differ. Every fork gets the
correction on its next push. The local `check_assignment.py` downloads the
current `_value_checks.py` and `grading.py` from main, falling back to the
bundled copy offline or with `--local-checks` / `DS217_LOCAL_CHECKS=1`, so local
and GitHub runs agree.

`scripts/build_assignment08_data.py` writes `08/assignment/data/imaging_exams.parquet`
from a fixed seed and rewrites the GENERATED block of `_value_checks.py` with the
values each artifact should hold, then copies `_value_checks.py` over its handout
twin. Rerun it after any change to the generator, and commit the Parquet file and
both copies together. The notebook downloads the Parquet file from `main` when a
student's `data/` lacks it.

A change to the check list or to `POINTS` belongs in `_value_checks.py` and
`grading.py` at once, in the same order: `grading.py` pairs the checks with
`POINTS` and raises when their counts differ. Keep the README's checkpoints and
completion contract in agreement with the checks; the self-test compares them.

## Checking the checks

```bash
uv run --python 3.13 --with numpy==2.3.3 --with pandas==3.0.5 --with pyarrow==25.0.0 --with 'pytest>=8,<9' \
    python 08/assignment_checks/_grader_selftest/run.py
```

It grades every kind of submission and confirms the handout's copy matches this
one byte for byte.
