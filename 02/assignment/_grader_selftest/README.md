# Assignment 02 grader self-test

This directory is instructor-only course-source validation. It materializes correct and adversarial submissions and checks that the public contract accepts and rejects them as intended.

Exclude this entire `_grader_selftest/` directory when creating the standalone student assignment repository. That repository contains only:

- `README.md`
- `.gitignore`
- `GIT_STATE_CHECK.md`
- `analysis_utils.py`
- `main.py`
- `PLATFORM_CHECK.md`
- `check_assignment.py`
- `_public_checks.py`
- `test_assignment.py` when the public managed-pytest contract is packaged

Instructor grading tests should independently enforce the written behavior against committed assignment work. They must not import `_public_checks.py` from a student repository. This self-test imports the course-source copy only to validate the public checker itself; it is not an instructor grading test.

Run from the repository root with managed Python 3.12.13:

```bash
python 02/assignment/_grader_selftest/run.py
```

Add `--pytest` to also run the public pytest facade against the correct fixture.

The current suite accepts both documented read-back comparison forms: an inline
comparison and a local name assigned from that comparison. It rejects 34
defective fixtures, including dead or hard-coded accumulation, duplicated
formatting arithmetic, falsey-zero handling, mutation and side effects,
forbidden or dynamically selected calls, indirect file I/O, extra/append-mode
opens, a published-spy special case, hard-coded driver output, and wrong
formatter order. These fixtures validate the public checker; they are not a
claim that instructor grading tests are secret.

The correct fixture is also executed by absolute script path from a separate
working directory. That run must still produce the exact 97-byte stdout and the
exact 70-byte report in the active working directory.
