# Assignment 04 grader self-test

This directory is instructor-only course-source material. Exclude the entire `_grader_selftest/` directory from the Classroom50 student starter. The discoverable production bundle may contain `autograder.py`, `grader_core.py`, its pinned grader requirements, protected-file manifest, and alternate fixture, but it lives in the teacher-controlled Classroom50 configuration repository rather than each student repository.

The self-test materializes disposable starter, correct, incomplete, stored-output, malformed, path-dependent, hard-coded, and other defective submissions. It verifies fresh notebook execution after deleting generated files and clearing stored output, a relocated nested launch, a second valid fixture, resubmission behavior, and the required `classroom50/result/v1` payload.

Run from the repository root:

```bash
uv run 04/assignment/_grader_selftest/run.py
```

Production execution requires nonempty `CLASSROOM`, `ASSIGNMENT`,
`SUBMISSION_TAG`, `COMMIT_URL`, and `RELEASE_URL`. `REVIEW_URL` falls back to
`COMMIT_URL`; the grader generates the UTC result `datetime`. Classroom50
launches `autograder.py` with plain Python from the student checkout. That
standard-library bootstrap installs the exact sibling `requirements.txt` into
the same interpreter before importing `grader_core.py`; PEP 723 metadata is
only a local `uv run` convenience.

This is local grader QA, not external Classroom50 configuration or certification. Published grader logic is discoverable and contains no secret solution, credential, or requirement outside the student README.
