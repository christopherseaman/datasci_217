# Classroom 50 pilot: Assignment 04

## Why Assignment 04

Assignment 04 is representative without being the most complex: it uses notebooks, generated data, pandas dependencies, output files, and pytest grading. It is also the first lecture for which Colab is intended to be a default execution path, allowing both new workflows to be tested together.

Assignment 04 is only the implementation pilot. Classroom 50 is intended for every active course assignment, including the terminal-based assignments in Lectures 01–03.

## Architecture decision

Use Classroom 50's teacher-controlled configuration repository and per-assignment autograder bundle. Do not copy grader-only tests or grading logic into student starter repositories.

This centralizes change control but does not make the grader confidential. Classroom 50 publishes manifests, autograder code, and per-assignment bundles through unauthenticated GitHub Pages. Treat all published tests, fixtures, and grader logic as discoverable. Never put solutions, credentials, confidential data, or tests whose value depends on secrecy in the published bundle.

The current `classroom.yml` pattern downloads tests from a public `UCSF-DataSci/ds217_25f_*` repository at grading time. Replace that design in the pilot with either:

1. Classroom 50 declarative Python tests, if the existing pytest suite maps cleanly; or
2. A per-assignment `autograder.py` and grader pytest files in the Classroom 50 config repository.

The second option is the likely fit for notebook/output-file grading.

The current runner launches the bundle's `autograder.py` with plain
`[sys.executable, entrypoint]` from the student checkout. Therefore each custom
bundle must provide a standard-library bootstrap that installs its exact sibling
requirements into that same interpreter before importing dependency-bearing
grader code. A sibling `requirements.txt` or PEP 723 block alone is not
production provisioning.

Accepted repositories may add exactly `.classroom50.yaml` and
`.github/workflows/autograde.yaml` as delivery-owned files. Exact inventory
checks ignore only the top-level `.git/**` repository metadata tree, accept
those two paths, and reject other root files, other `.github/**` files, a
nested `ordinary/.git/**` tree, and any student-authored `_grader_selftest/**`
tree.

## Pilot stages

### 1. Instructor sandbox

- Create a disposable GitHub organization or isolated test classroom.
- Confirm the organization has the Team tier required by Classroom 50, including verified-educator access if applicable.
- Install and authenticate the Classroom 50 teacher CLI.
- Initialize the Classroom 50 config repository and Pages assets.
- Create a two-person test roster using non-production accounts.
- Record every required organization permission and token scope.

### 2. Starter package

- Copy only student-facing Assignment 04 files into a clean template repository.
- Remove `.github/workflows/classroom.yml` and any embedded grader-only tests.
- Add local test instructions and a minimal public smoke test.
- Ensure `data/` and `output/` are generated reproducibly.
- Add `.classroom50.yaml` only through the normal acceptance workflow, not to the template by hand.

### 3. Autograder

- Place non-secret grader tests and fixtures under the Assignment 04 bundle in the config repository.
- Run pytest through a per-assignment `autograder.py`.
- Emit the required `classroom50/result/v1` result.
- Require nonempty runner context `CLASSROOM`, `ASSIGNMENT`, `SUBMISSION_TAG`,
  `COMMIT_URL`, and `RELEASE_URL`; use `REVIEW_URL` when nonempty and otherwise
  fall back to `COMMIT_URL`; generate `datetime` in UTC inside the grader.
- Leave `owner`, `assignment_type`, and `submitted_by` to the runner. Missing or
  empty required context is an infrastructure failure: exit nonzero and do not
  write a plausible local-default `result.json`.
- Use stable, rubric-aligned test names and explicit point weights.
- Set realistic timeouts and pin grading dependencies.
- Test `autograder.py` through plain Python in a fresh dependency-empty runtime;
  do not substitute `uv run` for the production entrypoint.

### 4. End-to-end scenarios

Test all of the following:

- assignment acceptance;
- starter repository correctness;
- submission using `gh student submit`;
- submission using a normal push to `main`;
- a correct notebook and output set;
- missing notebook cells or outputs;
- malformed/corrupt notebook JSON;
- an incomplete submission;
- a resubmission after feedback;
- score collection and export;
- instructor review link;
- student-visible failure messages;
- unavailable Pages asset or grading dependency;
- recovery/regrade without editing every student repository.

## Pass criteria

- No solution, credential, confidential data, or other secret is present in the student repository or published grader bundle.
- The assignment remains valid when students can inspect the published tests and grading logic.
- Students cannot modify the centrally managed grader used on their next submission.
- Correct, starter, and incomplete fixtures produce the expected scores.
- A student can complete acceptance and submission from documented commands without instructor intervention.
- The instructor can identify the latest score, submission history, and review diff.
- Grading configuration updates apply centrally to a subsequent submission.
- The gradebook can be exported in a form usable by the course's official grade system.
- Failure recovery and regrading are documented.
- The instructor accepts the GitHub organization permissions and maintenance cost.

## Migration sequence after the gate

Migrate by grader archetype so each pattern is implemented and reviewed once:

1. Assignment 01: recover or rebuild the historical terminal setup assignment and replace its obsolete grader design.
2. Assignments 02–03: terminal/script assignments using declarative `run` or `python` tests where possible.
3. Assignments 06–09: structurally similar notebook/output assignments that can reuse the Assignment 04 pilot pattern.
4. Assignment 10: modeling dependencies and network dataset download, with a separately pinned runtime.
5. Assignments 05 and 11: mixed or multi-notebook assessments with the largest custom and manual grading surfaces.

## Open operational decisions

- Production GitHub organization and classroom name.
- Individual versus group mode for each assignment.
- Roster source and identity reconciliation process.
- Late-submission rule and which submission is authoritative.
- LMS/grade-system import format.
- Retention policy for student repositories and release artifacts.
- Whether student CLI installation is acceptable or web/push submission must be the documented default.
- Whether the course's assessment policy permits discoverable autograder tests; use a different grading path where hidden tests are required.
