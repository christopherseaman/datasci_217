# 2026 Course Refresh Review Guide

This document summarizes the changes on `2026-refresh` relative to `main`. It is intended to help reviewers distinguish intentional course changes from items that still require reconciliation before release.

## Review Scope

The branch combines two related workstreams:

- Migration of the course website to Eleventy.
- Refresh of lecture demos, assignments, data, and grading infrastructure.

The lecture restoration and Lecture 11 rebuild are separated in the latest commits:

- `87c753a fix: restore original lecture materials`
- `759664b feat: rebuild lecture 11 workflow`

Use `git diff main...HEAD` to review the complete branch rather than reviewing only the latest commit.

## Priority Review Issue

Lecture 11 is internally aligned, but Lectures 02-10 are not uniformly release-ready. The branch restores much of the original student-facing assignment prose while retaining rebuilt fixtures, checkers, and grader bundles from the earlier refresh work. Several assignment READMEs therefore refer to files or workflows that no longer exist.

Before release, decide whether each restored assignment or its rebuilt package is authoritative, then align its instructions, starter files, dependencies, public checker, and grader.

Examples include:

- `02/assignment/README.md` describes a legacy shell/Git project while the package now contains `analysis_utils.py`, `main.py`, and a new checker.
- `03/assignment/README.md` references the removed health-data generators while the package now contains a small NumPy exercise.
- `04/assignment/README.md` requires the removed `data_generator.ipynb`.
- `05/assignment/README.md` describes the removed multi-part clinical-trial package and still links to `TIPS.md`. That file was moved unchanged to the site-excluded `src/content/docs/05/assignment/TIPS.md`, so the student-facing link is broken.
- `06/assignment/README.md` through `10/assignment/README.md` similarly refer to removed generators, notebooks, datasets, or grading paths.

The plans and audit files under `work/` describe intermediate refresh states. They should not be treated as certification of the final tree after the lecture restoration commit.

## Lecture Summary

| Lecture | Student-facing changes from `main` | Review status |
| --- | --- | --- |
| 01 | Adds a terminal-based readiness assignment with scripts, public tests, and an instructor self-test. No notebook requirement. | Review the new assignment as a complete addition. |
| 02 | Replaces legacy Classroom workflow/tests with local checks, starter modules, and an instructor self-test. | Instructions and rebuilt package need reconciliation. |
| 03 | Adds reproducibility terminology and a pinned Python/NumPy recreation path. Adds a smaller NumPy assignment package. | Lecture additions are targeted; assignment instructions and package need reconciliation. |
| 04 | Rebuilds three demos around notebook state, pandas fundamentals, and portable data I/O. Adds fixed assignment fixtures and grader infrastructure. | Demos are the main intentional teaching change; assignment instructions and package need reconciliation. |
| 05 | Adds definitions for raw/clean data, schema, provenance, and executable validation. Replaces the large legacy assignment package with one notebook and fixed fixtures. | Assignment scope and its role as a midterm need confirmation. |
| 06 | Retains the original lecture material. Replaces generator-based assignment infrastructure with fixed join/reshape fixtures and local grading tools. | Assignment instructions and package need reconciliation. |
| 07 | Adds guidance on question, audience, claim, grain, encoding, and accessibility. Adds deterministic assignment fixtures and grading tools. | Assignment instructions and package need reconciliation; generated demo outputs are excluded from Git. |
| 08 | Retains the original lecture material. Adds a deterministic support-request fixture and local grading tools. | Assignment instructions and package need reconciliation. |
| 09 | Adds source-versus-grid missingness, measurement-aware resampling, window semantics, and prediction-time availability. Replaces the three-part assignment with one notebook package. | Assignment instructions and package need reconciliation. |
| 10 | Retains the original lecture material. Adds fixed modeling fixtures and local grading tools. | Assignment instructions and package need reconciliation; dependency policy remains unresolved. |
| 11 | Rebuilds the end-to-end demos and assignment around frozen, reproducible datasets with a clear required/bonus boundary. | Internally aligned and locally validated; publication and platform checks remain. |

## Lecture 11 Detail

### Required demos

The four required notebooks use a compact NYC Yellow Taxi release:

- `11/demo/01_setup.ipynb`
- `11/demo/02_wrangling.ipynb`
- `11/demo/03_model_prep.ipynb`
- `11/demo/04_modeling.ipynb`

The workflow uses zone-hour observations, a next-hour pickup target, a `lag_168` baseline, MAE, and chronological train/validation/test periods. Required notebooks can rebuild prerequisites when prior notebook outputs are absent.

The frozen source artifacts and provenance are in:

- `11/demo/data/demo_release_manifest.json`
- `11/demo/data/taxi_zone_lookup.csv`
- `11/demo/data/yellow_taxi_2023_h1_event_sample.parquet`
- `11/demo/data/yellow_taxi_2023_h1_zone_hour_counts.parquet`
- `scripts/build_lecture11_demo_data.py`

### Optional geography

Geographic analysis is explicitly demo-only and non-graded:

- `11/demo/05_geo_bonus.ipynb`
- `11/demo/BONUS.md`

Geospatial dependencies and polygon operations are excluded from the four required demos and from the assignment.

### Assignment

The assignment remains a nine-phase workflow but now uses Chicago Beach Weather Sensors to forecast next-hour air temperature:

- `11/assignment/q1_setup_exploration.ipynb` through `11/assignment/q9_writeup.ipynb`
- `11/assignment/assignment.md`
- `11/assignment/report.md`

The release contains 50,895 observations from 2022-2024 and is pinned by a manifest:

- `11/assignment/data/chicago_beach_sensors_2022_2024.csv`
- `11/assignment/data/release_manifest.json`
- `scripts/build_assignment11_data.py`

The modeling contract uses chronological periods, a persistence baseline, one selected scikit-learn model, one final test evaluation, and a written report. Geographic work is not part of the assignment.

### Grading

The student-facing checker validates structure and cross-artifact consistency:

- `11/assignment/check_assignment.py`

The instructor-only Classroom 50 bundle applies deterministic reference checks:

- `11/assignment/_grader_selftest/classroom50_grader.py`
- `11/assignment/_grader_selftest/run.py`

Grading is artifact-only. It does not inspect notebook source or ASTs. Passing rows receive deterministic credit; failed or blocked rows are directed to targeted human review. Artifact checks cannot independently prove training-only fitting or the provenance of model-selection decisions.

Legacy GitHub Classroom workflow files and large committed example outputs were removed.

The canonical source/starter for each assignment is the corresponding
`NN/assignment` subtree in this repository. Each subtree now carries a portable
`.github/test` pytest entrypoint and an optional `.github/workflows/tests.yml`
workflow. Those nested workflows are intentionally dormant in this monorepo;
when an assignment is later copied to repository root, the same subtree becomes
the standalone assignment repository and its Actions feedback works without
mutable remote test downloads. The instructor-only `_grader_selftest` bundles
remain source material for the later TA grading runner and are excluded from
student-repository exports.

## Course-wide Infrastructure

- Legacy per-assignment GitHub Classroom workflows were removed from Assignments 02-11.
- Local public checkers and instructor self-test bundles were introduced.
- Assignments 04-11 retain instructor-only grader bundles as migration input;
  they are not part of the learner-facing repository export.
- The current portable learner contract is per-assignment pytest plus optional
  GitHub Actions; the `NN/assignment` directories are the source bundles from
  which standalone assignment repositories can be published later.
- Fixed local fixtures replace many random generators and large committed outputs.
- Instructor grader bundles must be excluded when student template repositories are created.

## Website Changes

The branch also replaces the prior site implementation with Eleventy:

- `.eleventy.js` and `.eleventyignore`
- `_data/nav.js`
- `_includes/`
- `css/`
- `.github/workflows/deploy.yml`

Assignments and demos are excluded from the generated site. Review whether lecture links to those files should point to repository sources, downloadable assets, or separately published content.

The current site also has concrete content defects to resolve:

- `_data/nav.js` mislabels Lectures 04-11; those labels are also used for generated page titles.
- `01/README.md` and `06/README.md` contain unresolved `attachment:` image URLs.
- `06/README.md` contains a legacy GitHub Classroom URL and an unrelated Notion bonus link.

## Validation Status

The following Lecture 11 checks were completed during implementation:

- Both frozen data releases rebuilt byte-for-byte.
- All assignment and demo Jupytext strict checks passed.
- Four required demo notebooks and the optional geography notebook executed successfully.
- The grader harness produced `100/100` for the correct fixture and `0/100` for the empty fixture.
- The grader harness covered all nine isolated rubric mutations plus blocking, malformed artifact, context, bootstrap, metrics, model-specification, and resubmission paths.
- The Eleventy build completed with 27 pages and 124 assets.
- `git diff --check` passed before the Lecture 11 commit.

The course-wide `scripts/course_audit.py` is not currently a reliable pass/fail release gate. It reports 46 errors because many expectations describe the intermediate rewritten Lectures 04-10 rather than the restored final lecture material. Its repository-wide warnings are still actionable: the current run also reports 22 warnings, including unresolved attachment URLs and legacy Classroom references.

## External Release Gates

These checks require publication or production services and remain open:

- Publish an immutable annual release tag and replace Lecture 11 URLs that still reference `main`.
- Run all supported notebooks in a fresh Colab environment.
- Confirm assignment save-back behavior from Colab.
- Provision the production Classroom 50 assignments and test submission, score collection, review links, resubmission, and regrading end to end.
- Confirm the course-wide dependency constraints and immutable grader-container digest.
- Validate the final deployed Eleventy site, navigation, media, and links.

## Suggested Review Order

1. Review the Priority Review Issue and choose the authoritative assignment design for Lectures 02-10.
2. Review Lecture 11's assignment contract in `11/assignment/assignment.md`.
3. Review the Lecture 11 grader boundary and rubric implementation.
4. Review frozen data provenance and reproduce both builders.
5. Execute the required notebooks from clean environments.
6. Review the Eleventy output and content-link strategy.
7. Complete the external release gates before merging to `main`.
