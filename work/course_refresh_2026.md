# DataSci 217 course refresh: 2026–27

Status: active implementation and independent verification

## Agreed scope

- Review the complete 11-lecture sequence and all associated material.
- Review lecture content, bonus material, demos, assignments, graders, dependencies, media, links, and instructor guidance together.
- Apply the same content-quality review to Lectures 01–03 as to the rest of the course; their terminal-only constraint removes Colab work, not curriculum review.
- Before Jupyter is introduced in Lecture 04, demos and assignments use Python scripts or shell files in a terminal.
- From Lecture 04 onward, demos and assignments may use Jupyter notebooks. Compatible demos use Google Colab as the default launch path and retain local Jupyter support.
- Move the entire course assignment workflow to Classroom 50, including the terminal-based assignments before Lecture 04.
- Pilot Classroom 50 with one representative assignment before reusing the pattern across the course.
- Preserve the course-design distinction between terminal-first Python and GUI-first Git unless that policy is explicitly changed; command-line Git currently belongs in bonus material.
- Treat Lectures 01–05 as the foundational path and Lectures 06–11 as the advanced path when checking completeness and prerequisites.
- Settle lecture content and concept order before redesigning its demos and assignment.

## Initial baseline

- The repository began with 11 lecture directories and assignment directories only for Lectures 02–11.
- Assignment 01 existed in repository history but had been removed “to reduce confusion.” It has now been redesigned, implemented, and independently verified against the refreshed Lecture 01 objectives.
- The initial tree contained 10 legacy assignment workflows. They are removed lecture by lecture as each assignment reaches its implementation gate.
- The structural audit currently parses 52 notebooks across demos and assignments; this count may remain stable while obsolete notebooks are replaced one-for-one.
- Lectures 01–03 contain no notebooks, consistent with the pre-Jupyter boundary.
- Lecture 04 demos have established the Colab-first/local-Jupyter notebook pattern; later notebook demos are converted only after their narratives stabilize.
- Classroom 50 centralizes grader control, but currently publishes assignment manifests, grader code, and per-assignment bundles through unauthenticated GitHub Pages. Grader logic must be treated as discoverable.

Detailed evidence and the structural scorecard are in [2026_refresh_audit.md](2026_refresh_audit.md).
The reusable team process is in [lecture_review_workflow.md](lecture_review_workflow.md).
The working prerequisite graph and course-level alignment contract are in [course_dependency_alignment.md](course_dependency_alignment.md).

## Environment boundary

| Material | Lectures 01–03 | Lectures 04–11 |
|---|---|---|
| Required demos | Terminal with `.py` or `.sh` files | Colab-first when notebook-friendly; local Jupyter supported |
| Assignments | Terminal with `.py` or `.sh` files | Notebook or terminal based on the learning objective |
| Shell and environment work | Local terminal | Local terminal whenever those tools are being taught |
| Git workflow | VS Code Source Control or GitHub Desktop; CLI Git is bonus | VS Code Source Control or GitHub Desktop; CLI Git is bonus |
| Assignment distribution and grading | Classroom 50 | Classroom 50 |

Do not create notebook wrappers for Lectures 01–03 merely for consistency. After Lecture 04, do not require a terminal-script version when a notebook is the appropriate teaching format.

## Deliverables

1. A reviewed lecture sequence with explicit objectives, prerequisites, topic progression, and core/bonus boundaries.
2. A review record for every lecture covering its content, demos, and assignment as one unit.
3. Reproducible instructor run sheets for all required demos.
4. A resolved Assignment 01 package and reviewed Assignments 02–11.
5. Classroom 50 starter templates, centrally controlled graders, and instructor/student workflows for every course assignment.
6. Colab-certified demo notebooks for Lectures 04–11 with local Jupyter fallbacks.
7. A supported Python/package matrix and clean-environment test record.
8. A release candidate of the course site with current links and no legacy GitHub Classroom workflow dependencies.

## Review method

### Course-level review

1. Identify the intended course outcomes and map them to the 11 lectures.
2. Build a prerequisite graph and find concepts that are used before introduction, repeated without purpose, or never applied.
3. Check that the sequence forms a coherent progression from terminal tools and Python through tabular analysis, visualization, aggregation, time series, modeling, and an end-to-end workflow.
4. Identify overlap, gaps, unnecessary library surveys, and material that belongs in bonus sections.
5. Map every graded assignment component to a lecture or course objective.
6. Confirm terminology, commands, APIs, and recommended workflows are consistent across lectures.

### Per-lecture review

Review the lecture narrative, bonus material, media, demos, and assignment together. Record:

- measurable learning objectives;
- explicit prerequisites and where they were previously introduced;
- topics to keep, revise, remove, combine, or move to bonus;
- demo coverage of the objectives;
- assignment and rubric coverage of the objectives;
- execution environment and student workflow;
- dependencies, data sources, resource demands, and expected outputs;
- current API and deprecation findings;
- internal/external link, media, accessibility, and licensing findings;
- common student failure modes and instructor troubleshooting notes;
- Classroom 50 grader type and validation fixtures;
- a disposition for each finding: fix, defer with rationale, move, or remove.

A lecture is approved only when its content, demos, assignment, and grader work as a coherent unit.

## Initial lecture review priorities

| # | Core role | Execution mode | Review focus |
|---|---|---|---|
| 01 | Setup, shell, Python, VS Code | Terminal only | Review all content and demos; recover or redesign the removed Assignment 01; validate the complete setup-to-submission workflow. |
| 02 | GUI Git workflow, functions, modules | Terminal Python; GUI Git | Remove command-line Git from the required path; review content volume and overlap with Lecture 01; execute every demo and assignment path; migrate its grader to Classroom 50. |
| 03 | Environments and NumPy | Terminal only | Review the environment strategy and NumPy progression; execute all demos and assignment cases; migrate its grader to Classroom 50. |
| 04 | Jupyter, pandas structures, data I/O | Jupyter introduced | Review the transition from terminal scripts to notebooks; use its demos for the Colab standard and its assignment for the Classroom 50 pilot. |
| 05 | Cleaning and preparation | Notebook plus terminal where useful | Separate instructional content from assessment administration; review the mixed script/notebook assignment and generated artifacts. |
| 06 | Merge, concatenate, reshape | Colab-first demos | Repair media and links; review merge cardinality, validation, demos, and notebook assignment together. |
| 07 | Visualization | Colab-first demos | Review required visualization libraries, principles, accessibility, rendering, and generated images; move the broad library survey to bonus where appropriate. |
| 08 | GroupBy, `pivot_table`, aggregation | Colab-first for analysis; terminal for remote tools | Keep aggregation central; review whether SSH, tmux, and remote-performance material supports the lecture or belongs in a separate local/bonus section. |
| 09 | Time series | Colab-first demos | Review pandas frequency aliases, time zones, resampling, data sizes, demos, and the multi-notebook assignment. |
| 10 | Statistical and predictive modeling | Colab-first for supported models | Reduce required library breadth; make statistical modeling and scikit-learn the core and review boosting/deep learning as optional material. |
| 11 | Complete data-science workflow | Colab/local Jupyter after data validation | Review dataset reliability, licensing, the nine-notebook assignment structure, modeling prerequisites, report requirements, and grading approach. |

## Work order

### 1. Establish the course map

- Confirm course outcomes and apply the working role of each lecture in `course_dependency_alignment.md`.
- Complete the prerequisite graph and objective-to-assignment map.
- Adopt one per-lecture review record and issue classification.
- Resolve duplicate or conflicting plans and identify the current source of truth.

### 2. Review Lectures 01–03 completely

- Review every lecture page, bonus page, demo guide, executable demo, assignment instruction, starter artifact, and test.
- Preserve the terminal-only execution model.
- Resolve Assignment 01 using the historical package as evidence rather than silently treating the lecture as unassigned.
- Validate the actual commands on each supported operating-system path.
- Confirm that students finish Lecture 03 prepared for the Jupyter transition.

### 3. Review Lecture 04 and establish notebook standards

- Review the Jupyter introduction and notebook-state model.
- Convert the Lecture 04 demos to the proposed Colab/local notebook contract.
- Pilot Assignment 04 in Classroom 50 because it exercises notebooks, generated data, output files, pandas, and pytest.
- Use pilot results to choose the reusable notebook grader pattern.

### 4. Review Lectures 05–11

- Apply the same content/demo/assignment review record to each lecture.
- Revisit Lecture 10 scope before certifying its environment.
- Review Lecture 11 after the upstream modeling and data decisions are stable.
- Track cross-lecture fixes when a later review exposes an earlier prerequisite problem.

### 5. Migrate every assignment to Classroom 50

- Recover or redesign Assignment 01, then migrate Assignments 01–11.
- Separate student starter files, public smoke tests, centrally managed grader files, reference solutions, and instructor records.
- Replace all legacy `.github/workflows/classroom.yml` files and course-specific `ds217_25f_*` fetches.
- Centralize assignment launch URLs instead of embedding them in lecture prose.
- Validate acceptance, normal `git push` submission, optional `gh student submit`, grading, feedback, resubmission, score collection, and regrading.
- Test starter, correct, partially correct, malformed, and dependency-failure submissions as applicable.

### 6. Certify Colab demos for Lectures 04–11

- Apply one notebook header, setup-cell, data, and verification pattern.
- Make small data and generators portable without manual uploads or Drive mounting.
- Run each notebook from a fresh Colab runtime and a clean local Jupyter environment.
- Add production Colab links only after the repository and release reference are settled.

### 7. Run course-wide quality assurance

- Execute all required demos in their documented environments.
- Run every assignment validation fixture through its local and Classroom 50 grader paths.
- Build the site and check navigation, links, media, and accessibility.
- Run an instructor and test-student walkthrough across terminal setup, Jupyter introduction, Colab demos, assignment acceptance, submission, feedback, and recovery.

## Classroom 50 plan

Assignment 04 is the pilot, but Classroom 50 is the submission and grading system for the entire course. The pilot choice does not limit rollout to notebook-based lectures.

### Migration groups

1. Assignment 01: recover or rebuild the terminal setup/readiness assignment and correct its obsolete grader design.
2. Assignments 02–03: terminal/script assignments; prefer declarative `run` or `python` tests where sufficient.
3. Assignments 04 and 06–09: notebook/output assignments that can reuse the pilot pattern.
4. Assignment 10: modeling dependencies and data acquisition with a separately pinned runtime.
5. Assignments 05 and 11: mixed or multi-notebook assessments requiring the most custom and manual grading review.

### Target architecture

| Layer | Contains | Must not contain |
|---|---|---|
| Student starter template | Instructions, starter files, dependencies, small data/generator, public smoke tests | Solutions, credentials, instructor records |
| Classroom 50 config repository | Assignment definitions, runtime settings, non-secret grader tests/fixtures, custom autograders, score collection | Solutions, credentials in published files, confidential tests, student work |
| Course site/data | Current assignment launch URLs and student workflow | Grader logic and retired assignment links in lecture prose |
| Instructor-only storage | Reference solutions, confidential validation, grading records, recovery notes | Student-facing links or published grader assets |

Classroom 50's configuration repository provides centralized control, not grader secrecy. Its published bundles are accessible without authentication. Use it only where an assignment remains valid if students inspect the tests and grader logic. Keep solutions and confidential validation outside the published bundle and use a different grading path if hidden tests are required.

Use declarative Classroom 50 tests when simple run or pytest checks express the rubric. Use a per-assignment `autograder.py` for notebooks, generated outputs, weighted partial credit, or custom failure handling. In both cases, emit the `classroom50/result/v1` result and verify the collected score history.

The detailed pilot is in [classroom50_pilot.md](classroom50_pilot.md). See the [Classroom 50 autograder documentation](https://github.com/foundation50/classroom50/wiki/Autograders) for its result contract and public Pages fetch model.

## Colab plan

- Lectures 01–03: no Colab badges, notebook wrappers, or notebook assignment variants.
- Lecture 04: explicitly teach Jupyter concepts, notebook state, execution order, saving, and portable paths.
- Lectures 04–11 demos: make **Open in Colab** the primary launch action for compatible notebooks, followed by local Jupyter instructions.
- Lectures 04–11 assignments: use notebooks where appropriate; Colab can be supported after its save-to-repository and submission workflow is validated.
- Shell, Git, SSH, tmux, and environment demonstrations remain in a real terminal even after Lecture 04 when those tools are the learning objective.

Each certified notebook must run top-to-bottom in a fresh Colab runtime and a clean local environment without credentials, Drive mounting, or manual uploads. Record package versions, data source, expected outputs, warnings, and validation evidence. Published notebooks and small data should use an immutable course release reference.

The detailed notebook contract is in [colab_standard.md](colab_standard.md).

## Acceptance criteria

### Lectures and materials

- Every lecture has measurable objectives and explicit prerequisites.
- The complete topic sequence has no unexplained prerequisite gaps.
- Core and bonus material are clearly separated.
- Examples use current, non-deprecated APIs.
- Links, media, accessibility, and licensing have been reviewed.

### Demos

- Every required demo identifies its environment, dependencies, data setup, expected result, and instructor troubleshooting guidance.
- Lectures 01–03 demos execute from the terminal with no notebook dependency.
- Compatible Lectures 04–11 demos run in both Colab and local Jupyter.
- Environment-specific demos clearly explain why they remain outside Colab.

### Assignments

- Assignment 01 has an explicit restore, redesign, or removal decision supported by the Lecture 01 objectives.
- Every assignment is reviewed for instructions, starter state, objective alignment, rubric, dependencies, grading, and recovery behavior.
- Every active assignment is distributed and graded through Classroom 50, regardless of terminal or notebook format.
- Grading is deterministic and tested against known fixtures.
- Grading does not depend on published Classroom 50 tests or fixtures remaining secret.
- Stored notebook outputs are not execution evidence. The grader executes a fresh copy; sensitive outputs are cleared, and ordinary outputs are retained only when a human rubric needs them. Generated files under `output/` remain a separate artifact contract.
- No active GitHub Classroom links, legacy workflows, or mutable `ds217_25f_*` grader fetches remain.

### Site and operations

- The Eleventy site builds successfully.
- Course navigation, assignment links, and Colab links point to the current release.
- Instructor and student workflows have been dry-run end to end.
- Historical announcements and obsolete plans are clearly archived or removed from current guidance.

## Decisions to record

1. **Resolved:** Assignment 01 was redesigned as a terminal readiness assignment and independently verified; it was not restored unchanged.
2. Source-of-truth syllabus and the intended role of the Lecture 05 and Lecture 11 assessments.
3. Supported Python minor version and package support policy.
4. Production GitHub organization, Classroom 50 classroom name, and staff access.
5. Individual/group mode, roster source, late-submission behavior, and authoritative submission rule.
6. Official grade-system import format and record-retention policy.
7. Whether discoverable Classroom 50 graders are acceptable for every assessment.
8. Whether Colab is supported for assignments or used only as the default for compatible demos.
9. Canonical repository and immutable course-release convention.
