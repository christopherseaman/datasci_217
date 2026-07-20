# 2026–27 curriculum audit

This is the working audit, not student-facing course material. Status values are **keep**, **revise**, **reduce**, or **investigate**.

## Baseline findings

- The repository has 11 lecture directories and current assignment directories for Lectures 02–11. Assignment 01 exists in repository history and remains referenced by an obsolete root Classroom link; it needs an explicit restore, redesign, or retirement decision during the Lecture 01 review.
- All ten assignment directories contain legacy GitHub Classroom workflows.
- There are 52 notebooks across demos and assignments; none has yet been certified for a clean Colab runtime.
- Dependency floors vary substantially between lectures and are mostly unbounded above.
- The root README contains only three historical Classroom acceptance links, while Lecture 06 contains another active-looking link.
- Several pages contain Notion links, `attachment:` image URLs, unresolved FIXME markers, and course-specific `25f` upstream grading repositories.
- Lecture 05 is labeled as a midterm and Lecture 11 as a final exam; assessment policy must be confirmed in the syllabus/source of truth.
- Lecture 10 attempts statsmodels, scikit-learn, XGBoost, TensorFlow/Keras, and PyTorch in one lecture. This is the clearest scope-reduction candidate.
- The working tree already contained an unrelated untracked `src/` directory when this audit began; this refresh does not modify it.

## Lecture-by-lecture review matrix

| # | Current focus | Initial status | Priority review actions |
|---|---|---|---|
| 01 | Setup, shell, Python, VS Code, GitHub | revise | Remove Notion/attachment artifacts; clarify WSL vs native shells; update Python installation; define a short local readiness check; replace GitHub Classroom language. |
| 02 | VS Code, Git, CLI, functions/modules | revise | Check scope and prerequisite overlap with Lecture 01; simplify the assignment; retain local execution as primary. |
| 03 | Environments, Python topics, NumPy | revise | Choose one primary environment workflow; make `python`/`python3` commands consistent; verify NumPy 2.x behavior and performance demos. |
| 04 | Jupyter, pandas structures, data I/O | revise | Use as the Colab and Classroom 50 pilot; add notebook-state guidance and Colab-safe data paths; resolve screenshot FIXMEs. |
| 05 | Cleaning and preparation; midterm | revise | Separate lecture material from assessment logistics; update missing-data APIs; reassess 100-point multi-file midterm size and generated artifacts. |
| 06 | Merge, concat, reshape | revise | Remove active Classroom link and attachment URL; verify merge validation/cardinality instruction; Colab-test all three demos. |
| 07 | Visualization ecosystem and principles | reduce | Focus required content on matplotlib/seaborn/pandas; move the larger library survey to bonus; test image accessibility and headless rendering. |
| 08 | GroupBy, pivot, remote workflows | revise | Decide whether SSH/tmux/performance belongs here or in a bonus/local lab; keep aggregation outcomes central. |
| 09 | Time series | keep/revise | Retain sequence; check pandas frequency aliases, timezone examples, data sizes, and notebook runtimes. |
| 10 | Statistics through deep learning | reduce | Make statistical modeling and scikit-learn the core; treat XGBoost and deep learning as survey or bonus material. |
| 11 | Full workflow and final exam | investigate | Check dataset availability, download reliability, runtime, licensing, assessment load, and whether nine notebooks are necessary. |

## Cross-course issues to resolve

### Curriculum alignment

- Add a common front-matter block or template for objectives, prerequisites, demos, assignment, and bonus material.
- Map every assignment item to a lecture objective and rubric row.
- Distinguish instruction from assessment administration; due dates and acceptance links should come from course data rather than lecture prose.

### Dependencies

- Select a tested Python minor version rather than promising “3.12+”.
- Create a shared constraints strategy for core packages while allowing lecture-specific requirements.
- Test pandas 3.x readiness before permitting it; existing text already anticipates removed APIs.
- Resolve inconsistent floors such as pandas 1.3/1.5/2.0 and Jupytext 1.13–1.16.
- Verify TensorFlow availability before retaining it in the required Lecture 10 environment.

### Repository hygiene

- Replace `attachment:` URLs with committed media paths.
- Remove or archive historical acceptance URLs and `25f` workflow references.
- Convert `FIXME.md` into tracked issues or an archive after triage.
- Decide whether generated reports, large CSV outputs, rendered plots, and paired `.md` notebook exports belong in source control.
- Consolidate duplicated root content between `README.md`, `index.md`, and older planning files.

## Review rubric for the detailed pass

Score every lecture from 0–2 on each item: absent, partial, or ready.

1. Objectives are measurable.
2. Prerequisites are explicit and previously taught.
3. Core content is necessary and appropriately scoped for the lecture's role.
4. Demo directly supports the objectives.
5. Assignment directly supports the objectives.
6. Commands and APIs are current.
7. Data and outputs are reproducible.
8. Local and Colab boundaries are clear.
9. Accessibility basics are present.
10. Instructor troubleshooting notes exist.

## Detailed pass 1: structural scorecard

Scoring uses the rubric above. These are initial evidence-based scores, not final pedagogical judgments. A total below 14 is a release blocker; 14–17 requires revision; 18–20 is ready after execution testing.

| # | Objectives | Prereqs | Scope | Demo | Assignment | Current APIs | Reproducible | Environment | Accessibility | Instructor notes | Total | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | 0 | 1 | 1 | 2 | 0 | 1 | 1 | 1 | 0 | 1 | 8 | blocker |
| 02 | 0 | 1 | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 11 | blocker |
| 03 | 0 | 1 | 1 | 2 | 2 | 1 | 1 | 2 | 1 | 2 | 13 | blocker |
| 04 | 0 | 2 | 1 | 2 | 2 | 1 | 1 | 1 | 0 | 2 | 12 | blocker |
| 05 | 0 | 2 | 1 | 2 | 2 | 1 | 1 | 1 | 1 | 2 | 13 | blocker |
| 06 | 2 | 2 | 1 | 2 | 2 | 1 | 1 | 1 | 0 | 2 | 14 | revise |
| 07 | 0 | 2 | 0 | 2 | 2 | 1 | 1 | 1 | 1 | 2 | 12 | blocker |
| 08 | 0 | 2 | 0 | 2 | 2 | 1 | 1 | 1 | 1 | 2 | 12 | blocker |
| 09 | 2 | 2 | 1 | 2 | 2 | 1 | 1 | 1 | 1 | 2 | 15 | revise |
| 10 | 0 | 2 | 0 | 2 | 2 | 1 | 0 | 1 | 1 | 2 | 11 | blocker |
| 11 | 2 | 2 | 1 | 2 | 2 | 1 | 0 | 1 | 1 | 1 | 13 | blocker |

### Scorecard interpretation

- The largest systematic gap is the absence of concise, measurable objectives in the student-facing lecture source. Only Lectures 06, 09, and 11 expose clear objective sections near the core material.
- Demo coverage is strong structurally: each lecture has a guide and executable artifacts. Scores remain provisional until clean-runtime execution.
- Assignment alignment is plausible from titles and task descriptions, but formal objective-to-rubric mapping is absent.
- Accessibility scores are low where images use weak filenames/alt text, remote-only images, or broken `attachment:` URLs.
- Environment scores distinguish documented local setup from a tested, supported dual Colab/local path; no notebook has passed that certification yet.

## Detailed pass 1: inventory

| # | Lecture lines | Demo files | Demo notebooks | Assignment files | Assignment notebooks | Legacy grading workflow | Unresolved content markers* |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | 807 | 6 | 0 | 0 | 0 | no | 7 |
| 02 | 894 | 4 | 0 | 6 | 0 | yes | due-date placeholders |
| 03 | 590 | 9 | 0 | 9 | 0 | yes | none outside teaching scaffolds |
| 04 | 723 | 8 | 3 | 8 | 2 | yes | 6 lecture FIXMEs |
| 05 | 812 | 7 | 3 | 38 | 4 | yes | midterm URL FIXME |
| 06 | 979 | 8 | 3 | 9 | 2 | yes | 12 attachment URLs |
| 07 | 844 | 20 | 3 | 9 | 2 | yes | none outside teaching scaffolds |
| 08 | 543 | 9 | 3 | 11 | 2 | yes | 2 media FIXMEs |
| 09 | 697 | 9 | 4 | 13 | 4 | yes | none outside teaching scaffolds |
| 10 | 891 | 8 | 3 | 7 | 1 | yes | none outside teaching scaffolds |
| 11 | 221 | 12 | 4 | 49 | 9 | yes | data/runtime review required |

\* `TODO` markers intentionally embedded in student scaffolds are not counted as defects.

## Prioritized issue register

### P0: must resolve before the course opens

| ID | Scope | Finding | Required disposition |
|---|---|---|---|
| P0-01 | Assignments 01–11 | Assignment 01 is absent from the current tree, while Assignments 02–11 depend on legacy GitHub Classroom workflows and course-specific `ds217_25f_*` grading sources. | Review and resolve Assignment 01, complete the Classroom 50 pilot, then migrate every active course assignment. |
| P0-02 | Root/06 | Historical and active-looking Classroom acceptance links are embedded in source. | Move assignment URLs to centralized course data; do not hard-code them in lecture prose. |
| P0-03 | 01/06 | Nineteen `attachment:` image URLs cannot work on the published site. | Map to committed media where available; replace or remove the rest. |
| P0-04 | 04 | Six explicit content FIXMEs remain in a core lecture. | Supply the visuals or rewrite the surrounding material so they are unnecessary. |
| P0-05 | 05 | The midterm link is literally `#FIXME:URL`. | Centralize assessment metadata and publish only after the production assignment exists. |
| P0-06 | 10 | Required environment includes TensorFlow alongside the full traditional modeling stack without a verified supported Python matrix. | Reduce core scope and verify package compatibility before freezing Python. |
| P0-07 | 11 | Final workflow relies on large external datasets/download scripts and has nine assignment notebooks. | Produce a deterministic reduced dataset, runtime budget, offline fallback, and grading plan. |
| P0-08 | Assignments 02–11 | Grading workflows fetch mutable tests from each upstream repository's `main` branch; action dependencies use movable major tags rather than commit SHAs. | Classroom 50 must centralize grader control, document that published grader bundles are discoverable, and pin the production grading supply chain. |

### P1: resolve during lecture revision

| ID | Scope | Finding | Required disposition |
|---|---|---|---|
| P1-01 | 01–11 | Objective formatting and measurability are inconsistent. | Add the standard lecture header and objective-to-assignment mapping. |
| P1-02 | 01–03 | `python` and `python3`, platform setup, venv, uv, and Conda guidance compete for attention. | Select one primary path per OS and make alternatives clearly optional. |
| P1-03 | 05–10 | Dependency minimums are inconsistent and have no shared upper-bound/constraints policy. | Establish tested course constraints plus minimal lecture requirements. |
| P1-04 | 07 | The required visualization ecosystem is too broad for one lecture. | Keep matplotlib, seaborn, and pandas plotting core; move Altair/Plotly/Bokeh/plotnine survey material to bonus. |
| P1-05 | 08 | Aggregation, remote shells, terminal multiplexers, and performance compete in one lecture. | Keep aggregation core; move remote-computing material to a separate local lab or bonus. |
| P1-06 | 09 | Time-zone and resampling material needs a current pandas frequency/deprecation pass. | Execute under the frozen pandas version and update aliases/examples. |
| P1-07 | 04–11 | Notebooks have not been certified for clean Colab and local execution. | Apply `work/colab_standard.md` and record validation evidence. |

### P2: repository maintenance

| ID | Scope | Finding | Required disposition |
|---|---|---|---|
| P2-01 | Root | `FIXME.md` mixes historical announcements with active maintenance notes. | Archive course announcements and move active work into this issue register. |
| P2-02 | Root/work | Multiple old plans describe repository states that no longer match reality. | Mark superseded plans as archival and link to `work/course_refresh_2026.md`. |
| P2-03 | 05/07/11 | Generated reports, plots, data, paired Markdown, and notebooks substantially inflate assignment packages. | Define which artifacts are source, generated, reference-only, or instructor-only. |
| P2-04 | Site | Root `README.md`, `index.md`, legacy HTML, and generated output overlap. | Confirm source-of-truth rules before content edits. |

## Audit execution status

- **Completed:** structural inventory, stale-platform scan, unresolved-marker scan, initial pedagogical scorecard, platform pilot design, Colab standards, notebook JSON parsing, baseline Eleventy build, parallel lecture review, detailed alignment matrices for Lectures 01–11, and the cross-lecture boundary reviews.
- **Runtime baseline:** Python 3.12.13 installed with uv; Node 24.18.0 installed with nvm; Eleventy 3.1.2 builds successfully.
- **Notebook metadata result:** all 52 notebooks parse as valid nbformat 4 JSON, but 23 notebooks lack kernelspec metadata and require normalization before Colab certification.
- **Site result:** the baseline build produced 28 pages and copied 124 assets without an Eleventy error. The refresh planning document was initially included because it was not ignored; it has now been added to `.eleventyignore`.
- **Next executable audit:** apply the documented GUI-first Git policy, finalize the canonical concept sequence, then build the objective→demo→assignment alignment matrix before editing course content.

## Assignment delivery audit

| Assignment | Legacy upstream | Local test material | Initial migration risk |
|---|---|---|---|
| 01 | removed from current tree | no current package | very high: must be recovered, redesigned, or intentionally retired |
| 02 | `ds217_25f_hw2` | yes | high: shell, Git, and platform-specific behavior |
| 03 | `ds217_25f_hw3` | yes | medium: scripts and deterministic NumPy outputs |
| 04 | `ds217_25f_hw04` | absent from student-facing inventory | medium: selected pilot; notebook plus generated outputs |
| 05 | `ds217_25f_midterm` | mixed local/instructor artifacts | very high: multi-file assessment with an inconsistent point total |
| 06 | `ds217_25f_hw06` | absent from student-facing inventory | medium: notebook/output grading and schema mismatches |
| 07 | `ds217_25f_hw07` | absent from student-facing inventory | medium: plot/image assertions can be brittle |
| 08 | `ds217_25f_hw08` | absent from student-facing inventory | medium: aggregation grain and many output artifacts |
| 09 | `ds217_25f_hw09` | absent from student-facing inventory | medium/high: multiple notebooks and time-zone behavior |
| 10 | `ds217_25f_hw10` | absent from student-facing inventory | high: network data, modeling dependencies, and evaluation leakage risk |
| 11 | `ds217_25f_final` | absent from student-facing inventory | very high: nine notebooks, many artifacts, and substantial manual evaluation |

The current workflows all use `actions/checkout@v4` and `actions/setup-python@v5`. Four also upload artifacts. None pins third-party actions to immutable commit SHAs. More importantly, each workflow downloads grading code from a mutable `main` branch. That permits central corrections, but it makes historical grading non-reproducible and couples student submissions to public upstream availability. Classroom 50's central bundle model is a better fit only if the config repository, release process, and grader versions are themselves retained and auditable.

Student-facing platform language also remains in Lecture 01, Assignment 08, and the final assignment. These references must be revised as part of migration rather than handled only at the root navigation level.

## Lectures 01–03 refresh record

Initial content revision completed on 2026-07-18.

### Completed

- Added measurable learning objectives, prerequisites/before-class guidance, and local-versus-Colab boundaries.
- Established the pre-Jupyter rule: Lectures 01–03 use only Python scripts and terminal workflows for demos and assignments.
- Replaced all seven broken Lecture 01 `attachment:` image URLs with committed media paths and useful alt text.
- Removed the stale Lecture 01 Notion/DLC entry point.
- Replaced student-facing GitHub Classroom language in the Lecture 01 setup demo with platform-neutral assignment-repository language.
- Centralized Assignment 02 deadline and late-policy language on the current course assignment page rather than leaving instructor placeholders.
- Made uv with Python 3.12 the primary Lecture 03 environment workflow, retained standard-library `venv` as the fallback, and labeled Conda optional.
- Updated the Conda example from Python 3.11 to the course's Python 3.12 baseline.

### Validation

- All Python files under Lectures 01–03 compile under Python 3.12.13.
- All ten Python demo scripts in Lectures 01–03 now execute successfully in isolated copies under Python 3.12.13; the NumPy demos use the Lecture 03 requirement.
- Both shell demo files pass `bash -n` syntax validation.
- The structural audit reports zero errors.
- Eleventy builds successfully after the changes.
- No student-facing `attachment:`, GitHub Classroom, Classroom acceptance, unresolved due-date placeholder, or Python 3.11 reference remains in Lectures 01–03 outside the legacy grading workflow files.

### Deferred to the Classroom 50 migration

- Assignment 02 and 03 still contain legacy `.github/workflows/classroom.yml` files. Removing them before the Classroom 50 pilot would leave no remote grader, so they remain explicitly tracked as P0 migration work.
- Full demo and assignment execution awaits isolated dependency environments and known-correct reference submissions; syntax validation alone does not certify teaching output.

### Execution defects corrected

- Lecture 02's module demo imported `python_functions_demo`, but the source file was named `03_python_functions_demo.py`; renamed the source module and updated references so the import works normally without advanced import machinery.
- Lecture 03's first two data-analysis demos expected a nonexistent `data/students.csv` path. They now locate the committed `students.csv` relative to the script and write reports under the demo directory regardless of the caller's working directory.
- Added concise instructor quick-run sections to all three demo guides, including the required order, environment commands, and the pre-Jupyter constraint.
- Assignment 02 exposes 17 pytest cases and Assignment 03 exposes 14; both suites collect under Python 3.12. Full scoring validation still requires known-correct and intentionally incomplete fixtures.

## Parallel pedagogical review synthesis

Review completed on 2026-07-18 using the process in `lecture_review_workflow.md`.

| Review lane | Coverage | Boundary follow-up |
|---|---|---|
| Terminal foundations | Lectures 01–03, demos, and assignments | Lecture 03 → 04 |
| Notebook/pandas foundation | Lectures 04–07, demos, and assignments | Lecture 07 → 08 |
| Advanced analysis and integration | Lectures 08–11, demos, and assignments | Lecture 08 → 09 → 10 → 11 |
| Course synthesis | concept homes, first-use order, prerequisite edges, scope, and assessment alignment | complete-course pass |

No lecture content was edited during this evidence-gathering pass.

### Course-wide conclusions

1. **Apply the documented Git policy when rewriting Lectures 01–02.** The recorded course design makes Git GUI-first and reserves command-line Git for bonus material, while the current Lecture 02 objectives and Assignment 02 assess command-line Git, branches, and shell behavior. Terminal-first Python does not make CLI Git part of the required path.
2. **Adopt a strict foundational spine.** Lectures 01–05 should establish Python execution, functions/modules, environments/NumPy, Jupyter/pandas, and cleaning. Several current demos and assignments jump ahead of this sequence.
3. **Give each major concept one canonical home.** Cleaning belongs in Lecture 05, merge and structural `pivot`/`melt` reshape in 06, visualization in 07, GroupBy and aggregating `pivot_table` in 08, time series in 09, modeling/evaluation in 10, and integrated workflow in 11. Earlier appearances may preview or consume supplied output, but must not require independent mastery.
4. **Define data grain before operations that change it.** Join cardinality, aggregation unit, time-series panel structure, prediction target, and prediction horizon are recurring missing prerequisites.
5. **Reduce survey breadth.** Advanced shell tricks, broad visualization-library tours, remote-computing tools, advanced MultiIndex/apply patterns, boosting, and deep learning should move to bonus unless they are made explicit course outcomes with adequate prerequisites.
6. **Rebuild demos after the concept sequence is accepted.** Each lecture should converge on two or three required demos. Several current demo guides duplicate earlier lectures, solve assignments, or introduce more concepts than the lecture can support.
7. **Rebuild assignments against an explicit alignment matrix.** There are repeated cases of untaught-but-assessed material, mismatched generated schemas, requirements not checked by tests, and tests that can be passed without demonstrating the intended competency.

### Recommended course spine

| Lecture | Canonical core role | Important scope boundary |
|---|---|---|
| 01 | Terminal orientation; run a Python script; scalar values, lists, conditions, and simple loops | Do not require functions, exceptions, recursion, file formats, or Git mastery. |
| 02 | Supported GUI Git workflow; functions; local modules; safe script entry point | Keep command-line Git in bonus material and teach the required workflow through VS Code Source Control or GitHub Desktop. |
| 03 | One reproducible environment workflow; dependency vocabulary; NumPy arrays, indexing, masks, reductions, shape, and axis | Keep Jupyter out; teach only bounded shell pipelines and one broadcasting/combining example if they remain objectives. |
| 04 | Notebook, cell, kernel/runtime, state and execution order; pandas Series/DataFrame; selection and portable I/O | Introduce Jupyter here; defer cleaning and aggregation to their own lectures. |
| 05 | Detect, handle, validate, and document data cleaning in one reproducible pipeline | Avoid aggregation-heavy work and modeling encodings as core requirements. |
| 06 | Row grain, keys, join cardinality, validated merge, concatenate, and wide/long reshape | Defer aggregation, resampling, and advanced MultiIndex work. |
| 07 | Question/audience/claim; honest and accessible charts; Matplotlib Figure/Axes; focused seaborn use; export | Move the multi-library survey, regression plots, rolling, and resampling to bonus or later lectures. |
| 08 | Unit of analysis; grouping key; split–apply–combine; named aggregation; output shape; transform; one aggregating `pivot_table` | Remote shells, tmux, performance, advanced apply, and advanced MultiIndex are not part of the core aggregation path. |
| 09 | Timestamp versus period; regular/irregular and single/panel series; datetime index; frequency; `asfreq`/resample; lag/difference; observation-count and elapsed-time trailing windows; information availability | Make missingness, entity boundaries, grouping semantics, and chronological holdout logic explicit before operations; keep EWM and non-trailing window variants in bonus. |
| 10 | Descriptive, inferential, and predictive questions; association versus causation; bounded OLS assumptions/uncertainty; target/horizon/availability; train/validation/test; baseline and one linear Pipeline; supplied classification metrics | Treat nonlinear models, boosting, and deep learning as enrichment; do not reuse test data for model selection. |
| 11 | One end-to-end question with a frozen entity/grain/key, target, horizon, availability cutoff, baseline, immutable licensed release, evaluation, and communication | Integrate prior skills rather than add new model families or a large artifact checklist; final demos/assignment wait for the exact project and dataset decision. |

### P0 findings by lecture

| Lecture | Finding | Required disposition |
|---|---|---|
| 01 | Objectives name collections/functions, but the lecture does not define functions; demos use functions, exceptions, comprehensions, file I/O, lambdas, sorting, and recursion before instruction. | Narrow the objectives and required demos to terminal/Python foundations; move later concepts to their canonical lectures. |
| 01 | The current Assignment 01 package is absent; the historical version required Git before Lecture 02 and used an unrelated historical grader. | Redesign a small readiness assignment or record an explicit retirement decision; do not restore it unchanged. |
| 02 | Core CLI Git conflicts with the recorded GUI-first Git policy. Functions appear after a long Python recap, module coverage is incomplete, and the `__main__` example is malformed. | Move CLI Git to bonus and restructure the required path around one GUI Git mental model followed by functions/modules. |
| 02 | Assignment instructions, repository location, tests, expected Git history, signatures, and modularity requirements conflict. | Rebuild as one coherent repository task; grade history only if Classroom 50 preserves the required evidence. |
| 03 | Broadcasting/combining are objectives but only bonus; CLI processing is treated as core without an objective; structured arrays are assessed but only taught in bonus. | Align objectives, core content, demos, and assignment around one bounded NumPy progression. |
| 03 | The demo guide is dominated by an Assignment 02 solution and repeated earlier material. | Replace with environment recreation, small-array concepts, and one applied terminal NumPy script. |
| 04 | Notebook state is named but not demonstrated through a concrete stale-state failure; the lecture reaches into cleaning and GroupBy before Lectures 05 and 08. | Teach state/restart/run-all directly and narrow pandas work to structures, selection, and portable I/O. |
| 04 | Assignment generator/path behavior, requested outputs, tests, and notebook-output instructions disagree. | Rebuild the assignment contract and validate it in both Colab and local Jupyter after content approval. |
| 05 | The assignment totals 125 despite presenting a 100-point assessment, and its notebooks reload raw data rather than forming the promised pipeline. | Correct the rubric and make the work a real chained cleaning pipeline. |
| 05 | Forward-filling cross-sectional rows and generic drop thresholds are presented without domain/grain safeguards. | Teach missing-data choices through data meaning and validation invariants rather than universal rules. |
| 06 | The first merge example is not an executable teaching example; cardinality omits `validate=` and unmatched-row inspection is underemphasized. | Make grain, key uniqueness, cardinality, `validate=`, and `indicator=True` the opening merge sequence. |
| 06 | Assignment schemas disagree with generated data; required artifacts and tests disagree; test data do not exercise orphan or duplicate keys. | Regenerate fixtures and tests around the actual join/reshape invariants. |
| 07 | Required scope spans too many plotting libraries, while assignment prompts, required outputs, and tests disagree. | Keep a focused Matplotlib/seaborn core and use a human rubric for communicative quality. |
| 08 | The `GroupBy.apply` compatibility explanation is backwards, and the assignment aggregates after a merge without defining the post-join unit of analysis. | Correct the API guidance; define grain before aggregation; prefer basic/named aggregation and `transform` in core. |
| 09 | Several examples create a Series and then assign named “columns,” so the examples are not valid as written. | Repair examples and establish single-series versus panel semantics before shift, rolling, and resampling. |
| 10 | Core material uses causal language without assumptions and mislabels a mean-response confidence interval as a prediction interval. | Define inference terms and limits before interpretation; correct uncertainty examples. |
| 10 | XGBoost uses the test set for early stopping and then reports that same test result. | Use train/validation/test separation and keep the untouched test set for final evaluation. |
| 11 | Cleaning occurs before station/time ordering, global forward-fill crosses stations, and observation-count windows are described as clock-time windows on irregular data. | Sort and clean within entity, then use a regular grid or offset-based windows with explicit semantics. |
| 11 | Feature work begins before defining prediction question, target timestamp, horizon, and information availability. | Define the prediction contract before feature engineering or splitting. |

### First-definition and first-use priorities

| Term or capability | Required home/definition | First independent use rule |
|---|---|---|
| path, working directory, relative path | Lecture 01 | Before globbing, file reads, or multi-file scripts. |
| function, parameter, argument, return value, local variable | Lecture 02 | Before students write helpers or import a local utility module. |
| module, import, main guard | Lecture 02 | Before Lecture 03 analysis scripts depend on reusable code. |
| environment, interpreter, package, dependency, direct/transitive dependency | Lecture 03 | Before students reproduce a NumPy environment. |
| ndarray, dtype, dimension, shape, axis | Lecture 03 | Before masks, reductions, reshape, or pandas comparisons. |
| notebook, cell, kernel/runtime, state, execution order | Lecture 04 | Before students are asked to diagnose, restart, or run a notebook. |
| Series, DataFrame, index, column, row, label versus position | Lecture 04 | Before unrestricted pandas selection or transformation. |
| row grain/unit of observation, key, cardinality | Lecture 06 | Before merging and before any post-merge aggregation. |
| Figure, Axes, encoding, data ink, accessibility | Lecture 07 | Before students design and critique explanatory charts. |
| grouping key, split–apply–combine, aggregation versus transform, MultiIndex | Lecture 08 | Before custom aggregation, `pivot_table` output, or index manipulation. |
| timestamp, period, frequency, regular/irregular series, panel | Lecture 09 | Before resampling, shifting, rolling, or pooling entities. |
| inference, estimand, association, causation, residual, confidence interval, prediction interval | Lecture 10 | Before coefficient or model-effect interpretation. |
| target, feature, prediction horizon, information availability, leakage, baseline | Lecture 10 and reinforced in 11 | Before feature engineering, temporal splitting, or model comparison. |

### Boundary contracts

- **Lecture 03 → 04:** Lecture 03 hands off scripts, imports, paths, environments, and ndarray shape/axis/dtype/indexing. Lecture 04 must define notebook/cell/kernel/runtime/state/order before free use, then explicitly bridge 1D arrays to Series and 2D arrays to DataFrames. Lecture 03 remains notebook-free.
- **Lecture 07 → 08:** Lecture 07 hands off question/audience/claim, prepared long-form data, Figure/Axes, and export. Lecture 08 begins with unit of analysis and grouping keys. Correlation, regression, rolling, and monthly resampling leave the Lecture 07 assignment.
- **Lecture 08 → 09:** Lecture 08 establishes result grain, output shape, aggregation versus transform, and index consequences. Lecture 09 adds time-index semantics rather than using period/month grouping as unexplained machinery in Lecture 08 demos.
- **Lecture 09 → 10:** Lecture 09 establishes single versus panel data, chronological order, temporal splitting, and information availability. Lecture 10 formalizes target/features, train/validation/test, baseline, evaluation, and leakage.
- **Lecture 10 → 11:** Lecture 10 hands off a complete modeling vocabulary and valid evaluation pattern. Lecture 11 applies it to a pinned dataset with entity-aware cleaning and a stated prediction horizon rather than adding another model survey.

### Ordered change set

1. Apply GUI-first Git to the required Lecture 02 path and move command-line Git to bonus.
2. Finalize the recommended course spine and canonical concept homes.
3. Build the complete prerequisite graph and objective→lecture→demo→assignment matrix from that approved spine.
4. Rewrite lecture narratives and bonus boundaries in course order, preserving the Lecture 03→04 execution boundary.
5. Reduce each lecture to two or three required demos and rewrite them against the accepted concept sequence.
6. Redesign Assignment 01 and repair Assignments 02–11 so every assessed capability has prior instruction and guided practice.
7. Only then package every assignment for Classroom 50 and certify compatible Lecture 04–11 demos for Colab/local Jupyter.
8. Run the independent boundary, execution, grader-fixture, site, link, media, and accessibility verification passes.

## Lecture 01 narrative revision

Core and bonus content revised on 2026-07-18 from the accepted Lecture 01 matrix in `work/reviews/lectures_01_03_alignment.md`.

### Content changes

- Reduced the core narrative from 825 to 496 lines while adding plain-language definitions for terminal, shell, command, prompt, working directory, absolute/relative path, scalar, list, element, index, condition, loop, traceback, and exception type before independent use.
- Declared one POSIX-style command subset through Bash on Linux/WSL/supported cloud or default zsh on macOS; native PowerShell is a setup bridge rather than a second assessed interface.
- Separated GitHub account/privacy readiness from Git mastery. Repositories, staging, commits, branches, and synchronization now begin in Lecture 02.
- Removed student-defined functions, exceptions, file formats, `pathlib`, comprehensions, lambdas, recursion, advanced shell expansion, unsupported professional claims, and course-calendar material from the required narrative.
- Added an explicit list/element/zero-based-index section before loops and membership-dependent reasoning.
- Replaced simulated error prose with a real traceback-reading workflow that identifies the exception, source line, correction, and rerun.
- Added `01/BONUS.md` for shell globbing/brace expansion/command substitution/search, the Python prompt, advanced f-string formats, and interactive input.
- Retained exactly three required demo callouts matching the approved demo contracts; the matching artifacts are recorded below.

### Validation

- `git diff --check` passes.
- The dependency-free course audit parses all 52 notebooks and reports zero structural errors. Existing Classroom, kernelspec, and Lecture 06 attachment warnings remain tracked and are unrelated to the Lecture 01 narrative change.
- Both referenced Lecture 01 media files exist.
- Independent post-edit pedagogical verification passed after one correction cycle. The final check parsed all Python fences with Python 3.12, syntax-checked all Bash fences, confirmed exact integrated output, verified the three-demo count, and found no remaining narrative blocker. Assignment 01 was handled as the separate downstream work item recorded below.

## Lecture 01 demo rebuild

The required demos were rebuilt on 2026-07-18 after the Lecture 01 narrative passed independent review.

### Artifact changes

- Replaced the former five-demo sequence with exactly three numbered terminal Python artifacts: first script and relative-path practice, values/lists/decision/direct-loop integration, and a real traceback fix-rerun sequence.
- Reduced the first script to the literal `print()` statement already introduced in the narrative; variables and f-strings begin in Demo 02 only after their lecture definitions.
- Made Demo 03 expose `IndentationError`, `NameError`, `TypeError`, `ValueError`, and `IndexError` one at a time as each preceding source defect is corrected.
- Replaced the long talking-points document with a concise command-and-output guide and removed the obsolete navigation, advanced-control-flow, and file-processing demo artifacts.
- Replaced the former GitHub/Git setup walkthrough with a readiness-only guide for Python, VS Code, Git installation, GitHub organization access, and a guided Classroom 50 repository-load check. Repository-change workflows remain in Lecture 02.
- Removed the unlinked legacy `01/bonus/advanced_topics.md`; `01/BONUS.md` is now the single optional extension and does not leak permissions, environment variables, imports, or other deferred material back into Lecture 01.

### Validation

- Both success scripts ran under managed CPython 3.12.13 and produced the documented output; the alternate Demo 02 list selected the documented lower branch and produced a mean of `6.0`.
- The Demo 01 working-directory sequence reproduced the expected missing-relative-path failure and successful corrected relative path.
- Demo 03 first failed syntax checking with the intended `IndentationError`; successive apply-patch corrections produced the four remaining real exception types, followed by the documented clean output. The source was then restored to its intentional `IndentationError`-first state.
- The final required artifact set contains three `.py` files and no notebooks or required shell scripts. Static inspection found no student-defined functions, Python file I/O, exception handling, dictionaries, comprehensions, `pathlib`, CSV/JSON processing, lambdas, recursion, advanced shell automation, or Git mastery.
- Independent demo verification reproduced both success paths, the alternate lower-threshold branch, and the complete five-exception correction sequence. Its one initial usability failure—VS Code still had the scratch folder open before Demos 02–03—was corrected by explicitly reopening the course repository, starting a new terminal, confirming `pwd`, and then entering `01/demo`; the verifier's final addendum is PASS.

## Lecture 02 narrative revision

The Lecture 02 core and bonus structure were rebuilt on 2026-07-18 after Lecture 01 narrative and demo verification.

### Content changes

- Made VS Code Source Control the single required Git interface; command-line Git and shell automation are explicitly optional bonus material and are not assumed by Lecture 03.
- Ordered the core as Git state model → one GUI edit/stage/commit/synchronize cycle → focused repository hygiene → GUI branch/merge/conflict → duplicated calculation → minimal dictionary → functions and edge cases → small text write/read-back → import-safe two-file program.
- Defined repository, working tree, diff, staging area, commit, branch, merge, conflict, remote, push, pull, synchronization, dictionary/key/value, function interface/implementation, parameter, argument, return value, local variable, docstring, driver script, side effect, module, top level, import statement, and main guard before independent use.
- Consolidated optional material into `advanced_git.md`, `bonus_python_concepts.md`, and `shell_automation.md`; removed the malformed redundant `advanced_python_cli.md`.
- Retained exactly three required live-demo contracts and an explicit Lecture 01 prerequisite/Lecture 03 handoff; no notebook or Colab workflow appears before Lecture 04.

### Validation

- Independent verification initially found four narrow failures: terms used before definition, one non-runnable isolated main-guard fence, missing text read-back, and a stale merge-menu location. Each was corrected without expanding scope.
- The final independent addendum is PASS: all 15 core and 13 bonus Python fences compile under Python 3.12.13; progressive examples and exact two-file output run; importing the driver is silent and creates no report; all 26 Bash fences pass syntax checking; links, media, the exact-three-demo count, and `git diff --check` pass.
- The verifier recorded one Lecture 03 follow-up: replace its required list comprehension with a plain loop rather than promoting comprehensions from Lecture 02 bonus into the core dependency chain.

## Assignment 01 rebuild

Assignment 01 was rebuilt on 2026-07-18 after the Lecture 01 narrative and three-demo set passed independent review.

### Artifact changes

- Added three incremental terminal-script tasks: dynamic readiness labels, a text-to-integer threshold plus zero-based indexing and a loop-and-decision measurement summary, and three prepared traceback corrections followed by a supplied fresh-execution output wrapper.
- Added named-file terminal-practice evidence without recursive operations. GitHub Desktop delivery is an exact, required, but unassessed platform checklist; Git concepts and independent synchronization remain in Lecture 02.
- Kept student-authored code within the Lecture 01 boundary: no functions, dictionaries, student-added imports, file I/O, exception handling, notebooks, Colab, third-party packages, shell pipes, redirection, or command-line Git.
- Added one standard-library checker used by `python check_assignment.py` and nine matching public pytest cases for Classroom 50. The checker fresh-executes scripts, varies top-level inputs in temporary copies, tests another working directory, enforces structural boundaries, and compares the stored artifact with fresh stdout rather than trusting it.
- Added instructor-only grader self-tests that materialize correct, partial, hard-coded, partial-slice-loop, dead-code/literal-output, dynamic-file-I/O, threshold-boundary, wrong-divisor, missing-loop, missing-else, forbidden-construct, label/rounding, stale-output, and broken-working-directory fixtures. Production-only fixtures remain external to the student starter and no discoverable test is represented as secret.

### Validation

- Managed CPython 3.12.13 accepted the correct fixture through all nine public pytest cases and the dependency-free checker.
- Independent verification initially found three false acceptances: processing the first item outside a sliced loop, preserving required debug expressions only in dead code while printing literal answers, and selecting file I/O dynamically through `getattr`. The public rules were hardened and disclosed in the README, and all three cases were added as permanent regressions.
- The final independent result is PASS: the correct solution passes the dependency-free checker and all nine public pytest cases under Python 3.12.13; all fourteen defective fixture scenarios are rejected through their intended actionable checks; the starter remains incomplete and `IndentationError`-first; no bytecode/cache artifact or trailing whitespace remains.
- The starter checker fails for the intended incomplete work, and `debug_report.py` remains restored to its intentional `IndentationError`-first state.
- The Classroom 50 remote acceptance/submission/grading loop remains an operational pilot item; this pass validates the assignment package and public grader contract, not the production platform configuration.

## Lecture 02 demo rebuild

The Lecture 02 demos were rebuilt on 2026-07-18 after the revised narrative and independent narrative verification passed.

### Artifact changes

- Replaced the former advanced shell demo and solution-scale Python programs with exactly three required demonstrations: one disposable VS Code GUI Git workflow, one duplication-to-functions refactor, and one import-safe two-file report program.
- Added a deterministic two-file Git seed. The guide creates both divergent commits through VS Code, resolves the prepared conflict to one exact final README line, and uses the shipped **Git: Merge...** Command Palette label. VS Code 1.86 replaced **Git: Merge Branch...** with that command, and the current built-in Git extension retains it: <https://code.visualstudio.com/updates/v1_86>, <https://github.com/microsoft/vscode/blob/main/extensions/git/package.nls.json>.
- Added a short duplicated calculation and a refactor using minimal dictionaries, `mean()`, `format_summary()`, one-sentence docstrings, explicit `None` behavior, and both numeric and no-measurements formatting paths.
- Added `analysis_utils.py` and a guarded `main.py` that overwrites, reads back, and prints one deterministic text report. Both import paths are silent and create no artifact.
- Removed old Python bytecode and all obsolete required demo files. The final set contains four `.py` files, two `.gitignore` files, one seed README, and one concise guide; it contains no shell script, notebook, dependency file, or generated output.

### Validation

- Managed CPython 3.12.13 compiled all four scripts and reproduced the exact documented output for both Demo 02 scripts.
- Direct checks passed for `mean([])`, ordinary and zero means, and the normal, zero, and empty formatting branches.
- A fresh temporary copy proved silent imports of both modules, no report on import, exact stdout/report bytes on direct execution, and identical overwrite behavior after replacing a stale report.
- A temporary Git repository reproduced both divergent commits, the intended merge conflict, the exact resolved README, the merge commit, and a clean final working tree.
- Static checks confirmed exactly three guide sections, exact local artifact references, and no command-line Git instruction, talking-point/timing language, path hacks, structured file formats, exception handling, comprehensions, generators, type annotations, timestamps, or other deferred scope.
- Independent verification initially found two stale GUI labels. The lecture and guide now use the shipped **Git: Merge...** and **Sync Changes** labels rather than the stale prose-documentation labels.
- The final independent demo result is PASS: all 28 Python fences compile under Python 3.12.13; exact outputs and ordinary/empty/zero cases pass; both imports are silent; direct execution overwrites a stale report with identical 70-byte stdout/file content; and a fresh Git rehearsal ends with six reachable commits, a two-parent merge, exact resolution, clean `main`, and matching local/remote refs.

## Lecture 03 narrative revision

The Lecture 03 core and bonus narrative were rebuilt on 2026-07-18 after the Lecture 01–02 prerequisite chain was independently verified.

### Content changes

- Reframed the core around exactly five measurable objectives: environment reproduction, one bounded shell pipeline, the NumPy array model and basic slicing, masks/vectorization/reductions, and reshape/transpose plus an import-safe terminal analysis.
- Established Python 3.12.13 and NumPy 2.0.2 as tested release candidates rather than the final course lock. The primary workflow uses uv, `.python-version`, `.venv`, and a deliberate `requirements.txt` containing only `numpy==2.0.2`; a concise standard-library `venv` fallback remains for an already-installed candidate interpreter.
- Defined interpreter, module, package, direct and transitive dependency, requirements file, lock artifact, environment, virtual environment, activation, standard input/output, pipe, pipeline, redirection, fixture, CSV, delimiter, field, header, ndarray, homogeneous, element, dimension, axis, shape, tuple, index, slice, view, copy, mask, vectorization, broadcasting, reduction, reshape, and transpose before independent use.
- Limited required shell processing to `head`, `tail`, `cut`, `sort`, `uniq`, `wc`, pipes, and overwrite redirection on a supplied comma-simple fixture. The narrative explicitly states that `cut` is not a general CSV parser.
- Limited core NumPy semantics to homogeneous 1D/2D arrays; `shape`, `ndim`, `size`, and `dtype`; positional indexing and basic slices; basic-slice views versus explicit copies; boolean masks; same-shape and scalar arithmetic; whole-array and axis means; compatible reshape; transpose; and the single scalar-to-1D broadcasting case.
- Clarified that basic slices are views but not all indexing is, an axis reduction removes the named dimension, and reshape returns a view when possible and a copy otherwise.
- Consolidated optional material into the single `03/BONUS.md` route. Advanced indexing, multidimensional broadcasting, concatenation/stacking, selected ufuncs, conditional selection, sorting/ranking, reproducible `np.random.default_rng()`, `np.isin()`, and structured-array recognition remain optional and are not prerequisites for Lecture 04.
- Retained exactly three required demo contracts: candidate-environment reproduction, the NumPy shape/view/copy mental model, and a bounded shell pipeline followed by a supplied-loader NumPy analysis.
- Made the Lecture 04 handoff explicit: a terminal script starts a fresh process while a notebook kernel retains state, and 1D/2D ndarrays provide the positional array models that pandas Series/DataFrame later add labels to. No notebook, Colab, or pandas API workflow is taught in Lecture 03.

### Validation

- All 14 core and 11 bonus Python fences compile under managed CPython 3.12.13. Progressive execution under a fresh environment containing exactly NumPy 2.0.2 reproduced all documented results, including the import-safe analysis's exact output; importing that analysis is silent.
- All 19 core Bash fences pass `bash -n`. The required `head`/`tail`/`cut`/`sort`/`uniq`/`wc` pipeline also executed successfully against the supplied `03/demo/students.csv`, producing eight subject groups that account for all 1,500 data rows.
- Static checks confirm exactly five objectives and three demo callouts, one valid local bonus link, no required comprehensions or exception handling, and no core `pathlib`, pandas APIs, structured arrays, legacy random API, `np.in1d`, advanced NumPy survey topics, `awk`, `sed`, `tr`, gnuplot, sparklines, Conda, downloaded installer pipeline, PowerShell workflow, notebook, or Colab execution.
- The cited Astral uv and NumPy 2.0 documentation pages were checked for exact-version environment commands, package installation, basic-slice view semantics, advanced-index copies, scalar broadcasting, reshape view/copy behavior, `default_rng`, and `np.isin`. The narrative has no local media dependency.
- `git diff --check` passes. The dependency-free course audit parses all 52 notebooks with zero errors; the 38 warnings are existing legacy Classroom, kernelspec, and Lecture 06 attachment follow-ups outside this narrative revision.
- Independent verification initially found two scope/order failures: the lone scalar broadcast appeared before reductions/reshape, and the integrated example accidentally broadcast a scalar over a 2D array. The broadcast section now follows reshape/transpose, and the integration reshapes to 1D before its scalar comparison; both demo contracts state that exact boundary.
- The final independent result is PASS: all 25 Python fences and 19 Bash fences execute or syntax-check under the candidate stack, the environment recreation/pipeline/import/output checks remain exact, objective/demo/scope counts pass, and five generated `.pyc` files in the stale demo directory were removed. No cache or bytecode artifact remains under `03/`.

### Downstream assignment status

- The legacy generated health-sensor assignment was subsequently replaced after the revised Lecture 03 narrative and demos were accepted. See **Assignment 03 rebuild** below for the implemented artifact and grader contract.

## Assignment 02 rebuild

Assignment 02 was rebuilt on 2026-07-18 after the revised Lecture 02 narrative and three-demo set passed independent review.

### Artifact changes

- Replaced the separate-repository, shell-scaffold, CSV/JSON grade-analysis project with one coherent Reusable Measurement Summary assignment in the provisioned Classroom 50 repository.
- Added three incremental parts: four short Git-state snapshots plus bounded README/`.gitignore` work, two reusable calculation/formatting functions, and an import-safe driver that overwrites and reads back one deterministic text report.
- Separated the required VS Code/GitHub Desktop branch-and-delivery workflow into `PLATFORM_CHECK.md`. It uses `feature/measurement-summary`, selective staging, focused documentation and Python commits, a GUI merge to `main`, and synchronization without forcing a conflict. Repository-history shape is not treated as Python grader evidence.
- Added dependency-free `check_assignment.py` and `_public_checks.py` plus nine matching public managed-pytest cases. Production grader tests remain independently implemented in the centrally managed Classroom 50 bundle and do not import an editable checker from a student repository.
- Removed the old GitHub Classroom workflow, remote-download tests, requirements files, TIPS document, shell scaffold, and broad student-grade data project. The starter has no notebook, shell script, dependency file, third-party package, or environment requirement.
- Added instructor-only `_grader_selftest/` validation with an explicit starter-packaging exclusion and no secrecy claim for published grader logic.
- Published the complete direct-call boundary in the student README: `len()` in `mean()`, `mean()` in `format_summary()`, and `format_summary()`/`open()`/`print()` plus the matching report-handle `write()`/`read()` calls in `main()`. Indirect or dynamically selected calls, dangerous built-ins, unapproved imports/calls, indirect file I/O, append mode, and extra opens are rejected as explicit course-scope violations.

### Validation

- Managed CPython 3.12.13 accepted the correct fixture through the dependency-free checker and all nine public pytest cases.
- Direct behavior checks cover ordinary, empty, zero, mixed-sign, decimal, repeated-call, nonmutation, and quiet-return cases. Per-run varied labels, records, numeric returns, and `None` returns prove that `format_summary()` calls `mean()` exactly once and builds each branch from that result rather than recognizing one published sentinel or recomputing the data.
- Fresh temporary subprocesses prove both imports are silent and artifact-free, direct execution has exact 97-byte stdout and exact 70-byte report content, stale content is overwritten, and report read-back participates in either the inline comparison or a separately assigned comparison printed afterward. Two varied formatter-result sets prove that the driver calls the imported formatter exactly once per supplied record in order and derives every report line from each returned value. The correct fixture also passes from a separate working directory.
- Independent verification initially exposed five false-acceptance categories and one false rejection: a dead loop plus mean lookup, a `mean()` call followed by duplicated formatting arithmetic, hard-coded handling of the one published formatter spy, a dummy mode-`w` open followed by the real append, indirect/dangerous call paths, and a compliant named read-back comparison. The AST and runtime checks were hardened without hiding any rule, and the named comparison now passes alongside the inline form.
- Thirty-four defective fixtures are rejected. The original eighteen cases remain, and sixteen permanent regressions cover dead-loop lookup; loop, `sum()`, and direct-arithmetic recomputation after a `mean()` call; `exec`/`eval`/`compile`/`__import__`; an indirect mean call; `__builtins__[...]`; an unapproved import; an unapproved direct call; dummy-write-plus-append; indirect file writing and reading; and published-spy special casing.
- The independent re-verification result is PASS: the compliant inline and named-comparison solutions pass, every original bypass is rejected, all nine public tests and all thirty-four defective fixtures behave as intended, and the README exposes every structural/call/open/read/write rule enforced by the checker.
- The intentionally incomplete starter fails eight actionable public checks and passes only the expected import-safety check; stored output is never trusted as evidence of current execution.
- All six Assignment 02 Python sources compile under Python 3.12.13, `git diff --check` passes, and no cache, bytecode, generated report, old workflow, requirements file, shell script, notebook, or structured-data artifact remains in the starter. The course audit parses all 52 notebooks with zero errors; its 38 warnings are existing later-lecture Classroom, kernelspec, and Lecture 06 attachment follow-ups.

## Lecture 03 demo rebuild

The Lecture 03 demos were rebuilt on 2026-07-18 after the revised narrative and three-demo contracts passed independent review.

### Artifact changes

- Replaced the stale assignment walkthrough, broad Python potpourri, performance survey, student-analysis solution, generated report, and 1,500-row dataset with exactly three required demonstrations: candidate-environment reproduction, the ndarray mental model, and a bounded shell pipeline followed by supplied-loader terminal analysis.
- Added exact candidate records for Python 3.12.13 and NumPy 2.0.2, a generated-artifact `.gitignore`, and an import-safe environment check. The guide uses uv as the primary route and includes a bounded standard-library `venv` fallback for an already-installed candidate interpreter.
- Added `ndarray_mental_model.py` with the approved sequence: metadata, positional selection, basic-slice view and explicit-copy behavior, one mask, same-shape arithmetic, whole/axis reductions, reshape/transpose, and one scalar-to-1D broadcast.
- Replaced the large student dataset with the exact six-row `observations.csv` fixture. `data_loader.py` alone owns CSV parsing and returns a homogeneous float64 `(6, 2)` ndarray; `array_summary.py` performs the approved reductions and reshapes to 1D before its scalar comparison.
- The final top-level set is exactly nine files: `.gitignore`, `.python-version`, `DEMO_GUIDE.md`, `array_summary.py`, `data_loader.py`, `environment_check.py`, `ndarray_mental_model.py`, `observations.csv`, and `requirements.txt`. It contains no notebook, shell script, generated output, environment, cache, or bytecode artifact.

### Validation

- A fresh temporary copy completed the primary uv workflow with managed CPython 3.12.13 and NumPy 2.0.2, then independently recreated the same environment from only `.python-version`, `requirements.txt`, and `environment_check.py`. A separate fresh copy also completed the documented standard-library fallback with the already-installed exact interpreter.
- All four Python sources compile. Imports are silent and artifact-free; the environment check, ndarray demonstration, and analysis match their documented stdout byte for byte; repeated analysis is identical; and all scripts run from a fresh copied directory.
- The supplied loader returns the exact float64 `(6, 2)` values. An alternate `(2, 3)` array produces the expected whole, column, and row means and review count; direct checks also confirm view mutation, copy independence, reduction shapes, and the bounded scalar-to-1D broadcast.
- The fixture has the exact schema and six data rows. The documented pipeline accounts for every row and normalizes to `3 north`, `2 south`, and `1 west`, with three output lines.
- Static checks confirm exactly three required demo sections; exactly 19 Bash fences, all passing `bash -n`; exclusive loader ownership of `csv.DictReader` and `open()`; and no comprehension, generator, exception handling, type annotation, path hack, pandas API, advanced NumPy survey, downloaded installer pipeline, or notebook execution.
- The official Astral uv and NumPy 2.0 documentation was checked for the exact interpreter-install/pin/environment/install commands and the indexing, view/copy, reduction, reshape, and scalar-broadcast semantics used by the demos. The linked uv installation page returns successfully.
- Independent read-only verification is PASS: it reproduced all three documented outputs byte for byte in a fresh copy, confirmed the exact fixture/pipeline/loader/alternate-array contracts and scope boundaries, and made no repository edits.
- `git diff --check` passes. The dependency-free course audit parses all 52 notebooks with zero errors; its 38 warnings are existing legacy Classroom, kernelspec, and Lecture 06 attachment follow-ups outside this demo rebuild.

## Assignment 03 rebuild

Assignment 03 was rebuilt on 2026-07-18 after the revised Lecture 03 narrative and three-demo sequence passed independent verification.

### Artifact changes

- Replaced the 50,000-row generated medical-data project, TIPS document, placeholder output, broad `awk` tasks, structured-array analysis, permissive dependency record, embedded GitHub Classroom workflow, and downloaded grader design with one coherent terminal NumPy assignment for the provisioned Classroom 50 repository.
- Added three incremental tasks: record and recreate the exact Python 3.12.13/NumPy 2.0.2 uv environment and save a fresh two-line probe; complete four exact bounded terminal commands over the pinned seven-line `observations.csv`; and implement seven directly testable homogeneous-array functions plus an import-safe eight-line terminal driver.
- Added an immutable supplied environment probe and CWD-safe `data_loader.py`. The loader alone owns `pathlib`/CSV parsing and returns the exact float64 `(6, 2)` ndarray used by the accepted demo contract. The student driver produces no stored report.
- Added `PLATFORM_CHECK.md` for the required but unassessed VS Code/GitHub Desktop branch, selective commits, GUI merge, synchronization, and Classroom 50 delivery. Activation/recreation and GUI operation are explicitly separate workflow evidence; the Python grader does not pretend to prove them. `.venv/` and `recreation-check/` are ignored recreation state and must never be submitted.
- Added a standard-library-plus-NumPy `check_assignment.py`/`_public_checks.py` implementation and ten matching public managed-pytest facade tests. The pipeline block is parsed and bounded before execution and is never sent to a shell. Production Classroom 50 tests remain independently implemented from the same written contract and do not trust an editable checker in the student repository.
- Published every enforced student-code boundary in the README: exact module docstrings, top-level layout/imports/signatures, no loops/comprehensions/generators/exceptions/path/file parsing or mutation, only the required NumPy calls/properties/indexing/arithmetic, no dynamic or indirect calls, and explicit exclusion of structured/random/fancy-index/sort/rank/stack/concatenate/multidimensional-broadcast/pandas/notebook/Colab work. The terminal task likewise excludes append/substitution/variables/compound commands/extra tools/CLI Git/`awk`/`sed`/`tr`.
- Added instructor-only `_grader_selftest/` material with an explicit starter-packaging exclusion and no secrecy claim for published grader logic.

### Validation

- A correct fixture under managed CPython 3.12.13 and NumPy 2.0.2 passes the dependency-free checker and all ten managed-pytest cases. The untouched starter fails the intended nine actionable checks and passes only quiet artifact-free imports.
- Runtime checks enforce the two exact records, immutable supplied files, exact saved probe, a matching fresh probe, and absence of a tracked `.venv` when Git metadata is available. The documentation states that these artifacts cannot prove local activation or recreation.
- The safe pipeline executor validates the exact four-command structure, runs tools directly without a shell, overwrites stale sentinels, and passes both the course fixture and a signed/decimal alternate fixture. Counts normalize to `3 north`, `2 south`, `1 west` for the base fixture; a malicious semicolon/extra-tool fixture is rejected without creating its marker.
- Fresh NumPy harnesses vary nonsquare shapes, signed values, decimals, integer dtypes, scalar thresholds at equality boundaries, offsets, and reshape dimensions. They verify float64 creation/metadata, basic selections, true view/copy memory relationships with no call-time mutation, 1D masks/selections/arithmetic, whole/axis reductions and result shapes, reshape/transpose, and the required reshape-to-1D whole mask count.
- Fresh subprocesses prove the three imports are silent and artifact-free in both project and alternate working directories. Loader/function spies exercise both a different-shape `(3, 2)` float32 result and a same-shape `(6, 2)` int16 result with different metadata, reductions, and count; they identity-check direct dataflow and verify one exact call order. The real driver then produces the exact eight output lines by absolute-path invocation from another working directory and creates no report or other artifact.
- Forty-two defective fixtures are rejected across wrong records/dependency/probe/stale evidence/tracked environment; blank/hard-coded/forbidden/append/missing-sort/header/malicious/base-only pipelines; partial/hard-coded/wrong-dtype/shape/loop/comprehension/view-as-copy/sharing-copy/mutation/mask/difference/broadcast/axis/reduction-shape/reshape/count defects; and hard-coded/ignored-result/wrong-order/top-level/import-artifact/CWD/output driver defects. Four permanent regressions additionally cover replacement of either required module docstring, a constant first selection in an otherwise correct dictionary, and same-shape special-casing in the driver.
- All eight Python files compile from source under the candidate interpreter. `git diff --check` passes, and the starter contains no legacy `.github` workflow/test directory, TIPS file, medical generator/analysis, placeholder, notebook, shell script, generated output, environment, recreation directory, cache, or bytecode artifact. The dependency-free course audit parses all 52 notebooks with zero errors; its 37 warnings are existing later-lecture Classroom, kernelspec, and Lecture 06 attachment follow-ups outside this assignment rebuild.

### Independent-QA correction cycle

- Independent QA found three remaining false-acceptance categories: `select_parts()` could substitute a constant while preserving the expected runtime dictionary, the single different-shape driver spy could be bypassed by special-casing the published `(6, 2)` shape, and the required module docstrings were not checked exactly.
- The structure check now requires the exact four-key selection dictionary to map directly and respectively to `values[0, 0]`, `values[1]`, `values[:, 1]`, and `values[:2, :2]`, and it compares both module docstrings to the published text.
- The driver dataflow check now runs both the different-shape spy and an independent same-shape spy whose dtype, metadata, reductions, shapes, and count all differ from the course fixture. Valid direct helper use passes both; shape-gated memorization fails the same-shape regression.
- The correct fixture passes all ten public checks and the local checker, the untouched starter still exposes nine actionable failures and one safe import, and all forty-two defective fixtures are rejected. Separate independent re-verification of this correction cycle remains pending.

## Lecture 04 narrative revision

The Lecture 04 core and bonus narrative were rebuilt on 2026-07-18 after the Lecture 03 narrative and demo handoff were accepted. Demo notebooks, the demo guide, assignment sources, graders, and generated HTML were intentionally left unchanged for later artifact rebuilds.

### Content changes

- Reframed the core around exactly five measurable objectives: the notebook execution model, a real stale-state failure and repair, the ndarray-to-labeled-pandas bridge, bounded selection/derivation/deterministic sorting, and a portable CSV round-trip verified from a fresh runtime.
- Defined notebook, cell, kernel, runtime, state, execution order, output, stale state, restart, and run all before independent notebook work. The stale-state sequence now distinguishes visible source, current in-memory values, stored output, a post-restart `NameError`, and the consistent restart-and-run-all repair.
- Made Colab the default launch experience for compatible demos and supplied a direct local-Jupyter/VS Code equivalence table. The narrative defines Colab's ephemeral filesystem, states that GitHub-opened Colab changes are not automatically saved back, rejects Drive mounting and manual uploads as defaults, and keeps assignment Colab support conditional on the save-to-repository and Classroom 50 pilot.
- Corrected the compatibility candidate to Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. The setup pattern conditionally installs only pandas 3.0.3 before import, identifies the combination as a candidate rather than a final lock, and explicitly forbids the yanked pandas 3.0.4 release.
- Pinned the portable I/O explanation to `anscombe.csv` from the `seaborn-data` repository at immutable commit `71e2436a092d714350de0fc409ca8a8714e7e78f`. The path contract searches for a future `04/demo/data/anscombe.csv` without claiming it exists, otherwise downloads the immutable HTTPS fixture, verifies SHA-256 `a0c1f636aa0347101de76271e7efe4c86a22ef28cda62886eaff23a1bf1924b1`, and creates runtime-local input/output directories without absolute-path or launch-directory assumptions.
- Added the explicit 1D ndarray to labeled Series and 2D ndarray to labeled DataFrame bridge. Required pandas work is limited to `index`, `columns`, `shape`, `dtypes`, bounded `head(3)`/`info()`/numeric `describe()` inspection, bracket selection, `.loc`, `.iloc`, one boolean mask, one arithmetic derived column, and a deterministic sort with a unique index-label tie-breaker.
- Narrowed core I/O to `read_csv`, `to_csv(index=False)`, read-back assertions, and fresh-runtime execution. Stored output is never execution evidence; sensitive output must be cleared; ordinary output remains only for human review; and generated `output/` files remain separate artifacts.
- Removed core missing-data handling, type conversion, grouped operations, alternate-format surveys, broad magic commands, interface-specific walkthroughs, summary/data-quality catalogs, and all plotting examples. The Lecture 05 handoff now begins with raw tables, row meaning, schema expectations, provenance, and explicit data-quality decisions.
- Consolidated optional material into one bounded bonus route: label alignment and explicit arithmetic axes, ranking tie rules, duplicate-label selection recognition, and reference-only Excel/JSON methods. It creates no required demo, assignment task, or Lecture 05 prerequisite.
- Replaced all six unresolved media/FIXME blocks with durable Markdown definitions and comparison tables. Removed the sole unreferenced `04/media/xkcd_1205.png` artifact; Lecture 04 now has no local media dependency.
- Retained exactly three required demo contracts in the narrative: Colab-first runtime/stale-state repair, NumPy-to-labeled-pandas selection, and a portable CSV round-trip. Existing demo artifacts remain knowingly misaligned until their separate rebuild.

### Validation

- All 15 core Python fences compile and execute progressively under managed CPython 3.12.13 with exactly NumPy 2.0.2 and pandas 3.0.3 from three clean shapes: repository root, a nested repository directory, and a non-repository directory. Every shape uses the immutable HTTPS fallback, verifies the pinned checksum and exact `(44, 3)` input schema, and produces the exact seven-row `dataset,x,y` selection for `x >= 13`. A separate fresh namespace raises the documented `NameError` when the dependent cell runs alone and produces exactly `36` after the ordered producer/dependent sequence.
- All six bonus Python fences compile under the candidate interpreter; the external-file Excel/JSON reference snippets are intentionally not executed or required.
- Static checks confirm exactly five objective rows and exactly three `LIVE DEMO` headings; no FIXME, TODO, media reference, timing estimate, or forbidden later-scope API/topic remains in the core. One bounded sentence uses the word `cleaning` only to state that inspection is not a cleaning decision; no cleaning API or decision is introduced. The bonus contains none of the explicitly deferred grouped, joining, structural, chunked, SQL, date-conversion, or missing-data APIs.
- The only pandas 3.0.4 core occurrence is the explicit prohibition. pandas 3.0.3 appears as the candidate table entry and conditional setup target.
- Scoped `git diff --check` passes for the Lecture 04 narrative, bonus, media deletion, and this audit entry. The dependency-free course audit checks all 11 lectures and parses all 52 notebooks with zero errors; its 37 warnings are existing legacy Classroom, later-notebook kernelspec, and Lecture 06 attachment follow-ups.
- Pedagogical acceptance is intentionally not self-certified. A separate reviewer must verify the revised narrative, bonus scope, and both Lecture 03→04 and Lecture 04→05 boundaries before demo or assignment implementation.

### Independent-review corrections

- Independent review rejected the first portable-data URL because the course repository path was not present upstream and the mutable branch address did not resolve. The replacement immutable `seaborn-data` URL resolved on 2026-07-18, matched the required checksum, contained exactly 44 rows with columns `dataset`, `x`, and `y`, and produced exactly seven rows for `x >= 13`.
- Added the missing bounded DataFrame inspection sequence immediately after structure: `head(3)`, direct `info()`, and numeric `describe()`. The narrative states that `info()` prints and returns `None`, must not be wrapped in `display()`, and does not authorize a cleaning decision.
- Normalized the Markdown hierarchy to one document-title H1. All core topics and handoff sections now use H2, with their internal topics at H3; the archived title convention is not applied because the active course map supersedes it.
- Re-executed all 15 core Python fences from clean repository-root, nested-repository, and non-repository working directories. All three runs exercised the HTTPS fallback and reached the exact seven-row round-trip result under the candidate stack.
- Separate independent re-verification remains required; these corrections are implementation evidence, not a pedagogical PASS.

## Lecture 04 demo rebuild

The three Lecture 04 demos were rebuilt on 2026-07-18 against the accepted narrative and independent alignment review. The core narrative, bonus, assignment, graders, and generated HTML were intentionally left unchanged.

### Artifact changes

- Replaced the three existing notebooks in place with exactly the required sequence: notebook runtime/stale-state failure and repair, NumPy-to-labeled-pandas construction and selection, and a checksum-pinned portable CSV round trip. Their filenames remain stable for existing links.
- Made each `.ipynb` the sole executable source. Removed all three same-stem Jupytext Markdown copies, rewrote the shared guide as facilitation and certification material rather than duplicated teaching code, and removed Jupytext and Matplotlib from the demo requirements.
- Standardized every notebook on the portable `Python 3` kernelspec, stable cell IDs, null execution counts, empty stored outputs, and one first-code-cell candidate setup. The setup conditionally installs only pandas 3.0.3 before import and prints Python, NumPy, and pandas versions.
- Added `.python-version` for Python 3.12.13, direct exact requirements for NumPy 2.0.2 and pandas 3.0.3, generated-output/checkpoint ignores, and the exact committed `data/anscombe.csv` fixture. Its SHA-256 is `a0c1f636aa0347101de76271e7efe4c86a22ef28cda62886eaff23a1bf1924b1`; the immutable upstream fallback remains executable when that local file is absent.
- Replaced the former plotting, broad pandas survey, missing-data cleanup, conversion, and grouped-sales examples. Core demo code now contains only the accepted notebook-state, bounded inspection, basic selection/filtering/derivation/sorting, and portable CSV APIs.
- Added development Colab badges targeting `eleventy`, explicit local-Jupyter equivalents, the real live mutation/restart protocol, exact checkpoints, output/privacy rules, and a certification record that remains pending. Publication must replace the development branch with one immutable release tag and fresh-run all three resulting badge URLs.
- Extended the dependency-free course audit only for Lecture 04 demo policy: exact notebook inventory, no paired Markdown, exact portable kernelspec, unique cell IDs, cleared execution state, fixture checksum, and obvious later-scope API violations.

### Implementation validation

- Disposable notebook copies executed top-to-bottom under managed CPython 3.12.13 with exactly NumPy 2.0.2 and pandas 3.0.3. All three notebooks printed the exact candidate versions and reached their final assertions; canonical notebook JSON remained unexecuted and output-free.
- A live kernel rehearsal using the actual Demo 1 producer, dependent, and observer cells produced `24`, then showed `rate = 3` with stale `total = 24`, raised a real `NameError` when the dependent cell ran after restart, and produced `36` from the repaired canonical order.
- Demo 3 passed from a disposable repository root using the committed fixture, from a nested repository working directory through upward search, and from a non-repository working directory through the immutable HTTPS fallback. A missing-fixture repository copy used the fallback, a corrupt fixture raised the intended checksum assertion, and a repeat run reproduced the same seven-row output bytes.
- The candidate Linux output contains exactly the seven documented `dataset,x,y` rows and has SHA-256 `1b6e792aedbed9dfa534b248716dad3f6559ff9f328d4b94f6b2d8f062c16ed1`. The semantic table remains the cross-platform invariant because the teaching write does not prescribe a line terminator.
- The dependency-free course audit parses all 52 notebooks with zero errors. Its 37 warnings are existing later-course Classroom, kernelspec, and Lecture 06 attachment follow-ups outside this rebuild.
- This is local implementation evidence, not independent pedagogical acceptance or Colab certification. A different reviewer must inspect the rebuilt artifacts, and fresh Colab execution from immutable-tag badges remains a publication gate.

## Lecture 05 narrative revision

The Lecture 05 core and bonus narrative were rebuilt on 2026-07-18 from the completed independent audit in `work/reviews/lectures_04_07_alignment.md`. Demo notebooks, the demo guide, assignment sources, graders, media, and generated HTML were intentionally left unchanged for later artifact rebuilds.

### Content changes

- Reframed the core around the audit's exact five measurable objectives: vocabulary and the raw/clean contract; a nonmutating reproducible issue audit; missing-data decisions grounded in variable meaning, row meaning, and purpose; targeted value/type/duplicate transformations; and executable invariants plus a clean artifact and decision log from a fresh runtime.
- Defined terms in dependency order before independent use: raw versus clean data, row meaning, schema, candidate identifier, tidy data, provenance/audit trail, audit, missing value, sentinel value, parse failure, lexical format check, numeric-but-noninteger value, duplicate identity, imputation, normalization, validation invariant, and decision log. Tidy structure is defined from Wickham's published formulation, while structural reshape remains deferred to Lecture 06.
- Added one self-contained raw fixture and preserved `raw_snapshot` plus a separate `working` copy. The progressive examples audit schema, pandas-recognized missingness, source-specific sentinels, nonnumeric parse failures, finite numeric-but-noninteger age values, invalid integer ranges, category-format inconsistencies, exact duplicates, and duplicate candidate identifiers before any mutation.
- Replaced universal deletion/filling advice with a structured decision table. The core states that no missingness percentage determines an action and uses `.ffill()`/`.bfill()` only as invalid counterexamples when entity and order semantics are absent.
- Limited required transformations to documented sentinel replacement, bounded string/category normalization and renaming, explicit numeric/date conversion, one raw-defined exact-duplicate decision, selection/filtering, and one review flag. Raw input is never overwritten, and no chained `inplace=True` operation appears.
- Added executable invariants for raw preservation, expected row-count relationship, required/unique candidate identifiers, allowed values, exact nullable-integer and datetime types, and valid ranges. The final pipeline writes clean data, issue audit, and decision log artifacts, rehydrates the clean CSV through an explicit nullable-integer/string/Boolean/date schema, compares normalized values and dtypes exactly, reruns the applicable invariants, and requires a fresh restart-and-run-all check.
- Made required demonstrations Colab-first with a local-Jupyter equivalent; retained assignment Colab as conditional on the repository-save/Classroom 50 pilot; and restated ephemeral-file, no-manual-upload/no-default-Drive, output/privacy, and fresh-grader evidence rules.
- Replaced the legacy unnumbered demo markers with exactly three linked H2 contracts: audit and decision table, targeted transformations, and a validated end-to-end pipeline.
- Removed the midterm placeholder, unsourced cleaning statistics, universal “drop if below 5%” guidance, unordered forward fill, automatic outlier deletion/capping, plotting, dummy encoding, sampling/bootstrap, GroupBy, configuration framework, and shell notebook-automation material from the core. The core now ends with an explicit Lecture 06 handoff.
- Rebuilt `05/BONUS.md` around only the approved bounded extensions: sourced MCAR/MAR/MNAR definitions and cautions; nullable dtypes and pandas 3 strings; advanced vectorized normalization; one bounded custom transform after vectorization; domain-aware anomaly review without automatic deletion; same-index `combine_first()` with alignment assertions and provenance; one small in-notebook configuration; and an explicit deferred-topic list.
- Removed all narrative FIXME, TODO, attachment, and media references without deleting Lecture 05 media during this narrative-only pass.
- The assessment role is intentionally unresolved outside the lecture prose: whether Lecture 05 remains a 100-point midterm and how much weight belongs to a human rubric remain later assignment/source-of-truth decisions.

### Implementation validation

- All 15 core Python fences compile and execute progressively under managed CPython 3.12.13 with NumPy 2.0.2 and pandas 3.0.3. The raw table remains unchanged, the issue audit and transformations produce the documented five-row clean result, every invariant passes, the function result equals the progressive result, and the three output artifacts round-trip in a fresh temporary working directory.
- All six bonus Python fences compile and execute progressively under the same candidate stack, including nullable types, Unicode/string normalization, the bounded identifier transform, anomaly flagging, same-index fallback, and the small configuration example.
- Static checks confirm exactly five objective rows, exactly three numbered `LIVE DEMO` H2 headings with local guide links, one H1 per narrative followed by H2/H3 nesting, and no FIXME, TODO, attachment, media, timing estimate, broad version pin, or executable later-scope API in the core.
- Scoped `git diff --check` passes. The dependency-free course audit continues to parse all 52 notebooks with zero errors; its warnings are existing legacy Classroom, kernelspec, and Lecture 06 attachment follow-ups outside this narrative pass.
- This is implementation evidence only. A different reviewer must independently verify the revised pedagogy, term order, bonus isolation, and Lecture 04→05→06 boundaries before demo or assignment implementation.

### Independent-review correction cycle

- Independent verification found four issues in the otherwise executable narrative: the exact-duplicate action was applied after normalization and coercion; pandas' date parser accepted non-zero-padded text despite the stated exact lexical contract; the CSV read-back checked only shape and columns and therefore lost the validated datetime type; and `age_text` was renamed before the prose said conversion made that rename appropriate.
- The audit now derives and preserves the exact-duplicate keep mask from untouched raw rows. Both the progressive path and `clean_person_records()` apply that mask after transformations, and a permanent regression proves that two raw-distinct rows differing only in whitespace/case both remain even when their cleaned values become equal.
- The date audit now defines an ASCII `YYYY-MM-DD` full-string pattern, counts nonmatching nonmissing text and exact-format impossible dates as parse failures, and parses only matching text. The transformation and reusable function use the same gate; a permanent regression rejects `2026-1-1` while accepting the lexical form of `2026-02-30` and then coercing it as an impossible calendar date.
- Numeric conversion now occurs under the source name `age_text`, is cast to the schema's nullable `Int64`, and is followed by the rename to `age` in both teaching paths. The export check rehydrates pandas strings, nullable `Int64` age, nullable Boolean review flags, and exact datetime text; it then requires exact frame equality and reruns clean-data invariants.
- A follow-up integer-schema review found that direct `Int64` conversion could raise on finite fractional input. The audit now counts numeric-but-noninteger age values separately from nonnumeric parse failures and integer range failures. The decision table forbids rounding, and both transformation paths retain only finite integer-valued parses before the safe `Int64` cast. A permanent `40.5` regression proves one audit count, a missing age, a true review flag, and preserved `Int64` dtype without an exception.
- All 15 core fences execute progressively with warnings promoted to errors under CPython 3.12.13, NumPy 2.0.2, and pandas 3.0.3. Every existing fixture count remains unchanged and the new numeric-but-noninteger count is zero for that fixture; the five-row result and three artifacts remain exact, the prior corrections and new fractional-age regression pass, and the round-trip dtypes are `string`/`Int64`/`datetime64[us]`/`boolean` as declared. `05/BONUS.md`, demos, assignment, media, and generated HTML remain untouched. Separate independent re-verification of this correction cycle remains pending.

## Lecture 06 narrative revision

The Lecture 06 core and bonus narrative were rebuilt on 2026-07-18 from the completed alignment audit in `work/reviews/lectures_04_07_alignment.md`. Demo notebooks, the demo guide, assignment sources, graders, media, generated HTML, and course navigation were intentionally left unchanged for later artifact rebuilds.

### Content changes

- Reframed the core around the audit's exact five measurable objectives: state row grain and test keys; predict cardinality and choose a join for a preservation goal; perform an explicit validated merge and inspect unmatched rows; concatenate with deliberate schema/index alignment; and complete a lossless structural `melt()`/`pivot()` round trip.
- Defined terms before independent use in the required order: row grain; identifier, candidate key, primary key, and foreign key; key checks; cardinality, preservation goal, and join types; explicit merge, validation, indicator, unmatched rows, and invariants; concatenation and provenance; wide/long form and identifier/measured variables; then melt/pivot uniqueness.
- Added a checksum-pinned, self-contained prepared fixture with explicit grains, single and composite key checks, many-to-one left-row preservation, explicit keys, suffixes, and row/key invariants. The narrative states pandas' missing-key matching behavior and requires missing required keys to be rejected or intentionally isolated before a merge.
- Operationalized all four cardinalities and the many-to-many per-key row formula `n_left(k) * n_right(k)`. Executable cases cover a duplicate right key rejected by `validate="many_to_one"`, an orphan foreign key exposed by `indicator=True`, pandas null-key matching, and the exact eight-row many-to-many result predicted from key counts.
- Taught vertical concatenation only for same-grain compatible partitions, with an ordinary source column for provenance and exact row/source-count checks. Separate alignment examples make missing positions from schema drift and intentionally mismatched horizontal index labels observable without turning those observations into cleaning decisions.
- Limited reshape to nonaggregating `melt()` and structural `pivot()`. The progressive path checks the long row count and identifier-variable uniqueness, proves exact wide-to-long-to-wide equality, and makes a duplicate combination fail rather than silently aggregate.
- Replaced the legacy demo markers with exactly three numbered H2 contracts: validated merge diagnostics, concat provenance/alignment, and a structural melt/pivot round trip.
- Restated the Colab-first/local-Jupyter equivalence, conditional pandas 3.0.3 setup before import, explicit pandas 3.0.4 prohibition, ephemeral-filesystem and pinned-input policy, fresh-output semantics, and assignment Colab support conditional on the repository-save/Classroom 50 pilot.
- Rebuilt `06/BONUS.md` around only index-based `merge()`/`join()` and advanced `concat()` provenance/integrity. It demonstrates index-to-index and column-to-index validation, `keys=` provenance, and `verify_integrity=True` without creating a required downstream prerequisite.
- Removed unresolved media/attachment prose, obsolete distribution links, advanced MultiIndex aggregation, `combine_first()`, aggregating pivot tables, GroupBy, resampling/time-series work, visualization, modeling, database, and performance material from the required narrative. Existing media and all non-narrative artifacts remain in place.

### Implementation validation

- All 16 core Python fences compile and execute progressively from a fresh temporary working directory with warnings promoted to errors under managed CPython 3.12.13, NumPy 2.0.2, and pandas 3.0.3. Both prepared-source checksums, key contracts, duplicate/null/orphan/validation failures, row-count invariants, concat alignment cases, exact reshape round trip, duplicate-pivot failure, and explicit-schema output read-backs pass.
- All six bonus Python fences compile and execute progressively under the same candidate stack. The index-based merges/joins preserve their stated rows, advanced source keys remain unique, overlapping concat labels raise the expected failure, and resetting meaningless labels produces the exact four-visit result.
- Static checks confirm exactly five objective rows, exactly three numbered `LIVE DEMO` H2 headings, one H1 in each narrative with valid H2/H3 nesting, dependency-ordered first definitions, and no FIXME, TODO, attachment reference, Notion reference, duration estimate, or executable later-scope API in the core.
- Scoped `git diff --check` passes. Under `06/`, only `README.md` and `BONUS.md` are modified; demos, assignment sources/tests, media, and generated HTML are unchanged. Lecture 07, the root course page, and course navigation have no scoped diff from this work.
- The dependency-free course audit checks all 11 lectures and parses all 52 notebooks with zero errors. Its 35 warnings are existing legacy Classroom references and later-notebook kernelspec gaps outside this narrative-only pass.
- This is implementation evidence only. Pedagogical acceptance is intentionally not self-certified; a separate reviewer must verify content completeness, term order, bonus isolation, and the Lecture 05→06→07 boundaries before demo or assignment implementation.

## Lecture 07 narrative revision

The Lecture 07 core and bonus narrative were rebuilt on 2026-07-18 from the completed alignment audit in `work/reviews/lectures_04_07_alignment.md`. Demo notebooks and guides, assignment sources and tests, media, generated HTML, navigation, and downstream lectures were intentionally left unchanged for later artifact rebuilds.

### Content changes

- Reframed the core around the audit's exact five measurable objectives: declare question/audience/claim and variable roles; choose and construct the five core static chart types; critique visual integrity; apply accessibility practices; and move from one bounded pandas/seaborn exploratory view to an annotated exported explanatory chart.
- Defined concepts in the accepted dependency order before independent construction: visualization; question, audience, and claim; displayed unit and plotting grain; variable roles; exploratory versus explanatory purpose; marks and encodings; line/bar/scatter/histogram/box chart jobs; bins and box-plot summaries; scale, axis, baseline, context, and integrity; accessibility; Figure/Axes; long-form data; then annotation, layout, export, and human visual QA.
- Added precise box-plot vocabulary for median, quartiles, IQR, default whiskers, and individual points beyond the whiskers. The prose explicitly rejects treating those points as automatic errors or deletion candidates and links the exact Matplotlib 3.10 reference.
- Added a visualization contract that precedes every tool choice. The running prepared example states one program-round summary per row, distinguishes descriptive evidence from causal claims, and uses fixed literal participant, program-round, and supplied-summary tables rather than random, mutable, or network-only data.
- Focused required construction on Matplotlib Figure/Axes with one line, bar, scatter, histogram, and box plot. Examples include complete units and labels, explicit histogram bins, a zero-based magnitude bar, nuanced nonzero-axis guidance for positional charts, colorblind-safe hues, and redundant color/marker/line-style encodings.
- Added bounded pandas and seaborn exploratory scatter views without correlation, regression, density estimation, confidence intervals, or inferential claims. The explanatory path returns to explicit Figure/Axes, adds a descriptive annotation and text alternative, exports one deterministic PNG, and separates executable properties from a human integrity/accessibility/layout rubric.
- Replaced the three legacy unnumbered markers with exactly three linked H2 contracts: critique and redesign, Figure and Axes fundamentals, and exploratory to explanatory.
- Restated the provisional Python 3.12.13, NumPy 2.0.2, pandas 3.0.3, Matplotlib 3.10.8, and seaborn 0.13.2 candidate; Colab-first/local-Jupyter equivalence; ephemeral-file/no-default-upload-or-Drive behavior; privacy; conditional assignment Colab support; and fresh-rendered-output evidence.
- Rebuilt `07/BONUS.md` around only extended critique, deliberate static small multiples, one bounded raw-observation seaborn view, PNG/SVG export, visual-QA additions, and a nonexecutable orientation to Altair and Plotly. Dashboards, animation, correlation/regression/inference, density estimation, time series, modeling, geospatial systems, and performance work are explicitly deferred.
- Removed the remote-image prose, broad visualization-library survey, invalid/deprecated plotting guidance, regression and density APIs, correlation heatmaps, confidence displays, animation, real-time plotting, color psychology, dashboards, and performance survey. Existing media and all non-narrative Lecture 07 artifacts remain in place.
- Made the Lecture 08 handoff explicit: students can state chart purpose and displayed unit, work from supplied long-form data, chart an already-supplied summary, and explain that later aggregation changes the represented unit and row count.

### Implementation validation

- All nine core and five bonus Python fences compile and execute progressively in fresh temporary working directories with an Agg backend under managed CPython 3.12.13, NumPy 2.0.2, pandas 3.0.3, Matplotlib 3.10.8, and seaborn 0.13.2. Lecture-code warnings were promoted to errors.
- Independent verification replaced the initial Matplotlib 3.10.0 candidate with 3.10.8. Version 3.10.8 preserves the required APIs and passes with Colab's newer pyparsing inventory without the test-only transitive constraint that 3.10.0 needed. Exact transitive constraints remain a course-level certification gate.
- The original seaborn box-plot exploration exposed seaborn 0.13.2's internal use of Matplotlib's pending `vert` deprecation under warnings-as-errors. The core now uses a supported seaborn scatter view with hue plus marker style; the required box plot remains a direct Matplotlib 3.10.8 `orientation=` example. No warning is suppressed by the teaching code.
- Executable checks verify prepared-table shapes; all chart titles, axes, and units; two-line identity and redundant encodings; zero-based bars; exact histogram edges and observation count; box/median artist counts; exploratory Axes labels; explanatory legend, annotation context, text-alternative limitation, and successful PNG/SVG writes.
- Two fresh explanatory runs produced the same 68,875-byte `(1184, 700)` PNG with SHA-256 `1a5132ea019960b070acf574cd8f89a9bfd91f35c9b8fddf751ef80f2f23caf9`. A visual smoke check found the title, labels, legend, redundant markers/styles, annotation, and data marks visible without clipping; this is implementation QA rather than pedagogical certification.
- Static checks confirm the five objectives verbatim, exactly three numbered `LIVE DEMO` H2 headings, one H1 per narrative with valid H2/H3 nesting, the required first-definition order, three valid local demo-guide links, and no FIXME, TODO, attachment, remote image, duration estimate, or executable cleaning/join/reshape/aggregation/time-series/modeling/interactive-library API in core.
- Scoped execution created no repository output. `07/README.md` and `07/BONUS.md` are the only Lecture 07 files edited; demos, assignments/tests, media, generated HTML, navigation, and downstream lecture sources remain untouched.
- This is implementation evidence only. Pedagogical acceptance is intentionally not self-certified; a separate reviewer must verify content completeness, chart/statistical explanations, accessibility guidance, bonus isolation, and the Lecture 06→07→08 boundaries before demo or assignment implementation.

## Lecture 08 narrative revision

The Lecture 08 core and bonus narrative were rebuilt on 2026-07-18 from the accepted aggregation outline in `work/reviews/lectures_08_11_alignment.md` and the verified Lecture 07 handoff. Demo notebooks and guides, assignment sources and tests, media, generated HTML, navigation, and Lecture 09 were intentionally left unchanged for later artifact rebuilds.

### Content changes

- Reframed the core around exactly five measurable objectives: predict grouping unit/key and input/output grain; choose `size`, `count`, or `nunique` and name a flat aggregation; distinguish aggregation from same-index `transform`; make grouped key/value columns, order, and index placement explicit; and build and interpret one aggregating pivot table.
- Defined the required concepts before independent use in the accepted order: input row grain, grouping key, group, grouping unit, output grain, aggregation, GroupBy object, split–apply–combine, deterministic data, categorical `observed` policy, missing-key `dropna` policy, the three count meanings, named aggregation, transform, index placement, two-key output grain, and aggregating pivot specification.
- Added one fixed twelve-row encounter table with repeated providers, missing ratings, four declared facility categories but only three observed facilities, all three service levels, and no South–Follow-up encounter. Every grouping and pivot call declares `observed=`, `sort=`, and the applicable missing-key policy instead of relying on defaults.
- Corrected categorical grouping semantics for pandas 3: `observed=True` includes only observed categorical values/combinations and is the pandas 3.0 default, while the core writes it explicitly. The bonus demonstrates the deliberate `observed=False` alternative by materializing the unused Remote level with count zero and separately contrasts `dropna=True` with `dropna=False` for a missing grouping key.
- Made counting question-driven and executable: group row counts are `[4, 4, 4]`, nonmissing rating counts are `[3, 2, 4]`, distinct-provider counts are `[2, 2, 2]`, and grouped encounter counts conserve all twelve source rows.
- Used named aggregation with flat output names and `as_index=False`, then contrasted it with a same-length, same-index facility-mean transform. A separate indexed one-key result and bounded flat two-key result expose key placement and the exact eight observed facility–service groups without making MultiIndex manipulation core.
- Added exactly one core `pivot_table()` call. Its index, columns, values, aggregation, ordering, categorical policy, and output meaning are stated before execution; every populated cell is checked against the equivalent two-key GroupBy result; and South–Follow-up remains missing because no input row represents that combination rather than being filled with a false zero.
- Replaced the legacy demo markers with exactly three linked H2 contracts: grouping grain and counts; named aggregation and transform; and one aggregating pivot with at most one already-familiar Lecture 07 chart and no new plotting objective.
- Restated Colab-first/local-Jupyter equivalence, conditional pandas 3.0.3 setup before import, explicit pandas 3.0.4 prohibition, deterministic/ephemeral-input policy, fresh execution evidence, and assignment Colab support conditional on the repository-save/Classroom 50 pilot.
- Rebuilt `08/BONUS.md` around only bounded advanced grouping/index-output policies: observed versus declared categories, missing grouping-key policy, whole-group filtering through a named rule, and inspection/reset of one two-level grouped result index. `GroupBy.apply`, advanced pivots, advanced MultiIndex manipulation, time series, statistics, plotting, performance, chunking, and parallelism are excluded.
- Explicitly deferred SSH/tmux to a separate optional local-terminal lab requiring an approved practice host and institution-specific authentication/security/platform instructions. It is not a notebook or Colab activity and cannot become a Lecture 08 or Lecture 09 prerequisite.
- Removed the former core and bonus surveys of custom apply functions, crosstabs, advanced pivot breadth, period-based grouping, rolling and lag features, statistics, visualization breadth, remote execution, SSH/screen/tmux commands, profiling, chunking, multiprocessing, Dask/cloud systems, and performance optimization. The required handoff now gives Lecture 09 grouping/result-grain prerequisites without preteaching time-series concepts.

### Implementation validation

- All 11 core Python fences compile and execute progressively from a fresh temporary working directory with warnings promoted to errors under managed CPython 3.12.13, NumPy 2.0.2, and pandas 3.0.3. The exact group identities/counts, named schemas and values, total conservation, transform row count/index, eight two-key groups, three-by-three pivot, absent Remote row, missing South–Follow-up cell, and GroupBy-to-pivot cell equality all pass.
- All six bonus Python fences compile and execute progressively under the same candidate stack. The observed/all-level contrast, missing-key conservation, whole-group filter's exact retained indices, and indexed/flat two-key layouts pass without warnings.
- Static checks confirm exactly five objective rows, exactly three numbered `LIVE DEMO` H2 headings, one H1 in each narrative with valid H2/H3 nesting, six valid local narrative links, and no FIXME, TODO, attachment, Notion, duration estimate, executable time-series API, `GroupBy.apply`, remote command, or performance implementation in core. All four official pandas API links return HTTP 200.
- Scoped `git diff --check` passes. Under `08/`, only `README.md` and `BONUS.md` are modified; demo and assignment artifacts remain unchanged. Execution created no repository output.
- The dependency-free course audit checks all 11 lectures and parses all 51 currently present notebooks with zero errors. Its 34 warnings are existing legacy Classroom references and later-notebook kernelspec gaps, including the untouched legacy Lecture 08 demos, outside this narrative-only pass.
- This is implementation evidence only. Pedagogical acceptance is intentionally not self-certified; a separate reviewer must verify content completeness, first-definition order, bonus isolation, pandas semantics, and the Lecture 07→08→09 boundaries before Lecture 08 demo or assignment implementation.

### Independent verification

- A reviewer who did not author the Lecture 08 changes read the complete core and bonus, checked the verified Lecture 07 handoff and accepted Lecture 09 entry contract, and confirmed that grouping/result-grain capabilities are introduced without preteaching time-series work.
- Exactly five measurable objectives and exactly three H2 `LIVE DEMO` contracts are present. Heading hierarchy, local links, term order, core/bonus disposition, and the explicit SSH/tmux deferral pass.
- All core Python fences execute progressively in a fresh temporary directory under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3 with warnings treated as errors. The three group sizes; size/count/nunique distinctions; named aggregation; transform length/index; two-key grain; three-by-three pivot; missing South–Follow-up cell; and core final invariants pass.
- All bonus Python fences execute independently under the same stack with warnings treated as errors. Observed versus declared categories, missing grouping-key handling, whole-group filtering, and the bounded two-level index layout pass.
- Static scope checks find no executable `GroupBy.apply`, time-series API, plotting API, remote command, performance implementation, unresolved marker, attachment, or duration estimate. pandas 3 categorical policy is stated explicitly rather than inherited from a default.
- Lecture 08 narrative and bonus therefore pass the independent narrative gate. The unchanged legacy demos and assignment remain unverified and must be rebuilt separately.

## Lecture 05 demo rebuild

The three required Lecture 05 notebooks and their guide were rebuilt on 2026-07-18 after the narrative passed independent verification.

### Artifact changes

- Replaced the legacy missing-data, transformation-survey, and configuration-workflow demos with exactly three canonical notebooks: `demo1_audit_decisions.ipynb`, `demo2_targeted_transformations.ipynb`, and `demo3_validated_pipeline.ipynb`.
- Removed paired same-stem Markdown copies. Added a portable Python 3 kernelspec, stable unique cell IDs, cleared outputs, null execution counts, exact direct-dependency pins, Python 3.12.13 metadata, local-notebook hygiene rules, and a guide with development Colab badges that must move to an immutable release tag before publication.
- Added one six-row course-authored synthetic fixture with checksum `7b3223154756aa59f2f00027ddbadaa225eeee51ad75d0df91de1fd8d14abe2d`. Repository launches use the committed file; standalone launches reconstruct the same supplied bytes. Both branches verify the checksum before parsing.
- Demo 1 states row meaning/schema/provenance, produces the exact fifteen-row issue audit and six-row decision table, and proves the raw table remains unchanged.
- Demo 2 makes an invalid cross-person forward-fill result visible without assigning it, then applies only documented sentinel, normalization, numeric/date, raw-derived exact-duplicate, and review-flag rules. A fractional-age regression proves that `40.5` becomes missing without rounding.
- Demo 3 runs raw→audit→decide→transform→validate→save, preserves normalization-colliding raw-distinct rows, writes clean/audit/decision artifacts, and verifies schema-aware readbacks.
- The course audit now enforces the exact Lecture 05 notebook set, fixture checksum, requirements, portable metadata/state, the single rejected forward-fill preview, and later-scope exclusions.

### Independent verification

- A reviewer who did not author the notebooks executed all three under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3 with lecture-code warnings promoted to errors.
- Repository-root, nested, standalone, missing-fixture, corrupt-fixture, output deletion/recreation, and deterministic repeat cases passed.
- Exact audit/log counts, raw checksum immutability, schema-aware clean-data readback, raw-derived duplicate handling, and fractional-age behavior passed.
- Notebook format/state/IDs/kernelspec, guide links and claims, privacy, and scope boundaries passed after one minimal `.gitignore` correction for Jupyter checkpoints.
- Fresh Colab runs and immutable-tag badge checks remain publication gates; they are not inferred from local execution.

## Lecture 06 demo rebuild

The three required Lecture 06 notebooks and their guide were rebuilt on 2026-07-18 after the narrative passed independent verification. They have now passed independent demo review.

### Artifact changes

- Replaced the legacy broad merge, pivot/melt, and concat/time-series notebooks with exactly three canonical notebooks: `demo1_validated_merge.ipynb`, `demo2_concat_alignment.ipynb`, and `demo3_structural_reshape.ipynb`.
- Removed paired same-stem Markdown copies and the former broad plotting/Jupyter requirement set. Added exact NumPy/pandas pins, Python 3.12.13 metadata, portable kernelspecs, stable unique cell IDs, cleared execution state, checkpoint/output hygiene, and development Colab badges with an immutable-tag publication gate.
- Added three checksum-pinned synthetic fixtures: one-row-per-visit data with one orphan site, versioned site metadata with one duplicate site key before the supplied current-record rule, and one-row-per-participant/site wide scores.
- Demo 1 states grains and keys, makes `validate="many_to_one"` reject the unfiltered history dimension, applies the supplied current-record rule, retests uniqueness, performs a diagnostic left merge, and verifies six preserved visit IDs, five matches, and the exact `V006`/`X` orphan.
- Demo 2 vertically stacks two same-grain partitions with source provenance, diagnoses column-alignment schema drift without cleaning it, horizontally aligns deliberate `visit_id` indexes, explains both missing positions, and verifies two schema-aware artifacts.
- Demo 3 states wide/long grains and identifier/measured variables, predicts and verifies six melted rows, reconstructs the source exactly with structural `pivot()`, makes a duplicate combination fail, and verifies the long-form artifact.
- The course audit now enforces the exact Lecture 06 notebook set, three fixture checksums, requirements, portable metadata/state, and exclusions for grouping/aggregating pivots, plotting, datetime/time-series work, and manual-upload/Drive paths.

### Local implementation validation

- All three notebooks execute from repository-root, nested `06/demo/`, and standalone layouts under the candidate stack with lecture-code warnings promoted to errors.
- An initial pandas 3 readback exposed two representation-level dtype differences: the provenance column's missing-value convention and the pivot-created column index dtype. Both contracts are now normalized explicitly, and all nine launch cases pass.
- Generated repository artifacts are exactly `combined_visits.csv`, `aligned_features.csv`, and `scores_long.csv`.
- `git diff --check` and the Lecture 06-specific dependency-free course audit pass with zero errors. A full-course audit run during the atomic Assignment 05 rebuild reported only the unrelated temporary absence of `05/assignment/`.

### Independent verification

- A reviewer who did not author the Lecture 06 demos executed all three through fresh notebook kernels from repository-root, nested `06/demo/`, and standalone layouts under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. A separate progressive-cell harness repeated the three-layout matrix with lecture warnings treated as errors.
- Missing and corrupted fixtures were exercised independently. Missing files were reconstructed byte-for-byte; corrupt committed fixtures were rejected before parsing. Deleted, stale, and corrupt generated artifacts were replaced deterministically, with identical hashes across layouts and repeat runs.
- Exact merge diagnostics, orphan identity, vertical provenance, alignment missingness, dtype-aware readbacks, six-row melt, exact pivot reconstruction, and duplicate-pivot failure all passed. Notebook state, IDs, kernelspecs, guide/checksum claims, scope exclusions, and the Lecture 05→06→07 boundaries also passed.
- Lecture 06 demos therefore pass the independent demo gate. Fresh Colab execution and immutable-release badge targets remain publication gates.

## Assignment 04 rebuild

Assignment 04 was rebuilt on 2026-07-18 from `work/reviews/assignment04_blueprint.md` as the course's first required notebook assignment and Classroom50 notebook-grading pilot. It requires clean local Jupyter; assignment Colab launch/save/submission remains gated on the course-level pilot.

### Artifact changes

- Replaced the legacy generated-data notebook, GitHub Classroom workflow/tests, broad tips, and later-scope cleaning/GroupBy work with one portable starter notebook, a synthetic checksum-pinned CSV fixture, public artifact checker, exact environment records, and an instructor-only disposable grader self-test bundle.
- The three tasks assess only fresh notebook dependency order; labeled Series/DataFrame construction and bracket/`.loc`/`.iloc` selection; and one portable CSV filter, arithmetic derived column, deterministic sort, `index=False` write, and readback.
- The student notebook has stable cell IDs, a portable Python 3 kernelspec, null execution counts, zero stored outputs, an immutable portable setup cell, and exactly three stable actionable public failures before completion.
- The central-grader prototype clears stored state, deletes generated CSVs, appends instructor-owned checks to a disposable copy, fresh-executes canonical and relocated checkouts, and repeats against a second valid fixture. It emits the discoverable `classroom50/result/v1` contract without trusting the editable public checker.

### Independent verification

- A reviewer who did not author Assignment 04 read the complete blueprint, README, platform checklist, starter notebook, public checker, protected-file manifest, central grader, and adversarial harness. The written tasks, automated checks, and human-review boundary align without introducing Lecture 05+ concepts or requiring Colab.
- The full self-test passed with exit code 0 under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. Correct source regenerated missing outputs and passed canonical, relocated/nested, alternate-fixture, unrelated-working-directory, and resubmission cases.
- The harness rejected stored-output fakery, `.loc`/`.iloc` boundary errors, Series/DataFrame return-type confusion, wrong arithmetic, `> 2`, a missing tie-breaker, serialized indexes, canonical-row hard-coding, absolute/Colab paths, edited fixture/manifest/checker, malformed JSON, and a missing supplied cell.
- Fixture, setup, protected-file, environment, JSON/state, package-hygiene, and scoped `git diff --check` checks passed. The repeated local kernel TCP-encryption warning comes from the disposable test harness and does not affect notebook results.
- Assignment 04 therefore passes the independent assignment gate. Actual Classroom50 provisioning/feedback/resubmission and any future Colab repository save-back remain pilot gates rather than inferred capabilities.

## Lecture 09 narrative revision

The Lecture 09 core and bonus narrative were rebuilt on 2026-07-18 from the reconciled temporal-data contract in `work/reviews/lectures_08_11_alignment.md` and the accepted Lecture 08 handoff. Demo notebooks and guides, assignment sources and tests, media, generated HTML, navigation, and Lectures 08 and 10 were intentionally left unchanged for later artifact rebuilds.

### Content changes

- Reframed the core around exactly five measurable objectives: classify timestamp/period, regular/irregular, and single/panel structure with explicit grain and sort keys; parse/localize/convert/sort/index within entity; choose `asfreq` or measurement-aware `resample`; build entity-scoped lag/difference and observation-count or elapsed-time windows; and audit availability plus a plausible chronological holdout.
- Defined the required concepts before independent use in the accepted order: temporal ordering, timestamp, period, entity, entity key, single series, panel, row grain, sort keys, regular/irregular observations, frequency, naive/aware timestamps, localization/conversion, DatetimeIndex, bounded interval, up/downsampling, `asfreq`/`resample`, source/grid missingness, measurement meaning, lag/lead/difference, trailing observation-count/elapsed-time windows, candidate feature, prediction timestamp, information availability, centered/future-derived candidates, future leakage, and chronological holdout.
- Added one fixed ten-row, two-station panel with explicit one-station-observation grain, entity–timestamp uniqueness, unambiguous Los Angeles-to-UTC conversion, within-station order, one source-missing temperature, and irregular one-/two-hour gaps. A source-row marker makes grid-created rows distinguishable from missing values already present in source data.
- Corrected pandas 3 frequency guidance to lowercase `h` and quarter/year-end `QE`/`YE`. Grouped hourly `asfreq()` retains both stations and identifies exactly four grid-created rows separately from the one source-missing value. Grouped left-closed/left-labeled two-hour `resample()` produces a station–interval grain, justified mean temperature and reading count, and conserves all ten source rows.
- Built valid DataFrame columns from station-scoped Series operations rather than assigning named fields into a Series. The first lag and difference in each station are missing. At South 21:00, the previous-two-observation mean is exactly 21.5 while the previous-two-elapsed-hour mean is exactly 22.0, proving that the two window meanings are not interchangeable on irregular data.
- Added a supplied 21:00 UTC availability inventory that keeps calendar time and the previous observation while rejecting a centered three-observation mean and next observation because both require 22:00 data. The chronological handoff keeps all earlier timestamps strictly before the later block and retains both stations without claiming a completed Lecture 10 evaluation design.
- Replaced the legacy demo markers with exactly three H2 contracts: classify/prepare temporal structure; frequency/resampling with measurement meaning; and past-only features/information availability. The third may use at most one already-familiar Lecture 07 temporal chart; plotting is not a new objective.
- Rebuilt `09/BONUS.md` around expanding, exponentially weighted, centered, and custom windows; advanced partial/clock-time selection; current calendar offsets; explicit DST ambiguity/nonexistence policy; and nonexecutable orientation to decomposition, STL, forecasting, ARIMA, exponential smoothing, and high-frequency/tick data. None is a required demo, assignment capability, or Lecture 10 prerequisite.
- Removed the broad core surveys of Python `datetime`, advanced time selection, centered/custom/expanding/EWM implementations, plotting and synthetic component generation, forecasting language, invalid Series-as-DataFrame examples, pooled-entity operations, obsolete frequency aliases, random/current-time examples, and unresolved media dependencies.
- Restated Colab-first/local-Jupyter equivalence, conditional pandas 3.0.3 setup before import, pandas 3.0.4 prohibition, ephemeral-file/no-default-upload-or-Drive behavior, fresh execution evidence, and assignment Colab support conditional on the repository-save/Classroom 50 pilot.

### Implementation validation

- All 11 core Python fences compile and execute progressively under managed CPython 3.12.13 with exactly NumPy 2.0.2 and pandas 3.0.3. Warnings are promoted to errors. Timestamp/period bounds, parsing/localization/conversion, entity–timestamp uniqueness, within-entity ordering, irregular gaps, bounded selection, source/grid missingness, grouped resample values/count conservation, per-entity lag/difference boundaries, both window semantics, availability decisions, and chronological ordering all pass.
- All seven bonus Python fences compile and execute progressively under the same exact stack with warnings promoted to errors. Expanding/EWM, centered/custom windows, partial-string/clock-time selection, `QE`/`YE` offsets, explicit spring/fall DST policies, and final invariants pass.
- Static checks confirm exactly five objective rows, exactly three numbered `LIVE DEMO` H2 headings, one H1 per narrative, valid H2/H3 nesting, and the existing local demo-guide target. No FIXME, TODO, attachment, Notion, duration estimate, or executable obsolete `H`/`Q`/`A` frequency alias remains.
- The dependency-free course audit runs directly under Python 3.12.13 because this environment does not provide the repository's `npm` wrapper. It checks all 11 lectures and parses all 48 currently present notebooks with zero errors. Its 31 warnings are existing legacy Classroom references and missing kernelspecs in untouched Lecture 07–11 artifacts, including the legacy Lecture 09 demos.
- Scoped `git diff --check` passes. Under `09/`, only `README.md` and `BONUS.md` are modified; demo and assignment artifacts remain unchanged. Execution created no repository output.
- This implementation evidence was subsequently checked by a separate reviewer; the independent result is recorded below.

### Independent verification

- A reviewer who did not author Lecture 09 read the complete core/bonus, the accepted Lecture 08 exit and Lecture 10 entry contracts, and corrected one prerequisite phrase so Lecture 09 now requires candidate/grouping-key knowledge on entry while defining entity keys inside Lecture 09.
- Exactly five objectives and exactly three H2 `LIVE DEMO` contracts match the accepted order. Timestamp/period, entity/panel/grain, parsing/timezone/sort, frequency/resampling/missingness, past-only windows, and information availability/chronological holdout are defined before independent use.
- All 11 core and seven bonus Python fences passed progressively under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3 with warnings treated as errors. Entity-scoped operations, pandas 3 aliases, `asfreq`/resample semantics, source-versus-grid missingness, window boundaries, and future-leakage exclusions passed.
- Core contains no modeling or new visualization objective. EWM, expanding, centered/custom windows, advanced selection/calendar/DST, decomposition/STL, forecasting/ARIMA, and high-frequency material remain bonus-only and are not downstream prerequisites.
- Heading hierarchy, local/external links, and the Lecture 08→09→10 boundary pass. Lecture 09 narrative therefore passes its independent gate; the legacy demos and assignment remain unaccepted and must be rebuilt separately.

## Lecture 07 demo rebuild

The three required Lecture 07 notebooks and their guide were rebuilt on 2026-07-18 after the narrative passed independent verification. The independent local demo gate subsequently passed; fresh Colab execution and immutable release-tag badges remain pending publication gates.

### Artifact changes

- Replaced the legacy Matplotlib survey, network-dependent statistical seaborn notebook, modern-library survey, paired Markdown copies, cleaning side demo, and committed Altair/interactive/scatter/damped-sine outputs with exactly three canonical notebooks: `demo1_critique_redesign.ipynb`, `demo2_figure_axes.ipynb`, and `demo3_explore_explain.ipynb`.
- Added exact Python 3.12.13, NumPy 2.0.2, pandas 3.0.3, Matplotlib 3.10.8, and seaborn 0.13.2 records; portable Python 3 kernelspecs; stable globally unique cell IDs; null execution counts; zero stored outputs; ignored runtime output; and development Colab badges with an immutable-tag publication gate.
- Added three checksum-pinned course-authored prepared fixtures: four program-period follow-up percentages, ten program-round scores, and ten participant practice/score rows. Repository launches find the committed data; standalone launches reconstruct identical embedded bytes; both paths verify checksums before parsing.
- Demo 1 states question/audience/claim/unit/grain/roles, makes five planted defects visible, and repairs the chart with a zero baseline, percentage units, descriptive claim, reduced decoration, color plus hatch, value labels, and a causal-limitation text alternative.
- Demo 2 defines Figure and Axes before use and constructs only the five core chart types. It uses a literal supplied two-row mean table rather than GroupBy, explicit histogram edges, descriptive scatter language, redundant line identity, and careful box-plot interpretation.
- Demo 3 uses one bounded seaborn exploratory scatter, then returns to Matplotlib for a coordinator-facing descriptive line chart with an exact seven-point annotation, redundant encoding, supporting-data CSV, and text-alternative file.
- The course audit now enforces the exact notebook/fixture/dependency/state package, globally unique IDs, expected output contracts, no committed generated output, and exclusions for advanced/interactive plotting libraries, network/random data, cleaning/join/reshape/grouping/aggregation, time-series work, correlation/regression/modeling, uploads, and Drive mounts.

### Local implementation validation

- All three notebooks executed progressively with warnings promoted to errors from repository-root, nested `07/demo/`, and standalone layouts under the exact candidate stack. All nine layout cases passed after correcting one generated JSON newline escape in Demo 3.
- All three notebooks also passed actual fresh candidate-kernel smoke execution. Fixture shapes/checksums, zero baseline, hatches, exact line identities, histogram edges/count conservation, box artists, annotation, schemas/readbacks, labels/units, legends, text limitations, and output dimensions passed.
- Newly rendered PNGs were inspected at original resolution. Titles, labels, units, legends, hatches/markers/styles, annotation, and marks were visible without clipping; the grouped bars started at zero and the explanatory claim remained descriptive. This is author-side visual smoke QA rather than independent acceptance.
- Generated outputs were removed after inspection. The dependency-free course audit parses 48 current notebooks with zero errors; its 31 warnings belong to untouched legacy later assignments/demos. Scoped `git diff --check` passes.
- Fresh Colab execution and immutable badge targets remain publication gates.

### Independent verification and correction

- A reviewer who did not author the Lecture 07 demos read the complete blueprint, accepted Lecture 07 core and bonus narratives, both adjacent lecture contracts, every notebook cell, the guide, exact environment records, and all fixture bytes. Scope exclusions and the Lecture 06→07→08 progression pass.
- All three notebooks fresh-executed from the repository root, `07/demo/`, a nested directory under `07/demo/`, and standalone disposable directories: twelve fresh-kernel cases passed under Python 3.12.13, NumPy 2.0.2, pandas 3.0.3, Matplotlib 3.10.8, and seaborn 0.13.2. All three also passed progressive execution with warnings promoted to errors.
- Every missing-fixture and corrupt-fixture case was exercised for every consuming notebook. Missing standalone fixtures were reconstructed byte-for-byte; corrupt committed fixtures stopped at checksum verification and were not replaced. Deleted, stale, and corrupt outputs were rebuilt while unrelated files and other notebooks' artifacts were preserved. Same-platform repeat runs produced identical hashes for all five outputs.
- Exact planted defects, corrected baseline/labels/hatches/value labels, five Figure/Axes chart contracts, supplied data values, histogram counts and edges, box medians, seaborn marks, explanatory annotation, supporting CSV, and text alternative all passed. Independent original-resolution visual review confirmed readable units and labels, distinguishable redundant encodings, accurate annotation, descriptive claims, and no clipping.
- Independent review made three narrow corrections: the landing Markdown in each notebook now names the question/audience/claim contract before free use; the repaired Demo 1 legend moved outside the Axes after visual review found that it covered two bars; and the course audit now asks Git whether `07/demo/output/` files are committed instead of rejecting legitimate ignored runtime output merely because it exists.
- Canonical notebook state, globally unique stable cell IDs, portable kernelspecs, dependency order, path behavior, guide/checksum/badge claims, no paired Markdown, no committed output, scoped course audit, and `git diff --check` pass. Lecture 07 demos therefore pass the independent local demo gate. Fresh Colab execution and immutable release-tag badge targets remain pending rather than inferred.

## Assignment 05 rebuild

Assignment 05 was rebuilt on 2026-07-18 from `work/reviews/assignment05_blueprint.md` as one neutral, documented cleaning-pipeline assignment. Whether it is labeled or weighted as a midterm remains a syllabus decision; the technical contract is independent of that policy overlay.

### Artifact changes

- Replaced the four legacy notebooks, shell/config/module exercises, large clinical generator/data, duplicated scripts/tests, committed reports/outputs, broad tips, and GitHub Classroom workflow with one local-Jupyter starter notebook, one 570-byte checksum-pinned synthetic fixture, one public checker, exact environment records, and an instructor-only Classroom50 grader/self-test bundle.
- The notebook follows exactly `raw → audit → decide → transform → validate → save`. Task 1 produces the exact fifteen-issue audit; Task 2 records eight ordered decisions before transforming a deep copy; Task 3 validates, writes, schema-aware reads back, and compares the issue audit, clean table, and provenance-rich decision log.
- The contract preserves raw bytes and `raw_snapshot`, derives exact-duplicate retention from untouched raw rows, distinguishes duplicate records from candidate-ID conflicts, rejects fractional ages without rounding, enforces exact ASCII `YYYY-MM-DD` plus calendar validity, retains reviewable missingness, and uses nullable pandas dtypes on readback.
- The neutral provisional overlay is 30/40/30 by task, with 85 automated and 15 human-reviewed points pending syllabus adjudication. Clean local Jupyter is mandatory; assignment Colab remains conditional on the repository-save/Classroom50 pilot.

### Independent verification and correction

- A reviewer who did not author Assignment 05 read the full blueprint, student instructions, platform guide, starter notebook, public checker, central grader, and adversarial harness. The task sequence, exact counts/values/dtypes, decision/human boundaries, and Lecture 05 scope align.
- The full adversarial self-test passed under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. It covers the untouched starter, correct solution, fresh/relocated execution, stale/deleted outputs, stored-output distrust, corrupt/missing fixtures, input mutation, raw-normalization collision, alternate nondefault-index values and schemas, forbidden APIs, and public-check independence. Correct work earns 85/85 automated points.
- Independent review found that the original result object used underscore-style fields and only printed an object, which did not satisfy the official `classroom50/result/v1` contract. The central reference now writes `result.json`, includes the required metadata, uses `max-score`/`test-name`/`passed` fields, preserves failure detail in logs, and exits successfully for captured test failures. The harness now validates a full emitted 85/85 payload against that shape.
- Independent review also found that `.gitignore` hid the three CSV artifacts the student instructions require committing. Assignment outputs are now visible to VS Code Source Control and GitHub Desktop; only caches, checkpoints, bytecode, and local environments are ignored. The corrected self-test still passes. The same submission-mechanics correction was applied to the accepted Assignment 06 blueprint before implementation.
- Fixture size and SHA-256, manifest, supplied-cell hashes, notebook IDs/state/kernelspec, exact dependencies, output hygiene, no cache artifacts, full-course audit, and scoped `git diff --check` pass.
- Assignment 05 therefore passes the independent assignment gate. Actual Classroom50 provisioning, score collection, feedback/resubmission behavior, and any future assignment Colab path remain pilot gates.

## Lecture 08 demo design verification

The implementation blueprint in `work/reviews/lecture08_demo_blueprint.md` was
independently reviewed on 2026-07-18 after the Lecture 08 narrative and its
Lecture 07/09 boundaries had passed. No file under `08/demo/` was edited during
this design gate.

### Independent verification and correction

- The exact three-demo progression matches all five verified Lecture 08
  objectives: predict group identities and grain; distinguish `size`, `count`,
  and `nunique`; create a flat named aggregation; contrast aggregation with a
  same-index `transform`; make a bounded two-key result deliberate; and compare
  one aggregating `pivot_table` with its equivalent GroupBy result.
- The synthetic fixture was reconstructed from the blueprint's literal text and
  independently hashed as
  `24a31904c1371553ff3af627dc21146ed743c8c0c47452ade3628c2fc199c5dc`.
  Under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3, it produced the exact
  three facility counts, `[560, 545, 545]` total charges, `[28.0, 26.0, 28.75]`
  mean waits, eight observed facility-service groups, and the specified 3-by-3
  pivot with South--Follow-up missing and the other eight cells populated.
- Independent review made design-only corrections before implementation:
  numeric fixture dtypes and `.gitignore` scope are now explicit; all reads and
  writes derive from one resolved demo root; present-but-corrupt fixtures cannot
  be silently replaced; CSV encoding, newlines, index serialization, and the
  missing-cell representation are deterministic; new terms must be defined
  before code use; Colab save-back limitations are stated; the wrong-grain
  diagnostic now uses an unambiguous three-value positional array rather than
  accidental Series label alignment; pivot occupancy is exact; and QA covers all
  four launch layouts, deterministic output hashes, and measured publication
  evidence.
- Remote computing, performance engineering, structural reshape, group
  filtering/`apply`, time series, modeling, network data, and a new visualization
  objective remain outside the required demos. Assignment Colab support remains
  conditional on the repository-save/Classroom50 pilot.
- The blueprint therefore passes the independent design gate and is ready for a
  separate implementation author. Fresh local execution, adversarial path/data/
  output testing, fresh Colab execution, and immutable release-tag badges remain
  later gates rather than inferred results.

## Assignment 06 rebuild

Assignment 06 was rebuilt on 2026-07-18 from the accepted technical handoff in
`work/reviews/assignment06_blueprint.md`. This section records author-side
implementation evidence only; it is not independent assignment acceptance.

### Artifact changes

- Replaced the legacy customer/product/purchase generator, paired Markdown and
  notebook sources, mutable remote data path, broad dependencies, GitHub
  Classroom workflow/tests, and later-scope date/month/GroupBy/aggregation work
  with one local-Jupyter starter, six small checksum-pinned synthetic fixtures,
  a dependency-free public checker, and a discoverable instructor-only grader
  and adversarial harness.
- The notebook has exactly 25 stable cells in the accepted order, a portable
  Python 3 kernelspec, null execution counts, no stored outputs, four protected
  cells, and the corrected split between Task 2's pre-observation schema-alignment
  prediction and the final post-observation explanation.
- Task 1 requires explicit grain/key/cardinality/preservation declarations, an
  observed pandas `MergeError` from the unfiltered duplicated station key, the
  supplied current-record rule, and a left many-to-one merge with explicit key,
  validation, indicator, six matched rows, and the single `SP106`/`X` orphan.
- Task 2 requires ordinary-column partition provenance, first-seen schema
  alignment without cleaning its structural missingness, and outer named-index
  feature alignment. Task 3 requires nonaggregating `melt`/`pivot`, exact wide
  reconstruction, and a natural duplicate-long-key `ValueError`. GroupBy and
  aggregation remain deferred to Lecture 08.
- The starter contains only `output/.gitkeep`. `.gitignore` excludes caches,
  checkpoints, bytecode, local environments, and local grader results but does
  not ignore `output/` or CSVs, so all five required artifacts remain visible in
  VS Code Source Control and GitHub Desktop.
- Student dependencies are exactly CPython 3.12.13, NumPy 2.0.2, and pandas
  3.0.3. Local Jupyter is mandatory; the assignment has no Colab badge, upload,
  Drive, network, or external Classroom50 configuration path.

### Implementation validation

- All six committed fixture byte hashes match the accepted manifest:
  `specimens.csv` `26eeae8...605`, `stations_history.csv` `dc6f75e5...08c`,
  batches A/B `1aaa71d0...943` and `8506512a...bc`, reviews
  `d7a1c957...d22`, and wide sensor scores `6eb9bfb9...701`.
- The untouched starter's dependency-free public checker passes environment,
  protected-file, and fixture integrity, then exits nonzero with exactly the
  expected unfinished-notebook and missing-artifact categories. It does not
  execute notebook code or claim a grade.
- The full adversarial harness passes under CPython 3.12.13, NumPy 2.0.2,
  pandas 3.0.3, nbclient 0.10.2, nbformat 5.10.4, and ipykernel 7.1.0 with
  `PYTHONDONTWRITEBYTECODE=1`. A correct disposable submission and a corrected
  resubmission each earn the provisional automated `90/90`; all six reusable
  functions pass alternate in-memory values, row orders, labels, schemas,
  multiple-orphan cases, and duplicate-key cases.
- The grader removes/replaces stale artifacts, clears saved notebook state,
  appends grader-owned checks, fresh-executes flattened and course-root copies
  from unrelated directories whose paths contain spaces, and verifies repeat
  deterministic output. Canonical artifact hashes are
  `1bc33aee...98c`, `78cbd883...666`, `19cb5d07...7e7`,
  `989affb1...6dd`, and `6eb9bfb9...701`.
- The tightened harness rejects 30/30 defective submissions. Cases cover the
  untouched starter; stored-output fakery; stale/missing artifacts; missing,
  corrupt, or extra fixtures; malformed/reordered/missing cells; protected setup
  or checker edits; wrong, absent, implicit, inner, or indicator-free merge
  contracts; manufactured merge/pivot failures; arbitrary or hard-coded station
  selection; input-mutating concat; cleaned schema gaps; positional index
  alignment; aggregating pivot, pre-deleted duplicates, and hard-coded reshape
  order; and remote or absolute paths. The two deliberately canonical-hard-coded
  solutions receive only `50/90` and `67/90`, proving alternate behavior is not
  inferred from canonical artifacts.
- The central command writes `./result.json` with the exact official
  `classroom50/result/v1` top-level metadata, total `score` and hyphenated
  `max-score`, and per-test `test-name`, `passed`, `score`, and `max-score`.
  Captured student-test failures write a valid `0/90` result and exit zero;
  failure detail remains in grader logs.
- Harness execution leaves no notebook checkpoints, Python bytecode caches,
  pytest caches, completed starter artifacts, or `result.json` in the assignment
  package. Static release checks confirm 25 cleared cells, six fixtures, only
  `output/.gitkeep`, absent legacy files, parseable checker/grader/harness source,
  Git-visible required output paths, and a clean scoped `git diff --check`.
- The dependency-free full-course audit parses 47 current notebooks with zero
  errors. Its 27 warnings are existing legacy Classroom references and missing
  kernelspecs in untouched Lecture 07–11 artifacts; none points to Assignment
  06. Classroom50 provisioning, policy mapping for the provisional 90/10 split,
  and any future Colab pilot remain external gates.

### Independent verification

- A reviewer who did not implement Assignment 06 read the accepted blueprint,
  complete student instructions and platform check, all 25 notebook cells, the
  public checker, central grader, adversarial harness, manifest, and exact
  fixture bytes. The three tasks progress from contract-first merge diagnostics
  through the two alignment modes to reversible nonaggregating reshape without
  assessing Lecture 08 or later material.
- The reviewer independently reran the full pinned harness. Correct and
  corrected-resubmission copies earned `90/90`; the untouched starter and all
  30 planted defects were rejected; six alternate-data function contracts,
  flattened/course-root layouts, unrelated working directories, paths with
  spaces, stale-output replacement, repeat determinism, and captured-failure
  recovery all passed.
- The emitted result file has the official `classroom50/result/v1` metadata,
  hyphenated `max-score`, exact per-test fields, and a zero process exit after a
  completed failing grade. The student-facing checker independently exits with
  only the two actionable starter categories—unfinished work and missing
  artifacts—and neither executes code nor claims a grade.
- Exact fixture/environment hashes, portable notebook state and IDs, protected
  cells/files, required artifact schemas and hashes, and scoped syntax/diff
  checks pass. `git check-ignore` confirms that none of the five required CSV
  outputs is hidden, and the released starter contains only `output/.gitkeep`.
  The full-course audit reports 47 notebooks, zero errors, and 27 warnings from
  untouched later legacy surfaces.
- Assignment 06 therefore passes the independent technical assignment gate.
  Its provisional diagnostic score still does not decide the historical
  competence/pass-fail conversion, and real Classroom50 provisioning plus any
  future assignment-Colab save-back path remain external policy/pilot gates.

## Lecture 08 demo rebuild

The three required Lecture 08 notebooks and their guide were rebuilt on
2026-07-18 from the independently accepted
`work/reviews/lecture08_demo_blueprint.md`. This is author-side implementation
evidence; a reviewer who did not implement the package must still run the local
demo gate.

### Artifact changes

- Replaced the legacy advanced GroupBy/crosstab notebooks, remote-computing and
  performance notebook, paired same-stem Markdown, and duplicate live-demo guide
  with exactly three canonical notebooks:
  `demo1_grouping_grain_counts.ipynb`,
  `demo2_named_aggregation_transform.ipynb`, and
  `demo3_aggregating_pivot.ipynb`.
- Added the exact 12-row synthetic encounter fixture with SHA-256
  `24a31904c1371553ff3af627dc21146ed743c8c0c47452ade3628c2fc199c5dc`,
  Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3 records, ignored runtime
  output, portable Python 3 kernelspecs, thirty globally unique stable cell IDs,
  null execution counts, and zero stored outputs.
- Each notebook resolves one demo root from repository-root, direct, nested, or
  standalone launches; reconstructs exact supplied bytes only when the fixture
  is absent; fails before parsing or output cleanup when a present fixture is
  corrupt; and owns only its documented CSV outputs.
- Demo 1 predicts North/South/West group identities and one-facility output
  grain before distinguishing `size`, selected-column `count`, and `nunique`.
  Demo 2 contrasts a three-row flat named aggregation with a twelve-row
  same-index transform, catches an unambiguous three-versus-twelve positional
  assignment failure, and produces the exact eight-row two-key summary. Demo 3
  builds exactly one aggregating pivot, compares all eight populated cells with
  an independently built GroupBy result, and preserves South--Follow-up as the
  only missing cell rather than zero.
- Rewrote `DEMO_GUIDE.md` as a directly actionable learner guide with exact
  launch links, checkpoint outcomes and explanations, fixture/output contracts,
  failure/rehearsal procedures, privacy and scope boundaries, and pending local,
  fresh-Colab, and immutable-badge certification rows. It contains no instructor
  talking points or invented certification measurements.
- Added narrow Lecture 08 package, fixture, dependency, notebook-state, global-ID,
  execution-contract, operation-count, terms-before-use, guide, tracked-output,
  and scope checks to `scripts/course_audit.py`.

### Local implementation validation

- All three notebooks passed progressive execution with warnings promoted to
  errors under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. The first pass
  exposed one pandas 3 schema detail: counting nullable `Int64` ratings returns
  nullable `Int64`. The facility-summary readback now preserves that dtype; no
  value or serialized byte changed.
- Twelve actual fresh-kernel cases passed: all three notebooks from repository
  root, `08/demo/`, `08/demo/data/` as a nested launch directory, and a
  standalone disposable layout. The repeated kernel TCP-encryption warning came
  from the disposable local test harness, not notebook source or execution.
- For every notebook, a missing standalone fixture was reconstructed to the
  exact committed checksum. A present corrupt fixture stopped at checksum
  verification without replacing the fixture or touching corrupt sentinel
  outputs. After fixture restoration, deleted/stale/corrupt owned outputs were
  rebuilt while an unrelated file was preserved.
- Same-platform repeat SHA-256 values were stable:
  `count_comparison.csv` `d32e1bc174e35e2608aaf8c3a9fe13e3aa5c406058a6c1810941927b44afc4ac`,
  `facility_summary.csv` `86c5ff037ab54d0ea48b9012fd9c8512b7ae70d37169ce385299e2890502b0a2`,
  `encounters_with_context.csv` `2021e98069a79dfa871961bdf2cbe8ed89fd40e799c56bd165097b3fc4d4bd76`,
  `facility_service_summary.csv` `c217d21df1e91dbaf30eb4de41f5b70e8240340235fef8a2dc3cf43961fb48aa`,
  and `mean_charge_pivot.csv` `8901d072de4156c7a2bcae4d15d8d0bef878fc11acb8cda7c37413e4ced21cd0`.
- Notebook-format validation, exact package tree, pins/checksum, expected pandas
  values and literal CSV bytes, terms-before-use, Colab/save-back language,
  scope exclusions, no paired Markdown, no tracked/generated output, and
  `git diff --check` pass. The dependency-free course audit currently parses 47
  notebooks with zero errors; its 27 warnings belong to untouched later legacy
  assignments/demos and course navigation.
- All five generated repository outputs were removed after validation. Fresh
  Colab execution and immutable release-tag badges remain publication gates and
  were not claimed. The Lecture 08 demos are ready for independent local QA.

## Assignment 07 design blueprint

Assignment 07 was specified on 2026-07-18 in
`work/reviews/assignment07_blueprint.md` after the Lecture 07 narrative and
demos passed their independent local gates. This is a design record only; no
student assignment source was implemented or accepted by this work.

- The legacy random, mutable-date sales generator, paired Markdown/notebook
  sources, six weakly checked image outputs, broad plotting gallery, joins,
  correlation/heatmap, GroupBy, rolling/resampling, dashboard work, and GitHub
  Classroom workflow are replaced in the design rather than preserved.
- The new contract has exactly three incremental competence tasks in one clean-
  local-Jupyter notebook: question/grain/roles plus bounded seaborn exploration;
  critique and repair of one executable supplied flawed bar chart; and one
  audience-specific accessible annotated Matplotlib explanation.
- Three assignment-only prepared CSVs have exact bytes and SHA-256 values:
  `format_completion.csv`
  `20ad900633154f5f3a2c09cfbc2f890f8423da0897d6345841745332110be66a`,
  `session_observations.csv`
  `fc4d69ab836288a2fe9c505c65c08413e137e51ab1914cd0e350f6e6636da096`,
  and `pathway_checkpoints.csv`
  `ec9a336b7fb97418a6f058704f2509c8cee6b13d744efb7a6e3e99224ef8c258`.
- The implementation contract fixes 23 cell IDs, three reusable chart
  functions, exactly two PNGs, exact supporting CSV/JSON/text evidence,
  GUI-visible outputs, protected surfaces, a dependency-free public checker,
  fresh central alternate-data grading, and the official hyphenated
  `classroom50/result/v1` `result.json` shape.
- The provisional diagnostic separates 80 executable automated points from 20
  human communication/visual points. Automation is explicitly prohibited from
  claiming honesty, accessibility, clarity, or audience fit from object/file
  proxies alone.
- Cleaning, joining/reshape, grouping/aggregation, time series,
  correlation/regression/inference/modeling, random/mutable/network data, and
  dashboard/interactive breadth remain outside Assignment 07. The terms ledger
  places definitions before independent use and preserves the Lecture
  06→07→08 boundary.
- Classroom50 production identifiers, pass/fail or grade conversion, human-score
  integration, late/resubmission/retention rules, and any future assignment
  Colab path remain policy or pilot gates rather than inferred capabilities. No
  timing claim was introduced.

### Assignment 07 independent design verification

An independent reviewer who did not author the blueprint rechecked it on
2026-07-18 against the accepted Lecture 07 narrative/demos, course dependency
map, review workflow, Colab boundary, Classroom50 pilot, and prior assignment
grader patterns. This verifies the design contract only; `07/assignment/` was
not implemented or accepted by this review.

- All three literal CSV payloads independently reproduce the stated final LF,
  98/309/181-byte sizes, SHA-256 values, pandas rows, changes, and nine-point
  canonical final gap.
- A pinned headless prototype on Python 3.12.13, NumPy 2.0.2, pandas 3.0.3,
  Matplotlib 3.10.8, seaborn 0.13.2, and Pillow 12.3.0 produced both contracted
  chart shapes within the PNG bounds and byte-identical hashes on a repeat run.
- The gate tightened the public PNG claim to structural proxies, made
  headless-grader ownership explicit, defined complete alternate-table
  preconditions and result-direction variation, froze a feasible AST call-name
  denylist, and required fresh human-review artifacts to leave the disposable
  execution tree through a grader-owned location.
- The revised blueprint passes the independent design gate: its tasks build
  contract → bounded exploration → critique/repair → explanation; terms precede
  independent use; exactly two PNGs and three evidence files remain GUI-visible;
  the fresh central result shape and human/automation boundary remain explicit;
  and no timing, unconditional assignment-Colab, Lecture 08+, cleaning, join,
  time-series, correlation, inference, or modeling requirement was introduced.

### Assignment 07 author implementation evidence

The accepted Assignment 07 blueprint was implemented on 2026-07-18. This is
author-side implementation and test evidence, not the required independent
acceptance review.

- The legacy paired source, generators, random sales/customer/product workflow,
  six legacy chart names, and complete GitHub Classroom workflow/test tree were
  removed atomically. The student surface is now the one 23-cell starter,
  instructions/platform guide, standard-library public checker, four pinned
  runtime records, three prepared fixtures plus manifest, and only
  `output/.gitkeep`; instructor-only grader assets remain under
  `_grader_selftest/`.
- Fixture bytes and hashes match the independently accepted contract:
  `format_completion.csv`
  `20ad900633154f5f3a2c09cfbc2f890f8423da0897d6345841745332110be66a`,
  `session_observations.csv`
  `fc4d69ab836288a2fe9c505c65c08413e137e51ab1914cd0e350f6e6636da096`,
  `pathway_checkpoints.csv`
  `ec9a336b7fb97418a6f058704f2509c8cee6b13d744efb7a6e3e99224ef8c258`,
  and manifest
  `1c3397cb2d98ae239f6a7cd254bb3aa9980d94cd23af4546c834a9262de0a28c`.
- The released notebook is nbformat 4.5 with exactly 23 globally unique
  prescribed IDs, portable Python 3 metadata, null execution counts, zero
  stored output, six frozen supplied cells, and actionable but incomplete
  student scaffolds. The supplied setup selects Agg before pyplot, validates
  every byte before pandas reads, deletes only five named stale artifacts, and
  uses an in-memory Jupyter display helper so teaching figures remain visible.
- `check_assignment.py` has SHA-256
  `9ca5df48763ccb048a49195c86941535ecfb9f91044bd19bf5ea5568973332b1`.
  It imports only the standard library, never executes notebook source, and
  independently checks fixture/notebook/protected-cell/source-scope/output/PNG-
  header/CSV/JSON/text contracts without assigning points or claiming visual
  quality.
- The independent central-grader candidate owns its own protected hashes and
  exact eight-package environment. It requires all official metadata plus an
  external grader-owned review directory; fresh-executes stripped disposable
  copies from flattened, course-root, relocated, nested-CWD, and path-with-
  spaces layouts; exercises canonical and alternate labels, reversed leader,
  and tie behavior; uses marker-geometry deduplication and Pillow decode; runs a
  second genuine kernel; and persists a fresh executed notebook plus two fresh
  PNGs outside the disposable tree before cleanup.
- The pinned reference submission scored 80/80 with exact test maxima
  10/15/25/25/5, while the untouched starter scored 0/80. Static adversarial
  cases rejected every protected-cell/file mutation, missing/stale artifact,
  malformed/duplicate/reordered/type-changed notebook, fixture inventory/byte
  mutation, extra/legacy output, submission sentinel, stored-output/broken-
  source case, and sampled APIs from every excluded family. Behavioral cases
  rejected dropped points, one encoding, truncated baseline, wrong legend,
  hard-coded leader, wrong tie target, and input mutation; a corrected
  resubmission returned 80/80. Official success/failure result topology,
  context passthrough, student-failure exit zero, missing-context nonzero, and
  durable review-bundle survival were exercised.
- On the pinned image, fresh exported proxies were 1,158×632 and 58,236 bytes
  for the critique PNG and 1,029×678 and 74,515 bytes for the explanatory PNG.
  Both decoded under Pillow, repeated byte-for-byte in fresh kernels on that
  image, and were manually inspected for gross clipping/overlap; these values
  are maintenance evidence rather than cross-platform student hash contracts.
  The first explanatory prototype's callout overlapped its title; the reference
  implementation moved it inside the plot and made tie targeting observable by
  matching annotation/arrow color to the selected path before the behavior gate
  passed.
- The dependency-free course audit now enforces the narrow Assignment 07
  starter, fixtures, protected cells, pins, instructor assets, local/GUI/
  Classroom50 workflow language, and no timing/Colab-badge boundary. It parses
  the repository with zero errors; remaining warnings concern untouched later
  legacy course surfaces. No Assignment 07 generated artifact remains in the
  released starter.

### Assignment 07 independent implementation verification

An independent reviewer who did not author `07/assignment/**` completed the
technical acceptance gate on 2026-07-18. **Result: PASS with no further source
correction.** Classroom50 production policy, human-score integration, and any
future assignment-Colab route remain separate pilot gates.

- The full release self-test was rerun under the exact Python 3.12.13 grader
  environment and exited zero. The reference and corrected submissions scored
  80/80; the starter and every protected-file, notebook-structure, fixture,
  output, scope, and behavioral mutant remained below full credit. Official
  success and student-failure result files use the exact
  `classroom50/result/v1` topology and hyphenated `max-score`; completed student
  grading exits zero while missing required runner context remains an
  infrastructure failure.
- A separate fresh reference submission was materialized and graded outside
  the harness's disposable tree. It scored 80/80 and produced a durable
  grader-owned bundle containing the executed notebook, manifest, and two
  freshly generated PNGs after central execution cleanup.
- Both fresh review PNGs were inspected at original resolution. The critique
  redesign has an honest zero baseline, explicit percentage unit, exact value
  labels, redundant hatch encoding, and an unclipped external legend. The
  explanatory chart has distinct color/marker/line encodings, readable title
  and axes, a visible nine-point final-gap annotation contained within the
  plot, and no gross clipping or overlap. This is a human visual check, not a
  cross-platform image-hash contract.
- The public checker was independently run on the untouched starter and failed
  only for the intended unfinished work and absent required artifacts. The
  five required submission outputs are not ignored, while the released output
  directory contains only `.gitkeep`; no cache, checkpoint, result, or generated
  review artifact remains in the student package.
- Scoped `git diff --check` and the course audit passed. The audit parsed all 44
  current notebooks with zero errors; its 20 warnings concern untouched legacy
  Assignment 09+ and Lecture 10–11 surfaces. Assignment 07 therefore passes its
  narrative, demo, design, implementation, adversarial-grader, visual-review,
  and repository-hygiene technical gates.

## Independent Lecture 08 demo verification

An independent reviewer who did not author `08/demo/**` completed the local
demo gate on 2026-07-18. **Result: PASS with no functional correction.** The
guide's independent-local certification row now contains execution evidence;
fresh Colab execution and immutable release-tag badges remain pending.

- The accepted matrix used 24 fresh candidate-kernel executions: all three
  notebooks ran from repository root, `08/demo/`, nested `08/demo/data/`, and a
  disposable standalone package, first normally and then with notebook-code
  warnings promoted to errors. All passed. The local kernel manager emitted its
  transport warning outside notebook code; the notebooks themselves emitted no
  warning or deprecation under the strict matrix.
- An additional 18 fresh-kernel adversarial cases covered every notebook with a
  missing fixture, deleted output, plausible stale CSV, binary-corrupt output,
  deterministic repeat, and present corrupt fixture. Missing fixtures were
  reconstructed to checksum
  `24a31904c1371553ff3af627dc21146ed743c8c0c47452ade3628c2fc199c5dc`.
  Present corrupt fixtures failed before parsing, replacement, or output
  cleanup. Each notebook replaced only its owned files and preserved an
  unrelated output sentinel.
- Independent pandas recomputation verified the three observed facility groups,
  12-row conservation, exact `size`/`count`/`nunique` values, the flat named
  aggregation, same-index 12-row transform, exact eight ordered two-key groups,
  and all eight GroupBy-to-pivot cell equalities. South--Follow-up remained the
  sole missing pivot cell and was never serialized or read back as zero.
- All five CSVs passed exact byte, SHA-256, newline, schema-aware dtype, column,
  row-order, value, path, and repeat-hash checks. Their independent hashes match
  the author record immediately above.
- The exact package tree, pins, fixture, portable kernelspecs, null execution
  state, zero stored outputs, and 30 globally unique cell IDs passed. No L08 ID
  collides with an ID in another repository notebook. The guide uses actionable
  learner checkpoints and official development Colab links without implying
  save-back; generated demo output is ignored and untracked.
- Grouping unit, output grain, and GroupBy object are defined before the first
  grouped operation; named aggregation and `transform` precede their first
  code use; structural `pivot` is contrasted with aggregating `pivot_table` and
  all five pivot choices are named before construction. The progression is
  cumulative and preserves the accepted Lecture 07→08→09 dependency boundary.
- Required code contains no cleaning/imputation, join or new structural reshape,
  group filtering or `GroupBy.apply`, advanced MultiIndex/crosstab work,
  statistics/modeling, time-series operation, plotting objective, remote or
  performance workflow, network/upload/Drive path, randomness, or mutable date.
  Exclusion-only guide text and the separate optional-terminal-lab boundary are
  not teaching uses.
- After generated outputs were inspected, all five were removed. The dependency-
  free course audit and scoped `git diff --check` passed; its remaining warnings
  belong to untouched later legacy surfaces and do not weaken this L08 gate.

## Assignment 08 design blueprint

Assignment 08 was specified on 2026-07-18 in
`work/reviews/assignment08_blueprint.md` after the Lecture 08 narrative and all
three demos passed their independent local gates. This is a design record only;
no `08/assignment/**` source was implemented or accepted by this work.

- The evidence audit covered the complete legacy Assignment 08 package, accepted
  Lecture 08 narrative/bonus/demo contract and verification evidence, the
  Lecture 07→08→09 dependency boundary, course workflow, Colab and Classroom50
  platform records, historical `070f7b6:CLAUDE.md`, prior Assignment 04–07
  blueprint patterns, and the course's McKinney GroupBy source notes.
- The legacy merge-at-ambiguous-grain start, random second generator notebook,
  paired Markdown, broad dependencies, advanced apply/filter/MultiIndex/
  crosstab/reshape/visualization/performance work, weak file-only tests, and
  mutable GitHub Classroom test fetch are replaced rather than retained.
- The new one-table design fixes three cumulative competence tasks: predict
  input/group/output grain and select `size`/`count`/`nunique`; produce flat
  named aggregation, same-index `transform`, and one bounded two-key result; and
  build exactly one aggregating pivot whose populated cells equal GroupBy and
  whose absent Harbor–Phone combination remains missing rather than zero.
- The assignment-only 469-byte synthetic support-request CSV has SHA-256
  `a9136161332c5da9f8f1251d869bbd014ed762751675fb757f81a79cff5352d6`;
  its exact 624-byte manifest has SHA-256
  `b2fee1c48fb678b81318d2f085c42e2f9b480bd6c4eed6f07ef118b9bfd70860`.
  Pinned pandas computation established all canonical counts, totals, means,
  transform values, eight observed two-key groups, and the sole missing pivot
  cell.
- The contract fixes 25 cell IDs, five pure public functions, five GUI-visible
  deterministic CSVs, protected setup/final verification, exact artifact bytes
  and hashes, a dependency-free public checker, fresh central Classroom50
  execution, the official hyphenated `classroom50/result/v1` result shape, and a
  disclosed alternate table with shuffled nondefault index and different labels,
  values, group sizes, category levels, and absent combination.
- A release self-test must cover canonical, alternate, path-layout, fixture,
  stale/repeat output, notebook-integrity, hard-coding, operation-choice,
  transform-alignment, pivot-equivalence, result-shape, and corrected-
  resubmission cases. Terms are defined before independent use; joins, cleaning,
  new structural reshape, apply/filter/MultiIndex/crosstab, visualization, time,
  statistics/modeling, remote/performance/network, random, and mutable-date work
  remain excluded.
- The provisional diagnostic keeps 90 automated points technically separate
  from 10 human reasoning points and does not infer a historical pass/fail
  threshold or grade conversion. Production Classroom50 identifiers, human-score
  integration, late/resubmission/regrade/retention rules, and any future
  immutable Assignment 08 Colab release remain explicit policy or pilot gates.
  No timing claim or unconditional Colab badge was introduced.
- Implementation remains gated on independent design review; this author did not
  modify the legacy assignment package or claim implementation acceptance.

## Lecture 09 demo design blueprint

Lecture 09 demos were specified on 2026-07-18 in
`work/reviews/lecture09_demo_blueprint.md` after the core/bonus narrative and
Lecture 08→09 boundary passed independent verification. This is a design record
only; the legacy `09/demo/**` package was not modified or accepted by this work.

- The legacy random and mutable-date medical generators, paired
  Markdown/notebook sources, notebook-form guide, broad plotting and seasonality
  survey, EWM/centered/expanding work, and unnecessary stats/visualization/
  Jupytext dependencies are replaced in the design rather than preserved.
- The exact three-notebook progression is structure and timezone-aware
  preparation → measurement-aware grouped frequency change with missingness
  provenance → entity-scoped past-only windows, availability rejection, and a
  chronological Lecture 10 handoff. No chart is required.
- The shared synthetic 310-byte station fixture has SHA-256
  `57dcdb82372805cf1dda83a7c227b463fe997cf1437275d64d01b9719ff26b54`.
  Pinned pandas 3.0.3 computation verified ten source rows, one source-missing
  value, four grid-created hourly rows, eight two-hour station bins, the exact
  21.5-versus-22.0 window contrast at South 21:00 UTC, and a seven/three-row
  chronological split with both entities retained.
- The handoff fixes exact package, pins, literal fixture, six CSV outputs,
  values, path resolution, checksum/corruption behavior, deterministic repeat,
  portable notebook state and global IDs, direct learner guide language,
  development badges, and an adversarial independent QA matrix.
- Pooled entities, automatic fill/interpolation, negative shifts/leads,
  centered/custom/EWM/expanding windows, advanced DST, decomposition,
  forecasting/modeling/statistics, remote/network/random/mutable data, and new
  visualization instruction remain excluded. Terms precede unrestricted use.
- Independent local execution, fresh Colab execution, canonical immutable
  release badges, and the course-wide version freeze remain publication gates.
  Assignment 09 and any unconditional assignment-Colab route remain outside
  this design. No timing claim was introduced.

## Independent Lecture 09 demo design verification

An independent reviewer who did not author the Lecture 09 demo blueprint
completed the design gate on 2026-07-18. **Result: PASS after one narrow
output-dtype clarification.** This accepts the technical design for
implementation; it does not certify the still-legacy files under `09/demo/**`.

- Under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3, the literal fixture is
  exactly 310 bytes with SHA-256
  `57dcdb82372805cf1dda83a7c227b463fe997cf1437275d64d01b9719ff26b54`.
  Exact-format parsing, one localization, one UTC conversion, and stable
  station/time sorting reproduce the declared string, UTC
  `datetime64[us, UTC]`, `float64`, and `int64` source dtypes and all ten
  prepared rows.
- The entity-time key is unique; time is monotonic within each station; each
  station has gaps `[1, 2, 1, 2]` hours and is irregular. Entity-scoped hourly
  `asfreq()` produces 14 rows, exactly four grid-created rows and the distinct
  North 19:00 UTC source-value-missing row. Grouped two-hour resampling produces
  the specified eight bins and conserves all ten source readings.
- Entity-scoped lag, difference, and past-only window calculations reproduce
  every declared value, including South 21:00 UTC's `21.5` observation-count
  mean versus `22.0` elapsed-time mean and South 22:00 UTC's `22.5` versus
  `23.0`. The availability inventory is exactly keep, keep, reject, reject, and
  the global 21:00 UTC cutoff yields seven earlier and three later rows with
  both entities in both blocks and strict chronological separation.
- The six reference serializations have exact `(bytes, SHA-256)` pairs:
  prepared `(431, a9e2b75c2f4e9f9a3778b53cd87e68d4a559511c368cad80b7153b1109a987ba)`,
  hourly `(788, 7054dfb410b36f35ef53ff4e02cc77fb633ff413e1dcd9f66d1807674053b40e)`,
  two-hour `(361, 0558659b66336e71c3c67769097aadf4e2616a2d4f913425bb498463528a9d6f)`,
  features `(633, 5a6524e8dbb37da3cc056cc648e5ff444c12cb485214bef1a95c3a67a22af3ab)`,
  availability `(326, d4125def8dcf8e23b9f33574f1dd9e14a5ed3f92889f88b455509110ad87e505)`,
  and blocks `(535, 7ea9752756ef882dbe19318bfbb1614c33ff6bbca45a5bbc5effe4bcad065a67)`.
- The narrow clarification enumerates exact dtypes for all six output frames.
  In particular, hourly reindexing inserts missing source markers and therefore
  promotes `source_row` from NumPy `int64` to `float64`; the blueprint now makes
  this expected result explicit instead of leaving implementation and readback
  assertions ambiguous.
- The exact package remains three clean-state, globally-IDed notebooks, one
  direct-action guide, one committed fixture, and six ignored generated CSVs.
  Colab-first/local-equivalent path discovery, missing-versus-corrupt fixture
  behavior, owned-output cleanup, unrelated-output preservation, repeat-byte
  stability, conditional assignment-Colab language, and development-to-
  immutable badge handling are specified without a timing claim.
- The verified Lecture 08 entry and Lecture 10 exit boundaries hold. Required
  code does not pool entities or introduce obsolete aliases, automatic fill or
  interpolation, advanced windows, advanced DST, forecasting, modeling,
  statistics, network or mutable data, or visualization. Fresh local notebook
  execution, fresh Colab execution, immutable release badges, and the
  course-wide version freeze remain implementation/publication gates.

## Independent Assignment 08 design verification

An independent reviewer who did not author the Assignment 08 blueprint completed
the design gate on 2026-07-18. **Result: PASS after four narrow feasibility
corrections.** This accepts the technical design for implementation; it does not
certify any still-legacy file under `08/assignment/**`.

- The exact support-request CSV, fixture manifest, and five canonical output
  blocks were reconstructed directly from the blueprint. All seven byte counts,
  final-LF contracts, and SHA-256 values match. Fresh pandas 3.0.3 computation
  also reproduced the declared input/category dtypes, nullable `Int64`
  satisfaction counts, group identities/order, all counts/totals/means, the
  15-row same-index transform, eight two-key groups, five output byte streams,
  GroupBy-to-pivot equality, and Harbor--Phone as the sole missing pivot cell.
- The disclosed alternate table reproduced Metro/Coast/Hill row counts
  `[4, 3, 3]`, satisfaction counts `[3, 2, 3]`, distinct-agent counts
  `[2, 2, 2]`, totals `[74, 105, 180]`, means `[18.5, 35.0, 60.0]`, the exact
  shuffled index, eight ordered two-key groups, the 3-by-3 pivot, and sole
  missing Coast--Voice cell. All populated alternate pivot cells equal the
  reusable two-key GroupBy result.
- The notebook topology has exactly 25 unique prescribed cell IDs, five public
  function interfaces, five tracked GUI-visible CSV artifacts, one protected
  definition ledger before independent use, one and only one student-editable
  `pd.pivot_table` call, and a cumulative count → aggregation/transform/two-key
  → pivot/equivalence progression. Explicit `observed=True`, `sort=True`,
  `dropna=True`, and deliberate `as_index=False` policies match the verified
  Lecture 08 narrative and demos.
- Removed the infeasible requirement to execute the notebook from a genuinely
  unrelated external working directory. The protected setup deliberately
  searches upward from the kernel working directory, while Classroom50 invokes
  the grader from the student checkout. Flattened, course-root, relocated,
  nested-within-assignment, and path-with-spaces layouts remain required.
- Clarified the unrelated-output case: setup must preserve a foreign sentinel,
  while final submission inventory must reject that extra file until removed.
  This tests non-destructive setup without weakening the exact five-output
  submission contract.
- Replaced vague retention of an executed notebook outside the disposable tree
  with a feasible human-review path: automated correctness comes from the fresh
  disposable execution, and the bounded human rubric reads student-authored
  Markdown through Classroom50's supplied `review` URL. Submitted stored output
  remains untrusted.
- Rechecked the official `classroom50/result/v1` contract. The grader must emit
  the seven context metadata values, total `score` and hyphenated `max-score`,
  and per-test `test-name`, `passed`, `score`, and `max-score`; completed grading
  exits zero even when tests fail. `owner` and `assignment_type` remain optional
  in grader output because the runner stamps them authoritatively, and the
  optional `submitted_by` object is runner-owned. Validation may not require the
  optional fields from the grader or let grader/student code invent them.
- Automation is limited to executable structure and behavior; the 10-point
  grain/count, aggregate/transform, pivot, and privacy/readability judgment stays
  human. The 90 automated points remain separate from that review and from the
  unresolved historical pass/fail conversion.
- No cleaning/imputation decision, join, new structural reshape, visualization,
  time-series operation, statistics/modeling, advanced GroupBy/MultiIndex/
  crosstab work, remote/performance workflow, random/network/mutable source, or
  timing claim entered required scope. Clean local Jupyter remains mandatory;
  production Classroom50 identifiers, score-policy integration, late/regrade/
  retention rules, and assignment Colab save-back/release remain explicit policy
  or pilot gates.

The blueprint now passes the independent design gate. Implementation may build
the exact package, but a different reviewer must still fresh-execute and
adversarially verify the resulting starter, checker, central grader, and
`result.json` success/failure paths before Assignment 08 is accepted.

## Lecture 09 demo implementation evidence

The exact three-notebook Lecture 09 package was implemented on 2026-07-18 from
the independently accepted `work/reviews/lecture09_demo_blueprint.md`. This is
author-side local evidence only; a different reviewer must still perform the
independent demo gate.

- The legacy random/mutable medical notebooks, paired Markdown sources,
  notebook-form guide, plotting/seasonality breadth, and broad dependencies were
  removed. The resulting exact package contains one direct-action guide, pins
  for Python 3.12.13/NumPy 2.0.2/pandas 3.0.3, one 310-byte checksum-pinned
  synthetic fixture, and exactly three clean-state notebooks with 29 globally
  unique cell IDs.
- One narrow assertion/guide correction was required during the first fresh
  kernel run. The accepted evidence had generalized North's gap order
  `[1, 2, 1, 2]` to both stations; the literal fixture gives South
  `[2, 1, 2, 1]`. Both histories still contain only one- and two-hour gaps and
  are irregular. The fixture, pedagogy, downstream temporal values, and all six
  accepted output hashes were unchanged.
- The author matrix completed 24 fresh candidate-kernel executions: all three
  notebooks ran normally and with notebook-code warnings promoted to errors
  from repository root, `09/demo/`, a nested `09/demo/data/` directory, and a
  disposable standalone package. A kernel transport warning and an ipykernel
  shutdown-only pending deprecation were outside notebook code; a strict rerun
  of all 12 layout/notebook pairs reset the in-memory warning filter only after
  the complete notebook source and passed with zero notebook-code warnings.
- Fifteen successful adversarial executions covered each notebook with a
  missing fixture, deleted output, binary-corrupt output, stale text output, and
  deterministic repeat. Three additional executions rejected a present corrupt
  fixture before parsing, replacement, or owned-output cleanup. Missing
  fixtures reconstructed to 310 bytes and SHA-256
  `57dcdb82372805cf1dda83a7c227b463fe997cf1437275d64d01b9719ff26b54`;
  every case preserved an unrelated output sentinel.
- External schema-aware readback reproduced the corrected exact dtypes and all
  six accepted `(bytes, SHA-256)` contracts: prepared `(431,
  a9e2b75c2f4e9f9a3778b53cd87e68d4a559511c368cad80b7153b1109a987ba)`,
  hourly `(788,
  7054dfb410b36f35ef53ff4e02cc77fb633ff413e1dcd9f66d1807674053b40e)`,
  two-hour `(361,
  0558659b66336e71c3c67769097aadf4e2616a2d4f913425bb498463528a9d6f)`,
  features `(633,
  5a6524e8dbb37da3cc056cc648e5ff444c12cb485214bef1a95c3a67a22af3ab)`,
  availability `(326,
  d4125def8dcf8e23b9f33574f1dd9e14a5ed3f92889f88b455509110ad87e505)`,
  and blocks `(535,
  7ea9752756ef882dbe19318bfbb1614c33ff6bbca45a5bbc5effe4bcad065a67)`.
- Runtime assertions cover entity/time uniqueness and ordering, exact timezone
  dtypes, one-versus-two-hour irregularity, 14 hourly rows with four
  grid-created and one source-value-missing row, eight measurement-aware
  two-hour bins, entity-scoped lag/difference/window values, the exact
  keep/keep/reject/reject availability inventory, and the seven/three-row
  chronological split with both stations retained.
- Narrow Lecture 09 demo enforcement was added to `scripts/course_audit.py` for
  the exact tree, pins, fixture/output hashes, direct-action guide, notebook
  metadata/state/global IDs, explicit pandas policies and operation counts, and
  advanced/later-scope exclusions. The dependency-free audit passed with 44
  notebooks, zero errors, and 20 warnings on untouched later/legacy surfaces.
- All six generated repository outputs and temporary execution artifacts were
  removed after validation. Fresh Colab execution, immutable release-tag
  badges, the course-wide version freeze, and independent local verification
  remain pending and are not claimed here.

## Independent Lecture 09 demo implementation verification

An independent reviewer who did not author the Lecture 09 demo implementation
completed the local implementation gate on 2026-07-18. **Result: PASS with no
notebook or teaching-content correction.** The only edits from this gate record
the completed local lifecycle state in the guide and require that state in the
Lecture 09 audit. This result does not certify Colab or a release badge.

- The reviewer read the accepted blueprint, verified core and bonus narratives,
  Lecture 08 entry and Lecture 10 exit contracts, historical `CLAUDE.md`, all
  eight package files, the Lecture 09 audit rules, and the author evidence. The
  exact package, pins, 310-byte fixture and checksum, three-notebook progression,
  learner-directed guide, and six-output ownership contract all match.
- Twenty-four independent fresh-kernel executions covered all three notebooks
  from repository root, `09/demo/`, nested `09/demo/data/`, and a disposable
  standalone package, first normally and then with notebook-code warnings
  promoted to errors. All passed under Python 3.12.13, NumPy 2.0.2, pandas
  3.0.3, nbclient 0.10.2, nbformat 5.10.4, and ipykernel 6.29.5; no notebook-code
  warning was emitted.
- Fifteen further fresh-kernel executions covered each notebook with a missing
  fixture, deleted output, plausible stale CSV, binary-corrupt output, and a
  deterministic repeat. Three corrupt-fixture executions failed at checksum
  verification before parsing, fixture replacement, or owned-output cleanup.
  Every successful missing-fixture case reconstructed the exact 310 bytes, and
  all cases preserved an unrelated binary output sentinel.
- Independent schema-aware reads reproduced all six accepted dtypes, byte
  counts, hashes, row and column orders, and values. The reviewer separately
  verified UTC localization/conversion, entity-time uniqueness, North gaps
  `[1, 2, 1, 2]`, South gaps `[2, 1, 2, 1]`, 14 entity-hour rows with four
  grid-created and one source-missing row, eight two-hour bins conserving ten
  readings, and the exact lag/difference and observation-count-versus-elapsed-
  time window values.
- The availability table is exactly keep, keep, reject, reject without computing
  the rejected future candidates. The chronological cutoff yields seven earlier
  and three later rows, retains both stations in both blocks, and has maximum
  earlier timestamp 20:00 UTC strictly before minimum holdout timestamp 21:00
  UTC.
- The three clean notebooks contain 8, 10, and 11 cells with 29 globally unique
  IDs, portable Python 3 kernelspecs, null execution counts, zero stored output,
  and compiling code. Terms precede their required operations. The guide has
  three explicit development badges, direct learner actions, correct visible
  results and hashes, no instructor/meta or lesson-duration language, and now
  records this independent local PASS.
- Required code keeps every temporal operation station-scoped and contains no
  automatic fill/interpolation, negative shift or computed lead, centered/
  custom/EWM/expanding window, advanced DST, obsolete pandas time alias,
  forecasting/modeling/statistics, visualization, network/runtime data,
  randomness, mutable date, upload/Drive, or credential path.
- The dependency-free course audit passes with 44 notebooks and zero errors.
  Its 20 warnings are confined to untouched legacy Classroom references and
  missing kernelspecs in Lecture 10--11 artifacts. Diff/whitespace checks pass;
  no generated Lecture 09 output is tracked, and final verification cleanup
  removed generated CSVs, notebook checkpoints, and Python caches.

Fresh Colab execution, replacement of all three development badges with one
immutable release tag, and the course-wide version freeze remain pending
publication gates. Assignment 09 is outside this verification.

## Assignment 08 implementation evidence

Assignment 08 was implemented on 2026-07-18 from the independently accepted
`work/reviews/assignment08_blueprint.md`. This is author-side local evidence;
a different reviewer must still perform the independent assignment gate.

- The legacy GitHub Classroom workflow and mutable remote test fetch, random
  data generator notebook/Markdown pair, duplicated assignment Markdown,
  `DATA_SCHEMA.md`, `TIPS.md`, broad dependencies, and advanced legacy task
  surface were removed atomically. The new student package has one canonical
  notebook, exact runtime records, local platform guide, public checker, one
  fixture/manifest pair, and a visible `output/.gitkeep`; the instructor-only
  self-test directory is excluded from student templates and submissions.
- The released starter has exactly 25 accepted unique cell IDs, notebook format
  4 minor 5, a portable Python 3 kernelspec, 16 code cells with null execution
  counts and zero stored outputs, six protected cells, actionable TODO
  scaffolds, and terms before independent use. It requires clean local Jupyter,
  contains no Colab badge or timing claim, and leaves only `.gitkeep` in the
  repository output directory.
- The assignment-only synthetic fixture is exactly 469 bytes with SHA-256
  `a9136161332c5da9f8f1251d869bbd014ed762751675fb757f81a79cff5352d6`;
  its exact 624-byte manifest has SHA-256
  `b2fee1c48fb678b81318d2f085c42e2f9b480bd6c4eed6f07ef118b9bfd70860`.
  Protected setup validates both before output cleanup, supports feasible
  flattened/course-root/nested/relocated/path-with-spaces launches, removes only
  the five owned artifacts, and preserves unrelated files.
- A disposable completed notebook reproduced the five accepted GUI-visible CSV
  contracts byte-for-byte: count summary
  `0735d0647dbbe2199b1de03e1061bf6c3a7a9d15bb553d128bdc1ab295ef2f36`,
  center summary
  `6c528bd229cd0ce2db2f4c90f09fd2a9ba670fb3aa659951bc113d70a33afad4`,
  context
  `391d56794e1537244c8d0b97f39e25e822b1e54d45b049fca98760ba646b1a7a`,
  two-key summary
  `41b74a8dac05eff1695e6b972b360bd2b1730e77f5e2060e42801533a07180da`,
  and pivot
  `1274782fc4e773bfd572736c0af106842d92751d672b6ad341207574e636dedf`.
- The dependency-free public checker validates environment/protected sources,
  exact fixture and notebook topology, bounded AST scope, all five function
  interfaces, explicit grouping policies, one and only one `pd.pivot_table`,
  the student-authored cell-for-cell comparison, and exact output inventory/
  bytes without executing code or claiming a score or prose judgment.
- The independent central grader does not import that checker. It clears stored
  state and outputs in disposable copies, fresh-executes the notebook, checks
  canonical dtypes/values/artifacts, and calls all five pure functions on the
  disclosed shuffled-index alternate table. Alternate counts, totals, means,
  transform index alignment, eight two-key groups, sole Coast--Voice missing
  cell, and all eight GroupBy-to-pivot equalities passed.
- The grader emits the five automated groups totaling 90 through the official
  hyphenated `classroom50/result/v1` fields. Correct and corrected submissions
  scored 90/90; completed failing grades wrote `result.json` and exited zero.
  Grader output omits optional `owner`, `assignment_type`, and `submitted_by`;
  the harness separately accepts those only when authoritatively runner-stamped.
  Human review remains outside `result.json` and uses the context-supplied
  `review` URL to inspect student-authored Markdown.
- The final adversarial harness passed with 45 rejected cases spanning untouched
  and broken stored-output submissions; missing/corrupt/line-ending-changed/
  extra fixtures and manifest edits; sentinel preservation versus final extra-
  file rejection; stale/deleted/legacy outputs; malformed/protected/reordered/
  duplicate-cell/checker edits; count-operation and hard-coding errors; implicit
  policies; named-aggregation, transform, two-key, and pivot defects; missing
  comparison; structural pivot; and join/apply/visualization/time/modeling/
  remote/random/absolute-path scope violations. Deterministic repeat and
  corrected resubmission also passed.
- Two narrow author-side enforcement corrections were required: network source
  scanning now distinguishes the valid `support_requests` variable from actual
  `requests.get/post/request/session` calls, and the central count-function check
  now rejects canonical label literals and function-local file I/O just as the
  other four functions do. No pinned Python, NumPy, pandas, nbclient, nbformat,
  ipykernel, notebook, fixture, output, or pedagogical contract correction was
  needed.
- The dependency-free course audit passed with 44 notebooks, zero errors, and 20
  warnings confined to untouched later/legacy surfaces. Scoped whitespace,
  protected-hash, starter-state, syntax, and diff checks passed. The exact
  `08/assignment/__pycache__/check_assignment.cpython-312.pyc` created by an
  early compile was listed and removed; no cache or generated CSV remains.

## Lecture 10 narrative revision

The Lecture 10 core and bonus narrative were rebuilt on 2026-07-18 from the
accepted modeling/evaluation outline in
`work/reviews/lectures_08_11_alignment.md`, the reconciled dependency map, and
the independently verified Lecture 09 handoff. The requested `10/LECTURE.md`
path did not exist; `10/README.md` is the canonical lecture narrative used by
the repository and was confirmed as the intended target. Legacy demo notebooks,
their guide/paired Markdown, assignment sources/tests, requirements, media,
generated HTML, navigation, and Lecture 11 were intentionally left unchanged
for later workflow stages.

### Author-side content changes

- Reframed the core around the five accepted measurable objectives: question
  classification and claim; bounded OLS association and uncertainty; target,
  timestamp, horizon, cutoff, and availability; split roles, train-only
  preprocessing, Pipeline, and leakage; and baseline/metric comparison,
  supplied binary-metric literacy, one final test evaluation, and limitations.
- Ordered and defined the required vocabulary before demanding use: model,
  descriptive/inferential/predictive question, sample/population/estimand,
  prediction target, association/causation, OLS response/explanatory variable/
  intercept/coefficient/fitted value/residual, assumptions, standard error and
  both interval meanings, prediction timestamp/target timestamp/horizon/
  feature/cutoff/availability, training/validation/test, exchangeability, split
  manifest, four leakage types, estimator/fit/predict/preprocessing/Pipeline/
  baseline, MAE/RMSE/R², evaluation, and binary positive-class metrics.
- Replaced the causal “why” framing with conditional noncausal language and a
  clear boundary: observational coefficient precision does not supply causal
  identification. The OLS path now states assumptions before interpretation,
  uses a deterministic teaching sample with a visible association, distinguishes
  mean-response confidence intervals from individual prediction intervals, and
  treats a residual plot as a warning diagnostic rather than proof.
- Added an explicit one-day prediction contract and deterministic chronological
  example. It splits on target timestamps into 22 training, seven validation,
  and eleven test rows; saves stable split roles; fits `StandardScaler` and
  `LinearRegression` together on training rows; compares a training-mean
  `DummyRegressor` and the Pipeline on validation MAE/RMSE/R²; and calls
  `predict()` on final test rows only after selection.
- Added a supplied ten-row binary-prediction table whose learned-model and
  most-frequent baseline accuracies both equal 0.8 while their precision/recall
  behavior differs. No second classifier is fit. Exactly three H2 `LIVE DEMO`
  contracts match the accepted later design handoff: bounded inference; split,
  availability, and leakage; and baseline plus one train-only linear Pipeline
  with supplied binary metrics.
- Rebuilt `10/BONUS.md` around only optional p-value literacy, Ridge/Lasso inside
  a Pipeline, split-aware cross-validation, and one Random Forest candidate with
  validation-set permutation importance. The prior grab bag of boosting/deep-
  learning frameworks, Bayesian search, stacking, automated feature engineering,
  time-series forecasting, deployment, and monitoring code was dropped rather
  than preserved without prerequisites or a downstream role.
- Restated Colab-first/local-Jupyter demo equivalence, conditional assignment
  Colab support, pinned/deterministic input, fresh execution evidence, and the
  unresolved statsmodels/scikit-learn/Matplotlib release pins without claiming
  certification. No lecture/assignment duration, instructor talking point, or
  unconditional Colab assignment path was introduced.

### Author-side validation

- All 13 core and four bonus Python fences executed progressively under Python
  3.12.13, NumPy 2.0.2, pandas 3.0.3, statsmodels 0.14.6, scikit-learn 1.9.0,
  and Matplotlib 3.11.1 with an Agg backend. A second run promoted warnings to
  errors and passed. Those additional library versions are execution evidence,
  not the course release lock.
- Exact coefficient/interval ordering, the wider individual interval, residual
  mean, one-day target horizon, 18/6/6 seeded ID split, 22/7/11 chronological
  split, split separation, training-only scaler state, validation selection,
  final prediction uniqueness, and supplied binary metric assertions all pass.
- `git diff --check` passes. The dependency-free course audit parses all 44
  current notebooks with zero errors; its 20 warnings concern the intentionally
  untouched legacy Assignment 09--11/Classroom surfaces and missing kernelspecs
  in the legacy Lecture 10--11 notebooks. Under `10/`, only `README.md` and
  `BONUS.md` are modified.

This is author-side narrative evidence only. A reviewer who did not author the
files must still verify term order, pedagogy, completeness, scope, both adjacent
boundaries, and absence of untaught required assessment before any Lecture 10
demo or assignment implementation begins.

### Independent Lecture 10 narrative verification

An independent reviewer who did not author `10/README.md` or `10/BONUS.md`
completed the narrative gate on 2026-07-19. **Result: PASS with no content
correction.** Lecture 10 demo design is unblocked; the legacy demos and
assignment remain unaccepted until their later workflow stages.

- The reviewer found no missing prerequisite, untaught required capability,
  causal overclaim, test-set reuse, framework-survey residue, timing claim, or
  Colab-policy overclaim. Required terms are introduced in a cumulative order
  before independent use, and the core/bonus plus Lecture 09→10→11 boundaries
  match the reconciled dependency contract.
- All 13 core and four bonus Python fences passed progressively in normal and
  warnings-as-errors runs under Python 3.12.13, NumPy 2.0.2, pandas 3.0.3,
  statsmodels 0.14.6, scikit-learn 1.9.0, and Matplotlib 3.11.1. These are
  execution-evidence versions, not the final course lock.
- Independent assertions reproduced 18/6/6 exchangeable and 22/7/11
  chronological splits, training-only scaler state, validation-based selection,
  final test MAE `0.45024`, wider individual than mean-response intervals, the
  supplied binary metrics, a rendered residual plot, and the bounded bonus
  permutation output.
- Scoped diff, heading, link, exclusion, and temporary-artifact checks passed.
  The global audit parsed 11 lectures and 41 current notebooks with zero errors;
  its 19 warnings concern intentionally untouched legacy Lecture 10 demos/
  assignment, Lecture 11, and course navigation surfaces rather than the
  accepted narrative.

## Independent Assignment 08 implementation verification

An independent reviewer who did not author the Assignment 08 blueprint or
implementation completed the local implementation gate on 2026-07-18.
**Result: PASS with no student-package or grader correction.** The only code
change from this gate adds the missing narrow Assignment 08 course-audit rule;
this section records the independent evidence.

- The reviewer read the accepted blueprint, verified Lecture 08 narrative,
  bonus, and demos plus the Lecture 07 entry and Lecture 09 exit boundary,
  platform and Colab policy documents, historical `CLAUDE.md`, every Assignment
  08 source, and the author evidence. The exact student inventory, pins,
  fixture/manifest bytes and hashes, local-only platform language, five visible
  outputs, and instructor-only grader surface match the accepted contract.
- The clean starter has exactly 25 ordered unique cells, 16 code cells with null
  execution counts and empty outputs, a portable Python 3 kernelspec, six
  protected cells, and actionable TODOs only in student surfaces. The term
  ledger precedes independent use, defines grain/group/count/aggregation/
  transform/pivot/absence concepts, and the three cumulative tasks remain
  within Lecture 08 scope.
- An independently inspected disposable correct submission passed the public
  checker while the untouched starter failed. AST reconstruction found exactly
  the five required reusable functions, five relevant GroupBy calls with
  literal `observed=True`, `sort=True`, and `dropna=True`, exactly two required
  flat `as_index=False` aggregations, and exactly one `pd.pivot_table` with the
  accepted literal roles and policies.
- Independent calls to all five functions on the disclosed alternate table
  reproduced Metro/Coast/Hill counts, totals, and means; preserved shuffled
  index labels `[42, 5, 91, 12, 63, 8, 77, 24, 3, 55]`; produced eight observed
  two-key rows; preserved Coast--Voice as absent; and matched all eight
  GroupBy-to-pivot cells without mutating the source.
- External schema-aware reads reproduced the exact five output sizes
  `[98, 174, 680, 210, 86]`, accepted hashes, ordered columns, extension and
  NumPy dtypes, UTF-8/LF bytes, and final newlines. The output directory contains
  only `.gitkeep` in the starter, and neither `.gitignore` nor guidance hides the
  five CSVs from GUI Git clients.
- The production-mirror central grader independently scored the correct
  submission 90/90 after fresh execution, alternate calls, all feasible path
  layouts, stale-output replacement, deterministic repeat, and schema-aware
  artifact comparison. A notebook with correct-looking stored output plus
  broken source and an edited always-success public checker earned less than
  full credit, confirming that neither submitted output nor the editable
  self-check is trusted.
- The complete pinned adversarial harness exited zero: correct and corrected
  submissions scored 90/90, all 45 named mutants were rejected, all layouts
  passed, the unrelated sentinel was preserved during setup but rejected as an
  extra final artifact, alternate behavior passed 5/5 functions and 8/8 pivot
  equivalences, and repeat execution was deterministic.
- Official behavior was cross-checked against the
  [Classroom50 Autograders wiki](https://github.com/foundation50/classroom50/wiki/Autograders).
  Success and completed-failure CLI runs both wrote the required hyphenated
  `classroom50/result/v1` object and exited zero; a forced result-write
  infrastructure failure exited 2. Context supplied `review` and other core
  fields were preserved, and optional runner-owned `owner`, `assignment_type`,
  and `submitted_by` fields are accepted but not grader-invented.
- Source and behavior checks found no cleaning, imputation, join, structural
  reshape, visualization, time-series, modeling/statistics, advanced grouping,
  remote/network, random-data, absolute-path, upload/Drive, timing, or Colab
  capability overclaim in the required work.
- `scripts/course_audit.py` now mirrors the established Assignment 07 pattern
  for Assignment 08: exact student inventory and small runtime files, protected
  file and fixture hashes, LF/final-newline fixtures, exact 25-cell topology and
  protected hashes, clean starter state, portable kernel, local-only/no-timing
  guidance, five GUI-visible artifact names, instructor assets, and central
  result/checker-independence markers. The dependency-free audit passes with 44
  notebooks and zero errors; its 20 warnings remain confined to untouched
  later/legacy surfaces. Scoped diff/whitespace and final hygiene checks pass,
  with no Assignment 08 cache, generated CSV, or `result.json` in the repository.

Production Classroom50 provisioning, competence/pass-fail and human-score
policy, resubmission/regrade/retention policy, live runner integration, fresh
Colab execution, and any immutable release badge remain pending external gates.

## Assignment 09 design and independent verification

Assignment 09 was specified on 2026-07-18 and independently accepted on
2026-07-19 in `work/reviews/assignment09_blueprint.md`. No file under
`09/assignment/**` was changed during this design gate. **Result: PASS after two
narrow integration/feasibility corrections; implementation is unblocked.**

- The replacement is one 26-cell notebook with five reusable functions, one
  380-byte synthetic temporal-panel fixture, one 473-byte manifest, six tracked
  deterministic CSVs, a dependency-free public checker, and an independent
  Classroom50 grader/self-test contract. Three cumulative tasks cover temporal
  preparation, measurement-aware frequency change, and entity-scoped past-only
  evidence plus availability and a chronological handoff.
- Under Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3, the independent reviewer
  reconstructed fixture hashes
  `c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4`
  and
  `27558bc4da7738775879501a6f11a0a9d874f3948823e54bb5e82ab91a02d703`.
  All six canonical output row counts, byte sizes, hashes, exact dtypes, row
  order, timezone semantics, grid provenance, resample conservation, lag/
  difference/window values, keep/keep/reject/reject decisions, and 8/4 blocks
  matched the blueprint.
- The disclosed America/Chicago alternate table also passed: ten prepared rows,
  both declared gap sequences, 14 hourly rows with four grid-created and one
  source-missing row, eight two-hour bins, exact Lab 17:00 feature values, and a
  strictly separated 7/3 split retaining both zones. A whole-hour UTC
  precondition now rejects off-grid labels before `asfreq()` can silently omit
  a source observation.
- The first narrow correction replaced invented `CLASSROOM50_*` context names
  with the official runner contract current on 2026-07-19: required
  `CLASSROOM`, `ASSIGNMENT`, `SUBMISSION_TAG`, `COMMIT_URL`, and `RELEASE_URL`;
  optional `REVIEW_URL` falling back to `COMMIT_URL`; and grader-generated UTC
  `datetime`. Runner-stamped identity fields remain outside student/grader
  responsibility, and completed student failure still exits zero.
- Terms precede independent use; the five function preconditions and alternate
  behavior are feasible; outputs remain visible in GUI Git; automation is 90
  points and human reasoning 10 without an inferred grade-conversion policy;
  clean local Jupyter is required; assignment Colab remains conditional; and no
  cleaning, pooling, advanced window/DST, plotting, modeling, mutable/network
  data, timing claim, or Lecture 10 capability enters required work.

## Cross-course Classroom50 runner-contract correction — 2026-07-19

This correction supersedes the earlier Assignment 04–08 runner-integration
claims in this audit. Those records remain evidence for grading behavior and
student-contract checks, but they did not establish the official runner
boundary while graders supplied local metadata defaults. In particular, the
Assignment 07 claims about a grader-owned external review directory and durable
review bundle are withdrawn: `CLASSROOM50_*` variables and
`CLASSROOM50_REVIEW_DIR` are not part of the documented runner contract and are
not production capabilities.

- The five graders now require nonempty `CLASSROOM`, `ASSIGNMENT`,
  `SUBMISSION_TAG`, `COMMIT_URL`, and `RELEASE_URL`. `REVIEW_URL` is optional
  and falls back to `COMMIT_URL`; each grader creates its own UTC `datetime`.
  Runner-owned identity fields are neither inferred nor emitted by graders.
- Missing or empty required context is an infrastructure failure: the command
  exits nonzero and does not create `result.json`. A completed student grading
  failure still writes a valid result and exits zero. The five harnesses test
  both states for all five required fields, for 50 induced infrastructure
  failures, plus both review-fallback states and a completed student failure per
  assignment.
- Assignment 07 now fresh-executes the submitted notebook, compares the
  committed artifacts with the fresh artifacts byte-for-byte, and exposes the
  official review URL only after that equality check. It neither creates nor
  claims an invented external review store.
- Standalone grader and harness scripts include PEP 723 metadata synchronized
  exactly with each grader's pinned `requirements.txt`: all 10 script records
  passed the synchronization check.
- Full pinned self-tests exited zero for all five assignments. Correct reference
  scores were 10/10, 85/85, 90/90, 80/80, and 90/90 for Assignments 04–08,
  respectively. Assignment 08 also returned 90/90 after correction, rejected
  all 45 named adversarial submissions, passed 5/5 alternate functions and 8/8
  pivot equivalences, and confirmed deterministic repeat execution across all
  supported layouts.
- All five public checkers rejected their untouched starters with exit 1 and
  only the intended incomplete-work or missing-artifact findings. The
  dependency-free course audit currently parses 41 notebooks with zero errors;
  its 19 warnings are outside this A04–08 correction. The notebook count is a
  concurrent-worktree observation while Assignment 09 is being consolidated,
  not a new course inventory baseline.

Live Classroom50 provisioning and runner execution still remain external gates,
including acceptance, push/tag/release/status/collection behavior, official
review-page access, human-score integration, and course policy decisions.

## Assignment 09 author implementation evidence

Assignment 09 was implemented on 2026-07-19 from the independently accepted
`work/reviews/assignment09_blueprint.md`. This is author-side local evidence
only. A reviewer who did not implement this package must still complete the
independent implementation gate.

- The legacy generator, three assignment notebooks, paired Markdown sources,
  GitHub Classroom workflow/tests, and broad dependencies were removed
  atomically. The student surface is now exactly one clean 26-cell notebook,
  two exact runtime records, local platform guidance, a standard-library public
  checker, one fixture/manifest pair, and `output/.gitkeep`. The discoverable
  instructor-only surface contains the central-grader mirror, author harness,
  synchronized grader requirements, and maintenance note.
- The synthetic fixture is exactly 380 bytes with SHA-256
  `c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4`;
  the manifest is exactly 473 bytes with SHA-256
  `27558bc4da7738775879501a6f11a0a9d874f3948823e54bb5e82ab91a02d703`.
  Both retain UTF-8/LF/final-newline bytes. Protected setup validates both
  before parsing or owned-output cleanup, removes only six named stale files,
  and preserves unrelated files until final exact-inventory validation.
- The starter has exactly 26 ordered globally unique IDs, a portable Python 3
  kernelspec, null execution counts, zero stored outputs, six protected cells,
  and actionable TODOs only on student surfaces. Terms precede demanding use.
  Clean local Jupyter is mandatory; no assignment Colab badge, save-back claim,
  duration, plotting, cleaning, advanced time-series, or Lecture 10 modeling
  scope was added.
- The public checker validates the exact package, fixture, topology, protected
  cells, five signatures, entity/time policies, pre-`asfreq` whole-hour
  rejection, exact resample aggregation roles, past-only window boundaries,
  validated merge, scope, and all six artifact byte contracts without executing
  notebook code or claiming a grade. The untouched starter exits 1 with exactly
  two actionable groups: unfinished notebook work and missing artifacts.
- The independent central-grader implementation does not import the public
  checker. It clears submitted state and owned outputs in disposable copies,
  fresh-executes the real notebook, calls all five functions on the disclosed
  shuffled-index America/Chicago table, checks canonical/alternate values and
  nonmutation, and exercises flattened/direct, spaces, nested, course-root, and
  relocated layouts. The off-grid alternate fails with an actionable
  `ValueError` before `asfreq` can lose a source row.
- The PEP 723-driven author harness completed with correct `90/90`, starter
  `0/90`, and corrected resubmission `90/90`. All six canonical artifact byte
  counts and hashes matched; all five alternate functions passed; deterministic
  repeat and stale/deleted/corrupt output replacement passed; and all 51 named
  fixture, template, source, scope, output, path, window, provenance, and
  hard-coding mutants were rejected.
- Official command behavior passed with the exact
  `classroom50/result/v1` fields and five automated groups totaling 90. Correct
  and completed student-failure runs write `result.json` and exit zero;
  `REVIEW_URL` falls back to `COMMIT_URL`; missing required
  `CLASSROOM`/`ASSIGNMENT`/`SUBMISSION_TAG`/`COMMIT_URL`/`RELEASE_URL` context
  and a forced result-write failure exit nonzero without a misleading result.
  Grader and harness PEP 723 dependencies match the instructor requirements.
  The standard-library checker also declares the assignment's NumPy/pandas
  runtime pins because it verifies that environment at its real `uv run`
  entrypoint even though its checker implementation does not import them.
- Narrow author corrections strengthened static checks for exact AST literal
  policies, cell types, zone/time sort keys, missingness provenance, aggregation
  roles, and entity-scoped past windows. Empty legacy `.github/` directories
  were removed after their files were deleted. No fixture, expected value,
  notebook topology, artifact contract, scoring allocation, or pedagogical
  boundary changed.
- `scripts/course_audit.py` now enforces only the accepted Assignment 09
  surface: exact inventory/pins/hashes, clean notebook topology and protected
  cells, local-only/no-timing guidance, visible output names, official runner
  markers, checker independence, and synchronized PEP 723 records. The audit
  parses 41 notebooks with zero errors; its 19 warnings are confined to
  untouched Lecture 10–11 or root legacy surfaces. Scoped whitespace, output
  visibility, starter-state, and cache/result hygiene checks pass.

Independent implementation review, live Classroom50 provisioning and runner
behavior, human-score and competence-policy integration, late/regrade/retention
rules, the course-wide package freeze, and any future assignment Colab
save-back/release path remain separate gates.

## Lecture 10 demo design blueprint

Lecture 10 demos were specified on 2026-07-19 in
`work/reviews/lecture10_demo_blueprint.md` after the core/bonus narrative and
Lecture 09 handoff passed independent verification. This is author-side design
evidence only. No file under `10/demo/**` or `10/assignment/**` was edited or
accepted by this work; a different reviewer must complete the design gate.

- The full legacy guide, requirements, three paired Markdown sources, and all
  three notebooks were reviewed. Notebook code matches the paired source fences
  exactly; the legacy package carries 193 stored outputs, a runtime California
  Housing fetch, broad stats/model/boosting/deep-learning dependencies,
  Jupytext conversion, instructor talking points, and duration estimates. The
  design replaces those surfaces rather than preserving them.
- The exact cumulative sequence is bounded OLS association and uncertainty;
  temporal prediction contract, exchangeable-versus-chronological splits and
  leakage rejection; then a training-mean baseline versus one train-only
  `Pipeline(StandardScaler, LinearRegression)`, validation selection, one final
  test evaluation, a familiar residual plot, and supplied binary metric
  literacy without classifier fitting.
- The two course-authored fixtures are exact and deterministic:
  `workshop_participants.csv` is 200 bytes with SHA-256
  `eefb5f1023e9b84106f407800fa0db72853b6876d58e61255c346ed2d2d32f05`;
  `station_next_day.csv` is 3,879 bytes with SHA-256
  `f95330b252c6e0f12026577602c69e21d01dbec232b5e523c6c41b0b62cf85a8`.
  Their literal blueprint blocks independently reproduce those byte/hash pairs.
- Repeated candidate-stack computations reproduced the study-hours coefficient
  `1.645244`, 95% CI `[1.111709, 2.178779]`, wider individual interval,
  18/6/6 exchangeable and 22/7/11 chronological roles, training target mean
  `12.112455`, train-only scaler state, validation Pipeline MAE `0.408166`
  versus baseline `2.598597`, final test MAE `0.450242`, all 11 test
  predictions, and supplied binary accuracy/precision/recall values.
- The handoff fixes the exact replacement tree, candidate pins, path and
  missing-versus-corrupt fixture behavior, 11/11/13 clean notebook topology,
  35 globally unique cell IDs, eight byte/hash-pinned CSV outputs, two exact-
  dimension visually reviewed PNG contracts, semantic readback dtypes, owned-
  output replacement, unrelated-output preservation, and repeat determinism.
- Installed signatures and official primary documentation were checked for the
  statsmodels prediction-interval API and the scikit-learn split, Pipeline,
  scaler, baseline, RMSE, R2, accuracy, precision, and recall paths under Python
  3.12.13, NumPy 2.0.2, pandas 3.0.3, statsmodels 0.14.6, scikit-learn 1.9.0,
  and Matplotlib 3.11.1. These versions remain execution candidates rather than
  a final course lock.
- The independent matrix requires 24 fresh layout/strict executions, missing
  and corrupt fixtures, stale/deleted/binary outputs, exact numeric/readback
  invariants, two original-detail visual inspections, scope and first-use
  ledgers, clean package/audit checks, a fresh-Colab matrix, and an immutable
  release-badge gate. Network/mutable/random data, XGBoost/boosting, deep
  learning, model surveys, classifier fitting, advanced bonus scope, instructor
  talking points, timing claims, and unconditional assignment-Colab support
  remain excluded.

Implementation remains blocked on independent design review. Later publication
still requires independent local implementation QA, fresh Colab execution,
one confirmed immutable course release for all badges, and the course-wide
version freeze. Assignment 10 remains a separate later workflow stage.

## Assignment 04–09 production-runner correction — 2026-07-19

This record supersedes the earlier cross-course claim that the Assignment
04–08 standalone grader entrypoints were production-ready because their PEP
723 dependencies matched `requirements.txt`. That check exercised an author
environment which provisioned dependencies before startup; it did not prove the
documented Classroom50 invocation of plain `[sys.executable, entrypoint]` in a
dependency-empty runner. Assignment 09 had the same unproved boundary. The
grading-behavior evidence remains valid, but that prior production-readiness
PASS is withdrawn.

- Assignments 04–09 now expose a standard-library `autograder.py` bootstrap as
  the production entrypoint. It checks for `pip`, invokes the same Python's
  `ensurepip` when needed, installs the exact sibling `requirements.txt` with
  that interpreter, and only then imports the dependency-bearing central
  grader. Its PEP 723 record intentionally has no dependencies because
  Classroom50 does not install them before invoking the entrypoint.
- A completed student grading failure writes the official
  `classroom50/result/v1` record and exits zero. Bootstrap, import, missing
  runner context, and result-write failures exit nonzero and do not leave a
  plausible `result.json`.
- Student package validation is exact in both public and central paths. It
  ignores only the top-level `.git/**` repository metadata tree, accepts only
  the optional delivery pair `.classroom50.yaml` and
  `.github/workflows/autograde.yaml`, and rejects any other root file,
  additional `.github/**` content, nested `ordinary/.git/**` content, or copied
  `_grader_selftest/**` material. Every Assignment 04–09 harness proves the
  accepted metadata case plus all four bypass classes against both validators.
- All six full author harnesses exited zero. Correct reference scores were
  10/10, 85/85, 90/90, 80/80, 90/90, and 90/90 for Assignments 04–09. Their
  official-command paths invoke the bootstrap with plain Python from the
  student working directory, rather than relying on `uv run` dependency
  provisioning.
- Six separate Python 3.12.13 virtual environments were created without
  `--seed`. Before invocation, `pip` and all course dependencies were absent.
  On the first plain-Python bootstrap invocation, every assignment installed
  the exact pinned sibling requirements and produced a valid UTC result for
  the untouched starter with exit zero: A04 1/10 with 6 tests, A05 0/85 with
  3, A06 0/90 with 3, A07 0/80 with 5, A08 0/90 with 5, and A09 0/90 with 5.
- The dependency-free course audit now checks the six exact instructor bundles,
  bootstrap/source markers, public and central inventory enforcement, and
  adversarial harness probes. It parses 41 notebooks with zero errors and 16
  warnings, all on untouched Assignment 10–11 or root legacy surfaces.

This is local production-contract evidence only. Live Classroom50 acceptance,
push/tag/release/status/collection behavior, review-page access, human-score
integration, and course policy remain external gates.

## Lecture 10 demo implementation and independent review — 2026-07-19

This record supersedes the design section's implementation-blocked status.
Independent design review first rejected four ambiguous contracts: an
inconsistent code-cell adjacency rule, a QA rule that also prohibited the
required seeded split, two names for the same audited feature, and an
unverifiable historical claim about test use. After those were corrected, a
second review caught the resulting stale availability-artifact hash. The final
design gate independently recomputed the corrected 416-byte artifact with
SHA-256
`e765b4426412525b06fcfad9717af158e786e7e0ff32c570341927192d6020f2`
and passed.

- The legacy boosting/deep-learning notebooks, paired Markdown, Jupytext path,
  broad dependencies, runtime dataset fetch, stored outputs, instructor talk,
  and timing material were replaced by exactly nine source files: three clean
  11/11/13-cell notebooks, one direct learner guide, five exact dependency
  pins, two checksum-pinned synthetic fixtures, and two hidden environment/
  output-hygiene records. All 35 cell IDs are globally unique; execution counts
  are null, outputs are empty, and only the three specified code/code
  adjacencies remain.
- The required sequence is bounded OLS association and intervals; prediction
  availability, leakage, and split choice; then validation-only selection
  between a training-mean baseline and one train-only linear Pipeline, followed
  by one frozen test evaluation and supplied binary-metric literacy. Terms
  precede demanding use. Advanced models, classifier fitting, forecasting,
  deployment, network/mutable/random observations, timing, and unconditional
  assignment-Colab claims remain excluded.
- A reviewer who did not author the package fresh-executed all three notebooks
  in course-root, demo-root, nested marked, flattened, and unmarked ephemeral
  layouts, normally and with notebook-code warnings promoted to errors: 30/30
  executions passed under Python 3.12.13 and the exact five direct pins.
  External kernel-shutdown warnings occurred only after notebook completion;
  notebook code emitted no warning.
- Independent adversarial checks passed missing and corrupt fixture ordering,
  deleted/stale/binary owned-output replacement, unrelated-output preservation,
  repeat determinism, and all exact output schemas, dtypes, bytes, and hashes.
  Both plots were inspected at original detail: 1,200 by 720 pixels, all 12/11
  points visible, exact labels and zero lines, legible contrast, no clipping,
  and bounded interpretations.
- Independent numeric recomputation matched the OLS study-hours coefficient
  `1.6452439696` and interval, exact seeded 18/6/6 membership, chronological
  22/7/11 roles, train-only scaler means, validation MAE `2.598597` versus
  `0.408166`, final test MAE `0.450242`, all 11 predictions, and both supplied
  binary metric rows. A through-validation probe observed no test access; the
  selected approach is assigned once and the one 11-row test prediction occurs
  only after that choice is frozen.
- The guide uses direct learner actions, documents Colab-first/local-Jupyter
  equivalence and ephemeral state, keeps assignment Colab conditional on the
  save-back/Classroom50 pilot, and contains no duration or instructor-facing
  talk. Scoped whitespace and hygiene pass. Generated output, QA images,
  checkpoints, caches, temporary scripts, and validation environments were
  removed. The course audit parses 41 notebooks with zero errors and 16
  warnings, all outside `10/demo`.

Lecture 10 demos therefore pass the independent local implementation gate.
Fresh live Colab execution, one immutable release tag for all badges, and the
course-wide dependency freeze remain publication gates. Assignment 10 proceeds
as a separate design and verification stage.

## Assignment 04–09 repository-metadata boundary correction — 2026-07-19

An independent adversarial review rejected the earlier inventory claim because
the public and central comprehensions excluded a file whenever `.git` appeared
in any relative path component. That correctly ignored the real top-level
repository metadata tree, but also allowed an unexpected file such as
`ordinary/.git/nested.txt` to disappear from exact student-package inventory.
The production-runner and grading-behavior evidence above is retained; only
that inventory-boundary claim required correction.

- All 12 public/central inventory predicates across Assignments 04–09 now
  ignore `.git` only when it is the first relative component. The optional
  delivery pair remains exact, and delivery symlinks, arbitrary root files,
  additional workflows, instructor-bundle injection, and nested `.git`
  directories remain visible to validation and are rejected.
- Each of the six full author harnesses now induces
  `ordinary/.git/nested.txt` after first proving that the delivery pair and a
  genuine top-level `.git/config` are accepted. Both the public checker and the
  independent central grading path must reject the induced nested file.
- The dependency-free course audit now requires the first-component predicate
  and the nested-metadata mutant in every Assignment 04–09 public/central/
  harness source set, preventing a return to parts-membership filtering.
- All six full pinned author harnesses exited zero after the correction. Correct
  reference scores remained 10/10, 85/85, 90/90, 80/80, 90/90, and 90/90 for
  Assignments 04–09; each harness accepted the delivery pair plus a genuine
  top-level `.git/config`, then rejected the nested mutant in both the public
  and central path. The course audit parsed 41 notebooks with zero errors and
  16 warnings, all on unchanged Assignment 10–11 or root legacy surfaces.

This correction is local author evidence until a separate reviewer reruns the
six inventory matrices. Live Classroom50 behavior and the external gates
listed above remain unclaimed.

## Independent Assignment 04–09 production and inventory QA — 2026-07-19

A reviewer who did not author either Classroom50 correction reran all six full
assignment harnesses and the stable production/inventory boundary. **Result:
PASS.** This supersedes the preceding author-only status without claiming a
live Classroom50 deployment.

- Correct reference scores remained 10/10, 85/85, 90/90, 80/80, 90/90, and
  90/90 for Assignments 04–09. Assignment 08 rejected 45/45 named mutants and
  Assignment 09 rejected 51/51; every full pinned harness exited zero.
- For every assignment, both public and central paths accepted the exact two
  regular delivery files and genuine top-level repository metadata. They
  rejected an arbitrary root file, an additional workflow, copied
  instructor-only bundle material, a delivery-file symlink, and a same-named
  repository-metadata directory nested below an ordinary student folder.
  Symlinked delivery metadata produced public exit 1 and a completed zero-score
  central result, rather than an infrastructure result or accidental pass.
- Six fresh Python 3.12.13 environments began without the assignment
  libraries. Plain-Python invocation of each bundle `autograder.py` installed
  its exact sibling pins and wrote a valid completed starter result with exit
  zero: A04 1/10; A05 0/85; A06 0/90; A07 0/80; A08 0/90; A09 0/90.
- Independent source inspection found only first-relative-component metadata
  exclusions; the former broad parts-membership predicate is absent. Protected
  checker hashes are synchronized wherever that assignment records one.
- The final course audit reports 11 lectures, 41 notebooks, zero errors, and
  16 later-scope warnings. Scoped whitespace passes. Six bytecode caches
  generated by the independent runs were enumerated and removed; a final scan
  found no cache, checkpoint, bytecode, `result.json`, or `release-body.md`
  artifact under Assignments 04–09.

Live acceptance, submission/tag/release/status/collection behavior, official
review-page access, human-score/course-policy integration, assignment Colab
save-back, immutable release badges, and the course-wide dependency freeze
remain external gates.

## Lecture 11 narrative rewrite and independent review — 2026-07-19

The legacy Lecture 11 outline was replaced by a dataset-agnostic culmination
narrative and independently re-reviewed. **Narrative result: PASS.** The gate
for dataset-specific demos and assignment design remains intentionally closed.

- The narrative now begins with measurable objectives, prerequisites, scope,
  and a project contract. The question, intended use, bounded claim, entity,
  unit/grain, key, target and target timestamp, horizon, availability cutoff,
  primary metric, and baseline must be fixed before data work begins.
- An immutable licensed release and manifest precede analysis. The required
  evidence includes source/release, license and attribution, retrieval date,
  checksum, byte size, schema, row count, time-zone interpretation,
  missing-value encodings, entity/time key, and source invariants.
- Entity/time parsing, stable within-entity ordering, key and grain checks,
  bounded cleaning, joins, grouped summaries, and time-available features now
  lead into fixed chronological train/validation/test roles. Selection compares
  one frozen baseline with one supported train-only Pipeline on validation;
  the selected approach is evaluated once on test.
- A predeclared error slice uses the frozen primary metric; signed residuals
  are defined separately as observed minus predicted and serve only as
  directional row evidence. Claims must link to inspectable artifacts and
  explicit limitations.
- The chapter ends with restartable fresh-Colab and clean-local-Jupyter
  evidence, stored-output distrust, and the supported GUI Git/Classroom50
  course-exit workflow. It does not teach shell Git as the required path.
- Duplicate phase maps, a generic checklist, mutable downloads, premature
  final-exam promises, correlation-first selection, categorical encoding,
  feature-importance theory, additional model families, and unrelated holiday
  material were removed. Heading and whitespace checks pass.

Before Lecture 11 demos or assignment design can begin, the release owner must
freeze the assessment role, exact prediction contract, immutable licensed data
bytes, split boundaries, metric, baseline and Pipeline, dependency/runtime
lock, required outputs, human-review boundary, and assignment-Colab decision.
No one of those choices was inferred during the narrative rewrite.

## Assignment 10 design and independent review — 2026-07-19

Assignment 10 now has an independently accepted candidate design. **Design
result: PASS for non-release implementation.** Certified release remains a
separate blocked gate.

- The three cumulative tasks assess bounded formula-based OLS inference; a
  prediction contract, feature availability, leakage, and chronological
  splitting; then validation-only selection between a training-mean baseline
  and one train-only linear Pipeline, followed by one frozen test evaluation
  and supplied binary-metric literacy.
- The design uses the `smf.ols` formula interface taught in Lecture 10. Seven
  public functions have exact argument, return, ordering, dtype, index,
  nonmutation, and error contracts. Fully disclosed renamed and boundary
  alternates prevent canonical-name or instance-only grading.
- Five static fixtures and eight deterministic CSV outputs were independently
  regenerated under the candidate direct pins. The timestamp fixture recipe
  now includes its exact UTC serialization format and reproduces 3,449 bytes
  with SHA-256
  `f14faf7da64347dfc255aa84b14e79eef7f2d0de94b394c747323319d937baa3`.
- The residual PNG is graded portably by signature, dimensions, full decode,
  nontrivial content, and live-Figure semantics. Cross-platform
  committed/fresh byte equality is not required; byte determinism applies only
  to two runs in the same certified central container. One notebook cell owns
  the single save operation.
- The blueprint distinguishes the distributed starter (`output/.gitkeep`
  only), solved/accepted output (nine generated files plus `.gitkeep`), and
  undistributed instructor assets. Only genuine top-level repository metadata
  receives inventory treatment; nested `.git`, alternate workflows, symlinks,
  arbitrary extras, and learner-injected grader material are rejected.
- The accepted notebook contract has 30 cells, eight protected and 22
  student-editable; automated scoring totals 90 and the nonduplicative human
  reasoning/communication review totals 10. Stored output is never execution
  evidence, local Jupyter is mandatory, and assignment Colab remains
  conditional on the save-back/Classroom50 pilot.

The course-wide transitive `constraints.txt` and immutable central-container
digest are intentionally unresolved. Candidate implementation may omit only
that instructor-owned constraints file. It may not freeze protected release
hashes or claim production-bootstrap acceptance, and production invocation
without the real lock must report a nonzero infrastructure failure. Locked
reruns, protected-hash freeze, live Classroom50 validation, and any required
reverification after locked numeric changes remain release gates.

## Assignment 10 implementation and independent QA — 2026-07-19

The independently accepted design was implemented and passed separate
ordinary-execution and integrity/packaging reviews. **Candidate result: PASS.**
Official release certification remains intentionally false.

The first implementation review found a substantive defect: changing the
protected README, checker, requirements file, or setup cell still produced
90/90 because the grader enforced topology and behavior but had no candidate
integrity profile. It also found that the standalone checker lacked the PEP
723 declaration required by the repository standards. The design was reopened,
narrowly corrected, and independently reverified before implementation work
resumed.

- The non-release candidate now has exact integrity maps for 12 immutable
  learner files and eight normalized protected notebook cells. The public
  checker covers the other 11 files and all eight cells; only the independent
  central grader protects the checker itself, and it reads the bytes without
  importing, executing, or trusting that checker.
- Any protected-file or protected-cell difference produces a valid completed
  Classroom50 result with Template 0/10, four blocked zero tests, and exact
  0/90. Independent QA reproduced that outcome for README, checker,
  requirements, platform guide, fixture, `.gitkeep`, and all eight cells, and
  proved that learner code never executed in those cases.
- The checker now has valid PEP 723 metadata for Python 3.12.13 and the exact
  five ordered learner pins while remaining standard-library-only in its own
  imports. Direct `uv` and provisioned plain-Python starter/solution paths
  return the specified readiness results.
- The full candidate harness exits zero with 90/90 for the solution and
  corrected resubmission, 20 isolated integrity rejections, three map-logic
  rejections, six PEP/static rejections, the prior 32 behavior/package
  rejections, seven disclosed alternates, eight deterministic CSVs, and one
  portable PNG. The absent-lock bootstrap exits 2 and writes no result.
- Independent ordinary execution passed normal and warning-as-error runs. It
  reproduced all CSV hashes, a fully decoded 720 by 480 RGBA PNG with retained
  live-Figure semantics, 29/29 training fits, 8/8 validation predictions, one
  11-row test prediction with the same selected estimator, and no refit.
- The candidate topology is exactly 18 files: 13 distributed learner files and
  five instructor-bundle files. The 19th target-release file is the unresolved
  instructor `constraints.txt`. An earlier author handoff saying 17 files was
  a summary typo; no topology defect was present. The final tree has no legacy
  workflow, cache, bytecode, checkpoint, result, generated learner output, or
  constraints placeholder.
- The final course audit parses 41 notebooks with zero errors and 14 warnings,
  all on still-legacy Lecture 11 assignment or root navigation surfaces.
  Scoped whitespace checks pass.

The current maps are explicitly candidate-nonrelease evidence. After the real
course-wide transitive lock, immutable container digest, final wording, and
platform gates pass, all protected values must be recomputed and frozen as
official release hashes, followed by the complete harness and another release
review. Live Classroom50 and conditional assignment-Colab pilots remain
external gates.
