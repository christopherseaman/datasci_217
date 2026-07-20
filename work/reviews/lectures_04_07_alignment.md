# Lectures 04–07 curriculum alignment review

Status: evidence and redesign contract only. This review does not modify lecture, demo, assignment, or test sources.

## Scope and governing decisions

This artifact applies the review process in `work/lecture_review_workflow.md` to Lectures 04–07 and incorporates the accepted course spine in `work/2026_refresh_audit.md`.

The governing decisions for this range are:

- Lecture 04 is the first Jupyter lecture. Lectures 01–03 remain script-and-terminal based; the current Lecture 03 states that boundary explicitly at `03/README.md:14-19`.
- Compatible required demos from Lecture 04 onward are Colab-first and also run in local Jupyter. Redesigned notebook assignments must execute in a clean Jupyter kernel and use the portable path/data contract; Colab becomes an assignment path only after repository-save and submission validation is approved. The current Lectures 04–07 contain no Colab instructions; Lecture 04 is VS Code-specific (`04/README.md:132-145`; `04/demo/DEMO_GUIDE.md:7-18`), and Lectures 06–07 instead lead with local virtual-environment instructions (`06/assignment/README.md:5-43`; `07/assignment/README.md:5-43`).
- Canonical concept homes in this range are: notebook state and pandas structures in Lecture 04; cleaning in Lecture 05; joins/concatenation/structural reshape in Lecture 06; visualization in Lecture 07; aggregation and aggregating pivots in Lecture 08.
- Each lecture should have two or three required demos. Enrichment can remain available, but it must not become a hidden assignment prerequisite.
- Assignment tests may check objective behavior and invariants. Communicative judgment, rationale, and visual quality require an explicit human rubric; image existence is not evidence of those competencies.

## Range-level dependency contract

| Boundary | Exact incoming capability | Where it is currently established | Capability this range must hand off |
|---|---|---|---|
| 03 → 04 | Run Python scripts; define/import functions and modules; use a project environment; inspect ndarray shape, dimensions, dtype, and axis; select with slices and boolean masks | `03/README.md:14-19`, `03/README.md:293-312`, `03/README.md:332-350`, `03/README.md:353-410` | Diagnose notebook state; create and inspect labeled pandas objects; select/filter them; perform portable CSV I/O in a notebook that runs cleanly from top to bottom |
| 04 → 05 | Use code/Markdown cells; explain kernel/runtime state; restart and run all; inspect `Series`/`DataFrame`; select by label/position; filter, sort, derive a column; read/write a portable CSV path | Partly present at `04/README.md:23-40`, `04/README.md:157-175`, `04/README.md:222-299`, `04/README.md:358-433`, and `04/README.md:601-634` | Audit a raw table, make documented cleaning decisions, validate invariants, and produce a clean artifact without relying on hidden notebook state |
| 05 → 06 | Preserve raw data; identify schema and row meaning; detect missing/sentinel/duplicate/type/category/range issues; transform and validate; preserve an audit trail | Current detect/handle/validate framing at `05/README.md:12-16`, validation at `05/README.md:568-638`, and pipeline at `05/README.md:638-716` | State table grain and keys, predict join cardinality, validate merges, inspect unmatched rows, concatenate compatible tables, and reshape structurally |
| 06 → 07 | Prepare one analysis-ready table; understand row grain, keys, merge row-count effects, wide/long form, and index alignment | Only partly present: relationship types at `06/README.md:164-201`, concat at `06/README.md:382-577`, wide/long at `06/README.md:583-673` | Choose and construct an honest, accessible chart for a question/audience/claim using a prepared table |
| 07 → 08 | State question, audience, and intended claim; distinguish exploratory from explanatory use; prepare long-form data; use `Figure`/`Axes`; label, annotate, and export | `Figure` is present at `07/README.md:202-212`; the question/audience/claim contract and exploratory/explanatory distinction are not currently taught as a core sequence | Lecture 08 can begin with unit of analysis, grouping key, split–apply–combine, result grain, aggregation versus transform, and output shape without reteaching plotting |

## Lecture 04: Jupyter, pandas structures, selection, and portable I/O

### Intended role

Lecture 04 converts the script/NumPy foundation into a reproducible notebook workflow, then introduces labeled pandas data structures, selection, and CSV round-tripping. It should not teach cleaning or aggregation.

### Current evidence and blockers

- There is no measurable objective block. The file opens with bonus topics and immediately begins the Jupyter narrative (`04/README.md:1-14`).
- Notebook and cell are defined, and kernel is identified as the Python interpreter (`04/README.md:23-40`). Kernel restart and run-all are described (`04/README.md:157-175`), but the lecture does not explicitly define **state**, **stale state**, or why execution order can make a notebook irreproducible.
- The demo guide does contain a guided restart failure (`04/demo/DEMO_GUIDE.md:103-114`), while the committed demo notebook only assigns and reuses a variable (`04/demo/demo1_jupyter_basics.md:60-69`). A learner running the notebook normally never sees the failure or repairs a stale-state bug.
- The delivery path is VS Code-first (`04/README.md:132-145`; `04/demo/DEMO_GUIDE.md:7-18`). There is no definition of Colab runtime, ephemeral filesystem, `/content`, upload/download, Drive mounting, or a portable local/Colab path contract.
- The NumPy-to-pandas bridge is too weak. The lecture says pandas is built on NumPy (`04/README.md:204-206`) and separately defines a Series as one-dimensional and a DataFrame as two-dimensional (`04/README.md:235-268`), but it never explicitly maps ndarray values/shape/dtype/axis to labeled values/index/columns/dtypes.
- The core contains Lecture 05 material—missing-data handling and type conversion (`04/README.md:469-523`)—and Lecture 08 material—GroupBy, filter, and transform (`04/README.md:574-595`). Demo 3 independently cleans missing values and performs grouped summaries (`04/demo/demo3_data_io.md:59-121`).
- CSV I/O is correctly placed, but the reference contains malformed inline-code fences at `04/README.md:619-620`, other formats expand the core at `04/README.md:636-660`, and all paths assume one working directory.
- The assignment assesses cleaning and GroupBy (`04/assignment/README.md:33-59`) rather than staying within the Lecture 04 boundary.
- The data generator writes `data/customer_purchases.csv` without creating `data/` (`04/assignment/data_generator.ipynb:78-81`). A fresh checkout or Colab session can fail before the assignment begins.
- Question 3 requires both category revenue and top products (`04/assignment/README.md:48-59`; `04/assignment/assignment.ipynb:321-388`), but the test checks only a category-like column and positive revenue-like values (`04/assignment/.github/test/test_assignment.py:70-96`). The top-product competency is not tested.
- Six unresolved media markers remain in the core (`04/README.md:16-31`, `04/README.md:208-226`, `04/README.md:362-369`).

### Proposed measurable objectives

By the end of Lecture 04, students should be able to:

1. Distinguish a script from a notebook and define **cell**, **kernel/runtime**, **state**, **execution order**, **output**, and **restart and run all** in plain language.
2. Reproduce a stale-state failure, diagnose the hidden dependency, and repair the notebook so it runs correctly after a fresh restart.
3. Create and inspect a `Series` and `DataFrame`, explaining their values, index, columns, shape, and dtypes in relation to a NumPy array.
4. Select rows and columns by label and position, filter with a boolean mask, sort rows, and create one derived column without an explicit loop.
5. Read a CSV from and write a CSV to a course-standard project path in both Colab and local Jupyter, then verify the notebook succeeds from a fresh runtime from top to bottom.

### Exact prerequisites

Incoming required skills:

- Run a `.py` file and interpret a traceback.
- Define and call functions and import a module.
- Activate a project environment and install/import NumPy.
- Explain ndarray dimension, shape, dtype, axis, slice, mask, and vectorized operation.
- Use relative paths and know the current working directory.

Lecture 04 must not assume:

- prior notebook, kernel, runtime, or cell-state knowledge;
- pandas cleaning, grouping, merging, reshaping, plotting, or datetime conversion;
- Colab file persistence or Google Drive behavior.

Outgoing required skills for Lecture 05:

- Run a notebook in order after restart with no hidden state.
- Construct, inspect, select, filter, sort, and derive a column in a DataFrame.
- Read and write a CSV using the standard portable project-root/path helper.
- Explain index labels versus integer positions.

### First-definition and first-independent-use ledger

| Term or skill | Current first definition/guided use | Current first independent use | Finding and required action |
|---|---|---|---|
| Notebook | Defined as mixed, interactively executed code/documentation at `04/README.md:12-14`; cells defined at `04/README.md:23-40` | Assignment immediately requires running two notebooks at `04/assignment/README.md:5-17` | Keep, but begin with a measurable notebook contract and Colab-first interface map before assignment use. |
| Cell | Code/Markdown cells and execution shortcuts at `04/README.md:23-40` | Students must run all generator cells at `04/assignment/README.md:7-13` | Aligned in order; add cell dependencies and output semantics. |
| Kernel/runtime | Kernel defined at `04/README.md:40` and expanded at `04/README.md:157-175` | Restart/run-all required in assignment notebook at `04/assignment/assignment.ipynb:13-16` | Define **runtime** as the Colab-facing name and explain that kernel/runtime owns state and installed packages. |
| State / stale state | Not explicitly defined; only implied by memory and restart (`04/README.md:157-175`) | Students are told to restart/run all but are not asked to diagnose state | Revise. Add a deliberate out-of-order success followed by fresh-runtime failure and repair. |
| Magic command | Defined at `04/README.md:100-113` | Demo uses `%matplotlib`, `%pwd`, and a shell pipe at `04/demo/DEMO_GUIDE.md:47-69` | Reduce to `%pip` and one location check. Avoid making shell pipelines a notebook prerequisite. |
| Series | Defined at `04/README.md:235-250`; guided example at `04/README.md:252-264` | Assignment only encounters Series indirectly through selection/aggregation | Add an explicit Series↔1D ndarray bridge and one required selection task. |
| DataFrame | Defined at `04/README.md:266-284`; guided example at `04/README.md:286-299` | Assignment loads and manipulates one at `04/assignment/README.md:19-59` | Keep; narrow independent use to inspection, selection, filtering, derivation, sorting, and I/O. |
| Index / label / position | Index is named in structure references (`04/README.md:230-245`); label versus position is defined at `04/README.md:358-399` | No assignment task or test requires `.loc`/`.iloc` (`04/assignment/.github/test/test_assignment.py:13-110`) | Taught but unassessed. Add one label selection and one positional inspection to the assignment/test contract. |
| Boolean mask | Explained through indexing at `04/README.md:393-432` | Assignment filters states/quantity at `04/assignment/README.md:41-42`; tests enforce the result at `04/assignment/.github/test/test_assignment.py:58-66` | Keep and align to a non-cleaning filter task. |
| Portable path | Not defined | Relative strings are used freely throughout; generator writes directly at `04/assignment/data_generator.ipynb:78-81` | P0 addition. Define one project-root convention that works in Colab and local Jupyter and creates required directories. |

### Objective-to-artifact alignment matrix

| Proposed objective | Revised lecture section | Required demo | Redesigned assignment evidence | Test/rubric evidence | Current status/action |
|---|---|---|---|---|---|
| 1. Notebook vocabulary and execution model | “From script process to notebook runtime” | Demo 1: runtime, cells, state, outputs | Short Markdown explanation identifying runtime, state, and why run-all matters | Human rubric checks definitions and explanation | **Partially aligned**: cell/kernel present; state/runtime not explicit. |
| 2. Diagnose stale state | “State, order, restart, and hidden dependencies” | Demo 1 contains deliberate out-of-order failure and repair | Repair one supplied hidden dependency; final notebook must execute cleanly | Execute notebook in a fresh runtime; human rubric checks diagnosis | **Taught but unpracticed** in current lecture; guide is stronger than committed notebook. |
| 3. Explain Series/DataFrame structure | “Labeled structures: NumPy bridge” | Demo 2: 1D array→Series and 2D records→DataFrame | Inspect shape/dtypes/index and select a Series/DataFrame result | Tests exact schema/result types or saved inspection table | **Partially aligned**: definitions strong, explicit bridge and assessment weak. |
| 4. Select, filter, sort, derive | “Selection before transformation” | Demo 2: `[]`, `.loc`, `.iloc`, mask, sort, one derived column | Produce a filtered, sorted table and one calculated column | Tests row membership/order, column values, and result schema | **Partially aligned**: current assignment filters and derives but skips `.loc`/`.iloc` and adds cleaning/GroupBy. |
| 5. Portable CSV round-trip and clean run | “Colab-first files, local equivalent, and restart verification” | Demo 3: read→inspect→write→restart/run-all | Generate `output/selected_records.csv` from a provided/pinned CSV | Fresh-runtime notebook execution plus file schema/value invariants | **Untaught but required operationally**: current paths are local-only and generator can fail on missing directory. |

### Recommended required demos

1. **Notebook runtime and stale state (Colab first, local Jupyter equivalent).** Create code and Markdown cells; demonstrate output persistence; deliberately run cells out of order; restart; observe failure; remove the hidden dependency; rerun all. Include the ephemeral Colab filesystem and output/privacy note.
2. **NumPy to labeled pandas.** Convert one 1D array to a Series and one small record collection to a DataFrame; inspect index/columns/shape/dtypes; practice `[]`, `.loc`, `.iloc`, masks, sorting, and one derived column.
3. **Portable CSV round-trip.** Resolve project paths, create directories, load a small pinned CSV, inspect it, select/filter it, save one output, and prove the same notebook runs from a fresh Colab runtime and local Jupyter kernel.

Required demos should not clean missing values, convert dates, group, aggregate, or plot beyond a supplied optional preview.

### Assignment redesign contract

**Purpose:** demonstrate notebook reproducibility and pandas structure/selection/I/O—not cleaning or aggregation.

**Student artifacts:** one `assignment.ipynb`, one provided/pinned input CSV, and two small output CSVs. The generator may be a supplied setup cell, but students should not need a second notebook merely to obtain data.

**Tasks:**

1. Explain the runtime/state hazard and repair one supplied hidden dependency.
2. Load the input; report shape, index, columns, and dtypes; use label and positional selection.
3. Apply a boolean filter, create one arithmetic column, sort deterministically, and save the selected result.
4. Restart and run all using the supported Colab/local path cell.

**Automated evidence:** execute from a clean runtime; verify directories are created; verify exact input checksum/version; test output schema, row membership, order, and arithmetic; ensure no current-session artifact is required.

**Human evidence:** notebook state explanation, readable Markdown, and a short local-versus-Colab path explanation.

**Remove from Assignment 04:** `.fillna`, `.dropna`, datetime conversion, type-cleaning decisions, GroupBy, top-product aggregation, and multi-output-format work. These belong in Lectures 05, 08, or bonus material. The current out-of-scope requirements are at `04/assignment/README.md:33-59`.

### Content disposition

| Material | Disposition | Evidence and rationale |
|---|---|---|
| Script versus notebook; code/Markdown cells | Keep/revise | Correct transition at `04/README.md:10-40`; add execution model and measurable objective. |
| VS Code Jupyter walkthrough | Consolidate as local-support note | Current core is interface-specific (`04/README.md:132-145`); Colab is the default compatible path. |
| Kernel management | Revise/reorder | Place directly after cells and define state/runtime before use (`04/README.md:157-175`). |
| Notebook output/privacy | Keep/revise | Important at `04/README.md:177-198`; distinguish clearing sensitive outputs from preserving nonsensitive evidence and from generated output files. |
| Magic-command catalog | Reduce/move to bonus | `%pip` and location are useful; `%timeit`, shell listing, and plotting magic are not central to the first notebook model (`04/README.md:100-130`). |
| Series/DataFrame/index/dtypes | Keep/reorder | Strong definitions at `04/README.md:222-299`; explicitly bridge from NumPy before multiple convenience APIs. |
| Selection, `.loc`, `.iloc`, masks, sort | Keep | Core daily use at `04/README.md:301-438` and `04/README.md:529-572`. |
| `.assign`, `.insert`, `.eval` | Reduce/move to bonus | Direct assignment is enough in core; four creation styles at `04/README.md:440-467` distract from the main model. |
| Missing-data handling and type conversion | Move to Lecture 05 | Current placement at `04/README.md:469-523` breaks the canonical cleaning home. |
| GroupBy/filter/transform | Move to Lecture 08 | Current placement at `04/README.md:574-595` pre-teaches the later lecture and becomes an assignment leak. |
| Excel/JSON/chunked I/O | Move to bonus | CSV is sufficient for the core transition; other formats are already named as bonus at `04/README.md:3-8` but duplicated in core at `04/README.md:636-660`. |
| Data-quality assessment | Move to Lecture 05 | Current section at `04/README.md:692-719` should become Lecture 05’s opening audit. |
| Unresolved screenshot/FIXME blocks | Replace or drop | Six markers remain and cannot be student-facing release content. |

### Ordered content outline

1. Why notebooks begin now: script process versus persistent notebook runtime.
2. Colab-first interface; local Jupyter equivalent.
3. Code cells, Markdown cells, outputs, kernel/runtime.
4. State, execution order, stale results, restart, run all.
5. Colab filesystem persistence, privacy, and course path convention.
6. NumPy-to-pandas bridge.
7. Series: values, index, dtype, name.
8. DataFrame: rows, columns, index, shape, dtypes.
9. Inspection: `head`, `info`, `describe` as bounded tools.
10. Column selection and Series-versus-DataFrame return type.
11. `.loc` labels, `.iloc` positions, boolean masks.
12. Sorting and one derived column.
13. CSV read/write with portable paths.
14. Fresh-runtime verification and handoff to cleaning.

## Lecture 05: data cleaning as a documented, validated pipeline

### Intended role

Lecture 05 teaches students to audit raw data, make domain- and row-meaning-aware cleaning decisions, transform deliberately, validate explicit invariants, and preserve a reproducible clean artifact plus an audit trail.

### Current evidence and blockers

- There is no measurable objective or prerequisite block. The lecture begins with a midterm placeholder and an informal statistic (`05/README.md:1-12`).
- The detect→handle→validate→transform frame is useful (`05/README.md:12`), but the current order begins with handling missing values before defining raw/clean data, schema, row meaning, identifiers, sentinels, or validation invariants.
- MCAR, MAR, and MNAR are expanded only in an image caption (`05/README.md:20-21`). They are not defined, distinguished, or used to justify a decision.
- A generic quality table recommends dropping when missingness is below 5% (`05/README.md:568-576`). Missingness proportion alone cannot determine whether deletion is defensible; variable meaning, mechanism, outcome relevance, and row loss matter.
- Forward fill is introduced on an unordered generic DataFrame (`05/README.md:84-118`) and the required assignment instructs students to forward-fill “time-series columns” in cross-sectional patient rows (`05/assignment/README.md:209-236`). Forward fill is only meaningful after entity and order semantics are established.
- The core expands into custom `apply`/lambda patterns (`05/README.md:193-253`), outlier rules (`05/README.md:332-365`), modeling encodings and multicollinearity (`05/README.md:367-433`), sampling/permutation/bootstrap (`05/README.md:513-565`), configuration systems (`05/README.md:718-753`), and command-line notebook automation (`05/README.md:756-808`). These obscure the foundational cleaning spine and introduce later-course terms.
- Demo 1 forward-fills dates across adjacent rows (`05/demo/demo1_missing_data.md:73-99`) without an entity/order contract.
- Demo 3’s guide promises configuration-driven processing (`05/demo/DEMO_GUIDE.md:84-109`), but the committed notebook performs direct sequential mutations and contains no configuration object (`05/demo/demo3_workflow.md:64-89`). The guide and artifact disagree.
- Demo 3 reports validation counts but does not assert invariants (`05/demo/demo3_workflow.md:92-104`), then adds period extraction, GroupBy, and IQR analysis (`05/demo/demo3_workflow.md:106-137`) before declaring the data “clean” (`05/demo/demo3_workflow.md:139-175`).
- Assignment 05 says “Total: 100 points” (`05/assignment/README.md:1-3`) but allocates 10+25+20+15+15+20+15+5 = 125 points across Questions 1–8 (`05/assignment/README.md:102-334`). The public test file also labels itself 100 points (`05/assignment/.github/test/test_assignment.py:1-6`) while its section comments reproduce the inconsistent allocations (`05/assignment/.github/test/test_assignment.py:29-220`).
- The assignment claims a Q3→Q7 pipeline (`05/assignment/README.md:33-38`), yet Q4, Q5, Q6, and Q7 each reload `data/clinical_trial_raw.csv` (`05/assignment/q4_exploration.ipynb:37`, `05/assignment/q5_missing_data.ipynb:35`, `05/assignment/q6_transformation.ipynb:34`, `05/assignment/q7_aggregation.ipynb:34`). The tests check output existence and minimal shape rather than stage-to-stage provenance (`05/assignment/.github/test/test_assignment.py:167-225`). Q8’s “pipeline execution” test only checks that a log exists (`05/assignment/.github/test/test_assignment.py:211-225`).
- Question 7 independently assesses GroupBy analysis before the canonical Lecture 08 home (`05/assignment/README.md:269-298`). Question 6 requires dummy variables before modeling (`05/assignment/README.md:239-266`).

### Proposed measurable objectives

By the end of Lecture 05, students should be able to:

1. Distinguish raw and cleaned data and define **schema**, **sentinel value**, **duplicate**, **missing value**, **imputation**, **validation invariant**, and **provenance/audit trail**.
2. Produce a reproducible audit of schema, missingness, sentinel codes, duplicate candidates, category inconsistencies, type failures, and invalid ranges without modifying the raw table.
3. Choose and justify a missing-data action using variable meaning, row meaning, and analysis purpose; identify when forward/backward fill is invalid because entity/order requirements are absent.
4. Standardize strings, categories, numeric/date types, sentinel values, and duplicate records while preserving raw input and recording each decision.
5. Express post-cleaning expectations as executable invariants and produce a clean dataset plus a decision log from a fresh runtime.

### Exact prerequisites

Incoming required skills:

- Restart and run all a notebook.
- Load and save CSV data through the course portable-path convention.
- Inspect DataFrame shape, columns, dtypes, index, head, and summary.
- Select/filter rows and columns; create a column; sort deterministically.
- Define/call a function when a repeated cleaning step benefits from one.

Lecture 05 must not assume:

- GroupBy aggregation, merging, MultiIndex, time-series panel ordering, statistical missingness theory, modeling encodings, train/test splitting, or command-line notebook automation.

Outgoing required skills for Lecture 06:

- Preserve raw and clean artifacts separately.
- State what one row represents and which columns are candidate identifiers.
- Identify schema, duplicate, missingness, sentinel, category, type, and range issues.
- Make and document a cleaning decision.
- Validate row-count, uniqueness, category, type, missingness, and range invariants.

### First-definition and first-independent-use ledger

| Term or skill | Current first definition/guided use | Current first independent use | Finding and required action |
|---|---|---|---|
| Raw versus clean data | Not explicitly defined; “clean” is used throughout from `05/README.md:12` onward | Assignment asks for a complete cleaning pipeline at `05/assignment/README.md:5-11` | Add first. Preserve immutable raw input and name derived clean artifacts. |
| Schema | Not defined in the lecture | Assignment expects type conversions and many fixed columns (`05/assignment/README.md:50-90`, `05/assignment/README.md:239-266`) | Define before audit; include column name, type, allowed values, nullability, and row meaning. |
| Missing value / missingness | Described generally at `05/README.md:14-27` | Assignment requires strategy choices at `05/assignment/README.md:209-236` | Keep, but distinguish absence, sentinel code, and invalid parse. |
| MCAR/MAR/MNAR | Acronyms only at `05/README.md:20-21` | No valid independent use | Move to bonus unless the lecture defines mechanisms and uses them in a bounded reasoning exercise. |
| Imputation | Defined at `05/README.md:84-96` | Assignment requires mean/median/forward fill at `05/assignment/README.md:215-224` | Revise: choose by meaning; never present a method list as universally interchangeable. |
| Sentinel value | Appears in a code comment at `05/README.md:177-180`; assignment gives sentinel codes at `05/assignment/README.md:82-90` | Students implement sentinel replacement in Q3 (`05/assignment/README.md:165-174`) | Define before code and distinguish sentinel from true numeric value and missing data. |
| Duplicate | Motivation and methods at `05/README.md:135-157` | Assignment removes “remaining duplicate rows” at `05/assignment/README.md:257-260` | Add record-identity question before `drop_duplicates`; exact duplicate is not the same as duplicate entity. |
| Validation invariant | Validation described as business constraints at `05/README.md:568-638`; term “invariant” absent | Tests mainly check file existence/minimal shape (`05/assignment/.github/test/test_assignment.py:167-225`) | Define an invariant as a condition that must be true after a stage; use executable assertions. |
| Provenance / audit trail | Decision documentation mentioned at `05/README.md:720-729`; guide promises audit trails at `05/demo/DEMO_GUIDE.md:122-129` | Assignment output contract does not require a decision ledger | Make core and assessed through a structured cleaning log. |
| Forward fill | Defined procedurally at `05/README.md:84-128` | Required in Q5 at `05/assignment/README.md:221-224` | Restrict independent use until entity and order are explicit; otherwise use as a counterexample. |

### Objective-to-artifact alignment matrix

| Proposed objective | Revised lecture section | Required demo | Redesigned assignment evidence | Test/rubric evidence | Current status/action |
|---|---|---|---|---|---|
| 1. Cleaning vocabulary and raw/clean contract | “What cleaning changes—and what it must preserve” | Demo 1 opens with raw copy, schema, and decision ledger | Markdown definitions and declared row meaning/identifiers | Human rubric; tests verify raw checksum unchanged | **Orphaned/absent** for schema, provenance, invariant; terminology must precede APIs. |
| 2. Audit before mutation | “Audit schema and quality dimensions” | Demo 1: structured audit | `output/audit.csv` or JSON with issue counts | Tests recompute issue counts from pinned raw data | **Partially aligned**: detection exists, but current assignment scatters audits across Q4/Q5. |
| 3. Justify missing-data action | “Missingness by meaning, mechanism, and consequence” | Demo 1 compares defensible and invalid strategies | Decision table with action and rationale per affected column | Human rubric; tests verify chosen transformations where uniquely specified | **Partially aligned**: methods taught; generic 5%/forward-fill rules are pedagogically unsafe. |
| 4. Normalize values and types | “Sentinels, strings, categories, numbers, dates, duplicates” | Demo 2: staged transformations | Cleaned table produced from prior audit, not raw reload | Tests types, allowed values, row identity, and known edge cases | **Aligned in topic, overbroad in scope**: remove dummy/modeling and generic outlier content. |
| 5. Validate and document pipeline | “Invariants, assertions, export, audit trail” | Demo 3: one restartable pipeline | `cleaned.csv` plus `cleaning_log.csv`; all cells rerun cleanly | Execute fresh; assert invariants; trace output to input checksum | **Partially aligned**: current demo prints counts, and current Q8 test does not execute the pipeline. |

### Recommended required demos

1. **Audit and decision table.** Load immutable raw data; state row meaning and candidate IDs; audit schema, sentinels, duplicates, missingness, categories, parse failures, and ranges; record decisions without mutating.
2. **Targeted transformations.** Apply built-in vectorized cleaning methods for sentinels, strings/categories, types/dates, and duplicates. Compare a defensible missing-value action with an invalid forward-fill example.
3. **Validated end-to-end pipeline.** Run raw→audit→clean→validate→save in one restartable notebook, use assertions for invariants, and emit a structured cleaning log. Keep aggregation and modeling out.

### Assignment redesign contract

If Lecture 05 remains the midterm, it can be broader than a regular assignment, but it should still assess Lecture 05’s cleaning objectives rather than repeat terminal setup or jump to aggregation/modeling.

**Student artifacts:** one Colab/local-compatible notebook, one pinned raw CSV, `output/cleaned_data.csv`, `output/audit.csv`, and `output/cleaning_log.csv` (or a documented JSON equivalent).

**Required stages:**

1. Declare row meaning, candidate identifier, schema expectations, and the immutable raw-input checksum.
2. Audit every specified issue category.
3. Record the chosen action and reason before changing each affected field.
4. Apply transformations in a visible sequence using functions only where they improve clarity.
5. Assert postconditions and save clean data plus the log.
6. Restart and run all in Colab and local Jupyter without a shell automation dependency.

**Automated evidence:** raw file unchanged; exact expected issue counts; known sentinel/type/category/duplicate edge cases corrected; no forbidden missing values; allowed categories/ranges/types satisfied; expected row-count relationship; clean output reproducible; notebook executes in a fresh runtime.

**Human evidence:** row-meaning statement, cleaning rationale, uncertainty/caveats, and audit-log quality.

**Remove from the required assignment:** executable shell setup, standalone Python config parsing, four separate notebooks, dummy variables, GroupBy reporting, top-patient analysis, and `nbconvert` shell orchestration. Current scope is visible at `05/assignment/README.md:15-38`, `05/assignment/README.md:102-176`, `05/assignment/README.md:239-334`.

**Scoring contract:** one source of truth totaling exactly 100 if this remains a 100-point exam. Every rubric row maps to one objective and distinguishes automated versus human evidence.

### Content disposition

| Material | Disposition | Evidence and rationale |
|---|---|---|
| Detect→handle→validate→transform | Keep/reorder | Strong organizing idea at `05/README.md:12`; add raw/clean/schema/row meaning first and move validation expectations before handling. |
| Missing-data detection and bounded imputation | Keep/revise | Core at `05/README.md:25-128`; ground each action in meaning and consequences. |
| MCAR/MAR/MNAR | Move to bonus or define fully | Acronyms alone at `05/README.md:20-21` add terminology without usable reasoning. |
| Universal “drop if <5%” guidance | Drop | Current table at `05/README.md:568-576` overgeneralizes a context-dependent decision. |
| Duplicate detection | Keep/revise | Current methods at `05/README.md:135-157`; add record identity and preservation checks. |
| Sentinel replacement; string/category/type/date normalization | Keep | Daily cleaning core across `05/README.md:160-190`, `05/README.md:255-332`, and `05/README.md:435-511`. |
| Custom `apply`/lambda catalog | Reduce/move to bonus | Prefer vectorized built-ins; current section `05/README.md:193-253` is larger than the required cleaning objective. |
| Generic outlier removal/capping | Move to bonus/revise | Current rules at `05/README.md:332-365` can erase legitimate extremes without domain rationale. |
| Dummy variables and multicollinearity | Move to Lecture 10 | Modeling vocabulary at `05/README.md:367-433` precedes the modeling lecture. |
| Sampling, shuffling, bootstrap | Move later or bonus | Current section at `05/README.md:513-565` is not necessary for cleaning and introduces train/test/bootstrap claims early. |
| Validation rules and data-quality report | Keep/strengthen | Correct home at `05/README.md:568-638`; change printed diagnostics to explicit invariants/assertions. |
| Configuration-driven processing | Move to bonus | Useful enrichment at `05/README.md:718-753`, but the current required demo does not implement what its guide promises. |
| `nbconvert`/shell pipeline | Move to bonus or repository tooling | Current section `05/README.md:756-808` is local-shell-specific and conflicts with Colab-first required work. |
| Aggregated customer summary in demo/assignment | Move to Lecture 08 | `05/demo/demo3_workflow.md:106-120` and `05/assignment/README.md:269-298` assess grouping before its canonical home. |

### Ordered content outline

1. Cleaning purpose: raw data, clean data, and preservation.
2. Row meaning, candidate identifiers, schema, and provenance.
3. Audit before mutation.
4. Missing values versus sentinel values versus parse failures.
5. Missing-data choices by variable meaning and consequence.
6. Exact versus entity duplicates.
7. Normalize strings and categories.
8. Convert numeric and datetime types with explicit failure handling.
9. Validate allowed values, ranges, types, uniqueness, missingness, and row counts.
10. Record decisions in a structured audit trail.
11. Compose one raw→audit→clean→validate→save notebook.
12. Restart/run-all verification and handoff to joins.

## Lecture 06: grain, validated joins, concatenation, and structural reshape

### Intended role

Lecture 06 teaches students to reason about what a row represents before combining data, validate key/cardinality expectations during merges, concatenate compatible tables with explicit alignment semantics, and reshape between wide and long forms without introducing aggregation.

### Current evidence and blockers

- Lecture 06 has an objective list (`06/README.md:23-30`), but verbs such as “master” are not measurable, and basic MultiIndex is included before the later aggregation lecture.
- The opening merge example is syntactically mangled by comments and statements running together (`06/README.md:85-103`), so the first executable model of the lecture is not trustworthy.
- Join types and many-to-one/many-to-many growth are explained (`06/README.md:107-201`), but **row grain**, **unit of observation**, **key uniqueness**, and **cardinality** are not defined as a pre-merge contract. One-to-one and one-to-many validation are not operationalized.
- The core never teaches `validate=` and does not include `indicator=True` in the lecture reference (`06/README.md:72-83`). Demo 1 does teach `indicator=True` (`06/demo/demo1_merge_operations.ipynb:265-310`), creating a lecture/demo mismatch, but still does not make `validate=` the normal safety control.
- `DataFrame.join()` and `combine_first()` appear in core (`06/README.md:309-376`) and again in bonus (`06/BONUS.md:226-390`, `06/BONUS.md:861-909`). This is duplication rather than deliberate reinforcement.
- Concatenation and wide/long explanations are useful (`06/README.md:382-577`, `06/README.md:583-673`), but structural `pivot()` is immediately expanded into aggregating `pivot_table()` (`06/README.md:677-750`) and GroupBy examples (`06/README.md:784-790`) before Lecture 08.
- The core’s basic MultiIndex section is generated through GroupBy (`06/README.md:915-958`), requiring a concept not yet canonically taught.
- Demo 3 is 596 lines and crosses into datetime indexes, `resample('QE')`, `combine_first`, GroupBy, MultiIndex, and year-over-year pivot tables (`06/demo/demo3_concat_timeseries.md:14-24`, `06/demo/demo3_concat_timeseries.md:115-147`, `06/demo/demo3_concat_timeseries.md:240-323`, `06/demo/demo3_concat_timeseries.md:444-517`). It is not a focused concat/index demo.
- Assignment schemas do not match the generator. The README promises customer `email`, `state`, `join_date`, and product `stock` (`06/assignment/README.md:145-177`), while the generator creates customer `name`, `city`, `signup_date` and product `product_name`, `category`, `price` only (`06/assignment/data_generator.md:66-73`, `06/assignment/data_generator.md:98-125`).
- The README requires `output/q1_validation.txt` (`06/assignment/README.md:76-90`, `06/assignment/README.md:181-189`) and the notebook scaffolds it (`06/assignment/assignment.md:175-200`), but the test explicitly says that validation is no longer required (`06/assignment/.github/test/test_assignment.py:83`).
- The assignment claims vertical and horizontal concat (`06/assignment/README.md:92-104`), but the test only checks the horizontal output, calculates `has_nan`, and never asserts it (`06/assignment/.github/test/test_assignment.py:183-196`).
- Generated tables contain valid foreign keys and unique dimension keys (`06/assignment/data_generator.md:142-149`, `06/assignment/data_generator.md:157-180`), so a wrong merge can pass structural tests. The tests mainly require nonempty data and expected-looking columns (`06/assignment/.github/test/test_assignment.py:41-80`, `06/assignment/.github/test/test_assignment.py:160-180`).
- Question 3 groups by month/category, uses `pivot_table`, and writes top/bottom analysis (`06/assignment/README.md:106-120`; `06/assignment/assignment.md:290-369`), assessing aggregation and early datetime grouping rather than structural reshape.
- The source contains an obsolete Classroom link and multiple unresolved `attachment:` images (`06/README.md:3-17`, `06/README.md:118-162`, and later attachment blocks).

### Proposed measurable objectives

By the end of Lecture 06, students should be able to:

1. State the row grain of each input table, identify candidate/foreign keys, and test whether the claimed keys are unique.
2. Predict one-to-one, one-to-many, many-to-one, or many-to-many merge behavior and choose an appropriate join type for a stated preservation goal.
3. Perform a merge with explicit keys, `validate=`, and `indicator=True`; inspect unmatched rows; and verify row-count/key invariants.
4. Concatenate tables vertically when their schemas represent the same row grain and horizontally when index alignment is deliberate, explaining the resulting missing values.
5. Convert a table between wide and long form with `melt()` and structural `pivot()`, explaining the uniqueness condition required for a lossless round trip.

### Exact prerequisites

Incoming required skills:

- Run a notebook from a clean runtime and use portable paths.
- Inspect DataFrame columns, shape, dtypes, index, missing values, and uniqueness.
- Select, filter, sort, and derive columns.
- Preserve raw versus cleaned data and validate explicit invariants.
- State what one row represents and identify candidate identifiers.

Lecture 06 must not assume:

- GroupBy aggregation, aggregating pivot tables, MultiIndex manipulation, datetime resampling, rolling analysis, or visualization design.

Outgoing required skills for Lecture 07:

- Produce one prepared analysis table with a stated row grain.
- Convert supplied wide data to long form suitable for seaborn.
- Explain when rows were added/lost through a join or concat.

Outgoing required skills for Lecture 08:

- State the unit of analysis before aggregation.
- Distinguish keys used to join from grouping keys used to summarize.
- Understand that a join can change the number/meaning of rows before any GroupBy.
- Use structural wide/long terminology; Lecture 08 adds aggregation and result grain.

### First-definition and first-independent-use ledger

| Term or skill | Current first definition/guided use | Current first independent use | Finding and required action |
|---|---|---|---|
| Row grain / unit of observation | Absent | Assignment merges and aggregates without declaring it (`06/assignment/README.md:76-120`) | P0 addition. This must be the first concept before keys or join type. |
| Key / foreign key | “Shared keys” named at `06/README.md:35`; merge syntax at `06/README.md:72-83` | Assignment independently joins three tables at `06/assignment/README.md:76-90` | Define candidate, primary, and foreign key operationally; require uniqueness checks. |
| Cardinality | Relationship types described at `06/README.md:164-201`; term not used | Students compare joins and multi-column merge without validating cardinality (`06/assignment/README.md:76-90`) | Define one-to-one, one-to-many, many-to-one, many-to-many and connect each to `validate=`. |
| Join type | Defined at `06/README.md:107-154` | Assignment compares join types at `06/assignment/README.md:78-88` | Keep, but choose from a preservation question after grain/cardinality rather than memorize four diagrams. |
| `validate=` | Absent from lecture/demo/assignment | Not required | P0 addition. Make it the default merge contract and assess failure on duplicate keys. |
| `indicator=True` / unmatched row | Absent from lecture; guided in demo notebook at `06/demo/demo1_merge_operations.ipynb:265-310` | Assignment requests a validation report but not source indicators (`06/assignment/assignment.md:175-200`) | Move into core lecture and require unmatched-row inspection. |
| Concatenation | Defined and contrasted with merge at `06/README.md:382-386` | Assignment requires vertical/horizontal concat at `06/assignment/README.md:92-104` | Keep; explicitly tie vertical concat to same grain/schema and horizontal concat to index alignment. |
| Wide / long | Defined at `06/README.md:583-673` | Assignment independently uses aggregating pivot and melt at `06/assignment/README.md:106-120` | Keep structural reshape; defer aggregation to Lecture 08. |
| `pivot()` | Defined with uniqueness constraint at `06/README.md:677-710` | Assignment uses `pivot_table`, not structural pivot (`06/assignment/assignment.md:308-345`) | Assess `pivot()` round-trip with unique combinations. Introduce `pivot_table()` only as a preview of Lecture 08. |
| MultiIndex | Objective at `06/README.md:29`; definition only near end at `06/README.md:915-958` | Assignment output via `pivot_table` can create index/columns before the definition is useful | Move to Lecture 08/bonus; retain only `reset_index()` as a practical cleanup when needed. |

### Objective-to-artifact alignment matrix

| Proposed objective | Revised lecture section | Required demo | Redesigned assignment evidence | Test/rubric evidence | Current status/action |
|---|---|---|---|---|---|
| 1. State grain and test keys | “Before combining: rows and keys” | Demo 1 begins with table contracts and deliberate duplicate key | Grain/key table for every input | Tests uniqueness claims and detects planted duplicate | **Untaught but operationally required** in current assignment. |
| 2. Predict cardinality and choose join | “Cardinality predicts row counts” | Demo 1 predicts then performs join types | Written prediction plus selected join for preservation goal | Test row counts, preserved IDs, and join type consequence | **Partially aligned**: relationship types present; no explicit contract or prediction evidence. |
| 3. Validate merge and inspect unmatched | “Safe merge workflow” | Demo 1 uses `validate=` and `indicator=True` | Merged table plus structured merge audit | Tests duplicate-key failure, unmatched IDs, row/key invariants | **Taught only in part of demo**: indicator present, validate absent, tests weak. |
| 4. Concatenate with alignment semantics | “Same grain: vertical; aligned index: horizontal” | Demo 2: vertical and horizontal concat | One vertical result and one deliberate alignment result | Tests exact row union/source labels and expected missing positions | **Partially aligned**: both assigned, vertical evidence and `has_nan` are not asserted. |
| 5. Lossless structural reshape | “Wide, long, melt, pivot” | Demo 3: wide→long→wide round trip | Long output and reconstructed wide output | Tests row count, unique identifier-variable pairs, and equality after round trip | **Partially aligned**: current work uses aggregating pivot/groupby and shape-only tests. |

### Recommended required demos

1. **Validated merge diagnostics.** Declare grain/keys; plant an orphan foreign key and duplicate dimension key; predict cardinality; use `validate=`; inspect `_merge`; check invariants; repair the duplicate-key problem.
2. **Concatenation and index alignment.** Vertically stack same-grain monthly files with a source column; horizontally align two small feature tables with deliberately mismatched labels; explain exactly where missing values come from.
3. **Structural wide/long round trip.** Use `melt` and `pivot` on unique identifier-variable combinations; deliberately create a duplicate combination, explain why `pivot` refuses it, and preview that Lecture 08 will aggregate duplicates.

Required demos should not use `resample`, GroupBy summaries, `combine_first`, advanced MultiIndex, year-over-year analysis, or plotting.

### Assignment redesign contract

**Student artifacts:** one Colab/local-compatible notebook, pinned input CSVs with explicit schema/grain notes, a merged output, a merge-audit output, a vertically concatenated output, a long-form output, and a reconstructed wide output.

**Tasks:**

1. State grain and expected key uniqueness for each input.
2. Diagnose a planted duplicate-key fixture and an orphan foreign key.
3. Perform the intended many-to-one merge with `validate=` and `indicator=True`, preserve the required base-table rows, and save the unmatched-row audit.
4. Vertically concatenate same-schema partitions and preserve source provenance.
5. Reshape a unique wide table to long and back without aggregation.

**Automated evidence:** exact fixture version; intentional orphan/duplicate cases; expected exception or diagnostic before repair; exact preserved IDs/row counts; correct `_merge` counts; concat source counts; structural round-trip equality. Tests must fail plausible but wrong joins and not merely look for columns.

**Human evidence:** grain statements, cardinality predictions, join choice rationale, and explanation of alignment-created missingness.

**Remove from required assignment:** monthly grouping, aggregating `pivot_table`, category summaries, top/bottom reports, and advanced analysis. Current leakage is at `06/assignment/README.md:106-120` and `06/assignment/assignment.md:290-369`.

### Content disposition

| Material | Disposition | Evidence and rationale |
|---|---|---|
| Row grain, identifiers, key uniqueness | Add/move earlier | Missing prerequisite for every merge and later GroupBy. |
| Four join types | Keep/revise | Current diagrams/examples at `06/README.md:37-154`; teach through preservation questions. |
| Many-to-one/many-to-many | Keep/revise | Useful warnings at `06/README.md:164-201`; connect to cardinality vocabulary and `validate=`. |
| Composite keys and suffixes | Keep | Daily-use extensions at `06/README.md:224-307`; place after validated basic merge. |
| `validate=` and `indicator=True` | Add to core | Safety tools must precede independent merges; indicator currently exists only in demo notebook. |
| `DataFrame.join()` and `combine_first()` | Move to bonus/consolidate | Duplicated between core (`06/README.md:309-376`) and bonus (`06/BONUS.md:226-390`, `06/BONUS.md:861-909`). |
| Vertical/horizontal concat | Keep/revise | Core at `06/README.md:382-577`; foreground same-grain and index-alignment conditions. |
| Wide/long, `melt`, structural `pivot` | Keep | Strong conceptual fit at `06/README.md:583-710`; remove embedded GroupBy examples. |
| Aggregating `pivot_table` | Move to Lecture 08 | Current `06/README.md:712-750` combines reshape with aggregation before split–apply–combine. |
| Basic/advanced MultiIndex | Move to Lecture 08/bonus | Current core `06/README.md:915-958` relies on GroupBy; advanced work is already in bonus. |
| Datetime index/resampling/time-series demo | Move to Lecture 09 | Current Demo 3 `06/demo/demo3_concat_timeseries.md:115-147` jumps ahead. |
| Attachment images and active Classroom URL | Replace/drop | Unresolved/nonportable assets and obsolete distribution links are not curriculum content. |

### Ordered content outline

1. Why combining tables is a question about row meaning.
2. Row grain and candidate/primary/foreign keys.
3. Key uniqueness checks.
4. Cardinality: one-to-one, one-to-many, many-to-one, many-to-many.
5. Preservation goal and join-type choice.
6. Explicit-key `merge` with `validate=`.
7. `indicator=True`, unmatched rows, suffixes, row/key invariants.
8. Composite keys.
9. Concatenation versus merge.
10. Vertical concat for same-grain/schema partitions and source provenance.
11. Horizontal concat and index alignment.
12. Wide versus long structure.
13. `melt` and lossless `pivot` with uniqueness requirement.
14. Handoff: aggregation of duplicate combinations belongs in Lecture 08.

## Lecture 07: visualization from question to accessible claim

### Intended role

Lecture 07 teaches students to move from a question and audience to an honest, accessible chart, using a focused pandas/Matplotlib/seaborn toolset and a clear exploratory-to-explanatory workflow.

### Current evidence and blockers

- There is no measurable objective or prerequisite block. The file provides an outline (`07/README.md:13-20`) and starts with Tufte principles before establishing question, audience, claim, variable roles, or exploratory versus explanatory purpose.
- Data-ink ratio, chartjunk, and lie factor are defined early (`07/README.md:41-93`). These can support critique, but they currently dominate the opening and are not converted into an assignment/rubric decision process.
- The truncated-axis example implies that starting at zero is generally the honest choice (`07/README.md:107-111`). Zero is often important for bar-length encodings, but line/scatter contexts require a more nuanced scale-and-context rule.
- Accessibility receives one color-palette bullet (`07/README.md:113-128`). The lecture does not define redundant encodings, contrast, readable labeling, alternative text/captions, or a student-facing accessibility checklist.
- The required core surveys pandas, Matplotlib, seaborn, Altair, plotnine, Bokeh, and Plotly (`07/README.md:147-194`, `07/README.md:557-834`). Demo 3 likewise covers Altair interactivity, regression transforms, time series, several export formats, plotnine, Bokeh, and Plotly (`07/demo/DEMO_GUIDE.md:29-39`; `07/demo/demo3_pandas_altair.md:1-9`, `07/demo/demo3_pandas_altair.md:145-296`). This is breadth without sufficient guided practice.
- The Matplotlib section defines a `Figure` but calls its children “subplots (individual plot areas)” (`07/README.md:202-212`); Demo 1 more accurately identifies them as `Axes` (`07/demo/demo1_matplotlib_basics.md:20-26`). The core should define Figure/Axes before using `ax` freely.
- The core reference recommends nonexistent `ax.set_style('seaborn')` (`07/README.md:244-257`) and deprecated `sns.distplot` (`07/README.md:508-540`). Bonus repeats an invalid `distplot(data=df, x=...)` call (`07/BONUS.md:314-316`).
- Demo 2 introduces regression plots before Lecture 10 (`07/demo/demo2_seaborn_statistical.md:135-162`) and uses `sns.load_dataset('tips')`, a network dependency (`07/demo/demo2_seaborn_statistical.md:97-108`). Pair/joint/KDE/violin material further expands required scope (`07/demo/demo2_seaborn_statistical.md:135-209`).
- Assignment README says there are four questions but documents only three (`07/assignment/README.md:58-113`).
- The README and tests require `q1_multi_panel.png` (`07/assignment/README.md:76-87`; `07/assignment/.github/test/test_assignment.py:52-58`), but the assignment notebook never saves that file. It only saves `q1_matplotlib_plots.png` (`07/assignment/assignment.md:94-120`) and lists the missing artifact later (`07/assignment/assignment.md:243-252`).
- The assignment requires correlation analysis (`07/assignment/README.md:89-100`; `07/assignment/assignment.md:164-189`) and rolling/monthly resampling (`07/assignment/assignment.md:191-217`) before the modeling and time-series lectures.
- The public tests check only image existence and byte size (`07/assignment/.github/test/test_assignment.py:43-137`). A blank, misleading, inaccessible, or mislabeled image can satisfy the grader.
- The generator uses `datetime.now()` for transaction and registration dates (`07/assignment/data_generator.md:30-32`, `07/assignment/data_generator.md:75-76`), so the data and expected time window change each run.

### Proposed measurable objectives

By the end of Lecture 07, students should be able to:

1. State a visualization’s question, audience, and intended claim; identify the variable roles and comparison the chart must support.
2. Select an appropriate line, bar, scatter, histogram, or box plot and construct it with the Matplotlib `Figure`/`Axes` model, complete labels, and deterministic data.
3. Critique and revise a chart for visual integrity, including scale, comparable baselines, truthful area/length encodings, context, and avoidance of unsupported claims.
4. Apply core accessibility practices: readable labels, sufficient contrast, colorblind-safe choices, redundant encoding when needed, and a concise text alternative/caption.
5. Use pandas/seaborn for a bounded exploratory view, then create and export one annotated explanatory chart tailored to an audience and claim.

### Exact prerequisites

Incoming required skills:

- Run a notebook from a clean runtime and use portable paths.
- Load a prepared DataFrame; inspect/select/filter/sort it.
- Understand the table’s row grain and use a supplied long-form table.
- Use only already-prepared summaries when a chart needs aggregation; do not require independent GroupBy mastery.

Lecture 07 must not assume:

- correlation interpretation, regression, rolling windows, resampling, inferential uncertainty, interactive-dashboard design, or grammar-of-graphics APIs.

Outgoing required skills for Lecture 08:

- State question, audience, claim, and unit displayed.
- Use a prepared long-form table and know what each mark represents.
- Make a basic chart from a grouped result supplied by Lecture 08 without visualization becoming a second independent learning objective.
- Explain that aggregation changes the unit/row count represented in the chart.

### First-definition and first-independent-use ledger

| Term or skill | Current first definition/guided use | Current first independent use | Finding and required action |
|---|---|---|---|
| Question / audience / claim | No core definition; “questions” appears only in chart-selection prose (`07/README.md:130-144`); audience appears late in Demo 3 (`07/demo/demo3_pandas_altair.md:480-482`) | Assignment prescribes plot types rather than asking students to justify a choice (`07/assignment/assignment.md:37-240`) | Add as the opening visualization contract and assess it. |
| Exploratory / explanatory | Not defined as a workflow; only image filenames/references are present | Assignment calls work exploration but also requires dashboard-like final images (`07/assignment/README.md:102-113`) | Define before APIs; require one of each with different purposes. |
| Encoding / mark | Encoding is introduced only in the Altair section (`07/README.md:615-645`) | No required core independent use except tool-specific Altair | Move the plain-language idea earlier—data mapped to position/length/color/shape—without requiring Altair. |
| Figure / Axes | Figure/subplot described at `07/README.md:202-212`; Demo 1 accurately defines Axes at `07/demo/demo1_matplotlib_basics.md:20-26` | Assignment uses `fig, axes = plt.subplots(...)` immediately at `07/assignment/assignment.md:37-64` | Define both precisely in core before any `ax` method; assess one single-Axes and one multi-Axes use only if both are objectives. |
| Data-ink / chartjunk / lie factor | Defined at `07/README.md:41-93` | Assignment does not ask students to critique or revise a misleading chart | Taught but unassessed. Use as critique vocabulary after question/audience/claim. |
| Accessibility | One colorblind-resource bullet at `07/README.md:113-128` | No assignment requirement or test | P0/P1 addition. Make a checklist and human-rubric row. |
| Correlation / regression | Mentioned in core/assignment before Lecture 10 (`07/README.md:121`; `07/assignment/assignment.md:164-189`; Demo 2 at `07/demo/demo2_seaborn_statistical.md:152-155`) | Students calculate correlation and create heatmap in Assignment 07 | Move later. Relationship visualization does not require inferential/regression interpretation. |
| Rolling / resampling | Not taught in Lecture 07 | Assignment requires rolling 7-day average and monthly resample (`07/assignment/assignment.md:191-217`) | **Untaught but assessed**; move to Lecture 09. |

### Objective-to-artifact alignment matrix

| Proposed objective | Revised lecture section | Required demo | Redesigned assignment evidence | Test/rubric evidence | Current status/action |
|---|---|---|---|---|---|
| 1. Question, audience, claim, variable roles | “Start with the communication contract” | Demo 1 critiques charts against three different audiences | Written contract before each submitted chart | Human rubric checks specificity and chart/claim fit | **Absent** in current core/assignment. |
| 2. Choose chart and use Figure/Axes | “Data roles, chart choice, Figure/Axes” | Demo 2 builds core chart types from prepared data | One exploratory chart and one explanatory chart | Tests files/data table; human rubric checks chart choice and construction | **Partially aligned**: tools practiced, current assignment prescribes rather than tests choice. |
| 3. Critique visual integrity | “Honesty: scales, comparisons, context, encodings” | Demo 1 before/after repair | Revise one supplied flawed chart and explain changes | Human rubric; test only deterministic source/output existence | **Taught but unassessed**; current zero-axis guidance needs nuance. |
| 4. Accessibility | “Readable by more people” | Every demo applies palette, contrast, redundant cue, labels, text alternative | Accessibility checklist and caption/alt text for final chart | Human rubric; optionally static checks for nonempty labels/metadata where feasible | **Orphaned/absent** beyond one palette bullet. |
| 5. Exploratory→explanatory workflow and export | “Explore, focus, annotate, export” | Demo 3 uses pandas/seaborn briefly, then Matplotlib refinement | One exploratory view, one annotated final chart, supporting data CSV | Tests deterministic input/output and supporting data; human rubric checks story/annotation | **Partially aligned**: export exists, but current six-image assignment and tests do not assess communication. |

### Recommended required demos

1. **Critique and redesign.** Begin with question/audience/claim; identify misleading scale, decoration, weak labeling, inaccessible color, and unsupported claim; revise one chart and explain every decision.
2. **Matplotlib Figure/Axes fundamentals.** Use prepared deterministic data to construct line, bar, scatter, and histogram examples; label, annotate, use one small-multiple example only if necessary, and export.
3. **Exploratory to explanatory with pandas/seaborn.** Make one quick exploratory view from long-form data, choose a focused finding, rebuild it as an accessible explanatory chart, add context/annotation, and export with a text alternative.

Altair, plotnine, Bokeh, Plotly, advanced seaborn, animation, dashboards, and interactivity should be optional demonstrations/bonus—not required demos.

### Assignment redesign contract

**Student artifacts:** one Colab/local-compatible notebook, one pinned prepared dataset (already joined/reshaped), one supporting-data CSV for the final chart, and two final image artifacts at most.

**Tasks:**

1. State question, audience, claim, unit shown, and variable roles.
2. Create one exploratory chart and record what it helps inspect without overstating a conclusion.
3. Critique one supplied flawed chart and produce a corrected version.
4. Create one explanatory chart with title, labels/units, accessible palette or redundant encoding, annotation/context, and text alternative/caption.
5. Restart and run all, then export deterministically.

**Automated evidence:** notebook clean execution; pinned data checksum; required supporting-data schema and values; image existence, dimensions, and sane size; deterministic filenames; no network-only dataset dependency.

**Human evidence:** question/audience/claim alignment, chart-type choice, integrity, accessibility, annotation, caption/alternative text, and clarity. Classroom 50 public tests cannot certify these judgments.

**Remove from required assignment:** correlation heatmap, regression, rolling average, resampling, six-image artifact checklist, and broad dashboard. Current out-of-scope work is at `07/assignment/README.md:76-113` and `07/assignment/assignment.md:123-240`.

### Content disposition

| Material | Disposition | Evidence and rationale |
|---|---|---|
| Question/audience/claim and exploratory/explanatory distinction | Add/move earlier | Missing organizing purpose for all later tool choices. |
| Data-ink, chartjunk, lie factor, integrity | Keep/reorder/revise | Useful definitions at `07/README.md:41-93`; place after communication contract and avoid presenting one universal scale rule. |
| Color and accessibility | Keep/strengthen | Current core only covers palette choice and one accessibility link (`07/README.md:113-128`). |
| Chart-selection guide | Keep/revise | Useful at `07/README.md:130-144`; base choice on variable role, comparison, and claim. |
| Matplotlib Figure/Axes, core chart types, labels, annotation, export | Keep/focus | Core practical skill at `07/README.md:196-334`; fix invalid API guidance. |
| pandas plotting | Keep as bounded exploratory shortcut | Useful at `07/README.md:336-420`; it should not become a second plotting API catalog. |
| seaborn scatter/box/hist/bar with long-form data | Keep/focus | Useful at `07/README.md:422-470`; connect to prepared long-form data from Lecture 06. |
| Pair/joint/violin/KDE/regression plots | Move to bonus/later | Current `07/README.md:472-555` and Demo 2 broaden scope and introduce statistical concepts. |
| Altair core tutorial and interactivity | Move to bonus | Current `07/README.md:557-743` is a separate grammar/API curriculum. |
| plotnine, Bokeh, Plotly survey | Move to bonus/drop from required path | Current `07/README.md:745-834` adds three more ecosystems without assessment depth. |
| Correlation/regression assignment | Move to Lecture 10 | Current `07/assignment/assignment.md:164-189` precedes modeling definitions. |
| Rolling/monthly time-series assignment | Move to Lecture 09 | Current `07/assignment/assignment.md:191-217` is untaught in this lecture. |
| Network datasets and mutable dates | Replace | `sns.load_dataset` and `datetime.now()` undermine offline/Colab reproducibility. |
| Obsolete/invalid APIs | Correct or drop | `ax.set_style` at `07/README.md:257`; `sns.distplot` at `07/README.md:519` and `07/BONUS.md:316`. |

### Ordered content outline

1. Why visualize: question, audience, claim.
2. Unit shown and variable roles.
3. Exploratory versus explanatory purpose.
4. Data-to-visual encoding: position, length, color, shape.
5. Choose line/bar/scatter/histogram/box plot by task.
6. Integrity: scales, comparable baselines, area/length, context, unsupported claims.
7. Accessibility: labels/units, contrast, color, redundant cues, text alternatives.
8. Matplotlib Figure and Axes model.
9. Construct and label core chart types.
10. pandas/seaborn for bounded exploration with long-form data.
11. Move from exploratory result to one explanatory chart.
12. Annotation, caption, export, and fresh-runtime verification.

## Boundary verification

### Lecture 03 → 04 handoff

| Check | Current result | Evidence | Required correction |
|---|---|---|---|
| Lecture 03 is explicitly the final pre-Jupyter lecture | Pass | `03/README.md:14-19` | Preserve this statement and keep notebooks out of Lecture 03. |
| Students arrive with scripts, functions/modules, environments, and NumPy structure | Pass with known L03 audit caveats | `03/README.md:3-19`, `03/README.md:293-410` | Lecture 04 should name these exact incoming capabilities, not merely “Lectures 1–3.” |
| Notebook/cell/kernel are defined before use | Partial pass | `04/README.md:23-40`; assignment use at `04/assignment/README.md:5-17` | Add runtime/state/output/order before assignment use. |
| Stale state is defined and practiced | Gap | Guide demonstrates a restart failure at `04/demo/DEMO_GUIDE.md:103-114`; notebook only reuses a variable at `04/demo/demo1_jupyter_basics.md:60-69` | Commit the failure/repair sequence as the required demo and assess clean execution. |
| NumPy is explicitly bridged to pandas | Gap | Only “built on NumPy” at `04/README.md:204-206`; separate Series/DataFrame definitions at `04/README.md:235-268` | Map 1D/2D values, shape/dtype/axis to labels/index/columns/dtypes. |
| Colab-first and local Jupyter equivalent are defined | Gap/P0 | Current material is VS Code-specific (`04/README.md:132-145`) | Add runtime/filesystem/path/persistence/setup contract and certify both paths. |
| Lecture 04 does not require later concepts | Fail | Cleaning at `04/README.md:469-523`; GroupBy at `04/README.md:574-595`; assignment at `04/assignment/README.md:33-59` | Remove cleaning/aggregation from required L04 lecture, demos, assignment, and tests. |
| Fresh checkout/runtime creates its own paths | Fail | Generator writes to an uncreated directory at `04/assignment/data_generator.ipynb:78-81` | Standard setup cell creates `data/` and `output/`, resolves project root, and verifies files. |

### Lecture 07 → 08 handoff

| Check | Current result | Evidence | Required correction |
|---|---|---|---|
| Lecture 07 hands off question/audience/claim | Fail | No core definition; audience appears only late in Demo 3 (`07/demo/demo3_pandas_altair.md:480-482`) | Make this the opening and assessed contract in Lecture 07. |
| Lecture 07 hands off prepared long-form visualization data | Partial | Lecture 06 explains wide/long at `06/README.md:583-673`; Lecture 07 uses seaborn but does not explicitly connect the handoff | Use a prepared long table and explain one row/mark in required Demo 3. |
| Figure/Axes and basic export are secure | Partial pass | `07/README.md:202-212`; `07/demo/demo1_matplotlib_basics.md:20-26`, `07/demo/demo1_matplotlib_basics.md:234-269` | Define Axes precisely in core and assess a focused use rather than six artifacts. |
| Lecture 07 assignment avoids aggregation/time/modeling prerequisites | Fail | Correlation at `07/assignment/assignment.md:164-189`; rolling/resample at `07/assignment/assignment.md:191-217` | Move these requirements to Lectures 10 and 09 respectively. |
| Lecture 08 begins with unit of analysis and grouping key | Gap | It begins with split–apply–combine at `08/README.md:22-69`, but does not first define the input/output unit | Add unit of analysis, grouping key, result grain, and expected row count before the first GroupBy. |
| Lecture 08 adds aggregation rather than reteaching visualization | Fail/overlap | Assignment 08 requires a pivot visualization at `08/assignment/README.md:117-128` | Visualization may consume the aggregation result as a supplied final step, but it should not be a separately graded Lecture 08 objective. |
| Pivot boundary is coherent | Gap | Lecture 06 teaches `pivot_table` at `06/README.md:712-750`; Lecture 08 claims pivot-table basics at `08/README.md:11-20` | Lecture 06 owns structural `melt`/`pivot`; Lecture 08 owns aggregating `pivot_table` after GroupBy and result-grain definitions. |

## Cross-range term-first-use priorities

| Term/capability | Canonical first definition | First independent-use gate |
|---|---|---|
| notebook, cell, kernel/runtime, state, execution order, output | Lecture 04 opening | Before any assignment notebook is run or repaired |
| Series, DataFrame, index, columns, labels, positions, dtypes | Lecture 04 NumPy bridge | Before selection/filtering assignment tasks |
| raw/clean data, schema, sentinel, imputation, duplicate identity, invariant, provenance | Lecture 05 before cleaning APIs | Before students choose/drop/fill/convert or submit clean data |
| row grain, candidate/foreign key, key uniqueness, cardinality, unmatched row | Lecture 06 before first merge | Before any merge, post-merge calculation, or aggregation |
| concat alignment, wide, long, `melt`, structural `pivot` | Lecture 06 after validated merge | Before students reshape independently or prepare seaborn data |
| question, audience, claim, exploratory/explanatory, encoding, Figure, Axes, accessibility | Lecture 07 before plotting APIs | Before chart selection, critique, or final visualization |
| unit of analysis, grouping key, split–apply–combine, aggregation, transform, result grain | Lecture 08 opening | Before GroupBy, aggregating pivot, or grouped visualization |

## Prioritized change contract

### P0: broken prerequisite or assessment alignment

1. Remove cleaning and GroupBy from Lecture/Assignment 04 and add Colab-first runtime/path/state instruction.
2. Replace Assignment 05’s 125-point/100-point contradiction and non-pipeline artifact chain with one coherent cleaning assessment.
3. Remove invalid cross-sectional forward fill and generic missingness thresholds as required decision rules.
4. Add row grain, key uniqueness, cardinality, `validate=`, and unmatched-row inspection before independent Lecture 06 merges.
5. Reconcile Assignment 06 schemas, validation artifact, generator fixtures, and tests; include orphan and duplicate-key cases.
6. Remove correlation/regression/rolling/resampling from Assignment 07 and add question/audience/claim plus accessibility evidence.
7. Replace image-existence-only grading with a hybrid automated/human contract for visual communication.

### P1: organization and scope

1. Add measurable objective and prerequisite blocks to Lectures 04, 05, and 07; revise Lecture 06 objectives.
2. Adopt the ordered outlines in this artifact and enforce one canonical concept home.
3. Reduce each lecture to the recommended two or three required demos.
4. Split structural reshape in Lecture 06 from aggregating pivots in Lecture 08.
5. Move configuration systems, advanced plotting libraries, MultiIndex depth, time operations, and modeling vocabulary to bonus/later lectures.
6. Standardize a Colab-first/local-Jupyter notebook header, portable-path helper, clean-runtime check, and deterministic data rule across 04–07.

### P2: release hygiene

1. Resolve or remove Lecture 04 FIXME media blocks.
2. Replace Lecture 06 `attachment:` images and obsolete Classroom link.
3. Correct malformed code/reference formatting and invalid/deprecated visualization APIs.
4. Remove mutable `datetime.now()` assignment generation and network-only demo datasets.

## Instructor decisions still required

1. Confirm whether Lecture 05 remains a 100-point midterm. The redesign works either way, but the breadth and human-rubric weight depend on that assessment decision.
2. **Resolved in the course map:** clear sensitive outputs; restart/run-all before submission; the grader executes a fresh copy and ignores stored output as execution evidence; retain ordinary output only when a human rubric needs it. Required files under `output/` remain a separate artifact contract.
3. **Resolved in the course map:** structural `melt`/`pivot` belongs in Lecture 06; aggregating `pivot_table` belongs in Lecture 08.
4. Approve human grading for visualization quality/accessibility. Public Classroom 50 tests can verify reproducibility and artifacts but cannot determine whether a chart communicates honestly.

## Post-edit verification gates

After source revisions, a reviewer who did not write them should verify:

- every proposed objective has a lecture definition/example, guided demo, assignment artifact, and test/rubric row;
- no `untaught but assessed` row remains;
- compatible required demo notebooks execute in a clean Colab runtime and local Jupyter; assignment notebooks execute locally and in Colab only when assignment Colab support is approved;
- notebooks do not depend on prior cell order, undeclared local files, network-only datasets, or mutable current dates;
- Assignment 05 totals exactly the declared score and its pipeline stages consume the prior stage’s artifact;
- Assignment 06 fails known wrong-cardinality, orphan-key, duplicate-key, wrong-join, and wrong-reshape fixtures;
- Assignment 07’s automated tests are limited to reproducible facts and its human rubric assesses question/audience/claim, integrity, accessibility, and clarity;
- the Lecture 03→04 and Lecture 07→08 boundary tables above pass without exceptions.
