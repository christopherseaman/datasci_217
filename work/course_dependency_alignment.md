# DataSci 217 dependency and alignment map

Status: working curriculum contract for the 2026–27 refresh

This document converts the pedagogical audit into an actionable course sequence. It is the cross-course companion to the detailed range matrices under `work/reviews/`.

Detailed evidence and artifact contracts:

- [`reviews/lectures_01_03_alignment.md`](reviews/lectures_01_03_alignment.md)
- [`reviews/lectures_04_07_alignment.md`](reviews/lectures_04_07_alignment.md)
- [`reviews/lectures_08_11_alignment.md`](reviews/lectures_08_11_alignment.md)

## Working decisions

- Lectures 01–03 use terminal-executed Python and shell files. Jupyter concepts and notebooks begin in Lecture 04.
- Required Git instruction is GUI-first through VS Code Source Control or GitHub Desktop. Command-line Git is bonus material unless a command is unavoidable for an instructor or recovery workflow.
- Lectures 01–05 form the foundational path. Lectures 06–11 form the advanced path.
- Each lecture has two or three required demos after its lecture narrative is stable.
- Every regular assignment normally has three competence-focused tasks with incremental complexity. Lectures 05 and 11 may use broader assessment, but must retain explicit objective alignment.
- Classroom 50 applies to Assignments 01–11. Its tests and grading bundles are considered discoverable, so public checks cannot be treated as secrets.
- Compatible notebook demos from Lecture 04 onward are Colab-first and also run in local Jupyter. Terminal-only tools remain in a terminal when they are the learning objective.
- Assignments from Lecture 04 onward may use notebooks, but Colab is not a required assignment path until saving back to the repository and the submission workflow have been validated.

## Resolved design decisions

- **Required early shell:** the shared POSIX command subset through Bash on Linux/WSL/supported cloud or the default zsh shell on macOS. Native PowerShell is a setup bridge unless equivalent required materials and tests are deliberately added later.
- **Git interface:** VS Code Source Control or GitHub Desktop for the required path; CLI Git remains bonus.
- **Reshape boundary:** structural `pivot`/`melt` belongs in Lecture 06; aggregating `pivot_table` belongs in Lecture 08 after grouping and result grain are defined.
- **Notebook execution evidence:** students restart and run all before submission, but stored outputs are never trusted as proof of execution. The grader executes a fresh copy. Sensitive outputs are cleared; ordinary outputs remain visible only where a human rubric needs to review a chart or explanation.
- **Visualization grading:** public checks may validate files, dimensions, labels, and deterministic data, but chart integrity, accessibility, and communicative quality receive a concise human checklist.
- **Lecture 03 broadcasting:** retain one simple compatible-shape broadcast in core so students can predict output shape; advanced rules and array combining remain bonus.
- **Dependency recording:** `requirements.txt` lists deliberate direct course dependencies. Do not describe an unreviewed `pip freeze` result as a direct-dependency list. Exact transitive locks belong in the release constraints/lock artifact.
- **Lecture Markdown format:** current lecture sources use standard Markdown: one H1 title followed by H2 sections and deeper nested headings as needed. The historical Notion-import heading convention in `CLAUDE.md` and the archived `implementation_plan.md` is not a 2026–27 release constraint.

## Candidate supported runtime

The compatibility target for the next execution pass is:

| Component | Candidate | Rationale |
|---|---|---|
| Python | 3.12.13 | Matches the repository's declared Python 3.12 minor line and Google's pin-able Colab 2026.04 runtime. It has run the dependency-free structural audit through uv; the full course stack is not yet certified. |
| NumPy | 2.0.2 | Matches Colab 2026.04 and supports the required Lecture 03 array concepts. |
| pandas | 3.0.3 | Current non-yanked pandas 3 release. pandas 3.0.4 is yanked for reported datetime-related segmentation faults. The content refresh should teach pandas 3 copy-on-write, string dtype, GroupBy, and frequency behavior rather than preserve pre-3.0 assumptions. |

Google documents that the Colab 2026.04 runtime uses Python 3.12.13 and NumPy 2.0.2, that past runtimes remain available for one year, and that notebooks may install the library versions they require: <https://research.google.com/colaboratory/runtime-version-faq.html>. pandas 3 supports Python 3.11+ and NumPy 1.26+, so this candidate combination is within its published support floor: <https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v3.0.0.html>. PyPI records pandas 3.0.4 as yanked for reported datetime-related segmentation faults, so it must not be installed or locked: <https://pypi.org/project/pandas/>. pandas 3.0.3 has passed an isolated import/version smoke test with the candidate Python and NumPy versions. The exact constraints are not final until every required notebook, assignment grader, plotting dependency, statsmodels example, and scikit-learn example passes in both environments.

## Canonical concept homes

| Concept family | Canonical home | Earlier use | Later use |
|---|---|---|---|
| paths, working directory, terminal execution | Lecture 01 | none | reinforce as an operational prerequisite |
| scalar values, lists, conditionals, simple loops | Lecture 01 | none | use freely from Lecture 02 |
| Git repository workflow | Lecture 02 | Lecture 01 may prepare accounts/tools only | use the supported GUI workflow for all submissions |
| minimal dictionaries/text I/O; functions, modules, imports, main guard | Lecture 02 | supplied code may contain them only when clearly marked | use freely from Lecture 03 |
| environment and dependency vocabulary | Lecture 03 | setup commands may be copied in Lecture 01 | use for reproducibility from Lecture 04 |
| ndarray, dtype, shape, axis, indexing, views/copies, masks, vectorization, reductions, reshape/simple broadcast | Lecture 03 | none | bridge explicitly to pandas in Lecture 04 |
| notebook/cell/kernel/runtime/state/order/output | Lecture 04 | no independent notebook use | require restart-and-run-all thereafter |
| Series, DataFrame, index/columns/dtypes, label/position selection, filtering, portable I/O | Lecture 04 | none | use freely from Lecture 05 |
| raw/clean data, schema, missingness/sentinels/duplicates, types, categories, validation invariants, provenance | Lecture 05 | fixed or supplied clean data only | reinforce in later workflows |
| row grain, keys, cardinality, merge, concatenate, structural `melt`/`pivot` reshape | Lecture 06 | Lecture 05 states row meaning and candidate identifiers; no independent joins | prerequisite for grouped and integrated work |
| chart purpose, integrity, accessibility, Figure/Axes, focused seaborn | Lecture 07 | small supplied plots may illustrate results | use for communication thereafter |
| grouping unit/key, grouped result grain, GroupBy, named aggregation, transform, aggregating `pivot_table` | Lecture 08 | fixed grouped outputs may be consumed | use freely from Lecture 09 |
| timestamp/period/frequency, entity boundaries, single/panel series, resample, lag, observation/time windows | Lecture 09 | dates may be parsed as values in earlier lectures | prerequisite for temporal modeling/workflows |
| inference/prediction, estimand/target/horizon/availability, association/causation, modeling assumptions/uncertainty, splitting, baselines, evaluation, leakage | Lecture 10 | descriptive relationships only | apply in Lecture 11 |
| integrated question→data→analysis/model→evaluation→communication workflow | Lecture 11 | component workflows are practiced earlier | course culmination |

## Capability dependency graph

```text
orient in a terminal
  → identify the working directory and resolve a relative path
  → create/edit/run a Python script and read a traceback
  → use values, lists, comparisons, conditions, and simple loops
  → define/call a function and distinguish parameter/argument/return
  → use a minimal dictionary and read/write one small text file at a resolved path
  → import a local module without unintended top-level work
  → create and verify an isolated environment
  → install and record a direct dependency
  → inspect an ndarray's dtype, dimensions, shape, and axes
  → select by position, slice, mask, reduce, reshape, and broadcast
  → distinguish a script process from notebook kernel state
  → restart and run a notebook top-to-bottom
  → distinguish Series/DataFrame and label/position selection
  → load, inspect, select, filter, sort, and save tabular data
  → preserve raw input, state row meaning and candidate identifiers, and record a cleaning decision
  → validate cleaning invariants and retain provenance
  → identify keys and join cardinality before merging
  → inspect unmatched rows and verify post-merge grain
  → reshape between wide and long forms
  → choose an honest accessible chart for a stated question
  → define grouping keys and predict grouped output grain
  → distinguish aggregation from transform and create one aggregating pivot table
  → distinguish timestamp/period and regular/irregular, single/panel data
  → identify entity/timestamp keys and distinguish observation-count from elapsed-time windows
  → resample, lag, difference, and roll without crossing entity boundaries
  → distinguish inference/prediction and association/causation
  → define an estimand, or define a prediction target, timestamp/horizon, features, and availability
  → split data appropriately and fit preprocessing only on training data
  → compare a baseline and model on untouched evaluation data
  → communicate an end-to-end result with limitations
```

The submission workflow is a parallel operational thread:

```text
prepare GitHub/VS Code access
  → open a Classroom 50 assignment repository
  → inspect changes in the GUI
  → stage and commit a coherent change
  → pull/push through the supported GUI
  → inspect grader feedback
  → revise and resubmit without losing work
```

Lecture 01's repository access and first synchronization are guided, unassessed onboarding. Independent staging, committing, branching, merging, and synchronization are taught and assessed only after the Git state model is introduced in Lecture 02.

## Lecture entry and exit contracts

| Lecture | Required on entry | Student can do independently on exit |
|---|---|---|
| 01 | operate a computer, browser, and text editor | navigate paths; create/edit/run a small script; use basic values, lists, conditions, and loops; interpret a simple traceback |
| 02 | Lecture 01 exit capabilities | complete the supported GUI Git loop; use a minimal dictionary and small text file; define/test functions; import a safe local module; use a main guard |
| 03 | Lecture 02 exit capabilities | create/verify/recreate one environment; explain direct dependencies; use core ndarray operations in a terminal script |
| 04 | Lecture 03 exit capabilities | explain notebook state; restart/run-all; construct and inspect Series/DataFrames; select/filter/sort; read/write a portable CSV path |
| 05 | Lecture 04 exit capabilities | preserve raw input; state row meaning and candidate identifiers; profile data quality; make justified missing/type/category decisions; validate invariants; rerun one cleaning pipeline from raw to clean with a decision log |
| 06 | Lecture 05 exit capabilities | state grain and keys; validate merge cardinality; inspect unmatched records; concatenate; reshape wide/long without silent duplication |
| 07 | Lecture 06 exit capabilities | select an appropriate chart; construct it with Figure/Axes or focused seaborn; preserve integrity/accessibility; annotate and export |
| 08 | Lecture 07 exit capabilities | state grouping unit/key; use named aggregation and transform; predict output shape/index; produce and interpret one aggregating pivot table |
| 09 | Lecture 08 exit capabilities | define entity/time-series structure and frequency; distinguish observation-count from elapsed-time windows; resample with correct semantics; create lags/differences/windows without entity leakage |
| 10 | Lecture 09 exit capabilities | classify descriptive/inferential/predictive questions; distinguish association from causation; interpret a bounded OLS association model and its uncertainty; build a train-only linear Pipeline; compare it with a baseline; interpret supplied binary metrics; evaluate test once |
| 11 | Lecture 10 exit capabilities | execute and communicate one reproducible end-to-end workflow with a defined question, grain, target/horizon, split, evaluation, and limitations |

## Alignment status rules

Every detailed range matrix uses these statuses:

- **aligned:** objective is defined, demonstrated, practiced, assessed, and checked appropriately;
- **partially aligned:** one or more layers are weak or ambiguous;
- **untaught but assessed:** assignment or grader requires independent use without prior instruction;
- **taught but unpracticed:** lecture defines the capability without a required guided demo;
- **practiced but unassessed:** useful practice exists but the assignment does not measure it;
- **orphaned:** material has no clear role in a current objective or later prerequisite.

No lecture is ready for artifact redesign while it contains an `untaught but assessed` row.

## Cross-course alignment summary

| Lecture | Narrative sequence | Required demos | Assignment contract | Current gate |
|---|---|---|---|---|
| 01 | core/bonus narrative independently verified | exact three terminal demos independently verified | Assignment 01 implemented and adversarially verified | complete |
| 02 | core/bonus narrative independently verified | exact three terminal/GUI demos independently verified | Assignment 02 implemented and adversarially verified | complete |
| 03 | core/bonus narrative independently verified | exact three terminal NumPy demos independently verified | Assignment 03 implemented and adversarially verified | complete |
| 04 | core/bonus narrative independently verified | exact three Colab-first/local notebooks independently verified | one notebook-state+pandas+portable-I/O assignment independently verified | run the Classroom50 notebook pilot; assignment Colab remains conditional |
| 05 | core/bonus narrative independently verified | exact three cleaning notebooks independently verified | one chained cleaning-pipeline assignment independently verified | technical artifact gate complete; syllabus role and Classroom50/Colab pilots remain |
| 06 | core/bonus narrative independently verified | exact three merge/concat/structural-reshape notebooks independently verified | validated merge/concat/structural-reshape assignment independently verified | technical artifact gate complete; provisional score-to-policy mapping remains |
| 07 | core/bonus narrative independently verified | exact three visualization notebooks independently verified | one visualization-evidence assignment independently verified | technical artifact gate complete; Classroom50 policy and conditional assignment-Colab pilot remain |
| 08 | core/bonus narrative independently verified | exact three grouping notebooks independently verified | one grouping/transform/aggregating-pivot assignment independently verified | technical artifact gate complete; Classroom50 policy and conditional assignment-Colab pilot remain; SSH/tmux stays outside the notebook sequence |
| 09 | core/bonus narrative independently verified | exact three time-series notebooks independently verified | independently accepted assignment blueprint; implementation active | implement and independently verify the assignment |
| 10 | core/bonus narrative independently verified | exact three-demo contract defined; design gate next | bounded OLS plus baseline/one linear Pipeline and supplied classification metrics | design demos; defer exact statsmodels/scikit-learn/Matplotlib pins to course-wide certification |
| 11 | audit and implementation-ready narrative outline complete | exact three-demo contract defined but dataset-specific implementation blocked | compact end-to-end capstone | narrative may proceed; demos/assignment wait for assessment role and frozen licensed data/project contract |

## Revision order

1. Finalize the three range matrices and reconcile their shared boundaries here.
2. Revise lecture narratives in order, moving advanced material to `BONUS.md` and removing orphaned material.
3. Recheck adjacent boundaries and the term ledger after each lecture revision.
4. Redesign the two or three required demos for each stable lecture.
5. Redesign and validate assignments against the accepted objectives.
6. Package Assignments 01–11 in Classroom 50 and certify Lecture 04–11 notebook demos for Colab/local Jupyter.
