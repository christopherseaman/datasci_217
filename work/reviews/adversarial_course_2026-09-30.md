# Whole-course adversarial review, 2026-09-30

Six scoped fixing teams and an independent referee reviewed the current Lectures 01–11, BONUS pages, demo instructions and executable sources, assignment handouts, and artifact graders. The earlier 04–11 report supplied context; fresh execution and counterexamples establish this pass's findings. Existing uncommitted work was preserved. No commits, pushes, handout publication, or Notion publication occurred.

The course progression holds. No required assignment skill was found only in BONUS or first taught after its corresponding demo break. Confirmed teaching and grading defects were repaired. The detailed topic, demo-task, and artifact-contract maps are in the linked team reports below.

## Coverage and order

| Lecture | Main progression and corresponding practice |
| --- | --- |
| 01 | Setup/fork/clone/sync → shell paths and files → Python values and operators → decisions, loops, and debugging. Four matching demos. |
| 02 | Git/branches/conflicts → containers, functions, and modules → strings, files, exceptions, and Markdown. Three clinic workflows. |
| 03 | uv projects and shell pipelines → array creation/types/arithmetic/indexing → copies, masks, reshaping, summaries, and ranking. |
| 04 | Notebook state → labeled pandas selection → derivation, deterministic ordering, and file round trips. |
| 05 | Contracts/missingness/types → transformations/text/categories → validation, raw preservation, provenance, and saved cleaning artifacts. |
| 06 | Keys/cardinality/joins → indexes and structural reshape → concatenation/alignment/source priority. |
| 07 | Visualization contracts and matplotlib → pandas/seaborn/statistical plots → critique, accessibility, Altair, and saved specifications. |
| 08 | Group mechanics/counts/aggregation/pivots → group coverage and result shapes → measured performance and local terminal rehearsal. |
| 09 | Dates/indexing/calendar rules/resampling → grids/gaps/windows → UTC/DST/entity-aware history/availability/holdouts. |
| 10 | OLS/uncertainty/availability/splits/cycles → train-fitted pipelines/baselines/evaluation → trees/boosting/networks. |
| 11 | One project walkthrough integrates release audit, cleaning/panel construction, availability-aware preparation, modeling, and evidence. |

There are 32 demo breaks: four in 01, three each in 02–10, and one in 11. Every lecture ends at its final marker. Concept explanations, concrete visuals/output, task-oriented reference material, and minimal snippets were reviewed before styling. The existing McKinney/shell progression needs no broad reorder; Lecture 03's saved shell-script example moved after its pipeline/timestamp prerequisites. Advanced alternatives remain optional. Size targets remain diagnostic: needed beginner setup in 01 and useful reference content in 03/09/10 were retained.

A focused follow-up independently rechecked humor, prose, topic structure, and prerequisite evidence. All 32 blocks contain a relevant comic or visual joke, including images inside Notion columns; no comic is reused across lectures. Lecture 01's installer comic moved beside Python installation rather than comparisons. Ordinary unbulleted prose paragraphs top out at 102 words; no massive, meandering, or off-topic prose section was found. Density varies locally around the Lecture 03 target, and the longer lectures retain focused setup, reference material, and concrete output.

The instructor's recorded dataset decisions remain in force: Lecture 11 keeps NYC taxis, and Assignment 11 keeps Chicago beach weather. These are explicit exceptions to the ordinary health-demo subject rule. Assignment 10 retains its approved **retrospective** 30/8/10 split; it must not be described as an operational prospective forecast. The prospective Demo 10 uses only labels available at its first validation cutoff.

## Teaching and demo repairs

### Classroom pacing follow-up

The first pacing pass below is historical. The instructor subsequently rejected moves that removed functionality from the main lectures. The restoration pass keeps nonduplicate functionality and short method examples in lectures, with extended workflows in demos; its current evidence and scope decisions are recorded after this first-pass account.

Actual teaching of 01–03 left room for only part of one demo. The class budget is now **90 minutes total, aiming for 60 minutes of lecture and 30 minutes of core demonstrations**. Reference completeness and concise prose did not establish classroom fit. Three scoped teams plus an independent reviewer reduced repeated lecture examples and separated every demo's core walkthrough from explicitly after-class independent practice. Lecture 11 preserves its project-workflow exception.

| Lecture | Before pacing | First pacing pass | Current corrected source |
| --- | ---: | ---: | ---: |
| 01 | 953 | 866 | 895 |
| 02 | 870 | 776 | 813 |
| 03 | 1,046 | 939 | 941 |
| 04 | 720 | 690 | 727 |
| 05 | 866 | 812 | 848 |
| 06 | 698 | 653 | 619 |
| 07 | 766 | 718 | 762 |
| 08 | 608 | 512 | 595 |
| 09 | 967 | 869 | 787 |
| 10 | 747 | 654 | 712 |
| 11 | 273 | 240 | 273 |

The first pass retained full setup, expected results, error corrections, and runnable independent practice. Early core routes use short executable examples rather than running long scripts and ignoring most output. Later core routes name actual notebook sections/cells; skipped variants provide no required core state. The first pass reserved extended environment setups, join/plot variations, SSH terminal rehearsal, and boosting/neural-network training for independent practice. The subsequent correction restores SSH/tmux/Jupyter lecture coverage and puts a first XGBoost and Keras fit in Demo 10.3's core. Lecture 11's original full project workflow is restored. Required assignment concepts remain in main lectures before their corresponding breaks.

All **25 required** 04–11 notebooks and their then-current core paths passed fresh Python 3.13/pandas 3.0.5 execution with meaningful output checks in the first pass. The unchanged optional geography pair retains its prior execution attribution. Setup/project/lock inputs and assignments/graders were unchanged in that pass; all 26 notebook installer cells matched the before snapshots. Its selected core paths comprised 193 code cells including setup and supplied data generators; this is a historical workload comparison, not a teaching-time measurement or evidence of the restored paths.

Current maps and evidence: [01–03](../../scratch/course0430/pacing0103/report.md), [04–07](../../scratch/course0430/pacing0407/report.md), [08–11](../../scratch/course0430/pacing0811/report.md), [independent challenge](../../scratch/course0430/pacing-referee/report.md), and [shared verification](../../scratch/course0430/pacing-shared/). The reviewer refuted a section-move bug, full-script core routes, and stale Git history expectations; each received an owning correction. The site-link test keeps asset-copy checks and verifies the clone screenshot still embedded in Lecture 01; the other detailed setup screenshots remain in its demo guide.

First-pass final integration passed: 01–03 demo entry points; 35 notebook pairs against fresh conversion; 22-page lint; all openings, 32 breaks, assets, and 54 unique comic IDs; and the Eleventy build at `scratch/course0430/site-pacing/`, with zero broken internal links/assets on all 26 primary pages. The real site-link test also passed. All 306 recorded assignment/grader/setup inputs matched their before hashes in that pass. Later validation must use the restored sources; older build paths describe earlier review states.

**Timing remains open:** rehearse the selected lecture and core demo route, including setup, questions, and transitions, against the 60/30 budget. Cold installs and novice interaction can change delivery time. No observed 90-minute fit or hosted-platform execution is claimed, and nothing has been published.

### Instructor-directed correction of pacing scope

The instructor rejected removing actual functionality merely to shorten a lecture. The revised rule is to discuss scope changes, remove genuine repetition, keep concepts and compact method examples in main teaching, and put extended workflows in demos. Git's nonduplicate guidance and Markdown's communication role are retained. Optional independent extensions explicitly link instructor-approved BONUS prerequisites and supply no required core state.

| Lecture | Corrected placement |
| --- | --- |
| 01 | Website upload and duck typing restored; all nonduplicate Git guidance retained. |
| 02 | Useful Markdown communication syntax, ignore patterns, editor/terminal conflict handling, imports, and append functionality restored. |
| 03 | venv/Conda described in main, full alternative workflows and approved deeper array selection/sorting in BONUS. The pandas preview is removed. Independent array extensions link their BONUS prerequisites. |
| 04 | Parquet concepts, backend, references, and minimal read/write calls in main; full typed-table round trip in Demo 3 independent practice. |
| 05 | All cleaning functionality retained, with method-sized examples and complete text-normalization/cleaning workflows in demos. |
| 06 | Accepted compaction retained; long error/repair scripts replaced by short method demonstrations and existing full-demo links. |
| 07 | Only repetition removed. Styling, libraries, panel scales, plotting methods, KDE bandwidth, accessibility, and Altair functionality remain in main. |
| 08 | SSH, keys, scp, tmux, tunnels, and Jupyter restored in main; full terminal/performance/filter workflows remain in demos. Short SSH examples distinguish laptop and server prompts. |
| 09 | Instructor-approved beginner core: read dated data; summarize with grids/resampling/windows; interpret clocks and use honest entity history. Frequency inference, specialized schedules, clock-time alternatives, percentage changes, and Grouper move to BONUS. Main retains gap runs, UTC/DST, reporting availability, and chronological splitting. Demo 1's resampling examples move to Demo 2; its lag/lead/alert workflow moves to Demo 3. |
| 10 | XGBoost/Keras API teaching restored; the core Demo 3 route fits a first XGBoost and Keras model. Extra fits/comparisons remain independent practice. |
| 11 | Main and all four required demo pairs restored exactly from received pre-pacing snapshots, including earlier correctness fixes. Full project workflow restored. |

The revisions are freshly verified. Existing real 01–03 entrypoints pass, including core/full routes and final optional prerequisite links. All eleven changed required notebooks pass fresh full execution; all seven changed compact routes pass. Forty changed 04–07 lecture examples, nineteen current 09 examples and five moved BONUS examples, and restored early/model snippets execute with meaningful output checks. Lecture 04's newly explicit `pyarrow==25.0.0` declaration/lock and actual notebook installer pass in a fresh uv project, preserving all other package versions. The independent referee ran final full 04.3, 07.2, and 09.2 plus the exact 09.1/09.2/09.3 and 10.3 cores, and verified Lecture 11 against all received snapshot files. SSH execution-context confusion, stale optional NumPy prerequisites, and an untaught Parquet-check method are fixed. No unresolved confirmed lecture/demo defect remains in the reviewed revisions.

Evidence: [01–03](../../scratch/course0430/revision0103/report.md), [04–07](../../scratch/course0430/revision0407/report.md), [08/10/11](../../scratch/course0430/revision081011/report.md), [09](../../scratch/course0430/revision09/report.md), [independent challenge](../../scratch/course0430/revision-referee/report.md), and [shared checks](../../scratch/course0430/revision-shared/). Current metrics, 22-page lint, fresh generation of all 35 notebook pairs after the final whitespace correction, openings/breaks/assets, all 32 humor blocks, and 54 unique comic IDs pass. All 273 recorded assignment/grader inputs are unchanged; only the authorized 04 demo project/lock differs among 306 protected inputs. The fresh Eleventy build at `scratch/course0430/site-revision-final/` passes, all 26 primary pages have zero broken internal links/assets, and the real site-link contract and final whitespace check pass. The earlier full-course grading evidence remains applicable to the unchanged grader/artifact contracts; this revision does not claim a new complete grading-suite run. Actual classroom timing and the previously recorded hosted-platform/server/human checks remain unverified; nothing was committed or published.

- **01–03:** removed pasteable shell comments and labeled annotated reference listings as reading-only; supplied missing chapter lists and later-demo terminal re-entry; added the macOS `code` PATH setup action; corrected shell shortcut behavior; removed repeated shortcut material. Qualified NumPy sorting/vectorization claims and explained standard deviation/percentile interpolation.
- **04–07:** corrected raw numeric auditing to reject fractions where the contract requires whole values; distinguished source-row grain from histogram/boxplot mark grain; corrected Unicode digit regex semantics and the modified-z distribution caveat.
- **08–11:** distinguished repeating calendar rules from equal elapsed spacing and Grouper from resample empty bins; fixed benchmark units; narrowed observational/causal and diabetes-score claims; corrected stale OLS coefficient/effect prose, estimator scoring, forest probability aggregation, and elapsed-hour/DST wording. Made complete event capture and immediate reporting explicit assumptions. Optional archive recovery now preserves existing work.
- **Course authoring README:** aligned uv project setup, Python 3.9-compatible artifact checks, exam exclusions, and the existing single-command Notion workflow. Publishing instructions remain conditional on the instructor's request and live reconciliation.

## Grading counterexamples and repairs

The graders read student-created artifacts only. Source instructions remain learning guidance, and submitted Python/notebooks are never executed or pattern-matched to decide automated points. Helpful failures name the affected artifact, observed versus expected evidence, and a correction. Equivalent formats and coherent downstream work retain credit.

| Defect | Current behavior demonstrated by real CLI results |
| --- | --- |
| 01–03 `zip(strict=True)` crashed on Python 3.9 | Explicit length invariants and ordinary zip; actual 3.13/3.9 reports agree. |
| 01 rejected case/numeric equivalents and cascaded total/classification mistakes | Equivalent reports earn full points; coherent wrong total→mean or classification→review count costs only the original 5-point error. |
| 02 charged an omitted report line twice or reused cutoff-range rejection downstream | Missing usable-count line: 92; coherent out-of-range cutoff/list: 95. |
| 03 wrong patient/hour choice also lost its correctly calculated mean | Each coherent choice/mean pair: 96, losing only the selection's 4 points. |
| 04 missing total column charged twice; empty/unrecognized fridge evidence received values credit | Genuine missing total: 98. Header-only/unknown fridge rows receive no reading-value credit. |
| 06/09 missing column hid independently wrong present sibling values | 06 omission: 92; omission plus wrong patient: 84. 09 missing collection time: 98; also wrong result time: 94. |
| 08/10 omitted columns charged both schema and values | 08 count-column omission: 94; also wrong present count: 90. 10 lower-bound omission: 98; also wrong upper bound: 95. Empty/unknown tables still earn no unsupported value credit. |
| 11 pandas duplicate-header mangling and NUL truncation concealed contradictions | Conflicting random-state copies: 74/75; unreadable NUL model metadata: 71/75. Equivalent numeric/date/Boolean duplicates retain credit. |

Root TA integration expectations were updated to match the charge-once rule: its running-total fixture loses the wrong classification's 5 points, while its correctly derived review count keeps credit. The fixture itself was preserved, and the real batch grading path was rerun.

All nine homework handouts carry byte-identical copies of their complete course-owned checker sets. Exams 05 and 11 ship no checker/workflow and retain 75 automated plus 25 human points. Fresh correct artifacts earn 100 for homework and 75 for exams. Assignment 01's full-score QA completion uses an existing trusted roster fixture; it is not verification of a new student's live identity.

## Fresh verification

- **Required demonstrations:** current 01–03 script/REPL/shell workflows executed, including actual Bash/zsh input, intentional errors/corrections, Git state, seeded NumPy outputs, and fresh uv recreation. All **26** 04–11 notebooks ran alone in fresh notebook-only folders; changed final sources were regenerated and rerun. Meaningful tables, figures, model outputs, and saved artifacts were inspected.
- **Notebook/setup integrity:** **35** Markdown/notebook pairs match fresh CLI conversion; **eight** local setup scripts fetch matching current bytes and preserve existing work; **eight** fresh demo projects run actual uv setup and current IPython `%pip` cells. Final installer cells/declarations match verified inputs. **17** supplied project locks pass `uv lock --check --offline`; all **16** supplied 04–11 interpreter pins are 3.13. Demo/Assignment 03 creates its pin as taught.
- **Grading:** 01–06 has **78** fresh cross-version CLI cases; the independent referee has **82** focused cases plus **16** final Boolean-boundary cases. Later grading alternatives, malformed/missing evidence, propagated versus independent errors, and PNG/model metadata checks also run on actual Python 3.13.14 and 3.9.25. Full updated self-tests 01–10 pass. The broad 11 reader repair's full suite recorded exit zero in 538.219 seconds; the subsequent four-line Boolean equivalence refinement passed current focused regressions, independent challenge, and full-credit/conflict/NUL CLI runs on both interpreters. Earlier suite output is not presented as a full-suite run of that last refinement.
- **Integration:** the current trusted entry point verifies empty/scaffold zero scores, totals, exam exclusions, checker identity, and poisoned submitted source. All **nine** actual homework workflows were replayed locally: complete download, download failure/fallback, incoherent-set recovery, and unusable fallback stopping the job (**36** cases). Real pytest rejects untouched scaffolds and passes all nine correct completions. The TA grading integration exercises local fork updates, preserved TA edits, source traps/symlinks, checker failures, interrupted runs, and all eleven untouched handouts.
- **Rendering:** lecture lint checks 22 pages with zero problems; all openings, local lecture/BONUS image paths, break counts, and **54** used comic IDs pass structural checks with no comic shared across lectures. The fresh Eleventy build passes, and all **26** primary course pages have zero broken internal links/assets. Final output is `scratch/course0430/site-final/`; build/link evidence is in the shared directory. `git diff --check` passes.

After the comic-placement follow-up, lint and structural checks passed again. A new build at `scratch/course0430/site-followup/` passes all 26 primary-page link/asset checks; the rendered Lecture 01 places the comic between Python installation and editor options. No executable demo, assignment, or grader source changed in this follow-up.

Notebook execution uses the pinned shared environment read-only, with installer cells replaced only in scratch copies; the actual cells were separately exercised in fresh environments. Unpublished download prefixes use current local `file://` inputs in isolated trials. No current-file result is inferred from a stale generated notebook or solely from process exit.

## Evidence and limits

Detailed maps, source changes, commands, outputs, and counterexamples:

- [Lecture 01](../../scratch/course0430/content01/report.md)
- [Lectures 02–03](../../scratch/course0430/content0203/report.md)
- [Lectures 04–07](../../scratch/course0430/content0407/report.md)
- [Lectures 08–11](../../scratch/course0430/content0811/report.md)
- [Assignments 01–06](../../scratch/course0430/grading0106/report.md)
- [Assignments 07–11](../../scratch/course0430/grading0711/report.md)
- [Independent challenge](../../scratch/course0430/referee/report.md)
- [Shared integration evidence](../../scratch/course0430/shared/)

This is a local Linux review. Hosted Colab, native macOS/Windows UI, live Actions, actual course-server SSH/scp/tunnels, and the exams' human judgments remain unverified. The 08 terminal exercise explicitly rehearses locally; actual remote validation requires the course server/account configuration. New download URLs await publication to `main`. These external checks are not claimed complete, and no deployment/publication was requested.
