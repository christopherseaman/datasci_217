# Lecture Review Progress Tracking

This is the durable checkpoint for the `2026-refresh` review. Update it after every substantive review or editing step and before every commit.

## Branch-only status

- Active branch: `2026-refresh`
- Current phase: assignment/grading design; the lecture content and instructor-directed xkcd/humor repairs pass their recorded lecture-only reviews, course-wide heading normalization remains open, and the demo audit is recorded as complete for its authorized scope
- Explicitly out of scope for this phase: merging to `main`; assignment source content remains unchanged while the portable repository/pytest/Actions shape is aligned
- Branch-only working records: `AGENTS.md`, `HANDOFF.md`, and this file
- Before the eventual merge to `main`: remove or explicitly exclude all three branch-only working records

## Review objective

Review all student-facing lecture content against `main`, the relevant Wes McKinney chapter extracts under `work/mckinney_content`, current package behavior, and the intended course sequence. Preserve inherited topic scope unless a change is explicitly justified. Correct factual/API defects and unintended LLM changes without silently redesigning a lecture.

## Demo audit scope and contract

The instructor authorized a separate demo-phase review after the lecture-only content checkpoint. The demo scope is every lecture's student-facing demo Markdown and its generated notebook/notebook source, with Lecture 11 treated as one capstone demo rather than three smaller demos. Assignments remain out of scope. Each demo should match the lecture material available at its position, usually landing near the first third, middle, and end of the lecture, and should introduce no concept before the lecture has taught it. Demo Markdown should generate a notebook that executes without errors when its documented packages and versions are available. The audit therefore checks generation fidelity, notebook execution, demo ordering and coverage, pandas 3/API compatibility, package-version alignment between lecture, demo, and local environment metadata, and stale or missing dependencies. The lecture/demo boundary is pedagogical: lecture code explains concepts, while demo notebooks own execution setup and state.

## Demo audit checkpoint (read-only)

The first demo audit covered all student-facing demo paths for Lectures 01–11 and excluded assignments. It checked the Markdown source, any paired notebook, the demo guide, local package metadata, and the lecture section named as the demo's owner. Four independent group reviews returned `NEEDS REPAIR`.

| Demo group | Execution/structure result | Main findings |
| --- | --- | --- |
| 01–03 | Scripts partly execute; no notebook source exists | Lecture 02 has a broken module import, a backup-directory bug, unsafe reset/force-push instructions, and guide/file-name drift. Lecture 03's analysis scripts look for a missing `data/students.csv`, mislabel non-Math subjects, and use a guide order that precedes the lecture's environment → NumPy sequence. Lecture 01's integration script imports `json`, `pathlib`, and `datetime` before those concepts are taught. Lecture 03's NumPy requirement is unbounded `numpy>=1.24.0` despite the lecture's tested NumPy 2.0.2/Python 3.12.13 contract. |
| 04–06 | Eight of nine notebooks execute under a temporary pandas 3.0.3 stack; one fails | Paired Markdown and notebooks are materially divergent in Lecture 04 and in Lectures 05–06. Lecture 05 Demo 2 fails on tied `qcut` edges; Lecture 05 has no demo requirements file. Lecture 04 and 06 requirements still permit pandas 2, Lecture 04 notebooks install packages during execution, Lecture 04 Demo 3 lacks its repository fixture, Lecture 05 Demo 1 previews a heatmap before visualization is taught, and Lecture 06 notebooks 1–2 carry stale Python 3.11 metadata. |
| 07–09 | Markdown-generated 08 notebooks execute; stored 07/09 notebooks fail on current APIs | Matplotlib `boxplot(labels=...)` fails and must use `tick_labels=`. Stored Lecture 08 notebooks retain stale uppercase `freq="H"` and diverge from Markdown. Lecture 09 Demo 1 teaches time zones before the lecture's time-zone section. Lecture 07's Altair selection APIs are deprecated. Requirements use broad pre-pandas-3 lower bounds, and Lecture 09's guide names optional Altair without declaring it. |
| 10–11 | Lecture 10 execution remains unverified on the available host; Lecture 11's capstone structure is strong | Lecture 10 Demo 2 uses the test set for XGBoost early stopping and reports it as final performance; Demo 3 repeatedly uses the test set for architecture/model comparison. Demo 1 labels mean-prediction confidence intervals as prediction intervals. The guide's data-size claims and Python 3.13 requirement conflict with actual datasets and the repository's Python 3.12 pin; `load_wine` is mislabeled as Wine Quality. Lecture 11's one-capstone-demo design and four required notebooks match the lecture, but its local `uv sync --no-project ... -r requirements.txt` command is invalid and its Colab links remain development URLs pending immutable-tag publication. |

The audit's cross-cutting recommendation is to make Markdown the authoritative source for every paired demo because the instructor's execution contract is “Markdown generates the notebook.” Where an existing notebook contains richer content than its Markdown (notably Lecture 04 and Lecture 06), reconcile that content into the Markdown source before regenerating the notebook; do not silently discard executable coverage. The repair batch will then regenerate paired notebooks and compare generated cells to their Markdown source.

The version baseline for the repair is the already tested course candidate: Python 3.12.13, NumPy 2.0.2, pandas 3.0.3, Matplotlib 3.11.1, scikit-learn 1.9.0, PyArrow 25.0.0, JupyterLab 4.4.10, and Jupytext 1.18.1 where those packages are used. Activity-specific libraries will be pinned only after their demo code is repaired and exercised; broad lower bounds that permit pandas 2 or pre-current plotting APIs are not acceptable. Lecture text will be updated only when its documented package/version contract is stale relative to the demo's tested contract. The first repair scope covers the P1 execution/safety failures, all source/notebook drift, pandas 3/API/version mismatches, and the directly misleading placement/guide claims; optional Colab publication remains a release operation rather than a local execution prerequisite.

No demo or lecture source was edited during this read-only audit; only this branch ledger records the findings. The next step is the bounded repair batch, followed by fresh Markdown-to-notebook generation, temporary-environment execution, lecture/demo alignment checks, and an independent adversarial demo review.

## Fresh core-lecture comparison against main

The instructor clarified that this pass compares the current lecture pairs, not every lecture-related Markdown page. The exact scope is the 11 pairwise comparisons from `main:01/README.md` versus current `01/README.md` through `main:11/README.md` versus current `11/README.md`—22 document snapshots representing 11 lectures. BONUS, POINTS, demos, assignments, and branch-only records are excluded from the comparison. The baseline is current commit `299cdbb` against `main` commit `7b49984`; `main` is an ancestor of the current branch.

The review distinguished intentional pandas 3/API modernization, prerequisite reordering, and approved content consolidation from accidental topic loss or unjustified expansion. It retained the established non-executable lecture boundary and treated the known one-H1 hierarchy gap as a separate format result. Four pair-range reviewers, a cross-course sequence reviewer, a quantitative reviewer, and an independent synthesis challenger all used this same scope.

### Core-pair verdict

**The Lecture 01–11 sequence and inherited core-topic coverage pass.** The initial comparison found one required preservation repair: Lecture 10 retained every required model family and was substantively more accurate, but it dropped two inherited Zoolander references despite the instructor's explicit preserve-all requirement. Both references are now restored without reviving the old deterministic model advice, and an independent challenger passed the repair. All 11 core pairs therefore pass their content comparison, including Lecture 05 with a minor discoverability discussion. The separate heading-format gate remains open.

| Lecture | Completeness versus `main` | Order and prerequisites | Size/density | Fun/engagement | Pair result |
| --- | --- | --- | --- | --- | --- |
| 01 | Full foundation retained and clarified | Strong terminal/script-first opening | Balanced | Retained | Approve |
| 02 | Full structured-Python/Git role retained; premature pandas use removed | Strong bridge from Lecture 01 | Substantially consolidated | Retained | Approve |
| 03 | Environment and NumPy contract expanded | Strong script/Git-to-reproducibility bridge | Denser, but justified | Retained | Approve |
| 04 | Notebook and pandas foundation modernized for pandas 3 | Strong; plotting remains deferred | Balanced | Slightly less immediate visual payoff | Approve |
| 05 | Cleaning core strengthened; peripheral core blocks removed | Excellent contract-to-save progression | More focused | Retained | Approve, with P3 notice option |
| 06 | Joins, concat, reshape, and index topics retained | Excellent cardinality-first progression | Balanced | Broken imported visuals removed; concept-serving humor retained | Approve |
| 07 | Visualization contract retained and improved | Strong matplotlib → pandas → seaborn sequence | Advanced survey consolidated | Retained | Approve |
| 08 | Aggregation and pivot scope retained with pandas 3 result-shape semantics | Strong | Balanced | Retained | Approve |
| 09 | Temporal core retained; specialized decomposition explicitly made optional | Strong | Balanced | Retained | Approve |
| 10 | All inherited model families and evaluation workflow retained | Strong statistics → ML → boosting → deep-learning order | Largest core lecture, but segmented and purposeful | Strong; all inherited Zoolander references restored | Approve after repair |
| 11 | Inherited integrative topic families retained in a question-led rewrite | Improved: question/evidence precede transformation and modeling | Expanded, appropriate for synthesis | Lighter than `main`, but still uses relevant visuals | Approve |

### Actionable comparison findings

1. **Resolved P1 preservation violation — two Lecture 10 Zoolander references are restored without restoring the deterministic model advice around them.** `main:10/README.md:462` used the “blue steel” reference in its Random Forest tip; current `10/README.md:484` now makes Blue Steel explicitly a style rather than a model-selection rule while retaining baseline-and-validation guidance. `main:10/README.md:637` compared XGBoost, LightGBM, and CatBoost to “blue steel, magnum, and le tigre”; current `10/README.md:667` restores that comparison while retaining dataset-specific benchmarking. Independent review confirmed that the two-line diff restores the main-carried references, leaves “start with Random Forest/XGBoost” prescriptions absent, and introduces no new factual or pedagogical issue.
2. **P3 discoverability option — Lecture 05 no longer signals two inherited peripheral sections.** `main:05/README.md:513-566` covered random sampling/permutation and `main:05/README.md:756-809` covered command-line notebook execution. The current core lecture appropriately focuses on cleaning and contains no replacement core signpost. This is not a cleaning-topic or sequence failure; if inherited-topic discoverability matters, add a brief optional pointer rather than restoring the two large blocks.
3. **Separate format gate — nine core READMEs still have multiple H1 headings.** Lectures 03 and 11 have exactly one H1; Lectures 01, 02, and 04–10 do not. This is an existing hierarchy/accessibility issue rather than content loss relative to `main`.

### Intentional differences that are not regressions

- Lecture 02's generic reporting example avoids using pandas before Lecture 04; the old example depended on an undeclared pandas setup and a term-like raw-data link. The prerequisite correction outweighs its small loss of data-science flavor.
- Lecture 06 removes 12 `attachment:` image references from `main`; those were non-repository Notion URIs. The current prose, tables, ASCII join diagrams, and retained humor preserve the instructional work, so the raw image-count drop is not a content defect.
- Lecture 07 keeps the advanced visualization ecosystem as a concise optional survey rather than several full executable walkthroughs. Lecture 09 explicitly makes decomposition/forecasting optional. Both are justified ownership and density changes.
- Lecture 11's old phase/checklist structure was intentionally replaced by a question-led integration path. Its crosswalk preserves cleaning, joins/reshape, aggregation, visualization, time series, modeling/evaluation, and communication without imposing a universal lifecycle.
- A prose-oriented metric increased for every current core README. The largest gains are Lectures 11, 10, and 03; large line-count reductions in Lectures 02 and 07 principally reflect code/reference consolidation rather than disappearing core concepts. These counts support the pair review but do not by themselves establish pedagogical quality.

### Focused Lecture 05 and Lecture 11 follow-up

The removed Lecture 05 sampling block taught `DataFrame.sample()` by count or fraction, weighted/replacement sampling, shuffling/permutation, a `groupby(...).apply(...)` stratified example, and bootstrap resampling. It was not required by the cleaning progression, and parts of its framing were misleading: a random sample is not automatically representative, shuffling does not safely “break temporal dependencies,” and the stratified example needs pandas-3-specific care. The removed notebook-automation block taught `jupyter nbconvert --execute`, output versus `--inplace`, shell exit handling, sequential notebook execution, and `--allow-errors`. This is operational tooling rather than cleaning content; `--inplace` is destructive and `--allow-errors` can mask a failed pipeline without careful framing. The current recommendation is to leave both large blocks out of core. Add a short pointer only if a later assessed activity actually expects either skill.

The focused Lecture 11 review confirms that the current question-led capstone is a substantial improvement over `main`'s universal nine-phase workflow and technique checklist. It preserves the inherited subject families while adding better intellectual owners for grain, provenance, evidence strength, prediction-time boundaries, baseline comparison, and limitations. No P0/P1 correctness or prerequisite defect was found. The remaining instructor choices are:

1. Make the unresolved parts of “how well” and “selected taxi zones” explicit. The preferred treatment is a short in-class decision prompt asking students to choose an evaluation measure and zone-selection rule, rather than silently supplying them or leaving the question apparently complete.
2. Teach three anchor decisions in the lecture half—question/claim, grain/provenance, and prediction contract—and treat the remaining subsections as synthesis/reference. This keeps the roughly 50/50 lecture/demo plan feasible without deleting the conceptual breadth.
3. Define the release manifest and one history-feature example locally. A parenthetical description of the manifest as expected filenames plus identities/checksums, and one concrete distinction among recent lag, weekly lag, and rolling history, would remove the only meaningful abstraction jumps.
4. Preserve the opening `fifteen_years_2x.png`. The instructor confirmed that it intentionally anchors the explanation for including so many xkcd comics throughout the course; the visual is therefore not a content or engagement finding. If that purpose needs to survive instructor handoff, an optional short presenter cue could make the connection explicit, but the planned spoken explanation is sufficient for the current lecture.
5. Optionally compress the six-line McKinney Chapters 12–13 recap. The stable conceptual attribution is reasonable, but the lecture stands on its own and does not need edition-specific chapter narration unless those chapters are assigned reading.

No Lecture 11 edit is authorized by this review checkpoint; these choices are recorded for discussion with the instructor.

Focused validation passed: `git diff --check` reported no whitespace errors; an independent read-only reviewer confirmed all main-carried Zoolander reference lines are represented and the old deterministic Random Forest/XGBoost prescriptions remain absent; and Eleventy wrote 30 pages and copied 124 assets. The only lecture source change is the two-line Lecture 10 restoration. No demo or assignment path changed.

### Sampling, notebook-automation, Lecture 11, and xkcd follow-up

The instructor chose to retain sampling in the course and identified notebook automation as Lecture 04 material. The lecture-only implementation now assigns the concepts as follows:

- Lecture 05 core teaches a reproducible row sample as a manual inspection aid. It explicitly says that randomness does not guarantee representativeness, whole-table validation still owns correctness, and ordered/time-series rows must not be shuffled casually.
- Lecture 05 bonus retains the design-heavy extensions without duplicating the core example: group-wise `GroupBy.sample`, weighted and systematic selection caveats, and replacement/bootstrap resampling. Bootstrap is distinguished from an ordinary sample, permutation, and train/test split.
- Lecture 04 core links `Restart & Run All` to an optional non-interactive workflow. Lecture 04 bonus now owns safe `jupyter nbconvert --execute` guidance, separate output files, stop-on-failure shell behavior, and explicit warnings about `--inplace`, `--allow-errors`, kernels, packages, paths, and working directories. The full unrelated section was removed from Lecture 05 bonus.
- Lecture 11 now pauses for students to choose the evaluation measure, selected-zone rule, and meaningful baseline; locally defines the release manifest, recent lag, weekly lag, and rolling history; and names question/claim, grain/provenance, and prediction contract as the three lecture anchors. Both xkcd images remain, including the intentionally personal/meta `fifteen_years_2x.png` opener.

Runtime validation used an isolated Python 3.12 environment with pandas 3.0.5, nbconvert 7.16.6, nbformat 5.10.4, and a temporary registered IPython kernel. The Lecture 05 sampling and group/bootstrap examples passed. The documented nbconvert command wrote the named separate notebook without changing its source, and a notebook with a failing cell returned nonzero. An initial trial without a registered kernel failed exactly at the kernel boundary, supporting the lecture's statement that kernel and package selection are part of the execution contract. An independent read-only challenge then passed all five changed lecture pages with no P0–P3 finding and confirmed that only redundant API/CLI variants, not concepts, were removed.

The final scoped gate found balanced fences and resolving local links in all six changed Markdown files (the five lecture pages plus this ledger), `git diff --check` passed, and Eleventy wrote 30 pages and copied 124 assets. Changed lecture paths are limited to Lectures 04, 05, and 11; no demo or assignment path changed.

The xkcd inventory covered all lecture Markdown under Lectures 01–11 while excluding demos and assignments. It found 25 embedded comic references representing 22 intended distinct comics after resolving repeated content. Three embedded duplicate groups need instructor disposition:

- xkcd 1513, “Code Quality,” is byte-identical in `01/README.md` and `11/README.md`. It fits Lecture 11 as a closing course-culture callback; the Lecture 01 copy is comparatively unmotivated.
- xkcd 1296, “Git Commit,” appears in both `02/README.md` and `02/bonus/advanced_git.md` through the same file. Keep the core placement; the bonus repetition adds no new work and bonus pages may remain dry.
- xkcd 1838, “Machine Learning,” appears twice in `10/README.md` at standard and 2× resolution. The second occurrence is incorrectly labeled xkcd 2400; xkcd 2400 is actually “Statistics.” Keep the opening occurrence and remove or replace the mislabeled repeat.

The same audit found two correctness issues rather than duplicates: `08/README.md` labels “Slope Hypothesis Testing” as xkcd 2582, but the comic is xkcd 2533 and xkcd 2582 is “Data Trap”; and `05/media/xkcd_2239.png` is a saved HTML 404 response rather than the intended xkcd 2239 “Data Error” image. Those factual/broken-asset repairs are recommended regardless of the duplicate-placement choice. Two commented Lecture 04 FIXME candidates and repeated URLs in noncanonical `08/addme*.md` notes are not embedded comic repetitions.

Lecture 11 is visually warm and strongly interactive but contains almost no written joke beyond vivid phrasing. That is acceptable for the capstone. If one more concept-serving line is desired, the preferred addition after “Those are choices, not mandatory project stages” is: “A capstone is not a technique scavenger hunt; the data gets a vote.” No humor or duplicate-comic edit is included in this checkpoint without instructor disposition.

The instructor then resolved every open comic disposition. Lecture 01 now bridges into conditionals with unique xkcd 1654, “Universal Install Script,” while xkcd 1513 remains only as Lecture 11's closing callback. The Lecture 02 bonus repetition of xkcd 1296 is removed while its core placement remains. Lecture 10 keeps xkcd 1838 at its opening and replaces the mislabeled repeated image with the actual xkcd 2400, “Statistics,” without changing any Zoolander reference. Lecture 08 correctly identifies “Slope Hypothesis Testing” as xkcd 2533. The invalid 146-byte HTML file at `05/media/xkcd_2239.png` is replaced by the official 740×282 xkcd 2239 “Data Error” PNG. Lecture 11 now embeds xkcd 2582, “Data Trap,” beside its warning against artifact-generating exploration and adds the instructor's tighter Pokémon/capstone line after the non-prescriptive transformation guidance. Official xkcd pages and image URLs confirmed all five titles, numbers, and source images.

Integrated validation now passes. A lecture-only gate checked 28 core, bonus, and points Markdown pages, found 414 balanced fenced blocks, resolved 85 local links/images, and confirmed exactly one embedded occurrence each of xkcd 1296, 1513, 1654, 1838, 2239, 2400, 2533, and 2582. The requested remote image URLs returned successfully. The repaired xkcd 2239 asset has a valid PNG signature, is 46,171 bytes at 740×282 pixels, renders as “Data Error,” and is copied byte-for-byte into the built site. `git diff --check` passed. Eleventy wrote 30 pages and copied 124 assets; the rendered pages contain each new label/image and the Pokémon line. The changed scope remains seven lecture files/assets plus this ledger, with no demo or assignment path changed. A fresh independent adversarial review passed all six instructor requirements with no P0–P3 finding, independently confirmed the official comic metadata and xkcd 2239 byte identity, found no unintended embedded-comic duplicate, and verified that every Zoolander reference plus Lecture 11's intentional fifteen-years opener remains intact.

## Recovered work state

The task transcript lost a substantial portion of the conversation twice, but the workspace retained proposed edits to `01/README.md` through `11/README.md`. No demo or assignment files were modified by this lecture pass. The recovered edits remain reviewable in Git.

### Decisions already established

- Lecture 03: reconcile environment, interpreter, version, package, and dependency guidance; pedagogical alternatives may vary when clearly labeled.
- Lecture 05: reconcile the cleaning workflow, but keep it approachable and proportionate to the lecture.
- Lecture 10: preserve the inherited model survey and topic scope; make only correctness and evaluation-workflow repairs.
- Lecture 11: use an approachable, question-led worked capstone; avoid a universal prescriptive lifecycle; divide time roughly evenly between lecture reasoning and a live workflow demonstration.
- The later assignment should be a somewhat guided mini research project using a different dataset from the Lecture 11 demo.
- Preserve the overall Lecture 01–11 order. Rebalance Lectures 01–03 around terminal/script fluency, structured/versioned programs, and reproducible NumPy work; introduce notebooks and pandas in Lecture 04 without requiring later concepts early.
- Treat pandas 3.x as the lecture API contract. Exact tested pins remain activity-specific and will be reconciled when demos and assignments enter scope.
- Lectures explain concepts and are not executable activities. Keep code examples correct and pedagogically ordered, but do not add procedural environment or activation steps merely to make an entire lecture runnable from top to bottom; demos and assignments own execution workflows.

## Pandas 3 organization follow-up

The user rejected the Lecture 03 reactivation detour as inconsistent with the lecture/demo boundary and asked whether pandas content can be simplified or better organized around pandas 3. The active follow-up will remove that procedural block, audit the pandas concept progression across lecture materials, compare it with the pandas 3 contract, and make only lecture-scoped organization or content repairs.

### Implementation checkpoint

The follow-up audit used current official pandas 3.0.5 documentation as the API authority and kept inherited topic scope while tightening ownership between lectures:

- Lecture 03 no longer adds an activation procedure solely to make later lecture snippets runnable; its environment material remains conceptual course content.
- Lecture 02 bonus material no longer introduces pandas before Lecture 04, and its duplicated lambda section plus malformed fences were repaired when the validation scope expanded to every built lecture page.
- Lecture 04 now introduces pandas 3's inferred `str` dtype and Copy-on-Write mental model before the core structures, corrects the visible dtype example, teaches direct owner assignment instead of chained assignment, and reduces GroupBy to a self-contained preview of Lecture 08.
- Lecture 05 distinguishes `Series.map`, `DataFrame.map`, and axis-oriented `apply`; keeps the contract-to-save cleaning workflow in core; and moves sampling/permutation plus notebook CLI automation to optional bonus reference.
- Lecture 06 makes label alignment a prerequisite for index-based combination, separates concat's `join=` alignment mode from relational joins, narrows `pivot_table` to an aggregation preview of Lecture 08, and removes repeated bonus coverage.
- Lecture 07 keeps the matplotlib → pandas → seaborn sequence while compressing the duplicated modern-library walkthrough to an optional ecosystem survey; executable extensions remain in bonus material.
- Lecture 08 organizes GroupBy by result shape, teaches pandas 3's `observed=True` and `include_groups=False` contracts without pandas 2 migration history, replaces view/`inplace` advice, and makes dtype/performance choices explicit rather than automatic.
- Lecture 09 separates timestamp offsets (`QE`, `YE`, `h`) from Period frequencies, distinguishes `asfreq()` from aggregation, and keeps decomposition/forecasting and other specialized topics explicitly optional.

No demo or assignment file was inspected or edited. The expanded structural gate, site build, and pandas 3 execution suite passed. Independent adversarial review passed the full change set and the final Lecture 02 bonus repair with no P0–P2 finding.

## Full adversarial lecture audit

The user requested a fresh review of all lecture content for completeness, prerequisite order, course flow, size, and sufficient fun/engagement. Eight independent read-only workstreams reviewed the 28 canonical lecture Markdown pages. Every workstream was explicitly constrained by the established boundary: lectures explain and illustrate topics and are not activities meant to execute top-to-bottom. Missing activation, installation, local imports, or shared runtime state was therefore not treated as a defect; individual claims and examples still had to be coherent and factual. All demo and assignment paths remained excluded. The user also confirmed that term-specific assignment and midterm URL placeholders are intentional repository slots, not findings.

### Verdict

**Changes required before lecture approval; no P0 finding.** The canonical sequence and inherited topic coverage pass. The main path is coherent:

`shell + scripts` → `Git + structured Python` → `environment concepts + NumPy` → `notebooks + pandas` → `cleaning` → `join/reshape` → `visualization` → `aggregation` → `time series` → `modeling` → `question-led capstone`

Comparison with `main` found no material accidental loss of inherited topic families. Most remaining correctness and safety debt is concentrated in bonus pages and the parallel `POINTS.md` copies, although Lecture 10 has one core completeness gap and several core wording issues.

### P1 — repair before approval

1. **Metric selection has no conceptual owner.** `10/README.md:272-307` relies on estimator-default `model.score()` and never explains how to choose among regression or classification measures, while `11/README.md:34-35` expects students to choose an evaluation measure. Lecture 10 already covers baselines, validation/CV selection, and one-time test reporting at `:478-508`; add a compact task/cost-aligned metrics bridge to that existing workflow without expanding into a full metrics course.
2. **Several statistical examples teach unsupported conclusions.** Remove the `p>0.05` joke at `06/README.md:785`; repair the generic normal-approximation interval at `07/BONUS.md:539-579`; and correct `09/BONUS.md:127-167`, where `seasonal=365` configures the STL smoother rather than annual periodicity and the ADF helper turns failure to reject a unit-root null into proof of nonstationarity. Lecture 10 should remain the inference vocabulary owner.
3. **Advanced Git and remote-notebook guidance crosses safety boundaries.** `02/bonus/advanced_git.md:111-115,395-425` presents commands that discard work or rewrite and force-push history without a verified disposable target or recovery workflow; it also teaches `git filter-branch`, which Git itself recommends against. `08/POINTS.md:121-129` binds Jupyter to `0.0.0.0` even though the canonical README correctly uses loopback plus SSH forwarding. Remove the destructive recipes or confine them to an explicitly disposable/recoverable setting, and synchronize the remote guidance to loopback.
4. **Some pandas examples are invalid under the declared pandas 3 contract.** `04/BONUS.md:13-45` uses unsupported DataFrame–Series arithmetic with `fill_value`; `08/BONUS.md:13-26` asks `GroupBy.agg` to return a multi-value Series; and the weighted pivot examples at `08/BONUS.md:97-106,378-389` try to access a weight column that `pivot_table(values='value')` does not pass to the aggregator. Replace them with explicit alignment, named/list aggregations, and a grouped weighted-total workflow.
5. **The advanced validation example records false results as passes.** `08/bonus/advanced_debugging.md:210-228` unconditionally records `passed: True` whenever a Boolean-returning validator does not raise, even though the supplied validators at `:249-268` return `False` for failure. Honor `bool(result)` or require and document an exception-based validator contract.
6. **Two Lecture 10 bonus examples are not viable current examples.** The custom attention layer at `10/BONUS.md:223-252` is dimensionally invalid for ordinary sequence inputs and is not an attention computation; the Featuretools example at `10/BONUS.md:455-490` uses removed entity APIs. Replace these with current Keras attention and Featuretools EntitySet/DFS APIs or label non-runnable material explicitly as pseudocode.

### P2 — important content and organization repairs

- `01/bonus/advanced_topics.md:85-97` mixes Python statements into a Bash fence. `02/bonus/bonus_python_concepts.md:162-190` promises a `mode` method it does not implement, and its later object-model table overstates identity-as-address and unconditional tuple hashability. `03/BONUS.md:121-158` uses deprecated `np.in1d` and demonstrates `argsort` only after sorting the source in place.
- `05/README.md:51-69` uses a Matplotlib missingness plot before the visualization lecture and names missingness mechanisms without a plain-language bridge. Treat the chart as an explicit Lecture 07 preview or keep this core example tabular. `05/BONUS.md:351-389` should not present `nbconvert --inplace` or `--allow-errors` without overwrite/failure warnings.
- `06/README.md:30-82` should teach pandas' null-key matching difference from SQL. `06/BONUS.md:222,472` overstates index preservation and carries a stale `Int64Index` representation; `06/POINTS.md:34,122-150` uses a false row-growth heuristic and overstates wide/long rules.
- `07/README.md:15-22` has an outline that does not match section order. `07/BONUS.md:336-349,675-688` labels a raw connecting line as a trend and stores the `ax.grid` method rather than grid state. `07/POINTS.md:26-33` loses the canonical README's nuance about zero baselines for bars versus contextual line-chart axes.
- `09/BONUS.md:265-271` silently assumes input columns named `value`, `volume`, and `count`; document that schema, or use `.size()`/named aggregation if `count` is intended to mean the number of observations. `09/README.md:674,677` contains two malformed image placeholders even though the intended image files exist.
- `10/README.md:257` incorrectly assigns `fit`, `predict`, and `transform` to every model; `:543-561` describes ordinary residual fitting without qualifying squared-error regression or introducing pseudo-residuals; and `:594-603` presents data-dependent XGBoost ranges as universal “sweet spots.” `10/BONUS.md:390-415` needs the untrusted-pickle boundary and current Keras save/export targets; `:535-559` should not turn an indiscriminate KS alert into automatic retraining.
- The four `POINTS.md` files are large parallel accounts—1,665 to 2,656 prose words each—and have already drifted from canonical guidance. Establish the README as authoritative and reduce points to presentation cues/links, or deprecate them.

### Size and engagement findings

- The median core lecture is 2,393 prose words. Lecture 10 is the largest at 3,967 words (1.66× median); retain its required model-family breadth but compress repeated API detail, comparisons, and pop-culture framing.
- Lecture 02 has 2,049 bonus words across three files and repeats lambda and shell-scripting material. Lecture 03 has the highest structural density—40 headings and 33 fenced examples in 1,358 prose words—and repeats its NumPy quick reference/image while carrying optional shell/visualization catalogs in core.
- Lecture 06's core/bonus/points bundle and Lectures 07–09's parallel points pages are the main total-volume problem; the core lengths of Lectures 04–07 and 09 are otherwise proportionate to their scope. Lecture 11 is appropriately sized for its integrative role.
- The course has enough fun. Lectures 07 and 11 are the strongest models because engagement serves the concept; Lectures 01, 02, 05, 08, and 09 are also engaging. Do not add decorative jokes merely to enliven bonus pages. Replace Lecture 06's fabricated “2007 schism” and statistical joke with concept-serving humor, and replace Lecture 03's author-facing drafting note. Preserve every existing Lecture 10 Zoolander joke and reference while compressing repeated API detail around them.

### Instructor disposition and authorized repair scope

The instructor accepted the correctness, safety, metric-selection, size, and organization findings with these controlling clarifications:

- Compress and consolidate repeated API detail, especially in the large parallel bonus and points material.
- Replace rather than merely remove the Lecture 06 jokes so the lecture does not lose its humor.
- Bonus pages may remain dry; engagement additions are unnecessary there.
- Preserve all Lecture 10 Zoolander jokes and references.
- Normalize Lecture 03 toward the current repository convention—one H1 title, H2 sections, and nested headings as useful—without treating the convention as a rigid template. Within that structure, prefer the established explanation → reference → brief example rhythm where it helps.
- Materially trim and deduplicate Lecture 02's bonus bundle, with clear ownership between Python-concept extensions, Python CLI material, and advanced Git.
- Keep the lecture/demo boundary intact: lectures explain; they are not notebooks or scripts that must run top-to-bottom. Do not add activation, installation, import, or shared-state glue solely for whole-lecture executability.
- Continue to treat term-specific assignment URLs as intentional placeholders and leave demos and assignments untouched.

The authorized implementation batch covers the recorded P1/P2 lecture findings and these editorial constraints without expanding inherited topic scope. It will be followed by independent adversarial review and integrated lecture-only validation before approval is requested again.

### Repair implementation checkpoint (pre-review)

The first repair batch changed 24 lecture Markdown pages and no demo or assignment file. The working diff is deliberately subtractive: approximately 1,500 lines were removed and 550 added before this ledger update. The largest reductions establish clear ownership instead of maintaining parallel explanations:

- Lecture 02's Python-concepts, Python-CLI, and advanced-Git bonus pages now have distinct roles; repeated lambda, function, main-guard, and shell material was removed, and unsafe discard/history-rewrite recipes were removed or replaced with coordinated safety guidance.
- Lecture 03 now uses one H1 title with H2 main sections and nested subsections, removes duplicated quick-reference material and the author-facing drafting note, and repairs the NumPy membership, indirect-sort, and large-memmap examples without adding execution-state glue.
- Lectures 04–10 repair the recorded pandas, statistics, safety, visualization, time-series, modeling, serialization, attention, Featuretools, and drift-response defects.
- `POINTS.md` for Lectures 06–09 now treats each README as authoritative and retains concise instructor cues rather than a second full lecture. The required three `LIVE DEMO` markers remain in each of Lectures 07–09's cue sheets.
- Lecture 06's two bad jokes were replaced with concept-serving levity. Lecture 10 retained all Zoolander material present at the start of that repair batch while its surrounding API guidance was consolidated; the later direct comparison with `main` above found two older references that were already absent from that batch baseline.

This is a pre-review implementation record, not a correctness or approval verdict. Independent reviewers must still challenge factual/API safety and pedagogical structure/size against the complete lecture context, after which all lecture-wide gates must run from the current sources.

### Post-repair adversarial verdict and validation

**The authorized content/API repair batch passes; full lecture approval remains pending course-wide heading normalization.** Two independent read-only reviewers challenged the integrated repair set. The first pass rejected it with one P1 and eight unique P2 findings across presentation-marker preservation, cue-sheet hierarchy, shell no-match behavior, null-key sentinel wording, Plotly OLS requirements, a local NumPy import, zero-weight handling, and tick-index requirements. All were repaired. A targeted correctness re-review then caught one version-specific mistake: `adfuller(..., result_object=False)` is valid in statsmodels 0.15 but not the course's pinned 0.14.6. The example was restored to the pinned-compatible call and the actual pinned stack was executed. Both final re-reviews passed with no remaining P0–P3 content/API finding.

The final content/size review found no accidental loss of an inherited topic family:

- Lecture 02's bonus bundle is roughly 46% smaller by the reviewer's prose measure and now has distinct Python-concept, Python-CLI, and Git ownership.
- The four points pages are compact cue sheets rather than parallel lectures. Raw word counts fell from 2,725 to 550 (Lecture 06), 2,337 to 446 (Lecture 07), 2,005 to 302 (Lecture 08), and 1,603 to 352 (Lecture 09). Each retains exactly three `LIVE DEMO` markers and one H1 title.
- Lecture 03 has one H1 with coherent H2/H3/H4 nesting; its optional shell/terminal-visualization material follows the core environment and NumPy sequence without becoming a prerequisite.
- Lecture 10 retained all 11 detected Zoolander reference lines present at the start of that repair batch and the statsmodels, scikit-learn, XGBoost, TensorFlow/Keras, and PyTorch families while adding the missing task/cost-aligned metric bridge. The later direct comparison with `main` above supersedes the broader preservation inference: two additional older references remain to be restored.
- Lecture 06 retains concept-serving humor in both repaired locations. No decorative humor was added to dry bonus pages.

Final integrated gates from the current sources:

- Scope: 24 lecture Markdown files plus this branch-only ledger changed; no demo or assignment path changed.
- Structure: all 28 canonical lecture pages passed with 413 balanced fenced blocks, 298 parseable Python fences after accounting for notebook magics, and 86 resolving relative links.
- Hierarchy/cues: Lecture 03 and the four points pages each have one H1; Lectures 06–09 each retain exactly three `LIVE DEMO` markers.
- Runtime: a targeted suite passed with warnings promoted to errors under Python 3.12, NumPy 2.0.2, pandas 3.0.5, SciPy 1.18.1, statsmodels 0.14.6, Plotly, and scikit-learn 1.9.0. It covered NumPy membership/indirect sorting, DataFrame–Series alignment, null-key merge behavior, the t interval, group-wise Plotly OLS, named GroupBy aggregation, weighted reshape and zero-weight detection, STL, ADF, and tick resampling/index validation.
- Safety: the advanced CLI no-match and match branches passed a targeted shell-logic check; destructive Git recipes remain removed.
- Build: Eleventy wrote 30 pages and copied 124 assets.
- Hygiene: `git diff --check` passed, and every changed path is inside the authorized lecture scope or this ledger.

### Course-wide Markdown hierarchy clarification

The current authority is `work/course_dependency_alignment.md`: lecture sources use standard Markdown with one H1 title, H2 sections, and deeper headings as useful. The multiple-H1 Notion-import convention in the archived `work/implementation_plan.md` is not a current release constraint.

A course-wide inventory exposed a remaining normalization gap that the prior structural gate did not enforce:

- 15 of the 28 canonical lecture pages currently have exactly one H1.
- Nine core READMEs still use multiple H1 headings: Lectures 01, 02, and 04–10.
- `03/BONUS.md` and `05/BONUS.md` use multiple H1 headings.
- `04/BONUS.md` and `06/BONUS.md` have no H1 title.
- Lecture 03's README and the revised Lectures 06–09 points pages already follow the intended hierarchy.

The previous fence, Python-syntax, relative-link, runtime, and build results remain valid, but the heading-format acceptance claim was too narrow. Full lecture approval therefore requires a lecture-only hierarchy pass across the 13 nonconforming pages, with semantic demotion rather than a blind replacement of heading markers, followed by another structural and pedagogical review.

## Course-flow revision checkpoint

The user approved a lecture-only sequencing pass after reviewing the completed content comparison with `main`. The implementation changes only lecture material:

- Lecture 01 now states that Lectures 01–03 use the REPL and scripts and removes the premature `datetime` import preview.
- Lecture 02 distinguishes review from new material, removes the pre-pandas Markdown example, substantially condenses repeated shell/Python reference material, and orders new Python concepts as objects/imports → collections/mutability → functions → file I/O → targeted exceptions → script entry points.
- Lecture 03 bridges early script/Git fluency to reproducible environments and NumPy, compresses repeated Python material, and marks terminal processing/visualization as optional reference ahead of the canonical visualization lecture.
- Lecture 04 introduces notebook mechanics with already-taught core Python before pandas, moves rich DataFrame display after pandas structures, declares the pandas 3.x lecture contract, and limits its final quality section to inspection before Lecture 05 decisions.
- Lecture 05 aligns its name transformation with the stated trim/preserve-case contract, uses pandas-3-native `DataFrame.map` wording, and makes categorical encoding a forward reference to Lecture 10 rather than assuming modeling vocabulary.
- Lecture 06 teaches merge cardinality and `validate=` before its first demo and reminds students that index-aligned combination builds on Lecture 04.
- Lectures 07–09 make the prepared-visualization → aggregation → time-series sequence and optional forecasting preview explicit while teaching current pandas 3 `GroupBy.apply` behavior.
- Lecture 10 adds a survey-depth statistical vocabulary bridge before inference terms, and Lecture 11 adds a non-prescriptive concept-to-lecture-to-notebook crosswalk.

No demo or assignment file was inspected or edited as part of this implementation batch. The initial integrated structural and build gates passed. An independent adversarial review then identified prerequisite, ordering, and pandas 3 contract defects; all were repaired, the gates were rerun, and the final targeted re-review passed with no P0–P2 finding. The lecture states below reflect the current checkpoint.

### Independent adversarial flow-review findings and repairs

The review rejected the first sequencing checkpoint with one P1 and six P2 lecture-scope findings:

- Lecture 04's first `Series` and `DataFrame` examples used `pd` before importing pandas in a fresh kernel.
- Lecture 03 deactivated the candidate environment before the substantive NumPy sequence without telling students to reactivate the environment they created.
- Lecture 02's shortened shell reference overstated what Lecture 01 had already taught and removed the compact `grep`/redirection/pipeline/`chmod` prerequisites needed by later lecture setup.
- Lecture 05 still lowercased human names in an earlier custom-function example, contradicting the preserve-case data contract in the compact pipeline.
- `06/POINTS.md` retained the pre-cardinality demo ordering and omitted executable `validate=` guidance.
- `08/POINTS.md` reversed the meaning of `observed=True` for categorical pivot tables.
- `09/POINTS.md` incorrectly claimed that uppercase hourly alias `H` still works under the pandas 3 course contract.

A targeted re-review then caught one more internal contradiction in `06/POINTS.md`: a customer-left/purchases-right merge was labeled many-to-one instead of one-to-many. The text now states both directional forms consistently with `validate=` and the canonical README.

All eight findings were repaired in the owning lecture layer. The independent reviewer passed the final repair set with no P0–P2 finding. No demo or assignment change was authorized or required.

## Current lecture-by-lecture assessment

A post-repair read-only audit confirmed that every original P1/P2 finding and every substantive repair action in the earlier per-lecture scorecard is closed. The old numeric scores described the pre-repair state and are superseded by this status matrix. Heading conformance is shown separately because it is the only remaining lecture release gate.

| Lecture | Content repair | Heading gate | Current assessment |
| --- | --- | --- | --- |
| 01 | Resolved | Open: README | Strong foundation; the mixed Bash/Python bonus fence is repaired and the unique conditional-comic bridge replaces the duplicate closing callback |
| 02 | Resolved | Open: README | Core role is sound; bonus ownership, API-contract, Git-safety, and duplicate-comic cleanup are complete |
| 03 | Resolved | Open: BONUS | Environment-to-NumPy order works; density, stale NumPy API, duplicated reference, and author-note issues are resolved |
| 04 | Resolved | Open: README and BONUS | Strong pandas 3 foundation; optional safe notebook automation now sits with the Jupyter owner rather than cleaning |
| 05 | Resolved | Open: README and BONUS | Excellent contract-driven cleaning owner; core sampling serves manual inspection, notebook automation moved to Lecture 04, and the broken validation comic is repaired |
| 06 | Resolved | Open: README and BONUS | Strong joins/reshape core; null-key guidance, points drift, stale bonus claims, and harmful humor are resolved |
| 07 | Resolved | Open: README | Excellent visualization contract; inference, trend/grid examples, and duplicate extensions are repaired |
| 08 | Resolved | Open: README | Strong result-shape-first core; aggregation, validation, remote-safety, parallel-page volume, and the mislabeled comic are resolved |
| 09 | Resolved | Open: README | Strong temporal core; optional STL/ADF/count guidance and broken image links are repaired |
| 10 | Resolved | Open: README | Inherited survey breadth and all main-carried Zoolander references are intact; evidence-based model selection and the corrected unique Statistics comic remain in place |
| 11 | Resolved | Pass | Question-led, non-prescriptive rewrite includes the approved prompt, local definitions, three delivery anchors, a concept-serving Data Trap comic, and the capstone joke; the intentional opening visual remains |

## Demo repair checkpoint

The authorized demo phase repaired the execution, source-fidelity, ordering,
and version findings from the read-only audit. Markdown is now authoritative
for all 27 paired notebooks (04–11, including the Lecture 09 guide and the
optional Lecture 11 geo notebook), and fresh Jupytext generation matches every
stored notebook's cell types and sources. The Lecture 01–03 script demos remain
script-based because notebooks are intentionally introduced only in Lecture 04.

- Pinned the active demo environments to CPython 3.12.13 and the pandas 3
  candidate (`pandas==3.0.3`, `numpy==2.0.2`), with activity-specific exact
  pins for plotting, statistics, modeling, and the Lecture 11 capstone.
- Reconciled the richer 04–11 notebook material into Markdown, regenerated
  notebooks without outputs, and added the missing 04 Anscombe fixture and 05
  requirements file. Lecture 05 Demo 1 keeps its tabular missingness analysis
  as the core path and adds an explicitly labeled Lecture 07 heatmap preview
  for visual learners, with the plotting dependencies pinned alongside it.
- Fixed pandas 3/API issues: tied `qcut`, current Matplotlib `tick_labels`,
  Altair point selections, lowercase hourly aliases, and current notebook
  metadata. Removed runtime package installation from 04 demos.
- Corrected Lecture 10 prediction-interval semantics and test-set leakage in
  boosting/deep-learning comparisons; corrected dataset and Python-version
  claims in the guides. Lecture 11 local setup now uses `uv venv` plus
  `uv pip install` and keeps development Colab links explicitly pending
  immutable-tag publication.
- Kept the usual three-demo cadence for Lectures 01–10, consolidating the
  foundational Lecture 01 guide into setup/CLI, Python/control flow, and one
  simple integration workflow. Lecture 11 remains one capstone with four
  required notebooks and one optional geo bonus.
- Simplified the Lecture 01 integration script to concepts already taught in
  Lecture 01. Its CLI path-error exercise captures the expected failure and
  then runs the corrected path so the demo exits successfully.
- Kept Lecture 04 Demo 1 focused on notebook mechanics and core Python by
  removing its pre-pandas NumPy/pandas version check; the version contract is
  checked in the later pandas demos and their guide.
- Corrected stale cross-lecture references in the Lecture 09 guide/demo
  (GroupBy is Lecture 08; cleaning is Lecture 05) and moved timezone material
  to the end demo, after the lecture introduces it. Lecture 05's guide no
  longer repeats Lecture 04's notebook-automation material.

Independent implementation validation:

- All 27 Markdown/notebook pairs regenerated with Jupytext 1.18.1; zero
  cell-type/source mismatches and no stored outputs or execution counts.
- Disposable Lecture 01–03 script runs passed under Python 3.12.13 and NumPy
  2.0.2; shell syntax and Python compilation checks passed.
- Repair agents executed fresh copies of all 04–09 notebooks, all three
  Lecture 10 demos, and required Lecture 11 notebooks in isolated pinned
  environments with zero cell errors. The optional Lecture 11 geo notebook was
  subsequently executed in the same temporary capstone layout with its
  separate geospatial dependencies; it is included as a notebook and Colab
  option, while remote Colab execution remains a release-time certification
  task.
- The public Colab links and the Lecture 11 notebook fallback data URLs now use
  the same `2026-refresh` branch. All five Colab URLs and all four fallback
  data files returned HTTP 200, and a fresh no-local-data execution of
  `01_setup.ipynb` passed. Before release, retarget the links to `main` or an
  immutable tag and perform one manual Colab smoke test.
- Additional scoped checks found no stale uppercase pandas hourly aliases,
  legacy Matplotlib `labels=`, deprecated Altair selection APIs, unsafe
  `nbconvert --inplace`/`--allow-errors` guide references, or broad pandas-2
  requirements. `git diff --check` passed.

An independent adversarial reviewer confirmed the repaired 05 guide, all
source/notebook pairs, current APIs, and the three-demo cadence. The review
also classified the intentionally captured Lecture 01 path error as
non-blocking teaching behavior; the current script now handles it explicitly.
The focused follow-up also passed the simplified Lecture 01 integration and
CLI demos with zero stderr and confirmed that Lecture 04 Demo 1 no longer
imports pandas before pandas is introduced in the lecture.
- The revised Lecture 05 Demo 1, including its visual-preview cell, and the
  other two Lecture 05 demos were regenerated and executed in a fresh pinned
  environment with zero cell errors.
- The two unreferenced legacy demo Markdown files (`07/demo/data_cleaning_viz_demo.md`
  and `08/demo/live_demo_guide.md`) were removed; the active demo guides are
  now the only student-facing demo entry points for those lectures.
No assignment paths were edited. The active demo phase is ready for a
checkpoint commit; the separate lecture heading-format gate and eventual
branch-only cleanup remain open.

## Current release recheck (2026-08-28)

The tested demo baseline above is now behind the current upstream releases:
PyPI lists pandas 3.0.5 as the latest pandas release, and Python.org lists
Python 3.14.7 as the latest stable interpreter (with Python 3.12.14 and
3.13.15 as the current maintenance releases for those lines). The repaired
demos therefore have execution evidence for pandas 3.0.3 / CPython 3.12.13,
not yet for pandas 3.0.5 / CPython 3.14.7.

The Python choice needs an ecosystem gate rather than a version-number-only
upgrade. The current TensorFlow release publishes wheels through CPython 3.13
but not 3.14, so a single 3.14 baseline would make Lecture 10's TensorFlow
demo conditional or require a newer TensorFlow release. The next dependency
refresh should bump pandas to 3.0.5 and either keep the conservative 3.12 line
(updating to 3.12.14) or move the entire tested matrix to 3.13.15 together
with a TensorFlow/package refresh and a complete rerun.

Altair remains the recommended required declarative visualization for this
Python/Jupyter course. Apache ECharts is JavaScript/DOM-first; a Python use
would add a wrapper such as pyecharts and a second rendering/tooling model.
It is a good optional ecosystem comparison or web-dashboard extension, not a
clean replacement for the lecture's current Altair teaching contract.

## Assignment/grading design discussion (2026-08-29)

The instructor proposed retiring GitHub Classroom/Classroom 50 in favor of
optional student GitHub Actions plus pytest, with a separate TA/instructor
runner that reads one assignment-source/fork manifest, fetches fresh checkouts,
executes trusted tests, and writes a report for grading triage. The assignment
inventory supports this direction, with two constraints:

- Assignments 01–03 already expose pytest facades, but Assignments 04–11 expose
  standalone public checkers while their instructor-only `_grader_selftest`
  bundles contain the stronger fresh-execution, artifact, alternate-input, and
  repeatability contracts. Running only `check_assignment.py` for 04–11 would
  silently under-grade the current behavior.
- The repository has no root pytest configuration or grading workflow. Existing
  Classroom 50 files are mostly self-test/reference bundles; production
  provisioning is external and not a viable dependency for this semester.

The source-location clarification is controlling: the canonical source/starter
for each assignment is the existing `01/assignment` through `11/assignment`
subtree in this repository. We do not create or require eleven external source
repositories during this refresh. When the course is ready to publish, each
subtree can be exported as the basis of its own assignment repository; its
local tests, dependency files, and workflow travel with that export. The
workbook/site monorepo is not expected to execute nested workflow files.

Recommended replacement shape:

1. Keep each `NN/assignment` subtree portable. Add an optional
   `NN/assignment/.github/workflows/tests.yml` that runs the public pytest
   contract on push, pull request, or manual dispatch when that subtree is
   exported as a standalone repository. Students may ignore the workflow; it
   is feedback, not submission.
2. Keep public tests/checkers inside the portable subtree. Where the rebuilt
   package currently exposes only `check_assignment.py` (Assignments 04–11),
   provide a thin pytest entrypoint that invokes that public checker without
   importing or exposing the instructor-only `_grader_selftest` bundle.
3. Add a small instructor-only `grading/` runner in this repository later. Its
   source manifest should identify the local source subtree now and optionally
   record the exported repository/ref once repositories exist. A semester roster
   file records reviewed student repository mappings; GitHub fork discovery may
   generate candidates but must not silently become the grading roster.
4. At snapshot time resolve exported source and student refs to full commit
   SHAs. Clone each into a fresh directory, validate the committed tree, run
   trusted tests against a controlled `SUBMISSION_ROOT`, and emit JSON/CSV plus
   JUnit XML and bounded logs. Never run student-supplied pytest configuration or
   tests as the grader's test suite.
5. Preserve the current assignment-specific contracts while migrating their
   trusted logic behind pytest adapters. Keep human review for prose, chart
   quality, workflow evidence, and other non-automatable points. A passing
   report is a fast-track signal, not an automatic final grade.

The remaining design gates are the private/public location of the student
roster, the sandbox/container available for arbitrary student code, the
source-test visibility policy, the commit/ref policy for resubmissions, and
whether the first pilot should cover a simple script assignment plus one
notebook-heavy assignment. The source-location comparison changed the design
record only; assignment source content was not changed during this step.

## Assignment source-layout comparison (2026-08-29)

The `main` branch used the same portable-subtree idea, but its workflows were
legacy Classroom jobs: they downloaded mutable tests with `curl`, used mixed
Python 3.10–3.12 versions, and in several cases executed notebooks before
running pytest. The refreshed branch intentionally removed those jobs and
replaced them with stronger local public checkers plus instructor self-tests.
The useful structural part to retain is the per-assignment location:

- `NN/assignment/README.md`, starter artifacts, fixtures, and dependency pins
  remain the source bundle;
- `NN/assignment/.github/test/` is the exported repository's public pytest
  entrypoint;
- `NN/assignment/.github/workflows/tests.yml` is optional learner feedback;
- `NN/assignment/_grader_selftest/` remains instructor-only and is excluded
  when a student repository is exported.

Nested `.github` directories are not active Actions workflows while they live
under this monorepo; they become active after the selected assignment subtree
is copied to repository root. No external source-repository URLs or student
roster are required until that export/publishing step.

## Assignment portable-test implementation checkpoint (2026-08-29)

The portable source layout is now materialized for all eleven assignment
subtrees. Each `NN/assignment` contains:

- `.github/test/requirements.txt` with the public pytest runner dependency;
- `.github/test/test_assignment.py` as the standalone-repository test entrypoint;
- `.github/workflows/tests.yml`, an optional push/pull-request/manual workflow
  using the course's Python 3.12.13 line and the assignment's own requirements.

Assignments 01–03 keep their existing root-level public pytest facades; the
new `.github/test` files load those facades so the export has the conventional
main-branch path without duplicating checks. Assignments 04–11 expose a thin
pytest adapter that runs the public `check_assignment.py` in a subprocess. The
adapter does not import or invoke `_grader_selftest`, so the instructor-only
bundle remains outside the learner-facing test contract. Public checker and
central-grader package inventories now explicitly allow these three portable
test/workflow files; otherwise an otherwise-correct exported submission would
be rejected as containing unexpected files.

The old student-facing Classroom/Classroom50 instructions were replaced with
repository and optional Actions language in the affected assignment README and
platform-check pages. `03/assignment/requirements.txt` was also repaired from
the placeholder `TODO` to its already-documented NumPy 2.0.2 pin so the new
workflow has a valid dependency file.

Validation for this checkpoint: all eleven new test entrypoints parse with the
standard-library AST parser; every assignment has the expected three-file
`.github` layout; no student-facing Assignment README or PLATFORM_CHECK page
still names Classroom; and `git diff --check` passes. The workflows are not
claimed to have executed from this monorepo because GitHub does not activate
nested workflow files; each must be smoke-tested after its subtree is exported
to repository root. Assignment task content and the instructor's core grading
rules were otherwise left unchanged; package inventories and integrity hashes
were updated to account for the portable public-test files and the refreshed
dependency pin.

## Assignment/test/content alignment audit (2026-08-29)

The portable test shape is implemented, but it must not be mistaken for a
completed assignment-content migration. The `.github/test` files are adapters:
Assignments 01–03 load the existing root public pytest facades, while
Assignments 04–11 run the public `check_assignment.py` command in a subprocess.
The adapters and checker expectations were updated for the rebuilt fixture and
dependency contracts, but the student-facing assignment source was not
reconciled in the same step.

Static comparison found the following release blockers:

- Assignment 02's README still describes the legacy separate repository,
  `setup_project.sh`, and `src/data_analysis*.py` project, while the package
  contains `analysis_utils.py`, `main.py`, and the rebuilt checks.
- Assignment 03's README still requires removed health-data generators and a
  50,000-row CSV; the package now contains a fixed NumPy exercise and
  `observations.csv`.
- Assignment 04's README and notebook still use the removed
  `data_generator.ipynb`/`customer_purchases.csv` contract, while its checker
  expects the fixed `a04-purchases-v1` fixture and `a04-*` notebook cell IDs.
- Assignment 05's README still describes the removed eight-part clinical-trial
  pipeline; the package is the compact fixed-people cleaning notebook.
- Assignments 06–09 retain README prose for removed generators, datasets, or
  notebook layouts, while their checkers target fixed fixtures and rebuilt
  notebook contracts. Assignment 07 also says “four questions” while listing
  three, and Assignment 09 still documents three old datasets.
- Assignment 10's README and notebook still promise/import XGBoost, but
  `requirements.txt` and the checker intentionally define the bounded
  statsmodels/scikit-learn contract and reject XGBoost in student cells.
- Assignment 11 is the exception: its frozen Chicago sensor release, nine
  phases, sklearn-only model boundary, and no-geo assignment scope are aligned
  with Lecture 11 and its capstone transfer. Its Colab/local-Jupyter wording
  remains a separate publication decision.

Therefore the current tests are adapted to the intended rebuilt contracts only
in part, and the assignments are not yet fully updated to match the final
lectures and demos. The next authorized assignment pass must choose each
rebuilt package as authoritative, rewrite its README/notebook prompts and
starter files to that contract, then update public/instructor checks together
and smoke-test each exported subtree. No assignment source or test assertion
was changed during this audit.

## Assignment/demo dependency pin refresh (2026-08-29)

PyPI now lists pandas 3.0.5 as the current 3.x release (3.0.4 was yanked), so
the active assignment and demo contracts were mechanically aligned from 3.0.3
to 3.0.5. The Assignment 07 contract also now uses Matplotlib 3.11.1, matching
the demo and the course candidate rather than its older 3.10.8 pin. These
updates include requirements files, public/instructor expectation strings and
protected-file hashes, Assignment 11's release metadata, demo
metadata/manifests, and the repository audit's active expectations. The
assignment workflows continue to use CPython 3.12.13; moving to 3.14 remains a
separate ecosystem decision because the course's TensorFlow path is not part of
this assignment packaging change.

This was a contract/data-record update, not an assignment execution pass. The
user explicitly kept assignment execution out of the current review scope.
Static validation confirmed no active `3.0.3` references remain outside the
archived `work/` material and historical ledger text, all refreshed protected
requirement hashes match their files, and the Assignment 11 manifest hash was
updated to match its refreshed bytes. Fresh assignment and demo execution
remains a later release gate.

An attempted Assignment 04 instructor self-test stopped before grading because
the host does not have `nbformat`; this is an environment limitation, not a
passing or failing assignment result.

## Assignment source reconciliation (2026-08-29)

The assignment review is now active. Assignments 02–10 were reconciled to the
rebuilt lecture/demo contracts rather than the legacy README and notebook
prompts. The restored contracts now cover, in sequence: Git-safe script
refactoring (02), terminal NumPy work (03), first notebooks and labeled data
(04), documented cleaning (05), validated combination and structural reshape
(06), visualization critique/redesign (07), explicit-grain grouping (08),
entity-aware temporal evidence (09), and bounded OLS/chronological evaluation
(10). Assignment 01 remains the terminal/Python readiness exercise, and
Assignment 11 remains the integrated Chicago forecasting capstone.

Student-facing instructions no longer promise the removed generators,
clinical-trial package, legacy notebook layouts, or XGBoost assignment path.
The local-first execution boundary is explicit. Assignment 11 is local-Jupyter
only; it does not add a demo or Colab execution option.
Active assignment requirements, platform notes, notebook assertions, and
public/instructor protected-file and cell digests are synchronized to the
current pandas 3.0.5/NumPy 2.0.2 candidate (with the existing activity-specific
Matplotlib, seaborn, scikit-learn, and statsmodels pins).

The 04 and 05 starter `.gitignore` files were also restored to the exact
cache/output-exclusion contracts enforced by their public checkers; the
Assignment 04 instructor protected-file manifest now records its current
platform and public-checker bytes.

Static validation passed for JSON, notebook IDs/order, stale learner-facing
contract removal, Python compilation of all non-intentionally-broken assignment
files, protected-source digests, and `git diff --check`. The public checkers
were smoke-tested from their starter state: remaining failures are expected
TODO/generated-artifact or unavailable-host-dependency messages, not source
integrity or contract-map mismatches. Fresh notebook execution remains a
release gate for an environment containing each assignment's exact pins; the
host here is Python 3.14 and does not have the NumPy/Jupyter stack required by
the notebooks. Assignment 01's deliberately broken `debug_report.py` is an
intentional debugging starter and is excluded from the compilation claim.

## Validation recovered from the interrupted work

- At the initial recovered checkpoint, only lecture README files were modified. Later lecture-only reconciliation also updated lecture `BONUS.md` files and `06/POINTS.md`; no demo or assignment file was modified.
- `git diff --check` passed.
- Eleven lecture READMEs had balanced Markdown fences and valid relative links after inherited attachment repairs.
- 173 Python fences parsed after accounting for IPython magics.
- An independent review confirmed that Lecture 10 retained every model-family section and that Lecture 11 matched the requested capstone direction.
- An independent exhaustive review executed the substantive Lecture 06 examples under pandas 3.0.3 and found no remaining P0–P2 issues.
- Targeted Lecture 08 bonus examples passed under pandas 3.0.3, including grouped windows, multi-key chunk aggregation, header-only input rejection, and the statistical-test sample-size guard; an independent exhaustive recheck found no remaining P0–P2 issues.
- Targeted and independent execution under pandas 3.0.3 confirmed Lecture 09's corrected feature-table examples, Period/DatetimeIndex aliases, resampling, rolling, and high-frequency examples; the independent scope review confirmed all unintended expansion was removed.
- An independent Lecture 10 review confirmed the inherited statsmodels, scikit-learn, XGBoost, TensorFlow/Keras, and PyTorch survey remains intact and the modern SHAP example is coherent.
- An independent Lecture 11 review confirmed the capstone direction, terminology, pinned interpreter, and Bash/WSL setup; no remaining P0–P2 lecture-content issue was found.
- A final adversarial review compared all Lectures 01–11 with `main` and the McKinney-derived references. After repairing Lecture 10's remaining deterministic model-selection claims and stale framework descriptions, every lecture passed with no actionable P0–P2 content finding.
- The final lecture-wide structural gate checked 23 Markdown files, parsed 295 Python fences after accounting for IPython magics, resolved 67 relative links, and passed `git diff --check`.
- The Eleventy build completed successfully with 29 pages and 124 copied assets. The branch-only tracking files also build as pages while present; their required pre-merge removal remains recorded above.
- The new-session handoff checkpoint passed `git diff --check`; the Eleventy build completed with 30 pages and 124 copied assets after adding branch-only `HANDOFF.md`.
- The course-flow implementation checkpoint checked the same 23 lecture Markdown files, parsed 291 Python fences after accounting for notebook magics, resolved 71 relative links, and passed `git diff --check`.
- The course-flow implementation checkpoint Eleventy build completed with 30 pages and 124 copied assets.
- The revised Lecture 05 compact audit-to-save example executed under pandas 3.0.5 in an isolated temporary directory; every invariant passed and the assertions confirmed that submitted name case is preserved.
- After adversarial repairs, the lecture-wide structural gate checked 23 Markdown files, parsed 292 Python fences after accounting for notebook magics, resolved 71 relative links, and passed `git diff --check`.
- The post-repair Eleventy build completed with 30 pages and 124 copied assets.
- A pandas 3.0.5 execution check confirmed the customer-left/purchases-right merge accepts `validate='one_to_many'` and the reversed inputs accept `validate='many_to_one'` with identical rows.
- The independent targeted re-review passed all eight repaired findings with no remaining P0–P2 issue.
- The pandas 3 organization checkpoint's structural gate checked 23 lecture Markdown files, parsed 282 Python fences after accounting for notebook magics, resolved 73 relative links, and passed `git diff --check`.
- Its Eleventy build completed with 30 pages and 124 copied assets.
- A pandas 3.0.5 contract suite verified inferred `str` versus explicit nullable `string`, Copy-on-Write assignment behavior, text/category `describe`, the Lecture 04 GroupBy preview, Lecture 06 cardinality/alignment/pivot behavior, categorical `observed=True`, `GroupBy.apply(include_groups=False)`, caller-controlled dtype conversion, current timestamp and Period aliases, `asfreq()` versus resampling aggregation, and the full Lecture 05 contract-to-save pipeline.
- The final validation scope expanded from the 23 canonical top-level lecture pages to all 28 built lecture Markdown pages, including nested bonus pages. It found and repaired a duplicated lambda section and two inherited unclosed fences in `02/bonus/advanced_python_cli.md`.
- The expanded structural gate checked all 28 pages, found 424 balanced fenced blocks, parsed all 307 Python fences after accounting for notebook magics, resolved 74 relative links, and passed `git diff --check`.
- The post-repair Eleventy build completed with 30 pages and 124 copied assets.
- The pandas 3.0.5 contract suite was rerun from the current lecture sources under Python 3.12 with NumPy 2.0.2; all string, Copy-on-Write, cleaning-pipeline, grouping, merge/alignment/pivot, dtype, offset, and frequency-conversion checks passed with warnings promoted to errors.
- Independent adversarial review passed the pandas 3 content/API/organization changes and the final nested Lecture 02 bonus repair with no P0–P2 finding.
- The subsequent full adversarial audit used eight independent workstreams across lecture groups, course sequencing, factual/API currency, quantitative size, and engagement. All used the non-executable lecture boundary and excluded demos, assignments, and intentional term-specific URL placeholders.
- The size audit measured 28 canonical lecture pages. Core prose ranged from 1,101 words (Lecture 08) to 3,967 words (Lecture 10), with a 2,393-word median; it also counted headings, fenced examples, images, bonus volume, and parallel points volume without treating examples as an execution contract.
- Official scikit-learn, pandas, NumPy, statsmodels, Git, Jupyter, Keras, and Featuretools documentation supported the current-API and safety findings. A targeted run under Python 3.12, NumPy 2.0.2, and pandas 3.0.5 reproduced the unsupported DataFrame–Series `fill_value`, multi-value GroupBy aggregation, weighted-pivot, and `np.in1d` deprecation findings.
- The full audit found no P0 issue and passed the canonical course sequence and inherited scope, but found P1 correctness/safety defects and P2 organization/content defects. Lecture content is therefore not yet approved.
- An independent challenge of the synthesized report caught two overstatements: Lecture 10 already owns baseline/CV/test-split workflow, and Lecture 09's tick example assumes rather than necessarily lacks a `count` column. Both findings were narrowed, and the targeted re-review passed with no remaining material synthesis defect.
- The repository-wide course audit remains unsuitable as a lecture-only release gate because its demo/assignment expectations target an intermediate branch state. No current audit error was newly introduced by the lecture README changes.

## Next action

1. Commit the reconciled assignment-source/pytest/Actions checkpoint after
   this validation.
2. Design and implement the TA grading runner against local assignment
   subtrees first; add exported repository coordinates and a semester roster
   only when the standalone repositories are created.
3. If desired before merge, normalize the 13 remaining nonconforming lecture
   pages to one H1 with semantic H2/H3/H4 nesting as a separate lecture-format
   change. Do not merge to `main`; remove or explicitly exclude `AGENTS.md`,
   `HANDOFF.md`, and this tracker during the eventual merge cleanup.

## Checkpoint log

| Commit | Scope | Result |
| --- | --- | --- |
| `9523a1d` | Add branch-only tracking instructions and progress ledger | Durable recovery process established |
| `f727982` | Snapshot recovered edits to Lectures 01–11 | Interrupted work preserved before further reconciliation |
| `b0fae9a` | Reconcile Lectures 02–04 | Paths, environment scope, pandas 3 API, inline Markdown, and recorded package installation repaired |
| `45cd90f` | Narrow and reconcile Lecture 05 | Compact pipeline independently executed under pandas 3.0.3; API/type/boundary findings resolved |
| `888462c` | Reconcile Lectures 06–08 | pandas 3 examples, aligned patching/window semantics, scope boundaries, cardinality guidance, edge cases, and server-specific setup wording reconciled |
| `ea359ab` | Reconcile Lectures 09–11 | Unintended L09 expansion removed, L09/L10 bonus APIs repaired, L10 scope preserved, and L11 capstone/setup made approachable and reproducible |
| `63da8b6` | Close the lecture-content review | Repair inherited L06 links and remaining L10 overgeneralizations; record the final adversarial review and validation gates |
| `5e851d0` | Finalize the review ledger | Record the clean lecture-only checkpoint and pending user acceptance |
| `8fd8465` | Add a new-session handoff | Preserve the missing-history diagnosis, replay experiment, and completed lecture-review state outside task history |
| `3f2c9cd` | Rebalance lecture flow and prerequisites | Lecture-only implementation and initial integrated validation passed |
| `f9dcd50` | Close adversarial lecture-flow findings | Eight prerequisite/API-contract findings repaired; final structural, build, pandas 3, and independent review gates passed |
| `a728700` | Reorganize lecture content for pandas 3 | Expanded validation and independent review passed; lecture-only checkpoint ready for acceptance |
| `b2c1f09` | Run full adversarial lecture audit | Sequence/scope pass; no P0; P1 correctness/safety and P2 content/organization repairs required before approval |
| `c6fdaf4` | Resolve adversarial lecture findings | Authorized lecture-only repairs, pinned-stack validation, integrated gates, and final independent re-reviews passed with no P0–P3 finding |
| `fe3f003` | Record the repaired lecture checkpoint | Preserve the completed repair evidence and remaining heading gate |
| `bdc072d` | Record the lecture heading-format gap | Identify the remaining one-H1 normalization scope |
| `299cdbb` | Refresh per-lecture repair status | Confirm content findings closed separately from heading conformance |
| `d92522e` | Scope the core-lecture comparison | Define pairwise `main` versus branch review across Lectures 01–11 |
| `36ea01c` | Record the core-lecture comparison | Preserve the pairwise content, flow, size, and engagement verdicts |
| `7e6ba44` | Restore Lecture 10 references | Restore every main-carried Zoolander reference before further review |
| `aa35ea2` | Refine lecture ownership and capstone | Place sampling and notebook automation with their owning lectures and clarify Lecture 11 |
| `demo-repair` | Repair and validate lecture demos | Reconcile Markdown/notebook sources, standardize the pandas 3 candidate, repair execution/API/order defects, and pass independent demo review |
