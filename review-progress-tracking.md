# Lecture Review Progress Tracking

This is the durable checkpoint for the `2026-refresh` review. Update it after every substantive review or editing step and before every commit.

## Branch-only status

- Active branch: `2026-refresh`
- Current phase: content/API repair batch passes, but course-wide lecture heading normalization is still required before approval; demos and assignments remain deferred
- Explicitly out of scope for this phase: demos, assignments, merging to `main`
- Branch-only working records: `AGENTS.md`, `HANDOFF.md`, and this file
- Before the eventual merge to `main`: remove or explicitly exclude all three branch-only working records

## Review objective

Review all student-facing lecture content against `main`, the relevant Wes McKinney chapter extracts under `work/mckinney_content`, current package behavior, and the intended course sequence. Preserve inherited topic scope unless a change is explicitly justified. Correct factual/API defects and unintended LLM changes without silently redesigning a lecture.

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
- Lecture 06's two bad jokes were replaced with concept-serving levity. Lecture 10 retains all existing Zoolander material while its surrounding API guidance is consolidated.

This is a pre-review implementation record, not a correctness or approval verdict. Independent reviewers must still challenge factual/API safety and pedagogical structure/size against the complete lecture context, after which all lecture-wide gates must run from the current sources.

### Post-repair adversarial verdict and validation

**The authorized content/API repair batch passes; full lecture approval remains pending course-wide heading normalization.** Two independent read-only reviewers challenged the integrated repair set. The first pass rejected it with one P1 and eight unique P2 findings across presentation-marker preservation, cue-sheet hierarchy, shell no-match behavior, null-key sentinel wording, Plotly OLS requirements, a local NumPy import, zero-weight handling, and tick-index requirements. All were repaired. A targeted correctness re-review then caught one version-specific mistake: `adfuller(..., result_object=False)` is valid in statsmodels 0.15 but not the course's pinned 0.14.6. The example was restored to the pinned-compatible call and the actual pinned stack was executed. Both final re-reviews passed with no remaining P0–P3 content/API finding.

The final content/size review found no accidental loss of an inherited topic family:

- Lecture 02's bonus bundle is roughly 46% smaller by the reviewer's prose measure and now has distinct Python-concept, Python-CLI, and Git ownership.
- The four points pages are compact cue sheets rather than parallel lectures. Raw word counts fell from 2,725 to 550 (Lecture 06), 2,337 to 446 (Lecture 07), 2,005 to 302 (Lecture 08), and 1,603 to 352 (Lecture 09). Each retains exactly three `LIVE DEMO` markers and one H1 title.
- Lecture 03 has one H1 with coherent H2/H3/H4 nesting; its optional shell/terminal-visualization material follows the core environment and NumPy sequence without becoming a prerequisite.
- Lecture 10 retains all 11 detected Zoolander reference lines and the statsmodels, scikit-learn, XGBoost, TensorFlow/Keras, and PyTorch families while adding the missing task/cost-aligned metric bridge.
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
| 01 | Resolved | Open: README | Strong foundation; the mixed Bash/Python bonus fence is repaired; defer pacing changes until delivery feedback |
| 02 | Resolved | Open: README | Core role is sound; bonus ownership, API-contract, and Git-safety cleanup is complete |
| 03 | Resolved | Open: BONUS | Environment-to-NumPy order works; density, stale NumPy API, duplicated reference, and author-note issues are resolved |
| 04 | Resolved | Open: README and BONUS | Strong pandas 3 foundation; the bonus alignment example is repaired |
| 05 | Resolved | Open: README and BONUS | Excellent contract-driven cleaning owner; missingness sequencing and non-destructive notebook guidance are clarified |
| 06 | Resolved | Open: README and BONUS | Strong joins/reshape core; null-key guidance, points drift, stale bonus claims, and harmful humor are resolved |
| 07 | Resolved | Open: README | Excellent visualization contract; inference, trend/grid examples, and duplicate extensions are repaired |
| 08 | Resolved | Open: README | Strong result-shape-first core; aggregation, validation, remote-safety, and parallel-page volume issues are resolved |
| 09 | Resolved | Open: README | Strong temporal core; optional STL/ADF/count guidance and broken image links are repaired |
| 10 | Resolved | Open: README | Inherited survey breadth is intact; metric selection, API/claim corrections, and repeated detail are repaired while preserving all Zoolander material |
| 11 | Pass unchanged | Pass | Question-led, non-prescriptive, and appropriately sized capstone |

## Deferred to the demo phase

- `09/demo/requirements.txt` permits pandas 2.0 even though the reviewed lecture now consistently targets pandas 3 aliases; reconcile the recorded demo environment before executing that demo.
- `11/demo/DEMO_GUIDE.md` uses an invalid `uv sync --no-project ... -r requirements.txt` command under uv 0.12.1; replace it with the tested `uv venv` plus `uv pip install -r` sequence during demo review.

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

1. Confirm the standard-Markdown decision and normalize the 13 remaining nonconforming lecture pages to one H1 with semantic H2/H3/H4 nesting.
2. Rerun course-wide heading, fence, link, syntax, build, and independent pedagogical gates before requesting lecture approval.
3. Keep demos and assignments deferred until lecture content is explicitly approved; review those as separate later phases.

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
