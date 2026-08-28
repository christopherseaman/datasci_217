# Lecture Review Progress Tracking

This is the durable checkpoint for the `2026-refresh` review. Update it after every substantive review or editing step and before every commit.

## Branch-only status

- Active branch: `2026-refresh`
- Current phase: lecture sequencing revision implemented and structurally validated; independent adversarial review pending; demos and assignments remain deferred
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

No demo or assignment file was inspected or edited as part of this implementation batch. The integrated structural and build gates passed; an independent adversarial review is the next action. The lecture states below remain the prior accepted checkpoint until that review finishes.

## Fresh read-only review results

| Lecture | Current review state | Required follow-up |
| --- | --- | --- |
| 01 | Pass | None currently identified |
| 02 | Pass | Illustrative `study_data.csv` path reconciled |
| 03 | Pass | Candidate environment explicitly scoped to Lecture 03 while allowing recorded later requirements |
| 04 | Pass | Removed `DataFrame.applymap` reference, repaired inline code, and made `%pip` use recorded requirements |
| 05 | Pass | Independent recheck executed the compact audit-to-save example under pandas 3.0.3; scope and prior API/type/boundary findings are resolved |
| 06 | Pass | Independent exhaustive recheck executed the substantive examples under pandas 3.0.3; bonus grouping, stacking, patching, cardinality guidance, and inherited course links are reconciled |
| 07 | Pass | Inherited optional survey retained; overlap with bonus material is now explicitly reference-only and unassessed |
| 08 | Pass | Independent exhaustive recheck found no remaining P0–P2 issues after pandas 3 API, grouped-window alignment, dependency, empty/short chunk, and server setup repairs |
| 09 | Pass | Removed unintended scope expansion; retained inherited runtime repairs and reconciled Period versus DatetimeIndex aliases under pandas 3.0.3 |
| 10 | Pass; scope preserved | Full inherited model survey retained; modern SHAP API, environment handoff, conditional model-selection guidance, and current framework descriptions independently rechecked |
| 11 | Pass | Approachable question-led capstone, non-prescriptive framing, roughly equal lecture/demo time, terminology, interpreter pin, and Bash/WSL assumptions independently rechecked |

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
- The repository-wide course audit remains unsuitable as a lecture-only release gate because its demo/assignment expectations target an intermediate branch state. No current audit error was newly introduced by the lecture README changes.

## Next action

1. Run an independent adversarial prerequisite and scope review of the validated lecture-only sequencing revision; repair any actionable findings before closing the pass.
2. Re-run the affected and lecture-wide validation gates after any repair.
3. Present the revised lecture flow for user acceptance.
4. Do not begin demo or assignment edits until the user moves the review to that phase. When that happens, start with the deferred environment/setup findings and course-wide pandas 3 pin reconciliation.

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
| This checkpoint | Add a new-session handoff | Preserve the missing-history diagnosis, replay experiment, and completed lecture-review state outside task history |
| Working checkpoint | Rebalance lecture flow and prerequisites | Lecture-only implementation and integrated validation passed; adversarial review pending |
