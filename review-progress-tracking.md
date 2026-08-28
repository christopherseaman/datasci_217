# Lecture Review Progress Tracking

This is the durable checkpoint for the `2026-refresh` review. Update it after every substantive review or editing step and before every commit.

## Branch-only status

- Active branch: `2026-refresh`
- Current phase: lecture content only
- Explicitly out of scope for this phase: demos, assignments, merging to `main`
- Branch-only working records: `AGENTS.md` and this file
- Before the eventual merge to `main`: remove or explicitly exclude both branch-only working records

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

## Fresh read-only review results

| Lecture | Current review state | Required follow-up |
| --- | --- | --- |
| 01 | Pass | None currently identified |
| 02 | Pass | Illustrative `study_data.csv` path reconciled |
| 03 | Pass | Candidate environment explicitly scoped to Lecture 03 while allowing recorded later requirements |
| 04 | Pass | Removed `DataFrame.applymap` reference, repaired inline code, and made `%pip` use recorded requirements |
| 05 | Pass | Independent recheck executed the compact audit-to-save example under pandas 3.0.3; scope and prior API/type/boundary findings are resolved |
| 06 | Pass | Independent exhaustive recheck executed the substantive examples under pandas 3.0.3; bonus grouping, stacking, patching, and cardinality guidance are reconciled |
| 07 | Pass | Inherited optional survey retained; overlap with bonus material is now explicitly reference-only and unassessed |
| 08 | Pass | Independent exhaustive recheck found no remaining P0–P2 issues after pandas 3 API, grouped-window alignment, dependency, empty/short chunk, and server setup repairs |
| 09 | Core pass; bonus no-pass | Replace removed/deprecated pandas offset aliases in `09/BONUS.md` |
| 10 | Pass; scope preserved | Repair incompatible SHAP bonus example; add a minimal environment handoff without expanding scope |
| 11 | Pass on capstone direction | Pin the demo interpreter and state the Bash/WSL platform assumption; optional brief definitions may improve approachability |

## Validation recovered from the interrupted work

- Only lecture README files were modified before this checkpoint.
- `git diff --check` passed.
- Eleven lecture READMEs had balanced Markdown fences and valid relative links after inherited attachment repairs.
- 173 Python fences parsed after accounting for IPython magics.
- An independent review confirmed that Lecture 10 retained every model-family section and that Lecture 11 matched the requested capstone direction.
- An independent exhaustive review executed the substantive Lecture 06 examples under pandas 3.0.3 and found no remaining P0–P2 issues.
- Targeted Lecture 08 bonus examples passed under pandas 3.0.3, including grouped windows, multi-key chunk aggregation, header-only input rejection, and the statistical-test sample-size guard; an independent exhaustive recheck found no remaining P0–P2 issues.
- The repository-wide course audit remains unsuitable as a lecture-only release gate because its demo/assignment expectations target an intermediate branch state. No current audit error was newly introduced by the lecture README changes.

## Next action

1. Reconcile Lectures 06–11 and their bonus material in small commits.
2. Re-run lecture-only syntax, fence, link, and diff validation.
3. Present the consolidated lecture review before touching demos or assignments.

## Checkpoint log

| Commit | Scope | Result |
| --- | --- | --- |
| `9523a1d` | Add branch-only tracking instructions and progress ledger | Durable recovery process established |
| `f727982` | Snapshot recovered edits to Lectures 01–11 | Interrupted work preserved before further reconciliation |
| `b0fae9a` | Reconcile Lectures 02–04 | Paths, environment scope, pandas 3 API, inline Markdown, and recorded package installation repaired |
| `45cd90f` | Narrow and reconcile Lecture 05 | Compact pipeline independently executed under pandas 3.0.3; API/type/boundary findings resolved |
| Pending | Reconcile Lectures 06–08 | pandas 3 examples, aligned patching/window semantics, scope boundaries, cardinality guidance, edge cases, and server-specific setup wording reconciled |
