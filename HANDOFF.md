# Codex Session and Lecture Review Handoff

Use this branch-only document to start a genuinely new Codex task. Do not resume or fork the affected task when testing whether Codex 0.150.1 fixes history replay; the purpose is to create a task whose history begins under the current version.

## Fresh-task kickoff

In the new task:

1. Read `AGENTS.md`, this file, and `review-progress-tracking.md` before changing course material.
2. Confirm the repository is `/home/christopher/projects/datasci_217`, the branch is `2026-refresh`, and the worktree is clean.
3. Confirm the CLI with `codex --version`, then run `codex doctor --json` and check `checks["app_server.status"].details["app-server version"]` for the persistent daemon. At this checkpoint both report `0.150.1`.
4. Run the history-replay experiment below before relying on cross-device task history.
5. Continue with the lecture-review next action. Do not edit demos or assignments, and do not merge to `main`, until the user explicitly accepts the lecture content and moves the review into the next phase.
6. Update `review-progress-tracking.md` after each substantive course-review step and before every commit. Commit coherent checkpoints frequently.

Suggested opening prompt:

> Read `AGENTS.md`, `HANDOFF.md`, and `review-progress-tracking.md`. This is a genuinely new task under the current Codex version. First help me test whether task history replays on a newly connected client; then present the completed lecture-content review for acceptance. Do not edit demos or assignments or merge to `main` yet.

## Missing-history incident

### Affected task

- Task ID: `01a03fa9-c597-7981-b453-262bade78747`
- Title: `217 Lectures`
- Created by Codex CLI `0.146.0`; later served by CLI/app-server `0.150.1`
- Screenshot of the truncated replay: `/home/christopher/.codex/attachments/278ed3fd-a983-4178-b11c-3d0e8a922816/codex-clipboard-7f9df8eb-519b-460a-b9a1-9e7a25f5fc85.jpg`

### Observed behavior

- The client left connected from the beginning retained the complete conversation.
- An already-subscribed iOS client continued receiving new messages live.
- A newly opened client, and the macOS client after reconnecting, replayed only the original user prompt, the first assistant response, and the first command.
- This happened twice during the lecture review, so chat history cannot be treated as the durable record of work.

### Local diagnosis

This is not an ordinary Unix file-permissions problem. The evidence points to a stalled durable-history projection:

- The complete, valid JSONL transcript is still present and growing in `/home/christopher/.codex/sessions/2026/08/26/rollout-2026-08-26T13-01-25-01a03fa9-c597-7981-b453-262bade78747.jsonl`. It is owned by `christopher:christopher` with mode `600` and was over 12 MB at the latest check.
- The app-server runs as the same user, and the relevant Codex directories and SQLite databases have compatible ownership and permissions.
- The reconnect-facing history database, `/home/christopher/.codex/thread_history_1.sqlite`, contains only the first turn for this task. Its projector stopped at rollout byte `255680`, ordinal `17`, far behind the rollout file.
- `/home/christopher/.codex/logs_2.sqlite` repeatedly records: `expected ordinal 17, got 18; 0 rejected rollout lines cannot cover that gap` for `codex_thread_store::local::live_writer`.
- At diagnosis time, related ordinal-gap warnings affected 35 local tasks, indicating a broader projection problem rather than a repository-specific issue.
- `codex doctor --json` reports overall status `ok`, including database integrity and rollout/state inventory. That does not contradict the finding: its checks do not establish that every task's paginated history projection has caught up.
- A read-only `codex migrate-rollouts --thread 01a03fa9-c597-7981-b453-262bade78747 --json` reported `already_paginated`; the problem is not a legacy rollout awaiting migration.

The version transition from task creation under `0.146.0` to serving under `0.150.1` may be relevant, but it is only a hypothesis. The local evidence establishes where replay fails, not why the ordinal was skipped.

Official OpenAI documentation identifies `codex remote-control` as experimental, `codex doctor` as the diagnostic command, and `codex resume`/`codex fork` as ways to continue or copy existing sessions. It does not currently document this projection failure or a supported repair: <https://learn.chatgpt.com/docs/developer-commands>.

### Safety boundaries

- Do not change permissions; they are already correct.
- Do not delete or directly edit the rollout or SQLite databases.
- Do not run `migrate-rollouts --apply` for the affected task; its rollout is already paginated.
- Preserve the rollout before any future repair experiment. For live SQLite databases, use a consistent SQLite backup after stopping the writer or an appropriate backup API rather than copying only the main database file while it is active.
- `codex resume` may reopen the local session but is not known to repair cross-device replay. A new task is required for the version-isolation experiment.

### New-task replay experiment

This experiment separates live delivery from durable replay:

1. Create a genuinely new task while CLI and app-server both report `0.150.1`; record its task ID and creation time here.
2. Exchange at least three turns and include at least one harmless read-only tool call, such as `git status --short --branch`, so the task has multiple persisted items. Do not mutate repository or external state merely to exercise history replay.
3. Only after those items finish, open that task on a client that was not already subscribed, or fully disconnect and reconnect a client.
4. Verify that the new client can see every earlier user message, assistant message, and tool call—not merely messages arriving after it connects.
5. Repeat once after another completed turn.
6. Record the result here and commit it before doing lengthy work in the new task.

Interpretation:

- If full replay works, the failure is likely tied to older task state or the `0.146.0` to `0.150.1` transition, though one passing task will not prove the projector is universally fixed.
- If replay truncates again, preserve the new rollout and task ID. That demonstrates the defect remains reproducible in a task created entirely under `0.150.1`.

Experiment record:

- New task ID: pending
- CLI/app-server versions: pending
- Clients used: pending
- Replay result: pending
- Relevant log error, if any: pending

## Lecture-content review handoff

### Scope and current state

- Branch: `2026-refresh`
- Lecture review status: Lectures 01–11 pass the completed content review; user acceptance is still pending.
- Local branch state before this handoff checkpoint: clean and eight commits ahead of `origin/2026-refresh`.
- Demos and assignments were not edited during the lecture pass.
- Merging is explicitly out of scope.
- `AGENTS.md`, `review-progress-tracking.md`, and this handoff are branch-only operational records. Remove or explicitly exclude them before an eventual merge to `main`.

The review compared the branch with `main`, relevant Wes McKinney-derived material under `work/mckinney_content`, current package behavior, and the intended course sequence. The goal was to undo unintended LLM scope changes while retaining justified correctness and reproducibility repairs.

### Decisions to preserve

- Lecture 03: distinguish environment, interpreter, Python version, package, and dependency concepts. Lectures may use clearly labeled pedagogical alternatives, while demos must execute with their recorded environments.
- Lecture 05: retain an approachable cleaning workflow without expanding it into a disproportionate production framework.
- Lecture 10: preserve the inherited model-family survey and topic scope. Changes are limited to correctness, current APIs/framework descriptions, and appropriately conditional evaluation guidance.
- Lecture 11: retain the McKinney-informed topics inside an approachable, question-led worked capstone. Avoid presenting one lifecycle as universally mandatory; divide time roughly evenly between lecture reasoning and a live workflow walkthrough.
- The eventual Lecture 11 assignment should be a somewhat guided mini research project using a dataset different from the demo. Dataset selection remains for the later demo/assignment phase.

### Completed checkpoints

| Commit | Scope |
| --- | --- |
| `9523a1d` | Add branch-only tracking instructions and progress ledger |
| `f727982` | Preserve recovered Lecture 01–11 edits after task-history loss |
| `b0fae9a` | Reconcile Lectures 02–04 |
| `45cd90f` | Reconcile and narrow Lecture 05 |
| `888462c` | Reconcile Lectures 06–08 |
| `ea359ab` | Reconcile Lectures 09–11 |
| `63da8b6` | Complete final adversarial lecture review and repairs |
| `5e851d0` | Finalize the durable lecture-review ledger |

### Final review and validation result

- All Lectures 01–11 passed the final adversarial comparison with no remaining actionable P0–P2 lecture-content findings.
- Lecture 10 retained its inherited statsmodels, scikit-learn, XGBoost, TensorFlow/Keras, and PyTorch survey; deterministic model-selection claims and stale framework descriptions were repaired.
- Lecture 11 now uses the requested approachable, non-prescriptive capstone structure with roughly equal lecture and demonstration time.
- The lecture-wide structural gate checked 23 Markdown files, parsed 295 Python fences after accounting for IPython magics, and resolved 67 relative links.
- `git diff --check` passed.
- The Eleventy build passed with 29 pages and 124 copied assets.
- Detailed lecture-by-lecture findings and validation evidence are in `review-progress-tracking.md`; treat that file as authoritative if this summary and the ledger ever diverge.

### Deferred until the demo phase

- `09/demo/requirements.txt` permits pandas 2.0 even though the reviewed lecture consistently uses pandas 3 aliases. Reconcile the demo environment before executing that demo.
- `11/demo/DEMO_GUIDE.md` contains an invalid `uv sync --no-project ... -r requirements.txt` command under uv 0.12.1. Replace it with the tested `uv venv` plus `uv pip install -r` sequence during demo review.

These are recorded findings only. Do not edit either file until the user accepts the lecture review and explicitly begins the demo phase.

## Next course-review action

Present a concise consolidated summary of the Lecture 01–11 review to the user for acceptance. If the user requests another lecture-content pass, keep it lecture-only and update the tracker before committing. Begin demos, and only later assignments, after explicit user direction.
