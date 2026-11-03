# Handoff

Updated 2026-11-03. Read [AGENTS.md](AGENTS.md) for the authoring and verification rules. This brief holds the current state, standing decisions, and open items; Git history and `work/reviews/` hold completed work.

## Overall goal

Prepare a coherent, practical introduction to Python and data-science tools for health data science master's students who are beginners in programming. Lectures explain concepts in order, with visuals, useful reference cards, minimal method examples, and relevant humor. Demos are reproducible independently after a missed class. Assignments are doable using preceding lectures and demos, with well-defined student-created artifacts and helpful, generous checks.

Each ordinary class has 90 minutes: about 60 of lecture and 30 of core demo walkthroughs. Lecture 11 is the project-workflow exception. Successful execution and shorter files do not establish a 90-minute classroom fit.

## Repo and release state

- On November 3, `main` and `origin/main` both contained the latest course-content commit `3472138`. The October 29–30 changes supersede the earlier review state; check `git status` and `git log` on arrival. Notion release verification is recorded below separately from Git state.
- Recorded handout releases for `UCSF-DataSci/ds217-26f-04` through `-11`: 05 on October 19; 06 and 11 on October 21; 08–10 on October 29; 04 and 07 on October 2 or 14. These are historical release dates, not a fresh parity check. Pushing the course updates the checks that homework forks download from `main`, never the handouts themselves.
- Assignment 05 had 8 forks at the October 19 check. That day its handout renamed the row-count artifact to `output/row_counts.csv` with the index `step`; earlier forks save `output/decision_log.csv` with the index `decision`. The checks accept both until the November cleanup in [TODO.md](TODO.md).
- Assignments 06 (the midterm), 08, 10, and 11 had no forks on October 20 and 21.

## Recent work, October 19 to 21

- Every lecture opens with a blue 🧭 callout, `# This Lecture`, holding `## Covers` (topics plus one Readings bullet) and `## Enables` (3 to 5 practical skills). AGENTS.md defines the shape.
- Lecture 06 was reworked from the instructor's notes:
    - One **key** idea replaces primary, candidate, and foreign keys; grain is the **unit of analysis**.
    - Join types sit in `pd.merge()` with a Venn image and left join as the course default.
    - Output tables drop the meaningless 0, 1, 2 index; headings name their function.
    - `combine_first()` fills missing values.
    - Two-column `set_index`, two-level `.loc`, and `stack()`/`unstack()` moved to BONUS and a Demo 2 extension; the main lecture keeps `reset_index()`.
    - The old join, wide/long, and MultiIndex images and the memes and comics are back, spaced at most 87 lines apart. Their sources are the old Notion export under `~/projects/sqrl/Notion/` and DataSci 223. The instructor dropped the beard join image and gave three memes plain captions, no jokes.
    - Demos 1 to 3 each carry two seaborn preview charts and cover `left_on=`/`right_on=` (Demo 1) and `%%bash wc -l` (Demo 3).
- Midterm grading charges each mistake once with simple rules. When `visits` is below the expected count and charged, rows missing from the two extracts cost nothing more. An October 21 audit (fresh completion 75/75) led to four more charge-once fixes. The self-test has 46 single-mistake cases and takes about 15 minutes; for a small grader change, run only the clean submission and the affected cases first (about 4 minutes), then the full suite.
- Exam handouts (06 and 11): each task's Checkpoint names its artifact, header, and line count. Check your work is Restart and Run All, then commit and push. CHECKS.md lists the checks run on the artifacts, the proof of work, and human review shows only category and full credit.
- The shark-repellent `xkcd/` toolkit searches, audits, and fetches comics (AGENTS.md has the commands). Comics 1205, 1319, and 1513 each appear in one lecture page plus an unused media file in another lecture.

## October 29: Lectures 08 to 10 rework

- Lectures, demos, BONUS, and Assignments 08 to 10 were rebuilt (commit 6d4da93) and Notion pages 08 to 10 plus their BONUS pages republished the same day. Demos and assignments use seeded causal-DAG generators; assignment Parquet lives in `NN/assignment/data/` and the build scripts in `scripts/build_assignmentNN_data.py`.
- Handout repos ds217-26f-08 to -10 were republished from main on October 29 (no forks existed).
- `notion_push.py` now uploads `<video>` and `<pdf>` tags and recreates `<bookmark url>` tags, so Notion-only media must be added to the repo before a republish. Lecture 08's instructor video and PDFs live in `08/media/`.
- Awaiting instructor review: Assignment 10's `_cause` now prefers a cause an earlier check already charged; Demo 10 call list targets patients under 150 who cross 150 next week.
- The instructor explicitly excluded Colab validation. Verify notebooks locally; do not treat hosted Colab certification as a pending task. Real remote-server workflows remain unverified.

## October 29–30 follow-through and November 3 release

- `0e2e91a` added mechanism visuals, comics, and visible demo outputs across 08–10: transform and filter-versus-mask diagrams, count/time-window and weighting diagrams, and estimator, network, and threshold-tradeoff figures. Assignment pointers were updated to identify the relevant teaching examples.
- `3472138` frames each lecture's three demos as one stakeholder project: a concrete request, data problems, methods, and decisions. The project framing retains independently runnable notebooks. Assignment 09's demo pointer was corrected after renumbering.
- November 3 live reconciliation confirmed that all six 08–10 lecture and BONUS pages already match the current sources; no replacement was necessary. All 211 code blocks matched, all 96 live media files matched local SHA-256 hashes, all three BONUS child-page IDs survived, and the bookmark matched. No synced blocks or unknown edits were found; the scoped media audit reported zero problems. Evidence: `scratch/nov03-release/notion-report.md` and its adjacent `notion/` directory (ignored workspace evidence).
- Local verification: all nine 08–10 notebook sources match their generated notebooks, and the three lecture pages pass lint. Four notebooks with edits after their saved executions were regenerated from current Markdown and rerun in fresh scratch folders: 08 Demo 3, 09 Demos 1–2, and 10 Demo 3. All passed with inspected outputs on Python 3.13.14/pandas 3.0.5; install cells were skipped after dependency verification. The other five notebooks' current code matches successful saved executions. Evidence: `scratch/handoff-validation-20261103/` and `scratch/pd/{08,09,10}-review/`. This is local notebook evidence, not a new classroom or pedagogical review.
- The unchanged `08/demo/build_features.py` entry point also ran against freshly generated data, including all 12 progress steps. Its `output/pilot_list.csv` has 659 distinct patients, the seven documented columns, Bayview/Excelsior clinics only, and latest A1c at least 9.0. Evidence is under `scratch/handoff-validation-20261103/helper/`; this does not verify the tmux UI or a remote server.
- Assignment repositories are not included in this documentation release; their last publication dates above do not establish parity with later local handout edits. HANDOFF.md is repository-only, not a mapped Notion or site page.
- Untracked `RESUME` and `refs/` were present at the start of the release. They are not part of this release and must not be added or removed incidentally.

## Coming up

- Lecture 06 and its demos were reviewed five times and the exam audited on October 21. Human exam judgments remain outside automated verification.
- November: remove the Assignment 05 legacy file name (TODO.md).
- Republish each handout only after a fresh reconcile, and only when the instructor asks.

## Working agreements

- Review priorities: lecture content, topic inclusion and ordering, pedagogical visuals, and humor first; then runnable demos that practice preceding content and show meaningful visual output; then assignments completable with lecture/demo methods. Passing execution checks alone does not establish teaching quality. Do not cut humor or add Colab validation work.
- Quick, well-specified edits go to a light model with exact instructions, followed by one check (lint or the single relevant test). Add review or completion passes only when the instructor asks.
- Commit, push, and publish changed lecture pages together; never leave Notion behind `main`.
- Subagent prompts must quote the instructor's request: agents see only their prompt, and they refused once when they lacked it.

## Decisions to preserve

- Keep non-repeated Git guidance, including website upload, and some duck typing in 01; keep Markdown as required communication material and non-repeated Git in 02. Students struggle with Git. Lecture 02's debugger stays an optional BONUS exercise; 02 keeps its Command-Line Catalog. VS Code commit and sync screenshots stay in 01 and 02, with Microsoft's merge-conflict link. Lecture 01 keeps its Quick references.
- In 03, mention venv/Conda in main and keep their workflows in BONUS. 2-D boolean indexing (steps, mask diagram, card, IndexError), the instructor's Examples of Wildcards table, and 2-D sorting (`np.sort(axis=)`, `argsort`, ordering rows) stay in main; 3-D and deeper selection stays in BONUS. The instructor accepted 03's length; do not reopen it for exceeding a soft size target.
- Keep Parquet in 04 and all cleaning functionality in 05. The 04 demo project pins pyarrow for real Parquet round trips.
- Keep Lecture 06's join-type comparison and cross-join audit; building both shapes is mentioned in the lecture and shown in Demo 2. Keep the accepted 06 compaction and only genuine repetition cleanup in 07.
- Keep SSH/tmux/Jupyter in 08. Lecture 08 keeps filter thresholds, within-group apply (sorting, `nlargest`), the MultiIndex result, and the `.values` length error. The real course server remains unverified; the terminal exercise rehearses locally.
- Lecture 09's blocks are dated data, then resampling/grids/windows, then clocks (time zones, clock-change nights; moved to block 3 on 2026-10-29) and honest entity history. Its Python `datetime` section stays in main. Frequency inference, specialized schedules, clock-time alternatives, percentage changes, and Grouper are BONUS.
- Keep XGBoost and Keras in the 10 lecture and demo, plus its Zoolander hook and the lab-analyzer, practice-exam, and scrambled-chart analogies. Lecture 11 stays at its pre-pacing version except the instructor-approved November 3 additions: "A Step-by-Step EDA Pass" (steps of EDA, pairplots, selection and measurement bias) and the EDA step in Demo 03_model_prep. Discuss removing functionality with the instructor; shorten examples and remove repetition instead.
- Humor follows Lecture 03: about one relevant comic per 150 lines, at least one per block, no comic reused across lectures. Size targets (about 850 lines, 3,000 prose words) are diagnostics.
- Lectures carry no links to other course pages; Notion's hierarchy connects them.
- Dataset exceptions: Lecture 11 uses NYC taxis; Assignment 11 uses Chicago beach weather. Assignment 10's 30/8/10 split is retrospective. Demo 10's prospective workflow uses only labels available at its first validation cutoff.
- Assignment 01's report gets full credit when the summary prints sit inside the loop.
- The midterm is `06/mid-term`. Its dataset keeps `E04` as an unmatched satellite code and 70/250 mmHg as explicit teaching boundaries. Missing gender in 05 is an audit flag, allowed in the final table.
- Already-distributed 01/02 handout repositories stay unchanged this term unless the instructor revises that.

## Source layout and working paths

- `NN/README.md`: lecture; `NN/BONUS.md`: optional material. `demo/DEMO_GUIDE.md` in 01 to 03: script walkthroughs. `demo/*.md` in 04 to 11: notebook sources; generate the paired `.ipynb` with Jupytext. Each local demo folder ships a uv project, lock, and `setup_demo.sh`.
- `NN/assignment/` and `NN/assignment_checks/` (06: `06/mid-term/` and `06/mid-term_checks/`); `05/mid-term/` and `06/assignment/` are inactive. `assignments.json` names each handout folder. Homework checkers are byte-identical in both locations; change both together and judge by emitted JSON.
- Older forks may run older workflows; do not assume every fork fetches current checks.
- TA grading: `uv run scripts/grade_submissions.py NN`, run from the ignored `grading/` folder. Live fork collection has not been exercised.
- Use ignored `scratch/` or `tmp/` for trials. Build each verification env fresh in `scratch/` with uv. `work/tufte/` holds local reference PDFs, not for redistribution.

## Still unverified

Classroom timing, native macOS and Windows workflows, the real course server (SSH, scp, tunnels), student-fork Actions runs, live TA fork collection, and human exam judgments. Hosted Colab validation is intentionally out of scope, not an outstanding task.

## Publishing

- Handouts: fetch and reconcile the target first, then `scripts/publish_assignment.sh NN UCSF-DataSci/ds217-26f-NN`. It mirrors the handout and excludes `_grader_selftest/`. Do not infer a handout release from a course push.
- Notion: each publish needs a request and fresh reconciliation of live content, child pages, links, images, and synced blocks. Use `python3 scripts/notion_push.py <page.md>` with `TMPDIR` set to a repo scratch folder, and verify with `python3 scripts/notion_check_pages.py`. Pace bulk work; Notion rate-limits concurrent calls. Commit, push, and publish changed lecture pages together.
