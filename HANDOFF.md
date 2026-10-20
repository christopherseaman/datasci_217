# Handoff

Updated 2026-10-20. Read [AGENTS.md](AGENTS.md) for the authoring and verification rules. This brief holds the current state, standing decisions, and open items; Git history and `work/reviews/` hold completed work.

## Overall goal

Prepare a coherent, practical introduction to Python and data-science tools for health data science master's students who are beginners in programming. Lectures explain concepts in order, with visuals, useful reference cards, minimal method examples, and relevant humor. Demos are reproducible independently after a missed class. Assignments are doable using preceding lectures and demos, with well-defined student-created artifacts and helpful, generous checks.

Each ordinary class has 90 minutes: about 60 of lecture and 30 of core demo walkthroughs. Lecture 11 is the project-workflow exception. Successful execution and shorter files do not establish a 90-minute classroom fit.

## Repo and release state

- As of October 20, `main` holds all work, and all mapped lecture and BONUS pages for 01 to 11 match it in Notion. Check `git status` and `git log` on arrival.
- Handout repositories `UCSF-DataSci/ds217-26f-04` through `-11` match their course folders as of their last publish: 05, 08, and 10 on October 19; 06 on October 20; the rest October 2 or 14. Pushing the course updates the checks that homework forks download from `main`, never the handouts themselves.
- Assignment 05 is live with 8 forks. On October 19 its handout renamed the row-count artifact to `output/row_counts.csv` with the index `step`; earlier forks save `output/decision_log.csv` with the index `decision`. The checks accept both until the November cleanup in [TODO.md](TODO.md).
- Assignments 06 (the midterm), 08, and 10 had no forks on October 20.

## Recent work, October 19 to 20

- Every lecture opens with `# Overview`: what it covers, its readings, and 3 to 5 practical skills taught in the main lecture or demos. AGENTS.md defines the shape. Lecture 01's Quick References now sit inside the Overview as a `##` section.
- Assignments 08 and 10: README steps and notebook TODOs name methods and arguments without paste-ready answer expressions. Grading is unchanged.
- Lecture 06 and the midterm had an adversarial review (six lenses, each claim checked by a skeptic). 7 of 11 claims were confirmed and fixed:
    - The grader charged one cleaning-rule mistake twice (in `issue_audit.csv` and a summary). `06/mid-term_checks/grading.py` now accepts summary values that follow from the student's own audit counts, so each such mistake costs 1 point; the self-test has 33 single-mistake cases.
    - `generate_data.py` moved all weekend week-12 follow-ups to Monday; follow-ups now count clinic days, about 5,500 per weekday. Fixture v7; data hashes and one count (date parse failures 169 to 170) changed.
    - HINTS.md credits Lecture 05 for `duplicated(keep=False)` and `isna().all()`, and names `one_to_one` for the A1c join; Task 2.2 copies both visit tables; Lecture 06 defines aggregation in bold at first use.
    - Refuted claims: integer values in the concat diagram, the `%%bash` file count, comic spacing, and follow-up noise size.
- The repo was cleaned of caches and build outputs (`__pycache__`, `.checks/`, demo `output/`, `_site/`, `dist/`).

## Coming up

- A further review of Lecture 06, its three demos, and the midterm before the midterm opens. The October 20 demo reviewer reported no defects but recorded no execution evidence, so rerun all three notebooks fresh. Untested: a student-style completion of the midterm from the handout alone after the v7 data change, the HINTS wording against the new grader, and the 25 human-review points.
- November: remove the Assignment 05 legacy file name (TODO.md).
- Republish each handout only after a fresh reconcile, and only when the instructor asks.

## Working agreements

- Quick, well-specified edits go to a light model with exact instructions, followed by one check (lint or the single relevant test). Add review or completion passes only when the instructor asks.
- Commit, push, and publish changed lecture pages together; never leave Notion behind `main`.
- Subagent prompts must quote the instructor's request: agents see only their prompt, and they refused once when they lacked it.

## Decisions to preserve

- Keep non-repeated Git guidance, including website upload, and some duck typing in 01; keep Markdown as required communication material and non-repeated Git in 02. Students struggle with Git. Lecture 02's debugger stays an optional BONUS exercise; 02 keeps its Command-Line Catalog. VS Code commit and sync screenshots stay in 01 and 02, with Microsoft's merge-conflict link. Lecture 01 keeps its Quick references.
- In 03, mention venv/Conda in main and keep their workflows in BONUS. 2-D boolean indexing (steps, mask diagram, card, IndexError), the instructor's Examples of Wildcards table, and 2-D sorting (`np.sort(axis=)`, `argsort`, ordering rows) stay in main; 3-D and deeper selection stays in BONUS. The instructor accepted 03's length; do not reopen it for exceeding a soft size target.
- Keep Parquet in 04 and all cleaning functionality in 05. The 04 demo project pins pyarrow for real Parquet round trips.
- Keep Lecture 06's join-type comparison and cross-join audit; building both shapes is mentioned in the lecture and shown in Demo 2. Keep the accepted 06 compaction and only genuine repetition cleanup in 07.
- Keep SSH/tmux/Jupyter in 08. Lecture 08 keeps filter thresholds, within-group apply (sorting, `nlargest`), the MultiIndex result, and the `.values` length error. The real course server remains unverified; the terminal exercise rehearses locally.
- Lecture 09's blocks are dated data, then resampling/grids/windows, then clocks and honest entity history. Its Python `datetime` section stays in main. Frequency inference, specialized schedules, clock-time alternatives, percentage changes, and Grouper are BONUS.
- Keep XGBoost and Keras in the 10 lecture and demo, plus its Zoolander hook and the lab-analyzer, practice-exam, and scrambled-chart analogies. Lecture 11's project lecture and four notebooks stay at their pre-pacing versions. Discuss removing functionality with the instructor; shorten examples and remove repetition instead.
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
- Use ignored `scratch/` or `tmp/` for trials. `scratch/` was emptied on October 20; build each verification env fresh there with uv. `work/tufte/` holds local reference PDFs, not for redistribution.

## Still unverified

Classroom timing, hosted Colab runtime and UI, native macOS and Windows workflows, the real course server (SSH, scp, tunnels), student-fork Actions runs, live TA fork collection, and human exam judgments.

## Publishing

- Handouts: fetch and reconcile the target first, then `scripts/publish_assignment.sh NN UCSF-DataSci/ds217-26f-NN`. It mirrors the handout and excludes `_grader_selftest/`. Do not infer a handout release from a course push.
- Notion: each publish needs a request and fresh reconciliation of live content, child pages, links, images, and synced blocks. Use `python3 scripts/notion_push.py <page.md>` with `TMPDIR` set to a repo scratch folder, and verify with `python3 scripts/notion_check_pages.py`. Pace bulk work; Notion rate-limits concurrent calls. Commit, push, and publish changed lecture pages together.
