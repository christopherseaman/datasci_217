# Handoff

Updated 2026-10-19. Read [AGENTS.md](AGENTS.md) for the authoring and verification rules. This brief holds the current state, standing decisions, and open items; Git history and `work/reviews/` hold completed work.

## Overall goal

Prepare a coherent, practical introduction to Python and data-science tools for health data science master's students who are beginners in programming. Lectures explain concepts in order, with visuals, useful reference cards, minimal method examples, and relevant humor. Demos are reproducible independently after a missed class. Assignments are doable using preceding lectures and demos, with well-defined student-created artifacts and helpful, generous checks.

Each ordinary class has 90 minutes: about 60 of lecture and 30 of core demo walkthroughs. Lecture 11 is the project-workflow exception. Successful execution and shorter files do not establish a 90-minute classroom fit.

## Repo and release state

- As of October 16, `main` holds all reviewed work through [`49602ac`](https://github.com/christopherseaman/datasci_217/commit/49602ac), and all 16 mapped lecture and BONUS pages for 04 to 11 match it in Notion. Check `git status` and `git log` on arrival.
- Handout repositories `UCSF-DataSci/ds217-26f-04` through `-11` received the October 2 release; 05 and 06 were updated on October 14. Pushing the course updates the checks that forks download from `main`, never the handouts themselves.
- Assignment 05 is live with student forks. On October 19 its handout renamed the row-count artifact to `output/row_counts.csv` with the index `step`; forks made earlier save `output/decision_log.csv` with the index `decision`. The checks accept both until the November cleanup in [TODO.md](TODO.md).
- Assignments 08 and 10 had no forks on October 19. Their scaffolds and README steps now name methods and arguments without paste-ready answers; republish each handout before it opens.

## Decisions to preserve

- Keep non-repeated Git guidance, including website upload, and some duck typing in 01; keep Markdown as required communication material and non-repeated Git in 02. Students struggle with Git. Lecture 02's debugger stays an optional BONUS exercise; 02 keeps its Command-Line Catalog. VS Code commit and sync screenshots stay in 01 and 02, with Microsoft's merge-conflict link. Lecture 01 keeps its Quick references opening.
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
- Use ignored `scratch/` or `tmp/` for trials. `scratch/py313-demo-env` is shared verification infrastructure; treat it as read-only. `work/tufte/` holds local reference PDFs, not for redistribution.

## Still unverified

Classroom timing, hosted Colab runtime and UI, native macOS and Windows workflows, the real course server (SSH, scp, tunnels), student-fork Actions runs, live TA fork collection, and human exam judgments.

## Publishing

- Handouts: fetch and reconcile the target first, then `scripts/publish_assignment.sh NN UCSF-DataSci/ds217-26f-NN`. It mirrors the handout and excludes `_grader_selftest/`. Do not infer a handout release from a course push.
- Notion: each publish needs a request and fresh reconciliation of live content, child pages, links, images, and synced blocks. Use `python3 scripts/notion_push.py <page.md>` with `TMPDIR` set to a repo scratch folder, and verify with `python3 scripts/notion_check_pages.py`. Pace bulk work; Notion rate-limits concurrent calls. Commit, push, and publish changed lecture pages together.
