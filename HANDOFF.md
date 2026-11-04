# Handoff

Updated 2026-11-04. Rules and settled decisions live in [AGENTS.md](AGENTS.md) (CLAUDE.md links to it); open work lives in [TODO.md](TODO.md). Nothing is mid-flight.

## State

- `main` holds all work. Notion lecture and BONUS pages for 08, 09, and 10 match it as of November 4; the page audit (`python3 scripts/notion_check_pages.py`) reports 0 problems.
- Handout repos: `ds217-26f-08` and `-09` were republished on November 4; `-10` last on October 30. No forks of 08 to 10 existed when their data and checks changed.
- Lectures 08 to 10 were reworked October 29 to November 4: causal-graph demo generators (`dowhy.gcm`), one stakeholder project per lecture's demos, hosted Parquet assignment data, 07-style handouts, and adversarial reviews of 08. Lecture 08 has four demos (Demo 0 is the EDA mini demo) and no demo terminal steps.
- Sizes (`python3 scripts/lecture_metrics.py`): 08 about 650 lines, 09 about 850, 10 about 875.

## Waiting on the instructor

- A course server for SSH practice; Demo 08.3's terminal steps return then (TODO.md).
