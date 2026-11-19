# Handoff

Updated 2026-11-19. Rules and settled decisions live in [AGENTS.md](AGENTS.md) (CLAUDE.md links to it); open work lives in [TODO.md](TODO.md). One task is open: the Lecture 09 arc, below.

## State

- `main` holds all work. Notion lecture and BONUS pages for 08, 09, and 10 match it as of November 4; the page audit (`python3 scripts/notion_check_pages.py`) reports 0 problems.
- Handout repos: `ds217-26f-08` and `-09` were republished on November 4; `-10` last on October 30. No forks of 08 to 10 existed when their data and checks changed.
- Lectures 08 to 10 were reworked October 29 to November 4: causal-graph demo generators (`dowhy.gcm`), one stakeholder project per lecture's demos, hosted Parquet assignment data, 07-style handouts, and adversarial reviews of 08. Lecture 08 has four demos (Demo 0 is the EDA mini demo) and no demo terminal steps.
- Sizes (`python3 scripts/lecture_metrics.py`): 08 about 650 lines, 09 about 850, 10 about 875.

## Waiting on the instructor

- A course server for SSH practice; Demo 08.3's terminal steps return then (TODO.md).

## Open: Lecture 09 sections do not build on each other

The instructor's words, 2026-11-18 and 19:

- "It feels like a collection of topics rather than a broad introduction to the topic of time series."
- "It feels like there is no arc connecting the methods."
- "Don't want to add lots of explanations and use cases."
- "Root cause is almost right: the sections don't build on each other. The new table for each section is just a symptom."
- The same problem "probably" holds in other lectures.

### Root cause

Each section stands alone. No section starts from what the section before it produced, and no later section needs an earlier one's result. A fresh toy table per section is a symptom of that. So are the glossary openings and the method-named headings. Fixing a symptom alone (one shared table, renamed headings) leaves the sections independent.

### Evidence in `09/README.md` at commit `22355e9`

- 0 of 7 topic boundaries use the previous topic's result. 8 of 8 `#` topics open on a bold definition of their own new term.
- 18 of 30 `#` and `##` headings name a pandas method. Four share one template: "Basic Resampling", "Basic Rolling Operations", "Basic Time Series Plots", "Basic Time Zone Operations".
- No snippet resamples and then rolls. The only place one method's output feeds another is the `daily_hr_rolling.png` caption.
- 16 snippets use 16 tables. About 30 datasets appear across 45 figures and snippets; 4 recur. Three tables are named `vitals`. In 7 sections the figure and the snippet under one heading use different data.
- The hourly monitor (January 1 to 28, 2024, outage January 5, fever January 15 to 17) is in 4 figures and 0 snippets.
- Gaps are taught in 8 places under different names (missing bins, `asfreq`, grid flags, gap runs, EDA spacing, line breaks). Past-only reading is warned about 3 times before its topic defines it.
- Three jobs are interleaved and never named: making time trustworthy (parse, sort, grid, gaps, zones), describing the series (bins, rolling, plots), building past-only features.
- Trend, seasonality and noise are defined in topic 1 and unused for 420 lines.
- Full tables, a 19-edit plan and its refutation: `scratch/l09_arc_diagnosis.json` (ignored by Git; keys `result.diags`, `result.plan`, `result.crit`).

### Tried and rejected

- Commit `22355e9` added `## Trend, Seasonality, and Noise` and `### Reference Card: Time Series Questions` to topic 1, moved calendar selection into "Date and Time Data Types", and put frequencies before resampling. The instructor: these are additions that do not address the cause.
- The questions card stays only if changes, not additions, in the later sections tie back to it. Otherwise delete it. It currently lists the topics in file order, which the style guide's no-outline rule bans.
- A plan to carry three series through the lecture (`monitor`, one P1/P2 panel, the illness-visit year) was called a symptom fix. It may still be part of the answer once sections build on each other.
- The centered-window row and the dashed line in `daily_hr_rolling.png` from `22355e9` were requested. Keep them.

### Constraints for the fix

- Changes, not additions: no new explanations, use cases, transitions, recaps or narration. Prose stays near 3,000 words (now 3,344 by `python3 scripts/lecture_metrics.py`).
- Keep every method, the EDA section, and the three demo breaks: all resampling and gap runs, then rolling and visualization, then time zones and past-only features (AGENTS.md, Decisions).
- Demos and Assignment 09 match the lecture now. A change to what the lecture teaches moves the demo content with it.
- Snippet outputs must be real output on pandas 3.0.5. Figures come from `09/scripts/generate_lecture_figures.py`, which asserts its drawn numbers.
- State the root cause and the proposed change to the instructor before editing. Two rounds of unrequested edits were rejected.
- Commit, push and publish together: `python3 scripts/notion_push.py 09/README.md`. Notion matches `22355e9`.

### Other lectures

Not assessed. The same test applies to each: at every topic boundary, does the next section start from the previous section's result, and does anything later use it? Lecture 08 names one table, `visits`, near `08/README.md` line 184; a diagnosis agent cited it as the contrast case.

