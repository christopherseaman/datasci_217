# TODO

- November 2026, after Assignment 05 is long past due: remove the `output/decision_log.csv` fallback from the 05 checks (`LEGACY_DECISION_FILE` in both copies of `_value_checks.py`) and the self-test's `legacy-decision-log-name` case, keeping only `output/row_counts.csv`.
- After this quarter, Lectures 01 to 07 (from the 2026-11 Unlearn interview feedback; Lecture 11's "A Step-by-Step EDA Pass" teaches both for the current cohort, so trim it to a short reference once these land):
    - 04: add a section on the steps of EDA (question and grain, structure with `info`/`describe`, missing values overall and by group, distributions and outliers, relationships), with a Demo 04 step.
    - 07: add pairwise plots (`sns.pairplot`) after "Watch the Grain", moving its row out of BONUS, with a core step in Demo 07.2 showing that r = 0.3 can hide a curve.
