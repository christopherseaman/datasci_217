# Lecture 05 Demo Guide: Data Cleaning and Preparation

Lecture 05 has three executable demonstrations. Markdown is authoritative and
each checked-in notebook is generated from its paired Markdown source with
Jupytext. The lecture explains the decisions; the demos exercise them
top-to-bottom in a fresh kernel.

## Tested environment

Use CPython 3.12.13 with NumPy 2.0.2, pandas 3.0.3, JupyterLab 4.4.10, and
Jupytext 1.18.1, as recorded in `requirements.txt`:

```bash
uv venv --python 3.12.13 .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
jupyter lab
```

To regenerate a notebook after editing Markdown, write a disposable output and
execute that copy; never overwrite the source notebook with outputs:

```bash
jupytext --to notebook --output demo2_transformations.generated.ipynb demo2_transformations.md
jupyter nbconvert --to notebook --execute \
  --output demo2_transformations.executed.ipynb \
  demo2_transformations.generated.ipynb
```

## Demo 1 — Missing data detective work

**Sources:** `demo1_missing_data.md` → `demo1_missing_data.ipynb`

Create a small patient table, quantify missingness, inspect a row-level
summary, and apply context-specific strategies: median imputation for age,
forward fill for ordered test dates, and a documented drop rule for rows
missing both critical measurements. The tabular summary is the core activity.
An optional heatmap preview is included immediately afterward for visual
learners; it previews the formal visualization workflow in Lecture 07 without
turning chart interpretation into a Lecture 05 prerequisite.

## Demo 2 — Transformation and cleaning pipeline

**Sources:** `demo2_transformations.md` → `demo2_transformations.ipynb`

Clean a survey table by standardizing column names and text, converting bad
numeric values with an explicit failure policy, replacing sentinels, and
creating categories, dummy variables, and categorical dtypes. The fixture has
tied income values, so `qcut(..., duplicates='drop')` is used without assuming
a fixed number of labels.

## Demo 3 — Complete data-cleaning workflow

**Sources:** `demo3_workflow.md` → `demo3_workflow.ipynb`

Run the end-to-end contract-driven sequence: detect issues, transform a copy,
validate the result, add analysis fields, identify outlier candidates, and
save cleaned data, summaries, and a report under a disposable `output/`
directory. This is the practical synthesis at the end of the lecture.

## Instructor checklist

- Run each notebook from a fresh kernel in order: missingness → transformations
  → complete workflow.
- Keep the original table separate from the working copy and explain why each
  cleaning decision is appropriate.
- Treat the exact pins in `requirements.txt` as the activity contract.
- Keep notebook automation (batch execution, failure handling, and output
  policies) with Lecture 04's optional Jupyter material; it is not repeated
  here.
- Clear outputs before committing; repository notebooks are output-free.
