# Lecture 05 Demo Guide: Data Cleaning and Preparation

Lecture 05 has three executable demonstrations. Markdown is authoritative and
each checked-in notebook is generated from its paired Markdown source with
Jupytext. The lecture explains the decisions; the demos exercise them
top-to-bottom in a fresh kernel.

## Tested environment

Use CPython 3.12.13 with NumPy 2.0.2, pandas 3.0.5, JupyterLab 4.4.10, and
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
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/05/demo/demo1_missing_data.ipynb)

Create a small patient table, quantify missingness, inspect a row-level
summary, and apply context-specific strategies: median imputation for age,
explicit date parsing and inspection, and a documented drop rule for rows
missing both critical measurements. Because the fixture documents neither
chronological row order nor entity boundaries, it deliberately does not fill
missing dates. An optional heatmap preview shows the same missingness mask and
is explicitly a Lecture 07 visualization preview, not a cleaning decision.

## Demo 2 — Transformation and cleaning pipeline

**Sources:** `demo2_transformations.md` → `demo2_transformations.ipynb`
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/05/demo/demo2_transformations.ipynb)

Clean a survey table by standardizing column names and text, converting bad
numeric values with an explicit failure policy, replacing sentinels, and
creating categories, dummy variables, and categorical dtypes. The demo briefly
contrasts range-based `cut` with quantile-based `qcut`; tied income values mean
`qcut(..., duplicates='drop')` is used without assuming a fixed number of
labels.

## Demo 3 — Complete data-cleaning workflow

**Sources:** `demo3_workflow.md` → `demo3_workflow.ipynb`
**Colab:** [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/05/demo/demo3_workflow.ipynb)

Run the end-to-end contract-driven sequence: detect issues, transform a copy,
validate the result, add row-level analysis fields, identify outlier
candidates, and save cleaned data and a report under a disposable `output/`
directory. Grouped summaries and temporal feature engineering are deferred to
later lectures.

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
