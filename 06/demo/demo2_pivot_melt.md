---
jupyter:
  jupytext:
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    name: python
    version: 3.12.13
---

# Demo 2: Survey Data Reshaping with `pivot()` and `melt()`

## Learning Objectives

- Distinguish wide and long data formats
- Transform wide data to long format with `melt()`
- Transform long data to wide format with `pivot()`
- Recognize when each format suits a downstream task

## Setup

```python
import pandas as pd
```

## Create a Wide Survey

Each row is one employee and each question is a column.

```python
survey_wide = pd.DataFrame({
    'employee_id': ['E001', 'E002', 'E003', 'E004'],
    'department': ['Engineering', 'Sales', 'Engineering', 'Marketing'],
    'Q1_workload': [4, 5, 3, 4],
    'Q2_management': [5, 4, 4, 5],
    'Q3_compensation': [3, 4, 3, 4],
})
survey_wide
```

Wide data is convenient for reports and spreadsheets. Long data is often more
convenient when a later tool expects one observation per row.

## Wide → Long with `melt()`

`melt()` keeps identifier columns fixed and turns selected value columns into
name/value rows.

```python
survey_long = survey_wide.melt(
    id_vars=['employee_id', 'department'],
    value_vars=['Q1_workload', 'Q2_management', 'Q3_compensation'],
    var_name='question',
    value_name='rating',
)
survey_long
```

The three question columns became one `question` column and one `rating`
column. The same structural operation makes filtering and plotting by question
easier. Aggregation is intentionally deferred to Lecture 08.

## Long → Wide with `pivot()`

`pivot()` reverses the operation when each index/column pair identifies exactly
one value.

```python
survey_wide_again = survey_long.pivot(
    index=['employee_id', 'department'],
    columns='question',
    values='rating',
).reset_index()
survey_wide_again.columns.name = None
survey_wide_again
```

The temporary index has two levels because both `employee_id` and `department`
were selected as row labels. `reset_index()` makes those labels ordinary
columns again.

## Duplicate Keys: Deferred Topic

If an index/column pair repeats, `pivot()` cannot choose one value and raises an
error. `pivot_table()` can aggregate duplicate observations, but the choice of
aggregation changes the question being answered. Lecture 06 only flags this
boundary; the single bounded preview and the aggregation workflow belong to
[Lecture 08](../../08/README.md#pivot-tables-and-cross-tabulations).

## A Structural Workflow

```python
survey_analysis = survey_wide.melt(
    id_vars=['employee_id', 'department'],
    value_vars=['Q1_workload', 'Q2_management', 'Q3_compensation'],
    var_name='question', value_name='rating',
)

question_labels = {
    'Q1_workload': 'Workload Balance',
    'Q2_management': 'Management Support',
    'Q3_compensation': 'Compensation',
}
survey_analysis['question'] = survey_analysis['question'].map(question_labels)
survey_analysis.head()
```

Use wide format for presentation and long format for filtering, plotting, or a
later aggregation step. Reshaping changes the structure, not the observations.

## Key Takeaways

1. `melt()` transforms selected wide columns into long rows.
2. `pivot()` transforms long rows back when key pairs are unique.
3. Duplicate-key aggregation is a Lecture 08 topic; do not add a reporting
   summary here before choosing the aggregation question.
