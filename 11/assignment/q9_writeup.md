---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Q9: Writeup

**15 points, human review**

Complete the root [`report.md`](report.md) using facts from your saved files. Concise, factual writing is welcome, and the model does not need to beat persistence. The [README's completion contract](README.md#completion-contract) says which report sections each review category reads and what earns full credit:

| Category | Points |
| --- | ---: |
| Justified cleaning and forecast decisions | 5 |
| Interpretation tied to evidence | 5 |
| Limitations and clear communication | 5 |

## 9.1 Required Structure

Keep exactly these level-two headings, in order:

1. Executive Summary
2. Data and Cleaning
3. Patterns
4. Forecast Design
5. Model Results
6. Limitations

Keep the six-column metrics table, with columns `Evaluation set`, `Model`, `MAE`, `RMSE`, `R2`, and `n`, and fill its four rows: the two rows of `q7_validation_metrics.csv` labeled Validation, then the two rows of `q8_test_metrics.csv` labeled Test. Keep all three image embeds:

- `![Release exploration](output/q1_visualizations.png)`
- `![Training patterns](output/q5_patterns.png)`
- `![Final model results](output/q8_final_visualizations.png)`

## 9.2 Your Results

Run this cell to see the numbers your report quotes, then copy them from here rather than recalculating them.

```python
import pandas as pd

release_audit = pd.read_csv("output/q1_release_audit.csv")
cleaning_audit = pd.read_csv("output/q2_cleaning_audit.csv")
monthly_summary = pd.read_csv("output/q5_monthly_station_summary.csv")
validation_metrics = pd.read_csv("output/q7_validation_metrics.csv")
test_metrics = pd.read_csv("output/q8_test_metrics.csv")
station_metrics = pd.read_csv("output/q8_station_metrics.csv")

display(release_audit)
display(cleaning_audit)
display(monthly_summary.head(12))
display(validation_metrics)
display(test_metrics)
display(station_metrics)
```

> **Checkpoint: `report.md`**

## Check Your Work

- [ ] The six headings appear exactly and in order.
- [ ] No bracketed placeholder from the scaffold remains.
- [ ] The four table rows match `q7_validation_metrics.csv` and `q8_test_metrics.csv`.
- [ ] All three images show when you preview `report.md` in VS Code.
- [ ] The limitations are specific to this release and this evaluation design.
- [ ] Every notebook was saved after **Run All**, so its outputs show.
