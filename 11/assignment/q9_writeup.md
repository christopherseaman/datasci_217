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

**6 points**

Complete root [`report.md`](report.md) using facts from your saved artifacts. This is a structural completeness check, not a subjective prose score. Concise, factual writing is welcome, and the student model does not need to beat persistence.

## Required Structure

Use exactly these level-two headings, in order:

1. Executive Summary
2. Data and Cleaning
3. Patterns
4. Forecast Design
5. Model Results
6. Limitations

Include the accepted six-column Markdown table with columns `Evaluation set`, `Model`, `MAE`, `RMSE`, `R2`, and `n`. It must contain exactly four data rows: the two rows from `q7_validation_metrics.csv`, labeled Validation, followed by the two rows from `q8_test_metrics.csv`, labeled Test. Also include all three required image embeds:

- `![Release exploration](output/q1_visualizations.png)`
- `![Training patterns](output/q5_patterns.png)`
- `![Final model results](output/q8_final_visualizations.png)`

## Artifact Cross-Check

```python
from pathlib import Path

import pandas as pd

release_audit = pd.read_csv("output/q1_release_audit.csv")
cleaning_audit = pd.read_csv("output/q2_cleaning_audit.csv")
validation_metrics = pd.read_csv("output/q7_validation_metrics.csv")
test_metrics = pd.read_csv("output/q8_test_metrics.csv")
station_metrics = pd.read_csv("output/q8_station_metrics.csv")

display(release_audit)
display(cleaning_audit)
display(validation_metrics)
display(test_metrics)
display(station_metrics)
```

Use displayed artifact values rather than hand-recalculating results.

## Final Checklist

- [ ] The six required headings appear exactly and in order.
- [ ] No bracketed starter placeholder remains.
- [ ] The six-column, four-row metrics table agrees with the Q7 and Q8 artifacts.
- [ ] All three required figure paths exist and render.
- [ ] Limitations are tied to this dataset and evaluation design.
- [ ] Submitted notebook outputs are cleared after the final end-to-end run.
