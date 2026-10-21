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
    version: 3.13
---

# Demo 3: Stacking Monthly Hospital Extracts with `concat()` and `combine_first()`

- A hospital quality team receives admissions one month at a time, plus monthly unit metrics from separate systems.
- Stack the monthly files and record where each row came from.
- Handle a March file whose columns changed: keep every column, or only the shared ones.
- Line up two systems by month, keeping every month or only the months both report.
- Fill missing census values from a backup source.

Run the cells from top to bottom; each **Expect** line says what you should see.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0 matplotlib==3.11.1 seaborn==0.13.2
```

## Core walkthrough

### 1. Load two monthly admission extracts

This supplied cell keeps any Parquet file already in `data/` and downloads the rest (`urlretrieve(url, path)` saves the file at a web address to `path`; `data_dir.glob("demo3_*.parquet")` lists the matching files).

```python
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/06/demo/data"
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
for filename in ["demo3_jan_admissions.parquet", "demo3_feb_admissions.parquet",
                 "demo3_mar_admissions.parquet", "demo3_census.parquet", "demo3_audits.parquet",
                 "demo3_adt_census.parquet", "demo3_finance_census.parquet",
                 "demo3_staffing.parquet", "demo3_experience.parquet", "demo3_units.parquet",
                 "demo3_q1_2025.parquet", "demo3_q1_2026.parquet"]:
    path = data_dir / filename
    if not path.exists():
        urlretrieve(f"{REPO_RAW}/{filename}", path)
print(len(list(data_dir.glob("demo3_*.parquet"))))
```

**Expect:** `12`, the number of `demo3_*.parquet` files in `data/`.

Each admission file has one row per admission; these are 3-row samples of the real monthly extracts. `los_days` is the length of stay in days.

```python
jan = pd.read_parquet(data_dir / "demo3_jan_admissions.parquet")
feb = pd.read_parquet(data_dir / "demo3_feb_admissions.parquet")
display(jan)
display(feb)
print(list(jan.columns) == list(feb.columns))
```

**Expect:** two 3-row tables with columns `admission_id`, `unit`, `age`, `diagnosis`, `los_days`, then `True`: same columns, same row meaning, so these are pieces of one table.

### 2. Count each extract's lines with `%%bash`

The monthly extracts usually arrive as CSV files. Save the two samples to `data/` without the row index, then count their lines from the shell. A cell whose first line is `%%bash` runs the whole cell in the shell, and `wc -l` counts the header line too.

```python
jan.to_csv(data_dir / "jan_admissions.csv", index=False)
feb.to_csv(data_dir / "feb_admissions.csv", index=False)
```

```python
%%bash
wc -l data/*_admissions.csv
```

**Expect:** `4 data/feb_admissions.csv`, `4 data/jan_admissions.csv`, and `8 total`: each file has 1 header line plus 3 admissions, so a file with N lines holds N - 1 rows.

### 3. Stack rows with `concat()`

```python
stacked = pd.concat([jan, feb])
print(list(stacked.index))
display(stacked)
```

**Expect:** 6 rows, and the index `[0, 1, 2, 0, 1, 2]`: each file brought its own row labels, so they repeat.

```python
stacked = pd.concat([jan, feb], ignore_index=True)
print(list(stacked.index))
```

**Expect:** `[0, 1, 2, 3, 4, 5]`. Use `ignore_index=True` when the old row numbers carry no information, as here.

### 4. Record where each row came from

Once stacked, nothing says which file a row came from. Add a label column to each piece **before** stacking; it is an ordinary column, so it survives saving to CSV and later merges.

```python
jan['source_file'] = 'demo3_jan_admissions'
feb['source_file'] = 'demo3_feb_admissions'

stacked = pd.concat([jan, feb], ignore_index=True)
display(stacked['source_file'].value_counts())
display(stacked)
```

**Expect:** `demo3_jan_admissions` 3 and `demo3_feb_admissions` 3, and a `source_file` column as the last column of the 6-row table.

### 5. March changed its columns

The March extract comes from an updated report: it no longer includes `age`, and it adds `payer`.

```python
mar = pd.read_parquet(data_dir / "demo3_mar_admissions.parquet")
mar['source_file'] = 'demo3_mar_admissions'
display(mar)
```

**Expect:** 3 March admissions with columns `admission_id`, `unit`, `diagnosis`, `los_days`, `payer`, `source_file`: no `age`, and a new `payer`.

`concat()` matches columns by **name**. With the default `join='outer'` it keeps every column from every file.

```python
admissions = pd.concat([jan, feb, mar], ignore_index=True)
print(admissions.shape)
display(admissions.isna().sum())
display(admissions)
```

**Expect:** `(9, 7)` and missing counts of `age` 3 and `payer` 6, everything else 0. The 3 March rows have no `age`, and the 6 January and February rows have no `payer`. These missing values come from the files' layouts, not from missing measurements, so note them rather than filling them.

`join='inner'` keeps only the columns every file shares.

```python
shared_only = pd.concat([jan, feb, mar], join='inner', ignore_index=True)
print(shared_only.shape)
print(list(shared_only.columns))
```

**Expect:** `(9, 5)` and `['admission_id', 'unit', 'diagnosis', 'los_days', 'source_file']`. `age` and `payer` are gone without any warning, which is why the outer result is the one to keep.

### 6. Side by side: line up two systems by month

ADT (the admission-discharge-transfer feed) reports a census every month. Infection control audits hand hygiene only in some months, and its file already includes July. Put `month` in the index of each so that `concat(axis=1)` lines rows up by label.

```python
census = pd.read_parquet(data_dir / "demo3_census.parquet").set_index('month')
audits = pd.read_parquet(data_dir / "demo3_audits.parquet").set_index('month')
display(census)
display(audits)
```

**Expect:** 6 census months (`2026-01` to `2026-06`) with `admissions` and `bed_days`, and 4 audit months (`2026-02`, `2026-03`, `2026-04`, `2026-07`) with `hand_hygiene_pct`.

```python
with_audits = pd.concat([census, audits], axis=1)
print(with_audits.shape)
display(with_audits)
```

**Expect:** `(7, 3)`. January, May, and June have `NaN` hand hygiene (no audit), and the new July row has `NaN` `admissions` and `bed_days`. The default `join='outer'` keeps every label from every input, and the `NaN` cells turn the counts into `float64` (`212.0`).

This chart is a preview: run the cell and read the picture; the plotting code is taught later. Light cells are the `NaN` values from the outer join.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(with_audits.isna(), cbar=False).set_title("Missing values after concat(axis=1)")
plt.show()
```

**Expect:** a 7-row by 3-column grid (`2026-01` to `2026-07`). `hand_hygiene_pct` is light for `2026-01`, `2026-05`, and `2026-06`; `admissions` and `bed_days` are light only for `2026-07`; every other cell is dark.

```python
audited_months = pd.concat([census, audits], axis=1, join='inner')
display(audited_months)
```

**Expect:** 3 rows, `2026-02`, `2026-03`, and `2026-04`: only the months present in both inputs, with no `NaN`.

### 7. Fill missing values with `combine_first()`

The live ADT export failed to report February and April and has not reported June yet. Finance keeps its own estimate for every month.

```python
primary = pd.read_parquet(data_dir / "demo3_adt_census.parquet").set_index('month')
finance = pd.read_parquet(data_dir / "demo3_finance_census.parquet").set_index('month')
display(primary)
display(finance)
```

**Expect:** 5 ADT months with `NaN` in both columns for February and April, and 6 finance months with no missing values.

```python
filled = primary.combine_first(finance)
display(filled)
```

**Expect:** 6 months. January, March, and May keep the ADT values (212, 225, 231 admissions), February and April take finance's estimates (200 and 238), and June comes only from finance (219). Values print as `212.0` because the missing values made the columns `float64`.

This chart is a preview: run the cell and read the picture; the plotting code is taught later.

```python
trend = pd.concat({'ADT only': primary['admissions'], 'after combine_first()': filled['admissions']},
                  names=['version']).reset_index()
sns.lineplot(data=trend, x='month', y='admissions', hue='version', marker='o').set_title("Monthly admissions")
plt.show()
```

**Expect:** `ADT only` has points at `2026-01`, `2026-03`, and `2026-05` only (212, 225, 231), joined by a straight line that skips February and April. `after combine_first()` has all 6 months, adding February (200), April (238), and June (219).

Compare `concat(axis=1)`, which puts the two sources side by side instead of filling one from the other:

```python
print(list(pd.concat([primary, finance], axis=1).columns))
```

**Expect:** `['admissions', 'bed_days', 'admissions', 'bed_days']`: duplicate column names and no filled values. For filling missing values, `combine_first()` is the right tool.

Record which source each month's numbers came from. Start by marking every month as an estimate, then relabel the months where ADT actually reported.

```python
filled['data_source'] = 'finance_estimate'
reported = primary[primary['admissions'].notna()].index
filled.loc[reported, 'data_source'] = 'adt_primary'
print(list(reported))
display(filled)
```

**Expect:** `['2026-01', '2026-03', '2026-05']`, and `data_source` reads `adt_primary` for those three months and `finance_estimate` for February, April, and June.

To keep the month labels in a saved file, turn them back into a column.

```python
filled_flat = filled.reset_index()
print(list(filled_flat.columns))
```

**Expect:** `['month', 'admissions', 'bed_days', 'data_source']`.

## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run the cells above again first.

### 8. `concat()` then `merge()`: add unit details

Stacking made one admissions table. Unit names and bed counts live in a separate lookup table, so they are joined by key with `merge()`. Each admission should find exactly one unit: many-to-one.

```python
units = pd.read_parquet(data_dir / "demo3_units.parquet")
display(units)

enriched = pd.merge(admissions, units, left_on='unit', right_on='unit_code',
                    how='left', validate='many_to_one', indicator=True)
print(len(enriched))
display(enriched['_merge'].value_counts())
```

**Expect:** the 3 units (ICU 12 beds, MED 30, SURG 24), then `9` rows (no admission gained or lost) with `both` 9, `left_only` 0, and `right_only` 0.

Flag stays of a week or longer and list the longest.

```python
enriched['long_stay'] = enriched['los_days'] >= 7
print(enriched['long_stay'].sum())
display(enriched.sort_values('los_days', ascending=False)[
    ['admission_id', 'unit_name', 'diagnosis', 'los_days', 'long_stay', 'source_file']
].head(3))
```

**Expect:** `2` long stays. The top three are A202 (Intensive Care, sepsis, 9 days, February), A303 (Intensive Care, stroke, 7 days, March), and A103 (Surgery, appendicitis, 6 days, January, not flagged).

`concat()` stacks pieces of one table; `merge()` joins different tables by key.

### 9. Three systems side by side

Staffing and patient experience each report one row per month, for the same 6 months as the census.

```python
staffing = pd.read_parquet(data_dir / "demo3_staffing.parquet").set_index('month')
experience = pd.read_parquet(data_dir / "demo3_experience.parquet").set_index('month')
display(staffing)
display(experience)
print(census.index.is_unique, staffing.index.is_unique, experience.index.is_unique)
```

**Expect:** 6 months of `nurse_hours`, 6 months of `top_box_pct` (the % of survey answers in the top category) and `responses`, then `True True True`: each month appears once in each system.

```python
monthly = pd.concat([census, staffing, experience], axis=1)
monthly['nurse_hours_per_bed_day'] = (monthly['nurse_hours'] / monthly['bed_days']).round(2)
print(monthly.shape)
display(monthly)
```

**Expect:** `(6, 6)`: one row per month with columns from all three systems plus the new ratio, and no `NaN` because every label matches. Nurse hours per bed day range from 7.71 in April, the busiest month, to 8.27 in February.

If the identifiers were in ordinary columns rather than the index, this would be a job for `merge()` on those columns.

### 10. Year over year: stack, then pivot

Compare the first quarter's admissions with the same months last year. Stack the two years with a `year` label column, then pivot to put the years side by side.

```python
q1_2025 = pd.read_parquet(data_dir / "demo3_q1_2025.parquet")
q1_2026 = pd.read_parquet(data_dir / "demo3_q1_2026.parquet")
q1_2025['year'] = 2025
q1_2026['year'] = 2026

both_years = pd.concat([q1_2025, q1_2026], ignore_index=True)
display(both_years)
```

**Expect:** 6 rows: `Jan`, `Feb`, `Mar` for 2025 (198, 185, 207 admissions), then the same months for 2026 (212, 198, 225).

```python
yoy = both_years.pivot(index='month', columns='year', values='admissions')
print(list(yoy.index))
yoy = yoy.loc[['Jan', 'Feb', 'Mar']]           # calendar order
yoy['growth_pct'] = ((yoy[2026] - yoy[2025]) / yoy[2025] * 100).round(1)
display(yoy)
```

**Expect:** `['Feb', 'Jan', 'Mar']` first, because `pivot()` sorts the labels alphabetically; after `.loc` the rows are in calendar order. Growth is 7.1% in January, 7.0% in February, and 8.7% in March, so admissions rose every month and March grew the most.

### 11. Source labels as index levels

`keys=` is the alternative: it labels each piece with an outer index level instead of a column. Here the outer label repeats what `source_file` already says, so you can compare the two.

```python
by_file = pd.concat([jan, feb], keys=['jan', 'feb'], names=['source', 'row'])
flat = by_file.reset_index()
display(flat[flat['source'] == 'feb'])
```

**Expect:** the 3 February admissions (A201, A202, A203), with `source` = `feb` and `row` = 0 to 2 as ordinary columns.

```python
print(list(by_file.reset_index().columns))
```

**Expect:** `['source', 'row', 'admission_id', 'unit', 'age', 'diagnosis', 'los_days', 'source_file']`: `reset_index()` turns both label levels into ordinary columns. The course uses the `source_file` column form, because it stays a plain column from the start.
