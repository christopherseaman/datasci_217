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

- A hospital quality team receives admissions one month at a time, plus monthly unit metrics from three separate systems.
- Stack the monthly files and record where each row came from.
- Handle a March file whose columns changed.
- Line up the systems' metrics by month, and patch a gappy census from a backup source.
- Compare this year's first quarter with last year's.

Run the cells from top to bottom; each **Expect** line says what you should see.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

## Core walkthrough

### 1. Two monthly admission extracts

Each file has one row per admission and the same columns. `los_days` is the length of stay in days.

```python
import pandas as pd

jan = pd.DataFrame({
    'admission_id': ['A101', 'A102', 'A103'],
    'unit': ['ICU', 'MED', 'SURG'],
    'age': [67, 45, 72],
    'diagnosis': ['sepsis', 'pneumonia', 'appendicitis'],
    'los_days': [4, 2, 6],
})
feb = pd.DataFrame({
    'admission_id': ['A201', 'A202', 'A203'],
    'unit': ['MED', 'ICU', 'MED'],
    'age': [58, 81, 39],
    'diagnosis': ['heart failure', 'sepsis', 'cellulitis'],
    'los_days': [5, 9, 1],
})
display(jan)
print(list(jan.columns) == list(feb.columns))
```

**Expect:** the 3 January admissions with columns `admission_id`, `unit`, `age`, `diagnosis`, `los_days`, then `True`: same columns, same row meaning, so these are pieces of one table.

### 2. Stack rows with `concat()`

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

### 3. Record where each row came from

Once stacked, nothing says which file a row came from. Add a label column to each piece **before** stacking; it is an ordinary column, so it survives saving to CSV and later merges.

```python
jan['source_file'] = 'jan_admissions.csv'
feb['source_file'] = 'feb_admissions.csv'

stacked = pd.concat([jan, feb], ignore_index=True)
display(stacked['source_file'].value_counts())
display(stacked)
```

**Expect:** `jan_admissions.csv` 3 and `feb_admissions.csv` 3, and a `source_file` column as the last column of the 6-row table.

### 4. March changed its columns

The March extract comes from an updated report: it no longer includes `age`, and it adds `payer`.

```python
mar = pd.DataFrame({
    'admission_id': ['A301', 'A302', 'A303'],
    'unit': ['SURG', 'MED', 'ICU'],
    'diagnosis': ['hernia', 'COPD', 'stroke'],
    'los_days': [3, 2, 7],
    'payer': ['Medicare', 'Private', 'Medicaid'],
})
mar['source_file'] = 'mar_admissions.csv'
print(list(mar.columns))
```

**Expect:** `['admission_id', 'unit', 'diagnosis', 'los_days', 'payer', 'source_file']`: no `age`, and a new `payer`.

`concat()` matches columns by **name**. With the default `join='outer'` it keeps every column from every file.

```python
admissions = pd.concat([jan, feb, mar], ignore_index=True)
print(admissions.shape)
display(admissions.isna().sum())
display(admissions)
```

**Expect:** `(9, 7)` and missing counts of `age` 3 and `payer` 6, everything else 0. The 3 March rows have no `age`, and the 6 January and February rows have no `payer`. These gaps come from the files' layouts, not from missing measurements, so note them rather than filling them.

### 5. Patch gaps with `combine_first()`

The census system (ADT, the admission-discharge-transfer feed) is the primary source, but it failed to report February and April and has not reported June yet. Finance keeps its own estimate for every month.

```python
primary = pd.DataFrame({
    'month': ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05'],
    'admissions': [212, None, 225, None, 231],
    'bed_days': [1010, None, 1102, None, 1125],
}).set_index('month')

finance = pd.DataFrame({
    'month': ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06'],
    'admissions': [210, 200, 220, 238, 230, 219],
    'bed_days': [1000, 960, 1090, 1175, 1120, 1068],
}).set_index('month')

filled = primary.combine_first(finance)
display(filled)
```

**Expect:** 6 months. January, March, and May keep the ADT values (212, 225, 231 admissions), February and April take finance's estimates (200 and 238), and June comes only from finance (219). Values print as `212.0` because the gaps made the columns `float64`.

Compare `concat(axis=1)`, which puts the two sources side by side instead of patching one with the other:

```python
print(list(pd.concat([primary, finance], axis=1).columns))
```

**Expect:** `['admissions', 'bed_days', 'admissions', 'bed_days']`: duplicate column names and no filled gaps. For patching, `combine_first()` is the right tool.

Record which source each month's numbers came from. Start by marking every month as an estimate, then relabel the months where ADT actually reported.

```python
filled['data_source'] = 'finance_estimate'
reported = primary[primary['admissions'].notna()].index
filled.loc[reported, 'data_source'] = 'adt_primary'
print(list(reported))
display(filled)
```

**Expect:** `['2026-01', '2026-03', '2026-05']`, and `data_source` reads `adt_primary` for those three months and `finance_estimate` for February, April, and June.

Before saving, turn the month labels back into a column so the file keeps them.

```python
filled_flat = filled.reset_index()
print(list(filled_flat.columns))
```

**Expect:** `['month', 'admissions', 'bed_days', 'data_source']`.

## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run the cells above again first.

### 6. `concat()` then `merge()`: add unit details

Stacking made one admissions table. Unit names and bed counts live in a separate lookup table, so they are joined by key with `merge()`. Each admission should find exactly one unit: many-to-one.

```python
units = pd.DataFrame({
    'unit_code': ['ICU', 'MED', 'SURG'],
    'unit_name': ['Intensive Care', 'Medicine', 'Surgery'],
    'beds': [12, 30, 24],
})

enriched = pd.merge(admissions, units, left_on='unit', right_on='unit_code',
                    how='left', validate='many_to_one', indicator=True)
print(len(enriched))
display(enriched['_merge'].value_counts())
```

**Expect:** `9` rows (no admission gained or lost) with `both` 9, `left_only` 0, and `right_only` 0.

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

### 7. Side by side: monthly metrics from three systems

Each system reports one row per month. Put `month` in the index of each so that `concat(axis=1)` lines rows up by label. The month labels are plain text such as `'2026-01'`.

```python
census = pd.DataFrame({
    'month': ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06'],
    'admissions': [212, 198, 225, 240, 231, 219],
    'bed_days': [1010, 955, 1102, 1180, 1125, 1068],
}).set_index('month')

staffing = pd.DataFrame({
    'month': ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06'],
    'nurse_hours': [8200, 7900, 8650, 9100, 8900, 8500],
}).set_index('month')

experience = pd.DataFrame({
    'month': ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06'],
    'top_box_pct': [71, 69, 73, 74, 72, 75],     # % of survey answers in the top category
    'responses': [120, 135, 150, 165, 180, 195],
}).set_index('month')

print(census.index.is_unique, staffing.index.is_unique, experience.index.is_unique)
```

**Expect:** `True True True`: each month appears once in each system.

```python
monthly = pd.concat([census, staffing, experience], axis=1)
monthly['nurse_hours_per_bed_day'] = (monthly['nurse_hours'] / monthly['bed_days']).round(2)
print(monthly.shape)
display(monthly)
```

**Expect:** `(6, 6)`: one row per month with columns from all three systems plus the new ratio. Nurse hours per bed day range from 7.71 in April, the busiest month, to 8.27 in February.

### 8. When the labels don't all match

Infection control audits hand hygiene only in some months, and its file already includes July.

```python
audits = pd.DataFrame({
    'month': ['2026-02', '2026-03', '2026-04', '2026-07'],
    'hand_hygiene_pct': [88, 91, 93, 90],
}).set_index('month')

with_audits = pd.concat([monthly, audits], axis=1)
print(with_audits.shape)
display(with_audits[['admissions', 'hand_hygiene_pct']])
```

**Expect:** `(7, 7)`: the 6 metric columns plus `hand_hygiene_pct`. January, May, and June have `NaN` hand hygiene (no audit), and the new July row has `NaN` for every other system. The default `join='outer'` keeps every label from every input.

```python
audited_months = pd.concat([monthly, audits], axis=1, join='inner')
print(list(audited_months.index))
```

**Expect:** `['2026-02', '2026-03', '2026-04']`: only the months present in both inputs.

If the identifiers were in ordinary columns rather than the index, this would be a job for `merge()` on those columns.

### 9. Year over year: stack, then pivot

Compare the first quarter's admissions with the same months last year. Stack the two years with a `year` label column, then pivot to put the years side by side.

```python
q1_2025 = pd.DataFrame({'month': ['Jan', 'Feb', 'Mar'], 'admissions': [198, 185, 207]})
q1_2026 = pd.DataFrame({'month': ['Jan', 'Feb', 'Mar'], 'admissions': [212, 200, 225]})
q1_2025['year'] = 2025
q1_2026['year'] = 2026

both_years = pd.concat([q1_2025, q1_2026], ignore_index=True)
print(both_years.shape)
```

**Expect:** `(6, 3)`.

```python
yoy = both_years.pivot(index='month', columns='year', values='admissions')
print(list(yoy.index))
yoy = yoy.loc[['Jan', 'Feb', 'Mar']]           # calendar order
yoy['growth_pct'] = ((yoy[2026] - yoy[2025]) / yoy[2025] * 100).round(1)
display(yoy)
```

**Expect:** `['Feb', 'Jan', 'Mar']` first, because `pivot()` sorts the labels alphabetically; after `.loc` the rows are in calendar order. Growth is 7.1% in January, 8.1% in February, and 8.7% in March, so admissions rose every month and March grew the most.

### 10. Source labels as index levels

`keys=` is the alternative: it labels each piece with an outer index level instead of a column. Here the outer label repeats what `source_file` already says, so you can compare the two.

```python
by_file = pd.concat([jan, feb], keys=['jan', 'feb'], names=['source', 'row'])
display(by_file.loc['feb'])
```

**Expect:** the 3 February admissions (A201, A202, A203), selected by the outer label `feb`, with row labels 0 to 2.

```python
print(list(by_file.reset_index().columns))
```

**Expect:** `['source', 'row', 'admission_id', 'unit', 'age', 'diagnosis', 'los_days', 'source_file']`: `reset_index()` turns both label levels into ordinary columns. The course uses the `source_file` column form, because it stays a plain column from the start.

### 11. Keep only shared columns

`join='inner'` keeps only the columns every file shares.

```python
shared_only = pd.concat([jan, feb, mar], join='inner', ignore_index=True)
print(shared_only.shape)
print(list(shared_only.columns))
```

**Expect:** `(9, 5)` and `['admission_id', 'unit', 'diagnosis', 'los_days', 'source_file']`. `age` and `payer` are gone without any warning, which is why the outer result is the one to keep.
