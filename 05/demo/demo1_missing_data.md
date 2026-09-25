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

# Demo 1: Missing Values, Duplicates, Sentinels, and Types

A clinic sends a small visit export. Before anyone averages a blood pressure, find the gaps, tell a double entry from a repeat visit, turn disguised missing values into real ones, and convert text columns to numbers and dates. Everything here comes from Lecture 05 up to the first demo break, plus Lectures 01 to 04.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-24 with Python 3.13, pandas 3.0.5, and NumPy 2.3.3; the whole notebook runs in a few seconds.

## Setup

The first cell installs pandas 3.0.5, the course version. Colab ships an older pandas (2.2). A `.venv` made with `uv venv --seed` includes pip, so the same `%pip` cell works locally too.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.` Locally, with the requirements already installed, the cell changes nothing and you can go on. In Colab, pip may also print a dependency conflict because some preinstalled packages expect pandas 2.2; that is expected, and this demo does not use them. If Colab asks you to restart the session (or says pandas was previously imported), choose **Runtime → Restart session**, then continue with the next cell. You do not need to rerun the install.

```python
import numpy as np
import pandas as pd

print('pandas', pd.__version__)
```

**Expect:** `pandas 3.0.5`.

## 1. The data contract

Write down what the table should look like before changing anything.

- **Row meaning:** one row is one clinic visit.
- **Candidate identifier:** `patient_id` + `visit_date`, because a patient can visit more than once.

| Column | Meaning | dtype | Rule |
| --- | --- | --- | --- |
| `patient_id` | study patient | `str` | `P` + three digits; never blank |
| `visit_date` | visit day | `datetime64` | a real calendar date |
| `age` | age at visit (years) | `Int64` | 0 to 120 when present |
| `sbp` | systolic blood pressure (mmHg) | `Int64` | 60 to 250 when present |
| `cholesterol` | total cholesterol (mg/dL) | `float64` | blank allowed |

## 2. Load the export

The export stores age as typed text (`age_text`) and dates as text. It also hides several problems that are not blank yet.

```python
visits = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P007', 'P008', 'P008'],
    'visit_date': ['2026-01-15', '2026-01-16', None, '2026-01-18', '2026-01-19',
                   '2026-02-30', '2026-01-21', '2026-01-21', '2026-01-22', '2026-03-05'],
    'age_text': ['45', 'unknown', '62', '34', 'forty', '58', '41', '41', '50', '50'],
    'sbp': [120, 135, None, -999, 1420, None, 125, 125, 130, 128],               # mmHg
    'cholesterol': [200, None, 185, 190, 220, None, None, None, 210, 205],       # mg/dL
})
print(visits)
print(visits.dtypes)
```

**Expect:** 10 rows. `age_text` and `visit_date` are `str`; `sbp` and `cholesterol` are `float64`, and `sbp` shows `-999.0` for P004 and `1420.0` for P005.

## 3. Count the gaps

```python
print(visits.isna().sum())                    # gaps per column
print((visits.isna().mean() * 100).round(1))  # percent missing per column
print(visits.isna().sum(axis=1))              # gaps per row
print('Rows with any gap:', visits.isna().any(axis=1).sum(), 'of', len(visits))
```

**Expect:** gaps of 1 in `visit_date`, 2 in `sbp`, and 4 in `cholesterol` (10.0, 20.0, and 40.0 percent), and `Rows with any gap: 5 of 10`. These counts miss `-999`, `1420`, `'unknown'`, `'forty'`, and `'2026-02-30'`, because none of them is blank yet.

## 4. Double entry or repeat visit?

A repeated `patient_id` is evidence to investigate, not an instruction to delete. Compare an exact-row check with the candidate identifier from the contract.

```python
print('Exact repeats:', visits.duplicated().sum())
print(visits[visits.duplicated(keep=False)])   # every row in a repeated set

print('Rows sharing a patient_id:', visits.duplicated(subset=['patient_id'], keep=False).sum())
print('Rows sharing patient_id + visit_date:',
      visits.duplicated(subset=['patient_id', 'visit_date'], keep=False).sum())
```

**Expect:** `Exact repeats: 1`, the two P007 rows (labels 6 and 7), `Rows sharing a patient_id: 4`, and `Rows sharing patient_id + visit_date: 2`.

P007's two rows match in every column: one visit entered twice. P008's two rows have different dates: two real visits, so both stay.

```python
clean = visits.drop_duplicates()
print(len(visits), 'rows before,', len(clean), 'after')
print(clean['patient_id'].value_counts())
```

**Expect:** `10 rows before, 9 after`. P008 is the only patient counted twice; row label 7 is gone.

## 5. Sentinels and impossible values

`-999` is the export's code for "not measured", and `1420` is not a possible systolic pressure. pandas averages both as if they were real readings.

```python
print('Mean SBP with the bad values:', clean['sbp'].mean())

clean['sbp'] = clean['sbp'].replace(-999, np.nan)   # a fixed code: replace it
clean['sbp'] = clean['sbp'].mask(clean['sbp'] > 250)  # a rule: blank anything above 250
print('Mean SBP after:', clean['sbp'].mean())
print(clean[['patient_id', 'sbp']])
```

**Expect:** `Mean SBP with the bad values: 151.28571428571428`, then `Mean SBP after: 127.6`. P004 and P005 now show `NaN`.

`1420` is probably `142` with an extra zero, but that is a guess. Blanking it and flagging the visit for review (step 6) keeps a guess out of the data.

## 6. Convert text columns

`to_numeric(..., errors='coerce')` turns `'unknown'` and `'forty'` into missing values, and `Int64` keeps whole numbers whole even with gaps.

```python
clean['age'] = pd.to_numeric(clean['age_text'], errors='coerce').astype('Int64')
clean['sbp'] = clean['sbp'].astype('Int64')
print(clean[['patient_id', 'age_text', 'age', 'sbp']])
```

**Expect:** `age` is `<NA>` for P002 (`'unknown'`) and P005 (`'forty'`), and `sbp` prints whole numbers such as `120` with `<NA>` in the gaps.

A date that does not exist becomes `NaT` with an explicit format. Separate a blank date (never recorded) from a date that failed to parse (recorded but wrong).

```python
parsed = pd.to_datetime(clean['visit_date'], format='%Y-%m-%d', errors='coerce')
date_audit = pd.DataFrame({
    'patient_id': clean['patient_id'],
    'raw_date': clean['visit_date'],
    'parsed_date': parsed,
    'parse_failed': parsed.isna() & clean['visit_date'].notna(),
})
print(date_audit)

clean['visit_date'] = parsed
clean['needs_review'] = (clean['age'].isna() | clean['visit_date'].isna() | clean['sbp'].isna()).astype('boolean')
print(clean.dtypes)
```

**Expect:** `parse_failed` is `True` only for P006 (`2026-02-30`). P003's date is also `NaT`, but it was never recorded, so it is not a parse failure. The dtypes now include `datetime64[us]`, `Int64`, and `boolean`.

## 7. Recount, then drop or fill

The disguised problems are now real missing values, so the counts go up.

```python
print(clean.isna().sum())
```

**Expect:** `visit_date 2`, `sbp 4`, `cholesterol 3`, and `age 2`. Before the conversions, only one date and two pressures looked missing.

For a blood-pressure and cholesterol analysis, a visit with neither measurement adds nothing. Drop only those rows.

```python
analysis = clean.dropna(subset=['sbp', 'cholesterol'], how='all')
print(len(clean), 'rows before,', len(analysis), 'after')
print(clean.loc[clean['sbp'].isna() & clean['cholesterol'].isna(), ['patient_id', 'age', 'sbp', 'cholesterol']])
```

**Expect:** `9 rows before, 8 after`. P006 is the one row with neither measurement. P003 and P004 lost their pressure but keep a cholesterol value, so they stay.

Fill cholesterol with its median, and record the rule in the table: a `cholesterol_imputed` column marks every filled value, because each one is a guess.

```python
analysis['cholesterol_imputed'] = analysis['cholesterol'].isna()
median_chol = analysis['cholesterol'].median()
analysis['cholesterol'] = analysis['cholesterol'].fillna(median_chol)
print('Rule: fill missing cholesterol with the median,', median_chol, 'mg/dL;',
      analysis['cholesterol_imputed'].sum(), 'values filled')
print(analysis[['patient_id', 'cholesterol', 'cholesterol_imputed']])
```

**Expect:** `Rule: fill missing cholesterol with the median, 202.5 mg/dL; 2 values filled`. P002 and P007 show `202.5` and `True`.

**What not to do:** forward fill copies the previous row down. These rows are different patients, so the previous row says nothing about this one.

```python
wrong = analysis[['patient_id', 'sbp']].copy()
wrong['sbp_ffill'] = analysis['sbp'].ffill()
print(wrong)
```

**Expect:** P003, P004, and P005 all receive `135`, P002's reading. Leave those gaps missing; `needs_review` already marks them.

## 8. The result

```python
print(analysis[['patient_id', 'visit_date', 'age', 'sbp', 'cholesterol', 'cholesterol_imputed', 'needs_review']])
print('Rows:', len(visits), 'raw,', len(analysis), 'for analysis;',
      analysis['needs_review'].sum(), 'flagged for review')
```

**Expect:** 8 rows and `Rows: 10 raw, 8 for analysis; 4 flagged for review` (P002, P003, P004, and P005). Every change is visible: the double entry is gone, sentinels and impossible values are missing rather than averaged, the filled cholesterol values are marked, and nothing was carried from one patient to another.
