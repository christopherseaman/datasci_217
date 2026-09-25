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

# Demo 3: From Raw Clinic Visits to a Validated Clean Table

This demo runs the whole cleaning pipeline from Lecture 05 on a clinic visit export: state the contract, load and preserve the raw table, audit it with validation rules, record each decision, transform a working copy, and save only after the checks pass. Everything here comes from Lecture 05 and Lectures 01 to 04.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. The files it writes go to `output/`, which disappears when a Colab runtime shuts down. Tested 2026-09-24 with Python 3.13, pandas 3.0.5, and NumPy 2.3.3; the whole notebook runs in a few seconds.

## Setup

The first cell installs pandas 3.0.5, the course version. Colab ships an older pandas (2.2). A `.venv` made with `uv venv --seed` includes pip, so the same `%pip` cell works locally too.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.` Locally, with the requirements already installed, the cell changes nothing and you can go on. In Colab, pip may also print a dependency conflict because some preinstalled packages expect pandas 2.2; that is expected, and this demo does not use them. If Colab asks you to restart the session (or says pandas was previously imported), choose **Runtime → Restart session**, then continue with the next cell. You do not need to rerun the install.

```python
from pathlib import Path

import pandas as pd

print('pandas', pd.__version__)
```

**Expect:** `pandas 3.0.5`.

## 1. The data contract

- **Row meaning:** one row is one clinic visit.
- **Candidate identifier:** `visit_id`, unique for every visit.

| Column | Meaning | dtype | Rule |
| --- | --- | --- | --- |
| `visit_id` | visit | `string` | unique; never blank |
| `patient_id` | study patient | `string` | `P` + three digits |
| `visit_date` | visit day | `datetime64` | exact `YYYY-MM-DD` text and a real date; missing allowed but flagged |
| `site` | clinic | `string` | `north`, `south`, or `west`; missing allowed but flagged |
| `age` | age at visit (years) | `Int64` | 0 to 120; missing allowed but flagged |
| `sbp` | systolic blood pressure (mmHg) | `Int64` | 60 to 250; missing allowed but flagged |
| `needs_review` | follow-up needed | `boolean` | `True` exactly when a date, site, age, or SBP is missing |

## 2. Load the raw table and keep it unchanged

The cell writes the export to `output/clinic_visits_raw.csv`, then reads every column as text, keeping blanks and the code `NA` exactly as written.

```python
export_text = """visit_id,patient_id,visit_date,site,age,sbp
V001,P001,2026-01-05,north,34,122
V002,P002,2026-01-06,South,41,135
V003,P0003,2026-01-07,north,52,128
V004,P004,2026-7-01,west,150,118
V005,P005,2026-02-30,south,67,141
V006,P006,2026-02-11,east,45,NA
V007,P007,2026-02-12,north,,131
V007,P007,2026-02-12,north,,131
V008,P008,2026-02-13,south,38,1320
V009,P009,2026-02-14,west,59,210
V010,P010,2026-02-15,north,29,117
V011,P011,2026-02-16, North ,71,138
V012,P012,2026-02-17,south,48,126
"""
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
raw_path = output_dir / 'clinic_visits_raw.csv'
with raw_path.open('w', encoding='utf-8') as file:
    file.write(export_text)

raw = pd.read_csv(raw_path, dtype='string', keep_default_na=False)
raw_snapshot = raw.copy(deep=True)
print(raw)
print('Blank cells:', (raw == '').sum().sum(), '| cells holding the text NA:', (raw == 'NA').sum().sum())
```

**Expect:** 13 rows, all `string`. V007's age is blank in both of its rows, and V006's SBP shows the text `NA`: `Blank cells: 2 | cells holding the text NA: 1`.

## 3. Audit with validation rules

Each rule gives one `True`/`False` per row. Blank and `NA` are the export's missing codes: allowed, but flagged later.

```python
allowed_sites = ['north', 'south', 'west']
missing_codes = ['', 'NA']
age_number = pd.to_numeric(raw['age'], errors='coerce')
sbp_number = pd.to_numeric(raw['sbp'], errors='coerce')

rules = pd.DataFrame({
    'patient_id P+3 digits': raw['patient_id'].str.fullmatch(r'P[0-9]{3}'),
    'date text YYYY-MM-DD': raw['visit_date'].str.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}'),
    'site allowed': raw['site'].isin(allowed_sites),
    'age 0-120 or missing': raw['age'].isin(missing_codes) | age_number.between(0, 120),
    'sbp 60-250 or missing': raw['sbp'].isin(missing_codes) | sbp_number.between(60, 250),
})
print((~rules).sum())               # rows failing each rule
print(raw[~rules.all(axis=1)])      # rows to review
```

**Expect:** failures `patient_id` 1 (V003's `P0003`), date text 1 (V004's `2026-7-01`), site 3 (`South`, `east`, and `' North '`), age 1 (V004's 150), and SBP 1 (V008's 1320). Six rows to review: V002, V003, V004, V006, V008, and V011.

Two problems pass every text rule. `2026-02-30` has the right shape but is not a real date, and V007 appears twice.

```python
exact_date = raw['visit_date'].str.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')
parsed = pd.to_datetime(raw['visit_date'].where(exact_date), format='%Y-%m-%d', errors='coerce')
print(raw.loc[parsed.isna(), ['visit_id', 'visit_date']])

print('Exact repeated rows:', raw.duplicated().sum())
print('visit_id unique:', raw['visit_id'].is_unique)
```

**Expect:** V004 and V005 cannot be parsed, `Exact repeated rows: 1`, and `visit_id unique: False`.

## 4. Flag unusual readings

The IQR rule flags candidates; it does not decide.

```python
q1 = sbp_number.quantile(0.25)
q3 = sbp_number.quantile(0.75)
iqr = q3 - q1
sbp_flag = (sbp_number < q1 - 1.5 * iqr) | (sbp_number > q3 + 1.5 * iqr)
print('Fences:', q1 - 1.5 * iqr, 'to', q3 + 1.5 * iqr)
print(raw.loc[sbp_flag, ['visit_id', 'sbp']])
```

**Expect:** `Fences: 104.375 to 159.375`, and two flagged visits. V008's 1320 already fails the range rule: almost certainly a typing error. V009's 210 is inside the contract and possible in severe hypertension, so it stays; it is a reading to confirm, not to delete.

## 5. Spot-check random rows

`head()` shows only the top of the file. Three random rows show the middle.

```python
print(raw.sample(n=3, random_state=42))
```

**Expect:** V011, V009, and V001, in that order. One random draw already turns up `' North '` and the 210 reading, so problems are not confined to the top of the file.

## 6. Record each decision

One row per decision: which field, what is wrong, what to do, and why. The table is saved with the cleaned data so someone else can repeat the steps.

```python
decisions = pd.DataFrame({
    'field': ['all columns', 'patient_id', 'site', 'site', 'visit_date', 'age', 'sbp', 'sbp'],
    'issue': ['V007 exported twice', 'P0003 fails P + 3 digits', 'case and spaces', 'east is not a study site',
              'not exact or not a real date', 'blank or outside 0-120', 'NA code or outside 60-250',
              '210 flagged by IQR'],
    'action': ['keep the first raw copy', 'correct to P003', 'strip and lowercase', 'set to missing',
               'set to missing', 'set to missing', 'set to missing', 'keep'],
    'reason': ['identical in every raw column', 'the clinic roster lists P003 and no P0003',
               'same three clinics typed differently', 'no east clinic; ask the source',
               'the intended date cannot be known', '150 is probably a typo, but a guess is not data',
               '1320 is probably 132, but a guess is not data', 'inside the contract and clinically possible'],
})
print(decisions)
```

**Expect:** 8 decisions. Every "set to missing" row also sets `needs_review` in the next step.

## 7. Transform a working copy

```python
working = raw.copy(deep=True)
working['patient_id'] = working['patient_id'].replace({'P0003': 'P003'})
working['site'] = working['site'].str.strip().str.lower()
working['site'] = working['site'].where(working['site'].isin(allowed_sites))
working['visit_date'] = parsed
working['age'] = age_number.where(age_number.between(0, 120)).astype('Int64')
working['sbp'] = sbp_number.where(sbp_number.between(60, 250)).astype('Int64')
working['needs_review'] = working[['visit_date', 'site', 'age', 'sbp']].isna().any(axis=1).astype('boolean')
print(working)
print(working.dtypes)
```

**Expect:** 13 rows still (the repeated V007 is still there), with `<NA>` or `NaT` in place of every bad value, and `needs_review` `True` for V004, V005, V006, both V007 rows, and V008.

## 8. Check the contract before saving

Each invariant is one named `True`/`False`, collected in a Series so it prints as a report. A function lets the same checks run again after a fix.

```python
def run_checks(table):
    """Return one named True/False per contract rule."""
    return pd.Series({
        'visit IDs unique': table['visit_id'].is_unique,
        'patient IDs are P + 3 digits': table['patient_id'].str.fullmatch(r'P[0-9]{3}').all(),
        'sites allowed when present': table['site'].dropna().isin(allowed_sites).all(),
        'ages 0-120 when present': table['age'].dropna().between(0, 120).all(),
        'SBP 60-250 when present': table['sbp'].dropna().between(60, 250).all(),
        'only the repeated row removed': len(table) == len(raw) - raw.duplicated().sum(),
    })


checks = run_checks(working)
print(checks)
```

**Expect:** two `False` checks, `visit IDs unique` and `only the repeated row removed`. The duplicate decision from step 6 was never applied.

This cell fails on purpose: the `assert` gate stops before any file is written. `try`/`except` prints the error so the rest of the notebook still runs.

```python
try:
    assert checks.all(), checks[~checks]
    working.to_csv(output_dir / 'clinic_visits_clean.csv', index=False)
except AssertionError as error:
    print('AssertionError: nothing saved. Failed checks:')
    print(error)
```

**Expect:** `AssertionError: nothing saved. Failed checks:` and the two `False` checks. `to_csv()` never ran.

**The fix:** apply the duplicate decision. Decide from the raw rows, not the cleaned ones: two rows that only became identical after cleaning were different in the export. `reset_index(drop=True)` renumbers the rows so the saved file reads back with the same labels.

```python
clean = working[~raw.duplicated()].reset_index(drop=True)
checks = run_checks(clean)
print(checks)

assert checks.all(), checks[~checks]
clean.to_csv(output_dir / 'clinic_visits_clean.csv', index=False)
decisions.to_csv(output_dir / 'decision_log.csv', index=False)
print('Saved', len(clean), 'visits;', clean['needs_review'].sum(), 'flagged for review')
```

**Expect:** every check `True`, then `Saved 12 visits; 5 flagged for review`.

## 9. Read the saved file back

A round trip proves the file holds what the table held. Give `read_csv` the intended dtypes, and put dates in `parse_dates`.

```python
round_trip = pd.read_csv(
    output_dir / 'clinic_visits_clean.csv',
    dtype={'visit_id': 'string', 'patient_id': 'string', 'site': 'string',
           'age': 'Int64', 'sbp': 'Int64', 'needs_review': 'boolean'},
    parse_dates=['visit_date'],
)
print(round_trip.dtypes)
print('Round trip equals clean:', round_trip.equals(clean))
print('Raw table unchanged:', raw.equals(raw_snapshot))
print('Rows:', len(raw), 'raw,', len(clean), 'clean')
```

**Expect:** the same dtypes as in step 7, `Round trip equals clean: True`, `Raw table unchanged: True`, and `Rows: 13 raw, 12 clean`. The checks show the table matches its contract; the decision table shows why each value changed.
