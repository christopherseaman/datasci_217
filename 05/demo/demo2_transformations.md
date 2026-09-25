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

# Demo 2: Reshaping a Clinic Intake Export

A clinic's intake form exports column names with spaces and units, smoking status and site typed several ways, age as text, and pain as `'7/10'`. This demo renames the columns, normalizes the text, turns labels and text into numbers, groups values into bands, and prepares indicator columns for a model. Everything here comes from Lecture 05 up to the second demo break, plus Lectures 01 to 04.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-24 with Python 3.13, pandas 3.0.5, and NumPy 2.3.3; the whole notebook runs in a few seconds.

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

## 1. Load the intake export

The cell writes the export to `output/intake_raw.csv` (Lecture 02's file writing), then reads it the Lecture 04 way.

```python
export_text = """Patient ID ,AGE,SBP (mmHg),DBP (mmHg),Smoking Status,Site,Pain
P101,34,118,76,never,North,2/10
P102,thirty-two,142,84,Current,north ,7/10
P103,58,130,78,FORMER, South,0/10
P104,45,126,92, never,south,5/10
P105,71,152,88,current,NORTH,8/10
P106,unknown,130,82,Former,West,3/10
P107,29,120,72,never,west,1/10
P108,63,130,90,never,South,4/10
P109,52,130,80,former,north,6/10
"""
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
raw_path = output_dir / 'intake_raw.csv'
with raw_path.open('w', encoding='utf-8') as file:
    file.write(export_text)

intake = pd.read_csv(raw_path)
print(intake)
print(intake.columns.tolist())
```

**Expect:** 9 patients and 7 columns. The first label is `'Patient ID '`, with a trailing space.

That trailing space breaks selection. This cell fails on purpose and prints the error instead of stopping.

```python
try:
    intake['Patient ID']
except KeyError as error:
    print('KeyError:', error)
```

**Expect:** `KeyError: 'Patient ID'`. The fix is to rename the columns.

## 2. Rename the columns

Function rules first (`str.strip`, `str.lower`, then spaces to underscores), then a dictionary for the two labels that still carry units.

```python
intake = intake.rename(columns=str.strip).rename(columns=str.lower)
intake.columns = intake.columns.str.replace(' ', '_')
print(intake.columns.tolist())

intake = intake.rename(columns={'sbp_(mmhg)': 'sbp', 'dbp_(mmhg)': 'dbp'})
print(intake.columns.tolist())
```

**Expect:** first `['patient_id', 'age', 'sbp_(mmhg)', 'dbp_(mmhg)', 'smoking_status', 'site', 'pain']`, then the same list with `'sbp'` and `'dbp'`. `intake['patient_id']` now works.

## 3. Normalize the text columns

```python
print(intake['site'].value_counts())

intake['site'] = intake['site'].str.strip().str.lower()
intake['smoking_status'] = intake['smoking_status'].str.strip().str.lower()
print(intake['site'].value_counts())
print(intake['smoking_status'].value_counts())
```

**Expect:** before, 9 different spellings, each counted once; two look like `north`, but one has a trailing space. After, `north 4`, `south 3`, `west 2`, and smoking status `never 4`, `former 3`, `current 2`.

## 4. Convert age, and leave unreadable ages missing

`'thirty-two'` and `'unknown'` cannot be read as numbers. Leave them missing and count them. Filling them with an average would put two invented ages into the age bands in step 6.

```python
intake['age'] = pd.to_numeric(intake['age'], errors='coerce').astype('Int64')
print(intake[['patient_id', 'age']])
print('Ages missing after conversion:', intake['age'].isna().sum())
```

**Expect:** `<NA>` for P102 and P106, and `Ages missing after conversion: 2`.

## 5. Map and apply the clinic's rules

A dictionary turns each smoking label into a code, a `lambda` pulls the number out of `'7/10'`, and a function staging blood pressure reads two columns per row.

```python
intake['smoking_code'] = intake['smoking_status'].map({'never': 0, 'former': 1, 'current': 2})
intake['pain_score'] = intake['pain'].apply(lambda text: int(text.split('/')[0]))


def bp_stage(row):
    """Stage one reading from both pressures (simplified ACC/AHA)."""
    if row['sbp'] >= 140 or row['dbp'] >= 90:
        return 'stage 2'
    elif row['sbp'] >= 130 or row['dbp'] >= 80:
        return 'stage 1'
    else:
        return 'below stage 1'


intake['bp_stage'] = intake.apply(bp_stage, axis=1)
print(intake[['patient_id', 'smoking_status', 'smoking_code', 'pain', 'pain_score', 'sbp', 'dbp', 'bp_stage']])
```

**Expect:** `smoking_code` 0, 2, 1, 0, 2, 1, 0, 0, 1 down the rows; `pain_score` matches the number before each slash; P104 (126/92) and P108 (130/90) reach `stage 2` on the diastolic pressure.

A label the dictionary does not know becomes missing, which is how `map` tells you a spelling slipped through:

```python
print(pd.Series(['never', 'Former', 'current']).map({'never': 0, 'former': 1, 'current': 2}))
```

**Expect:** `0.0`, `NaN`, `2.0`. The unnormalized `'Former'` has no code.

## 6. Group values into bands

Age bands with edges chosen to match a clinical table (`cut`), then SBP quartiles chosen by the data (`qcut`).

```python
intake['age_band'] = pd.cut(intake['age'], bins=[17, 39, 64, 120], labels=['18-39', '40-64', '65+'])
print(intake[['patient_id', 'age', 'age_band']])
```

**Expect:** P101 and P107 are `18-39`, P105 is `65+`, and the two missing ages stay `NaN` instead of landing in a band.

Four readings were recorded as exactly 130, a common rounding habit. Asking `qcut` for quartiles fails on purpose here, because two quartile edges are both 130:

```python
try:
    pd.qcut(intake['sbp'], q=4)
except ValueError as error:
    print('ValueError:', error)
```

**Expect:** `ValueError: Bin edges must be unique: Index([118.0, 126.0, 130.0, 130.0, 152.0], dtype='float64', name='sbp').`, followed by the hint `You can drop duplicate edges by setting the 'duplicates' kwarg`. That hint is the fix: `duplicates='drop'` merges the repeated edge and leaves three bins:

```python
intake['sbp_band'] = pd.qcut(intake['sbp'], q=4, duplicates='drop')
print(intake['sbp_band'].value_counts())
```

**Expect:** `(126.0, 130.0]` 4, `(117.999, 126.0]` 3, `(130.0, 152.0]` 2. Three bins, not four; report them that way.

## 7. Store repeated labels as categories

```python
print('site as str:', intake['site'].memory_usage(deep=True), 'bytes')
intake['site'] = intake['site'].astype('category')
print('site as category:', intake['site'].memory_usage(deep=True), 'bytes')
print(intake['site'].cat.categories)
print(intake['site'].cat.codes.tolist())
```

**Expect:** fewer bytes as a category: `616` then `302` in the course environment, or `249` then `180` in Colab, whose `pyarrow` package stores text more compactly. The categories are `['north', 'south', 'west']` and the codes are `[0, 0, 1, 1, 0, 2, 2, 1, 0]`.

## 8. Indicator columns for a model

```python
smoking_dummies = pd.get_dummies(intake['smoking_status'], prefix='smoking', drop_first=True, dtype='int64')
print(pd.concat([intake[['patient_id', 'smoking_status']], smoking_dummies], axis=1))
```

**Expect:** two columns, `smoking_former` and `smoking_never`. `drop_first=True` dropped the alphabetically first label, `current`, so current smokers (P102 and P105) are 0 in both columns: `current` is the reference.

```python
print(intake.dtypes)
```

**Expect:** `age` is `Int64`, `smoking_code` and `pain_score` are `int64`, `age_band`, `sbp_band`, and `site` are `category`, and `bp_stage` is `str`.
