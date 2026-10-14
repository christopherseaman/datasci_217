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

# Demo 2: Transforming a Clinic Intake Export

A clinic's intake form exports column names with spaces and units, smoking status and site typed several ways, age as text, and pain as `'7/10'`. This demo renames the columns, normalizes the text, turns labels and text into numbers, groups values into bands, flags text with a pattern, stores repeated labels as categories, and prepares indicator columns for a model.

Run the cells from top to bottom; after each step, an **Expect** line says what you should see.

```python
# Installs the course's pandas, pyarrow, and thefuzz (for the optional last practice) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0 thefuzz==0.22.1
```

## Core walkthrough

### 1. Load the intake export

The cell writes the export to `output/intake_raw.csv` (Lecture 02's file writing), then reads it the Lecture 04 way.

```python
from pathlib import Path

import pandas as pd

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
display(intake)
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

### 2. Rename the columns

Function rules first (`str.strip`, `str.lower`, then spaces to underscores), then a dictionary for the two labels that still carry units.

```python
intake = intake.rename(columns=str.strip).rename(columns=str.lower)
intake.columns = intake.columns.str.replace(' ', '_')
print(intake.columns.tolist())

intake = intake.rename(columns={'sbp_(mmhg)': 'sbp', 'dbp_(mmhg)': 'dbp'})
print(intake.columns.tolist())
```

**Expect:** first `['patient_id', 'age', 'sbp_(mmhg)', 'dbp_(mmhg)', 'smoking_status', 'site', 'pain']`, then the same list with `'sbp'` and `'dbp'`. `intake['patient_id']` now works.

### 3. Normalize the text columns

```python
display(intake['site'].value_counts())

intake['site'] = intake['site'].str.strip().str.lower()
intake['smoking_status'] = intake['smoking_status'].str.strip().str.lower()
display(intake['site'].value_counts())
display(intake['smoking_status'].value_counts())
```

**Expect:** before, 9 different spellings, each counted once; two look like `north`, but one has a trailing space. After, `north 4`, `south 3`, `west 2`, and smoking status `never 4`, `former 3`, `current 2`.

### 4. Convert age, and leave unreadable ages missing

`'thirty-two'` and `'unknown'` cannot be read as numbers. Leave them missing and count them. Filling them with an average would put two invented ages into the age bands in step 6.

```python
intake['age'] = pd.to_numeric(intake['age'], errors='coerce').astype('Int64')
display(intake[['patient_id', 'age']])
print('Ages missing after conversion:', intake['age'].isna().sum())
```

**Expect:** `<NA>` for P102 and P106, and `Ages missing after conversion: 2`.

### 5. Map and apply the clinic's rules

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
display(intake[['patient_id', 'smoking_status', 'smoking_code', 'pain', 'pain_score', 'sbp', 'dbp', 'bp_stage']])
```

**Expect:** `smoking_code` 0, 2, 1, 0, 2, 1, 0, 0, 1 down the rows; `pain_score` matches the number before each slash; P104 (126/92) and P108 (130/90) reach `stage 2` on the diastolic pressure.

A label the dictionary does not know becomes missing, which is how `map` tells you a spelling slipped through:

```python
display(pd.Series(['never', 'Former', 'current']).map({'never': 0, 'former': 1, 'current': 2}))
```

**Expect:** `0.0`, `NaN`, `2.0`. The unnormalized `'Former'` has no code.

### 6. Group values into bands

Age bands with edges chosen to match a clinical table (`cut`), and SBP groups with edges chosen by the data so each holds about the same number of rows (`qcut`).

```python
intake['age_band'] = pd.cut(intake['age'], bins=[17, 39, 64, 120], labels=['18-39', '40-64', '65+'])
display(intake[['patient_id', 'age', 'age_band']])
```

**Expect:** P101 and P107 are `18-39`, P105 is `65+`, and the two missing ages stay `NaN` instead of landing in a band.

```python
intake['sbp_group'] = pd.qcut(intake['sbp'], q=3, labels=['low', 'middle', 'high'])
display(intake[['patient_id', 'sbp', 'sbp_group']])
display(intake['sbp_group'].value_counts())
```

**Expect:** `low` for 118, 120, and 126; `high` for 142 and 152; `middle` for the four readings of 130. Counts are `middle 4`, `low 3`, `high 2`, uneven because tied values cannot be split.

### 7. Flag text with a pattern

`str.contains` reads its pattern as a regular expression, and `|` means "or".

```python
intake['ever_smoker'] = intake['smoking_status'].str.contains('former|current')
display(intake[['patient_id', 'smoking_status', 'ever_smoker']])
print('Ever smokers:', intake['ever_smoker'].sum())
```

**Expect:** `True` for P102, P103, P105, P106, and P109, `False` for the four `never` rows, and `Ever smokers: 5`.

### 8. Store repeated labels as categories

```python
intake['site'] = intake['site'].astype('category')
print(intake['site'].cat.categories)
print(intake['site'].cat.codes.tolist())
```

**Expect:** categories `['north', 'south', 'west']` and codes `[0, 0, 1, 1, 0, 2, 2, 1, 0]`: each row stores a small number that points to its label.

### 9. Indicator columns for a model

```python
smoking_dummies = pd.get_dummies(intake['smoking_status'], prefix='smoking', drop_first=True, dtype='int64')
display(intake[['patient_id', 'smoking_status']].join(smoking_dummies))
```

**Expect:** two columns, `smoking_former` and `smoking_never`. `drop_first=True` dropped the alphabetically first label, `current`, so current smokers (P102 and P105) are 0 in both columns: `current` is the reference.

```python
display(intake.dtypes)
```

**Expect:** `age` is `Int64`, `smoking_code` and `pain_score` are `int64`, `age_band`, `sbp_group`, and `site` are `category`, `ever_smoker` is `bool`, and `bp_stage` is `str`.

## Independent practice

Continue on your own after class, outside the live route. These cells reuse the core results; if the runtime closed, run the cells above again first.

### Practice 1: SBP quantiles with repeated edges

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
display(intake['sbp_band'].value_counts())
```

**Expect:** `(126.0, 130.0]` 4, `(117.999, 126.0]` 3, `(130.0, 152.0]` 2. Three bins, not four; report them that way.

### Practice 2: Compare category storage

```python
intake['site_text'] = intake['site'].astype('str')
print('site as str:', intake['site_text'].memory_usage(deep=True), 'bytes')
print('site as category:', intake['site'].memory_usage(deep=True), 'bytes')
```

**Expect:** fewer bytes as a category. About `247` then `180` bytes with `pyarrow` installed, as the install cell and `uv sync` provide.

### Practice 3: Normalize multiword clinic labels

A second export uses full clinic names with repeated spaces. Keep its original labels beside the normalized ones so you can inspect exactly what changed.

```python
clinic_labels = pd.DataFrame({'raw_label': [' North  Clinic ', 'South   Clinic', ' NORTH CLINIC ']})
clinic_labels['normalized'] = (clinic_labels['raw_label'].str.strip().str.lower()
                               .str.replace(r' +', '_', regex=True))
display(clinic_labels)
display(clinic_labels['normalized'].value_counts())
assert clinic_labels['normalized'].tolist() == ['north_clinic', 'south_clinic', 'north_clinic']
```

**Expect:** three rows preserving the original text, with normalized labels `north_clinic`, `south_clinic`, and `north_clinic`. The counts are `north_clinic 2` and `south_clinic 1`; spaces and letter case no longer create extra clinics.

### Practice 4 (optional extension): Match misspelled clinic names

Prerequisite: BONUS.md, Fuzzy matching for near-duplicates. Nothing in the core route or the assignment needs this.

Normalizing case and spaces cannot fix a typo. `thefuzz` scores how similar two strings are (0 to 100), and `process.extractOne` returns the best canonical name above a cutoff. Names below the cutoff stay missing for a person to review.

```python
from thefuzz import fuzz, process

canonical = ['north_clinic', 'south_clinic', 'west_clinic']
typed = pd.DataFrame({'typed_name': ['north_clinic', 'nrth_clinic', 'south_clinc', 'sotuh_clinic', 'west_clinic', 'wset_clinic', 'central_clinic']})


def best_match(name, cutoff=80):
    """Return the closest canonical name scoring at least cutoff, else None."""
    hit = process.extractOne(name, canonical, scorer=fuzz.ratio, score_cutoff=cutoff)
    return hit[0] if hit else None


typed['matched'] = typed['typed_name'].apply(best_match)
typed['best_score'] = typed['typed_name'].apply(lambda name: process.extractOne(name, canonical, scorer=fuzz.ratio)[1])
display(typed)
```

**Expect:** 7 rows. `north_clinic` and `west_clinic` match themselves at 100. The typos `nrth_clinic` (96), `south_clinc` (96), `sotuh_clinic` (92), and `wset_clinic` (91) map to their canonical names. `central_clinic` scores 72 against `west_clinic`, below the cutoff of 80, so `matched` is `NaN` and the row goes to a person. Check `best_score` before trusting a match: a lower cutoff (70) would wrongly send `central_clinic` to `west_clinic`.
