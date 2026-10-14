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

# Demo 1: Joining Patient and Lab Tables with `pd.merge()`

- A clinic keeps a patient registry and a separate lab results table.
- Compare an inner join with a left join, then audit both tables with an outer join to find the patients with no labs and the labs with no patient.
- Match monthly visit counts to targets on three keys.
- Catch a registry export that would silently duplicate lab rows.

Run the cells from top to bottom; each **Expect** line says what you should see.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

## Core walkthrough

### 1. Load the tables

This supplied cell keeps any CSV already in `data/` and downloads the rest (`urlretrieve(url, path)` saves the file at a web address to `path`).

```python
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/06/demo/data"
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
for filename in ["demo1_patients.csv", "demo1_labs.csv", "demo1_registry.csv",
                 "demo1_visits.csv", "demo1_targets.csv",
                 "demo1_central_a1c.csv", "demo1_poc_a1c.csv"]:
    path = data_dir / filename
    if not path.exists():
        urlretrieve(f"{REPO_RAW}/{filename}", path)
print(sorted(p.name for p in data_dir.glob("demo1_*.csv")))
```

**Expect:** a list of the 7 `demo1_*.csv` file names.

The registry has one row per patient, so `patient_id` is its primary key. The lab table has one row per test; its `patient_id` is a foreign key that can repeat.

```python
patients = pd.read_csv(data_dir / "demo1_patients.csv")
labs = pd.read_csv(data_dir / "demo1_labs.csv")
display(patients)
display(labs)
print(patients['patient_id'].is_unique, labs['patient_id'].is_unique)
```

**Expect:** two tables (5 patients with `patient_id`, `birth_year`, `sex`, `clinic`, `bmi`; 7 lab rows with `units`), then `True False`: patient IDs are unique in the registry but repeat in the lab table.

Before merging, note three things the join types will reveal:

- P001 has **three** lab results, so patients to labs is **one-to-many**: one registry row can match many lab rows.
- P004 and P005 have **no** lab results yet.
- P006 has a lab result but **no registry record**, for example a referral that was never registered.

### 2. Inner join vs left join: who drops out

**Question:** "Which lab results attach to a registered patient, and which patients have none?" An inner join keeps only matching keys. A left join keeps every patient.

```python
inner_merge = pd.merge(patients, labs, on='patient_id', how='inner')
left_merge = pd.merge(patients, labs, on='patient_id', how='left')
print('inner:', len(inner_merge), 'left:', len(left_merge))
display(left_merge[left_merge['lab_id'].isna()][['patient_id', 'clinic', 'lab_id', 'test']])
```

**Expect:** `inner: 6 left: 8`. The inner join silently dropped P004 and P005. The left join keeps them as the 2 rows shown, with `NaN` in `lab_id` and `test`. Neither join keeps P006's lab, because P006 is not in the left table.

### 3. Outer join with `indicator=True`: see what matches

**Question:** "Which patients and labs match, and which are left over?" An outer join keeps every row from both tables; `indicator=True` adds a `_merge` column saying where each row came from.

```python
audit = pd.merge(patients, labs, on='patient_id', how='outer', indicator=True)
print(len(audit))
display(audit['_merge'].value_counts())
display(audit[audit['_merge'] != 'both'][['patient_id', 'clinic', 'lab_id', '_merge']])
```

**Expect:** 9 rows. `_merge` counts: `both` 6, `left_only` 2, `right_only` 1. The leftover table has 3 rows: P004 and P005 as `left_only` (registered, no labs) and P006 as `right_only` (a lab with no registry record, so `clinic` is `NaN`). Use `indicator=True` whenever a merge's row count surprises you.

### 4. Merging on several keys: visits against targets

Clinic operations tracks monthly visits by clinic and service. Targets are set per clinic, month, and service, so a visit count matches a target only when **all three** keys agree.

```python
visits = pd.read_csv(data_dir / "demo1_visits.csv")
targets = pd.read_csv(data_dir / "demo1_targets.csv")
display(visits)
display(targets)

keys = ['clinic', 'month', 'service']
vs_target = pd.merge(visits, targets, on=keys, how='left')
vs_target['pct_of_target'] = (vs_target['visits'] / vs_target['target'] * 100).round(1)
display(vs_target)
```

**Expect:** 5 January visit rows and 6 target rows (4 for January, 2 for February). The merge has 5 rows. `pct_of_target` is 102.5 for North primary care, 92.3 for North cardiology, 95.0 for South primary care, and 105.6 for South cardiology. East has no target, so its `target` and `pct_of_target` are `NaN`.

### 5. Check the cardinality before you trust a merge

Next month's registry export arrives. Looking up each lab's patient is a **many-to-one** merge (many lab rows, one registry row per patient). Watch the row count.

```python
registry = pd.read_csv(data_dir / "demo1_registry.csv")
print('missing keys:', labs['patient_id'].isna().sum(), registry['patient_id'].isna().sum())
unchecked = pd.merge(labs, registry, on='patient_id', how='left')
print('lab rows before:', len(labs), 'after:', len(unchecked))
```

**Expect:** `missing keys: 0 0`, so no blank key can match another blank. Then `lab rows before: 7 after: 8`: one lab row was photocopied, and nothing raised an error.

Find the repeated key:

```python
print(registry['patient_id'].is_unique)
display(registry[registry.duplicated(subset=['patient_id'], keep=False)])
```

**Expect:** `False`, then 2 rows for P003: a `retired` North record and a `current` South record. P003 moved clinics, and the export kept both rows.

**Intentional error:** declare the contract with `validate='many_to_one'`, and pandas refuses the merge. The `try`/`except` (Lecture 02) catches the `MergeError` so the notebook keeps running.

```python
try:
    pd.merge(labs, registry, on='patient_id', how='left', validate='many_to_one')
except pd.errors.MergeError as error:
    print('MergeError:', error)
```

**Expect:** `MergeError: Merge keys are not unique in right dataset; not a many-to-one merge`, followed by a `Duplicates in right:` list that names `P003`.

**The fix:** apply a documented rule (keep only `current` records), confirm the key is now unique, and merge again with both the check and the audit column.

```python
current = registry[registry['record_status'] == 'current']
print(current['patient_id'].is_unique)

labs_with_clinic = pd.merge(labs, current, on='patient_id', how='left',
                            validate='many_to_one', indicator=True)
print(len(labs_with_clinic))
display(labs_with_clinic['_merge'].value_counts())
display(labs_with_clinic[['lab_id', 'patient_id', 'clinic', '_merge']])
```

**Expect:** `True`, then 7 rows (one per lab, as it should be), with `both` 6, `left_only` 1, and `right_only` 0 (pandas lists every `_merge` label, even an empty one). P003's lab `L04` now shows the `South` clinic, and P006's lab is the one `left_only` row.

## Independent practice

Continue on your own after class. These cells reuse the core results; if the runtime closed, run the cells above again first.

### 6. Right join: every lab result

**Question:** "Show every lab result, even when the patient is not registered."

```python
right_merge = pd.merge(patients, labs, on='patient_id', how='right')
print(len(right_merge))
orphans = right_merge[right_merge['birth_year'].isna()]
display(orphans[['lab_id', 'patient_id', 'test', 'value', 'units']])
```

**Expect:** 7 rows in the right join, one per lab. The orphan table has 1 row: `L05`, `P006`, `A1c`, `6.4`, `%`. An orphaned lab is a data-quality issue to send back to registration.

### 7. Why all three keys matter

Check that each key combination appears once in each table, then confirm the merge is one-to-one with `validate`.

```python
print(visits.duplicated(subset=keys).sum(), targets.duplicated(subset=keys).sum())
checked = pd.merge(visits, targets, on=keys, how='left', validate='one_to_one')
print(len(checked))
```

**Expect:** `0 0`, then `5`: no repeated key combination, so `validate='one_to_one'` passes.

Merging on `clinic` alone pairs every January count with every target for that clinic, including February's.

```python
wrong = pd.merge(visits, targets, on='clinic')
print(len(wrong))
print(list(wrong.columns))
```

**Expect:** `12` rows from 5 visit counts, with the key columns that were left out renamed `month_x`, `month_y`, `service_x`, and `service_y`.

### 8. Which clinic-service pairs have no target? A cross join

A missing target leaves no row to find. Build the expected grid of every clinic with every service using `how='cross'`, then left-merge January's targets onto it.

```python
clinic_list = visits[['clinic']].drop_duplicates()
services = visits[['service']].drop_duplicates()
expected = pd.merge(clinic_list, services, how='cross')
print(len(expected))

jan_targets = targets[targets['month'] == '2026-01']
coverage = pd.merge(expected, jan_targets, on=['clinic', 'service'],
                    how='left', indicator=True)
display(coverage[coverage['_merge'] == 'left_only'][['clinic', 'service']])
```

**Expect:** `6` (3 clinics × 2 services), then 2 rows: East primary care and East cardiology have no January target.

### 9. Overlapping column names: two A1c sources

The central lab and a point-of-care (POC) device both report A1c in a column named `a1c`. `a1c` is not the key, so pandas must rename one of them.

```python
central = pd.read_csv(data_dir / "demo1_central_a1c.csv")
poc = pd.read_csv(data_dir / "demo1_poc_a1c.csv")
display(pd.merge(central, poc, on='patient_id', validate='one_to_one'))
```

**Expect:** 4 rows with columns `patient_id`, `a1c_x`, `collected`, `a1c_y`, `device`. Which `a1c` is which? The names do not say.

```python
paired = pd.merge(central, poc, on='patient_id', validate='one_to_one',
                  suffixes=('_lab', '_poc'))
paired['poc_minus_lab'] = (paired['a1c_poc'] - paired['a1c_lab']).round(1)
display(paired[['patient_id', 'a1c_lab', 'a1c_poc', 'device', 'poc_minus_lab']])
```

**Expect:** columns `a1c_lab` and `a1c_poc`, and `poc_minus_lab` of -0.2, 0.2, -0.7, and 0.1. P003's device reading is 0.7 points below the lab, the largest gap, and it came from device `DCA-2`: a question for the lab's quality team. This comparison needs the descriptive suffixes to be readable.

### 10. Putting it together: a checked patient-lab table

Merge the registry with the labs, keep every patient, check the cardinality, and flag who has results.

```python
patient_labs = pd.merge(patients, labs, on='patient_id', how='left',
                        validate='one_to_many', indicator=True)
patient_labs['has_lab'] = patient_labs['_merge'] == 'both'
display(patient_labs['has_lab'].value_counts())
display(patient_labs.sort_values(['clinic', 'patient_id'])[
    ['clinic', 'patient_id', 'test', 'value', 'units', 'collected', 'has_lab']
])
```

**Expect:** `True` 6 and `False` 2, then 8 rows sorted by clinic (North first) and patient: North holds P001's three labs, P003's A1c, and P005 with no lab; South holds P002's two labs and P004 with no lab.
