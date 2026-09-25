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

A clinic keeps a patient registry and a separate lab results table. This demo joins them every way Lecture 06 teaches, finds the patients with no labs and the labs with no patient, matches monthly visit counts to targets on several keys, names overlapping columns, and catches a registry export that would silently duplicate lab rows. Everything here comes from Lecture 06 up to the first demo break, plus Lectures 01 to 05.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-25 with Python 3.13 and pandas 3.0.5; the whole notebook runs in a few seconds. The patient IDs and values are synthetic.

## Setup

The first cell installs pandas 3.0.5, the course version. Colab ships an older pandas (2.2). A `.venv` made with `uv venv --seed` includes pip, so the same `%pip` cell works locally too.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.`, perhaps after a notice that a newer pip is available; neither needs any action. Locally, with the requirements already installed, the cell changes nothing and you can go on. In Colab, pip may also print a dependency conflict because some preinstalled packages expect pandas 2.2; that is expected, and this demo does not use them. If Colab asks you to restart the session (or says pandas was previously imported), choose **Runtime → Restart session**, then continue with the next cell. You do not need to rerun the install.

```python
import pandas as pd

print('pandas', pd.__version__)
```

**Expect:** `pandas 3.0.5`.

## 1. The two tables

The registry has one row per patient, so `patient_id` is its primary key. The lab table has one row per test; its `patient_id` is a foreign key that can repeat.

```python
patients = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
    'birth_year': [1958, 1971, 1964, 1990, 1983],
    'clinic': ['North', 'South', 'North', 'South', 'North'],
})

labs = pd.DataFrame({
    'lab_id': ['L01', 'L02', 'L03', 'L04', 'L05', 'L06', 'L07'],
    'patient_id': ['P001', 'P001', 'P002', 'P003', 'P006', 'P001', 'P002'],
    'test': ['A1c', 'LDL', 'A1c', 'A1c', 'A1c', 'A1c', 'LDL'],
    'value': [7.2, 142.0, 5.6, 8.1, 6.4, 6.9, 118.0],     # A1c in %, LDL in mg/dL
    'collected': ['2026-01-12', '2026-01-12', '2026-01-20', '2026-02-03',
                  '2026-02-05', '2026-04-14', '2026-04-22'],
})

print(patients)
print(labs)
print(patients['patient_id'].is_unique, labs['patient_id'].is_unique)
```

**Expect:** 5 patients and 7 lab rows, then `True False`: patient IDs are unique in the registry but repeat in the lab table.

Before merging, note three things the join types will reveal:

- P001 has **three** lab results, so patients to labs is **one-to-many**: one registry row can match many lab rows.
- P004 and P005 have **no** lab results yet.
- P006 has a lab result but **no registry record**, for example a referral that was never registered.

## 2. Inner join: only matching keys

**Question:** "Which lab results can I attach to a registered patient?"

```python
inner_merge = pd.merge(patients, labs, on='patient_id', how='inner')
print(len(inner_merge))
inner_merge
```

**Expect:** 6 rows. P001 appears 3 times, P002 twice, P003 once. P004 and P005 (no labs) and P006's lab (no registry record) are gone.

Always check row counts: if you expected every patient, the inner join silently dropped two of them.

## 3. Left join: every patient

**Question:** "Show every registered patient, with labs where they exist."

```python
left_merge = pd.merge(patients, labs, on='patient_id', how='left')
print(len(left_merge))
left_merge
```

**Expect:** 8 rows: the 6 matched rows plus P004 and P005 with `NaN` in `lab_id`, `test`, `value`, and `collected`. P006's lab is still excluded because P006 is not in the left table.

A missing lab value is a clinical to-do list: these patients are due for their first A1c.

```python
no_labs = left_merge[left_merge['lab_id'].isna()]
no_labs[['patient_id', 'clinic']]
```

**Expect:** 2 rows: P004 (South) and P005 (North).

## 4. Right join: every lab result

**Question:** "Show every lab result, even when the patient is not registered."

```python
right_merge = pd.merge(patients, labs, on='patient_id', how='right')
print(len(right_merge))
orphans = right_merge[right_merge['birth_year'].isna()]
orphans[['lab_id', 'patient_id', 'test', 'value']]
```

**Expect:** 7 rows in the right join, one per lab. The orphan table has 1 row: `L05`, `P006`, `A1c`, `6.4`. An orphaned lab is a data-quality issue to send back to registration.

## 5. Outer join with `indicator=True`: the full audit

**Question:** "Show everything, and say where each row came from."

```python
audit = pd.merge(patients, labs, on='patient_id', how='outer', indicator=True)
print(len(audit))
print(audit['_merge'].value_counts())
```

**Expect:** 9 rows, with `both` 6, `left_only` 2, and `right_only` 1: the 6 matched labs, the 2 patients without labs, and the 1 orphaned lab.

```python
audit[audit['_merge'] != 'both'][['patient_id', 'clinic', 'lab_id', '_merge']]
```

**Expect:** 3 rows: P004 and P005 as `left_only`, and P006 as `right_only` with `NaN` clinic.

`indicator=True` is the quickest way to see what matched and what didn't. Use it whenever a merge's row count surprises you.

## 6. Check the cardinality before you trust a merge

Patients to labs should be one-to-many. `validate=` turns that expectation into a check: the merge runs only if the left keys are unique.

```python
checked = pd.merge(patients, labs, on='patient_id', how='left', validate='one_to_many')
print(len(checked))
```

**Expect:** `8`, the same as the unchecked left join, because the registry has no repeated `patient_id`.

Next month's registry export arrives. Looking up each lab's patient is a **many-to-one** merge (many lab rows, one registry row per patient). Watch the row count.

```python
registry = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P003', 'P004', 'P005'],
    'clinic': ['North', 'South', 'North', 'South', 'South', 'North'],
    'record_status': ['current', 'current', 'retired', 'current', 'current', 'current'],
})

print('missing keys:', labs['patient_id'].isna().sum(), registry['patient_id'].isna().sum())
unchecked = pd.merge(labs, registry, on='patient_id', how='left')
print('lab rows before:', len(labs), 'after:', len(unchecked))
```

**Expect:** `missing keys: 0 0`, so no blank key can match another blank. Then `lab rows before: 7 after: 8`: one lab row was photocopied, and nothing raised an error.

Find the repeated key:

```python
print(registry['patient_id'].is_unique)
registry[registry.duplicated(subset=['patient_id'], keep=False)]
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
print(labs_with_clinic['_merge'].value_counts())
labs_with_clinic[['lab_id', 'patient_id', 'clinic', '_merge']]
```

**Expect:** `True`, then 7 rows (one per lab, as it should be), with `both` 6, `left_only` 1, and `right_only` 0 (pandas lists every `_merge` label, even an empty one). P003's lab `L04` now shows the `South` clinic, and P006's lab is the one `left_only` row.

## 7. Merging on several keys: visits against targets

Clinic operations tracks monthly visits by clinic and service. Targets are set per clinic, month, and service, so a visit count matches a target only when **all three** keys agree.

```python
visits = pd.DataFrame({
    'clinic': ['North', 'North', 'South', 'South', 'East'],
    'month': ['2026-01'] * 5,
    'service': ['primary_care', 'cardiology', 'primary_care', 'cardiology', 'primary_care'],
    'visits': [410, 120, 380, 95, 150],
})

targets = pd.DataFrame({
    'clinic': ['North', 'North', 'South', 'South', 'North', 'South'],
    'month': ['2026-01', '2026-01', '2026-01', '2026-01', '2026-02', '2026-02'],
    'service': ['primary_care', 'cardiology', 'primary_care', 'cardiology',
                'primary_care', 'primary_care'],
    'target': [400, 130, 400, 90, 420, 410],
})

keys = ['clinic', 'month', 'service']
print(visits.duplicated(subset=keys).sum(), targets.duplicated(subset=keys).sum())
```

**Expect:** `0 0`: each key combination appears once in each table, so this is a one-to-one merge.

```python
vs_target = pd.merge(visits, targets, on=keys, how='left', validate='one_to_one')
vs_target['pct_of_target'] = (vs_target['visits'] / vs_target['target'] * 100).round(1)
vs_target
```

**Expect:** 5 rows. `pct_of_target` is 102.5 for North primary care, 92.3 for North cardiology, 95.0 for South primary care, and 105.6 for South cardiology. East has no target, so its `target` and `pct_of_target` are `NaN`.

Why all three keys matter: merging on `clinic` alone pairs every January count with every target for that clinic, including February's.

```python
wrong = pd.merge(visits, targets, on='clinic')
print(len(wrong))
print(list(wrong.columns))
```

**Expect:** `12` rows from 5 visit counts, with the key columns that were left out renamed `month_x`, `month_y`, `service_x`, and `service_y`.

## 8. Which clinic-service pairs have no target? A cross join

A missing target leaves no row to find. Build the expected grid of every clinic with every service using `how='cross'`, then left-merge January's targets onto it.

```python
clinic_list = pd.DataFrame({'clinic': ['North', 'South', 'East']})
services = pd.DataFrame({'service': ['primary_care', 'cardiology']})
expected = pd.merge(clinic_list, services, how='cross')
print(len(expected))

jan_targets = targets[targets['month'] == '2026-01']
coverage = pd.merge(expected, jan_targets, on=['clinic', 'service'],
                    how='left', indicator=True)
coverage[coverage['_merge'] == 'left_only'][['clinic', 'service']]
```

**Expect:** `6` (3 clinics × 2 services), then 2 rows: East primary care and East cardiology have no January target.

## 9. Overlapping column names: two A1c sources

The central lab and a point-of-care (POC) device both report A1c in a column named `a1c`. `a1c` is not the key, so pandas must rename one of them.

```python
central = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P004'],
    'a1c': [7.2, 5.6, 8.1, 6.3],
    'collected': ['2026-01-12', '2026-01-20', '2026-02-03', '2026-02-10'],
})
poc = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P004'],
    'a1c': [7.0, 5.8, 7.4, 6.4],
    'device': ['DCA-1', 'DCA-1', 'DCA-2', 'DCA-2'],
})

pd.merge(central, poc, on='patient_id', validate='one_to_one')
```

**Expect:** 4 rows with columns `patient_id`, `a1c_x`, `collected`, `a1c_y`, `device`. Which `a1c` is which? The names do not say.

```python
paired = pd.merge(central, poc, on='patient_id', validate='one_to_one',
                  suffixes=('_lab', '_poc'))
paired['poc_minus_lab'] = (paired['a1c_poc'] - paired['a1c_lab']).round(1)
paired[['patient_id', 'a1c_lab', 'a1c_poc', 'device', 'poc_minus_lab']]
```

**Expect:** columns `a1c_lab` and `a1c_poc`, and `poc_minus_lab` of -0.2, 0.2, -0.7, and 0.1. P003's device reading is 0.7 points below the lab, the largest gap, and it came from device `DCA-2`: a question for the lab's quality team. This comparison needs the descriptive suffixes to be readable.

## 10. Putting it together: a checked patient-lab table

Merge the registry with the labs, keep every patient, check the cardinality, and flag who has results.

```python
patient_labs = pd.merge(patients, labs, on='patient_id', how='left',
                        validate='one_to_many', indicator=True)
patient_labs['has_lab'] = patient_labs['_merge'] == 'both'
print(patient_labs['has_lab'].value_counts())
patient_labs.sort_values(['clinic', 'patient_id'])[
    ['clinic', 'patient_id', 'test', 'value', 'collected', 'has_lab']
]
```

**Expect:** `True` 6 and `False` 2, then 8 rows sorted by clinic (North first) and patient: North holds P001's three labs, P003's A1c, and P005 with no lab; South holds P002's two labs and P004 with no lab.
