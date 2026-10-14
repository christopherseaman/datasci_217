# Assignment 05: Clean a Lab Export

## Overview

Clean a diabetes clinic network's lab export the way Lecture 05 does: count and drop missing values, remove repeated rows, turn the export's `-999` code into real gaps, convert types, standardize text, add categories and indicator columns, validate every rule, flag outliers, and clean the table. At each stage you save a small summary an analyst would hand over, such as a missing-value report, row counts, or category counts, not the table itself; the checks compare those summaries with the same pipeline run on the same data.

`generate_data.py` writes `data/lab_visits.csv`, a synthetic export of about 59,000 visits by 18,000 patients. Each row is one clinic visit: `patient_id` (`D` and five digits), `visit_date`, `clinic`, `gender`, `age` in years, `weight_kg`, `hba1c_pct` (**HbA1c**, the percent of hemoglobin carrying glucose, a three-month average of blood sugar), and `glucose_mg_dl` (fasting glucose in mg/dL). Most patients come back every three to four months. Like many real exports, it has problems you can only find with code: blanks, rows sent twice, `-999` for a measurement that was not taken, ages typed as `unknown`, dates that do not exist, IDs and labels spelled several ways, and decimal points left out (`64` for 6.4).

```text
patient_id,visit_date,clinic,gender,age,weight_kg,hba1c_pct,glucose_mg_dl
D10632,2024-01-02,Sunset,female,72,73.4,6.3,118
D10720,2024-01-02,Sunset,F,46,96.0,5.9,106
D11156,2024-01-02,Excelsior,M,71,89.6,5.2,62
```

The cleaned table follows this data contract. One row is one visit; `patient_id` and `visit_date` together identify it, because a patient visits more than once.

| Column | Meaning | dtype | Rule |
| --- | --- | --- | --- |
| `patient_id` | study patient | `str` | `D` + five digits |
| `visit_date` | visit day | `datetime64` | a real calendar date; never blank |
| `clinic` | clinic name | `str` | one spelling per clinic; `Unknown` when the export left it blank |
| `gender` | gender code | `str` | `F`, `M`, or `X` (nonbinary); blank when the export's code is none of these |
| `age` | age at visit (years) | `Int64` | blank allowed |
| `weight_kg` | weight (kg) | `float64` | 30 to 250 when present |
| `hba1c_pct` | HbA1c (%) | `float64` | 4 to 20 when present |
| `glucose_mg_dl` | fasting glucose (mg/dL) | `float64` | blank allowed; flagged, never deleted, when unusual |

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open the cloned folder itself, not a folder above it.
2. In the integrated terminal, create the environment, activate it, and install the packages `pyproject.toml` and `uv.lock` list. Do not run `uv init`: the project files already exist.

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```

    - Expect: `uv venv` prints `Using CPython 3.13.x`, and `uv sync` lists `+ pandas==3.0.5` and `+ pyarrow==25.0.0` among the packages it installs.
3. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`.
4. Run the notebook's first code cell. It runs `generate_data.py` when `data/` is missing (the same as `python3 generate_data.py` in the terminal), then compares the file with `data_manifest.json`.
    - Expect: `NumPy: 2.3.3`, `pandas: 3.0.5`, and `data/lab_visits.csv matches data_manifest.json`. The data is about 2.7 MB and stays out of Git (`.gitignore` lists `data/`); the same generator writes the same bytes on every computer.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── generate_data.py        # supplied: writes data/lab_visits.csv; keep unchanged
├── data_manifest.json      # supplied: the checksum of the data generate_data.py writes
├── data/
│   └── lab_visits.csv      # written by the first cell; not committed
├── pyproject.toml          # supplied: the project's packages, numpy, pandas, pyarrow, and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── missing_report.csv      # you generate in Task 1.1
    ├── row_counts.csv          # you generate in Task 2.1
    ├── typed_profile.csv       # you generate in Task 2.3
    ├── clinic_counts.csv       # you generate in Task 3.1
    ├── gender_counts.csv       # you generate in Task 3.1
    ├── a1c_counts.csv          # you generate in Task 3.2
    ├── coded_sample.csv        # you generate in Task 3.3
    ├── validation_report.csv   # you generate in Task 4.1
    ├── glucose_fences.csv      # you generate in Task 4.2
    ├── decision_log.csv        # you generate in Task 5.1
    └── clean_summary.csv       # you generate in Task 5.2
```

Each task changes the same table, `labs`, and later tasks build on earlier ones, so run the cells in order. After a mistake, click **Restart**, then **Run All**. Every file is a few hundred bytes to a few KB; all eleven together are about 5 KB.

## Task 1: Find and drop missing values

### 1.1 Count the gaps

The first Task 1 cell reads the file with `labs = pd.read_csv(DATA_PATH)`, sets `rows_read = len(labs)`, and shows the first rows and dtypes.

- Expect: `shape: (59447, 8)`. `age` is `str`, because some ages are the word `unknown`, and `weight_kg` is `float64`, with `-999` counted as a weight.

1. Build a table with one row per column of `labs`:

    ```python
    missing_report = pd.DataFrame({
        "missing_count": labs.isna().sum(),
        "missing_pct": (labs.isna().mean() * 100).round(1),
    })
    ```

2. Name the index with `missing_report.index.name = "column"`.
3. Set `rows_with_gap` to the number of rows with at least one missing value: `labs.isna().any(axis=1).sum()`.
    - Expect: `rows with a gap: 7792`, and:

    ```text
                   missing_count  missing_pct
    column                                   
    patient_id                 0          0.0
    visit_date               206          0.3
    clinic                   240          0.4
    gender                   132          0.2
    age                       88          0.1
    weight_kg                388          0.7
    hba1c_pct               4272          7.2
    glucose_mg_dl           3484          5.9
    ```

    Only empty cells count so far: `-999` and `unknown` look like values until Task 2. Each `missing_pct` is `missing_count / 59447 * 100`.
4. Save with `missing_report.to_csv(MISSING_PATH)`, keeping the index: it holds the column names.

> **Checkpoint: `output/missing_report.csv`**
> The header line `column,missing_count,missing_pct`, then the eight rows above.

### 1.2 Drop visits with no lab results

A visit with neither an HbA1c nor a glucose result is a no-show, so it has nothing to analyze.

1. Drop it: `labs = labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="all")`. `how="all"` drops a row only when every listed column is missing, so a visit with one of the two results stays. The cell then sets `rows_with_labs = len(labs)`.
    - Expect: `rows with labs: 58554`. Sanity check: no more rows are dropped than the smaller of the two missing counts above (3,484).

Task 1.2 writes no file; `output/row_counts.csv` in Task 2.1 records the count.

## Task 2: Remove repeats, sentinels, and wrong types

### 2.1 Remove exact repeats and save the row counts

1. Set `exact_repeats` to `labs.duplicated().sum()`, the rows identical to an earlier row.
    - Expect: `exact repeats: 524`.
2. Display the first rows whose `patient_id` and `visit_date` appear more than once: `labs[labs.duplicated(subset=["patient_id", "visit_date"], keep=False)].head(6)`.
    - Expect: pairs of identical rows, such as D15568 on 2024-01-05 twice. A patient's visits on different dates are repeat visits, not repeated entries, so they stay.
3. Keep the first copy of each repeat: `labs = labs.drop_duplicates()`.
4. Record the three row counts in one Series, named so the file reads as a log:

    ```python
    row_counts = pd.Series({"read": rows_read, "with_labs": rows_with_labs, "unique": len(labs)}, name="rows")
    row_counts.index.name = "step"
    ```

    - Expect: `read 59447`, `with_labs 58554`, `unique 58030`. Invariants: the counts never grow from one step to the next, and `with_labs - unique` equals `exact_repeats`.
5. Save with `row_counts.to_csv(ROWS_PATH)`.

> **Checkpoint: `output/row_counts.csv`**
> The header line `step,rows`, then `read`, `with_labs`, and `unique` with their counts.

### 2.2 Replace the `-999` code

The export writes `-999` in `weight_kg` and `glucose_mg_dl` for a measurement that was not taken. Left in, it pulls every mean far below the real one.

1. Replace it in those two columns only, and assign the result back:

    ```python
    labs = labs.replace({"weight_kg": {-999: np.nan}, "glucose_mg_dl": {-999: np.nan}})
    ```

    - Expect: `weight_kg` now has 946 missing values and `glucose_mg_dl` 3079, and `labs["weight_kg"].min()` is 29.9, an impossible weight Task 4 catches.

### 2.3 Convert types and save a profile

1. Convert `age` to whole numbers that allow a gap: `labs["age"] = pd.to_numeric(labs["age"], errors="coerce").astype("Int64")`. `errors="coerce"` turns text that is not a number into a missing value.
2. Convert `visit_date` to dates: `labs["visit_date"] = pd.to_datetime(labs["visit_date"], format="%Y-%m-%d", errors="coerce")`. A date that does not exist, such as `2025-02-30`, becomes `NaT`.
    - Expect: `age` is `Int64` and `visit_date` is `datetime64[us]`.
3. Profile the five converted columns, one row per column:

    ```python
    typed = labs[["visit_date", "age", "weight_kg", "hba1c_pct", "glucose_mg_dl"]]
    typed_profile = pd.DataFrame({
        "dtype": typed.dtypes.astype(str),
        "missing_count": typed.isna().sum(),
        "min": typed.min(),
        "max": typed.max(),
        "mean": typed.mean(numeric_only=True).round(2),
    })
    typed_profile.index.name = "column"
    ```

    `numeric_only=True` leaves `visit_date` out of the means, so its `mean` is blank. pandas lines the five Series up by column name, which sorts the rows alphabetically.
    - Expect:

    ```text
                            dtype  missing_count                  min                  max    mean
    column
    age                     Int64            213                   25                   86   55.13
    glucose_mg_dl         float64           3079                 62.0                412.0  129.37
    hba1c_pct             float64           3347                  3.9                103.0    6.69
    visit_date     datetime64[us]            295  2024-01-02 00:00:00  2026-06-30 00:00:00    <NA>
    weight_kg             float64            946                 29.9               1295.0   86.84
    ```

    The `max` of 103.0 for HbA1c and 1295.0 for weight are decimal slips (10.3, 129.5); keep them for now, since Task 4 counts them and Task 5 blanks them. Sanity checks: no `min` is `-999`; `age`'s missing count is its blanks plus its `unknown` cells; `visit_date`'s is its blanks plus the dates that do not exist.
4. Save with `typed_profile.to_csv(PROFILE_PATH)`.

> **Checkpoint: `output/typed_profile.csv`**
> The header line `column,dtype,missing_count,min,max,mean`, then the five rows above.

## Task 3: Standardize text and encode categories

### 3.1 Clean the text columns and count them

1. Display `labs["clinic"].value_counts(dropna=False)`.
    - Expect: 21 spellings of four clinics, such as `Mission`, `mission`, `MISSION`, ` Mission` with a leading space, and `Mission ` with a trailing one, plus `NaN`.
2. Clean the IDs: `labs["patient_id"] = labs["patient_id"].str.strip().str.upper()`. `d10007` and ` D11126` become `D10007` and `D11126`.
3. Clean the clinic names with `labs["clinic"].str.strip().str.title()`, then fill the blanks with `.fillna("Unknown")`, and assign the result back to `labs["clinic"]`.
4. Map every spelling of gender to one code. A label the dictionary does not list, such as `U`, `unknown`, or `?`, becomes `NaN`:

    ```python
    gender_codes = {"f": "F", "female": "F", "m": "M", "male": "M", "x": "X", "nb": "X", "nonbinary": "X"}
    labs["gender"] = labs["gender"].str.strip().str.lower().map(gender_codes)
    ```

5. Count the cleaned columns, blanks included, and save both counts with their index:

    ```python
    clinic_counts = labs["clinic"].value_counts(dropna=False)
    gender_counts = labs["gender"].value_counts(dropna=False)
    clinic_counts.to_csv(CLINIC_PATH)
    gender_counts.to_csv(GENDER_PATH)
    ```

    - Expect: `clinic_counts` shows `Mission` 20225, `Sunset` 15923, `Excelsior` 12952, `Bayview` 8698, and `Unknown` 232; `gender_counts` shows `F` 29012, `M` 26876, `X` 1765, and `NaN` 377. Invariant: each count table sums to 58030, the rows of `labs`.

> **Checkpoint: `output/clinic_counts.csv` and `output/gender_counts.csv`**
> The header lines `clinic,count` and `gender,count`. The gender file's last row has a blank label: those are the `NaN` codes.

### 3.2 Bin HbA1c

The clinic sorts HbA1c into the standard bands: below 5.7% is normal, 5.7% to 6.4% is prediabetes, and 6.5% or more is diabetes.

1. Add `a1c_category` with `pd.cut(labs["hba1c_pct"], bins=[0, 5.6, 6.4, 20], labels=["normal", "prediabetes", "diabetes"])`. Each bin includes its right edge, `(5.6, 6.4]`, so with one-decimal results 5.6 is normal, 5.7 is prediabetes, and 6.5 is diabetes. A value outside every bin, such as a slip of 64, gets no category.
    - Worked example on four made-up values: `pd.cut(pd.Series([5.6, 5.7, 6.4, 6.5]), bins=[0, 5.6, 6.4, 20], labels=["normal", "prediabetes", "diabetes"])` gives `normal, prediabetes, prediabetes, diabetes`.
2. Count it with `a1c_counts = labs["a1c_category"].value_counts(dropna=False)` and save with `a1c_counts.to_csv(A1C_PATH)`.
    - Expect: `diabetes` 25763, `prediabetes` 14827, `normal` 14045, and `NaN` 3395: the 3347 blank HbA1c values plus the 48 above 20. Invariant: the four counts sum to 58030.

> **Checkpoint: `output/a1c_counts.csv`**
> The header line `a1c_category,count`, then the four rows above.

### 3.3 Add clinic indicators and save the sample

1. Build one 0/1 column per clinic: `clinic_dummies = pd.get_dummies(labs["clinic"], prefix="clinic", dtype="int64")`.
2. Add them with `labs = labs.join(clinic_dummies)`.
    - Expect: `shape: (58030, 14)`, with the new columns `clinic_Bayview`, `clinic_Excelsior`, `clinic_Mission`, `clinic_Sunset`, and `clinic_Unknown`; each row has one 1.
3. Take the visits of the twelve sample patients the first cell lists in `SAMPLE_IDS`, in order: `coded_sample = labs[labs["patient_id"].isin(SAMPLE_IDS)].sort_values(["patient_id", "visit_date"])`. These patients carry the export's problems in a few visits each: an ID in lower case or with a space, a blank or oddly spelled clinic, a gender of `nb` or `U`, HbA1c values on the bin edges, a slip, `-999`, and dates that are blank or do not exist.
4. Save with `coded_sample.to_csv(SAMPLE_PATH, index=False)`. The index holds only row numbers, so leave it out.
    - Expect: 56 rows and 14 columns. D10112 has 7 visits, one at `Unknown`; D10007 has 4, including the one saved as `d10007`. A visit whose ID still has a space or a lower-case `d` does not match `isin()`, so a short sample means Task 3.1's ID cleaning missed it.

> **Checkpoint: `output/coded_sample.csv`**
> 56 visits with 14 columns: the eight from Task 2, `a1c_category`, and the five indicators. Every ID is upper case without spaces, and every gender is `F`, `M`, `X`, or blank.

## Task 4: Validate the visits

### 4.1 Count rule failures

1. Build one column of `True`/`False` per rule. The two range rules let a missing value pass, because `between()` is `False` for `NaN`:

    ```python
    rules = pd.DataFrame({
        "patient_id_format": labs["patient_id"].str.fullmatch(r"D[0-9]{5}", na=False),
        "visit_date_present": labs["visit_date"].notna(),
        "gender_valid": labs["gender"].isin(["F", "M", "X"]),
        "hba1c_in_range": labs["hba1c_pct"].between(4, 20) | labs["hba1c_pct"].isna(),
        "weight_in_range": labs["weight_kg"].between(30, 250) | labs["weight_kg"].isna(),
    })
    ```

2. Count the failures per rule with `validation_report = (~rules).sum()`, then name it: `validation_report.index.name = "rule"` and `validation_report.name = "failing_rows"`.
3. Display the first failing rows with `labs[~rules.all(axis=1)].head(10)`.
    - Expect:

    ```text
    rule
    patient_id_format     119
    visit_date_present    295
    gender_valid          377
    hba1c_in_range         49
    weight_in_range        70
    Name: failing_rows, dtype: int64
    ```

    Cross-checks with earlier files: `visit_date_present` equals the `visit_date` missing count in `typed_profile.csv`, and `gender_valid` equals the blank row of `gender_counts.csv`. `between()` includes both ends, so an HbA1c of exactly 4.0 or 20.0 and a weight of exactly 30.0 or 250.0 pass, while 3.9, 20.1, 29.9, and 250.1 fail.
4. Save with `validation_report.to_csv(VALIDATION_PATH)`, keeping the index.

> **Checkpoint: `output/validation_report.csv`**
> The header line `rule,failing_rows`, then the five rows above.

### 4.2 Flag unusual glucose values

Flag unusual values rather than deleting them: a fasting glucose of 412 mg/dL is rare but real, and the clinic wants to see it.

1. Compute the quartiles of `labs["glucose_mg_dl"]` with `quantile(0.25)` and `quantile(0.75)`, then `iqr = q3 - q1`, `lower = q1 - 1.5 * iqr`, and `upper = q3 + 1.5 * iqr`.
    - Expect: `Q1 102.0, Q3 153.0, fences 25.5 to 229.5`.
2. Add the flag: `labs["glucose_outlier"] = (labs["glucose_mg_dl"] < lower) | (labs["glucose_mg_dl"] > upper)`. A comparison with `NaN` is `False`, so a blank glucose is not flagged.
3. Collect the fences and the number of flagged visits:

    ```python
    glucose_fences = pd.Series({"q1": q1, "q3": q3, "lower": lower, "upper": upper,
                                "outliers": labs["glucose_outlier"].sum()}, name="value")
    glucose_fences.index.name = "measure"
    ```

    - Expect: `outliers` 486.0, all above `upper`; the Series holds floats, so the count shows as 486.0. Sanity check: the outliers are a small share of the 54951 glucose values present, and none is blank.
4. Save with `glucose_fences.to_csv(FENCES_PATH)`.

> **Checkpoint: `output/glucose_fences.csv`**
> The header line `measure,value`, then `q1`, `q3`, `lower`, `upper`, and `outliers`.

## Task 5: Clean and summarize

### 5.1 Apply the decisions and log them

1. Work on a copy, so `labs` stays as validated: `clean = labs.copy()`.
2. Keep only the visits that can be identified, a valid ID and a real date: `clean = clean[rules["patient_id_format"] & rules["visit_date_present"]]`. A visit that breaks only a range or gender rule stays.
3. Before blanking, count the impossible values among the kept visits: `hba1c_outside = ((clean["hba1c_pct"] < 4) | (clean["hba1c_pct"] > 20)).sum()`, and `weight_outside` the same way with 30 and 250.
4. Blank the impossible values, keeping the visit: `clean["hba1c_pct"] = clean["hba1c_pct"].where(clean["hba1c_pct"].between(4, 20))`, and the same for `weight_kg` with 30 and 250. `where()` keeps a value where the condition is `True` and puts `NaN` elsewhere. Keep the other blanks as they are: nothing says what they were.
5. Number the rows again with `clean = clean.reset_index(drop=True)`.
6. Log the three decisions:

    ```python
    decision_log = pd.Series({"visits_dropped": len(labs) - len(clean), "hba1c_blanked": hba1c_outside,
                              "weight_blanked": weight_outside}, name="rows")
    decision_log.index.name = "decision"
    ```

    - Expect: `shape: (57618, 15)`, and `visits_dropped` 412, `hba1c_blanked` 48, `weight_blanked` 70. Sanity checks: `visits_dropped` is at least the larger and at most the sum of `patient_id_format` and `visit_date_present` in the validation report (some visits fail both); each blanked count is at most its rule's failing rows.
7. Save with `decision_log.to_csv(DECISION_PATH)`. It does not depend on the checks in Task 5.2, so it is saved first.

> **Checkpoint: `output/decision_log.csv`**
> The header line `decision,rows`, then the three rows above.

### 5.2 Validate, then save the summary

1. Collect the contract's checks in one Series:

    ```python
    checks = pd.Series({
        "one row per visit": not clean.duplicated(subset=["patient_id", "visit_date"]).any(),
        "patient IDs valid": clean["patient_id"].str.fullmatch(r"D[0-9]{5}").all(),
        "visit dates present": clean["visit_date"].notna().all(),
        "hba1c in range": clean["hba1c_pct"].dropna().between(4, 20).all(),
        "weight in range": clean["weight_kg"].dropna().between(30, 250).all(),
    })
    ```

2. Stop if any fails: `assert checks.all(), checks[~checks]`.
    - Expect: all five are `True`, and the `assert` prints nothing. If it stops, the message names the failing check; every earlier file is already saved, so fix that step and run the cells again.
3. Summarize the cleaned number columns: `clean_summary = clean[["age", "weight_kg", "hba1c_pct", "glucose_mg_dl"]].describe().round(2).T`. Round before `.T`: `.T` gives every column the `object` dtype, which `round()` leaves unchanged. `.T` turns the table so each column of `clean` is a row.
    - Expect:

    ```text
                     count    mean    std   min    25%    50%    75%    max
    age            57406.0   55.13   17.7  25.0   40.0   55.0   70.0   86.0
    weight_kg      56608.0   85.94  17.09  30.0   74.2   86.0   97.7  250.0
    hba1c_pct      54252.0    6.64   1.32   4.0    5.6    6.3    7.5   20.0
    glucose_mg_dl  54564.0  129.35  36.79  62.0  102.0  124.0  153.0  412.0
    ```

    Sanity checks: every `count` is below 57618, the rows of `clean`, because blanks stay blank; every `min` and `max` is inside the contract's range.
4. Save with `clean_summary.to_csv(SUMMARY_PATH)`. It is the only file saved after the `assert`, because it describes the validated table.

> **Checkpoint: `output/clean_summary.csv`**
> The header line `,count,mean,std,min,25%,50%,75%,max`, then the four rows above.

## Check your work

- Run the checks any time, even partway through: `python3 check_assignment.py` in the terminal (with the environment active) prints your score and what to fix.
- When **Restart** and **Run All** finish without errors, the notebook's last cell runs the same checks.
- Both use the latest checks from the course repository, the same checks GitHub runs on each push. Commit `assignment.ipynb` and the `output/` files, then push; `data/` stays out of Git.
- Expect: a passing run ends with `Score: 100/100`.

What each check looks for: [CHECKS.md](CHECKS.md)
