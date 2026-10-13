# Assignment 05: Clean a Lab Export

## Overview

Clean a diabetes clinic's lab export the way Lecture 05 does: count and drop missing values, remove repeated rows, turn the export's `-999` code into real gaps, convert types, standardize text, add categories and indicator columns, validate every rule, and save a cleaned table, saving each stage for the checks.

`data/lab_visits.csv` is synthetic. Each row is one clinic visit: `patient_id` (`D` and four digits), `visit_date`, `clinic`, `sex`, `age` in years, `weight_kg`, `hba1c_pct` (**HbA1c**, the percent of hemoglobin carrying glucose, a three-month average of blood sugar), and `glucose_mg_dl` (fasting glucose in mg/dL). Like many real exports, it has blanks, two rows entered twice, `-999` for a measurement that was not taken, an age typed as text, a date that does not exist, and clinic and sex labels spelled several ways.

```text
patient_id,visit_date,clinic,sex,age,weight_kg,hba1c_pct,glucose_mg_dl
D2001,2026-03-02,Mission,F,54,82.5,7.8,162
D2002,2026-03-02,mission ,f,61,95.1,6.1,118
D2003,2026-03-03,Sunset,M,47,-999,8.9,201
```

The cleaned table follows this data contract. One row is one visit; `patient_id` and `visit_date` together identify it, because a patient can visit more than once.

| Column | Meaning | dtype | Rule |
| --- | --- | --- | --- |
| `patient_id` | study patient | `str` | `D` + four digits |
| `visit_date` | visit day | `datetime64` | a real calendar date; never blank |
| `clinic` | clinic name | `str` | one spelling per clinic; `Unknown` when the export left it blank |
| `sex` | sex code | `str` | `F` or `M`; blank when the export's code is neither |
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
4. Run the notebook's first code cell.
    - Expect: `NumPy: 2.3.3`, `pandas: 3.0.5`, and `data file found: True`.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/
│   └── lab_visits.csv      # supplied lab export; keep it exactly as handed out
├── pyproject.toml          # supplied: the project's packages, numpy, pandas, pyarrow, and ipykernel
├── uv.lock                 # supplied: the exact versions `uv sync` installs
├── .python-version         # supplied: tells uv to use Python 3.13
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── missing_report.csv      # you generate in Task 1.1
    ├── labs_typed.csv          # you generate in Task 2.3
    ├── labs_coded.csv          # you generate in Task 3.3
    ├── validation_report.csv   # you generate in Task 4.1
    └── labs_clean.csv          # you generate in Task 5.2
```

Each task changes the same table, `labs`, and later tasks build on earlier ones, so run the cells in order. After a mistake, click **Restart**, then **Run All**.

## Task 1: Find and drop missing values

### 1.1 Count the gaps

The first Task 1 cell reads the file with `labs = pd.read_csv(DATA_PATH)` and shows its first rows and dtypes.

- Expect: `shape: (22, 8)`. `age` is `str`, because one age is the word `unknown`, and `weight_kg` is `float64`, with `-999` counted as a weight.

1. Build a table with one row per column of `labs`:

    ```python
    missing_report = pd.DataFrame({
        "missing_count": labs.isna().sum(),
        "missing_pct": (labs.isna().mean() * 100).round(1),
    })
    ```

2. Name the index with `missing_report.index.name = "column"`.
3. Set `rows_with_gap` to the number of rows with at least one missing value: `labs.isna().any(axis=1).sum()`.
    - Expect: `rows with a gap: 5`, and:

    ```text
                   missing_count  missing_pct
    column                                   
    patient_id                 0          0.0
    visit_date                 1          4.5
    clinic                     1          4.5
    sex                        0          0.0
    age                        0          0.0
    weight_kg                  0          0.0
    hba1c_pct                  2          9.1
    glucose_mg_dl              2          9.1
    ```

    Only empty cells count so far: `-999` and `unknown` look like values until Task 2.
4. Save with `missing_report.to_csv(MISSING_PATH)`, keeping the index: it holds the column names.

> **Checkpoint: `output/missing_report.csv`**
> The header line `column,missing_count,missing_pct`, then the eight rows above.

### 1.2 Drop visits with no lab results

A visit with neither an HbA1c nor a glucose result is a no-show, so it has nothing to analyze.

1. Drop it: `labs = labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="all")`. `how="all"` drops a row only when every listed column is missing, so a visit with one of the two results stays.
    - Expect: `shape: (21, 8)`. D2011 is gone; D2004 (no HbA1c) and D2017 (no glucose) stay.

Task 1.2 writes no file; `output/labs_typed.csv` in Task 2.3 shows the result.

## Task 2: Remove repeats, sentinels, and wrong types

### 2.1 Remove exact repeats

1. Set `exact_repeats` to `labs.duplicated().sum()`, the rows identical to an earlier row.
    - Expect: `exact repeats: 2`.
2. Display every row whose `patient_id` and `visit_date` appear more than once: `labs[labs.duplicated(subset=["patient_id", "visit_date"], keep=False)]`.
    - Expect: four rows, two each for D2006 and D2016. D2001 is not among them: its two visits are on different dates, so it is a repeat visit, not a repeated entry.
3. Keep the first copy of each repeat: `labs = labs.drop_duplicates()`.
    - Expect: `shape: (19, 8)`, with both of D2001's visits.

### 2.2 Replace the `-999` code

The export writes `-999` in `weight_kg` and `glucose_mg_dl` for a measurement that was not taken. Left in, it pulls every mean far below zero.

1. Replace it in those two columns only, and assign the result back:

    ```python
    labs = labs.replace({"weight_kg": {-999: np.nan}, "glucose_mg_dl": {-999: np.nan}})
    ```

    - Expect: `weight_kg` now has 2 missing values (D2003 and D2017) and `glucose_mg_dl` 2 (D2010 and D2017), and `labs["weight_kg"].min()` is 61.0.

### 2.3 Convert types and save

1. Convert `age` to whole numbers that allow a gap: `labs["age"] = pd.to_numeric(labs["age"], errors="coerce").astype("Int64")`. `errors="coerce"` turns text that is not a number into a missing value.
2. Convert `visit_date` to dates: `labs["visit_date"] = pd.to_datetime(labs["visit_date"], format="%Y-%m-%d", errors="coerce")`. A date that does not exist becomes `NaT`.
    - Expect: `age` is `Int64` and D2005's age is `<NA>`; `visit_date` is `datetime64[us]`, and D2009's `2026-02-30` is now `NaT`, beside D2013's date that was blank already.
3. Save with `labs.to_csv(TYPED_PATH, index=False)`. The index holds only row numbers, so leave it out.

> **Checkpoint: `output/labs_typed.csv`**
> The header line `patient_id,visit_date,clinic,sex,age,weight_kg,hba1c_pct,glucose_mg_dl`, then 19 visits. No `-999` and no `unknown` remain; D2005's age, D2009's and D2013's dates, and the four values from 2.2 are blank.

## Task 3: Standardize text and encode categories

### 3.1 Clean the text columns

1. Display `labs["clinic"].value_counts(dropna=False)`.
    - Expect: nine spellings of three clinics, such as `Mission`, `mission`, `MISSION`, and ` Mission` with a leading space, plus `NaN`.
2. Clean the IDs: `labs["patient_id"] = labs["patient_id"].str.strip().str.upper()`. `d2014 ` becomes `D2014`.
3. Clean the clinic names with `labs["clinic"].str.strip().str.title()`, then fill the one blank with `.fillna("Unknown")`, and assign the result back to `labs["clinic"]`.
    - Expect: `value_counts()` shows `Mission` 8, `Sunset` 6, `Excelsior` 4, and `Unknown` 1.
4. Map every spelling of sex to one code. A label the dictionary does not list, such as `u`, becomes `NaN`:

    ```python
    sex_codes = {"f": "F", "female": "F", "m": "M", "male": "M"}
    labs["sex"] = labs["sex"].str.strip().str.lower().map(sex_codes)
    ```

    - Expect: `value_counts(dropna=False)` shows `F` 10, `M` 8, and `NaN` 1 (D2012).

### 3.2 Bin HbA1c

The clinic sorts HbA1c into the standard bands: below 5.7% is normal, 5.7% to 6.4% is prediabetes, and 6.5% or more is diabetes.

1. Add `a1c_category` with `pd.cut(labs["hba1c_pct"], bins=[0, 5.6, 6.4, 20], labels=["normal", "prediabetes", "diabetes"])`. Each bin includes its right edge, `(5.6, 6.4]`, so with one-decimal results 5.7 is prediabetes and 6.5 is diabetes. A value outside every bin gets no category.
    - Expect: `value_counts(dropna=False)` shows `diabetes` 11, `normal` 3, `prediabetes` 3, and `NaN` 2: D2004, whose HbA1c is blank, and D2007, whose `64` is above 20.

### 3.3 Add clinic indicators and save

1. Build one 0/1 column per clinic: `clinic_dummies = pd.get_dummies(labs["clinic"], prefix="clinic", dtype="int64")`.
2. Add them with `labs = labs.join(clinic_dummies)`.
    - Expect: `shape: (19, 13)`, with the new columns `clinic_Excelsior`, `clinic_Mission`, `clinic_Sunset`, and `clinic_Unknown`; each row has one 1.
3. Save with `labs.to_csv(CODED_PATH, index=False)`.

> **Checkpoint: `output/labs_coded.csv`**
> 19 visits with 13 columns: the eight from Task 2, `a1c_category`, and the four indicators. Every ID is upper case without spaces, every clinic has one spelling, D2008's clinic is `Unknown`, and every sex is `F`, `M`, or blank.

## Task 4: Validate the visits

### 4.1 Count rule failures

1. Build one column of `True`/`False` per rule. The two range rules let a missing value pass, because `between()` is `False` for `NaN`:

    ```python
    rules = pd.DataFrame({
        "patient_id_format": labs["patient_id"].str.fullmatch(r"D[0-9]{4}", na=False),
        "visit_date_present": labs["visit_date"].notna(),
        "sex_valid": labs["sex"].isin(["F", "M"]),
        "hba1c_in_range": labs["hba1c_pct"].between(4, 20) | labs["hba1c_pct"].isna(),
        "weight_in_range": labs["weight_kg"].between(30, 250) | labs["weight_kg"].isna(),
    })
    ```

2. Count the failures per rule with `validation_report = (~rules).sum()`, then name it: `validation_report.index.name = "rule"` and `validation_report.name = "failing_rows"`.
3. Display the failing rows with `labs[~rules.all(axis=1)]`.
    - Expect: six rows, D2007, D2009, D2010, D2012, D204, and D2013, and:

    ```text
    rule
    patient_id_format     1
    visit_date_present    2
    sex_valid             1
    hba1c_in_range        1
    weight_in_range       1
    Name: failing_rows, dtype: int64
    ```

4. Save with `validation_report.to_csv(VALIDATION_PATH)`, keeping the index.

> **Checkpoint: `output/validation_report.csv`**
> The header line `rule,failing_rows`, then the five rows above.

### 4.2 Flag unusual glucose values

Flag unusual values rather than deleting them: a fasting glucose of 412 mg/dL is rare but real, and the clinic wants to see it.

1. Compute the quartiles of `labs["glucose_mg_dl"]` with `quantile(0.25)` and `quantile(0.75)`, then `iqr = q3 - q1`, `lower = q1 - 1.5 * iqr`, and `upper = q3 + 1.5 * iqr`.
    - Expect: `Q1 118.0, Q3 162.0, fences 52.0 to 228.0`.
2. Add the flag: `labs["glucose_outlier"] = (labs["glucose_mg_dl"] < lower) | (labs["glucose_mg_dl"] > upper)`. A comparison with `NaN` is `False`, so a blank glucose is not flagged.
    - Expect: two flagged visits, D2006 (244) and D2014 (412).

Task 4.2 writes no file; `output/labs_clean.csv` carries the flag.

## Task 5: Save the cleaned table

### 5.1 Apply the decisions

1. Work on a copy, so `labs` stays as validated: `clean = labs.copy()`.
2. Keep only the visits that can be identified, a valid ID and a real date: `clean = clean[rules["patient_id_format"] & rules["visit_date_present"]]`. A visit that breaks only a range or sex rule stays.
    - Expect: 16 visits; D204, D2009, and D2013 are gone.
3. Blank the impossible values, keeping the visit: `clean["hba1c_pct"] = clean["hba1c_pct"].where(clean["hba1c_pct"].between(4, 20))`, and the same for `weight_kg` with 30 and 250. `where()` keeps a value where the condition is `True` and puts `NaN` elsewhere. Keep the other blanks as they are: nothing says what they were.
    - Expect: D2007's `hba1c_pct` and D2010's `weight_kg` are `NaN`.
4. Number the rows again with `clean = clean.reset_index(drop=True)`.

### 5.2 Validate before saving, and save

1. Collect the contract's checks in one Series:

    ```python
    checks = pd.Series({
        "one row per visit": not clean.duplicated(subset=["patient_id", "visit_date"]).any(),
        "patient IDs valid": clean["patient_id"].str.fullmatch(r"D[0-9]{4}").all(),
        "visit dates present": clean["visit_date"].notna().all(),
        "hba1c in range": clean["hba1c_pct"].dropna().between(4, 20).all(),
        "weight in range": clean["weight_kg"].dropna().between(30, 250).all(),
    })
    ```

2. Stop if any fails: `assert checks.all(), checks[~checks]`.
    - Expect: all five are `True`, and the `assert` prints nothing.
3. Save with `clean.to_csv(CLEAN_PATH, index=False)`.

> **Checkpoint: `output/labs_clean.csv`**
> 16 visits with 14 columns: those of `labs_coded.csv` and `glucose_outlier`. D2007's HbA1c and D2010's weight are blank, and `glucose_outlier` is `True` for D2006 and D2014 only.

### 5.3 Read it back

1. Read the file with the intended types: `back = pd.read_csv(CLEAN_PATH, dtype={"age": "Int64"}, parse_dates=["visit_date"])`.
    - Expect: `shape: (16, 14)`; `age` is `Int64`, `visit_date` is `datetime64[us]`, and `glucose_outlier` is `bool`.

## Check your work

- Run the checks any time, even partway through: `python3 check_assignment.py` in the terminal (with the environment active) prints your score and what to fix.
- When **Restart** and **Run All** finish without errors, the notebook's last cell runs the same checks.
- Both use the latest checks from the course repository, the same checks GitHub runs on each push. Commit `assignment.ipynb` and the `output/` files, then push.
- Expect: a passing run ends with `Score: 100/100`.

What each check looks for: [CHECKS.md](CHECKS.md)
