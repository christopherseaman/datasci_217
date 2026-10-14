# Assignment 05: Clean a Lab Export

## Overview

Clean a diabetes clinic network's lab export and save a small summary of each stage for the checks.

`generate_data.py` writes `data/lab_visits.csv`, a synthetic export of about 59,000 visits by 18,000 patients, most seen about every three months. It has blanks, rows sent twice, `-999` for a measurement not taken, ages typed as `unknown`, dates that do not exist, IDs and labels spelled several ways, and dropped decimal points (`64` for 6.4).

```text
patient_id,visit_date,clinic,gender,age,weight_kg,hba1c_pct,glucose_mg_dl
D10632,2024-01-02,Sunset,female,72,73.4,6.3,118
D10720,2024-01-02,Sunset,F,46,96.0,5.9,106
D11156,2024-01-02,Excelsior,M,71,89.6,5.2,62
```

The cleaned table follows this data contract. One row is one visit, identified by `patient_id` and `visit_date` together.

| Column | Meaning | dtype | Rule |
| --- | --- | --- | --- |
| `patient_id` | study patient | `str` | `D` + five digits |
| `visit_date` | visit day | `datetime64` | a real calendar date; never blank |
| `clinic` | clinic name | `str` | one spelling per clinic; `Unknown` when the export left it blank |
| `gender` | gender code | `str` | `F`, `M`, or `X` (nonbinary); otherwise blank |
| `age` | age at visit (years) | `Int64` | blank allowed |
| `weight_kg` | weight (kg) | `float64` | 30 to 250 when present |
| `hba1c_pct` | **HbA1c** (%), a three-month average of blood sugar | `float64` | 4 to 20 when present |
| `glucose_mg_dl` | fasting glucose (mg/dL) | `float64` | blank allowed; unusual values flagged, not deleted |

## Setup

1. Fork the assignment repository on GitHub and clone your fork as in Lecture 01 (Command Palette → **Git: Clone**). Open the cloned folder itself.
2. In the integrated terminal, create the environment and install the supplied packages. Do not run `uv init`.

    ```bash
    uv venv --seed
    source .venv/bin/activate
    uv sync
    ```

    - Expect: `uv sync` lists `+ pandas==3.0.5` and `+ pyarrow==25.0.0`.
3. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`.
4. Run the first code cell. It runs `generate_data.py` when `data/` is missing and checks the file against `data_manifest.json`.
    - Expect: `NumPy: 2.3.3`, `pandas: 3.0.5`, and `data/lab_visits.csv matches data_manifest.json`.

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── generate_data.py        # supplied: writes data/lab_visits.csv
├── data_manifest.json      # supplied: that file's checksum
├── data/
│   ├── lab_visits.csv      # written by the first cell; not committed
│   └── labs_clean.parquet  # you generate in Task 5.3; not committed
├── pyproject.toml, uv.lock, .python-version  # supplied: packages and Python 3.13
├── CHECKS.md               # supplied: what each check looks for
├── check_assignment.py     # supplied: run it to check your work
├── grading.py, _value_checks.py, test_assignment.py, .github/  # supplied: the checks; keep unchanged
└── output/
    ├── missing_report.csv      # you generate in Task 1.1
    ├── typed_profile.csv       # you generate in Task 2
    ├── category_counts.csv     # you generate in Task 3.2
    ├── validation_report.csv   # you generate in Task 4
    ├── decision_log.csv        # you generate in Task 5.1
    └── clean_summary.csv       # you generate in Task 5.2
```

Each task changes `labs`, so run the cells in order; after a mistake, **Restart** and **Run All**.

## Task 1: Find missing values and repeats

### 1.1 Count the gaps

The first Task 1 cell reads the file with `labs = pd.read_csv(DATA_PATH)` and sets `rows_read = len(labs)`.

- Expect: `shape: (59447, 8)`. `age` is `str` (some ages are `unknown`); `weight_kg` is `float64`, with `-999` counted as a weight.

1. Build a table with one row per column of `labs`:

    ```python
    missing_report = pd.DataFrame({
        "missing_count": labs.isna().sum(),
        "missing_pct": (labs.isna().mean() * 100).round(1),
    })
    ```

2. Name the index with `missing_report.index.name = "column"`.
    - Expect:

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

3. Save with `missing_report.to_csv(MISSING_PATH)`.

> **Checkpoint: `output/missing_report.csv`**
> Header `column,missing_count,missing_pct` and the eight rows above.

### 1.2 Drop no-lab visits and exact repeats

1. Drop the no-shows, visits with neither lab result: `labs = labs.dropna(subset=["hba1c_pct", "glucose_mg_dl"], how="all")`. The cell then sets `rows_with_labs = len(labs)`.
    - Expect: `rows with labs: 58554`.
2. Set `exact_repeats` to `labs.duplicated().sum()`, the rows identical to an earlier row.
    - Expect: `exact repeats: 524`.
3. Keep the first copy of each repeat: `labs = labs.drop_duplicates()`. The cell then sets `rows_unique = len(labs)`.
    - Expect: `unique rows: 58030`.

## Task 2: Replace codes and convert types

1. Replace `-999` with a missing value in `weight_kg` and `glucose_mg_dl` only:

    ```python
    labs = labs.replace({"weight_kg": {-999: np.nan}, "glucose_mg_dl": {-999: np.nan}})
    ```

2. Convert `age`: `labs["age"] = pd.to_numeric(labs["age"], errors="coerce").astype("Int64")`.
3. Convert `visit_date`: `labs["visit_date"] = pd.to_datetime(labs["visit_date"], format="%Y-%m-%d", errors="coerce")`.
    - Expect: `age` is `Int64` and `visit_date` is `datetime64[us]`.
4. Profile the five converted columns, one row per column:

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

    The `max` values 103.0 and 1295.0 are decimal slips; Task 5 blanks them.
5. Save with `typed_profile.to_csv(PROFILE_PATH)`.

> **Checkpoint: `output/typed_profile.csv`**
> Header `column,dtype,missing_count,min,max,mean` and the five rows above.

## Task 3: Standardize text and encode categories

### 3.1 Clean the text columns

1. Display `labs["clinic"].value_counts(dropna=False)`.
    - Expect: 21 spellings of four clinics, such as `mission`, ` Mission`, and `Mission `, plus `NaN`.
2. Clean the IDs: `labs["patient_id"] = labs["patient_id"].str.strip().str.upper()`.
3. Clean the clinic names with `labs["clinic"].str.strip().str.title()`, then fill the blanks with `.fillna("Unknown")`, and assign the result back to `labs["clinic"]`.
4. Map every spelling of gender to one code. A label the dictionary does not list, such as `U`, `unknown`, or `?`, becomes `NaN`:

    ```python
    gender_codes = {"f": "F", "female": "F", "m": "M", "male": "M", "x": "X", "nb": "X", "nonbinary": "X"}
    labs["gender"] = labs["gender"].str.strip().str.lower().map(gender_codes)
    ```

### 3.2 Bin HbA1c and count the categories

HbA1c bands: below 5.7% is normal, 5.7% to 6.4% is prediabetes, and 6.5% or more is diabetes.

1. Add `a1c_category` with `pd.cut(labs["hba1c_pct"], bins=[0, 5.6, 6.4, 20], labels=["normal", "prediabetes", "diabetes"])`. Each bin includes its right edge, so 5.6 is normal; a slip such as 64 gets no category.
2. Build one 0/1 column per category: `coded = pd.get_dummies(labs[["clinic", "gender", "a1c_category"]], dtype="int64")`. A blank gets 0 in every column.
3. Count each indicator's visits with `category_counts = coded.sum()`, then name it: `category_counts.index.name = "indicator"` and `category_counts.name = "visits"`.
    - Expect:

    ```text
    indicator
    clinic_Bayview               8698
    clinic_Excelsior            12952
    clinic_Mission              20225
    clinic_Sunset               15923
    clinic_Unknown                232
    gender_F                    29012
    gender_M                    26876
    gender_X                     1765
    a1c_category_normal         14045
    a1c_category_prediabetes    14827
    a1c_category_diabetes       25763
    Name: visits, dtype: int64
    ```

4. Save with `category_counts.to_csv(CATEGORY_PATH)`.

> **Checkpoint: `output/category_counts.csv`**
> Header `indicator,visits` and the eleven rows above.

## Task 4: Flag outliers and count rule failures

1. Compute the quartiles of `labs["glucose_mg_dl"]` with `quantile(0.25)` and `quantile(0.75)`, then `iqr = q3 - q1`, `lower = q1 - 1.5 * iqr`, and `upper = q3 + 1.5 * iqr`.
    - Expect: `Q1 102.0, Q3 153.0, fences 25.5 to 229.5`.
2. Flag the unusual values, keeping them in the table: `labs["glucose_outlier"] = (labs["glucose_mg_dl"] < lower) | (labs["glucose_mg_dl"] > upper)`.
3. Build one `True`/`False` column per rule, `True` where the visit passes; a missing value passes the range rules:

    ```python
    rules = pd.DataFrame({
        "patient_id_format": labs["patient_id"].str.fullmatch(r"D[0-9]{5}", na=False),
        "visit_date_present": labs["visit_date"].notna(),
        "gender_valid": labs["gender"].isin(["F", "M", "X"]),
        "hba1c_in_range": labs["hba1c_pct"].between(4, 20) | labs["hba1c_pct"].isna(),
        "weight_in_range": labs["weight_kg"].between(30, 250) | labs["weight_kg"].isna(),
        "glucose_within_fences": ~labs["glucose_outlier"],
    })
    ```

4. Count the failures per rule with `validation_report = (~rules).sum()`, then name it: `validation_report.index.name = "rule"` and `validation_report.name = "failing_rows"`.
    - Expect:

    ```text
    rule
    patient_id_format        119
    visit_date_present       295
    gender_valid             377
    hba1c_in_range            49
    weight_in_range           70
    glucose_within_fences    486
    Name: failing_rows, dtype: int64
    ```

5. Save with `validation_report.to_csv(VALIDATION_PATH)`.

> **Checkpoint: `output/validation_report.csv`**
> Header `rule,failing_rows` and the six rows above.

## Task 5: Clean and summarize

### 5.1 Apply the decisions and log them

1. Work on a copy: `clean = labs.copy()`.
2. Keep the visits with a valid ID and a real date: `clean = clean[rules["patient_id_format"] & rules["visit_date_present"]]`.
3. Before blanking, count the impossible values among the kept visits: `hba1c_outside = ((clean["hba1c_pct"] < 4) | (clean["hba1c_pct"] > 20)).sum()`, and `weight_outside` the same way with 30 and 250.
4. Blank those values, keeping the visit: `clean["hba1c_pct"] = clean["hba1c_pct"].where(clean["hba1c_pct"].between(4, 20))`, and the same for `weight_kg` with 30 and 250.
5. Number the rows again with `clean = clean.reset_index(drop=True)`.
6. Log the row counts and the decisions:

    ```python
    decision_log = pd.Series({
        "rows_read": rows_read, "rows_with_labs": rows_with_labs, "rows_unique": rows_unique,
        "visits_dropped": len(labs) - len(clean), "hba1c_blanked": hba1c_outside, "weight_blanked": weight_outside,
    }, name="rows")
    decision_log.index.name = "decision"
    ```

    - Expect: `shape: (57618, 10)`; the three counts from Task 1.2, then `visits_dropped` 412 (2 visits fail both rules), `hba1c_blanked` 48, and `weight_blanked` 70.
7. Save with `decision_log.to_csv(DECISION_PATH)`.

> **Checkpoint: `output/decision_log.csv`**
> Header `decision,rows` and the six counts above.

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
    - Expect: no output. A failure names the check; fix that step and run the cells again.
3. Summarize the cleaned number columns: `clean_summary = clean[["age", "weight_kg", "hba1c_pct", "glucose_mg_dl"]].describe().round(2).T`. `.T` makes each column a row.
    - Expect:

    ```text
                     count    mean    std   min    25%    50%    75%    max
    age            57406.0   55.13   17.7  25.0   40.0   55.0   70.0   86.0
    weight_kg      56608.0   85.94  17.09  30.0   74.2   86.0   97.7  250.0
    hba1c_pct      54252.0    6.64   1.32   4.0    5.6    6.3    7.5   20.0
    glucose_mg_dl  54564.0  129.35  36.79  62.0  102.0  124.0  153.0  412.0
    ```

4. Save with `clean_summary.to_csv(SUMMARY_PATH)`.

> **Checkpoint: `output/clean_summary.csv`**
> Header `,count,mean,std,min,25%,50%,75%,max` and the four rows above.

### 5.3 Save the cleaned table as Parquet

Parquet keeps each column's dtype.

1. The cell sets `parquet_path = Path("data") / "labs_clean.parquet"`. Save with `clean.to_parquet(parquet_path)`.
2. Read it back with `clean_back = pd.read_parquet(parquet_path)`, then print `clean_back.dtypes` and `clean_back.equals(clean)`.
    - Expect: `age` is `Int64`, `visit_date` is `datetime64[us]`, `a1c_category` is `category`, `glucose_outlier` is `bool`, and `same as clean: True`.

> **Checkpoint: `data/labs_clean.parquet`**
> About 0.5 MB, with 57618 rows and the 10 columns of `clean`.

## Check your work

- Run `python3 check_assignment.py` in the terminal any time; it prints your score and what to fix.
- The notebook's last cell, `run_checks()`, runs the same checks after **Restart** and **Run All**.
- GitHub runs the latest checks on each push. Commit `assignment.ipynb` and the `output/` files, then push.
- Expect: `Score: 100/100`.

What each check looks for: [CHECKS.md](CHECKS.md)
