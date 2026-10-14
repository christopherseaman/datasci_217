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

Run the first Task 1 cell as it is; it sets `rows_read`.

- Expect: `shape: (59447, 8)`. `age` is `str` (some ages are `unknown`); `weight_kg` is `float64`, with `-999` counted as a weight.

1. Build `missing_report` with `pd.DataFrame()` from a dictionary of two Series, so it has one row per column of `labs`:
    - `missing_count`: the number of blanks in each column;
    - `missing_pct`: the percent of blanks in each column, rounded to 1 decimal.

    Lecture 05's card "Finding missing values" has both.
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

1. Drop the no-shows, visits with neither lab result, with `dropna()`: `subset` names the two lab columns, and `how="all"` drops a row only when both are missing. Assign the result back to `labs`. The cell then sets `rows_with_labs`.
    - Expect: `rows with labs: 58554`.
2. Set `exact_repeats` to the number of rows identical to an earlier row, with `duplicated()` and `.sum()`.
    - Expect: `exact repeats: 524`.
3. Keep the first copy of each repeat with `drop_duplicates()`, and assign the result back to `labs`. The cell then sets `rows_unique`.
    - Expect: `unique rows: 58030`.

## Task 2: Replace codes and convert types

1. Replace `-999` with `np.nan` in `weight_kg` and `glucose_mg_dl` only, with `labs.replace()` and a nested dictionary: column name, then `{-999: np.nan}`. Assign the result back to `labs`.
2. Convert `age` to numbers with `pd.to_numeric()`, letting `errors` turn `unknown` into a missing value, then to the nullable `Int64` dtype with `.astype()`. Assign it back to `labs["age"]`.
3. Convert `visit_date` with `pd.to_datetime()`, giving the `YYYY-MM-DD` layout as `format` and using `errors` so a date that does not exist becomes `NaT`. Assign it back.
    - Expect: `age` is `Int64` and `visit_date` is `datetime64[us]`.
4. The cell selects the five converted columns as `typed`. Build `typed_profile` with `pd.DataFrame()`, one row per column of `typed`, from these five Series in this order:
    - `dtype`: `typed.dtypes`, turned to text with `.astype(str)`;
    - `missing_count`: the blanks per column, as in Task 1.1;
    - `min` and `max`: each column's smallest and largest value;
    - `mean`: each column's mean, with `numeric_only=True` to skip the date column, rounded to 2 decimals.
5. Name the index `column`, as in Task 1.1.
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
6. Save with `typed_profile.to_csv(PROFILE_PATH)`.

> **Checkpoint: `output/typed_profile.csv`**
> Header `column,dtype,missing_count,min,max,mean` and the five rows above.

## Task 3: Standardize text and encode categories

### 3.1 Clean the text columns

1. Display `labs["clinic"].value_counts(dropna=False)`.
    - Expect: 22 rows: 21 spellings of four clinics, such as `mission`, ` Mission`, and `Mission `, plus `NaN`.
2. Clean the IDs with `.str.strip()` and `.str.upper()`, and assign the result back to `labs["patient_id"]`.
3. Clean the clinic names with `.str.strip()` and `.str.title()`, fill the blanks with `Unknown` using `.fillna()`, and assign the result back to `labs["clinic"]`.
4. Write a dictionary `gender_codes` from each lower-case spelling to its code:
    - `f` and `female` to `F`;
    - `m` and `male` to `M`;
    - `x`, `nb`, and `nonbinary` to `X`.
5. Clean `gender` with `.str.strip()` and `.str.lower()`, then `.map(gender_codes)`, and assign the result back. A label the dictionary does not list, such as `U`, `unknown`, or `?`, becomes `NaN`.

### 3.2 Bin HbA1c and count the categories

HbA1c bands: from 0 to below 5.7% is normal, 5.7% to 6.4% is prediabetes, and 6.5% to 20% is diabetes. A slip above 20, such as 64, gets no category.

1. Add the column `a1c_category` to `labs` by binning `hba1c_pct` with `pd.cut()`. Choose `bins` edges that match the bands, and pass the three band names as `labels`. Each bin includes its right edge, and HbA1c is recorded to one decimal.
2. Build `coded`, one 0/1 column per category of `clinic`, `gender`, and `a1c_category`, with `pd.get_dummies()` on those three columns and `dtype="int64"`. A blank gets 0 in every column.
3. Set `category_counts` to the visits per indicator column of `coded`, with `.sum()`. Name its index `indicator` and the Series `visits`, through `.index.name` and `.name`.
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

1. Set `q1` and `q3` to the quartiles of `glucose_mg_dl` with `quantile()`. Set `iqr` to their difference, and the fences `lower` and `upper` to 1.5 IQRs below Q1 and above Q3.
    - Expect: `Q1 102.0, Q3 153.0, fences 25.5 to 229.5`.
2. Flag the unusual values, keeping them in the table: add `glucose_outlier`, `True` where `glucose_mg_dl` is below `lower` or above `upper`. Join the two comparisons with `|`, each in parentheses. A missing glucose is not an outlier.
3. Build `rules` with `pd.DataFrame()`: one `True`/`False` column per rule of the data contract, `True` where the visit passes. Pick each test from Lecture 05's validation card. Use these six names in this order:
    - `patient_id_format`: the whole ID matches `r"D[0-9]{5}"`, and a missing ID fails;
    - `visit_date_present`: the date is not missing;
    - `gender_valid`: the code is `F`, `M`, or `X`, so a blank fails;
    - `hba1c_in_range`: the value is from 4 to 20, or missing;
    - `weight_in_range`: the weight is from 30 to 250, or missing;
    - `glucose_within_fences`: the visit is not a glucose outlier.
4. Set `validation_report` to the number of failing rows per rule: flip `rules` with `~`, then `.sum()`. Name its index `rule` and the Series `failing_rows`.
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

1. Set `clean` to a copy of `labs` with `.copy()`.
2. Keep the visits that pass both the ID rule and the date rule from Task 4: join the two `rules` columns with `&` and select those rows. Assign the result back to `clean`.
3. Before blanking, count the impossible values among the kept visits in `clean`. Set `hba1c_outside` to the number of `hba1c_pct` values below 4 or above 20, joining the two comparisons with `|` as in Task 4 step 2. Set `weight_outside` the same way with 30 and 250.
4. Blank those values, keeping the visit: `.where()` keeps the values where its mask is `True` and turns the rest into `NaN`. Give it a mask of the 4 to 20 range, assign the result back to `clean["hba1c_pct"]`, and do the same for `weight_kg` with 30 and 250.
5. Number the rows again with `reset_index()`, using `drop` so the old index does not become a column. Assign the result back to `clean`.
6. Build `decision_log` with `pd.Series()` from a dictionary of six counts, in this order, and `name="rows"`:
    - `rows_read`, `rows_with_labs`, and `rows_unique`, the variables from Task 1;
    - `visits_dropped`: `len(labs)` minus `len(clean)`;
    - `hba1c_blanked` and `weight_blanked`: `hba1c_outside` and `weight_outside`.
7. Name the index `decision`.
    - Expect: `shape: (57618, 10)`; `rows_read` 59447, `rows_with_labs` 58554, `rows_unique` 58030, then `visits_dropped` 412 (2 visits fail both rules), `hba1c_blanked` 48, and `weight_blanked` 70. `hba1c_blanked` is one fewer than Task 4's 49 because one bad HbA1c sits on a visit dropped in step 2.
8. Save with `decision_log.to_csv(DECISION_PATH)`.

> **Checkpoint: `output/decision_log.csv`**
> Header `decision,rows` and the six counts above.

### 5.2 Validate, then save the summary

1. Build `checks` with `pd.Series()` from a dictionary of five `True`/`False` values, one per rule of the data contract, each tested on `clean`. Each one reduces a column test to one value, as in Lecture 05's card "Validation checks":
    - `one row per visit`: `duplicated()` with `subset` set to both key columns flags no row (`.sum()` is 0);
    - `patient IDs valid`: every ID matches `r"D[0-9]{5}"`;
    - `visit dates present`: no date is missing;
    - `hba1c in range`: every recorded HbA1c is from 4 to 20;
    - `weight in range`: every recorded weight is from 30 to 250.
2. Stop if any check fails, with the `assert` line from Lecture 05's card "Validation checks".
    - Expect: `checks` shows five `True` values, and the `assert` prints nothing. A failure names the check; fix that step and run the cells again.
3. Set `clean_summary` to `describe()` of the four number columns of `clean` (`age`, `weight_kg`, `hba1c_pct`, `glucose_mg_dl`), rounded to 2 decimals, then turned with `.T` so each column is a row.
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

1. The cell sets `parquet_path`. Save `clean` there with `to_parquet()`.
2. Set `clean_back` to the file read back with `pd.read_parquet()`. The cell prints its dtypes and whether it equals `clean`.
    - Expect: `age` is `Int64`, `visit_date` is `datetime64[us]`, `a1c_category` is `category`, `glucose_outlier` is `bool`, and `same as clean: True`.

> **Checkpoint: `data/labs_clean.parquet`**
> Not committed or checked; about 0.5 MB, with the 57618 rows and 10 columns of `clean`.

## Check your work

- Run `python3 check_assignment.py` in the terminal any time; it prints your score and what to fix.
- The notebook's last cell, `run_checks()`, runs the same checks after **Restart** and **Run All**.
- GitHub runs the latest checks on each push. Commit `assignment.ipynb` and the `output/` files, then push.
- Expect: `Score: 100/100`.

What each check looks for: [CHECKS.md](CHECKS.md)
