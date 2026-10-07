---
notion:
  title_line: "# DLC: Jupyter Workflows and Advanced Pandas Operations"
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-8119-83f4-fb2c4b73430e"
  url: "https://app.notion.com/p/3d2d9fdd1a1a811983f4fb2c4b73430e"
---

# DLC: Jupyter Workflows and Advanced Pandas Operations

# Running Notebooks Non-Interactively

- `jupyter nbconvert --execute` runs every cell in order from the command line, as a reproducibility check or one step in a batch workflow.
- It comes with JupyterLab; in a project without it, add it once with `uv add nbconvert`.
- It writes a separate output notebook, so the source stays unchanged.

```bash
jupyter nbconvert --execute --to notebook \
    --output executed_analysis.ipynb analysis.ipynb
```

A cell error makes the command fail with a nonzero exit status, so `set -euo pipefail` stops a script there instead of running the next notebook. Save this as `run_notebooks.sh` and run it with `bash run_notebooks.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

jupyter nbconvert --execute --to notebook \
    --output executed_prepare.ipynb prepare.ipynb
jupyter nbconvert --execute --to notebook \
    --output executed_analyze.ipynb analyze.ipynb
```

- Avoid `--inplace` unless overwriting the source is deliberate and recoverable.
- Avoid `--allow-errors` in validation: its output notebook can contain failed cells.
- The working directory, kernel, and installed packages all affect the result.

# More Ways to Filter Rows

These build the same kind of mask with less typing, or select by position instead of label.

## Reference Card: Filter shortcuts

- `df.query("temp_c >= 37 and age < 50")`: Filter with an expression string that names columns directly; `and`/`or` replace `&`/`|`. Returns a filtered `DataFrame`.
- `df["col"].between(left, right)`: Test an inclusive range; returns a Boolean `Series`.
- `df.isin([...])`: Test every cell against a list; returns a Boolean `DataFrame`.
- `df.iloc[mask.to_numpy()]`: `.iloc` accepts positions only, so convert a Boolean Series to a plain array first.

## Code Snippet: Filter shortcuts

```python
visits = pd.DataFrame(
    {"age": [34, 58, 41], "temp_c": [36.8, 38.1, 37.2], "clinic": ["North", "South", "East"]},
    index=pd.Index(["P001", "P002", "P003"], name="patient_id"),
)
display(visits.query("temp_c >= 37 and age < 50"))
display(visits.loc[visits["age"].between(40, 60)])

over_40 = visits["age"] > 40
# visits.iloc[over_40] raises ValueError: iLocation based boolean indexing cannot use an indexable as a mask
display(visits.iloc[over_40.to_numpy()])
```

```text
            age  temp_c clinic
patient_id                    
P003         41    37.2   East
            age  temp_c clinic
patient_id                    
P002         58    38.1  South
P003         41    37.2   East
            age  temp_c clinic
patient_id                    
P002         58    38.1  South
P003         41    37.2   East
```

# Reindexing, Aligning, and Arithmetic Methods

Arithmetic lines up labels automatically; these tools make the label set explicit before you combine tables.

## Reference Card: Reindexing and arithmetic methods

- `s.reindex(labels)`: Conform to a list of labels, in that order; a new label gets `NaN` (or `fill_value=`), and a label left out is dropped.
- `df.reindex(columns=[...])`: The same for columns.
- `a.align(b, join="inner")`: Return both objects conformed to a shared label set; `join="outer"` keeps every label.
- `df.add(other, fill_value=0)`: On DataFrames, a cell missing from one side counts as `0`; a cell missing from both stays `NaN`.
- `df.radd()`, `.rsub()`, `.rmul()`, `.rdiv()`, `.rpow()`: Reversed arguments, so `df.rdiv(1)` is `1 / df`.
- `df.floordiv()`, `df.pow()`: Method forms of `//` and `**`, each with `fill_value=` and `axis=`.

## Code Snippet: Reindex to a roster

```python
sbp = pd.Series([128, 142, 150], index=["P001", "P002", "P003"])
roster = ["P001", "P002", "P003", "P004"]
display(sbp.reindex(roster))
```

```text
P001    128.0
P002    142.0
P003    150.0
P004      NaN
dtype: float64
```

`P004` is on the roster without a reading, so the gap is visible instead of the patient silently missing.

## Code Snippet: Keep only shared labels

```python
week4 = pd.Series([124, 136, 131], index=["P001", "P002", "P004"])
left, right = sbp.align(week4, join="inner")
display(right - left)
```

```text
P001   -4
P002   -6
dtype: int64
```

## Code Snippet: Fill missing cells on both axes

```python
doses = pd.DataFrame({"am": [1, 2], "pm": [1, 1]}, index=["P001", "P002"])
extra = pd.DataFrame({"am": [1], "noon": [1]}, index=["P002"])
display(doses + extra)
display(doses.add(extra, fill_value=0))
```

```text
       am  noon  pm
P001  NaN   NaN NaN
P002  3.0   NaN NaN
       am  noon   pm
P001  1.0   NaN  1.0
P002  3.0   1.0  1.0
```

`P001`'s `noon` cell exists in neither table, so it stays `NaN` even with `fill_value=0`.

## Assignment Under Copy-on-Write

- With Copy-on-Write, changing a subset never changes the DataFrame it came from.
- So chained assignment such as `visits[visits["temp_c"] >= 38]["flag"] = "fever"` never updates `visits`.
- Update the owner in one statement: `.loc[row_mask, column] = value`, or `.iloc[row_positions, column_positions] = value`.
- For a separate result, transform the subset and assign the returned object to a name.

Details: the [pandas 3.0 release notes](https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v3.0.0.html), [string-dtype migration guide](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html), and [Copy-on-Write guide](https://pandas.pydata.org/docs/user_guide/copy_on_write.html).

# Function Application and Method Chaining

- Reach for these when vectorized arithmetic cannot express the logic, or to write a readable chain of steps.
- A `lambda` is a one-line function without a name: `lambda d: d["weight_kg"] / d["height_m"] ** 2` takes `d` and returns the expression.

## Reference Card: Apply, map, and column helpers

- `df.apply(func)`: Column-wise by default; add `axis="columns"` for row-wise logic.
- `series.map(func)`: Element-level transform; also accepts a dict or Series as a lookup.
- `df.map(func)`: Element-wise DataFrame transform; slow on large tables.
- `df.assign(name=lambda d: ...)`: A new DataFrame with added columns; each `lambda` sees the columns built so far, and the original is unchanged.
- `df.insert(loc, column, value)`: Insert a column at integer position `loc`; changes `df` itself and returns `None`.
- `df.eval("new = expression")`: Compute a column from an expression string that names columns directly; returns a new DataFrame.
- `.assign()`, `.pipe()`, `.rename()`: Chain helpers for step-by-step pipelines.

## Code Snippet: Apply, map, and chain

```python
doses = pd.DataFrame({"am_mg": [5, 10, 20], "pm_mg": [5, 5, 10]}, index=["P001", "P002", "P003"])
display(doses.apply(lambda col: col.max() - col.min()))
display(doses.apply(lambda row: row.sum(), axis="columns"))
display(doses.map(lambda x: f"{x} mg"))

share = (
    doses.assign(daily_mg=lambda d: d["am_mg"] + d["pm_mg"])
         .pipe(lambda d: d / d["daily_mg"].max())
)
display(share)
```

```text
am_mg    15
pm_mg     5
dtype: int64
P001    10
P002    15
P003    30
dtype: int64
      am_mg  pm_mg
P001   5 mg   5 mg
P002  10 mg   5 mg
P003  20 mg  10 mg
         am_mg     pm_mg  daily_mg
P001  0.166667  0.166667  0.333333
P002  0.333333  0.166667  0.500000
P003  0.666667  0.333333  1.000000
```

## Adding Columns with `assign()`, `insert()`, and `eval()`

```python
patients = pd.DataFrame({
    "patient_id": ["P001", "P002", "P003"],
    "weight_kg": [70.0, 82.5, 64.0],
    "height_m": [1.75, 1.80, 1.62],
})
augmented = patients.assign(
    bmi=lambda d: d["weight_kg"] / d["height_m"] ** 2,
    overweight=lambda d: d["bmi"] >= 25,
)
display(augmented)
patients.insert(1, "weight_lb", patients["weight_kg"] * 2.2046)
print(patients.columns.tolist())
display(patients.eval("height_cm = height_m * 100"))
```

```text
  patient_id  weight_kg  height_m        bmi  overweight
0       P001       70.0      1.75  22.857143       False
1       P002       82.5      1.80  25.462963        True
2       P003       64.0      1.62  24.386526       False
['patient_id', 'weight_lb', 'weight_kg', 'height_m']
  patient_id  weight_lb  weight_kg  height_m  height_cm
0       P001   154.3220       70.0      1.75      175.0
1       P002   181.8795       82.5      1.80      180.0
2       P003   141.0944       64.0      1.62      162.0
```

`assign()` left `patients` unchanged; `insert()` changed it in place.

# More Ranking Options

## Reference Card: Ranking

- `method="average"` (default), `"min"`, `"max"`: Ties share the mean, best, or worst place.
- `method="first"`: Ties broken by order of appearance; every rank is distinct.
- `method="dense"`: Like `"min"`, but the next value takes the next whole number (1, 1, 2), with no gap.
- `pct=True`: Rank as a fraction of the count, a percentile.
- `df.rank(axis="columns")`: Rank across each row instead of down each column.

## Code Snippet: Tie rules side by side

```python
s = pd.Series([142, 118, 142, 130], index=["P003", "P001", "P002", "P004"])
display(pd.DataFrame({
    "average": s.rank(ascending=False),
    "first": s.rank(ascending=False, method="first"),
    "dense": s.rank(ascending=False, method="dense"),
    "pct": s.rank(pct=True),
}))
```

```text
      average  first  dense    pct
P003      1.5    1.0    1.0  0.875
P001      4.0    4.0    3.0  0.250
P002      1.5    2.0    1.0  0.875
P004      3.0    3.0    2.0  0.500
```

# Covariance

**Covariance** is the unscaled version of Pearson's _r_, in the product of the two columns' units (here mmHg²), so its size depends on the units.

## Reference Card: Covariance

- `df["a"].cov(df["b"])`: Covariance of two columns.
- `df.cov()`: Covariance of every pair of numeric columns; the diagonal holds each column's variance.
- `df.corrwith(series)`: Correlation of every column with one Series.

## Code Snippet: Covariance of blood-pressure visits

```python
bp = pd.DataFrame(
    {"baseline": [128, 142, 150], "week_4": [124, 136, 138], "week_8": [121, 138, 131]},
    index=["P001", "P002", "P003"],
)
display(bp.cov())
display(bp.corrwith(bp["baseline"]))
```

```text
          baseline     week_4  week_8
baseline     124.0  82.000000    67.0
week_4        82.0  57.333333    55.0
week_8        67.0  55.000000    73.0
baseline    1.000000
week_4      0.972522
week_8      0.704211
dtype: float64
```

# Handling Duplicate Index Labels

A patient seen twice can have two rows with one label. Selecting that label returns several values, and summaries count both.

## Reference Card: Duplicate index labels

- `df.index.is_unique`: `False` when any label repeats.
- `s["P001"]`: A `Series` when the label repeats, a single value when it does not.
- `df.index.duplicated()`: Boolean mask, `True` for each repeat after the first.
- `df.reset_index()`: Turns the labels into a column with a fresh, unique RangeIndex.

## Code Snippet: Select repeated labels

```python
temps = pd.Series(
    [36.8, 37.9, 37.2, 38.4, 36.6],
    index=["P001", "P001", "P002", "P002", "P003"],
    name="temp_c",
)
print(temps.index.is_unique)
display(temps["P001"])
print(temps["P003"])
display(temps[~temps.index.duplicated()])
```

```text
False
P001    36.8
P001    37.9
Name: temp_c, dtype: float64
36.6
P001    36.8
P002    37.2
P003    36.6
Name: temp_c, dtype: float64
```

# Other File Formats and Databases

pandas reads and writes many formats with one pattern: `pd.read_FORMAT()` returns a DataFrame, and `df.to_FORMAT()` writes one. The snippets below use this table:

```python
visits = pd.DataFrame({
    "patient_id": ["P001", "P002", "P003"],
    "clinic": ["North", "South", "North"],
    "temp_c": [36.8, 38.1, 37.2],
})
```

## Reference Card: Readers and writers

| Format | Read | Write | Needs |
| --- | --- | --- | --- |
| JSON | `pd.read_json(path)` | `df.to_json(path, orient="records")` | Nothing extra |
| Excel | `pd.read_excel(path, sheet_name=...)` | `df.to_excel(path, sheet_name=..., index=False)` | `uv add openpyxl` |
| Pickle | `pd.read_pickle(path)` | `df.to_pickle(path)` | Nothing extra; Python only, and only from sources you trust |
| Feather | `pd.read_feather(path)` | `df.to_feather(path)` | `pyarrow`, as for Parquet |
| HDF5 | `pd.read_hdf(path, key)` | `df.to_hdf(path, key=...)` | `uv add tables` |
| SQL database | `pd.read_sql(query, con)` | `df.to_sql(table, con, index=False)` | `sqlite3` comes with Python; other databases need a driver |

## JSON

- **JSON** (JavaScript Object Notation) is the text format most web APIs return: lists in `[...]` and key-value objects in `{...}`, like Python lists and dicts.
- `orient="records"` writes one object per row, the most common shape from an API.

```python
visits.to_json("visits.json", orient="records", indent=2)
print(open("visits.json").read()[:60])
display(pd.read_json("visits.json"))
```

```text
[
  {
    "patient_id":"P001",
    "clinic":"North",
    "te
  patient_id clinic  temp_c
0       P001  North    36.8
1       P002  South    38.1
2       P003  North    37.2
```

Deeply nested JSON needs flattening first; see `pd.json_normalize()`.

## Excel

- Excel needs the `openpyxl` package in the kernel's environment (`uv add openpyxl` locally, `%pip install openpyxl` in Colab); without it pandas raises `ModuleNotFoundError`.
- `pd.ExcelWriter` writes several sheets to one workbook.
- `sheet_name=None` reads every sheet into a dictionary of DataFrames keyed by sheet name.

```python
with pd.ExcelWriter("clinic.xlsx") as writer:
    visits.to_excel(writer, sheet_name="visits", index=False)
    visits.loc[visits["temp_c"] >= 38.0].to_excel(writer, sheet_name="fever", index=False)

sheets = pd.read_excel("clinic.xlsx", sheet_name=None)
print(list(sheets))
display(sheets["fever"])
```

```text
['visits', 'fever']
  patient_id clinic  temp_c
0       P002  South    38.1
```

## Binary Formats

- **Parquet** is the usual choice.
- **Feather**: a fast format from the same Arrow project, for short-term files.
- **Pickle**: saves any Python object exactly, but only Python reads it, a newer pandas may not, and loading one can run code, so never load one from an untrusted source.
- **HDF5**: many tables in one file; needs the `tables` package.

```python
visits.to_feather("visits.feather")
visits.to_pickle("visits.pkl")
print(pd.read_feather("visits.feather").equals(visits))
print(pd.read_pickle("visits.pkl").equals(visits))
```

```text
True
True
```

## SQL Databases

- Hospital records usually live in a **relational database**, queried with SQL.
- `pd.read_sql(query, con)` runs a query and returns the result as a DataFrame.
- Python's built-in `sqlite3` opens a SQLite database, a single file; a server database such as PostgreSQL needs its own driver or SQLAlchemy (`uv add sqlalchemy`).

```python
import sqlite3

con = sqlite3.connect("clinic.db")
visits.to_sql("visits", con, index=False, if_exists="replace")
fevers = pd.read_sql("SELECT patient_id, temp_c FROM visits WHERE temp_c >= 38.0", con)
con.close()
display(fevers)
```

```text
  patient_id  temp_c
0       P002    38.1
```

`if_exists="replace"` overwrites a table of the same name, so rerunning the cell is safe in this example and destructive in a real database.

# Large and Messy CSV Files

## Reading Large Files in Chunks

`chunksize=` makes `read_csv()` return one DataFrame of that many rows at a time, so a file larger than memory is processed piece by piece.

```python
chunk_iter = pd.read_csv("all_visits.csv", chunksize=10000)
results = []

for chunk in chunk_iter:
    fevers = chunk[chunk["temp_c"] >= 38.0].groupby("clinic").size()
    results.append(fevers)

fevers_by_clinic = pd.concat(results).groupby(level=0).sum()
```

- Use chunking when a file exceeds memory or you only need totals.
- Sums and counts add up across chunks; chunk means cannot be averaged.

## Turning Off the Default Missing Markers

The defaults sometimes misfire: a country column with `NA` for Namibia reads as missing. `keep_default_na=False` turns every default marker off, blanks included, so only your `na_values` count as missing.
