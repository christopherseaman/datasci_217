# Bonus: Optional pandas Extensions

This file extends Lecture 04's labeled-data model. None of these sections is required by the Lecture 04 demos, assignment, or Lecture 05.

Return to [README.md](README.md) for the required notebook-state, Series/DataFrame, selection, sorting, and portable CSV workflow.

## Label alignment and explicit broadcasting

NumPy arithmetic is driven primarily by compatible shapes. pandas arithmetic also considers labels.

When two Series use different index labels, pandas aligns equal labels before calculating. The result contains the union of labels. `NaN` marks a label that had no partner for that calculation; handling such markers is outside this optional example.

```python
import pandas as pd

left = pd.Series([10, 20, 30], index=["north", "south", "west"])
right = pd.Series([1, 2, 3], index=["south", "west", "east"])

print(left + right)
```

Only `south` and `west` have partners in both Series.

Arithmetic methods can make the intended axis explicit when combining a DataFrame and a Series:

```python
measurements = pd.DataFrame(
    {
        "baseline": [10, 20, 30],
        "follow_up": [20, 30, 40],
    },
    index=["north", "south", "west"],
)

column_offsets = pd.Series(
    {"baseline": 2, "follow_up": 5}
)

adjusted = measurements.sub(column_offsets, axis="columns")
print(adjusted)
```

The explicit `axis="columns"` says that the Series labels should match DataFrame column labels. This is optional label reasoning, not a new required arithmetic pattern.

## Ranking after deterministic sorting

Sorting changes row order. Ranking instead returns an order number for each original row.

```python
scores = pd.Series(
    [88, 95, 88, 72],
    index=["obs-001", "obs-002", "obs-003", "obs-004"],
)

print(scores.rank(ascending=False))
print(scores.rank(ascending=False, method="min"))
print(scores.rank(ascending=False, method="dense"))
```

Tie methods answer different questions:

- the default gives tied values their average rank;
- `method="min"` gives every tie the best occupied rank; and
- `method="dense"` does not leave gaps after a tie.

Choose and document the tie rule when rank values will be interpreted or shared.

## Duplicate index labels

Lecture 04 uses unique row labels so one label identifies one row. pandas also permits duplicate index labels, but selection can then return a different shape depending on the label.

```python
readings = pd.Series(
    [10, 12, 20],
    index=["north", "north", "south"],
    name="reading",
)

print("unique index:", readings.index.is_unique)
print("north result:")
print(readings.loc["north"])
print("south result:")
print(readings.loc["south"])
```

`readings.loc["north"]` returns a Series because two rows share that label. `readings.loc["south"]` returns one scalar value. This shape change is why the required Lecture 04 examples use unique row labels.

## Optional Excel and JSON reference

CSV is the only required Lecture 04 file format. The following methods are references for learners who already have an external need for another format.

### Excel workbooks

```python
# Requires an appropriate optional Excel engine in the active environment.
worksheet = pd.read_excel("data/workbook.xlsx", sheet_name="Measurements")
worksheet.to_excel(
    "output/measurements.xlsx",
    sheet_name="Measurements",
    index=False,
)
```

Excel support has optional package dependencies. It is not installed or assessed for Lecture 04.

### JSON records

```python
records = pd.read_json("data/records.json", orient="records")
records.to_json(
    "output/records.json",
    orient="records",
    indent=2,
)
```

JSON can represent structures that are not simple rectangular tables, so the correct orientation depends on the producer and consumer. That design choice is not a Lecture 04 requirement.

## Scope boundary

These extensions remain optional:

- label alignment and an explicit arithmetic axis;
- ranking with documented tie behavior;
- recognition of duplicate-label selection behavior; and
- reference-only Excel and JSON methods.

They introduce no required demo step, assignment requirement, or prerequisite for the next lecture.
