# Bonus: Bounded Data-Cleaning Extensions

This optional material extends Lecture 05's documented cleaning pipeline. None of it is required by the Lecture 05 demos, assignment, or Lecture 06.

Return to [README.md](README.md) for the required raw/clean contract, audit-before-mutation workflow, targeted transformations, validation invariants, decision log, and fresh-runtime pipeline.

## MCAR, MAR, and MNAR as analysis assumptions

A **missing-data mechanism** describes how the probability that a value is missing relates to observed and unobserved data:

- **Missing completely at random (MCAR):** missingness does not depend on observed or unobserved values.
- **Missing at random (MAR):** after conditioning on observed information, missingness does not additionally depend on the unseen value.
- **Missing not at random (MNAR):** missingness can still depend on the unseen value after accounting for observed information.

These are assumptions about a data-generating process, not labels that a missing-value count can prove. They can guide an analysis plan, but they do not create an automatic rule to drop or impute values.

The framework originates with Rubin's [Inference and Missing Data](https://www.ets.org/research/policy_research_reports/publications/article/1976/itce.html). The US Agency for Healthcare Research and Quality gives accessible definitions and applied examples in [Types of Missing Data](https://www.ncbi.nlm.nih.gov/books/NBK493614/).

This lecture does not assess mechanism modeling, multiple imputation, or sensitivity analysis.

## Nullable dtypes and pandas 3 strings

A **nullable dtype** represents ordinary values and missing values without forcing every column into a generic object representation. Specify one when the schema needs a nullable integer, Boolean, or string:

```python
import pandas as pd

typed = pd.DataFrame(
    {
        "visit_count": pd.Series([1, None, 3], dtype="Int64"),
        "consented": pd.Series([True, None, False], dtype="boolean"),
        "site_label": pd.Series(["North", None, "South"], dtype="string"),
    }
)

print(typed)
print(typed.dtypes)
```

The capitalized `Int64` name identifies pandas' nullable integer dtype. The explicit `string` dtype is a nullable `StringDtype`. pandas 3 also infers a default `str` dtype for ordinary string data; do not write validation that assumes every text column has legacy `object` dtype.

See the pandas documentation for [nullable integer data](https://pandas.pydata.org/docs/user_guide/integer_na.html), [nullable Boolean data](https://pandas.pydata.org/docs/user_guide/boolean.html), and the [pandas 3 string migration guide](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html).

## Advanced normalization with vectorized string methods

Normalization should have a documented equivalence rule. Unicode normalization, whitespace collapsing, and case normalization can be composed with vectorized string methods:

```python
normalization_input = pd.Series(
    ["  São   Paulo ", "SÃO PAULO", None],
    dtype="string",
)

normalized_text = (
    normalization_input
    .str.normalize("NFKC")
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.casefold()
)

print(normalized_text)
```

Unicode normalization does not decide whether accents, punctuation, abbreviations, or alternate names represent the same domain value. Preserve the original field when the normalized representation may need review. Python's `unicodedata` behavior follows the [Unicode normalization forms](https://unicode.org/reports/tr15/).

## A bounded custom transform after vectorization

Use built-in vectorized operations first. A small custom function is appropriate when a documented domain rule cannot be stated clearly with those operations alone.

This example first performs vectorized whitespace and case normalization, then applies one bounded identifier rule:

```python
submitted_ids = pd.Series(
    [" subject-7 ", "PARTICIPANT-42", "bad-id", None],
    dtype="string",
)

prepared_ids = submitted_ids.str.strip().str.casefold()


def normalize_participant_id(value):
    if pd.isna(value):
        return pd.NA

    prefix, separator, number = value.partition("-")
    if separator and prefix in {"subject", "participant"} and number.isdigit():
        return f"P{int(number):04d}"
    return pd.NA


normalized_ids = prepared_ids.map(normalize_participant_id).astype("string")
print(normalized_ids)
```

The function has one input, one documented output form, and an explicit result for unsupported values. The decision log should record that unsupported values became missing and require review.

## Domain-aware anomaly review

An **anomaly** is a value that violates or challenges an expected domain rule. It is not automatically an error or a row to delete.

Use a supplied specification to flag values while preserving them for review:

```python
measurements = pd.DataFrame(
    {
        "record_id": ["R1", "R2", "R3", "R4"],
        "temperature_c": [36.8, 42.5, 34.0, 37.1],
    }
)

documented_min = 35.0
documented_max = 42.0

measurements["outside_documented_range"] = ~measurements[
    "temperature_c"
].between(documented_min, documented_max)

anomaly_review = measurements.loc[
    measurements["outside_documented_range"]
].copy()

anomaly_review
```

The next action might be source verification, correction from authoritative evidence, retention with a caveat, or exclusion for a stated purpose. The numeric rule alone cannot choose among them.

## Same-index fallback with `combine_first`

`combine_first()` fills missing positions in one object from nonmissing positions in another. Restrict this optional pattern to sources already proven to represent the same rows in the same index and columns.

```python
primary = pd.DataFrame(
    {"status": ["active", pd.NA, "complete"]},
    index=["R1", "R2", "R3"],
)
backup = pd.DataFrame(
    {"status": [pd.NA, "pending", "complete"]},
    index=["R1", "R2", "R3"],
)

assert primary.index.equals(backup.index)
assert primary.columns.equals(backup.columns)

filled_from_backup = primary.isna() & backup.notna()
combined = primary.combine_first(backup)

print(combined)
print(filled_from_backup)
```

The provenance log must identify the fallback source and affected cells. If row identity or alignment is not already established, stop; Lecture 06 teaches key-based combination. See the official [`DataFrame.combine_first` reference](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.combine_first.html).

## A small in-notebook cleaning configuration

A small dictionary can keep repeated, already-decided rules visible in one notebook. It should not replace the audit or rationale.

```python
cleaning_config = {
    "rename": {"Site Name": "site"},
    "sentinels": {"site": {"": pd.NA, "N/A": pd.NA}},
    "allowed_site": ["north", "south", "west"],
}

configured_input = pd.DataFrame(
    {"Site Name": [" North ", "SOUTH", "N/A"]}
)

configured = configured_input.rename(columns=cleaning_config["rename"])
configured["site"] = configured["site"].replace(
    cleaning_config["sentinels"]["site"]
)
configured["site"] = configured["site"].str.strip().str.lower()

assert configured["site"].dropna().isin(
    cleaning_config["allowed_site"]
).all()
configured
```

Keep the configuration small, local, and directly connected to the decision log. External configuration files, dynamic dispatch, and general cleaning frameworks are outside this lecture.

## Explicitly deferred material

The following topics are not Lecture 05 bonus prerequisites:

- joins, concatenation, key alignment, and structural `melt`/`pivot` reshape — Lecture 06;
- visualization and plotting design — Lecture 07;
- GroupBy, named aggregation, and aggregating `pivot_table` — Lecture 08;
- entity/time ordering, time-based filling, resampling, and rolling analysis — Lecture 09;
- feature encodings, train/test splitting, and modeling — Lecture 10;
- fuzzy record linkage, automatic anomaly deletion, and statistical outlier catalogs;
- multiple imputation and missing-mechanism modeling; and
- shell-driven notebook execution, large configuration systems, or performance tuning.

Return to the required [Lecture 05 narrative](README.md) before proceeding to Lecture 06.
