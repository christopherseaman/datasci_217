---
notion:
  title_line: "# DLC: Advanced Data Cleaning"
  role: bonus
  status: mapped
  page_id: "286d9fdd-1a1a-80ca-b36b-f85d415c2e53"
  url: "https://app.notion.com/p/286d9fdd1a1a80cab36bf85d415c2e53"
---

# DLC: Advanced Data Cleaning

_These are power-user features for when you need to go beyond basic data cleaning. Master the core content first!_

# Modern Pandas Extension Types

The core lecture introduces nullable `Int64`, `string`, and `boolean`. This bonus adds nullable floats and their memory and interoperability implications.

_Fun fact: For years, pandas had to convert integers to floats when there was missing data. Extension types finally fixed this: no more mysterious float64 columns!_

## Float64 and convert_dtypes()

### Reference Card: Extra nullable conversions

- `astype('Float64')`: Nullable float; missing shows as `<NA>` instead of `NaN`.
- `df.convert_dtypes()`: Convert every column to the best nullable type in one call (`Int64`, `string`, `boolean`, `Float64`).
- `pd.NA`: The missing marker used by nullable types; `np.nan` remains the marker for NumPy `float64`.

### Code Snippet: Convert every column at once

```python
df = pd.DataFrame({'age': [25, 30, None, 45], 'name': ['Ana', 'Bo', 'Cy', None], 'member': [True, False, None, True]})
print(df.convert_dtypes().dtypes)
```

```text
age         Int64
name       string
member    boolean
dtype: object
```

Extension types give consistent missing-data semantics, but measure memory and speed on the actual data rather than assuming they are smaller or faster. Under pandas 3, inferred text uses the `str` dtype; an explicit `string` dtype remains useful when nullable-string semantics are part of the data contract.

# Advanced Regular Expressions for Text Data

Regular expressions (regex) are powerful for complex pattern matching, but they can be overkill for simple tasks. The core lecture uses `[0-9]` and `{n}` with `str.fullmatch()`; the syntax below goes further.

_Warning: Regular expressions are write-only code. You write them once, and six months later you have no idea what they do. Comment generously!_

## Regex syntax and extraction

### Reference Card: Regex syntax

- `\d`: Any Unicode decimal digit; use `[0-9]` for only the digits 0 through 9
- `\w`: Any word character (letter, digit, underscore)
- `\s`: Any whitespace
- `+`: One or more of previous
- `*`: Zero or more of previous
- `{n,m}`: Between n and m of previous
- `[abc]`: Any of a, b, or c
- `^`: Start of string
- `$`: End of string
- `()`: Capture group
- `df.replace(pattern, replacement, regex=True)`: Replace regex matches in text values

### Code Snippet: Extract phone numbers and validate emails

```python
# Extract phone numbers from text
import re
text = pd.Series(['Call me at 415-555-1234', 'My number is (555) 555-5678', 'No phone here'])

# Pattern for phone numbers
pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
phones = text.str.extract(f'({pattern})')
print(phones)
```

```text
                0
0    415-555-1234
1  (555) 555-5678
2             NaN
```

```python
# Validate email addresses
emails = pd.Series(['alice@test.com', 'invalid.email', 'bob@example.org'])
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
valid = emails.str.match(email_pattern)
print(valid)  # [True, False, True]
```

# Advanced Outlier Detection Methods

Beyond simple threshold-based outlier detection, statistical methods can identify unusual values.

## Statistical outlier methods

### Reference Card: Outlier detection methods

- **IQR rule**: taught in the lecture (Flag unusual values).
- **Z-score**: values with |z| > 3; `scipy.stats.zscore` needs SciPy, which `uv add scipy` installs.
- **Modified z-score**: `0.6745 * (x - median) / MAD`, flagged above 3.5; the median and median absolute deviation (MAD) resist extreme values.
- **Isolation Forest**: machine-learning approach (`sklearn`).

### Code Snippet: Compare z-score and modified z-score

`value` holds twenty cholesterol readings (mg/dL), 18 near 185 and two far outside, 450 and 460:

```python
from scipy import stats
z = np.abs(stats.zscore(value))
mad = (value - value.median()).abs().median()
modified_z = 0.6745 * (value - value.median()) / mad
print(value[z > 3].tolist(), value[modified_z.abs() > 3.5].tolist())
```

```text
[460] [450, 460]
```

The z-score rule misses 450: the two extreme values inflate the standard deviation they are measured against, so 450 scores only 2.93. The median-based rule catches both.

# Complex String Transformations

Advanced string operations for specialized text cleaning tasks.

## Joining Values

The lecture's string card teaches `str.split()`; these methods put strings back together.

### Reference Card: String joining

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `series.str.cat(sep=' ')` | Combine all non-missing strings into one | One string |
| `series.str.join(sep)` | Join the strings within each list value | String `Series` |

### Code Snippet: Rejoin split parts

```python
parts = pd.Series(['Alice Smith', 'Bob Jones']).str.split(' ')
print(parts.str.join('_').tolist())
print(pd.Series(['Alice', 'Bob']).str.cat(sep=' & '))
```

```text
['Alice_Smith', 'Bob_Jones']
Alice & Bob
```

## Extracting and normalizing text

### Reference Card: Advanced string methods

- `str.extract(pattern, expand=True)`: Extract regex groups into columns
- `str.extractall(pattern)`: Extract all matches (returns MultiIndex)
- `str.normalize('NFKD')`: Unicode normalization
- `str.translate(table)`: Character-level replacement
- `str.encode()` / `str.decode()`: Character encoding conversion

### Code Snippet: Parse addresses and normalize unicode

```python
# Extract multiple components from structured text
addresses = pd.Series(['123 Main St, Boston, MA 02101',
                       '456 Oak Ave, Cambridge, MA 02138'])

# Pattern with multiple capture groups
pattern = r'(\d+)\s+([A-Za-z\s]+),\s+([A-Za-z]+),\s+([A-Z]{2})\s+(\d{5})'
components = addresses.str.extract(pattern)
components.columns = ['number', 'street', 'city', 'state', 'zip']
print(components)

# Unicode normalization (remove accents)
text = pd.Series(['café', 'naïve', 'résumé'])
normalized = text.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
print(normalized)  # ['cafe', 'naive', 'resume']
```

# Advanced Duplicate Handling

The core lecture finds exact repeats and repeated identifiers with `duplicated()`. Near-duplicates, such as `John Smith` and `Jon Smith`, differ by a typo, so exact comparison misses them.

## Fuzzy matching for near-duplicates

### Reference Card: Fuzzy matching

- Fuzzy matching for near-duplicates (requires `fuzzywuzzy` or similar; `uv add fuzzywuzzy` installs it)

### Code Snippet: Find near-duplicate names

```python
# Fuzzy string matching for near-duplicates
from fuzzywuzzy import fuzz
names = pd.Series(['John Smith', 'Jon Smith', 'Jane Doe'])

def find_similar(s, threshold=80):
    for i, name1 in enumerate(s):
        for j, name2 in enumerate(s[i+1:], i+1):
            ratio = fuzz.ratio(name1, name2)
            if ratio >= threshold:
                print(f"Similar: '{name1}' and '{name2}' ({ratio}% match)")

find_similar(names)
```

# Data Type Optimization

Reduce memory usage by choosing optimal data types.

## Downcasting numbers and categorizing strings

### Reference Card: Memory-efficient dtypes

- `pd.to_numeric(downcast='integer')`: Use smallest int type
- `pd.to_numeric(downcast='float')`: Use smallest float type
- `astype('category')`: For repeated string values
- `astype('Int8')`, `astype('Int16')`, etc.: Specific sizes

### Code Snippet: Shrink a DataFrame's memory footprint

```python
# Before optimization
df = pd.DataFrame({'A': range(1000), 'B': ['cat', 'dog', 'cat', 'dog'] * 250})
print(f"Original memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Optimize numeric column
df['A'] = pd.to_numeric(df['A'], downcast='integer')

# Optimize string column
df['B'] = df['B'].astype('category')

print(f"Optimized memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
```

# Conditional Data Replacement

Use `np.where()` and `np.select()` for complex conditional replacements.

## Vectorized conditional logic

### Reference Card: np.where and np.select

- `np.where(condition, if_true, if_false)`: Two outcomes, like `if`/`else`.
- `np.select(conditions_list, choices_list, default)`: Several outcomes; the first `True` condition wins, like `if`/`elif`, and `default` covers rows where none is `True`.

### Code Snippet: Stage blood pressure without apply

The lecture's `bp_stage()` runs once per row through `apply(axis=1)`. The same rule written as whole-column conditions gives the same stages in one step.

```python
vitals = pd.DataFrame({'sbp': [118, 142, 134, 126], 'dbp': [76, 84, 78, 92]})  # mmHg

# Two outcomes: np.where
vitals['sbp_140_plus'] = np.where(vitals['sbp'] >= 140, 'yes', 'no')

# Several outcomes: the first True condition wins, as in if/elif
conditions = [
    (vitals['sbp'] >= 140) | (vitals['dbp'] >= 90),
    (vitals['sbp'] >= 130) | (vitals['dbp'] >= 80),
]
choices = ['stage 2', 'stage 1']
vitals['bp_stage'] = np.select(conditions, choices, default='below stage 1')
print(vitals)
```

```text
   sbp  dbp sbp_140_plus       bp_stage
0  118   76           no  below stage 1
1  142   84          yes        stage 2
2  134   78           no        stage 1
3  126   92           no        stage 2
```

# Configuration-Driven Cleaning

Configuration files can make repeated pipelines more maintainable and reproducible. If a pipeline is reused across sources, a small dictionary or reviewed configuration file can hold genuinely changeable contract values so transformation and validation do not drift apart.

Keep genuinely changeable rules separate from the transformation logic, but do not turn every implementation constant into an option. Changing a rule still requires a documented decision and a fresh validation run.

## Configuration Guidance

Use a Python dictionary for a small, local configuration; use a reviewed CSV, JSON, or text file when parameters must be shared. Keep transformations in functions and document the source of each cleaning rule.

# When to Use These Techniques

- **Regular Expressions**: Email validation, phone number extraction, parsing log files, complex text cleaning.
- **Advanced Outlier Detection**: Financial data, scientific measurements, when IQR/percentile methods aren't appropriate.
- **Complex String Operations**: Parsing addresses, standardizing names, cleaning web-scraped data.
- **Fuzzy Matching**: Merging datasets with typos, de-duplicating user input, matching company names.
- **Memory Optimization**: Working with large datasets (>1GB), when speed is critical, preparing data for deployment.
- **Conditional Replacement**: Clinical staging rules on large tables, deriving new categories, data validation with multiple rules.
- **Configuration-Driven Cleaning**: The same cleaning rules reused across sites, sources, or repeated data deliveries.

# Optional Reference: Sampling Designs and Resampling

The core lecture uses `df.sample()` to spot-check rows. The techniques below show additional designs and resampling patterns; each one answers a different selection question.

## Stratified Sampling

Stratified sampling divides the sampling frame into defined strata, then samples within each stratum. Use `GroupBy.sample` when the design calls for a fixed number or fraction from every group (Lecture 08 teaches `groupby()`). The strata and allocation are analytical choices; every group must have enough rows unless sampling with replacement is deliberate. For a train/test split that preserves a label's proportions, see `sklearn.model_selection.train_test_split(..., stratify=labels, random_state=...)`.

```python
frame = pd.DataFrame({
    'site': ['north'] * 4 + ['south'] * 4,
    'value': [3, 5, 4, 8, 2, 7, 6, 9],
})

by_site = frame.groupby('site', group_keys=False).sample(
    n=2,
    random_state=42,
)
print(by_site)
```

## Weighted and Systematic Sampling

- `df.sample(weights='weight')` uses caller-supplied selection weights, which must be validated and justified by the sampling design.
- Systematic sampling chooses a random start and then every _step_-th row. Ordering or periodic structure can make it biased, so `df.iloc[start::step]` is only appropriate when that risk has been considered.

```python
weighted_frame = frame.assign(weight=[1, 1, 1, 1, 2, 2, 2, 2])
weighted = weighted_frame.sample(n=3, weights='weight', random_state=42)

rng = np.random.default_rng(42)
step = 2
start = rng.integers(step)
systematic = frame.iloc[start::step]
print(weighted)
print(systematic)
```

## Shuffling and Permutation

Shuffling changes row order while retaining every row; it is useful when order is not meaningful. `df.sample(frac=1, random_state=42)` returns a shuffled DataFrame. `np.random.default_rng().permutation()` returns a permutation of positions, which can be reused to reorder aligned arrays or a DataFrame with `.iloc`.

```python
shuffled = frame.sample(frac=1, random_state=42)
positions = np.random.default_rng(42).permutation(len(frame))
same_rows_new_order = frame.iloc[positions]
print(shuffled)
print(same_rows_new_order)
```

Do not shuffle before time-aware analysis or temporal validation, where original order is part of the design.

## Bootstrap Sampling

`df.sample(n=len(df), replace=True)` draws a same-sized resample with expected duplicates. Repeating the draw to estimate a statistic's uncertainty is a bootstrap procedure with assumptions of its own; it is not an ordinary train/test split or proof of representativeness.

```python
bootstrap_draw = frame.sample(
    n=len(frame),
    replace=True,
    random_state=42,
)
print(bootstrap_draw)
```
