Bonus Content: Advanced Data Cleaning

*These are power-user features for when you need to go beyond basic data cleaning. Master the core content first!*

## Modern Pandas Extension Types

Pandas extension types solve longstanding issues with missing data and memory efficiency. While useful, these are advanced features that go beyond basic data cleaning workflows.

*Fun fact: For years, pandas had to convert integers to floats when there was missing data. Extension types finally fixed this - no more mysterious float64 columns!*

### Extension Types for Better Missing Data Handling

Traditional NumPy-based types couldn't represent missing integers or booleans. Extension types provide proper NA support across all data types.

**Reference:**

- `astype('Int64')` - Nullable integer (note capital I)
- `astype('Float64')` - Nullable float
- `astype('boolean')` - Nullable boolean
- `astype('string')` - Efficient string type
- `pd.NA` - Missing value marker for extension types
- vs `np.nan` - Old-style missing value (float only)

**Brief Example:**

```python
# Old way: integers become floats with missing data
s_old = pd.Series([1, 2, None])
print(s_old.dtype)  # float64 (forced conversion!)

# New way: integers stay integers
s_new = pd.Series([1, 2, None], dtype='Int64')
print(s_new.dtype)  # Int64
print(s_new)  # [1, 2, <NA>]

# Boolean with proper missing values
bools = pd.Series([True, False, None], dtype='boolean')
print(bools)  # [True, False, <NA>]
```

### Why Use Extension Types?

Extension types provide better memory efficiency, faster operations, and proper missing data handling.

**When to use:**
- **Int64, Int32, Int16, Int8**: Integer data that might have missing values
- **Float64, Float32**: When you need explicit control over precision
- **boolean**: Boolean data with potential missing values
- **string**: Large text datasets (uses less memory than object dtype)
- **category**: Repeated string values (huge memory savings)

**Brief Example:**

```python
# Convert existing DataFrame to extension types
df = pd.DataFrame({
    'age': [25, 30, None, 45],
    'name': ['Alice', 'Bob', 'Charlie', None],
    'is_member': [True, False, None, True]
})

# Convert to extension types
df['age'] = df['age'].astype('Int64')
df['name'] = df['name'].astype('string')
df['is_member'] = df['is_member'].astype('boolean')

print(df.dtypes)
print(df)
```

## Advanced Regular Expressions for Text Data

Regular expressions (regex) are powerful for complex pattern matching, but they can be overkill for simple tasks.

*Warning: Regular expressions are write-only code - you write them once, and six months later you have no idea what they do. Comment generously!*

**Reference:**

- `\d` - Any digit (0-9)
- `\w` - Any word character (letter, digit, underscore)
- `\s` - Any whitespace
- `+` - One or more of previous
- `*` - Zero or more of previous
- `{n,m}` - Between n and m of previous
- `[abc]` - Any of a, b, or c
- `^` - Start of string
- `$` - End of string
- `()` - Capture group

**Brief Example:**

```python
# Extract phone numbers from text
import re
text = pd.Series(['Call me at 555-1234', 'My number is (555) 555-5678', 'No phone here'])

# Pattern for phone numbers
pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
phones = text.str.extract(f'({pattern})')
print(phones)

# Validate email addresses
emails = pd.Series(['alice@test.com', 'invalid.email', 'bob@example.org'])
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
valid = emails.str.match(email_pattern)
print(valid)  # [True, False, True]
```

## Advanced Outlier Detection Methods

Beyond simple threshold-based outlier detection, statistical methods can identify unusual values.

**Reference:**

- **IQR Method**: Values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
- **Z-Score Method**: Values with |z-score| > 3
- **Modified Z-Score**: More robust for skewed data
- **Isolation Forest**: Machine learning approach (sklearn)

**Brief Example:**

```python
# IQR-based outlier detection
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
print(f"Found {len(outliers)} outliers")

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['value']))
outliers_z = df[z_scores > 3]
```

## Complex String Transformations

Advanced string operations for specialized text cleaning tasks.

**Reference:**

- `str.extract(pattern, expand=True)` - Extract regex groups into columns
- `str.extractall(pattern)` - Extract all matches (returns MultiIndex)
- `str.normalize('NFKD')` - Unicode normalization
- `str.translate(table)` - Character-level replacement
- `str.encode()` / `str.decode()` - Character encoding conversion

**Brief Example:**

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

## Advanced Duplicate Handling

More sophisticated approaches to finding and handling duplicates.

**Reference:**

- `subset=['col1', 'col2']` - Check specific columns only
- `keep='first'` - Keep first occurrence (default)
- `keep='last'` - Keep last occurrence
- `keep=False` - Mark all duplicates as True
- Fuzzy matching for near-duplicates (requires `fuzzywuzzy` or similar)

**Brief Example:**

```python
# Find ALL duplicates (including first occurrence)
df = pd.DataFrame({'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
                   'score': [85, 90, 88, 92, 90]})

all_dupes = df[df.duplicated(subset=['name'], keep=False)]
print(all_dupes)  # Shows all Alice and Bob rows

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

## Data Type Optimization

Reduce memory usage by choosing optimal data types.

**Reference:**

- `pd.to_numeric(downcast='integer')` - Use smallest int type
- `pd.to_numeric(downcast='float')` - Use smallest float type
- `astype('category')` - For repeated string values
- `astype('Int8')`, `astype('Int16')`, etc. - Specific sizes

**Brief Example:**

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

## Conditional Data Replacement

Use `np.where()` and `np.select()` for complex conditional replacements.

**Reference:**

- `np.where(condition, if_true, if_false)` - Simple if-else
- `np.select(conditions_list, choices_list, default)` - Multiple conditions
- `pd.Series.where(condition, other)` - Keep values where True
- `pd.Series.mask(condition, other)` - Replace values where True

**Brief Example:**

```python
# Simple conditional replacement
df = pd.DataFrame({'score': [55, 65, 75, 85, 95]})
df['grade'] = np.where(df['score'] >= 70, 'Pass', 'Fail')

# Multiple conditions
conditions = [
    df['score'] >= 90,
    df['score'] >= 80,
    df['score'] >= 70,
    df['score'] >= 60
]
choices = ['A', 'B', 'C', 'D']
df['letter_grade'] = np.select(conditions, choices, default='F')
print(df)
```

## When to Use These Techniques

**Regular Expressions:** Email validation, phone number extraction, parsing log files, complex text cleaning.

**Advanced Outlier Detection:** Financial data, scientific measurements, when IQR/percentile methods aren't appropriate.

**Complex String Operations:** Parsing addresses, standardizing names, cleaning web-scraped data.

**Fuzzy Matching:** Merging datasets with typos, de-duplicating user input, matching company names.

**Memory Optimization:** Working with large datasets (>1GB), when speed is critical, preparing data for deployment.

**Conditional Replacement:** Complex business logic, deriving new categories, data validation with multiple rules.
