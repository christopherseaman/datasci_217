# Lecture 02 bonus: Advanced Python function and collection patterns

This material is optional and not required for assignments. It assumes that you can already define and call a function, distinguish parameters from arguments and return values, handle empty input, use a dictionary, import a local module, and protect driver code with a main guard.

The examples extend those core skills instead of replacing the straightforward patterns used in the lecture.

# Default and keyword arguments

A **default argument value** is used when a caller omits the corresponding argument:

```python
def format_measurement(value, digits=1):
    """Return a measurement formatted to the requested precision."""
    return f"{value:.{digits}f}"


print(format_measurement(21.456))
print(format_measurement(21.456, digits=2))
```

Expected output:

```text
21.5
21.46
```

`digits=2` is a **keyword argument**. It names the parameter at the call site, which can make optional settings easier to read.

Avoid mutable default values such as an empty list. The same default object can be reused across calls. Use `None` and create a new list inside the function instead:

```python
def append_label(label, labels=None):
    """Return a new label list containing label."""
    if labels is None:
        labels = []

    result = list(labels)
    result.append(label)
    return result
```

# Type annotations

**Type annotations** document the kinds of values an interface expects and returns. Python does not enforce them automatically at runtime.

```python
def mean(values: list[float]) -> float | None:
    """Return the arithmetic mean, or None for empty input."""
    if not values:
        return None

    return sum(values) / len(values)
```

Annotations help readers and static-analysis tools, but they do not replace tests or input validation.

# Tuples and unpacking

A **tuple** is an ordered collection that cannot be modified after creation. It can represent a small fixed result:

```python
def range_summary(values):
    """Return the minimum and maximum, or None for empty input."""
    if not values:
        return None

    return min(values), max(values)


result = range_summary([18, 21, 24])

if result is not None:
    minimum, maximum = result
    print(f"Minimum: {minimum}")
    print(f"Maximum: {maximum}")
```

Assigning the two tuple elements to `minimum` and `maximum` is **unpacking**.

# Comprehensions

A **list comprehension** creates a list from an iterable. Keep it short enough to read without tracing several conditions at once:

```python
measurements = [18, 21, 24, 19]
review_values = [value for value in measurements if value >= 20]
print(review_values)
```

A **dictionary comprehension** creates key-value associations:

```python
labels = ["morning", "evening", "overnight"]
label_lengths = {label: len(label) for label in labels}
print(label_lengths)
```

When a comprehension needs nested loops, several branches, or side effects, write a normal loop instead.

# Functions as values and sort keys

Python functions are values, so a function can be passed to another function. `sorted()` accepts a `key` function that returns the value used for ordering:

```python
records = [
    {"label": "Morning", "mean": 21.0},
    {"label": "Evening", "mean": 22.7},
    {"label": "Overnight", "mean": 19.5},
]


def mean_value(record):
    """Return the mean used to order a record."""
    return record["mean"]


ordered_records = sorted(records, key=mean_value, reverse=True)

for record in ordered_records:
    print(record["label"])
```

For a very small one-use expression, a **lambda expression** creates an unnamed function:

```python
ordered_records = sorted(
    records,
    key=lambda record: record["mean"],
    reverse=True,
)
```

Prefer a named function when the rule needs explanation, testing, or reuse.

# Exceptions as interface contracts

Returning `None` is one reasonable empty-input contract. Another is to raise an exception when the caller has violated a requirement.

```python
def require_mean(values):
    """Return the arithmetic mean; reject empty input."""
    if not values:
        raise ValueError("values must contain at least one number")

    return sum(values) / len(values)
```

A caller can handle the specific failure:

```python
try:
    result = require_mean([])
except ValueError as error:
    print(f"Cannot calculate mean: {error}")
else:
    print(f"Mean: {result:.1f}")
```

Catch only exceptions you can handle meaningfully. Avoid a broad `except:` block that hides unexpected programming errors.

# Mutability and aliases

Lists and dictionaries are **mutable**: their contents can change. Two names can refer to the same mutable object, creating aliases:

```python
original = [18, 21]
alias = original
alias.append(24)

print(original)
```

Expected output:

```text
[18, 21, 24]
```

Make a shallow copy when the outer list should change independently:

```python
original = [18, 21]
copied = list(original)
copied.append(24)

print(original)
print(copied)
```

Expected output:

```text
[18, 21]
[18, 21, 24]
```

Nested mutable objects require more careful copying. Lecture 03 revisits shared data and copies in the context of NumPy array views.

# Small assertions while developing

An `assert` statement checks an assumption and raises `AssertionError` when the condition is false:

```python
assert mean([18, 21, 24]) == 21
assert mean([]) is None
```

Assertions are useful for quick development checks. They are not a substitute for user-facing validation or a complete automated test suite.

# Optional practice

1. Add a keyword-only precision option to a formatting function.
2. Return a fixed two-value tuple and unpack it at the call site.
3. Replace one short transformation loop with a readable comprehension.
4. Sort a list of dictionaries first with a named key function and then with a lambda.
5. Compare a `None`-returning interface with a `ValueError`-raising interface.
6. Demonstrate aliasing and copying with a list without changing the original accidentally.
