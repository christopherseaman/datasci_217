# Bonus Python Concepts

*Optional extensions for students who want to explore Python concepts beyond the core lecture.*

## Function design

The core lecture introduces functions and script entry points. These patterns deepen that material without repeating the basic syntax.

### Flexible arguments

```python
def calculate_stats(*numbers):
    """Return simple statistics for any number of numeric arguments."""
    if not numbers:
        return None
    total = sum(numbers)
    return {"sum": total, "average": total / len(numbers), "count": len(numbers)}


def create_profile(**details):
    """Collect arbitrary named fields into a new dictionary."""
    return dict(details)
```

Default parameters are evaluated when the function is defined, so use `None` when a mutable default should be created per call:

```python
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
```

### Documentation and validation

Docstrings describe a function's contract. Keep examples and accepted values aligned with the implementation:

```python
def analyze_data(data, method="mean"):
    """Return the mean or median of a non-empty numeric sequence."""
    if not data:
        raise ValueError("data cannot be empty")
    if method == "mean":
        return sum(data) / len(data)
    if method == "median":
        ordered = sorted(data)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2
    raise ValueError("method must be 'mean' or 'median'")
```

## Conditional expressions

A conditional (ternary) expression is useful for a short, readable choice. Prefer a regular `if` statement when conditions become nested or complex.

```python
age = 25
status = "adult" if age >= 18 else "minor"

temperatures = [15, 25, 35, 5, 45]
categories = [
    "hot" if temp > 30 else "cold" if temp < 10 else "moderate"
    for temp in temperatures
]
```

## Python's object model

In Python, values are objects with a type, identity, and value. `id(value)` exposes an identity token for the lifetime of that object; it is not a promise that the object resides at that numeric memory address.

```python
value = [1, 2]
alias = value
copy = value.copy()
print(value is alias)  # True
print(value is copy)   # False
```

## Mutability and hashability

Mutable objects can change in place; immutable objects cannot. A tuple is immutable, but it is hashable only when all of its elements are hashable. This is why a tuple containing a list cannot be used as a dictionary key.

```python
items = [1, 2]
items.append(3)

point = (1, 2)
lookup = {point: "origin-adjacent"}

unhashable = (1, [2])
# {unhashable: "not allowed"}  # TypeError: list is unhashable
```

## Exception handling patterns

Catch the narrowest expected exception, add context, and let unexpected failures remain visible:

```python
def parse_score(text):
    try:
        score = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid score: {text!r}") from exc
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    return score
```

Custom exceptions can make a library's public failure modes clearer:

```python
class DataValidationError(ValueError):
    """Raised when input data violates an application contract."""


def require_columns(columns, required):
    missing = set(required) - set(columns)
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")
```

## Practice prompts

1. Add keyword-only options to a reusable function.
2. Document and validate a small data-processing function.
3. Find a mutable-default-argument bug and repair it.
4. Define a custom exception for one domain-specific validation rule.
