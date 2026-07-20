# Lecture 03 bonus: Additional NumPy Patterns

This is the single authoritative bonus route for Lecture 03. It assumes the core environment and ndarray model are already secure. None of these patterns is required for Assignment 03 or assumed by Lecture 04.

Use the candidate environment from the main lecture:

```python
import numpy as np
```

# Advanced integer indexing

Core basic slices return views. **Advanced indexing** selects with integer or boolean arrays and returns a copy.

```python
values = np.array([10, 20, 30, 40, 50])
positions = np.array([4, 1, 3])
selected = values[positions]

selected[0] = 999
print(values)
print(selected)
```

Expected output:

```text
[10 20 30 40 50]
[999  20  40]
```

The independent result is useful for reordering, but it has different view/copy behavior from a basic slice.

# Multidimensional broadcasting

Core broadcasting uses only a scalar and a 1D array. More generally, NumPy compares shapes from the right. Two compared dimensions are compatible when they are equal or one of them is `1`.

A 1D offset with shape `(3,)` can align with the last dimension of a `(2, 3)` table:

```python
table = np.array(
    [
        [10, 20, 30],
        [40, 50, 60],
    ]
)
column_offsets = np.array([1, 2, 3])

adjusted = table + column_offsets
print(adjusted)
```

Expected output:

```text
[[11 22 33]
 [41 52 63]]
```

Predict both input shapes and the output shape before using multidimensional broadcasting. A shorter expression is not an improvement when its alignment is unclear or it creates an unnecessarily large intermediate array.

# Concatenation and stacking

**Concatenation** joins arrays along an existing axis. **Stacking** joins arrays along a new axis.

```python
first = np.array([1, 2, 3])
second = np.array([4, 5, 6])

joined = np.concatenate([first, second])
rows = np.stack([first, second], axis=0)

print(joined)
print(joined.shape)
print(rows)
print(rows.shape)
```

Expected output:

```text
[1 2 3 4 5 6]
(6,)
[[1 2 3]
 [4 5 6]]
(2, 3)
```

State the intended output shape first. Silent shape surprises become harder to diagnose in later data workflows.

# Selected universal functions

A NumPy **universal function**, or **ufunc**, applies an element-wise operation and follows NumPy's broadcasting rules.

```python
values = np.array([1.0, 4.0, 9.0, 16.0])
roots = np.sqrt(values)

left = np.array([1, 8, 3])
right = np.array([4, 2, 6])
pairwise_maximum = np.maximum(left, right)

print(roots)
print(pairwise_maximum)
```

Expected output:

```text
[1. 2. 3. 4.]
[4 8 6]
```

# Conditional selection

`np.where(condition, when_true, when_false)` chooses element-wise results without changing the source:

```python
values = np.array([-2, 0, 5, -1])
nonnegative = np.where(values >= 0, values, 0)

print(nonnegative)
```

Expected output:

```text
[0 0 5 0]
```

Boolean reductions answer whether any or all positions satisfy a condition:

```python
scores = np.array([18, 21, 24])
print((scores >= 20).any())
print((scores >= 15).all())
```

Expected output:

```text
True
True
```

# Sorting and indirect ordering

`np.sort()` returns sorted values. `np.argsort()` returns the positions that would put values in sorted order.

```python
values = np.array([30, 10, 20])
ordered_values = np.sort(values)
order = np.argsort(values)

print(ordered_values)
print(order)
print(values[order])
```

Expected output:

```text
[10 20 30]
[1 2 0]
[10 20 30]
```

Indirect ordering is powerful, but the relationship between positions and values must remain explicit.

# Reproducible random generation

Use `np.random.default_rng()` rather than the legacy global random interface. A **seed** initializes a generator so a fresh generator produces the same sequence under the documented NumPy version.

```python
rng = np.random.default_rng(seed=42)
values = rng.integers(0, 10, size=5)

print(values)
```

With NumPy 2.0.2, the expected output is:

```text
[0 7 6 4 4]
```

Pinned software and a recorded seed support reproduction, but a random generator is not a substitute for a small deterministic teaching fixture.

# Set-like array operations

`np.isin()` returns a boolean array with the same shape as its first input, marking elements found in the supplied test values:

```python
values = np.array([10, 20, 30, 40])
allowed = np.array([20, 40])
membership = np.isin(values, allowed)

print(membership)
print(values[membership])
```

Expected output:

```text
[False  True False  True]
[20 40]
```

Other set-like helpers include `np.unique()`, `np.intersect1d()`, and `np.setdiff1d()`. They often sort their results, so check their documented ordering when order matters.

# Structured arrays: recognize, do not require

A **structured array** stores named fields in one ndarray dtype. It can be useful when a NumPy-only binary representation is required, but it is not the course's bridge to ordinary tabular analysis.

```python
record_dtype = np.dtype(
    [
        ("site", "U5"),
        ("score", "f8"),
    ]
)
records = np.array(
    [
        ("north", 18.0),
        ("south", 21.0),
    ],
    dtype=record_dtype,
)

print(records["site"])
print(records["score"])
```

Expected output:

```text
['north' 'south']
[18. 21.]
```

Lecture 04's labeled pandas objects are the required course path for tables. Structured arrays remain optional and must not appear as an untaught assignment requirement.

# Scope boundary and references

This bonus deliberately omits a broad linear-algebra survey, memory-mapped files, terminal plotting, and alternative environment managers. Add those tools only for a project with a concrete need and its own tested dependency contract.

Official references for the retained patterns:

- [NumPy indexing](https://numpy.org/doc/2.0/user/basics.indexing.html)
- [NumPy broadcasting](https://numpy.org/doc/2.0/user/basics.broadcasting.html)
- [NumPy random Generator](https://numpy.org/doc/2.0/reference/random/generator.html)
- [`numpy.isin`](https://numpy.org/doc/2.0/reference/generated/numpy.isin.html)
