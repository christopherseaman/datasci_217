---
marp: true
theme: sqrl
paginate: true
class: invert
---

# Lecture 06: Are you ready to wrangle?!?

1. Introduction to Data Wrangling with pandas
2. Combining and Reshaping Data
3. Practical Data Cleaning Techniques

---

[Slides 1-9 remain unchanged]

---

## Reshaping Data: Melt

Melt transforms "wide" format data into "long" format.

Before:
```python
df = pd.DataFrame({
    'A': ['a', 'b', 'c'],
    'B': [1, 3, 5],
    'C': [2, 4, 6]
})
print(df)
```
```
   A  B  C
0  a  1  2
1  b  3  4
2  c  5  6
```

After:
```python
melted = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
print(melted)
```
```
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
3  a        C      2
4  b        C      4
5  c        C      6
```

---

## Reshaping Data: Pivot

Pivot transforms "long" format data into "wide" format.

Before (using melted data from previous slide):
```python
print(melted)
```
```
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
3  a        C      2
4  b        C      4
5  c        C      6
```

After:
```python
pivoted = melted.pivot(index='A', columns='variable', values='value')
print(pivoted)
```
```
variable  B  C
A            
a         1  2
b         3  4
c         5  6
```

---

## Stacking and Unstacking

Stacking rotates from columns to index, unstacking does the opposite.

Original DataFrame:
```python
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
print(df)
```
```
   A  B
x  1  3
y  2  4
```

Stacked:
```python
stacked = df.stack()
print(stacked)
```
```
x  A    1
   B    3
y  A    2
   B    4
dtype: int64
```

Unstacked (back to original):
```python
unstacked = stacked.unstack()
print(unstacked)
```
```
   A  B
x  1  3
y  2  4
```

---

## Regular Expressions (Regex) in pandas

- Powerful pattern matching tool, similar to command-line use
- Used with string methods in pandas for advanced text processing
- Common patterns:
  - `\d`: any digit
  - `\w`: any word character
  - `\s`: any whitespace
  - `+`: one or more
  - `*`: zero or more
  - `[]`: character set
  - `()`: capturing group

---

## String Manipulation with Regex

Example: Extracting information from text

```python
df = pd.DataFrame({
    'text': [
        'Contact: john@email.com, Phone: 123-456-7890',
        'Meeting on 2023/05/15 with Jane (jane@company.com)'
    ]
})

# Extract email addresses
df['email'] = df['text'].str.extract(r'([\w\.-]+@[\w\.-]+)')

# Extract phone numbers
df['phone'] = df['text'].str.extract(r'(\d{3}-\d{3}-\d{4})')

# Extract dates
df['date'] = df['text'].str.extract(r'(\d{4}/\d{2}/\d{2})')

print(df)
```
```
                                               text               email         phone        date
0  Contact: john@email.com, Phone: 123-456-7890    john@email.com  123-456-7890        NaN
1  Meeting on 2023/05/15 with Jane (jane@company.com)  jane@company.com          NaN  2023/05/15
```

---

## Advanced Categorical Data Operations

Example: Managing categories in a DataFrame

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'A', 'B', 'D', 'E']
})
df['category'] = df['category'].astype('category')

# Add new category
df['category'] = df['category'].cat.add_categories(['F'])

# Remove unused categories
df['category'] = df['category'].cat.remove_unused_categories()

# Rename categories
df['category'] = df['category'].cat.rename_categories({'A': 'Alpha', 'B': 'Beta'})

print(df['category'].cat.categories)
print(df)
```
```
Index(['Alpha', 'Beta', 'C', 'D', 'E'], dtype='object')
  category
0    Alpha
1     Beta
2        C
3    Alpha
4     Beta
5        D
6        E
```

---
