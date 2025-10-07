---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# My First Data Analysis

This notebook demonstrates:
- Loading data
- Basic exploration
- Simple visualization

```python
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

print("Setup complete!")
```

```python
# Display plots inline
%matplotlib inline

# Check working directory
%pwd
```

```python
# Create sample data
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000]
}

df = pd.DataFrame(data)
df
```

```python
# Summary statistics
print(df.describe())

# Quick visualization
df.plot(x='name', y='salary', kind='bar')
plt.title('Salaries by Person')
plt.show()
```

```python
# Test variable persistence
test_var = "I'm in memory"
print(test_var)
```

```python
# This will work until kernel restarts
print(test_var)
```
