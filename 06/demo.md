# Mini-Demos

## 1. Pandas Basics

Let's start by creating a simple DataFrame and performing some basic operations. We'll create a DataFrame with information about people, then demonstrate how to select columns, filter rows, and add new columns.

```python
import pandas as pd
import numpy as np

# Create a simple DataFrame with information about people
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'city': ['New York', 'San Francisco', 'London', 'Sydney']
}
df = pd.DataFrame(data)

# Display the entire DataFrame
print("DataFrame:")
print(df)

# Select a single column (returns a Series)
print("\nSelect 'name' column:")
print(df['name'])

# Filter rows based on a condition
print("\nFilter rows where age > 30:")
print(df[df['age'] > 30])

# Add a new column based on an existing one
# Here we're creating a boolean column for whether someone is an adult
df['is_adult'] = df['age'] >= 18
print("\nDataFrame with new 'is_adult' column:")
print(df)
```

## 2. Data Types and Conversion

Now, let's explore how pandas handles different data types. We'll create a DataFrame with mixed types and then convert them to more appropriate types. This is a common task in data cleaning.

```python
import pandas as pd

# Create a DataFrame with mixed types
data = {
    'date': ['2023-05-01', '2023-05-02', '2023-05-03'],  # Strings that look like dates
    'value': ['100', '200', '300'],  # Strings that look like numbers
    'category': [1, 2, 3]  # Numbers that we want to treat as categories
}
df = pd.DataFrame(data)

# Display original DataFrame and its data types
print("Original DataFrame and types:")
print(df.dtypes)
print(df)

# Convert types
df['date'] = pd.to_datetime(df['date'])  # Convert to datetime
df['value'] = pd.to_numeric(df['value'])  # Convert to numeric
df['category'] = df['category'].astype('category')  # Convert to categorical

# Display converted DataFrame and its new data types
print("\nConverted DataFrame and types:")
print(df.dtypes)
print(df)
```

## 3. Handling Missing Data

Missing data is a common issue in real-world datasets. Let's create a DataFrame with some missing values and demonstrate different methods for handling them.

```python
import pandas as pd
import numpy as np

# Create a DataFrame with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],  # np.nan represents a missing value
    'B': [5, np.nan, np.nan, 8],
    'C': ['a', 'b', 'c', None]  # None also represents a missing value
})

# Display the DataFrame and count of missing values
print("Original DataFrame:")
print(df)
print("\nMissing values in each column:")
print(df.isnull().sum())

# Demonstrate different methods of handling missing data
print("\nDrop rows with any missing values:")
print(df.dropna())

print("\nFill numeric missing values with mean of the column:")
print(df.fillna(df.mean()))

print("\nFill missing values with the previous value (forward fill):")
print(df.fillna(method='ffill'))
```

## 4. Data Aggregation and Grouping

Grouping and aggregating data is a powerful way to summarize information. Let's create a DataFrame and perform some groupby operations.

```python
import pandas as pd

# Create a sample DataFrame with categories and values
data = {
    'category': ['A', 'B', 'A', 'B', 'A', 'C'],
    'value1': [10, 20, 30, 40, 50, 60],
    'value2': [100, 200, 300, 400, 500, 600]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Group by 'category' and calculate mean of other columns
print("\nGroup by 'category' and calculate mean:")
print(df.groupby('category').mean())

# Group by 'category' and apply different aggregation functions to different columns
print("\nGroup by 'category' and aggregate different functions:")
print(df.groupby('category').agg({
    'value1': 'sum',  # Sum of value1 for each category
    'value2': ['min', 'max']  # Min and max of value2 for each category
}))
```

## 5. Quick Data Visualization

Visualizing data can provide quick insights. Let's create a simple time series dataset and visualize it in different ways using pandas' built-in plotting capabilities.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample DataFrame with years and sales data
data = {
    'year': range(2010, 2023),
    'sales': [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
}
df = pd.DataFrame(data)

# Create a line plot
df.plot(x='year', y='sales', title='Sales Trend')
plt.show()

# Create a bar plot
df.plot(x='year', y='sales', kind='bar', title='Yearly Sales')
plt.show()

# Create a scatter plot
plt.figure()  # Create a new figure
plt.scatter(df['year'], df['sales'])
plt.title('Sales vs Year')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()
```