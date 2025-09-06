# Lecture 05: Live Demo Instructions

## NumPy Demo

> In this demo, we'll explore the basics of NumPy, a fundamental library for numerical computing in Python. We'll create arrays, examine their properties, and perform some basic operations.

1. Open a new Jupyter notebook or Python interactive shell.

2. :star: Import NumPy:
   ```python
   import numpy as np
   ```

3. :star: Create a simple NumPy array:
   ```python
   arr = np.array([1, 2, 3, 4, 5])
   print("Simple array:", arr)
   ```

> Here we've created a 1-dimensional array. NumPy arrays are homogeneous, meaning all elements must be of the same type.

4. :star: Demonstrate array attributes:
   ```python
   print("Shape:", arr.shape)
   print("Dimensions:", arr.ndim)
   print("Size:", arr.size)
   print("Data type:", arr.dtype)
   ```

> These attributes give us important information about the array's structure and contents.

5. :star: Create a 2D array:
   ```python
   arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
   print("2D array:\n", arr_2d)
   ```

> NumPy can handle multi-dimensional arrays, which are crucial for many scientific computing tasks.

6. Demonstrate indexing and slicing:
   ```python
   print("First element of second row:", arr_2d[1, 0])
   print("First column:\n", arr_2d[:, 0])
   ```

> Indexing and slicing in NumPy is powerful and allows us to easily access specific elements or subsets of our data.

7. Perform basic operations:
   ```python
   print("Array + 10:\n", arr_2d + 10)
   print("Array * 2:\n", arr_2d * 2)
   print("Sum of all elements:", np.sum(arr_2d))

8. :star: Demonstrate flatten() vs ravel():
   ```python
   flat_arr = arr.flatten()
   ravel_arr = arr.ravel()
   
   print("\nFlattened array:", flat_arr)
   print("Raveled array:", ravel_arr)
   
   # Modify the flattened and raveled arrays
   flat_arr[0] = 99
   ravel_arr[0] = 88
   
   print("\nAfter modification:")
   print("Original array:\n", arr)
   print("Flattened array:", flat_arr)
   print("Raveled array:", ravel_arr)
   ```

> Note how modifying the raveled array affects the original, while the flattened array remains independent.

9. Demonstrate vertical stacking:
   ```python
   arr1 = np.array([1, 2, 3])
   arr2 = np.array([4, 5, 6])
   
   stacked_v = np.vstack((arr1, arr2))
   print("\nVertically stacked arrays:\n", stacked_v)
   ```

10. Demonstrate horizontal stacking:
   ```python
   stacked_h = np.hstack((arr1, arr2))
   print("\nHorizontally stacked arrays:", stacked_h)
   ```

11. Demonstrate stacking 2D arrays:
   ```python
   arr3 = np.array([[7, 8, 9], [10, 11, 12]])
   stacked_2d = np.vstack((arr, arr3))
   print("\nStacked 2D arrays:\n", stacked_2d)
   ```

> These operations are crucial for reshaping and combining arrays in NumPy, which is often necessary in data preprocessing and analysis tasks.

   ```

> NumPy operations are vectorized, meaning they're applied to all elements simultaneously, which is much faster than using Python loops.

## Pandas Demo

> Now we'll move on to Pandas, a powerful library for data manipulation and analysis. We'll create a DataFrame, which is the primary Pandas data structure, and perform some common operations.

1. :star: Import Pandas:
   ```python
   import pandas as pd
   ```

2. :star: Create a simple DataFrame:
   ```python
   data = {'Name': ['Alice', 'Bob', 'Charlie'],
           'Age': [25, 30, 35],
           'City': ['New York', 'San Francisco', 'Los Angeles']}
   df = pd.DataFrame(data)
   print("DataFrame:\n", df)
   ```

> A DataFrame is a 2-dimensional labeled data structure. It's similar to a spreadsheet or SQL table.

3. :star: Demonstrate basic DataFrame operations:
   ```python
   print("\nDataFrame info:")
   df.info()
   
   print("\nDataFrame description:")
   print(df.describe())
   
   print("\nSelect 'Name' column:")
   print(df['Name'])
   ```

> These operations give us a quick overview of our data and allow us to select specific columns.

4. :star: Filter data:
   ```python
   print("\nPeople older than 28:")
   print(df[df['Age'] > 28])
   ```

> We can easily filter our data based on conditions.

5. Add a new column:
   ```python
   df['Country'] = 'USA'
   print("\nDataFrame with new column:\n", df)
   ```

> Adding new columns to a DataFrame is straightforward.

6. Demonstrate groupby:
   ```python
   df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40], labels=['Young', 'Middle'])
   print("\nAge groups:\n", df.groupby('Age_Group').mean())
   ```

> The groupby operation is powerful for aggregating and analyzing data.

## NumPy and Pandas Integration Demo

> NumPy and Pandas work well together. Let's see how we can use them in combination.

1. Create a NumPy array and convert it to a Pandas Series:
   ```python
   np_arr = np.array([1, 2, 3, 4, 5])
   pd_series = pd.Series(np_arr)
   print("NumPy array:", np_arr)
   print("Pandas Series:\n", pd_series)
   ```

> A Pandas Series is essentially a labeled NumPy array.

2. :star: Perform NumPy operations on a Pandas DataFrame:
   ```python
   df_num = pd.DataFrame(np.random.rand(3, 3), columns=['A', 'B', 'C'])
   print("Original DataFrame:\n", df_num)
   
   print("\nSquare root:\n", np.sqrt(df_num))
   print("\nMean of each column:\n", np.mean(df_num, axis=0))
   ```

> We can use NumPy functions directly on Pandas DataFrames.

3. Use NumPy to create a boolean mask:
   ```python
   mask = df_num > 0.5
   print("Boolean mask:\n", mask)
   print("\nFiltered DataFrame:\n", df_num[mask])
   ```

> Boolean masking is a powerful technique for filtering data.

## Shell Commands Demo

> Finally, let's look at some shell commands that are useful for data processing.

1. Prepare a sample CSV file named 'data.csv' with the following content:
   ```
   Name,Age,City
   Alice,25,New York
   Bob,30,San Francisco
   Charlie,35,Los Angeles
   David,28,Chicago
   Eve,32,Boston
   ```

> This is a simple CSV file we'll use for our demonstrations.

2. :star: Demonstrate `cut`:
   ```bash
   echo "Extracting names and ages:"
   cut -d',' -f1,2 data.csv
   ```

> The `cut` command is useful for extracting specific columns from tabular data.

3. :star: Demonstrate `tr`:
   ```bash
   echo "Converting to uppercase:"
   cat data.csv | tr '[:lower:]' '[:upper:]'
   ```

> `tr` is used for translating or deleting characters. Here we're using it to convert text to uppercase.

4. :star: Demonstrate `sed`:
   ```bash
   echo "Replacing 'New York' with 'NYC':"
   sed 's/New York/NYC/' data.csv
   ```

> `sed` is a stream editor that can perform various text transformations. Here we're using it for a simple find-and-replace operation.

5. Combine commands:
   ```bash
   echo "Extracting ages, sorting, and showing top 3:"
   cut -d',' -f2 data.csv | sort -n | head -n 3
   ```

> By combining these commands with pipes, we can perform more complex data processing tasks.

Remember to practice these demos beforehand to ensure smooth execution during the lecture. Have the sample data and commands ready in a text file for quick access if needed.