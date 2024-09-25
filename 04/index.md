## Dictionary Comprehensions (advanced)

- Similar to list comprehensions, but create dictionaries
- Syntax: `{key_expr: value_expr for item in iterable if condition}`

```python
square_dict = {x: x**2 for x in range(5)}
name_lengths = {name: len(name) for name in ['Alice', 'Bob', 'Charlie']}
```

---
## Generator Expressions (advanced)

- Similar to list comprehensions, but generate items one at a time
- More memory-efficient for large datasets
- Created using parentheses `()` instead of square brackets

```python
gen = (x**2 for x in range(1000000))
print(next(gen))  # 0
print(next(gen))  # 1
```

---
## The `yield` keyword

- `yield` is used to define generator functions, it returns a generator object
- When called, it runs until it hits a `yield` statement, then pauses and returns the yielded value
- The function's state is saved, allowing it to resume where it left off on the next call

Example:
```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for number in countdown(5):
    print(number)
# Output: 5 4 3 2 1
```

---
# LIVE DEMO!!!
---
## Python: Files & Functions

- Interacting with files
- Python functions, modules
- Common file operations 
- Reading a file line-by-line
- Splitting lines into arrays
---

## Interacting with Files

Basic file operations:
- Opening a file: `open(filename, mode)`
- Reading from a file: `file.read()`, `file.readline()`, `file.readlines()`
- Writing to a file: `file.write()`, `file.writelines()`
- Closing a file: `file.close()`

Always use the `with` statement for automatic file closing:

```python
with open('example.txt', 'r') as file:
    content = file.read()
```

---

## File Modes

Common file modes:
- `'r'`: Read (default)
- `'w'`: Write (overwrites existing content)
- `'a'`: Append
- `'r+'`: Read and write
- `'b'`: Binary mode (e.g., `'rb'`, `'wb'`)

Example:
```python
with open('example.txt', 'w') as file:
    file.write('Hello, World!')
```

---

## Reading a File Line-by-Line

Method 1: Using a for loop
```python
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())
```

Method 2: Using `readline()`
```python
with open('example.txt', 'r') as file:
    while True:
        line = file.readline()
        if not line:
            break
        print(line.strip())
```

---

## Splitting Lines into Arrays

Using the `split()` method:

```python
with open('data.txt', 'r') as file:
    for line in file:
        # Split by whitespace (default)
        items = line.split()
        
        # Split by specific delimiter
        items = line.split(',')
        
        print(items)
```

---

## Common File Operations

- Check if a file exists:
  ```python
  import os
  os.path.exists('file.txt')
  ```

- Delete a file:
  ```python
  import os
  os.remove('file.txt')
  ```

- Rename a file:
  ```python
  import os
  os.rename('old_name.txt', 'new_name.txt')
  ```

---
## Common Directory Operations
- Create a new directory:
  ```python
  import os
  os.mkdir('new_directory')
  ```

- Create nested directories:
  ```python
  import os
  os.makedirs('path/to/new/directory')
  
  # Can also allow the directory to already exist
  os.makedirs('path/to/new/directory', exist_ok = True)
  ```

---

## Working with Directories

- Get current working directory:
  ```python
  import os
  current_dir = os.getcwd()
  ```

- Change current working directory:
  ```python
  import os
  os.chdir('/path/to/new/directory')
  ```

- List contents of a directory:
  ```python
  import os
  contents = os.listdir('/path/to/directory')
  ```

- Check if a path is a directory:
  ```python
  import os
  is_dir = os.path.isdir('/path/to/check')
  ```

---

## Python Functions

Defining a function:
```python
def greet(name):
    return f"Hello, {name}!"

# Calling the function
message = greet("Alice")
print(message)  # Output: Hello, Alice!
```

Function with default parameters:
```python
def greet(name="World"):
    return f"Hello, {name}!"

print(greet())  # Output: Hello, World!
print(greet("Bob"))  # Output: Hello, Bob!
```

---
## Function Arguments

Positional arguments:
```python
def add(a, b):
    return a + b

result = add(3, 5)  # result = 8
```

Keyword arguments:
```python
def greet(first_name, last_name):
    return f"Hello, {first_name} {last_name}!"

message = greet(last_name="Doe", first_name="John")
print(message)  # Output: Hello, John Doe!
```

---

## *args and **kwargs

`*args`: Variable number of positional arguments
```python
def sum_all(*args):
    return sum(args)

result = sum_all(1, 2, 3, 4)  # result = 10
```

`**kwargs`: Variable number of keyword arguments
```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="New York")
```

---
## Command Line Arguments in Python

You can pass arguments to python just like any other command

- Two main methods:
  1. `sys.argv`: Argument order matters
	  `python script.py arg1 arg2`
  2. `argparse`: Arguments are explicitly named
	  `python script.py -two arg2 -one arg1`

---
## Using sys.argv

```python
import sys

script_name = sys.argv[0]
arguments = sys.argv[1:]

print(f"Script: {script_name}")
print(f"Args: {arguments}")
```

Usage: `python script.py arg1 arg2`

---
## Using argparse Module

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Name to greet")
parser.add_argument("-c", "--count", type=int, default=1)

args = parser.parse_args()

for _ in range(args.count):
    print(f"Hello, {args.name}!")
```

Usage: `python script.py Alice -c 3`

---
## Key Benefits of argparse

- Automatic help messages
- Type conversion
- Optional and positional arguments
- Default values

Example: `python script.py -h`

---
## Python Modules

Importing modules:
```python
import math
print(math.pi)  # Output: 3.141592653589793

from math import sqrt
print(sqrt(16))  # Output: 4.0

from math import *  # Import all (use cautiously)
```

---
## Modules are just `.py` files!

Creating your own module:
1. Create a file `mymodule.py`
2. Define functions in the file
3. Import and use in another file:
   ```python
   import mymodule
   mymodule.my_function()
   ```

---

## Summary

- File operations: open, read, write, close
- Reading files line-by-line
- Splitting lines into arrays
- Defining and using functions
- Function arguments: positional, keyword, *args, **kwargs
- Working with modules

---