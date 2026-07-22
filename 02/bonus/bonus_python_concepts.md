# Bonus Python Concepts

*This content is optional and not required for assignments. It's here for students who want to dive deeper into Python programming concepts.*

## Functions

**Conceptual Description:**
Functions organize code into reusable units with clear interfaces. Lambda expressions provide concise syntax for simple function definitions, particularly useful with higher-order functions.

**Reference:**

- `def function_name(parameters): ...` - Function definition
- `return value` - Return value
- `if __name__ == "__main__":` - Main guard for script execution
- `import module` - Import functions from other modules (could be module.py)

**Brief Example:**

```python
# Function definition
def calculate_average(grades):
    """Calculate the average of a list of grades."""
    if not grades:
        return 0
    return sum(grades) / len(grades)

# Main guard for script execution
if __name__ == "__main__":
    # This code runs when script is executed directly
    grades = [85, 92, 78, 96, 88]
    average = calculate_average(grades)
    print(f"Average grade: {average:.1f}")
```

**Library Usage:**

```python
# Other scripts can import and use these functions, e.g., analysis.py
from analysis import calculate_average
result = calculate_average([90, 95, 87])
```

## Advanced Function Concepts

### Lambda Functions

Lambda functions provide concise anonymous function syntax for simple operations used once. They're particularly useful with higher-order functions like `sorted()`, `filter()`, and `map()` for data transformations and custom sorting logic.

**Reference:**
- `lambda parameters: expression` - Anonymous function syntax
- `sorted(iterable, key=function)` - Sort with custom key function
- `filter(function, iterable)` - Filter items based on condition
- `map(function, iterable)` - Apply function to each item
- `functools.reduce(function, iterable)` - Cumulative operations

**Brief Example:**
```python
# Lambda with sorting
students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
sorted_students = sorted(students, key=lambda x: x[1], reverse=True)

# Lambda with filter and map
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_squares = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))

# Lambda with data processing
data = ["  Alice  ", "  Bob", "Charlie   "]
clean_names = list(map(lambda x: x.strip().title(), data))

# Complex lambda for data analysis
sales_data = [{"product": "A", "sales": 100}, {"product": "B", "sales": 200}]
top_products = sorted(sales_data, key=lambda x: x["sales"], reverse=True)
```

### Module vs Script Execution

Understanding the difference between importing a module and running a script directly is crucial for creating reusable code.

**The `if __name__ == "__main__":` Pattern:**

```python
# mymodule.py
def useful_function():
    return "This function can be imported"

def main():
    """Main function for script execution"""
    print("Running as a script")
    result = useful_function()
    print(result)

# This only runs when script is executed directly, not when imported
if __name__ == "__main__":
    main()
```

**Usage Examples:**

```python
# As a script:
# $ python mymodule.py
# Output: Running as a script
#         This function can be imported

# As an import:
# >>> import mymodule
# >>> mymodule.useful_function()
# 'This function can be imported'
# (No "Running as a script" output)
```

**Benefits:**
- **Code Reusability**: Functions can be imported without running script logic
- **Testing**: Test functions individually without running main script
- **Library Creation**: Turn any script into an importable module
- **Command Line Tools**: Create scripts that work both as tools and libraries

### Advanced Function Features

**Default Parameters:**
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))           # Hello, Alice!
print(greet("Bob", "Hi"))       # Hi, Bob!
```

**Variable Arguments:**
```python
def calculate_stats(*numbers):
    """Accept any number of arguments"""
    if not numbers:
        return None
    return {
        'sum': sum(numbers),
        'average': sum(numbers) / len(numbers),
        'count': len(numbers)
    }

stats = calculate_stats(1, 2, 3, 4, 5)
print(stats)  # {'sum': 15, 'average': 3.0, 'count': 5}
```

**Keyword Arguments:**
```python
def create_profile(**details):
    """Accept any number of keyword arguments"""
    profile = {}
    for key, value in details.items():
        profile[key] = value
    return profile

profile = create_profile(name="Alice", age=25, city="San Francisco")
print(profile)  # {'name': 'Alice', 'age': 25, 'city': 'San Francisco'}
```

### Function Documentation

**Docstrings:**
```python
def analyze_data(data, method="mean"):
    """
    Analyze numerical data using specified method.
    
    Args:
        data (list): List of numerical values
        method (str): Analysis method ('mean', 'median', 'mode')
    
    Returns:
        float: Analysis result
    
    Raises:
        ValueError: If data is empty or method is invalid
    
    Example:
        >>> analyze_data([1, 2, 3, 4, 5])
        3.0
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    if method == "mean":
        return sum(data) / len(data)
    elif method == "median":
        sorted_data = sorted(data)
        n = len(sorted_data)
        return sorted_data[n // 2] if n % 2 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2
    else:
        raise ValueError(f"Unknown method: {method}")
```

## When to Use These Concepts

**Use Functions When:**
- Code is repeated in multiple places
- Logic is complex and benefits from organization
- Testing individual components is important
- Creating reusable utilities

**Use Lambda Functions When:**
- Simple, one-line operations
- Sorting with custom keys
- Quick transformations with map/filter
- Callback functions for data processing

**Use `if __name__ == "__main__"` When:**
- Creating scripts that might be imported
- Building command-line tools
- Testing code interactively
- Separating library code from execution code

## Practice Exercises

1. Create a data analysis module with functions that can be imported or run as a script
2. Use lambda functions to sort complex data structures
3. Build a utility module with the main guard pattern
4. Practice writing comprehensive docstrings
5. Create functions with default parameters and variable arguments

## Ternary Expressions (Conditional Expressions)

Ternary expressions provide a concise way to write simple conditional assignments in a single line. This syntax is particularly useful in data science for applying transformations, setting default values, and creating derived columns in datasets.

**Reference:**

- `value_if_true if condition else value_if_false` - Ternary expression syntax
- Condition is evaluated first, then appropriate value is returned
- Can be nested but should be kept simple for readability
- Useful for data cleaning, default value assignment, and transformations
- More concise than full if-else statements for simple cases

**Brief Example:**

```python
# Basic ternary expressions
age = 25
status = "adult" if age >= 18 else "minor"

score = 85
grade = "pass" if score >= 70 else "fail"

# Data science applications
import pandas as pd
data = pd.DataFrame({'temperature': [15, 25, 35, 5, 45]})

# Create categorical variables
data['temp_category'] = ['hot' if temp > 30 else 'cold' if temp < 10 else 'moderate'
                        for temp in data['temperature']]

# Handle missing data with defaults
user_input = None
default_value = 100
result = user_input if user_input is not None else default_value

# Data transformation
prices = [19.99, 25.50, 12.00, 99.99]
discounted = [price * 0.8 if price > 20 else price for price in prices]

print(f"Status: {status}")           # adult
print(f"Grade: {grade}")             # pass
print(f"Result: {result}")           # 100
```

**When to Use Ternary Expressions:**
- Simple conditional assignments
- Data cleaning and transformation
- Setting default values
- Creating derived columns in datasets
- Replacing short if-else statements

**When NOT to Use:**
- Complex conditions with multiple clauses
- When readability would be compromised
- Nested ternary expressions (use regular if-else instead)

Remember: Good function design makes code more readable, testable, and reusable!

## Python Object Model

**Python Object Model:**

(advanced)

In Python, everything is an object with three fundamental properties:

```
Every Python Object Has:
┌─────────────────────────────────────┐
│  Object Identity (id)               │
│  └─ Memory address (never changes)  │
├─────────────────────────────────────┤
│  Object Type (type)                 │
│  └─ Defines behavior and operations │
├─────────────────────────────────────┤
│  Object Value (value)               │
│  └─ The actual data content         │
└─────────────────────────────────────┘

Examples:
- Integer 42: id=140712234567568, type=<class 'int'>, value=42
- String "hello": id=140712234567712, type=<class 'str'>, value="hello"
- List [1,2,3]: id=140712234567856, type=<class 'list'>, value=[1,2,3]
```

This unified object model means integers, strings, functions, and classes all share the same fundamental structure, enabling consistent behavior and introspection across all Python data types.

## Mutable vs Immutable Objects

**Mutable vs Immutable Objects:**

(kind of advanced)

| Aspect | Mutable (Lists) | Immutable (Tuples) |
|--------|----------------|-------------------|
| **Can Change After Creation?** | ✅ Yes | ❌ No |
| **Add Items** | `list.append(item)` | ❌ Creates new tuple |
| **Remove Items** | `list.remove(item)` | ❌ Creates new tuple |
| **Modify Items** | `list[0] = new_value` | ❌ TypeError |
| **Memory Usage** | Higher (flexible size) | Lower (fixed size) |
| **Performance** | Slower access | Faster access |
| **Use Cases** | Dynamic data, calculations | Fixed records, coordinates |

**Examples:**
```python
# Mutable List Operations
grades = [85, 92, 78]
grades.append(96)         # ✅ Works: [85, 92, 78, 96]
grades[0] = 90           # ✅ Works: [90, 92, 78, 96]

# Immutable Tuple Operations
coordinates = (40.7, -74.0)
coordinates.append(100)  # ❌ AttributeError: no append method
coordinates[0] = 41.0    # ❌ TypeError: doesn't support assignment

# Converting between types
list(coordinates)        # ✅ Creates new list: [40.7, -74.0]
tuple(grades)           # ✅ Creates new tuple: (90, 92, 78, 96)
```

## Advanced Mutable vs Immutable Analysis

**List vs Tuple Detailed Comparison:**

| Feature | Lists `[]` | Tuples `()` |
|---------|------------|-------------|
| **Mutability** | Mutable (can change) | Immutable (cannot change) |
| **Syntax** | `[1, 2, 3]` | `(1, 2, 3)` or `1, 2, 3` |
| **Memory Usage** | More memory overhead | Less memory overhead |
| **Performance** | Slower iteration | Faster iteration |
| **Methods Available** | Many: append, remove, insert, etc. | Few: count, index |
| **Use Case** | Dynamic collections | Fixed data records |
| **Hashable** | ❌ No (can't be dict keys) | ✅ Yes (can be dict keys) |
| **Best For** | Shopping carts, todo lists | Coordinates, RGB colors |

**Performance Comparison:**
```python
import timeit

# Create large collections
large_list = list(range(1000000))
large_tuple = tuple(range(1000000))

# Iteration speed test
list_time = timeit.timeit(lambda: [x for x in large_list], number=10)
tuple_time = timeit.timeit(lambda: [x for x in large_tuple], number=10)

print(f"List iteration: {list_time:.4f}s")
print(f"Tuple iteration: {tuple_time:.4f}s")  # Usually ~10% faster
```

**When to Choose:**
- **Lists**: When you need to add/remove/modify items frequently
- **Tuples**: When data structure is fixed (coordinates, database records, function returns)

## Advanced Error Handling Patterns

Advanced error handling goes beyond basic try/except to include exception details, cleanup code, and complex exception management patterns essential for robust data science applications.

**Reference:**

- `except ExceptionType as e:` - Capture exception details for debugging
- `else:` - Execute code only if no exception occurs
- `finally:` - Always execute cleanup code (runs regardless of exceptions)
- `raise Exception("message")` - Manually raise exceptions
- Multiple exception handling in single try block
- Custom exception classes for specific error conditions

**Comprehensive Exception Handling:**

```python
# Advanced error handling with multiple exception types
try:
    filename = input("Enter filename: ")
    with open(filename, 'r') as file:
        data = file.read()
        number = int(data.strip())
        result = 100 / number
except FileNotFoundError as e:
    print(f"File error: {e}")
    result = None
except ValueError as e:
    print(f"Invalid number format: {e}")
    result = None
except ZeroDivisionError as e:
    print(f"Math error: {e}")
    result = None
else:
    # Only runs if no exceptions occurred
    print("All operations completed successfully")
finally:
    # Always runs for cleanup
    print("Operation finished")
```

**File handling with comprehensive error management:**
```python
def process_data_file(filename):
    """Process data file with comprehensive error handling."""
    data = None
    try:
        with open(filename, 'r') as file:
            data = file.read()
            # Process the data
            lines = data.split('\n')
            numbers = [float(line) for line in lines if line.strip()]
            return sum(numbers) / len(numbers)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except ValueError as e:
        print(f"Error: Invalid data format - {e}")
        return None
    except ZeroDivisionError:
        print("Error: No valid numbers found in file")
        return None
    finally:
        print(f"Finished processing {filename}")
```

**Custom Exception Classes:**
```python
class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def validate_grade(grade):
    """Validate grade with custom exception."""
    try:
        grade_num = float(grade)
        if grade_num < 0 or grade_num > 100:
            raise DataValidationError(f"Grade {grade_num} is out of range (0-100)")
        return grade_num
    except ValueError:
        raise DataValidationError(f"'{grade}' is not a valid number")

# Usage with custom exceptions
try:
    user_grade = validate_grade("105")
except DataValidationError as e:
    print(f"Validation failed: {e}")
```

**Key Advanced Patterns:**
- **`finally` blocks**: Code that always executes, regardless of exceptions
- **Resource management**: Proper file handling with cleanup
- **Graceful degradation**: Continuing execution with fallback values
- **Exception chaining**: Preserving original error context
- **Custom exceptions**: Domain-specific error types for better debugging