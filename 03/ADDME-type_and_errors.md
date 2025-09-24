
## Type Checking and Debugging

Understanding data types is crucial for debugging. Python's dynamic typing means variables can change type, so type checking is essential.

**Reference:**

- `type(variable)` - Get variable type
- `isinstance(variable, type)` - Check if variable is specific type
- `print(f"Type: {type(var)}")` - Debug type information
- `print(f"Value: {var}")` - Debug variable values
- `print(f"Debug: {var} = {value}, type = {type(value)}")` - Complete debugging

**Brief Example:**

```python
# Type checking for debugging
user_input = "42"  # This is a string, not a number!
print(f"Input: {user_input}")
print(f"Type: {type(user_input)}")  # <class 'str'>

# Convert and verify
number = int(user_input)
print(f"Converted: {number}")
print(f"New type: {type(number)}")  # <class 'int'>

# Debugging data processing
data = [1, 2, "3", 4, 5]  # Mixed types!
for item in data:
    print(f"Item: {item}, Type: {type(item)}")
    if isinstance(item, str):
        print(f"  Converting string '{item}' to int")
        item = int(item)
```

## Error Handling Basics

Error handling prevents crashes when unexpected things happen. Python's try/except statements catch errors gracefully.

**Reference:**

- `try: ... except: ...` - Basic exception handling
- `except ValueError:` - Catch specific exception types
- Common exceptions: `ValueError`, `TypeError`, `FileNotFoundError`

**Brief Example:**

```python
# Basic error handling
try:
    number = int("not_a_number")
except ValueError:
    print("Could not convert to number")

try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
```
