# Python Syntax Quick Reference

## Basic Data Types and Variables

### Variable Assignment
```python
# String variables
name = "Alice"
message = 'Hello World'

# Numeric variables  
age = 25            # Integer
height = 5.6        # Float (decimal)
temperature = -10.5 # Can be negative

# Boolean variables
is_student = True
has_graduated = False
```

### String Operations
```python
# String manipulation
full_name = "john doe"
cleaned = full_name.title()     # "John Doe"
upper_case = full_name.upper()  # "JOHN DOE"
words = full_name.split()       # ["john", "doe"]

# String formatting
name = "Alice"
age = 25
message = f"Hello, {name}! You are {age} years old."
```

## Control Structures

### Conditional Statements
```python
# If-elif-else structure
temperature = 25

if temperature > 30:
    status = "hot"
elif temperature > 20:
    status = "warm"
elif temperature > 10:
    status = "cool"
else:
    status = "cold"

print(f"Temperature is {status}")
```

### Loops

#### For Loops
```python
# Loop through a list
temperatures = [18, 25, 32, 15, 28]
for temp in temperatures:
    print(f"Temperature: {temp}°C")

# Loop with range
for day in range(1, 8):  # 1 through 7
    print(f"Day {day}")

# Loop with enumeration
for index, temp in enumerate(temperatures):
    print(f"Reading {index + 1}: {temp}°C")
```

#### While Loops
```python
# Count down example
count = 5
while count > 0:
    print(f"Countdown: {count}")
    count -= 1
print("Launch!")
```

## Functions

### Function Definition
```python
def greet_person(name, age=25):
    """
    Greet a person with their name and age.
    
    Args:
        name (str): Person's name
        age (int): Person's age (default: 25)
        
    Returns:
        str: Greeting message
    """
    return f"Hello, {name}! You are {age} years old."

# Function call
message = greet_person("Alice", 30)
print(message)

# Using default parameter
message2 = greet_person("Bob")
print(message2)
```

### Functions with Multiple Returns
```python
def analyze_number(num):
    """
    Analyze a number and return various properties.
    
    Args:
        num (float): Number to analyze
        
    Returns:
        tuple: (is_positive, is_even, squared_value)
    """
    is_positive = num > 0
    is_even = num % 2 == 0
    squared = num ** 2
    
    return is_positive, is_even, squared

# Unpack multiple return values
positive, even, square = analyze_number(4)
print(f"Positive: {positive}, Even: {even}, Squared: {square}")
```

## Data Collections

### Lists
```python
# Create lists
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]
mixed = [1, "hello", 3.14, True]

# List operations
numbers.append(6)           # Add to end
numbers.insert(0, 0)        # Insert at index 0
first_name = names[0]       # Access by index
last_three = numbers[-3:]   # Get last 3 elements

# List comprehension
squares = [x**2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]
```

### Dictionaries
```python
# Create dictionary
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Access values
name = person["name"]
age = person.get("age", 0)  # Get with default value

# Add/modify values
person["job"] = "Data Scientist"
person["age"] = 31

# Iterate through dictionary
for key, value in person.items():
    print(f"{key}: {value}")
```

## Input and Output

### User Input
```python
# Get string input
name = input("What is your name? ")

# Get numeric input with error handling
try:
    age = int(input("What is your age? "))
    print(f"You are {age} years old")
except ValueError:
    print("Please enter a valid number")
```

### File Operations
```python
# Reading files
with open("data.txt", "r") as file:
    content = file.read()
    lines = file.readlines()

# Writing files
data = ["line 1", "line 2", "line 3"]
with open("output.txt", "w") as file:
    for line in data:
        file.write(line + "\n")
```

## Common Patterns

### Error Handling
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("This always runs")
```

### List Processing
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter even numbers
evens = [x for x in numbers if x % 2 == 0]

# Transform all numbers
doubled = [x * 2 for x in numbers]

# Find maximum/minimum
max_num = max(numbers)
min_num = min(numbers)

# Sum and average
total = sum(numbers)
average = total / len(numbers)
```

### String Processing
```python
text = "Hello, World! How are you today?"

# Split into words
words = text.split()

# Count occurrences
word_count = len(words)
char_count = len(text)

# Clean and normalize
cleaned = text.lower().replace(",", "").replace("!", "")
```

## Useful Built-in Functions

```python
# Mathematical functions
abs(-5)         # Absolute value: 5
round(3.7)      # Round to integer: 4
round(3.14159, 2)  # Round to 2 decimals: 3.14

# Type checking and conversion
type(42)        # <class 'int'>
str(42)         # Convert to string: "42"
int("42")       # Convert to integer: 42
float("3.14")   # Convert to float: 3.14

# Sequence functions
len([1,2,3])    # Length: 3
sorted([3,1,2]) # Sort: [1, 2, 3]
reversed([1,2,3])  # Reverse iterator
```

## Command Line Integration

### System Arguments
```python
import sys

# Access command line arguments
script_name = sys.argv[0]
arguments = sys.argv[1:]  # All arguments except script name

if len(sys.argv) > 1:
    first_arg = sys.argv[1]
    print(f"First argument: {first_arg}")
```

### Argument Parsing
```python
import argparse

parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('--input', help='Input file path')
parser.add_argument('--output', default='result.txt', help='Output file path')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

args = parser.parse_args()

if args.verbose:
    print(f"Processing {args.input} -> {args.output}")
```

## Common Mistakes and Solutions

### Indentation
```python
# Wrong - inconsistent indentation
if True:
print("This will cause an IndentationError")

# Right - consistent indentation (4 spaces recommended)
if True:
    print("This works correctly")
```

### Variable Scope
```python
# Global vs local variables
global_var = "I'm global"

def my_function():
    local_var = "I'm local"
    print(global_var)    # Can access global
    print(local_var)     # Can access local

my_function()
print(global_var)       # Can access global
# print(local_var)      # Error! Can't access local outside function
```

### Mutable vs Immutable
```python
# Strings are immutable
text = "hello"
text.upper()      # Returns "HELLO" but doesn't change text
print(text)       # Still "hello"
text = text.upper()  # Must reassign to change

# Lists are mutable
numbers = [1, 2, 3]
numbers.append(4)    # Changes the original list
print(numbers)       # Now [1, 2, 3, 4]
```