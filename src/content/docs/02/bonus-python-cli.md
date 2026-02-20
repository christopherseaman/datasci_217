---
title: "Bonus: Advanced Python & CLI"
---

# Advanced Python CLI Topics

## Lambda Expressions and Functional Programming

**Conceptual Description:**
Lambda expressions provide concise syntax for creating anonymous functions, particularly useful for short operations that are used only once. They're commonly employed with higher-order functions like `sorted()`, `filter()`, and `map()` to create compact, readable code for data transformations.

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
## Lambda Functions and Functional Programming

**Conceptual Description:**
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
```
## Advanced List Comprehensions and Function-based Operations

**Conceptual Description:**
Advanced list comprehensions combine transformation, filtering, and function application in a single expression. Function-based list comprehensions apply custom functions to each element, enabling complex data transformations that go beyond simple arithmetic operations.

**Reference:**
- `[function(item) for item in iterable if condition]` - Function-based list comprehension
- `[expr1 if condition else expr2 for item in iterable]` - Conditional expressions in comprehensions
- Nested comprehensions: `[[expr for inner in outer] for outer in iterable]`
- Multiple iteration: `[expr for item1 in iter1 for item2 in iter2]`

**Brief Example:**
```python
# Function-based list comprehensions
def grade_to_letter(grade):
    if grade >= 90:
        return 'A'
    elif grade >= 80:
        return 'B'
    else:
        return 'C'

grades = [85, 92, 78, 96, 88]
letter_grades = [grade_to_letter(g) for g in grades]

# Advanced conditional comprehensions
passing_grades = [g for g in grades if g >= 80]
letter_grades = ['A' if g >= 90 else 'B' if g >= 80 else 'C' for g in grades]

# Nested comprehensions for matrix operations
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
```

## Advanced Shell Scripting

**Conceptual Description:**
Advanced shell scripting combines multiple commands with control flow structures to automate complex data processing workflows. Scripts can handle file processing, data transformation, and system operations with proper error handling and logging.

**Reference:**
- `#!/bin/bash` - Shebang line for bash scripts
- `$1, $2, $3...` - Command line arguments
- `$@` - All arguments
- `$#` - Number of arguments
- `$?` - Exit code of last command
- `if [ condition ]; then ... fi` - Conditional execution
- `for variable in list; do ... done` - Loop execution
- `while [ condition ]; do ... done` - While loops
- `case $variable in pattern) ... ;; esac` - Pattern matching

**Brief Example:**
```bash
#!/bin/bash
# Process multiple data files
for file in data/*.csv; do
    echo "Processing $file"
    head -1 "$file" > "processed/$(basename "$file")"
done
```

## Complex Command Chaining and Redirection

**Conceptual Description:**
Command chaining allows complex data processing pipelines by connecting multiple commands. Redirection controls where input comes from and where output goes, enabling powerful data transformations and file processing workflows.

**Reference:**
- `command1 | command2` - Pipe output to next command
- `command1 && command2` - Run command2 only if command1 succeeds
- `command1 || command2` - Run command2 only if command1 fails
- `command > file` - Redirect output to file
- `command >> file` - Append output to file
- `command < file` - Use file as input
- `command 2> file` - Redirect error output
- `command &> file` - Redirect both stdout and stderr
- `command <<< "text"` - Here string input

**Brief Example:**
```bash
# Complex data processing pipeline
grep "error" logfile.txt | wc -l    # Count error lines
ls *.csv | head -5 > filelist.txt   # Save first 5 CSV files to list

# Error handling with redirection
backup_script.sh > backup.log 2>&1  # Capture both stdout and stderr

# Process substitution for complex operations
diff <(sort file1.txt) <(sort file2.txt)  # Compare sorted files
```

## Process Substitution and Advanced Pipelines

**Conceptual Description:**
Process substitution allows using command output as file arguments, enabling complex data comparisons and processing. Advanced pipelines combine multiple tools for sophisticated data manipulation tasks.

**Reference:**
- `<(command)` - Process substitution for input
- `>(command)` - Process substitution for output
- `tee` - Split output to multiple destinations
- `xargs` - Build command lines from input
- `parallel` - Execute commands in parallel

**Brief Example:**
```bash
# Compare two data files after processing
diff <(grep "sales" data1.csv | sort) <(grep "sales" data2.csv | sort)

# Process multiple files in parallel
find . -name "*.log" | xargs -I {} sh -c 'echo "Processing {}"; wc -l {}'

# Split output for monitoring and logging
long_running_command | tee output.log | grep "progress"
```

## Additional Shell Scripting Fundamentals

### Advanced Shell Scripting Concepts

Shell scripting automates repetitive tasks and creates reusable command sequences. Scripts combine multiple commands with control flow to handle complex data processing workflows.

**Reference:**

- `#!/bin/bash` - Shebang line for bash scripts
- `$1, $2, $3...` - Command line arguments
- `$@` - All arguments
- `$#` - Number of arguments
- `$?` - Exit code of last command
- `if [ condition ]; then ... fi` - Conditional execution
- `for variable in list; do ... done` - Loop execution

**Brief Example:**

```bash
#!/bin/bash
# Process multiple data files
for file in data/*.csv; do
    echo "Processing $file"
    head -1 "$file" > "processed/$(basename "$file")"
done
```

### Enhanced Command Chaining and Redirection

Command chaining allows complex data processing pipelines by connecting multiple commands. Redirection controls where input comes from and where output goes, enabling powerful data transformations.

**Reference:**

- `command1 | command2` - Pipe output to next command
- `command1 && command2` - Run command2 only if command1 succeeds
- `command1 || command2` - Run command2 only if command1 fails
- `command > file` - Redirect output to file
- `command >> file` - Append output to file
- `command < file` - Use file as input

**Brief Example:**

```bash
grep "error" logfile.txt | wc -l    # Count error lines
ls *.csv | head -5 > filelist.txt   # Save first 5 CSV files to list