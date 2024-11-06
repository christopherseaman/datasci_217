# Everyday Python

- Quick summary of where we've been
- Debugging
  - Linting (`pylint` & `ruff`)
  - Debugging techniques
    - Reading error messages
    - Debugging statements
    - Using the VS Code debugger
- Error Handling
  - Try/except blocks
  - Exception types
  - Custom exceptions
- Machine Learning
  - Framework Overview
    - Keras/TensorFlow: High-level, production-ready
    - PyTorch: Research-focused, flexible
  - Key Concepts
    - Data preprocessing & normalization
    - Model architecture (layers, activations)
    - Training & evaluation
    - Model deployment considerations

---

## Basic Python

- Variables and Data Types
  - Integers, floats, strings
  - Variables are dynamically typed
  - Type conversion and checking
  - String operations and f-strings

- Control Structures
  - If/elif/else conditionals
  - For and while loops
  - Break and continue statements
  - Compound conditions with `and`, `or`, `not`

---

## Functions and Methods

- Functions and Methods
  - Function definition with `def`
  - Parameters and return values
  - Default arguments
  - Command line arguments

- Packages and Modules
  - Installing packages
  - Importing with aliases
  - Specific functions and classes
  - Managing virtual environments

---

## Data Structures

- Lists
  - Creation and indexing
  - List methods (append, extend, pop)
  - List slicing and operations
  - List comprehensions
  - Sorting and searching

- Dictionaries
  - Key-value pairs
  - Dictionary methods
  - Nested dictionaries
  - Dictionary comprehensions
  - Default dictionaries

---

## Data Structures II

- Sets
  - Unique elements
  - Set operations (union, intersection)
  - Set methods
  - Set comprehensions

- Tuples
  - Immutable sequences
  - Tuple packing/unpacking
  - Named tuples
  - Using tuples as dictionary keys

---

## File Operations

- File Handling
  - Opening and closing files
  - Reading and writing text files
  - Context managers (`with` statement)
  - Binary file operations
  - CSV and JSON handling

- Path Operations
  - Path manipulation with `os.path`
  - Modern path handling with `pathlib`
  - Directory operations
  - File system navigation

---

## Numerical Packages

- NumPy
  - Arrays and operations
  - Broadcasting
  - Mathematical functions
  - Array manipulation

- Pandas
  - Series and DataFrames
  - Data loading and saving
  - Data cleaning and transformation
  - Grouping and aggregation
  - Time series functionality

---

## Data Visualization

- Matplotlib
- Seaborn statistical plots
- Interactive visualization
- Customizing plots

---

## Statistical Methods

- Time Series Analysis
  - DateTime handling
  - Resampling and rolling windows
  - Seasonal decomposition
  - ARIMA models
- statsmodels
  - Linear regression
  - Generalized linear models
  - Statistical tests
  - Model diagnostics
- Machine Learning
  - scikit-learn basics
  - Model selection and evaluation
  - Feature engineering
  - Cross-validation

---

## Data Science Fundamentals

- Jupyter Notebooks
  - Remote access and configuration
  - Magic commands
  - Cell execution and kernel management
- NumPy
  - Array operations and broadcasting
  - Mathematical functions
  - Array manipulation and indexing
  - Universal functions (ufuncs)
- Pandas
  - Series and DataFrame objects
  - Data loading and manipulation
  - Missing data handling
  - Grouping and aggregation

---

## Running System Commands (1/3)

The `subprocess` module provides a powerful interface to run external commands:

```python
import subprocess

# Basic command execution
result = subprocess.run(['ls', '-l'], capture_output=True, text=True)
print(result.stdout)

# Shell commands (use shell=True)
files = subprocess.check_output('find . -name "*.py"', shell=True).decode()

# Handling errors
try:
    subprocess.run(['nonexistent_command'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Command failed with exit code {e.returncode}")
```

<!--
- Run system commands from Python
- Capture command output
- Handle command failures
-->

---

## Running System Commands (2/3)

Common subprocess patterns:

```python
# Running command and capturing output
def get_git_branch():
    result = subprocess.run(['git', 'branch', '--show-current'],
                          capture_output=True, text=True)
    return result.stdout.strip()

# Running command with input
def compress_file(filename):
    with open(filename, 'rb') as f:
        proc = subprocess.run(['gzip', '-c'],
                            stdin=f,
                            stdout=subprocess.PIPE)
        return proc.stdout

# Running command with environment variables
env = os.environ.copy()
env['DEBUG'] = '1'
subprocess.run(['my_script.py'], env=env)
```

<!--
- Capture command output
- Provide input to commands
- Modify environment variables
-->

---

## Running System Commands (3/3)

Best practices and common pitfalls:

```python
# DON'T: Unsafe command construction
cmd = f"find {user_input} -type f"  # Command injection risk!
subprocess.run(cmd, shell=True)

# DO: Safe command construction
subprocess.run(['find', user_input, '-type f'])

# DON'T: Ignore errors
subprocess.run(['risky_command'])  # Might fail silently

# DO: Handle errors properly
try:
    subprocess.run(['risky_command'], check=True)
except subprocess.CalledProcessError as e:
    logging.error(f"Command failed: {e}")

# DO: Set timeouts for long-running commands
try:
    subprocess.run(['slow_command'], timeout=60)
except subprocess.TimeoutExpired:
    print("Command timed out")
```

<!--
- Avoid command injection
- Handle errors explicitly
- Set appropriate timeouts
- Use logging for errors
-->

---

## Code Quality Tools: Linters (1/5)

A "linter" is a program that highlights potential errors before you even try running the code. There are linters for pretty much every language you can think of, even Markdown. For Python, the linter that I recommend trying is `ruff`. It is much faster than `pylint` and I find it gets "confused" less often about the code context.

```python
# Installing linters
pip install pylint ruff

# Running pylint
pylint my_script.py

# Running ruff
ruff check .

# Example pylint output:
************* Module my_script
my_script.py:10:0: C0303: Trailing whitespace (trailing-whitespace)
my_script.py:15:0: C0116: Missing function docstring (missing-docstring)
```

<!--
- Linters catch common mistakes
- Enforce coding standards
- Improve code quality
- Prevent bugs before runtime
-->

---

## Understanding Errors (2/5)

```python
# 1. Print Debugging
def calculate_total(items):
    print(f"Debug: items received = {items}")  # Debug print
    total = 0
    for item in items:
        print(f"Debug: processing item = {item}")  # Debug print
        total += item['price']
    return total

# 2. Interactive Debugging with pdb (advanced)
def process_data(data):
    results = []
    for item in data:
        breakpoint()  # Starts interactive debugger
        result = complex_calculation(item)
        results.append(result)
    return results

# 3. Common Error Patterns
# NameError: Using undefined variables
def process_stats():
    total = count + 1  # count is not defined
    return total

# TypeError: Mixing incompatible types
def calculate_average(numbers):
    total = "0"  # String instead of number
    for num in numbers:
        total += num  # Can't add number to string
    return total / len(numbers)

# IndexError: Invalid list access
def get_first_elements(list1, list2):
    return [list1[0], list2[0]]  # Error if any list is empty

# KeyError: Missing dictionary key
def get_user_info(user_dict):
    return f"{user_dict['name']} is {user_dict['age']}"  # Error if keys don't exist

# AttributeError: Invalid object attributes
class User:
    def __init__(self, name):
        self.name = name

user = User("Alice")
email = user.email  # email attribute doesn't exist

# ValueError: Invalid type conversion
def parse_user_data(data_str):
    user_id = int(data_str)  # Error if data_str isn't a valid integer
    return user_id
```

<!--
- Use print statements strategically
- Interactive debugging with pdb
- Common error patterns
- Debug step by step
-->

---

## VS Code Debugger (3/5)

See the docs at [https://code.visualstudio.com/Docs/editor/debugging](https://code.visualstudio.com/Docs/editor/debugging)

```python
# debug_example.py
def process_list(items):
    total = 0
    for i, item in enumerate(items):
        # Set a breakpoint here in VS Code
        value = complex_calculation(item)
        total += value
    return total

def complex_calculation(x):
    # Step through this function
    intermediate = x * 2
    result = intermediate + 5
    return result

# Test data with potential issues
numbers = [1, 2, "3", 4]  # Bug: string in list
result = process_list(numbers)

<!--
- Set breakpoints in VS Code
- Step through code execution
- Inspect variables in debug view
- Configure debug settings
-->

---

![bg contain](media/debug_start.png)

---

![bg contain](media/debug_vscode.png)

---

## Try/Except Basics (4/5)

```python
def safe_divide(x, y):
    try:
        result = x / y
        return result
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return None
    except TypeError as e:
        print(f"Error: Invalid types - {e}")
        return None
    finally:
        print("Division operation attempted")

# Example usage:
print(safe_divide(10, 0))       # Handles ZeroDivisionError
print(safe_divide("10", 2))     # Handles TypeError
```

<!--
- Handle specific exceptions
- Use descriptive error messages
- Implement cleanup with finally
- Return meaningful results
-->

---

## Exception Types (5/5)

```python
# Common built-in exceptions and when they occur
def demonstrate_exceptions():
    # IndexError
    list_demo = [1, 2, 3]
    try:
        value = list_demo[5]
    except IndexError as e:
        print(f"Index error: {e}")
    
    # TypeError
    try:
        result = "2" + 2
    except TypeError as e:
        print(f"Type error: {e}")
    
    # ValueError
    try:
        number = int("abc")
    except ValueError as e:
        print(f"Value error: {e}")
    
    # FileNotFoundError
    try:
        with open("nonexistent.txt") as f:
            content = f.read()
    except FileNotFoundError as e:
        print(f"File error: {e}")
```

<!--
- Choose appropriate exceptions
- Handle multiple error types
- Provide context in messages
- Learn from common errors
-->

---

## Deep Learning Frameworks: TensorFlow vs PyTorch

### Framework Comparison

#### TensorFlow/Keras
- **Origin**: Google
- **Strengths**
  - Production-ready
  - Static computation graphs
  - Enterprise & industry standard
  - Keras as high-level API
- **Best For**
  - Web/mobile app deployment
  - Large-scale industrial applications
  - TensorBoard visualization

#### PyTorch
- **Origin**: Facebook
- **Strengths**
  - Dynamic computation graphs
  - Pythonic, flexible design
  - Easier debugging
  - Research community favorite
- **Best For**
  - Academic research
  - Rapid prototyping
  - Custom architectures

### Model Training Lifecycle

1. **Data Preparation**
   - Collect and clean dataset
   - Normalize input features
   - Split into train/validation sets
   - Create data loaders

2. **Model Architecture**
   - Define neural network layers
   - Choose activation functions
   - Configure model complexity
   - Consider regularization techniques

3. **Training Process**
   - Select loss function
   - Choose optimization algorithm
   - Set hyperparameters
   - Implement training loop
   - Monitor training metrics

4. **Evaluation**
   - Validate on test set
   - Compute accuracy/loss
   - Analyze model performance
   - Detect overfitting

### Complete PyTorch Example

For a complete, commented example of training a neural network on the MNIST dataset, see the accompanying script:
[`pytorch_mnist_example.py`](pytorch_mnist_example.py)

This script demonstrates:
- Data loading with torchvision
- Neural network design
- Training and validation process
- Performance metrics calculation

### Key Considerations

- **Similarities**
  - GPU acceleration
  - High-level neural network APIs
  - Extensive documentation
  - Active communities

- **Differences**
  - Graph construction (static vs dynamic)
  - Debugging approach
  - Research vs production focus

### Practical Recommendations

- Start with simple architectures
- Always normalize data
- Use validation sets
- Monitor training metrics
- Experiment with both frameworks
