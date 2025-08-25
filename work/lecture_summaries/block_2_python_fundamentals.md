# BLOCK 2: PYTHON FUNDAMENTALS (Lectures 4-6)
## Core Programming and Scientific Computing Skills

### BLOCK OVERVIEW
This block establishes solid Python programming foundations and introduces scientific computing concepts. Students progress from basic Python syntax to NumPy array operations, building the programming skills necessary for data manipulation and analysis.

**Block Learning Objectives:**
- Master Python data types, control structures, and functions
- Implement file operations and error handling patterns
- Apply NumPy for numerical computing and array operations
- Develop debugging and code quality practices
- Understand virtual environments and package management

---

## LECTURE 4: Python Basics and Programming Concepts
**Duration**: 90 minutes | **Content Reduction**: 12% from current Lectures 01 & 04

### Learning Objectives
By the end of this lecture, students will be able to:
- Use Python data types effectively (strings, numbers, booleans)
- Implement control structures (conditionals, loops) for data processing
- Write and call functions with parameters and return values
- Work with Python's core data structures (lists, dictionaries, tuples, sets)
- Import and use Python modules and packages

### Content Sources & Integration
**Primary Sources:**
- Current Lecture 01: Python basics (full retention, reorganized)
- Current Lecture 03: Data structures (90% retention)
- Current Lecture 04: Functions and modules (80% retention)

**Content Trimming (12% reduction):**
- **REMOVE**: Command-line integration details (covered in Block 1)
- **CONDENSE**: Installation and setup (prerequisites from Block 1)
- **STREAMLINE**: Focus on data science relevant examples
- **DEFER**: Advanced list comprehensions to later lectures

### Detailed Content Structure

#### Python Fundamentals Review & Extension (25 min)
**Data Types & Variables:**
- Numbers: integers, floats, mathematical operations
- Strings: creation, manipulation, f-strings
- Booleans: logical operations, truthiness
- Dynamic typing and duck typing concepts

**Operators & Expressions:**
- Arithmetic, comparison, logical operators
- String concatenation and formatting
- Variable assignment and naming conventions

#### Control Structures for Data Processing (25 min)
**Conditional Logic:**
- `if`, `elif`, `else` statements
- Compound conditions with `and`, `or`, `not`
- Practical applications in data filtering

**Iteration Patterns:**
- `for` loops with ranges and sequences
- `while` loops for conditional iteration
- Loop control: `break`, `continue`
- Nested loops for multi-dimensional data

#### Core Data Structures (30 min)
**Lists: Sequential Data Handling**
- Creation, indexing, slicing
- Methods: `append`, `extend`, `insert`, `remove`, `pop`
- List operations for data collection
- Sorting: `sort()` vs `sorted()`

**Dictionaries: Key-Value Data Mapping**
- Creation and access patterns
- Methods: `keys()`, `values()`, `items()`
- Dictionary comprehensions (basic)
- Use cases in data science

**Tuples & Sets: Specialized Collections**
- Tuples: immutable sequences, unpacking
- Sets: unique elements, set operations
- When to use each data structure

#### Functions & Modules (10 min)
**Function Design:**
- Definition, parameters, return values
- Default arguments and keyword arguments
- Function scope and variable visibility
- Documentation with docstrings

**Module System:**
- Importing standard library modules
- Package installation with pip
- Common data science imports preview

### Advanced Topics Introduced
- Nested data structures for complex data
- String methods for text processing
- Range objects and memory efficiency
- Basic error types (preview)

### Prerequisites
- Block 1: Command Line Mastery
- Basic programming concepts helpful but not required

### Assessment Integration
Students complete programming exercises demonstrating:
- Data type manipulation
- Control structure implementation
- Function writing and usage
- Data structure selection and usage

---

## LECTURE 5: File Operations, Error Handling, and Best Practices
**Duration**: 90 minutes | **Content Reduction**: 8% from current Lectures 04 & 09

### Learning Objectives
By the end of this lecture, students will be able to:
- Read from and write to files using proper Python patterns
- Handle exceptions gracefully using try/except blocks
- Debug Python code using print statements and IDE tools
- Apply code quality practices and linting tools
- Manage command-line arguments and system integration

### Content Sources & Integration
**Primary Sources:**
- Current Lecture 04: File operations (full retention)
- Current Lecture 09: Error handling and debugging (80% retention)
- Current Lecture 04: Command-line arguments (full retention)

**Content Trimming (8% reduction):**
- **CONDENSE**: Advanced debugging techniques (focus on practical methods)
- **STREAMLINE**: Linting tool configuration (provide preconfigured setup)
- **DEFER**: Complex subprocess operations

### Detailed Content Structure

#### File Operations & Data Persistence (30 min)
**File I/O Fundamentals:**
- Opening files: modes (`r`, `w`, `a`, `b`)
- Context managers: `with` statement importance
- Reading: `read()`, `readline()`, `readlines()`
- Writing: `write()`, `writelines()`

**Practical File Patterns:**
- Processing files line by line
- CSV handling with built-in tools
- JSON data reading and writing
- File path operations with `os.path`

**Directory Operations:**
- Creating directories: `os.mkdir()`, `os.makedirs()`
- Listing contents: `os.listdir()`, `glob` module
- File existence checking
- Working directory management

#### Error Handling & Robustness (25 min)
**Exception Fundamentals:**
- Understanding Python error types
- Common exceptions: `FileNotFoundError`, `ValueError`, `TypeError`
- The exception hierarchy

**Try/Except Patterns:**
- Basic try/except structure
- Handling specific exception types
- Multiple exception handling
- `finally` blocks for cleanup
- `else` clause usage

**Defensive Programming:**
- Input validation strategies
- Graceful error recovery
- Error logging and reporting
- When to catch vs when to let errors propagate

#### Debugging & Code Quality (20 min)
**Debugging Strategies:**
- Print debugging: strategic placement and formatting
- VS Code debugger: breakpoints, variable inspection
- Interactive debugging with `breakpoint()`
- Reading error messages and tracebacks

**Code Quality Tools:**
- Linting with `ruff` or `pylint`
- Code formatting with `black` (introduction)
- Common code smells and fixes
- Documentation best practices

#### System Integration (15 min)
**Command-Line Interface:**
- `sys.argv` for basic argument handling
- `argparse` for structured CLI tools
- Environment variables with `os.environ`
- Exit codes and status reporting

**Subprocess Operations:**
- Running external commands with `subprocess`
- Capturing output and handling errors
- Security considerations
- Integration with shell scripts

### Advanced Topics Introduced
- Context manager creation
- Custom exception classes
- Logging module basics
- Testing concepts (preview)

### Prerequisites
- Lecture 4: Python Basics and Programming Concepts
- Block 1: Command Line Mastery

### Assessment Integration
Students create robust programs demonstrating:
- Proper file handling with error checking
- Exception handling implementation
- Debugging skill application
- Command-line tool creation

---

## LECTURE 6: NumPy and Scientific Computing Introduction
**Duration**: 90 minutes | **Content Reduction**: 10% from current Lecture 05

### Learning Objectives
By the end of this lecture, students will be able to:
- Create and manipulate NumPy arrays efficiently
- Perform mathematical operations on arrays using vectorization
- Apply array indexing, slicing, and reshaping operations
- Understand broadcasting rules for array operations
- Use NumPy for basic scientific computing tasks

### Content Sources & Integration
**Primary Sources:**
- Current Lecture 05: NumPy introduction (90% retention)
- Current Lecture 05: Array operations (full retention)
- Current Lecture 05: Mathematical functions (80% retention)

**Content Trimming (10% reduction):**
- **CONDENSE**: Advanced indexing techniques (cover in later lectures)
- **DEFER**: Complex mathematical functions to specialized courses
- **STREAMLINE**: Focus on data science applications

### Detailed Content Structure

#### NumPy Fundamentals (25 min)
**Array Creation & Properties:**
- `ndarray` object: dimensions, shape, dtype
- Array creation: `array()`, `zeros()`, `ones()`, `arange()`
- Array attributes: `shape`, `size`, `ndim`, `dtype`
- Memory efficiency vs Python lists

**Data Types & Memory Management:**
- NumPy data types: int, float, bool, complex
- Choosing appropriate data types
- Memory layout and performance implications
- Array copying vs views

#### Array Operations & Vectorization (30 min)
**Element-wise Operations:**
- Arithmetic operations: `+`, `-`, `*`, `/`, `**`
- Comparison operations and boolean arrays
- Universal functions (ufuncs)
- Performance benefits of vectorization

**Mathematical Functions:**
- Basic math: `np.sqrt()`, `np.exp()`, `np.log()`
- Trigonometric functions
- Statistical functions: `mean()`, `std()`, `sum()`
- Aggregation along axes

**Broadcasting Rules:**
- Compatible array shapes
- Broadcasting examples and applications
- Common broadcasting patterns
- Avoiding broadcasting pitfalls

#### Array Manipulation (25 min)
**Indexing & Slicing:**
- Basic indexing: single elements, slices
- Multi-dimensional indexing
- Boolean indexing for data filtering
- Fancy indexing with arrays

**Shape Manipulation:**
- Reshaping: `reshape()`, `flatten()`, `ravel()`
- Array stacking: `vstack()`, `hstack()`, `concatenate()`
- Splitting arrays: `split()`, `hsplit()`, `vsplit()`
- Transposition and axis manipulation

#### Scientific Computing Applications (10 min)
**Real-world Examples:**
- Data normalization and standardization
- Distance calculations
- Basic statistics on datasets
- Image processing concepts (as arrays)

**Integration with Python:**
- Converting between lists and arrays
- NumPy array methods vs functions
- Memory considerations
- When to use NumPy vs pure Python

### Advanced Topics Introduced
- Linear algebra operations preview
- Random number generation
- Array masking and advanced selection
- Performance optimization concepts

### Prerequisites
- Lecture 4: Python Basics and Programming Concepts
- Lecture 5: File Operations, Error Handling, and Best Practices
- Basic mathematics (algebra, basic statistics)

### Assessment Integration
Students complete exercises demonstrating:
- Array creation and manipulation
- Vectorized operations implementation
- Broadcasting application
- Scientific computing problem solving

---

## BLOCK 2 INTEGRATION ASSESSMENT

### Capstone Project: Data Processing Pipeline
Students create a complete Python program that demonstrates:

1. **Data Input & Validation**
   - Read data from multiple file formats
   - Implement proper error handling
   - Validate data integrity

2. **Data Processing with NumPy**
   - Convert data to NumPy arrays
   - Perform mathematical operations
   - Apply statistical analysis
   - Handle missing data

3. **Program Structure & Quality**
   - Well-organized functions
   - Proper documentation
   - Error handling throughout
   - Command-line interface

4. **Testing & Debugging**
   - Include test cases
   - Demonstrate debugging process
   - Handle edge cases gracefully

### Block Learning Outcomes Validation
- **Programming Proficiency**: Solid Python foundation for data science
- **Scientific Computing**: NumPy array operations and mathematical thinking
- **Software Engineering**: Error handling, debugging, and code quality
- **Problem Solving**: Breaking down complex problems into manageable components

### Preparation for Block 3
Students now have the programming foundation necessary to:
- Work with specialized data science libraries
- Implement complex data manipulation workflows
- Handle real-world data processing challenges
- Apply mathematical concepts to data problems

---

## BLOCK 2 SUMMARY

**Total Duration**: 4.5 hours (3 × 90-minute lectures)
**Content Reduction**: 10% average across block
**Skills Emphasis**: Programming fundamentals and numerical computing
**Assessment Strategy**: Progressive skill building with practical applications

This block transforms students from programming novices to competent Python practitioners with solid foundations in scientific computing. The emphasis on practical applications and error handling prepares students for the complexities of real-world data science work, while NumPy introduction establishes the mathematical computing foundation essential for advanced data science techniques.