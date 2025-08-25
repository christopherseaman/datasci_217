# Lecture 1: Practice Exercises

## Exercise 1: Command Line Mastery

### Setup Challenge - File Operations Practice
Create the following directory structure using only command line commands:

```bash
# Command line navigation and file operations practice
mkdir datasci_bootcamp
cd datasci_bootcamp
mkdir -p week_01/{scripts,data,results}
mkdir -p week_02/{scripts,data,results}
mkdir -p resources/{documentation,references}
```

### Command Line Skills Practice
**Navigation Practice:**
1. Navigate to different directories using relative and absolute paths
2. Use `pwd` to verify your current location
3. Practice `cd ..`, `cd ~`, and `cd /path/to/directory`

**File Operations Practice:**
1. Create sample Python scripts in the scripts directories
2. Use `ls` with different flags (`-l`, `-la`, `-lh`) to explore the structure  
3. Practice file operations: `cp`, `mv`, `rm` (carefully!)
4. Use `touch` to create empty files for testing

**Text Processing Practice:**
1. Use `echo` to create simple data files
2. Practice `cat`, `head`, `tail` on your created files
3. Try `grep` to search for patterns in files

## Exercise 2: Python Variables and Control Structures

### Temperature Analysis Script with Python Fundamentals
Create a Python script named `temperature_analyzer.py` that demonstrates:

**Variables and Data Types:**
```python
# String variables for categories
category_hot = "hot"
category_warm = "warm" 
category_cool = "cool"
category_cold = "cold"

# Numeric variables for thresholds
hot_threshold = 30.0
warm_threshold = 20.0
cool_threshold = 10.0
```

**Control Structures Practice:**
1. Use `if-elif-else` statements to categorize temperatures
2. Implement `for` loops to process multiple readings
3. Use `while` loops for user input validation
4. Practice nested control structures

### Functions and Modular Programming
Extend your temperature analyzer with functions:

```python
def categorize_temperature(temp):
    """Demonstrate function definition and return values."""
    # Function implementation here
    
def process_temperature_data(readings):
    """Practice with lists and function parameters."""
    # Function implementation here
    
def save_results_to_file(results, filename):
    """Practice file operations within functions.""" 
    # Function implementation here
```

## Exercise 3: Python Scripts and Command Line Integration

### File Processing Pipeline with Scripts
Create a complete workflow that demonstrates:

**Python Scripts Integration:**
1. Write Python scripts that accept command line arguments
2. Practice `sys.argv` for basic argument handling
3. Use `argparse` for professional argument processing
4. Create scripts that can be run from any directory

**File Operations in Python:**
```python
import os
import sys

# Demonstrate file operations
def organize_data_files():
    """Practice Python file operations."""
    # Create directories
    # Move files
    # Process file contents
    
def process_multiple_files():
    """Practice processing multiple files."""
    # Loop through files
    # Read and analyze each file
    # Combine results
```

### Advanced Integration Challenge
Create a complete data processing pipeline:

1. **Command Line Setup:** Use command line to create project structure
2. **Python Script Creation:** Write modular Python scripts with functions  
3. **File Operations:** Process multiple data files programmatically
4. **Variable Management:** Use appropriate data types for different information
5. **Control Structures:** Implement decision-making and iteration logic
6. **Integration:** Run Python scripts from command line with parameters

### Assessment Preparation Exercises

**Skills Alignment Practice:**
- **Command Line Navigation:** Practice all file system operations
- **Python Variables:** Work with strings, numbers, booleans, lists
- **Control Structures:** Master if/elif/else, for loops, while loops  
- **Functions:** Write reusable code with proper parameters and returns
- **File Operations:** Read from and write to files using Python
- **Script Execution:** Run Python scripts from command line with arguments

This comprehensive practice ensures you master all fundamental skills needed for data science workflows.