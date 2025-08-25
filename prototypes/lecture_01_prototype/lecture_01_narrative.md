# Lecture 1: Python Fundamentals and Essential Command Line

## Overview

Welcome to your journey into data science! This lecture establishes the fundamental skills that will support everything you do throughout this course and in your professional data science career. Think of this as building your digital toolbox - we're not just learning isolated commands and syntax, but developing a way of thinking about how to communicate with computers effectively.

Data scientists spend much of their time working with files, running scripts, and managing data through text-based interfaces. The command line isn't just a relic from computing's past - it's a powerful, precise tool that lets you automate repetitive tasks, process large datasets efficiently, and work seamlessly across different computing environments. When combined with Python's intuitive syntax and powerful libraries, these skills form the foundation for sophisticated data analysis and machine learning workflows.

By the end of this lecture, you'll understand not just how to run Python code and navigate file systems, but why these skills are essential for reproducible research, collaborative development, and scaling your analysis from small experiments to production systems.

## Learning Objectives

By the end of this lecture, students will be able to:

- Navigate the command line confidently using essential commands for file operations and text manipulation
- Run Python scripts from the command line and understand the Python execution environment
- Use fundamental Python syntax for variables, data types, control structures, and functions
- Create simple Python scripts for data science applications with proper file organization
- Set up and maintain their development environment using command line tools and Python
- Integrate command line operations with Python scripts for efficient workflows

## Prerequisites

This lecture assumes basic computer literacy but no prior programming experience. Students should:

- Have access to a computer with terminal/command line capabilities
- Be comfortable with basic file operations (saving, opening, organizing files)
- Have Python 3.8+ installed (installation guidance provided)
- Be willing to learn text-based interfaces alongside graphical tools

## Core Concepts

### The Command Line: Your Gateway to Computer Power

The command line interface (CLI) might seem intimidating at first - a blank prompt waiting for your instructions. But this simplicity is its strength. Unlike graphical interfaces that limit you to predetermined actions, the command line gives you direct access to your computer's capabilities.

In data science, you'll often work with files too large for spreadsheet programs, process hundreds of files at once, or run analyses on remote servers where graphical interfaces aren't available. The command line excels in these scenarios because it's designed for automation, precision, and remote access.

```bash
# Your first command - find out where you are
pwd
# pwd stands for "print working directory"
# This shows your current location in the file system
```

Understanding file system navigation is crucial because data science projects involve managing many different types of files - data files, scripts, results, documentation. Let's explore how to move around and understand your digital environment.

#### Understanding Your Digital Environment

Every computer organizes files in a hierarchical structure - think of it as a filing cabinet with folders inside folders. Your current location in this structure is called your "working directory."

```bash
# See what's in your current location
ls
# On Windows PowerShell: dir

# The ls command reveals the contents of your current directory
# You'll see files and folders that you can navigate to
```

The command line uses paths to describe locations. There are two types: absolute paths (which start from the root of the file system) and relative paths (which start from your current location).

```bash
# Navigate to different locations
cd Documents          # Relative path - go to Documents folder in current location
cd /home/username/Documents  # Absolute path - exact location from root
cd ..                 # Go up one level (parent directory)
cd ~                  # Go to your home directory
cd                    # Also goes to home directory
```

Understanding these concepts is essential because Python scripts need to know where to find data files, where to save results, and how to reference other code files.

#### File Operations: Managing Your Data Science Projects

Data science projects typically involve multiple file types - Python scripts, data files, configuration files, results, and documentation. Learning to create, organize, and manipulate these files efficiently is fundamental to professional workflows.

```bash
# Create directories for project organization
mkdir data_science_project
mkdir data_science_project/data
mkdir data_science_project/scripts
mkdir data_science_project/results

# Create files
touch data_science_project/scripts/analysis.py
# On Windows: New-Item data_science_project/scripts/analysis.py
```

The ability to quickly create organized project structures becomes invaluable when you're working on multiple analyses or collaborating with teams.

```bash
# Copy and move files - essential for organizing and backing up work
cp analysis.py analysis_backup.py      # Create backup
mv old_analysis.py archive/             # Move to archive folder

# View file contents quickly
head data.csv           # See first 10 lines
tail error.log          # See last 10 lines
cat small_file.txt      # Display entire file
```

These operations become second nature once you realize they're faster than clicking through graphical interfaces, especially when dealing with many files.

#### Text Processing: The Foundation of Data Manipulation

Before you import data into Python, you often need to understand its structure, clean it, or extract specific parts. Command line text processing tools excel at these tasks.

```bash
# Search for patterns - crucial for exploring unknown datasets
grep "error" log_file.txt              # Find lines containing "error"
grep -i "temperature" weather_data.csv # Case-insensitive search for temperature

# Combine commands for powerful analysis
cat data.csv | grep "2023" | head -5   # Show first 5 lines from 2023
```

The pipe symbol (`|`) connects commands, allowing you to chain operations together. This concept mirrors how you'll later combine functions in Python to create data processing pipelines.

### Python: From Installation to Problem Solving

Python's popularity in data science comes from its readable syntax, extensive libraries, and strong community support. But before we explore libraries like pandas and NumPy, we need to master Python's fundamentals.

#### Setting Up Your Python Environment

```bash
# Verify your Python installation
python3 --version
# Should show something like: Python 3.9.7

# Run Python interactively for quick experiments
python3
>>> print("Hello, Data Science!")
>>> exit()
```

The Python interactive interpreter (REPL) is perfect for testing small code snippets and exploring new concepts. However, for anything substantial, you'll write scripts that can be saved, shared, and run repeatedly.

#### Python Fundamentals: Building Blocks for Data Analysis

Python's syntax is designed to be readable and expressive. Let's explore the concepts that form the foundation of every data science script.

```python
# Variables and data types - the building blocks of all programs
name = "Alice"                    # String - for text data
age = 25                         # Integer - for whole numbers
height = 5.6                     # Float - for decimal numbers
is_student = True                # Boolean - for True/False values

# Python is dynamically typed - variables can change type
data_point = 42                  # Start as integer
data_point = "forty-two"         # Now it's a string
data_point = 42.0               # Now it's a float
```

Understanding data types is crucial in data science because datasets contain different types of information - numerical measurements, categorical labels, timestamps, and boolean indicators.

```python
# String manipulation - essential for data cleaning
full_name = "john doe"
cleaned_name = full_name.title()    # "John Doe"
last_name = full_name.split()[1]    # "doe"

# F-strings for readable output
print(f"The person's name is {cleaned_name}")
```

String manipulation skills are essential because real-world data often comes with inconsistent formatting, extra whitespace, or mixed cases that need standardization.

#### Control Structures: Making Decisions and Repeating Actions

Data analysis involves making decisions based on data values and performing operations repeatedly across datasets. Python's control structures make these operations straightforward.

```python
# Conditional statements - making decisions based on data
temperature = 25

if temperature > 30:
    comfort_level = "hot"
elif temperature > 20:
    comfort_level = "warm"
elif temperature > 10:
    comfort_level = "cool"
else:
    comfort_level = "cold"

print(f"Temperature is {temperature}°C, feeling {comfort_level}")
```

This pattern appears constantly in data science - categorizing data, filtering observations, and applying different logic based on data values.

```python
# Loops - performing operations repeatedly
# Processing multiple data points
temperatures = [18, 25, 32, 15, 28]

for temp in temperatures:
    if temp > 25:
        print(f"{temp}°C is above average")
    
# Generating data ranges
for day in range(1, 8):  # Days 1 through 7
    print(f"Day {day} temperature check")
```

Loops are fundamental to data processing - whether you're analyzing each row in a dataset, processing multiple files, or generating synthetic data for testing.

#### Functions: Creating Reusable Tools

As your analysis becomes more complex, you'll find yourself repeating similar operations. Functions let you capture these operations for reuse, making your code more organized and maintainable.

```python
def analyze_temperature(temp):
    """
    Categorize a temperature reading.
    
    This function takes a temperature value and returns
    a comfort level category. It's useful for standardizing
    how we classify temperature data across different analyses.
    
    Args:
        temp (float): Temperature in Celsius
        
    Returns:
        str: Comfort level category
    """
    if temp > 30:
        return "hot"
    elif temp > 20:
        return "warm"
    elif temp > 10:
        return "cool"
    else:
        return "cold"

# Use the function
reading = 23
category = analyze_temperature(reading)
print(f"Temperature {reading}°C is {category}")
```

Functions become especially important in data science for creating repeatable analysis steps, data cleaning procedures, and custom calculations that aren't available in standard libraries.

### Integrating Command Line and Python

The real power comes from combining command line skills with Python programming. This integration allows you to create automated workflows that can process data, run analyses, and generate reports without manual intervention.

```python
# Python script: temperature_analyzer.py
import sys

def analyze_temperature(temp):
    if temp > 30:
        return "hot"
    elif temp > 20:
        return "warm"
    elif temp > 10:
        return "cool"
    else:
        return "cold"

def main():
    if len(sys.argv) != 2:
        print("Usage: python temperature_analyzer.py <temperature>")
        return
    
    try:
        temperature = float(sys.argv[1])
        result = analyze_temperature(temperature)
        print(f"Temperature {temperature}°C is {result}")
    except ValueError:
        print("Please provide a valid number")

if __name__ == "__main__":
    main()
```

```bash
# Run the script from command line
python3 temperature_analyzer.py 25
# Output: Temperature 25.0°C is warm
```

This pattern - creating Python scripts that accept command line arguments - is fundamental to building data science pipelines that can be automated and integrated into larger workflows.

## Hands-On Practice

### Exercise 1: Command Line Navigation Challenge

Practice the essential file operations that you'll use throughout your data science career.

```bash
# Create a structured project directory
mkdir datasci_project
cd datasci_project
mkdir data scripts results documentation

# Create some sample files
touch data/sample_data.csv
touch scripts/analysis.py
touch documentation/README.md

# Practice navigation
ls -la                    # List all files with details
cd scripts && pwd         # Navigate and confirm location
cd ../data && ls          # Navigate to different directory
cd .. && find . -name "*.py"  # Return to root and find Python files
```

**Expected Outcome**: Students can confidently create organized project structures and navigate between directories without using graphical file managers.

### Exercise 2: Hello Data Science Script

Create your first Python script and run it from the command line.

```python
# Create file: hello_ds.py
#!/usr/bin/env python3
"""
My first data science script!

This script demonstrates basic Python concepts and file organization.
"""

def greet_data_scientist(name):
    """
    Generate a personalized greeting for a data scientist.
    """
    return f"Hello, {name}! Welcome to the world of data science!"

def analyze_number(num):
    """
    Perform basic analysis on a number.
    """
    analysis = {
        "original": num,
        "squared": num ** 2,
        "is_even": num % 2 == 0,
        "is_positive": num > 0
    }
    return analysis

def main():
    print("=== Data Science Introduction ===")
    
    # Get user input
    name = input("What's your name? ")
    greeting = greet_data_scientist(name)
    print(greeting)
    
    # Demonstrate number analysis
    try:
        number = float(input("Enter a number to analyze: "))
        results = analyze_number(number)
        
        print(f"\nAnalysis of {results['original']}:")
        print(f"- Squared: {results['squared']}")
        print(f"- Even number: {results['is_even']}")
        print(f"- Positive: {results['is_positive']}")
    except ValueError:
        print("That wasn't a valid number, but that's okay!")
    
    print("Keep learning and exploring!")

if __name__ == "__main__":
    main()
```

```bash
# Make it executable and run it
chmod +x hello_ds.py
python3 hello_ds.py
```

**Expected Outcome**: Students understand how to create, organize, and execute Python scripts while seeing fundamental programming concepts in action.

### Exercise 3: Project Euler Problem - Mathematical Problem Solving

Solve a classic computational problem that demonstrates loops, conditionals, and mathematical thinking.

```python
# Create file: euler_problem_1.py
"""
Project Euler Problem 1: Multiples of 3 or 5

If we list all the natural numbers below 10 that are multiples of 3 or 5,
we get 3, 5, 6, and 9. The sum of these multiples is 23.

Find the sum of all the multiples of 3 or 5 below 1000.
"""

def is_multiple_of_3_or_5(number):
    """
    Check if a number is a multiple of 3 or 5.
    
    Args:
        number (int): Number to check
        
    Returns:
        bool: True if multiple of 3 or 5, False otherwise
    """
    return number % 3 == 0 or number % 5 == 0

def solve_euler_1(limit):
    """
    Find sum of multiples of 3 or 5 below the given limit.
    
    Args:
        limit (int): Upper limit (exclusive)
        
    Returns:
        int: Sum of qualifying multiples
    """
    total = 0
    multiples = []
    
    for number in range(1, limit):
        if is_multiple_of_3_or_5(number):
            multiples.append(number)
            total += number
    
    return total, multiples

def main():
    print("Project Euler Problem 1: Multiples of 3 or 5")
    print("=" * 45)
    
    # Test with small example first
    test_limit = 10
    test_sum, test_multiples = solve_euler_1(test_limit)
    print(f"\nTest case (below {test_limit}):")
    print(f"Multiples: {test_multiples}")
    print(f"Sum: {test_sum}")
    
    # Solve the actual problem
    actual_limit = 1000
    actual_sum, _ = solve_euler_1(actual_limit)
    print(f"\nActual problem (below {actual_limit}):")
    print(f"Sum of all multiples of 3 or 5: {actual_sum}")

if __name__ == "__main__":
    main()
```

```bash
# Run the solution
python3 euler_problem_1.py
```

**Expected Outcome**: Students experience the satisfaction of solving a mathematical problem computationally while practicing function design, loops, conditionals, and problem decomposition.

### Exercise 4: Integration Challenge - File Processing Pipeline

Combine command line and Python skills to create a data processing workflow.

```python
# Create file: file_processor.py
#!/usr/bin/env python3
"""
File Processing Pipeline

This script demonstrates how Python can work with command line arguments
and file operations to create automated data processing workflows.
"""

import sys
import os
from datetime import datetime

def count_lines_and_words(filename):
    """
    Count lines and words in a text file.
    
    Args:
        filename (str): Path to the file
        
    Returns:
        dict: Statistics about the file
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            line_count = len(lines)
            word_count = sum(len(line.split()) for line in lines)
            
            return {
                "filename": filename,
                "lines": line_count,
                "words": word_count,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    except FileNotFoundError:
        return {"error": f"File '{filename}' not found"}
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

def save_results(results, output_file):
    """
    Save processing results to a file.
    """
    with open(output_file, 'w') as file:
        file.write("File Processing Results\n")
        file.write("=" * 25 + "\n\n")
        
        for result in results:
            if "error" in result:
                file.write(f"ERROR: {result['error']}\n")
            else:
                file.write(f"File: {result['filename']}\n")
                file.write(f"Lines: {result['lines']}\n")
                file.write(f"Words: {result['words']}\n")
                file.write(f"Processed: {result['processed_at']}\n")
                file.write("-" * 20 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 file_processor.py <file1> [file2] [file3] ...")
        print("Example: python3 file_processor.py data.txt readme.md")
        return
    
    files_to_process = sys.argv[1:]
    results = []
    
    print("Processing files...")
    for filename in files_to_process:
        print(f"  Processing: {filename}")
        result = count_lines_and_words(filename)
        results.append(result)
        
        if "error" not in result:
            print(f"    Lines: {result['lines']}, Words: {result['words']}")
        else:
            print(f"    {result['error']}")
    
    # Save results
    output_file = "processing_results.txt"
    save_results(results, output_file)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
```

```bash
# Create some sample files to process
echo "This is a sample data file with multiple lines." > sample_data.txt
echo "Line two contains more information." >> sample_data.txt
echo "Final line completes our test data." >> sample_data.txt

echo "# Project README" > README.md
echo "This project demonstrates file processing." >> README.md

# Run the processing script
python3 file_processor.py sample_data.txt README.md

# View the results
cat processing_results.txt

# Use command line to organize results
mkdir processed_files
mv processing_results.txt processed_files/
ls processed_files/
```

**Expected Outcome**: Students see how Python scripts can accept command line arguments, process multiple files, and integrate with command line workflows to create professional data processing pipelines.

## Common Pitfalls and Troubleshooting

### Issue: "Command not found" or "python is not recognized"

**Symptoms**: When trying to run `python` or basic commands, you receive error messages saying the command doesn't exist.

**Solution**: 
1. Verify Python installation: `python3 --version`
2. On Windows, ensure Python is in your PATH environment variable
3. Use `python3` instead of `python` on Unix systems
4. Check if you need to restart your terminal after installation

**Why This Happens**: The operating system can't find the program you're trying to run because it's not in the system's PATH or isn't properly installed.

### Issue: "Permission denied" when running scripts

**Symptoms**: Error messages about lacking permissions to execute files or access directories.

**Solution**:
```bash
# Make script executable
chmod +x script.py

# Or run with python explicitly
python3 script.py
```

**Why This Happens**: Unix-like systems require explicit permission to execute files as programs. Windows typically doesn't have this issue, but may have similar security restrictions.

### Issue: "No such file or directory" despite file existing

**Symptoms**: Scripts or commands can't find files that you can see in your file manager.

**Solution**:
```bash
# Check your current location
pwd

# List files in current directory
ls -la

# Use absolute paths or navigate to correct directory
cd /path/to/your/files
python3 script.py
```

**Why This Happens**: Your script is looking for files relative to your current working directory, which might not be where you think it is. This is one of the most common beginner mistakes.

### Issue: Python syntax errors with indentation

**Symptoms**: `IndentationError` or `TabError` messages when running Python scripts.

**Solution**: Python uses indentation to define code blocks. Be consistent with either spaces or tabs (spaces are recommended).

```python
# Wrong - inconsistent indentation
if temperature > 25:
print("Hot")    # This line should be indented

# Right - consistent indentation
if temperature > 25:
    print("Hot")    # Properly indented with 4 spaces
```

**Why This Happens**: Unlike many programming languages that use brackets `{}` to group code, Python uses indentation. This makes code readable but requires attention to spacing.

## Real-World Applications

These fundamental skills form the backbone of professional data science workflows. Here's how they connect to real industry scenarios:

**Data Pipeline Development**: Large-scale data science projects often involve processing hundreds or thousands of files. The command line skills you've learned let you efficiently navigate large directory structures, while Python scripting automates repetitive processing tasks. For example, a genomics researcher might use command line tools to organize thousands of DNA sequence files, then run Python scripts to analyze each one automatically.

**Remote Computing**: Many data science computations run on remote servers or cloud platforms where graphical interfaces aren't available. Your command line skills allow you to connect to these systems, upload data, run analyses, and download results entirely through text-based interfaces.

**Reproducible Research**: The combination of organized file structures, command line automation, and Python scripting enables reproducible research. Other researchers can run your exact analysis by executing your scripts with their data, following the same directory structure you've established.

**Version Control Integration**: Professional data science teams use version control systems like Git (which you'll learn about next lecture). These tools work primarily through command line interfaces, making your shell navigation skills essential for collaboration.

**Data Quality Assessment**: Before diving into complex analyses, data scientists often need to quickly examine file contents, count records, check for missing values, or identify data quality issues. Command line text processing tools excel at these exploratory tasks, letting you understand dataset characteristics before writing more sophisticated Python analyses.

## Assessment Integration

### Formative Assessment

Quick check questions to gauge your understanding:

1. **Command Line Comprehension**: "If you're in directory `/home/user/projects` and run `cd ../documents`, where will you end up? Why?"

2. **Python Application**: "Look at this code: `temperature = '25'`. What data type is `temperature`? How would you convert it to perform mathematical operations?"

3. **Integration Thinking**: "You have 50 CSV files in a directory and need to count the rows in each. Describe how you would approach this using both command line tools and Python."

### Summative Assessment Preview

Your assignment combines these skills in a practical project:

- **Command Line Portfolio**: Create organized project directories and demonstrate file operations
- **Python Problem Solving**: Solve computational problems showing understanding of variables, loops, and functions
- **Integration Project**: Build a script that processes files and accepts command line arguments

This mirrors real-world data science tasks where you'll manage complex projects with multiple data sources, analytical scripts, and results files.

## Further Reading and Resources

### Essential Resources
- [The Linux Command Line](http://linuxcommand.org/tlcl.php) - Comprehensive guide to shell usage, chapters 1-8 cover fundamentals
- [Python.org Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide) - Official Python learning path with tutorials and exercises
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) - Practical Python applications including file processing

### Advanced Topics
- [Real Python Command Line Applications](https://realpython.com/command-line-interfaces-python-argparse/) - Building more sophisticated command line tools
- [The Missing Semester](https://missing.csail.mit.edu/) - Advanced command line tools and productivity techniques
- [Python Path and Environment Management](https://realpython.com/python-path/) - Understanding how Python finds modules and packages

### Practice Environments
- [Command Line Challenge](https://cmdchallenge.com/) - Interactive shell exercises
- [Python.org Online Console](https://www.python.org/shell/) - Web-based Python interpreter for quick testing
- [Project Euler](https://projecteuler.net/) - Mathematical programming challenges

## Next Steps

In our next lecture, we'll build on today's foundation by exploring **Data Structures and Version Control**. The file organization and Python scripting skills you've developed today will be essential as we learn to:

- Manage code changes with Git version control
- Work with Python's built-in data structures (lists, dictionaries, sets)
- Collaborate on code projects using GitHub
- Write more sophisticated Python programs using advanced control structures

The command line confidence and Python fundamentals from today's lecture directly enable next week's topics. You'll use your shell navigation skills to manage Git repositories, and your Python knowledge will expand to include the data structures that make complex data analysis possible.

Start practicing these skills daily - even five minutes of command line navigation or Python experimentation will build the muscle memory that makes advanced data science techniques feel natural.

---

*Lecture Format: Notion-Compatible Narrative with Embedded Interactive Code*
*Progressive Complexity: Fundamentals → Integration → Real-World Applications*
*Version: 1.0 - Week 2 Prototype Implementation*