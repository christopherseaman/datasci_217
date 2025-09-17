Python and Command Line Foundations

Welcome to your journey into data science! Today we're building the foundation you'll use every single day as a data scientist: basic Python programming and command line navigation. Think of this as learning to walk before we run.

# Why Both Python and Command Line?

You might wonder: "Why do I need to learn two things at once?" Here's the reality - professional data scientists constantly switch between Python scripts and command line operations. You'll write Python to analyze data, then use command line to organize files, run scripts, and manage projects.

It's like being bilingual in the data world. Python speaks to your data, command line speaks to your computer.

*[xkcd 353: "Python" - There's always a relevant XKCD!]*

# Command Line Essentials

## What is the Command Line?

The command line is your direct conversation with your computer. Instead of clicking icons, you type commands. Why? Because it's faster, more precise, and works exactly the same way on every computer.

Think of it as texting your computer instead of playing charades with icons.

## Navigation Commands

**Reference:**
- `pwd` - Print working directory (where am I?)  
- `ls` - List contents (what's here?)
- `ls -la` - List with details (show me everything)
- `cd [path]` - Change directory (go somewhere)
- `cd ..` - Go up one level
- `cd ~` - Go to home directory

**Brief Example:**
```bash
pwd                    # Shows: /Users/yourname
ls                     # Shows files in current directory  
cd Documents           # Move to Documents folder
pwd                    # Shows: /Users/yourname/Documents
```

# LIVE DEMO!
*Navigation practice: We'll walk through your file system together*

## File Operations  

**Reference:**
- `mkdir [name]` - Make directory
- `mkdir -p [path/to/nested]` - Make nested directories
- `touch [filename]` - Create empty file
- `cp [source] [destination]` - Copy file
- `mv [source] [destination]` - Move/rename file
- `rm [filename]` - Remove file (careful!)

**Brief Example:**
```bash
mkdir my_data_project     # Create project folder
cd my_data_project       # Enter the folder
touch analysis.py        # Create Python file
mkdir data              # Create data subfolder
```

## Viewing Files

**Reference:**
- `cat [filename]` - Show entire file contents
- `head [filename]` - Show first 10 lines
- `head -n 5 [filename]` - Show first 5 lines  
- `tail [filename]` - Show last 10 lines
- `tail -n 20 [filename]` - Show last 20 lines

**Brief Example:**
```bash
head data.csv           # Quick peek at data file
tail -n 5 results.txt   # See the last few results
```

## Getting Help

**Reference:**
- `man [command]` - Manual page for command
- `[command] --help` - Quick help for command
- `which [command]` - Find where command is located

Your best friends when you're stuck!

# Python Basics

## Running Python

You have three ways to run Python:

1. **Interactive mode** (REPL): Just type `python` and start experimenting
2. **Script mode**: Write code in a file, run with `python filename.py`
3. **Jupyter notebooks**: We'll meet these later!

**Reference:**
```bash
python                  # Start interactive Python
python script.py        # Run a Python script
exit()                 # Quit interactive Python
```

## Variables and Data Types

Python stores information in variables - think of them as labeled boxes.

**Reference:**
```python
# Numbers
age = 25                    # Integer  
height = 5.8               # Float

# Text  
name = "Alice"             # String
email = 'alice@ucsf.edu'   # Single or double quotes

# True/False
is_student = True          # Boolean
has_data = False           # Boolean
```

**Brief Example:**
```python
student_name = "Bob"
student_age = 23
print(f"Student {student_name} is {student_age} years old")
# Output: Student Bob is 23 years old
```

## Basic Operations

**Reference:**
```python
# Math
result = 10 + 5         # Addition: 15
result = 10 - 3         # Subtraction: 7  
result = 4 * 6          # Multiplication: 24
result = 15 / 4         # Division: 3.75
result = 15 // 4        # Integer division: 3
result = 15 % 4         # Remainder: 3

# String operations
full_name = first + " " + last        # Concatenation
message = f"Hello {name}!"            # f-string formatting (preferred)
```

**Brief Example:**
```python
# Calculate BMI
weight_kg = 70
height_m = 1.75
bmi = weight_kg / (height_m ** 2)
print(f"BMI is {bmi:.1f}")
```

# LIVE DEMO!
*We'll create our first data science calculation together*

## Printing and Basic Input

**Reference:**
```python
print("Hello world")                    # Basic printing
print("Value:", 42)                     # Multiple values
print(f"Result: {result:.2f}")          # Formatted printing

# Basic input (we'll use this rarely)
name = input("Enter your name: ")       # Gets text from user
```

The `f""` format is your best friend - it lets you embed variables directly in text!

# Putting It All Together

Let's see how command line and Python work together:

**Brief Example:**
```bash
# Command line: Set up workspace
mkdir data_analysis
cd data_analysis
touch calculate_stats.py

# Then edit calculate_stats.py with:
```

```python
# Python: Do the calculation
data_value = 42
result = data_value * 1.5
print(f"Processed value: {result}")
```

```bash
# Command line: Run the analysis  
python calculate_stats.py
```

This is the basic workflow you'll use hundreds of times!

# Key Takeaways

1. **Command line** = talking directly to your computer
2. **Python variables** = labeled storage boxes for your data  
3. **f-strings** = easy way to format text with variables
4. **Workflow** = command line for organization, Python for analysis

You now have the fundamental skills to navigate your computer and write basic Python. These might seem simple, but they're the building blocks for everything else we'll do.

Next week: We'll learn how to save and share our work with Git and GitHub!

# Practice Challenge

Before our next class, try this workflow:
1. Create a folder called `practice` using command line
2. Make a Python script that calculates something interesting to you
3. Run it from command line
4. Use `head` to look at the first few lines of your script

Remember: Every data scientist started exactly where you are now. The key is practice, not perfection!

*[xkcd 1513: "Code Quality" - Perfect is the enemy of good enough!]*