Python and Command Line Foundations

Welcome to your journey into data science! Today we're building the foundation you'll use every single day as a data scientist: basic Python programming and command line navigation. Think of this as learning to walk before we run.

## Class Structure

- **Lectures** cover new material with hands-on demos
- **Lectures end** with a practical assignment
- **Lab** for help completing the practical assignment
- **Assignments** are always due the following week unless otherwise noted

# Getting Started: Your First Steps

Before we dive into the technical content, let's make sure you have everything you need to succeed. This section covers the essential setup that every data scientist needs.

## Getting to the Command Line

The command line is your direct conversation with your computer. Here's how to access it on different systems:

**Windows Users:**
- **PowerShell** (built-in): Press `Win + X`, then select "Windows PowerShell"
- **Windows Subsystem for Linux (WSL)** (recommended): Run `wsl --install` in PowerShell as Administrator
- **GitHub Codespaces** (cloud option): No installation needed

**Mac Users:**
- **Terminal** (built-in): Press `Cmd + Space`, type "Terminal", press Enter
- **GitHub Codespaces** (cloud option): No installation needed

**Cloud Options:**
- **GitHub Codespaces**: Free tier available, works on any device with internet

## Installing Python

Python is the language of data science. Here's how to get it on your system:

**Windows:**
```bash
# Option 1: Official installer from python.org
# Download Python 3.12+ from https://python.org

# Option 2: Using winget (Windows Package Manager)
winget install -e --id Python.Python.3.12
```

**Mac:**
```bash
# Option 1: Official installer from python.org
# Download Python 3.12+ from https://python.org

# Option 2: Using Homebrew (recommended)
# First install Homebrew from https://brew.sh
brew install python3
```

**WSL Ubuntu:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

**Verify Installation:**
```bash
python3 --version
# Should show: Python 3.12.x (or similar)
```

## Text Editor Options

You'll need a good text editor to write Python code. Here are your options:

**Visual Studio Code (Recommended):**
- Free, powerful, and perfect for data science
- Available on all platforms
- Built-in Python support
- Can open files from command line with `code filename.py`

**Other Options:**
- **Sublime Text**: Fast and lightweight
- **PyCharm**: Full-featured Python IDE
- **nano**: Simple command-line editor for quick fixes

## Why VS Code?

![IDE Choice Guidance](media/IDE_choice.png)

VS Code strikes the perfect balance between simplicity and power. It's what most professional data scientists use, and it's what we'll use in this course.

# Why Both Python and Command Line?

You might wonder: "Why do I need to learn two things at once?" Here's the reality - professional data scientists constantly switch between Python scripts and command line operations. You'll write Python to analyze data, then use command line to organize files, run scripts, and manage projects.

It's like being bilingual in the data world. Python speaks to your data, command line speaks to your computer.

![xkcd 353: Python](media/xkcd_353.png)

# Command Line Essentials

## What is the Command Line?

The command line is your direct conversation with your computer. Instead of clicking icons, you type commands. Why? Because it's faster, more precise, and works exactly the same way on every computer.

Think of it as texting your computer instead of playing charades with icons.

![Unix System Reference](media/its-a-unix-system.jpeg)

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

LIVE DEMO!
*Navigation practice: We'll walk through your file system together*

# File and Directory Operations

**Reference:**
- `mkdir [name]` - Make directory
- `mkdir -p [path/to/nested]` - Make nested directories
- `touch [filename]` - Create empty file
- `cp [source] [destination]` - Copy file
- `mv [source] [destination]` - Move/rename file
- `rm [filename]` - Remove file (careful!)
- `rm -r [directory]` - Remove directory and contents (very careful!)

**Extended Examples for Data Science Workflows:**
```bash
# Create a typical data science project structure
mkdir data-science-project
cd data-science-project
mkdir data scripts results docs

# Create placeholder files for our project
touch data/raw_data.csv
touch scripts/analysis.py
touch docs/project_notes.md

# View our project structure
ls -la
# You'll see: data/ scripts/ results/ docs/ and our files

# Copy important files to backup location
cp data/raw_data.csv data/raw_data_backup.csv

# Rename a file to be more descriptive
mv scripts/analysis.py scripts/customer_analysis.py
```

**Common File Operation Patterns:**
```bash
# Pattern 1: Organizing downloaded data files
mkdir -p project/data/{raw,processed,cleaned}
mv ~/Downloads/*.csv project/data/raw/

# Pattern 2: Creating dated backup directories
mkdir backups/$(date +%Y-%m-%d)
cp -r project/ backups/$(date +%Y-%m-%d)/

# Pattern 3: Finding and organizing files by type
mkdir analysis/{python,jupyter,results}
find . -name "*.py" -exec cp {} analysis/python/ \;
```

**Brief Example:**
```bash
mkdir my_data_project     # Create project folder
cd my_data_project       # Enter the folder
touch analysis.py        # Create Python file
mkdir data              # Create data subfolder
```

# Viewing Files

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

# Getting Help

**Reference:**
- `man [command]` - Manual page for command
- `[command] --help` - Quick help for command
- `which [command]` - Find where command is located

Your best friends when you're stuck!

## Make it Stop!

Sometimes you need to stop what you're doing:

**Command Line:**
- `Control-c` - Cancel the current command
- `exit` - Close the terminal

**Python:**
- `Control-c` - Cancel the current operation
- `exit()` - Quit interactive Python

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

**Interactive Mode Example:**
```python
$ python
>>> print("Hello, World!")
Hello, World!
>>> exit()
```

**Script Mode Example:**
```bash
$ python my_script.py
```

## Python Syntax Overview

Python has some unique syntax rules that are essential to understand:

**Indentation Matters!**
Python uses indentation (spaces or tabs) to group code together. This is different from most programming languages:

```python
# Correct indentation
if x > 0:
    print("Positive")    # This line is indented
    print("Still positive")  # This line is also indented

# Wrong indentation (will cause an error)
if x > 0:
print("This will cause an IndentationError")
```

**Comments Use `#`**
```python
# This is a comment - Python ignores this line
print("This is code")  # Comments can also go at the end of lines
```

**Key Syntax Rules:**
- Use 4 spaces for indentation (not tabs)
- No semicolons needed at the end of lines
- Case-sensitive: `Name` and `name` are different variables
- Use quotes for strings: `"Hello"` or `'Hello'`

## Variables and Data Types

Python stores information in variables - think of them as labeled boxes that you can put different types of information in.

**Core Data Types in Detail:**

# Numbers - The Foundation of Data Science
```python
# Integers (whole numbers)
student_count = 150
year = 2024
temperature_celsius = -5

# Floats (decimal numbers)
average_grade = 87.3
height_meters = 1.75
pi_approximation = 3.14159

# Scientific notation for very large/small numbers
population = 1.4e9          # 1.4 billion
atom_mass = 1.67e-27        # Very small number
```

# Text - Essential for Data Labels and Categories
```python
# Strings for text data
student_name = "Alice Johnson"
department = "Data Science"
file_path = "/Users/alice/projects/analysis.py"

# String methods you'll use constantly
name_upper = student_name.upper()        # "ALICE JOHNSON"
name_lower = student_name.lower()        # "alice johnson"
name_title = student_name.title()        # "Alice Johnson"
clean_name = "  Bob Smith  ".strip()     # Removes whitespace: "Bob Smith"
```

# Boolean - Essential for Data Filtering
```python
# True/False values for logical operations
has_complete_data = True
missing_values = False
analysis_ready = True and has_complete_data    # True
needs_cleaning = missing_values or not analysis_ready  # False
```

**Variable Naming Best Practices:**
```python
# Good variable names (descriptive and clear)
student_age = 22
average_test_score = 85.7
data_file_path = "student_grades.csv"

# Avoid these (unclear or confusing)
a = 22                  # What does 'a' represent?
x1 = 85.7              # Meaningless variable name
temp = "grades.csv"     # 'temp' usually means temporary
```

**Understanding Variable Types (Debugging Foundation):**
```python
# Check what type a variable is (essential for debugging!)
student_name = "Alice"
student_age = 22
grade_average = 87.5

print(type(student_name))    # <class 'str'>
print(type(student_age))     # <class 'int'>
print(type(grade_average))   # <class 'float'>

# This is crucial when data doesn't behave as expected!
mysterious_data = "22"       # Looks like a number, but it's text
print(type(mysterious_data)) # <class 'str'> - Aha! That's the problem
```

## Basic Operations

**Reference:**
```python
# Math operations
result = 10 + 5         # Addition: 15
result = 10 - 3         # Subtraction: 7  
result = 4 * 6          # Multiplication: 24
result = 15 / 4         # Division: 3.75
result = 15 // 4        # Integer division: 3
result = 15 % 4         # Remainder: 3
result = 2 ** 3         # Power: 8

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

## Control Structures

Control structures let your programs make decisions and repeat actions - essential for data analysis!

### Comparison Operators

**Reference:**
```python
# Equality and inequality
x == y          # Equal to
x != y          # Not equal to
x < y           # Less than
x > y           # Greater than
x <= y          # Less than or equal
x >= y          # Greater than or equal

# Membership testing
x in [1, 2, 3]  # Is x in the list?
x not in [1, 2, 3]  # Is x NOT in the list?
```

### If Statements

**Basic If Statements:**
```python
# Simple decision making
score = 85

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
else:
    print("Grade: F")
```

**Compound Conditions:**
```python
# Multiple conditions with and/or
age = 25
has_license = True

if age >= 18 and has_license:
    print("Can drive")
elif age >= 16 and not has_license:
    print("Can learn to drive")
else:
    print("Cannot drive")
```

### For Loops

**Basic For Loops:**
```python
# Count from 0 to 4
for i in range(5):
    print(f"Count: {i}")

# Loop through a list
grades = [85, 92, 78, 96, 88]
for grade in grades:
    print(f"Grade: {grade}")
```

**Practical Data Science Example:**
```python
# Calculate average grade
grades = [85, 92, 78, 96, 88]
total = 0
count = 0

for grade in grades:
    total += grade
    count += 1

average = total / count
print(f"Average grade: {average:.1f}")
```

LIVE DEMO!
*We'll create our first data science calculation together*

## Printing and Basic Input

**Essential Output Formatting for Data Science:**

```python
# Basic printing - your daily communication tool
print("Hello world")                    # Basic printing
print("Value:", 42)                     # Multiple values
print("Processing complete!")           # Status updates

# F-string formatting - the data scientist's best friend
student_name = "Alice"
test_score = 87.3
class_average = 82.1

print(f"Student: {student_name}")                    # Basic variable insertion
print(f"Score: {test_score}")                        # Number display
print(f"Score: {test_score:.1f}")                    # One decimal place: 87.3
print(f"Score: {test_score:.0f}%")                   # No decimals: 87%
print(f"Above average by {test_score - class_average:.1f} points")  # Calculations inside f-strings
```

**Advanced F-String Patterns for Data Analysis:**
```python
# Currency formatting (useful for business data)
revenue = 15432.50
print(f"Revenue: ${revenue:,.2f}")                   # $15,432.50

# Percentage formatting
success_rate = 0.847
print(f"Success rate: {success_rate:.1%}")           # 84.7%

# Scientific notation for very large/small numbers
population = 1400000000
print(f"Population: {population:.2e}")               # 1.40e+09

# Padding and alignment for clean output tables
print(f"{'Name':<15} {'Score':>8} {'Grade':>8}")    # Column headers
print(f"{'Alice':<15} {87.3:>8.1f} {'B+':>8}")      # Left/right aligned data
print(f"{'Bob':<15} {92.1:>8.1f} {'A-':>8}")

# Date formatting (preview for later lectures)
from datetime import datetime
today = datetime.now()
print(f"Analysis run on: {today:%Y-%m-%d %H:%M}")    # 2024-01-15 14:30
```

**Basic Input (Rare in Data Science, but Good to Know):**
```python
# Interactive input - mainly for testing and debugging
name = input("Enter your name: ")                    # Gets text from user
age_str = input("Enter your age: ")                  # Always returns string!
age = int(age_str)                                   # Convert to number
print(f"Hello {name}, you are {age} years old")

# Be careful: input() always returns strings
user_number = input("Enter a number: ")              # This is text: "42"
print(type(user_number))                             # <class 'str'>
actual_number = float(user_number)                   # Convert to number: 42.0
print(type(actual_number))                           # <class 'float'>
```

**Why F-Strings Matter in Data Science:**
F-strings let you create clear, readable output that tells the story of your data. Instead of printing raw numbers, you can provide context, explanations, and professional formatting that makes your analysis understandable to anyone.

# Debugging and Error Handling Basics

**Reading Python Error Messages (Essential Skill!):**

When Python encounters a problem, it tells you exactly what went wrong. Learning to read these messages will save you hours of frustration.

```python
# Common error: trying to use an undefined variable
print(student_naem)  # Typo in variable name
```
```
NameError: name 'student_naem' is not defined
```

**How to Read This Error:**
1. **Error Type**: `NameError` - Python doesn't recognize the variable name
2. **Error Message**: tells you exactly what's wrong
3. **Your Action**: Check spelling, make sure you defined the variable first

**More Common Errors You'll Encounter:**

```python
# Type errors - mixing incompatible data types
age = "25"                    # This is text, not a number
next_year = age + 1          # Can't add number to text
```
```
TypeError: can only concatenate str (not "int") to str
```

**How to Fix It:**
```python
age = "25"                    # Text
age_number = int(age)         # Convert to number
next_year = age_number + 1    # Now this works!
print(f"Next year you'll be {next_year}")
```

**Value Errors - Wrong Type of Value:**
```python
bad_number = int("hello")     # Can't convert "hello" to a number
```
```
ValueError: invalid literal for int() with base 10: 'hello'
```

**Debugging Strategy for Beginners:**
1. **Read the error message carefully** - Python is usually very specific
2. **Check variable names for typos** - most common beginner mistake
3. **Use `print()` to check variable values and types**
4. **Check your data types** with `type(variable_name)`

**Defensive Programming Example:**
```python
# Always check what type your data is when debugging
user_input = "42"
print(f"Input: {user_input}")
print(f"Type: {type(user_input)}")       # Shows: <class 'str'>

# Convert and verify
number = int(user_input)
print(f"Converted: {number}")
print(f"New type: {type(number)}")       # Shows: <class 'int'>

# Now you can safely do math
result = number * 2
print(f"Result: {result}")
```

**Error Prevention Tips:**
- **Use descriptive variable names** - reduces typos
- **Check types when debugging** - use `type()` function
- **Test with small examples first** - don't write 50 lines then run
- **One step at a time** - add complexity gradually

# Simple Workflow Example

Let's see how command line and Python work together in a basic data science workflow:

**Basic Data Calculation Workflow:**
```bash
# Command line: Set up workspace
mkdir data_analysis
cd data_analysis
touch calculate_stats.py
```

```python
# Python: calculate_stats.py
# Simple statistical analysis
sales_data = [1200, 1500, 1800, 1100, 1650, 1750]
total_sales = sum(sales_data)
average_sales = total_sales / len(sales_data)
best_day = max(sales_data)

print(f"Weekly Sales Analysis")
print(f"Total sales: ${total_sales:,}")
print(f"Average daily sales: ${average_sales:.2f}")
print(f"Best day: ${best_day}")
```

```bash
# Command line: Run the analysis
python calculate_stats.py
```

**Output:**
```
Weekly Sales Analysis
Total sales: $9,000
Average daily sales: $1,500.00
Best day: $1750
```

**Key Workflow Principles:**
1. **Start small** - test logic with simple data first
2. **Build incrementally** - add complexity step by step
3. **Test frequently** - run your code after every few changes
4. **Save your work** - use meaningful file names and organize results
5. **Document as you go** - use print statements to explain what's happening

# Key Takeaways

By the end of today's session, you should feel comfortable with:

1. **Command line navigation** = Your direct conversation with the computer
   - Navigate directories with `cd`, `pwd`, `ls`
   - Create project structures with `mkdir`, `touch`
   - View files with `head`, `tail`, `cat`

2. **Python fundamentals** = The foundation of data analysis
   - Variables store different types of data (numbers, text, booleans)
   - F-strings create professional, readable output
   - Error messages are your friends - they tell you exactly what's wrong

3. **Integration workflow** = The data scientist's daily routine
   - Use command line to organize files and run scripts
   - Use Python to process data and generate insights
   - Test small, build incrementally, save your work

4. **Debugging mindset** = Essential problem-solving skills
   - Read error messages carefully
   - Check variable types when things go wrong
   - Use `print()` statements to understand what's happening

**Why This Matters:**
These might seem like simple tools, but they're the foundation everything else builds on. Every advanced data science technique - machine learning, statistical analysis, data visualization - starts with these basic skills.

**Professional Reality Check:**
Real data scientists spend 80% of their time doing exactly these things: organizing files, reading data, cleaning it up, and generating clear reports. The fancy algorithms are just 20% of the work!

Next week: We'll learn how to save and share our work with Git and GitHub!

# Practical Exercise

Now let's put everything together with a hands-on exercise that combines command line skills and Python programming.

## Exercise 1: Setup and Hello World

1. **Get everything installed** (or use the cloud) and play around a bit
2. **Create an account on GitHub**
   - Apply for [GitHub Education](https://docs.github.com/en/education/explore-the-benefits-of-teaching-and-learning-with-github-education/github-education-for-students/apply-to-github-education-as-a-student)
   - [Join the course and see next week's assignment](https://classroom.github.com/a/Z2sWwnXF)

3. **Create a Python script that prints "Hello, Data Science!"**
   - Save it as `hello_ds.py`
   - Run it from the command line
   - Use command line to create a `scripts` folder and move your file into it

## Exercise 2: Mathematical Problem Solving

Write a Python script to solve the following problem:

> If we list all the natural numbers below 10 that are multiples of 3 or 5, we get (3, 5, 6, 9). The sum of these multiples is 23. Find the sum of all the multiples of 3 or 5 below 1000.

**Hint:** Use a for loop and if statements to check each number!

## Exercise 3: Email Submission

Email me:
1. **GitHub username**
2. **Answer to the problem above + script you wrote to solve it**
3. **Brief introduction** (who are you, why are you here, anything you're specifically hoping to get out of the course)

# Comprehensive Practice Challenge

**Challenge Level 1: Basic Skills (Everyone Should Complete)**
1. Create a project structure using command line:
   ```bash
   mkdir my_data_practice
   cd my_data_practice
   mkdir data scripts results
   touch scripts/my_analysis.py
   ```

2. Write a Python script that analyzes something you care about:
   - Personal budget calculation
   - Grade point average
   - Favorite sports team statistics
   - Recipe ingredient costs

3. Run your script and save the output:
   ```bash
   python scripts/my_analysis.py
   python scripts/my_analysis.py > results/my_results.txt
   ```

4. Use command line to examine your work:
   ```bash
   head scripts/my_analysis.py    # Look at your code
   cat results/my_results.txt     # Look at your results
   ```

**Challenge Level 2: Intermediate Practice (For Extra Credit)**
Create a multi-step analysis that:
1. Creates sample data in a text file
2. Reads that data with Python
3. Calculates multiple statistics
4. Generates a formatted report
5. Saves results to multiple files

**Challenge Level 3: Advanced Exploration (For the Ambitious)**
Research and try:
- Using different f-string formatting options for currency, percentages
- Handling errors with try/except blocks
- Creating data visualizations with matplotlib (we'll cover this formally later)
- Building a simple data dashboard

**Debugging Practice:**
Intentionally create these errors and practice fixing them:
1. NameError (misspell a variable name)
2. TypeError (try to add a string and number)
3. ValueError (convert "hello" to an integer)

For each error:
- Read the error message
- Identify what's wrong
- Fix the problem
- Verify it works

**Self-Assessment Questions:**
- Can you create a directory structure from command line?
- Can you write a Python script that calculates averages and percentages?
- Can you read and fix basic Python error messages?
- Can you run your scripts and view the results using command line?

If you answered "yes" to all four, you're ready for next week!

**Getting Help:**
- Stuck on command line? Try `man [command]` or `[command] --help`
- Stuck on Python? Try the error message first, then search for the error type
- Still stuck? That's normal! Bring your questions to next class

Remember: Every data scientist started exactly where you are now. The difference between a beginner and an expert isn't talent - it's practice and persistence!

![xkcd 1513: Code Quality](media/xkcd_1513.png)

**What's Next:**
Next week we'll build on these foundations by learning Git and GitHub - the tools that let data scientists collaborate, save versions of their work, and share their findings with the world. Everything you've learned today will make that much easier!

# Additional Resources

## Command Line Resources
- [LinuxCommand.org](http://linuxcommand.org/lc3_learning_the_shell.php) - Excellent beginner tutorial
- [The Linux Command Line book](http://linuxcommand.org/tlcl.php) - Free online book
- [The Missing Semester](https://missing.csail.mit.edu/) - MIT's course on essential tools

## Python Resources
- [Official Python documentation](https://docs.python.org/3/) - The definitive reference
- [Whirlwind Tour of Python](https://jakevdp.github.io/WhirlwindTourOfPython/) - Quick overview by Jake VanderPlas
- [Think Python](https://greenteapress.com/wp/think-python/) - Free book by Allen Downey
- [Hitchhiker's Guide to Python!](https://docs.python-guide.org/) - Best practices and conventions
- [Codecademy Python course](https://www.codecademy.com/learn/learn-python-3) - Interactive learning

## Getting Help
- **RTFM**: `man COMMAND` - Read the manual for any command
- **Ask the computer**: `COMMAND --help` - Quick help for any command
- **Ask a bigger computer**: Claude, ChatGPT, or Stack Overflow
- **Come to lab!** - We're here to help

## Wrap-up

- We've covered Python basics and essential command line operations
- Assignment: Practice these concepts with provided exercises
- Next lecture: Version control with Git, shell scripting and more Python