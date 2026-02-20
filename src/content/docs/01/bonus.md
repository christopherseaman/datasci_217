---
title: "Bonus: Advanced Topics"
---


These topics are **optional** and not required for future lectures. Explore them if you're curious or want to deepen your understanding!

## Advanced ls Options

Beyond the basic `ls`, there are many useful variations:

```bash
ls -la          # Long format with hidden files
ls -lh          # Human-readable file sizes  
ls -lt          # Sort by modification time
ls -lr          # Reverse order
ls *.py         # List only Python files
ls -R           # Recursive (show subdirectories too)
```

## Python REPL Advanced Features

The interactive Python environment has helpful features:

```python
# Special variables
_                   # Result of last expression
__name__           # Current module name

# Useful functions for exploration
dir()              # List available names
help(print)        # Get help on function
type(variable)     # Check variable type
len(text)          # Length of strings/lists
```

## Command History and Shortcuts

Make your command line experience smoother:

```bash
# History navigation
↑/↓ arrows         # Previous/next command
history            # Show recent commands
!123               # Run command #123 from history
!!                 # Run last command again

# Editing shortcuts
Ctrl+A             # Beginning of line
Ctrl+E             # End of line  
Ctrl+U             # Clear entire line
Tab                # Auto-complete (your best friend!)
```

## Python help() and dir() Functions

These built-in functions are incredibly useful for learning:

```python
# Explore what's available
help(str)          # Help on string methods
dir(str)           # List all string methods
help(str.split)    # Help on specific method

# Interactive help
help()             # Enter help mode
# Type "quit" to exit help mode

# Quick information
print.__doc__      # Function documentation
str.upper.__doc__   # Method documentation
```

## File Permissions Basics

Understanding what you can and can't do with files:

```bash
ls -l              # Shows permissions (rwxrwxrwx format)
chmod +x script.py # Make file executable
chmod 644 file.txt # Set specific permissions

# Permission format: owner-group-others
# r = read (4), w = write (2), x = execute (1)
# 644 = owner can read/write, others can read
```

## Environment Variables Preview

Your computer stores settings in environment variables:

```bash
echo $HOME         # Your home directory
echo $PATH         # Where computer looks for commands
env                # Show all environment variables

# In Python, you can access these too:
import os
print(os.environ['HOME'])    # Access environment variable
```

## Advanced Python String Operations

Strings have many useful methods:

```python
text = "Hello, Data Science!"

# Case operations
text.upper()           # "HELLO, DATA SCIENCE!"
text.lower()           # "hello, data science!"
text.title()           # "Hello, Data Science!"

# Checking content
text.startswith("Hello")  # True
text.endswith("!")        # True
"Data" in text            # True

# Cleaning
text.strip()           # Remove whitespace from ends
text.replace("Hello", "Hi")  # Replace text
```

## Command Line Wildcards and Patterns

Advanced file selection:

```bash
ls *.txt           # All .txt files
ls data_*          # Files starting with "data_"
ls ??.py           # Python files with 2-character names
ls [abc]*          # Files starting with a, b, or c
ls *[0-9].csv      # CSV files ending with a number
```

## Python Number Formats and Operations

More mathematical operations:

```python
import math

# Advanced math
math.sqrt(16)      # Square root: 4.0
math.ceil(4.3)     # Round up: 5
math.floor(4.7)    # Round down: 4
math.pi            # 3.141592653589793

# Number formatting
value = 3.14159
f"{value:.2f}"     # "3.14" (2 decimal places)
f"{value:.0f}"     # "3" (no decimal places)
f"{1234:,}"        # "1,234" (comma separator)

# Scientific notation
large_number = 1.23e6    # 1,230,000
small_number = 1.23e-3   # 0.00123
```

## Why These Topics Are Bonus

These advanced features are powerful, but can be overwhelming when you're just starting. Focus on the core concepts first, then come back to these when:

- You feel comfortable with the basics
- You encounter a specific need for these features  
- You want to optimize your workflow
- You're curious about "how the pros do it"

Remember: Being productive with the basics is better than being confused by the advanced features!

## When You Might Need These

- **Advanced ls**: When working with large directories or complex file structures
- **REPL features**: When exploring new Python libraries or debugging
- **History shortcuts**: When you find yourself retyping the same commands
- **Help functions**: When working with unfamiliar Python functions
- **Permissions**: When setting up scripts or working on shared systems
- **Environment variables**: When configuring development environments
- **Advanced strings**: When processing messy text data
- **Wildcards**: When working with many files with similar names
- **Advanced math**: When doing complex calculations

## Practice Suggestions

If you want to explore these topics:

1. **Start small**: Pick one advanced feature and use it for a week
2. **Practice with real files**: Use your actual project files, not made-up examples
3. **Ask "why?"**: Understand when and why you'd use each feature
4. **Document what works**: Keep notes on useful shortcuts and commands

The key is building confidence with the basics first, then gradually adding more advanced techniques as needed!