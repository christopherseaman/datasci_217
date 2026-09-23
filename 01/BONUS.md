---
notion:
  title_line: "# DLC: Python, the Command Line, and VS Code"
  role: bonus
  status: mapped
  page_id: "3d6d9fdd-1a1a-81d0-97ab-c3098e3d1397"
  url: "https://app.notion.com/p/3d6d9fdd1a1a81d097abc3098e3d1397"
---

# DLC: Python, the Command Line, and VS Code

These topics are **optional** and not required for future lectures. Explore them if you're curious or want to deepen your understanding!

# Python on Native Windows PowerShell

The course demos use WSL, but uv can install Python for PowerShell itself:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new PowerShell terminal, then run:

```powershell
uv python install 3.13 --default
uv python update-shell
```

Reopen the terminal and check `python --version`. Use `python` in place of `python3` in this environment; the shell commands in Lecture 01 still need WSL.

# Other Editors

- **Sublime Text**: fast and lightweight
- **PyCharm**: full-featured Python IDE
- **nano**: simple command-line editor for quick fixes
- **Vim / Neovim**: modal terminal editors; open with `vim filename.py` or `nvim filename.py`. Press `i` to insert text, then `Esc` and `:wq` to save and quit (the answer to "how do I get out of Vim?").

# Advanced ls Options

Beyond the basic `ls`, there are many useful variations:

```bash
ls -la          # Long format with hidden files
ls -lh          # Human-readable file sizes
ls -lt          # Sort by modification time
ls -lr          # Reverse order
ls *.py         # List only Python files
ls -R           # Recursive (show subdirectories too)
```

# Python REPL Advanced Features

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

# Command History and Shortcuts

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
Ctrl+U             # Delete from cursor back to start of line
Tab                # Auto-complete (your best friend!)
```

# Python help() and dir() Functions

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

# File Permissions Basics

Understanding what you can and can't do with files:

```bash
ls -l              # Shows permissions (rwxrwxrwx format)
chmod +x script.py # Make file executable
chmod 644 file.txt # Set specific permissions

# Permission format: owner-group-others
# r = read (4), w = write (2), x = execute (1)
# 644 = owner can read/write, others can read
```

# Environment Variables Preview

Your computer stores settings in environment variables:

```bash
echo $HOME         # Your home directory
echo $PATH         # Where computer looks for commands
env                # Show all environment variables
```

```python
# In Python, you can access these too:
import os
print(os.environ['HOME'])    # Access environment variable
```

# Advanced Python String Operations

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

# Command Line Wildcards and Patterns

Advanced file selection:

```bash
ls *.txt           # All .txt files
ls data_*          # Files starting with "data_"
ls ??.py           # Python files with 2-character names
ls [abc]*          # Files starting with a, b, or c
ls *[0-9].csv      # CSV files ending with a number
```

# Python Number Operations

More mathematical operations:

```python
import math

# Advanced math
math.sqrt(16)      # Square root: 4.0
math.ceil(4.3)     # Round up: 5
math.floor(4.7)    # Round down: 4
math.pi            # 3.141592653589793

# Scientific notation
large_number = 1.23e6    # 1,230,000
small_number = 1.23e-3   # 0.00123
```

