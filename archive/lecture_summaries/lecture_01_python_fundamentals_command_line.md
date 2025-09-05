# Lecture 1: Python Fundamentals + Essential Command Line

## Learning Objectives
By the end of this lecture, students will be able to:
- Navigate the command line using essential commands for file operations and text manipulation
- Run Python scripts from the command line and understand the Python execution environment
- Use basic Python syntax for variables, data types, control structures, and functions
- Create simple Python scripts for data science applications
- Set up their development environment with command line tools and Python

## Content Consolidation Details

### Primary Sources (Current Lectures)
- **Lecture 01 (90%)**: Python basics, command line navigation, file operations, Python installation
- **Lecture 02 (25%)**: Environment setup, Python package management basics
- **Lecture 03 (30%)**: Advanced command line operations, shell scripting fundamentals

### Secondary Integration
- **Lecture 04 (15%)**: File operations in Python, basic system interaction

## Specific Topics Covered

### Command Line Essentials (45 minutes)
1. **Getting to the Command Line**
   - Terminal access on different platforms (Windows PowerShell/WSL, Mac Terminal, Linux)
   - Understanding shells (bash, zsh, PowerShell)
   - GitHub Codespaces as a cloud option

2. **Navigation and File System**
   - `pwd`, `ls`/`dir`, `cd` commands
   - Understanding paths (relative vs absolute)
   - Special directories: `~`, `.`, `..`

3. **File Operations**
   - Creating: `mkdir`, `touch`/`New-Item`
   - Copying and moving: `cp`, `mv`
   - Viewing content: `cat`, `head`, `tail`
   - Deleting: `rm` (with safety considerations)

4. **Text Manipulation Basics**
   - Pattern searching: `grep`/`Select-String`
   - Basic pipes: `|`
   - Output redirection: `>`, `>>`

### Python Fundamentals (60 minutes)
1. **Installation and Setup**
   - Installing Python on different platforms
   - Verifying installation with `python --version`
   - Running Python interactively vs scripts

2. **Core Syntax and Data Types**
   - Python syntax rules (indentation, comments)
   - Variables and dynamic typing
   - Basic data types: integers, floats, strings
   - String operations and f-string formatting

3. **Control Structures**
   - Conditional statements: `if`, `elif`, `else`
   - Comparison operators: `==`, `!=`, `<`, `>`, `<=`, `>=`
   - Logical operators: `and`, `or`, `not`
   - For loops with `range()`
   - While loops (brief introduction)

4. **Functions and Code Organization**
   - Defining functions with `def`
   - Parameters and return values
   - Basic function documentation
   - Importing and using modules

### Integration and Practical Application (15 minutes)
1. **Running Python from Command Line**
   - Executing Python scripts: `python script.py`
   - Command line arguments basics
   - Understanding error messages

2. **Development Workflow**
   - Text editor options (focus on VS Code)
   - Creating and organizing project files
   - Basic debugging with print statements

## Content to Trim (25% reduction from source lectures)

### From Lecture 01
- **Remove (10 minutes)**: Detailed PowerShell equivalents - focus on Unix-style commands
- **Reduce (5 minutes)**: Multiple shell options discussion - emphasize bash/zsh
- **Simplify (5 minutes)**: Installation methods - focus on primary methods per platform

### From Lecture 02  
- **Remove (8 minutes)**: Git installation and configuration details (move to Lecture 2)
- **Remove (7 minutes)**: Virtual environments setup (move to Lecture 2)

### From Lecture 03
- **Remove (15 minutes)**: Advanced shell scripting, cron jobs, environment variables (move to Lecture 5)
- **Remove (10 minutes)**: File compression/decompression (optional supplementary material)

## Practical Exercises and Hands-on Components

### Command Line Practice (20 minutes)
1. **Navigation Challenge**
   - Create directory structure: `projects/datasci_217/week1`
   - Navigate using relative and absolute paths
   - List files with different options

2. **File Operations Workshop**
   - Create sample data files
   - Practice copying, moving, and organizing files
   - Use basic text viewing commands

### Python Programming Lab (25 minutes)
1. **Hello Data Science Script**
   - Create and run `hello_ds.py`
   - Practice file creation and execution
   - Debug common syntax errors

2. **Problem-Solving Exercise**
   - Project Euler Problem 1: Sum of multiples of 3 or 5 below 1000
   - Implement using loops and conditionals
   - Practice function creation and testing

3. **Integration Challenge**
   - Create script that reads command line arguments
   - Process data and write results to file
   - Use command line to run and verify results

## Prerequisites and Dependencies

### Assumed Knowledge
- Basic computer literacy (file systems, text editing)
- Willingness to work with text-based interfaces
- No prior programming experience required

### Technical Requirements
- Computer with terminal access (Windows/Mac/Linux)
- Python 3.8+ installation
- Text editor (VS Code recommended)
- GitHub account for future assignments

### Preparation Materials
- Pre-lecture setup guide for Python installation
- Command line cheat sheet for reference
- Access to practice environment (local or cloud)

## Assessment Components

### Formative Assessment (During Class)
- Interactive command line exercises
- Live coding demonstrations
- Peer problem-solving activities
- Q&A sessions for immediate feedback

### Summative Assessment (Assignment)
1. **Command Line Competency**
   - Create specific directory structure
   - Perform file operations sequence
   - Document process with screenshots

2. **Python Programming Project**
   - Solve mathematical problem with Python
   - Create well-documented script
   - Demonstrate proper execution from command line

3. **Integration Task**
   - Combine command line and Python skills
   - Create automation script for file processing
   - Submit via GitHub (introduction to version control)

## Key Success Metrics
- Students can confidently navigate file system via command line
- Students can write and execute basic Python programs
- Students understand the relationship between command line and Python development
- Students demonstrate problem-solving approach using programming concepts
- 90% of students successfully complete and submit the practical assignment

## Resources and References

### Essential Resources
- [The Linux Command Line](http://linuxcommand.org/tlcl.php) - Chapters 1-5
- [Python.org Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)

### Supplementary Materials
- Interactive Python tutorials (CodeAcademy, Python.org)
- Command line practice environments
- Troubleshooting guides for different operating systems
- Video demonstrations for complex concepts

### Help and Support
- Lab sessions for hands-on assistance
- Course forum for peer and instructor support
- Office hours for individualized help
- Recommended study groups for collaborative learning