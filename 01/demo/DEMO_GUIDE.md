# Live Demo!

# Demo 1: Setup and command-line navigation
**Guide:** `01_github_vscode_setup_guide.md` (Live walkthrough, no script)

**Key Steps:**
1. GitHub account creation
   - Professional identity in tech
   - Username best practices
   - Employers look at GitHub profiles

2. Email privacy and GitHub Education
   - Noreply email setup
   - Email privacy in commits
   - GitHub Student Pack: https://education.github.com/students
   - Free tools and resources available

3. VS Code setup
   - Coding environment for students
   - Python extension installation
   - Widespread industry adoption

4. Git configuration in VS Code
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "noreply@users..."
   ```
   - Create and run hello.py
   - Integration between tools

**Check Understanding:**
- Verify students can see their noreply email
- Confirm hello.py runs successfully

## Command-line navigation
**Script:** `02_cli_navigation_demo.sh`

**Key Steps:**
1. Basic navigation commands
   - pwd (where am I?)
   - ls (what's here?)
   - mkdir (create directories)
   - cd (change directories)

2. Quick project creation
   ```bash
   ./02_cli_navigation_demo.sh
   ```
   - Simple directory structure
   - File creation with echo redirection

3. File viewing
   - cat command to display file contents
   - Basic CSV data example

4. Path problem demonstration
   - The first run captures the expected wrong-path error
   - The script rewrites the path and reruns successfully
   - Teaches relative path concept

**Discussion Points:**
- Why the Python script couldn't find the file
- How file paths work from different directories

# Demo 2: Python basics, debugging, and control structures
**Script:** `03_python_basics_demo.py`

**Key Steps:**
1. Set expectations
   - Intentional mistakes in demo
   - Learning exercise to spot and understand fixes

2. Variables and Types
   ```bash
   python3 03_python_basics_demo.py
   ```
   - Typo error shown: "student_naem" (commented out with fix)
   - Error message explanations and debugging tips

3. F-strings
   - Missing 'f' mistake (commented out with fix)
   - Common error pattern
   - Professional formatting

4. Indentation and Lists
   - Python's unique indentation-based syntax
   - IndexError: lists start at 0, not 1 (commented out with fix)

5. Basic Operations and Error Handling
   - Math operations and string handling
   - Division by zero and off-by-one errors (commented out with fixes)
   - Step-by-step debugging approach
   - Data type checking importance

**Interactive Elements:**
- Have students identify errors as they occur (now shown as commented-out code)
- Discuss data types and their importance
- Practice reading error messages and understanding fixes

## Control structures
**Script:** `04_control_structures_demo.py`

**Key Steps:**
1. Comparisons and If Statements
   ```bash
   python3 04_control_structures_demo.py
   ```
   - = assigns, == compares (commented out with fix)
   - Order-matters error in conditions (commented out with fix)

2. For Loops
   - 0-based vs 1-based indexing confusion (commented out with fix)
   - enumerate() as Python's elegant solution

3. While Loops
   - Commented-out infinite loop (commented out with fix)
   - Ensuring loops can end

4. Nested Loops and Comprehensions
   - List comprehensions as Pythonic approach
   - Data science application

5. Practical Example
   - Real grading system
   - Complete logic walkthrough
   - Edge case handling for empty lists

**Interactive Elements:**
- Have students predict output before execution
- Compare different coding approaches
- Practice identifying logic errors in commented-out code

# Demo 3: Complete integration
**Script:** `05_integration_workflow_demo.py`

**Key Steps:**
1. Set the scene
   - Everything comes together
   - Day-in-the-life of a data scientist

2. Project Setup and Data Creation
   ```bash
   python3 05_integration_workflow_demo.py
   ```
   - A small list-based fixture keeps attention on the workflow
   - The command line supplies the project location; Python analyzes the values
   - The script uses only Lecture 01 variables, lists, loops, conditionals, and arithmetic

3. Analysis
   - Print each student's status using a loop and a threshold
   - Calculate class average, minimum, maximum, and passing count

4. Saving Results
   - Redirect the script's printed report to `results.txt`
   - Re-run the same script whenever the input list changes

**Discussion Points:**
- Potential additions to the analysis
- Scaling considerations for larger datasets

# Take-away Message:

1. **Errors are teachers** - "Every error message is a learning opportunity"
2. **Organization matters** - "Start clean, stay clean"
3. **Test small, build big** - "Always verify each piece works"
4. **Debugging is detective work** - "Use print(), check types, read errors"
5. **This is real** - "You've seen actual data science workflow"
