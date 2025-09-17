# Live Demo!

# Demo 1: GitHub & VS Code Setup
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

# Demo 2: Command Line Navigation
**Script:** `02_cli_navigation_demo.sh`

**Key Steps:**
1. Connect to previous demo
   - Real command line operations
   - Essential CLI skills for data science

2. Navigation and Project Structure
   ```bash
   ./02_cli_navigation_demo.sh
   ```
   - Organized folder structure as industry standard
   - pwd, ls, mkdir, cd commands

3. File Operations
   - Creating files with cat and heredoc
   - Real CSV data and Python script creation

4. Path Problems
   - Path error occurs naturally
   - Most common error in data science
   - Relative vs absolute paths concept

5. File Viewing and Navigation
   - cat, head, tail commands
   - cd navigation practice
   - Directory cleanup options

**Discussion Points:**
- Why the script failed initially
- Solutions for fixing relative path issues

# Demo 3: Python Basics with Debugging
**Script:** `03_python_basics_demo.py`

**Key Steps:**
1. Set expectations
   - Intentional mistakes in demo
   - Learning exercise to spot and understand fixes

2. Variables and Types
   ```bash
   python3 03_python_basics_demo.py
   ```
   - Typo error happens: "student_naem"
   - Error message hints

3. F-strings
   - Missing 'f' mistake
   - Common error pattern
   - Professional formatting

4. Indentation and Lists
   - Python's unique indentation-based syntax
   - IndexError: lists start at 0, not 1

5. Real-world Debugging
   - Actual data with actual problems
   - Step-by-step debugging approach
   - Data validation importance

**Interactive Elements:**
- Have students identify errors as they occur
- Discuss data types and their importance

# Demo 4: Control Structures
**Script:** `04_control_structures_demo.py`

**Key Steps:**
1. Comparisons and If Statements
   ```bash
   python3 04_control_structures_demo.py
   ```
   - = assigns, == compares
   - Order-matters error in conditions

2. For Loops
   - 0-based vs 1-based indexing confusion
   - enumerate() as Python's elegant solution

3. While Loops
   - Commented-out infinite loop
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

# Demo 5: Complete Integration
**Script:** `05_integration_workflow_demo.py`

**Key Steps:**
1. Set the scene
   - Everything comes together
   - Day-in-the-life of a data scientist

2. Project Setup and Data Creation
   ```bash
   python3 05_integration_workflow_demo.py
   ```
   - Professional project structure
   - Intentional bad data as realistic scenario

3. Data Loading with Validation
   - Validation catching problems
   - Time savings in debugging

4. Analysis
   - Simple but properly organized statistics
   - Complete analysis for each student

5. Saving Results
   - Multiple output formats for different audiences
   - Logging for reproducibility in research

6. Organization and Automation
   - Final directory tree
   - Portfolio-worthy organization
   - Automation as goal: write once, run many times

**Discussion Points:**
- Potential additions to the analysis
- Scaling considerations for larger datasets

# Post-Demo Wrap-up

# Key Messages to Reinforce:
1. **Errors are teachers** - "Every error message is a learning opportunity"
2. **Organization matters** - "Start clean, stay clean"
3. **Test small, build big** - "Always verify each piece works"
4. **Debugging is detective work** - "Use print(), check types, read errors"
5. **This is real** - "You've seen actual data science workflow"

# Call to Action:
- "Run these demos yourself - break them, fix them, modify them"
- "The homework builds directly on these concepts"
- "Office hours if you get stuck - bring your errors!"
