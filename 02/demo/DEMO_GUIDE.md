# Live Demo! - Lecture 2

# Demo 1: Git & GitHub Workflow
**Type:** Live walkthrough (no script needed)

**Key Steps:**
1. **Repository Setup (5 minutes)**
   ```bash
   # Create project directory
   mkdir data_analysis_project
   cd data_analysis_project
   
   # Initialize Git repository
   git init
   git status
   
   # Configure Git (one-time setup)
   git config --global user.name "Your Name"
   git config --global user.email "your.email@ucsf.edu"
   ```

2. **VS Code Git Integration (5 minutes)**
   - Create project structure: `mkdir data scripts results`
   - Create files: `touch README.md scripts/analysis.py data/sample_data.csv`
   - Open VS Code: `Ctrl+Shift+G` for Source Control panel
   - Stage files: Click `+` next to each file
   - Commit: "Initial project setup" + `Ctrl+Enter`
   - Make changes and track progress with visual diff

3. **GitHub Collaboration (5 minutes)**
   - Create repository on GitHub (github.com → "+" → "New repository")
   - Connect local to GitHub:
     ```bash
     git remote add origin https://github.com/YOUR_USERNAME/data-analysis-project.git
     git push -u origin main
     ```
   - Demonstrate collaboration features: file history, issues, pull requests

4. **Advanced Git Workflow (5 minutes)**
   - Branch and merge workflow:
     ```bash
     git checkout -b feature-analysis
     # Make changes, commit
     git checkout main
     git merge feature-analysis
     ```
   - Professional commit messages and best practices

**Check Understanding:**
- Verify students can see staged changes in VS Code
- Confirm successful push to GitHub
- Test repository cloning

# Demo 2: Command Line Mastery
**Script:** `02_cli_advanced_demo.sh`

**Key Steps:**
1. **Essential Navigation Commands (2 minutes)**
   ```bash
   # Show current location and navigate
   pwd                                    # Show current directory
   ls -la                                 # List files with details
   cd data/raw                            # Navigate to subdirectory
   pwd                                    # Confirm location
   cd ../..                               # Go up two levels
   cd ~                                   # Go to home directory
   cd -                                   # Return to previous directory
   ```

2. **Project Setup and Navigation (3 minutes)**
   ```bash
   ./02_cli_advanced_demo.sh
   ```
   - Create complex directory structure: `mkdir -p data/{raw,processed} scripts results logs`
   - Generate sample data files (students.csv, courses.csv)
   - Demonstrate file operations with wildcards

3. **Text Processing and Search (4 minutes)**
   - File viewing: `head`, `tail`, `cat`
   - Text search: `grep -i "math" data/raw/*.csv`
   - Data extraction: `cut -d',' -f4 data/raw/students.csv`
   - Line counting: `wc -l data/raw/*.csv`

4. **Advanced Data Processing (4 minutes)**
   - AWK operations: `awk -F',' '$3 > 85 {print $1, $3}' data/raw/students.csv`
   - Average calculations: `awk -F',' 'NR>1 {sum+=$3; count++} END {print "Average:", sum/count}'`
   - File operations with timestamps: `timestamp=$(date +"%Y%m%d_%H%M%S")`

5. **Command Chaining and Automation (4 minutes)**
   - Pipe operations: `grep "error" logfile.txt | wc -l`
   - Output redirection: `ls *.csv | head -5 > filelist.txt`
   - Report generation with multiple commands
   - Error handling and validation

6. **History Navigation and Shortcuts (3 minutes)**
   ```bash
   # Command history navigation
   history                                 # Show command history
   !<number>                              # Execute command by number
   !!                                     # Repeat last command
   
   # Keyboard shortcuts
   # Up arrow - Previous command
   # Down arrow - Next command
   # Ctrl+A - Move to beginning of line
   # Ctrl+E - Move to end of line
   # Ctrl+K - Delete from cursor to end
   # Ctrl+U - Delete from cursor to beginning
   # Tab - Auto-complete commands/files
   # Ctrl+R - Reverse search through history
   # Ctrl+C - Cancel current command
   ```

**Interactive Elements:**
- Have students predict command outputs
- Practice command chaining with real data
- Build script step-by-step together

# Demo 3: Python Functions and Modules
**Scripts:** `03_python_functions_demo.py` and `03_module_usage_demo.py`

**Key Steps:**
1. **Start with Inline Script (4 minutes)**
   ```bash
   python3 03_python_functions_demo.py
   ```
   - Create sample student data
   - Basic data processing with loops
   - Simple calculations and output
   - File I/O operations with CSV

2. **Refactor into Functions (4 minutes)**
   - Extract common operations: `calculate_average()`, `find_highest_grade()`
   - Add function parameters and return values
   - Implement error handling: `safe_calculate_average()`
   - Data validation: `validate_student_data()`

3. **Add Professional Structure (3 minutes)**
   - Add `__main__` execution control
   - Script execution vs. import behavior
   - File I/O operations: `save_results_to_file()`
   - Professional script organization

4. **Create Module and Import (4 minutes)**
   ```bash
   python3 03_module_usage_demo.py
   ```
   - Import functions from first script
   - Build new functions using imported ones
   - Advanced analysis: `analyze_grade_distribution()`
   - Comprehensive reporting: `generate_detailed_report()`

**Interactive Elements:**
- Build functions step-by-step
- Practice function design and testing
- Demonstrate code organization principles

# Take-away Message:

1. **Git enables collaboration** - "Version control is essential for team projects"
2. **CLI automates workflows** - "Command line tools scale to any project size"
3. **Functions promote reuse** - "Write once, use many times"
4. **Modules organize code** - "Good structure makes code maintainable"
5. **Integration is powerful** - "Tools work better together than alone"
