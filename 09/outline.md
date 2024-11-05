# Revisiting the Command Line

## Basic Navigation

- `pwd` - Print Working Directory
- `ls` - List directory contents
  - `-l` - Long format
  - `-a` - Show hidden files
- `cd` - Change Directory
  - `cd ~` - Home directory
  - `cd ..` - Parent directory
  - `cd -` - Previous directory

---

## File Operations

- `mkdir` - Create directory
- `touch` - Create empty file or update timestamp
- `cp` - Copy files/directories
  - `cp source destination`
  - `cp -r` - Copy directories recursively
- `mv` - Move/rename files
- `rm` - Remove files
  - `rm -r` - Remove directories recursively
  - `rm -f` - Force remove without confirmation

---

## File Viewing

- `cat` - Display file contents
- `head` - Show first lines of file
  - `head -n N` - Show first N lines
- `tail` - Show last lines of file
  - `tail -n N` - Show last N lines
  - `tail -f` - Follow file updates

---

## Text Processing

- `grep` - Search for patterns
  - `grep "pattern" file`
  - `-i` - Case insensitive
  - `-r` - Recursive search
  - `-n` - Show line numbers
- `cut` - Extract columns from files
  - `cut -d',' -f1,3` - Extract columns 1 and 3 using comma delimiter
  - `cut -c5-10` - Extract characters 5-10 from each line
- `tr` - Translate characters
  - `tr 'a-z' 'A-Z'` - Convert to uppercase
  - `tr -d '0-9'` - Delete digits
  - `tr -s ' '` - Squeeze repeated spaces
- `sed` - Stream editor for text manipulation
  - `sed 's/old/new/'` - Replace first occurrence
  - `sed 's/old/new/g'` - Replace all occurrences
  - `sed '/pattern/d'` - Delete lines matching pattern

---

## File Links

- `ln` - Create links
  - `ln -s target link_name` - Create symbolic link
  - `ln target link_name` - Create hard link

### Environment Variables

- `echo $VARIABLE` - Display variable value
- `export VARIABLE=value` - Set environment variable
- `env` - Display all environment variables
- `.env` files for project-specific variables
  - Create: `touch .env`
  - Format: `VARIABLE_NAME=value`
  - **Never commit .env files to version control!**

---

## Shell Scripts

- First line: `#!/bin/bash` (shebang)
- Make executable: `chmod +x script.sh`
- Run: `./script.sh` or `bash script.sh`
- Arguments: `$1`, `$2`, etc. ($0 is script name)
- `$#` - Number of arguments passed

---

## File Permissions

- `chmod` - Change file mode
  - `chmod +x file` - Make executable
  - `chmod u+w file` - Add write permission for user
  - Numeric mode: `chmod 644 file` (owner rw, group/others r)

### Task Scheduling

- `cron` - Schedule recurring tasks
  - Edit: `crontab -e`
  - Format: `* * * * * command`
  - Fields: minute hour day_of_month month day_of_week
  - Example: `0 2 * * * backup.sh` (run at 2 AM daily)

---

## Remote Access

- `ssh` - Secure shell
  - `ssh user@host` - Connect to remote host
- `scp` - Secure copy
  - `scp file user@host:/path` - Copy to remote
  - `scp user@host:/path file` - Copy from remote
  - `scp -r` - Copy directories

---

## Session Management

- `tmux` - Modern terminal multiplexer ([`tmux` cheat sheet](https://devhints.io/tmux))
  - `tmux` - Start new session
  - `tmux new -s name` - Start named session
  - `tmux attach -t name` - Attach to session
  - `tmux ls` - List sessions
  - `Ctrl-b d` - Detach
- `screen` - Terminal multiplexer
  - `screen` - Start new session
  - `screen -S name` - Start named session
  - `screen -r` - Reattach
  - `Ctrl-a d` - Detach

---

## Compression

- `tar` - Archive files
  - `tar -cvf archive.tar files` - Create archive
  - `tar -xvf archive.tar` - Extract archive
  - `tar -czvf archive.tar.gz files` - Create compressed archive
  - `tar -xzvf archive.tar.gz` - Extract compressed archive
- `zip/unzip`
  - `zip archive.zip files` - Create zip archive
  - `zip -r archive.zip directory` - Zip directory
  - `unzip archive.zip` - Extract zip archive

---

## Pipes and Redirection

- `|` - Pipe output to another command
  - Example: `cat file.txt | grep "pattern"`
- `>` - Redirect output (overwrite)
  - Example: `echo "text" > file.txt`
- `>>` - Redirect output (append)
  - Example: `echo "more text" >> file.txt`
- `2>&1` - Redirect stderr to stdout

---



- Manipulation
  - `find`
  - `grep`
  - `sed`
- System monitoring
  - `ps`
  - `top` or `htop`
  - `watch`
  - `df` and `du`

---



## Basic Python

- Variables and Data Types
  - Integers, floats, strings
  - Variables are dynamically typed
  - Type conversion and checking
  - String operations and f-strings

- Control Structures
  - If/elif/else conditionals
  - For and while loops
  - Break and continue statements
  - Compound conditions with `and`, `or`, `not`

---

## Functions and Methods

- Functions and Methods
  - Function definition with `def`
  - Parameters and return values
  - Default arguments
  - Command line arguments

- Packages and Modules
  - Installing packages
  - Importing with aliases
  - Specific functions and classes
  - Managing virtual environments

---

## Data Structures

- Lists
  - Creation and indexing
  - List methods (append, extend, pop)
  - List slicing and operations
  - List comprehensions
  - Sorting and searching

- Dictionaries
  - Key-value pairs
  - Dictionary methods
  - Nested dictionaries
  - Dictionary comprehensions
  - Default dictionaries

---

## Data Structures II

- Sets
  - Unique elements
  - Set operations (union, intersection)
  - Set methods
  - Set comprehensions

- Tuples
  - Immutable sequences
  - Tuple packing/unpacking
  - Named tuples
  - Using tuples as dictionary keys

---

## File Operations

- File Handling
  - Opening and closing files
  - Reading and writing text files
  - Context managers (`with` statement)
  - Binary file operations
  - CSV and JSON handling

- Path Operations
  - Path manipulation with `os.path`
  - Modern path handling with `pathlib`
  - Directory operations
  - File system navigation

---

## Numerical Packages

- NumPy
  - Arrays and operations
  - Broadcasting
  - Mathematical functions
  - Array manipulation

- Pandas
  - Series and DataFrames
  - Data loading and saving
  - Data cleaning and transformation
  - Grouping and aggregation
  - Time series functionality

---

## Data Visualization

- Matplotlib
- Seaborn statistical plots
- Interactive visualization
- Customizing plots

---

## Statistical Methods

- Time Series Analysis
  - DateTime handling
  - Resampling and rolling windows
  - Seasonal decomposition
  - ARIMA models
- statsmodels
  - Linear regression
  - Generalized linear models
  - Statistical tests
  - Model diagnostics
- Machine Learning
  - scikit-learn basics
  - Model selection and evaluation
  - Feature engineering
  - Cross-validation

---

## Data Science Fundamentals

- Jupyter Notebooks
  - Remote access and configuration
  - Magic commands
  - Cell execution and kernel management
- NumPy
  - Array operations and broadcasting
  - Mathematical functions
  - Array manipulation and indexing
  - Universal functions (ufuncs)
- Pandas
  - Series and DataFrame objects
  - Data loading and manipulation
  - Missing data handling
  - Grouping and aggregation

---

## Error Handling

- Reading error messages
- VS Code debugger
- Try/except