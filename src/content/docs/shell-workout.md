---
title: Shell Workout
---

## Basic Exercises

1. **Print a String with Quotation Marks**
    - Write a command to print the string `Use " man echo"` _including the quotation marks_
    - **Hint:** Use double quotes in the inner string and wrap the whole thing in single quotes

2. **Sleep Command**
    - Read `man sleep` and figure out how to make the terminal "sleep" for 5 seconds before running another command
    - Example: Sleep before printing "Hello, World!"

3. **Canceling a Long-Running Command**
    - Execute the command `sleep 5000`
    - Realize that's well over an hour
    - Learn how to cancel the running command

4. **File Manipulation Sequence**
    - Create an empty file called `foo`
    - Rename `foo` to `bar`
    - Copy `bar` to `baz`

5. **List Specific Files**
    - List only the files starting with "b" in the current directory
    - **Hint:** Use a wildcard

6. **Efficient File Removal**
    - Remove both `bar` and `baz` using a _single_ `rm` command
    - **Hint:** Check if those are the only two files in the directory starting with "b"

7. **Create a Text File**
    - Create a file `hello.txt` containing the text "Hello, World!"
    - Use `echo` and a redirect `>`

8. **Simple Greeting Shell Script**
    - Write a shell script that greets the user
    - **Hint:** The current user is stored in the `$USER` environment variable in most shells

## Intermediate Exercises

9. **File Content Extraction**
    - Use `head` and `tail` to display:
        - First 5 lines of a text file
        - Last 3 lines of a text file
    - **Bonus:** Combine these in a single command pipeline

10. **Find Large Files**
    - Create a command that finds all files in the current directory larger than 1MB
    - **Hint:** Use `find` with size parameters

11. **Directory Analysis Script**
    - Write a shell script that:
        - Takes a directory path as an argument
        - Counts the number of files in that directory
        - Prints the total size of all files in that directory
        - Displays the 3 largest files in that directory

12. **Log File Analysis with grep**
    - Use `grep` to:
        - Find all lines containing the word "error" in a log file
        - Count the number of error lines
        - Make the search case-insensitive

13. **File Existence and Permissions Script**
    - Create a shell script that:
        - Checks if a file exists
        - If it exists, display its permissions
        - If it doesn't exist, create an empty file with read and write permissions for the owner

14. **Text Transformation with tr**
    - Use `tr` to:
        - Convert a text file to uppercase
        - Remove all digits from a text file
        - Squeeze multiple spaces into a single space

15. **File Size Ranking Pipeline**
    - Write a command that uses pipes to:
        - List all files in a directory
        - Sort them by size
        - Display the top 5 largest files

## Advanced Exercises

16. **Backup Utility Script**
    - Create a shell script that acts as a simple backup utility:
        - Takes a source directory and destination directory as arguments
        - Uses `tar` to create a compressed backup
        - Names the backup with the current date (use `date +%Y-%m-%d`)
        - Optionally, keep only the last 5 backups by removing older ones

17. **System Resource Monitor**
    - Write a script that monitors system resources:
        - Use `top` or `ps` to capture CPU and memory usage
        - Write the output to a log file
        - If memory usage exceeds 80%, send an alert message

18. **Python Function Analyzer**
    - Create a command pipeline that:
        - Finds all Python files in a directory
        - Extracts all function definitions
        - Counts the number of functions
        - Sorts functions by name

19. **Selective File Archiver**
    - Write a shell script that:
        - Takes a file extension as an argument
        - Finds all files with that extension in the current directory and subdirectories
        - Calculates and displays the total size of those files
        - Creates a compressed archive of those files

20. **Environment Variable Manager**
    - Design a script that:
        - Checks the current environment variables
        - Allows adding a new environment variable
        - Allows removing an existing environment variable
        - Displays all current environment variables

## Bonus Challenges

21. **tmux Monitoring Session**
    - Create a tmux session script that:
        - Opens multiple panes
        - Runs different monitoring commands in each pane
        - Automatically attaches to the session

22. **Comprehensive System Information Script**
    - Write a script that displays:
        - Current user
        - Disk usage
        - CPU and memory usage
        - Top 5 running processes
        - Network interface information

23. **Log Rotation Script**
    - Develop a log rotation script that:
        - Takes a log file path as an argument
        - Compresses old log files
        - Keeps only the last N log files
        - Can be scheduled with cron

24. **Remote File Synchronization**
    - Create a script using `scp` that:
        - Takes source and destination paths
        - Synchronizes files
        - Handles different remote server configurations
        - Provides verbose output

25. **Interactive System Maintenance Menu**
    - Design an interactive menu-driven shell script that:
        - Provides options for system maintenance tasks
        - Includes options like disk cleanup, process management, backup
        - Uses `select` for menu creation
        - Implements error handling and input validation
