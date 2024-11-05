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

## Getting Acclimated with the Command Line

**Opinionated recommendation**: best way to learn the command line is to use it regularly, so set up a linux server for yourself

---

## Server Options

- Repurpose Old Hardware
  - Old Mac or PC works great
  - No additional cost
- Cloud Options
  - Google Cloud (free tier available)
  - AWS/Azure (pay-as-you-go)
  - GitHub Codespaces (auto-shutdown)
  - **Note:** Remember to turn off pay-as-you-go instances when not in use!

---

## Accessing Your Server

- GUI Options
  - Remote Desktop (RDP)
  - VNC Viewer
- Command Line
  - SSH for direct terminal access
  - VS Code Terminal

---

## Remote Access Setup

- VS Code (Recommended)
  - Install on server
  - Run: `code tunnel service install`
  - Follow authentication prompts
  - Open VS Code or [https://vscode.dev](https://vscode.dev) -> Remote Explorer
- Advanced Options
  - `ssh`: install `sshd` or `openssh`
  - `rdp`: install `xrdp` ([or use the latest script here](https://c-nergy.be/repository.html))

May need to configure firewall settings or use a VPN

- VPN options:
  - Tailscale (moderate)
  - Wireguard (advanced)

---

## A Few Powerful Command Line Tools

- Manipulation
  - `find`
  - `grep` (again)
  - `sed`
- System monitoring
  - `ps`
  - `top` or `htop`
  - `watch`
  - `df` and `du`

---

## Finding Files with `find`

- Search for files in a directory hierarchy
- Powerful options for filtering and actions

```bash
# Find files by name
find . -name "*.py"

# Find files modified in last 24 hours
find . -mtime -1

# Find and execute command on results
find . -name "*.txt" -exec grep "pattern" {} \;
```

---

## Text Processing with `grep`

- Search for patterns in files
- Supports regular expressions
- Recursive search through directories

```bash
# Basic pattern search
grep "error" logfile.txt

# Case-insensitive recursive search
grep -ri "warning" /var/log/

# Show context around matches
grep -C 2 "exception" app.log
```

---

## Text Manipulation with `sed`

- Stream editor for filtering and transforming text
- Common uses: substitution, deletion, insertion

```bash
# Replace text
sed 's/old/new/g' file.txt

# Delete lines matching pattern
sed '/DEBUG/d' logfile.txt

# Add text at beginning of each line
sed 's/^/PREFIX: /' file.txt
```

---

## System Monitoring Commands

### `ps` - Process Status

```bash
# Show all processes
ps aux

# Show process tree
ps axjf
```

### `top` - Dynamic Process Viewer

- Interactive process viewer
- Real-time system statistics
- Press 'q' to quit, 'h' for help

---

## `time` - How long did that take?

- Measure program execution time
- Shows real, user, and system time
- Built into shell or `/usr/bin/time`

```bash
# Basic usage
time sleep 2

# Detailed statistics with /usr/bin/time
/usr/bin/time -v ls -R /

# Time a complex pipeline
time (find . -type f | wc -l)
```

---

## `watch` - Execute Command Periodically

```bash
# Update every 2 seconds (default)
watch df -h

# Update every 5 seconds
watch -n 5 'ps aux | grep python'
```

### `df` and `du` - Disk Usage

```bash
# Show disk space usage
df -h

# Show directory sizes
du -sh *
```

---

## Examples

These examples demonstrate how to:

1. Combine multiple commands into powerful pipelines
2. Use tmux for managing long-running processes
3. Monitor progress of background tasks
4. Create persistent development environments
5. Process files in bulk with real-time monitoring
6. Measure and analyze program performance
7. Create automated code review tools
8. Build real-time monitoring dashboards
9. Combine multiple tools for complex analysis
10. Use tmux for organized multi-pane workflows

---

## Example 1: Automated Script Generation

Create a shell script that finds and processes log files:

```bash
# Find all Python files, extract function definitions, and create a script
find . -name "*.py" | xargs grep "^def" | sed 's/def /function: /' > functions.sh

# Make it executable and add shebang
sed -i '1i #!/bin/bash' functions.sh && chmod +x functions.sh

# Run it
./functions.sh
```

## Example 2: Long-running Task with Progress Monitoring

Using tmux to run a task and watch its progress:

```bash
# Start new tmux session
tmux new -s backup_session

# In first pane: Start backup
tar -czf backup.tar.gz /large/directory/

# Split pane vertically: Ctrl-b %
# In second pane: Monitor progress
watch -n 1 'du -sh backup.tar.gz'

# Detach from session: Ctrl-b d
# Reattach later: tmux attach -t backup_session
```

## Example 3: Log Analysis Pipeline

Find and analyze error patterns in logs:

```bash
# Find all logs, grep for errors, sort by frequency
find /var/log -name "*.log" -type f -exec grep -i "error" {} \; | \
sort | uniq -c | sort -nr > error_summary.txt

# Watch for new errors in real-time
tmux new-session \
'tail -f /var/log/*.log | grep -i "error" --line-buffered | \
tee -a error_log.txt'
```

## Example 4: System Resource Monitor

Create a custom system monitoring dashboard:

```bash
# Create a tmux session with multiple panes
tmux new-session -d 'top' \; \
  split-window -v 'watch df -h' \; \
  split-window -h 'watch "ps aux | sort -rk 3 | head -n 5"' \; \
  select-layout even-vertical

# Attach to the session
tmux attach
```

## Example 5: Automated File Processing

Process multiple files with progress tracking:

```bash
# Start tmux session
tmux new-session -d

# Process files in background
tmux send-keys 'for f in *.txt; do 
  echo "Processing $f..."
  sed -i "s/old/new/g" "$f"
  sleep 1
done' C-m

# Split and monitor progress
tmux split-window -v
tmux send-keys 'watch -n 1 "ls -l *.txt | wc -l"' C-m

# Attach to session
tmux attach
```

## Example 6: Development Environment Setup

Combine tools to set up a development environment:

```bash
# Create new tmux session
tmux new-session -d -s dev

# Split window into panes
tmux split-window -v
tmux split-window -h

# Run different commands in each pane
tmux select-pane -t 0
tmux send-keys 'watch npm run build' C-m
tmux select-pane -t 1
tmux send-keys 'tail -f logs/development.log' C-m
tmux select-pane -t 2
tmux send-keys 'python manage.py runserver' C-m

# Attach to session
tmux attach -t dev
```

## Example 7: Performance Analysis Pipeline

Analyze command execution times and system load:

```bash
# Start tmux session for monitoring
tmux new-session -d -s perf

# Run command multiple times and collect timing
tmux send-keys 'for i in {1..10}; do
  /usr/bin/time -v ./script.py 2>> timing.log
  sleep 1
done' C-m

# Split window and monitor system load
tmux split-window -v
tmux send-keys 'watch -n 1 "grep -A 5 \"Maximum resident set\" timing.log | tail -n 6"' C-m

# Split again for CPU monitoring
tmux split-window -h
tmux send-keys 'top -b -n 1 | head -n 12' C-m

# Attach to session
tmux attach -t perf
```

## Example 8: Automated Code Review

Find and analyze code patterns:

```bash
# Create a tmux session
tmux new-session -d -s code_review

# Find all Python files and analyze imports
tmux send-keys 'find . -name "*.py" -exec grep "^import" {} \; | \
sort | uniq -c | sort -nr > imports.txt' C-m

# Find long functions (>50 lines)
tmux split-window -v
tmux send-keys 'for f in $(find . -name "*.py"); do
  echo "=== $f ==="
  awk "/def /,/^$/" "$f" | grep -v "^$" | grep -B1 -A50 "def " 
done > long_functions.txt' C-m

# Monitor for TODO comments
tmux split-window -h
tmux send-keys 'watch -n 5 "find . -type f -exec grep -l \"TODO\" {} \; | wc -l"' C-m

# Attach to session
tmux attach -t code_review
```

## Example 9: Real-time Log Analysis Dashboard

Create a multi-pane monitoring setup:

```bash
# Start new tmux session
tmux new-session -d -s logs

# Watch for errors
tmux send-keys 'tail -f /var/log/syslog | grep -i "error" --color' C-m

# Split and watch warnings
tmux split-window -v
tmux send-keys 'tail -f /var/log/syslog | grep -i "warning" --color' C-m

# Split and show error counts
tmux split-window -h
tmux send-keys 'watch -n 10 "grep -i \"error\" /var/log/syslog | cut -d\" \" -f5 | sort | uniq -c | sort -nr"' C-m

# Create summary window
tmux split-window -v
tmux send-keys 'while true; do
  clear
  echo "=== System Status ==="
  uptime
  echo "=== Disk Space ==="
  df -h | grep -v "tmpfs"
  echo "=== Memory Usage ==="
  free -h
  sleep 30
done' C-m

# Attach to session
tmux attach -t logs
```
