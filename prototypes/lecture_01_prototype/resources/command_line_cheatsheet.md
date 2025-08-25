# Command Line Quick Reference

## Essential Navigation Commands

| Command | Description | Example |
|---------|-------------|---------|
| `pwd` | Print working directory | `pwd` |
| `ls` | List directory contents | `ls -la` |
| `cd` | Change directory | `cd Documents` |
| `cd ..` | Go up one level | `cd ..` |
| `cd ~` | Go to home directory | `cd ~` |

## File Operations

| Command | Description | Example |
|---------|-------------|---------|
| `mkdir` | Create directory | `mkdir project` |
| `touch` | Create empty file | `touch script.py` |
| `cp` | Copy files | `cp file.py backup.py` |
| `mv` | Move/rename files | `mv old.py new.py` |
| `rm` | Remove files (careful!) | `rm unwanted.txt` |

## Text Processing

| Command | Description | Example |
|---------|-------------|---------|
| `cat` | Display file contents | `cat data.csv` |
| `head` | Show first lines | `head -10 data.csv` |
| `tail` | Show last lines | `tail -5 log.txt` |
| `grep` | Search patterns | `grep "error" log.txt` |
| `\|` | Pipe commands | `cat file.txt \| grep pattern` |

## Python Execution

| Command | Description | Example |
|---------|-------------|---------|
| `python3` | Run Python interpreter | `python3` |
| `python3 script.py` | Run Python script | `python3 analysis.py` |
| `python3 --version` | Check Python version | `python3 --version` |
| `python3 -c "code"` | Run Python one-liner | `python3 -c "print('hello')"` |