---
marp: true
theme: gaia
paginate: true
---
# Lecture 4

- Lies from the previous lecture
- Python
	- File operations
	- Functions & methods
- Command line
	- Remote access with `ssh`
	- Persistent connections
	- Remote Jupyter notebooks
	- Brief: CUDA and GPUs with Python
	- Brief: Submitting jobs to the university HPC cluster
---
# Lies!

- Persistent environment variables
- Setting environment variables from `.env` in the shell

---
## Persistent Environment Variables:
Setting default editor to nano

1. Open your shell configuration file: `nano ~/.bashrc`
2. Add this line at the end of the file: `export EDITOR=nano`
3. Save and exit (Ctrl+X, then Y, then Enter)
4. Reload the configuration: `source ~/.bashrc`
---
## THAT DIDN'T WORK! Why?

Modifying only `.bashrc` won't work for all scenarios because:

1. Different shells use different configuration files
2. Some programs may not read `.bashrc`
3. Operating systems may have different default behaviors

For example, if a user is using Zsh (default on macOS since Catalina) instead of Bash, changes in `.bashrc` won't affect their environment.

### Find out which shell you're using with `echo $SHELL`

---
## Configuration Files: `bash` (most common)

- `.bashrc`: Executed for interactive non-login shells
- `.bash_profile`: Executed for login shells
- `.bash_login`: Executed for login shells if `.bash_profile` doesn't exist
- `.profile`: Executed for login shells if neither `.bash_profile` nor `.bash_login` exist

---
## Configuration Files: `zsh` (MacOS default)

- `.zshenv`: Executed for all shells (login, interactive, or script)
- `.zprofile`: Executed for login shells
- `.zshrc`: Executed for interactive shells
- `.zlogin`: Executed for login shells, after `.zshrc`
- `.zlogout`: Executed when a login shell exits
---
## Configuration Files: Others

- `fish`
   - `config.fish`: Executed for all shells
   - `fish_variables`: Stores universal variables

- `tcsh`
   - `.tcshrc`: Executed for all shells
   - `.login`: Executed for login shells, after `.tcshrc`

- `ksh` (Korn Shell)
   - `.kshrc`: Executed for interactive shells
   - `.profile`: Executed for login shells
   
---
## Configuration File Takeaways

To ensure changes apply across different shells and scenarios:
- For `bash` users: Modify both `.bashrc` and `.bash_profile`
- For `zsh` users (e.g, macOS): Focus on `.zshenv`, `.zshrc`, and `.zprofile`
- For cross-shell compatibility use shell-specific files to source a common configuration


---
## Persistent Environment Variables (again)
Setting default editor to nano

1. Use `echo $SHELL` to learn which config file to change
2. Open your shell configuration file: `nano ~/<CONFIG_FILE>`
3. Add this line at the end of the file: `export EDITOR=nano`
4. Save and exit (Ctrl+X, then Y, then Enter)
5. Reload the configuration: `source ~/<CONFIG_FILE>`

---
## Setting Variables from `.env` in the Shell

There is **NOT** a single command to load a `.env` file, so let's define one in our shell config using [`set`'s `allexport` option](https://linuxcommand.org/lc3_man_pages/seth.html):

```bash
# Add this to the shell configuration file, e.g., .bashrc for bash
load_env () {
    set -o allexport # enable the "allexport" option
    source $1        # set env var's from .env file
    set +o allexport # disable the "allexport" option
}

# Usage
load_env /path/to/.env
```

---
# LIVE DEMO

---
## Python: Files & Functions

- Interacting with files
- Python functions, modules
- Common file operations 
- Reading a file line-by-line
- Splitting lines into arrays
---

## Interacting with Files

Basic file operations:
- Opening a file: `open(filename, mode)`
- Reading from a file: `file.read()`, `file.readline()`, `file.readlines()`
- Writing to a file: `file.write()`, `file.writelines()`
- Closing a file: `file.close()`

Always use the `with` statement for automatic file closing:

```python
with open('example.txt', 'r') as file:
    content = file.read()
```

---

## File Modes

Common file modes:
- `'r'`: Read (default)
- `'w'`: Write (overwrites existing content)
- `'a'`: Append
- `'r+'`: Read and write
- `'b'`: Binary mode (e.g., `'rb'`, `'wb'`)

Example:
```python
with open('example.txt', 'w') as file:
    file.write('Hello, World!')
```

---

## Reading a File Line-by-Line

Method 1: Using a for loop
```python
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())
```

Method 2: Using `readline()`
```python
with open('example.txt', 'r') as file:
    while True:
        line = file.readline()
        if not line:
            break
        print(line.strip())
```

---

## Splitting Lines into Arrays

Using the `split()` method:

```python
with open('data.txt', 'r') as file:
    for line in file:
        # Split by whitespace (default)
        items = line.split()
        
        # Split by specific delimiter
        items = line.split(',')
        
        print(items)
```

---

## Common File Operations

- Check if a file exists:
  ```python
  import os
  os.path.exists('file.txt')
  ```

- Delete a file:
  ```python
  import os
  os.remove('file.txt')
  ```

- Rename a file:
  ```python
  import os
  os.rename('old_name.txt', 'new_name.txt')
  ```

---
## Common Directory Operations
- Create a new directory:
  ```python
  import os
  os.mkdir('new_directory')
  ```

- Create nested directories:
  ```python
  import os
  os.makedirs('path/to/new/directory')
  
  # Can also allow the directory to already exist
  os.makedirs('path/to/new/directory', exist_ok = True)
  ```

---

## Working with Directories

- Get current working directory:
  ```python
  import os
  current_dir = os.getcwd()
  ```

- Change current working directory:
  ```python
  import os
  os.chdir('/path/to/new/directory')
  ```

- List contents of a directory:
  ```python
  import os
  contents = os.listdir('/path/to/directory')
  ```

- Check if a path is a directory:
  ```python
  import os
  is_dir = os.path.isdir('/path/to/check')
  ```

---

## Python Functions

Defining a function:
```python
def greet(name):
    return f"Hello, {name}!"

# Calling the function
message = greet("Alice")
print(message)  # Output: Hello, Alice!
```

Function with default parameters:
```python
def greet(name="World"):
    return f"Hello, {name}!"

print(greet())  # Output: Hello, World!
print(greet("Bob"))  # Output: Hello, Bob!
```

---
## Function Arguments

Positional arguments:
```python
def add(a, b):
    return a + b

result = add(3, 5)  # result = 8
```

Keyword arguments:
```python
def greet(first_name, last_name):
    return f"Hello, {first_name} {last_name}!"

message = greet(last_name="Doe", first_name="John")
print(message)  # Output: Hello, John Doe!
```

---

## `*args` and `**kwargs`

`*args`: Variable number of positional arguments
```python
def sum_all(*args):
    return sum(args)

result = sum_all(1, 2, 3, 4)  # result = 10
```

`**kwargs`: Variable number of keyword arguments
```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="New York")
```
---
## Command Line Arguments in Python

You can pass arguments to python just like any other command

- Two main methods:
  1. `sys.argv`: Argument order matters
	  `python script.py arg1 arg2`
  2. `argparse`: Arguments are explicitly named
	  `python script.py -two arg2 -one arg1`

---
## Using sys.argv

```python
import sys

script_name = sys.argv[0]
arguments = sys.argv[1:]

print(f"Script: {script_name}")
print(f"Args: {arguments}")
```

Usage: `python script.py arg1 arg2`

---
## Using argparse Module

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Name to greet")
parser.add_argument("-c", "--count", type=int, default=1)

args = parser.parse_args()

for _ in range(args.count):
    print(f"Hello, {args.name}!")
```

Usage: `python script.py Alice -c 3`

---
## Key Benefits of argparse

- Automatic help messages
- Type conversion
- Optional and positional arguments
- Default values

Example: `python script.py -h`

---
## Python Modules

Importing modules:
```python
import math
print(math.pi)  # Output: 3.141592653589793

from math import sqrt
print(sqrt(16))  # Output: 4.0

from math import *  # Import all (use cautiously)
```

---
## Modules are just `.py` files!

Creating your own module:
1. Create a file `mymodule.py`
2. Define functions in the file
3. Import and use in another file:
   ```python
   import mymodule
   mymodule.my_function()
   ```

---
---
## Summary

- File operations: open, read, write, close
- Reading files line-by-line
- Splitting lines into arrays
- Defining and using functions
- Function arguments: positional, keyword, *args, **kwargs
- Working with modules
---
# LIVE DEMO!!!
---
## Jupyter Notebooks

- Jupyter basics
- Remote Jupyter
	- No longer supported at Wynton
	- [Paperspace](https://paperspace.com) - free option
	- $\$\$ (advanced) [AWS](https://docs.aws.amazon.com/dlami/latest/devguide/setup-jupyter.html) and [GCP](https://cloud.google.com/deep-learning-vm/docs/jupyter)
---
## What is Jupyter Notebook?

- Interactive computing environment for Python, R, Julia, …
- Combines code execution, rich text, mathematics, plots and rich media
- Uses:
  - Data cleaning and transformation
  - Numerical simulation
  - Statistical modeling
  - Machine learning
- File format: `.ipynb` (IPython Notebook)
- Key features:
  - In-line code execution
  - Markdown support
  - Code and output in the same document
  - Easy sharing and collaboration

---
## Creating a Jupyter Notebook

From the Terminal:
1. Install Jupyter: `pip install jupyter`
2. Navigate to your project directory
3. Start Jupyter: `jupyter notebook`
4. In the browser interface click "New" > "Python 3" 

From VS Code:
1. Install "Jupyter" extension
2. Command palette: "Jupyter: Create New Blank Notebook"
4. Select Python kernel when prompted

Both methods create a `.ipynb` file in your current directory.

---
## Connecting to a Remote Jupyter Notebook with VS Code
1. Start Jupyter on remote server:
   ```bash
   jupyter notebook --no-browser --port=PORTNUMBER # Often 8888
   ```

2. In VS Code:
   - Command palette: "Jupyter: Specify local or remote Jupyter server"
   - Enter the remote server's URL (e.g., `http://localhost:8888`)
   - Provide the token or password if prompted

3. Open or create a notebook
   - It will now run on the remote server
   - Any files or artifacts also have to be on the remote server

Note: Ensure proper port forwarding if using SSH

---

# LIVE DEMO!!!

---
## Spooky Action at a Distance

- `ssh`
- `scp`

---
# SSH 

- [UCSF "Wynton" HPC](https://wynton.ucsf.edu/hpc/about/join.html) (IT approval required)
- [Super Dimension Fortress](https://sdf.org) Remote Learning Lab
- [Google Cloud Shell](https://cloud.google.com/free/docs/compute-getting-started)
- [GitHub Codespaces](https://cli.github.com/manual/gh_codespace_ssh)
- Your own machine! (easiest with macOS & Linux)

```shell
ssh user@host.address 
# Then enter your password or automatically
# connect with a pre-shared key
```

---
## Super Dimension Fortress Remote Learning Lab

Offers basic access to a learning environment for free.

Open command line:

```shell
ssh new@sdf.org
# Then follow the instructions
```

---
## Your own Mac

![bg contain](mac_ssh.png)

---
## Google Cloud Shell

- Free temporary virtual machinen
- Persistent 5gb storage

1. Open [Google Cloud Console](https://console.cloud.google.com)
2. Click the button at the top right that looks like a shell
 ![](media/Pasted%20image%2020241001193551.png)
---
## Google Cloud Free Tier

If you want an always-on option, Google Cloud offers a [free tier](https://cloud.google.com/free/docs/free-cloud-features#compute) for their Compute VM service:

- One instance: `e2-micro`
- Region: `us-west1`, `us-central1`, or `us-east1`
- Storage: 30gb persistent
- Always-on vs. Cloud Shell only active when you are 

---
## GitHub CLI to SSH into Codespaces

1. Install GitHub CLI
2. Authenticate with GitHub
3. Create or select a Codespace
4. Connect via SSH

---

# GitHub: Install and Authenticate CLI

[GitHub offers a Command Line Interface](https://github.com/cli/cli), which includes many git commands as well as `ssh` access to Codespaces

```bash
# Install GitHub CLI (example for macOS with Homebrew)
brew install gh
winget install --id GitHub.cli

# Authenticate
gh auth login
```

Follow the prompts to complete authentication.

---

# GitHub: Create and Connect to Codespace

```bash
# Create a new Codespace
gh codespace create

# List available Codespaces
gh codespace list

# SSH into a Codespace
gh codespace ssh -c CODESPACE_NAME
```

Replace `CODESPACE_NAME` with your Codespace's name.

---
## Persistent Sessions on Remote Machines

- Challenge: SSH connections can drop unexpectedly
- Solution: Tools for maintaining persistent sessions
  - Screen
  - Tmux
  - Mosh (Mobile Shell)

**NOTE:** None of these will persist across machine restarts

---
## Screen

- Basic usage:
  ```bash
  screen              # Start a new session
  screen -S name      # Start a named session
  screen -ls          # List sessions
  screen -r [name]    # Reattach to a session
  ```
- Within a screen session:
  - `Ctrl-a d`: Detach from session
  - `Ctrl-a c`: Create a new window
  - `Ctrl-a n`: Next window
  - `Ctrl-a p`: Previous window

---
## Tmux

- Similar to Screen, but with more features
- Basic usage:
  ```bash
  tmux                # Start a new session
  tmux new -s name    # Start a named session
  tmux ls             # List sessions
  tmux attach -t name # Attach to a session
  ```
- Within a tmux session:
  - `Ctrl-b d`: Detach from session
  - `Ctrl-b c`: Create a new window
  - `Ctrl-b %`: Split pane vertically
  - `Ctrl-b "`: Split pane horizontally

---
## Mosh (Mobile Shell)

- Alternative to SSH, more resilient to network issues
- Maintains connection despite IP changes or sleep/wake
- Basic usage:
  ```bash
  mosh username@remote-server
  ```
- Requires installation on both client and server (advanced)
- Uses SSH for initial authentication

---
## Comparison

| Feature | Screen | Tmux | Mosh |
|---------|--------|------|------|
| Persistence | Yes | Yes | Yes |
| Split panes | Limited | Yes | No |
| Network resilience | No | No | Yes |
| Scroll back | Yes | Yes | Limited |
| Learning curve | Moderate | Steeper | Easy |

---
# LIVE DEMO!!!