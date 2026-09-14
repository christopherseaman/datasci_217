# Lecture 01 Live Demo Guide

There are three live demos, grouped from five runnable pieces:

1. Setup, VS Code, and shell navigation (`01_github_vscode_setup_guide.md` plus `02_cli_navigation_demo.sh`)
2. Python basics and control structures (`03_python_basics_demo.py` plus `04_control_structures_demo.py`)
3. A small end-to-end workflow (`05_integration_workflow_demo.py`)

These are scripts and walkthroughs, not notebooks. Run them from a disposable
copy if you want to keep the repository clean.

## Demo 1: Setup, VS Code, and shell navigation

### Setup and VS Code walkthrough

Use [VS Code](https://code.visualstudio.com/) as the editor and terminal for
the walkthrough. On Windows, use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
for the Bash commands in this course; run `wsl --install` from Administrator
PowerShell, restart if prompted, and open the Ubuntu terminal.

Work through `01_github_vscode_setup_guide.md`:

- create or open a GitHub account and find the privacy-preserving `noreply` email;
- install the Python extension in VS Code;
- configure Git with your name and GitHub `noreply` email;
- create `hello.py` containing `print("Hello, Data Science!")` and run it;
- verify `python3 --version`, `git --version`, and `pwd` in the VS Code terminal.

### Shell navigation script

Run the script from a disposable directory:

```bash
mkdir -p scratch/lecture01-cli
cd scratch/lecture01-cli
bash /path/to/datasci_217/01/demo/02_cli_navigation_demo.sh
```

It practices `pwd`, `ls`, `mkdir`, `cd`, `cat`, `echo`-based file creation,
and relative paths. The intentional path failure is part of the exercise: read
the error, identify the working directory, and rerun with the corrected path.

**Try it:** Before running the script, predict which files and directories it
will create. Afterward, use `find` or `ls -R` to check your prediction.

## Demo 2: Python basics and control structures

### Python basics and debugging

```bash
python3 03_python_basics_demo.py
```

The script covers variables, types, f-strings, lists, arithmetic, and reading
tracebacks. Several mistakes are commented out beside their fixes so you can
predict the error before enabling a line.

**Try it:** Explain the typo, missing `f` prefix, zero-based index, and
division-by-zero guard before reading the printed explanation.

### Control structures

```bash
python3 04_control_structures_demo.py
```

The script practices comparisons, `if`/`elif`/`else`, `for`, `enumerate`,
`while`, nested loops, `break`, `continue`, and a small grading summary using
only Lecture 01 concepts.

**Try it:** Predict the output of one loop and one conditional branch before
running the script. Then change a threshold and rerun it.

## Demo 3: Complete integration

```bash
python3 05_integration_workflow_demo.py
```

This list-based fixture combines the command line, variables, lists, loops,
conditionals, arithmetic, and redirected output:

```bash
python3 05_integration_workflow_demo.py > results.txt
```

The script prints each student's status, then reports the average, minimum,
maximum, and passing count. Change one input value and rerun the command to see
how the result changes.
