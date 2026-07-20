# Lecture 02 bonus: Optional shell automation

This material is optional and not required for assignments. The required Lecture 02 Python work uses `.py` scripts run from the terminal; it does not require writing shell scripts. Lecture 03 introduces the small command-line pipeline it needs directly.

Practice only inside a disposable project directory. Shell scripts can affect many files quickly, so inspect paths and expansions before adding file-changing commands.

# A small Bash script

A shell script stores terminal commands in a text file. A **shebang** on the first line can select the program that interprets it; `#!/usr/bin/env bash` locates Bash through the current environment. The setting `set -u` stops the script when it uses an unset variable. This is one useful check, not complete error handling.

Save this example as `show_project.sh`:

```bash
#!/usr/bin/env bash

set -u

echo "Working directory:"
pwd

echo "Python files:"
find . -maxdepth 2 -name "*.py" -print
```

Run the script by passing its path to Bash:

```bash
bash show_project.sh
```

Making a script executable with `chmod` is optional; invoking `bash` directly is sufficient for this example.

# Positional arguments

A **positional argument** is text supplied after the script name. Inside a Bash script, `$1` is the first argument. The redirection `>&2` sends a message to the standard error stream, and `exit 1` stops the script with a nonzero failure status.

Save this as `show_named_file.sh`:

```bash
#!/usr/bin/env bash

set -u

file_path="$1"

if [[ ! -f "$file_path" ]]; then
    echo "File not found: $file_path" >&2
    exit 1
fi

echo "First lines of $file_path:"
head "$file_path"
```

Run it with an explicit path:

```bash
bash show_named_file.sh README.md
```

Double quotes around `"$file_path"` keep a path containing spaces together as one argument.

# Exit status and conditional chaining

A command returns an **exit status**. Status `0` conventionally indicates success; a nonzero status indicates failure.

`&&` runs the next command only when the previous command succeeds:

```bash
python main.py && echo "Python program completed"
```

`||` runs the next command only when the previous command fails:

```bash
python main.py || echo "Python program failed" >&2
```

Use these operators for small, visible sequences. A longer workflow is easier to understand when its checks and errors are written explicitly in a script.

# Pipes and redirection

A **pipe** sends the standard output of one command to the standard input of another:

```bash
find . -maxdepth 2 -name "*.py" -print | sort
```

Redirection sends output to a file:

```bash
python main.py > run_output.txt
```

`>` replaces the destination file. `>>` appends instead. Error output is a separate stream and is not captured by `>` unless it is redirected deliberately.

# A safe project check

This script checks for required files without creating, moving, or removing anything:

```bash
#!/usr/bin/env bash

set -u

missing=0

for file_path in README.md main.py analysis_utils.py; do
    if [[ -f "$file_path" ]]; then
        echo "Found: $file_path"
    else
        echo "Missing: $file_path" >&2
        missing=1
    fi
done

exit "$missing"
```

Read-only checks are a good first automation exercise because a mistaken path reports the wrong file without deleting or overwriting it.

# Further exploration

Useful later topics include:

- `set -e` and `set -o pipefail`, including their edge cases;
- functions within Bash scripts;
- temporary directories created with `mktemp`;
- cleanup with `trap`; and
- dedicated tools for repeatable workflows, such as task runners and continuous integration.

Treat recursive file operations, unverified wildcards, and downloaded copy-and-paste scripts as high-risk until you can state their exact targets and effects.
