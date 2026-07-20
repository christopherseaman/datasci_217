# Lecture 01 bonus: Additional Terminal and Python Patterns

This material is optional. It extends the required terminal-and-script workflow but is not assumed by Lecture 02 and is not assessed in Assignment 01.

# Additional shell patterns

## Globbing

The shell can expand patterns before it runs a command:

```bash
ls *.py
ls data/*.csv
```

`*` matches zero or more characters. Always inspect a glob with `ls` before combining it with a command that moves, changes, or removes files.

## Brace expansion

Bash and zsh can generate several related names:

```bash
mkdir -p project/{data,scripts,results}
```

The shell expands this into three directory paths before `mkdir` runs. The `-p` option creates any missing parent directories and does not treat an already-existing directory as an error.

## Command substitution

`$(...)` replaces itself with a command's output:

```bash
mkdir "backup-$(date +%Y-%m-%d)"
```

This is convenient for automation, but it is less transparent than using a fixed scratch name while learning paths.

## Finding files

`find` searches a directory tree:

```bash
find . -name "*.py"
```

`.` selects the current directory as the search root. `-name "*.py"` selects paths whose names match the pattern. Keep search and file-changing operations separate until you can verify every matched path.

## Recursive operations

Options such as `cp -r` and `rm -r` operate on an entire directory tree. They are intentionally excluded from required Lecture 01 practice because a mistaken path can affect many files. Prefer a disposable scratch directory and named single-file operations while learning.

# Python's interactive prompt

The interactive prompt is useful for tiny experiments:

```bash
python
```

```text
>>> 2 + 3
5
>>> exit()
```

The prompt does not preserve a readable program automatically. Put any result you want to reproduce into a `.py` script.

# Additional f-string formatting

F-strings support separators, percentages, and scientific notation:

```python
revenue = 15432.5
success_rate = 0.847
small_value = 0.0000123

print(f"Revenue: ${revenue:,.2f}")
print(f"Success rate: {success_rate:.1%}")
print(f"Small value: {small_value:.2e}")
```

These formats are useful when they clarify meaning. They are not substitutes for correct calculations or clear labels.

# Interactive input

`input()` always returns a string:

```python
age_text = input("Age: ")
age = int(age_text)
print(f"Age next year: {age + 1}")
```

Most course analyses use files or supplied data rather than interactive prompts. Automatic validation of unsuitable input requires exception handling, which is introduced later.
