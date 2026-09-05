---
notion:
  role: bonus
  status: unmapped
  page_id: null
  url: null
  target:
    role: dlc_subpage
    parent_page_id: "271d9fdd-1a1a-8036-9c2b-c4a66ae97d9d"
    anchor: top
  note: "Lecture 02 has optional material in Notion, but no separate bonus page was found."
---

# Bonus Content: Advanced Git Concepts

*This content is optional and not required for assignments. It's here for students who want to dive deeper into Git.*

## Command Line Git (Power User Track)

While VS Code's Git interface is excellent for daily use, command line Git offers more power and precision. Here's what power users should know:

### Essential Command Line Git

**Repository Setup:**
```bash
# Initialize new repository
git init

# Clone existing repository
git clone https://github.com/username/repo.git

# Add remote origin
git remote add origin https://github.com/username/repo.git
```

**Daily Workflow:**
```bash
# Check status
git status

# Stage files
git add file.txt
git add .                # Stage all changes
git add -A               # Stage all changes including deletions

# Commit changes
git commit -m "Descriptive commit message"
git commit -am "Stage and commit modified files"

# Push/pull changes
git push origin main
git pull origin main
```

**Viewing History:**
```bash
# Show commit history
git log --oneline
git log --graph --oneline --all

# Show changes in files
git diff                 # Working directory vs staging
git diff --staged        # Staging vs last commit
git diff HEAD~1          # Compare with previous commit
```

## Advanced Branching Strategies

### Feature Branch Workflow

```bash
# Create and switch to new branch
git checkout -b feature/user-authentication
# or in newer Git versions:
git switch -c feature/user-authentication

# Work on feature, make commits
git add .
git commit -m "Add login form"
git commit -m "Add password validation"

# Switch back to main
git checkout main
# or: git switch main

# Merge feature branch
git merge feature/user-authentication

# Delete feature branch
git branch -d feature/user-authentication
```

### Why Use Feature Branches?

1. **Isolation:** Work on features without affecting main code
2. **Collaboration:** Multiple people can work on different features
3. **Review:** Feature branches enable pull request reviews
4. **Rollback:** Easy to abandon a feature if it doesn't work out

### Git Flow Model

For larger projects, consider the Git Flow model:

- **main/master:** Production-ready code
- **develop:** Integration branch for features
- **feature/*:** Individual feature development
- **release/*:** Release preparation
- **hotfix/*:** Critical fixes to production

```bash
# Example Git Flow workflow
git checkout develop
git checkout -b feature/data-visualization
# ... work on feature ...
git checkout develop
git merge feature/data-visualization
git branch -d feature/data-visualization
```

## Advanced Git Operations

### Undoing Changes

**Review before undoing working-directory changes:**
```bash
git diff file.txt                  # Review edits first
git restore --staged file.txt     # Unstage, keep edits
```

**Undo staged changes:**
```bash
git reset file.txt                 # Unstage file
git reset                          # Unstage all files
```

**Undo commits:**
```bash
git reset --soft HEAD~1            # Undo last commit, keep changes staged
git reset --mixed HEAD~1           # Undo last commit, unstage changes
```

**Revert published commits:**
```bash
git revert HEAD                    # Create new commit that undoes last commit
git revert abc123                  # Revert specific commit by hash
```

### Interactive Rebase

Clean up commit history before sharing:

```bash
# Rebase last 3 commits interactively
git rebase -i HEAD~3
```

Options in interactive rebase:
- **pick:** Keep commit as-is
- **reword:** Change commit message
- **edit:** Amend commit content
- **squash:** Combine with previous commit
- **drop:** Delete commit entirely

### Stashing Changes

Temporarily save work without committing:

```bash
# Stash current changes
git stash
git stash push -m "Work in progress on user auth"

# List stashes
git stash list

# Apply stash
git stash pop                      # Apply and remove stash
git stash apply                    # Apply but keep stash

# Stash specific files
git stash push -m "Message" file1.txt file2.txt
```

## Git Hooks and Automation

### Pre-commit Hooks

Automate code quality checks:

**.git/hooks/pre-commit** (make executable):
```bash
#!/bin/sh
# Run tests before allowing commit

echo "Running tests..."
python -m pytest tests/

if [ $? -ne 0 ]; then
    echo "Tests failed! Commit aborted."
    exit 1
fi

echo "Running linting..."
flake8 src/

if [ $? -ne 0 ]; then
    echo "Linting failed! Commit aborted."
    exit 1
fi

echo "All checks passed!"
```

### Using pre-commit Framework

Install the pre-commit package:
```bash
pip install pre-commit
```

**.pre-commit-config.yaml:**
```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
-   repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
    -   id: black
        language_version: python3
```

```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## SSH Keys vs HTTPS

### Setting Up SSH Keys

More secure and convenient than HTTPS with passwords:

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@ucsf.edu"

# Add to SSH agent
ssh-add ~/.ssh/id_ed25519

# Copy public key to clipboard (Mac)
pbcopy < ~/.ssh/id_ed25519.pub

# Copy public key to clipboard (Linux)
cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard
```

Add the public key to your GitHub account:
1. GitHub Settings → SSH and GPG keys
2. New SSH key → paste public key
3. Test: `ssh -T git@github.com`

### Convert HTTPS to SSH

```bash
# Check current remote
git remote -v

# Change to SSH
git remote set-url origin git@github.com:username/repo.git
```

## Advanced Collaboration

### Pull Request Best Practices

1. **Small, Focused PRs:** Easier to review and less likely to have conflicts
2. **Descriptive Titles:** Summarize what the PR does
3. **Good Descriptions:** Explain why the change is needed
4. **Link Issues:** Reference related issues with #123
5. **Request Reviewers:** Get feedback before merging

### Handling Merge Conflicts

```bash
# When merge conflicts occur
git status                         # See which files have conflicts

# Edit conflicted files, look for:
<<<<<<< HEAD
Your changes
=======
Their changes
>>>>>>> branch-name

# After resolving conflicts
git add conflicted_file.txt
git commit -m "Resolve merge conflict in conflicted_file.txt"
```

### Advanced Merging Strategies

```bash
# Merge without fast-forward (preserves branch history)
git merge --no-ff feature-branch

# Squash merge (combines all commits into one)
git merge --squash feature-branch
git commit -m "Add feature: description"

# Rebase instead of merge (linear history)
git checkout feature-branch
git rebase main
git checkout main
git merge feature-branch
```

## Git Configuration

### Global Configuration

```bash
# User information
git config --global user.name "Your Name"
git config --global user.email "your.email@ucsf.edu"

# Default editor
git config --global core.editor "code --wait"

# Default branch name
git config --global init.defaultBranch main

# Helpful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
```

### Repository-Specific Configuration

```bash
# Set different email for work projects
git config user.email "work.email@company.com"

# Set up different merge tools
git config merge.tool vimdiff
```

## Git Performance and Large Files

### Speeding Up Git

```bash
# For large repositories
git config core.preloadindex true
git config core.fscache true
git config gc.auto 256

# Shallow clone for huge repositories
git clone --depth 1 https://github.com/user/huge-repo.git
```

### Git LFS (Large File Storage)

For tracking large files (datasets, models, media):

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.csv"
git lfs track "*.pkl"
git lfs track "data/**"

# Add and commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for data files"

# Work normally - LFS handles large files automatically
git add large_dataset.csv
git commit -m "Add training dataset"
```

## Troubleshooting Common Issues

### Detached HEAD State

```bash
# If you're in detached HEAD
git checkout -b temp-branch        # Create branch from current state
git checkout main                  # Switch to main
git merge temp-branch              # Merge your work
git branch -d temp-branch          # Clean up
```

### Accidental Commits

```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Safely undo a published commit with a new inverse commit
git revert HEAD

# Amend last commit message
git commit --amend -m "New commit message"

# Add forgotten files to last commit
git add forgotten_file.txt
git commit --amend --no-edit
```

### Large Repository Issues

For sensitive or oversized files accidentally committed, stop and coordinate
with the repository administrator. History rewriting affects every clone; do
not treat legacy history-rewrite commands as routine recipes. Make a verified
backup and follow the hosting provider's current removal workflow in a
disposable clone.

## When NOT to Use These Advanced Features

- **Don't rebase public branches:** Others might have based work on them
- **Don't force push to shared branches:** it can cause others to lose work.
- **Don't rewrite history casually:** coordinate, back up, and use a disposable clone.
- **Don't overcomplicate:** Simple workflows are often better for small teams

## Resources for Deep Learning

- **Official Git Documentation:** https://git-scm.com/docs
- **Pro Git Book:** https://git-scm.com/book (free online)
- **Interactive Git Tutorial:** https://learngitbranching.js.org/
- **Git Flow Tutorial:** https://github.com/nvie/gitflow
- **Advanced Git Videos:** Search for "Advanced Git" on YouTube

## Practice Exercises

1. Create a feature branch, make commits, and practice different merge strategies
2. Set up pre-commit hooks for a Python project
3. Practice interactive rebase to clean up commit history
4. Set up SSH keys for passwordless Git operations
5. Practice resolving merge conflicts in a safe test repository

Remember: These are power-user features. Master the basics first!

## Professional Git Workflow

Professional Git workflows emphasize clear commit messages, logical change organization, and effective collaboration patterns. These practices ensure project history remains understandable and maintainable.

**Reference:**

- **Commit Messages**: Present tense, descriptive, under 50 characters
- **Atomic Commits**: One logical change per commit
- **Branch Strategy**: Feature branches for development
- **Pull Requests**: Code review before merging
- **Conflict Resolution**: Merge conflicts handled systematically
- **History Management**: Clean, linear history when possible

**Brief Example:**

```bash
# Good commit message format
git commit -m "Add data validation to analysis script

- Validate input file exists before processing
- Check data format matches expected schema
- Add error handling for malformed data

Fixes issue #123"
```

---


# Advanced Python CLI Topics

*Optional reference for students interested in command-line data workflows.*

This page owns shell and CLI-specific extensions. Python function design and
object-model extensions live in the [Python concepts section](#bonus-python-concepts)
below; the core lecture already introduces ordinary functions, lambdas, and the
main guard.

## Command-line essentials

Use these as a reference while practicing in a disposable directory.

### Navigation

```bash
pwd                 # print the current directory
ls                  # list its contents
cd data             # move into data
cd ..               # move up one directory
```

### Files and directories

```bash
mkdir results       # create a directory
touch notes.txt     # create an empty file (or update its timestamp)
cp notes.txt copy.txt
mv copy.txt archive.txt  # rename; use a directory as the destination to move
rm archive.txt      # remove a file; check the path first
```

### Inspecting and searching text

```bash
cat notes.txt           # print a small text file
head -n 5 data.csv      # first five lines
tail -n 5 data.csv      # last five lines
grep -i 'error' log.txt # search, ignoring case
wc -l data.csv          # count lines
```

### Directory trees, history, and shortcuts

```bash
tree .              # show this directory's hierarchy, when tree is installed
history             # list prior commands
```

The ↑ and ↓ keys cycle through earlier commands, `Tab` completes names, and
`Ctrl+R` searches command history.

## Shell scripts with arguments

Shell scripts can turn a repeatable pipeline into a small command-line tool.
Quote paths and validate inputs before processing them.

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s FILE\n' "$0" >&2
    exit 2
fi

input=$1
[[ -f "$input" ]] || { printf 'not a file: %s\n' "$input" >&2; exit 1; }

# grep status 1 means "no matches," which is a valid clean result. Capture the
# matches so statuses above 1 can still stop the script as real failures.
if matches=$(grep -i 'error' "$input"); then
    :
else
    status=$?
    if (( status != 1 )); then
        exit "$status"
    fi
fi
printf '%s' "$matches" | sort | tee errors.txt
```

Useful shell variables include `$1` (the first argument), `$@` (all
arguments), `$#` (argument count), and `$?` (the previous command's status).

## Pipelines, redirection, and process substitution

Pipes connect stdout to stdin. `&&` continues only after success, `||` handles
failure, and explicit redirections make output destinations clear.

```bash
grep -i 'error' logfile.txt | wc -l > error-count.txt
backup_script.sh > backup.log 2>&1
diff <(sort file1.txt) <(sort file2.txt)
```

For larger batches, `find` can safely pass null-delimited paths to a loop:

```bash
find data -name '*.csv' -print0 |
while IFS= read -r -d '' file; do
    printf 'processing %s\n' "$file"
done
```

## Command-line data processing

Small Unix tools are useful for inspection before a Python program takes over:

```bash
cut -d',' -f1,3 data.csv |        # select fields
  tr '[:lower:]' '[:upper:]' |   # normalize case
  sort -t',' -k2,2n |            # order by field 2
  head -n 10 > results.csv
```

`grep`, `cut`, `sort`, `uniq`, `tr`, `sed`, and `awk` each do one focused
transformation. Check quoting and delimiters for the actual input format.

## Calling Python from a shell

The shell is often the orchestrator while Python owns domain logic. Pass input
through arguments or standard input rather than relying on hidden state:

```bash
python3 summarize.py data.csv --output summary.json
python3 -c 'import sys; print(sum(map(float, sys.stdin)))' < values.txt
```

## Further directions

Explore `xargs`, `tee`, process substitution, and shell completion in a
disposable practice directory. For substantial transformations, prefer a
tested Python script so parsing, errors, and edge cases are explicit.

---


# Bonus Python Concepts

*Optional extensions for students who want to explore Python concepts beyond the core lecture.*

## Function design

The core lecture introduces functions and script entry points. These patterns deepen that material without repeating the basic syntax.

### Flexible arguments

```python
def calculate_stats(*numbers):
    """Return simple statistics for any number of numeric arguments."""
    if not numbers:
        return None
    total = sum(numbers)
    return {"sum": total, "average": total / len(numbers), "count": len(numbers)}


def create_profile(**details):
    """Collect arbitrary named fields into a new dictionary."""
    return dict(details)
```

Default parameters are evaluated when the function is defined, so use `None` when a mutable default should be created per call:

```python
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
```

### Documentation and validation

Docstrings describe a function's contract. Keep examples and accepted values aligned with the implementation:

```python
def analyze_data(data, method="mean"):
    """Return the mean or median of a non-empty numeric sequence."""
    if not data:
        raise ValueError("data cannot be empty")
    if method == "mean":
        return sum(data) / len(data)
    if method == "median":
        ordered = sorted(data)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2
    raise ValueError("method must be 'mean' or 'median'")
```

## Conditional expressions

A conditional (ternary) expression is useful for a short, readable choice. Prefer a regular `if` statement when conditions become nested or complex.

```python
age = 25
status = "adult" if age >= 18 else "minor"

temperatures = [15, 25, 35, 5, 45]
categories = [
    "hot" if temp > 30 else "cold" if temp < 10 else "moderate"
    for temp in temperatures
]
```

## Python's object model

In Python, values are objects with a type, identity, and value. `id(value)` exposes an identity token for the lifetime of that object; it is not a promise that the object resides at that numeric memory address.

```python
value = [1, 2]
alias = value
copy = value.copy()
print(value is alias)  # True
print(value is copy)   # False
```

## Mutability and hashability

Mutable objects can change in place; immutable objects cannot. A tuple is immutable, but it is hashable only when all of its elements are hashable. This is why a tuple containing a list cannot be used as a dictionary key.

```python
items = [1, 2]
items.append(3)

point = (1, 2)
lookup = {point: "origin-adjacent"}

unhashable = (1, [2])
# {unhashable: "not allowed"}  # TypeError: list is unhashable
```

## Exception handling patterns

Catch the narrowest expected exception, add context, and let unexpected failures remain visible:

```python
def parse_score(text):
    try:
        score = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid score: {text!r}") from exc
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    return score
```

Custom exceptions can make a library's public failure modes clearer:

```python
class DataValidationError(ValueError):
    """Raised when input data violates an application contract."""


def require_columns(columns, required):
    missing = set(required) - set(columns)
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")
```

## Practice prompts

1. Add keyword-only options to a reusable function.
2. Document and validate a small data-processing function.
3. Find a mutable-default-argument bug and repair it.
4. Define a custom exception for one domain-specific validation rule.
