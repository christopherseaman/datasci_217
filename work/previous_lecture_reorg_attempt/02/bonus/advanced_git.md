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

**Undo working directory changes:**
```bash
git checkout -- file.txt          # Discard changes to file
git checkout .                     # Discard all changes
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
git reset --hard HEAD~1            # Undo last commit, lose changes (DANGEROUS!)
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

# Completely remove last commit (DANGEROUS!)
git reset --hard HEAD~1

# Amend last commit message
git commit --amend -m "New commit message"

# Add forgotten files to last commit
git add forgotten_file.txt
git commit --amend --no-edit
```

### Large Repository Issues

```bash
# Clean up repository
git gc --aggressive --prune=now

# Remove files from Git history (NUCLEAR OPTION)
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch large_file.csv' \
--prune-empty --tag-name-filter cat -- --all

# Force push (after filter-branch)
git push --force --all
```

## When NOT to Use These Advanced Features

- **Don't rebase public branches:** Others might have based work on them
- **Don't force push to shared branches:** Can cause others to lose work  
- **Don't use filter-branch lightly:** Can rewrite entire repository history
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