Version Control and Project Setup

Today we're learning the most important skill for collaborative data science: version control with Git and GitHub. Think of Git as "track changes" for your entire project, but much more powerful and designed for team collaboration.

By the end of today, you'll know how to save your work, share it with others, and never lose progress again. Plus, you'll have a professional portfolio on GitHub!

*[xkcd 1597: "Git" - "This is git. It tracks collaborative work on projects through a beautiful distributed graph theory tree model." "Cool. How do we use it?" "No idea. Just memorize these shell commands and try not to break anything."]*

Don't worry - we're taking a different approach than that xkcd suggests!

# Why Version Control Matters

## The Problem Without Version Control

Picture this: You're working on a data analysis. You create these files:
- `analysis_v1.py`
- `analysis_v2.py`  
- `analysis_v2_final.py`
- `analysis_v2_final_ACTUALLY_FINAL.py`
- `analysis_fixed_broken_computer_recovery.py`

Sound familiar? Now imagine collaborating with teammates doing the same thing. Chaos!

## The Git Solution

Git tracks every change to every file in your project. You can:
- See exactly what changed and when
- Go back to any previous version
- Work on features in parallel without conflicts
- Collaborate with teammates seamlessly
- Never lose work (it's all backed up on GitHub)

It's like having infinite "undo" for your entire project, plus collaboration superpowers.

# Git Concepts - The Mental Model

## Repository (Repo)
Your project folder that Git tracks. Contains your files plus a hidden `.git` folder with all the version history.

Think: "This entire folder is under Git management."

## Commit  
A saved snapshot of your project at a specific point in time. Like saving a game - you can always come back to this exact state.

Think: "I'm saving my progress with a description of what I accomplished."

## Remote
The version of your repository stored on GitHub (or similar service). Your local computer has a copy, GitHub has a copy, your teammates have copies.

Think: "The shared version everyone can access."

## Branch
A parallel timeline for your project. The main branch contains your official version, feature branches contain experimental work.

Think: "I'm trying something new without risking the working version."

*We'll focus on the main branch today - branches come later!*

# GUI-First Git with VS Code

## Why Start with GUI?

Command line Git is powerful, but the visual interface helps you understand what's happening. VS Code's Git integration shows you:
- Which files changed (visual diff)
- What you're about to commit
- The status of everything at a glance

Once you understand Git concepts, you can choose GUI or command line based on the task.

## Setting Up Git in VS Code

**Reference:**
1. Install VS Code (if not already done)
2. Open VS Code → View → Source Control (or Ctrl+Shift+G)
3. If first time: VS Code will prompt to configure Git username/email

**Brief Example:**
```
Git configuration (one-time setup):
- Full Name: Alice Smith
- Email: alice.smith@ucsf.edu (use your actual UCSF email)
```

# LIVE DEMO!
*Setting up Git configuration and exploring VS Code's Git interface*

# GitHub Account Setup

## Creating Your GitHub Account

**Reference:**
1. Go to github.com
2. Sign up with your UCSF email (or personal email)
3. Choose a professional username (you'll use this for years!)
4. Verify your email address

**Username Tips:**
- Use your name or initials: `alice-smith`, `asmith-ucsf`
- Avoid hard-to-remember numbers: `alice_smith_9847`
- Keep it professional - future employers will see this
- You can change it later, but links might break

## GitHub Student Pack (Optional Bonus)
With your .edu email, you can get free premium features. We don't need them for class, but they're nice to have!

# Your First Repository

## Creating a Repository on GitHub

**Reference:**
1. Log into GitHub
2. Click green "New" button (or + icon → New repository)
3. Repository name: `datasci-practice` (or similar)
4. Description: "Practice repository for DataSci 217"
5. Choose Public (shows your work to the world)
6. Check "Add a README file"
7. Click "Create repository"

**Brief Example:**
```
Repository settings:
- Name: datasci-practice
- Description: Practice repository for DataSci 217  
- Public ✓ (showcase your work!)
- Initialize with README ✓
```

## Cloning to Your Computer

**Reference (VS Code method):**
1. On GitHub: Click green "Code" button → copy HTTPS URL
2. VS Code: Command Palette (Ctrl+Shift+P) → "Git: Clone"
3. Paste URL, choose location on your computer
4. VS Code opens the cloned repository

**Brief Example:**
```
Clone URL: https://github.com/yourusername/datasci-practice.git
Local location: Choose a folder you'll remember (Desktop, Documents, etc.)
```

# LIVE DEMO!
*Creating repository on GitHub and cloning to VS Code*

# Basic Git Workflow in VS Code

## The Three-Stage Workflow

1. **Working Directory**: Your files as you edit them
2. **Staging Area**: Files you've marked "ready to commit"  
3. **Repository**: Committed (saved) snapshots

Think of staging as "putting items in your shopping cart" and committing as "buying everything in the cart."

## Making Your First Changes

**Reference:**
1. Edit files in VS Code (create new files, modify existing ones)
2. VS Code Source Control shows changed files with "M" (modified) or "U" (untracked)
3. Click "+" next to files to stage them
4. Type commit message in text box
5. Click checkmark to commit
6. Click "Sync Changes" to push to GitHub

**Brief Example Workflow:**
```
1. Create new file: practice_analysis.py
2. Add some Python code
3. Stage file: Click + in Source Control
4. Commit message: "Add first analysis script"
5. Commit: Click checkmark ✓
6. Push: Click "Sync Changes"
```

# Essential Git Operations

## Staging and Committing

**What to stage:**
- Complete logical units of work
- Files that work together
- Related changes

**Good commit messages:**
- "Add data loading function"
- "Fix bug in calculation logic"  
- "Update analysis with new dataset"

**Avoid:**
- "Update" (what did you update?)
- "Fix" (fix what?)
- "asdkfjasldkf" (just... no)

## Viewing Changes (Diff)

**Reference:**
- Click on modified file in Source Control → see side-by-side diff
- Green lines: additions
- Red lines: deletions  
- White lines: unchanged

This visual diff is invaluable for reviewing your work before committing!

## Git Status in VS Code

**Visual indicators:**
- Blue dot: file modified
- Green U: new file (untracked)
- Red trash icon: file deleted
- Number badge: how many files changed

Bottom status bar shows:
- Current branch name
- Sync status (ahead/behind remote)

# LIVE DEMO!
*Complete workflow: make changes, stage, commit, push to GitHub*

# Project Environment Management

## Why Virtual Environments?

Python projects need specific versions of packages. Different projects need different packages. Virtual environments create isolated Python installations for each project.

Think: "Each project gets its own Python toolkit, separate from others."

## Creating Virtual Environments

**Reference (using conda):**
```bash
conda create -n datasci-practice python=3.11
conda activate datasci-practice
```

**Reference (using venv):**
```bash
python -m venv datasci-practice
# Mac/Linux: source datasci-practice/bin/activate  
# Windows: datasci-practice\Scripts\activate
```

**Brief Example:**
```bash
# Create environment
conda create -n datasci-practice python=3.11

# Activate environment  
conda activate datasci-practice

# Install packages
conda install pandas numpy matplotlib

# Deactivate when done
conda deactivate
```

## Package Installation Basics

**Reference:**
```bash
# With conda (preferred for data science)
conda install pandas numpy matplotlib

# With pip  
pip install pandas numpy matplotlib

# Install from requirements file
pip install -r requirements.txt
```

The key insight: Always work in virtual environments for projects!

# Integrating Git and Environments

## .gitignore File

Some files shouldn't be tracked by Git:
- Virtual environment folders
- Data files (often too large)
- Temporary files  
- Secret keys/passwords

**Reference (.gitignore contents):**
```
# Virtual environments
venv/
env/
datasci-practice/

# Data files (often too large for Git)
*.csv
data/

# Python generated files
__pycache__/
*.pyc

# Jupyter notebook outputs
.ipynb_checkpoints/

# Operating system files
.DS_Store
Thumbs.db
```

## requirements.txt File

Track which packages your project needs:

**Reference:**
```bash
# Create requirements file
pip freeze > requirements.txt

# Install from requirements file (on new computer/teammate)
pip install -r requirements.txt
```

This lets anyone recreate your exact environment!

# Collaboration Basics

## README.md Files

Every repository should have a README explaining:
- What the project does
- How to set it up  
- How to run it
- Who to contact for questions

**Brief Example:**
```markdown
# My Data Science Practice

This repository contains practice exercises for DataSci 217.

## Setup
1. Clone this repository
2. Create virtual environment: `conda create -n practice python=3.11`
3. Activate environment: `conda activate practice`
4. Install packages: `pip install -r requirements.txt`

## Usage
Run analysis: `python analysis.py`

## Contact  
Alice Smith - alice.smith@ucsf.edu
```

Markdown is just formatted text - we'll learn more next week!

# Key Takeaways

1. **Git tracks every change** to your project over time
2. **GitHub stores your work** in the cloud and enables collaboration  
3. **VS Code's GUI** makes Git visual and approachable
4. **Virtual environments** keep projects separate and reproducible
5. **Good habits early** save massive headaches later

You now have professional-grade project management skills. Every change you make is tracked, your work is backed up, and you can collaborate with teammates seamlessly.

Next week: We'll dive deeper into Python data structures and file operations!

# Practice Challenge

Before next class:
1. Create a second repository called `week-02-practice`
2. Add a Python file with a simple calculation
3. Create a requirements.txt file (even if empty)
4. Write a descriptive README.md
5. Make several commits with good commit messages
6. Check that everything appears correctly on GitHub

Remember: This workflow becomes automatic with practice. Professional data scientists do this hundreds of times per week!

*[xkcd 1296: "Git Commit" - Shows a humorous flowchart of git commit decision-making]*