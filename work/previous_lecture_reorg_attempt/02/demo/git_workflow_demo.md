# LIVE DEMO: Git Workflow with VS Code

*This guide is for the instructor to follow during the live demonstration. Students can refer to this afterward.*

## Demo Setup (Pre-class)

1. **Instructor GitHub account** ready
2. **VS Code** open and ready
3. **Screen sharing** configured for clear visibility
4. **Demo repository** name ready: `datasci217-demo-week02`

## Part 1: Creating Repository on GitHub (5 minutes)

### Show the GitHub Web Interface

**Narration:** "Let's create a new repository together. I'm going to github.com and clicking the green 'New' button..."

**Steps:**
1. Navigate to github.com
2. Click "New" button (or + icon → New repository)
3. **Repository name:** `datasci217-demo-week02`
4. **Description:** "Live demo for DataSci 217 - Git workflow with VS Code"
5. **Public repository** (check the box)
6. **Initialize with README** ✓
7. Click "Create repository"

**Key Points to Emphasize:**
- "Public repositories are great for building your professional portfolio"
- "The README file gives us something to start with"
- "Notice the HTTPS clone URL - we'll use this in VS Code"

## Part 2: Cloning with VS Code (8 minutes)

### Demonstrate VS Code Git Integration

**Narration:** "Now let's get this repository onto our computer using VS Code..."

**Steps:**
1. Open VS Code
2. **Command Palette:** Ctrl+Shift+P (or Cmd+Shift+P on Mac)
3. Type: "Git: Clone"
4. **Paste the HTTPS URL** from GitHub
5. **Choose location** (Desktop or Documents folder)
6. **Open in VS Code** when prompted

**Show the Interface:**
- Point out the **Source Control** icon (third icon in sidebar)
- Show the **file explorer** with the README.md
- Open README.md to show current content

**Key Points to Emphasize:**
- "VS Code automatically detected this is a Git repository"
- "The Source Control panel shows us the status of our repository"
- "We can see we're on the 'main' branch at the bottom"

## Part 3: Making Our First Changes (10 minutes)

### Create Project Structure

**Narration:** "Let's set up a proper data science project structure..."

**Create Files Step by Step:**

1. **Create .gitignore file:**
   - Right-click → New File → ".gitignore"
   - Add basic content:
   ```
   __pycache__/
   *.pyc
   .DS_Store
   data/*.csv
   ```

2. **Create simple Python file:**
   - Create folder: "src"  
   - Create file: "src/hello_analysis.py"
   - Add content:
   ```python
   def analyze_data():
       print("Hello from DataSci 217!")
       numbers = [1, 2, 3, 4, 5]
       print(f"Sum: {sum(numbers)}")
       print(f"Average: {sum(numbers)/len(numbers)}")

   if __name__ == "__main__":
       analyze_data()
   ```

3. **Update README.md:**
   - Replace content with:
   ```markdown
   # DataSci 217 - Week 02 Demo
   
   This is our live demo repository for learning Git workflow.
   
   ## Files
   - `src/hello_analysis.py` - Simple analysis script
   - `.gitignore` - Files Git should ignore
   
   ## Usage
   ```
   python src/hello_analysis.py
   ```
   
   ## Author
   DataSci 217 Class Demo
   ```

### Show the Source Control Changes

**Narration:** "Look what happened in our Source Control panel..."

**Point Out:**
- Files marked with "U" (untracked)
- Number badge showing how many files changed
- Diff view when clicking on files

## Part 4: The Git Workflow - Stage, Commit, Push (12 minutes)

### Demonstrate Staging

**Narration:** "Now we'll stage our changes. Think of this as putting items in a shopping cart before checkout..."

**Steps:**
1. **Click the Source Control icon**
2. **Show the Changes section** with unstaged files
3. **Stage files one by one:**
   - Click "+" next to .gitignore
   - Click "+" next to README.md  
   - Click "+" next to hello_analysis.py
4. **Show Staged Changes section**

**Alternative:** Show staging all at once with the "+" next to "Changes"

### Demonstrate Committing

**Narration:** "Now we commit - this saves our changes to the Git history..."

**Steps:**
1. **Type commit message** in the text box: "Add initial project structure and analysis script"
2. **Click the checkmark** ✓ to commit
3. **Show the result:** staged changes disappear, clean working directory

**Key Points to Emphasize:**
- "Good commit messages describe what you accomplished"
- "Notice the source control panel is now clean"
- "Our changes are saved locally, but not yet on GitHub"

### Demonstrate Pushing

**Narration:** "Finally, let's push our changes to GitHub to share them..."

**Steps:**
1. **Click "Sync Changes"** (or the cloud icon)
2. **Show the push happening**
3. **Switch to GitHub in browser**
4. **Refresh the page** to show the changes

**Wow Moment:** "Look! Our changes are now on GitHub for the world to see!"

## Part 5: Making More Changes - The Continuous Workflow (8 minutes)

### Show Daily Workflow

**Narration:** "Let's simulate daily work - making changes, committing, pushing..."

**Make Changes:**
1. **Edit hello_analysis.py** to add a new function:
   ```python
   def calculate_stats(numbers):
       return {
           'count': len(numbers),
           'sum': sum(numbers),
           'mean': sum(numbers) / len(numbers),
           'min': min(numbers),
           'max': max(numbers)
       }
   ```

2. **Update the main function** to use it:
   ```python
   def analyze_data():
       print("Hello from DataSci 217!")
       numbers = [1, 2, 3, 4, 5, 10, 15, 20]
       stats = calculate_stats(numbers)
       
       print("Statistical Analysis:")
       for key, value in stats.items():
           print(f"{key}: {value}")
   ```

### Show the Diff View

**Narration:** "VS Code shows us exactly what changed..."

**Steps:**
1. **Click on the modified file** in Source Control
2. **Show the side-by-side diff:**
   - Green lines = additions
   - Red lines = deletions
   - White lines = unchanged

**Key Points:**
- "This diff view is invaluable for reviewing your work"
- "You can see exactly what you're about to commit"

### Complete the Workflow

**Steps:**
1. **Stage the changes**
2. **Commit with message:** "Add statistical analysis function"
3. **Push to GitHub**
4. **Show the updated code on GitHub**

## Part 6: Creating a .gitignore and Requirements File (5 minutes)

### Demonstrate Project Best Practices

**Narration:** "Let's add professional project setup..."

**Create requirements.txt:**
```
# Basic packages for data analysis
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
```

**Improve .gitignore:**
```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Virtual environments  
venv/
env/
.venv/

# Data files
*.csv
*.xlsx
data/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/settings.json
.idea/
```

**Commit these changes:** "Add requirements and improve gitignore"

## Part 7: Q&A and Common Issues (7 minutes)

### Address Common Questions

**"What if I make a mistake?"**
- Show how to unstage files (click "-" next to staged file)
- Mention that Git keeps history - very hard to lose work

**"What if VS Code doesn't show Git options?"**
- Check that folder is a Git repository (look for .git folder)
- Make sure Git is installed on the system

**"What's the difference between Git and GitHub?"**
- Git = version control system (local)
- GitHub = hosting service (remote)
- Analogy: Git is like Microsoft Word, GitHub is like Google Drive

### Show Git Status in VS Code

**Point out visual indicators:**
- **Bottom status bar:** branch name, sync status
- **File explorer:** blue dots for modified files
- **Source control badge:** number of changed files

## Wrap-up and Key Takeaways (5 minutes)

### Emphasize the Workflow

**"This is the basic workflow you'll use hundreds of times:"**
1. Make changes to your files
2. Review changes in Source Control
3. Stage the changes you want to commit
4. Write a descriptive commit message
5. Commit to save locally
6. Push to share on GitHub

### Professional Benefits

**"Why this matters for your career:"**
- Every tech company uses Git
- GitHub is your professional portfolio
- Version control saves you from disasters
- Collaboration becomes seamless

### Next Steps

**"For next class:"**
- Practice this workflow with Assignment 02
- Don't worry about memorizing - it becomes automatic
- Focus on good commit messages
- Ask questions in Discord if you get stuck

## Demo Repository Cleanup (After Class)

1. **Archive the demo repository** or delete it
2. **Save any useful examples** for future classes
3. **Note any questions** that came up for next year's improvements

## Technical Notes for Instructor

### Backup Plans

- **VS Code issues:** Have GitHub Desktop ready as alternative
- **Network issues:** Have offline demo repository prepared
- **Permission issues:** Test clone location beforehand

### Visual Presentation

- **Zoom in** on VS Code interface (Ctrl/Cmd + Plus)
- **Use large fonts** in terminal/editor
- **Highlight mouse cursor** for better visibility
- **Speak out loud** what you're clicking

### Timing Flexibility

- **Core content:** Parts 1-4 (essential workflow)
- **Nice to have:** Parts 5-6 (if time permits)
- **Buffer time:** Part 7 (can be shortened if needed)

This demo gives students confidence that Git isn't scary and shows them the exact workflow they'll use daily!