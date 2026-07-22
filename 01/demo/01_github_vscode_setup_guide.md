# Part 1: GitHub Account Creation

## Browser Demo Steps:
1. **Navigate to github.com**
   - "Let's start by creating your professional presence in tech"
   - "GitHub is where all code lives - from Linux to NASA's Mars rover"

2. **Show the Sign Up process**
   - Click "Sign up" button
   - **Username selection:**
     - "Choose professionally - employers will see this"
     - "Good: alice-smith, asmith2024"
     - "Avoid: coolgamer123, random numbers"
   - **Email:**
     - "Use your UCSF email or personal - you can add both"
     - "You can hide your email later (I'll show you)"

3. **Email Privacy Settings (Important!)**
   - Navigate to: Settings → Emails
   - Show the "Keep my email addresses private" checkbox
   - **Show the noreply email:**
     - "GitHub gives you a proxy email"
     - "Mine is: 86775+christopherseaman@users.noreply.github.com"
     - "Use this for Git configuration - never expose your real email"

4. **GitHub Student Developer Pack (Optional but Valuable!)**
   - Navigate to: https://education.github.com/students
   - "Free access to dozens of developer tools"
   - "GitHub Pro features (unlimited private repos)"
   - "Free domain name, cloud credits, and more"
   - **How to apply:**
     - Click "Get your pack"
     - Sign in with your GitHub account
     - Verify with your .edu email
     - "Takes 1-3 days for approval"
   - "Not required for class, but great professional resources"

5. **Quick tour of GitHub:**
   - Show your profile
   - Show a popular repository (e.g., microsoft/vscode)
   - "This is where we'll save your homework"
   - "Version control = never lose work again"

## Key Talking Points:
- "GitHub is your coding portfolio"
- "Every commit is saved forever"
- "Employers look at GitHub profiles"
- "We'll use GitHub Classroom for assignments"

---

# Part 2: VS Code Setup & Configuration

## VS Code Demo Steps:

1. **Open VS Code**
   - "This is your new home for writing code"
   - "Free, powerful, what professionals use"

2. **Show the Interface:**
   - **Explorer** (Ctrl/Cmd+Shift+E): "Your file tree"
   - **Search** (Ctrl/Cmd+Shift+F): "Find anything in your project"
   - **Source Control** (Ctrl/Cmd+Shift+G): "Git integration built-in"
   - **Extensions** (Ctrl/Cmd+Shift+X): "Superpowers for your editor"

3. **Essential Extensions to Install (live):**
   ```
   Python (by Microsoft)
   - "IntelliSense, debugging, everything Python"
   
   GitLens (optional but helpful)
   - "See who wrote what code and when"
   
   Rainbow CSV (for data science)
   - "Makes CSV files actually readable"
   ```

4. **Configure Git in VS Code:**
   - Open Terminal in VS Code (Ctrl/Cmd+`)
   - "VS Code has a built-in terminal!"
   
   ```bash
   # Configure Git (students follow along)
   git config --global user.name "Your Name"
   git config --global user.email "12345+yourusername@users.noreply.github.com"
   ```
   
   - "Remember: use the noreply email from GitHub!"
   - "This links your commits to your GitHub profile"

5. **Create a Test File:**
   - File → New File → Save as `hello.py`
   ```python
   print("Hello, Data Science!")
   ```
   - "Run it with the play button or right-click → Run Python File"
   - Show output in terminal

---

# Part 3: Quick Integration Test

## Verify Everything Works:
1. **In VS Code Terminal:**
   ```bash
   # Check Python
   python3 --version
   
   # Check Git  
   git --version
   
   # Check current directory
   pwd
   ```

2. **Create a Practice Repository:**
   - "Let's create your first repo together"
   - In VS Code: View → Source Control
   - Initialize Repository
   - "This creates a local Git repository"
   - "Next week: we'll push this to GitHub"

---

# Common Issues & Solutions

## If Git isn't configured:
- "No problem, let's do it together"
- Walk through git config commands again

## If Python isn't found:
- "Let's check your Python installation"
- Try `python` vs `python3`
- "We'll fix this in office hours if needed"

## If VS Code won't open:
- "You can use any editor for now"
- "But VS Code is what we'll support in class"

---

# Student Engagement Points

**After GitHub setup:**
"Raise your hand when you see your noreply email"

**After VS Code extensions:**
"Who sees the Python extension installed? Look for the Python logo in the sidebar"

**After Git config:**
"Type 'git config --list' - do you see your name and email?"

---

# Key Messages

1. **GitHub = Your professional identity**
   - "Start building your portfolio from day 1"
   
2. **VS Code = Your development environment**
   - "Same tool used at Microsoft, Google, startups"
   
3. **Privacy matters**
   - "Never put your real email in public commits"
   
4. **It's okay to struggle with setup**
   - "Everyone goes through this"
   - "Once it's set up, it just works"

---

# Transition to Demo 2

"Now that we have our tools ready, let's see how the command line and Python work together..."

→ Move to `02_cli_navigation_demo.py`