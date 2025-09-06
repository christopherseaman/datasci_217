# `git`
1. Clone repo
2. Create scratch repo
		1. Create using `git init`
		2. Edit, add, commit
		3. Open in VS Code
		4. View git history
		5. Edit, add
      6. Status
      7. commit
		8. Create remote (VS Code)
	3. Push/pull
	4. Create branch `git checkout -b feature-branch`
3. Git log
4. Checkout
5. Rebase
	1. `git rebase branch-name`
	2. `git rebase -i HEAD~3` 
## Git Conflict Creation and Resolution Guide

### 1. GitHub and Local (VS Code/nano)

#### Creating the Conflict:
1. On GitHub:
   - Edit the README.md file directly on GitHub
   - Add a line: "This line was added on GitHub"
   - Commit the change

2. Locally (VS Code or nano):
   - Without pulling the changes, edit the same README.md file
   - Add a different line: "This line was added locally"
   - Commit the change

3. Try to push the local changes

#### Resolving the Conflict:
1. In VS Code:
   - Pull the changes
   - VS Code will show the conflict in the editor
   - Use VS Code's merge conflict resolution tools to choose which changes to keep
   - Stage the resolved file, commit, and push

2. Using nano (or any text editor):
   - Pull the changes
   - Open the file and manually edit to resolve the conflict
   - Remove conflict markers (<<<<<<<, =======, >>>>>>>)
   - Save the file, stage, commit, and push

### 2. Two Branches on GitHub

#### Creating the Conflict:
1. Create two branches from main: branch-A and branch-B
2. In branch-A:
   - Edit README.md
   - Add a line: "This line was added in branch-A"
   - Commit the change
3. In branch-B:
   - Edit the same part of README.md
   - Add a line: "This line was added in branch-B"
   - Commit the change
4. Create a pull request to merge branch-A into main
5. Create another pull request to merge branch-B into main

#### Resolving the Conflict:
1. GitHub will show a conflict in the second pull request
2. Use GitHub's web interface to resolve the conflict:
   - Click on "Resolve conflicts"
   - Edit the file in the web editor
   - Choose which changes to keep or combine them
   - Mark as resolved, commit the changes
3. Complete the pull request

### 3. Command Line

#### Creating the Conflict:
1. Create and switch to a new branch:
   ```
   git checkout -b feature-branch
   ```
2. Edit README.md, add a line, and commit
3. Switch back to main:
   ```
   git checkout main
   ```
4. Edit the same part of README.md, add a different line, and commit

#### Resolving the Conflict:
1. Try to merge the feature branch:
   ```
   git merge feature-branch
   ```
2. Git will report a conflict
3. Open the file in a text editor
4. Manually resolve the conflict, removing conflict markers
5. Stage the resolved file:
   ```
   git add README.md
   ```
6. Complete the merge:
   ```
   git commit
   ```

Remember to explain each step during the demo, highlighting how conflicts appear in different environments and the tools available for resolution.

# Markdown

Walk through https://github.com/christopherseaman/datasci_217

# Python packages

## Virtual environments
1. Create (shell + vscode, not needed on codespaces)
2. Activate
3. `which python3`
4. `pip install`
5. Deactivate
6. `pip install` (will fail on modern systems)
## Package demo
`pip install plotly pandas jupyter`
```
# NOTE: Requires Jupyter Notebook & Jupyter Renderers extensions
# to be installed when run using VS Code

import plotly.express as px
import pandas as pd

# Load a sample dataset
df = px.data.gapminder()

# Create an animated scatter plot
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])

# Show the plot
fig.show()
```
## `.gitignore`
