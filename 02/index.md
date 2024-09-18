---
marp: true
theme: gaia
paginate: true
---

# Lecture 02
- `git` and GitHub
- Markdown
- Python environments and packages
---
## Offline discussion
- CLE?
- [Discord?](https://discord.gg/QdTYgR45er)

---
## Notes from last lecture

- `wsl --install`
- Open `Ubuntu` from Start
- Files from Windows available at `/mnt/c/Users/<user name>`

---
## Random links for the day

- [Data scientists work alone and that's bad | Ethan Rosenthal](https://www.ethanrosenthal.com/2023/01/10/data-scientists-alone)
- [I Used Computer Vision To Destroy My Childhood High Score in a DS Game](https://betterprogramming.pub/using-computer-vision-to-destroy-my-childhood-high-score-in-a-ds-game-38ebd53a1d64) 

---
## Resources for `git`, `markdown`, and `pip`

* `git`
	* GitHub's [Git Guides](https://github.com/git-guides), part of which is included in this week's assignment
	* Atlassian has an [excellent tutorial on git](https://www.atlassian.com/git/tutorials/what-is-version-control), especially _Getting Started_ and _Collaborating_ 
* Markdown
	- [https://www.markdownguide.org/basic-syntax/](https://www.markdownguide.org/basic-syntax/) (cheat sheet)
	- https://commonmark.org/help/tutorial (self-guided tutorial)
	- [https://www.markdowntutorial.com](https://www.markdowntutorial.com/) (self-guided tutorial) 
- `python` + packages
	- _Whirlwind Tour of Python_, VanderPlas - author’s [website](https://jakevdp.github.io/WhirlwindTourOfPython/)

---
## Configuring with your name and email

We need to tell git who we are. We do this using `git config` to

`git config --global user.name "<YOUR NAME>"`

`git config --global user.email "<YOUR EMAIL>"` to set your email address

---

## Important note!
Having your email address listed in a public repository is a **bad idea**. You will get targeted for spam or worse. GitHub will set up an anonymous proxy email address automatically, you can find it https://github.com/settings/emails while logged in.  

---

![bg contain](media/github_email.png)

---
## Cloud options

We’ll work with GitHub here, but other options include [GitLab](https://gitlab.com/) and [Bitbucket](https://bitbucket.org/). UCSF also has an [internally-facing version of GitHub](https://it.ucsf.edu/search?search=github), which you should definitely use if you’re working on anything PHI-related. Access to [UCSF’s GitHub](https://git.ucsf.edu) and High-Performance Computing (Wynton) must be requested from IT.

---
## `git init` and `git clone`

You can create a new repository in the current directory with the `git init` command. This adds a hidden folder `.git` with the configuration for the repo. You can later add the repo to a remote host (like GitHub) where others can access it

To copy a remote repository to your local machine use the `git clone` command. This will copy the repository and all its version history to a subdirectory with the same name as the repository you’ve cloned.

---
To clone the notebooks that accompany the book Python for Data Analysis:

`git clone` `[https://github.com/wesm/pydata-book.git](https://github.com/wesm/pydata-book.git)`
![](media/git_clone.png)

---
## Commit

You save a snapshot of your work using `git commit` commands. Each commit will need a short message to describe the changes you’ve made.

1. `git status` - to see what changes you’ve made since the last commit
2. `git add <FILE>` - add files you’ve changed to _staging_ (included in the next commit)
3. `git commit -m <MESSAGE>` - commit your work with a quick summary
---
## Push ⇄ pull

If you have a `remote` set up, it does what it says on the tin. Cloning from GitHub always adds GitHub as a `remote`

- `git push` will add your local commits to the remote copy
- `git pull` will download any changes from the remote to your local copy
- `git sync` (not all systems) will perform both a pull and push

---
## Push ⇄ pull 

From the command line, GitHub no longer allows password-based pushes, so you will need to either:
1. Use the [GitHub CLI](https://cli.github.com/)
3. Set up a [Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) or [ssh key](https://github.com/settings/keys)
4. Use a GUI tool, e.g.,  VS Code or GitHub Desktop

---
## Fork, branch, and merge

Sometimes you want to work on something outside the “main” flow of a repository. Maybe there’s an analysis or model you’re working on that isn’t complete. By creating a separate **branch** of the repository, you can do your work without worrying about breaking the **main** or **trunk** branch of the repository. If you want to make your own work based on another repository, you can **fork** it, creating a copy that you own going forward.

---
## Git Checkout
`git checkout` is used to switch between branches, restore working tree files, or examine specific commits.

  ```
  # Switch to an existing branch
  git checkout <branch-name>
  
  # Create and switch to a new branch:
  git checkout -b <new-branch-name>

  # Discard changes to a file
  git checkout -- <file-name>
```

---
## If I could turn back time...

You can view the commit history of your current branch using `git log` and see each commit's message and hash identifier.
![](Pasted%20image%2020240918090552.png)

---
## If I could turn back time...

```
  # Examine a specific commit
  git checkout <commit-hash>
  ```
  ![](Pasted%20image%2020240918090702.png)
On most systems you can use a "short hash" with the first 7 characters:
![](Pasted%20image%2020240918090806.png)

---
## Git Reset

`git reset` allows you to undo changes in your repository

- `git reset --soft <commit>` - Moves HEAD to `<commit>`, changes remain staged
- `git reset --hard <commit>`- Moves HEAD to `<commit>`, discards all changes **(be careful!)**

Note! You can use `HEAD~N` to step `N` commits backwards

---
## Git Reset Examples

```
# Moves HEAD to specified commit, unstages changes
git reset <commit>

# Undo the last commit, keep changes staged
git reset --soft HEAD~1

# Undo the last two commits, discard all changes
git reset --hard HEAD~2
```

---
## Choosing the Right Git Operation
- `git merge` - Integrating completed features into the main branch
	- **Use case:** Combining work from different branches
	- **Best for:** Preserving full history of parallel development
- `git reset` - Undoing recent commits or changes
	- **Use case:** Correcting mistakes, reorganizing work
	- **Best for:** Local changes that haven't been pushed
- `git rebase` (advanced) - Cleaning up commit history before sharing
	- **Use case:** Creating a linear project history
	- **Best for:** Local branches before merging to main

---
## Best practices

- Always pull before starting new work
- When in doubt, create a backup branch first

---
## Branch workflow

It makes it a lot easier when collaborating with others to keep a clean and functional `main` branch. If you’re an imperfect human, then you probably can’t ensure that every commit you make along the way is also clean and functional. Mistakes happen.

One solution is to use a **branch workflow**, where work-in-progress happens in dedicated branches. Once a piece of work is deemed complete, it can be merged into the main branch.

---
## Branch workflow II

A best practice when merging work into the main branch is to use a **pull request** or **PR**. This is the only way to merge branches on GitHub.

A pull request signals that your work may be complete and you’d like someone else to review it and give feedback. This ensures not just that the changes you’ve made are correct, but that they are understandable to others. Once the reviewer gives the 👍 (and conflicts are resolved), your development branch can be merged into the main branch.

We will walk through the process together in a few minutes...

---

![bg contain](media/git_branches.png)

---
## Sensitive information

Never ever. Not once. This goes for passwords, PII, and PHI. Don’t put it on GitHub.

We will see ways to store sensitive information using environment variables, secrets,  and `.env` files in a future lecture. 

---

## Getting into conflict

Sometimes your repos will get into states that can’t be resolved automatically and non-destructively. Generally, this occurs when the local and remote both have changes to the same file.

---

## Conflict resolution

There are a few ways to resolve git conflicts:

- `git restore <FILE>` - discard all changes to `<FILE>` since the last commit
- `git rebase` - discard all changes
- `git stash` - save the current state in a local “stash”, then rebase the repo to the last commit

Read more on merge conflicts in [Atlassian’s tutorial](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts)

---

![bg contain](media/xkcd_git.png)

---
## Best Practices

- Communicate with your team about merge conflicts
- Pull changes frequently to minimize large conflicts
- Always pull before starting new work
- When in doubt, create a backup branch first
- Use feature branches to isolate work
---
## Conflict resolution: command line

1. Pull the latest changes: `git pull origin main`
2. If conflicts occur, Git will notify you
3. Open the conflicting file(s) in a text editor
4. Look for conflict markers: `<<<<<<<`, `=======`, `>>>>>>>`
5. Manually edit the file to resolve the conflict
6. Save the file
7. Stage the resolved file: `git add <filename>`
8. Commit the changes: `git commit -m "Resolve merge conflict"`
9. Push the changes: `git push origin main`

---

## Conflict resolution: VS Code

1. Pull changes in VS Code's Source Control panel
2. VS Code will highlight conflicts in the editor
3. Click on "Accept Current Change", "Accept Incoming Change", "Accept Both Changes", or manually edit
4. After resolving all conflicts, stage the changes
5. Commit and push the resolved conflicts

---

## Conflict resolution: GitHub web interface

1. When a pull request has conflicts, GitHub will notify you
2. Click on "Resolve conflicts" button
3. GitHub's web editor will show the conflicts
4. Manually edit the file to resolve conflicts
5. Click "Mark as resolved" for each file
6. Commit the changes
7. Complete the merge
---

## LIVE DEMO

---
## Markdown

Markdown is a lightweight markup language for writing documents. The format was created as an alternative to HTML, while retaining most of the capabilities. It’s the most common format in many tools, including GitHub, Notion, and Google Docs (when enabled).

---
## Paragraphs

Start a new paragraph by separating it from the previous one with a blank line

```markdown
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 

This is a new paragraph!
```

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

This is a new paragraph!

---
## Headers

Starting a line with hash symbols `#` will creating headings within your document:

## `# Header 1 (biggest)`
### `## Header 2`
#### `### Header 3 (smallest commonly supported)`

---
## Font Styles

There is some variability in how these are applied between Slack, Notion, GitHub, etc.

- **Bold** - double-asterisks `**` around a word to **`**bold**`** (some apps allow single astersisks)
- _italic_ - single underscores `_` around a word to _`_italicize_`_ (some apps confusingly use single asterisks for italic)
- Blockquote - prefix text with a greater-than sign to `>` blockquote (Notion uses the pipe | symbol). To blockquote multiple paragraphs, include a ‘>’ on the blank line between them

```markdown
> This is Blockquoted
> and it continues...
```

> This is Blockquoted
> and it continues...

---
## Lists - unordered

**Unordered lists** start with an asterisks, hyphen, or plus sign. Indent with additional spaces to make sublists:

```markdown
* one
  * two
* three
```

- one
    - two
- three

---
## Lists - ordered

**Ordered lists** start with numbers and indent similarly to undordered lists, but it actually doesn’t matter which digits you use.

```markdown
1. asdf
  3. jfjf
7. btbk
```

1. asdf
    1. jfjf
2. btbk

---
## Lists - checklists

**Checklists** start with bracket pairs `[ ]`, completed items with an x inside `[x]`. **NOTE:** VS Code needs extensions to support completed items within a checklist.

```markdown
[] do this
[x] this is done
```

- [ ] do this
- [x] this is done

---
## Code

Code is marked with surrounding backticks `` `code goes here` `` and can be included inside a paragraph  Larger blocks of code, spanning multiple lines begin and end with three backticks

````JSON
```
This is a large block of code

across multiple lines
```
````

```JSON
This is a large block of code

across multiple lines
```

---
## Links

You can create links by surrounding the link text with [] brackets, then the url surrounded by parentheses:

```markdown
Neat collection of [data science notes](badmath.org/datasci)
```

Neat collection of [data science notes](badmath.org/datasci)

---
## Images

Insert images similar to how you insert links

```
# General format
![Alt text](image_url_or_path "Optional title")

# Web-hosted image
![Cute cat](https://example.com/cat.jpg "A cute cat")

# Local image
![My screenshot](screenshots/pandas_install.png "Pandas installation")
```

---
## readme.md

A file names `readme.md` in the root of your repository is treated specially. This is your repository's introduction to the world. Please write something!

You can see how I've used mine to show the syllabus in the [course repository](https://github.com/christopherseaman/datasci_217)

---
## VS Code Extensions

Recommended to get VS Code working best with Markdown:
- [GitHub Markdown Preview](https://marketplace.visualstudio.com/items?itemName=bierner.github-markdown-preview) for better Code/GitHub compatibility
- [Markdown All-in-One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) for convenience features (e.g., continue lists on enter)

---
## LIVE DEMO

---
## Python Virtual Environments

Virtual environments are isolated Python installations that allow you to manage project-specific dependencies without interfering with system-wide packages.

### Creating a virtual environment

```bash
python -m venv myproject_env
```

---
### Activating the environment

On Windows:
```bash
myproject_env\Scripts\activate
```

On macOS and Linux:
```bash
source myproject_env/bin/activate
```

---
### Deactivating the environment

```bash
deactivate
```

---
## Python Packages

Packages are collections of Python modules that extend the language's functionality.

The best code is code you didn't have to write, that is maintained by someone else who is a specialist in the field.

---

![bg contain](media/python_import.webp)

---
## Installing packages with pip

`pip` is the package installer for Python. Use it to install packages from the Python Package Index (PyPI).

```bash
pip install package_name
pip install package_name==1.2.3
```

---
## Requirement.txt

This is a common file format for saving which packages/versions your project works with.

It can be written manually or generated using 
```bash
pip freeze > requirements.txt
```

Install packages listed in a requirements.txt file:
```bash
pip install -r requirements.txt
```

---
## Common Data Science Packages

- `numpy`: Numerical computing
- `pandas` or `polars`: Data manipulation and analysis
- `matplotlib`: Data visualization
- `scikit-learn`: Machine learning basics
- `statsmodels`: Statistical modeling

---
## Importing libraries

More on this in the future...

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
```

---
## Please, `ignore` the mess...

Programming with virtual environments often means installing packages in the same directory as your repository, but you **DON'T** want to commit these files to `git`. They take up extra space and may be specific to your machine/architecture.

The `.gitignore` file tells Git which files or folders to ignore in a project.

---
## Example `.gitignore` for Python:

```gitignore
# Ignore this filename wherever it show up
.DS_Store

# Ignore these folders anywhere
venv/ 

# Ignore file pattern
*.venv

```

---
## Remember to add it!
VS Code and GitHub have helpers to help you add files/folders to

```bash
git add .gitignore
git commit -m "Add .gitignore file"
```


---

## LIVE DEMO

---
## GitHub Classroom

- Accepting assignments creates a fork of the assignment repo
- Complete assignments by committing to your copy of the repo
- I've recently added another file to the assignment, so you may have to sync up with my changes
---

![bg contain](Pasted%20image%2020240918103045.png)

---
## Practical

- Accept the [assignment on GitHub Classroom](https://classroom.github.com/a/Z2sWwnXF)
- Sync with upstream changes if needed
- Read through the Readme.MD describing `git` and GitHub
- Complete the assignment steps listed in `assignment.md`