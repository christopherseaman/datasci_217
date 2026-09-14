# GitHub, VS Code, and WSL Setup

Use this checklist to prepare the tools used in Lecture 01. The commands are
safe to repeat, except where GitHub asks you to choose an account name.

## 1. GitHub account and email privacy

1. Open [github.com](https://github.com/) and create or open your account.
2. Choose a professional username; it will be part of your public portfolio.
3. Open [GitHub email settings](https://github.com/settings/emails).
4. Enable **Keep my email addresses private** and copy your GitHub `noreply`
   address. Use that address in Git configuration instead of your personal
   email.
5. The [GitHub Student Developer Pack](https://education.github.com/students)
   is optional.

## 2. Choose a shell

The course shell examples use Bash or a compatible POSIX shell.

- macOS: open Terminal.
- Linux: open your terminal.
- Windows: install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
  from Administrator PowerShell with `wsl --install`, restart if prompted,
  and open the installed Ubuntu terminal.
- Any platform: GitHub Codespaces is an optional browser-based alternative.

In the shell, check that the basic commands are available:

```bash
pwd
ls
python3 --version
git --version
```

Native Windows PowerShell uses different command names and syntax. Use WSL for
the Bash examples in this course, or translate each command deliberately.

## 3. Install and open VS Code

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Install the **Python** extension by Microsoft.
3. Optional extensions: GitLens and Rainbow CSV.
4. Open a course or practice folder in VS Code.
5. Open the integrated terminal with **View → Terminal**.

Useful interface areas are Explorer, Search, Source Control, Run and Debug,
and Extensions. The keyboard shortcuts vary by operating system.

## 4. Configure Git in the VS Code terminal

Replace the placeholders with your own name and GitHub `noreply` address:

```bash
git config --global user.name "Your Name"
git config --global user.email "12345+yourusername@users.noreply.github.com"
git config --list --global
```

Confirm that the displayed email is the privacy-preserving GitHub address.

## 5. Run a first Python file

Create `hello.py` in VS Code:

```python
print("Hello, Data Science!")
```

Run it with the play button, **Run Python File**, or the terminal:

```bash
python3 hello.py
```

Next, initialize a disposable practice repository from the VS Code terminal:

```bash
mkdir lecture01-practice
cd lecture01-practice
git init
printf 'print("Hello, Git!")\n' > hello.py
git status
```

Keep the practice repository separate from assignment repositories.
