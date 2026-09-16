# GitHub, VS Code, and WSL Setup

Use this checklist to prepare the tools used in Lecture 01. The commands are safe to repeat, except where GitHub asks you to choose an account name.

## 1. GitHub account and email privacy

1. Open [github.com](https://github.com/) and create or open your account.
2. Choose a professional username; it will be part of your public portfolio.
3. Open [GitHub email settings](https://github.com/settings/emails).
4. Enable **Keep my email addresses private** and copy your GitHub `noreply`
   address. Use that address in Git configuration instead of your personal
   email.
5. The [GitHub Student Developer Pack](https://education.github.com/students)
   is optional.

## 2. Install and open VS Code

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Install the **Python** extension by Microsoft.
3. Optional extensions: GitLens and Rainbow CSV.
4. Open a course or practice folder in VS Code.
5. Open the integrated terminal with **Terminal → New Terminal** (`Ctrl+Shift+backtick` on all platforms).

Useful interface areas are Explorer, Search, Source Control, Run and Debug, and Extensions. Open the Command Palette with **View → Command Palette** (`Ctrl+Shift+P` on Windows/Linux; `Cmd+Shift+P` on macOS). Open Source Control with **View → Source Control** (`Ctrl+Shift+G` on all platforms, including macOS).

## 3. Choose a shell

The course shell examples use Bash or a compatible POSIX shell.

- In VS Code, open your project folder, then **Terminal → New Terminal**.
- Windows: install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
  from Administrator PowerShell with `wsl --install`, restart if prompted,
  and choose **Ubuntu (WSL)** as VS Code's default terminal profile.
- Native-terminal alternative: open Terminal on macOS/Linux or Ubuntu on Windows, then `cd` to your project folder.
- Any platform: GitHub Codespaces is an optional browser-based alternative.

In the shell, check that the basic commands are available:

```bash
pwd
ls
python3 --version
git --version
```

Native Windows PowerShell uses different command names and syntax. Use WSL for the Bash examples in this course, or translate each command deliberately.

## 4. Configure Git in the VS Code terminal

Replace the placeholders with your own name and GitHub `noreply` address:

```bash
git config --global user.name "Your Name"
git config --global user.email "12345+yourusername@users.noreply.github.com"
git config --list --global
```

Confirm that the displayed email is the privacy-preserving GitHub address.

## 5. Fork, clone, and save a change

1. Open the assignment repository on GitHub. Select **Fork**, choose your account, and select **Create fork**.

    ![GitHub's Fork button](../assignment/media/github-fork.png)

2. From your fork, copy **Code → HTTPS**. Confirm that the URL contains your username as the owner.

    ![Copy the HTTPS URL from your fork](../assignment/media/github-clone-url.png)

3. Open **View → Command Palette** (`Ctrl+Shift+P` on Windows/Linux; `Cmd+Shift+P` on macOS), choose **Git: Clone**, paste the URL, choose a local folder, and open it.

    ![VS Code's Clone from URL prompt](../assignment/media/vscode-clone.png)

4. In Explorer, create `practice.txt` and write a sentence about what you want to learn.
5. Open **View → Source Control** (`Ctrl+Shift+G` on all platforms, including macOS), review the change, stage it with **+**, enter a commit message, and select **Commit**.
6. Select **Sync Changes**, then check your fork on GitHub to see the file there.

You have a copy on GitHub and a working copy on your computer. Lecture 02 develops the Git concepts behind this workflow.

Screenshots show example repositories; paste your own fork's URL. Sources: [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and [VS Code documentation](https://code.visualstudio.com/docs/sourcecontrol/quickstart).
