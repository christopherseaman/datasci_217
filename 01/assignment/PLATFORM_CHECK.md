# Assignment 01: Work in Your Fork

A fork is your copy on GitHub; a clone is the working copy on your computer. Complete and submit the assignment in your fork.

## Fork and open the assignment

1. Open the assignment repository URL and sign in to GitHub.
2. Select **Fork**, choose your own account as the owner, and select **Create fork**.

    ![GitHub's Fork button](media/github-fork.png)

3. On your fork, select **Code → HTTPS** and copy the URL. Check that the owner is your username.

    ![Copy the HTTPS URL from your fork's Code menu](media/github-clone-url.png)

4. In VS Code's Command Palette, run **Git: Clone**, paste your fork's URL, choose a local folder, and open the cloned repository. Sign in to GitHub if prompted.

    ![VS Code's Clone from URL prompt; paste your fork's GitHub URL here](media/vscode-clone.png)

5. Open **Terminal → New Terminal**. Use `pwd` and `ls` to confirm that you are in the folder containing this assignment's `README.md`.

Screenshots show example repositories; use your own fork's URL. Sources: [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and [VS Code documentation](https://code.visualstudio.com/docs/sourcecontrol/quickstart).

## Deliver the completed files

After generating `output/readiness.txt` and passing `python3 check_assignment.py`:

1. Open VS Code's **Source Control** view and review your changes.
2. Stage (**+**) the three student scripts, `terminal-practice/source.txt`, `terminal-practice/path-check.txt`, `output/readiness.txt`, and `output/student_identity.txt`.
3. Enter a descriptive message, such as `Complete Assignment 01`, and select **Commit** on `main`.
4. Select **Sync Changes** to push to your fork.
5. Open your fork on GitHub and confirm that `output/readiness.txt` ends with `Next checkpoint: 5`.

Your fork is the submission; no pull request to the course repository is needed. The optional **Actions** check provides feedback on the committed artifacts. Lecture 02 explains the Git model behind this workflow.
