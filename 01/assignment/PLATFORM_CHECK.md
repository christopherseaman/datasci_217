# Assignment 01 platform checkpoint

This is one exact, guided GUI synchronization checklist whose sole purpose is to deliver Assignment 01. It is required but unassessed. You are not expected to explain repositories, staging, commits, branches, remotes, push, or pull yet; Lecture 02 defines and practices those concepts.

Use GitHub Desktop for this first delivery unless the instructor gives you an equivalent accessibility workflow.

## Open the assigned repository

1. Open the Assignment 01 repository URL supplied by the instructor and sign in with the course GitHub account.
2. Open the repository and confirm that it contains the Assignment 01 files.
3. On that repository page, select **Code → Open with GitHub Desktop**.
4. Approve the browser prompt to open GitHub Desktop.
5. In GitHub Desktop, choose a local folder you can find again and select **Clone**.
6. Select **Repository → Open in Visual Studio Code**.
7. In VS Code, open **Terminal → New Terminal**, run `pwd`, and confirm that the terminal is in the assigned repository before beginning the assignment.

## Deliver the completed files

Complete the assignment and make `python check_assignment.py` report `All public checks passed.` Then:

1. Return to GitHub Desktop and select the **Changes** tab.
2. Confirm that the changed-file list contains only your Assignment 01 work. It should include the three student scripts, `terminal-practice/source.txt`, `terminal-practice/path-check.txt`, and `output/readiness.txt`.
3. In the **Summary** box, enter exactly `Complete Assignment 01`.
4. Select **Commit to main**.
5. Select **Push origin**.
6. Return to the assigned repository page in the browser and refresh it.
7. Open `output/readiness.txt` on the repository page and confirm that its final line is `Next checkpoint: 5`.
8. Optionally open the repository's **Actions** tab to inspect the public pytest feedback; a green run is useful evidence but is not a submission requirement.

If any named button is unavailable, stop before trying terminal Git commands. Record a screenshot and the exact message, then contact the instructor. The delivery checkpoint is handled separately from the Python pass/fail result.
