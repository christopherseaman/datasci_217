# Lecture 01 onboarding and setup readiness

Complete this checklist before the required demos. It verifies access and tools; it does not teach or assess Git.

## GitHub and Classroom 50 access

1. Sign in to the GitHub account you will use for the course and verify its email address.
2. Accept the instructor's invitation to the course GitHub organization.
3. Open [Classroom 50](https://classroom50.org/) and select **Sign in with GitHub**.
4. Approve the requested GitHub access, then open the instructor-provided readiness-assignment link and accept the assignment.
5. Open the repository that Classroom 50 creates. Confirm that you can see its `README.md` while signed in to your own account.
6. Stop after confirming access. Assignment 01 supplies one exact, guided, and unassessed GUI synchronization checklist solely to deliver that work. Lecture 02 teaches the Git concepts and independent workflow.

If the course organization or repository is missing, record the GitHub username currently signed in and the exact message shown. Send both to the instructor.

GitHub email privacy is optional. To note GitHub's private email address for later, open **GitHub → Settings → Emails** and record the `noreply` address shown for your own account. Do not copy another person's address.

## Local tool check

1. Open a POSIX-style terminal: Bash on Linux/WSL or the supported cloud environment, or zsh on macOS.
2. Check Python:

   ```bash
   python --version
   ```

   If that command is not available, try `python3 --version` and use `python3` consistently for Lectures 01–02.
3. Confirm that the reported version is Python 3.12.
4. Open VS Code and install the **Python** extension published by Microsoft.
5. In VS Code, select **Terminal → New Terminal** and run the same Python version command there.
6. Confirm that Git is installed for Lecture 02:

   ```bash
   git --version
   ```

The version check is the only terminal Git command in Lecture 01.
