# Assignment 02 platform delivery

This is the required GUI delivery workflow for Assignment 02. It is reviewed separately from the public Python checker because a grading checkout may not preserve every branch reference or enough history to certify GUI competence.

Use VS Code Source Control for these steps. GitHub Desktop is an acceptable equivalent when it exposes the same working-tree, diff, staging, commit, branch, merge, and remote states. Do not use command-line Git and do not create another repository.

## Open the provisioned repository

1. Open the Assignment 02 repository URL supplied by the instructor.
2. Open the repository locally with the supported GUI workflow.
3. Open the repository folder in VS Code and confirm that it contains this `PLATFORM_CHECK.md`.
4. Open Source Control, switch to `main` if necessary, and select **Sync Changes** before editing.
5. Confirm that Source Control lists no unfinished changes.

## Create the feature branch

1. Open the VS Code Command Palette and run **Git: Create Branch**.
2. Enter the exact branch name `feature/measurement-summary` and choose `main` as its source if prompted.
3. Confirm that the branch indicator now shows `feature/measurement-summary`.

## Commit the documentation change

1. Complete the README description and Run sections, `.gitignore`, and `GIT_STATE_CHECK.md`.
2. Open each changed file from Source Control and inspect its diff.
3. Stage only those three documentation/configuration files. Leave unfinished Python files under **Changes**.
4. Inspect **Staged Changes** and commit with `Complete repository documentation`.

## Commit the Python change

1. Complete `analysis_utils.py` and `main.py`.
2. Run `python main.py`, then run `python check_assignment.py` until all public checks pass.
3. Inspect the Python diffs in Source Control.
4. Stage `analysis_utils.py`, `main.py`, and the freshly generated `report.txt`.
5. Commit with `Implement reusable measurement summary`.
6. Select **Publish Branch** or **Sync Changes** so the remote receives the feature-branch commits.

## Merge and deliver main

1. Confirm that the feature-branch working tree is clean.
2. Switch to `main` with the branch control in the VS Code Status Bar.
3. Open the Command Palette, run **Git: Merge...**, and select `feature/measurement-summary`.
4. A conflict is not required. If an unexpected conflict appears, resolve the intended final content, remove all conflict markers, inspect the resolution, and complete the merge through Source Control.
5. Select **Sync Changes**.
6. Open the assigned repository in the browser and confirm that `main` contains the completed documentation, both Python files, and `report.txt`.
7. Optionally open the repository's **Actions** tab to inspect the public pytest feedback.

If a required GUI control is unavailable, stop before trying terminal Git commands. Record the exact message and contact the instructor.
