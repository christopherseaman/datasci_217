# Assignment 03 platform delivery

This required GUI delivery workflow is an unassessed platform checklist, separate from the public code checker. A grading checkout can validate files and behavior, but it cannot prove local environment activation/recreation or which Git interface you operated.

Use VS Code Source Control. GitHub Desktop is an acceptable equivalent when it exposes the same working-tree, diff, staging, commit, branch, merge, and remote states. Do not use command-line Git.

## Open the provisioned repository

1. Open the Assignment 03 repository URL supplied by the instructor.
2. Open the repository with the supported GUI workflow.
3. Open the repository folder in VS Code and confirm that it contains this `PLATFORM_CHECK.md`.
4. Switch to `main` if necessary, select **Sync Changes**, and confirm that Source Control lists no unfinished changes.

## Create the feature branch

1. Open the VS Code Command Palette and run **Git: Create Branch**.
2. Enter `feature/numpy-analysis` and choose `main` as its source if prompted.
3. Confirm that the branch indicator shows `feature/numpy-analysis`.

## Commit the environment and pipeline records

1. Complete Task 1, including the recreation check, then complete and run the four Task 2 commands.
2. Confirm through the Explorer and Source Control that neither `.venv/` nor `recreation-check/` appears as a submitted change.
3. Inspect the diffs for `.python-version`, `requirements.txt`, `PIPELINE.md`, and `output/environment_check.txt`.
4. Stage only those four files and commit with `Record reproducible terminal workflow`.

The four generated pipeline text files may remain untracked working evidence; the checker reruns the documented commands and does not require them as submission artifacts.

## Commit the NumPy implementation

1. Complete `array_analysis.py` and `analysis.py`.
2. Run `python analysis.py`, then `python check_assignment.py` until all public checks pass.
3. Inspect both Python diffs in Source Control.
4. Stage only `array_analysis.py` and `analysis.py` and commit with `Implement NumPy array analysis`.
5. Select **Publish Branch** or **Sync Changes** so the remote receives both commits.

## Merge and deliver main

1. Confirm that the feature-branch working tree has no unfinished required changes.
2. Switch to `main` with the branch control in the VS Code Status Bar.
3. Open the Command Palette, run **Git: Merge...**, and select `feature/numpy-analysis`.
4. A conflict is not required. If one appears, resolve the intended final content, remove every conflict marker, inspect the resolution, and complete the merge through Source Control.
5. Select **Sync Changes**.
6. In the browser, confirm that `main` contains the two exact environment records, completed pipeline block, saved environment probe, and both completed Python files. Confirm that `.venv/` and `recreation-check/` are absent.
7. Optionally open the repository's **Actions** tab to inspect the public pytest feedback.

If a required GUI control is unavailable, stop before trying terminal Git commands. Record the exact message and contact the instructor.
