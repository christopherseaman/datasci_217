# Lecture 02 demonstration guide

Run all Python examples from their own project directories. These demos use terminal-executed `.py` files.

# Demo 1 — One GUI Git state cycle

Use a disposable private repository. Do not initialize a repository inside the course repository.

## Create and publish the seed repository

1. Copy the two files from `01_git_gui_seed/` into a new folder named `measurement-summary` outside the course repository.
2. Open `measurement-summary` with **File → Open Folder** in VS Code.
3. Open Source Control. Select **Initialize Repository**.
4. Confirm that the branch name in the Status Bar is `main`. If it is not, open the Command Palette, run **Git: Rename Branch**, and enter `main`.
5. Under **Changes**, open `README.md` and `.gitignore` to inspect their diffs.
6. Select the `+` beside each file. They move from **Changes** to **Staged Changes**.
7. Enter `chore: seed measurement summary` as the commit message and select **Commit**.
8. Select **Publish Branch** (or **Publish to GitHub**), choose a private disposable repository, and complete the GitHub sign-in prompt if it appears.
9. Select **Sync Changes** if VS Code still shows an outgoing commit. The local `main` branch and its remote now contain the seed commit.

## Inspect, stage, commit, and synchronize focused changes

1. Add this line immediately below `Values: 18, 21, 24` in `README.md`:

   ```text
   Units: degrees Celsius
   ```

2. Add `.DS_Store` on a new line at the end of `.gitignore`.
3. Open each changed file from Source Control and inspect its line-by-line diff.
4. Select the `+` beside `README.md` only. Confirm that `README.md` is under **Staged Changes** while `.gitignore` remains under **Changes**.
5. Commit the staged file with `docs: add measurement units`.
6. Stage `.gitignore` and commit it separately with `chore: ignore macOS metadata`.
7. Select **Sync Changes**. The remote receives both focused commits.

The working tree is clean when Source Control lists no changed files. Staging selected the proposed content of each commit; synchronizing moved the resulting commits to the remote.

## Create both sides of a prepared conflict

1. Confirm that the current branch in the VS Code Status Bar is `main` and that Source Control is clean.
2. Open the Command Palette and run **Git: Create Branch**.
3. Name the branch `feature/summary-status` and select `main` as its source if prompted. VS Code switches to the new branch.
4. In `README.md`, replace `Summary: draft` with:

   ```text
   Summary: ready for team review
   ```

5. Inspect the diff, stage `README.md`, and commit with `docs: mark summary ready`.
6. Select **Publish Branch** so the feature commit also has a remote copy.
7. Select the branch name in the Status Bar and switch back to `main`.
8. In `README.md`, replace `Summary: draft` with:

   ```text
   Summary: owned by the course team
   ```

9. Inspect the diff, stage `README.md`, and commit with `docs: identify summary owner`.

Both branches now contain a different committed replacement for the same original line.

## Merge and resolve the conflict

1. Keep `main` checked out.
2. Open the Command Palette, run **Git: Merge...**, and select `feature/summary-status`.
3. Open the conflicted `README.md` from Source Control. Use the merge editor or edit the file directly.
4. Replace the competing versions and any conflict markers with this one exact line:

   ```text
   Summary: ready for team review by the course team.
   ```

5. Save `README.md`. Confirm that no `<<<<<<<`, `=======`, or `>>>>>>>` markers remain.
6. Inspect the resolved diff, stage `README.md`, and complete the merge commit with `merge: combine summary status and owner`.
7. Select **Sync Changes**.

The final `README.md` is:

```text
# Measurement Summary

Values: 18, 21, 24
Units: degrees Celsius
Summary: ready for team review by the course team.
```

# Demo 2 — From duplication to functions

## Run the duplicated calculation

1. Open `02_functions/` in VS Code.
2. Start a new integrated terminal and confirm that its working directory ends in `02_functions`.
3. Run:

   ```bash
   python duplicated_summary.py
   ```

4. Verify the exact output:

   ```text
   Morning mean: 21.0
   Evening mean: 22.7
   ```

The two blocks repeat the same total-and-divide algorithm. A change to that algorithm would have to be made twice.

## Trace the refactor

1. Open `functions_summary.py` beside `duplicated_summary.py`.
2. In `mean(values)`, locate the one-sentence docstring, the `values` parameter, the local variables `total` and `value`, and both return paths.
3. Locate `mean(record["values"])` inside `format_summary(record)`. The selected list is the argument whose value is assigned to the parameter `values` for that call.
4. Follow `mean([])`: an empty list is false in a condition, so the function reaches `return None` before division.
5. Follow `format_summary()` for the `Overnight` record: `average is None` selects the no-measurements text.
6. Run:

   ```bash
   python functions_summary.py
   ```

7. Verify the exact output:

   ```text
   Morning mean: 21.0
   Evening mean: 22.7
   Overnight mean: no measurements
   ```

`mean()` returns a reusable value; `format_summary()` decides how to represent either a numeric mean or the explicit `None` result as text.

# Demo 3 — From functions to an import-safe module

1. Open `03_module/` in VS Code.
2. Start a new integrated terminal and confirm that its working directory ends in `03_module`.
3. Confirm that `analysis_utils.py` and `main.py` are in this same directory. No import-path modification is needed.
4. Remove an old `report.txt` if this directory has been used before.
5. Check that importing the driver is silent:

   ```bash
   python -c "import main"
   ```

6. Confirm that the command printed nothing and did not create `report.txt`. Importing defines functions, but the main guard prevents `main()` from running.
7. Run the driver:

   ```bash
   python main.py
   ```

8. Verify the exact terminal output:

   ```text
   Morning mean: 21.0
   Evening mean: 22.7
   Overnight mean: no measurements
   ```

9. Open `report.txt`. Its exact contents are the same three lines, including a newline after the final line.
10. Run `python main.py` again. Mode `"w"` replaces the prior report, so the terminal output and report remain identical rather than accumulating duplicate lines.

`analysis_utils.py` owns reusable definitions and has no output side effects. `main.py` owns the records, file write, read-back, terminal output, and guarded call to `main()`.
