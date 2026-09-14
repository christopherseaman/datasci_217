# Assignment 04 platform delivery

This required checklist covers artifact delivery and GUI workflow. The public checker is the grading contract: graders run a trusted copy against committed CSVs without executing the notebook. Do not use command-line Git.

Assignment 04 is local-first. Use the supplied notebook or another editor to create the committed CSV artifacts; grading reads committed artifacts without executing the notebook. Lecture 04 demos may be shown in Colab, but this assignment should not be uploaded by itself.

## Open the assignment repository

1. Open the `04/assignment` subtree, or its exported standalone repository, with GitHub Desktop or VS Code Source Control.
2. Open that repository locally with GitHub Desktop or VS Code Source Control.
3. Open the repository folder in VS Code and confirm it contains this `PLATFORM_CHECK.md`.
4. Switch to `main` if necessary, select **Sync Changes**, and confirm there are no unfinished changes.
5. For `assignment.ipynb`, open it in the local notebook interface and select the Python 3 environment created from `.python-version` and `requirements.txt`.

## Create the milestone artifacts

1. Complete the three tasks without editing the supplied setup cell.
2. Save the completed notebook source, create `output/labeled_block.csv` and `output/selected_purchases.csv`, and run `python check_assignment.py`.
3. Continue only when the checker prints `All public checks passed.`

The automated grader reads the committed CSV files directly. A notebook rerun is optional local QA.

## Inspect, commit, and push

1. Open the Source Control changes list.
2. Confirm that the fixture, manifest, environment records, checker, and platform guide are unchanged.
3. Inspect the notebook diff and both generated CSV files. Ensure they contain only synthetic course data and no credentials or private information.
4. Stage `assignment.ipynb`, `output/labeled_block.csv`, and `output/selected_purchases.csv`.
5. Commit with the summary `Complete Assignment 04 notebook`.
6. Select **Sync Changes** or **Push origin** so the assignment repository receives the commit.
7. Open the assigned repository in the browser and confirm the notebook and both CSV files are present.
8. Optionally open the repository's **Actions** tab to inspect the public pytest feedback. Actions are optional feedback, not a submission requirement.

If a required GUI or local-Jupyter control is unavailable, stop before trying terminal Git or moving the assignment to Colab. Record the exact message and contact the instructor.
