# Assignment 09 local platform check

The completed notebook/source is required coursework, but automated grading does not execute it. Assignment Colab is not part of this repository contract.

## Prepare the environment

From the repository root, create or refresh the Assignment 09 environment with the Python version recorded in `.python-version` and the packages in `requirements.txt`. The course uses `uv`:

```text
uv venv --python 3.12.13
uv pip install -r 09/assignment/requirements.txt
```

Select that interpreter as the `Python 3` kernel in VS Code/Jupyter. The first notebook code cell verifies Python 3.12.13, NumPy 2.0.2, and pandas 3.0.5; it does not install packages.

## Prepare the artifacts

1. If you use the notebook, open it from the repository root, from `09/assignment/`, or from a directory nested inside the assignment.
2. Create the six CSV artifacts.
3. From `09/assignment/`, run:

   ```text
   python check_assignment.py
   ```

5. If the checker reports a fix, regenerate the artifacts and run it again.

## Commit and submit

Use VS Code Source Control or GitHub Desktop for the required Git path. Confirm all six CSVs in `output/` are visible, commit them, and push. The repository ignore rules intentionally do not hide these files.

Optionally inspect the repository's Actions feedback after pushing. If a check
fails, regenerate the artifacts, rerun the public checker,
then commit and push the corrected files. Command-line Git is optional bonus
knowledge and is not assessed here.

Do not add private data, credentials, notebook checkpoints, environments, or extra output files. Colab save-back and submission are not claimed by this assignment.
