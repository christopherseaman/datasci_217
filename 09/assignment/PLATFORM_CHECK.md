# Assignment 09 local platform check

Clean local Jupyter or the VS Code notebook interface is required. Assignment Colab is not supported until the course repository-save and Classroom50 pilot is accepted.

## Prepare the environment

From the repository root, create or refresh the Assignment 09 environment with the Python version recorded in `.python-version` and the packages in `requirements.txt`. The course uses `uv`:

```text
uv venv --python 3.12.13
uv pip install -r 09/assignment/requirements.txt
```

Select that interpreter as the `Python 3` kernel in VS Code/Jupyter. The first notebook code cell verifies Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3; it does not install packages.

## Verify a clean run

1. Open `assignment.ipynb` from the repository root, from `09/assignment/`, or from a directory nested inside the assignment.
2. Restart the kernel and run all cells in order.
3. Confirm the final cell reports local readiness.
4. From `09/assignment/`, run:

   ```text
   python check_assignment.py
   ```

5. If the checker reports a fix, correct the source, restart the kernel, and run all again. Do not rely on stored output.

## Commit and submit

Use VS Code Source Control or GitHub Desktop for the required Git path. Confirm `assignment.ipynb` and all six CSVs in `output/` are visible, commit them, and push. The repository ignore rules intentionally do not hide these files.

Submit with the course's Classroom50 instructions. Open the supplied review/feedback link, revise the same notebook, restart and run all, rerun the public checker, then commit and push a corrected resubmission. Command-line Git is optional bonus knowledge and is not assessed here.

Do not add private data, credentials, notebook checkpoints, environments, or extra output files. Colab save-back and submission are not claimed by this assignment.
