# Assignment 05 platform check

Complete this check locally before preparing the artifacts. Colab is not an
assignment or submission path for Assignment 05.

1. Open a terminal in the assignment directory.
2. Confirm `python --version` reports `Python 3.14`.
3. Activate a fresh `.venv` and run `python -m pip install -r requirements.txt`.
4. Run the following commands:

   ```bash
   python -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)"
   python -c "from pathlib import Path; print(Path('data/people_raw.csv').is_file())"
   ```

   The first command must print `2.3.3 3.0.5`; the second must print `True`.

5. For `assignment.ipynb`, open it with the `.venv` Python 3 kernel; execution is optional local QA.
6. Create the three CSV artifacts and run `python check_assignment.py`. The starter should show task-specific
   `[FIX]` messages until the TODOs and generated artifacts are complete.

If a version is wrong, recreate the virtual environment rather than installing
additional unrecorded packages. If the data check is false, return to the
assignment directory; do not replace the portable path logic with an absolute
path.
