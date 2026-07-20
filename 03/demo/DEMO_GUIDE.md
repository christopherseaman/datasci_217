# Lecture 03 demonstration guide

Run these three required demonstrations from `03/demo` in the POSIX-style shell established in Lecture 01: Bash on Linux, WSL, or the supported cloud environment, and default zsh on macOS. Native PowerShell is not a second required command interface. All paths below use forward slashes.

Start from a fresh copy in which `.venv/`, `recreation-check/`, and `site_counts.txt` do not exist. The candidate versions used here are Python 3.12.13 and NumPy 2.0.2; they are not yet the final course lock.

# Demo 1 — Reproduce the candidate environment

Open `03/demo` as the project folder in VS Code. Start a new integrated terminal and confirm the working directory:

```bash
pwd
ls
```

Inspect the interpreter used before activation:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

For only these two pre-environment checks, use `python3` instead if that was the working command established in Lectures 01–02. After activation, use `python` consistently.

Confirm that uv is already installed. Installation belongs to course readiness and follows the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/); this demonstration does not run a downloaded installer pipeline.

```bash
uv --version
```

Inspect the deliberate project records:

```bash
cat .python-version
cat requirements.txt
```

Their exact contents are:

```text
3.12.13
numpy==2.0.2
```

Install and pin the exact candidate interpreter, then create the project environment:

```bash
uv python install 3.12.13
uv python pin 3.12.13
uv venv --python 3.12.13 .venv
source .venv/bin/activate
```

Verify the active interpreter before installing the direct requirement:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

The version must be `Python 3.12.13`. The executable path depends on the project location, but it must end inside `03/demo/.venv/bin/python`.

Install only from the deliberate direct-dependency file and run the check program:

```bash
uv pip install -r requirements.txt
python environment_check.py
```

Expected program output:

```text
Python: 3.12.13
NumPy: 2.0.2
```

Leave the first environment:

```bash
deactivate
```

Create a separate disposable project directory and copy only the records and check program needed to reproduce the environment:

```bash
mkdir recreation-check
cp .python-version requirements.txt environment_check.py recreation-check/
cd recreation-check
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
python -c "import sys; print(sys.executable)"
python environment_check.py
```

The executable path must now point inside `recreation-check/.venv/bin/python`, and the check program must reproduce the same two exact output lines. Leave that environment and return to `03/demo`:

```bash
deactivate
cd ..
```

The `.venv/` directories are generated from the committed version and requirement records; they are not course source files. Before repeating the demonstration, confirm `pwd`, then remove only the generated `.venv/` and `recreation-check/` directories through the VS Code Explorer.

## Standard-library fallback reference

This is a reference for a machine where uv is unavailable and the candidate interpreter is already installed. It is not a second required demonstration. Confirm the exact interpreter before creating the environment:

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python environment_check.py
deactivate
```

The first command must report Python 3.12.13, and the program output must match the two candidate-version lines above.

# Demo 2 — Build the ndarray mental model

Remain in `03/demo` with its primary `.venv` active. If it is no longer active, reactivate it:

```bash
source .venv/bin/activate
```

Open `ndarray_mental_model.py`. Its operations follow the required sequence: metadata, positional selection, view/copy behavior, one mask, same-shape arithmetic, reductions, reshape/transpose, and one scalar-to-1D broadcast.

Before running, record the predicted shape of each array result. Run the program:

```bash
python ndarray_mental_model.py
```

Expected output:

```text
Metadata
scores: [18. 21. 24. 19.]
score table:
[[18. 21. 24.]
 [20. 22. 26.]]
scores metadata: shape=(4,), ndim=1, size=4, dtype=float64
table metadata: shape=(2, 3), ndim=2, size=6, dtype=float64
Selection
first score: 18.0
second row: [20. 22. 26.]
third column value: 24.0
middle scores: [21. 24.]
second column: [21. 22.]
View and copy
source after view mutation: [10 99 30 40]
source after copy mutation: [10 20 30 40]
copy after mutation: [99 30]
Mask
mask: [False  True  True False]
masked values: [21 24]
Same-shape arithmetic
change: [ 2 -1  3]
Reductions
overall mean: 25.0
column means: [15. 25. 35.]
column means shape: (3,)
row means: [20. 30.]
row means shape: (2,)
Reshape and transpose
grid:
[[1 2 3]
 [4 5 6]]
grid shape: (2, 3)
transpose:
[[1 4]
 [2 5]
 [3 6]]
transpose shape: (3, 2)
Scalar-to-1D broadcast
adjusted scores: [19 22 25]
adjusted shape: (3,)
```

The view mutation changes its source. The explicit copy mutation leaves its source unchanged. For reductions, the selected axis disappears from the result shape. The only broadcasting pattern taught here is a scalar combined with a 1D array.

# Demo 3 — Run a bounded pipeline and terminal analysis

The committed `observations.csv` fixture is deliberately small. Each field contains no comma, quoted newline, or other CSV complication. Preview its beginning and end:

```bash
head -n 3 observations.csv
tail -n 2 observations.csv
```

Select the site field after removing the one-line header, then order and count it:

```bash
tail -n +2 observations.csv | cut -d',' -f1
tail -n +2 observations.csv | cut -d',' -f1 | sort | uniq -c
```

Save the count result with overwrite redirection:

```bash
tail -n +2 observations.csv | cut -d',' -f1 | sort | uniq -c > site_counts.txt
cat site_counts.txt
wc -l site_counts.txt
```

`uniq -c` may pad its count field differently across supported systems. The exact count/name pairs are:

```text
3 north
2 south
1 west
```

`wc -l` must report three lines in `site_counts.txt`. This is a bounded pipeline for the supplied fixture, not a general CSV parser.

`data_loader.py` is supplied infrastructure. Do not implement or modify its CSV parsing. Its teaching interface is `load_measurements("observations.csv")`, which returns a homogeneous float64 ndarray with shape `(6, 2)`.

Verify that importing the supplied loader and analysis produces no terminal output or report:

```bash
python -c "import data_loader; import array_summary"
```

Check the supplied loader's return contract:

```bash
python -c "from data_loader import load_measurements; data = load_measurements('observations.csv'); print(data.shape); print(data.dtype)"
```

Expected output:

```text
(6, 2)
float64
```

Run the import-safe terminal analysis:

```bash
python array_summary.py
```

Expected output:

```text
Measurements shape: (6, 2)
Measurements dtype: float64
Overall mean: 25.0
Column means: [20. 30.]
Column means shape: (2,)
Row means: [15. 25. 35. 15. 25. 35.]
Row means shape: (6,)
Values at or above 30: 6
```

The analysis receives a 2D array from the supplied loader. It calculates whole-array and axis reductions, reshapes the array to 1D, and only then applies the scalar mask. Run the program again; a fresh script process must produce the same exact output.

No notebook is created or executed in these demonstrations. Lecture 04 introduces notebook state and labeled pandas objects after this terminal workflow is secure.
