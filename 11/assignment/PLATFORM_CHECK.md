# Platform Check

Local Jupyter is the grading reference. Colab assignment support is pending save-back validation, so use the local setup unless your instructor announces otherwise.

From `11/assignment`, run:

```bash
uv venv --python 3.12.13
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
python -c "import numpy,pandas,sklearn,matplotlib,jupyterlab,jupytext; print(numpy.__version__, pandas.__version__, sklearn.__version__, matplotlib.__version__, jupyterlab.__version__, jupytext.__version__)"
./download_data.sh
```

Expected output versions:

```text
Python 3.12.13
2.0.2 3.0.3 1.9.0 3.11.1 4.4.10 1.18.1
```

If `uv` reports that Python is missing, run `uv python install 3.12.13`, then repeat setup. Do not replace exact pins with version ranges.

The release is a CSV read directly by pandas, so no parquet or geographic package is needed.
