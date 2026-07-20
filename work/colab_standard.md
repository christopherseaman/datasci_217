# Google Colab standard for DataSci 217 demos

## Scope

Colab is the default launch experience for notebook demos in Lectures 04–11. Local Jupyter remains supported.

Lectures 01–03 are strictly pre-notebook: every demo and assignment must use Python scripts and the terminal. Do not provide Colab or Jupyter alternatives before Lecture 04 introduces notebooks. Shell, Git, SSH, tmux, and virtual-environment exercises remain local-first throughout the course.

## Candidate 2026–27 runtime target

- Python 3.12.13
- NumPy 2.0.2
- pandas 3.0.3

Google's pin-able Colab 2026.04 runtime supplies Python 3.12.13 and NumPy 2.0.2 and remains available for one year: <https://research.google.com/colaboratory/runtime-version-faq.html>. Its published package inventory contains pandas 2.2.2, so the course setup cell must install pandas 3.0.3 before importing pandas when this runtime is selected: <https://raw.githubusercontent.com/googlecolab/backend-info/77d5dbef56d73b96db5efef2280679cb548c9bd9/pip-freeze.txt>. pandas 3 supports Python 3.11+ and NumPy 1.26+: <https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v3.0.0.html>.

pandas 3.0.4 was removed from the candidate on 2026-07-18 because PyPI marks that release as yanked for reported datetime-related segmentation faults: <https://pypi.org/project/pandas/>. The replacement 3.0.3 combination has passed an isolated import/version smoke test with Python 3.12.13 and NumPy 2.0.2, but still requires the complete two-environment certification below.

This is a compatibility candidate, not the final release lock. Freeze exact versions only after every required notebook and grader passes locally and in a fresh Colab runtime. Avoid upgrading unrelated preinstalled Colab packages.

For assignment notebooks, students restart and run all before submission. Stored outputs may remain when a human needs to review a chart or written interpretation, but the grader ignores them and executes a fresh copy. Stored output is never accepted as proof that the submitted code runs.

## Notebook contract

Every Colab-ready demo must:

1. Use Python 3 and a portable kernelspec.
2. Avoid absolute paths and assumptions about the launch directory.
3. Install only missing/non-default dependencies in one clearly labeled setup cell.
4. Acquire data reproducibly from committed small files or a stable HTTPS source.
5. Set random seeds when generated values affect teaching output.
6. Avoid credentials, mounted Drive, or private data by default.
7. Run top-to-bottom in a fresh runtime without manual file uploads.
8. Complete within a documented runtime and reasonable memory limit.
9. Preserve compatibility with local Jupyter/VS Code.
10. Include a final verification cell or clearly observable expected result.

## Standard header

Each notebook should begin with:

- lecture and demo title;
- learning objectives;
- “Open in Colab” badge in the paired demo guide/course page;
- note that changes in Colab are not saved back to GitHub;
- tested Python/package date;
- approximate runtime;
- link to local setup instructions.

## Setup-cell pattern

Prefer a small conditional or quiet install rather than reinstalling the complete data-science stack. The notebook should print key versions after setup. Repository files should be fetched from an immutable course release/tag when the course release is frozen; during development they may use the working branch.

## Data strategy

- Small static data: commit beside the notebook and fetch from the raw GitHub URL.
- Generated data: use a deterministic generator inside the notebook or a shared module fetched from the release.
- Medium public data: download from a stable, documented source and cache only within the runtime.
- Large data: provide a reduced teaching sample; do not make Drive mounting the default.
- Network-dependent examples: include a small fallback dataset where practical.

## Initial conversion inventory

| Lecture | Demo notebooks | Colab disposition |
|---|---:|---|
| 04 | 3 | Pilot all; establishes the base pattern. |
| 05 | 3 | Convert after checking generated files and pipeline paths. |
| 06 | 3 | Convert; expected to be low risk. |
| 07 | 3 | Convert; check interactive renderers and exported assets. |
| 08 | 3 | Convert aggregation demos; keep remote/performance portions explicitly environment-sensitive. |
| 09 | 4 | Convert the three topic demos; assess whether the notebook demo guide is redundant. |
| 10 | 3 | Convert stats/ML; deep-learning demo is optional pending scope and runtime decisions. |
| 11 | 4 | Convert only after data download, runtime, and memory are stabilized. |

## Validation matrix

For every certified notebook record:

- notebook path;
- paired Markdown path;
- dependency source;
- data source;
- Colab pass/fail and date;
- local pass/fail and date;
- runtime;
- maximum observed memory where relevant;
- warnings/deprecations;
- expected generated files.

## Badge format

Use the official Colab URL form:

`https://colab.research.google.com/github/christopherseaman/datasci_217/blob/<release-or-branch>/<path>.ipynb`

Do not add production badges until the canonical organization/repository and release branch are confirmed.
