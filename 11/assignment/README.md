# Final Project: Chicago Beach Weather Forecasting

## Files

```text
assignment/
├── assignment.md          # Provided artifact schemas and forecasting rules
├── q1_*.md … q9_*.md       # Provided question scaffolds
├── q1_*.ipynb … q9_*.ipynb # Paired notebook scaffolds
├── report.md              # Provided report scaffold to complete
├── data/                  # Provided frozen release and provenance
├── requirements.txt       # Provided pinned environment
├── download_data.sh        # Provided release verification helper
├── check_assignment.py    # Provided completion checker
└── output/                # Generated CSV and PNG artifacts to submit
```

## Setup

In VS Code, open the assignment folder and choose **Terminal → New Terminal**. You can also use your native terminal or WSL Ubuntu; change to the assignment directory before running these commands.

From `11/assignment`:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
./download_data.sh
jupyter lab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. If Python 3.13 is missing, run `uv python install 3.13`. Select this environment as the notebook kernel. The pinned versions are NumPy 2.3.3, pandas 3.0.5, scikit-learn 1.9.0, Matplotlib 3.11.1, JupyterLab 4.4.10, and Jupytext 1.18.1.

The release and provenance manifest are committed under `data/`. `download_data.sh` verifies these files without downloading or replacing them. Start with [`assignment.md`](assignment.md) for the forecasting question, fixed inputs, and artifact schemas, then complete the nine notebook scaffolds in order. See [`HINTS.md`](HINTS.md) for nudges.

## Questions

| Question | Points | Notebook | Main result |
|---|---:|---|---|
| Q1 | 7 | [`q1_setup_exploration.ipynb`](q1_setup_exploration.ipynb) | Audit and explore the frozen release |
| Q2 | 9 | [`q2_data_cleaning.ipynb`](q2_data_cleaning.ipynb) | Clean timestamps and sensor values |
| Q3 | 11 | [`q3_data_wrangling.ipynb`](q3_data_wrangling.ipynb) | Build a complete station-hour panel |
| Q4 | 14 | [`q4_feature_engineering.ipynb`](q4_feature_engineering.ipynb) | Build leakage-safe forecast features |
| Q5 | 7 | [`q5_pattern_analysis.ipynb`](q5_pattern_analysis.ipynb) | Describe training-only patterns |
| Q6 | 11 | [`q6_modeling_preparation.ipynb`](q6_modeling_preparation.ipynb) | Create fixed chronological splits |
| Q7 | 13 | [`q7_modeling.ipynb`](q7_modeling.ipynb) | Select and validate one sklearn model |
| Q8 | 13 | [`q8_results.ipynb`](q8_results.ipynb) | Evaluate the untouched test period |
| Q9 | 15 human | [`q9_writeup.ipynb`](q9_writeup.ipynb) | Complete `report.md` |

## Check Your Work

From this assignment directory:

```bash
./download_data.sh
jupytext --to ipynb --test-strict q*.md
python check_assignment.py
```

### Completion contract

Submit every CSV and PNG listed under Q1–Q8 in the [artifact contract](assignment.md#artifact-contract), at its exact `output/` path, and complete root `report.md` as specified under Q9. The contract gives each filename, column sequence, row identity, calculation, timestamp convention, and missing-value rule. CSVs use no extra index except the labeled Q5 correlation matrix. Predictions must align with the documented split rows, and metrics must agree with the saved predictions. The PNG check verifies file format; inspect the images yourself for readable labels and the requested content.

The saved artifacts earn up to 85 automated points: Q1 7, Q2 9, Q3 11, Q4 14, Q5 7, Q6 11, Q7 13, and Q8 13. Q9's `report.md` earns 15 human-review points: 5 for justified decisions, 5 for evidence-based interpretation, and 5 for limitations and clear communication. Its automated structure, metrics-table, and image-link checks are readiness feedback. There is no model-performance threshold.

## Submit

Commit and push the nine `.md`/`.ipynb` pairs, root `report.md`, and all required files under `output/` using VS Code Source Control. Alternatively, use **Add file → Upload files** on the GitHub website and commit the files at their required paths. Clear notebook outputs before submission; retain the CSV and PNG files. Keep supplied `data/` files unchanged. GitHub Actions runs the checks automatically on every push. If your fork has Actions disabled, enable it once in the Actions tab. Review the feedback, then regenerate, check, commit, and push corrected artifacts if needed.
