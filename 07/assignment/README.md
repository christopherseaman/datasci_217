# Assignment 07: Visualization Critique, Redesign, and Explanation

## Files

```text
assignment/
├── assignment.ipynb     # Provided notebook scaffold to complete
├── data/                # Provided fixtures
├── requirements.txt     # Provided pinned environment
├── check_assignment.py  # Provided completion checker
└── output/              # Generated artifacts to submit
```

## Setup

In VS Code, open the assignment folder and choose **Terminal → New Terminal**. You can also use your native terminal or WSL Ubuntu; change to the assignment directory before running these commands.

From this assignment directory, create a Python 3.13 environment and install the pinned requirements:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. If Python 3.13 is missing, run `uv python install 3.13`. Select this environment as the kernel when opening the notebook in VS Code or Jupyter. The course uses pandas 3.0.5.

Keep the supplied `data/` files unchanged. Open the complete assignment directory; the setup cell locates and verifies its fixtures in either a standalone assignment repository or the course repository. Restore missing or checksum-mismatched fixtures before continuing.

## Prepared fixtures

- `format_completion.csv`: four rows, one per delivery format and stage;
  categorical format and stage with quantitative prepared completion percent.
- `session_observations.csv`: twelve rows, one per synthetic learning session;
  identifier, categorical pathway, and two quantitative fields.
- `pathway_checkpoints.csv`: eight rows, one per pathway and checkpoint;
  categorical pathway, ordered checkpoint, and quantitative prepared
  completion percent.

The rows describe only the prepared fixtures. They do not establish cause, population effects, prediction, or general patterns.

## Question 1: Bounded exploration

### 1.1 Build and export the exploratory chart

State an exploratory question, the one-session row/mark grain, variable roles, one observation restricted to the twelve supplied rows, and a limitation that rejects causal and generalized conclusions. Implement `build_exploratory_chart(session_table, pathway_order)` as one Altair scatterplot specification of activities completed against reflection score. Use typed quantitative positions and nominal pathway color and point-shape encodings, preserve the caller's two-label order, label units, and include tooltips. Export `exploratory_chart.to_dict()` as `output/exploratory_spec.json`, embedding the plotted session rows.

> **Checkpoint — `output/exploratory_spec.json`**

## Question 2: Critique and redesign

### 2.1 Explain and repair the supplied chart

Inspect the supplied four-bar comparison for a learning-support coordinator. It intentionally has an unsupported causal title, truncated baseline, missing unit, color-only category encoding, and distracting decoration. Explain each problem and a repair without changing the prepared values.

Implement `build_critique_redesign(summary_table, format_order, stage_order)`. Use a zero baseline, explicit percentage unit, course colors plus hatches, value labels, restrained decoration, and an outside legend. Preserve arbitrary valid caller labels and order. Save the canonical result as `output/critique_redesign.png`.

> **Checkpoint — `output/critique_redesign.png`**

## Question 3: Audience-focused explanation

### 3.1 Prepare the supporting evidence

State the question, learning-support coordinator audience and follow-up use, bounded intended claim, unit, grain, roles, comparison, chart rationale, and causal limitation. Copy and export the exact supporting data.

> **Checkpoint — `output/explanatory_supporting_data.csv`**

### 3.2 Build the explanatory chart

Implement `build_explanatory_chart(checkpoint_table, pathway_order)` so the two ordered paths have redundant color, marker, and line-style cues. Derive the leader, checkpoint count, final absolute gap, title, and annotation from any valid two-pathway input; on a final tie, attach the annotation to the second requested pathway. Save the canonical result as `output/pathway_explanatory.png`.

> **Checkpoint — `output/pathway_explanatory.png`**

### 3.3 Export the explanation and inspect the chart

Export the evidence JSON and a matching text alternative. Finish the visual-review checklist with observable evidence rather than yes/no answers.

> **Checkpoint — `output/visualization_evidence.json`**

> **Checkpoint — `output/explanatory_text_alternative.txt`**

## Check Your Work

Run this from the assignment directory after saving your artifacts:

```bash
python check_assignment.py
```

Fix each failed check, regenerate the affected files, and run the checker again. It reads saved artifacts without running your code.

### Completion contract

All six files below are graded. JSON and text files use UTF-8; the CSV contains the supplied data with its named columns and no duplicate rows.

| Artifact | Completion criteria |
|---|---|
| `output/exploratory_spec.json` | An Altair point specification embedding exactly the 12 session rows. Encode `activities_completed` and `reflection_score` as quantitative x/y, and `pathway` as nominal color and shape. |
| `output/critique_redesign.png` | The saved redesign in PNG format. |
| `output/pathway_explanatory.png` | The saved explanatory chart in PNG format. |
| `output/explanatory_supporting_data.csv` | Exactly the supplied pathway rows and columns `pathway`, `checkpoint_number`, `completion_percent`. |
| `output/visualization_evidence.json` | An object with five `critique` entries, one per category: `unsupported claim`, `truncated baseline`, `missing unit`, `color-only encoding`, `distracting decoration`; each has nonblank `problem` and `repair`. Include nonblank `question`, `audience`, `intended_claim`, `displayed_unit`, `grain`, and `text_alternative`; `variable_roles` maps `pathway` to `categorical`, `checkpoint_number` to `ordered`, and `completion_percent` to `quantitative`. |
| `output/explanatory_text_alternative.txt` | Nonblank text matching the JSON `text_alternative`, allowing different line endings and trailing newlines. |

Task 1 is worth 25 points, Task 2 is worth 37, and Task 3 is worth 38: 100 total. The PNG check verifies file format; inspect your charts for clarity, accessibility, honest scales, clipping, and overlap.

## Submit

In VS Code Source Control, inspect your completed notebook and required `output/` files, then commit and push them. Alternatively, use **Add file → Upload files** on the GitHub website and commit the files at their required paths. Keep private data, credentials, virtual environments, and notebook checkpoints out of your submission. GitHub Actions runs the assignment checks automatically on every push. If your fork has Actions disabled, enable it once in the Actions tab. Review the feedback, then regenerate, check, commit, and push corrected artifacts if needed.
