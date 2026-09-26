# Assignment 07: Critique, Redesign, and Explain Cardiac Rehab Charts

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/                   # supplied; keep these files exactly as handed out
│   ├── rehab_patients.csv      # Task 1
│   ├── session_attendance.csv  # Task 2
│   └── followup_goals.csv      # Task 3
├── requirements.txt        # supplied: numpy, pandas, matplotlib, Altair, and ipykernel
├── .python-version         # supplied: tells uv to use Python 3.13
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── exploratory_spec.json             # you generate in Task 1
    ├── visualization_evidence.json       # you generate in Task 2 and save again in Task 3
    ├── critique_redesign.png             # you generate in Task 2
    ├── explanatory_supporting_data.csv   # you generate in Task 3
    ├── explanatory_chart.png             # you generate in Task 3
    └── explanatory_text_alternative.txt  # you generate in Task 3
```

## The data

All three files are synthetic and come from one hospital's cardiac rehabilitation service. After a heart attack or heart surgery, patients join a 36-session exercise program, either at the hospital (`Center-based`) or at home with weekly phone coaching (`Home-based`).

- `data/rehab_patients.csv` (Task 1): one row per patient: a synthetic `patient_id`, the `program`, the `sessions_attended` out of 36, and `walk_distance_m`, how far the patient walked in six minutes at discharge, in meters.
- `data/session_attendance.csv` (Task 2): one row per program and quarter: the `program`, the `quarter` (`Q1` or `Q2`), and `attended_pct`, the percentage of scheduled sessions that patients attended.
- `data/followup_goals.csv` (Task 3): one row per program and follow-up visit: the `program`, the `visit_number` (1 to 4, the visits in order), the `patients_seen` at that visit, and `goal_met_pct`, the percentage of those patients who met the weekly goal of 150 minutes of exercise.

```text
program,visit_number,patients_seen,goal_met_pct
Home-based,1,48,58
Home-based,2,46,63
```

Patients chose their program, and a few dozen rows describe only this service, so these data can describe what happened but cannot show that one program causes better results.

## Setup

Fork the assignment repository on GitHub and clone your fork the way Lecture 01 did: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open it. Then open **Terminal → New Terminal** in VS Code at the assignment directory (Ctrl+Shift+backtick, also Control on Mac). If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first. Run `ls data` and expect the three CSV files above. This clone is a new repository, so before your first commit run Lecture 02's two `git config user.name "..."` and `git config user.email "..."` lines in this terminal, with your name and GitHub noreply email.

Create the project environment, activate it, and install the supplied requirements, as in Lecture 03:

```bash
uv venv --seed --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` includes **ipykernel**, the package that lets a notebook run on this environment's Python (Lecture 04), so this one install is all the notebook needs. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`. If VS Code offers to install the **Jupyter** extension, accept.

> **Windows:** work in the **WSL: Ubuntu** window from Lecture 01's setup. In PowerShell instead, activate with `.\.venv\Scripts\Activate.ps1`.

Run the notebook's first code cell. It prints the NumPy, pandas, matplotlib, and Altair versions and `data files found: True`. `False` means the notebook is not running from the assignment directory, so open the folder itself in VS Code, not a folder above it.

## Task 1: Explore the rehab patients with Altair

A rehab nurse asks whether patients who attend more sessions walk farther at discharge, and whether that looks different in the two programs. An exploratory scatter plot is a first look at that question, not an answer to it.

### 1.1 Load the patients and build the scatter plot

In the Task 1.1 cell:

1. Read the patients with `patients = pd.read_csv(PATIENTS_PATH)`. Print the shape and the dtypes; expect `(12, 4)`.
2. Build `exploratory_chart` from `alt.Chart(patients)` with `.mark_point(filled=True, size=90)` and these encodings, as Lecture 07's "Encode the study table" snippet does:
    - `x`: `sessions_attended` as quantitative (`:Q`), titled with its unit, such as `Sessions attended (of 36)`;
    - `y`: `walk_distance_m` as quantitative, titled `Six-minute walk distance (m)`, with `scale=alt.Scale(zero=False)`;
    - `color` and `shape`: both `program` as nominal (`:N`), so the programs differ in grayscale as well as in color;
    - `tooltip`: `patient_id` and the three plotted columns.

    Add a title with `.properties(title=...)`.
3. Save it with `exploratory_chart.save(EXPLORATORY_SPEC_PATH)`, which writes the chart's Vega-Lite specification with the twelve rows embedded.

The cell ends by showing the chart: twelve filled points rising from lower left to upper right, one color and shape per program.

> **Checkpoint: `output/exploratory_spec.json`**
> A Vega-Lite specification whose `mark` has the type `point`, whose `encoding` has the `x`, `y`, `color`, and `shape` entries above, and whose embedded data holds the twelve patients.

### 1.2 Describe what the chart shows

Replace the TODO in the Markdown cell under 1.2 with a few sentences: the question the chart explores, what one point represents, one pattern you see, and why twelve patients cannot show that attending more sessions causes a longer walk.

Task 1.2 has no file of its own to check.

## Task 2: Critique and redesign a misleading chart

### 2.1 Inspect the draft chart

Run the Task 2.1 cell. It reads `data/session_attendance.csv` and draws a draft chart from the service's dashboard. The numbers are right, but the chart has five problems, one in each of the five categories Demo 3 of Lecture 07 uses in its critique:

| Category | Where to look |
| --- | --- |
| `unsupported claim` | The title |
| `truncated baseline` | Where the y-axis starts |
| `missing unit` | The y-axis label |
| `color-only encoding` | What tells the two programs apart |
| `distracting decoration` | Edges, hatches, and grid lines that carry no data |

### 2.2 Write the critique

In the Task 2.2 cell, fill in each entry's `problem` (what is wrong, and how it could mislead a reader) and `repair` (what the redesign changes). Keep the five `category` values as they are. Demo 3 of Lecture 07 writes a critique the same way. Running the cell prints the critique and saves it.

> **Checkpoint: `output/visualization_evidence.json`**
> An object with the key `critique`: a list of five entries, one per category above, each with the keys `category`, `problem`, and `repair`. Task 3.3 saves this file again with more keys and keeps `critique`.

### 2.3 Redesign the chart

In the Task 2.3 cell, reuse `quarters`, `x`, `width`, `home`, and `center` from the Task 2.1 cell to draw the same four bars on `redesign_ax`, repairing each problem, as Lecture 07's "Grouped Bars That Work in Grayscale" snippet does:

- start the y-axis at zero with `redesign_ax.set_ylim(0, 100)`, because bar length encodes the value;
- label the y-axis with its unit, such as `Scheduled sessions attended (%)`, and the x-axis `Quarter`;
- give each program its own color and its own hatch, such as `ORANGE` with `'..'` for Home-based and `BLUE` with `'//'` for Center-based;
- write each value on its bar with `redesign_ax.bar_label(bars, fmt='%d%%')`;
- use a title that describes what the bars show without claiming a cause;
- leave out the thick edges and the heavy grid, hide the top and right spines, and place the legend, titled `Program`, just outside the right edge.

Save with `redesign_fig.savefig(REDESIGN_PATH, dpi=150, bbox_inches="tight")`. The cell shows the chart and prints `y-axis runs from 0.0 to 100.0`.

> **Checkpoint: `output/critique_redesign.png`**
> Your redesigned chart as a PNG image.

## Task 3: Explain one finding to the program coordinator

The rehab program coordinator is deciding where to add follow-up support. Your explanatory chart shows how often patients in each program met the weekly exercise goal at the four follow-up visits.

### 3.1 Write the contract and save the supporting data

In the Task 3.1 cell:

1. Read `followup = pd.read_csv(FOLLOWUP_PATH)` and print it.
2. Write the chart's contract as five strings: `question`, `audience` (who reads the chart and what they will use it for), `intended_claim` (the one descriptive conclusion the chart supports), `displayed_unit` (what the y-axis measures, with its unit), and `grain` (what one row of the plotting table and one point on a line represent).
3. Fill in `data_types` with each plotted column's data type, in Lecture 07's words: `categorical`, `quantitative`, `ordinal`, or `temporal`. `visit_number` is the order of the visits, not a date.
4. Select the three plotted columns, `program`, `visit_number`, and `goal_met_pct`, into `supporting_data`, and save it to `SUPPORTING_DATA_PATH` with `index=False`.

The cell reads the file back and prints `shape: (8, 3)` and `same as saved: True`.

> **Checkpoint: `output/explanatory_supporting_data.csv`**
> The header line `program,visit_number,goal_met_pct`, then the eight rows of `data/followup_goals.csv` without `patients_seen`, starting with `Home-based,1,58`.

### 3.2 Draw the explanatory chart

The Task 3.2 cell supplies `home_goals` and `center_goals`, each program's four rows. Draw one line per program on `explanatory_ax`, as Lecture 07's "Redundant Cues on a Line Chart" snippet does:

- `visit_number` on x and `goal_met_pct` on y;
- each program with its own color, marker, and line style, such as `ORANGE`, `'s'`, and `'--'` for Home-based and `BLUE`, `'o'`, and `'-'` for Center-based, named by a direct label at its right end or by a legend;
- one annotation whose arrow points to Center-based's visit-4 point, `(4, 79)`, and whose text states the gap at visit 4: 79% against 70%, 9 points;
- axis labels with units, such as `Follow-up visit` and `Patients meeting the exercise goal (%)`, ticks at visits 1 to 4, and a title that states the finding without claiming a cause;
- the top and right spines hidden.

Save with `explanatory_fig.savefig(EXPLANATORY_PATH, dpi=150, bbox_inches="tight")`.

> **Checkpoint: `output/explanatory_chart.png`**
> Your explanatory line chart as a PNG image.

### 3.3 Write the text alternative and save the evidence

In the Task 3.3 cell:

1. Write `text_alternative`, one paragraph that names the chart type, both axes and their units, both programs, how each changes from visit 1 to visit 4, the gap at visit 4, and a limitation: patients chose their program, so the gap does not show that one program causes more exercise. Lecture 07's "Make the chart accessible" section has an example.
2. Add the keys `question`, `audience`, `intended_claim`, `displayed_unit`, `grain`, `data_types`, and `text_alternative` to `visualization_evidence`, each holding the variable of the same name. `critique` is already there.
3. Save `visualization_evidence` to `EVIDENCE_PATH` with `json.dump(..., indent=2, ensure_ascii=False)`, as the Task 2.2 cell does. This replaces the Task 2.2 file and keeps its critique.
4. Write `text_alternative` to `TEXT_ALTERNATIVE_PATH` with `open(..., "w", encoding="utf-8")` and `file.write()`.

The cell reads the JSON back and prints its eight keys.

> **Checkpoint: `output/visualization_evidence.json`**
> An object with the eight keys `critique`, `question`, `audience`, `intended_claim`, `displayed_unit`, `grain`, `data_types`, and `text_alternative`. `data_types` maps `program`, `visit_number`, and `goal_met_pct` to their data types.

> **Checkpoint: `output/explanatory_text_alternative.txt`**
> The same paragraph as `text_alternative` in the JSON.

### 3.4 Review the saved charts

Open `output/critique_redesign.png` and `output/explanatory_chart.png`, and answer the questions in the Markdown cell under 3.4 with what you see in them. Task 3.4 has no file of its own to check.

## Check your work

Click **Restart**, then **Run All**. The last cell prints `Fresh-run check passed`, or names the task to fix. Then, with the environment active, run the checks from the assignment directory:

```bash
python check_assignment.py
```

`check_assignment.py` runs the same checks GitHub runs. They read only the six files in `output/` and compare them with the supplied data. They never run or read your notebook, so any way of producing correct files counts.

Each check prints `PASS` or `FIX` and the points it earned, and a `FIX` says what to fix on the line beneath it. When the next checks need the same fix, such as a missing file, they say `(same fix as above)`. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/4  exploratory spec: point mark
         output/exploratory_spec.json is missing; run the Task 1.1 cell to write it, then commit it.
```

Below the score, `Left to fix` lists the checks still failing and the points they are worth. Fix what they name, rerun the notebook and then the checks, and repeat until every check passes. A clean run ends with:

```text
[PASS]  2/2  text alternative file

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points.
- Spacing, line endings, quoting, key order, and column order never cost points, and extra keys in the JSON files are ignored.
- Numbers are compared as numbers, so `79`, `79.0`, and `79.00` are the same value.
- Program names, categories, keys, column names, and data types are compared in any letter case. A data type may carry a note, as in `ordinal (visit order)`. `nominal` and `qualitative` count as categorical, `numeric` and `continuous` as quantitative, and `ordered` as ordinal, alone or beside categorical, as in `categorical (ordered)`.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- The exploratory spec is compared on the three plotted columns only, so leaving out `patient_id` or adding a column costs nothing.
- The PNG checks confirm that each chart was saved as a PNG image, and the text checks confirm that each answer is filled in. The wording and the look are yours, so open both PNG files and check them against Tasks 2.3 and 3.2 yourself.

Every push also runs GitHub Actions, which downloads the course's current copy of the checks and reruns them on the files you committed and pushed. That run is what counts, and a check corrected after handout reaches you there on your next push.

### Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/exploratory_spec.json` | Its mark is `point`. | exploratory spec: point mark | 4 |
| `output/exploratory_spec.json` | Its embedded rows hold the twelve patients' `program`, `sessions_attended`, and `walk_distance_m`. | exploratory spec: embedded patient rows | 5 |
| `output/exploratory_spec.json` | It encodes `sessions_attended` as quantitative `x`. | exploratory spec: x encoding | 4 |
| `output/exploratory_spec.json` | It encodes `walk_distance_m` as quantitative `y`. | exploratory spec: y encoding | 4 |
| `output/exploratory_spec.json` | It encodes `program` as nominal `color`. | exploratory spec: color encoding | 4 |
| `output/exploratory_spec.json` | It encodes `program` as nominal `shape`. | exploratory spec: shape encoding | 4 |
| `output/critique_redesign.png` | It is a PNG image. | critique redesign: PNG image | 12 |
| `output/visualization_evidence.json` | Its `critique` has an `unsupported claim` entry with a `problem` and a `repair`. | critique: unsupported claim | 5 |
| `output/visualization_evidence.json` | Its `critique` has a `truncated baseline` entry with a `problem` and a `repair`. | critique: truncated baseline | 5 |
| `output/visualization_evidence.json` | Its `critique` has a `missing unit` entry with a `problem` and a `repair`. | critique: missing unit | 5 |
| `output/visualization_evidence.json` | Its `critique` has a `color-only encoding` entry with a `problem` and a `repair`. | critique: color-only encoding | 5 |
| `output/visualization_evidence.json` | Its `critique` has a `distracting decoration` entry with a `problem` and a `repair`. | critique: distracting decoration | 5 |
| `output/explanatory_chart.png` | It is a PNG image. | explanatory chart: PNG image | 8 |
| `output/explanatory_supporting_data.csv` | Its columns are `program`, `visit_number`, and `goal_met_pct`. | supporting data: columns | 4 |
| `output/explanatory_supporting_data.csv` | It holds the eight rows of `data/followup_goals.csv`, unchanged. | supporting data: rows and values | 6 |
| `output/visualization_evidence.json` | Its `question` is filled in. | evidence: question | 2 |
| `output/visualization_evidence.json` | Its `audience` is filled in. | evidence: audience | 2 |
| `output/visualization_evidence.json` | Its `intended_claim` is filled in. | evidence: intended_claim | 2 |
| `output/visualization_evidence.json` | Its `displayed_unit` is filled in. | evidence: displayed_unit | 2 |
| `output/visualization_evidence.json` | Its `grain` is filled in. | evidence: grain | 2 |
| `output/visualization_evidence.json` | Its `text_alternative` is filled in. | evidence: text_alternative | 2 |
| `output/visualization_evidence.json` | Its `data_types` gives `program` as categorical. | data type: program | 2 |
| `output/visualization_evidence.json` | Its `data_types` gives `visit_number` as ordinal. | data type: visit_number | 2 |
| `output/visualization_evidence.json` | Its `data_types` gives `goal_met_pct` as quantitative. | data type: goal_met_pct | 2 |
| `output/explanatory_text_alternative.txt` | It holds the same text as `text_alternative` in the JSON. | text alternative file | 2 |

Extra files are ignored.

## Submit

Before you commit a notebook, follow Lecture 04's "Before You Commit a Notebook": click **Clear All Outputs**, then save. In VS Code Source Control, stage `assignment.ipynb` and all six files in `output/`. Commit with `Complete Assignment 07 charts` and select **Sync Changes**. Keep `.venv/` out of the commit; `.gitignore` already lists it.

Confirm the notebook and the six output files on `main` in the repository browser. GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If a run cannot download the course's current checks, it grades with the copy in your repository and says so in its log. If your local run and the GitHub run ever disagree, the GitHub run counts, because it uses the course's current checks. If a required VS Code control is unavailable, record its message and contact the instructor.
