# Assignment 04: Vaccine Fridges and a Clinic Supply Order

## Files

```text
assignment/
├── assignment.ipynb        # the notebook you complete
├── data/supply_order.csv   # supplied order lines; keep this file exactly as handed out
├── requirements.txt        # supplied: numpy, pandas, and ipykernel
├── .python-version         # supplied: tells uv to use Python 3.13
├── check_assignment.py     # supplied: run it to check your work; keep unchanged
├── grading.py, _value_checks.py  # supplied: the checks themselves; keep unchanged
├── test_assignment.py, .github/  # supplied: run the checks on GitHub; keep unchanged
└── output/
    ├── fridge_block.csv        # you generate in Task 2
    └── selected_supplies.csv   # you generate in Task 3
```

## The data

Both datasets are synthetic and come from one primary-care clinic.

- **Vaccine fridge log** (Task 2). Vaccines must stay between 2 and 8 °C, so staff read each vaccine refrigerator's thermometer at the start and the end of the day. The notebook supplies four fridges' readings as a NumPy array: one row per fridge, with the morning reading (`am_temp_c`) and the afternoon reading (`pm_temp_c`) in °C.
- **Supply order** (Task 3). `data/supply_order.csv` is one week's order of exam-room supplies. Each row is one order line: a catalog `item_id`, the `item` description, the `quantity` ordered, and the `unit_price_usd` in US dollars.

```text
item_id,item,quantity,unit_price_usd
C3150,Nitrile exam gloves (box of 100),6,9.50
C1022,Blood pressure cuff (adult),1,24.00
```

## Setup

Fork the assignment repository on GitHub and clone your fork the way Lecture 01 did: Command Palette → **Git: Clone**, paste your fork's URL, pick a folder, and open it. Then open **Terminal → New Terminal** in VS Code at the assignment directory (Ctrl+Shift+backtick, also Control on Mac). If you use a native terminal or WSL Ubuntu instead, `cd` into the assignment directory first. Run `ls data` and expect `supply_order.csv`. This clone is a new repository, so before your first commit run Lecture 02's two `git config user.name "..."` and `git config user.email "..."` lines in this terminal, with your name and GitHub noreply email.

Create the project environment, activate it, and install the supplied requirements, as in Lecture 03:

```bash
uv venv --seed --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` includes **ipykernel**, the package that lets a notebook run on this environment's Python (Lecture 04), so this one install is all the notebook needs. Open `assignment.ipynb`, click **Select Kernel** at the top right, and choose the Python inside this project's `.venv`. If VS Code offers to install the **Jupyter** extension, accept.

> **Windows:** work in the **WSL: Ubuntu** window from Lecture 01's setup. In PowerShell instead, activate with `.\.venv\Scripts\Activate.ps1`.

Run the notebook's first code cell. It prints the NumPy and pandas versions and `data file found: True`. `False` means the notebook is not running from the assignment directory, so open the folder itself in VS Code, not a folder above it.

## Task 1: Put the cells in running order

### 1.1 Reorder the cells

The first Task 1 code cell uses `vials_on_hand`, which the second cell defines, so after a restart the first cell stops with `NameError: name 'vials_on_hand' is not defined`. Move the whole producer cell above the dependent cell, as in Lecture 04's "A producer cell and a dependent cell" snippet: in VS Code, drag it by the bar to the left of its code, or click into it, press `Esc` for command mode, and press `Alt+Up` (`Option+Up` on Mac). Do not copy the definition into another cell. After **Restart**, then **Run All**, the two cells print:

```text
vials_on_hand: 3
doses_available: 30
```

### 1.2 Explain the repair

Replace the TODO in the Markdown cell under 1.2 with a few sentences that explain:

- the difference between the order the cells appear in and the order the kernel ran them;
- why the dependent cell can print a result while it sits above the cell it needs;
- why output saved under a cell does not prove the notebook works now; and
- what you changed, and how Restart and Run All shows that it worked.

Task 1 has no file of its own to check, but **Run All** stops at the `NameError` until 1.1 is done, so Tasks 2 and 3 depend on it.

## Task 2: Label the fridge readings

### 2.1 Build the Series and DataFrame

In the Task 2.1 cell, keep the supplied arrays and replace each `None`:

- `latest_by_area`: a Series from `latest_values`, the latest fridge reading in each clinic area in °C, with the index `pharmacy`, `pediatrics`, `family_medicine`, `urgent_care` and the name `temp_c`.
- `fridge_log`: a DataFrame from `fridge_readings`, with the index `FRG-101`, `FRG-102`, `FRG-103`, `FRG-104` and the columns `am_temp_c`, `pm_temp_c`. Then name its index: `fridge_log.index.name = "fridge_id"`.
- `am_series`: the `am_temp_c` column selected with one label in brackets, which gives a Series.
- `am_table`: the same column selected with a list of one label, which gives a one-column DataFrame.

The cell then prints `fridge_log`:

```text
           am_temp_c  pm_temp_c
fridge_id                      
FRG-101          4.1        5.6
FRG-102          3.8        6.2
FRG-103          5.0        7.4
FRG-104          2.9        4.4
```

It also prints `shape: (4, 2)`, two `float64` columns, `<class 'pandas.Series'>`, and `<class 'pandas.DataFrame'>`.

### 2.2 Select two fridges and save them

Select the rows `FRG-102` through `FRG-103` and both columns twice, as Lecture 04's "Compare label and position selection" snippet does:

- `label_block`: by label with `.loc`, as in `fridge_log.loc["FRG-102":"FRG-103", ["am_temp_c", "pm_temp_c"]]`. A label slice includes its end label.
- `position_block`: the same block by position with `.iloc[1:3, 0:2]`. A position slice stops before its end position.

The cell prints `same block: True` when the two match. Then save `label_block` to `FRIDGE_OUTPUT_PATH` with `to_csv()`. Keep the index this time: the fridge IDs are meaningful labels, so leave out `index=False`.

> **Checkpoint: `output/fridge_block.csv`**
> The header line `fridge_id,am_temp_c,pm_temp_c`, then one line each for `FRG-102` (3.8 and 6.2 °C) and `FRG-103` (5.0 and 7.4 °C).

## Task 3: Select, total, and sort the supply order

The clinic manager wants to review every order line that buys two or more units, largest dollar amount first.

### 3.1 Read, select, and add the line total

In the Task 3.1 cell:

1. Read the order with `supplies = pd.read_csv(DATA_PATH)`. Print its shape and dtypes and display its first rows; expect `(12, 4)`.
2. Build the mask `quantity_at_least_two = supplies["quantity"] >= 2`.
3. Select the rows where the mask is `True` and the four columns `item_id`, `item`, `quantity`, `unit_price_usd` in one `.loc`, and end with `.copy()`, since the next step adds a column to it. Name the result `selected_supplies`.
4. Add the derived column `line_total_usd`, the line's `quantity` times its `unit_price_usd`.

The cell prints `selected lines: 9`.

### 3.2 Sort, save, and read back

In the Task 3.2 cell:

1. Sort `selected_supplies` by `line_total_usd` from highest to lowest, and break ties with `item_id` from A to Z: `by=["line_total_usd", "item_id"]` with `ascending=[False, True]`. Four pairs of lines tie (at 57.00, 36.00, 24.00, and 13.00 dollars), so the second key decides their order. Assign the sorted result back to `selected_supplies`.
2. Save it to `SUPPLIES_OUTPUT_PATH` with `index=False`: its index holds only the row numbers the lines had in `supplies`.
3. Read the file back into `round_trip` with `pd.read_csv()`. The cell prints `round-trip shape: (9, 5)`.

> **Checkpoint: `output/selected_supplies.csv`**
> The header line `item_id,item,quantity,unit_price_usd,line_total_usd`, then nine order lines in this order of `item_id`: C1833, C3150, C3012, C4105, C2210, C2877, C2655, C2904, C2318. The first line is `C1833,Specimen cups (case of 100),2,28.5,57.0`.

## Check your work

Click **Restart**, then **Run All**. The last cell prints `Fresh-run check passed`, or names the task to fix. Then, with the environment active, run the checks from the assignment directory:

```bash
python check_assignment.py
```

`check_assignment.py` runs the same checks GitHub runs. They read only the two CSV files in `output/` and compare them with the supplied readings and order. They never run or read your notebook, so any way of producing correct files counts.

Each check prints `PASS` or `FIX` and the points it earned, and a `FIX` says what to fix on the line beneath it. When the next checks need the same fix, they say `(same fix as above)`. Before Task 2, for example, the fridge block checks report:

```text
[FIX ]  0/10 fridge block: fridge_id index column
         output/fridge_block.csv is missing; run the Task 2 cells to write it, then commit it.
[FIX ]  0/10 fridge block: rows FRG-102 and FRG-103  (same fix as above)
```

Below the score, `Left to fix` lists the checks still failing and the points they are worth. Fix what they name, rerun the notebook and then the checks, and repeat until every check passes. A clean run ends with:

```text
[PASS]  4/4  selected supplies: ties in item_id order

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points.
- Spacing, line endings, quoting, and column order never cost points.
- Numbers are compared as numbers, so `57`, `57.0`, and `57.00` are the same value.
- IDs, item descriptions, and column names are compared in any letter case.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.

Every push also runs GitHub Actions, which downloads the course's current copy of the checks and reruns them on the files you committed and pushed. That run is what counts, and a check corrected after handout reaches you there on your next push.

### Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/fridge_block.csv` | It has a `fridge_id` column holding the saved index. | fridge block: fridge_id index column | 10 |
| `output/fridge_block.csv` | Its rows are `FRG-102` and `FRG-103`, and no others. | fridge block: rows FRG-102 and FRG-103 | 10 |
| `output/fridge_block.csv` | Its `am_temp_c` values match the supplied array: 3.8 and 5.0. | fridge block: am_temp_c values | 10 |
| `output/fridge_block.csv` | Its `pm_temp_c` values match the supplied array: 6.2 and 7.4. | fridge block: pm_temp_c values | 10 |
| `output/selected_supplies.csv` | It has an `item_id` column. | selected supplies: item_id column | 2 |
| `output/selected_supplies.csv` | It has an `item` column. | selected supplies: item column | 2 |
| `output/selected_supplies.csv` | It has a `quantity` column. | selected supplies: quantity column | 2 |
| `output/selected_supplies.csv` | It has a `unit_price_usd` column. | selected supplies: unit_price_usd column | 2 |
| `output/selected_supplies.csv` | It has a `line_total_usd` column. | selected supplies: line_total_usd column | 2 |
| `output/selected_supplies.csv` | It has no columns beyond those five. | selected supplies: no extra columns | 2 |
| `output/selected_supplies.csv` | Line `C1833` is there with its supplied item, quantity, and unit price, and its `line_total_usd` is quantity times unit price. | selected supplies: line C1833 | 4 |
| `output/selected_supplies.csv` | Line `C3150` is there with its supplied item, quantity, and unit price, and its `line_total_usd` is quantity times unit price. | selected supplies: line C3150 | 4 |
| `output/selected_supplies.csv` | Line `C3012` is there with its supplied item, quantity, and unit price, and its `line_total_usd` is quantity times unit price. | selected supplies: line C3012 | 4 |
| `output/selected_supplies.csv` | Line `C4105` is there with its supplied item, quantity, and unit price, and its `line_total_usd` is quantity times unit price. | selected supplies: line C4105 | 4 |
| `output/selected_supplies.csv` | Line `C2210` is there with its supplied item, quantity, and unit price, and its `line_total_usd` is quantity times unit price. | selected supplies: line C2210 | 4 |
| `output/selected_supplies.csv` | Line `C2877` is there with its supplied item, quantity, and unit price, and its `line_total_usd` is quantity times unit price. | selected supplies: line C2877 | 4 |
| `output/selected_supplies.csv` | Line `C2655` is there with its supplied item, quantity, and unit price, and its `line_total_usd` is quantity times unit price. | selected supplies: line C2655 | 4 |
| `output/selected_supplies.csv` | Line `C2904` is there with its supplied item, quantity, and unit price, and its `line_total_usd` is quantity times unit price. | selected supplies: line C2904 | 4 |
| `output/selected_supplies.csv` | Line `C2318` is there with its supplied item, quantity, and unit price, and its `line_total_usd` is quantity times unit price. | selected supplies: line C2318 | 4 |
| `output/selected_supplies.csv` | It holds no line with quantity 1, and no line twice. | selected supplies: no other lines | 4 |
| `output/selected_supplies.csv` | Its lines run from the highest `line_total_usd` to the lowest. | selected supplies: highest line total first | 4 |
| `output/selected_supplies.csv` | Lines with the same `line_total_usd` are in `item_id` order. | selected supplies: ties in item_id order | 4 |

Extra files are ignored.

## Submit

Before you commit a notebook, follow Lecture 04's "Before You Commit a Notebook": click **Clear All Outputs**, then save. In VS Code Source Control, stage `assignment.ipynb` and both files in `output/`. Commit with `Complete Assignment 04 notebook` and select **Sync Changes**. Keep `.venv/` out of the commit; `.gitignore` already lists it.

Confirm the notebook and both CSV files on `main` in the repository browser. GitHub Actions runs the checks automatically on every push; enable Actions once if GitHub prompts you in a fork. If a run cannot download the course's current checks, it grades with the copy in your repository and says so in its log. If your local run and the GitHub run ever disagree, the GitHub run counts, because it uses the course's current checks. If a required VS Code control is unavailable, record its message and contact the instructor.
