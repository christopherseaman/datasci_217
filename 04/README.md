---
notion:
  title_line: "# Pandas on Jupyter: Data Structures & I/O"
  role: lecture
  status: mapped
  page_id: "281d9fdd-1a1a-800a-897d-cafb5971c23f"
  url: "https://app.notion.com/p/281d9fdd1a1a800a897dcafb5971c23f"
---

# Pandas on Jupyter: Data Structures & I/O

See [BONUS.md](BONUS.md) for the optional extensions.

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/04/demo/demo1_jupyter_basics.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/04/demo/demo2_pandas_basics.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/04/demo/demo3_data_io.ipynb)

**Run locally:**

```shell
curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/04/demo/setup_demo.sh | sh
cd ~/04-demo
uv venv --seed
source .venv/bin/activate
uv sync
```

Then open the `04-demo` folder in VS Code and choose its `.venv` as the notebook kernel.

This lecture covers:

- McKinney, _Python for Data Analysis_ (3rd ed.): 2.2 (running the Jupyter notebook), 5.1 (Series and DataFrame), 5.2 (dropping columns; indexing, selection, and filtering, including pitfalls with chained indexing; sorting), 5.3 (descriptive statistics, unique values, and value counts), 6.1 (reading and writing CSV files), 6.2 (Parquet), 7.2 (removing duplicates with `duplicated()`), and Appendix B.2 and B.5 (magic commands, and timing code with `%timeit`)

# Jupyter Notebooks: Interactive Data Analysis

A **Jupyter notebook** (`.ipynb` file) is a document of **cells**: **code cells** run Python and show each result beneath the cell, and **Markdown cells** hold notes. The code runs in a **kernel**, a Python process that keeps values between cells, so a clinic's data loads once and can be inspected, fixed, and rechecked beside the notes that explain it.

## Opening and Running a Notebook

Assignments run notebooks in VS Code on your computer. Demos also open in **Google Colab**, a free notebook service in the browser whose kernel runs on a Google machine called a **runtime**; files made there disappear when the runtime shuts down.

![VS Code notebook: add code or Markdown, run cells, run all, and select a kernel](media/vscode-jupyter-kernel-picker.png)

In VS Code, open the `.ipynb` file (install the **Jupyter** extension if VS Code offers it), click **Select Kernel** (top right), and choose the Python in the project's `.venv`. That environment also needs **ipykernel**, the package that lets Jupyter start a kernel from it: `uv add ipykernel` in the project folder installs it and records it in `pyproject.toml`, and a project that already lists it, such as an assignment handout, needs only `uv sync` (Lecture 03). **Run All** runs every cell in order; **Clear All Outputs** erases the results saved under the cells.

<callout icon="⚠️" color="yellow_bg">
	## The kernel, not the terminal, picks a notebook's Python
	`ModuleNotFoundError: No module named 'pandas'` in a notebook means its kernel runs a Python without pandas, even when the terminal's environment is active. Click the kernel name at the top right and choose the project's `.venv`; if the error stays, run `uv sync` in the project folder.
</callout>

### Reference Card: Notebook controls

| Task | VS Code | Colab | Result |
| --- | --- | --- | --- |
| Create a notebook | Command Palette → **Create: New Jupyter Notebook** | **File → New notebook in Drive** | New `.ipynb` file |
| Run a cell | `Shift+Enter` (run, move on) or `Ctrl+Enter` (run, stay) | Same keys | Output appears below the cell |
| Add a cell | **+ Code** / **+ Markdown** | **+ Code** / **+ Text** | New cell |
| Delete a cell | Trash icon, or `DD` in command mode | Trash icon | Cell removed |
| Move a cell | Drag the bar at the cell's left, or `Alt+Up` / `Alt+Down` in command mode (`Option` on Mac) | `Ctrl+M K` (up) / `Ctrl+M J` (down), or the arrows on the cell's toolbar | Cell moves, code and output together |
| Prepare an environment | `uv add ipykernel` in the project, once; `uv sync` when `pyproject.toml` already lists it | Nothing to do | The environment can run notebook cells |
| Choose Python | **Select Kernel** | Managed for you by the runtime | Which interpreter runs the cells |
| Keep your changes | `Ctrl+S` (`Cmd+S` on macOS) | **File → Save a copy in Drive** | Edits saved; Colab does not save back to the course repository |

Shortcuts such as `A` (add a cell above), `B` (below), `DD` (delete), and `Alt+Up` (move up) work only in **command mode**: press `Esc` so the cell is selected but not being edited. In Colab, press `Ctrl+M` first, then the letter.

### Code Snippet: A notebook cell

```python
# Cell 1: the kernel keeps these values
patient_id = "P001"
temps_c = [36.8, 37.4, 38.1]
```

```python
# Cell 2: a later cell can use them, and its output appears below it
average = sum(temps_c) / len(temps_c)
print(f"{patient_id} average temperature: {average:.1f} °C")
```

```text
P001 average temperature: 37.4 °C
```

### Alternative: JupyterLab

JupyterLab is Jupyter's own browser interface, started with `jupyter lab` in an activated environment that lists `jupyterlab`. It shows the same cells, each with a run number such as `[4]` beside it and its output beneath.

![JupyterLab: file browser at left, notebook cells and output in the center](media/jupyterlab-interface.png)

## Kernel State and Execution Order

The kernel's **state** is every name and value it currently holds: running a cell changes it, and editing a cell without running it does not. The number beside a cell, such as the `[4]` in the JupyterLab screenshot, is its **execution count**: the order the kernel actually ran it, which follows your clicks rather than the page, so a notebook can look correct and still depend on a cell you later changed or deleted. The result saved under a cell is **stored output**: a record of its last run, not proof that the notebook works now.

| Step | You do | Execution count | Output under the cell |
| --- | --- | --- | --- |
| 1 | Run `days = 12` and `doses_per_day = 2` | `[1]` | none |
| 2 | Run `total_doses = days * doses_per_day` and `print(total_doses)` | `[2]` | `24` |
| 3 | Edit the first cell to `doses_per_day = 3` but do not run it | still `[1]` | `24`, now stale |
| 4 | Restart & Run All | `[1]`, `[2]` | `36` |

### Reference Card: Kernel actions

| Action | Where | When to use it | Result |
| --- | --- | --- | --- |
| Interrupt | VS Code: **Interrupt** (■); Colab: **Runtime → Interrupt execution** | A cell runs far longer than expected; the notebook's Ctrl+C from Lecture 01 | Stops the cell; state is kept |
| Restart | VS Code: **Restart**; Colab: **Runtime → Restart session** | Values look wrong, the kernel is stuck, or "it worked before but now it doesn't" | Empty state; cells and stored output stay on the page |
| Run All | VS Code: **Run All**; Colab: **Runtime → Run all** | After a restart | Every cell runs top to bottom |
| Restart & Run All | VS Code: **Restart**, then **Run All**; Colab: **Runtime → Restart session and run all** | Before you commit or submit | Shows the notebook works from a fresh start |

### Code Snippet: A producer cell and a dependent cell

```python
# Cell 1 (producer): defines names
days = 12
doses_per_day = 3
```

```python
# Cell 2 (dependent): needs the names from Cell 1
total_doses = days * doses_per_day
print("total doses:", total_doses)  # total doses: 36
```

If Cell 2 sits above Cell 1, Restart & Run All stops with `NameError: name 'days' is not defined`. Fix it by moving the producer cell above the dependent cell (**Move a cell** in the controls card), not by copying the definition into another cell.

![xkcd 2200: Unreachable State. Cells run out of order can leave the kernel in a state no top-to-bottom run would reach, and Restart & Run All brings it back](media/xkcd_2200.png)

## Jupyter Magic Commands

**Magic commands** are notebook-only shortcuts that start with `%`. `%pwd` and `%ls` work like the Lecture 01 shell commands, showing where the notebook runs and which files it can see, so check them first when a notebook cannot find a file. `%timeit` times one line of Python by running it many times, and `%pip install` installs a package into the kernel's environment.

### Reference Card: Magic commands

| Command | Arguments | Typical output / effect |
| --- | --- | --- |
| `%pwd` | None | Current working directory, as a quoted string |
| `%ls` | None | Directory contents |
| `%timeit expression` | Python expression | Timing summary |
| `%pip install -q --no-warn-conflicts pandas==3.0.5` | Package and exact version; `-q` prints less; `--no-warn-conflicts` skips warnings about other installed packages | Package installed into the kernel's environment; restart the kernel if it was already imported |
| `%pip show package_name` | Package name | Installed version and location |

### Code Snippet: Where is the notebook running?

A cell shows the value of its last line only, so give `%pwd` its own cell:

```python
%pwd
```

```text
'/content'
```

That is Colab's working directory; in VS Code, it is usually the notebook's folder.

```python
%timeit sum(range(100))
```

```text
693 ns ± 3.46 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
```

Times vary by machine.

### Code Snippet: Install a package into the kernel

Each demo notebook's first cell installs the course's pandas, since Colab ships an older one:

```python
%pip install -q --no-warn-conflicts pandas==3.0.5
```

```text
Note: you may need to restart the kernel to use updated packages.
```

A package that was already imported keeps its old version until the kernel restarts, which is what the note means. If Colab asks you to restart after the install, choose **Runtime → Restart session** and run the notebook from the top; the install then finishes at once. `--no-warn-conflicts` hides pip's complaint that Colab's own `google-colab` package wants an older pandas; the demos do not use it.

<callout icon="💡" color="blue_bg">
	## On your computer, add packages with `uv add`
	`%pip install` suits Colab. In a local project, run `uv add` in the terminal instead: it records the package in `pyproject.toml`, while `uv sync` removes any package that `pyproject.toml` does not list (Lecture 03). Locally, the demos' `%pip` cell runs the pip that `uv venv --seed` put in `.venv` and changes nothing, since `uv sync` already installed pandas 3.0.5; if `.venv` was made without `--seed`, it prints `No module named pip` instead, which is just as harmless.
</callout>

_Think of magic commands as the Konami code of Jupyter: instead of 30 extra lives, you get shell shortcuts and a stopwatch._

## Notebooks vs Scripts

The same Python runs in a notebook cell and in a `.py` script, but four things behave differently:

| | Notebook cell | `.py` script |
| --- | --- | --- |
| A bare last expression, such as `len(temps_c)` | Shown below the cell automatically | Shown nowhere |
| `display(table)` | Draws a table (pandas DataFrame, next topic) as a formatted grid, like the JupyterLab screenshot (a Series stays plain text); `print()` shows plain text | Unavailable; use `print()` |
| Magic commands (`%pwd`, `%ls`, `%pip`, `%timeit`) | Work | `SyntaxError` |
| Values from earlier runs | Kept by the kernel until it restarts | Every run starts fresh |

Scripts suit analyses that rerun unattended; notebooks suit exploring and explaining. BONUS.md shows how to run a whole notebook from the terminal.

### Code Snippet: Three ways to show a result

```python
print(temps_c)    # Plain text, works everywhere: [36.8, 37.4, 38.1]
display(temps_c)  # Same list; a DataFrame would draw as a formatted table
max(temps_c)      # Last line: shown automatically as 38.1
```

A cell shows only its last line's value automatically, so anything earlier needs `print()` or `display()`.

_Think of `print()` as the reliable Honda Civic that works almost anywhere, while `display()` is the sports car: prettier, but happiest in Jupyter._

## Notebook Outputs and Git

<callout icon="⚠️" color="yellow_bg">
	## Outputs are saved in the notebook file!
	A notebook saves each cell's output inside the `.ipynb` file, next to the code. Anything a cell printed, such as a patient's name or a password, is committed with the notebook and stays in Git history.
</callout>

_Notebooks are like that one friend who screenshots everything you text them._

### Code Snippet: What Git actually commits

```python
patient_name = "Example Patient"
blood_pressure = "120/80"
print(patient_name, blood_pressure)  # Example Patient 120/80
```

Open the `.ipynb` file in a text editor and that line is right there, in the file Git will commit:

```json
{"cell_type": "code",
 "execution_count": 1,
 "outputs": [{"name": "stdout", "output_type": "stream",
              "text": ["Example Patient 120/80\n"]}]}
```

### Before You Commit a Notebook

For course assignments, keep the requested outputs from synthetic data as evidence of your results. For a notebook whose outputs should not be shared:

1. **Clear all outputs**: click **Clear All Outputs** in VS Code (Colab: **Edit → Clear all outputs**).
2. **Check for sensitive data**: make sure no personal information, passwords, or confidential data is visible.
3. **Save the notebook**: the outputs are removed from the file.

Then check the notebook's diff in VS Code Source Control (Lecture 02) before you commit.

> Never be afraid to make a mistake. Unless it's in Git. Then be afraid. Be very afraid.

# LIVE DEMO!

[Open Demo 1 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/04/demo/demo1_jupyter_basics.ipynb)

# Introduction to Pandas

**pandas** is the Python library for labeled tables, built on NumPy. A clinic's visit table mixes text IDs, whole-number ages, and decimal temperatures, and pandas keeps them side by side, so you ask for patient P002's temperature by label rather than by position, and each patient's values stay together when you sort or filter.

Import it with `import pandas as pd`. Every output below comes from pandas 3.0.5, the course version; pandas 2 prints some results differently.

_Fun fact: the name comes from **panel data**, an econometrics term for datasets that follow the same subjects over time (think of a longitudinal cohort study), and it is also a play on "Python data analysis." No bears were involved. 🐼_

![xkcd 2180: Spreadsheets. A spreadsheet quietly grows into a program; pandas lets you write the real code instead](media/xkcd_2180.png)

## Series and DataFrames

- A **Series** is one column of values plus an **index**, a label for each value, like a dictionary (Lecture 02) whose keys stay in order.
- A **DataFrame** is a table whose columns share one row index. Each column is a Series with its own **dtype** (data type), so text, numbers, and `True`/`False` sit side by side.

```text
Series temp_c         DataFrame visits
index  value          index  age  temp_c  smoker   <- column labels
P001   36.8           P001    34    36.8   False
P002   38.1           P002    58    38.1    True
P003   37.2           P003    41    37.2   False
                                  ^ the temp_c column is itself a Series
```

### Reference Card: Series attributes and methods

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `pd.Series(data, index=None, name=None)` | `data` may be a list, a NumPy array, or a dict (its keys become the index); optional labels and name | New `Series` |
| `series.index` | Access index labels | `Index(['P001', 'P002', 'P003'], dtype='str')` |
| `series.values` | Access underlying values without labels | NumPy array or extension array, depending on dtype |
| `series.name` | Get/set Series name | Series name |
| `series.dtype` | Get data type | Data type |
| `series.size` | Number of elements | Integer element count |
| `series.head(n=5)` | First n elements | `Series` with original labels |
| `series.tail(n=5)` | Last n elements | `Series` with original labels |
| `series.describe()` | Summarize values according to dtype | Statistics as a `Series` |

### Code Snippet: Create and inspect a Series

```python
temp_c = pd.Series([36.8, 38.1, 37.2], index=["P001", "P002", "P003"], name="temp_c")
print(temp_c)
print(temp_c["P002"])  # 38.1
```

```text
P001    36.8
P002    38.1
P003    37.2
Name: temp_c, dtype: float64
38.1
```

_Think of Series inside DataFrames like Russian nesting dolls: one labeled column fits inside the larger labeled table._

### Reference Card: DataFrame attributes and methods

| Item | Purpose / arguments | Output / note |
| --- | --- | --- |
| `pd.DataFrame(data, index=None, columns=None)` | Supply data, such as a dict of lists, and optional row/column labels | New `DataFrame` |
| `pd.DataFrame(array_2d, index=[...], columns=[...])` | Build from a 2D NumPy array, as in Lecture 03 | Labeled `DataFrame` |
| `df.index.name = "patient_id"` | Label the row index | Shown above the index; becomes the index column's header when you save a CSV |
| `df.index` | Access row index | Index labels |
| `df.columns` | Access column names | Column labels |
| `df.values` | Get values as NumPy array | NumPy array of values |
| `df.shape` | Count rows and columns | `(rows, columns)` tuple |
| `df.dtypes` | Data types per column | Series of column dtypes |
| `df.info()` | Inspect columns, **non-null counts** (values present, not missing), dtypes, and memory | Prints a summary; returns `None` |
| `df.describe()` | Summarize numeric columns | count, mean, std, min, quartiles, max per column; `std` is the sample SD (`ddof=1`, divides by n − 1), unlike NumPy's `.std()` default (`ddof=0`) |
| `df.head(n=5)` | First n rows | `DataFrame` with original labels |
| `df.tail(n=5)` | Last n rows | `DataFrame` with original labels |
| `df.sample(n=5)` | Random n rows | Random sample DataFrame |

### Code Snippet: Create and inspect a DataFrame

```python
visits = pd.DataFrame(
    {"age": [34, 58, 41], "temp_c": [36.8, 38.1, 37.2], "smoker": [False, True, False]},
    index=["P001", "P002", "P003"],
)
visits.index.name = "patient_id"
print(visits)
print(visits.shape)  # (3, 3)
print(visits.dtypes)
```

```text
            age  temp_c  smoker
patient_id                     
P001         34    36.8   False
P002         58    38.1    True
P003         41    37.2   False
(3, 3)
age         int64
temp_c    float64
smoker       bool
dtype: object
```

### Code Snippet: Summarize a DataFrame

```python
visits.info()
print(visits.describe())
```

Expected output: `info()` reports three rows and three non-null values per column. The numeric summary includes:

| Statistic | `age` | `temp_c` |
| --- | ---: | ---: |
| count | 3 | 3 |
| mean | 44.333333 | 37.366667 |
| std | 12.342339 | 0.665833 |
| min / max | 34 / 58 | 36.8 / 38.1 |

Every column has 3 non-null values, so nothing is missing; the memory figure varies with installed packages. `describe()` summarizes only the numeric columns, so `smoker` is left out, and its `std` row is the sample standard deviation, which divides by n − 1 (`0.665833` for `temp_c`); NumPy's `np.std(visits["temp_c"])` divides by n and gives about `0.544` unless you pass `ddof=1` (Lecture 03).

_Pro tip: DataFrames are like Excel spreadsheets, but with superpowers. They can handle millions of rows without breaking a sweat, and they never ask you to "save as" or complain about circular references._

## Selecting Columns

Brackets select columns by label. One label gives a Series; a list of labels (double brackets) gives a DataFrame, even when the list holds one name.

### Reference Card: Column selection

| Expression | Arguments | Output |
| --- | --- | --- |
| `df["column_name"]` | One label | `Series` |
| `df[["col1", "col2"]]` | List of labels | `DataFrame` |
| `df.column_name` | Identifier that does not conflict with an attribute | `Series`; fails on names with spaces or names shared with a DataFrame method, so prefer brackets |
| `df.select_dtypes(include=["number"])` | Dtype selector | Matching-column `DataFrame` |

### Code Snippet: Select Series and DataFrames

```python
print(type(visits["temp_c"]))     # one label: Series
print(type(visits[["temp_c"]]))   # a list of one label: DataFrame
print(visits[["age", "temp_c"]])
```

```text
<class 'pandas.Series'>
<class 'pandas.DataFrame'>
            age  temp_c
patient_id             
P001         34    36.8
P002         58    38.1
P003         41    37.2
```

_Think of column selection like picking your team for dodgeball: sometimes you want just your star player (single column), and sometimes you want your entire A-team (multiple columns)._

## Selecting with `.loc` and `.iloc`

Brackets pick columns. To pick rows, or rows and columns together, use `.loc` with labels or `.iloc` with integer positions, which works like `arr[row, col]` on a 2-D NumPy array (Lecture 03).

| Selector | Uses | Same cell | Slice ending |
| --- | --- | --- | --- |
| `.loc` | Row and column labels | `visits.loc["P002", "temp_c"]` → `38.1` | `visits.loc["P001":"P002"]` includes `P002` |
| `.iloc` | Integer positions | `visits.iloc[1, 1]` → `38.1` | `visits.iloc[0:2]` stops before position `2` |

### Reference Card: Selection by label and position

- `df.loc[row_label, column_label]`: One value, by labels.
- `df.loc["P001":"P002", ["age", "temp_c"]]`: A label slice (includes the end label) and a list of columns; returns a `DataFrame`.
- `df.loc["P002"]`: One whole row, as a `Series`.
- `df.loc[:, ["age"]]`: `:` means every row.
- `df.iloc[1, 1]`, `df.iloc[0:2, 0:2]`: The same selections by integer position; slices stop before the end position.
- `df1.equals(df2)`: `True` when two tables (or two Series) have the same labels, values, and dtypes; a quick check that two selections match.

### Code Snippet: Compare label and position selection

```python
print(visits.loc["P002", "temp_c"])                     # 38.1 (row label, column label)
print(visits.iloc[1, 1])                                # 38.1 (row position 1, column position 1)
by_label = visits.loc["P001":"P002", ["age", "temp_c"]]  # label slice includes P002
by_position = visits.iloc[0:2, 0:2]                     # position slice stops before 2
print(by_label)
print(by_label.equals(by_position))
```

```text
38.1
38.1
            age  temp_c
patient_id             
P001         34    36.8
P002         58    38.1
True
```

Both slices hold the same two rows, P001 and P002, so `.equals()` returns `True`.

### Common Mistakes: Labels vs Positions

- **`.loc`** = **L**abels; **`.iloc`** = **i**nteger **loc**ations (0, 1, 2, ... like list positions).
- `visits.loc[1, "age"]` raises `KeyError: 1`: no row is _labeled_ 1.
- `visits.iloc["P002", 0]` raises `ValueError`: `.iloc` accepts positions only.

_Indexing in pandas is like a choose-your-own-adventure book: there are multiple ways to reach the same destination, and sometimes you end up in a completely different story than you intended._

## Filtering Rows with a Boolean Mask

Comparing a column, as in `visits["temp_c"] >= 38.0`, gives a **mask**: a Boolean Series that carries the table's index, so each `True` or `False` stays attached to its patient. Like `arr[arr > 5]` in Lecture 03, the mask keeps the rows where it is `True`; give it a descriptive name, then pass it to `.loc` with the columns you want.

```text
temp_c >= 38.0     has_fever      visits.loc[has_fever, ["age", "temp_c"]]
P001  36.8   ->    P001  False
P002  38.1   ->    P002  True  -> P002   58   38.1
P003  37.2   ->    P003  False
```

### Reference Card: Boolean masks

- `mask = df["col"] >= value`: Test every row; returns a Boolean `Series` with the same index.
- `df.loc[mask]`: Keep the rows where `mask` is `True`.
- `df.loc[mask, ["col1", "col2"]]`: Keep matching rows and only the listed columns.
- `(df["a"] > 1) & (df["b"] < 5)`, `(...) | (...)`: Combine tests with AND / OR; parentheses are required, as in Lecture 03.
- `mask.sum()`: Count the `True` rows.

### Code Snippet: Keep patients with a fever

```python
has_fever = visits["temp_c"] >= 38.0
print(has_fever)
print(visits.loc[has_fever, ["age", "temp_c"]])
```

```text
patient_id
P001    False
P002     True
P003    False
Name: temp_c, dtype: bool
            age  temp_c
patient_id             
P002         58    38.1
```

![xkcd 2618: Selection Bias. The rows a filter keeps decide the answer, so name each mask and count what it kept](media/xkcd_2618.png)

# LIVE DEMO!

[Open Demo 2 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/04/demo/demo2_pandas_basics.ipynb)

# Deriving and Ordering Data

A **derived column** is computed from columns the table already has, such as a temperature in Fahrenheit or a change from baseline, because a clinic export rarely stores the number you report. **Sorting** then reorders whole rows by a column, so the patients who matter most, such as the highest temperatures, come first with their other values attached.

```text
visits                 add temp_f                    sort by temp_f (highest first)
     age  temp_c            age  temp_c  temp_f           age  temp_c  temp_f
P001  34    36.8      P001   34    36.8   98.24     P002   58    38.1  100.58
P002  58    38.1      P002   58    38.1  100.58     P003   41    37.2   98.96
P003  41    37.2      P003   41    37.2   98.96     P001   34    36.8   98.24
```

## Adding Columns

Assign to a new column name with brackets. As with NumPy's vectorized arithmetic in Lecture 03, pandas computes the whole column at once with no loop, matching rows by index label.

### Reference Card: Adding, updating, and removing columns

- `df["new"] = expression`: Add a column, or replace it if the name exists; values line up by index label.
- `df.drop(columns=["col1", "col2"])`: Return a new DataFrame without the named columns; `columns=` names what to leave out, as one label or a list, and the original table keeps every column.
- `df.loc[mask, "col"] = value`: Update only the rows where `mask` is `True`, in the original table.
- `subset = df.loc[mask].copy()`: Make a separate table you intend to modify, and say so explicitly.

### Code Snippet: Derive and flag

```python
visits["temp_f"] = visits["temp_c"] * 9 / 5 + 32
visits["flag"] = "ok"
visits.loc[visits["temp_c"] >= 38.0, "flag"] = "fever"
print(visits)
```

```text
            age  temp_c  smoker  temp_f   flag
patient_id                                    
P001         34    36.8   False   98.24     ok
P002         58    38.1    True  100.58  fever
P003         41    37.2   False   98.96     ok
```

### Common Mistake: Chained Assignment

In Lecture 03, a NumPy slice was a view, so changing the slice changed the original array. pandas 3 uses **Copy-on-Write**: every selection behaves like a separate copy, so two bracket steps in a row change a temporary copy. pandas warns with `ChainedAssignmentError`, and `visits` stays unchanged.

```python
visits[visits["temp_c"] >= 38.0]["flag"] = "fever"     # warning; visits is not updated
visits.loc[visits["temp_c"] >= 38.0, "flag"] = "fever"  # one step: updates visits
```

To change a separate table, such as the fever patients only, copy it first, as with NumPy arrays in Lecture 03: `fever_visits = visits.loc[has_fever].copy()`.

## Sorting Rows

`sort_values()` reorders whole rows, so each patient's other columns and index label travel with the sorted value. It returns a **new** DataFrame and leaves the original in its old order; assign the result to a name to keep it. When two rows share a value (a **tie**), sorting by that value alone does not guarantee their order. Add a unique second key, such as an ID, for a **deterministic sort**: the same rows always appear in the same order.

### Reference Card: Sorting

- `df.sort_values("col")`: Sort rows by one column, smallest first; returns a new DataFrame.
- `df.sort_values("col", ascending=False)`: Largest first.
- `df.sort_values(by=["col1", "col2"], ascending=[False, True])`: Sort by `col1` descending, then break ties with `col2` ascending, one direction per key. `by` may also name the row index, such as `"patient_id"` in `visits`.
- `df.sort_index()`: Sort rows by their index labels.

### Code Snippet: Break a tie with a unique ID

```python
vitals = pd.DataFrame({
    "patient_id": ["P003", "P001", "P002", "P004"],
    "systolic": [142, 118, 142, 130],
})
by_pressure = vitals.sort_values(
    by=["systolic", "patient_id"],
    ascending=[False, True],
)
print(by_pressure)
```

```text
  patient_id  systolic
2       P002       142
0       P003       142
3       P004       130
1       P001       118
```

P002 and P003 tie at 142, and `patient_id` puts P002 first. The index labels (2, 0, 3, 1) show where each row started. `vitals` itself is unchanged.

# Data Loading and Storage

**Data loading** turns a file into a DataFrame, and **storage** writes a table back out. The most common file is a **CSV file** (comma-separated values), the format `cut -d','` split in Lecture 03: a **header** line of column names, then one line per record. `pd.read_csv()` loads a clinic export in one call, detecting each column's type, and `df.to_csv()` saves a result for the next step.

![xkcd 1906: Making Progress. Hours of work can still end with the same problems, now in a spreadsheet](media/xkcd_1906.png)

## Reading and Writing CSV Files

Health data files mark missing values in many ways: a blank, `NA`, `NULL`, `?`. pandas already treats blanks and common markers such as `NA`, `N/A`, and `NULL` as missing and prints each one as **`NaN`** (_Not a Number_). Anything else is read as ordinary text, and a single `?` turns a whole numeric column into text (`str`). List the extra markers with `na_values` when you read.

`visits.csv`:

```text
patient_id,age,temp_c,clinic
P001,34,36.8,North
P002,58,?,South
P003,41,37.2,North
P004,,38.4,NULL
P003,41,37.2,North
```

| Read with | `temp_c` dtype | `?` becomes | `NULL` becomes |
| --- | --- | --- | --- |
| `pd.read_csv("visits.csv")` | `str` (text) | the text `"?"` | `NaN` (missing) |
| `pd.read_csv("visits.csv", na_values=["?"])` | `float64` | `NaN` (missing) | `NaN` (missing) |

A path such as `"data/visits.csv"` is **relative** to the notebook's working directory (`%pwd`); from the wrong folder, `pd.read_csv()` raises `FileNotFoundError: [Errno 2] No such file or directory: 'data/visits.csv'`. It also accepts a web address (URL), which suits Colab, where the files on your computer are not available.

_CSV stands for "Comma-Separated Values," unless someone used semicolons, or tabs, or pipes, or any other delimiter they felt like using that day._

### Reference Card: CSV and Parquet input and output

| Task | Call | Key arguments | Result |
| --- | --- | --- | --- |
| Read CSV | `pd.read_csv(path)` | `path` may be a file path or URL; `na_values=["?"]` adds missing markers to the defaults (blank, `NA`, `N/A`, `NULL`, ...); `index_col="patient_id"` makes a column the row index; `sep=";"` reads other delimiters | New `DataFrame` |
| Write CSV | `df.to_csv(path)` | Writes the row index as the first column, headed by `df.index.name` | CSV file; returns `None` |
| Write CSV | `df.to_csv(path, index=False)` | Leaves the row index out | CSV with only the data columns |
| Read Parquet | `pd.read_parquet(path)` | Reads a **Parquet** file: a compressed format that stores each column with its dtype, so nothing is re-guessed; needs the `pyarrow` package | New `DataFrame` |
| Write Parquet | `df.to_parquet(path, index=False)` | `index=False` leaves the row index out, as with `to_csv` | Parquet file |

### Code Snippet: Read with missing markers

```python
visits = pd.read_csv("visits.csv", na_values=["?"])
print(visits)
print(visits.dtypes)
```

```text
  patient_id   age  temp_c clinic
0       P001  34.0    36.8  North
1       P002  58.0     NaN  South
2       P003  41.0    37.2  North
3       P004   NaN    38.4    NaN
4       P003  41.0    37.2  North
patient_id        str
age           float64
temp_c        float64
clinic            str
dtype: object
```

`age` became `float64` because `NaN` is a floating-point value; Lecture 05 shows how to keep whole numbers when values are missing.

### Code Snippet: Keep or drop the index when writing

```python
by_patient = pd.read_csv("visits.csv", na_values=["?"], index_col="patient_id")
by_patient.to_csv("with_index.csv")             # first line: patient_id,age,temp_c,clinic
by_patient.to_csv("no_index.csv", index=False)  # first line: age,temp_c,clinic
```

Keep the index when it holds meaningful labels such as patient IDs. Use `index=False` when the index is just the default 0, 1, 2, ... row numbers. Reading the saved file back with `pd.read_csv()`, a **round trip**, confirms that the columns you meant to write are there.

## Preserving Types with Parquet

**Parquet** stores columns with their dtypes and missing values, rather than writing everything as text. Use it when another Python analysis needs the same table back; CSV suits spreadsheet exchange. The `pyarrow` package reads and writes Parquet; Demo 3's setup installs it, and the supplied local project declares it. For another local project, `uv add pyarrow==25.0.0` adds the course version.

| Format | Saved values | What the next read does |
| --- | --- | --- |
| CSV | Text fields, including empty fields for missing values | Guesses each column's dtype again |
| Parquet | Typed columns and missing values | Restores the saved dtypes |

### Reference Card: Parquet Round Trips

- `df.to_parquet(path, index=False)`: Save typed columns; omit an index that only counts rows.
- `pd.read_parquet(path)`: Read them back into a DataFrame; no CSV `na_values` rules are needed.
- `df.dtypes`: Inspect the restored types; `df.equals(original)` checks values, types, and row labels.

### Code Snippet: Write a Typed Table

```python
visits.to_parquet("visits.parquet", index=False)
```

Expected result: `visits.parquet` contains the five visits, including their missing age, temperature, and clinic values.

### Code Snippet: Read the Typed Columns

```python
print(pd.read_parquet("visits.parquet").dtypes)
```

```text
patient_id        str
age           float64
temp_c        float64
clinic            str
dtype: object
```

Demo 3's independent practice checks a complete saved-table round trip.

_Pro tip: if you're ever stuck with a weird file format, remember: "There's a pandas function for that!"_ pandas also has readers such as `pd.read_excel()` and `pd.read_json()`; BONUS.md covers them along with performance tips.

![xkcd 927: Standards. Each file format was meant to be the one everyone uses, which is why pandas has a reader for so many of them](media/xkcd_927.png)

## Inspecting a Loaded Table

Each check below is one line; its output shows what Lecture 05's cleaning tools must fix.

### Reference Card: First-look inspection

| Question | Call | Typical output |
| --- | --- | --- |
| Size, types, typical values? | `df.shape`, `df.info()`, `df.describe()` | `(rows, columns)`; non-null counts and dtypes; numeric summary (DataFrame card above) |
| What is missing? | `df.isna().sum()` | Missing count per column |
| Which categories? | `df["col"].value_counts()` | Count per value, most common first; `dropna=False` also counts missing |
| Distinct values? | `df["col"].unique()` / `df["col"].nunique()` | The distinct values, including missing, such as `['North', 'South', nan]` / how many, excluding missing: `2` |
| Repeated records? | `df.duplicated().sum()` | Rows identical to an earlier row; `df.duplicated()` alone is a Boolean mask |
| Typical value of one column? | `df["col"].mean()`, `.median()`, `.min()`, `.max()` | One number; missing values are skipped |
| Which row holds the extreme? | `df["col"].idxmax()` / `df["col"].idxmin()` | Index label of the largest / smallest value (the first one if tied) |
| Mean of every numeric column? | `df.mean(numeric_only=True)` | `Series` with one mean per numeric column; text columns are left out |

### Code Snippet: Count gaps, categories, and repeats

```python
print(visits.isna().sum())
print(visits["clinic"].value_counts(dropna=False))
print(visits.duplicated().sum())
```

```text
patient_id    0
age           1
temp_c        1
clinic        1
dtype: int64
clinic
North    3
South    1
NaN      1
Name: count, dtype: int64
1
```

The row labeled 4 repeats row 2: the same visit entered twice. Because `visits.duplicated()` is a mask, `visits.loc[visits.duplicated()]` shows the repeated row.

### Code Snippet: Summarize one column

```python
print(visits["temp_c"].mean())    # P002's missing temperature is skipped
print(visits["temp_c"].max())
print(visits["temp_c"].idxmax())  # index label of the highest temperature
print(visits.mean(numeric_only=True))
```

```text
37.400000000000006
38.4
3
age       43.5
temp_c    37.4
dtype: float64
```

pandas skips missing values in summaries, so the mean averages the four recorded temperatures; NumPy's `.mean()` returns `nan` when any value is missing. The trailing `...006` is binary rounding: most decimals cannot be stored exactly, so format the value with `:.1f` (Lecture 02) when you report it. The duplicated P003 visit is counted twice here, one more reason to find repeats before summarizing.

# LIVE DEMO!

[Open Demo 3 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/04/demo/demo3_data_io.ipynb)
