---
notion:
  title_line: "# Data Aggregation and Group Operations"
  role: lecture
  status: mapped
  page_id: "2a1d9fdd-1a1a-80f8-b1e8-f7b20e4a2e84"
  url: "https://app.notion.com/p/2a1d9fdd1a1a80f8b1e8f7b20e4a2e84"
---

# Data Aggregation and Group Operations

See [BONUS.md](BONUS.md) for the optional extensions.

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo1_groupby_operations.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo2_coverage_result_shapes.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo3_remote_performance.ipynb)

**Run locally:**

```shell
curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/08/demo/setup_demo.sh | sh
cd ~/08-demo
uv venv --seed
source .venv/bin/activate
uv sync
```

→ Then open the `08-demo` folder in VS Code.

_"Aggregation" comes from the Latin "aggregare," to add to a flock, which is what a groupby does with scattered rows._

This lecture covers:

- McKinney, _Python for Data Analysis_ (3rd ed.):
    - 7.5 (computations with categoricals)
    - 10.1 to 10.5 (group operations, data aggregation, apply, group transforms, and pivot tables and cross-tabulation)
- Shotts, _The Linux Command Line_:
    - Chapter 16 (secure communication with remote hosts: `ssh` and `scp`)
- MIT, _The Missing Semester of Your CS Education_:
    - "Command-line Environment" (terminal multiplexers and remote machines)

# The Split-Apply-Combine Paradigm

- **Split-apply-combine**: the pattern behind every grouped summary; pandas runs it with `groupby()`. It answers questions such as "What is the average wait at each clinic?" from a visit log with one row per visit, returning one row per clinic.
- **Split**: sort rows into **groups** by a **grouping key**, a column whose values decide which group each row joins (here, `clinic`).
- **Apply**: run a calculation on each group separately (here, the mean of `wait_min`).
- **Combine**: collect the per-group results into a new table.
- Grouping usually changes the **grain**, what one row represents (Lectures 06 and 07), so name it before and after.

```text
INPUT: one row per visit    SPLIT by clinic          APPLY mean     COMBINE: one row per clinic
clinic  wait_min
North   12            ┌──>  North: 12, 15, 6  ──>   11.0  ──┐     clinic  wait_min
North   15            │                                     │     East         9.0
North    6            ├──>  South: 20, 30     ──>   25.0  ──┼──>  North       11.0
South   20            │                                     │     South       25.0
South   30            └──>  East: 9           ──>    9.0  ──┘
East     9
```

# Basic GroupBy Operations

- **GroupBy object**: what `df.groupby('clinic')` returns, a record of which rows belong to which group, not yet a summary table.
- **Aggregation**: a calculation that reduces each group's values to one number, such as a mean or a count. Nothing is calculated until you choose a column and an aggregation.

## Aggregating Each Group

The snippets in this lecture use `visits`, six clinic visits:

|  | clinic | visit_type | patient_id | wait_min | satisfaction |
| --- | --- | --- | --- | --- | --- |
| 0 | North | New | P01 | 12 | 4.0 |
| 1 | North | Follow-up | P02 | 15 | NaN |
| 2 | North | Follow-up | P01 | 6 | 5.0 |
| 3 | South | New | P03 | 20 | 3.0 |
| 4 | South | New | P04 | 30 | NaN |
| 5 | East | Follow-up | P05 | 9 | 4.0 |

### Reference Card: GroupBy Aggregation

| Task | Call | Purpose and key arguments | Output |
| --- | --- | --- | --- |
| Split | `df.groupby('key')` | Group rows by one key; pass a list for several keys | `DataFrameGroupBy`; nothing computed yet |
| Select | `grouped['col']` / `grouped[['a', 'b']]` | Choose the columns to summarize; `.mean()` on text columns raises `TypeError`, and `.sum()` glues text together (`P01P02P01`) | `SeriesGroupBy` / `DataFrameGroupBy` |
| Summarize | `.mean()`, `.median()`, `.sum()`, `.min()`, `.max()` | One summary per group of the selected numeric column | One row per group |
| Count | `.size()` | Rows per group, including rows with missing values | One count per group |
| Count | `['col'].count()` | Non-missing values of `col` per group | One count per group |
| Count | `['col'].nunique()` | Distinct non-missing values of `col` per group | One count per group |
| Several at once | `.agg(name=('col', 'func'), ...)` | **Named aggregation**: each keyword names an output column; its value is a `(source column, function)` pair | One flat column per name |
| Several at once | `grouped['col'].agg(['mean', 'std'])` | A list of functions for one column | One column per function, named `mean`, `std` |
| Several at once | `.agg({'col': ['mean', 'max']})` | Several functions per column | Two-level column labels |
| Result shape | `groupby(..., as_index=False)` | Keep keys as ordinary columns | Flat table, `0..n-1` index |
| Result shape | `groupby(..., sort=True)` | Sort rows by key (the default); categorical keys follow category order | Ordered rows |

### Code Snippet: One Mean per Clinic

```python
grouped = visits.groupby('clinic')
print(grouped)
display(grouped['wait_min'].mean())
```

```text
<pandas.api.typing.DataFrameGroupBy object at 0x...>
```

| clinic | wait_min |
| --- | --- |
| East | 9.0 |
| North | 11.0 |
| South | 25.0 |

The GroupBy object shows no numbers; `.mean()` on the selected column runs the apply and combine steps, with the clinics in alphabetical order.

### Code Snippet: Count and Summarize Each Clinic

```python
summary = visits.groupby('clinic', as_index=False).agg(
    visits=('patient_id', 'size'),
    patients=('patient_id', 'nunique'),
    rated=('satisfaction', 'count'),
    mean_wait=('wait_min', 'mean'),
)
display(summary)
```

|  | clinic | visits | patients | rated | mean_wait |
| --- | --- | --- | --- | --- | --- |
| 0 | East | 1 | 1 | 1 | 9.0 |
| 1 | North | 3 | 2 | 2 | 11.0 |
| 2 | South | 2 | 2 | 1 | 25.0 |

North had three visits (`size`) from two patients (`nunique`: P01 came twice), and only two of those visits have a satisfaction score (`count` skips `NaN`).

![xkcd 2533: Slope Hypothesis Testing. Measuring the same people again adds rows, not people, which is why size and nunique answer different questions](media/xkcd_2533.png)

## Grouping by Two Keys

Pass a list of keys to get one group per observed combination, with a MultiIndex (Lecture 06), one index level per key.

### Reference Card: Two-Key Results

- `df.groupby(['a', 'b'])['col'].mean()`: One value per observed `(a, b)` pair; MultiIndex `Series`.
- `result.unstack()`: Move the inner level (`b`) into columns so the table reads like a grid; absent pairs become `NaN`.
- `wide.stack()`: Move the columns back into the inner index level; absent pairs come back as `NaN` rows.
- `df.groupby(['a', 'b'], as_index=False)`: Keep `a` and `b` as ordinary columns; one flat row per pair.

### Code Snippet: Mean Wait by Clinic and Visit Type

```python
by_type = visits.groupby(['clinic', 'visit_type'])['wait_min'].mean()
display(by_type)
display(by_type.unstack())
```

| clinic | visit_type | wait_min |
| --- | --- | --- |
| East | Follow-up | 9.0 |
| North | Follow-up | 10.5 |
|  | New | 12.0 |
| South | New | 25.0 |

| clinic | Follow-up | New |
| --- | --- | --- |
| East | 9.0 | NaN |
| North | 10.5 | 12.0 |
| South | NaN | 25.0 |

East had no new-patient visits, so East–New is `NaN`: there was nothing to average, which is not a zero-minute wait (those exist only in clinic brochures).

# Pivot Tables and Cross-Tabulations

![The greatest research skill you can have is being a nosy bitch who wants to find out: a pivot table is how you find out](media/research.png)

- **Pivot table**: summarizes one column by two keys at once, one down the rows and one across the columns, such as mean wait by clinic and visit type. It is the two-key `groupby(...).mean().unstack()` in a single call.
- **Cross-tabulation** (crosstab): the special case that counts the rows in each combination.

```text
LONG: one row per visit                    PIVOT TABLE: mean wait_min, one row per clinic
clinic  visit_type  wait_min               visit_type  Follow-up   New
North   New         12                     clinic
North   Follow-up   15  ┐ mean → 10.5      East              9.0   NaN   ← no East new-patient visits
North   Follow-up    6  ┘                  North            10.5  12.0
South   New         20  ┐ mean → 25.0      South             NaN  25.0
South   New         30  ┘
East    Follow-up    9
```

## Basic Pivot Tables

### Reference Card: Pivot Tables and Crosstabs

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `pd.pivot_table(df, values=..., index=..., columns=..., aggfunc='mean')` | Group by the `index` and `columns` keys and summarize `values` in each cell; `aggfunc` picks the summary (`'mean'`, `'sum'`, `'count'`, ...). Unlike `pivot()` (Lecture 06), repeated key pairs are summarized, not an error | Wide table; absent combinations are `NaN` |
| `pd.pivot_table(..., aggfunc=['sum', 'mean'])` | Several summaries at once; `result['sum']` selects one | Two-level column labels |
| `pd.pivot_table(..., sort=True)` | Order the rows and columns by key (the default); `sort=False` leaves both in first-appearance order | Predictable row and column order |
| `pd.crosstab(df['a'], df['b'])` | Count rows for each combination of two columns; pass a list such as `[df['a'], df['c']]` for two-level rows | Counts; absent combinations are 0 |
| `pd.crosstab(df['a'], df['b'], values=df['v'], aggfunc='mean')` | Summarize a third column instead of counting, like `pivot_table` | Summary table; absent combinations are `NaN` |

### Code Snippet: A Pivot Table and Its GroupBy Twin

```python
mean_wait = pd.pivot_table(visits, values='wait_min', index='clinic',
                           columns='visit_type', aggfunc='mean')
display(mean_wait)

same = visits.groupby(['clinic', 'visit_type'])['wait_min'].mean().unstack()
print(mean_wait.equals(same))

display(pd.crosstab(visits['clinic'], visits['visit_type']))
```

| clinic | Follow-up | New |
| --- | --- | --- |
| East | 9.0 | NaN |
| North | 10.5 | 12.0 |
| South | NaN | 25.0 |

```text
True
```

| clinic | Follow-up | New |
| --- | --- | --- |
| East | 1 | 0 |
| North | 2 | 1 |
| South | 0 | 2 |

## Totals and Absent Cells

- **Margins**: a total row and column computed from the underlying rows, not from the cells. In a table of means, a clinic's `Total` is the mean of all its visits, not the average of its cell means.

### Reference Card: Pivot Table Totals and Fills

| Option | Purpose and key arguments | Output effect |
| :--- | :--- | :--- |
| `margins=True, margins_name='Total'` | Add margins; `margins_name` labels them (default `All`); `pd.crosstab()` takes the same two options | Extra `Total` row and column |
| `fill_value=0` | Replace absent cells with 0, only for counts and sums | No-`NaN` cells |

### Code Snippet: Totals Where Zero Is Real

```python
total_wait = pd.pivot_table(visits, values='wait_min', index='clinic',
                            columns='visit_type', aggfunc='sum',
                            fill_value=0, margins=True, margins_name='Total')
display(total_wait)
```

| clinic | Follow-up | New | Total |
| --- | --- | --- | --- |
| East | 9 | 0 | 9 |
| North | 21 | 12 | 33 |
| South | 0 | 50 | 50 |
| Total | 30 | 62 | 92 |

<callout icon="⚠️" color="yellow_bg">
	## `fill_value=0` only where zero is real
	An absent cell means no rows had that combination. For a count or a sum that is 0, as East's total new-patient wait above. For a mean or any other measurement it is not: leave it `NaN`.
</callout>

# LIVE DEMO!

[Open Demo 1 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo1_groupby_operations.ipynb)

# Which Groups Appear in a Summary

- **Group coverage**: which groups a summary lists. Two pandas defaults shrink it without warning:
    - Rows with a missing key are dropped before the split, so visits with no clinic recorded go uncounted.
    - A categorical key (Lecture 05) lists only the categories that occur, so a new clinic with no visits yet has no row.

## Missing Keys and Unused Categories

| What the April visit log held | The default summary | What a handed-off report should show |
| --- | --- | --- |
| North 3, South 2, East 1 visits | three rows, alphabetical | three rows, in your reporting order |
| West opened in March, no visits yet | no West row at all | `West 0` |
| One visit with no clinic recorded | dropped before the split, uncounted | a `NaN` row you can see and explain |

### Reference Card: Group Coverage Options

- `df.groupby('key', dropna=False)`: Keep rows with a missing key as a `NaN` group, listed last. The default, `dropna=True`, leaves them out.
- `pd.Categorical(values, categories=[...], ordered=True)`: Declare the allowed values and their reporting order; with the default `sort=True`, groups follow this order, not alphabetical order. `ordered=True` records that the order is meaningful. Prints `Categories (4, str): ['North' < 'South' < 'East' < 'West']` under the values.
- `df.groupby('key', observed=False)`: For a categorical key, list every defined category; an unused one shows 0 for counts and sums and `NaN` for means. The pandas 3 default, `observed=True`, lists only categories present.
- `pd.pivot_table(..., observed=False, dropna=False)`: The same two choices for a pivot table's rows and columns.

### Code Snippet: A Reporting Order and an Unused Clinic

```python
levels = ['North', 'South', 'East', 'West']   # reporting order; West has no visits yet
cat_visits = visits.copy()
cat_visits['clinic'] = pd.Categorical(visits['clinic'], categories=levels, ordered=True)
display(cat_visits.groupby('clinic', observed=True)['wait_min'].count())
display(cat_visits.groupby('clinic', observed=False)['wait_min'].count())
```

| clinic | wait_min |
| --- | --- |
| North | 3 |
| South | 2 |
| East | 1 |

| clinic | wait_min |
| --- | --- |
| North | 3 |
| South | 2 |
| East | 1 |
| West | 0 |

### Code Snippet: A Visit with No Clinic Recorded

```python
unrecorded = visits.copy()
unrecorded.loc[5, 'clinic'] = None     # the clinic was not written down for this visit
display(unrecorded.groupby('clinic')['wait_min'].count())
display(unrecorded.groupby('clinic', dropna=False)['wait_min'].count())
```

| clinic | wait_min |
| --- | --- |
| North | 3 |
| South | 2 |

| clinic | wait_min |
| --- | --- |
| North | 3 |
| South | 2 |
| NaN | 1 |

That visit was East's only one, so the default report drops East entirely: six visits went in, five are counted, and nothing says so.

![xkcd 2523: Endangered. A flu lineage going extinct is good news; a clinic silently vanishing from your report is not](media/xkcd_2523.png)

# GroupBy Result-Shape Choices

- **Result shape**: how many rows a GroupBy method returns, and at which grain. Decide the grain the answer needs, then choose the method that returns it.

| Operation | Question it answers (clinic visits) | Rows returned | Grain of the result |
| --- | --- | --- | --- |
| `agg` | What is the mean wait at each clinic? | 3 | One row per clinic |
| `transform` | How does each visit compare with its clinic's mean? | 6, same index as `visits` | One row per visit |
| `filter` | Which visits belong to clinics with at least two visits? | 5 (East dropped) | One row per visit |
| `apply` | Which two visits (the whole rows) had the longest waits at each clinic? | Depends on the function (5 here) | Depends on the function (up to two visits per clinic here) |

- Use `agg`, `transform`, or `filter` whenever one fits; a single number such as the longest wait is just `agg('max')`.

## Transform Operations

- **transform**: computes a statistic for each group and copies it back to every row of that group. The result has the table's length and index, so it can become a new column.
- **z-score**: `(value - group_mean) / group_std`, a row's distance from its group's mean in standard deviations; 0 is the mean, 1 is one standard deviation above it.

### Reference Card: Transform Operations

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `grouped.transform('mean')` | Broadcast each group's mean to its original rows | Series aligned to original index |
| `grouped.transform('std')` | Broadcast within-group spread | Series aligned to original index |
| `grouped.transform(lambda x: x - x.mean())` | Compute a custom within-group value | Same row count as input |
| `df['col'].fillna(df.groupby('key')['col'].transform('median'))` | **Group-specific fill**: replace each missing value with its own group's median | Series aligned to original index |

### Code Snippet: Compare Each Visit with Its Clinic

```python
with_context = visits.copy()   # visits itself stays unchanged
with_context['clinic_mean_wait'] = visits.groupby('clinic')['wait_min'].transform('mean')
with_context['wait_vs_clinic'] = with_context['wait_min'] - with_context['clinic_mean_wait']
display(with_context[['clinic', 'patient_id', 'wait_min', 'clinic_mean_wait', 'wait_vs_clinic']])
```

|  | clinic | patient_id | wait_min | clinic_mean_wait | wait_vs_clinic |
| --- | --- | --- | --- | --- | --- |
| 0 | North | P01 | 12 | 11.0 | 1.0 |
| 1 | North | P02 | 15 | 11.0 | 4.0 |
| 2 | North | P01 | 6 | 11.0 | -5.0 |
| 3 | South | P03 | 20 | 25.0 | -5.0 |
| 4 | South | P04 | 30 | 25.0 | 5.0 |
| 5 | East | P05 | 9 | 9.0 | 0.0 |

East's one visit sits exactly at its clinic's mean: it is easy to be average when you are the whole group.

### Code Snippet: Fill Missing Scores with the Clinic Median

```python
clinic_median = visits.groupby('clinic')['satisfaction'].transform('median')
display(visits['satisfaction'].fillna(clinic_median))
```

|  | satisfaction |
| --- | --- |
| 0 | 4.0 |
| 1 | 4.5 |
| 2 | 5.0 |
| 3 | 3.0 |
| 4 | 3.0 |
| 5 | 4.0 |

North's missing score becomes 4.5, the median of North's 4.0 and 5.0, and South's becomes 3.0. A whole-column `fillna(visits['satisfaction'].median())` would give both 4.0.

<callout icon="⚠️" color="yellow_bg">
	## A group summary does not line up with the rows
	`visits['clinic_mean_wait'] = visits.groupby('clinic')['wait_min'].mean()` fills every row with `NaN`, without an error: pandas matches a new column to rows by index label (Lecture 04), and the summary is labeled by clinic while the rows are labeled 0 to 5. `.mean().values` fails instead: `ValueError: Length of values (3) does not match length of index (6)`. Use `transform('mean')`.
</callout>

## Filter Operations

- **filter**: runs a test on each whole group and keeps every original row of the groups that pass. Rows are kept or dropped, never changed.

### Reference Card: Filter Operations

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `grouped.filter(lambda g: len(g) >= n)` | Keep groups with at least `n` rows | Original rows from passing groups |
| `grouped.filter(lambda g: g['col'].sum() > threshold)` | Keep groups meeting a total threshold | Original rows from passing groups |
| `grouped.filter(lambda g: g['col'].mean() > threshold)` | Keep groups meeting a mean threshold | Original rows from passing groups |

### Code Snippet: Keep Clinics by Size, Mean, or Total

```python
busy = visits.groupby('clinic').filter(lambda g: len(g) >= 2)
slow = visits.groupby('clinic').filter(lambda g: g['wait_min'].mean() > 10)
heavy = visits.groupby('clinic').filter(lambda g: g['wait_min'].sum() > 40)
display(slow[['clinic', 'patient_id', 'wait_min']])
print(len(busy), heavy['clinic'].unique().tolist())
```

|  | clinic | patient_id | wait_min |
| --- | --- | --- | --- |
| 0 | North | P01 | 12 |
| 1 | North | P02 | 15 |
| 2 | North | P01 | 6 |
| 3 | South | P03 | 20 |
| 4 | South | P04 | 30 |

```text
5 ['South']
```

East (one visit, mean 9) fails both the size and the mean test. Only South's waits total more than 40 minutes (North's add up to 33).

# LIVE DEMO!

[Open Demo 2 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo2_coverage_result_shapes.ipynb)

# Apply: Your Own Function per Group

- **apply**: hands each group to your function as a small DataFrame and stitches the results together. The result can be a number, a summary, or several original rows, so its shape depends on what your function returns. It is the slower fallback: use it when no built-in aggregation fits.

## Sorting and Keeping Top Rows Within Groups

### Reference Card: Apply Operations

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `grouped.apply(func, include_groups=False)` | Run your own function on each group's columns, excluding the grouping column | Depends on `func` |
| `grouped.apply(lambda g: g.sort_values('col'), include_groups=False)` | Sort rows inside each group | Every row, grouped and sorted |
| `grouped.apply(lambda g: g.nlargest(2, 'col'), include_groups=False)` | Keep the top two rows in each group | Up to two rows per group |

### Code Snippet: Sort Within Clinics and Keep the Two Longest Waits

```python
by_wait = visits.groupby('clinic').apply(
    lambda g: g.sort_values('wait_min'), include_groups=False,
)
top2 = visits.groupby('clinic').apply(
    lambda g: g.nlargest(2, 'wait_min'), include_groups=False,
)
display(by_wait[['patient_id', 'wait_min']])
display(top2[['patient_id', 'wait_min']])
```

| clinic |  | patient_id | wait_min |
| --- | --- | --- | --- |
| East | 5 | P05 | 9 |
| North | 2 | P01 | 6 |
|  | 0 | P01 | 12 |
|  | 1 | P02 | 15 |
| South | 3 | P03 | 20 |
|  | 4 | P04 | 30 |

| clinic |  | patient_id | wait_min |
| --- | --- | --- | --- |
| East | 5 | P05 | 9 |
| North | 1 | P02 | 15 |
|  | 0 | P01 | 12 |
| South | 4 | P04 | 30 |
|  | 3 | P03 | 20 |

- The index has two levels (a MultiIndex): the clinic, then each visit's original row number.
- North's rows come back in wait order, and East keeps its single visit because `nlargest(2)` returns as many rows as the group has.
- In pandas 3, `include_groups=False` is the default and the only allowed value, so your function never receives the clinic column; writing it out keeps the code working on older pandas, such as Colab's 2.2.

![xkcd 1172: Workflow. Every change breaks someone's workflow, including pandas 3's change to what apply hands your function](media/xkcd_1172.png)

# Performance Optimization

- **Performance optimization**: making an analysis run faster or use less memory, which matters once a summary that takes a second on a class example takes minutes on a year of hospital lab results.
- One `.agg()` call computing several summaries beats calling `groupby()` again for each summary, because every new `groupby()` call redoes the split.
- Built-in aggregations named by string (`'mean'`, `'std'`, `transform('mean')`) run as fast compiled code; a Python `lambda` or `apply` runs once per group and is often 20 to 100 times slower.
- A repeated text key stored as `category` (Lecture 05) uses far less memory.

![Performance benchmarks on about 100 million rows (lower is better): three separate groupby calls vs one agg, a per-group z-score with apply vs transform, and memory before and after categorical keys](media/perf_combined.png)

## Measuring Speed and Memory

### Reference Card: Measure Before Optimizing

- `%timeit expression`: Time one line in a notebook by running it many times; output looks like `8.8 ms ± 0.1 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)`.
- `df.memory_usage(deep=True)`: Bytes used by each column, counting the text inside it; shows which column to shrink.
- `df['key'].astype('category')`: Store a repeated text key as a few labels plus a small code per row; same groups, less memory.
- `grouped['col'].agg(['mean', 'std', 'count'])`: Several summaries from one split instead of one `groupby()` per summary.

![xkcd 1319: Automation. Making code faster is work too, so measure first and spend the effort only where the time goes](media/xkcd_1319.png)

# Remote Computing with SSH

- **SSH** (Secure Shell): opens an encrypted terminal on another computer, a **server** or **host**, so you can run analyses too large or too long for a laptop.
- **Protected health information** (PHI), records that can identify a person, often may not leave an approved secure server, so you bring your code to the data and take back only permitted results.

## Connect and Copy Files

The server's operator supplies its hostname, your account, and how to log in. Once connected, shell commands (`pwd`, `ls`, `cd`, `python3`) run on the server, and the prompt shows where you are; `exit` returns to your laptop:

```text
you@laptop:~$ ssh jdoe@analysis.example.org
The authenticity of host 'analysis.example.org' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
jdoe@analysis:~$ hostname
analysis
jdoe@analysis:~$ exit
you@laptop:~$
```

- **Host key fingerprint**: the code shown on your first connection that identifies the server. Type `yes` only if it matches the fingerprint the administrator published; SSH remembers the server after that.
- A password prompt shows nothing as you type; that is normal.
- **SSH key pair**: what most servers use instead of a password. `ssh-keygen` creates the **public key**, `~/.ssh/id_ed25519.pub`, a padlock you install on every server you use, and the **private key**, `~/.ssh/id_ed25519`, the only key that opens it.

<callout icon="⚠️" color="yellow_bg">
	## Never share your private key
	Never email, upload, or commit `~/.ssh/id_ed25519`: anyone holding it can log in as you. Only the `.pub` file goes to servers.
</callout>

### Reference Card: SSH File and Shell Commands

| Command | Purpose and key arguments | Result |
| :--- | :--- | :--- |
| `ssh user@host` | Open a remote shell | Remote prompt |
| `ssh -p port user@host` | Connect through a nondefault port | Remote prompt |
| `ssh user@host 'command'` | Run one command remotely | Command output locally |
| `scp local user@host:path` | Copy a local file to the server | Remote file |
| `scp user@host:path local` | Copy a remote file back | Local file |
| `ssh-keygen -t ed25519` | Create a public/private key pair; `-C "email"` labels it | `~/.ssh/id_ed25519` and `~/.ssh/id_ed25519.pub` |
| `ssh-keygen -t ed25519 -f path` | Save the pair at `path` instead of the default, such as a practice key that must not replace your real one | `path` and `path.pub` |
| `ssh-copy-id user@host` | Install the public key where supported | Passwordless key login |
| `ssh-add` | Unlock your private key once per login session; the SSH agent keeps it unlocked | Later `ssh` and `scp` stop asking for the passphrase |

### Code Snippet: Create a Key Pair

Run this once on your laptop, then install only the public half using the server's instructions or `ssh-copy-id`:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

- Press Enter for the default location, and set a passphrase that protects the private key file.
- **SSH agent**: a background program on your laptop (macOS runs one for you). `ssh-add` asks for the passphrase once, and the agent keeps the key unlocked until you log out. If `ssh-add` cannot connect to your authentication agent (common in WSL), skip it and type the passphrase when asked.
- `Overwrite (y/n)?` means a key already exists: answer `n` and keep using it, because replacing it breaks access to servers that trust its public half.

### Code Snippet: Upload a File and Download a Result

Run these from your laptop's prompt, not inside an `ssh` session. The first copies `data.csv` into the server's existing `~/data/` folder; the second copies `analysis.csv` into your current local folder:

```bash
scp data.csv username@server.example:~/data/
scp username@server.example:~/results/analysis.csv ./
```

![Punk is whatever makes you happy that irritates people who are used to having total control: a job that keeps running after you log off is a little punk](media/punk.jpg)

## Keep Long Jobs Alive with tmux or screen

- When an SSH connection drops (laptop sleeps, Wi-Fi changes), programs tied to that terminal can stop, including an analysis three hours into a four-hour run.
- **Terminal multiplexer**: a program such as `tmux` that keeps a shell running on the server by itself. You **detach**, disconnect, and later **attach** again to find the job still running. A session survives disconnects, not a server restart.
- `screen` is an older alternative; use whichever the server provides. [Tmux Fundamentals](https://linuxhandbook.com/courses/tmux/) is a guided introduction.

```text
laptop ── ssh ──> server
                   └─ tmux session "analysis"   (keeps running after you disconnect)
                        └─ python3 analysis.py
```

### Reference Card: tmux Sessions

- Install: servers usually have tmux already. To practice on your laptop, `brew install tmux` (macOS) or `sudo apt install tmux` (Ubuntu or WSL); `tmux -V` prints the version.
- `tmux new -s analysis`: Start a session named `analysis`; a status bar appears at the bottom.
- `Ctrl+b`, then `d`: Detach; the session keeps running and you return to the normal prompt.
- `tmux ls` (short for `tmux list-sessions`): List sessions; output looks like `analysis: 1 windows (created Fri Sep 18 15:32:45 2026)`.
- `tmux attach -t analysis`: Reattach to the session.
- `exit` inside the session: End it when the job is done.
- `tmux kill-session -t analysis`: End a detached session from outside, stopping anything still running in it.
- Alternative: `screen -S analysis`; detach with `Ctrl+a`, then `d`; reattach with `screen -r analysis`.

### Code Snippet: Keep an Analysis Running

On the server, after `ssh`, in a project folder whose environment you built there with `uv venv --seed` and `uv sync` (Lecture 03):

```bash
tmux new -s analysis
source .venv/bin/activate
time python3 analysis.py
```

Press `Ctrl+b`, then `d`, to detach; now it is safe to disconnect. Later, after reconnecting with `ssh`, find the session and attach to it; when the job ends, `time` reports how long it ran:

```bash
tmux ls
tmux attach -t analysis
```

# LIVE DEMO!

[Open Demo 3 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo3_remote_performance.ipynb)
