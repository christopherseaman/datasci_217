---
notion:
  title_line: "Data Aggregation and Group Operations"
  role: lecture
  status: mapped
  page_id: "2a1d9fdd-1a1a-80f8-b1e8-f7b20e4a2e84"
  url: "https://app.notion.com/p/2a1d9fdd1a1a80f8b1e8f7b20e4a2e84"
---

Data Aggregation and Group Operations

See [BONUS.md](BONUS.md) for the optional extensions.

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo1_groupby_operations.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo2_pivot_tables.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo3_remote_performance.ipynb)

*Fun fact: The term "aggregation" comes from the Latin "aggregare" meaning "to add to a flock." In data science, we're literally gathering scattered data points into meaningful groups - turning a flock of individual observations into organized insights.*

# The Split-Apply-Combine Paradigm

*Reality check: GroupBy operations are the bread and butter of data analysis. Master this concept and you'll be able to answer almost any "what if we group by..." question that comes your way.*

A clinic manager asks, "What is the average wait at each clinic?" The visit log has one row per visit, but the answer needs one number per clinic. By hand, you would sort the visits into piles by clinic, average each pile, and copy the averages into a small table. pandas does the same three steps with `groupby()`.

Hadley Wickham named this pattern **split-apply-combine**:

- **Split**: sort rows into **groups** using a **grouping key**, a column whose values decide which group each row joins (here, `clinic`).
- **Apply**: run a calculation on each group separately (here, the mean of `wait_min`).
- **Combine**: collect the per-group results into a new table.

Lecture 06 reshaped wide tables into long ones so that labels became values you can group by. Lectures 06 and 07 called what one row of a table represents its grain. Grouping usually changes the grain, so name it before and after. Here the input has one row per visit; the result has one row per clinic.

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

`df.groupby('clinic')` performs only the split. It returns a **GroupBy object**: a record of which rows belong to which group, not a summary table. Nothing is calculated until you pick a column and an **aggregation**, a calculation that reduces each group's values to one number such as a mean or a count.

## Aggregating Each Group

### Code Snippet: One Mean per Clinic

```python
import pandas as pd
import numpy as np

visits = pd.DataFrame({
    'clinic': ['North', 'North', 'North', 'South', 'South', 'East'],
    'visit_type': ['New', 'Follow-up', 'Follow-up', 'New', 'New', 'Follow-up'],
    'patient_id': ['P01', 'P02', 'P01', 'P03', 'P04', 'P05'],
    'wait_min': [12, 15, 6, 20, 30, 9],
    'satisfaction': [4, np.nan, 5, 3, np.nan, 4],
})

grouped = visits.groupby('clinic')
print(grouped)
print(grouped['wait_min'].mean())
```

```text
<pandas.api.typing.DataFrameGroupBy object at 0x...>
clinic
East      9.0
North    11.0
South    25.0
Name: wait_min, dtype: float64
```

Printing the GroupBy object shows no numbers. Selecting `wait_min` and calling `.mean()` runs the apply and combine steps, and the result is sorted by clinic name.

### Reference Card: GroupBy Aggregation

| Task | Call | Purpose and key arguments | Output |
| --- | --- | --- | --- |
| Split | `df.groupby('key')` | Group rows by one key; pass a list for several keys | `DataFrameGroupBy`; nothing computed yet |
| Select | `grouped['col']` / `grouped[['a', 'b']]` | Choose the columns to summarize | `SeriesGroupBy` / `DataFrameGroupBy` |
| Summarize | `.mean()`, `.median()`, `.sum()`, `.min()`, `.max()` | One summary per group of the selected numeric column | One row per group |
| Count | `.size()` | Rows per group, including rows with missing values | One count per group |
| Count | `['col'].count()` | Non-missing values of `col` per group | One count per group |
| Count | `['col'].nunique()` | Distinct non-missing values of `col` per group | One count per group |
| Several at once | `.agg(name=('col', 'func'), ...)` | **Named aggregation**: output name, source column, function | One flat column per name |
| Several at once | `.agg({'col': ['mean', 'max']})` | Several functions per column | Two-level column labels |
| Result shape | `groupby(..., as_index=False)` | Keep keys as ordinary columns | Flat table, `0..n-1` index |
| Result shape | `groupby(..., sort=True)` | Sort rows by key (the default); categorical keys follow category order | Ordered rows |

### Code Snippet: Count and Summarize Each Clinic

```python
summary = visits.groupby('clinic', as_index=False).agg(
    visits=('patient_id', 'size'),
    patients=('patient_id', 'nunique'),
    rated=('satisfaction', 'count'),
    mean_wait=('wait_min', 'mean'),
)
print(summary)
```

```text
  clinic  visits  patients  rated  mean_wait
0   East       1         1      1        9.0
1  North       3         2      2       11.0
2  South       2         2      1       25.0
```

North had three visits from two patients (P01 came twice), and only two of those visits have a satisfaction score.

### Common Mistakes: Text Columns and Counts

- `visits.groupby('clinic').mean()` raises `TypeError: dtype 'str' does not support operation 'mean'` because `patient_id` and `visit_type` are text. Select numeric columns first.
- `.sum()` over text does not fail: it glues strings together (`P01P02P01`). Select columns before summing.
- `size` and `count` differ only when values are missing. Use `size` for "how many rows?" and `count` for "how many recorded values?"

## Grouping by Two Keys

Pass a list of keys to get one group per observed combination. The result has a **MultiIndex** (Lecture 06): one index level per key. `.unstack()` moves the inner level into columns so the table reads like a grid. Combinations with no rows appear as `NaN`.

### Reference Card: Two-Key Results

- `df.groupby(['a', 'b'])['col'].mean()`: One value per observed `(a, b)` pair; MultiIndex `Series`.
- `result.unstack()`: Move the inner level (`b`) into columns; absent pairs become `NaN`.
- `wide.stack()`: Move the columns back into the inner index level; absent pairs come back as `NaN` rows.
- `df.groupby(['a', 'b'], as_index=False)`: Keep `a` and `b` as ordinary columns; one flat row per pair.

### Code Snippet: Mean Wait by Clinic and Visit Type

```python
by_type = visits.groupby(['clinic', 'visit_type'])['wait_min'].mean()
print(by_type)
print(by_type.unstack())
```

```text
clinic  visit_type
East    Follow-up      9.0
North   Follow-up     10.5
        New           12.0
South   New           25.0
Name: wait_min, dtype: float64
visit_type  Follow-up   New
clinic
East              9.0   NaN
North            10.5  12.0
South             NaN  25.0
```

East had no new-patient visits, so East–New is `NaN`. There was nothing to average, which is not the same as a zero-minute wait.

## Missing Keys and Unused Categories

Two defaults decide which groups appear. Rows with a missing key are left out. For a categorical key (Lecture 05), only the categories that occur in the data appear, so a clinic with no visits yet is missing from the report. Declaring the categories yourself also sets the order of the results: with the default `sort=True`, groups follow the order you list in `categories=`, not alphabetical order.

### Reference Card: Which Groups Appear

- `df.groupby('key', dropna=False)`: Keep rows with a missing key as a `NaN` group, listed last. The default, `dropna=True`, leaves them out.
- `df.groupby('key', observed=False)`: For a categorical key, list every defined category; an unused one shows 0 for counts and sums and `NaN` for means. The pandas 3 default, `observed=True`, lists only categories present. For plain text keys, `observed` does nothing.
- `pd.Categorical(values, categories=[...], ordered=True)`: Declare the allowed values and their reporting order; `ordered=True` records that the order is meaningful. Prints `Categories (4, str): ['North' < 'South' < 'East' < 'West']` under the values. A value missing from `categories=` becomes `NaN` with a warning, so list every value that occurs.

### Code Snippet: A Reporting Order and an Unused Clinic

```python
levels = ['North', 'South', 'East', 'West']   # reporting order; West has no visits yet
cat_visits = visits.copy()
cat_visits['clinic'] = pd.Categorical(visits['clinic'], categories=levels, ordered=True)
print(cat_visits.groupby('clinic', observed=True)['wait_min'].count())
print(cat_visits.groupby('clinic', observed=False)['wait_min'].count())
```

```text
clinic
North    3
South    2
East     1
Name: wait_min, dtype: int64
clinic
North    3
South    2
East     1
West     0
Name: wait_min, dtype: int64
```

The same order carries into `pivot_table` rows and columns.

# GroupBy Result-Shape Choices

Before choosing a method, decide what one row of the answer should represent: its grain. Aggregation changes the grain; transform and filter keep the original rows.

| Operation | Question it answers (clinic visits) | Rows returned | Grain of the result |
| --- | --- | --- | --- |
| `agg` | What is the mean wait at each clinic? | 3 | One row per clinic |
| `transform` | How does each visit compare with its clinic's mean? | 6, same index as `visits` | One row per visit |
| `filter` | Which visits belong to clinics with at least two visits? | 5 (East dropped) | One row per visit |
| `apply` | Which visit (the whole row) had the longest wait at each clinic? | Depends on the function (3 here) | Depends on the function (one visit per clinic here) |

Use `agg`, `transform`, or `filter` whenever one fits; a single number such as the longest wait is just `agg('max')`. `apply` is the fallback for custom per-group work, such as returning whole rows. It is slower, and its output shape depends on what your function returns.

## Transform Operations

**transform** computes a statistic for each group and copies it back to every row of that group. The result has the same length and index as the original table, so it can become a new column that compares each row with its group.

### Reference Card: Transform Operations

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `grouped.transform('mean')` | Broadcast each group's mean to its original rows | Series aligned to original index |
| `grouped.transform('std')` | Broadcast within-group spread | Series aligned to original index |
| `grouped.transform(lambda x: x - x.mean())` | Compute a custom within-group value | Same row count as input |
| `grouped.agg(['mean', 'std'])` | Compare with a reducing summary | One row per group |

### Code Snippet: Compare Each Visit with Its Clinic

```python
with_context = visits.copy()
with_context['clinic_mean_wait'] = visits.groupby('clinic')['wait_min'].transform('mean')
with_context['wait_vs_clinic'] = with_context['wait_min'] - with_context['clinic_mean_wait']
print(with_context[['clinic', 'patient_id', 'wait_min', 'clinic_mean_wait', 'wait_vs_clinic']])
```

```text
  clinic patient_id  wait_min  clinic_mean_wait  wait_vs_clinic
0  North        P01        12              11.0             1.0
1  North        P02        15              11.0             4.0
2  North        P01         6              11.0            -5.0
3  South        P03        20              25.0            -5.0
4  South        P04        30              25.0             5.0
5   East        P05         9               9.0             0.0
```

The new columns go on a `.copy()` (Lecture 04), so `visits` is unchanged for the next examples.

### Common Mistake: Attaching a Summary to Rows

```python
clinic_means = visits.groupby('clinic')['wait_min'].mean()   # 3 rows: East, North, South
wrong = visits.copy()
wrong['clinic_mean_wait'] = clinic_means
print(wrong[['clinic', 'wait_min', 'clinic_mean_wait']])
```

```text
  clinic  wait_min  clinic_mean_wait
0  North        12               NaN
1  North        15               NaN
2  North         6               NaN
3  South        20               NaN
4  South        30               NaN
5   East         9               NaN
```

pandas matches a new column to rows by index label (Lecture 04). The summary's labels are clinic names, and the rows are labeled 0-5, so nothing matches and every value is `NaN`, with no error. Copying the three numbers by position with `clinic_means.values` fails instead: `ValueError: Length of values (3) does not match length of index (6)`. `transform('mean')` returns one value per visit with the original index, so it lines up.

## Filter Operations

**filter** runs a test on each whole group and keeps every row of the groups that pass. Rows are kept or dropped, never changed.

### Reference Card: Filter Operations

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `grouped.filter(lambda x: len(x) > n)` | Keep groups with more than `n` rows | Original rows from passing groups |
| `grouped.filter(lambda x: x['col'].sum() > threshold)` | Keep groups meeting a total threshold | Filtered DataFrame |
| `grouped.filter(lambda x: x['col'].mean() > threshold)` | Keep groups meeting a mean threshold | Filtered DataFrame |

### Code Snippet: Keep Clinics with at Least Two Visits

```python
busy = visits.groupby('clinic').filter(lambda g: len(g) >= 2)
print(busy[['clinic', 'patient_id', 'wait_min']])
```

```text
  clinic patient_id  wait_min
0  North        P01        12
1  North        P02        15
2  North        P01         6
3  South        P03        20
4  South        P04        30
```

## Apply Operations

**apply** hands each group to your function as a small DataFrame and stitches the results together. `nlargest(1, 'wait_min')` returns the row with the longest wait.

### Reference Card: Apply Operations

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `grouped.apply(func, include_groups=False)` | Run your own function on each group; the function does not receive the grouping column | Depends on `func` |
| `grouped.apply(lambda x: x.sort_values('col'), include_groups=False)` | Sort rows inside each group | Combined DataFrame |
| `grouped.apply(lambda x: x.nlargest(2, 'col'), include_groups=False)` | Keep top two rows in each group | Combined DataFrame |

### Code Snippet: The Longest-Wait Visit at Each Clinic

```python
longest = visits.groupby('clinic').apply(
    lambda g: g.nlargest(1, 'wait_min'), include_groups=False
)
print(longest[['patient_id', 'wait_min']])
```

```text
         patient_id  wait_min
clinic
East   5        P05         9
North  1        P02        15
South  4        P04        30
```

The index has two levels: the clinic and each row's original index. `include_groups=False` means your function does not receive the `clinic` column. It is the only setting pandas 3 allows; writing it out makes pandas 2.2, which still passes the column by default, give the same result.

# LIVE DEMO!

# Pivot Tables and Cross-Tabulations

![Research vs. Practical](media/research.png)

*Think of pivot tables as the data analyst's Swiss Army knife - they can reshape, summarize, and analyze data in ways that would take dozens of lines of code to accomplish manually.*

In Lecture 06, `pivot()` rearranged a long table into a wide one and raised an error when an index/column pair appeared more than once. A **pivot table** handles those repeats: it groups rows by one key for the rows and another for the columns, aggregates each combination, and lays the results out as a grid. It is the two-key `groupby(...).mean().unstack()` from Grouping by Two Keys in a single call, with optional row and column totals called **margins**. A **cross-tabulation** (crosstab) is the special case that counts rows in each combination.

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
| `pd.pivot_table(df, values=..., index=..., columns=..., aggfunc='mean')` | Group by the `index` and `columns` keys and summarize `values` in each cell; `aggfunc` picks the summary (`'mean'`, `'sum'`, `'count'`, ...) | Wide table; absent combinations are `NaN` |
| `pd.pivot_table(..., aggfunc=['sum', 'mean'])` | Several summaries at once; `result['sum']` selects one | Two-level column labels |
| `pd.crosstab(df['a'], df['b'])` | Count rows for each combination of two columns; pass a list such as `[df['a'], df['c']]` for two-level rows | Counts; absent combinations are 0 |
| `pd.crosstab(df['a'], df['b'], values=df['v'], aggfunc='mean')` | Summarize a third column instead of counting, like `pivot_table` | Summary table; absent combinations are `NaN` |
| `pd.crosstab(df['a'], df['b'], margins=True)` | Add an `All` row and column of totals | Counts with totals |

### Code Snippet: A Pivot Table and Its GroupBy Twin

```python
mean_wait = pd.pivot_table(visits, values='wait_min', index='clinic',
                           columns='visit_type', aggfunc='mean')
print(mean_wait)

same = visits.groupby(['clinic', 'visit_type'])['wait_min'].mean().unstack()
print(mean_wait.equals(same))   # True

print(pd.crosstab(visits['clinic'], visits['visit_type'], margins=True))
```

```text
visit_type  Follow-up   New
clinic
East              9.0   NaN
North            10.5  12.0
South             NaN  25.0
True
visit_type  Follow-up  New  All
clinic
East                1    0    1
North               2    1    3
South               0    2    2
All                 3    3    6
```

### Missing Cells: Absent Is Not Zero

A `NaN` cell means no rows had that combination. `fill_value=0` is right for counts and sums, where "no visits" really is 0 visits (the crosstab above already shows 0). It is wrong for means, minimums, or other measurements: filling East–New with 0 would report a zero-minute wait that never happened.

## Advanced Pivot Operations

### Reference Card: Advanced Pivot Options

| Option | Purpose and key arguments | Output effect |
| :--- | :--- | :--- |
| `margins=True, margins_name='Total'` | Add a row and column of totals computed from the underlying rows; `margins_name` labels them (default `All`) | Extra `Total` row and column |
| `fill_value=0` | Replace absent cells with 0, only for counts and sums | No-`NaN` cells |
| `observed=True, sort=True` | For categorical keys: show only combinations present, in category order (both are the defaults) | Smaller, ordered table |
| `dropna=False` | Keep all-missing result rows and columns and show missing keys as a `NaN` row (also counted in margins) | Larger table |

### Code Snippet: Totals Where Zero Is Real

```python
total_wait = pd.pivot_table(visits, values='wait_min', index='clinic',
                            columns='visit_type', aggfunc='sum',
                            fill_value=0, margins=True, margins_name='Total')
print(total_wait)
```

```text
visit_type  Follow-up  New  Total
clinic
East                9    0      9
North              21   12     33
South               0   50     50
Total              30   62     92
```

East had no new-patient visits, so its total new-patient wait really is 0 minutes. The mean table above correctly leaves that cell `NaN`.

# LIVE DEMO!

# Performance Optimization

![xkcd 1319: Automation](media/xkcd_1319.png)

A grouped summary that takes a second on a class example can take many minutes on a year of hospital lab results. Before changing code, **measure** by timing the step and checking memory. You already have both tools. `%timeit` (Lecture 04) runs a line several times and reports the typical time. `df.memory_usage(deep=True)` (Lecture 05) reports bytes per column. Results depend on data size, dtypes, number of groups, and hardware, so measure your own workload.

![Performance benchmarks on about 100 million rows (lower is better): three separate groupby calls vs one agg, a per-group z-score with apply vs transform, and memory before and after categorical keys](media/perf_combined.png)

The benchmark (about 100 million rows) shows three changes that usually matter:

- One `.agg()` call computing several summaries beats calling `groupby()` again for each summary, because every new `groupby()` call redoes the split.
- Built-in aggregations named by string (`'mean'`, `'std'`, `transform('mean')`) run as fast compiled code. A Python `lambda` or `apply` runs once per group and can be 20-50× slower.
- A repeated text key stored as `category` (Lecture 05) uses far less memory.

## Measure, Then Optimize

### Reference Card: Measure, Then Optimize

| Task | Tool | Use when | Typical output |
| --- | --- | --- | --- |
| Measure time | `%timeit expression` | Comparing two ways to get the same result in a notebook | `8.8 ms ± 0.1 ms per loop ...` |
| Measure time | `time python analysis.py` | Timing a whole script in the terminal | `real 0m12.3s` |
| Measure memory | `df.memory_usage(deep=True)` | Finding the largest columns | Bytes per column |
| Faster | One `.agg(...)` with several summaries | You need several statistics per group | One pass, one table |
| Faster | String aggregations and `transform('mean')` instead of `lambda`/`apply` | The statistic is built in | Same numbers, much faster |
| Smaller | `.astype('category')` on repeated text keys | Few distinct values repeated many times | Less memory |

### Code Snippet: Measure Before and After

```python
rng = np.random.default_rng(0)
n = 1_000_000
labs = pd.DataFrame({
    'patient_id': rng.integers(0, 10_000, n),
    'clinic': rng.choice(['North', 'South', 'East', 'West'], n),   # one random clinic per row
    'glucose': rng.normal(100, 15, n).round(1),
})
print(round(labs['clinic'].memory_usage(deep=True) / 1e6, 1))  # 12.5 (MB as text)
labs['clinic'] = labs['clinic'].astype('category')
print(round(labs['clinic'].memory_usage(deep=True) / 1e6, 1))  # 1.0 (MB as category)

by_patient = labs.groupby('patient_id')['glucose']
%timeit by_patient.agg('std')              # about 9 ms
%timeit by_patient.agg(lambda s: s.std())  # about 450 ms: same values, ~50x slower
```

When data still do not fit in memory, see chunked reading and parallel processing in [BONUS.md](BONUS.md#scaling-past-memory-chunks-and-processes). The simpler move is often a bigger computer, which is the next topic.

# Remote Computing with SSH

![xkcd 2523: Endangered](media/xkcd_2523.png)

Some analyses cannot run on a laptop. The dataset may be too large or the job may run all night. Very often in health research, the data are not allowed to leave an approved secure server because they contain **protected health information** (PHI), health records that can identify a person. The answer is to bring your code to the data: log in to the other computer, run the work there, and bring back only results you are allowed to take. Use the hostname, account, and authentication instructions supplied by whoever operates the server.

**SSH** (Secure Shell) opens an encrypted terminal on another computer, called the **server** or **host**. Once connected, you type the same shell commands from Lectures 01-02 (`pwd`, `ls`, `cd`, `python`), but they run on the server. The prompt shows where you are:

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

The first time you connect, SSH shows the server's **host key fingerprint**, a code that identifies that server, and asks whether to trust it. Compare it with the fingerprint the server's administrator published; type `yes` only if they match. SSH remembers the server after that. If it asks for a password, nothing appears as you type; that is normal.

Most servers use an **SSH key pair** instead of a password. `ssh-keygen` creates two files. `~/.ssh/id_ed25519.pub`, the **public key**, works like a padlock: give copies to every server you use. `~/.ssh/id_ed25519`, the **private key**, is the only key that opens that padlock, so it stays on your laptop and is never shared, uploaded, or committed.

## Connect and Copy Files

### Reference Card: SSH File and Shell Commands

| Command | Purpose and key arguments | Result |
| :--- | :--- | :--- |
| `ssh user@host` | Open a remote shell | Remote prompt |
| `ssh -p port user@host` | Connect through a nondefault port | Remote prompt |
| `ssh user@host 'command'` | Run one command remotely | Command output locally |
| `scp local user@host:path` | Copy a local file to the server | Remote file |
| `scp user@host:path local` | Copy a remote file back | Local file |
| `ssh-keygen -t ed25519` | Create a public/private key pair | `~/.ssh/id_ed25519` and `~/.ssh/id_ed25519.pub` |
| `ssh-copy-id user@host` | Install the public key where supported | Passwordless key login |

### Code Snippet: Set Up a Key, Connect, and Copy Files

```bash
# Create a key once, then follow the server's instructions to install the public key
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-copy-id username@server.example

# Connect or run a single remote command
ssh username@server.example
ssh username@server.example 'hostname && uptime'

# Copy inputs to the server and results back
scp data.csv username@server.example:~/data/
scp username@server.example:~/results/analysis.csv ./
```

`ssh-keygen` first asks where to save the key; press Enter to accept the default location. It then asks for a passphrase that protects the private key file; set one. Run `ssh-add` and type the passphrase once: the **SSH agent**, a background program on your laptop, then remembers the unlocked key until you log out or restart, so `ssh`, `scp`, and `ssh-copy-id` stop asking for the passphrase every time. macOS runs an agent for you. If `ssh-add` says it cannot connect to your authentication agent (common in WSL), skip it and type the passphrase when each command asks.

## Keep Long Jobs Alive with tmux or screen

![Punk vs. Process](media/punk.png)

When an SSH connection drops (laptop sleeps, Wi-Fi changes), the server stops the programs started from that connection, including an analysis three hours into a four-hour run. A **terminal multiplexer** such as `tmux` keeps a shell running on the server on its own. You **detach**, disconnect, and later **attach** again to find the job still running. A tmux session survives disconnects, not a server restart. `screen` is an older alternative; use whichever the server provides. For a guided introduction, see [Tmux Fundamentals](https://linuxhandbook.com/courses/tmux/).

```text
laptop ── ssh ──> server
                   └─ tmux session "analysis"   (keeps running after you disconnect)
                        └─ python analysis.py
```

### Reference Card: tmux Sessions

- `tmux new -s analysis`: Start a session named `analysis`; a status bar appears at the bottom.
- `Ctrl+b`, then `d`: Detach; the session keeps running and you return to the normal prompt.
- `tmux ls` (short for `tmux list-sessions`): List sessions; output looks like `analysis: 1 windows (created Fri Sep 18 15:32:45 2026)`.
- `tmux attach -t analysis`: Reattach to the session.
- `exit` inside the session: End it when the job is done.
- `tmux kill-session -t analysis`: End a detached session from outside, stopping anything still running in it.
- Alternative: `screen -S analysis`; detach with `Ctrl+a`, then `d`; reattach with `screen -r analysis`.

### Code Snippet: Keep an Analysis Running

```bash
tmux new -s analysis          # on the server, after ssh
source .venv/bin/activate     # the project's environment on the server
time python analysis.py       # start the long job; time reports how long it took
# press Ctrl+b, then d to detach; now it is safe to disconnect
tmux ls                       # later, after reconnecting with ssh
tmux attach -t analysis
```

## Run Jupyter Through an SSH Tunnel

`jupyter lab` (Lecture 04) starts Jupyter as a small web server that your browser talks to. Programs like this listen on a numbered **port**; Jupyter uses 8888 by default. On a shared server, start Jupyter so it accepts connections only from `127.0.0.1`, also called **localhost** ("this same computer"), so nobody else on the network can reach it. Then open an **SSH tunnel**, a private pipe inside your SSH connection that carries anything sent to port 8888 on your laptop to port 8888 on the server. Your browser acts as if Jupyter were local; the kernel, files, memory, and CPU are on the server.

```text
laptop browser → localhost:8888 ══ SSH tunnel ══> server localhost:8888 → Jupyter
```

### Reference Card: Jupyter Through a Tunnel

- `jupyter lab --ip=127.0.0.1 --port=8888 --no-browser` (on the server, inside tmux): Start Jupyter for localhost only; it prints a URL ending in `?token=...`, a password generated for this session.
- `ssh -N -L 8888:127.0.0.1:8888 user@host` (in a second, local terminal): `-L laptop_port:server_address:server_port` builds the tunnel; `-N` opens no remote shell, so the terminal shows nothing while the tunnel is open.
- Paste the printed `http://127.0.0.1:8888/lab?token=...` URL into your laptop's browser.
- `Ctrl+C` in the tunnel terminal: Close the tunnel; Jupyter keeps running in tmux.
- If port 8888 is already in use on your laptop (for example, by a local Jupyter), use `-L 8889:127.0.0.1:8888` and change 8888 to 8889 in the URL.

### Code Snippet: Start Jupyter and Open the Tunnel

On the server, start a persistent session first, then launch Jupyter inside it:

```bash
tmux new -s notebooks
source .venv/bin/activate     # an environment with JupyterLab installed
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
```

In a second, local terminal:

```bash
ssh -N -L 8888:127.0.0.1:8888 username@server.example
```

# LIVE DEMO!
