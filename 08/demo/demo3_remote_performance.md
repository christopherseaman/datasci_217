---
jupyter:
  jupytext:
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    name: python
    version: 3.13
---

# Demo 3: Apply, Measure, and Keep a Long Job Running

One million synthetic fasting-glucose results, then a practice session for a remote server.

- **Parts 1 and 2** run in this notebook: a custom `apply` summary per clinic, then timings of grouped summaries and the memory a repeated text key costs.
- **Part 3** runs in a terminal on your own computer: it creates a practice SSH key pair and keeps a long job alive in tmux, the steps you repeat on a remote server.

Run the notebook cells from top to bottom; each **Expect** line says what the output should show. Timings depend on the computer, so compare the ratio between two timings, not the exact milliseconds. Patient IDs and values are synthetic.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

## Part 1: Apply Runs Your Own Function per Group

`apply` hands each group to your function as a small DataFrame and stitches the results together. Use it when no built-in aggregation fits; its output shape depends on what your function returns.

### Build One Million Lab Results

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 1_000_000
labs = pd.DataFrame({
    "patient_id": rng.integers(0, 10_000, n),                # 10,000 patients, about 100 results each
    "clinic": rng.choice(["North", "South", "East", "West"], n),
    "glucose": rng.normal(100, 15, n).round(1),              # fasting glucose, mg/dL
})
# One age per patient (drawn last so the glucose values stay the same), placed after patient_id
labs.insert(1, "age", rng.integers(18, 96, 10_000)[labs["patient_id"]])
print(labs.shape)
print(f"Patients: {labs['patient_id'].nunique():,}")
display(labs.head())
```

**Expect:** `(1000000, 4)`, `Patients: 10,000`, and five rows with patient number, age (18 to 95), clinic, and a glucose value around 100 mg/dL. The grain is one row per lab result.

### A Custom Summary per Clinic

```python
def glucose_summary(group):
    """Summarize one clinic's fasting glucose (mg/dL) as a Series."""
    q25 = group["glucose"].quantile(0.25)
    q75 = group["glucose"].quantile(0.75)
    return pd.Series({
        "results": len(group),
        "median": group["glucose"].median(),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "over_126": (group["glucose"] > 126).sum(),
    })

clinic_summary = labs.groupby("clinic").apply(glucose_summary, include_groups=False)
display(clinic_summary)

# Checkpoint: the medians match the built-in aggregation
builtin = labs.groupby("clinic")["glucose"].median()
print("Same medians as agg('median')?", clinic_summary["median"].equals(builtin))
```

**Expect:** one row per clinic, like `agg`, with the columns your function named. Counts show `.0` because a `Series` that holds a median stores all its values as floats. Each clinic has about 250,000 results (East 249,785, North 249,815, South 250,430, West 249,970), a median near 100 mg/dL, an interquartile range of about 20 mg/dL, and about 10,300 to 10,450 results over 126 mg/dL. Then `True`.

### The Highest Glucose in Each Clinic

```python
highest = labs.groupby("clinic").apply(
    lambda g: g.nlargest(2, "glucose"), include_groups=False
)
display(highest[["patient_id", "age", "glucose"]])
print("Rows:", len(highest))
```

**Expect:** 8 rows: the two highest glucose results in each clinic, with the clinic as the outer index level and each result's original row number as the inner one. North has the highest value, 175.0 mg/dL (patient 9336, age 84). When the function returns whole rows, `apply` returns rows; when it returns one `Series` per group, it returns one row per group. When `agg` or `transform` can do the job, prefer them: they run as compiled code instead of once per group.

## Part 2: Measure, Then Optimize

### One `.agg()` Instead of Three `groupby()` Calls

Before timing two ways of getting an answer, check that they give the same answer.

```python
def three_calls(df):
    """Mean, SD, and count of glucose per patient, one groupby() call each."""
    mean = df.groupby("patient_id")["glucose"].mean()
    sd = df.groupby("patient_id")["glucose"].std()
    count = df.groupby("patient_id")["glucose"].count()
    return mean, sd, count

def one_agg(df):
    """The same three summaries from one groupby() and one .agg() call."""
    return df.groupby("patient_id")["glucose"].agg(["mean", "std", "count"])

# Checkpoint: identical results
mean, sd, count = three_calls(labs)
together = one_agg(labs)
print(mean.equals(together["mean"]), sd.equals(together["std"]), count.equals(together["count"]))
```

**Expect:** `True True True`.

```python
%timeit three_calls(labs)
%timeit one_agg(labs)
```

**Expect:** two lines such as `55.8 ms ± 217 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)` and `32.2 ms ± 112 μs per loop (...)`. The single `.agg()` is about 1.5 to 2 times faster, because every `groupby()` call splits the million rows again.

### Built-in Aggregations vs a Lambda

```python
by_patient = labs.groupby("patient_id")["glucose"]
fast = by_patient.agg("std")
slow = by_patient.agg(lambda s: s.std())
print("Largest difference:", (fast - slow).abs().max())
```

**Expect:** a number around `1e-14`: the same values, apart from rounding in the last decimal place.

```python
%timeit by_patient.agg("std")
%timeit by_patient.agg(lambda s: s.std())
```

**Expect:** about 10 ms against about 450 ms, so the lambda is roughly 50 times slower. It runs as Python once for each of the 10,000 patients; `"std"` runs as one compiled loop over all of them.

### `transform`: Built-in vs Lambda

```python
centered_fast = labs["glucose"] - by_patient.transform("mean")
centered_slow = by_patient.transform(lambda s: s - s.mean())
print("Largest difference:", (centered_fast - centered_slow).abs().max())
print("Same index as labs?", centered_fast.index.equals(labs.index))
```

**Expect:** a difference around `6e-14`, then `True`: both give each result's distance from its patient's own mean, one value per lab row.

```python
%timeit labs["glucose"] - by_patient.transform("mean")
%timeit by_patient.transform(lambda s: s - s.mean())
```

**Expect:** about 10 ms against more than a second: over 100 times slower.

### Memory: Text Key vs `category`

```python
display(labs.memory_usage(deep=True))

text_mb = labs["clinic"].memory_usage(deep=True) / 1e6
labs["clinic"] = labs["clinic"].astype("category")
category_mb = labs["clinic"].memory_usage(deep=True) / 1e6
print(f"\nclinic as text:     {text_mb:.1f} MB")
print(f"clinic as category: {category_mb:.1f} MB")
```

**Expect:** `clinic` is the largest column: 53,500,245 bytes (about 12.5 million in Colab, as explained below), against 8,000,000 for each number column. Then `53.5 MB` as text and `1.0 MB` as a category: four labels stored once, plus one small code per row. Colab has the `pyarrow` package installed, which stores text more compactly, so there the text line reads about `12.5 MB`; the category version is still far smaller.

## Part 3: Keep a Long Job Running (Terminal)

This part runs on your own computer, which plays the server: you run the commands you would type after `ssh`, and closing a terminal window stands in for a dropped connection. This rehearses key creation and tmux locally; it does not test an SSH login, file copy, or tunnel. Those need an actual server account and its administrator's connection instructions. Type the commands below into a terminal, not into this notebook.

Check that tmux is installed:

```shell
tmux -V
```

**Expect:** a version such as `tmux 3.6`; any 3.x works. If you see `command not found`, install it with `sudo apt install tmux` (Ubuntu or WSL) or, with Homebrew on macOS, `brew install tmux`.

### Step 1: Make a Practice Key Pair

Work in a new folder, and use `-f demo_key` so the practice pair is saved there under its own name. Without `-f`, `ssh-keygen` offers `~/.ssh/id_ed25519`, and saving over that file would replace your real key.

```shell
mkdir -p ~/ds217_ssh_practice
cd ~/ds217_ssh_practice
ssh-keygen -t ed25519 -f demo_key -C "ds217 practice key"
```

When it asks for a passphrase, type one (such as `practice`) and press Enter; nothing appears as you type. Type it again to confirm.

**Expect:** (your fingerprint and picture will differ)

```text
Generating public/private ed25519 key pair.
Your identification has been saved in demo_key
Your public key has been saved in demo_key.pub
The key fingerprint is:
SHA256:Quuk5GjxuLAT+BgTU7tk7mIv5JZ9MdQsKg0F0JYgl1M ds217 practice key
The key's randomart image is:
+--[ED25519 256]--+
...
+----[SHA256]-----+
```

```shell
ls -l demo_key*
cat demo_key.pub
```

**Expect:**

```text
-rw------- 1 you you 464 Sep 24 23:50 demo_key
-rw-r--r-- 1 you you 100 Sep 24 23:50 demo_key.pub
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICm6gIA8vEmA6qytDJwgDG4zKtuAyeK7Iq3OA9m5BPFe ds217 practice key
```

`demo_key` is the private key: `-rw-------` means only you can read it, and it never leaves this computer. The one line in `demo_key.pub` is the padlock: on a real server, `ssh-copy-id` installs it, or the server's instructions tell you where to paste it.

### Step 2: Keep a Job Running in tmux

<!-- #region -->
In VS Code (or any editor), save this stand-in for a long analysis as `long_job.py` in `~/ds217_ssh_practice`:

```python
import time

# One progress line per second for two minutes
for step in range(1, 121):
    print("step", step, "of 120")
    time.sleep(1)  # wait one second
print("done")
```
<!-- #endregion -->

Start a session, then start the job inside it:

```shell
tmux new -s analysis
time python3 long_job.py
```

**Expect:** a status bar across the bottom with `[analysis]` at its left, then `step 1 of 120`, `step 2 of 120`, ... once a second. (If `python3` is not found, run `uv run python3 long_job.py` instead.)

Press `Ctrl+b`, let go, then press `d`.

**Expect:** `[detached (from session analysis)]` and your normal prompt. The job is still running inside the session.

```shell
tmux ls
```

**Expect:** `analysis: 1 windows (created Thu Sep 24 23:50:11 2026)`, with your date and time.

Now rehearse losing your terminal: open a **new** terminal window, then close the old one. (Open the new one first; on Windows, closing every WSL window can shut WSL down.) In the new window:

```shell
tmux ls
tmux attach -t analysis
```

**Expect:** the session is still listed, and after you attach, the count has kept going while no window was showing it: if you were away 30 seconds, it is about 30 steps further along. When the job finishes, it prints `done`, and `time` reports that it ran for about two minutes: bash prints a line such as `real 2m0.130s`, and zsh (the macOS default) ends its line with `2:00.13 total`.

End the session and check that nothing is left running:

```shell
exit
tmux ls
```

**Expect:** `[exited]`. If this was your only session, `tmux ls` then says `no server running on /tmp/tmux-1000/default` (the path differs by computer); any other sessions you already had stay listed.

### Optional Extension: Run Jupyter Inside tmux

Prerequisite: BONUS.md, Run Jupyter Through an SSH Tunnel. Not needed for the core route.

- On a server, Jupyter runs inside tmux and your browser reaches it through an SSH tunnel. This rehearses everything except the tunnel.
- `~/08-demo` has JupyterLab because its `pyproject.toml` lists it. Colab users: run the local setup commands first.

```shell
cd ~/08-demo
tmux new -s notebooks
source .venv/bin/activate
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
```

**Expect:** log lines ending in a URL such as `http://127.0.0.1:8888/lab?token=...`. Copy that URL into your browser, and JupyterLab opens. If port 8888 is busy, Jupyter picks 8889 and prints that URL instead.

Detach with `Ctrl+b`, then `d`, and reload the browser page.

**Expect:** JupyterLab still works, because Jupyter keeps running inside tmux.

Clean up: `tmux attach -t notebooks`, press `Ctrl+C` twice to stop Jupyter, then type `exit`.

**Expect:** `[exited]` after `exit`. The practice files remain in `~/ds217_ssh_practice`; keep its private key out of Git, just like a real one.
