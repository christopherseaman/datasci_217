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

# Demo 3: A1c Features for a Diabetes Model

The diabetes care team wants one row per patient, with each patient's number of A1c tests, mean, highest, and latest value, to feed a risk model. The source is the health system's 2025 lab extract, 2,000,000 synthetic results with one row per result. The notebook reviews the extreme values, builds the features, and measures which code is fast enough for a table this size; then a terminal session runs the whole build as a long job inside tmux, the way it would run on a server.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

This supplied cell downloads `generate_data.py` and `build_features.py` when they are missing, runs `generate_data.build()` to write the tables into `data/` if they are not there yet (about 30 seconds), then loads the lab results.

```python
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/08/demo"
for filename in ["generate_data.py", "build_features.py"]:
    if not Path(filename).exists():
        urlretrieve(f"{REPO_RAW}/{filename}", filename)
if not all(Path(f"data/{name}.parquet").exists() for name in ["clinics", "visits", "labs"]):
    import generate_data
    generate_data.build()

labs = pd.read_parquet("data/labs.parquet")
print(labs.shape)
```

Expect `(2000000, 6)`.

## Core walkthrough

### 1. Name the grain

```python
display(labs.head())
display(labs["test_name"].value_counts())
print(f"Patients: {labs['patient_id'].nunique():,}")
print(f"No result (cancelled samples): {labs['result'].isna().sum():,}")
```

Expect five results with patient, clinic, collection time, test, result, and unit, then four tests from Creatinine (887,471 results) and Glucose (887,362), drawn together as one panel, down to LDL (106,462), then 189,607 patients and 10,941 results with no value. One row per result, so one blood draw can fill several rows.

### 2. Review each clinic's two highest glucose results

A lab's critical-value review starts with the most extreme results at each site. `apply` hands each clinic's rows to `nlargest`.

```python
glucose = labs[labs["test_name"] == "Glucose"]
top2_raw = glucose.groupby("clinic").apply(lambda g: g.nlargest(2, "result"), include_groups=False)
display(top2_raw[["patient_id", "collected_at", "result"]])
```

Expect 16 rows, two per clinic, indexed by clinic and then each result's original row number. The other clinics' highest results are 385 to 654 mg/dL, emergencies worth a call, but Excelsior's two are `9999.0`: no blood holds that much glucose. Parkside's two belong to one patient, P113888, 9 days apart. 9999 is the code one Excelsior analyzer writes for a failed run, so it is not a result.

Mark the error codes as missing (Lecture 05), then repeat the review with the `apply`-free version: sort once, then keep the first two rows of each clinic.

```python
print("Error codes:", (labs["result"] == 9999).sum())
labs.loc[labs["result"] == 9999, "result"] = np.nan

glucose = labs[labs["test_name"] == "Glucose"]
top2 = glucose.groupby("clinic").apply(lambda g: g.nlargest(2, "result"), include_groups=False)
top2_sorted = glucose.sort_values("result", ascending=False).groupby("clinic").head(2)
display(top2_sorted[["clinic", "patient_id", "result"]].sort_values("clinic"))
print("Same rows both ways?", sorted(top2.index.get_level_values(1)) == sorted(top2_sorted.index))
```

Expect `Error codes: 23`, then 16 rows with a flat index; Excelsior's highest is now a real result, 564.0 mg/dL. Then `True`: both ways keep the same rows.

Plot the two reviews side by side, one dot per result.

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
sns.scatterplot(data=top2_raw.reset_index(), x="result", y="clinic", s=60, ax=axes[0])
axes[0].set(title="Before: two highest glucose per clinic", xlabel="Glucose (mg/dL)", ylabel="")
sns.scatterplot(data=top2.reset_index(), x="result", y="clinic", s=60, ax=axes[1])
axes[1].set(title="After marking 9999 as missing", xlabel="Glucose (mg/dL)")
plt.show()
```

Expect the left panel's dots bunched under 700 at the left edge, except Excelsior's dot alone at 9999 (its two results are identical, so they overlap). On the right the x-axis runs from about 380 to 660, and every clinic's two dots sit between 385 and 654 mg/dL.

### 3. Each patient's latest A1c

The model needs the most recent A1c, not the average of old ones. Sort by collection time, then keep each patient's last row.

```python
a1c = labs[labs["test_name"] == "A1c"].dropna(subset=["result"])
latest = a1c.sort_values("collected_at").groupby("patient_id").tail(1)
display(latest.head())
print(f"A1c results: {len(a1c):,}  patients: {a1c['patient_id'].nunique():,}  latest rows: {len(latest):,}")
print(f"Latest A1c of 9% or more: {(latest['result'] >= 9).mean():.1%}")
```

Expect five latest rows (patients whose only A1c was drawn on January 2), then 118,032 A1c results from 90,281 patients and exactly one latest row per patient. 1.9% of patients have a latest A1c of 9% or more, poorly controlled diabetes; most of the others had one yearly screening test.

The red lines mark the diabetes cutoff (6.5%) and poor control (9%); `ax.axvline(x)` draws a vertical line at `x`.

```python
ax = latest["result"].plot.hist(bins=60, figsize=(8, 3.5))
for cutoff in [6.5, 9]:
    ax.axvline(cutoff, color="red", linestyle="--")
ax.set(title="Latest A1c per patient, 2025", xlabel="A1c (%)", ylabel="Patients")
plt.show()
```

Expect a tall peak near 5.3%, below the diabetes cutoff at 6.5%, and a second, lower hump near 7.5% from patients with diabetes. The thin tail past the 9% line is the 1.9%.

### 4. One split, several summaries, timed

Two ways to get the same count, mean, standard deviation, and maximum per patient. Check that they agree before timing them.

```python
def four_calls(df):
    """Count, mean, SD, and max of A1c per patient, one groupby() call each."""
    return [df.groupby("patient_id")["result"].agg(f) for f in ["count", "mean", "std", "max"]]

def one_agg(df):
    """The same four summaries from one groupby() and one .agg() call."""
    return df.groupby("patient_id")["result"].agg(["count", "mean", "std", "max"])

together = one_agg(a1c)
print(all(part.equals(together[name]) for part, name in zip(four_calls(a1c), together.columns)))
```

Expect `True`.

```python
%timeit four_calls(a1c)
%timeit one_agg(a1c)
```

Expect two lines such as `176 ms ± 799 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)` and `50.7 ms ± 120 μs per loop (...)`. The one `.agg()` is about 3.5 times faster, because every `groupby()` call splits the 118,032 rows again. Timings depend on the computer, so compare the ratio, not the milliseconds.

### 5. Built-in aggregation vs a lambda

```python
by_patient = a1c.groupby("patient_id")["result"]
print("Largest difference:", (by_patient.agg("std") - by_patient.agg(lambda s: s.std())).abs().max())
```

Expect a number near `1e-15`: the same values, apart from rounding in the last decimal place.

The lambda runs as Python once per patient, so one run takes seconds; `-n 1 -r 3` asks `%timeit` for three single runs instead of its usual dozens.

```python
%timeit by_patient.agg("std")
%timeit -n 1 -r 3 by_patient.agg(lambda s: s.std())
```

Expect about 3 ms against about 4 s: the lambda is over 1,000 times slower, because it runs once for each of 90,281 patients while `"std"` is one compiled loop over all of them.

### 6. Memory: text keys vs `category`

```python
before = labs.memory_usage(deep=True)
compact = labs.copy()
for column in ["clinic", "test_name", "unit"]:
    compact[column] = compact[column].astype("category")
after = compact.memory_usage(deep=True)
memory = pd.DataFrame({"text": before / 1e6, "category": after / 1e6}).drop(index="Index")
ax = memory.plot.barh(figsize=(7, 3.5))
ax.set(title="Memory per column, 2,000,000 lab results", xlabel="MB")
plt.show()
print(f"Total: {before.sum() / 1e6:.0f} MB as text, {after.sum() / 1e6:.0f} MB with category keys")

same = compact.groupby("test_name", observed=True)["result"].mean().equals(labs.groupby("test_name")["result"].mean())
print("Same test means?", same)
```

Expect the `clinic`, `test_name`, and `unit` bars to shrink from 25 to 32 MB each to 2.0 MB each (one small code per row plus a few labels), while `patient_id` and the number columns stay the same. The total falls from 150 MB to 68 MB, and the test means are unchanged (`True`). `patient_id` stays text: with 189,607 distinct values, it is not a repeated key.

### 7. A practice key pair (terminal)

Steps 7 and 8 run in VS Code's terminal on your own computer, in the `~/08-demo` folder where you ran this notebook. Colab users first set it up there with `curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/08/demo/setup_demo.sh | sh`, then run this notebook once so `data/` exists. Type the commands into the terminal, not into the notebook. Your own computer plays the server: you run the commands you would type after `ssh`.

Make the key pair in a practice folder, and use `-f demo_key` so it is saved there under its own name. Without `-f`, `ssh-keygen` offers `~/.ssh/id_ed25519`, and saving over that file would replace your real key.

```shell
mkdir -p ~/ds217_ssh_practice
cd ~/ds217_ssh_practice
ssh-keygen -t ed25519 -f demo_key -C "ds217 practice key"
```

When it asks for a passphrase, type one (such as `practice`) and press Enter; nothing appears as you type. Type it again to confirm.

Expect (your fingerprint and picture will differ):

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

Expect:

```text
-rw------- 1 you you 464 Oct 29 14:50 demo_key
-rw-r--r-- 1 you you 100 Oct 29 14:50 demo_key.pub
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICm6gIA8vEmA6qytDJwgDG4zKtuAyeK7Iq3OA9m5BPFe ds217 practice key
```

`demo_key` is the private key: `-rw-------` means only you can read it, and it never leaves this computer. The one line in `demo_key.pub` is the padlock: on a real server, `ssh-copy-id` installs it, or the server's instructions say where to paste it.

### 8. Run the feature build inside tmux (terminal)

`build_features.py` rebuilds the A1c features one month at a time, prints one progress line per month, and writes the year-end table to `output/a1c_features.csv`. It pauses 10 seconds per month to stand in for a job that takes hours, so it runs about two minutes.

Check that tmux is installed:

```shell
tmux -V
```

Expect a version such as `tmux 3.6`; any 3.x works. If you see `command not found`, install it with `sudo apt install tmux` (Ubuntu or WSL) or `brew install tmux` (macOS).

Start a session in the demo folder, then start the job inside it:

```shell
cd ~/08-demo
tmux new -s features
source .venv/bin/activate
time python3 build_features.py
```

Expect a status bar across the bottom with `[features]` at its left, then `month 1 of 12: 8,627 patients`, and a new line every 10 seconds.

Press `Ctrl+b`, let go, then press `d`.

Expect `[detached (from session features)]` and your normal prompt. The job is still running inside the session.

```shell
tmux ls
```

Expect `features: 1 windows (created Thu Oct 29 14:52:11 2026)`, with your date and time.

Now rehearse losing your connection: open a **new** terminal window, then close the old one. (Open the new one first; on Windows, closing every WSL window can shut WSL down.) In the new window:

```shell
tmux ls
tmux attach -t features
```

Expect the session still listed, and after you attach, more month lines than when you left: the job kept running with no window showing it. It ends with `month 12 of 12: 90,281 patients` and `done: wrote output/a1c_features.csv`, and `time` reports about two minutes: bash prints a line such as `real 2m1.9s`, and zsh (the macOS default) ends its line with `2:01.93 total`.

End the session and check the result:

```shell
exit
tmux ls
head -3 ~/08-demo/output/a1c_features.csv
```

Expect `[exited]`, then `no server running on /tmp/tmux-1000/default` if this was your only session (the path differs by computer), then the header `patient_id,tests,mean_a1c,max_a1c,latest_a1c` and two patients' features.

## Independent practice

Run the core walkthrough first.

### 9. Each patient's two most recent A1c results

Sort inside each patient with `apply`, keep the last two rows, and see the result's two-level index.

```python
dm = a1c[a1c["patient_id"].isin(latest.loc[latest["result"] >= 6.5, "patient_id"])]
last_two = dm.groupby("patient_id").apply(
    lambda g: g.sort_values("collected_at").tail(2), include_groups=False,
)
display(last_two[["collected_at", "result"]].head(6))

change = last_two.groupby("patient_id")["result"].agg(["first", "last", "count"])
change = change[change["count"] == 2]
change["change"] = change["last"] - change["first"]
print(f"Patients with two results: {len(change):,}")
print(f"Improved by 0.5 points or more: {(change['change'] <= -0.5).mean():.1%}")
```

Expect a two-level index (patient, then original row number) with up to two rows per patient, oldest first; P000021 went from 8.2 to 7.3, P000025 rose from 7.5 to 8.0, and P000035 has only one result. 14,878 patients with a latest A1c of 6.5% or more have two results, and 10.7% of them improved by half a point or more between their last two tests.

### 10. `transform`: built-in vs lambda

```python
centered_fast = a1c["result"] - by_patient.transform("mean")
centered_slow = by_patient.transform(lambda s: s - s.mean())
print("Largest difference:", (centered_fast - centered_slow).abs().max())
print("Same index as a1c?", centered_fast.index.equals(a1c.index))
```

Expect a difference near `4e-15` and `True`.

The lambda takes more than 10 seconds on all 90,281 patients, so `-r 1` times one run.

```python
%timeit a1c["result"] - by_patient.transform("mean")
%timeit -n 1 -r 1 by_patient.transform(lambda s: s - s.mean())
```

Expect about 3 ms against about 15 s: over 4,000 times slower.

### 11. A custom summary per clinic

```python
def a1c_summary(group):
    """Summarize one clinic's A1c results as a Series."""
    return pd.Series({
        "results": len(group),
        "median": group["result"].median(),
        "share_9_or_more": (group["result"] >= 9).mean(),
    })

clinic_summary = a1c.groupby("clinic").apply(a1c_summary, include_groups=False)
display(clinic_summary.round(3))
print("Same medians as agg('median')?", clinic_summary["median"].equals(a1c.groupby("clinic")["result"].median()))
```

Expect one row per clinic (the function returns one `Series` per group), medians from 5.6% (Marina) to 6.4% (Bayview), 2.4% to 5.3% of results at 9% or more, and `True`. Bayview and Excelsior, in the most deprived neighborhoods, have the highest medians and shares; Marina, in the least deprived, the lowest. `results` shows `.0` because a `Series` holding a share stores every value as a float.

### 12. Category keys also group faster

```python
%timeit labs.groupby("test_name")["result"].mean()
%timeit compact.groupby("test_name", observed=True)["result"].mean()
```

Expect the category version about 2 times faster: pandas groups small integer codes instead of comparing text.

## Optional extension: run Jupyter inside tmux

Prerequisite: BONUS.md, Run Jupyter Through an SSH Tunnel.

On a server, Jupyter runs inside tmux and your browser reaches it through an SSH tunnel. This rehearses everything except the tunnel, in the same `~/08-demo` folder, whose `pyproject.toml` lists JupyterLab.

```shell
cd ~/08-demo
tmux new -s notebooks
source .venv/bin/activate
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
```

Expect log lines ending in a URL such as `http://127.0.0.1:8888/lab?token=...`. Copy that URL into your browser, and JupyterLab opens. If port 8888 is busy, Jupyter picks 8889 and prints that URL instead.

Detach with `Ctrl+b`, then `d`, and reload the browser page. Expect JupyterLab still working, because Jupyter keeps running inside tmux.

Clean up: `tmux attach -t notebooks`, press `Ctrl+C` twice to stop Jupyter, then type `exit`. Expect `[exited]`. The practice key stays in `~/ds217_ssh_practice`; keep it out of Git, like a real one.
