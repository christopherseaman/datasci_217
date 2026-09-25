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

# Demo 3: Measure, Then Optimize, and Keep a Long Job Running

Part 1 runs here in the notebook: it times grouped summaries of one million synthetic fasting-glucose results and measures how much memory a repeated text key costs. Part 2 runs in a terminal on your own computer: it creates a practice SSH key pair and keeps a long job alive in tmux, the steps you will repeat on a remote server. Everything here comes from Lecture 08 up to the third demo break, plus Lectures 01 to 07.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03), and run Part 1 from top to bottom; after each step, an **Expect** line says what you should see. Timings depend on the computer, so compare the ratios between two timings, not the exact milliseconds; Colab is usually slower than a recent laptop. Part 1 takes about a minute, most of it in the `%timeit` cells. Part 2 needs a terminal on your own computer (macOS Terminal, Linux, WSL Ubuntu on Windows, or VS Code's integrated terminal), not Colab. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-25 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, OpenSSH 10.2, and tmux 3.6. The patient IDs and values are synthetic.

## Setup

The first cell installs pandas 3.0.5, the course version. Colab ships an older pandas (2.2). A `.venv` made with `uv venv --seed` includes pip, so the same `%pip` cell works locally too.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.`, perhaps after a notice that a newer pip is available; neither needs any action. Locally, with the requirements already installed, the cell changes nothing and you can go on. In Colab, pip may also print a dependency conflict because some preinstalled packages expect pandas 2.2; that is expected, and this demo does not use them. If Colab asks you to restart the session (or says pandas was previously imported), choose **Runtime → Restart session**, then continue with the next cell. You do not need to rerun the install.

```python
import numpy as np
import pandas as pd

print('pandas', pd.__version__)
```

**Expect:** `pandas 3.0.5`.

## Part 1: Measure, Then Optimize

### Build One Million Lab Results

```python
rng = np.random.default_rng(0)
n = 1_000_000
labs = pd.DataFrame({
    "patient_id": rng.integers(0, 10_000, n),                # 10,000 patients, about 100 results each
    "clinic": rng.choice(["North", "South", "East", "West"], n),
    "glucose": rng.normal(100, 15, n).round(1),              # fasting glucose, mg/dL
})
print(labs.shape)
print(f"Patients: {labs['patient_id'].nunique():,}")
labs.head()
```

**Expect:** `(1000000, 3)`, `Patients: 10,000`, and five rows of patient numbers, clinic names, and glucose values around 100 mg/dL. The grain is one row per lab result.

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

**Expect:** about 10 ms against about 450 ms, so the lambda is roughly 50 times slower. It runs as Python once for each of the 10,000 patients; `"std"` runs as one compiled loop over all of them. This cell takes several seconds, because `%timeit` runs the slow version seven times.

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

**Expect:** about 10 ms against more than a second: over 100 times slower. This cell takes about ten seconds.

### Memory: Text Key vs `category`

```python
print(labs.memory_usage(deep=True))

text_mb = labs["clinic"].memory_usage(deep=True) / 1e6
labs["clinic"] = labs["clinic"].astype("category")
category_mb = labs["clinic"].memory_usage(deep=True) / 1e6
print(f"\nclinic as text:     {text_mb:.1f} MB")
print(f"clinic as category: {category_mb:.1f} MB")
```

**Expect:** `clinic` is the largest column: 53,500,245 bytes (about 12.5 million in Colab, as explained below), against 8,000,000 for each number column. Then `53.5 MB` as text and `1.0 MB` as a category: four labels stored once, plus one small code per row. Colab has the `pyarrow` package installed, which stores text more compactly, so there the text line reads about `12.5 MB`; the category version is still far smaller.

## Part 2: Keep a Long Job Running (Terminal)

This part runs on your own computer, which plays the server: you run the commands you would type after `ssh`, and closing a terminal window stands in for a dropped connection. Type the commands below into a terminal, not into this notebook.

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

**Expect:** a status bar across the bottom with `[analysis]` at its left, then `step 1 of 120`, `step 2 of 120`, ... once a second. (If `python3` is not found, run `uv run python long_job.py` instead.)

Press `Ctrl+b`, let go, then press `d`.

**Expect:** `[detached (from session analysis)]` and your normal prompt. The job is still running inside the session.

```shell
tmux ls
```

**Expect:** `analysis: 1 windows (created Thu Sep 24 23:50:11 2026)`, with your date and time.

Now drop the connection: open a **new** terminal window, then close the old one. (Open the new one first; on Windows, closing every WSL window can shut WSL down.) In the new window:

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

**Expect:** `[exited]`, then `no server running on /tmp/tmux-1000/default` (the path differs by computer): no sessions are left.

### Step 3: Run Jupyter Inside tmux

On a server, Jupyter runs inside tmux and your browser reaches it through an SSH tunnel. On your own computer you can rehearse everything except the tunnel. Go to a project folder whose `.venv` has JupyterLab (the Lecture 03 setup; add it with `uv pip install jupyterlab` if needed):

```shell
tmux new -s notebooks
source .venv/bin/activate
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
```

**Expect:** log lines ending in a URL such as `http://127.0.0.1:8888/lab?token=...`. Copy that URL into your browser, and JupyterLab opens. (If port 8888 is busy, Jupyter picks 8889 and prints that URL instead.)

Detach with `Ctrl+b`, then `d`, and reload the browser page.

**Expect:** JupyterLab still works, because Jupyter keeps running inside tmux. On a server, the one extra step is the `ssh -N -L 8888:127.0.0.1:8888 ...` tunnel from the lecture, in a second terminal on your laptop.

Clean up: `tmux attach -t notebooks`, press `Ctrl+C` twice to stop Jupyter, type `exit`, and delete the practice folder:

```shell
rm -r ~/ds217_ssh_practice
```

**Expect:** `[exited]` after `exit`, and no output from `rm`. The practice key pair is gone, and your real `~/.ssh` was never touched.
