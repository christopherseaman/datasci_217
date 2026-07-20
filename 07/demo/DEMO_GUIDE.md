# Lecture 07 demo guide

These three demonstrations move from a communication contract to an honest accessible chart: state the question/audience/claim and displayed unit, critique defects, construct focused static charts, then turn one bounded exploration into an explanatory export. Colab is the default launch experience; the same notebooks run top-to-bottom in local Jupyter.

## Launch the demos

The development badges point to the `eleventy` branch. Work opened from GitHub in Colab is not automatically saved back to GitHub.

| Demo | Colab | Local notebook | Purpose |
|---|---|---|---|
| 1. Critique and redesign | [![Open Demo 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/07/demo/demo1_critique_redesign.ipynb) | `demo1_critique_redesign.ipynb` | Diagnose scale, label, encoding, clutter, and claim defects, then repair one chart |
| 2. Figure/Axes fundamentals | [![Open Demo 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/07/demo/demo2_figure_axes.ipynb) | `demo2_figure_axes.ipynb` | Construct and check the five core static chart types with prepared data |
| 3. Exploratory to explanatory | [![Open Demo 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/07/demo/demo3_explore_explain.ipynb) | `demo3_explore_explain.ipynb` | Use one bounded seaborn view, then build an annotated accessible Matplotlib explanation |

Before publication, replace `eleventy` in all three badge targets with one immutable release tag. Open and fresh-run every resulting URL before calling the demos certified.

## Environment candidate

The compatibility candidate is Python 3.12.13, NumPy 2.0.2, pandas 3.0.3, Matplotlib 3.10.8, and seaborn 0.13.2. It is not the final course lock until local-Jupyter and fresh-Colab certification are complete. Do not install pandas 3.0.4.

Each notebook begins with one supplied setup cell. It conditionally installs only missing or mismatched course packages before the first course-package import, then prints and asserts the versions actually in use. It does not reinstall the complete Colab package collection.

For local use, start from this directory, create a Python 3.12.13 environment, and install the four deliberate direct dependencies from `requirements.txt`. Open the notebook with the course Jupyter or VS Code host and select that environment as the Python 3 kernel. Jupyter hosting and kernel-support packages are platform tooling rather than lecture imports; record their versions during certification.

The portable kernelspec is named `Python 3`, not after a local virtual environment. The notebook is the sole executable teaching source; there are no paired same-stem Markdown copies.

## Pinned synthetic fixtures and paths

All fixtures are course-authored, prepared, complete, and non-identifying:

| Fixture | SHA-256 | Grain or role |
|---|---|---|
| `data/followup_summary.csv` | `928e929c0779800eb9f5b4cfbfadcafe0a0fc160d315265ce36182047c32c6f9` | one row per program and observation period; supplied descriptive percentages |
| `data/program_progress.csv` | `c48d53634f711d4f60b32f230633a47c77e56d8b1eac5f8c84fbad3858f85b36` | one row per program and study round; supplied descriptive scores |
| `data/participant_scores.csv` | `8eecd1393f3dbd4599269ba41724b28325ea2035f926ffeca674c2150abfc165` | one row per participant; prepared practice and score values |

Inside the repository, notebooks search upward for `07/demo/` and the committed files. When only a notebook is present, they reconstruct the exact supplied bytes under a runtime-local `data/` directory. Both branches verify checksums before pandas reads the files. A corrupted committed file stops execution; it is never silently replaced. No manual upload, Drive mount, credential, random generation, mutable date, or network data fetch is required.

Repository executions write generated artifacts under ignored `07/demo/output/`. A standalone notebook writes under `output/` relative to its launch directory. Each notebook creates the directory and removes only its own stale outputs before rebuilding them.

## Demo 1: critique and redesign

Start with the written question, clinic-operations audience, descriptive claim, program-period summary grain, and categorical/quantitative roles. Display the intentionally flawed grouped bar chart and ask students to find all five planted defects before revealing the repair:

1. unsupported causal title;
2. truncated magnitude baseline;
3. missing percentage unit;
4. program identity encoded by color alone; and
5. distracting chart furniture.

The repaired grouped bars start at zero, label the percentage and period, use a descriptive prepared-summary title, pair color with hatch, reduce decoration, and provide a text alternative. The visible result should show Standard changing from 68% to 69% and Reminder from 67% to 74% without claiming the reminder caused the difference.

Expected output: `output/followup_redesign.png`.

Pause for human visual QA: can the audience recover the correct comparison, distinguish programs without color alone, read the labels, and understand the causal limitation?

## Demo 2: Figure/Axes fundamentals

Define Figure and Axes before using `fig` and `ax`. For every chart, state the grain, variable roles, and intended comparison before the API call.

- Line: two prepared program score paths across five genuinely ordered rounds, with color plus marker/line style.
- Bar: two supplied literal prepared means, `65.6` and `73.2`, with a zero baseline. Do not compute them through GroupBy.
- Scatter: ten participant practice/score points, with color plus marker and no correlation or causal claim.
- Histogram: ten scores with exact edges `[60, 65, 70, 75, 80, 85]` and participant-count units.
- Box plot: two program distributions with median/quartile/IQR/whisker meaning and no automatic outlier deletion claim.

Expected output: `output/core_line_chart.png`. The remaining four figures are live teaching state rather than a graded-looking artifact set.

Pause for human visual QA: does each chart type support the stated comparison, use complete labels and units, apply the bar-baseline rule correctly, and remain within descriptive evidence?

## Demo 3: exploratory to explanatory

Begin with one seaborn scatter from the supplied participant rows. Program uses both hue and marker style. Record only a descriptive observation; do not calculate correlation, fit a trend, estimate uncertainty, or claim cause.

Then restate the coordinator audience and prepared program-round claim. Build two score paths through explicit Figure/Axes objects, annotate the exact seven-point round-5 separation, preserve redundant encodings, and export a descriptive chart plus the exact plotted table and a useful text alternative.

Expected outputs:

- `output/program_progress_explanatory.png`;
- `output/explanatory_supporting_data.csv`; and
- `output/explanatory_text_alternative.txt`.

Pause for the full human rubric: question/audience/claim fit, displayed unit, chart choice, scale and context, accessibility, annotation usefulness, layout, and whether the text alternative names the chart, axes, pattern, and limitation.

## Likely failure modes

- **Version assertion fails:** select Python 3.12.13 and rerun the setup cell; do not work around the assertion by editing expected versions.
- **Fixture checksum fails:** restore the supplied CSV. Do not update the checksum to bless a changed file.
- **A chart is blank or stale:** restart and run all; the notebook deletes its own prior output before rendering.
- **Labels or annotation are clipped:** inspect the newly rendered export at its saved size; adjust layout rather than trusting the inline display.
- **The two lines or programs are hard to distinguish:** check marker/line style or hatch in addition to color.
- **A claim sounds causal or inferential:** return to the written data grain and replace the claim with a bounded descriptive statement.
- **A standalone notebook cannot find data:** keep the notebook writable in its launch directory so it can create the exact embedded fixture bytes under `data/`.

## Destructive and repeat-run rehearsal

Use disposable copies for destructive checks.

- Corrupt each fixture separately and confirm that every consuming notebook stops at checksum verification.
- Remove each fixture in a disposable repository copy and confirm that the embedded fallback recreates identical bytes.
- Launch from the repository root, `07/demo/`, nested under the demo directory, and outside the repository.
- Delete, stale, or corrupt generated outputs; restart/run-all and confirm complete deterministic replacement.
- Repeat clean runs. Supporting text/CSV bytes must be identical; PNG dimensions/content properties must remain stable even if platform font rendering changes exact PNG bytes.

## Scope and privacy policy

The required demos do not clean, join, reshape, group, aggregate, pivot, resample, roll, calculate correlation, regress, model, use network datasets, or introduce interactive plotting libraries. Advanced static layouts and optional library orientation remain in `../BONUS.md`; time-series work belongs to Lecture 09 and modeling to Lecture 10.

- The fixtures contain no real person or protected information.
- Never put credentials, tokens, private records, or identifying data in notebook source or output.
- Stored notebook output is never execution proof. Canonical notebooks have cleared outputs and null execution counts.
- Executed certification copies and generated `output/` files are disposable.
- GitHub source opened in Colab is not automatically updated by edits made in the Colab tab.
- Assignment Colab submission remains conditional on the repository-save/Classroom50 pilot.

## Certification record

Do not mark a row as passing without independent evidence from that environment.

| Notebook | Paired Markdown | Local candidate | Fresh Colab | Badge release ref |
|---|---|---|---|---|
| `demo1_critique_redesign.ipynb` | none — canonical notebook policy | pass — independent, 2026-07-18 | pending | development: `eleventy` |
| `demo2_figure_axes.ipynb` | none — canonical notebook policy | pass — independent, 2026-07-18 | pending | development: `eleventy` |
| `demo3_explore_explain.ipynb` | none — canonical notebook policy | pass — independent, 2026-07-18 | pending | development: `eleventy` |

For each certification run, record the notebook path, environment and plotting backend, Python/package versions, launch working directory, fixture paths/checksums, generated files, executable result, human visual result, tester, date, and immutable release ref. Do not treat this guide, authored assertions, or committed notebook output as independent certification.

Independent local evidence on 2026-07-18 used the `Agg` backend with Python 3.12.13, NumPy 2.0.2, pandas 3.0.3, Matplotlib 3.10.8, seaborn 0.13.2, nbclient 0.10.2, and ipykernel 6.29.5. A reviewer who did not author the notebooks fresh-executed all three from the repository root, `07/demo/`, a nested directory under `07/demo/`, and standalone disposable directories. The committed fixture paths and checksums passed; standalone copies reconstructed identical bytes. All five expected generated files passed executable, destructive-output, deterministic-repeat, and independent human visual checks. The local release ref was the development branch `eleventy`; fresh Colab execution and immutable release-tag badges remain pending publication gates.
