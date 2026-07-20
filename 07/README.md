# Visualization from Question to Accessible Claim

Lecture 07 turns a prepared table into an honest, readable visual argument. The central skill is not memorizing plotting libraries. It is deciding what comparison a chart must support, constructing that chart deliberately, and checking whether another person can interpret it without being misled.

Optional extensions are collected in [BONUS.md](BONUS.md). They are not prerequisites for the required demos, assignment, or Lecture 08.

## Prerequisites

Before starting this lecture, students should be able to:

- restart and run a notebook from top to bottom using portable paths;
- inspect, select, filter, and sort a prepared DataFrame;
- state what one row represents and distinguish row labels from columns;
- use a supplied long-form table; and
- read a supplied summary without independently creating it through aggregation.

Lecture 07 does not require students to clean, join, reshape, or aggregate data. It also does not assume time-series analysis, correlation or regression interpretation, statistical inference, modeling, interactive-dashboard design, or performance engineering.

## Learning objectives

By the end of Lecture 07, students should be able to:

1. State a visualization’s question, audience, and intended claim; identify the variable roles and comparison the chart must support.
2. Select an appropriate line, bar, scatter, histogram, or box plot and construct it with the Matplotlib `Figure`/`Axes` model, complete labels, and deterministic data.
3. Critique and revise a chart for visual integrity, including scale, comparable baselines, truthful area/length encodings, context, and avoidance of unsupported claims.
4. Apply core accessibility practices: readable labels, sufficient contrast, colorblind-safe choices, redundant encoding when needed, and a concise text alternative/caption.
5. Use pandas/seaborn for a bounded exploratory view, then create and export one annotated explanatory chart tailored to an audience and claim.

## Colab-first execution and evidence

Required Lecture 07 demonstrations are Colab-first and also run in local Jupyter or the VS Code notebook interface. The provisional 2026–27 compatibility candidate is:

| Component | Candidate |
|---|---|
| Python | 3.12.13 |
| NumPy | 2.0.2 |
| pandas | 3.0.3 |
| Matplotlib | 3.10.8 |
| seaborn | 0.13.2 |

This is not the final release lock. A required notebook must select the named Python runtime, conditionally install missing or mismatched course packages before importing them, print the versions actually in use, and pass in both a fresh Colab runtime and clean local Jupyter before publication. Avoid reinstalling unrelated Colab packages.

Colab's filesystem is ephemeral. Required notebooks reacquire pinned prepared data and create output directories in code; manual upload and mounted Drive are not defaults. Changes made in a Colab notebook opened from GitHub are not automatically saved back to the repository.

Assignment notebooks remain runnable in clean local Jupyter. Colab becomes an assignment submission path only after the repository-save and Classroom 50 pilot is approved. Remove credentials, private records, and sensitive output before sharing. Stored cell output is not execution evidence: a grader runs a fresh copy, and a human reviewer inspects the newly rendered chart.

## Start with a visualization contract

A **visualization** maps data values to visible properties so that a reader can make a comparison. Before choosing an API or chart type, write four plain-language statements.

A **question** names what the chart should help the reader compare or understand. An **audience** names the people who will use the chart and the context they bring. A **claim** is the specific descriptive conclusion the finished chart is intended to support. A chart can reveal a pattern, but a visual pattern alone does not prove why that pattern occurred.

For the running example:

- **Question:** How do observed assessment scores change across five study rounds for two programs?
- **Audience:** A course coordinator deciding what result deserves follow-up.
- **Intended claim:** In this prepared descriptive dataset, the guided program shows a larger observed increase and ends with the higher score.
- **Comparison:** Compare the two score paths at the same round and compare each program's first and last rounds.

### State the unit and grain shown

The **unit displayed** is what one mark or summarized position in the chart represents. Its **grain** is the corresponding row meaning in the plotting table. State both because a chart made from participant rows answers a different question from a chart made from program summaries.

The running line-chart table has grain **one row per program and study round**. Each plotted point represents one already-prepared program-round score. Lecture 08 will teach how such a summary is produced; Lecture 07 only consumes it.

### Identify variable roles

A variable's **role** describes how it participates in the comparison:

- a **categorical variable** places observations into named groups, such as program;
- a **quantitative variable** records a numeric magnitude, such as score;
- an **ordered variable** has a meaningful sequence, such as study round; and
- an **identifier** distinguishes rows but is not automatically a meaningful visual encoding.

For the running claim, `round_number` is ordered, `score` is quantitative, and `program` is categorical.

### Separate exploratory and explanatory work

An **exploratory visualization** helps the analyst inspect patterns, distributions, or unexpected values while the question is still being refined. It may be quick, but it still needs truthful scales and labels.

An **explanatory visualization** communicates one selected finding to a named audience. It removes irrelevant alternatives, adds context and annotation, and uses a title or caption that states what the reader should notice without overstating the evidence.

Exploration can produce many views. The final explanatory chart should have one clear job.

### Think in marks and encodings

A **mark** is a visible object such as a point, line, or rectangle. An **encoding** maps a data value to a visible property such as horizontal position, vertical position, length, color, marker shape, or line style.

Position along a common scale usually supports more precise comparison than area or decorative volume. Color can distinguish categories, but color alone is fragile: some readers cannot distinguish the selected hues, and grayscale reproduction may remove the distinction. When category identity matters, pair color with a redundant encoding such as marker shape, line style, direct labeling, or position.

## Choose a chart that matches the comparison

The five required chart types have different jobs.

| Chart | Appropriate comparison | Required cautions |
|---|---|---|
| **Line chart** | Change across a genuinely ordered sequence | Do not connect categories whose order has no meaning; identify every line |
| **Bar chart** | Magnitudes across discrete categories | Bar length encodes magnitude, so comparable bars normally share a zero baseline |
| **Scatter plot** | How two quantitative variables vary together | A visible relationship is descriptive and does not establish causation |
| **Histogram** | Distribution of one quantitative variable | Results depend on bin boundaries and width; show the measurement unit |
| **Box plot** | Compact comparison of quantitative distributions | Explain its summary and inspect the underlying observations before labeling points as errors |

A **bin** is an interval that collects numeric observations for one histogram bar. Wider or shifted bins can hide or emphasize structure, so the bin specification is part of the chart's meaning.

In a box plot, the **median** is the middle ordered value. The first and third **quartiles**, Q1 and Q3, mark the lower and upper edges of the central half of the data. The **interquartile range**, or **IQR**, is `Q3 - Q1`. With the common `1.5 × IQR` rule documented in Matplotlib's [`Axes.boxplot()` reference](https://matplotlib.org/3.10.8/api/_as_gen/matplotlib.axes.Axes.boxplot.html), each **whisker** reaches the farthest observed value still inside the corresponding cutoff; observations beyond the whiskers appear as individual points. Those points are not automatically data errors, grounds for deletion, or proof of an unusual mechanism. They are observations to investigate in context.

## Protect visual integrity

A **scale** maps data values to positions, lengths, or other visual magnitudes. An **axis** displays a scale and should name the variable and unit. A **baseline** is the reference value from which a visual length or change is judged. **Context** is the information a reader needs to interpret the comparison, such as the population, period or round range, units, denominator, or whether values are raw observations or prepared summaries.

**Visual integrity** means that the chart's visible comparisons faithfully represent the documented data, scale, context, and question.

Use these checks:

- Preserve comparable scales when panels or marks are meant to be compared.
- Start ordinary magnitude bars at zero because truncating the axis changes their apparent lengths. A nonzero bar baseline requires an exceptional, explicitly justified design.
- Do not apply the zero rule mechanically to line or scatter plots. Their positions may use a narrower range when the range and context remain visible and the choice does not exaggerate a claim.
- Avoid area, volume, or pictogram encodings when the data are intended to be compared by simple length.
- Name units, population, and whether values are observations or summaries.
- Do not hide inconvenient observations merely to make the pattern cleaner.
- Phrase descriptive findings as observations, not causal, inferential, or predictive conclusions.

## Make the chart accessible

An accessible chart is designed so more readers can recover its comparison.

- Use readable type, complete labels, and adequate contrast against the background.
- Use a colorblind-safe palette, but do not treat palette choice as the whole accessibility task.
- Add redundant encoding when color distinguishes important categories. The running line chart uses both color and marker/line style.
- Prefer direct labels or a clearly associated legend over a distant decoding task.
- Do not rely on hover interaction to reveal essential values.
- Provide a concise **text alternative** or caption that states the chart type, axes, main pattern, and a relevant limitation.

Example text alternative for the final chart:

> Line chart of prepared score by study round for standard and guided programs. Both rise across five rounds; the guided series rises from 61 to 79 and finishes seven points above the standard series. These are descriptive prepared summaries and do not establish a causal program effect.

## LIVE DEMO 1: Critique and redesign

[Open the Lecture 07 demo guide](demo/DEMO_GUIDE.md).

The first required demonstration starts from a supplied misleading chart and its data. Students state the question, audience, claim, unit, and variable roles; diagnose an unjustified bar baseline, incomplete labels, color-only categories, decoration, and an unsupported causal title; then rebuild one honest, accessible chart and supply a text alternative.

## Construct charts with Matplotlib Figure and Axes

A Matplotlib **Figure** is the complete canvas that can be saved. An **Axes** is one plotting area inside that Figure, including its data region, scales, labels, title, and plotted marks. The variable name `ax` conventionally refers to one Axes; it is not an abbreviation for “axis.” An individual x-axis or y-axis is one component of an Axes.

`plt.subplots()` returns both objects. Create them explicitly, draw with `ax` methods, and save with the Figure.

The first executable cell checks the provisional stack. In Colab, the notebook's earlier setup cell must install mismatched packages before this import cell runs.

```python
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

assert platform.python_version() == "3.12.13"
assert np.__version__ == "2.0.2"
assert pd.__version__ == "3.0.3"
assert matplotlib.__version__ == "3.10.8"
assert sns.__version__ == "0.13.2"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
```

### Use deterministic prepared data

**Deterministic data** have fixed documented values, so rerunning the notebook produces the same plotting table and expected comparisons. The examples use literal prepared records rather than random generation or a network-only source.

A **long-form table** stores one observed or prepared value per row, with separate columns for identifiers, categories, and measurements. The table `progress_long` has grain one row per program and round. The table `participant_scores` has grain one row per participant. `program_summary` is an already-prepared summary supplied for a bar-chart example; this lecture does not recreate it.

```python
progress_long = pd.DataFrame(
    {
        "program": ["Standard"] * 5 + ["Guided"] * 5,
        "round_number": [1, 2, 3, 4, 5] * 2,
        "score": [62, 65, 67, 70, 72, 61, 66, 71, 75, 79],
    }
).astype({"program": "string"})

participant_scores = pd.DataFrame(
    {
        "participant_id": [
            "S01", "S02", "S03", "S04", "S05",
            "G01", "G02", "G03", "G04", "G05",
        ],
        "program": ["Standard"] * 5 + ["Guided"] * 5,
        "practice_hours": [2.0, 3.0, 3.5, 4.0, 5.0, 2.5, 3.5, 4.0, 5.0, 6.0],
        "score": [61, 65, 66, 67, 69, 64, 70, 73, 77, 82],
    }
).astype(
    {
        "participant_id": "string",
        "program": "string",
    }
)

program_summary = pd.DataFrame(
    {
        "program": ["Standard", "Guided"],
        "mean_score": [65.6, 73.2],
    }
).astype({"program": "string"})

assert progress_long.shape == (10, 3)
assert participant_scores.shape == (10, 4)
assert program_summary.shape == (2, 2)
```

### Line chart for an ordered sequence

```python
standard_rounds = progress_long.loc[
    progress_long["program"].eq("Standard")
]
guided_rounds = progress_long.loc[
    progress_long["program"].eq("Guided")
]

line_figure, line_ax = plt.subplots(figsize=(7, 4))
line_ax.plot(
    standard_rounds["round_number"],
    standard_rounds["score"],
    color=BLUE,
    marker="o",
    linestyle="-",
    label="Standard",
)
line_ax.plot(
    guided_rounds["round_number"],
    guided_rounds["score"],
    color=ORANGE,
    marker="s",
    linestyle="--",
    label="Guided",
)
line_ax.set(
    title="Prepared scores across study rounds",
    xlabel="Study round",
    ylabel="Prepared score (points)",
    xticks=[1, 2, 3, 4, 5],
)
line_ax.legend(title="Program")
line_figure.tight_layout()

assert len(line_ax.lines) == 2
assert line_ax.get_xlabel() == "Study round"
assert line_ax.get_ylabel() == "Prepared score (points)"
```

### Bar chart for category magnitudes

```python
bar_figure, bar_ax = plt.subplots(figsize=(6, 4))
bars = bar_ax.bar(
    program_summary["program"],
    program_summary["mean_score"],
    color=[BLUE, ORANGE],
)
bar_ax.set(
    title="Prepared mean score by program",
    xlabel="Program",
    ylabel="Prepared mean score (points)",
    ylim=(0, 80),
)
bar_ax.bar_label(bars, fmt="%.1f")
bar_figure.tight_layout()

assert bar_ax.get_ylim()[0] == 0
assert len(bars) == 2
```

### Scatter plot for two quantitative variables

```python
scatter_figure, scatter_ax = plt.subplots(figsize=(6, 4))
for program, color, marker in [
    ("Standard", BLUE, "o"),
    ("Guided", ORANGE, "s"),
]:
    subset = participant_scores.loc[
        participant_scores["program"].eq(program)
    ]
    scatter_ax.scatter(
        subset["practice_hours"],
        subset["score"],
        color=color,
        marker=marker,
        s=55,
        label=program,
    )

scatter_ax.set(
    title="Practice hours and observed score",
    xlabel="Practice (hours)",
    ylabel="Observed score (points)",
)
scatter_ax.legend(title="Program")
scatter_figure.tight_layout()

assert len(scatter_ax.collections) == 2
```

### Histogram for one quantitative distribution

The explicit boundaries below define five bins: `[60, 65)`, `[65, 70)`, `[70, 75)`, `[75, 80)`, and `[80, 85]`.

```python
histogram_figure, histogram_ax = plt.subplots(figsize=(6, 4))
bin_edges = [60, 65, 70, 75, 80, 85]
counts, returned_edges, patches = histogram_ax.hist(
    participant_scores["score"],
    bins=bin_edges,
    color=GREEN,
    edgecolor="white",
)
histogram_ax.set(
    title="Distribution of observed participant scores",
    xlabel="Observed score (points)",
    ylabel="Participants (count)",
)
histogram_figure.tight_layout()

assert int(counts.sum()) == len(participant_scores)
assert returned_edges.tolist() == bin_edges
assert len(patches) == 5
```

### Box plot for compact distribution comparison

```python
standard_scores = participant_scores.loc[
    participant_scores["program"].eq("Standard"),
    "score",
]
guided_scores = participant_scores.loc[
    participant_scores["program"].eq("Guided"),
    "score",
]

box_figure, box_ax = plt.subplots(figsize=(6, 4))
box_artists = box_ax.boxplot(
    [standard_scores, guided_scores],
    tick_labels=["Standard", "Guided"],
    orientation="vertical",
    whis=1.5,
    patch_artist=True,
)
for patch, color in zip(box_artists["boxes"], [BLUE, ORANGE]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

box_ax.set(
    title="Observed score distributions by program",
    xlabel="Program",
    ylabel="Observed score (points)",
)
box_figure.tight_layout()

assert len(box_artists["boxes"]) == 2
assert len(box_artists["medians"]) == 2
```

## LIVE DEMO 2: Figure and Axes fundamentals

[Open the Lecture 07 demo guide](demo/DEMO_GUIDE.md).

The second required demonstration begins with the plotting contract and fixed prepared data, creates each of the five core chart types through explicit Figure/Axes objects, labels every scale and unit, applies the bar-baseline rule with nuance, and checks observable chart properties rather than trusting stored output.

## Use pandas and seaborn for bounded exploration

pandas plotting is a concise exploratory shortcut around Matplotlib. seaborn accepts long-form DataFrames and provides focused defaults for categorical comparisons. Neither tool chooses the question, audience, claim, unit, scale, or interpretation for you.

The next two views inspect the prepared participant rows. They do not calculate correlations, fit trend lines, or estimate uncertainty.

```python
pandas_figure, pandas_ax = plt.subplots(figsize=(6, 4))
participant_scores.plot.scatter(
    x="practice_hours",
    y="score",
    color=BLUE,
    ax=pandas_ax,
)
pandas_ax.set(
    title="Exploratory view of practice and score",
    xlabel="Practice (hours)",
    ylabel="Observed score (points)",
)
pandas_figure.tight_layout()

seaborn_figure, seaborn_ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(
    data=participant_scores,
    x="practice_hours",
    y="score",
    hue="program",
    style="program",
    palette={"Standard": BLUE, "Guided": ORANGE},
    markers={"Standard": "o", "Guided": "s"},
    ax=seaborn_ax,
)
seaborn_ax.set(
    title="Exploratory relationship view by program",
    xlabel="Practice (hours)",
    ylabel="Observed score (points)",
)
seaborn_ax.legend(title="Program")
seaborn_figure.tight_layout()

assert pandas_ax.get_xlabel() == "Practice (hours)"
assert seaborn_ax.get_ylabel() == "Observed score (points)"
```

## Move from exploration to one explanatory chart

An **annotation** adds focused context to a mark, such as a label, arrow, or reference line. **Layout** is the arrangement of plot area, title, legend, labels, and explanatory text so that they do not overlap or compete. **Export** writes the complete Figure to a file with deliberate dimensions, format, and filename.

The explanatory chart below returns to the declared audience and claim. It uses color plus marker/line style, labels units, describes prepared summaries rather than claiming cause, and annotates the largest observed separation.

```python
explanatory_figure, explanatory_ax = plt.subplots(figsize=(8, 4.8))
explanatory_ax.plot(
    standard_rounds["round_number"],
    standard_rounds["score"],
    color=BLUE,
    marker="o",
    linestyle="-",
    linewidth=2,
    label="Standard",
)
explanatory_ax.plot(
    guided_rounds["round_number"],
    guided_rounds["score"],
    color=ORANGE,
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Guided",
)
explanatory_ax.annotate(
    "Largest observed separation: 7 points",
    xy=(5, 79),
    xytext=(3.15, 81.5),
    arrowprops={"arrowstyle": "->", "color": "#333333"},
)
explanatory_ax.set(
    title="Guided scores rise more in the prepared five-round summary",
    xlabel="Study round",
    ylabel="Prepared score (points)",
    xticks=[1, 2, 3, 4, 5],
    xlim=(0.8, 5.2),
    ylim=(58, 84),
)
explanatory_ax.legend(title="Program", frameon=False)
explanatory_ax.spines[["top", "right"]].set_visible(False)

explanatory_text_alternative = (
    "Line chart of prepared score by study round for standard and guided "
    "programs. Both rise across five rounds; guided rises from 61 to 79 "
    "and finishes seven points above standard. The summaries are descriptive "
    "and do not establish a causal program effect."
)

explanatory_figure.tight_layout()
output_directory = Path("output")
output_directory.mkdir(parents=True, exist_ok=True)
explanatory_path = output_directory / "lecture07_explanatory.png"
explanatory_figure.savefig(explanatory_path, dpi=150, bbox_inches="tight")

assert explanatory_ax.get_title().startswith("Guided scores rise")
assert explanatory_ax.get_xlabel() == "Study round"
assert explanatory_ax.get_ylabel() == "Prepared score (points)"
assert len(explanatory_ax.lines) == 2
assert explanatory_ax.get_legend() is not None
assert "do not establish a causal" in explanatory_text_alternative
assert explanatory_path.is_file()
assert explanatory_path.stat().st_size > 1_000
```

### Apply a human visual-QA rubric

Automated assertions can verify labels, object counts, paths, and file creation. They cannot certify that a chart is honest, legible, accessible, or useful. A human reviewer should answer every item:

- **Contract:** Are the question, audience, intended claim, unit, grain, and variable roles explicit?
- **Choice:** Does the chart type support the intended comparison without adding an unsupported one?
- **Integrity:** Are scales, baselines, encodings, units, denominators, and context truthful and comparable?
- **Claim:** Does the title/caption remain descriptive and within the evidence?
- **Accessibility:** Are labels readable, contrast sufficient, colors distinguishable, and important categories redundantly encoded?
- **Text alternative:** Does it name the chart, axes, main pattern, and a limitation without merely repeating the title?
- **Layout:** Are marks, labels, annotation, and legend unclipped and free of confusing overlap at the exported size?
- **Reproducibility:** Does restart-and-run-all recreate the chart from pinned prepared data without hidden state or an old output file?

## LIVE DEMO 3: Exploratory to explanatory

[Open the Lecture 07 demo guide](demo/DEMO_GUIDE.md).

The third required demonstration makes one bounded pandas or seaborn exploratory view from supplied long-form data, selects a descriptive finding, rebuilds it through Figure/Axes as one annotated accessible explanatory chart, exports it, writes a text alternative, and completes both executable checks and the human visual-QA rubric.

## Handoff to Lecture 08

After this lecture, students should be able to:

- state a chart's question, audience, intended claim, and displayed unit;
- use prepared long-form plotting data and explain what each mark represents;
- construct and critique one basic chart from an already-supplied summary; and
- explain that aggregation can change the unit and row count represented in a chart.

Lecture 08 introduces grouping keys, aggregation, result grain, and aggregating pivot tables. A chart may communicate that grouped result, but visualization does not become a second Lecture 08 objective.

## Core scope boundary

Required Lecture 07 work is limited to the visualization contract, the five core static chart types, visual integrity, accessibility, focused Figure/Axes construction, bounded pandas/seaborn exploration, annotation, export, and human visual QA on deterministic prepared data.

Cleaning and missing-data decisions remain in Lecture 05. Joins and structural reshape remain in Lecture 06. GroupBy and aggregating pivots belong to Lecture 08; time-series operations to Lecture 09; correlation, regression, inference, prediction, and modeling to Lecture 10. Interactive libraries, dashboards, animation, geospatial systems, and performance engineering are not core Lecture 07 requirements.
