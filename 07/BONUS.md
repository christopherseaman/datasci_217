# Bonus: Bounded Static Visualization Extensions

This optional material extends Lecture 07 without creating another required visualization curriculum. It assumes the core question/audience/claim, integrity, accessibility, Figure/Axes, and visual-QA workflow.

Return to [README.md](README.md) for the required five chart types and three demonstrations. Nothing in this bonus is required by the Lecture 07 assignment or by Lecture 08.

## Extend a critique without changing the evidence

A deeper critique can examine design choices that the core rubric only touches briefly:

- Do small panels use comparable scales and category order?
- Does an aspect ratio make a change appear steeper or flatter than the documented values justify?
- Does a nonzero or logarithmic scale have a visible rationale and readable tick labels?
- Are dual axes creating an accidental comparison between unrelated scales?
- Do annotations clarify selected evidence rather than cover contrary observations?
- Does the caption distinguish raw observations from supplied summaries?

The goal remains revision, not decoration. A redesign should make the intended comparison easier while leaving the underlying values and evidentiary limits unchanged.

## Use a small-multiple layout deliberately

A **small multiple** repeats the same visual form for related subsets. Shared limits and parallel labels support comparison; inconsistent scales can defeat the purpose.

The bonus examples use the same provisional stack as the core and a headless backend so they can be checked in a clean environment.

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
```

```python
bonus_scores = pd.DataFrame(
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

assert bonus_scores.shape == (10, 4)
```

```python
small_figure, small_axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(9, 4),
    sharex=True,
    sharey=True,
)

for ax, program, color, marker in zip(
    small_axes,
    ["Standard", "Guided"],
    [BLUE, ORANGE],
    ["o", "s"],
):
    subset = bonus_scores.loc[bonus_scores["program"].eq(program)]
    ax.scatter(
        subset["practice_hours"],
        subset["score"],
        color=color,
        marker=marker,
        s=55,
    )
    ax.set(
        title=program,
        xlabel="Practice (hours)",
        xlim=(1.5, 6.5),
        ylim=(58, 84),
    )

small_axes[0].set_ylabel("Observed score (points)")
small_figure.suptitle("Same scales support comparison across program panels")
small_figure.tight_layout()

assert small_axes[0].get_xlim() == small_axes[1].get_xlim()
assert small_axes[0].get_ylim() == small_axes[1].get_ylim()
```

Small multiples are useful when direct overlap would be crowded. They are not automatically better than one focused Axes.

## Add one bounded seaborn view

`stripplot()` can reveal individual observations that a compact distribution summary may hide. This example disables jitter so the teaching result is deterministic and retains direct access to every supplied row.

```python
strip_figure, strip_ax = plt.subplots(figsize=(6, 4))
sns.stripplot(
    data=bonus_scores,
    x="program",
    y="score",
    hue="program",
    palette={"Standard": BLUE, "Guided": ORANGE},
    jitter=False,
    size=7,
    legend=False,
    ax=strip_ax,
)
strip_ax.set(
    title="Individual observed scores by program",
    xlabel="Program",
    ylabel="Observed score (points)",
)
strip_figure.tight_layout()

assert strip_ax.get_ylabel() == "Observed score (points)"
assert len(strip_ax.collections) == 2
```

This is an optional raw-observation view, not a regression, density estimate, significance display, or causal comparison.

## Refine static design and export

Useful bounded refinements include direct labels, restrained reference lines with documented meanings, shared panel scales, deliberate whitespace, and vector export for static line art. Each addition must help the stated comparison.

PNG is a raster format suited to a fixed screen image. SVG is a vector format suited to scalable static marks and text. PDF can preserve vector content inside a document workflow. Always inspect the actual exported artifact because format support does not guarantee unclipped labels or readable size.

```python
bonus_output = Path("output")
bonus_output.mkdir(parents=True, exist_ok=True)
small_png = bonus_output / "lecture07_small_multiples.png"
small_svg = bonus_output / "lecture07_small_multiples.svg"

small_figure.savefig(small_png, dpi=150, bbox_inches="tight")
small_figure.savefig(small_svg, bbox_inches="tight")

assert small_png.is_file() and small_png.stat().st_size > 1_000
assert small_svg.is_file() and small_svg.stat().st_size > 1_000
assert small_svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")
```

## Orient to Altair and Plotly without adding a requirement

Altair and Plotly can be useful when a later project genuinely needs declarative specifications, browser rendering, tooltips, filtering, or another bounded interaction. They are optional orientations here, not parallel required tutorials.

Before choosing either tool, answer:

- What reader task cannot the static explanatory chart support?
- Is the essential comparison still visible without hover?
- Can the result be exported and reviewed in the submission environment?
- Are the package version, renderer, and data source reproducible in fresh Colab and local Jupyter?
- Does interaction add evidence, or merely add controls?

No Altair or Plotly code is required in Lecture 07. Production dashboards require design, accessibility, deployment, and testing work beyond this course lecture.

## Bonus visual-QA additions

In addition to the core rubric, review:

- whether small panels use identical scales when visual comparison requires them;
- whether a logarithmic scale is necessary, labeled, and understandable to the audience;
- whether annotations remain legible in PNG, SVG, and any document export;
- whether an interactive orientation has a meaningful noninteractive fallback; and
- whether optional styling changes preserve the same values, units, and evidentiary limits.

## Bonus scope boundary

This bonus is limited to extended critique, bounded static design, one raw-observation seaborn view, static export, and orientation to Altair or Plotly.

It does not teach or assess correlation, regression, confidence intervals, statistical significance, density estimation, time-series operations, modeling, geospatial systems, animation, real-time displays, dashboards, large-data sampling, performance optimization, or deployment. Those topics require their own questions, prerequisites, validation, and later canonical homes.
