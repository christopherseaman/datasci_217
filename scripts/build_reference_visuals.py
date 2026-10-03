# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["matplotlib==3.11.1", "numpy==2.4.6", "pandas==3.0.5", "scipy>=1.17,<2", "seaborn==0.13.2", "altair==5.5.0", "typing-extensions>=4.10", "vl-convert-python>=1.8,<2"]
# ///
"""Render worked examples; Altair 5.5's generated typing needs Python 3.13."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    import altair as alt
    import pandas as pd
    import seaborn as sns

    # Lecture 07's density figure: the glucose readings its snippets use.
    glucose = pd.Series([84, 88, 91, 93, 95, 96, 98, 101, 156, 161, 166, 172], name="fasting_glucose")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    glucose.plot.density(ax=axes[0], title="pandas density")
    sns.kdeplot(x=glucose, ax=axes[1], label="default")
    sns.kdeplot(x=glucose, bw_adjust=0.5, ax=axes[1], label="bw_adjust=0.5")
    axes[1].legend()
    sns.histplot(x=glucose, kde=True, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("Fasting glucose (mg/dL)")
    fig.tight_layout()
    output = ROOT / "07/media/distribution_reference.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Execute only this bounded Altair example, not the lecture as a notebook.
    lecture = (ROOT / "07/README.md").read_text()
    source = re.search(r"```python\n(.*?)\n```", lecture.split("### Code Snippet: Encode the study table", 1)[1], re.S)[1]
    # The lecture lists the snippet's six study rows in prose, not code.
    study = pd.DataFrame({
        "age": [38, 52, 67, 41, 55, 70],
        "systolic_bp": [118, 129, 141, 124, 136, 150],
        "clinic": ["North"] * 3 + ["South"] * 3,
    })
    namespace = {"alt": alt, "pd": pd, "study": study}
    exec(source, namespace)
    output = ROOT / "07/media/altair_study_reference.png"
    namespace["scatter"].save(str(output), scale_factor=2)
    assert output.stat().st_size > 0

    x = np.arange(1, 7)
    y = np.array([3, 6, 5, 9, 8, 11])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, label="Observed", color="#0072B2", zorder=3)
    ax.plot(x, fitted, label=f"OLS fit: y = {intercept:.2f} + {slope:.2f}x", color="#D55E00")
    ax.vlines(x, fitted, y, colors="0.4", linestyles="dashed", label="Vertical residuals")
    ax.set(xlabel="Feature x", ylabel="Outcome y", title="Least squares minimizes squared vertical residuals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "10/media/ols_residuals.png", dpi=150)
    plt.close(fig)
    assert np.isclose((y - fitted).sum(), 0)


if __name__ == "__main__":
    main()
