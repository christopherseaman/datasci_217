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
    # Execute only these two bounded examples, not the lecture as a notebook.
    lecture = (ROOT / "07/README.md").read_text()
    for heading, variable, filename in (
        ("Code Snippet: Density Comparisons", "fig", "distribution_reference.png"),
        ("Code Snippet: Encode the study table", "scatter", "altair_study_reference.png"),
    ):
        import pandas as pd
        import seaborn as sns

        source = re.search(r"```python\n(.*?)\n```", lecture.split("### " + heading, 1)[1], re.S)[1]
        namespace = {"np": np, "pd": pd, "sns": sns, "plt": plt}
        exec(source.replace("plt.show()", ""), namespace)
        output = ROOT / "07/media" / filename
        if variable == "fig":
            namespace[variable].savefig(output, dpi=150, bbox_inches="tight")
            plt.close(namespace[variable])
        else:
            namespace[variable].save(str(output), scale_factor=2)
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
