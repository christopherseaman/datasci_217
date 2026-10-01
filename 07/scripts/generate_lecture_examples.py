"""Render Lecture 07's example figures into ../media/.

Five images are the output of the README snippet each one illustrates, so they
are drawn by running those snippets, never a copy of them that could drift.
chart_selection.png is drawn here. distribution_reference.png and
altair_study_reference.png come from scripts/build_reference_visuals.py.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

LECTURE_DIR = Path(__file__).resolve().parents[1]
MEDIA_DIR = LECTURE_DIR / "media"

# Snippet heading -> the image it draws, in lecture order. None runs a snippet only for
# the names later snippets use: weeks and north, then weekly.
SNIPPET_FIGURES = {
    "One Mark Type per Panel": "matplotlib_subplots.png",
    "Axes Customization": "matplotlib_customization.png",
    "Visual Styles": "matplotlib_styles.png",
    "The Index Becomes the x-Axis": None,
    "Several Plot Kinds in One Grid": "pandas_plotting.png",
    "Statistical Plots": "seaborn_statistical.png",
}


def render_snippets():
    """Run each snippet in one shared namespace, as a notebook would, and save its figure."""
    readme = (LECTURE_DIR / "README.md").read_text(encoding="utf-8")
    snippets = dict(re.findall(r"^### Code Snippet: (.+?)\n.*?```python\n(.*?)```", readme, re.S | re.M))
    namespace = {"np": np, "pd": pd, "plt": plt, "sns": sns}
    for heading, image in SNIPPET_FIGURES.items():
        with plt.rc_context():  # keeps a snippet's sns.set_style() out of the next figure
            exec(snippets[heading].replace("plt.show()", ""), namespace)
            if image:
                plt.gcf().savefig(MEDIA_DIR / image, dpi=150, bbox_inches="tight")
                print("Generated:", image)
        plt.close("all")


def draw_chart_selection():
    """The chart-selection guide: six chart types, each on random data, so every run redraws it."""
    with plt.rc_context():
        sns.set_palette("husl")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        x = np.arange(12)
        y = np.random.randint(10, 50, 12) + np.sin(x) * 5
        axes[0, 0].plot(x, y, marker='o', linewidth=2, color='#1f77b4')
        axes[0, 0].fill_between(x, y, alpha=0.3)
        axes[0, 0].set_title('Line Chart: Time Series', fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].bar(['A', 'B', 'C', 'D'], [23, 45, 56, 38], color='#ff7f0e')
        axes[0, 1].set_title('Bar Chart: Categories', fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        x_scatter = np.random.randn(100)
        y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5
        axes[0, 2].scatter(x_scatter, y_scatter, alpha=0.6, color='#2ca02c')
        axes[0, 2].set_title('Scatter: Relationships', fontweight='bold')
        axes[0, 2].grid(alpha=0.3)

        axes[1, 0].hist(np.random.normal(50, 15, 1000), bins=30, color='#d62728', alpha=0.7)
        axes[1, 0].set_title('Histogram: Distribution', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        data_box = [np.random.normal(i * 10, 5, 50) for i in range(3)]
        axes[1, 1].boxplot(data_box, tick_labels=['Group A', 'Group B', 'Group C'])
        axes[1, 1].set_title('Box Plot: Distribution + Outliers', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        im = axes[1, 2].imshow(np.random.randn(10, 10), cmap='coolwarm', aspect='auto')
        axes[1, 2].set_title('Heatmap: 2D Patterns', fontweight='bold')
        plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()
        fig.savefig(MEDIA_DIR / 'chart_selection.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    print("Generated: chart_selection.png")


if __name__ == '__main__':
    render_snippets()
    draw_chart_selection()
