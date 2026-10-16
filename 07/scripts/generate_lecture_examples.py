"""Render Lecture 07's example figures into ../media/.

The multi-panel reference figures use the lecture's small example tables;
they combine methods that the lecture now teaches in separate short snippets.
chart_selection.png is drawn here. distribution_reference.png and
altair_study_reference.png come from scripts/build_reference_visuals.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

LECTURE_DIR = Path(__file__).resolve().parents[1]
MEDIA_DIR = LECTURE_DIR / "media"

def render_reference_figures():
    """Build the reference panels from the lecture's stated example inputs."""
    weeks = [1, 2, 3, 4]
    north = [42, 45, 51, 48]
    south = [30, 33, 31, 36]
    pressures = [118, 124, 126, 129, 131, 133, 134, 136, 139, 142, 147, 155]
    weekly = pd.DataFrame({'North': north, 'South': south}, index=pd.Index(weeks, name='week'))

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(MEDIA_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('Generated:', filename)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(weeks, north)
    axes[0, 0].set(title='Weekly flu visits', xlabel='Week', ylabel='Flu visits')
    axes[0, 1].hist(pressures, bins=4)
    assert [bar.get_height() for bar in axes[0, 1].patches] == [3, 5, 2, 2]
    axes[0, 1].set(title='Blood-pressure distribution', xlabel='Systolic BP (mmHg)', ylabel='Readings')
    axes[1, 0].scatter([34, 45, 52, 61, 68, 74], [116, 123, 128, 134, 139, 146])
    axes[1, 0].set(title='Blood pressure by age', xlabel='Age (years)', ylabel='Systolic BP (mmHg)')
    axes[1, 1].bar(['North', 'South', 'East'], [42, 30, 25])
    axes[1, 1].set(title='Flu visits by clinic', xlabel='Clinic', ylabel='Flu visits')
    save(fig, 'matplotlib_subplots.png')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(weeks, north, color='blue', label='North')
    ax.plot(weeks, south, color='red', linestyle='--', label='South')
    ax.set(title='Weekly flu visits by clinic', xlabel='Week', ylabel='Flu clinic visits')
    ax.set_xticks(weeks)
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, 'matplotlib_customization.png')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for offset, (style, color, label) in enumerate(zip(
            ['o-', 's--', '^-.', '*:'], ['blue', 'red', 'green', 'purple'],
            ['circles', 'squares', 'triangles', 'stars'])):
        ax.plot(weeks, np.array(weeks) + offset, style, color=color, label=label)
    ax.set(title='Different line styles and markers', xlabel='Week', ylabel='Example value (arbitrary units)')
    ax.legend()
    save(fig, 'matplotlib_styles.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    weekly.plot(ax=axes[0, 0], marker='o', ylabel='Flu visits', title='Line: visits each week')
    weekly.plot.bar(ax=axes[0, 1], ylabel='Flu visits', title='Bar: visits each week')
    weekly.plot.scatter(x='North', y='South', ax=axes[1, 0], title='Scatter: one point per week')
    axes[1, 0].set(xlabel='North flu visits', ylabel='South flu visits')
    weekly.plot.box(ax=axes[1, 1], title='Box: spread of weekly visits')
    axes[1, 1].set(xlabel='Clinic', ylabel='Flu visits')
    save(fig, 'pandas_plotting.png')

    health = pd.read_parquet(LECTURE_DIR / 'demo/healthexp.parquet')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.scatterplot(data=health, x='Spending_USD', y='Life_Expectancy', hue='Country', style='Country', ax=axes[0])
    axes[0].set(xlabel='Health spending per person (USD)', ylabel='Life expectancy (years)')
    axes[0].legend(title='Country', loc='upper left', bbox_to_anchor=(1, 1))
    sns.lineplot(data=health, x='Year', y='Life_Expectancy', hue='Country', style='Country', ax=axes[1])
    axes[1].set(ylabel='Life expectancy (years)')
    sns.boxplot(data=health, x='Life_Expectancy', y='Country', ax=axes[2])
    axes[2].set(xlabel='Life expectancy (years)', ylabel='')
    save(fig, 'seaborn_statistical.png')


def draw_chart_selection():
    """The chart-selection guide: six chart types on reproducible example data."""
    with plt.rc_context():
        rng = np.random.default_rng(42)
        sns.set_palette("husl")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        x = np.arange(12)
        y = rng.integers(10, 50, 12) + np.sin(x) * 5
        axes[0, 0].plot(x, y, marker='o', linewidth=2, color='#1f77b4')
        axes[0, 0].fill_between(x, y, alpha=0.3)
        axes[0, 0].set_title('Line Chart: Time Series', fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].bar(['A', 'B', 'C', 'D'], [23, 45, 56, 38], color='#ff7f0e')
        axes[0, 1].set_title('Bar Chart: Categories', fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        x_scatter = rng.standard_normal(100)
        y_scatter = 2 * x_scatter + rng.standard_normal(100) * 0.5
        axes[0, 2].scatter(x_scatter, y_scatter, alpha=0.6, color='#2ca02c')
        axes[0, 2].set_title('Scatter: Relationships', fontweight='bold')
        axes[0, 2].grid(alpha=0.3)

        axes[1, 0].hist(rng.normal(50, 15, 1000), bins=30, color='#d62728', alpha=0.7)
        axes[1, 0].set_title('Histogram: Distribution', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        data_box = [rng.normal(i * 10, 5, 50) for i in range(3)]
        axes[1, 1].boxplot(data_box, tick_labels=['Group A', 'Group B', 'Group C'])
        axes[1, 1].set_title('Box Plot: Distribution + Outliers', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        im = axes[1, 2].imshow(rng.standard_normal((10, 10)), cmap='coolwarm', aspect='auto')
        axes[1, 2].set_title('Heatmap: 2D Patterns', fontweight='bold')
        plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()
        fig.savefig(MEDIA_DIR / 'chart_selection.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    print("Generated: chart_selection.png")


if __name__ == '__main__':
    render_reference_figures()
    draw_chart_selection()
