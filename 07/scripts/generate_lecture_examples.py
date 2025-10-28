"""
Generate visual outputs for all code examples in Lecture 07
Saves images to ../media/ directory
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set output directory
MEDIA_DIR = Path(__file__).parent.parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)

# Set default style
plt.style.use('default')
sns.set_palette("husl")


def example_1_subplots():
    """Matplotlib: Figures and Subplots example"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Line plot
    axes[0, 0].plot([1, 2, 3, 4], [1, 4, 2, 3], linewidth=2, color='#1f77b4')
    axes[0, 0].set_title('Line Plot', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # Histogram
    axes[0, 1].hist(np.random.normal(0, 1, 1000), bins=30, color='#ff7f0e', alpha=0.7)
    axes[0, 1].set_title('Histogram', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Scatter plot
    axes[1, 0].scatter(np.random.randn(100), np.random.randn(100), alpha=0.6, color='#2ca02c')
    axes[1, 0].set_title('Scatter Plot', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Bar chart
    axes[1, 1].bar(['A', 'B', 'C'], [3, 7, 2], color='#d62728')
    axes[1, 1].set_title('Bar Chart', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'matplotlib_subplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: matplotlib_subplots.png")


def example_2_customization():
    """Matplotlib: Customizing Plots example"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Plot with customization
    ax.plot(x, y1, label='sin(x)', color='#1f77b4', linewidth=2.5)
    ax.plot(x, y2, label='cos(x)', color='#ff7f0e', linewidth=2.5, linestyle='--')

    # Customize appearance
    ax.set_title('Trigonometric Functions', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X values', fontsize=12)
    ax.set_ylabel('Y values', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'matplotlib_customization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: matplotlib_customization.png")


def example_3_colors_markers():
    """Matplotlib: Colors, Markers, and Line Styles"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 10, 20)

    # Different styles
    ax.plot(x, x, 'r-', label='Solid red', linewidth=2)
    ax.plot(x, x + 1, 'b--', label='Dashed blue', linewidth=2)
    ax.plot(x, x + 2, 'g-.', label='Dash-dot green', linewidth=2)
    ax.plot(x, x + 3, 'mo', label='Circles magenta', markersize=8)
    ax.plot(x, x + 4, 'ks', label='Squares black', markersize=8)
    ax.plot(x, x + 5, 'c^', label='Triangles cyan', markersize=8)

    ax.set_title('Colors, Markers, and Line Styles', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X values', fontsize=12)
    ax.set_ylabel('Y values', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'matplotlib_styles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: matplotlib_styles.png")


def example_4_seaborn_statistical():
    """Seaborn: Statistical Visualizations"""
    np.random.seed(42)

    # Generate sample data
    tips = pd.DataFrame({
        'total_bill': np.random.gamma(2, 10, 200),
        'tip': np.random.gamma(2, 2, 200),
        'day': np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], 200)
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot with regression
    sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0, 0],
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[0, 0].set_title('Regression Plot', fontweight='bold', fontsize=12)

    # Box plot by category
    sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Box Plot by Day', fontweight='bold', fontsize=12)

    # Violin plot
    sns.violinplot(data=tips, x='day', y='tip', ax=axes[1, 0], palette='muted')
    axes[1, 0].set_title('Violin Plot', fontweight='bold', fontsize=12)

    # Distribution plot
    sns.histplot(data=tips, x='total_bill', kde=True, ax=axes[1, 1], color='#1f77b4')
    axes[1, 1].set_title('Distribution with KDE', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'seaborn_statistical.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: seaborn_statistical.png")


def example_5_chart_types():
    """Chart Selection Guide Visual"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Line chart - time series
    x = np.arange(12)
    y = np.random.randint(10, 50, 12) + np.sin(x) * 5
    axes[0, 0].plot(x, y, marker='o', linewidth=2, color='#1f77b4')
    axes[0, 0].fill_between(x, y, alpha=0.3)
    axes[0, 0].set_title('Line Chart: Time Series', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # Bar chart - categories
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 38]
    axes[0, 1].bar(categories, values, color='#ff7f0e')
    axes[0, 1].set_title('Bar Chart: Categories', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Scatter plot - relationships
    x_scatter = np.random.randn(100)
    y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5
    axes[0, 2].scatter(x_scatter, y_scatter, alpha=0.6, color='#2ca02c')
    axes[0, 2].set_title('Scatter: Relationships', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    # Histogram - distribution
    data_hist = np.random.normal(50, 15, 1000)
    axes[1, 0].hist(data_hist, bins=30, color='#d62728', alpha=0.7)
    axes[1, 0].set_title('Histogram: Distribution', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Box plot - distribution with outliers
    data_box = [np.random.normal(i*10, 5, 50) for i in range(3)]
    axes[1, 1].boxplot(data_box, labels=['Group A', 'Group B', 'Group C'])
    axes[1, 1].set_title('Box Plot: Distribution + Outliers', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    # Heatmap - 2D patterns
    heatmap_data = np.random.randn(10, 10)
    im = axes[1, 2].imshow(heatmap_data, cmap='coolwarm', aspect='auto')
    axes[1, 2].set_title('Heatmap: 2D Patterns', fontweight='bold')
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'chart_selection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: chart_selection.png")


def example_6_pandas_plotting():
    """Pandas built-in plotting"""
    np.random.seed(42)

    # Create sample DataFrame with positive values for area plot
    dates = pd.date_range('2024-01-01', periods=100)
    df = pd.DataFrame({
        'A': np.abs(np.cumsum(np.random.randn(100))) + 10,
        'B': np.abs(np.cumsum(np.random.randn(100))) + 15,
        'C': np.abs(np.cumsum(np.random.randn(100))) + 12
    }, index=dates)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Line plot
    df.plot(ax=axes[0, 0], linewidth=2)
    axes[0, 0].set_title('Pandas Line Plot', fontweight='bold', fontsize=12)
    axes[0, 0].grid(alpha=0.3)

    # Area plot (stacked requires all positive or all negative)
    df.plot.area(ax=axes[0, 1], alpha=0.5, stacked=True)
    axes[0, 1].set_title('Pandas Area Plot', fontweight='bold', fontsize=12)
    axes[0, 1].grid(alpha=0.3)

    # Bar plot
    df.iloc[::10].plot.bar(ax=axes[1, 0])
    axes[1, 0].set_title('Pandas Bar Plot', fontweight='bold', fontsize=12)
    axes[1, 0].grid(alpha=0.3)

    # Histogram
    df.plot.hist(ax=axes[1, 1], bins=20, alpha=0.7)
    axes[1, 1].set_title('Pandas Histogram', fontweight='bold', fontsize=12)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'pandas_plotting.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: pandas_plotting.png")


if __name__ == '__main__':
    print("Generating lecture example visualizations...")
    print(f"Output directory: {MEDIA_DIR}\n")

    example_1_subplots()
    example_2_customization()
    example_3_colors_markers()
    example_4_seaborn_statistical()
    example_5_chart_types()
    example_6_pandas_plotting()

    print("\n✅ All example images generated successfully!")
    print(f"Images saved to: {MEDIA_DIR}")
