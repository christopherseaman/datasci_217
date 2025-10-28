"""
Generate Tufte visualization examples for Lecture 07
Saves images to ../media/ directory
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set output directory
MEDIA_DIR = Path(__file__).parent.parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_data_ink_ratio():
    """Generate data-ink ratio comparison"""
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # LOW data-ink ratio (chartjunk)
    ax1.bar(categories, values, color='red', edgecolor='black', linewidth=2)
    ax1.set_title('LOW Data-Ink Ratio (Chartjunk)', fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', linestyle='-', linewidth=1.5)
    ax1.set_facecolor('#f0f0f0')
    ax1.spines['top'].set_linewidth(3)
    ax1.spines['right'].set_linewidth(3)
    ax1.spines['left'].set_linewidth(3)
    ax1.spines['bottom'].set_linewidth(3)

    # HIGH data-ink ratio (Tufte-inspired)
    ax2.barh(categories, values, color='#2E7D32')
    ax2.set_title('HIGH Data-Ink Ratio (Tufte-Inspired)', fontsize=14, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    for i, v in enumerate(values):
        ax2.text(v + 1, i, str(v), va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'tufte_data_ink_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: tufte_data_ink_ratio.png")


def generate_small_multiples():
    """Generate small multiples example"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle('Sales Trends by Region (Small Multiples)', fontsize=16, fontweight='bold')

    regions = ['North', 'South', 'East', 'West', 'Central', 'Overseas']
    np.random.seed(42)

    for idx, (ax, region) in enumerate(zip(axes.flat, regions)):
        x = np.arange(12)
        y = np.random.randint(50, 150, 12) + idx * 10

        ax.plot(x, y, linewidth=2, color='#1976D2')
        ax.fill_between(x, y, alpha=0.3, color='#1976D2')
        ax.set_title(region, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'tufte_small_multiples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: tufte_small_multiples.png")


def generate_bar_chart_comparison():
    """Generate before/after bar chart comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    categories = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    values = [42, 68, 55, 73, 61]

    # BEFORE: Chartjunk
    bars1 = ax1.bar(categories, values,
                    color=['red', 'blue', 'green', 'yellow', 'purple'],
                    edgecolor='black', linewidth=3)
    ax1.set_title('BEFORE: Chartjunk Example', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Sales (in thousands)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, which='both', linestyle='-', linewidth=2, alpha=0.7)
    ax1.set_facecolor('#e0e0e0')
    for bar in bars1:
        bar.set_hatch('//')

    # AFTER: Clean design
    bars2 = ax2.barh(range(len(categories)), values, color='#2E7D32')
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels(categories)
    ax2.set_xlabel('Sales (in thousands)', fontsize=12)
    ax2.set_title('AFTER: Tufte-Inspired Design', fontsize=14, fontweight='bold', pad=20)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    for i, v in enumerate(values):
        ax2.text(v + 1.5, i, f'{v}k', va='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'tufte_bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: tufte_bar_comparison.png")


def generate_lie_factor_example():
    """Generate lie factor (truncated axis) example"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    revenue = [98, 99, 101, 103, 105, 106]

    # BEFORE: Truncated axis (misleading)
    ax1.plot(months, revenue, marker='o', linewidth=3, markersize=10, color='red')
    ax1.set_ylim(95, 110)
    ax1.set_title('BEFORE: Truncated Axis (Misleading!)', fontsize=14, fontweight='bold', color='red')
    ax1.set_ylabel('Revenue ($M)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.text(3, 108, '🚨 Exaggerates growth!', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # AFTER: Honest scale
    ax2.plot(months, revenue, marker='o', linewidth=2, markersize=8, color='#1976D2')
    ax2.set_ylim(0, 120)
    ax2.set_title('AFTER: Honest Scale (Tufte-Approved)', fontsize=14, fontweight='bold', color='green')
    ax2.set_ylabel('Revenue ($M)', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'tufte_lie_factor.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: tufte_lie_factor.png")


def generate_color_palettes():
    """Generate color palette guide"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sequential
    sequential_colors = sns.color_palette("Blues", 5)
    axes[0, 0].bar(range(5), [1, 2, 3, 4, 5], color=sequential_colors)
    axes[0, 0].set_title('Sequential: Ordered Data (Low → High)', fontweight='bold')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)

    # Diverging
    diverging_colors = sns.color_palette("RdBu_r", 7)
    values = [-3, -2, -1, 0, 1, 2, 3]
    axes[0, 1].bar(range(7), values, color=diverging_colors)
    axes[0, 1].set_title('Diverging: Data with Midpoint (e.g., Profit/Loss)', fontweight='bold')
    axes[0, 1].set_ylabel('Change')
    axes[0, 1].axhline(y=0, color='black', linewidth=1)
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)

    # Qualitative
    qualitative_colors = sns.color_palette("Set2", 4)
    axes[1, 0].bar(range(4), [5, 7, 6, 8], color=qualitative_colors)
    axes[1, 0].set_title('Qualitative: Categorical Data (No Order)', fontweight='bold')
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_xticklabels(['Cat A', 'Cat B', 'Cat C', 'Cat D'])
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    # Colorblind-safe
    colorblind_safe = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC',
                       '#CA9161', '#949494', '#ECE133', '#56B4E9']
    axes[1, 1].bar(range(len(colorblind_safe)), [1]*len(colorblind_safe),
                   color=colorblind_safe)
    axes[1, 1].set_title('Colorblind-Safe Palette (Accessible)', fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'color_palettes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: color_palettes.png")


def generate_tufte_principles_comparison():
    """Generate comprehensive Tufte principles comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    data = [10, 20, 30, 40]

    # BAD: Multiple violations
    bars1 = ax1.bar(['A', 'B', 'C', 'D'], data,
                    color=['red', 'blue', 'green', 'yellow'],
                    edgecolor='black', linewidth=3)
    ax1.set_title('Sales by Region', fontsize=18, style='italic')
    ax1.set_ylim(5, 45)
    ax1.grid(True, which='both', linestyle='-', linewidth=2)
    ax1.set_facecolor('#f0f0f0')
    for bar in bars1:
        bar.set_hatch('///')

    ax1.text(0.5, 0.95, '❌ Chartjunk, Truncated Axis, Poor Colors',
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
             fontsize=10, fontweight='bold')

    # GOOD: Tufte-approved
    bars2 = ax2.barh(range(len(data)), data, color='#2E7D32')
    ax2.set_yticks(range(len(data)))
    ax2.set_yticklabels(['Region A', 'Region B', 'Region C', 'Region D'])
    ax2.set_xlabel('Sales (thousands)', fontsize=12)
    ax2.set_title('Sales by Region', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim(0, 45)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    for i, v in enumerate(data):
        ax2.text(v + 0.8, i, f'{v}k', va='center', fontweight='bold', fontsize=11)

    ax2.text(0.5, 0.95, '✅ High Data-Ink, Honest Scale, Clear Labels',
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
             fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'tufte_principles_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: tufte_principles_comparison.png")


if __name__ == '__main__':
    print("Generating Tufte visualization examples...")
    print(f"Output directory: {MEDIA_DIR}")

    generate_data_ink_ratio()
    generate_small_multiples()
    generate_bar_chart_comparison()
    generate_lie_factor_example()
    generate_color_palettes()
    generate_tufte_principles_comparison()

    print("\n✅ All images generated successfully!")
    print(f"Images saved to: {MEDIA_DIR}")
