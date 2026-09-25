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
    """Weekly flu visits at six clinics, one panel each, on one shared y-scale"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle('Weekly flu visits at six clinics (shared y-axis)', fontsize=16, fontweight='bold')

    clinics = ['North', 'South', 'East', 'West', 'Central', 'Harbor']
    sizes = [60, 35, 45, 25, 70, 30]      # peak weekly visits per clinic
    peak_weeks = [6, 7, 5, 8, 6, 9]
    rng = np.random.default_rng(42)
    weeks = np.arange(1, 13)

    for ax, clinic, size, peak in zip(axes.flat, clinics, sizes, peak_weeks):
        season = np.exp(-((weeks - peak) ** 2) / 8)   # one seasonal peak
        visits = np.round(10 + size * season + rng.normal(0, 3, weeks.size))
        ax.plot(weeks, visits, linewidth=2, color='#1976D2', marker='o', markersize=3)
        ax.set_title(clinic, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Week')
    for ax in axes[:, 0]:
        ax.set_ylabel('Flu visits')
    axes[0, 0].set_ylim(0, 90)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'tufte_small_multiples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: tufte_small_multiples.png")


def generate_bar_chart_comparison():
    """Before/after bar chart: decoration that encodes nothing, then direct labels"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    clinics = ['North', 'South', 'East', 'West', 'Central']
    patients = [420, 680, 550, 730, 610]

    # BEFORE: every bar its own color and the same hatch, so neither encodes anything
    bars1 = ax1.bar(clinics, patients,
                    color=['red', 'blue', 'green', 'yellow', 'purple'],
                    edgecolor='black', linewidth=3)
    ax1.set_title('BEFORE: Chartjunk', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Patients seen in March', fontsize=12)
    ax1.set_ylim(0, 1000)
    ax1.grid(True, which='both', linestyle='-', linewidth=2, alpha=0.7)
    ax1.set_facecolor('#e0e0e0')
    for bar in bars1:
        bar.set_hatch('//')

    # AFTER: one color, sorted, values written on the bars
    order = sorted(range(len(clinics)), key=lambda i: patients[i])
    sorted_clinics = [clinics[i] for i in order]
    sorted_patients = [patients[i] for i in order]
    ax2.barh(sorted_clinics, sorted_patients, color='#2E7D32')
    ax2.set_xlabel('Patients seen in March', fontsize=12)
    ax2.set_title('AFTER: High data-ink ratio', fontsize=14, fontweight='bold', pad=20)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    for i, v in enumerate(sorted_patients):
        ax2.text(v + 10, i, str(v), va='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(MEDIA_DIR / 'tufte_bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: tufte_bar_comparison.png")


def generate_lie_factor_example():
    """Hand-hygiene bars on a 95% baseline (lie factor about 96), then on zero"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    months = ['March', 'April']
    compliance = [96, 97]
    colors = ['#90A4AE', '#1976D2']

    # BEFORE: the y-axis starts at 95%, so April's bar is twice as tall
    ax1.bar(months, compliance, color=colors, width=0.6)
    ax1.set_ylim(95, 97.5)
    ax1.set_title('BEFORE: y-axis starts at 95%', fontsize=14, fontweight='bold', color='#C62828')
    ax1.set_ylabel('Hand-hygiene compliance (%)', fontsize=12)
    ax1.text(0.5, 97.25, 'April looks twice as high', ha='center', fontsize=12)

    # AFTER: bars start at zero, so length matches the data
    bars = ax2.bar(months, compliance, color=colors, width=0.6)
    ax2.bar_label(bars, fmt='%d%%', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.set_title('AFTER: y-axis starts at 0%', fontsize=14, fontweight='bold', color='#2E7D32')
    ax2.set_ylabel('Hand-hygiene compliance (%)', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

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
