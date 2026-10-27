"""Render Lecture 07's example figures into ../media/.

The multi-panel reference figures use the lecture's small example tables;
they combine methods that the lecture now teaches in separate short snippets.
chart_selection.png and the concept figures are drawn here. distribution_reference.png and
altair_study_reference.png, and altair_three_steps.png come from scripts/build_reference_visuals.py.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

LECTURE_DIR = Path(__file__).resolve().parents[1]
MEDIA_DIR = LECTURE_DIR / "media"
sys.path.insert(0, str(LECTURE_DIR / "demo"))
from generate_data import build  # noqa: E402

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
    axes[0, 0].set_xticks(weeks)
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
    ax.set_xticks(weeks)
    ax.legend()
    save(fig, 'matplotlib_styles.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    weekly.plot(ax=axes[0, 0], marker='o', ylabel='Flu visits', title='Line: visits each week')
    axes[0, 0].set_xticks(weeks)
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


OKABE_ITO = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'vermillion': '#D55E00'}
CLEAN = {'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.grid': False,
         'axes.spines.top': False, 'axes.spines.right': False}


def save_clean(fig, filename):
    fig.savefig(MEDIA_DIR / filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Generated:', filename)


def draw_chart_selection():
    """The chart-selection guide: six chart jobs, each on labeled clinic data."""
    blue, orange, green, vermillion = OKABE_ITO.values()
    rng = np.random.default_rng(42)
    with plt.rc_context(CLEAN):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        weeks = np.arange(1, 13)
        flu = [18, 22, 27, 35, 44, 51, 48, 40, 33, 27, 22, 19]
        axes[0, 0].plot(weeks, flu, marker='o', linewidth=2, color=blue)
        axes[0, 0].set_xticks(weeks)
        axes[0, 0].set(xlabel='Week', ylabel='Flu visits (North clinic)')
        axes[0, 0].set_title('Line Chart: Time Series', fontweight='bold')

        axes[0, 1].bar(['North', 'South', 'East', 'West'], [42, 30, 25, 38], color=orange)
        axes[0, 1].set(xlabel='Clinic', ylabel='Flu visits in March')
        axes[0, 1].set_title('Bar Chart: Categories', fontweight='bold')

        age = rng.uniform(25, 80, 60)
        sbp = 100 + 0.55 * age + rng.normal(0, 8, age.size)
        axes[0, 2].scatter(age, sbp, alpha=0.7, color=green)
        axes[0, 2].set(xlabel='Age (years)', ylabel='Systolic BP (mmHg)')
        axes[0, 2].set_title('Scatter: Relationships', fontweight='bold')

        axes[1, 0].hist(rng.normal(128, 15, 500), bins=25, color=vermillion, edgecolor='white')
        axes[1, 0].set(xlabel='Systolic BP (mmHg)', ylabel='Patients')
        axes[1, 0].set_title('Histogram: Distribution', fontweight='bold')

        clinics = ['North', 'South', 'East']
        by_clinic = [rng.normal(mean, 12, 50) for mean in (132, 125, 140)]
        axes[1, 1].boxplot(by_clinic, tick_labels=clinics)
        axes[1, 1].set(xlabel='Clinic', ylabel='Systolic BP (mmHg)')
        axes[1, 1].set_title('Box Plot: Distribution + Outliers', fontweight='bold')

        peaks = {'North': 6, 'South': 8, 'East': 5, 'West': 9}
        grid = np.array([10 + 40 * np.exp(-((weeks - peak) ** 2) / 6) for peak in peaks.values()]).round()
        im = axes[1, 2].imshow(grid, cmap='Blues', aspect='auto',
                               extent=(0.5, 12.5, len(peaks) - 0.5, -0.5))
        axes[1, 2].set_yticks(range(len(peaks)), list(peaks))
        axes[1, 2].set_xticks(weeks)
        axes[1, 2].set(xlabel='Week', ylabel='Clinic')
        axes[1, 2].set_title('Heatmap: 2D Patterns', fontweight='bold')
        fig.colorbar(im, ax=axes[1, 2], label='Flu visits')

        fig.tight_layout()
        save_clean(fig, 'chart_selection.png')


def render_concept_figures():
    """Concept figures whose numbers match the lecture text and snippets."""
    blue, orange, green, vermillion = OKABE_ITO.values()
    with plt.rc_context(CLEAN):
        # Exploratory vs explanatory: same weekly counts, two purposes.
        weeks = np.arange(1, 9)
        north = [12, 18, 27, 39, 46, 38, 29, 21]
        south = [8, 10, 13, 17, 22, 28, 33, 36]
        assert weeks[np.argmax(north)] == 5 and weeks[np.argmax(south)] == 8
        with plt.rc_context({'axes.spines.top': True, 'axes.spines.right': True}):
            fig = plt.figure(figsize=(13, 4.5))
            left = fig.add_subplot(1, 2, 1)
        left.plot(weeks, north, label='North')
        left.plot(weeks, south, label='South')
        left.set_title('weekly')
        left.legend()
        right = fig.add_subplot(1, 2, 2)
        right.plot(weeks, south, color='0.6', linewidth=2)
        right.plot(weeks, north, color=blue, linewidth=3)
        right.text(8.15, north[-1], 'North', color=blue, va='center', fontweight='bold')
        right.text(8.15, south[-1], 'South', color='0.4', va='center')
        right.annotate('Peak: 46 visits in week 5', xy=(5, 46), xytext=(5.6, 52),
                       arrowprops=dict(arrowstyle='->'))
        right.set_xticks(weeks)
        right.set_ylim(0, 58)
        right.set(xlabel='Week', ylabel='Flu clinic visits')
        right.set_title("North's flu visits peaked in week 5; South was still rising in week 8",
                        loc='left', fontsize=11, fontweight='bold')
        left.set(xlabel='week', ylabel='visits')
        fig.tight_layout()
        for x_pos, heading in [(0.27, 'Exploratory'), (0.77, 'Explanatory')]:
            fig.text(x_pos, 1.02, heading, ha='center', fontsize=13, fontweight='bold')
        save_clean(fig, 'exploratory_vs_explanatory.png')

        # Encoding accuracy: one set of values, four encodings.
        clinics = ['North', 'South', 'East', 'West']
        visits = [42, 35, 30, 27]
        colors = [blue, orange, green, vermillion]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].scatter(visits, clinics, color=blue, s=60, zorder=3)
        axes[0].set_xlim(0, 45)
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].set_xlabel('Visits')
        axes[1].barh(clinics, visits, color=blue)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Visits')
        axes[2].pie(visits, labels=clinics, colors=colors, startangle=90, counterclock=False,
                    wedgeprops=dict(edgecolor='white'))
        radii = np.sqrt(np.array(visits) / np.pi) * 0.25
        for i, (name, r, c) in enumerate(zip(clinics, radii, colors)):
            axes[3].add_patch(plt.Circle((i * 2.2, 0), r, color=c))
            axes[3].text(i * 2.2, -1.5, name, ha='center')
        axes[3].set_xlim(-1.5, 8.1)
        axes[3].set_ylim(-1.8, 1.3)
        axes[3].set_aspect('equal', adjustable='datalim')
        axes[3].axis('off')
        for ax, title in zip(axes, ['Position', 'Length', 'Angle', 'Area']):
            ax.set_title(title, fontweight='bold')
        fig.suptitle('North 42, South 35, East 30, West 27 visits: ranking is easy on the left, hard on the right',
                     y=1.03)
        fig.tight_layout()
        save_clean(fig, 'encoding_accuracy.png')

        # Unit and grain: readings vs clinic-month means.
        visits_df = pd.DataFrame({
            'patient_id': [101, 101, 103, 103, 105, 105, 102, 102, 104, 104, 106, 106],
            'clinic': ['North'] * 6 + ['South'] * 6,
            'visit_month': [1, 2] * 6,
            'systolic_bp': [142, 136, 150, 144, 134, 130, 128, 131, 138, 135, 122, 127],
        })
        means = visits_df.groupby(['clinic', 'visit_month'])['systolic_bp'].mean().round(1)
        print(means.to_dict())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for (clinic, color, offset) in [('North', blue, -0.08), ('South', orange, 0.08)]:
            rows = visits_df[visits_df['clinic'] == clinic]
            marker = 'o' if clinic == 'North' else 's'
            axes[0].scatter(rows['visit_month'] + offset, rows['systolic_bp'], color=color,
                            marker=marker, s=50, label=clinic)
            m = means[clinic]
            bars = axes[1].bar(np.array(m.index) + offset * 2.5, m.values, width=0.4, color=color,
                               label=clinic, hatch='//' if clinic == 'North' else '..', edgecolor='white')
            axes[1].bar_label(bars, fmt='%.1f')
        axes[0].set(title='One mark = one reading', xlabel='Visit month', ylabel='Systolic BP (mmHg)')
        axes[1].set(title='One mark = one clinic-month mean', xlabel='Visit month',
                    ylabel='Mean systolic BP (mmHg)')
        axes[1].set_ylim(0, 165)  # bar length encodes magnitude, so start at zero
        for ax in axes:
            ax.set_xticks([1, 2])
            ax.legend(title='Clinic', frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        fig.tight_layout()
        save_clean(fig, 'unit_grain.png')

        # pandas index before and after set_index.
        weekly = pd.DataFrame({'week': [1, 2, 3, 4], 'North': [42, 45, 51, 48], 'South': [30, 33, 31, 36]})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        weekly.plot(ax=axes[0], color=[green, blue, orange])
        axes[0].set_xticks([0, 1, 2, 3])
        axes[0].set(title='Before set_index: week drawn as a line', xlabel='Row number', ylabel='Flu visits')
        weekly.set_index('week').plot(ax=axes[1], marker='o', color=[blue, orange])
        axes[1].set_xticks([1, 2, 3, 4])
        axes[1].set(title="After set_index('week')", xlabel='Week', ylabel='Flu visits')
        for ax in axes:
            ax.legend(frameon=False)
        fig.tight_layout()
        save_clean(fig, 'pandas_index_before_after.png')

        # sharey: unshared panels make East look as busy as North.
        monthly = pd.DataFrame({'North': [210, 235, 260, 245, 225, 200],
                                'South': [118, 124, 130, 122, 115, 120],
                                'East': [22, 30, 38, 34, 27, 24]},
                               index=pd.Index(range(1, 7), name='month'))
        fig = plt.figure(figsize=(12, 6.5))
        for subfig, share, title in zip(fig.subfigures(1, 2, wspace=0.05), [False, True],
                                        ['sharey=False: each panel rescales', 'sharey=True: one y-scale']):
            axes = subfig.subplots(3, 1, sharex=True, sharey=share)
            monthly.plot(subplots=True, ax=axes, color=[blue, orange, green], marker='o', legend=False)
            for ax, name in zip(axes, monthly.columns):
                ax.set_ylabel('Visits')
                ax.set_title(name, loc='left', fontsize=10, fontweight='bold')
            axes[-1].set_xticks(range(1, 7))
            axes[-1].set_xlabel('Month')
            if share:
                axes[0].set_ylim(0, 280)
            subfig.suptitle(title, fontweight='bold')
        save_clean(fig, 'sharey_comparison.png')

        # seaborn grain: default CI band vs errorbar=None.
        readings = pd.DataFrame({
            'week': [1, 1, 1, 2, 2, 2],
            'patient_id': ['A', 'B', 'C', 'A', 'B', 'C'],
            'systolic_bp': [150, 138, 142, 144, 136, 139],
        })
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        sns.lineplot(data=readings, x='week', y='systolic_bp', ax=axes[0], color=blue, seed=0)
        sns.scatterplot(data=readings, x='week', y='systolic_bp', ax=axes[0], color='0.3', zorder=3)
        sns.lineplot(data=readings, x='week', y='systolic_bp', errorbar=None, ax=axes[1], color=blue, marker='o')
        drawn = axes[1].get_lines()[0].get_ydata()
        assert np.allclose(drawn, [143.333333, 139.666667])
        for week, mean in zip([1, 2], drawn):
            axes[1].annotate(f'{mean:.1f}', (week, mean), xytext=(0, 8), textcoords='offset points', ha='center')
        axes[0].set_title('Default: mean + 95% CI')
        axes[1].set_title('errorbar=None: mean only')
        for ax in axes:
            ax.set_xticks([1, 2])
            ax.set(xlabel='Week', ylabel='Systolic BP (mmHg)')
        fig.tight_layout()
        save_clean(fig, 'seaborn_grain.png')

        # Correlation heatmap on the demos' patients table, so lecture and demo numbers agree.
        paths = build(LECTURE_DIR.parent / 'scratch' / 'lecture07_data')
        vitals = pd.read_parquet(paths['patients'])[
            ['age', 'systolic_bp', 'bmi', 'steps_per_day', 'fasting_glucose']]
        corr = vitals.corr()
        print(corr.round(2))
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, ax=ax)
        ax.set_title('Correlations among five measures (2,400 synthetic patients)')
        fig.tight_layout()
        save_clean(fig, 'correlation_heatmap.png')

        # Accessible charts, saved in grayscale to show the cues survive without color.
        visit = np.arange(1, 6)
        nurse = [153, 148, 143, 139, 135]
        standard = [155, 152, 149, 146, 144]
        assert standard[-1] - nurse[-1] == 9
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        axes[0].plot(visit, standard, color=orange, marker='s', linestyle='--')
        axes[0].plot(visit, nurse, color=blue, marker='o', linestyle='-')
        axes[0].text(5.08, standard[-1], 'Standard care', va='center')
        axes[0].text(5.08, nurse[-1], 'Nurse-led', va='center')
        axes[0].set_xticks(visit)
        axes[0].set_xlim(0.8, 6)
        axes[0].set(title='Mean systolic BP by follow-up visit', xlabel='Visit',
                    ylabel='Mean systolic BP (mmHg)')
        x = np.arange(2)
        width = 0.35
        north_bars = axes[1].bar(x - width / 2, [58, 64], width, label='North', color=blue, hatch='//')
        south_bars = axes[1].bar(x + width / 2, [52, 55], width, label='South', color=orange, hatch='..')
        axes[1].bar_label(north_bars, fmt='%d%%')
        axes[1].bar_label(south_bars, fmt='%d%%')
        axes[1].set_xticks(x, ['2024-25', '2025-26'])
        axes[1].set_ylim(0, 100)
        axes[1].set(title='Flu vaccine uptake by season', xlabel='Season', ylabel='Vaccinated (%)')
        axes[1].legend(title='Clinic', frameon=False)
        fig.tight_layout()
        path = MEDIA_DIR / 'accessible_charts.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        rgb = plt.imread(path)[..., :3]
        plt.imsave(path, rgb @ [0.299, 0.587, 0.114], cmap='gray', vmin=0, vmax=1)
        print('Generated: accessible_charts.png (grayscale)')

        # The lecture's weekly chart with North's peak annotated.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([1, 2, 3, 4], [42, 45, 51, 48], 's--', color=blue, markersize=6, label='North')
        ax.plot([1, 2, 3, 4], [30, 33, 31, 36], 'o-', color=orange, markersize=6, label='South')
        ax.set(title='Weekly flu visits by clinic', xlabel='Week', ylabel='Flu clinic visits')
        ax.annotate('Peak: 51 visits', xy=(3, 51), xytext=(1.5, 50), arrowprops=dict(arrowstyle='->'))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([1, 2, 3, 4])
        ax.legend(frameon=False)
        fig.tight_layout()
        save_clean(fig, 'annotated_flu_line.png')


if __name__ == '__main__':
    render_reference_figures()
    draw_chart_selection()
    render_concept_figures()
