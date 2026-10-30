#!/usr/bin/env python3
"""Measure the Lecture 08 performance comparisons on the Demo 3 lab extract and draw media/perf_combined.png.

Panels: four groupby() calls vs one .agg() for per-patient A1c summaries, a lambda vs the
built-in "std" per patient, and memory of the lab table with text vs category keys.

Usage, from the repository root (DATA_DIR defaults to 08/demo/data and is built when missing):
  08/demo/.venv/bin/python 08/perf_benchmark.py [DATA_DIR]
"""

import sys
import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DEMO = HERE / 'demo'
SLOW, FAST = '#9a9a9a', '#1f6fb4'


def load_labs(data):
    if not (data / 'labs.parquet').exists():
        sys.path.insert(0, str(DEMO))
        import generate_data
        generate_data.build(data)
    return pd.read_parquet(data / 'labs.parquet')


def best_seconds(func, repeat):
    return min(timeit.repeat(func, number=1, repeat=repeat))


def measure(labs):
    a1c = labs[labs['test_name'] == 'A1c'].dropna(subset=['result'])
    by_patient = a1c.groupby('patient_id')['result']
    stats = ['count', 'mean', 'std', 'max']
    compact = labs.copy()
    for column in ['clinic', 'test_name', 'unit']:
        compact[column] = compact[column].astype('category')
    return {
        'split': (best_seconds(lambda: [a1c.groupby('patient_id')['result'].agg(f) for f in stats], 7),
                  best_seconds(lambda: a1c.groupby('patient_id')['result'].agg(stats), 7)),
        'lambda': (best_seconds(lambda: by_patient.agg(lambda s: s.std()), 3),
                   best_seconds(lambda: by_patient.agg('std'), 7)),
        'memory': (labs.memory_usage(deep=True).sum() / 1e6, compact.memory_usage(deep=True).sum() / 1e6),
        'rows': len(a1c), 'patients': a1c['patient_id'].nunique(),
    }


def seconds_label(s):
    return f'{s * 1000:.0f} ms' if s < 1 else f'{s:.1f} s'


def panel(ax, title, labels, values, fmt):
    bars = ax.barh(labels, values, color=[SLOW, FAST], height=0.55)
    ax.invert_yaxis()
    ax.set_title(title, loc='left', fontsize=12)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.set_xlim(0, max(values) * 1.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                fmt(value), va='center', fontsize=11)


def plot(m, outfile):
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), dpi=150)
    slow, fast = m['split']
    panel(axes[0], f'Four summaries per patient: {slow / fast:.1f}x faster with one split',
          ['4 groupby() calls', 'one .agg() call'], [slow, fast], seconds_label)
    slow, fast = m['lambda']
    panel(axes[1], f'SD per patient: built-in about {round(slow / fast, -2):,.0f}x faster than a lambda',
          ['agg(lambda s: s.std())', "agg('std')"], [slow, fast], seconds_label)
    before, after = m['memory']
    panel(axes[2], f'Lab table memory: {before / after:.1f}x smaller with category keys',
          ['text keys', 'category keys'], [before, after], lambda v: f'{v:.0f} MB')
    fig.suptitle(f"{m['rows']:,} A1c results from {m['patients']:,} patients; 2,000,000-row lab table",
                 x=0.01, ha='left', fontsize=10, color='#555555')
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


if __name__ == '__main__':
    measured = measure(load_labs(Path(sys.argv[1]) if len(sys.argv) > 1 else DEMO / 'data'))
    print(measured)
    plot(measured, HERE / 'media' / 'perf_combined.png')
    print('Wrote', HERE / 'media' / 'perf_combined.png')
