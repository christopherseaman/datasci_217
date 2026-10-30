"""Rebuild one row of A1c features per patient month by month through 2025, printing progress, and save the year-end table.

Run it from the demo folder in a terminal, inside tmux on a server:

    python3 build_features.py

It reads data/labs.parquet (written by generate_data.py) and writes output/a1c_features.csv.
"""

import time
from pathlib import Path

import pandas as pd

labs = pd.read_parquet('data/labs.parquet', columns=['patient_id', 'collected_at', 'test_name', 'result'])
a1c = labs[labs['test_name'] == 'A1c'].dropna(subset=['result']).sort_values('collected_at')

for month in range(1, 13):
    next_month = pd.Timestamp('2025-01-01') + pd.DateOffset(months=month)
    so_far = a1c[a1c['collected_at'] < next_month]
    features = so_far.groupby('patient_id', as_index=False).agg(
        tests=('result', 'count'),
        mean_a1c=('result', 'mean'),
        max_a1c=('result', 'max'),
        latest_a1c=('result', 'last'),
    )
    print(f'month {month} of 12: {len(features):,} patients', flush=True)
    # shortcut: the pause stands in for the hours a real feature build takes; delete it for real work.
    time.sleep(10)

Path('output').mkdir(exist_ok=True)
features.to_csv('output/a1c_features.csv', index=False)
print('done: wrote output/a1c_features.csv')
