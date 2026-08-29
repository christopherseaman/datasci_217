# Lecture 11 Demo Guide

These notebooks form one forecasting story, but every required notebook can run
top-to-bottom in a fresh local or Colab runtime. Each uses committed compact
data when available and otherwise downloads it from raw GitHub. No Drive mount
or manual upload is needed. The optional geography notebook is included in the
same notebook set and can be run after Demo 4 when its separate setup cell has
installed the geo packages.

The links below target the public `2026-refresh` branch while this review is in
progress, so they are the Colab execution option for the current notebooks. At
release time, retarget them to `main` or an immutable annual tag and recheck the
remote runtime; that publication check is separate from the local notebook
contract.

| Demo | Assignment pattern | Colab |
|---|---|---|
| 1. Release, grain, exploration, cleaning | Q1-Q2 | [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/2026-refresh/11/demo/01_setup.ipynb) |
| 2. Complete panel and past-only features | Q3-Q4 | [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/2026-refresh/11/demo/02_wrangling.ipynb) |
| 3. Training-only patterns and temporal split | Q5-Q6 | [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/2026-refresh/11/demo/03_model_prep.ipynb) |
| 4. Baseline, pipeline, frozen choice, report | Q7-Q9 | [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/2026-refresh/11/demo/04_modeling.ipynb) |
| 5. Optional zone-error choropleth | Non-graded | [Open in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/2026-refresh/11/demo/05_geo_bonus.ipynb) |

## Project Contract

- Grain: one `(pickup_zone_id, target_hour_utc)`.
- Target: next-hour `pickup_count`.
- Baseline: `lag_168`.
- Primary metric: MAE.
- Secondary metric: RMSE.
- Split: before May / May / June in New York local time.
- Predictions: clipped to nonnegative values.
- Reproducibility seed: `217` where an estimator supports `random_state`.

## Local Run

From `11/demo/`:

```bash
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab
```

The notebooks create `output/` as needed. Demo 2 writes a compact model table;
Demo 3 writes a small split manifest; Demo 4 writes Parquet predictions plus small
CSV and PNG evidence. Required notebooks rebuild prerequisites when those local
outputs are absent.

## Data Roles

- `yellow_taxi_2023_h1_event_sample.parquet` is a deterministic 30,000-event sample
  for audit and cleaning practice. It cannot reproduce full-panel counts.
- `yellow_taxi_2023_h1_zone_hour_counts.parquet` is the complete 12-zone hourly
  panel derived from all official January-June 2023 source rows.
- `demo_release_manifest.json` records hashes, source evidence, selection rules,
  and builder versions.
- `taxi_zone_lookup.csv` supplies human-readable labels and supports the optional map.
