"""Render Lecture 11's generated figure into ../media/.

The pairplot uses the taxi demo's training rows, rebuilt exactly as Demo 3 builds them,
and asserts the correlations the lecture snippet prints.

Run from the repository root:
    uv run --project 11/demo python 11/scripts/generate_lecture_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

LECTURE_DIR = Path(__file__).resolve().parents[1]
PANEL = LECTURE_DIR / "demo" / "data" / "yellow_taxi_2023_h1_zone_hour_counts.parquet"


def training_rows():
    """Demo 3's training rows: past-only lags, sorted by target hour then zone."""
    table = pd.read_parquet(PANEL).sort_values(["pickup_zone_id", "target_hour_utc"]).copy()
    table["hour_of_day"] = table["target_hour_utc"].dt.tz_convert("America/New_York").dt.hour
    grouped = table.groupby("pickup_zone_id", sort=False)["pickup_count"]
    for lag in (1, 24, 168):
        table[f"lag_{lag}"] = grouped.shift(lag)
    table = table.dropna(subset=["lag_168"]).sort_values(["target_hour_utc", "pickup_zone_id"])
    train = table[table["target_hour_utc"] < pd.Timestamp("2023-05-01", tz="America/New_York")]
    return train.reset_index(drop=True)


def taxi_pairplot():
    train = training_rows()
    assert len(train) == 32_532
    cols = ["pickup_count", "lag_1", "lag_168", "hour_of_day"]
    sample = train[cols].sample(2000, random_state=217)
    corr = sample.corr().round(2)
    print(corr)
    assert corr.loc["pickup_count", "lag_168"] == 0.94
    assert corr.loc["pickup_count", "hour_of_day"] == 0.62
    grid = sns.pairplot(sample, corner=True, height=1.9, plot_kws={"s": 8, "alpha": 0.4})
    grid.figure.savefig(LECTURE_DIR / "media" / "taxi_pairplot.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)
    print("wrote taxi_pairplot.png")


if __name__ == "__main__":
    taxi_pairplot()
