"""Run one import-safe NumPy analysis over the supplied fixture."""

from data_loader import load_measurements


def summarize(measurements):
    """Return shape-aware reductions and a scalar-to-1D mask count."""
    overall_mean = measurements.mean()
    column_means = measurements.mean(axis=0)
    row_means = measurements.mean(axis=1)
    review_values = measurements.reshape(measurements.size)
    review_mask = review_values >= 30
    review_count = int(review_mask.sum())
    return {
        "overall_mean": overall_mean,
        "column_means": column_means,
        "row_means": row_means,
        "review_count": review_count,
    }


def main():
    """Load the supplied fixture and print its deterministic summary."""
    measurements = load_measurements("observations.csv")
    summary = summarize(measurements)

    print(f"Measurements shape: {measurements.shape}")
    print(f"Measurements dtype: {measurements.dtype}")
    print(f'Overall mean: {summary["overall_mean"]:.1f}')
    print(f'Column means: {summary["column_means"]}')
    print(f'Column means shape: {summary["column_means"].shape}')
    print(f'Row means: {summary["row_means"]}')
    print(f'Row means shape: {summary["row_means"].shape}')
    print(f'Values at or above 30: {summary["review_count"]}')


if __name__ == "__main__":
    main()
