"""Demonstrate the required Lecture 03 ndarray operations in order."""

import numpy as np


def main():
    """Run the deterministic ndarray mental-model demonstration."""
    print("Metadata")
    scores = np.array([18, 21, 24, 19], dtype=np.float64)
    score_table = np.array(
        [
            [18, 21, 24],
            [20, 22, 26],
        ],
        dtype=np.float64,
    )
    print(f"scores: {scores}")
    print("score table:")
    print(score_table)
    print(
        "scores metadata: "
        f"shape={scores.shape}, ndim={scores.ndim}, "
        f"size={scores.size}, dtype={scores.dtype}"
    )
    print(
        "table metadata: "
        f"shape={score_table.shape}, ndim={score_table.ndim}, "
        f"size={score_table.size}, dtype={score_table.dtype}"
    )

    print("Selection")
    print(f"first score: {scores[0]}")
    print(f"second row: {score_table[1]}")
    print(f"third column value: {score_table[0, 2]}")
    print(f"middle scores: {scores[1:3]}")
    print(f"second column: {score_table[:, 1]}")

    print("View and copy")
    view_source = np.array([10, 20, 30, 40])
    middle_view = view_source[1:3]
    middle_view[0] = 99
    print(f"source after view mutation: {view_source}")

    copy_source = np.array([10, 20, 30, 40])
    middle_copy = copy_source[1:3].copy()
    middle_copy[0] = 99
    print(f"source after copy mutation: {copy_source}")
    print(f"copy after mutation: {middle_copy}")

    print("Mask")
    mask_scores = np.array([18, 21, 24, 19])
    review_mask = mask_scores >= 20
    print(f"mask: {review_mask}")
    print(f"masked values: {mask_scores[review_mask]}")

    print("Same-shape arithmetic")
    baseline = np.array([18, 21, 24])
    follow_up = np.array([20, 20, 27])
    change = follow_up - baseline
    print(f"change: {change}")

    print("Reductions")
    measurements = np.array(
        [
            [10, 20, 30],
            [20, 30, 40],
        ],
        dtype=np.float64,
    )
    overall_mean = measurements.mean()
    column_means = measurements.mean(axis=0)
    row_means = measurements.mean(axis=1)
    print(f"overall mean: {overall_mean:.1f}")
    print(f"column means: {column_means}")
    print(f"column means shape: {column_means.shape}")
    print(f"row means: {row_means}")
    print(f"row means shape: {row_means.shape}")

    print("Reshape and transpose")
    values = np.array([1, 2, 3, 4, 5, 6])
    grid = values.reshape(2, 3)
    transposed = grid.T
    print("grid:")
    print(grid)
    print(f"grid shape: {grid.shape}")
    print("transpose:")
    print(transposed)
    print(f"transpose shape: {transposed.shape}")

    print("Scalar-to-1D broadcast")
    broadcast_scores = np.array([18, 21, 24])
    adjusted_scores = broadcast_scores + 1
    print(f"adjusted scores: {adjusted_scores}")
    print(f"adjusted shape: {adjusted_scores.shape}")


if __name__ == "__main__":
    main()
