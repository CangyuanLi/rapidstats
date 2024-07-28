from pathlib import Path

import numpy as np
import polars as pl

BASE_PATH = Path(__file__).resolve().parents[0]

np.random.seed(208)

N_ROWS = 1_000_000


def main():
    y_true = np.random.choice([True, False], N_ROWS)
    y_score = np.random.rand(N_ROWS)
    y_pred = y_score >= 0.5

    df = pl.DataFrame({"y_true": y_true, "y_score": y_score, "y_pred": y_pred})

    df.write_parquet(BASE_PATH / "benchmark_data.parquet")


if __name__ == "__main__":
    main()
