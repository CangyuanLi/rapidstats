import math

import polars as pl
from polars.series.series import ArrayLike


def _bin_width_to_count(x: ArrayLike, bin_width: float) -> int:
    x = pl.Series(x)

    return math.ceil((x.max() - x.min()) / bin_width)


def freedman_diaconis(x: ArrayLike) -> int:
    x = pl.Series(x)

    iqr = x.quantile(0.75, interpolation="linear") - x.quantile(
        0.25, interpolation="linear"
    )

    bin_width = 2.0 * iqr * x.len() ** (-1.0 / 3.0)

    return _bin_width_to_count(x, bin_width)


def doane(x: ArrayLike) -> int:
    x = pl.Series(x)
    x_len = x.len()

    if x_len <= 2:
        raise ValueError("Doane's rule requires at least 3 observations")

    g1 = abs(x.skew())
    sg1 = math.sqrt(6.0 * (x_len - 2) / ((x_len + 1.0) * (x_len + 3)))

    return int(1 + math.log2(x_len) + math.log2(1 + (g1 / sg1)))


def rice(x: ArrayLike) -> int:
    return math.ceil(2 * (len(x) ** (1 / 3)))


def sturges(x: ArrayLike) -> int:
    return math.ceil(math.log2(len(x))) + 1


def scott(x: ArrayLike) -> int:
    x = pl.Series(x)

    bin_width = (3.49 * x.std()) / (len(x) ** (1 / 3))

    return _bin_width_to_count(x, bin_width)


def sqrt(x: ArrayLike) -> int:
    return math.ceil(math.sqrt(len(x)))
