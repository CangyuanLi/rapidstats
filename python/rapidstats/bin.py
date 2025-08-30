import math

import polars as pl
from polars.series.series import ArrayLike


def _bin_width_to_count(x: ArrayLike, bin_width: float) -> int:
    x = pl.Series(x)

    return math.ceil((x.max() - x.min()) / bin_width)


def freedman_diaconis(x: ArrayLike) -> int:
    r"""The Freedman-Diaconis rule defines the bin width as

    \[
        h = 2\frac{IQR(x)}{\sqrt[3]{n}}
    \]

    where $x$ is the input array and $n$ is the length of $x$.

    The bin width is converted to a bin count via

    \[
        k = \lceil \frac{\max{x} - \min{x}}{h} \rceil
    \]

    Parameters
    ----------
    x : ArrayLike

    Returns
    -------
    int
        Bin count
    """
    x = pl.Series(x)

    iqr = x.quantile(0.75, interpolation="linear") - x.quantile(
        0.25, interpolation="linear"
    )

    bin_width = 2.0 * iqr * x.len() ** (-1.0 / 3.0)

    return _bin_width_to_count(x, bin_width)


def doane(x: ArrayLike) -> int:
    r"""Doane's rule defines the bin count as

    \[
        k = 1 + \log_{2}(n) + \log_{2}\left(1 + \frac{|g_{1}|}{\sigma_{g_{1}}}\right)
    \]

    where

    \[
        \sigma_{g_{1}} = \sqrt{\frac{6(n-2)}{(n+1)(n+3)}}
    \]

    Parameters
    ----------
    x : ArrayLike

    Returns
    -------
    int
        Bin count

    Raises
    ------
    ValueError
        If $n < 2$
    """
    x = pl.Series(x)
    x_len = x.len()

    if x_len <= 2:
        raise ValueError("Doane's rule requires at least 3 observations")

    g1 = abs(x.skew())
    sg1 = math.sqrt(6.0 * (x_len - 2) / ((x_len + 1.0) * (x_len + 3)))

    return int(1 + math.log2(x_len) + math.log2(1 + (g1 / sg1)))


def rice(x: ArrayLike) -> int:
    r"""Rice's rule defines the bin count as

    \[
        k = \lceil 2n^{\frac{1}{3}} \rceil
    \]

    Parameters
    ----------
    x : ArrayLike

    Returns
    -------
    int
        Bin count
    """
    return math.ceil(2 * (len(x) ** (1 / 3)))


def sturges(x: ArrayLike) -> int:
    r"""Sturges' rule defines the bin count as

    \[
        k = \lceil 1 + \log_{2}(n) \rceil
    \]

    Parameters
    ----------
    x : ArrayLike

    Returns
    -------
    int
        Bin count
    """
    return math.ceil(math.log2(len(x))) + 1


def scott(x: ArrayLike) -> int:
    r"""Scott's rule defines the bin width as

    \[
        h = 3.49\sigma n^{-\frac{1}{3}}
    \]

    The bin count is given by

    \[
        k = \lceil \frac{\max{x} - \min{x}}{h} \rceil
    \]

    Parameters
    ----------
    x : ArrayLike

    Returns
    -------
    int
        Bin count
    """
    x = pl.Series(x)

    bin_width = (3.49 * x.std()) / (len(x) ** (1 / 3))

    return _bin_width_to_count(x, bin_width)


def sqrt(x: ArrayLike) -> int:
    r"""The square root rule defines the bin count as

    \[
        k = \lceil\sqrt{n}\rceil
    \]

    Parameters
    ----------
    x : ArrayLike

    Returns
    -------
    int
        Bin count
    """
    return math.ceil(math.sqrt(len(x)))
