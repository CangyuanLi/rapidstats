from __future__ import annotations

from typing import Callable

import narwhals.stable.v1 as nw
import narwhals.stable.v1.typing as nwt
import polars as pl

from ._rustystats import _thin_points_greedy


class ScreenTransform:
    """Transforms the data from raw units into "screen" space, e.g. pixels.

    Parameters
    ----------
    width : float
        The width of the screen
    height : float
        The height of the screen
    xmin : float | None, optional
        The min x value. If None, is the min observed x value, by default None
    xmax : float | None, optional
        The max x value. If None, is the max observed x value, by default None
    ymin : float | None, optional
        The min y value. If None, is the min observed y value, by default None
    ymax : float | None, optional
        The max y value. If None, is the max observed y value, by default None
    """

    def __init__(
        self,
        width: float,
        height: float,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
    ):
        self.width = width
        self.height = height
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __call__(self, df: nwt.IntoDataFrameT, x: str, y: str) -> nwt.IntoDataFrameT:
        nw_df = nw.from_native(df)

        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax

        if xmin is None:
            xmin = nw_df[x].min()

        if xmax is None:
            xmax = nw_df[x].max()

        if ymin is None:
            ymin = nw_df[y].min()

        if ymax is None:
            ymax = nw_df[y].max()

        if xmax < xmin:
            raise ValueError("xmax must be >= xmin")

        if ymax < ymin:
            raise ValueError("ymax must be >= ymin")

        sx = self.width / (xmax - xmin)
        sy = self.height / (ymax - ymin)

        return nw_df.with_columns(
            nw.col(x).__sub__(xmin).__mul__(sx), nw.col(y).__sub__(ymin).__mul__(sy)
        ).to_native()


def thin_points(
    df: nwt.IntoDataFrameT,
    x: str,
    y: str,
    min_distance: float,
    always_keep: str | None = None,
    order: str | None = None,
    transform: (
        Callable[[nwt.IntoDataFrameT, str, str], nwt.IntoDataFrameT] | None
    ) = None,
) -> nwt.IntoDataFrameT:
    """Given a set of points, select points such that each point is visually distinct
    from the other.

    Parameters
    ----------
    df : nwt.IntoDataFrameT
        The DataFrame containing the points
    x : str
        The column denoting the x-axis
    y : str
        The column denoting the y-axis
    min_distance : float
        The minimum distance between each point
    always_keep : str | None, optional
        A boolean column denoting whether the point should always be kept regardless of
        distance. If None, no points are always kept (equivalent to a boolean column of
        all false), by default None
    order : str | None, optional
        A u64 column (lower is better) that controls which points in a cluster are kept.
        If None, points are kept in insertion order, by default None
    transform : Callable[[nwt.IntoDataFrameT, str, str], nwt.IntoDataFrameT] | None, optional
        A callable that accepts `df`, `x`, and `y` used to transform the data before
        applying the thinning algorithm. For instance, `ScreenTransform` can map the
        data to pixel space, so that `min_distance` can refer to pixels instead of raw
        units. If None, no transformations are applied, by default None

    Returns
    -------
    nwt.IntoDataFrameT
        The original DataFrame filtered to the thinned points
    """
    if transform is not None:
        sanitized = transform(df, x, y)

    to_select = [
        c
        for c in [
            x,
            y,
            always_keep,
            order,
        ]
        if c is not None
    ]

    sanitized = nw.from_native(sanitized).select(to_select)

    sanitized = sanitized.to_polars().with_columns(
        pl.col(x, y).cast(pl.Float64),
    )

    if always_keep is None:
        always_keep = "__rapidstats_always_keep__"
        sanitized = sanitized.with_columns(pl.lit(False).alias(always_keep))
    else:
        sanitized = sanitized.with_columns(pl.col(always_keep).cast(pl.Boolean))

    if order is not None:
        sanitized = sanitized.with_columns(pl.col(order).cast(pl.UInt64))

    to_keep = _thin_points_greedy(
        df=sanitized,
        x=x,
        y=y,
        min_distance=min_distance,
        always_keep=always_keep,
        order=order,
    )

    return nw.from_native(df).filter(to_keep).to_native()
