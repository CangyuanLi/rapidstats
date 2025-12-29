import polars as pl
import polars.testing

from rapidstats.viz import ScreenTransform, thin_points


def test_basic_thinning_input_order_wins():
    df = pl.DataFrame({"x": [0.0, 0.1, 10.0], "y": [0.0, 0.1, 10.0]})
    out = thin_points(df, x="x", y="y", min_distance=1.0)

    # first two are within ~0.141 < 1.0, so keep first, drop second; keep third
    polars.testing.assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "x": [0.0, 10.0],
                "y": [0.0, 10.0],
            }
        ),
    )


def test_always_keep_is_never_dropped_and_suppresses_others():
    df = pl.DataFrame(
        {
            "x": [0.0, 0.1, 10.0],
            "y": [0.0, 0.1, 10.0],
            "always_keep": [False, True, False],
        }
    )
    out = thin_points(df, x="x", y="y", min_distance=1.0, always_keep="always_keep")

    # always_keep point (0.1,0.1) must remain.
    # It is within 1.0 of (0,0), so (0,0) should be suppressed if always_keep is processed first.
    polars.testing.assert_frame_equal(
        out.select("x", "y"),
        pl.DataFrame(
            {
                "x": [0.1, 10.0],
                "y": [0.1, 10.0],
            }
        ),
    )


def test_order_controls_winner_in_cluster():
    df = pl.DataFrame(
        {
            "x": [0.0, 0.1, 10.0],
            "y": [0.0, 0.1, 10.0],
            # lower is earlier/better
            "order": [5, 1, 0],
        }
    )
    out = thin_points(df, x="x", y="y", min_distance=1.0, order="order")

    # (0.0,0.0) and (0.1,0.1) are within 1.0; point with smaller order=1 wins over order=5
    # (10,10) is far away and is kept
    polars.testing.assert_frame_equal(
        out.select("x", "y"),
        pl.DataFrame(
            {
                "x": [0.1, 10.0],
                "y": [0.1, 10.0],
            }
        ),
    )


def test_order_tie_breaker_is_deterministic_by_insertion_order():
    df = pl.DataFrame(
        {
            "x": [0.0, 0.1, 10.0],
            "y": [0.0, 0.1, 10.0],
            "order": [1, 1, 0],  # tie for first two
        }
    )
    out = thin_points(df, x="x", y="y", min_distance=1.0, order="order")

    # With same order, deterministic tie-break should pick the earlier row (0.0,0.0)
    polars.testing.assert_frame_equal(
        out.select("x", "y"),
        pl.DataFrame(
            {
                "x": [0.0, 10.0],
                "y": [0.0, 10.0],
            }
        ),
    )


def test_transform_changes_which_points_conflict():
    # In raw data space, these are far apart in x but identical in y.
    # A screen transform with a small width can compress x so they become close in screen coords.
    df = pl.DataFrame({"x": [0.0, 50.0, 100.0], "y": [0.0, 0.0, 0.0]})

    # Without transform, min_distance=10 keeps all (distance between neighbors is 50).
    out_raw = thin_points(df, x="x", y="y", min_distance=10.0)
    assert out_raw.height == 3

    # With screen transform mapping x to width=10, x span=100 -> scale=0.1, so:
    # x_screen = [0, 5, 10]. Neighbor distance is 5.
    # With min_distance=6, middle conflicts with both ends; input order keeps first and third.
    tf = ScreenTransform(
        width=10.0, height=10.0
    )  # adjust constructor to your implementation
    out_screen = thin_points(df, x="x", y="y", min_distance=6.0, transform=tf)
    assert out_screen.height == 2
    assert out_screen["x"].to_list() == [0.0, 100.0]


def test_all_columns_preserved():
    df = pl.DataFrame(
        {"x": [0.0, 0.1, 10.0], "y": [0.0, 0.1, 10.0], "extra": ["a", "b", "c"]}
    )
    out = thin_points(df, x="x", y="y", min_distance=1.0)

    assert out.columns == df.columns
