import numpy as np
import polars as pl
import pytest
import scipy.stats

import rapidstats

np.random.seed(208)


def reference_percentile_interval(bootstrap_stats, confidence_level):
    alpha = (1 - confidence_level) / 2
    interval = alpha, 1 - alpha

    def percentile_fun(a, q):
        return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(bootstrap_stats, interval[0] * 100)
    ci_u = percentile_fun(bootstrap_stats, interval[1] * 100)

    return (ci_l, ci_u)


def test_percentile_interval():
    n = 1_000
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2
    bootstrap_stats = np.random.uniform(size=n)

    rs = rapidstats._bootstrap._percentile_interval(bootstrap_stats, alpha)
    rs = (rs[0], rs[2])
    ref = reference_percentile_interval(bootstrap_stats, confidence_level)

    for a, b in zip(rs, ref):
        assert pytest.approx(a) == b


def reference_bca_interval(data, bootstrap_stats, confidence_level):
    alpha = (1 - confidence_level) / 2

    interval = scipy.stats._resampling._bca_interval(
        (data,),
        statistic=np.mean,
        axis=-1,
        alpha=alpha,
        theta_hat_b=bootstrap_stats,
        batch=None,
    )[:2]

    percentile_fun = scipy.stats._resampling._percentile_along_axis

    ci_l = percentile_fun(bootstrap_stats, interval[0] * 100)
    ci_u = percentile_fun(bootstrap_stats, interval[1] * 100)

    return (ci_l, ci_u)


def test_bca_interval():
    n = 1_000
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2
    bootstrap_stats = np.random.uniform(size=n)
    data = np.random.uniform(size=n)

    def _mean(df):
        return df["x"].mean()

    rs = rapidstats._bootstrap._bca_interval(
        original_stat=np.mean(data),
        bootstrap_stats=bootstrap_stats.tolist(),
        jacknife_stats=rapidstats._bootstrap._jacknife(
            pl.DataFrame({"x": data.tolist()}), _mean
        ),
        alpha=alpha,
    )
    rs = (rs[0], rs[2])

    ref = reference_bca_interval(data, bootstrap_stats, confidence_level)

    for a, b in zip(rs, ref):
        assert pytest.approx(a, rel=1e-5) == b
