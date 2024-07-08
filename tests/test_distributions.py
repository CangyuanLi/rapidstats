import pytest
import scipy.stats

import rapidstats

Q = [i / 100 for i in range(100)]


@pytest.mark.parametrize("q", Q)
def test_norm_ppf(q: float):
    ref = scipy.stats.norm.ppf(q)
    rs = rapidstats.norm.ppf(q)

    pytest.approx(ref) == rs


@pytest.mark.parametrize("q", Q)
def test_norm_cdf(q: float):
    ref = scipy.stats.norm.cdf(q)
    rs = rapidstats.norm.cdf(q)

    pytest.approx(ref) == rs
