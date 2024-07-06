import pytest
import scipy.stats

import rapidstats


@pytest.mark.parametrize("q", [i / 100 for i in range(100)])
def test_norm_ppf(q: float):
    ref = scipy.stats.norm.ppf(q)
    rs = rapidstats.norm.ppf(q)

    pytest.approx(ref) == rs
