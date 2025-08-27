import pytest
import scipy.stats

import rapidstats

Q = [i / 100 for i in range(100)]


@pytest.mark.parametrize("q", Q)
def test_norm_ppf(q: float):
    ref = scipy.stats.norm.ppf(q)
    rs = rapidstats.norm.ppf(q)

    assert pytest.approx(ref) == rs


@pytest.mark.parametrize("q", Q)
def test_norm_cdf(q: float):
    ref = scipy.stats.norm.cdf(q)
    rs = rapidstats.norm.cdf(q)

    assert pytest.approx(ref) == rs


def test_random_seed_sequence():
    r = rapidstats.Random(208)
    run1 = r.poisson(1, 100)
    run2 = r.poisson(1, size=100)

    assert not (pytest.approx(run1) == run2)


def test_random_reproducible():
    assert pytest.approx(rapidstats.Random(208).poisson(1, 100)) == rapidstats.Random(
        208
    ).poisson(1, 100)
