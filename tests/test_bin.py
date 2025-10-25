import numpy as np
import pytest

import rapidstats as rs

RNG = np.random.default_rng(208)
SIZE = 1_000

UNIFORM = RNG.uniform(size=SIZE)
EXPONENTIAL = RNG.exponential(size=SIZE)
HEAVILY_ZERO = [0] * SIZE + [1] * (SIZE // 200)

ARRAYS = [UNIFORM, EXPONENTIAL, HEAVILY_ZERO]


@pytest.mark.parametrize("x", ARRAYS)
def test_freedman_diaconis(x):
    rs.bin.freedman_diaconis(x)


@pytest.mark.parametrize("x", ARRAYS)
def test_doane(x):
    rs.bin.doane(x)


@pytest.mark.parametrize("x", ARRAYS)
def test_rice(x):
    rs.bin.rice(x)


@pytest.mark.parametrize("x", ARRAYS)
def test_sturges(x):
    rs.bin.sturges(x)


@pytest.mark.parametrize("x", ARRAYS)
def test_scott(x):
    rs.bin.scott(x)


@pytest.mark.parametrize("x", ARRAYS)
def test_sqrt(x):
    rs.bin.sqrt(x)
