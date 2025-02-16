import numpy as np
import pytest
import sklearn.metrics

import rapidstats

np.random.seed(208)

N_ROWS = 1_000

X = np.random.rand(N_ROWS)
Y = np.random.rand(N_ROWS)


def test_auc():
    indices = X.argsort()

    # trapezoidal
    ref = sklearn.metrics.auc(X[indices], Y[indices])
    res = rapidstats.auc(X, Y)

    assert pytest.approx(ref) == res
