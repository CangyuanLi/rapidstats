# rapidstats:
[![PyPI version](https://badge.fury.io/py/rapidstats.svg)](https://badge.fury.io/py/rapidstats)
![PyPI - Downloads](https://img.shields.io/pypi/dm/rapidstats)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://github.com/CangyuanLi/rapidstats/actions/workflows/tests.yaml/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

<p align="center">
  <a href="https://cangyuanli.github.io/rapidstats/latest/">Documentation</a>
<br>
</p>

## What is it?

**rapidstats** is a minimal library that implements fast statistical routines in Rust and Polars. While similar in spirit, it **does not** aim to be a complete re-implementation of libraries like **scikit-learn** or **scipy**. Only functions that can be significantly faster (e.g. a bootstrap class that offers optimized Rust kernels for metrics such as ROC-AUC) or significantly more ergonomic (e.g. dataframe-first encoders and scalers) are added.

*This library is in an alpha state. Although all functions are tested against existing libraries, use at your own risk. The API is subject to change very frequently.*

# Usage:

## Dependencies

**rapidstats** has a minimal set of dependencies. It only depends on **polars**, **narwhals** (for dataframe compatibility), and **tqdm** (for progress bars). You may install **pyarrow** (`pip install rapidstats[pyarrow]`) to allow functions to take **numpy** arrays, **pandas** objects, and other objects that may be converted through Arrow.

## Installing

The easiest way is to install **rapidstats** is from PyPI using pip:

```sh
pip install rapidstats
```

# Performance

**rapidstats** is very fast. For example, say you wanted the confusion matrix metrics for a 50,000 row dataset. You aren't sure what exact threshold you want yet, so you decide to compute the metrics for multiple thresholds, let's say 500. With sklearn, this takes 40 seconds. With **rapidstats**, this takes just .2 seconds, a 198x speedup! Furthermore, **rapidstats** can use a cumuluative sum algorithm that computes the metrics at all possible thresholds, not just these particular 500. So finding the metrics for 500 or 50,000 metrics takes the exact same amount of time. In addition, even just looping the **rapidstats** version is a 58x speedup, since **rapidstats** applies several optimizations, such as computing the basic confusion matrix (TP, FP, FN, TN) using a nice bincount trick and avoiding re-computing this basic matrix for each different metric.

![](https://github.com/CangyuanLi/rapidstats/raw/master/docs/images/cm_at_t.png)

Similarly, calculating the bootstrapped (100 iterations) ROC-AUC of a 25,000 sample dataset takes only .15 seconds, compared to .83 seconds for the equivalent sklearn + scipy operation, a speedup of 5.3x.

![](https://github.com/CangyuanLi/rapidstats/raw/master/docs/images/bs_roc_auc.png)
