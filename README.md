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

**rapidstats** is a minimal library that implements fast statistical routines in Rust and Polars. Currently, its core purpose is to provide the Bootstrap class. Unlike scipy.stats.bootstrap, the Bootstrap class contains specialized functions (for e.g. confusion matrix, ROC-AUC, and so on) implemented in Rust as well as a generic interface for any Python callable. This makes it significantly faster than passing in a callable on the Python side. For example, bootstrapping confusion matrix statistics can be up to 40x faster.

*This library is in an alpha state. Although all functions are tested against existing libraries, use at your own risk. The API is subject to change very frequently.*

# Usage:

## Dependencies

**rapidstats** has a minimal set of dependencies. It only depends on Polars and tqdm (for progress bars). You may install pyarrow (`pip install rapidstats[pyarrow]`) to allow functions to take NumPy arrays, Pandas objects, and other objects that may be converted through Arrow.

## Installing

The easiest way is to install **rapidstats** is from PyPI using pip:

```sh
pip install rapidstats
```

# Notes

The purpose of **rapidstats** is not to replace scipy or scikit-learn, which would be a massive undertaking. At the moment, only functions that are not easily achievable using existing libraries or functions that are significantly more performant (e.g. the core loop is implemented in Rust) will be added.
