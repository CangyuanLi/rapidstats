from __future__ import annotations

import copy
import inspect
import logging
import math
import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, TypedDict

import narwhals.stable.v1 as nw
import narwhals.stable.v1.typing as nwt
import polars as pl
from polars.series.series import ArrayLike
from tqdm.auto import tqdm

from ._corr import correlation_matrix
from .metrics import roc_auc

logger = logging.getLogger(__name__)


class Estimator(Protocol):
    """A class that implements a `.fit(X, y, **kwargs)` method."""

    def fit(self, X, y, **kwargs): ...


def _write_list(lst, path: str | Path):
    with open(path, "w") as f:
        json.dump(lst, f)


def _read_list(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def _copy(estimator: Estimator):
    if inspect.isclass(estimator):
        raise TypeError("Must pass in class instance, not class")

    if hasattr(estimator, "__sklearn_clone__"):
        return estimator.__sklearn_clone__()

    return copy.deepcopy(estimator)


def _roc_auc(est, X, y) -> float:
    y_score = est.predict_proba(X)[:, 0]

    return roc_auc(y, y_score)


class EarlyStopping:
    """A callback that activates early stopping.

    Parameters
    ----------
    X : Optional[Any], optional
        The evaluation dataset the model should predict. If None, it will first
        look for the existence of an `eval_set` parameter. If `eval_set` is not
        available, it will use the training data, by default None
    y : Optional[Any], optional
        The evaluation ground truth target, by default None
    metric : Callable[[Estimator, Any, Any], float], optional
        A callable that takes in the estimator, `X`, `y` and returns a float, by default
        a function that calls `.predict_proba(X)[:, 0]` on the fit estimator to obtain
        y_pred and and returns the ROC-AUC
    max_delta : float, optional
        The maximum difference between the best iteration and the worst iteration
        before stopping, by default 0.1
    direction : Literal["maximize", "minimize"], optional
        Whether the metric should be maximized or minimized, by default "maximize"
    """

    def __init__(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        metric: Callable[[Estimator, Any, Any], float] = _roc_auc,
        max_delta: float = 0.1,
        direction: Literal["maximize", "minimize"] = "maximize",
    ):
        self.X = X
        self.y = y
        self.metric_func = metric
        self.max_delta = max_delta
        self.direction = direction
        self.maximize = direction == "maximize"

        self._stop = False
        self.best_metric = -math.inf if self.maximize else math.inf
        self.best_iteration = 0
        self.metrics: list[float] = []

    def __call__(self, rfe_state: RFEState):
        eval_set = (
            (rfe_state["X"], rfe_state["y"])
            if rfe_state["eval_set"] is None
            else rfe_state["eval_set"][0]
        )
        X = self.X if self.X is not None else eval_set[0]
        y = self.y if self.y is not None else eval_set[1]
        est = rfe_state["estimator"]

        metric = self.metric_func(est, X, y)
        self.metrics.append(metric)

        if self.maximize:
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_iteration = rfe_state["iteration"]

            if self.best_metric - metric >= self.max_delta:
                self._stop = True
        else:
            if metric < self.best_metric:
                self.best_metric = metric
                self.best_iteration = rfe_state["iteration"]

            if metric - self.best_metric >= self.max_delta:
                self._stop = True

    def stop(self) -> bool:
        return self._stop


class ModelCheckpoint:
    """A callback that saves out the fit estimator every RFE iteration. It first checks
    if the estimator has a "save_model" method and saves it out as "{iteration}.sav" if
    so. If not, it saves it out as "{iteration}.pkl".

    Parameters
    ----------
    out_dir : str | Path
        The directory to save the model to
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)

    def __call__(self, rfe_state: RFEState):
        est = rfe_state["estimator"]
        iteration = rfe_state["iteration"]

        if hasattr(est, "save_model"):
            est.save_model(self.out_dir / f"{iteration}.sav")
        else:
            with open(self.out_dir / f"{iteration}.pkl", "wb") as f:
                pickle.dump(est, f)


def _get_step(n_features: int, step: float) -> int:
    if 0 < step < 1:
        step = int(max(1, step * n_features))

    return step


def _get_max_iterations(n_features: int, n_features_to_select: int, step: float):
    i = 1
    while n_features >= n_features_to_select:
        n_features -= _get_step(n_features, step)
        i += 1

    return i


class RFEState(TypedDict):
    """The state at each RFE iteration.

    Attributes
    ----------
    estimator: Estimator
        The fit estimator
    X: Any
        A DataFrame of features reduced to the features at that iteration
    y: Any
        The target
    eval_set: Optional[list[tuple[Any, Any]]]
        Data to use for evaluation. If present, will be reduced to the features at that
        iteration
    features: list[str]
        The features at that iteration
    iteration: int
        The iteration number, starting from 0
    """

    estimator: Estimator
    X: Any
    y: Any
    eval_set: Optional[list[tuple[Any, Any]]]
    features: list[str]
    iteration: int


def _get_feature_importance(est):
    if hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
    elif hasattr(est, "coef_"):
        importances = est.coef_
    else:
        raise AttributeError("Could not find either `feature_importances_` or `coef_`.")

    if hasattr(importances, "ravel"):
        if callable(importances.ravel):
            importances = importances.ravel()

    return importances


def _rfe_get_feature_importance(rfe_state: RFEState) -> ArrayLike:
    return _get_feature_importance(rfe_state["estimator"])


class RFE:
    """Performs recursive feature elimination. Recursively drop the least important
    features until only `n_features_to_select` features remain. This is done by
    training `estimator` on the intial set of features, returning the importances
    through `importance`, and dropping the bottom `step` features, repeated until
    the stopping condition is met.

    Parameters
    ----------
    estimator : Estimator
        An unfit estimator used to train the model each iteration
    n_features_to_select : float, optional
        The desired final number of features, by default 1
    step : float, optional
        The number (if an int) or percent (if a float) of features to eliminate each
        iteration. If a percent, the denominator is the number of remaining features
        each iteration, by default 1
    importance : Callable[[RFEState], ArrayLike], optional
        A callable that takes RFEState as input and returns an ArrayLike in the
        same order as the features representing the importance of each feature, by
        default a function that attempts to read the "feature_importances_"
        attribute of the fitted estimator
    callbacks : Optional[Iterable[Callable[[RFEState], Any]]], optional
        An iterable of callbacks to run each iteration, by default None
    quiet : bool, optional
        Whether to display information like progress bars, by default False

    Attributes
    ----------
    unfit_estimator : Estimator,
    n_features_to_select : float
    step : float
    importance : Callable[[RFEState], ArrayLike]
    callbacks : Optional[Iterable[Callable[[RFEState], Any]]]
    quiet : bool
    selected_features_ : list[str]
        The selected features sorted alphabetically, available only after fitting
    """

    def __init__(
        self,
        estimator: Estimator,
        n_features_to_select: float = 1,
        step: float = 1,
        importance: Callable[[RFEState], ArrayLike] = _rfe_get_feature_importance,
        callbacks: Optional[Iterable[Callable[[RFEState], Any]]] = None,
        quiet: bool = False,
    ):
        self.unfit_estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.importance = importance
        self.callbacks = [] if callbacks is None else callbacks
        self.quiet = quiet

    def fit(
        self,
        X: nwt.IntoDataFrame,
        y: Any,
        **fit_kwargs,
    ):
        """Fit from a DataFrame of features and a target.

        Parameters
        ----------
        X : nwt.IntoDataFrame
            A DataFrame of features
        y : Any
            A target

        Returns
        -------
        Self
        """
        X_nw = nw.from_native(X, eager_only=True)

        if "eval_set" in fit_kwargs:
            eval_set = [
                (nw.from_native(x_val), y_val)
                for x_val, y_val in fit_kwargs["eval_set"]
            ]
        else:
            eval_set = None

        n_features = X_nw.shape[1]
        n_features_to_select = self.n_features_to_select
        stop = False
        step = self.step
        remaining_features = X_nw.columns.copy()

        if 0 < n_features_to_select < 1:
            n_features_to_select = int(n_features * n_features_to_select)

        iteration = 0
        with tqdm(
            total=_get_max_iterations(n_features, n_features_to_select, step),
            disable=self.quiet,
        ) as pbar:
            while len(remaining_features) >= n_features_to_select:
                features = remaining_features
                X_loop = X_nw.select(features).to_native()

                if eval_set is not None:
                    fit_kwargs["eval_set"] = [
                        (X_val.select(features).to_native(), y_val)
                        for X_val, y_val in eval_set
                    ]

                est = _copy(self.unfit_estimator).fit(
                    X_loop,
                    y,
                    **fit_kwargs,
                )

                state = RFEState(
                    estimator=est,
                    X=X_loop,
                    y=y,
                    eval_set=fit_kwargs.get("eval_set", None),
                    features=features,
                    iteration=iteration,
                )

                for callback in self.callbacks:
                    callback(state)

                    if hasattr(callback, "stop"):
                        if callable(callback.stop):
                            stop = callback.stop()

                            if stop:
                                break

                if stop:
                    break

                len_features = len(features)
                real_step = _get_step(len_features, step)
                k = len_features - real_step

                if k <= 0:
                    break

                remaining_features = (
                    pl.LazyFrame(
                        {"importance": self.importance(state), "feature": features}
                    )
                    .sort(pl.col("importance").abs(), descending=True)
                    .select("feature")
                    .head(k)
                    .collect()
                    .get_column("feature")
                    .to_list()
                )

                iteration += 1
                pbar.update(1)

        self._estimator_ = est
        self.selected_features_ = sorted(features)

        return self

    def transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        """Reduce X to the selected features.

        Parameters
        ----------
        X : nwt.IntoFrameT

        Returns
        -------
        nwt.IntoFrameT
        """
        return nw.from_native(X).select(self.selected_features_).to_native()

    def fit_transform(
        self, X: nwt.IntoDataFrameT, y: Any, **fit_kwargs
    ) -> nwt.IntoDataFrameT:
        """Equivalent to calling `.fit(X, y, **fit_kwargs).transform(X)`.

        Parameters
        ----------
        X : nwt.IntoDataFrameT
        y : Any

        Returns
        -------
        nwt.IntoDataFrameT
        """
        return self.fit(X, y, **fit_kwargs).transform(X)

    def save(self, path: str | Path):
        """Saves the fit selector.

        Parameters
        ----------
        path : str | Path
            The file to save to

        Returns
        -------
        Self
        """
        _write_list(self.selected_features_, path)

        return self

    @staticmethod
    def load(path: str | Path):
        """Loads in an already fit selector.

        Parameters
        ----------
        path : str | Path
            The file to load from

        Returns
        -------
        Self
        """
        x = RFE(estimator=None)
        x.selected_features_ = _read_list(path)

        return x


class NFEState(TypedDict):
    """A dictionary storing NFE state.

    Attributes
    ----------
    estimator: Estimator
        A fit estimator
    X: Any
        A DataFrame of features
    y: Any
        A target
    """

    estimator: Estimator
    X: Any
    y: Any


def _nfe_get_feature_importance(nfe_state: NFEState) -> ArrayLike:
    return _get_feature_importance(nfe_state["estimator"])


class NFE:
    """Performs noise feature elimination. A model is estimated with the features
    and a column of random noise added. Only features that have a higher importance
    than the random noise are selected.

    Parameters
    ----------
    estimator : Estimator
        An unfit estimator used to train the model
    importance : Callable[[NFEState], ArrayLike], optional
        A callable that determines how feature importance is computed. It should accept
        NFEState and return an ArrayLike of importances in the same order as features
        are encountered in X, by default a function that attempts to read a
        "feature_importances_" attribute from the fitted estimator
    seed : Optional[int], optional
        The seed, by default 208

    Attributes
    ----------
    unfit_estimator : Estimator
    importance : Callable[[NFEState], ArrayLike]
    seed : Optional[int]
    selected_features_ : list[str]
        The selected features sorted alphabetically, available only after fitting
    """

    _NOISE_COL = "__rapidstats_nfe_random_noise__"

    def __init__(
        self,
        estimator: Estimator,
        importance: Callable[[NFEState], ArrayLike] = _nfe_get_feature_importance,
        seed: Optional[int] = 208,
    ):
        self.unfit_estimator = estimator
        self.importance = importance
        self.seed = seed

    def _add_noise(self, df: nw.DataFrame) -> nw.DataFrame:
        noise_col = self._NOISE_COL

        n_rows = df.shape[0]

        return df.with_row_index(noise_col).with_columns(
            nw.col(noise_col)
            .sample(n_rows, with_replacement=True, seed=self.seed)
            .__truediv__(n_rows)
            .alias(noise_col)
        )

    def fit(self, X: nwt.IntoDataFrame, y: Any, **fit_kwargs):
        """Fit from a DataFrame of features and a target.

        Parameters
        ----------
        X : nwt.IntoDataFrame
            A DataFrame of features
        y : Any
            The target
        fit_kwargs : dict[str, Any]
            Any kwargs to pass to the estimator's `.fit()` method

        Returns
        -------
        Self
        """
        X_nw = nw.from_native(X, eager_only=True).pipe(self._add_noise)

        if "eval_set" in fit_kwargs:
            fit_kwargs["eval_set"] = [
                (
                    nw.from_native(x_val, eager_only=True)
                    .pipe(self._add_noise)
                    .to_native(),
                    y_val,
                )
                for x_val, y_val in fit_kwargs["eval_set"]
            ]

        X_train = X_nw.to_native()
        est = self.unfit_estimator.fit(X_train, y, **fit_kwargs)

        state = NFEState(estimator=est, X=X_train, y=y)

        nfe_features = (
            pl.LazyFrame(
                {"feature": X_train.columns, "importance": self.importance(state)}
            )
            .with_columns(pl.col("importance").abs())
            .filter(
                pl.col("importance").gt(
                    pl.col("importance").filter(pl.col("feature").eq(self._NOISE_COL))
                )
            )
            .collect()["feature"]
            .sort()
            .to_list()
        )

        self.selected_features_ = nfe_features

        return self

    def transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        """Reduce X to the selected features.

        Parameters
        ----------
        X : nwt.IntoFrameT

        Returns
        -------
        nwt.IntoFrameT
        """
        return nw.from_native(X).select(self.selected_features_).to_native()

    def fit_transform(
        self, X: nwt.IntoDataFrameT, y: Any, **fit_kwargs
    ) -> nwt.IntoDataFrameT:
        """Equivalent to calling `.fit(X, y, **fit_kwargs).transform(X)`.

        Parameters
        ----------
        X : nwt.IntoDataFrameT
        y : Any

        Returns
        -------
        nwt.IntoDataFrameT
        """
        return self.fit(X, y, **fit_kwargs).transform(X)

    def save(self, path: str | Path):
        """Saves the fit selector.

        Parameters
        ----------
        path : str | Path
            The file to save to

        Returns
        -------
        Self
        """
        _write_list(self.selected_features_, path)

        return self

    @staticmethod
    def load(path: str | Path):
        """Loads in an already fit selector.

        Parameters
        ----------
        path : str | Path
            The file to load from

        Returns
        -------
        Self
        """
        x = NFE(estimator=None)
        x.selected_features_ = _read_list(path)

        return x


class CFE:
    """Performs correlation feature elimination. The algorithm is as follows:
    Given a correlation matrix, identify all the non-identity pairs that are
    >= `threshold`. Compute the number of times each feature is highly correlated
    with another feature. Drop the feature with the highest count. Re-compute these
    counts and drop until we are left with a set of features that satisfy our
    threshold condition.

    Parameters
    ----------
    threshold : float, optional
        Drop features that have correlations >= `threshold`, by default 0.99
    seed : Optional[int], optional
        The seed, by default 208

    Attributes
    ----------
    threshold : float
    seed : Optional[int]
    selected_features_ : list[str]
        The selected features sorted alphabetically, available only after fitting
    """

    def __init__(self, threshold: float = 0.99, seed: Optional[int] = 208):
        self.threshold = threshold
        self.seed = seed

    @staticmethod
    def _find_drop(corr_mat: nw.DataFrame, seed: Optional[int]) -> tuple[str, int]:
        c1 = "c1"
        c2 = "c2"

        c1_counts = corr_mat.group_by(c1).agg(nw.len().alias("count_c1"))
        c2_counts = corr_mat.group_by(c2).agg(nw.len().alias("count_c2"))

        counts = (
            c1_counts.join(c2_counts, left_on=c1, right_on=c2, how="full")
            .with_columns(
                nw.coalesce(c1, c2).alias("feature"),
                nw.sum_horizontal("count_c1", "count_c2").alias("count"),
            )
            .select("feature", "count")
            .filter(nw.col("count").__eq__(nw.col("count").max()))
            # We need to sort by "feature" because the order after the join is not
            # always the same, making multiple runs even with the same seed not
            # reproducible without the sort.
            .sort("feature")
            # We could take the first or last, but let's sample so that we don't
            # introduce bias based on the alphabetical order.
            .sample(1, seed=seed)
        )

        return (counts["feature"].item(), counts["count"].item())

    def fit_from_correlation_matrix(
        self, corr_mat: nwt.IntoFrame, index: str = "", transform: bool = True
    ):
        """Fit directly from a correlation matrix.

        Parameters
        ----------
        corr_mat : nwt.IntoFrame
            A correlation matrix
        index : str, optional
            The column identifying the features, by default ""
        transform : bool, optional
            Whether to transfrom the correlation matrix to long form. A wide form
            correlation matrix has columns that are features and an "index" that lists
            the features. If False, the correlation matrix must already be in long form
            with at least 3 columns, "c1", "c2", and "correlation" , by default True

        Returns
        -------
        Self
        """
        c1 = "c1"
        c2 = "c2"
        cm_nw = nw.from_native(corr_mat).lazy()

        if transform:
            cm_nw = cm_nw.unpivot(index=index).rename(
                {index: c1, "variable": c2, "value": "correlation"}
            )

        features = (
            nw.concat(
                [
                    cm_nw.select(c1).rename({"c1": "x"}),
                    cm_nw.select(c2).rename({"c2": "x"}),
                ],
                how="vertical",
            )
            .unique()
            .collect()["x"]
            .to_list()
        )

        cm_nw = (
            cm_nw.with_columns(nw.col("correlation").abs())
            .filter(
                nw.col(c1).__ne__(nw.col(c2)),
                nw.col("correlation").is_null().__invert__(),
                nw.col("correlation").is_nan().__invert__(),
                nw.col("correlation").__ge__(self.threshold),
            )
            .collect()
        )

        drop_list = []
        i = 0
        while cm_nw.shape[0] > 0:
            to_drop, count = self._find_drop(cm_nw, self.seed)

            logger.info(
                f"Iteration {i}: Dropping {to_drop}, correlated with {count} other features"
            )

            cm_nw = cm_nw.filter(
                nw.col(c1)
                .__eq__(to_drop)
                .__or__(nw.col(c2).__eq__(to_drop))
                .__invert__()
            )

            drop_list.append(to_drop)
            i += 1

        self.selected_features_ = sorted(list(set(features) - set(drop_list)))

        return self

    def fit(self, X: nwt.IntoFrame):
        """Fit from a DataFrame of features.

        Parameters
        ----------
        X : nwt.IntoFrame

        Returns
        -------
        Self
        """
        corr_mat = correlation_matrix(X)

        self.fit_from_correlation_matrix(corr_mat)

        return self

    def transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        """Reduce X to the selected features.

        Parameters
        ----------
        X : nwt.IntoFrameT

        Returns
        -------
        nwt.IntoFrameT
        """
        return nw.from_native(X).select(self.selected_features_).to_native()

    def fit_transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        """Equivalent to calling `.fit(X).transform(X)`.

        Parameters
        ----------
        X : nwt.IntoFrameT

        Returns
        -------
        nwt.IntoFrameT
        """
        return self.fit(X).transform(X)

    def save(self, path: str | Path):
        """Saves the fit selector.

        Parameters
        ----------
        path : str | Path
            The file to save to

        Returns
        -------
        Self
        """
        _write_list(self.selected_features_, path)

        return self

    @staticmethod
    def load(path: str | Path):
        """Loads in an already fit selector.

        Parameters
        ----------
        path : str | Path
            The file to load from

        Returns
        -------
        Self
        """
        x = CFE()
        x.selected_features_ = _read_list(path)

        return x
