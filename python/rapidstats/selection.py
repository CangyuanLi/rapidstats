from __future__ import annotations

import copy
import inspect
import math
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, TypedDict

import narwhals.stable.v1 as nw
import narwhals.stable.v1.typing as nwt
import polars as pl
from tqdm.auto import tqdm

from .metrics import roc_auc


class Estimator(Protocol):
    def fit(self, X, y, eval_set, **kwargs): ...


def _copy(estimator: Estimator):
    if inspect.isclass(estimator):
        raise TypeError("Must pass in class instance, not class")

    if hasattr(estimator, "__sklearn_clone__"):
        return estimator.__sklearn_clone__()

    return copy.deepcopy(estimator)


class EarlyStoppingProtocol(Protocol):
    def stop(self) -> bool: ...

    def __call__(self, *args, **kwargs): ...


def _roc_auc(est, X, y) -> float:
    y_score = est.predict_proba(X)[:, 0]

    return roc_auc(y, y_score)


class EarlyStopping:
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
    estimator: Estimator
    X: Any
    y: Any
    eval_set: Optional[list[tuple[Any, Any]]]
    features: list[str]
    iteration: int


def _get_feature_importance(rfe_state: RFEState) -> pl.Series:
    est = rfe_state["estimator"]

    if hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
    elif hasattr(est, "coef_"):
        importances = est.coef_
    else:
        raise AttributeError("Could not find either `feature_importances_` or `coef_`.")

    if hasattr(importances, "ravel"):
        importances = importances.ravel()

    return importances


class RFE:
    def __init__(
        self,
        estimator: Estimator,
        n_features_to_select: float = 1,
        step: float = 1,
        importance: Callable[[RFEState], Iterable[float]] = _get_feature_importance,
        callbacks: Optional[Iterable[Callable[[RFEState]]]] = None,
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

                state = {
                    "estimator": est,
                    "X": X_loop,
                    "y": y,
                    "eval_set": fit_kwargs.get("eval_set", None),
                    "features": features,
                    "iteration": iteration,
                }

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

        self.estimator_ = est
        self.feature_names_in_ = features

        return self
