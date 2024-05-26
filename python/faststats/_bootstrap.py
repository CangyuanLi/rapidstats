import dataclasses
from typing import Optional

import polars as pl
import scipy.stats
from polars.series.series import ArrayLike

from ._rustystats import _bootstrap_confusion_matrix

# from ._utils import _run_concurrent


@dataclasses.dataclass
class BootstrappedConfusionMatrix:
    tn: tuple[float, float, float]
    fp: tuple[float, float, float]
    fn: tuple[float, float, float]
    tp: tuple[float, float, float]
    tpr: tuple[float, float, float]
    fpr: tuple[float, float, float]
    fnr: tuple[float, float, float]
    tnr: tuple[float, float, float]
    prevalence: tuple[float, float, float]
    prevalence_threshold: tuple[float, float, float]
    informedness: tuple[float, float, float]
    precision: tuple[float, float, float]
    false_omission_rate: tuple[float, float, float]
    plr: tuple[float, float, float]
    nlr: tuple[float, float, float]
    acc: tuple[float, float, float]
    balanced_accuracy: tuple[float, float, float]
    f1: tuple[float, float, float]
    folkes_mallows_index: tuple[float, float, float]
    mcc: tuple[float, float, float]
    threat_score: tuple[float, float, float]
    markedness: tuple[float, float, float]
    fdr: tuple[float, float, float]
    npv: tuple[float, float, float]
    dor: tuple[float, float, float]


# def _run_bootstrap():
#     pass


# def bootstrap(
#     data,
#     stat_func,
#     iterations: int = 1_000,
#     confidence: float = 0.95,
#     seed: int = None,
#     **exeuctor_kwargs,
# ):
#     for i in range(iterations):
#         data.sample()
#     _run_concurrent(stat_func)


class Bootstrap:
    def __init__(
        self,
        iterations: int = 1_000,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        self.iterations = iterations
        self.z = scipy.stats.norm.ppf(confidence)
        self.seed = seed

    def confusion_matrix(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> BootstrappedConfusionMatrix:
        df = pl.DataFrame({"y_true": y_true, "y_pred": y_pred}).with_columns(
            pl.col("y_true", "y_pred").cast(pl.Boolean)
        )

        return BootstrappedConfusionMatrix(
            *_bootstrap_confusion_matrix(df, self.iterations, self.z, self.seed)
        )
