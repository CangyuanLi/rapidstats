from ._rustystats import _norm_cdf, _norm_ppf


class norm:
    @staticmethod
    def ppf(q: float) -> float:
        return _norm_ppf(q)

    @staticmethod
    def cdf(q: float) -> float:
        return _norm_cdf(q)
