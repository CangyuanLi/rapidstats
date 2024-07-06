from ._rustystats import _norm_ppf


class norm:
    @staticmethod
    def ppf(q: float) -> float:
        return _norm_ppf(q)
