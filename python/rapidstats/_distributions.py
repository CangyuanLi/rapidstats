from ._rustystats import _norm_cdf, _norm_ppf


class norm:
    """Functions for working with a normal continuous random variable."""

    @staticmethod
    def ppf(q: float) -> float:
        r"""The percent point function. Also called the quantile, percentile, inverse
        CDF, or inverse distribution function. Computes the value of a random variable
        such that its probability is \( \leq q \). If `q` is 0, it returns negative
        infinity, if `q` is 1, it returns infinity. Any number outside of [0, 1] will
        result in NaN.

        Parameters
        ----------
        q : float
            Probability value

        Returns
        -------
        float
            Likelihood a random variable is realized in the range at or below `q` for
            the normal distribution.
        """
        return _norm_ppf(q)

    @staticmethod
    def cdf(x: float) -> float:
        r"""The cumulative distribution function.

        Parameters
        ----------
        x : float

        Returns
        -------
        float
            The probability a random variable will take a value \( \leq x \)
        """
        return _norm_cdf(x)
