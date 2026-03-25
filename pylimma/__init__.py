"""
pylimma — Native Python implementation of limma's core differential expression pipeline.

Core workflow::

    fit = lm_fit(E, design)
    fit = contrasts_fit(fit, contrasts)   # optional
    fit = ebayes(fit)
    results = top_table(fit, coef=1)

Pingouin is used for multiple testing correction in top_table().
All other statistical operations use numpy/scipy.
"""

from .lmfit import lm_fit
from .contrasts import make_contrasts, contrasts_fit
from .ebayes import ebayes, squeeze_var, fit_f_dist
from .toptable import top_table

__all__ = [
    "lm_fit",
    "make_contrasts",
    "contrasts_fit",
    "ebayes",
    "squeeze_var",
    "fit_f_dist",
    "top_table",
]
