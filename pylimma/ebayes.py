"""Empirical Bayes moderation of gene-wise variances.

Mirrors R limma ``eBayes()``, ``squeezeVar()``, ``fitFDist()``, and the
B-statistic machinery (``tmixture.vector`` / ``tmixture.matrix``).

Key idea
--------
Gene-wise residual variances σ̂²_g are assumed to follow a scaled
F-distribution a priori.  Empirical Bayes estimation of the prior
degrees-of-freedom (df_prior) and prior variance (s2_prior) allows us to
form *posterior* (squeezed) variances:

    s2_post_g = (df_residual * σ̂²_g + df_prior * s2_prior)
                / (df_residual + df_prior)

Moderated t-statistics then use s2_post instead of σ̂²:

    t_mod_g = β̂_g / (stdev_unscaled_g * sqrt(s2_post_g))

with degrees of freedom  df_total = df_residual + df_prior.

Reference
---------
Smyth, G.K. (2004) Linear models and empirical Bayes methods for assessing
differential expression in microarray experiments.  Statistical Applications
in Genetics and Molecular Biology 3, Article 3.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import special as sp_special
from scipy import stats as sp_stats

from ._utils import is_fullrank


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ebayes(
    fit: dict[str, Any],
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
    trend: bool = False,
    robust: bool = False,
) -> dict[str, Any]:
    """Empirical Bayes moderation of linear model fits.

    Parameters
    ----------
    fit:
        Output of :func:`~pylimma.lmfit.lm_fit` (or :func:`~pylimma.contrasts.contrasts_fit`).
    proportion:
        Prior probability that a gene is differentially expressed.  Used for
        computing B-statistics.
    stdev_coef_lim:
        Limits for the unscaled standard deviation of a DE log-fold-change
        used in B-statistic computation.
    trend:
        If ``True``, allow the prior variance to depend on average
        expression (``fit['Amean']``).  Currently only ``trend=False`` is
        supported; set to ``True`` raises ``NotImplementedError``.
    robust:
        If ``True`` use robust hyperparameter estimation (not yet
        implemented; raises ``NotImplementedError``).

    Returns
    -------
    A copy of *fit* augmented with:

    - ``s2_prior``   – estimated prior variance
    - ``df_prior``   – estimated prior degrees of freedom
    - ``s2_post``    – posterior (squeezed) variances, shape (n_genes,)
    - ``df_total``   – total degrees of freedom for moderated t, shape (n_genes,)
    - ``t``          – moderated t-statistics, shape (n_genes, n_coef)
    - ``p_value``    – two-tailed p-values, shape (n_genes, n_coef)
    - ``lods``       – B-statistics (log-odds of DE), shape (n_genes, n_coef)
    - ``F``          – moderated F-statistics (if design full-rank), shape (n_genes,)
    - ``F_p_value``  – p-values for F-statistics, shape (n_genes,)
    """
    if trend:
        raise NotImplementedError("trend=True is not yet implemented in pylimma")
    if robust:
        raise NotImplementedError("robust=True is not yet implemented in pylimma")

    import copy
    fit = copy.copy(fit)

    coeff = fit["coefficients"]          # (n_genes, n_coef)
    stdev_u = fit["stdev_unscaled"]      # (n_genes, n_coef)
    sigma = fit["sigma"]                 # (n_genes,)
    df_res = fit["df_residual"]          # (n_genes,)

    if coeff is None or stdev_u is None or sigma is None or df_res is None:
        raise ValueError("fit is missing required fields from lm_fit")
    if np.all(df_res == 0):
        raise ValueError("No residual degrees of freedom in linear model fits")
    if not np.any(np.isfinite(sigma)):
        raise ValueError("No finite residual standard deviations")

    # Squeeze variances
    sq = squeeze_var(sigma**2, df_res)
    s2_prior = sq["var_prior"]
    df_prior = sq["df_prior"]
    s2_post = sq["var_post"]

    fit["s2_prior"] = s2_prior
    fit["df_prior"] = df_prior
    fit["s2_post"] = s2_post

    # Moderated t-statistics
    # t_gk = beta_gk / (stdev_unscaled_gk * sqrt(s2_post_g))
    t = coeff / (stdev_u * np.sqrt(s2_post)[:, None])
    df_total = df_res + df_prior
    df_pooled = np.nansum(df_res)
    df_total = np.minimum(df_total, df_pooled)
    fit["df_total"] = df_total

    fit["t"] = t
    fit["p_value"] = 2.0 * sp_stats.t.sf(np.abs(t), df=df_total[:, None])

    # B-statistics
    var_prior_lim = np.array(stdev_coef_lim) ** 2 / np.median(s2_prior)
    var_prior = _tmixture_matrix(t, stdev_u, df_total, proportion, var_prior_lim)
    if np.any(np.isnan(var_prior)):
        var_prior[np.isnan(var_prior)] = 1.0 / np.median(s2_prior)
        warnings.warn("Estimation of var_prior failed — set to default value")
    fit["var_prior"] = var_prior

    # r_gk = (stdev_unscaled_gk^2 + var_prior_k) / stdev_unscaled_gk^2
    r = (stdev_u**2 + var_prior[None, :]) / stdev_u**2
    t2 = t**2
    kernel = (1.0 + df_total[:, None]) / 2.0 * np.log(
        (t2 + df_total[:, None]) / (t2 / r + df_total[:, None])
    )
    lods = np.log(proportion / (1.0 - proportion)) - np.log(r) / 2.0 + kernel
    fit["lods"] = lods

    # Moderated F-statistics across all non-intercept coefficients
    design = fit.get("design")
    if design is not None and is_fullrank(design):
        fit["F"], fit["F_p_value"] = _moderated_f(fit)

    return fit


# ---------------------------------------------------------------------------
# squeeze_var
# ---------------------------------------------------------------------------


def squeeze_var(
    var: np.ndarray,
    df: np.ndarray | float,
    covariate: np.ndarray | None = None,
) -> dict[str, Any]:
    """Empirical Bayes posterior variances.

    Mirrors R limma ``squeezeVar()``.

    Parameters
    ----------
    var:
        Sample variances (σ̂²_g), shape (n_genes,).
    df:
        Degrees of freedom for each variance (scalar or array of length n_genes).
    covariate:
        Optional covariate for a mean-variance trend (not yet implemented).

    Returns
    -------
    dict with keys ``df_prior``, ``var_prior``, ``var_post``.
    """
    var = np.asarray(var, dtype=float)
    df = np.broadcast_to(np.asarray(df, dtype=float), var.shape).copy()
    n = len(var)

    if n == 0:
        raise ValueError("var is empty")
    if n < 3:
        return {"var_post": var.copy(), "var_prior": var.copy(), "df_prior": 0.0}

    # Guard against inf/nan variance when df==0
    var[df == 0] = 0.0

    fit_f = fit_f_dist(var, df, covariate=covariate)
    df_prior = fit_f["df2"]
    if np.isnan(df_prior):
        raise RuntimeError("Could not estimate prior df")

    var_post = _squeeze_var_inner(var, df, fit_f["scale"], df_prior)
    return {"df_prior": df_prior, "var_prior": fit_f["scale"], "var_post": var_post}


def _squeeze_var_inner(
    var: np.ndarray,
    df: np.ndarray,
    var_prior: np.ndarray | float,
    df_prior: float,
) -> np.ndarray:
    """Compute posterior variances given hyperparameters."""
    if np.isfinite(df_prior):
        return (df * var + df_prior * var_prior) / (df + df_prior)
    # df_prior is infinite → posterior = prior
    n = len(var)
    if np.isscalar(var_prior) or len(np.atleast_1d(var_prior)) == 1:
        return np.full(n, float(var_prior))
    return np.asarray(var_prior, dtype=float).copy()


# ---------------------------------------------------------------------------
# fit_f_dist — moment estimation of a scaled F distribution
# ---------------------------------------------------------------------------


def fit_f_dist(
    x: np.ndarray,
    df1: np.ndarray | float,
    covariate: np.ndarray | None = None,
) -> dict[str, Any]:
    """Moment estimation of the parameters of a scaled F-distribution.

    Mirrors R limma ``fitFDist()``.

    Estimates the scale factor *s0²* and denominator degrees of freedom *df2*
    such that  x_g / s0²  ~  F(df1, df2).

    Parameters
    ----------
    x:
        Observed variances (numerator of F-statistics).
    df1:
        Known numerator degrees of freedom (scalar or array matching *x*).
    covariate:
        Not yet implemented; pass ``None``.

    Returns
    -------
    dict with keys ``scale`` (s0²) and ``df2``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n == 0:
        return {"scale": np.nan, "df2": np.nan}
    if n == 1:
        return {"scale": float(x[0]), "df2": 0.0}

    df1 = np.broadcast_to(np.asarray(df1, dtype=float), x.shape).copy()

    # Remove non-finite, negative, or zero-df entries
    ok = np.isfinite(df1) & (df1 > 1e-15) & np.isfinite(x) & (x > -1e-15)
    nok = ok.sum()
    if nok <= 1:
        return {"scale": float(x[ok][0]) if nok == 1 else np.nan, "df2": 0.0}

    x_ok = np.maximum(x[ok], 0.0)
    df1_ok = df1[ok]

    # Offset zeros away from zero (mirroring R)
    m = np.median(x_ok)
    if m == 0:
        warnings.warn(
            "More than half of residual variances are exactly zero: eBayes unreliable"
        )
        m = 1.0
    elif np.any(x_ok == 0):
        warnings.warn(
            "Zero sample variances detected, have been offset away from zero"
        )
    x_ok = np.maximum(x_ok, 1e-5 * m)

    # Work on log scale
    z = np.log(x_ok)
    # e = log(x) + log(Γ(df1/2)) - log(Γ(df1/2 - 1/2))  — "log modified digamma"
    # R calls this  z + logmdigamma(df1/2)  where logmdigamma(a) = log(a-1/2) - digamma(a)
    # Actually in limma: e = z + logmdigamma(df1/2)
    # logmdigamma(a) = log( E[log(F(df1,df2))] adjustment) = log(2*a) - digamma(2*a) ?
    # Looking at R source: e <- z + logmdigamma(df1/2)
    # and logmdigamma is defined as log(x) - digamma(x) where x=df1/2  (see utility.R)
    e = z + _log_mdigamma(df1_ok / 2.0)

    if covariate is not None:
        raise NotImplementedError("covariate trend not yet implemented")

    emean = np.mean(e)
    evar = np.sum((e - emean) ** 2) / (nok - 1)

    # Trigamma correction
    evar = evar - np.mean(sp_special.polygamma(1, df1_ok / 2.0))  # mean trigamma

    if evar > 0:
        df2 = 2.0 * _trigamma_inverse(evar)
        s20 = np.exp(emean - _log_mdigamma(df2 / 2.0))
    else:
        df2 = np.inf
        s20 = np.mean(x_ok)

    return {"scale": float(s20), "df2": float(df2)}


# ---------------------------------------------------------------------------
# Helper: log modified digamma  logmdigamma(x) = log(x) - digamma(x)
# ---------------------------------------------------------------------------

def _log_mdigamma(x: np.ndarray) -> np.ndarray:
    """log(x) - digamma(x).

    This is the function ``logmdigamma`` from R limma utility.R.
    For large x: log(x) - digamma(x) ≈ 1/(2x) + 1/(12x²) - ...
    """
    return np.log(x) - sp_special.digamma(x)


# ---------------------------------------------------------------------------
# trigammaInverse — Newton's method
# ---------------------------------------------------------------------------


def _trigamma_inverse(x: np.ndarray | float) -> np.ndarray | float:
    """Solve  trigamma(y) = x  for y.

    Mirrors R limma ``trigammaInverse()``.  Uses Newton's method on
    1/trigamma(y) (which is convex and nearly linear), so iteration
    converges monotonically.
    """
    scalar = np.isscalar(x)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.where(x > 0, 0.5 + 1.0 / x, np.nan)

    for _ in range(50):
        tri = sp_special.polygamma(1, y)   # trigamma
        tetra = sp_special.polygamma(2, y)  # tetragamma
        dif = tri * (1.0 - tri / x) / tetra
        y += dif
        if np.nanmax(-dif / y) < 1e-8:
            break

    return float(y[0]) if scalar else y


# ---------------------------------------------------------------------------
# tmixture — prior variance estimation from mixture model
# ---------------------------------------------------------------------------


def _tmixture_vector(
    tstat: np.ndarray,
    stdev_unscaled: np.ndarray,
    df: np.ndarray,
    proportion: float,
    v0_lim: tuple[float, float] | None = None,
) -> float:
    """Estimate the prior variance of DE coefficients from a t-statistic mixture.

    Mirrors R limma ``tmixture.vector()``.
    """
    # Remove NAs
    ok = np.isfinite(tstat)
    tstat = tstat[ok]
    stdev_unscaled = stdev_unscaled[ok]
    df = df[ok]

    n_genes = len(tstat)
    ntarget = int(np.ceil(proportion / 2.0 * n_genes))
    if ntarget < 1:
        return np.nan

    p = max(ntarget / n_genes, proportion)

    tstat = np.abs(tstat)
    max_df = float(np.max(df))

    # Equalise df by converting tail probabilities
    need_conv = df < max_df
    if np.any(need_conv):
        tail_p = sp_stats.t.logsf(tstat[need_conv], df=df[need_conv])
        tstat[need_conv] = sp_stats.t.isf(np.exp(tail_p), df=max_df)
        df[need_conv] = max_df

    # Select top ntarget statistics
    order = np.argsort(tstat)[::-1][:ntarget]
    tstat = tstat[order]
    v1 = stdev_unscaled[order] ** 2

    # Compare to expected order statistics
    r = np.arange(1, ntarget + 1)
    p0 = 2.0 * sp_stats.t.sf(tstat, df=max_df)
    ptarget = ((r - 0.5) / n_genes - (1.0 - p) * p0) / p
    v0 = np.zeros(ntarget)
    pos = ptarget > p0
    if np.any(pos):
        qtarget = sp_stats.t.isf(ptarget[pos] / 2.0, df=max_df)
        v0[pos] = v1[pos] * ((tstat[pos] / qtarget) ** 2 - 1.0)

    if v0_lim is not None:
        v0 = np.clip(v0, v0_lim[0], v0_lim[1])

    return float(np.mean(v0))


def _tmixture_matrix(
    tstat: np.ndarray,
    stdev_unscaled: np.ndarray,
    df: np.ndarray,
    proportion: float,
    v0_lim: tuple[float, float] | None = None,
) -> np.ndarray:
    """Estimate prior variances for each coefficient column.

    Mirrors R limma ``tmixture.matrix()``.
    """
    n_coef = tstat.shape[1]
    v0 = np.zeros(n_coef)
    for j in range(n_coef):
        v0[j] = _tmixture_vector(
            tstat[:, j].copy(),
            stdev_unscaled[:, j].copy(),
            df.copy(),
            proportion,
            v0_lim,
        )
    return v0


# ---------------------------------------------------------------------------
# Moderated F-statistic
# ---------------------------------------------------------------------------


def _moderated_f(fit: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Compute moderated F-statistics across all non-intercept coefficients.

    Implements the same logic as R limma's ``classifyTestsF(..., fstat.only=TRUE)``.
    """
    coeff = fit["coefficients"]
    stdev_u = fit["stdev_unscaled"]
    s2_post = fit["s2_post"]
    df_total = fit["df_total"]
    design = fit.get("design")

    # Identify intercept column (all-ones design column)
    non_intercept: list[int] = list(range(coeff.shape[1]))
    if design is not None:
        for j in range(design.shape[1]):
            if np.all(design[:, j] == 1.0):
                if j < len(non_intercept):
                    non_intercept.remove(j)
                break

    if not non_intercept:
        return np.full(coeff.shape[0], np.nan), np.full(coeff.shape[0], np.nan)

    c = coeff[:, non_intercept]
    s = stdev_u[:, non_intercept]

    # Standardised coefficients
    with np.errstate(invalid="ignore", divide="ignore"):
        z = c / (s * np.sqrt(s2_post)[:, None])

    df1 = len(non_intercept)
    df2 = df_total  # shape (n_genes,)

    F_stat = np.nanmean(z**2, axis=1)
    F_pval = sp_stats.f.sf(F_stat, dfn=df1, dfd=df2)

    return F_stat, F_pval
