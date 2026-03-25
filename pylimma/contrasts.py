"""Contrast operations.

Mirrors R limma ``makeContrasts()`` and ``contrasts.fit()``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy.linalg import cholesky


def make_contrasts(coef_names: list[str], **contrasts: str) -> np.ndarray:
    """Build a contrast matrix from symbolic contrast expressions.

    Parameters
    ----------
    coef_names:
        Ordered list of coefficient names matching the columns of the fit
        object (e.g. ``["Intercept", "groupB", "groupC"]``).
    **contrasts:
        Named contrast expressions referencing elements of ``coef_names``.
        For example ``AvsB="groupB - groupA"`` or ``BvsC="groupB - groupC"``.

    Returns
    -------
    np.ndarray of shape (n_coef, n_contrasts) with column names stored as
    ``result.contrast_names`` attribute.

    Examples
    --------
    >>> C = make_contrasts(["Intercept", "groupB", "groupC"],
    ...                    BvsA="groupB", CvsA="groupC", BvsC="groupB - groupC")
    """
    n_coef = len(coef_names)
    contrast_names = list(contrasts.keys())
    n_cont = len(contrast_names)
    C = np.zeros((n_coef, n_cont))
    name_to_idx = {name: i for i, name in enumerate(coef_names)}

    for j, (cont_name, expr) in enumerate(contrasts.items()):
        C[:, j] = _parse_contrast_expr(expr, name_to_idx, n_coef)

    return C, contrast_names


def contrasts_fit(fit: dict[str, Any], contrasts: np.ndarray) -> dict[str, Any]:
    """Re-express a fitted model in terms of contrasts.

    Transforms ``fit['coefficients']`` and ``fit['stdev_unscaled']`` so that
    each column represents a contrast rather than an original coefficient.

    Parameters
    ----------
    fit:
        Output of :func:`~pylimma.lmfit.lm_fit` (or after a previous
        ``contrasts_fit`` call).
    contrasts:
        Contrast matrix, shape (n_coef, n_contrasts).  Rows correspond to
        the coefficients in *fit*; columns to the contrasts of interest.
        Can also be a ``(matrix, names)`` tuple as returned by
        :func:`make_contrasts`.

    Returns
    -------
    New fit dict with updated ``coefficients``, ``stdev_unscaled``, and
    ``cov_coefficients``.  Any existing test statistics (``t``, ``p_value``,
    ``lods``, ``F``, ``F_p_value``) are removed so that :func:`~pylimma.ebayes.ebayes`
    must be re-run.
    """
    import copy

    fit = copy.copy(fit)  # shallow copy — don't mutate caller's dict

    # Accept (matrix, names) tuple from make_contrasts
    contrast_names: list[str] | None = None
    if isinstance(contrasts, tuple):
        contrasts, contrast_names = contrasts

    contrasts = np.asarray(contrasts, dtype=float)
    if contrasts.ndim == 1:
        contrasts = contrasts[:, None]

    coeff = fit["coefficients"]
    stdev = fit["stdev_unscaled"]
    cov_coef = fit.get("cov_coefficients")

    n_coef = coeff.shape[1]
    if contrasts.shape[0] != n_coef:
        raise ValueError(
            f"contrasts has {contrasts.shape[0]} rows but fit has {n_coef} coefficients"
        )

    # Remove any existing test statistics
    for key in ("t", "p_value", "lods", "F", "F_p_value"):
        fit.pop(key, None)

    # Store contrast matrix
    fit["contrasts"] = contrasts

    # New coefficients:  beta_new = beta_old @ C
    fit["coefficients"] = coeff @ contrasts

    # Correlation matrix of original coefficients
    if cov_coef is None:
        # Assume orthogonal
        orthog = True
        var_coef = np.nanmean(stdev**2, axis=0)
        cov_coef = np.diag(var_coef)
    else:
        cov_coef = np.asarray(cov_coef, dtype=float)
        # Replace NaN with 0 for correlation computation
        cov_nan = np.isnan(cov_coef)
        cov_coef_clean = np.where(cov_nan, 0.0, cov_coef)
        # Correlation matrix
        d = np.sqrt(np.diag(cov_coef_clean))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov_coef_clean / np.outer(d, d)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        # Check orthogonality
        lower = corr[np.tril_indices_from(corr, k=-1)]
        orthog = np.all(np.abs(lower) < 1e-12)
        cov_coef = cov_coef_clean

    # New cov_coefficients:  C' cov_coef C  (scaled by σ² later by eBayes)
    # Use Cholesky so we propagate numerical precision correctly
    # cov_coef may have NaN rows/cols for non-estimable coefs — replace with 0
    cov_clean = np.where(np.isnan(cov_coef), 0.0, cov_coef)
    try:
        R_chol = cholesky(cov_clean, lower=False)  # upper triangular
        RC = R_chol @ contrasts
        fit["cov_coefficients"] = RC.T @ RC
    except Exception:
        fit["cov_coefficients"] = contrasts.T @ cov_clean @ contrasts

    # New stdev_unscaled
    if orthog:
        # Simple: sqrt( stdev^2 @ C^2 )
        fit["stdev_unscaled"] = np.sqrt(stdev**2 @ contrasts**2)
    else:
        # General: for each gene i, U_i = || R_corr * diag(stdev_i) * C ||_col
        n_genes = stdev.shape[0]
        n_cont = contrasts.shape[1]
        U = np.ones((n_genes, n_cont))
        d = np.sqrt(np.diag(cov_clean))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_mat = cov_clean / np.outer(d, d)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        np.fill_diagonal(corr_mat, 1.0)
        try:
            R_corr = cholesky(corr_mat, lower=False)
        except Exception:
            R_corr = np.eye(n_coef)
        for i in range(n_genes):
            s_i = stdev[i]
            if np.any(np.isnan(s_i)):
                # Replace NA stdevs temporarily with large value
                s_i = np.where(np.isnan(s_i), 1e30, s_i)
            RUC = R_corr @ (s_i[:, None] * contrasts)
            U[i] = np.sqrt(np.sum(RUC**2, axis=0))
        fit["stdev_unscaled"] = U

    # Restore NaN for non-estimable contrasts (stdev > 1e20 sentinel)
    large = fit["stdev_unscaled"] > 1e20
    fit["coefficients"][large] = np.nan
    fit["stdev_unscaled"][large] = np.nan

    return fit


# ---------------------------------------------------------------------------
# Internal: parse a simple arithmetic expression over coefficient names
# ---------------------------------------------------------------------------

def _parse_contrast_expr(
    expr: str, name_to_idx: dict[str, int], n_coef: int
) -> np.ndarray:
    """Parse an expression like "groupB - groupA" into a contrast vector."""
    vec = np.zeros(n_coef)

    # Tokenise: split on + and - while keeping the sign
    expr = expr.strip()
    # Normalise to have explicit leading +
    if not expr.startswith(("+", "-")):
        expr = "+" + expr
    # Split into tokens like ['+groupB', '-groupA']
    tokens = re.findall(r"[+\-][^+\-]+", expr)

    for token in tokens:
        token = token.strip()
        sign = 1.0 if token[0] == "+" else -1.0
        term = token[1:].strip()

        # Optional coefficient like "2*groupB"
        m = re.match(r"^([\d.]+)\s*\*\s*(.+)$", term)
        if m:
            scale = float(m.group(1))
            name = m.group(2).strip()
        else:
            scale = 1.0
            name = term

        if name not in name_to_idx:
            raise ValueError(
                f"Contrast references '{name}' which is not in coef_names: "
                f"{list(name_to_idx)}"
            )
        vec[name_to_idx[name]] += sign * scale

    return vec
