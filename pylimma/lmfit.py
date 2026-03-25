"""Gene-wise linear model fitting.

Mirrors R limma ``lmFit`` / ``lm.series``.

The core idea: fit the same linear model  y_g = X β_g + ε_g  independently
for every gene g, where X is the shared design matrix (samples × coefficients)
and y_g is the vector of expression values across samples.

When there are no missing values and no per-gene observation weights, a single
QR decomposition of X suffices for all genes (fast path).  When missing values
or per-gene weights are present we fall back to a gene-by-gene loop.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.linalg import qr as scipy_qr, solve_triangular

from ._utils import is_fullrank, non_estimable


def lm_fit(
    E: np.ndarray,
    design: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a linear model gene-wise.

    Parameters
    ----------
    E:
        Expression matrix, shape (n_genes, n_samples).  Values should be on a
        log scale (log2 CPM, log2 intensity, etc.).
    design:
        Design matrix, shape (n_samples, n_coef).  Defaults to a single
        intercept column (grand mean model).
    weights:
        Observation weights, broadcastable to (n_genes, n_samples).  Rows
        with weight <= 0 are treated as missing.

    Returns
    -------
    dict with keys:

    - ``coefficients``    – (n_genes, n_coef) estimated β
    - ``stdev_unscaled``  – (n_genes, n_coef) unscaled standard deviations,
                            i.e. sqrt(diag((X'X)^{-1})) per gene
    - ``sigma``           – (n_genes,) residual standard deviation
    - ``df_residual``     – (n_genes,) residual degrees of freedom
    - ``cov_coefficients``– (n_coef, n_coef) covariance of coefficients
                            (unscaled, i.e. times σ²)
    - ``pivot``           – column pivot from QR decomposition
    - ``rank``            – column rank of design matrix
    - ``Amean``           – (n_genes,) row means of E (average log-expression)
    - ``design``          – the design matrix used
    """
    E = np.asarray(E, dtype=float)
    n_genes, n_samples = E.shape

    if design is None:
        design = np.ones((n_samples, 1))
    design = np.asarray(design, dtype=float)

    if design.shape[0] != n_samples:
        raise ValueError(
            f"design has {design.shape[0]} rows but E has {n_samples} columns"
        )

    coef_names = _coef_names(design, design.shape[1])

    ne = non_estimable(design, coef_names)
    if ne:
        warnings.warn(f"Coefficients not estimable: {' '.join(ne)}")

    n_coef = design.shape[1]

    # Handle weights
    W: np.ndarray | None = None
    if weights is not None:
        W = np.broadcast_to(np.asarray(weights, dtype=float), (n_genes, n_samples)).copy()
        W[W <= 0] = np.nan
        E = E.copy()
        E[~np.isfinite(W)] = np.nan

    Amean = np.nanmean(E, axis=1)

    # Decide fast path vs gene-by-gene
    has_na = not np.all(np.isfinite(E))
    has_gene_weights = W is not None and not _is_array_weights(W)

    if not has_na and not has_gene_weights:
        result = _lm_series_fast(E, design, W, coef_names)
    else:
        result = _lm_series_loop(E, design, W, coef_names)

    result["Amean"] = Amean
    result["design"] = design
    return result


# ---------------------------------------------------------------------------
# Fast path: single QR for all genes
# ---------------------------------------------------------------------------

def _lm_series_fast(
    E: np.ndarray,
    design: np.ndarray,
    W: np.ndarray | None,
    coef_names: list[str],
) -> dict[str, Any]:
    n_genes, n_samples = E.shape
    n_coef = design.shape[1]

    if W is not None:
        # All genes share the same weight vector (array weights)
        w = W[0]
        sqrt_w = np.sqrt(w)
        Xw = design * sqrt_w[:, None]
        Yw = E * sqrt_w[None, :]
    else:
        Xw = design
        Yw = E

    # QR decomposition of (weighted) design matrix
    Q, R, pivot = scipy_qr(Xw, pivoting=True)
    rank = int(np.linalg.matrix_rank(Xw))

    # Coefficients: solve R * beta = Q' * Y  for all genes simultaneously
    # Yw shape: (n_genes, n_samples); Q shape: (n_samples, n_samples)
    QtY = Q.T @ Yw.T  # (n_samples, n_genes)
    beta_pivot = solve_triangular(R[:rank, :rank], QtY[:rank, :])  # (rank, n_genes)

    # Place estimates back in original column order
    coefficients = np.full((n_genes, n_coef), np.nan)
    coefficients[:, pivot[:rank]] = beta_pivot.T

    # Residuals and sigma
    df_residual = n_samples - rank
    if df_residual > 0:
        residuals = QtY[rank:, :]  # (n_samples-rank, n_genes)
        sigma = np.sqrt(np.mean(residuals**2, axis=0))
    else:
        sigma = np.full(n_genes, np.nan)

    # Covariance of coefficients: (R'R)^{-1}
    R_est = R[:rank, :rank]
    cov_coef_full = np.linalg.inv(R_est.T @ R_est)
    # Re-order to original column order
    inv_pivot = np.argsort(pivot[:rank])
    cov_coef = cov_coef_full[np.ix_(inv_pivot, inv_pivot)]

    # stdev_unscaled: sqrt(diag(cov_coef)) broadcast to all genes
    stdev_base = np.sqrt(np.diag(cov_coef))
    stdev_unscaled = np.tile(stdev_base, (n_genes, 1))
    # NaN where coefficient not estimable
    stdev_unscaled[:, pivot[rank:]] = np.nan

    # Build full cov_coef with NaN rows/cols for non-estimable
    cov_full = np.full((n_coef, n_coef), np.nan)
    cov_full[np.ix_(pivot[:rank], pivot[:rank])] = cov_coef

    return dict(
        coefficients=coefficients,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        df_residual=np.full(n_genes, float(df_residual)),
        cov_coefficients=cov_full,
        pivot=pivot,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Gene-by-gene loop (NAs or per-gene weights)
# ---------------------------------------------------------------------------

def _lm_series_loop(
    E: np.ndarray,
    design: np.ndarray,
    W: np.ndarray | None,
    coef_names: list[str],
) -> dict[str, Any]:
    n_genes, n_samples = E.shape
    n_coef = design.shape[1]

    coefficients = np.full((n_genes, n_coef), np.nan)
    stdev_unscaled = np.full((n_genes, n_coef), np.nan)
    sigma = np.full(n_genes, np.nan)
    df_residual = np.zeros(n_genes)

    for i in range(n_genes):
        y = E[i]
        obs = np.isfinite(y)
        if W is not None:
            w_i = W[i]
            obs = obs & np.isfinite(w_i) & (w_i > 0)
        if obs.sum() == 0:
            continue

        X_i = design[obs]
        y_i = y[obs]

        if W is not None:
            w_i_obs = W[i, obs]
            sqrt_w = np.sqrt(w_i_obs)
            X_i = X_i * sqrt_w[:, None]
            y_i = y_i * sqrt_w

        Q_i, R_i, pivot_i = scipy_qr(X_i, pivoting=True)
        rank_i = int(np.linalg.matrix_rank(X_i))
        n_obs = obs.sum()

        QtY_i = Q_i.T @ y_i
        beta_pivot_i = solve_triangular(R_i[:rank_i, :rank_i], QtY_i[:rank_i])

        coefficients[i, pivot_i[:rank_i]] = beta_pivot_i

        df_i = n_obs - rank_i
        df_residual[i] = df_i
        if df_i > 0:
            resid_i = QtY_i[rank_i:]
            sigma[i] = np.sqrt(np.mean(resid_i**2))

        R_est_i = R_i[:rank_i, :rank_i]
        cov_i = np.linalg.inv(R_est_i.T @ R_est_i)
        inv_piv_i = np.argsort(pivot_i[:rank_i])
        cov_i = cov_i[np.ix_(inv_piv_i, inv_piv_i)]
        stdev_unscaled[i, pivot_i[:rank_i]] = np.sqrt(np.diag(cov_i))

    # Global cov_coef from design (without NA/weights, for reference)
    Q, R, pivot = scipy_qr(design, pivoting=True)
    rank = int(np.linalg.matrix_rank(design))
    R_est = R[:rank, :rank]
    cov_coef_full = np.linalg.inv(R_est.T @ R_est)
    inv_pivot = np.argsort(pivot[:rank])
    cov_coef_r = cov_coef_full[np.ix_(inv_pivot, inv_pivot)]
    cov_full = np.full((n_coef, n_coef), np.nan)
    cov_full[np.ix_(pivot[:rank], pivot[:rank])] = cov_coef_r

    return dict(
        coefficients=coefficients,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        df_residual=df_residual,
        cov_coefficients=cov_full,
        pivot=pivot,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coef_names(design: np.ndarray, n_coef: int) -> list[str]:
    if hasattr(design, "columns"):
        return list(design.columns)
    return [f"x{i+1}" for i in range(n_coef)]


def _is_array_weights(W: np.ndarray) -> bool:
    """True if all rows of W are identical (array-level weights, not gene-level)."""
    return bool(np.all(W == W[0]))
