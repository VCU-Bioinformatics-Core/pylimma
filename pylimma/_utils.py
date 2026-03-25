"""Internal utilities mirroring limma's helper functions."""

import numpy as np


def is_fullrank(X: np.ndarray) -> bool:
    """Check whether a numeric matrix has full column rank.

    Mirrors R limma ``is.fullrank()``.  Uses the ratio of the largest to
    smallest eigenvalue of X'X; the matrix is considered full-rank when
    the ratio exceeds 1e-13.
    """
    X = np.asarray(X, dtype=float)
    eigenvalues = np.linalg.eigvalsh(X.T @ X)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return bool(eigenvalues[0] > 0 and abs(eigenvalues[-1] / eigenvalues[0]) > 1e-13)


def non_estimable(X: np.ndarray, col_names: list[str] | None = None) -> list[str] | None:
    """Return names of non-estimable (redundant) columns in a design matrix.

    Mirrors R limma ``nonEstimable()``.  Returns ``None`` if the matrix has
    full column rank, or a list of column names that are linearly dependent.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    if col_names is None:
        col_names = [str(i + 1) for i in range(p)]

    _, R, pivot = np.linalg.qr(X, mode="complete"), None, None
    # Use scipy for column-pivoted QR to get rank and pivot information
    from scipy.linalg import qr as scipy_qr
    _, _, pivot = scipy_qr(X, pivoting=True)
    rank = np.linalg.matrix_rank(X)

    if rank < p:
        redundant = [col_names[pivot[i]] for i in range(rank, p)]
        return redundant
    return None
