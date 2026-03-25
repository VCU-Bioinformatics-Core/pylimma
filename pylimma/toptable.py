"""Summary table of top differentially expressed genes.

Mirrors R limma ``topTable()``.

Multiple testing correction is performed using :func:`pingouin.multicomp`,
which wraps ``statsmodels.stats.multitest.multipletests`` and supports
the same adjustment methods as R's ``p.adjust()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pingouin


# Mapping from limma/R style method names to pingouin/statsmodels names
_METHOD_MAP: dict[str, str] = {
    "BH": "fdr_bh",
    "fdr": "fdr_bh",
    "fdr_bh": "fdr_bh",
    "BY": "fdr_by",
    "fdr_by": "fdr_by",
    "bonferroni": "bonferroni",
    "Bonferroni": "bonferroni",
    "holm": "holm",
    "Holm": "holm",
    "hochberg": "fdr_bh",   # R's hochberg ≈ BH for most purposes; use BH
    "none": "none",
    "None": "none",
}


def top_table(
    fit: dict[str, Any],
    coef: int | str | None = None,
    number: int = 10,
    adjust_method: str = "BH",
    sort_by: str = "B",
    p_value: float = 1.0,
    lfc: float = 0.0,
    confint: bool = False,
    gene_names: list[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Extract a table of the top differentially expressed genes.

    Parameters
    ----------
    fit:
        Output of :func:`~pylimma.ebayes.ebayes` (must contain ``t``,
        ``p_value``, ``lods`` fields).
    coef:
        Which coefficient (column index or name) to report.  Defaults to the
        last non-intercept coefficient.  Pass a list to get an F-test table.
    number:
        Maximum number of genes to return.
    adjust_method:
        Multiple testing correction method passed to
        :func:`pingouin.multicomp`.  Accepts R-style names (``"BH"``,
        ``"bonferroni"``, ``"none"``) and pingouin/statsmodels names
        (``"fdr_bh"``, ``"holm"``, etc.).
    sort_by:
        Sort column: ``"B"`` (B-statistic), ``"P"`` (p-value),
        ``"logFC"``, ``"t"``, ``"AveExpr"``, or ``"none"``.
    p_value:
        Return only genes with adjusted p-value ≤ this threshold.
    lfc:
        Return only genes with |logFC| ≥ this threshold (log2 scale).
    confint:
        If ``True``, include 95 % confidence interval columns ``CI.L``
        and ``CI.R``.
    gene_names:
        Optional sequence of gene identifiers for the row labels.

    Returns
    -------
    pandas DataFrame, sorted by *sort_by*, with at most *number* rows.
    Columns: ``logFC``, ``AveExpr``, ``t``, ``P.Value``, ``adj.P.Val``,
    ``B`` (and ``CI.L``, ``CI.R`` if *confint* is True).
    """
    if fit.get("t") is None and fit.get("F") is None:
        raise ValueError("Need to run ebayes() first — fit has no t or F statistics")

    coeff = fit["coefficients"]   # (n_genes, n_coef)
    n_genes, n_coef = coeff.shape

    # Resolve gene names / row labels
    if gene_names is None:
        gene_names = fit.get("gene_names")
    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    else:
        gene_names = list(gene_names)

    # Resolve coef
    coef = _resolve_coef(fit, coef, n_coef)

    # Multi-coef path → F-statistic table
    if isinstance(coef, list) and len(coef) > 1:
        return _top_table_f(
            fit, coef, number, gene_names, adjust_method, sort_by, p_value, lfc
        )

    # Single-coef path → t-statistic table
    if isinstance(coef, list):
        coef = coef[0]

    return _top_table_t(
        fit,
        coef=coef,
        number=number,
        gene_names=gene_names,
        adjust_method=adjust_method,
        sort_by=sort_by,
        p_value=p_value,
        lfc=lfc,
        confint=confint,
    )


# ---------------------------------------------------------------------------
# Single-coefficient (t-test) table
# ---------------------------------------------------------------------------


def _top_table_t(
    fit: dict[str, Any],
    coef: int,
    number: int,
    gene_names: list[str],
    adjust_method: str,
    sort_by: str,
    p_value: float,
    lfc: float,
    confint: bool,
) -> pd.DataFrame:
    logfc = fit["coefficients"][:, coef]
    tstat = fit["t"][:, coef]
    pval = fit["p_value"][:, coef]
    ave_expr = fit.get("Amean", np.full(len(logfc), np.nan))

    lods = fit.get("lods")
    b_stat = lods[:, coef] if lods is not None else None

    # Multiple testing correction via pingouin
    adj_pval = _adjust_pvalues(pval, adjust_method)

    # Build DataFrame
    df = pd.DataFrame(
        {
            "logFC": logfc,
            "AveExpr": ave_expr,
            "t": tstat,
            "P.Value": pval,
            "adj.P.Val": adj_pval,
        },
        index=gene_names,
    )
    if b_stat is not None:
        df["B"] = b_stat

    # Optional confidence intervals
    if confint:
        s2_post = fit.get("s2_post")
        stdev_u = fit.get("stdev_unscaled")
        df_total = fit.get("df_total")
        if s2_post is not None and stdev_u is not None and df_total is not None:
            from scipy import stats as sp_stats
            se = np.sqrt(s2_post) * stdev_u[:, coef]
            t_crit = sp_stats.t.ppf(0.975, df=df_total)
            df["CI.L"] = logfc - t_crit * se
            df["CI.R"] = logfc + t_crit * se

    # Filter by thresholds
    keep = np.ones(len(df), dtype=bool)
    if p_value < 1.0:
        keep &= adj_pval <= p_value
    if lfc > 0.0:
        keep &= np.abs(logfc) >= lfc
    df = df[keep]

    if len(df) == 0:
        return df

    # Sort
    sort_by = _normalise_sort(sort_by)
    if sort_by == "B" and "B" not in df.columns:
        sort_by = "P"
    df = _sort_df(df, sort_by)

    return df.head(number)


# ---------------------------------------------------------------------------
# Multi-coefficient (F-test) table
# ---------------------------------------------------------------------------


def _top_table_f(
    fit: dict[str, Any],
    coef: list[int],
    number: int,
    gene_names: list[str],
    adjust_method: str,
    sort_by: str,
    p_value: float,
    lfc: float,
) -> pd.DataFrame:
    f_stat = fit.get("F")
    f_pval = fit.get("F_p_value")
    if f_stat is None:
        raise ValueError("F-statistics not found in fit — ensure design is full-rank")

    coeff_cols = fit["coefficients"][:, coef]
    ave_expr = fit.get("Amean", np.full(len(f_stat), np.nan))

    adj_pval = _adjust_pvalues(f_pval, adjust_method)

    df = pd.DataFrame(coeff_cols, index=gene_names)
    df.columns = [str(c) for c in coef]
    df["AveExpr"] = ave_expr
    df["F"] = f_stat
    df["P.Value"] = f_pval
    df["adj.P.Val"] = adj_pval

    keep = np.ones(len(df), dtype=bool)
    if p_value < 1.0:
        keep &= adj_pval <= p_value
    if lfc > 0.0:
        keep &= np.any(np.abs(coeff_cols) >= lfc, axis=1)
    df = df[keep]

    if len(df) == 0:
        return df

    sort_by_f = "P" if sort_by == "B" else _normalise_sort(sort_by)
    df = _sort_df(df, sort_by_f)

    return df.head(number)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adjust_pvalues(pvals: np.ndarray, method: str) -> np.ndarray:
    """Apply multiple testing correction using pingouin.multicomp."""
    method_pg = _METHOD_MAP.get(method, method)

    if method_pg == "none":
        return pvals.copy()

    finite = np.isfinite(pvals)
    adj = pvals.copy()
    if np.any(finite):
        _, adj_finite = pingouin.multicomp(pvals[finite], method=method_pg)
        adj[finite] = adj_finite

    return adj


def _resolve_coef(
    fit: dict[str, Any], coef: int | str | list | None, n_coef: int
) -> int | list[int]:
    if coef is None:
        # Default: all non-intercept coefficients
        design = fit.get("design")
        all_coefs = list(range(n_coef))
        if design is not None:
            for j in range(design.shape[1]):
                if np.all(design[:, j] == 1.0):
                    if j in all_coefs:
                        all_coefs.remove(j)
                    break
        if len(all_coefs) == 1:
            return all_coefs[0]
        return all_coefs

    if isinstance(coef, (list, tuple)):
        return [int(c) for c in coef]

    return int(coef)


def _normalise_sort(sort_by: str) -> str:
    aliases = {"M": "logFC", "A": "AveExpr", "Amean": "AveExpr", "p": "P", "T": "t"}
    return aliases.get(sort_by, sort_by)


def _sort_df(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "B" and "B" in df.columns:
        return df.sort_values("B", ascending=False)
    if sort_by == "P" and "P.Value" in df.columns:
        return df.sort_values("P.Value", ascending=True)
    if sort_by == "logFC" and "logFC" in df.columns:
        return df.reindex(df["logFC"].abs().sort_values(ascending=False).index)
    if sort_by == "t" and "t" in df.columns:
        return df.reindex(df["t"].abs().sort_values(ascending=False).index)
    if sort_by == "AveExpr" and "AveExpr" in df.columns:
        return df.sort_values("AveExpr", ascending=False)
    return df
