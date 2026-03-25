"""Tests for top_table, including numerical parity with R topTable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma import lm_fit, ebayes, top_table


REF = Path(__file__).parent / "reference_data"
ATOL = 1e-5


@pytest.fixture(scope="module")
def eb_fit(ref_E, ref_design):
    fit = lm_fit(ref_E, ref_design)
    return ebayes(fit)


@pytest.fixture(scope="module")
def r_toptable() -> pd.DataFrame:
    return pd.read_csv(REF / "toptable.csv", index_col=0)


def test_top_table_returns_dataframe(eb_fit):
    result = top_table(eb_fit, coef=1, number=10)
    assert isinstance(result, pd.DataFrame)


def test_top_table_columns(eb_fit):
    result = top_table(eb_fit, coef=1, number=10)
    for col in ("logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B"):
        assert col in result.columns, f"Missing column '{col}'"


def test_top10_gene_order(eb_fit, r_toptable):
    """Top 10 genes by B-statistic match R topTable (default sort)."""
    result = top_table(eb_fit, coef=1, number=10, sort_by="B")
    r_top10 = r_toptable.head(10)

    # Row indices (gene numbers as strings) should match
    py_idx = [int(i) for i in result.index]
    r_idx  = [int(i) - 1 for i in r_top10.index]  # R is 1-based row names
    assert py_idx == r_idx, (
        f"Top-10 gene order mismatch.\n  Python: {py_idx}\n  R: {r_idx}"
    )


def test_logfc_parity(eb_fit, r_toptable):
    result = top_table(eb_fit, coef=1, number=100, sort_by="P")
    r_sorted = r_toptable.sort_values("P.Value")
    np.testing.assert_allclose(
        result["logFC"].values, r_sorted["logFC"].values, atol=ATOL,
        err_msg="logFC does not match R topTable"
    )


def test_pvalue_parity(eb_fit, r_toptable):
    result = top_table(eb_fit, coef=1, number=100, sort_by="P")
    r_sorted = r_toptable.sort_values("P.Value")
    np.testing.assert_allclose(
        result["P.Value"].values, r_sorted["P.Value"].values, atol=ATOL,
        err_msg="P.Value does not match R topTable"
    )


def test_adj_pvalue_parity(eb_fit, r_toptable):
    """Adjusted p-values from pingouin.multicomp should match R p.adjust(BH)."""
    result = top_table(eb_fit, coef=1, number=100, sort_by="P")
    r_sorted = r_toptable.sort_values("P.Value")
    np.testing.assert_allclose(
        result["adj.P.Val"].values, r_sorted["adj.P.Val"].values, atol=ATOL,
        err_msg="adj.P.Val from pingouin.multicomp does not match R p.adjust(BH)"
    )


def test_lfc_filter(eb_fit):
    """Filtering by lfc removes genes below threshold."""
    result = top_table(eb_fit, coef=1, number=100, lfc=1.0)
    assert np.all(np.abs(result["logFC"]) >= 1.0)


def test_pvalue_filter(eb_fit):
    """Filtering by p_value removes non-significant genes."""
    result = top_table(eb_fit, coef=1, number=100, p_value=0.05)
    assert np.all(result["adj.P.Val"] <= 0.05)


def test_sort_by_p(eb_fit):
    result = top_table(eb_fit, coef=1, number=20, sort_by="P")
    pvals = result["P.Value"].values
    assert np.all(pvals[:-1] <= pvals[1:]), "Results not sorted by P.Value"


def test_sort_by_logfc(eb_fit):
    result = top_table(eb_fit, coef=1, number=20, sort_by="logFC")
    lfc = np.abs(result["logFC"].values)
    assert np.all(lfc[:-1] >= lfc[1:]), "Results not sorted by |logFC|"


def test_number_limit(eb_fit):
    result = top_table(eb_fit, coef=1, number=5)
    assert len(result) == 5


def test_confint(eb_fit):
    result = top_table(eb_fit, coef=1, number=10, confint=True)
    assert "CI.L" in result.columns
    assert "CI.R" in result.columns
    # CI should straddle the logFC
    assert np.all(result["CI.L"] <= result["logFC"])
    assert np.all(result["CI.R"] >= result["logFC"])


def test_adjust_method_bonferroni(eb_fit):
    result_bh = top_table(eb_fit, coef=1, number=100, adjust_method="BH")
    result_bon = top_table(eb_fit, coef=1, number=100, adjust_method="bonferroni")
    # Bonferroni is more conservative than BH
    assert np.all(result_bon["adj.P.Val"].values >= result_bh["adj.P.Val"].values - 1e-10)


def test_adjust_method_none(eb_fit):
    result_none = top_table(eb_fit, coef=1, number=100, adjust_method="none")
    np.testing.assert_allclose(
        result_none["adj.P.Val"].values, result_none["P.Value"].values, atol=1e-12
    )
