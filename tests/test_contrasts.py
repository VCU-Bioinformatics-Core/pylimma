"""Numerical parity tests for contrasts_fit vs R limma contrasts.fit."""

import numpy as np
import pytest
from pylimma import lm_fit, make_contrasts, contrasts_fit


ATOL = 1e-6


@pytest.fixture(scope="module")
def fit_after_contrasts(ref_E, ref_design):
    fit = lm_fit(ref_E, ref_design)
    # Contrast: groupB coefficient (index 1)
    C, names = make_contrasts(["Intercept", "groupB"], BvsA="groupB")
    return contrasts_fit(fit, (C, names))


def test_contrasts_coefficients(fit_after_contrasts, ref_cf_coef):
    # Our output is (n_genes, n_contrasts); R CSV loads as 1D for single contrast
    np.testing.assert_allclose(
        fit_after_contrasts["coefficients"].squeeze(), ref_cf_coef.squeeze(), atol=ATOL,
        err_msg="contrasts_fit coefficients do not match R output"
    )


def test_contrasts_stdev_unscaled(fit_after_contrasts, ref_cf_stdev_unscaled):
    np.testing.assert_allclose(
        fit_after_contrasts["stdev_unscaled"].squeeze(), ref_cf_stdev_unscaled.squeeze(), atol=ATOL,
        err_msg="contrasts_fit stdev_unscaled does not match R output"
    )


def test_contrasts_clears_test_stats(ref_E, ref_design):
    """contrasts_fit must remove any existing eBayes test statistics."""
    from pylimma import ebayes
    fit = lm_fit(ref_E, ref_design)
    fit = ebayes(fit)
    assert "t" in fit

    C, names = make_contrasts(["Intercept", "groupB"], BvsA="groupB")
    fit2 = contrasts_fit(fit, (C, names))
    for key in ("t", "p_value", "lods", "F", "F_p_value"):
        assert key not in fit2, f"Key '{key}' should have been removed by contrasts_fit"


def test_make_contrasts_subtraction():
    """make_contrasts parses A - B correctly."""
    C, names = make_contrasts(["A", "B", "C"], AvsB="A - B", CvsA="C - A")
    assert names == ["AvsB", "CvsA"]
    np.testing.assert_array_equal(C[:, 0], [1, -1, 0])
    np.testing.assert_array_equal(C[:, 1], [-1, 0, 1])


def test_make_contrasts_scaled():
    """make_contrasts handles scalar multipliers."""
    C, names = make_contrasts(["A", "B"], half="0.5*A - 0.5*B")
    np.testing.assert_allclose(C[:, 0], [0.5, -0.5])


def test_contrasts_fit_then_ebayes_t(ref_E, ref_design, ref_cf_t, ref_cf_pvalue):
    """Full pipeline: lm_fit -> contrasts_fit -> ebayes t-stats match R."""
    from pylimma import ebayes
    fit = lm_fit(ref_E, ref_design)
    C, names = make_contrasts(["Intercept", "groupB"], BvsA="groupB")
    fit = contrasts_fit(fit, (C, names))
    fit = ebayes(fit)

    np.testing.assert_allclose(
        fit["t"].squeeze(), ref_cf_t.squeeze(), atol=ATOL,
        err_msg="t-statistics after contrasts_fit + ebayes differ from R"
    )
    np.testing.assert_allclose(
        fit["p_value"].squeeze(), ref_cf_pvalue.squeeze(), atol=ATOL,
        err_msg="p-values after contrasts_fit + ebayes differ from R"
    )
