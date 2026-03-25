"""Numerical parity tests for lm_fit vs R limma lmFit."""

import numpy as np
import pytest
from pylimma import lm_fit


ATOL = 1e-6


def test_coefficients(ref_E, ref_design, ref_lmfit_coef):
    fit = lm_fit(ref_E, ref_design)
    np.testing.assert_allclose(
        fit["coefficients"], ref_lmfit_coef, atol=ATOL,
        err_msg="coefficients do not match R lmFit output"
    )


def test_sigma(ref_E, ref_design, ref_lmfit_sigma):
    fit = lm_fit(ref_E, ref_design)
    np.testing.assert_allclose(
        fit["sigma"], ref_lmfit_sigma, atol=ATOL,
        err_msg="sigma does not match R lmFit output"
    )


def test_df_residual(ref_E, ref_design, ref_lmfit_df_residual):
    fit = lm_fit(ref_E, ref_design)
    np.testing.assert_allclose(
        fit["df_residual"], ref_lmfit_df_residual, atol=1e-10,
        err_msg="df_residual does not match R lmFit output"
    )


def test_stdev_unscaled(ref_E, ref_design, ref_lmfit_stdev_unscaled):
    fit = lm_fit(ref_E, ref_design)
    np.testing.assert_allclose(
        fit["stdev_unscaled"], ref_lmfit_stdev_unscaled, atol=ATOL,
        err_msg="stdev_unscaled does not match R lmFit output"
    )


def test_cov_coefficients(ref_E, ref_design, ref_lmfit_cov_coef):
    fit = lm_fit(ref_E, ref_design)
    np.testing.assert_allclose(
        fit["cov_coefficients"], ref_lmfit_cov_coef, atol=ATOL,
        err_msg="cov_coefficients does not match R lmFit output"
    )


def test_amean(ref_E, ref_design):
    fit = lm_fit(ref_E, ref_design)
    expected = np.mean(ref_E, axis=1)
    np.testing.assert_allclose(fit["Amean"], expected, atol=1e-12)


def test_output_shapes(ref_E, ref_design):
    fit = lm_fit(ref_E, ref_design)
    n_genes, n_samples = ref_E.shape
    n_coef = ref_design.shape[1]
    assert fit["coefficients"].shape == (n_genes, n_coef)
    assert fit["stdev_unscaled"].shape == (n_genes, n_coef)
    assert fit["sigma"].shape == (n_genes,)
    assert fit["df_residual"].shape == (n_genes,)
    assert fit["cov_coefficients"].shape == (n_coef, n_coef)


def test_no_design_defaults_to_intercept(ref_E):
    fit = lm_fit(ref_E)
    assert fit["coefficients"].shape == (ref_E.shape[0], 1)
    np.testing.assert_allclose(
        fit["coefficients"][:, 0], np.mean(ref_E, axis=1), atol=1e-12
    )


def test_na_handling():
    """Fallback loop path handles NaN values correctly."""
    rng = np.random.default_rng(0)
    E = rng.normal(size=(10, 6))
    E[0, 2] = np.nan  # one missing value
    design = np.column_stack([
        np.ones(6),
        np.array([0, 0, 0, 1, 1, 1], dtype=float),
    ])
    fit = lm_fit(E, design)
    # Gene 0 should have df_residual = n_obs - rank = 5 - 2 = 3 (one obs dropped)
    assert fit["df_residual"][0] == 3
    # Other genes have df = 4
    assert np.all(fit["df_residual"][1:] == 4)
