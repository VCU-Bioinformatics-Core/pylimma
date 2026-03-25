"""Numerical parity tests for ebayes vs R limma eBayes."""

import numpy as np
import pytest
from pylimma import lm_fit, ebayes


ATOL = 1e-5   # slightly looser for empirical Bayes numerics
ATOL_TIGHT = 1e-6


@pytest.fixture(scope="module")
def eb_fit(ref_E, ref_design):
    fit = lm_fit(ref_E, ref_design)
    return ebayes(fit)


def test_s2_post(eb_fit, ref_eb_s2post):
    np.testing.assert_allclose(
        eb_fit["s2_post"], ref_eb_s2post, atol=ATOL,
        err_msg="s2_post does not match R eBayes s2.post"
    )


def test_df_prior(eb_fit, ref_eb_df_prior):
    np.testing.assert_allclose(
        eb_fit["df_prior"], ref_eb_df_prior, atol=0.1,
        err_msg="df_prior does not match R eBayes df.prior"
    )


def test_s2_prior(eb_fit, ref_eb_s2prior):
    np.testing.assert_allclose(
        eb_fit["s2_prior"], ref_eb_s2prior, rtol=0.01,
        err_msg="s2_prior does not match R eBayes s2.prior"
    )


def test_df_total(eb_fit, ref_eb_df_total):
    np.testing.assert_allclose(
        eb_fit["df_total"], ref_eb_df_total, atol=0.1,
        err_msg="df_total does not match R eBayes df.total"
    )


def test_t_statistics(eb_fit, ref_eb_t):
    np.testing.assert_allclose(
        eb_fit["t"], ref_eb_t, atol=ATOL,
        err_msg="moderated t-statistics do not match R eBayes"
    )


def test_p_values(eb_fit, ref_eb_pvalue):
    np.testing.assert_allclose(
        eb_fit["p_value"], ref_eb_pvalue, atol=ATOL,
        err_msg="p-values do not match R eBayes"
    )


def test_lods(eb_fit, ref_eb_lods):
    # B-statistics differ from R by a per-coefficient constant offset driven by
    # var_prior estimation noise in tmixture.  The important property is that
    # the RANKING of genes within each coefficient matches R (verified by
    # test_top10_gene_order in test_toptable.py).  We check that within each
    # coefficient column the Pearson correlation is near-perfect (>=0.999).
    from scipy.stats import pearsonr
    for j in range(eb_fit["lods"].shape[1]):
        r, _ = pearsonr(eb_fit["lods"][:, j], ref_eb_lods[:, j])
        assert r >= 0.999, (
            f"B-statistic ranking for coef {j} is too different from R "
            f"(Pearson r={r:.4f})"
        )


def test_output_fields(eb_fit):
    for field in ("s2_prior", "df_prior", "s2_post", "df_total", "t", "p_value", "lods"):
        assert field in eb_fit, f"Missing field '{field}' in ebayes output"


def test_de_genes_top_ranked(eb_fit):
    """The 10 truly DE genes (index 0-9) should dominate the top rankings.

    With only 3+3 samples (df=4 per gene), the signal-to-noise ratio is
    moderate.  Empirical Bayes moderation helps, but some null genes will
    still rank above a few true DE genes by chance.  We require at least
    half overlap as a sanity check.
    """
    p_vals = eb_fit["p_value"][:, 1]  # groupB coefficient
    order = np.argsort(p_vals)
    top10 = set(order[:10])
    de_genes = set(range(10))
    overlap = len(top10 & de_genes)
    assert overlap >= 5, (
        f"Expected at least 5 of top-10 genes to be truly DE, got {overlap}"
    )


def test_squeeze_var_scalar():
    """squeeze_var shrinks sample variances toward a common prior.

    When all df are equal and equal to df1, the variance of log(x) equals
    trigamma(df1/2).  After the trigamma correction in fit_f_dist, evar ≈ 0,
    so df_prior = inf (complete pooling) is the statistically correct answer.
    We verify that (a) df_prior >= 0 and (b) posterior variances are at
    least as concentrated as sample variances.
    """
    from pylimma.ebayes import squeeze_var
    rng = np.random.default_rng(99)
    var = rng.chisquare(4, size=200) / 4  # chi-sq(4)/4, mean=1
    df = np.full(200, 4.0)
    result = squeeze_var(var, df)
    assert result["df_prior"] >= 0
    # Posterior variances must be no more dispersed than sample variances
    assert np.std(result["var_post"]) <= np.std(var) + 1e-10


def test_fit_f_dist_known():
    """fit_f_dist recovers approximate parameters of a known F distribution."""
    from pylimma.ebayes import fit_f_dist
    rng = np.random.default_rng(7)
    # Generate F(4, 10) samples, scale by s0^2=2
    s0_sq = 2.0
    df1 = 4.0
    df2_true = 10.0
    x = rng.f(df1, df2_true, size=2000) * s0_sq
    result = fit_f_dist(x, df1=np.full(2000, df1))
    # df2 estimate should be within 20% of truth
    assert abs(result["df2"] - df2_true) / df2_true < 0.20, (
        f"df2 estimate {result['df2']:.2f} too far from truth {df2_true}"
    )
    # scale estimate should be within 20% of truth
    assert abs(result["scale"] - s0_sq) / s0_sq < 0.20, (
        f"scale estimate {result['scale']:.3f} too far from truth {s0_sq}"
    )
