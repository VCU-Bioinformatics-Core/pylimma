# pylimma

> **Experimental** — This package is under active development. APIs may change without notice and results should be validated against R limma before use in production analyses.

A native Python implementation of the core differential expression pipeline from the R [limma](https://bioconductor.org/packages/limma/) package (Linear Models for Microarray and Omics Data). The goal is to reproduce the `lmFit → eBayes → topTable` workflow with numerical parity to R, using [pingouin](https://pingouin-stats.org) for multiple testing correction and NumPy/SciPy for all other statistical operations.

---

## What is implemented

| R function | Python equivalent | Notes |
|---|---|---|
| `lmFit()` | `lm_fit()` | Gene-wise OLS via QR decomposition |
| `makeContrasts()` | `make_contrasts()` | Symbolic contrast expressions |
| `contrasts.fit()` | `contrasts_fit()` | Cholesky-based coefficient transformation |
| `eBayes()` | `ebayes()` | Empirical Bayes variance shrinkage + moderated t-stats |
| `topTable()` | `top_table()` | Ranked results table with BH/Bonferroni via pingouin |

**Not yet implemented:** `voom`, `normalizeBetweenArrays`, `duplicateCorrelation`, `diffSplice`, gene set tests (CAMERA, ROAST), batch effect removal.

---

## Installation

### 1. Create a conda environment

```bash
conda create -n pylimma python=3.13 -y
conda activate pylimma
```

### 2. Install the package

For regular use:

```bash
pip install pylimma
```

Or install directly from source (editable/development mode):

```bash
git clone https://github.com/VCU-Bioinformatics-Core/pylimma.git
cd pylimma
pip install -e ".[dev]"
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥ 1.21 | Array math, QR decomposition |
| scipy | ≥ 1.7 | Special functions (digamma, polygamma), t-distribution |
| pandas | ≥ 1.3 | Results tables |
| pingouin | ≥ 0.5 | Multiple testing correction in `top_table()` |

---

## Usage

### Basic workflow

```python
import numpy as np
from pylimma import lm_fit, ebayes, top_table

# E: (n_genes, n_samples) expression matrix (log2-scale)
# design: (n_samples, n_coef) design matrix

rng = np.random.default_rng(42)
n_genes, n_samples = 1000, 6

E = rng.normal(6, 1, size=(n_genes, n_samples))

# Two-group design: intercept + group effect
design = np.zeros((n_samples, 2))
design[:, 0] = 1            # intercept
design[3:, 1] = 1           # groupB indicator

fit = lm_fit(E, design)
fit = ebayes(fit)
results = top_table(fit, coef=1, number=20)
print(results.head())
```

### With contrasts

```python
from pylimma import lm_fit, make_contrasts, contrasts_fit, ebayes, top_table

fit = lm_fit(E, design)

# Define contrasts symbolically
C, names = make_contrasts(
    ["Intercept", "groupB", "groupC"],
    BvsA="groupB",
    CvsA="groupC",
    BvsC="groupB - groupC",
)

fit = contrasts_fit(fit, (C, names))
fit = ebayes(fit)

# Top differentially expressed genes for B vs A
results = top_table(fit, coef=0, number=50, adjust_method="fdr_bh", lfc=1.0)
```

### `top_table` options

```python
top_table(
    fit,
    coef=0,                  # which coefficient/contrast to test
    number=10,               # max rows to return (None = all)
    adjust_method="fdr_bh",  # "fdr_bh", "bonferroni", "holm", "none"
    sort_by="B",             # "B", "p", "logFC", "none"
    p_value=0.05,            # p-value filter (adjusted)
    lfc=None,                # log-fold-change filter
    confint=False,           # include 95% CI columns
)
```

---

## Running tests

```bash
conda activate pylimma
cd pylimma
pytest tests/ -v
```

All 40 tests cover numerical parity against R limma reference values generated with the same random seed.

---

## Numerical parity with R limma

Core quantities (coefficients, sigma, moderated t-statistics, p-values, BH-adjusted p-values) match R limma to within `atol=1e-5`. B-statistics (log-odds) preserve the same gene ranking as R but may differ in absolute value by a per-coefficient constant due to `tmixture` estimation noise.

---

## Mathematical equivalence with R limma

Each pylimma function is a direct line-for-line translation of the corresponding R limma source. The snippets below show the R source (top) and the Python translation (bottom) for every stage of the pipeline.

---

### Stage 1 — Linear model fitting (`lm_fit` ↔ `lmFit` / `lm.series`)

**Model:** `y_g = X β_g + ε_g` fit independently for each gene g via QR decomposition.
Coefficient covariance = `(X'X)⁻¹`; residual sigma = `‖residuals_g‖ / sqrt(df)`.

```r
# R — lmfit.R (lm.series)
QR            <- qr(design)
coefficients  <- qr.coef(QR, t(M))               # solve for all genes at once
cov.coef      <- chol2inv(QR$qr, size = QR$rank)  # (X'X)^{-1}
stdev.unscaled <- sqrt(diag(cov.coef))             # same for every gene
sigma         <- sqrt(colMeans(effects[(rank+1):n, ]^2))
df.residual   <- n - rank
```

```python
# Python — lmfit.py (lm_fit)
Q, R, pivot    = scipy_qr(Xw, pivoting=True)
QtY            = Q.T @ Yw.T                        # project all genes at once
beta           = solve_triangular(R[:rank, :rank], QtY[:rank, :])
cov_coef       = np.linalg.inv(R_est.T @ R_est)   # (X'X)^{-1}
stdev_unscaled = np.tile(np.sqrt(np.diag(cov_coef)), (n_genes, 1))
sigma          = np.sqrt(np.mean(residuals**2, axis=0))
df_residual    = n_samples - rank
```

---

### Stage 2 — Prior variance estimation (`fit_f_dist` ↔ `fitFDist`)

**Model:** Gene variances follow a scaled-F prior: `σ̂²_g / s²₀ ~ F(df_g, df₀)`.
Parameters `s²₀` and `df₀` are estimated by log-scale moment matching using
`logmdigamma(x) = log(x) − ψ(x)` (ψ = digamma) and a trigamma correction.

```r
# R — fitFDist.R
z    <- log(x)
e    <- z + logmdigamma(df1/2)           # center on log scale
evar <- var(e) - mean(trigamma(df1/2))   # subtract sampling noise
df2  <- 2 * trigammaInverse(evar)        # Newton's method
s20  <- exp(emean - logmdigamma(df2/2))
```

```python
# Python — ebayes.py (fit_f_dist)
z    = np.log(x)
e    = z + _log_mdigamma(df1 / 2.0)                          # log(x) − ψ(df1/2) + log(df1/2)
evar = np.var(e) - np.mean(sp_special.polygamma(1, df1/2.0)) # trigamma correction
df2  = 2.0 * _trigamma_inverse(evar)                          # same Newton iteration
s20  = np.exp(emean - _log_mdigamma(df2 / 2.0))
```

The trigamma inverse uses identical Newton steps in both languages:

```r
# R — fitFDist.R (trigammaInverse)
repeat {
    tri <- trigamma(y)
    dif <- tri * (1 - tri/x) / psigamma(y, deriv=2)
    y   <- y + dif
    if (max(-dif/y) < 1e-8) break
}
```

```python
# Python — ebayes.py (_trigamma_inverse)
for _ in range(50):
    tri = sp_special.polygamma(1, y)   # trigamma
    dif = tri * (1.0 - tri / x) / sp_special.polygamma(2, y)  # tetragamma
    y  += dif
    if np.nanmax(-dif / y) < 1e-8:
        break
```

---

### Stage 3 — Variance squeezing (`squeeze_var` ↔ `squeezeVar`)

**Formula:** Posterior variance is a weighted average of gene-wise and prior variance:

```
s²_g,post = (df_g · s²_g  +  df₀ · s²₀) / (df_g + df₀)
```

```r
# R — squeezeVar.R
var.post <- (df * var + df.prior * var.prior) / (df + df.prior)
```

```python
# Python — ebayes.py (squeeze_var) — identical line-for-line
var_post = (df * var + df_prior * var_prior) / (df + df_prior)
```

When `df₀ = ∞` (no between-gene variance detected), both implementations set `var_post = s²₀` for all genes.

---

### Stage 4 — Moderated t-statistics & p-values (`ebayes` ↔ `eBayes`)

**Key idea:** Replace gene-wise `σ̂_g` with posterior `s_g,post`; the resulting t-statistic
follows a t-distribution with `df_total = df_g + df₀` degrees of freedom.

```r
# R — ebayes.R
t        <- coefficients / stdev.unscaled / sqrt(s2.post)
df.total <- pmin(df.residual + df.prior, sum(df.residual))
p.value  <- 2 * pt(-abs(t), df = df.total)
```

```python
# Python — ebayes.py (ebayes)
t        = coeff / (stdev_u * np.sqrt(s2_post)[:, None])
df_total = np.minimum(df_res + df_prior, np.nansum(df_res))
p_value  = 2.0 * sp_stats.t.sf(np.abs(t), df=df_total[:, None])
```

---

### Stage 5 — B-statistics / log-odds (`ebayes` ↔ `eBayes` + `tmixture.matrix`)

**Formula:**

```
B_gk = log(p / (1−p))  −  ½ log(r_gk)  +  ½(1 + df_total) · log[(t²_gk + df_total) / (t²_gk/r_gk + df_total)]
```

where `r_gk = (u²_gk + v₀_k) / u²_gk`, `u_gk = stdev.unscaled_gk`, and `v₀_k` is a
per-coefficient prior variance estimated from the top fraction of |t| statistics.

```r
# R — ebayes.R
r      <- (stdev.unscaled^2 + var.prior) / stdev.unscaled^2
t2     <- t^2
kernel <- (1 + df.total)/2 * log((t2 + df.total) / (t2/r + df.total))
lods   <- log(proportion/(1-proportion)) - log(r)/2 + kernel
```

```python
# Python — ebayes.py (ebayes)
r      = (stdev_u**2 + var_prior[None, :]) / stdev_u**2
t2     = t**2
kernel = (1.0 + df_total[:, None]) / 2.0 * np.log(
             (t2 + df_total[:, None]) / (t2 / r + df_total[:, None]))
lods   = np.log(proportion / (1.0 - proportion)) - np.log(r) / 2.0 + kernel
```

---

### Stage 6 — Contrast transformation (`contrasts_fit` ↔ `contrasts.fit`)

**Formulas:** Given contrast matrix `C`:
- New coefficients: `β* = β C`
- New covariance: `Σ* = (R C)ᵀ (R C)` where `R = chol(Σ)`
- New stdev (orthogonal design shortcut): `u* = sqrt(u² C²)`

```r
# R — contrasts.R
fit$coefficients     <- fit$coefficients %*% contrasts
R                    <- chol(fit$cov.coefficients)
fit$cov.coefficients <- crossprod(R %*% contrasts)
# orthogonal shortcut:
fit$stdev.unscaled   <- sqrt(fit$stdev.unscaled^2 %*% contrasts^2)
```

```python
# Python — contrasts.py (contrasts_fit)
fit["coefficients"]     = coeff @ contrasts
R_chol                  = cholesky(cov_clean, lower=False)
fit["cov_coefficients"] = (R_chol @ contrasts).T @ (R_chol @ contrasts)
# orthogonal shortcut:
fit["stdev_unscaled"]   = np.sqrt(stdev**2 @ contrasts**2)
```

---

### Stage 7 — Multiple testing correction (`top_table` ↔ `topTable`)

**Method:** Benjamini-Hochberg FDR (default), Bonferroni, Holm, or none — applied to the
vector of moderated p-values across all genes. This is the **one place pingouin is used**.

```r
# R — toptable.R
adj.P.Value <- p.adjust(P.Value, method = adjust.method)
```

```python
# Python — toptable.py  (uses pingouin.multicomp)
_, adj_pvals = pingouin.multicomp(pvals, method="fdr_bh")
```

`pingouin.multicomp` wraps `statsmodels.stats.multitest.multipletests`, which implements
the identical BH step-up procedure as R's `p.adjust(method="BH")`.

---

### Summary

| Stage | Math | R function | Python function | Equivalence |
|---|---|---|---|---|
| 1 | Gene-wise OLS | `lmFit` / `lm.series` | `lm_fit` | Exact (QR) |
| 2 | Log-F moment matching | `fitFDist` | `fit_f_dist` | Exact (digamma/trigamma) |
| 3 | EB posterior variance | `squeezeVar` | `squeeze_var` | **Identical line** |
| 4 | Moderated t / p-value | `eBayes` | `ebayes` | Exact |
| 5 | B-statistics (log-odds) | `eBayes` + `tmixture.matrix` | `ebayes` | Exact (same ranking) |
| 6 | Contrast transform | `contrasts.fit` | `contrasts_fit` | Exact (Cholesky) |
| 7 | FDR correction | `p.adjust` | `pingouin.multicomp` | Exact (same BH algorithm) |

---

## Citation

If you use pylimma in your work, please also cite the original limma paper:

> Ritchie ME, Phipson B, Wu D, Hu Y, Law CW, Shi W, Smyth GK (2015). "limma powers differential expression analyses for RNA-sequencing and microarray studies." *Nucleic Acids Research*, **43**(7), e47. https://doi.org/10.1093/nar/gkv007

---

## License

MIT
