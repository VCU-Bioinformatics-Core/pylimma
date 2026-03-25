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

## Citation

If you use pylimma in your work, please also cite the original limma paper:

> Ritchie ME, Phipson B, Wu D, Hu Y, Law CW, Shi W, Smyth GK (2015). "limma powers differential expression analyses for RNA-sequencing and microarray studies." *Nucleic Acids Research*, **43**(7), e47. https://doi.org/10.1093/nar/gkv007

---

## License

MIT
