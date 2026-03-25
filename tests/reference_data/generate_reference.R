#!/usr/bin/env Rscript
# Generate reference data for pylimma numerical parity tests.
# Run from this directory:  Rscript generate_reference.R
#
# Requires: limma (Bioconductor)

library(limma)

set.seed(42)

# ---------------------------------------------------------------------------
# Two-group comparison: 100 genes, 6 samples (3 per group)
# First 10 genes are truly DE (mean shift of 2 in group B)
# ---------------------------------------------------------------------------
n_genes  <- 100L
n_A      <- 3L
n_B      <- 3L
n_samples <- n_A + n_B

E <- matrix(rnorm(n_genes * n_samples, mean = 5, sd = 1), nrow = n_genes)
# Add signal to first 10 genes in group B
E[1:10, (n_A + 1):n_samples] <- E[1:10, (n_A + 1):n_samples] + 2

group <- factor(rep(c("A", "B"), c(n_A, n_B)))
design <- model.matrix(~ group)
colnames(design) <- c("Intercept", "groupB")

# --- lmFit ---
fit <- lmFit(E, design)

# --- eBayes ---
fit_eb <- eBayes(fit)

# --- contrasts.fit ---
# Test a contrast: groupB - groupA is already groupB coefficient,
# but let's construct a proper contrast to exercise contrasts.fit
C <- makeContrasts(BvsA = groupB, levels = design)
fit_c  <- contrasts.fit(fit, C)
fit_ceb <- eBayes(fit_c)

# --- topTable ---
tt <- topTable(fit_eb, coef = "groupB", number = Inf, sort.by = "P")

# ---------------------------------------------------------------------------
# Save as CSV and plain text for numpy loading
# ---------------------------------------------------------------------------
outdir <- "."

write_matrix <- function(m, fname) {
  write.table(m, file = file.path(outdir, fname),
              sep = ",", row.names = FALSE, col.names = FALSE)
}
write_vec <- function(v, fname) {
  write(v, file = file.path(outdir, fname), ncolumns = length(v), sep = ",")
}

write_matrix(E,                    "E.csv")
write_matrix(design,               "design.csv")

# lmFit outputs
write_matrix(fit$coefficients,     "lmfit_coef.csv")
write_matrix(fit$stdev.unscaled,   "lmfit_stdev_unscaled.csv")
write_vec(fit$sigma,               "lmfit_sigma.csv")
write_vec(fit$df.residual,         "lmfit_df_residual.csv")
write_matrix(fit$cov.coefficients, "lmfit_cov_coef.csv")

# eBayes outputs
write_matrix(fit_eb$t,             "eb_t.csv")
write_matrix(fit_eb$p.value,       "eb_pvalue.csv")
write_vec(fit_eb$s2.post,          "eb_s2post.csv")
write_vec(fit_eb$df.total,         "eb_df_total.csv")
write(fit_eb$df.prior,             file = file.path(outdir, "eb_df_prior.csv"))
write(fit_eb$s2.prior,             file = file.path(outdir, "eb_s2prior.csv"))
write_matrix(fit_eb$lods,          "eb_lods.csv")

# contrasts.fit + eBayes outputs
write_matrix(fit_c$coefficients,   "cf_coef.csv")
write_matrix(fit_c$stdev.unscaled, "cf_stdev_unscaled.csv")
write_matrix(fit_ceb$t,            "cf_t.csv")
write_matrix(fit_ceb$p.value,      "cf_pvalue.csv")

# topTable output
write.csv(tt, file = file.path(outdir, "toptable.csv"), row.names = TRUE)

cat("Reference data written to", outdir, "\n")
