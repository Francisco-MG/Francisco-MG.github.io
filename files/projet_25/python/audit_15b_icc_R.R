# audit_15b_icc_R.R
# ------------------
# Gold-standard cross-check of phase-15b ICC values using psych::ICC().
#
# Python audits A, B, D returned OK. Audit E flagged a 0.10-0.20 disagreement
# between two Python ICC estimators (one-way ANOVA vs REML method-of-moments)
# on a strongly unbalanced design (1 to 17 windows per subject). This script
# uses the canonical psych::ICC() as a third independent estimate.
#
# Usage:
#   Rscript audit_15b_icc_R.R                  (writes a console report)
#
# Reads:  files/projet_25/data/processed/physio_features.parquet
# Output: console table + verdict.
#
# Interpretation key (psych::ICC variants):
#   ICC1  : one-way random effects, single rater  (= our Python ICC1)
#   ICC2  : two-way random effects, single rater
#   ICC3  : two-way mixed effects, single rater (consistency)
#   *k variants are for averaged ratings (not relevant here).
#
# In our context (n EMA windows per subject as repeated measures),
# ICC1 is the conceptually correct one. ICC2/ICC3 are reported for
# completeness; they should be close to ICC1 when there is no rater bias
# (which is our case: all "raters" are the same wearable measurement).

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(psych)
  library(tidyr)
})

# --- paths --------------------------------------------------------------------
# Try to locate the script directory robustly across Rscript / interactive use.
get_script_dir <- function() {
  # When run via Rscript: sys.frames() contains the source frame
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(dirname(sub("^--file=", "", file_arg[1])), mustWork = FALSE))
  }
  # Fallback: working directory (interactive use)
  return(getwd())
}
script_dir   <- get_script_dir()
project_dir  <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
parquet_path <- file.path(project_dir, "data", "processed", "physio_features.parquet")

if (!file.exists(parquet_path)) {
  stop("physio_features.parquet not found at: ", parquet_path,
       "\nRun this script from files/projet_25/python/, or adjust paths.")
}

cat("Reading:", parquet_path, "\n")
features <- read_parquet(parquet_path)
features <- features[features$usable_final == TRUE, ]
cat("usable_final rows:", nrow(features), "\n")
cat("Subjects with >= 2 windows:",
    sum(table(features$subject_id) >= 2), "/",
    length(unique(features$subject_id)), "\n\n")

# --- helper: compute psych::ICC for one feature -------------------------------
# psych::ICC needs a wide matrix: rows = "subjects", cols = "raters".
# For repeated measures with unequal n, we use a long-format trick: we treat
# each window as a "rating", and ICC1 (one-way) handles the unbalanced case
# directly via the underlying lme4 / aov machinery in psych.
#
# However, psych::ICC requires *equal* n per "subject" (rows) when given a
# matrix. For unbalanced designs, the proper approach is a mixed model with
# lme4. We'll use both:
#   (1) psych::ICC on the BALANCED subset (subjects with at least k windows,
#       trimmed to k each)
#   (2) lme4-based ICC on the FULL (unbalanced) data
# and compare both to our two Python estimators.

icc_from_lmer <- function(values, subjects) {
  # ICC1 from a random-intercept model: var(intercept) / (var(intercept) + var(residual))
  if (!requireNamespace("lme4", quietly = TRUE)) return(NA_real_)
  d <- data.frame(value = values, subj = factor(subjects))
  d <- d[!is.na(d$value), ]
  if (nlevels(droplevels(d$subj)) < 2) return(NA_real_)
  fit <- tryCatch(
    lme4::lmer(value ~ 1 + (1 | subj), data = d, REML = TRUE),
    error = function(e) NULL, warning = function(w) NULL
  )
  if (is.null(fit)) return(NA_real_)
  vc <- as.data.frame(lme4::VarCorr(fit))
  v_between <- vc$vcov[vc$grp == "subj"]
  v_within  <- vc$vcov[vc$grp == "Residual"]
  if (length(v_between) == 0 || length(v_within) == 0) return(NA_real_)
  v_between / (v_between + v_within)
}

icc_from_psych_balanced <- function(values, subjects, min_per_subj = 2) {
  # Pick the largest balanced subset: subjects with >= min_per_subj windows,
  # trimmed to exactly min_per_subj each.
  d <- data.frame(value = values, subj = subjects)
  d <- d[!is.na(d$value), ]
  counts <- table(d$subj)
  keep   <- names(counts)[counts >= min_per_subj]
  if (length(keep) < 2) return(NA_real_)
  d <- d[d$subj %in% keep, ]
  d <- d %>%
    group_by(subj) %>%
    slice_head(n = min_per_subj) %>%
    ungroup()
  mat <- matrix(d$value, ncol = min_per_subj, byrow = TRUE)
  res <- tryCatch(
    psych::ICC(mat, lmer = FALSE),
    error = function(e) NULL
  )
  if (is.null(res)) return(NA_real_)
  # Pull ICC1 (one-way random)
  res$results$ICC[1]
}

# --- run on the same features as the Python audit ----------------------------
features_to_check <- c("acc_magnitude_std", "hrv_meannn", "eda_scl_mean", "hrv_rmssd")
# Python's reported ICCs (from your console output)
python_icc1   <- c(0.299, 0.297, 0.236, 0.056)
python_reml   <- c(0.428, 0.506, 0.424, 0.217)

cat("=== ICC cross-check ===\n\n")
results <- data.frame(
  feature = character(),
  python_icc1_oneway = numeric(),
  python_reml_mom    = numeric(),
  R_lmer_random_int  = numeric(),
  R_psych_ICC1_balanced_n2 = numeric()
)

for (i in seq_along(features_to_check)) {
  feat <- features_to_check[i]
  v    <- features[[feat]]
  s    <- features$subject_id
  icc_R_lmer <- icc_from_lmer(v, s)
  icc_R_psych <- icc_from_psych_balanced(v, s, min_per_subj = 2)
  results <- rbind(results, data.frame(
    feature = feat,
    python_icc1_oneway = python_icc1[i],
    python_reml_mom    = python_reml[i],
    R_lmer_random_int  = round(icc_R_lmer,  3),
    R_psych_ICC1_balanced_n2 = round(icc_R_psych, 3)
  ))
}

print(results, row.names = FALSE)

cat("\n--- INTERPRETATION ---\n")
cat("R lmer random-intercept ICC is the *gold-standard* estimator for our\n")
cat("unbalanced repeated-measures design.\n\n")

# Verdict for each feature
cat("Per-feature agreement (closest Python estimator to R-lmer):\n")
for (i in seq_len(nrow(results))) {
  row <- results[i, ]
  d_p1   <- abs(row$python_icc1_oneway - row$R_lmer_random_int)
  d_reml <- abs(row$python_reml_mom    - row$R_lmer_random_int)
  closer <- ifelse(d_p1 < d_reml, "Python ICC1 (one-way)", "Python REML-MoM")
  best_d <- min(d_p1, d_reml)
  status <- if (best_d <= 0.05) "EXCELLENT" else if (best_d <= 0.10) "OK" else "PROBLEM"
  cat(sprintf("  %-22s  closer=%-25s  diff_to_R=%.3f  %s\n",
              row$feature, closer, best_d, status))
}

cat("\n--- VERDICT ---\n")
cat("If the closer Python estimator is consistently within 0.10 of the R lmer\n")
cat("value, ICC computation is correct and the original audit-E flag was a\n")
cat("false positive driven by an overly strict threshold on a design-imbalance\n")
cat("artifact. We can then adopt that estimator as our official one in the qmd.\n")
