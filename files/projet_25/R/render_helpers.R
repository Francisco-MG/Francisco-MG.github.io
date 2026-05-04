# render_helpers.R
# Small utilities used in projet25_ema_physio_adarp.qmd.
# Kept minimal and dependency-light; loaded via source() from the qmd.

# Robust parquet loader: returns an empty tibble with a warning if the file
# is missing (useful in demo mode where some intermediate parquets may not
# be present). Requires the 'arrow' package.
safe_read_parquet <- function(path, cols = NULL) {
  stopifnot(requireNamespace("arrow", quietly = TRUE))
  if (!file.exists(path)) {
    warning("Missing parquet: ", path, " -> returning empty tibble")
    return(tibble::tibble())
  }
  if (is.null(cols)) {
    arrow::read_parquet(path)
  } else {
    arrow::read_parquet(path, col_select = tidyselect::any_of(cols))
  }
}

# One-line summary of wear statistics per subject, formatted for a gt/kbl table.
summarise_wear_by_subject <- function(nonwear_df) {
  stopifnot(requireNamespace("dplyr", quietly = TRUE))
  nonwear_df %>%
    dplyr::group_by(subject_id) %>%
    dplyr::summarise(
      n_sessions     = dplyr::n(),
      total_hours    = round(sum(total_duration_h, na.rm = TRUE), 1),
      wear_hours     = round(sum(wear_h,          na.rm = TRUE), 1),
      median_pct_wear = round(stats::median(pct_wear, na.rm = TRUE), 1),
      .groups = "drop"
    )
}

# Per-subject EMA availability vs usable windows.
summarise_ema_coverage <- function(ema_windows_qc_df) {
  stopifnot(requireNamespace("dplyr", quietly = TRUE))
  ema_windows_qc_df %>%
    dplyr::group_by(subject_id) %>%
    dplyr::summarise(
      n_ema           = dplyr::n(),
      n_matched       = sum(session_idx >= 0),
      n_usable_init   = sum(usable, na.rm = TRUE),
      n_usable_final  = sum(usable_final, na.rm = TRUE),
      pct_final       = round(100 * mean(usable_final, na.rm = TRUE), 1),
      .groups = "drop"
    )
}

# Compact message used in the qmd preamble to remind readers of data mode.
data_mode_banner <- function(data_dir) {
  is_demo <- grepl("demo", data_dir, fixed = TRUE)
  if (is_demo) {
    "> **Demo mode.** This render uses the slim reproducibility subset (~3 subjects).
> The narrative text reflects results from the full ADARP run; see `README.md`."
  } else {
    "> **Full mode.** This render uses all ADARP subjects processed locally."
  }
}
