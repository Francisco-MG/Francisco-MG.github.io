"""
extract_features.py
-------------------
Orchestrate feature extraction for all EMA windows in the ADARP pipeline.

Input:
    data/processed/ema_windows_qc.parquet  (produced by prepare_data.py)

Output:
    data/processed/physio_features.parquet      (one row per EMA)
    data/processed/extraction_qc_report.parquet (feature-level QC summary)

Each EMA row is enriched with:
    - 10 HRV features (time, frequency, non-linear)
    - 6 EDA features (tonic + phasic + composite)
    - 5 ACC activity features
    - 3 quality-control flags (bvp_quality_pct, eda_quality_pct, feature_usable)

Usage:
    python extract_features.py              # full run, only on usable_final windows
    python extract_features.py --all        # include all windows (even non-usable)
    python extract_features.py --demo       # read+write the demo subset
    python extract_features.py --standardize # also compute within/between subject stats

Design notes:
    - Median QC: feature_usable = (bvp_quality_pct >= 50) AND (eda_quality_pct >= 50).
      Windows that fail are kept with their (noisy) features, flagged for optional
      exclusion in phase 15c sensitivity analyses.
    - Within-subject z-scores and between-subject means are appended as extra
      columns with suffixes _wz and _bm, respectively.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ingest_e4 import load_signal
from extract_hrv import extract_hrv_features
from extract_eda import extract_eda_features
from extract_activity import extract_activity_features

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_DIR   = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
DEMO_DIR      = PROJECT_DIR / "data" / "demo"

# Median QC thresholds
BVP_QC_MEDIAN = 50.0  # % of 10-s segments with acceptable peak density
EDA_QC_MEDIAN = 50.0  # % of 10-s segments in plausible range


def log(msg: str) -> None:
    print(f"[extract_features] {msg}", flush=True)


# -- single-window extraction --------------------------------------------------

def _extract_one_window(
    session_path: Path,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> Dict[str, float]:
    """Load BVP, EDA, ACC from a session, slice by timestamps, extract features."""
    features: Dict[str, float] = {}

    # BVP
    try:
        bvp = load_signal(session_path, "BVP")
        if len(bvp) > 0:
            mask = (bvp.index >= t_start) & (bvp.index <= t_end)
            bvp_win = bvp.loc[mask, "bvp"].values
            features.update(extract_hrv_features(bvp_win, sr_hz=int(bvp.attrs.get("sample_rate_hz", 64))))
        else:
            features.update(extract_hrv_features(np.array([]), sr_hz=64))
    except Exception as e:
        features.update(extract_hrv_features(np.array([]), sr_hz=64))

    # EDA
    try:
        eda = load_signal(session_path, "EDA")
        if len(eda) > 0:
            mask = (eda.index >= t_start) & (eda.index <= t_end)
            eda_win = eda.loc[mask, "eda"].values
            features.update(extract_eda_features(eda_win, sr_hz=int(eda.attrs.get("sample_rate_hz", 4))))
        else:
            features.update(extract_eda_features(np.array([]), sr_hz=4))
    except Exception:
        features.update(extract_eda_features(np.array([]), sr_hz=4))

    # ACC
    try:
        acc = load_signal(session_path, "ACC")
        if len(acc) > 0:
            mask = (acc.index >= t_start) & (acc.index <= t_end)
            acc_win = acc.loc[mask, ["acc_x", "acc_y", "acc_z"]].values
            features.update(extract_activity_features(acc_win, sr_hz=int(acc.attrs.get("sample_rate_hz", 32))))
        else:
            features.update(extract_activity_features(np.empty((0, 3)), sr_hz=32))
    except Exception:
        features.update(extract_activity_features(np.empty((0, 3)), sr_hz=32))

    # Feature-level QC
    bvp_q = features.get("bvp_quality_pct", 0.0)
    eda_q = features.get("eda_quality_pct", 0.0)
    features["feature_usable"] = bool((bvp_q >= BVP_QC_MEDIAN) and (eda_q >= EDA_QC_MEDIAN))
    return features


# -- batch extraction ----------------------------------------------------------

def extract_all(windows_df: pd.DataFrame, only_usable_final: bool = True) -> pd.DataFrame:
    """
    Extract features for every EMA window in windows_df.

    If only_usable_final=True, only windows with usable_final=True are processed;
    the others get NaN features (for a consistent row-aligned output).
    """
    n = len(windows_df)
    log(f"Extracting features for {n} windows "
        f"(processing only usable_final={'yes' if only_usable_final else 'all'})")

    feature_rows = []
    for _, w in tqdm(windows_df.iterrows(), total=n, desc="  windows"):
        # Skip if not usable_final and only_usable_final=True
        if only_usable_final and not w.get("usable_final", False):
            feature_rows.append({"feature_usable": False})
            continue
        if not w["session_path"]:
            feature_rows.append({"feature_usable": False})
            continue

        feats = _extract_one_window(
            Path(w["session_path"]),
            pd.Timestamp(w["window_start_utc"]),
            pd.Timestamp(w["window_end_utc"]),
        )
        feature_rows.append(feats)

    feats_df = pd.DataFrame(feature_rows)
    # Join back onto the original windows table, preserving row order
    out = pd.concat([windows_df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
    return out


# -- standardisation -----------------------------------------------------------

FEATURE_COLS = [
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_meannn",
    "hrv_lf", "hrv_hf", "hrv_lfhf",
    "hrv_sd1", "hrv_sd2", "hrv_sd1sd2",
    "eda_scl_mean", "eda_scl_slope",
    "eda_scr_count", "eda_scr_mean_amp", "eda_scr_rate_per_min",
    "eda_symp_activation",
    "acc_magnitude_mean", "acc_magnitude_std", "acc_activity_count",
    "acc_pct_sedentary", "acc_pct_active",
]


def add_standardised_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add within-subject z-scores (_wz) and between-subject means (_bm).

    Within-subject z-score: (x - subject_mean) / subject_sd
    Between-subject mean:   subject_mean (repeated per row)

    These enable disentangling between- and within-person effects in MLM
    (Curran & Bauer, 2011).
    """
    out = df.copy()
    for col in FEATURE_COLS:
        if col not in out.columns:
            continue
        grouped = out.groupby("subject_id")[col]
        subj_mean = grouped.transform("mean")
        subj_sd   = grouped.transform("std")
        out[f"{col}_wz"] = (out[col] - subj_mean) / subj_sd
        out[f"{col}_bm"] = subj_mean
    return out


# -- QC report -----------------------------------------------------------------

def build_qc_report(feats_df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject summary of feature extraction QC."""
    rows = []
    for sid, grp in feats_df.groupby("subject_id"):
        n = len(grp)
        n_feat_ok = int(grp["feature_usable"].sum())
        rows.append({
            "subject_id": int(sid),
            "n_windows": n,
            "n_feature_usable": n_feat_ok,
            "pct_feature_usable": round(100.0 * n_feat_ok / n, 1) if n else np.nan,
            "median_bvp_quality_pct": round(float(grp["bvp_quality_pct"].median()), 2),
            "median_eda_quality_pct": round(float(grp["eda_quality_pct"].median()), 2),
            "median_n_peaks":         round(float(grp["n_peaks_detected"].median()), 0),
            "pct_nan_hrv_hf":         round(100.0 * float(grp["hrv_hf"].isna().mean()), 1),
            "pct_nan_scr":            round(100.0 * float(grp["eda_scr_count"].isna().mean()), 1),
        })
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


# -- main ----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ADARP phase-15b feature extraction")
    parser.add_argument("--all", action="store_true",
                        help="Process all windows, not just usable_final ones.")
    parser.add_argument("--demo", action="store_true",
                        help="Read from data/demo and write there.")
    parser.add_argument("--standardize", action="store_true", default=True,
                        help="Compute within-subject z-scores and between-subject means (default: on).")
    args = parser.parse_args(argv)

    data_dir = DEMO_DIR if args.demo else PROCESSED_DIR
    in_path  = data_dir / "ema_windows_qc.parquet"
    out_path = data_dir / "physio_features.parquet"
    qc_path  = data_dir / "extraction_qc_report.parquet"

    if not in_path.exists():
        log(f"ERROR: input not found: {in_path}")
        log("Run prepare_data.py first.")
        return 1

    log(f"Reading windows from {in_path}")
    windows = pd.read_parquet(in_path)
    log(f"  -> {len(windows)} windows total, "
        f"{int(windows['usable_final'].sum())} usable_final")

    t0 = time.time()
    feats = extract_all(windows, only_usable_final=not args.all)
    log(f"  -> extraction done in {time.time() - t0:.1f}s")

    if args.standardize:
        log("Adding within-subject z-scores and between-subject means")
        feats = add_standardised_columns(feats)

    feats.to_parquet(out_path, index=False)
    log(f"Features written: {out_path} ({len(feats)} rows, {feats.shape[1]} columns)")

    qc = build_qc_report(feats[feats["usable_final"] == True])
    qc.to_parquet(qc_path, index=False)
    log(f"QC report written: {qc_path}")
    log("")
    log("QC summary:")
    log(qc.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
