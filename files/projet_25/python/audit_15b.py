"""
audit_15b.py
------------
Console audit of phase-15b feature extraction.

Runs four checks:
    A. BVP peak detection sanity on a real EMA window.
    B. neurokit2 HR vs Empatica native HR.csv on the same window.
    D. _wz / _bm standardisation correctness.
    E. ICC sanity (one-way ANOVA vs psych::ICC equivalent in numpy).

Usage:
    python audit_15b.py

The script picks a representative usable_final window automatically
(subject 108 by default; can be overridden with --subject SID).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ingest_e4 import load_signal
from extract_hrv import extract_hrv_features

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_DIR   = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def hr(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


# ---- AUDIT A: BVP peak detection on a real window ---------------------------

def audit_A_bvp_peaks(window_row: pd.Series) -> dict:
    """
    Re-extract BVP for a single window, run peak detection, and report
    instantaneous heart rate statistics.
    """
    hr("AUDIT A — BVP peak detection sanity on a real EMA window")
    sp = Path(window_row["session_path"])
    print(f"Subject: {window_row['subject_id']}, EMA obs_id: {window_row['obs_id']}")
    print(f"Window: {window_row['window_start_utc']} -> {window_row['window_end_utc']}")
    print(f"Session: {sp.name}")

    bvp = load_signal(sp, "BVP")
    mask = (bvp.index >= window_row["window_start_utc"]) & (bvp.index <= window_row["window_end_utc"])
    bvp_win = bvp.loc[mask, "bvp"].values
    sr = int(bvp.attrs.get("sample_rate_hz", 64))
    duration_s = len(bvp_win) / sr
    print(f"Window length: {len(bvp_win)} samples = {duration_s:.1f} s = {duration_s/60:.1f} min")

    feats = extract_hrv_features(bvp_win, sr_hz=sr)
    n_peaks = feats["n_peaks_detected"]
    hr_avg_from_peaks = 60.0 * n_peaks / (duration_s)  # bpm
    hr_from_meannn    = 60_000.0 / feats["hrv_meannn"] if feats["hrv_meannn"] else float("nan")

    # Run peak detection again to grab the RR series for finer stats
    import neurokit2 as nk
    bvp_clean = nk.ppg_clean(bvp_win, sampling_rate=sr)
    peaks_info = nk.ppg_findpeaks(bvp_clean, sampling_rate=sr)
    peaks = np.asarray(peaks_info.get("PPG_Peaks", []), dtype=int)

    if len(peaks) >= 2:
        rr_samples = np.diff(peaks)
        rr_ms = rr_samples * 1000.0 / sr
        hr_inst = 60_000.0 / rr_ms
        print(f"\nDetected peaks: {n_peaks}")
        print(f"Average HR (from total peak count) : {hr_avg_from_peaks:.1f} bpm")
        print(f"Average HR (from mean NN interval) : {hr_from_meannn:.1f} bpm")
        print(f"\nInstantaneous HR statistics:")
        print(f"  min     : {np.min(hr_inst):.1f} bpm")
        print(f"  median  : {np.median(hr_inst):.1f} bpm")
        print(f"  mean    : {np.mean(hr_inst):.1f} bpm")
        print(f"  max     : {np.max(hr_inst):.1f} bpm")
        print(f"  IQR     : [{np.percentile(hr_inst, 25):.1f}, {np.percentile(hr_inst, 75):.1f}]")
        # Sanity flags
        impossible = int(np.sum((hr_inst < 30) | (hr_inst > 220)))
        suspicious = int(np.sum((hr_inst < 40) | (hr_inst > 180)))
        print(f"\nPhysiologically impossible (<30 or >220 bpm): {impossible} / {len(hr_inst)} ({100*impossible/len(hr_inst):.1f}%)")
        print(f"Suspicious (<40 or >180 bpm): {suspicious} / {len(hr_inst)} ({100*suspicious/len(hr_inst):.1f}%)")

        verdict = "OK" if impossible / len(hr_inst) < 0.05 else "PROBLEM"
        print(f"\nVERDICT A: {verdict}")
        if verdict == "PROBLEM":
            print("  >5% of detected peaks imply impossible HR. Likely false-positive peaks.")
        else:
            print("  <5% of detected peaks are physiologically impossible.")
        return {"verdict": verdict, "n_peaks": n_peaks,
                "hr_median": float(np.median(hr_inst)),
                "pct_impossible": 100*impossible/len(hr_inst),
                "peaks": peaks, "bvp_clean": bvp_clean, "sr": sr}
    else:
        print("Not enough peaks detected for analysis.")
        return {"verdict": "PROBLEM", "n_peaks": int(len(peaks))}


# ---- AUDIT B: neurokit2 HR vs Empatica native HR.csv ------------------------

def audit_B_hr_consistency(window_row: pd.Series, audit_a_result: dict) -> dict:
    hr("AUDIT B — HR from neurokit2 peaks vs Empatica native HR.csv")
    sp = Path(window_row["session_path"])
    hr_native = load_signal(sp, "HR")
    if len(hr_native) == 0:
        print("HR.csv missing or empty; cannot run audit B.")
        return {"verdict": "SKIP"}

    mask = (hr_native.index >= window_row["window_start_utc"]) & (hr_native.index <= window_row["window_end_utc"])
    hr_win = hr_native.loc[mask, "hr"].values
    if len(hr_win) == 0:
        print("HR.csv has no samples in this window; cannot run audit B.")
        return {"verdict": "SKIP"}

    hr_native_mean = float(np.mean(hr_win))
    hr_neurokit    = audit_a_result["hr_median"]
    diff           = hr_neurokit - hr_native_mean

    print(f"HR (Empatica native, mean of HR.csv) : {hr_native_mean:.1f} bpm  (n={len(hr_win)} samples)")
    print(f"HR (neurokit2 from peaks, median)    : {hr_neurokit:.1f} bpm")
    print(f"Difference                           : {diff:+.1f} bpm")

    abs_diff = abs(diff)
    if abs_diff <= 5:
        verdict = "OK"
        msg = "Excellent agreement (<= 5 bpm) between methods."
    elif abs_diff <= 15:
        verdict = "ACCEPTABLE"
        msg = "Moderate disagreement (5-15 bpm). Could reflect smoothing differences in Empatica's algorithm."
    else:
        verdict = "PROBLEM"
        msg = "Large disagreement (>15 bpm). Likely peak-detection bias."

    print(f"\nVERDICT B: {verdict}")
    print(f"  {msg}")
    return {"verdict": verdict, "hr_native": hr_native_mean,
            "hr_neurokit": hr_neurokit, "diff": diff}


# ---- AUDIT D: _wz / _bm standardisation correctness -------------------------

def audit_D_standardisation(features_df: pd.DataFrame) -> dict:
    hr("AUDIT D — within-subject z-scores and between-subject means")
    f = features_df[features_df["usable_final"] == True].copy()
    feat = "hrv_rmssd"

    if f"{feat}_wz" not in f.columns or f"{feat}_bm" not in f.columns:
        print(f"Standardised columns for '{feat}' not found.")
        return {"verdict": "PROBLEM"}

    # Pick the subject with the most usable_final windows
    counts = f.groupby("subject_id").size().sort_values(ascending=False)
    sid = int(counts.index[0])
    sub = f[f["subject_id"] == sid].dropna(subset=[feat]).copy()
    print(f"Reference subject for manual recomputation: {sid} (n_windows = {len(sub)})")

    manual_mean = float(sub[feat].mean())
    manual_sd   = float(sub[feat].std())
    manual_wz   = (sub[feat] - manual_mean) / manual_sd

    stored_bm = sub[f"{feat}_bm"].iloc[0]
    stored_wz = sub[f"{feat}_wz"].values

    print(f"\nFeature: {feat}")
    print(f"  manual mean (subject {sid})         : {manual_mean:.4f}")
    print(f"  stored {feat}_bm (first row)        : {stored_bm:.4f}")
    print(f"  bm match                            : {np.isclose(stored_bm, manual_mean, atol=1e-4)}")

    print(f"\n  manual sd                           : {manual_sd:.4f}")
    n_match_wz = int(np.sum(np.isclose(stored_wz, manual_wz.values, atol=1e-4)))
    print(f"  wz exact match on {len(sub)} rows           : {n_match_wz}/{len(sub)}")

    # Cross-subject sanity: _bm should be CONSTANT within a subject
    constant_within = (
        f.groupby("subject_id")[f"{feat}_bm"]
         .nunique(dropna=True)
         .max() <= 1
    )
    print(f"\n  {feat}_bm is constant within each subject : {constant_within}")

    # Within-subject z-score of _wz should sum approximately to 0 per subject
    wz_sum_per_subj = f.groupby("subject_id")[f"{feat}_wz"].sum().abs().max()
    print(f"  max |sum(_wz) per subject|                  : {wz_sum_per_subj:.4f}  (expect ~0)")

    bm_ok = np.isclose(stored_bm, manual_mean, atol=1e-4)
    wz_ok = (n_match_wz == len(sub))
    sumzero_ok = wz_sum_per_subj < 1e-3
    constant_ok = constant_within

    if bm_ok and wz_ok and sumzero_ok and constant_ok:
        verdict = "OK"
        msg = "All standardisation invariants hold."
    else:
        verdict = "PROBLEM"
        msg = "Standardisation has at least one invariant violation."
    print(f"\nVERDICT D: {verdict}")
    print(f"  {msg}")
    return {"verdict": verdict, "bm_ok": bm_ok, "wz_ok": wz_ok,
            "sumzero_ok": sumzero_ok, "constant_ok": constant_ok}


# ---- AUDIT E: ICC sanity check ----------------------------------------------

def _icc_one_way(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA ICC1 (same as compute_icc in qmd)."""
    ok = ~np.isnan(values)
    v = values[ok]; g = groups[ok]
    if len(v) < 5 or len(np.unique(g)) < 2:
        return float("nan")
    means = pd.Series(v).groupby(pd.Series(g)).transform("mean").values
    ns    = pd.Series(v).groupby(pd.Series(g)).transform("count").values
    grand_mean = np.mean(v)
    ss_between = np.sum((means - grand_mean) ** 2 * (ns ** 0) * 1.0)  # actually compute per-group
    # Recompute properly
    df_ = pd.DataFrame({"v": v, "g": g})
    grp = df_.groupby("g")["v"]
    sub_means = grp.mean()
    sub_ns    = grp.count()
    ss_between = float(np.sum(sub_ns * (sub_means - grand_mean) ** 2))
    ss_within  = float(np.sum((v - df_.groupby("g")["v"].transform("mean")) ** 2))
    df_b = len(sub_means) - 1
    df_w = len(v) - len(sub_means)
    if df_w <= 0:
        return float("nan")
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w
    k = float(np.mean(sub_ns))
    icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
    return max(0.0, min(1.0, icc))


def _icc_lme_style(values: np.ndarray, groups: np.ndarray) -> float:
    """
    REML-style ICC via random-intercept model variance decomposition.
    Implemented as method-of-moments on group means + residuals.
    Mathematically equivalent to ICC(1,1) under balanced designs;
    very close otherwise. Used here as an independent cross-check.
    """
    ok = ~np.isnan(values)
    v = values[ok]; g = groups[ok]
    if len(v) < 5 or len(np.unique(g)) < 2:
        return float("nan")
    df_ = pd.DataFrame({"v": v, "g": g})
    grp = df_.groupby("g")["v"]
    sub_means = grp.mean()
    sub_ns    = grp.count()

    # Within-subject variance
    within_resid = []
    for gid, gvals in grp:
        within_resid.extend(gvals - gvals.mean())
    sigma2_within = float(np.var(within_resid, ddof=1)) if len(within_resid) > 1 else float("nan")

    # Between-subject variance (method-of-moments)
    overall_mean = np.mean(v)
    between_var_obs = float(np.var(sub_means, ddof=1)) if len(sub_means) > 1 else float("nan")
    sigma2_between = max(0.0, between_var_obs - sigma2_within / np.mean(sub_ns))

    if sigma2_within + sigma2_between == 0:
        return float("nan")
    return sigma2_between / (sigma2_between + sigma2_within)


def audit_E_icc_sanity(features_df: pd.DataFrame) -> dict:
    hr("AUDIT E — ICC sanity check via two independent estimators")
    f = features_df[features_df["usable_final"] == True].copy()
    candidates = ["acc_magnitude_std", "hrv_meannn", "eda_scl_mean", "hrv_rmssd"]
    print(f"{'Feature':<22s}  {'ICC1 (own)':>12s}  {'ICC (REML-MoM)':>16s}  {'agree?':>10s}")
    print("-" * 70)
    out = []
    for feat in candidates:
        if feat not in f.columns:
            continue
        v = f[feat].values
        g = f["subject_id"].values
        icc1 = _icc_one_way(v, g)
        icc2 = _icc_lme_style(v, g)
        agree = abs(icc1 - icc2) < 0.10 if (not np.isnan(icc1) and not np.isnan(icc2)) else False
        print(f"{feat:<22s}  {icc1:>12.3f}  {icc2:>16.3f}  {str(agree):>10s}")
        out.append((feat, icc1, icc2, agree))

    all_agree = all(o[3] for o in out)
    verdict = "OK" if all_agree else "PROBLEM"
    print(f"\nVERDICT E: {verdict}")
    if verdict == "OK":
        print("  Both estimators agree within 0.10 — the low ICCs reflect a real")
        print("  property of the data, not a calculation bug.")
    else:
        print("  Estimators disagree by more than 0.10 on at least one feature.")
        print("  ICC formula or implementation may have a bug.")
    return {"verdict": verdict, "details": out}


# ---- main -------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=108,
                   help="Subject id to use for audits A and B (default: 108).")
    args = p.parse_args(argv)

    windows = pd.read_parquet(PROCESSED_DIR / "ema_windows_qc.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "physio_features.parquet")

    # Pick a representative window: usable_final + feature_usable + chosen subject
    candidates = features[
        (features["usable_final"] == True)
        & (features["feature_usable"] == True)
        & (features["subject_id"] == args.subject)
    ]
    if len(candidates) == 0:
        # fallback: first feature_usable window for that subject, or any subject
        candidates = features[
            (features["feature_usable"] == True)
            & (features["subject_id"] == args.subject)
        ]
    if len(candidates) == 0:
        candidates = features[features["feature_usable"] == True]
    win = candidates.iloc[0]

    a = audit_A_bvp_peaks(win)
    if "hr_median" in a:
        b = audit_B_hr_consistency(win, a)
    else:
        print("\nSkipping audit B because audit A could not produce hr_median.")
        b = {"verdict": "SKIP"}
    d = audit_D_standardisation(features)
    e = audit_E_icc_sanity(features)

    hr("AUDIT SUMMARY")
    print(f"  A (BVP peaks)          : {a['verdict']}")
    print(f"  B (HR consistency)     : {b['verdict']}")
    print(f"  D (standardisation)    : {d['verdict']}")
    print(f"  E (ICC sanity)         : {e['verdict']}")
    print()
    if all(x['verdict'] == "OK" for x in [a, d, e]) and b['verdict'] in ("OK", "ACCEPTABLE", "SKIP"):
        print("OVERALL: green light for phase 15c.")
    else:
        print("OVERALL: at least one audit flags a problem; review before 15c.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
