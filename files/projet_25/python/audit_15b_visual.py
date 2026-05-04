"""
audit_15b_visual.py
-------------------
Visual audit of phase-15b feature extraction (audit C).

Produces three PNGs in data/processed/audit/:
    - audit_C1_bvp_peaks.png     : BVP signal with detected peaks
    - audit_C2_eda_decomp.png    : raw EDA, tonic SCL, phasic + SCR peaks
    - audit_C3_acc_activity.png  : ACC magnitude with sedentary/active epochs

Look at these by eye. Each plot includes a verdict hint at the top.

Usage:
    python audit_15b_visual.py
    python audit_15b_visual.py --subject 108
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ingest_e4 import load_signal

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_DIR   = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
AUDIT_DIR     = PROCESSED_DIR / "audit"


def plot_C1_bvp(window_row: pd.Series, out_path: Path) -> None:
    import neurokit2 as nk
    sp = Path(window_row["session_path"])
    bvp = load_signal(sp, "BVP")
    mask = (bvp.index >= window_row["window_start_utc"]) & (bvp.index <= window_row["window_end_utc"])
    bvp_win = bvp.loc[mask, "bvp"].values
    sr = int(bvp.attrs.get("sample_rate_hz", 64))

    bvp_clean = nk.ppg_clean(bvp_win, sampling_rate=sr)
    peaks_info = nk.ppg_findpeaks(bvp_clean, sampling_rate=sr)
    peaks = np.asarray(peaks_info.get("PPG_Peaks", []), dtype=int)

    # Plot only first 60 seconds for readability
    n_show = min(60 * sr, len(bvp_clean))
    t = np.arange(n_show) / sr
    peaks_in_view = peaks[peaks < n_show]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, bvp_clean[:n_show], lw=0.6, color="steelblue", label="BVP (cleaned)")
    ax.scatter(peaks_in_view / sr, bvp_clean[peaks_in_view],
               color="red", s=15, label=f"Detected peaks (n={len(peaks_in_view)} in 60 s)")
    hr_60s = len(peaks_in_view) * (60.0 / (n_show / sr))
    ax.set_title(
        f"AUDIT C1 — BVP peak detection (subject {window_row['subject_id']}, "
        f"obs {window_row['obs_id']})\n"
        f"VERDICT HINT: peaks should align with systolic upstrokes. "
        f"Inferred HR over visible 60 s = {hr_60s:.0f} bpm — physiologically credible if 50-110."
    )
    ax.set_xlabel("Time (s, from window start)")
    ax.set_ylabel("BVP (cleaned, a.u.)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_C2_eda(window_row: pd.Series, out_path: Path) -> None:
    import neurokit2 as nk
    sp = Path(window_row["session_path"])
    eda = load_signal(sp, "EDA")
    mask = (eda.index >= window_row["window_start_utc"]) & (eda.index <= window_row["window_end_utc"])
    eda_win = eda.loc[mask, "eda"].values
    sr = int(eda.attrs.get("sample_rate_hz", 4))

    eda_clean = nk.eda_clean(eda_win, sampling_rate=sr)
    try:
        decomp = nk.eda_phasic(eda_clean, sampling_rate=sr, method="cvxeda")
    except Exception:
        decomp = nk.eda_phasic(eda_clean, sampling_rate=sr, method="smoothmedian")
    tonic  = decomp["EDA_Tonic"].values
    phasic = decomp["EDA_Phasic"].values

    scr_peaks_info, scr_info = nk.eda_peaks(phasic, sampling_rate=sr, amplitude_min=0.05)
    scr_peak_idx = np.where(scr_peaks_info["SCR_Peaks"] == 1)[0]

    t = np.arange(len(eda_clean)) / sr / 60.0  # minutes
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, eda_clean, color="darkgreen", lw=0.8)
    axes[0].set_ylabel("EDA raw (µS)"); axes[0].set_title(
        f"AUDIT C2 — EDA decomposition (subject {window_row['subject_id']}, "
        f"obs {window_row['obs_id']})\n"
        f"VERDICT HINT: red dots should sit on phasic peaks (bumps), not on noise."
    )

    axes[1].plot(t, tonic, color="navy", lw=1.0, label="Tonic SCL (cvxEDA)")
    axes[1].set_ylabel("SCL (µS)")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, phasic, color="purple", lw=0.8, label="Phasic component")
    axes[2].scatter(scr_peak_idx / sr / 60.0, phasic[scr_peak_idx],
                    color="red", s=25, zorder=5,
                    label=f"SCRs (amp ≥ 0.05 µS, n={len(scr_peak_idx)})")
    axes[2].set_ylabel("Phasic (µS)")
    axes[2].set_xlabel("Time (min, from window start)")
    axes[2].axhline(0, color="grey", lw=0.5)
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_C3_acc(window_row: pd.Series, out_path: Path) -> None:
    sp = Path(window_row["session_path"])
    acc = load_signal(sp, "ACC")
    mask = (acc.index >= window_row["window_start_utc"]) & (acc.index <= window_row["window_end_utc"])
    acc_win = acc.loc[mask, ["acc_x", "acc_y", "acc_z"]].values
    sr = int(acc.attrs.get("sample_rate_hz", 32))
    acc_g = acc_win / 64.0  # to g
    magnitude = np.sqrt(np.sum(acc_g ** 2, axis=1))
    magnitude_no_g = magnitude - np.mean(magnitude)
    t = np.arange(len(magnitude_no_g)) / sr / 60.0

    # Per-10-s epoch counts
    epoch_len = sr * 10
    n_epochs = len(magnitude_no_g) // epoch_len
    counts = np.empty(n_epochs)
    for i in range(n_epochs):
        seg = magnitude_no_g[i * epoch_len:(i + 1) * epoch_len]
        counts[i] = np.sum(np.abs(seg - np.mean(seg))) * 100.0
    epoch_t = np.arange(n_epochs) * 10.0 / 60.0  # in minutes

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(t, magnitude_no_g, color="darkorange", lw=0.4)
    axes[0].set_ylabel("|ACC| - g (g)")
    axes[0].set_title(
        f"AUDIT C3 — Accelerometer activity (subject {window_row['subject_id']}, "
        f"obs {window_row['obs_id']})\n"
        f"VERDICT HINT: sedentary epochs should look flat, active should oscillate."
    )

    colors = ["lightblue" if c < 500 else ("orange" if c < 1000 else "red") for c in counts]
    axes[1].bar(epoch_t, counts, width=10/60.0, color=colors, edgecolor="grey", lw=0.3)
    axes[1].axhline(500,  color="grey",  lw=0.5, linestyle="--", label="Sedentary < 500")
    axes[1].axhline(1000, color="black", lw=0.5, linestyle="--", label="Active >= 1000")
    axes[1].set_ylabel("Activity count (10 s epoch)")
    axes[1].set_xlabel("Time (min, from window start)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=108)
    args = p.parse_args(argv)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(PROCESSED_DIR / "physio_features.parquet")

    cands = features[(features["usable_final"] == True)
                     & (features["feature_usable"] == True)
                     & (features["subject_id"] == args.subject)]
    if len(cands) == 0:
        cands = features[features["feature_usable"] == True]
    win = cands.iloc[0]

    out1 = AUDIT_DIR / "audit_C1_bvp_peaks.png"
    out2 = AUDIT_DIR / "audit_C2_eda_decomp.png"
    out3 = AUDIT_DIR / "audit_C3_acc_activity.png"

    print(f"Generating visual audits for subject {win['subject_id']}, obs {win['obs_id']}")
    print(f"  -> {out1}")
    plot_C1_bvp(win, out1)
    print(f"  -> {out2}")
    plot_C2_eda(win, out2)
    print(f"  -> {out3}")
    plot_C3_acc(win, out3)
    print("\nOpen these PNGs and inspect visually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
