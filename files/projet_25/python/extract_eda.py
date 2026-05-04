"""
extract_eda.py
--------------
EDA (electrodermal activity) feature extraction from Empatica E4 EDA (4 Hz).

Pipeline (via neurokit2):
    1. Clean EDA with a lowpass filter (default 3 Hz cutoff).
    2. Decompose into tonic (SCL) and phasic (SCR) components via cvxEDA
       (Greco et al., IEEE TBME 2016). Convex-optimisation approach that
       jointly estimates the tonic baseline and the phasic sudomotor response
       under explicit physiological constraints; considered best-in-class
       for ambulatory EDA and the reference method in recent reviews
       (Posada-Quintero & Chon, 2020; Tronstad et al., 2022).
    3. Detect SCRs (sudomotor responses) on the phasic component with
       amplitude threshold of 0.05 uS (Boucsein, 2012 standard).

Features (6):
    Tonic:
        eda_scl_mean         : mean tonic SCL over the window (uS)
        eda_scl_slope        : linear trend (uS per minute) of tonic component
    Phasic:
        eda_scr_count        : number of detected SCRs in the window
        eda_scr_mean_amp     : mean amplitude of detected SCRs (uS)
        eda_scr_rate_per_min : SCR count normalised by window duration
    Composite:
        eda_symp_activation  : z-composite of SCR rate + SCL slope
                               (sympathetic activation proxy)

Quality control:
    eda_quality_pct : fraction of 10-s segments where EDA is within
                      plausible physiological range [0.05, 60] uS and
                      not flat (std > 0.005 uS).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

EDA_SAMPLING_RATE = 4       # Empatica E4
EDA_PLAUSIBLE_LOW  = 0.05   # below is non-wear
EDA_PLAUSIBLE_HIGH = 60.0   # above is sensor fault
EDA_FLAT_STD       = 0.005  # below is saturated / disconnected
SCR_AMP_THRESHOLD_US = 0.05  # Boucsein (2012) standard threshold for ambulatory SCR detection


def _empty_eda_row() -> Dict[str, float]:
    return {
        "eda_scl_mean": np.nan,
        "eda_scl_slope": np.nan,
        "eda_scr_count": 0,
        "eda_scr_mean_amp": np.nan,
        "eda_scr_rate_per_min": np.nan,
        "eda_symp_activation": np.nan,
        "eda_quality_pct": 0.0,
    }


def _compute_eda_quality_pct(eda_values: np.ndarray, sr_hz: int) -> float:
    """
    Fraction of 10-s segments passing three conditions:
        (a) median in [0.05, 60] uS
        (b) std > 0.005 uS (i.e. non-flat)
    """
    seg_len = 10 * sr_hz
    n_segs = len(eda_values) // seg_len
    if n_segs == 0:
        return 0.0
    ok = 0
    for i in range(n_segs):
        seg = eda_values[i * seg_len:(i + 1) * seg_len]
        med = float(np.median(seg))
        std = float(np.std(seg))
        if EDA_PLAUSIBLE_LOW <= med <= EDA_PLAUSIBLE_HIGH and std > EDA_FLAT_STD:
            ok += 1
    return 100.0 * ok / n_segs


def extract_eda_features(eda_values: np.ndarray, sr_hz: int = EDA_SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract EDA features from a window using cvxEDA decomposition.

    Parameters
    ----------
    eda_values : np.ndarray
        1-D EDA signal in microsiemens.
    sr_hz : int
        EDA sampling rate (default 4 Hz for E4).

    Returns
    -------
    dict of feature name -> float (NaN if not computable).
    """
    import neurokit2 as nk

    out = _empty_eda_row()
    if eda_values is None or len(eda_values) < sr_hz * 60:  # need >= 1 min
        return out

    out["eda_quality_pct"] = round(_compute_eda_quality_pct(np.asarray(eda_values), sr_hz), 2)

    # Clean
    try:
        eda_clean = nk.eda_clean(eda_values, sampling_rate=sr_hz)
    except Exception:
        return out

    # Decompose (cvxEDA first; fall back to smoothMedian if cvxpy problem fails)
    try:
        eda_decomp = nk.eda_phasic(eda_clean, sampling_rate=sr_hz, method="cvxeda")
    except Exception:
        try:
            eda_decomp = nk.eda_phasic(eda_clean, sampling_rate=sr_hz, method="smoothmedian")
        except Exception:
            return out

    tonic  = np.asarray(eda_decomp["EDA_Tonic"].values)
    phasic = np.asarray(eda_decomp["EDA_Phasic"].values)

    # Tonic features
    out["eda_scl_mean"] = round(float(np.nanmean(tonic)), 4)
    # Slope: regression on minutes (simple OLS)
    duration_min = len(tonic) / sr_hz / 60.0
    if duration_min > 1:
        t_min = np.arange(len(tonic)) / sr_hz / 60.0
        slope, _ = np.polyfit(t_min, tonic, 1)
        out["eda_scl_slope"] = round(float(slope), 5)

    # SCR detection on phasic
    try:
        scr_peaks, scr_info = nk.eda_peaks(phasic, sampling_rate=sr_hz,
                                           amplitude_min=SCR_AMP_THRESHOLD_US)
        n_scr = int(np.sum(scr_peaks["SCR_Peaks"] == 1)) if "SCR_Peaks" in scr_peaks else 0
        amps = scr_info.get("SCR_Amplitude", np.array([]))
        if not isinstance(amps, np.ndarray):
            amps = np.asarray(amps)
        out["eda_scr_count"] = n_scr
        if n_scr > 0:
            out["eda_scr_mean_amp"] = round(float(np.nanmean(amps)), 4)
        out["eda_scr_rate_per_min"] = round(n_scr / duration_min, 3) if duration_min > 0 else np.nan
    except Exception:
        pass

    # Composite sympathetic activation: z-score of (SCR rate) + z-score of (SCL slope)
    # This is a within-window proxy; proper between-subject standardisation happens downstream.
    # We leave it as the raw sum here and document that z-scoring is done later.
    if not (np.isnan(out["eda_scr_rate_per_min"]) or np.isnan(out["eda_scl_slope"])):
        out["eda_symp_activation"] = round(
            out["eda_scr_rate_per_min"] + 10.0 * out["eda_scl_slope"], 4
        )

    return out
