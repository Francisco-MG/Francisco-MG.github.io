"""
extract_hrv.py
--------------
HRV feature extraction from Empatica E4 BVP signal (64 Hz).

Pipeline (via neurokit2):
    1. Clean BVP with a bandpass filter (default 0.5 - 8 Hz).
    2. Detect systolic peaks (nk.ppg_findpeaks).
    3. Compute R-R (peak-to-peak) intervals in ms.
    4. Derive HRV indices over the whole window.

Features (10):
    Time-domain (nk.hrv_time):
        hrv_rmssd   : root mean square of successive differences (ms)
        hrv_sdnn    : standard deviation of NN intervals (ms)
        hrv_pnn50   : percentage of NN intervals differing by >50 ms
        hrv_meannn  : mean NN interval (ms)
    Frequency-domain (nk.hrv_frequency, Welch PSD):
        hrv_lf      : low-frequency power (0.04 - 0.15 Hz, ms^2)
        hrv_hf      : high-frequency power (0.15 - 0.40 Hz, ms^2)
        hrv_lfhf    : LF / HF ratio
    Non-linear (nk.hrv_nonlinear, Poincare plot):
        hrv_sd1     : short-term variability (ms)
        hrv_sd2     : long-term variability (ms)
        hrv_sd1sd2  : SD1/SD2 ratio

Quality control:
    n_peaks_detected : number of systolic peaks found in the window
    bvp_quality_pct  : fraction of 10-s segments with acceptable peak density
                       (>= 6 peaks per 10 s = 36 bpm floor)

HRV feature computation requires at least ~120 NN intervals for frequency-
domain metrics to be stable (~2 min of clean signal). Windows with fewer
peaks produce NaN in the frequency/non-linear blocks; time-domain metrics
still compute on as few as 5-10 intervals.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# neurokit2 is imported lazily to keep module import cheap during CI.

HRV_SAMPLING_RATE = 64  # Empatica E4 BVP
BP_LOW_HZ  = 0.5
BP_HIGH_HZ = 8.0
MIN_PEAKS_FOR_FREQ = 30      # below this, return NaN for freq & nonlinear blocks
MIN_SEGMENT_PEAKS_10S = 6    # heart rate ~36 bpm floor for QC


def _empty_hrv_row() -> Dict[str, float]:
    return {
        "hrv_rmssd": np.nan, "hrv_sdnn": np.nan, "hrv_pnn50": np.nan,
        "hrv_meannn": np.nan, "hrv_lf": np.nan, "hrv_hf": np.nan,
        "hrv_lfhf": np.nan, "hrv_sd1": np.nan, "hrv_sd2": np.nan,
        "hrv_sd1sd2": np.nan, "n_peaks_detected": 0, "bvp_quality_pct": 0.0,
    }


def _compute_bvp_quality_pct(peaks: np.ndarray, n_samples: int, sr_hz: int) -> float:
    """Fraction of 10-second segments with >=6 detected peaks."""
    if n_samples == 0:
        return 0.0
    seg_len = 10 * sr_hz
    n_segs = n_samples // seg_len
    if n_segs == 0:
        return 0.0
    ok = 0
    for i in range(n_segs):
        start = i * seg_len
        end = start + seg_len
        count = np.sum((peaks >= start) & (peaks < end))
        if count >= MIN_SEGMENT_PEAKS_10S:
            ok += 1
    return 100.0 * ok / n_segs


def extract_hrv_features(bvp_values: np.ndarray, sr_hz: int = HRV_SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract HRV features from a BVP window.

    Parameters
    ----------
    bvp_values : np.ndarray
        1-D BVP signal for the window.
    sr_hz : int
        BVP sampling rate (default 64 Hz for E4).

    Returns
    -------
    dict of feature name -> float (NaN if not computable).
    """
    import neurokit2 as nk

    out = _empty_hrv_row()
    if bvp_values is None or len(bvp_values) < sr_hz * 30:  # need >= 30 s
        return out

    # Clean & detect systolic peaks
    try:
        bvp_clean = nk.ppg_clean(bvp_values, sampling_rate=sr_hz,
                                 heart_rate=None)
        peaks_info = nk.ppg_findpeaks(bvp_clean, sampling_rate=sr_hz)
        peaks = np.asarray(peaks_info.get("PPG_Peaks", []), dtype=int)
    except Exception:
        return out

    out["n_peaks_detected"] = int(len(peaks))
    out["bvp_quality_pct"] = round(_compute_bvp_quality_pct(peaks, len(bvp_values), sr_hz), 2)

    if len(peaks) < 5:
        return out

    # Build peaks dict expected by nk.hrv_*
    peaks_dict = {"PPG_Peaks": peaks}

    # Time-domain HRV
    try:
        hrv_t = nk.hrv_time(peaks_dict, sampling_rate=sr_hz, show=False)
        out["hrv_rmssd"]  = float(hrv_t.get("HRV_RMSSD",  np.nan).iloc[0]) if "HRV_RMSSD"  in hrv_t else np.nan
        out["hrv_sdnn"]   = float(hrv_t.get("HRV_SDNN",   np.nan).iloc[0]) if "HRV_SDNN"   in hrv_t else np.nan
        out["hrv_pnn50"]  = float(hrv_t.get("HRV_pNN50",  np.nan).iloc[0]) if "HRV_pNN50"  in hrv_t else np.nan
        out["hrv_meannn"] = float(hrv_t.get("HRV_MeanNN", np.nan).iloc[0]) if "HRV_MeanNN" in hrv_t else np.nan
    except Exception:
        pass

    # Frequency + non-linear require more peaks
    if len(peaks) < MIN_PEAKS_FOR_FREQ:
        return out

    try:
        hrv_f = nk.hrv_frequency(peaks_dict, sampling_rate=sr_hz, show=False,
                                 psd_method="welch")
        out["hrv_lf"]   = float(hrv_f.get("HRV_LF",   np.nan).iloc[0]) if "HRV_LF"   in hrv_f else np.nan
        out["hrv_hf"]   = float(hrv_f.get("HRV_HF",   np.nan).iloc[0]) if "HRV_HF"   in hrv_f else np.nan
        out["hrv_lfhf"] = float(hrv_f.get("HRV_LFHF", np.nan).iloc[0]) if "HRV_LFHF" in hrv_f else np.nan
    except Exception:
        pass

    try:
        hrv_nl = nk.hrv_nonlinear(peaks_dict, sampling_rate=sr_hz, show=False)
        sd1 = float(hrv_nl.get("HRV_SD1", np.nan).iloc[0]) if "HRV_SD1" in hrv_nl else np.nan
        sd2 = float(hrv_nl.get("HRV_SD2", np.nan).iloc[0]) if "HRV_SD2" in hrv_nl else np.nan
        out["hrv_sd1"] = sd1
        out["hrv_sd2"] = sd2
        out["hrv_sd1sd2"] = sd1 / sd2 if (sd2 and not np.isnan(sd2) and sd2 != 0) else np.nan
    except Exception:
        pass

    return out
