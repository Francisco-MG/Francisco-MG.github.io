"""
extract_activity.py
-------------------
Actigraphy feature extraction from Empatica E4 3-axis accelerometer (32 Hz).

ACC units are 1/64 g per axis (Empatica convention). We convert to g and
work with the magnitude of the vector after removing the gravity component.

Pipeline:
    1. Convert to g: x/64, y/64, z/64.
    2. Compute vector magnitude: sqrt(x^2 + y^2 + z^2).
    3. Remove gravity with a simple highpass (mean subtraction per window).
    4. Compute "activity counts" per 10-s epoch (Choi et al., 2011):
         - Sum of absolute deviations from the mean per epoch.
    5. Classify epochs:
         - Sedentary : activity_count <  500
         - Active    : activity_count >= 1000
         - Light     : else

Features (5):
    acc_magnitude_mean   : mean of gravity-removed magnitude (g)
    acc_magnitude_std    : standard deviation of magnitude (g)
    acc_activity_count   : mean per-10s activity count over the window
    acc_pct_sedentary    : percentage of 10-s epochs classified sedentary
    acc_pct_active       : percentage of 10-s epochs classified active

Note:
    Choi et al. (2011) cutpoints are originally for waist-worn ActiGraph
    devices at 100 Hz; thresholds here are rescaled heuristics for
    wrist-worn E4 at 32 Hz. The relative ordering (sedentary < active)
    is preserved and valid for within-subject contrast, which is what
    matters for MLM. Absolute cutpoints should not be compared to
    ActiGraph literature.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

ACC_SAMPLING_RATE = 32
ACC_UNIT_TO_G = 1.0 / 64.0
EPOCH_SECONDS = 10
CUTPOINT_SEDENTARY = 500
CUTPOINT_ACTIVE    = 1000


def _empty_acc_row() -> Dict[str, float]:
    return {
        "acc_magnitude_mean": np.nan,
        "acc_magnitude_std":  np.nan,
        "acc_activity_count": np.nan,
        "acc_pct_sedentary":  np.nan,
        "acc_pct_active":     np.nan,
    }


def extract_activity_features(
    acc_xyz: np.ndarray,
    sr_hz: int = ACC_SAMPLING_RATE,
) -> Dict[str, float]:
    """
    Extract actigraphy features from a 3-axis ACC window.

    Parameters
    ----------
    acc_xyz : np.ndarray of shape (n, 3)
        Raw ACC samples in E4 units (1/64 g per axis).
    sr_hz : int
        ACC sampling rate (default 32 Hz for E4).

    Returns
    -------
    dict of feature name -> float.
    """
    out = _empty_acc_row()
    if acc_xyz is None or len(acc_xyz) < sr_hz * EPOCH_SECONDS:
        return out

    acc = np.asarray(acc_xyz) * ACC_UNIT_TO_G  # to g

    # Magnitude, gravity removed
    magnitude = np.sqrt(np.sum(acc ** 2, axis=1))
    magnitude_no_gravity = magnitude - np.mean(magnitude)

    out["acc_magnitude_mean"] = round(float(np.mean(np.abs(magnitude_no_gravity))), 5)
    out["acc_magnitude_std"]  = round(float(np.std(magnitude_no_gravity)), 5)

    # Per-epoch activity counts
    epoch_len = sr_hz * EPOCH_SECONDS
    n_epochs = len(magnitude_no_gravity) // epoch_len
    if n_epochs == 0:
        return out

    counts = np.empty(n_epochs)
    for i in range(n_epochs):
        seg = magnitude_no_gravity[i * epoch_len:(i + 1) * epoch_len]
        # Sum of absolute deviations, scaled to give an ActiGraph-like magnitude
        counts[i] = np.sum(np.abs(seg - np.mean(seg))) * 100.0

    out["acc_activity_count"] = round(float(np.mean(counts)), 2)
    out["acc_pct_sedentary"]  = round(100.0 * float(np.mean(counts <  CUTPOINT_SEDENTARY)), 2)
    out["acc_pct_active"]     = round(100.0 * float(np.mean(counts >= CUTPOINT_ACTIVE)),    2)

    return out
