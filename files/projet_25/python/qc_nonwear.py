"""
qc_nonwear.py
-------------
Non-wear detection for Empatica E4 recordings, EDA-based.

Rule (Kleckner et al., 2018; standard in the field):
    A sample is flagged NON-WEAR if EDA < EDA_LOW_THRESHOLD (default 0.05 uS).
    Non-wear *bouts* require contiguous non-wear of at least MIN_NONWEAR_SECONDS.

The rationale: a wristband that is not in skin contact shows near-zero EDA,
because the electrodermal circuit is open. Actual skin contact produces
baseline EDA around 0.5-5 uS.

Functions:
    flag_nonwear(eda_df, ...)   -> eda_df with 'nonwear' bool column
    nonwear_intervals(eda_df)   -> list of (start_ts, end_ts) non-wear bouts
    session_wear_stats(eda_df)  -> dict (duration_h, wear_h, pct_wear)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

EDA_LOW_THRESHOLD_US = 0.05          # microsiemens
MIN_NONWEAR_SECONDS  = 5 * 60        # 5 minutes


def flag_nonwear(
    eda_df: pd.DataFrame,
    threshold_us: float = EDA_LOW_THRESHOLD_US,
    min_seconds: int = MIN_NONWEAR_SECONDS,
) -> pd.DataFrame:
    """
    Flag non-wear samples in an EDA DataFrame indexed by timestamp_utc.

    A sample is non-wear if it belongs to a contiguous run of samples below
    the threshold whose total duration is >= min_seconds.
    """
    if len(eda_df) == 0 or "eda" not in eda_df.columns:
        return eda_df.assign(nonwear=False)

    sr = eda_df.attrs.get("sample_rate_hz", 4.0)
    min_samples = int(min_seconds * sr)

    below = eda_df["eda"].values < threshold_us

    # Label contiguous runs of 'below' and count their lengths.
    # Diff-based run detection: where does the boolean change?
    change_idx = np.where(np.diff(below.astype(int)) != 0)[0] + 1
    run_starts = np.concatenate(([0], change_idx))
    run_ends   = np.concatenate((change_idx, [len(below)]))

    nonwear = np.zeros(len(below), dtype=bool)
    for s, e in zip(run_starts, run_ends):
        if below[s] and (e - s) >= min_samples:
            nonwear[s:e] = True

    out = eda_df.copy()
    out["nonwear"] = nonwear
    return out


def nonwear_intervals(flagged_eda: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Extract list of (start, end) non-wear bouts from a flagged EDA DataFrame."""
    if len(flagged_eda) == 0 or "nonwear" not in flagged_eda.columns:
        return []
    nw = flagged_eda["nonwear"].values
    if not nw.any():
        return []
    ts = flagged_eda.index.values

    change_idx = np.where(np.diff(nw.astype(int)) != 0)[0] + 1
    run_starts = np.concatenate(([0], change_idx))
    run_ends   = np.concatenate((change_idx, [len(nw)]))

    bouts = []
    for s, e in zip(run_starts, run_ends):
        if nw[s]:
            bouts.append((pd.Timestamp(ts[s]), pd.Timestamp(ts[e - 1])))
    return bouts


def session_wear_stats(flagged_eda: pd.DataFrame) -> dict:
    """Return a dict: total_duration_h, wear_h, nonwear_h, pct_wear."""
    if len(flagged_eda) == 0:
        return {"total_duration_h": 0.0, "wear_h": 0.0, "nonwear_h": 0.0, "pct_wear": np.nan}
    sr = flagged_eda.attrs.get("sample_rate_hz", 4.0)
    n = len(flagged_eda)
    n_nw = int(flagged_eda["nonwear"].sum())
    total_h = n / sr / 3600.0
    nonwear_h = n_nw / sr / 3600.0
    wear_h = total_h - nonwear_h
    pct = 100.0 * wear_h / total_h if total_h > 0 else np.nan
    return {
        "total_duration_h": round(total_h, 3),
        "wear_h": round(wear_h, 3),
        "nonwear_h": round(nonwear_h, 3),
        "pct_wear": round(pct, 1),
    }


def window_wear_fraction(
    flagged_eda: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> float:
    """
    Fraction of a [t_start, t_end] window that is 'wear' (1 - fraction of nonwear).
    Returns NaN if the window is entirely outside the EDA index or empty.
    """
    if len(flagged_eda) == 0:
        return np.nan
    mask = (flagged_eda.index >= t_start) & (flagged_eda.index <= t_end)
    sub = flagged_eda.loc[mask]
    if len(sub) == 0:
        return np.nan
    return float(1.0 - sub["nonwear"].mean())
