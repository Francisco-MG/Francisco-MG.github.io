"""
sync_windows.py
---------------
Temporal synchronization between EMA prompts and E4 sessions.

For each EMA entry, find the E4 session that was active at that moment
(i.e. [session_start_utc, session_end_utc] contains the EMA timestamp),
and define a physiological window used for later feature extraction.

Default strategy: 'pre-event' window of 30 minutes ending at the EMA timestamp.
    window_start_utc = ema_timestamp - 30 min
    window_end_utc   = ema_timestamp

We record, per EMA:
    - session_id (row index in sessions_inventory) if matched, else -1
    - window_start_utc / window_end_utc (whatever the match)
    - window_covered_fraction : how much of the 30 min falls inside the session
    - usable : boolean; True if window_covered_fraction >= USABLE_MIN_COVERAGE

Later QC in extract_physio_features (15b) adds non-wear-based rejection on
top of 'usable'.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

PRE_WINDOW_MINUTES = 30
USABLE_MIN_COVERAGE = 0.8  # at least 80 % of the 30 min must fall inside a session


@dataclass(frozen=True)
class WindowSpec:
    pre_window_minutes: int = PRE_WINDOW_MINUTES
    usable_min_coverage: float = USABLE_MIN_COVERAGE


def _find_session(
    ema_ts_utc: pd.Timestamp,
    subject_id: int,
    sessions_df: pd.DataFrame,
) -> Optional[int]:
    """Find first session (by original index) that contains the EMA timestamp."""
    sub = sessions_df[
        (sessions_df["subject_id"] == subject_id)
        & (sessions_df["session_start_utc"] <= ema_ts_utc)
        & (sessions_df["session_end_utc"]   >= ema_ts_utc)
    ]
    if len(sub) == 0:
        return None
    return int(sub.index[0])


def _window_coverage(
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
    sess_start: pd.Timestamp,
    sess_end: pd.Timestamp,
) -> float:
    """Fraction of [win_start, win_end] covered by [sess_start, sess_end]."""
    overlap_start = max(win_start, sess_start)
    overlap_end   = min(win_end, sess_end)
    if overlap_end <= overlap_start:
        return 0.0
    total = (win_end - win_start).total_seconds()
    cov   = (overlap_end - overlap_start).total_seconds()
    return cov / total if total > 0 else 0.0


def build_windows_index(
    ema_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    spec: WindowSpec = WindowSpec(),
) -> pd.DataFrame:
    """
    Build a window index: one row per EMA, annotated with the matching session
    (if any) and a physiological window for later feature extraction.

    Required ema_df columns:
        subject_id, obs_id, databurst, timestamp_utc, stress_event, stress_composite
    Required sessions_df columns:
        subject_id, session_start_utc, session_end_utc, session_path

    Returns the ema_df with added columns:
        session_idx, session_path, window_start_utc, window_end_utc,
        window_covered_fraction, usable
    """
    if len(ema_df) == 0:
        return ema_df.copy()

    win_delta = pd.Timedelta(minutes=spec.pre_window_minutes)

    # Find matching session and compute coverage row-by-row.
    session_idxs: list = []
    coverages: list = []
    sess_paths: list = []
    win_starts: list = []
    win_ends: list = []

    for _, ema in ema_df.iterrows():
        ts = ema["timestamp_utc"]
        sid = ema["subject_id"]

        # The window anchors on the EMA timestamp whatever the session status.
        w_start = ts - win_delta
        w_end   = ts
        win_starts.append(w_start)
        win_ends.append(w_end)

        idx = _find_session(ts, sid, sessions_df)
        if idx is None:
            # Try: any session that overlaps the *window*, not just the EMA instant.
            candidates = sessions_df[
                (sessions_df["subject_id"] == sid)
                & (sessions_df["session_start_utc"] <= w_end)
                & (sessions_df["session_end_utc"]   >= w_start)
            ]
            if len(candidates) == 0:
                session_idxs.append(-1)
                coverages.append(0.0)
                sess_paths.append("")
                continue
            idx = int(candidates.index[0])

        sess_row = sessions_df.loc[idx]
        cov = _window_coverage(
            w_start, w_end,
            sess_row["session_start_utc"], sess_row["session_end_utc"],
        )
        session_idxs.append(idx)
        coverages.append(round(cov, 3))
        sess_paths.append(str(sess_row.get("session_path", "")))

    out = ema_df.copy()
    out["session_idx"] = session_idxs
    out["session_path"] = sess_paths
    out["window_start_utc"] = win_starts
    out["window_end_utc"] = win_ends
    out["window_covered_fraction"] = coverages
    out["usable"] = [c >= spec.usable_min_coverage for c in coverages]
    return out
