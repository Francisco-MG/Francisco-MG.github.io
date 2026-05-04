"""
ingest_ema.py
-------------
Parser for the 'Final Phone Survey Stress Events.xlsm' file.

This file is a hybrid: it mixes a codebook (rows 0-19) with per-subject
EMA-like stress event reports (rows 20-151), then a completion summary.

Per-subject sections are delimited by a 'Caseid=XXX' row followed by a
header row 'Obs, newdate, time submitted, Databurst, more_stressed, ...'
and then data rows until an empty row.

Variables decoded (from rows 0-19 codebook):
    more_stressed   : 1 = yes, 2 = no          -> recoded to 1/0
    more_overwhelm  : 1 = yes, 2 = no          -> recoded to 1/0
    more_anxious    : 1 = yes, 2 = no          -> recoded to 1/0
    Databurst       : 1 = morning, 2 = midday, 3 = afternoon, 4 = evening
    reason_stress   : 1..9 categorical stress reason

Outcome variables produced:
    stress_event      (binary) : 1 if more_stressed == 1, 0 otherwise
    stress_composite  (0..3)   : sum of more_stressed + more_overwhelm + more_anxious
                                 (after recoding 2 -> 0)

TIMESTAMP RECONSTRUCTION - IMPORTANT LIMITATION:
    Inspection of the raw xlsm shows that the 'time submitted' column has
    number_format 'mmss.0' (minutes:seconds), NOT an hour-of-day.
    Its values are survey-completion offsets (e.g. "10:25.59" means the
    participant took 10 minutes 25.59 seconds to complete the survey), not
    clock times. The actual hour-of-day of each EMA prompt is therefore
    NOT DIRECTLY AVAILABLE in the public Stress Events file.

    We reconstruct a canonical local timestamp per EMA using the 'databurst'
    column, which the codebook documents as:
        1 = morning     -> imputed 09:00 local
        2 = midday      -> imputed 12:00 local
        3 = afternoon   -> imputed 15:00 local
        4 = evening     -> imputed 20:00 local

    This is a conservative approximation and is documented in the DMP.
    Downstream windowing (sync_windows) should take this into account:
    the 30-min 'pre-event' window may not precisely cover the actual moment
    of stress; we therefore widen QC tolerance in later feature extraction.

    Local timezone: America/Los_Angeles (WSU Pullman, WA).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Los_Angeles"  # WSU Pullman, WA

# Canonical local hour (24h) imputed for each databurst value (codebook).
# Used because 'time submitted' in the xlsm is a survey-completion offset
# (mm:ss), not an hour-of-day.
DATABURST_LOCAL_HOUR = {
    1: 9,   # morning
    2: 12,  # midday
    3: 15,  # afternoon
    4: 20,  # evening
}

# Column indices in the raw xlsm (0-based) after reading with header=None.
# Determined by inspection of the 'Obs ... time_stress_4' header rows.
COL_OBS          = 0
COL_NEWDATE      = 1
COL_TIME         = 2
COL_DATABURST    = 3
COL_MORESTRESSED = 4
COL_MOREOVERWHELM= 5
COL_MOREANXIOUS  = 6
COL_REASONSTRESS = 7
COL_REASONOTHER  = 8
COL_TSTRESS_1    = 9
COL_TSTRESS_2    = 10
COL_TSTRESS_3    = 11
COL_TSTRESS_4    = 12


def _is_caseid_row(row: pd.Series) -> Optional[int]:
    """Return subject id if row[0] is like 'Caseid=104', else None."""
    val = row.iloc[0]
    if isinstance(val, str) and "caseid" in val.lower():
        try:
            return int(val.split("=")[1].strip())
        except (IndexError, ValueError):
            return None
    return None


def _is_data_row(row: pd.Series) -> bool:
    """A data row has a numeric Obs in column 0 and a real date in column 1."""
    obs = row.iloc[COL_OBS]
    newdate = row.iloc[COL_NEWDATE]
    return (
        pd.notna(obs)
        and pd.notna(newdate)
        and isinstance(obs, (int, float, np.integer, np.floating))
        and not isinstance(obs, bool)
    )


def _recode_binary(v) -> float:
    """1 -> 1, 2 -> 0, '.'/NaN/anything else -> NaN."""
    if pd.isna(v):
        return np.nan
    try:
        iv = int(v)
        if iv == 1:
            return 1.0
        if iv == 2:
            return 0.0
    except (ValueError, TypeError):
        pass
    return np.nan


def _combine_date_databurst(newdate, databurst) -> Optional[pd.Timestamp]:
    """
    Build a local-wall-clock Timestamp from a date and a databurst code.
    The original 'time submitted' field is a survey duration (mm:ss), not
    an hour-of-day, so it cannot be used to reconstruct the true prompt time.
    We impute a canonical hour per databurst (see DATABURST_LOCAL_HOUR).
    """
    if pd.isna(newdate) or pd.isna(databurst):
        return None
    try:
        date_part = pd.to_datetime(newdate).date()
    except Exception:
        return None
    try:
        db = int(databurst)
    except (ValueError, TypeError):
        return None
    hour = DATABURST_LOCAL_HOUR.get(db)
    if hour is None:
        return None
    return pd.Timestamp.combine(date_part, pd.Timestamp(f"2000-01-01 {hour:02d}:00:00").time())


def load_ema_stress_events(xlsm_path: Path) -> pd.DataFrame:
    """
    Parse the Stress Events xlsm into a tidy long DataFrame.

    Returns columns:
        subject_id, obs_id, databurst, timestamp_local, timestamp_utc,
        more_stressed, more_overwhelm, more_anxious,
        stress_event, stress_composite,
        reason_stress, reason_other
    """
    xlsm_path = Path(xlsm_path)
    if not xlsm_path.exists():
        raise FileNotFoundError(f"EMA file not found: {xlsm_path}")

    raw = pd.read_excel(xlsm_path, sheet_name=0, header=None, engine="openpyxl")

    current_subject: Optional[int] = None
    records = []

    for _, row in raw.iterrows():
        sid = _is_caseid_row(row)
        if sid is not None:
            current_subject = sid
            continue

        if current_subject is None:
            continue

        if not _is_data_row(row):
            continue

        ts_local = _combine_date_databurst(row.iloc[COL_NEWDATE], row.iloc[COL_DATABURST])
        if ts_local is None:
            continue

        ms  = _recode_binary(row.iloc[COL_MORESTRESSED])
        mo  = _recode_binary(row.iloc[COL_MOREOVERWHELM])
        ma  = _recode_binary(row.iloc[COL_MOREANXIOUS])

        composite_vals = [v for v in (ms, mo, ma) if not pd.isna(v)]
        composite = sum(composite_vals) if composite_vals else np.nan

        records.append({
            "subject_id": current_subject,
            "obs_id": int(row.iloc[COL_OBS]),
            "databurst": int(row.iloc[COL_DATABURST]) if pd.notna(row.iloc[COL_DATABURST]) else np.nan,
            "timestamp_local": ts_local,
            "more_stressed":  ms,
            "more_overwhelm": mo,
            "more_anxious":   ma,
            "stress_event":   1.0 if ms == 1.0 else (0.0 if ms == 0.0 else np.nan),
            "stress_composite": composite,
            "reason_stress":  row.iloc[COL_REASONSTRESS] if pd.notna(row.iloc[COL_REASONSTRESS]) else np.nan,
            "reason_other":   row.iloc[COL_REASONOTHER]  if pd.notna(row.iloc[COL_REASONOTHER])  else np.nan,
        })

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df

    # Coerce mixed-type columns to stable types for Parquet compatibility.
    # reason_other may contain numeric codes (-1, -3 for 'not applicable')
    # or free-text strings (when reason_stress == 9 'other').
    df["reason_other"] = df["reason_other"].astype(str).where(df["reason_other"].notna(), None)
    df["reason_stress"] = pd.to_numeric(df["reason_stress"], errors="coerce")

    # Localize to America/Los_Angeles, then convert to UTC.
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    df["timestamp_utc"] = (
        df["timestamp_local"]
        .dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )

    df = df.sort_values(["subject_id", "timestamp_utc"]).reset_index(drop=True)
    return df
