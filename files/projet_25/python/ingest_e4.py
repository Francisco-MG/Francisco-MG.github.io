"""
ingest_e4.py
------------
Ingestion module for Empatica E4 raw data in ADARP structure.

ADARP sensor data structure:
    Sensor Data/
    ├── Part 101C/
    │   ├── <DeviceID>_<YYMMDD>-<HHMMSS>/
    │   │   ├── ACC.csv, BVP.csv, EDA.csv, HR.csv, IBI.csv, tags.csv, TEMP.csv, info.txt
    │   └── ...
    └── ...

Empatica E4 CSV format (except IBI and tags):
    Row 1: initial timestamp (Unix UTC, seconds)
    Row 2: sample rate (Hz)
    Row 3+: data values (single column, or 3 for ACC)

IBI.csv: no sample rate. Two columns: (seconds_from_start, ibi_duration_seconds).
tags.csv: one column of Unix timestamps (button presses).

Signal sample rates:
    ACC  : 32 Hz (3 axes, unit 1/64 g)
    BVP  : 64 Hz
    EDA  : 4  Hz  (microsiemens)
    HR   : 1  Hz  (derived by E4)
    TEMP : 4  Hz  (Celsius)

Functions:
    parse_session_folder_name(name) -> dict
    scan_sessions(sensor_root)      -> pd.DataFrame  (inventory)
    load_signal(session_path, name) -> pd.DataFrame  (timestamp, value[s])
    load_session(session_path)      -> dict of DataFrames
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# -- constants -----------------------------------------------------------------

EXPECTED_SIGNALS = ["ACC", "BVP", "EDA", "HR", "TEMP"]  # have sampling rate header
EXPECTED_EVENT_FILES = ["IBI", "tags"]                  # no sampling rate header
ALL_E4_FILES = EXPECTED_SIGNALS + EXPECTED_EVENT_FILES

SUBJECT_FOLDER_RE = re.compile(r"^Part\s+(\d+)C?$", re.IGNORECASE)
SESSION_FOLDER_RE = re.compile(r"^(?P<device>[A-Za-z0-9]+)_(?P<date>\d{6})-(?P<time>\d{6})$")


# -- path parsing --------------------------------------------------------------

def parse_subject_folder_name(name: str) -> Optional[int]:
    """Extract subject id from 'Part 101C' -> 101. Return None if not matching."""
    m = SUBJECT_FOLDER_RE.match(name.strip())
    return int(m.group(1)) if m else None


def parse_session_folder_name(name: str) -> Optional[dict]:
    """
    Parse 'A01f4c_190429-184613' ->
        {'device_id': 'A01F4C',
         'session_start_utc': Timestamp('2019-04-29 18:46:13', tz='UTC')}
    Return None if not matching.

    Device IDs are normalised to UPPERCASE to collapse case variants
    (e.g. 'A01d53' and 'A01D53' correspond to the same physical device;
    the variation is a data-entry artifact in ADARP folder naming).
    """
    m = SESSION_FOLDER_RE.match(name.strip())
    if not m:
        return None
    yymmdd = m.group("date")
    hhmmss = m.group("time")
    # Format: YY-MM-DD HH:MM:SS. YY assumed 20YY (ADARP is 2019-2020).
    dt_str = f"20{yymmdd[0:2]}-{yymmdd[2:4]}-{yymmdd[4:6]} {hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
    try:
        ts = pd.Timestamp(dt_str, tz="UTC")
    except Exception:
        return None
    return {"device_id": m.group("device").upper(), "session_start_utc": ts}


# -- CSV reading ---------------------------------------------------------------

def _read_header(csv_path: Path) -> tuple[float, float]:
    """Read rows 1-2 of an E4 signal CSV: (unix_ts_start, sample_rate_hz)."""
    with open(csv_path, "r") as f:
        ts_line = f.readline().strip()
        sr_line = f.readline().strip()
    # For ACC these are 3-column headers with identical values; take the first.
    ts_val = float(ts_line.split(",")[0])
    sr_val = float(sr_line.split(",")[0])
    return ts_val, sr_val


def load_signal(session_path: Path, signal_name: str) -> pd.DataFrame:
    """
    Load a single E4 signal file and return a DataFrame with a UTC timestamp index.

    Returns columns depend on signal:
        ACC  : ['acc_x', 'acc_y', 'acc_z'] (units: 1/64 g)
        BVP  : ['bvp']
        EDA  : ['eda']
        HR   : ['hr']
        TEMP : ['temp']
    """
    if signal_name not in EXPECTED_SIGNALS:
        raise ValueError(f"Unknown signal '{signal_name}'. Use: {EXPECTED_SIGNALS}")

    fpath = session_path / f"{signal_name}.csv"
    if not fpath.exists():
        return pd.DataFrame()

    ts_start, sr_hz = _read_header(fpath)
    # Read data (skip the 2 header rows). ACC has 3 columns.
    data = pd.read_csv(fpath, header=None, skiprows=2)
    n = len(data)
    if n == 0:
        return pd.DataFrame()

    timestamps = pd.to_datetime(
        ts_start + np.arange(n) / sr_hz, unit="s", utc=True
    )

    if signal_name == "ACC":
        data.columns = ["acc_x", "acc_y", "acc_z"]
    else:
        data.columns = [signal_name.lower()]

    data.index = timestamps
    data.index.name = "timestamp_utc"
    data.attrs["sample_rate_hz"] = sr_hz
    data.attrs["session_start_utc"] = pd.Timestamp(ts_start, unit="s", tz="UTC")
    return data


def load_ibi(session_path: Path) -> pd.DataFrame:
    """Load IBI.csv: (seconds_from_start, ibi_seconds). Adds a UTC timestamp."""
    fpath = session_path / "IBI.csv"
    if not fpath.exists():
        return pd.DataFrame()

    with open(fpath, "r") as f:
        first = f.readline().strip()
    ts_start = float(first.split(",")[0])

    df = pd.read_csv(fpath, header=None, skiprows=1, names=["t_offset_s", "ibi_s"])
    if len(df) == 0:
        return pd.DataFrame()

    df["timestamp_utc"] = pd.to_datetime(ts_start + df["t_offset_s"], unit="s", utc=True)
    df = df.set_index("timestamp_utc")
    df.attrs["session_start_utc"] = pd.Timestamp(ts_start, unit="s", tz="UTC")
    return df[["ibi_s"]]


def load_tags(session_path: Path) -> pd.DataFrame:
    """Load tags.csv: button-press Unix timestamps (one per row)."""
    fpath = session_path / "tags.csv"
    if not fpath.exists():
        return pd.DataFrame(columns=["timestamp_utc"])
    try:
        df = pd.read_csv(fpath, header=None, names=["ts"])
        if len(df) == 0:
            return pd.DataFrame(columns=["timestamp_utc"])
        df["timestamp_utc"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df[["timestamp_utc"]]
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["timestamp_utc"])


def load_session(session_path: Path) -> Dict[str, pd.DataFrame]:
    """Load all signals + IBI + tags from one session folder."""
    out: Dict[str, pd.DataFrame] = {}
    for sig in EXPECTED_SIGNALS:
        out[sig] = load_signal(session_path, sig)
    out["IBI"] = load_ibi(session_path)
    out["tags"] = load_tags(session_path)
    return out


# -- session inventory ---------------------------------------------------------

def _session_duration_from_signal(session_path: Path) -> tuple[Optional[pd.Timestamp], float, int]:
    """
    Derive session end + duration from EDA (most reliable, 4 Hz).
    Returns (end_utc, duration_hours, n_samples_eda). If EDA missing, falls back to BVP.
    """
    for sig in ["EDA", "BVP", "TEMP"]:
        fpath = session_path / f"{sig}.csv"
        if not fpath.exists():
            continue
        try:
            ts_start, sr_hz = _read_header(fpath)
            with open(fpath, "r") as f:
                n = sum(1 for _ in f) - 2  # minus the 2 header rows
            if n <= 0:
                continue
            duration_s = n / sr_hz
            end = pd.Timestamp(ts_start + duration_s, unit="s", tz="UTC")
            return end, duration_s / 3600.0, n
        except Exception:
            continue
    return None, 0.0, 0


def scan_sessions(sensor_root: Path) -> pd.DataFrame:
    """
    Walk the ADARP Sensor Data tree and build an inventory of all sessions.

    Returns a DataFrame with one row per session:
        subject_id, device_id, session_start_utc, session_end_utc,
        duration_hours, n_files_ok, missing_files, session_path
    """
    rows: List[dict] = []
    sensor_root = Path(sensor_root)

    if not sensor_root.exists():
        raise FileNotFoundError(f"Sensor root does not exist: {sensor_root}")

    for subject_dir in sorted(sensor_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = parse_subject_folder_name(subject_dir.name)
        if subject_id is None:
            continue

        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            parsed = parse_session_folder_name(session_dir.name)
            if parsed is None:
                continue

            present_files = {f.stem for f in session_dir.glob("*.csv")}
            missing = sorted(set(ALL_E4_FILES) - present_files)

            end_utc, dur_hours, _ = _session_duration_from_signal(session_dir)

            rows.append({
                "subject_id": subject_id,
                "device_id": parsed["device_id"],
                "session_start_utc": parsed["session_start_utc"],
                "session_end_utc": end_utc,
                "duration_hours": round(dur_hours, 2),
                "n_files_ok": len(ALL_E4_FILES) - len(missing),
                "missing_files": ",".join(missing) if missing else "",
                "session_path": str(session_dir),
            })

    df = pd.DataFrame(rows).sort_values(["subject_id", "session_start_utc"]).reset_index(drop=True)
    return df
