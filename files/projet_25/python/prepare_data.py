"""
prepare_data.py
---------------
Orchestrator for the ADARP phase-15a pipeline.

Steps:
    1. Scan ADARP Sensor Data -> sessions_inventory.parquet
    2. Parse Stress Events xlsm -> ema_clean.parquet
    3. Build EMA x session windows -> windows_index.parquet
    4. Per-session EDA non-wear QC -> nonwear_report.parquet
    5. Union of EMA usability + wear fraction -> ema_windows_qc.parquet

Usage:
    python prepare_data.py                    # full ADARP run, writes to data/processed/
    python prepare_data.py --demo SUB1 SUB2   # write a slimmed demo dataset to data/demo/

Paths are configured at the top of the file.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ingest_e4 import scan_sessions, load_signal
from ingest_ema import load_ema_stress_events
from qc_nonwear import flag_nonwear, session_wear_stats
from sync_windows import build_windows_index, WindowSpec

# ---- CONFIGURATION LOCALE ----------------------------------------------------
# Adapt this single variable to your machine.
ADARP_ROOT = Path(r"C:\Users\fmarting\Documents\Travail_Data\6640290")
# ------------------------------------------------------------------------------

SENSOR_DIR = ADARP_ROOT / "Sensor Data"
EMA_FILE   = ADARP_ROOT / "Final Phone Survey Stress Events" / "Final Phone Survey Stress Events.xlsm"

# Output paths (relative to this script's parent = projet_25/)
SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_DIR   = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
DEMO_DIR      = PROJECT_DIR / "data" / "demo"


def log(msg: str) -> None:
    print(f"[prepare_data] {msg}", flush=True)


# -- step 1: sessions inventory ------------------------------------------------

def step_inventory(sensor_dir: Path, out_dir: Path) -> pd.DataFrame:
    log(f"Step 1/4: scanning sessions in {sensor_dir}")
    t0 = time.time()
    sessions = scan_sessions(sensor_dir)
    log(f"  -> {len(sessions)} sessions across {sessions['subject_id'].nunique()} subjects "
        f"({time.time() - t0:.1f}s)")
    out = out_dir / "sessions_inventory.parquet"
    sessions.to_parquet(out, index=False)
    log(f"  written: {out}")
    return sessions


# -- step 2: EMA --------------------------------------------------------------

def step_ema(ema_path: Path, out_dir: Path) -> pd.DataFrame:
    log(f"Step 2/4: parsing EMA stress events from {ema_path.name}")
    ema = load_ema_stress_events(ema_path)
    log(f"  -> {len(ema)} EMA entries from {ema['subject_id'].nunique()} subjects")
    out = out_dir / "ema_clean.parquet"
    ema.to_parquet(out, index=False)
    log(f"  written: {out}")
    return ema


# -- step 3: windows ----------------------------------------------------------

def step_windows(ema: pd.DataFrame, sessions: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    log("Step 3/4: building EMA x session windows (30 min pre-event)")
    windows = build_windows_index(ema, sessions, WindowSpec())
    n_matched = int((windows["session_idx"] >= 0).sum())
    n_usable  = int(windows["usable"].sum())
    log(f"  -> {n_matched}/{len(windows)} EMA matched to a session, "
        f"{n_usable} usable (>=80% window coverage)")
    out = out_dir / "windows_index.parquet"
    windows.to_parquet(out, index=False)
    log(f"  written: {out}")
    return windows


# -- step 4: non-wear QC per session ------------------------------------------

def step_nonwear(sessions: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    log("Step 4/4: EDA-based non-wear QC per session")
    records = []
    for _, sess in tqdm(sessions.iterrows(), total=len(sessions), desc="  sessions"):
        sp = Path(sess["session_path"])
        try:
            eda = load_signal(sp, "EDA")
        except Exception:
            eda = pd.DataFrame()
        if len(eda) == 0:
            records.append({
                "session_idx": int(sess.name),
                "subject_id": int(sess["subject_id"]),
                "total_duration_h": 0.0, "wear_h": 0.0, "nonwear_h": 0.0,
                "pct_wear": np.nan,
            })
            continue
        flagged = flag_nonwear(eda)
        stats = session_wear_stats(flagged)
        records.append({
            "session_idx": int(sess.name),
            "subject_id": int(sess["subject_id"]),
            **stats,
        })
    df = pd.DataFrame(records)
    out = out_dir / "nonwear_report.parquet"
    df.to_parquet(out, index=False)
    log(f"  -> {len(df)} sessions QC'd, median pct_wear = {df['pct_wear'].median():.1f}%")
    log(f"  written: {out}")
    return df


# -- step 5: join windows + wear fraction per EMA window ----------------------

def step_window_qc(windows: pd.DataFrame, sessions: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    For each EMA window with session_idx >= 0, compute the *wear fraction of the
    30 min window itself* (not just of the session). This is a sharper QC.
    """
    log("Step 5: per-window wear fraction (fine-grained QC)")
    wear_frac = []
    # Cache EDA per session to avoid reloading for multiple EMA in the same session.
    eda_cache: dict[int, pd.DataFrame] = {}

    for _, w in tqdm(windows.iterrows(), total=len(windows), desc="  windows"):
        sidx = int(w["session_idx"])
        if sidx < 0:
            wear_frac.append(np.nan)
            continue
        if sidx not in eda_cache:
            sp = Path(sessions.loc[sidx, "session_path"])
            try:
                eda = load_signal(sp, "EDA")
                eda_cache[sidx] = flag_nonwear(eda) if len(eda) else pd.DataFrame()
            except Exception:
                eda_cache[sidx] = pd.DataFrame()
        flagged = eda_cache[sidx]
        if len(flagged) == 0:
            wear_frac.append(np.nan)
            continue
        mask = (flagged.index >= w["window_start_utc"]) & (flagged.index <= w["window_end_utc"])
        sub = flagged.loc[mask]
        if len(sub) == 0:
            wear_frac.append(np.nan)
        else:
            wear_frac.append(float(1.0 - sub["nonwear"].mean()))

    out_df = windows.copy()
    out_df["window_wear_fraction"] = wear_frac
    out_df["usable_final"] = out_df["usable"] & (pd.Series(wear_frac).fillna(0) >= 0.7).values
    out = out_dir / "ema_windows_qc.parquet"
    out_df.to_parquet(out, index=False)
    log(f"  -> {int(out_df['usable_final'].sum())}/{len(out_df)} EMA windows usable_final")
    log(f"  written: {out}")
    return out_df


# -- demo extractor -----------------------------------------------------------

def make_demo(subjects: list[int], full_processed: Path, demo_dir: Path) -> None:
    """
    Build a slim demo dataset in demo_dir from the full processed outputs,
    filtered on the requested subjects.
    """
    log(f"Building demo dataset for subjects {subjects}")
    demo_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["sessions_inventory.parquet", "ema_clean.parquet",
                  "windows_index.parquet", "nonwear_report.parquet",
                  "ema_windows_qc.parquet"]:
        src = full_processed / fname
        if not src.exists():
            log(f"  WARN: {src.name} not found, skipping")
            continue
        df = pd.read_parquet(src)
        if "subject_id" in df.columns:
            df = df[df["subject_id"].isin(subjects)].copy()
        dst = demo_dir / fname
        df.to_parquet(dst, index=False)
        log(f"  {fname}: {len(df)} rows -> {dst}")


# -- main ---------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ADARP phase-15a data preparation")
    parser.add_argument("--demo", nargs="+", type=int, default=None,
                        help="Subject ids to include in the demo subset, e.g. --demo 102 108 111")
    parser.add_argument("--skip-full", action="store_true",
                        help="Skip the full run (use if you only want to rebuild the demo).")
    args = parser.parse_args(argv)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_full:
        if not SENSOR_DIR.exists():
            log(f"ERROR: Sensor Data directory not found: {SENSOR_DIR}")
            return 1
        if not EMA_FILE.exists():
            log(f"ERROR: EMA xlsm not found: {EMA_FILE}")
            return 1

        sessions = step_inventory(SENSOR_DIR, PROCESSED_DIR)
        ema      = step_ema(EMA_FILE, PROCESSED_DIR)
        windows  = step_windows(ema, sessions, PROCESSED_DIR)
        _        = step_nonwear(sessions, PROCESSED_DIR)
        _        = step_window_qc(windows, sessions, PROCESSED_DIR)

    if args.demo:
        make_demo(args.demo, PROCESSED_DIR, DEMO_DIR)

    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
