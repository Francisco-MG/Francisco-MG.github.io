"""
inspect_results.py
------------------
Quick inspection of the parquets produced by prepare_data.py.
Paste the console output back to the chat.
"""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED = PROJECT_DIR / "data" / "processed"

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# --- 1. Sessions: per-subject summary -----------------------------------------
print("=" * 78)
print("SESSIONS INVENTORY — per subject")
print("=" * 78)
s = pd.read_parquet(PROCESSED / "sessions_inventory.parquet")
per_subj = s.groupby("subject_id").agg(
    n_sessions=("session_start_utc", "count"),
    n_devices=("device_id", "nunique"),
    total_hours=("duration_hours", "sum"),
    median_duration_h=("duration_hours", "median"),
    first_session=("session_start_utc", "min"),
    last_session=("session_end_utc",   "max"),
).round(2)
per_subj["span_days"] = (per_subj["last_session"] - per_subj["first_session"]).dt.total_seconds() / 86400
per_subj["span_days"] = per_subj["span_days"].round(1)
print(per_subj.to_string())
print()
print(f"TOTAL: {len(s)} sessions, {s['subject_id'].nunique()} subjects, "
      f"{s['duration_hours'].sum():.1f} cumulative hours")

# --- 2. EMA ------------------------------------------------------------------
print()
print("=" * 78)
print("EMA — stress events")
print("=" * 78)
e = pd.read_parquet(PROCESSED / "ema_clean.parquet")
print(f"Total EMA: {len(e)}, subjects: {sorted(e['subject_id'].unique())}")
print()
print("stress_event distribution:")
print(e["stress_event"].value_counts(dropna=False).to_string())
print()
print("stress_composite distribution:")
print(e["stress_composite"].value_counts(dropna=False).sort_index().to_string())
print()
print("EMA per subject:")
print(e.groupby("subject_id").size().to_string())

# --- 3. Windows QC -----------------------------------------------------------
print()
print("=" * 78)
print("EMA WINDOWS QC — per subject")
print("=" * 78)
w = pd.read_parquet(PROCESSED / "ema_windows_qc.parquet")
qc_per_subj = w.groupby("subject_id").agg(
    n_ema=("obs_id", "count"),
    n_matched=("session_idx", lambda x: (x >= 0).sum()),
    n_usable_cov=("usable", "sum"),
    n_usable_final=("usable_final", "sum"),
    median_wear_frac=("window_wear_fraction", "median"),
).round(2)
print(qc_per_subj.to_string())
print()
print(f"TOTAL usable_final: {w['usable_final'].sum()}/{len(w)} "
      f"({100*w['usable_final'].mean():.1f}%)")

# --- 4. Non-wear -------------------------------------------------------------
print()
print("=" * 78)
print("NON-WEAR REPORT — distribution of per-session % wear")
print("=" * 78)
nw = pd.read_parquet(PROCESSED / "nonwear_report.parquet")
print(nw["pct_wear"].describe().round(2).to_string())
print()
print("Sessions with pct_wear < 50%:")
low = nw[nw["pct_wear"] < 50].sort_values("pct_wear")
print(f"  count: {len(low)}")
if len(low) > 0:
    print(low.head(10).to_string())

# --- 5. Multi-device subjects ------------------------------------------------
print()
print("=" * 78)
print("DEVICE USAGE — subjects with multiple E4 devices")
print("=" * 78)
multi = s.groupby("subject_id")["device_id"].nunique()
multi = multi[multi > 1]
if len(multi) > 0:
    print(multi.to_string())
    for sid in multi.index:
        devs = s[s["subject_id"] == sid]["device_id"].value_counts()
        print(f"  Subject {sid}: {devs.to_dict()}")
else:
    print("All subjects used a single device.")
