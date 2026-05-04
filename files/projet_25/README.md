# Projet 25 — EMA + Wearable Physiology (ADARP)

**Ecological Momentary Assessment and Empatica E4 wearable data:
a reproducible pipeline for stress research in alcohol use disorder.**

This project demonstrates an end-to-end data-engineering and
signal-processing stack for ambulatory mental-health research: raw
multimodal wearable physiology plus self-report prompts, converted
into a rigorously synchronised, QC'd, feature-extracted dataset ready
for multilevel modelling and machine learning.

Data source: ADARP public dataset (Sah et al., Zenodo
[10.5281/zenodo.6640290](https://doi.org/10.5281/zenodo.6640290), CC-BY 4.0).

---

## Project status

| Phase | Content | Status |
|---|---|---|
| 15a | Data pipeline: ingestion, sync, QC, DMP | **done** |
| 15b | Physiological feature extraction (HRV, EDA, actigraphy) | **done** |
| 15c | Multilevel modelling (R) + ML prediction (Python) | planned |
| 15d | Synthetic data extension (parametric bootstrap) | optional |

Each phase produces its own rendered Quarto report in `../projects/`.

---

## Repository layout

```
projet_25/
├── DMP.md                     Data Management Plan (concise, Science Europe style)
├── README.md                  This file
├── .gitignore                 Excludes raw data and processed parquets
├── python/
│   ├── requirements.txt
│   │
│   │ -- Phase 15a (pipeline) --
│   ├── ingest_e4.py           E4 signal loader + session inventory
│   ├── ingest_ema.py          Stress Events xlsm parser
│   ├── qc_nonwear.py          EDA-based non-wear detection
│   ├── sync_windows.py        EMA <-> session synchronisation, pre-event windows
│   ├── prepare_data.py        Orchestrator for phase 15a (CLI)
│   ├── inspect_results.py     Optional: console summary of 15a outputs
│   │
│   │ -- Phase 15b (feature extraction) --
│   ├── extract_hrv.py         HRV time / frequency / non-linear (NeuroKit2)
│   ├── extract_eda.py         SCL / SCR via cvxEDA decomposition
│   ├── extract_activity.py    Actigraphy (Choi-like cutpoints)
│   └── extract_features.py    Orchestrator for phase 15b (CLI)
│
├── R/
│   └── render_helpers.R       Helpers used inside the Quarto reports
└── data/
    ├── demo/                  Versioned slim subset for public reproducibility
    └── processed/             Full-run outputs, local only (gitignored)
```

The two rendered reports live at repository level, not inside this folder:

```
../projects/
├── projet25_ema_physio_adarp.qmd     Phase 15a report
└── projet25b_feature_extraction.qmd  Phase 15b report
```

---

## Local setup (once)

```bash
cd Francisco-MG.github.io/files/projet_25/python
pip install -r requirements.txt
```

**Windows note.** The `cvxopt` package (required by NeuroKit2 for the
cvxEDA decomposition) sometimes fails to build from source on Windows.
If `pip install` fails on `cvxopt`, use:

```bash
conda install -c conda-forge cvxopt
```

Open `prepare_data.py` and verify the top-of-file `ADARP_ROOT` variable
points to the directory that contains `Sensor Data/` and
`Final Phone Survey Stress Events/`.

---

## Pipeline execution

### Phase 15a — data preparation

```bash
python prepare_data.py
```

Produces five parquet files in `data/processed/`:

1. `sessions_inventory.parquet` — one row per E4 session.
2. `ema_clean.parquet` — tidy EMA outcomes (`stress_event`,
   `stress_composite`, imputed timestamps).
3. `windows_index.parquet` — each EMA paired with its 30-min pre-event
   window.
4. `nonwear_report.parquet` — per-session wear statistics.
5. `ema_windows_qc.parquet` — final per-window QC flag
   (`usable_final`).

Expected wall-time: roughly 2–5 minutes on a modern laptop.

### Phase 15b — feature extraction

```bash
python extract_features.py
```

Reads `ema_windows_qc.parquet`, extracts 21 physiological features per
usable window (plus within-subject z-scores and between-subject means
for MLM compatibility), and writes:

1. `physio_features.parquet` — per-window features + QC flags.
2. `extraction_qc_report.parquet` — per-subject feature QC summary.

Expected wall-time: 5–15 minutes (dominated by HRV computation on
30-min BVP windows).

Optional flags:
- `--all`     : process every window, not just `usable_final==TRUE`.
- `--demo`    : read from `data/demo/` and write there.
- Standardisation (`_wz`, `_bm`) is on by default.

### Building the demo subset

After a full run, extract a slim reproducibility demo:

```bash
python prepare_data.py --skip-full --demo 102 108 111
python extract_features.py --demo
```

Writes filtered parquets into `data/demo/`. These are intentionally
committable to the public repository (under 100 MB).

---

## Report rendering

Open either `.qmd` in RStudio and click **Render**. No Python kernel
needed — the reports are pure R and read the parquets produced by the
Python layer.

To render from the demo subset, toggle `DATA_MODE <- "demo"` in the
setup chunk at the top of each `.qmd`.

---

## Attribution

If you use this pipeline or the derived outputs, please cite:

- Martin-Gomez F. (2026). *EMA + Wearable Physiology: a reproducible
  pipeline (projet 25).* GitHub repository.
- Sah R. K. et al. (2022). *ADARP Dataset (1.0).* Zenodo.
  [https://doi.org/10.5281/zenodo.6640290](https://doi.org/10.5281/zenodo.6640290)
- Makowski, D. et al. (2021). *NeuroKit2: A Python toolbox for
  neurophysiological signal processing.* Behavior Research Methods.
- Greco, A. et al. (2016). *cvxEDA: A convex optimization approach to
  electrodermal activity processing.* IEEE TBME.

Code is released under the MIT licence. Demo data is redistributed
under the CC-BY 4.0 licence inherited from the source dataset.
