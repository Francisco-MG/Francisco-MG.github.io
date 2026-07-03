"""
50_senegal_v4_ml.py
===================
ML + interpretability companion to volet 4 (learning outcomes).

Purpose
-------
The psychometric part (in the .qmd) answers *"is the measure fair?"* (DIF).
This script answers a different, complementary question: *"which structural
factors best PREDICT a pupil reaching the proficiency threshold?"* — and, via
SHAP, *"in which direction and how strongly does each factor act on the model's
prediction?"*. Bringing the two together triangulates the equity story: the
same factors flagged by the psychometric analysis (home language, disability,
governance) should surface as strong predictors here.

Why Python here (not R): gradient boosting (LightGBM) paired with SHAP is the
most mature, fastest path for this specific task. Everything else in the series
stays in R; exchange is by CSV, exactly as in Projet 20.

Pipeline
--------
  in : files/projet_50/eleves_ml.csv   (exported by the .qmd: measured ability
       `theta_hat`, binary `atteint_seuil`, and structural covariates)
  out: files/projet_50/shap_importance.csv   (feature, mean |SHAP|)
       files/projet_50/shap_langue_direction.csv  (mean SHAP by home language)
       files/projet_50/shap_beeswarm.png          (visual summary)

IMPORTANT INTERPRETIVE CAVEAT (carried into the R read-back): SHAP explains the
MODEL's predictions, i.e. predictive ASSOCIATIONS — not causal effects. A large
SHAP importance means the factor helps predict proficiency, not that acting on it
would change proficiency. This distinction is stated explicitly in the report.

Dependencies: see requirements.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")                      # headless: save figures without a display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("files/projet_50")
IN_CSV   = DATA_DIR / "eleves_ml.csv"
SEED     = 54

# Structural predictors (deliberately EXCLUDING the latent school ability from
# which theta was generated — including it would be circular and trivial).
FEATURES = ["milieu", "langue_maison", "genre", "handicap",
            "reform", "g_ia", "remoteness_s", "ptr", "att_school"]
CATEGORICAL = ["milieu", "langue_maison", "genre"]
TARGET = "atteint_seuil"


def load():
    df = pd.read_csv(IN_CSV)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")   # LightGBM handles categoricals natively
    print(f"Loaded {len(df)} pupils from {IN_CSV}")
    return df


def train(df):
    X, y = df[FEATURES], df[TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=-1)
    model.fit(Xtr, ytr, categorical_feature=CATEGORICAL)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print(f"Hold-out AUC = {auc:.3f}")
    return model, X


def explain(model, X):
    """SHAP values for the positive class, with version-robust handling."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # Binary classifiers may return a list [class0, class1] or a single array.
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                       # (n, features, classes) in newer shap
        sv = sv[:, :, 1]
    return sv


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"{IN_CSV} introuvable. Lance d'abord le chunk R 'export-ml' du .qmd.")

    df = load()
    model, X = train(df)
    sv = explain(model, X)

    # (1) Global importance = mean absolute SHAP per feature.
    imp = (pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0)})
             .sort_values("mean_abs_shap", ascending=False))
    imp.to_csv(DATA_DIR / "shap_importance.csv", index=False)
    print(imp.to_string(index=False))

    # (2) Direction for home language: mean SHAP per category tells us which
    #     groups the model pushes UP or DOWN on proficiency — the equity signal.
    li = FEATURES.index("langue_maison")
    dir_langue = (pd.DataFrame({"langue_maison": df["langue_maison"].astype(str),
                                "shap_langue": sv[:, li]})
                    .groupby("langue_maison")["shap_langue"].mean().reset_index())
    dir_langue.to_csv(DATA_DIR / "shap_langue_direction.csv", index=False)

    # (3) Visual summary (beeswarm) for the appendix.
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "shap_beeswarm.png", dpi=130)
    print("Wrote shap_importance.csv, shap_langue_direction.csv, shap_beeswarm.png")


if __name__ == "__main__":
    main()
