"""Train Chokepoint scorers.

Three rankers are produced, all attached to the same vendor_scores.parquet:

  baseline_score  : minmax(betweenness_centrality). No ML.
  iso_score       : IsolationForest unsupervised anomaly score on the full
                    engineered feature set.
  model_score     : Supervised GradientBoosting regressor trained on the
                    TRAIN portion of the simulated coverage_drop labels.
                    Predicts coverage_drop for every vendor.

Only the train split of labels is used to fit the supervised model; the test
split is held out for `models/eval.py`.

Artifacts written to models/artifacts/:
  scaler.joblib, isoforest.joblib, supervised.joblib
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("data/processed/vendor_features.parquet")
LABELS_PATH = Path("data/processed/vendor_labels.parquet")
SCORES_PATH = Path("data/processed/vendor_scores.parquet")
ARTIFACT_DIR = Path("models/artifacts")

RAW_FEATURE_COLS: list[str] = [
    "degree_centrality",
    "betweenness_centrality",
    "pagerank",
    "eigenvector_centrality",
    "articulation_point",
    "agency_count",
    "naics_count",
    "critical_naics_count",
    "contract_count",
    "total_award_value",
    "avg_award_size",
    "sole_source_ratio",
    "critical_sole_source_count",
    "mean_pair_redundancy",
    "hhi_score",
    "naics_hhi",
    "critical_naics_market_share",
]

MODEL_FEATURE_COLS: list[str] = [
    "degree_centrality",
    "betweenness_centrality",
    "pagerank",
    "eigenvector_centrality",
    "articulation_point",
    "log_agency_count",
    "log_naics_count",
    "log_total_award_value",
    "log_contract_count",
    "log_avg_award_size",
    "sole_source_ratio",
    "mean_pair_redundancy",
    "hhi_score",
    "naics_hhi",
    "critical_naics_market_share",
    "sole_source_breadth",
    "footprint",
    "log_critical_naics_count",
    "log_critical_sole_source_count",
    "critical_breadth",
]

ISO_CONTAMINATION = 0.05
ISO_N_ESTIMATORS = 300
GB_N_ESTIMATORS = 400
GB_MAX_DEPTH = 4
GB_LEARNING_RATE = 0.05
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform skewed magnitudes and add interaction features.

    Raw counts and award value are heavy right-tailed; without log scaling
    the IsolationForest collapses onto magnitude. `sole_source_breadth` and
    `footprint` are the interaction signals that matter for chokepoint-ness.
    """
    out = df.copy()
    out["log_agency_count"] = np.log1p(out["agency_count"].astype(float))
    out["log_naics_count"] = np.log1p(out["naics_count"].astype(float))
    out["log_total_award_value"] = np.log1p(out["total_award_value"].astype(float))
    out["log_contract_count"] = np.log1p(out["contract_count"].astype(float))
    out["log_avg_award_size"] = np.log1p(out["avg_award_size"].astype(float))
    out["sole_source_breadth"] = out["sole_source_ratio"] * out["log_naics_count"]
    out["footprint"] = out["log_agency_count"] * out["log_naics_count"]
    out["log_critical_naics_count"] = np.log1p(
        out["critical_naics_count"].astype(float)
    )
    out["log_critical_sole_source_count"] = np.log1p(
        out["critical_sole_source_count"].astype(float)
    )
    # critical_breadth = sole-source pressure within the defense-critical
    # NAICS slice. Captures "this vendor sole-sources defense-critical
    # categories" independent of total breadth.
    out["critical_breadth"] = (
        out["sole_source_ratio"] * out["log_critical_naics_count"]
    )
    return out


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def main() -> None:
    if not FEATURES_PATH.exists():
        raise SystemExit(f"Missing {FEATURES_PATH}. Run `make features`.")
    if not LABELS_PATH.exists():
        raise SystemExit(f"Missing {LABELS_PATH}. Run `make labels`.")

    df = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    df = df.merge(labels, on="vendor_id", how="left")
    log.info("Loaded %d vendors  (train=%d  test=%d  none=%d)",
             len(df),
             int((df["split"] == "train").sum()),
             int((df["split"] == "test").sum()),
             int((df["split"] == "none").sum()))

    df = engineer_features(df)
    x_all = df[MODEL_FEATURE_COLS].to_numpy(dtype=float)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("chokepoint")
    with mlflow.start_run(run_name="rf+isoforest_v2"):
        scaler = StandardScaler().fit(x_all)
        x_scaled = scaler.transform(x_all)

        # --- IsolationForest (unsupervised) ---
        log.info("Fitting IsolationForest")
        iforest = IsolationForest(
            n_estimators=ISO_N_ESTIMATORS,
            contamination=ISO_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ).fit(x_scaled)
        iso_raw = -iforest.decision_function(x_scaled)

        # --- Supervised GradientBoosting (trained on TRAIN split only) ---
        # Target: naics_affected_count. Using the absolute count (not the
        # fraction `coverage_drop`) prevents the model from inflating tiny
        # vendors that sole-source 1/1 pair — those have coverage_drop=1 but
        # naics_affected_count=1. Larger structural chokepoints score
        # naics_affected_count in the 5..15+ range.
        train_mask = (df["split"] == "train").to_numpy()
        y_train = df.loc[train_mask, "naics_affected_count"].to_numpy(dtype=float)
        log.info(
            "Fitting GradientBoostingRegressor on %d train samples "
            "(target=naics_affected_count, mean=%.2f, max=%.0f)",
            train_mask.sum(),
            float(y_train.mean()),
            float(y_train.max()),
        )
        gb = GradientBoostingRegressor(
            n_estimators=GB_N_ESTIMATORS,
            max_depth=GB_MAX_DEPTH,
            learning_rate=GB_LEARNING_RATE,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ).fit(x_scaled[train_mask], y_train)
        gb_pred = gb.predict(x_scaled)

        # Normalize to [0, 1]
        df["baseline_score"] = minmax(df["betweenness_centrality"]).values
        df["iso_score"] = minmax(pd.Series(iso_raw, index=df.index)).values
        df["model_score"] = minmax(pd.Series(gb_pred, index=df.index)).values

        rho_iso, _ = spearmanr(df["model_score"], df["iso_score"])
        rho_base, _ = spearmanr(df["model_score"], df["baseline_score"])
        log.info(
            "Spearman(model, iso)=%.3f  Spearman(model, baseline)=%.3f",
            rho_iso,
            rho_base,
        )

        # Persist scores. Ground-truth labels (coverage_drop,
        # naics_affected_count) stay in vendor_labels.parquet; downstream code
        # joins on vendor_id when needed.
        cols_out = [
            "vendor_id",
            "vendor_name",
            "baseline_score",
            "iso_score",
            "model_score",
            *RAW_FEATURE_COLS,
        ]
        df[cols_out].to_parquet(SCORES_PATH, index=False)
        log.info("Wrote %s (%d rows)", SCORES_PATH, len(df))

        joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
        joblib.dump(iforest, ARTIFACT_DIR / "isoforest.joblib")
        joblib.dump(gb, ARTIFACT_DIR / "supervised.joblib")
        log.info("Saved scaler + isoforest + supervised to %s", ARTIFACT_DIR)

        # Feature importances for the headline model
        importances = dict(zip(MODEL_FEATURE_COLS, gb.feature_importances_))
        log.info(
            "Supervised feature importances:\n%s",
            "\n".join(
                f"  {k:<24} {v:.4f}"
                for k, v in sorted(importances.items(), key=lambda kv: -kv[1])
            ),
        )

        mlflow.log_params(
            {
                "iso_contamination": ISO_CONTAMINATION,
                "iso_n_estimators": ISO_N_ESTIMATORS,
                "gb_n_estimators": GB_N_ESTIMATORS,
                "gb_max_depth": GB_MAX_DEPTH,
                "gb_learning_rate": GB_LEARNING_RATE,
                "n_features": len(MODEL_FEATURE_COLS),
                "n_train": int(train_mask.sum()),
                "n_vendors": len(df),
            }
        )
        mlflow.log_metric("rho_model_iso", float(rho_iso))
        mlflow.log_metric("rho_model_baseline", float(rho_base))
        mlflow.log_artifact(str(SCORES_PATH))

        top = df.nlargest(10, "model_score")[
            ["vendor_name", "model_score", "iso_score", "baseline_score", "coverage_drop"]
        ]
        log.info("Top 10 by model_score:\n%s", top.to_string(index=False))


if __name__ == "__main__":
    main()
