from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_table, periodic_activity_mask
from preprocessing import AXES, DEFAULT_FS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORM_CSV = PROJECT_ROOT / "data" / "processed" / "gaitex_norm_left_foot.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"


def make_estimator(contamination: float = 0.05):
    estimator = IsolationForest(
        n_estimators=300,
        contamination=float(contamination),
        random_state=42,
    )
    return make_pipeline(StandardScaler(), estimator)


def train_model(
    norm_csv: str | Path = DEFAULT_NORM_CSV,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    axes: list[str] | None = None,
    window_s: float = 2.0,
    step_s: float = 0.5,
    fs: float = DEFAULT_FS,
    contamination: float = 0.05,
    save: bool = True,
) -> dict:
    axes = axes or AXES.copy()
    norm_csv = Path(norm_csv)
    model_path = Path(model_path)
    if not norm_csv.exists():
        raise FileNotFoundError(f"Norme introuvable: {norm_csv}")

    norm_df = pd.read_csv(norm_csv)
    feature_df, feature_columns = build_feature_table(
        norm_df,
        axes=axes,
        window_s=window_s,
        step_s=step_s,
        fs=fs,
    )
    periodic_mask = periodic_activity_mask(feature_df).to_numpy(dtype=bool)
    ignored_non_periodic_windows = int((~periodic_mask).sum())
    feature_df = feature_df.loc[periodic_mask].reset_index(drop=True)
    if feature_df.empty:
        raise ValueError("Aucune fenetre periodique detectee dans la norme NG.")

    pipeline = make_estimator(contamination=contamination)
    x_train = feature_df[feature_columns]
    pipeline.fit(x_train)

    train_scores = -pipeline.decision_function(x_train)
    train_predictions = pipeline.predict(x_train)
    artifact = {
        "pipeline": pipeline,
        "model_type": "isolation_forest",
        "axes": axes,
        "window_s": float(window_s),
        "step_s": float(step_s),
        "fs": float(fs),
        "feature_columns": feature_columns,
        "contamination": float(contamination),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_summary": {
            "norm_csv": str(norm_csv),
            "training_rows": int(len(norm_df)),
            "training_windows": int(len(feature_df)),
            "ignored_non_periodic_windows": ignored_non_periodic_windows,
            "subjects": int(norm_df["subject_id"].nunique()) if "subject_id" in norm_df.columns else None,
            "mean_score": float(train_scores.mean()),
            "max_score": float(train_scores.max()),
            "anomaly_rate": float((train_predictions == -1).mean()),
        },
    }

    if save:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, model_path)
    return artifact


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable: {model_path}")
    return joblib.load(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainement Isolation Forest pour signaux IMU.")
    parser.add_argument("--train", action="store_true", help="Entrainer et sauvegarder le modele.")
    parser.add_argument("--norm", default=str(DEFAULT_NORM_CSV), help="CSV de norme NG prepare.")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH), help="Chemin du modele joblib.")
    parser.add_argument("--axes", default="yaw,pitch,roll", help="Axes utilises: yaw,pitch,roll ou pitch,roll.")
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--step-s", type=float, default=0.5)
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()

    if not args.train:
        parser.error("Ajoutez --train pour entrainer le modele.")

    axes = [axis.strip() for axis in args.axes.split(",") if axis.strip()]
    artifact = train_model(
        norm_csv=args.norm,
        model_path=args.output,
        axes=axes,
        window_s=args.window_s,
        step_s=args.step_s,
        contamination=args.contamination,
    )
    summary = artifact["train_summary"]
    print(f"Modele sauvegarde: {args.output}")
    print(f"Fenetres d'entrainement: {summary['training_windows']}")
    print(f"Taux de fenetres atypiques sur train: {summary['anomaly_rate']:.3f}")


if __name__ == "__main__":
    main()
