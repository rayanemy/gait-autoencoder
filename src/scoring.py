from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_feature_table, periodic_activity_mask
from model import DEFAULT_MODEL_PATH, load_model_artifact
from preprocessing import DEFAULT_FS, read_prepared_or_single_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GWO_CSV = PROJECT_ROOT / "data" / "processed" / "gaitex_test_gwo_left_foot.csv"
DEFAULT_SCORES_CSV = PROJECT_ROOT / "artifacts" / "scores.csv"


SCORE_COLUMNS = [
    "window_id",
    "start_s",
    "end_s",
    "center_s",
    "anomaly_score",
    "is_periodic",
    "is_anomaly",
]


def score_signal(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    axes = artifact["axes"]
    feature_df, _ = build_feature_table(
        df,
        axes=axes,
        window_s=float(artifact["window_s"]),
        step_s=float(artifact["step_s"]),
        fs=float(artifact.get("fs", DEFAULT_FS)),
    )

    feature_columns = artifact["feature_columns"]
    missing = [col for col in feature_columns if col not in feature_df.columns]
    if missing:
        raise ValueError(f"Features manquantes pour le scoring: {', '.join(missing)}")

    is_periodic = periodic_activity_mask(feature_df).to_numpy(dtype=bool)
    scores = np.full(len(feature_df), np.nan, dtype=float)
    predictions = np.ones(len(feature_df), dtype=int)

    if is_periodic.any():
        x = feature_df.loc[is_periodic, feature_columns]
        pipeline = artifact["pipeline"]
        scores[is_periodic] = -pipeline.decision_function(x)
        predictions[is_periodic] = pipeline.predict(x)

    result = feature_df[
        [
            col
            for col in ["window_id", "sequence_id", "subject_id", "scenario", "start_s", "end_s", "center_s"]
            if col in feature_df.columns
        ]
    ].copy()
    result["anomaly_score"] = scores
    result["is_periodic"] = is_periodic
    result["is_anomaly"] = (predictions == -1) & is_periodic
    result = result.sort_values(["sequence_id", "start_s"] if "sequence_id" in result.columns else "start_s")

    ordered = [col for col in SCORE_COLUMNS if col in result.columns]
    extras = [col for col in result.columns if col not in ordered]
    return result[ordered + extras]


def summarize_scores(scores: pd.DataFrame) -> dict[str, float | int]:
    total = int(len(scores))
    if "is_periodic" in scores.columns:
        periodic_mask = scores["is_periodic"].astype(bool)
    else:
        periodic_mask = pd.Series(True, index=scores.index)
    periodic = int(periodic_mask.sum())
    abnormal = int(scores["is_anomaly"].sum()) if total else 0
    denom = periodic if periodic else total
    periodic_scores = (
        scores.loc[periodic_mask, "anomaly_score"]
        if periodic
        else scores.get("anomaly_score", pd.Series(dtype=float))
    )
    return {
        "total_windows": total,
        "periodic_windows": periodic,
        "ignored_windows": total - periodic,
        "anomaly_windows": abnormal,
        "anomaly_percent": float(abnormal / denom * 100.0) if denom else 0.0,
        "max_score": float(periodic_scores.max()) if len(periodic_scores) else 0.0,
        "mean_score": float(periodic_scores.mean()) if len(periodic_scores) else 0.0,
    }


def score_csv(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    input_path: str | Path = DEFAULT_GWO_CSV,
    output_path: str | Path = DEFAULT_SCORES_CSV,
    smoothing: bool = True,
) -> pd.DataFrame:
    artifact = load_model_artifact(model_path)
    df = read_prepared_or_single_csv(
        input_path,
        source_name=input_path,
        target_fs=float(artifact.get("fs", DEFAULT_FS)),
        smoothing=smoothing,
    )
    scores = score_signal(df, artifact)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoring anomalie IMU avec Isolation Forest.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Modele joblib.")
    parser.add_argument("--input", default=str(DEFAULT_GWO_CSV), help="CSV a scorer.")
    parser.add_argument("--output", default=str(DEFAULT_SCORES_CSV), help="CSV de scores.")
    args = parser.parse_args()

    scores = score_csv(args.model, args.input, args.output)
    summary = summarize_scores(scores)
    print(f"Scores sauvegardes: {args.output}")
    print(
        f"Fenetres: {summary['total_windows']} "
        f"(periodiques: {summary['periodic_windows']}, ignorees: {summary['ignored_windows']})"
    )
    print(
        f"Fenetres anormales: {summary['anomaly_windows']} "
        f"({summary['anomaly_percent']:.1f}% des fenetres periodiques)"
    )
    print(f"Score moyen: {summary['mean_score']:.4f}")
    print(f"Score max: {summary['max_score']:.4f}")


if __name__ == "__main__":
    main()
