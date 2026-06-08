from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


DEFAULT_FS = 50.0
AXES = ["yaw", "pitch", "roll"]
LEFT_FOOT_QUATERNION_COLUMNS = [
    "XSens_Foot_Left_QX",
    "XSens_Foot_Left_QY",
    "XSens_Foot_Left_QZ",
    "XSens_Foot_Left_QW",
]


def _normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {_normalise_column_name(col): col for col in df.columns}


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = _column_lookup(df)
    for candidate in candidates:
        found = lookup.get(_normalise_column_name(candidate))
        if found is not None:
            return found
    return None


def has_angle_columns(df: pd.DataFrame) -> bool:
    lookup = _column_lookup(df)
    return all(axis in lookup for axis in AXES)


def _seek_start(source) -> None:
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:
            pass


def infer_subject_id(source_name: str | Path | None) -> str:
    if source_name is None:
        return "unknown"
    stem = Path(str(source_name)).stem
    match = re.search(r"registered_([^_]+)_(?:ng|gwo)$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"generated_([^_]+)", stem, flags=re.IGNORECASE)
    if match:
        return "synthetic"
    return stem


def infer_scenario(source_name: str | Path | None, fallback: str = "unknown") -> str:
    if source_name is None:
        return fallback
    text = str(source_name).lower()
    if "_ng" in text or "/ng/" in text.replace("\\", "/"):
        return "NG - marche normale"
    if "_gwo" in text or "/gwo/" in text.replace("\\", "/"):
        return "GWO - marche avec orthèse"
    if "generated_pronation" in text:
        return "Pronation synthétique"
    if "generated_supination" in text:
        return "Supination synthétique"
    if "generated_toe_walk" in text:
        return "Marche sur les pointes synthétique"
    if "generated_incline_walk" in text:
        return "Marche inclinée synthétique"
    return fallback


def extract_angles_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with time_s, yaw, pitch, roll from CSV angles or Xsens quaternions."""
    time_col = _find_column(
        df,
        ["time_s", "time [s]", "time", "timestamp_s", "timestamp", "seconds", "t"],
    )

    result = pd.DataFrame()
    if time_col is not None:
        result["time_s"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        result["time_s"] = np.arange(len(df), dtype=float) / DEFAULT_FS

    lookup = _column_lookup(df)
    if has_angle_columns(df):
        for axis in AXES:
            result[axis] = pd.to_numeric(df[lookup[axis]], errors="coerce")
        return result

    quat_cols = []
    for column in LEFT_FOOT_QUATERNION_COLUMNS:
        found = _find_column(df, [column])
        if found is None:
            raise ValueError(
                "CSV inexploitable: colonnes yaw/pitch/roll absentes et quaternion "
                f"de semelle gauche manquant ({column})."
            )
        quat_cols.append(found)

    quat = df[quat_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid = np.isfinite(quat).all(axis=1)
    angles = np.full((len(df), 3), np.nan)
    if valid.any():
        rotation = Rotation.from_quat(quat[valid])
        angles[valid] = rotation.as_euler("zyx", degrees=True)

    result["yaw"] = angles[:, 0]
    result["pitch"] = angles[:, 1]
    result["roll"] = angles[:, 2]
    return result


def preprocess_signal(
    df: pd.DataFrame,
    target_fs: float = DEFAULT_FS,
    smoothing: bool = True,
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """Apply the common preprocessing required by the MVP."""
    clean = extract_angles_from_raw(df)
    clean = clean[["time_s", *AXES]].copy()

    for col in ["time_s", *AXES]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(subset=["time_s", *AXES])
    if clean.empty or len(clean) < 3:
        raise ValueError("Signal trop court ou incomplet apres nettoyage.")

    clean = clean.sort_values("time_s")
    clean = clean.drop_duplicates(subset="time_s", keep="first")
    clean["time_s"] = clean["time_s"] - clean["time_s"].iloc[0]

    for axis in AXES:
        radians = np.unwrap(np.deg2rad(clean[axis].to_numpy(dtype=float)))
        clean[axis] = np.rad2deg(radians)

    duration = float(clean["time_s"].iloc[-1])
    if duration <= 0:
        raise ValueError("Duree du signal invalide.")

    step = 1.0 / float(target_fs)
    grid = np.arange(0.0, duration + step * 0.5, step)
    resampled = pd.DataFrame({"time_s": grid})
    source_time = clean["time_s"].to_numpy(dtype=float)
    for axis in AXES:
        resampled[axis] = np.interp(grid, source_time, clean[axis].to_numpy(dtype=float))

    if smoothing and smoothing_window > 1:
        for axis in AXES:
            resampled[axis] = (
                resampled[axis]
                .rolling(window=int(smoothing_window), center=True, min_periods=1)
                .mean()
            )

    return resampled


def load_imu_csv(
    source,
    source_name: str | Path | None = None,
    scenario: str | None = None,
    label: int | None = None,
    target_fs: float = DEFAULT_FS,
    smoothing: bool = True,
) -> pd.DataFrame:
    """Read, convert and preprocess an IMU CSV, then attach simple metadata."""
    _seek_start(source)
    raw = pd.read_csv(source)
    processed = preprocess_signal(raw, target_fs=target_fs, smoothing=smoothing)

    source_label = Path(str(source_name)).name if source_name is not None else "uploaded_csv"
    if scenario is None and "scenario" in raw.columns and raw["scenario"].notna().any():
        scenario = str(raw["scenario"].dropna().iloc[0])
    scenario = scenario or infer_scenario(source_name)

    if label is None and "label" in raw.columns and raw["label"].notna().any():
        label = int(raw["label"].dropna().iloc[0])
    if label is None:
        label = 0 if "normale" in scenario.lower() or "ng" in scenario.lower() else 1

    if "subject_id" in raw.columns and raw["subject_id"].notna().any():
        subject_id = str(raw["subject_id"].dropna().iloc[0])
    else:
        subject_id = infer_subject_id(source_name)

    processed["label"] = int(label)
    processed["subject_id"] = subject_id
    processed["scenario"] = scenario
    processed["source_file"] = source_label
    return processed


def read_prepared_or_single_csv(
    source,
    source_name: str | Path | None = None,
    target_fs: float = DEFAULT_FS,
    smoothing: bool = True,
) -> pd.DataFrame:
    """Load prepared multi-subject data as-is, otherwise preprocess a single sequence."""
    _seek_start(source)
    raw = pd.read_csv(source)
    if has_angle_columns(raw) and "source_file" in raw.columns and raw["source_file"].nunique() > 1:
        return raw
    if has_angle_columns(raw) and "time_s" in raw.columns and raw["time_s"].is_monotonic_increasing:
        expected = ["label", "subject_id", "scenario", "source_file"]
        if all(col in raw.columns for col in expected):
            return raw
    _seek_start(source)
    return load_imu_csv(
        source,
        source_name=source_name,
        target_fs=target_fs,
        smoothing=smoothing,
    )
