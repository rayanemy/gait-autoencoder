from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from preprocessing import AXES, DEFAULT_FS


# A gait window must both move enough and look rhythmic. Static pauses, setup
# movements and non-periodic transitions are excluded before training/scoring.
PERIODIC_MIN_AMPLITUDE_DEG = 4.0
PERIODIC_RELATIVE_FRAC = 0.30
PERIODIC_MIN_AUTOCORR = 0.30
_PERIODIC_AXIS_PRIORITY = ("pitch", "roll", "yaw")


def periodic_activity_mask(
    features: pd.DataFrame,
    min_amplitude_deg: float = PERIODIC_MIN_AMPLITUDE_DEG,
    relative_frac: float = PERIODIC_RELATIVE_FRAC,
    min_autocorr: float = PERIODIC_MIN_AUTOCORR,
) -> pd.Series:
    """Return True for windows that contain periodic gait-like movement."""
    if len(features) == 0:
        return pd.Series([], dtype=bool)

    axis = next((a for a in _PERIODIC_AXIS_PRIORITY if f"{a}_amplitude" in features.columns), None)
    if axis is None:
        return pd.Series(True, index=features.index)

    amplitude = features[f"{axis}_amplitude"].to_numpy(dtype=float)
    periodicity_col = f"{axis}_periodicity"
    if periodicity_col in features.columns:
        periodicity = features[periodicity_col].fillna(0.0).to_numpy(dtype=float)
    else:
        periodicity = np.ones(len(features), dtype=float)

    out = np.zeros(len(features), dtype=bool)
    if "sequence_id" in features.columns:
        groups = features.groupby("sequence_id", sort=False).indices.values()
    else:
        groups = [np.arange(len(features))]

    for positions in groups:
        reference = float(np.percentile(amplitude[positions], 90))
        amplitude_threshold = max(min_amplitude_deg, relative_frac * reference)
        out[positions] = (
            (amplitude[positions] >= amplitude_threshold)
            & (periodicity[positions] >= min_autocorr)
        )
    return pd.Series(out, index=features.index)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if not np.isfinite(corr) else corr


def _periodicity(values: np.ndarray, fs: float) -> tuple[float, float]:
    centered = values - np.nanmean(values)
    if len(centered) < 8 or np.nanstd(centered) < 1e-9:
        return 0.0, float("nan")

    min_lag = max(3, int(round(0.30 * fs)))
    max_lag = min(len(centered) - 3, int(round(1.30 * fs)))
    if max_lag <= min_lag:
        return 0.0, float("nan")

    best_corr = 0.0
    best_lag = min_lag
    for lag in range(min_lag, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-9:
            continue
        corr = float(np.dot(left, right) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_corr, best_lag / fs


def _window_features(window: pd.DataFrame, axes: list[str], fs: float) -> dict[str, float]:
    features: dict[str, float] = {}
    for axis in axes:
        values = window[axis].to_numpy(dtype=float)
        features[f"{axis}_mean"] = float(np.mean(values))
        features[f"{axis}_std"] = float(np.std(values, ddof=0))
        features[f"{axis}_min"] = float(np.min(values))
        features[f"{axis}_max"] = float(np.max(values))
        features[f"{axis}_amplitude"] = features[f"{axis}_max"] - features[f"{axis}_min"]
        features[f"{axis}_rms"] = float(np.sqrt(np.mean(np.square(values))))
        velocity = np.diff(values) * fs
        features[f"{axis}_velocity_std"] = float(np.std(velocity, ddof=0)) if len(velocity) else 0.0
        periodicity, period_s = _periodicity(values, fs)
        features[f"{axis}_periodicity"] = periodicity
        features[f"{axis}_period_s"] = period_s

    for left, right in combinations(axes, 2):
        features[f"{left}_{right}_corr"] = _safe_corr(
            window[left].to_numpy(dtype=float),
            window[right].to_numpy(dtype=float),
        )
    return features


def _sequence_groups(df: pd.DataFrame):
    if "source_file" in df.columns:
        yield from df.groupby("source_file", sort=False)
    else:
        yield "sequence_1", df


def build_feature_table(
    df: pd.DataFrame,
    axes: list[str] | None = None,
    window_s: float = 2.0,
    step_s: float = 0.5,
    fs: float = DEFAULT_FS,
) -> tuple[pd.DataFrame, list[str]]:
    axes = axes or AXES.copy()
    missing = [axis for axis in axes if axis not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour les features: {', '.join(missing)}")

    window_n = max(3, int(round(window_s * fs)))
    step_n = max(1, int(round(step_s * fs)))
    rows: list[dict[str, object]] = []
    feature_columns: list[str] | None = None

    for sequence_id, group in _sequence_groups(df):
        group = group.sort_values("time_s").dropna(subset=["time_s", *axes]).reset_index(drop=True)
        if len(group) < 3:
            continue

        if len(group) < window_n:
            starts = [0]
        else:
            starts = list(range(0, len(group) - window_n + 1, step_n))

        for start_idx in starts:
            end_idx = min(start_idx + window_n, len(group))
            window = group.iloc[start_idx:end_idx]
            if len(window) < 3:
                continue

            feature_values = _window_features(window, axes=axes, fs=fs)
            if feature_columns is None:
                feature_columns = list(feature_values.keys())

            start_s = float(window["time_s"].iloc[0])
            end_s = float(window["time_s"].iloc[-1])
            row: dict[str, object] = {
                "window_id": len(rows),
                "sequence_id": str(sequence_id),
                "start_s": start_s,
                "end_s": end_s,
                "center_s": (start_s + end_s) / 2.0,
            }
            for optional in ["subject_id", "scenario", "label"]:
                if optional in group.columns and group[optional].notna().any():
                    row[optional] = group[optional].dropna().iloc[0]
            row.update(feature_values)
            rows.append(row)

    if not rows:
        raise ValueError("Aucune fenetre exploitable pour l'extraction de features.")

    features = pd.DataFrame(rows)
    feature_columns = feature_columns or [
        col
        for col in features.columns
        if col not in {"window_id", "sequence_id", "start_s", "end_s", "center_s"}
    ]
    return features, feature_columns
