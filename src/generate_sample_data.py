from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

from preprocessing import DEFAULT_FS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORM_CSV = PROJECT_ROOT / "data" / "processed" / "gaitex_norm_left_foot.csv"
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

# Cadence physiologique d'un pied lors d'une marche normale (~0.8-1.0 Hz).
GAIT_BAND_HZ = (0.5, 2.0)
# Bruit capteur léger ajouté pour éviter des signaux parfaitement périodiques.
SENSOR_NOISE_DEG = 0.3
RANDOM_SEED = 7


def _fallback_base(duration_s: float = 40.0, fs: float = DEFAULT_FS) -> pd.DataFrame:
    time_s = np.arange(0.0, duration_s, 1.0 / fs)
    gait = 2 * np.pi * 0.85 * time_s
    return pd.DataFrame(
        {
            "time_s": time_s,
            "yaw": 4.0 * np.sin(gait * 0.5),
            "pitch": 7.6 + 12.0 * np.sin(gait) + 2.0 * np.sin(gait * 2.0),
            "roll": 7.0 * np.sin(gait + 0.8),
        }
    )


def load_base_signal(norm_csv: str | Path = DEFAULT_NORM_CSV, duration_s: float = 40.0) -> pd.DataFrame:
    norm_csv = Path(norm_csv)
    if not norm_csv.exists():
        return _fallback_base(duration_s=duration_s)

    norm = pd.read_csv(norm_csv)
    if "source_file" in norm.columns:
        first_source = norm["source_file"].dropna().iloc[0]
        norm = norm[norm["source_file"] == first_source].copy()
    norm = norm.sort_values("time_s")
    norm = norm[norm["time_s"] <= duration_s].copy()
    if len(norm) < 3:
        return _fallback_base(duration_s=duration_s)
    return norm[["time_s", "yaw", "pitch", "roll"]].reset_index(drop=True)


def gait_stance_weight(pitch: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """Poids [0, 1] verrouillé sur le cycle de marche réel (max en appui, min en oscillation).

    On extrait la phase instantanée du cycle à partir de l'axe sagittal (pitch),
    filtré autour de la cadence, via la transformée de Hilbert. Les perturbations
    sont ainsi synchronisées avec le pas réel plutôt qu'avec une sinusoïde arbitraire.
    """
    pitch = np.asarray(pitch, dtype=float)
    centered = pitch - np.mean(pitch)
    if len(centered) < 9 or np.std(centered) < 1e-6:
        return np.full(len(pitch), 0.5)

    nyq = fs / 2.0
    low, high = GAIT_BAND_HZ[0] / nyq, min(GAIT_BAND_HZ[1] / nyq, 0.99)
    try:
        b, a = butter(2, [low, high], btype="band")
        filtered = filtfilt(b, a, centered)
        phase = np.unwrap(np.angle(hilbert(filtered)))
    except ValueError:
        return np.full(len(pitch), 0.5)

    # cos(phase) ramène la phase sur [0, 1] : 1 = appui, 0 = oscillation.
    return 0.5 * (1.0 + np.cos(phase))


def _finish_sample(
    df: pd.DataFrame,
    scenario: str,
    filename: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = df.copy()
    for axis in ("yaw", "pitch", "roll"):
        result[axis] = result[axis] + rng.normal(0.0, SENSOR_NOISE_DEG, size=len(result))
    result["label"] = 1
    result["subject_id"] = "synthetic"
    result["scenario"] = scenario
    result["source_file"] = filename
    return result


def generate_samples(
    output_dir: str | Path = DEFAULT_SAMPLE_DIR,
    norm_csv: str | Path = DEFAULT_NORM_CSV,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    base = load_base_signal(norm_csv=norm_csv)
    pitch = base["pitch"].to_numpy(dtype=float)
    pitch_mean = float(np.mean(pitch))
    # Poids d'appui synchronisé sur le cycle de marche réel.
    stance = gait_stance_weight(pitch, fs=DEFAULT_FS)

    samples: dict[str, pd.DataFrame] = {}

    # --- Pronation : effondrement médial, éversion marquée surtout en appui. ---
    pronation = base.copy()
    pronation["roll"] = base["roll"] - 12.0 - 6.0 * stance
    pronation["pitch"] = pitch_mean + 0.95 * (base["pitch"] - pitch_mean)
    samples["generated_pronation.csv"] = _finish_sample(
        pronation, "Pronation synthétique", "generated_pronation.csv", rng
    )

    # --- Supination : inversion excessive (miroir de la pronation). ---
    supination = base.copy()
    supination["roll"] = base["roll"] + 12.0 + 6.0 * stance
    supination["pitch"] = pitch_mean + 0.95 * (base["pitch"] - pitch_mean)
    samples["generated_supination.csv"] = _finish_sample(
        supination, "Supination synthétique", "generated_supination.csv", rng
    )

    # --- Marche sur les pointes : flexion plantaire maintenue, amplitude réduite
    #     (pas d'attaque talon), roll atténué. ---
    toe_walk = base.copy()
    toe_walk["pitch"] = pitch_mean + 0.45 * (base["pitch"] - pitch_mean) + 14.0
    toe_walk["roll"] = base["roll"] * 0.55
    samples["generated_toe_walk.csv"] = _finish_sample(
        toe_walk, "Marche sur les pointes synthétique", "generated_toe_walk.csv", rng
    )

    # --- Marche inclinée (montée) : dorsiflexion accrue et soutenue (offset stable,
    #     non une dérive linéaire), amplitude sagittale légèrement augmentée. ---
    incline = base.copy()
    incline["pitch"] = pitch_mean + 1.1 * (base["pitch"] - pitch_mean) + 12.0
    samples["generated_incline_walk.csv"] = _finish_sample(
        incline, "Marche inclinée synthétique", "generated_incline_walk.csv", rng
    )

    written: dict[str, Path] = {}
    for filename, frame in samples.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        written[filename] = path
        print(f"Sample sauvegarde: {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère des marches synthétiques de démonstration.")
    parser.add_argument("--output-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--norm", default=str(DEFAULT_NORM_CSV))
    args = parser.parse_args()
    generate_samples(output_dir=args.output_dir, norm_csv=args.norm)


if __name__ == "__main__":
    main()
