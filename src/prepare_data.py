from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocessing import DEFAULT_FS, load_imu_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_NG_DIR = DATA_DIR / "ng"
DEFAULT_GWO_DIR = DATA_DIR / "gwo"
DEFAULT_PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_NORM_OUTPUT = DEFAULT_PROCESSED_DIR / "gaitex_norm_left_foot.csv"
DEFAULT_GWO_OUTPUT = DEFAULT_PROCESSED_DIR / "gaitex_test_gwo_left_foot.csv"


def prepare_folder(
    input_dir: str | Path,
    scenario: str,
    label: int,
    target_fs: float = DEFAULT_FS,
    smoothing: bool = True,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Aucun CSV trouve dans {input_dir}")

    frames = []
    for path in paths:
        print(f"Preparation: {path.name}")
        prepared = load_imu_csv(
            path,
            source_name=path,
            scenario=scenario,
            label=label,
            target_fs=target_fs,
            smoothing=smoothing,
        )
        frames.append(prepared)
    return pd.concat(frames, ignore_index=True)


def prepare_data(
    ng_dir: str | Path = DEFAULT_NG_DIR,
    gwo_dir: str | Path = DEFAULT_GWO_DIR,
    norm_output: str | Path = DEFAULT_NORM_OUTPUT,
    gwo_output: str | Path = DEFAULT_GWO_OUTPUT,
    target_fs: float = DEFAULT_FS,
    smoothing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    norm_output = Path(norm_output)
    gwo_output = Path(gwo_output)
    norm_output.parent.mkdir(parents=True, exist_ok=True)
    gwo_output.parent.mkdir(parents=True, exist_ok=True)

    norm = prepare_folder(
        ng_dir,
        scenario="NG - marche normale",
        label=0,
        target_fs=target_fs,
        smoothing=smoothing,
    )
    gwo = prepare_folder(
        gwo_dir,
        scenario="GWO - marche avec orthèse",
        label=1,
        target_fs=target_fs,
        smoothing=smoothing,
    )

    columns = ["time_s", "yaw", "pitch", "roll", "label", "subject_id", "scenario", "source_file"]
    norm[columns].to_csv(norm_output, index=False)
    gwo[columns].to_csv(gwo_output, index=False)

    print(f"Norme sauvegardee: {norm_output} ({len(norm)} lignes)")
    print(f"Test GWO sauvegarde: {gwo_output} ({len(gwo)} lignes)")
    return norm, gwo


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare les CSV GAITEX NG et GWO.")
    parser.add_argument("--ng-dir", default=str(DEFAULT_NG_DIR))
    parser.add_argument("--gwo-dir", default=str(DEFAULT_GWO_DIR))
    parser.add_argument("--norm-output", default=str(DEFAULT_NORM_OUTPUT))
    parser.add_argument("--gwo-output", default=str(DEFAULT_GWO_OUTPUT))
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--no-smoothing", action="store_true")
    args = parser.parse_args()

    prepare_data(
        ng_dir=args.ng_dir,
        gwo_dir=args.gwo_dir,
        norm_output=args.norm_output,
        gwo_output=args.gwo_output,
        target_fs=args.fs,
        smoothing=not args.no_smoothing,
    )


if __name__ == "__main__":
    main()
