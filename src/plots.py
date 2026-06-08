from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_signal(df: pd.DataFrame, axes: list[str] | None = None):
    axes = axes or ["yaw", "pitch", "roll"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for axis in axes:
        if axis in df.columns:
            ax.plot(df["time_s"], df[axis], label=axis)
    ax.set_title("Signal IMU nettoye")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Angle (degrés)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return fig


def plot_scores(scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(scores["center_s"], scores["anomaly_score"], label="Score d'anomalie")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", label="Seuil modèle")
    ax.set_title("Score d'anomalie par fenetre")
    ax.set_xlabel("Temps central (s)")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return fig


def plot_overlay(df: pd.DataFrame, scores: pd.DataFrame, axes: list[str] | None = None):
    axes = axes or ["yaw", "pitch", "roll"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for axis in axes:
        if axis in df.columns:
            ax.plot(df["time_s"], df[axis], label=axis)
    for _, row in scores[scores["is_anomaly"]].iterrows():
        ax.axvspan(row["start_s"], row["end_s"], color="#c43e3e", alpha=0.15)
    ax.set_title("Signal avec zones anormales")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Angle (degrés)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return fig
