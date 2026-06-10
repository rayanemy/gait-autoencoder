from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - fallback if plotly is absent.
    go = None

from generate_sample_data import generate_samples
from model import load_model_artifact, train_model
from prepare_data import prepare_data
from preprocessing import AXES, DEFAULT_FS, read_prepared_or_single_csv
from scoring import score_signal, summarize_scores


NORM_CSV = PROJECT_ROOT / "data" / "processed" / "gaitex_norm_left_foot.csv"
GWO_CSV = PROJECT_ROOT / "data" / "processed" / "gaitex_test_gwo_left_foot.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
SAMPLE_FILES = {
    "Pronation synthétique": SAMPLE_DIR / "generated_pronation.csv",
    "Supination synthétique": SAMPLE_DIR / "generated_supination.csv",
    "Marche sur les pointes synthétique": SAMPLE_DIR / "generated_toe_walk.csv",
    "Marche inclinée synthétique": SAMPLE_DIR / "generated_incline_walk.csv",
}

SCENARIO_LABELS = {
    "GWO - marche avec orthese": "GWO - marche avec orthèse",
    "Pronation synthetique": "Pronation synthétique",
    "Supination synthetique": "Supination synthétique",
    "Marche sur les pointes synthetique": "Marche sur les pointes synthétique",
    "Marche inclinee synthetique": "Marche inclinée synthétique",
    "NG - marche normale": "NG - marche normale",
}


st.set_page_config(
    page_title="Semelle connectée - Détection IMU",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --fallback-accent: #2563eb;
      --fallback-accent-2: #16a34a;
      --fallback-warning: #b45309;
      --fallback-danger: #b91c1c;
      --fallback-ink: currentColor;
      --fallback-surface: color-mix(in srgb, currentColor 5%, transparent);
      --fallback-surface-strong: color-mix(in srgb, currentColor 8%, transparent);
      --accent: var(--primary-color, var(--fallback-accent));
      --accent-2: var(--fallback-accent-2);
      --warning: var(--fallback-warning);
      --danger: var(--fallback-danger);
      --ink: var(--text-color, var(--fallback-ink));
      --muted: color-mix(in srgb, var(--text-color, var(--fallback-ink)) 68%, transparent);
      --line: color-mix(in srgb, var(--text-color, var(--fallback-ink)) 20%, transparent);
      --soft: transparent;
      --surface: var(--secondary-background-color, var(--fallback-surface));
      --surface-strong: color-mix(
        in srgb,
        var(--secondary-background-color, var(--fallback-surface-strong)) 88%,
        var(--background-color, var(--fallback-surface-strong))
      );
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --fallback-accent: #60a5fa;
        --fallback-accent-2: #22c55e;
        --fallback-warning: #f59e0b;
        --fallback-danger: #f87171;
      }
    }
    .main .block-container {
      padding-top: 1.0rem;
      max-width: 1180px;
    }
    h1, h2, h3 {
      letter-spacing: 0;
      color: var(--ink);
    }
    .project-kicker {
      color: var(--accent);
      font-weight: 700;
      font-size: 0.86rem;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      margin-bottom: 0.2rem;
    }
    .lead {
      color: var(--muted);
      font-size: 1.04rem;
      line-height: 1.6;
      margin: 0.15rem 0 0.95rem 0;
      max-width: 880px;
    }
    .section-band {
      background: var(--soft);
      border-top: 2px solid var(--line);
      border-bottom: 0px solid var(--line);
      padding: 0.9rem 1rem;
      margin: 1.1rem 0;
    }
    .status-box {
      border: 1px solid var(--line);
      border-left: 5px solid var(--accent);
      border-radius: 6px;
      padding: 0.85rem 1rem;
      background: var(--surface-strong);
      color: var(--ink);
    }
    .status-box strong {
      display: block;
      margin-bottom: 0.2rem;
    }
    .normal-box {
      border-left-color: var(--accent-2);
      background: color-mix(in srgb, var(--accent-2) 10%, var(--surface-strong));
    }
    .anomaly-box {
      border-left-color: var(--danger);
      background: color-mix(in srgb, var(--danger) 10%, var(--surface-strong));
    }
    .muted-note {
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.45;
    }
    .pipeline {
      display: flex;
      align-items: stretch;
      gap: 0.65rem;
      margin: 1rem 0 1.25rem 0;
      flex-wrap: wrap;
    }
    .pipeline-step {
      flex: 1 1 145px;
      min-width: 145px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      padding: 0.78rem 0.85rem;
    }
    .pipeline-step strong {
      display: block;
      color: var(--ink);
      font-size: 0.96rem;
      margin-bottom: 0.2rem;
    }
    .pipeline-step span {
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.25;
    }
    .explain-box {
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      border-radius: 6px;
      padding: 0.75rem 0.9rem;
      margin: 0.75rem 0 1rem 0;
      background: var(--surface-strong);
      color: var(--ink);
    }
    .explain-box strong {
      display: block;
      margin-bottom: 0.25rem;
    }
    .explain-box p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .term-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 0.7rem;
      margin: 0.75rem 0 1rem 0;
    }
    .term-card {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      padding: 0.75rem 0.85rem;
      color: var(--ink);
    }
    .term-card strong {
      display: block;
      margin-bottom: 0.25rem;
    }
    .term-card span {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.35;
    }
    .pipeline-arrow {
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-weight: 800;
      min-width: 1.2rem;
    }
    .decision-tree {
      display: grid;
      grid-template-columns: 1fr 1fr 1.1fr 1fr;
      gap: 0.65rem;
      align-items: center;
      margin: 0.8rem 0 1.1rem 0;
    }
    .decision-node {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
      padding: 0.8rem 0.9rem;
    }
    .decision-node strong {
      display: block;
      margin-bottom: 0.25rem;
    }
    .decision-node span {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.35;
    }
    .decision-branches {
      display: grid;
      gap: 0.55rem;
    }
    .decision-path {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface-strong);
      padding: 0.7rem 0.8rem;
      color: var(--ink);
    }
    .decision-normal {
      border-left: 4px solid var(--accent-2);
    }
    .decision-anomaly {
      border-left: 4px solid var(--danger);
    }
    @media (max-width: 720px) {
      .pipeline-arrow {
        display: none;
      }
      .decision-tree {
        grid-template-columns: 1fr;
      }
    }
    .stMetric,
    div[data-testid="stMetric"] {
      background: var(--surface-strong) !important;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0.65rem 0.75rem;
      color: var(--ink);
    }
    div[data-testid="stMetric"] * {
      background: transparent !important;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p {
      color: var(--muted) !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] div,
    div[data-testid="stMetricValue"] p {
      color: var(--ink) !important;
    }
    .stButton>button {
      border-radius: 6px;
      font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def explain(title: str, body: str, class_name: str = "explain-box") -> None:
    st.markdown(
        f"""
        <div class="{class_name}">
          <strong>{title}</strong>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def glossary_cards() -> None:
    st.markdown(
        """
        <div class="term-grid">
          <div class="term-card"><strong>Fenêtre</strong><span>Durée du morceau de signal analysé à chaque fois. À 2 s et 50 Hz, une fenêtre contient environ 100 mesures.</span></div>
          <div class="term-card"><strong>Pas</strong><span>Décalage entre deux fenêtres. Un pas de 0,5 s crée des fenêtres qui se chevauchent, donc le suivi temporel est plus fin.</span></div>
          <div class="term-card"><strong>Features</strong><span>Résumé numérique d’une fenêtre : moyenne, écart-type, amplitude, RMS, vitesse angulaire, périodicité et corrélations entre axes.</span></div>
          <div class="term-card"><strong>Contamination</strong><span>Pour Isolation Forest, proportion attendue de fenêtres atypiques. Elle règle surtout la sensibilité du seuil.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout_kwargs() -> dict[str, object]:
    dark = st.get_option("theme.base") == "dark"
    return {
        "template": "plotly_dark" if dark else "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#f8fafc" if dark else "#172033"},
        "xaxis": {"gridcolor": "rgba(148,163,184,0.28)", "zerolinecolor": "rgba(148,163,184,0.45)"},
        "yaxis": {"gridcolor": "rgba(148,163,184,0.28)", "zerolinecolor": "rgba(148,163,184,0.45)"},
    }


def pipeline_diagram() -> None:
    st.markdown(
        """
        <div class="pipeline" role="img" aria-label="Pipeline de détection d'anomalies IMU">
          <div class="pipeline-step"><strong>Semelle gauche</strong><span>IMU + carte SD</span></div>
          <div class="pipeline-arrow">&rarr;</div>
          <div class="pipeline-step"><strong>Angles</strong><span>yaw, pitch, roll à 50 Hz</span></div>
          <div class="pipeline-arrow">&rarr;</div>
          <div class="pipeline-step"><strong>Fenêtres temporelles</strong><span>features de marche</span></div>
          <div class="pipeline-arrow">&rarr;</div>
          <div class="pipeline-step"><strong>Périodicité</strong><span>non rythmique ignorée</span></div>
          <div class="pipeline-arrow">&rarr;</div>
          <div class="pipeline-step"><strong>Isolation Forest</strong><span>normal ou écart à la norme</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_tree_diagram() -> None:
    st.markdown(
        """
        <div class="decision-tree" role="img" aria-label="Diagramme de décision Isolation Forest">
          <div class="decision-node"><strong>Fenêtre IMU</strong><span>2 s de yaw, pitch, roll nettoyés.</span></div>
          <div class="decision-node"><strong>Périodicité</strong><span>Si la fenêtre n'est pas rythmique, elle est ignorée.</span></div>
          <div class="decision-node"><strong>Isolation Forest</strong><span>Forêt d'arbres d'isolation, pas arbre supervisé.</span></div>
          <div class="decision-branches">
            <div class="decision-path decision-normal"><strong>Score ≤ seuil</strong><span>Fenêtre considérée normale.</span></div>
            <div class="decision-path decision-anomaly"><strong>Score &gt; seuil</strong><span>Fenêtre anormale par rapport à GAITEX.</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_signal(df: pd.DataFrame, axes: list[str], title: str):
    if go is None:
        st.line_chart(df.set_index("time_s")[axes])
        return
    fig = go.Figure()
    colors = {"yaw": "#2563eb", "pitch": "#16a34a", "roll": "#dc2626"}
    for axis in axes:
        fig.add_trace(
            go.Scatter(
                x=df["time_s"],
                y=df[axis],
                mode="lines",
                name=axis,
                line={"width": 2, "color": colors.get(axis)},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Temps (s)",
        yaxis_title="Angle (degrés)",
        height=410,
        margin={"l": 20, "r": 20, "t": 55, "b": 35},
        legend_title_text="Axes",
        **plotly_layout_kwargs(),
    )
    st.plotly_chart(fig, width="stretch")


def plot_scores(scores: pd.DataFrame):
    plottable = scores.copy()
    if "is_periodic" in plottable.columns:
        plottable = plottable[plottable["is_periodic"].astype(bool)].copy()
    plottable = plottable.dropna(subset=["anomaly_score"])
    if plottable.empty:
        st.info("Aucune fenêtre périodique à tracer pour ce signal.")
        return
    if go is None:
        st.line_chart(plottable.set_index("center_s")["anomaly_score"])
        return
    fig = go.Figure()
    marker_colors = plottable["is_anomaly"].map({True: "#dc2626", False: "#2563eb"})
    fig.add_trace(
        go.Scatter(
            x=plottable["center_s"],
            y=plottable["anomaly_score"],
            mode="lines+markers",
            name="Score d'anomalie",
            line={"color": "#2563eb", "width": 2},
            marker={"size": 6, "color": marker_colors},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", annotation_text="seuil modèle")
    fig.update_layout(
        title="Score d'anomalie par fenêtre temporelle",
        xaxis_title="Centre de la fenêtre (s)",
        yaxis_title="Score",
        height=360,
        margin={"l": 20, "r": 20, "t": 55, "b": 35},
        **plotly_layout_kwargs(),
    )
    st.plotly_chart(fig, width="stretch")


def plot_overlay(df: pd.DataFrame, scores: pd.DataFrame, axes: list[str]):
    if go is None:
        st.line_chart(df.set_index("time_s")[axes])
        return
    fig = go.Figure()
    colors = {"yaw": "#2563eb", "pitch": "#16a34a", "roll": "#dc2626"}
    for axis in axes:
        fig.add_trace(
            go.Scatter(
                x=df["time_s"],
                y=df[axis],
                mode="lines",
                name=axis,
                line={"width": 2, "color": colors.get(axis)},
            )
        )
    y_min = float(df[axes].min().min())
    y_max = float(df[axes].max().max())
    for _, row in scores[scores["is_anomaly"]].iterrows():
        fig.add_shape(
            type="rect",
            x0=float(row["start_s"]),
            x1=float(row["end_s"]),
            y0=y_min,
            y1=y_max,
            fillcolor="#dc2626",
            opacity=0.16,
            line_width=0,
            layer="below",
        )
    fig.update_layout(
        title="Signal IMU avec zones anormales en surbrillance",
        xaxis_title="Temps (s)",
        yaxis_title="Angle (degrés)",
        height=430,
        margin={"l": 20, "r": 20, "t": 55, "b": 35},
        legend_title_text="Axes",
        **plotly_layout_kwargs(),
    )
    st.plotly_chart(fig, width="stretch")


def dataframe_summary(df: pd.DataFrame) -> dict[str, object]:
    duration = float(df.groupby("source_file")["time_s"].max().sum()) if "source_file" in df.columns else float(df["time_s"].max())
    subjects = int(df["subject_id"].nunique()) if "subject_id" in df.columns else None
    return {
        "rows": int(len(df)),
        "duration": duration,
        "subjects": subjects,
        "columns": ", ".join(df.columns[:10]) + ("..." if len(df.columns) > 10 else ""),
    }


def humanize_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "scenario" in result.columns:
        result["scenario"] = result["scenario"].replace(SCENARIO_LABELS)
    return result


def choose_display_sequence(
    df: pd.DataFrame,
    scores: pd.DataFrame | None = None,
    key: str = "display_sequence",
) -> tuple[str | None, pd.DataFrame, pd.DataFrame | None]:
    if "source_file" not in df.columns or df["source_file"].nunique() <= 1:
        return None, df, scores
    options = sorted(df["source_file"].dropna().unique().tolist())
    selected = options[0]
    st.caption(f"Graphiques affichés sur : {selected}. Le résumé reste calculé sur toutes les séquences.")
    display_df = df[df["source_file"] == selected].copy()
    display_scores = None
    if scores is not None and "sequence_id" in scores.columns:
        display_scores = scores[scores["sequence_id"] == selected].copy()
    return selected, display_df, display_scores


def get_source_data(source_choice: str, sample_choice: str | None, uploaded_file):
    if source_choice == "GWO - marche avec orthèse":
        if not GWO_CSV.exists():
            raise FileNotFoundError("Le fichier GWO préparé est introuvable.")
        return read_prepared_or_single_csv(GWO_CSV, source_name=GWO_CSV), "GWO - marche avec orthèse"
    if source_choice == "Exemple synthétique":
        sample_path = SAMPLE_FILES[sample_choice]
        if not sample_path.exists():
            raise FileNotFoundError(f"Exemple synthétique introuvable : {sample_path.name}")
        return read_prepared_or_single_csv(sample_path, source_name=sample_path), sample_choice
    if uploaded_file is None:
        raise ValueError("Aucun CSV utilisateur chargé.")
    return read_prepared_or_single_csv(uploaded_file, source_name=uploaded_file.name), uploaded_file.name


axes = AXES.copy()
window_s = 2.0
step_s = 0.5
contamination = 0.05


with st.sidebar:
    st.header("1. Source de données")
    source_choice = st.radio(
        "Marche à scorer",
        ["GWO - marche avec orthèse", "Exemple synthétique", "Upload CSV"],
        index=0,
        help="Choisissez le signal à comparer à la norme GAITEX. Aucun de ces signaux n'est utilisé pour entraîner le modèle.",
    )
    st.caption("GWO est une marche avec orthèse : c'est un exemple réel de marche modifiée disponible dans le projet.")
    sample_choice = None
    uploaded_file = None
    if source_choice == "Exemple synthétique":
        sample_choice = st.selectbox(
            "Exemple",
            list(SAMPLE_FILES.keys()),
            help="Ces fichiers sont fabriqués à partir d'un signal normal puis déformés pour illustrer un écart à la norme.",
        )
    if source_choice == "Upload CSV":
        uploaded_file = st.file_uploader(
            "CSV IMU",
            type=["csv"],
            help="Format accepté : time_s,yaw,pitch,roll ou quaternions Xsens de la semelle gauche.",
        )

    st.header("2. Scoring")
    st.caption("Modèle : Isolation Forest chargé depuis artifacts/model.joblib. S'il est absent, il est entraîné automatiquement depuis la norme NG.")
    run_button = st.button("Lancer le scoring", type="primary", width="stretch", key="sidebar_run_scoring")


st.markdown('<div class="project-kicker">Semelle connectée (P2I2 - 221C)</div>', unsafe_allow_html=True)
st.title("Détection d'anomalies de marche par IMU gauche")
st.markdown(
    """
    <p class="lead">
    Démonstration machine learning par <a href="https://github.com/rayanemy" target="_blank">@rayanemy</a> et <a href="https://github.com/dhia9" target="_blank">@dhia9</a> : les données IMU sauvegardées sur carte SD par la semelle gauche
    sont converties en angles, comparées à une norme GAITEX, puis scorées par Isolation Forest.
    </p>
    """,
    unsafe_allow_html=True,
)

pipeline_diagram()
explain(
    "Lecture du pipeline",
    "Le modèle apprend seulement les fenêtres périodiques de marche normale NG. Les zones non rythmiques, par exemple avant le départ de la marche, sont exclues avant l'entraînement et avant le scoring.",
)

if not NORM_CSV.exists():
    st.warning("La norme GAITEX préparée est absente.")
    if st.button("Préparer data/ng et data/gwo", width="content"):
        with st.spinner("Préparation des données Xsens de la semelle gauche vers yaw/pitch/roll..."):
            prepare_data()
            load_csv_cached.clear()
        st.success("Données préparées.")
    st.stop()

norm_df = load_csv_cached(str(NORM_CSV))
summary = dataframe_summary(norm_df)

st.markdown('<div class="section-band"><strong>1. Norme GAITEX utilisée pour l’apprentissage</strong></div>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="muted-note">
    La norme est construite à partir de data/ng. Chaque CSV brut fournit les quaternions de la semelle gauche, puis le pipeline garde yaw, pitch et roll à 50 Hz avec les métadonnées du sujet.
    </p>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(4)
cols[0].metric("Lignes", f"{summary['rows']:,}".replace(",", " "))
cols[1].metric("Durée approx.", f"{summary['duration']:.0f} s")
cols[2].metric("Sujets", "n/a" if summary["subjects"] is None else str(summary["subjects"]))
cols[3].metric("Fréquence cible", f"{DEFAULT_FS:.0f} Hz")
explain(
    "Préparation du signal",
    "Chaque fichier est remis au propre avant le modèle : temps trié, doublons retirés, quaternions convertis en yaw/pitch/roll, interpolation à 50 Hz et léger lissage.",
)

missing_samples = [path for path in SAMPLE_FILES.values() if not path.exists()]
if missing_samples:
    st.info("Les exemples synthétiques ne sont pas encore générés.")
    if st.button("Générer les exemples synthétiques"):
        generate_samples()
        st.success("Exemples synthétiques générés.")

artifact = None
model_message = ""

try:
    if MODEL_PATH.exists():
        artifact = load_model_artifact(MODEL_PATH)
        model_message = "Modèle chargé depuis artifacts/model.joblib."
    else:
        with st.spinner("Aucun modèle sauvegardé : entraînement depuis la norme..."):
            artifact = train_model(
                norm_csv=NORM_CSV,
                model_path=MODEL_PATH,
                axes=axes,
                window_s=window_s,
                step_s=step_s,
                contamination=contamination,
            )
        model_message = "Modèle entraîné et sauvegardé."
except Exception as exc:
    st.error(f"Impossible de charger ou d'entraîner le modèle : {exc}")
    st.stop()

train_summary = artifact.get("train_summary", {})
st.markdown('<div class="section-band"><strong>2. Modèle Isolation Forest</strong></div>', unsafe_allow_html=True)
st.markdown('<p class="muted-note"> Le modèle ne reçoit que des fenêtres périodiques extraites de la marche normale NG. Les débuts, fins et transitions non rythmiques ne servent pas au fit ; le modèle apprend donc la marche réellement active.</p>', unsafe_allow_html=True)

model_cols = st.columns(6)
model_cols[0].metric("Modèle", "IForest")
model_cols[1].metric("Axes", ", ".join(artifact["axes"]))
model_cols[2].metric("Fenêtre", f"{artifact['window_s']:.1f} s")
model_cols[3].metric("Pas", f"{artifact['step_s']:.2f} s")
model_cols[4].metric("Train périodique", str(train_summary.get("training_windows", "n/a")))
model_cols[5].metric("NG ignorées", str(train_summary.get("ignored_non_periodic_windows", "n/a")))
st.markdown(f'<p class="muted-note">{model_message} Le modèle apprend uniquement les fenêtres périodiques de marche normale NG de la semelle gauche.</p>', unsafe_allow_html=True)
explain(
    "Ce que le modèle regarde",
    "Il ne compare pas chaque point du signal. Il résume chaque fenêtre de marche avec quelques mesures simples, puis cherche si ce résumé s'éloigne de la norme GAITEX.",
)

if not run_button:
    st.markdown('<div class="section-band"><strong>3. Lancer le scoring</strong></div>', unsafe_allow_html=True)
    explain(
        "Étape suivante",
        "Choisissez une source dans la barre latérale, puis lancez le scoring. L'application nettoie le signal, garde les fenêtres de marche rythmique et les compare au modèle appris sur NG.",
    )

if run_button:
    try:
        with st.spinner("Scoring en cours..."):
            source_df, source_name = get_source_data(source_choice, sample_choice, uploaded_file)
            scores = score_signal(source_df, artifact)
    except Exception as exc:
        st.error(f"Scoring impossible : {exc}")
        st.stop()

    score_summary = summarize_scores(scores)
    st.markdown('<div class="section-band"><strong>3. Résultats du scoring</strong></div>', unsafe_allow_html=True)
    explain(
        "Comment lire le score",
        "Chaque ligne correspond à une fenêtre temporelle. Les fenêtres non périodiques sont retirées : leur score reste vide et elles ne peuvent pas être des anomalies. Sur les fenêtres périodiques, un score positif franchit le seuil d'anomalie.",
    )
    result_cols = st.columns(6)
    result_cols[0].metric("Fenêtres totales", score_summary["total_windows"])
    result_cols[1].metric("Périodiques", score_summary["periodic_windows"])
    result_cols[2].metric("Ignorées", score_summary["ignored_windows"])
    result_cols[3].metric("Anormales", score_summary["anomaly_windows"])
    result_cols[4].metric("% anormal", f"{score_summary['anomaly_percent']:.1f} %")
    result_cols[5].metric("Score max", f"{score_summary['max_score']:.3f}")

    if score_summary["anomaly_windows"] == 0:
        st.markdown(
            '<div class="status-box normal-box"><strong>Aucune anomalie importante détectée.</strong>'
            "Cette marche reste proche de la norme GAITEX apprise par le modèle.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-box anomaly-box"><strong>Des fenêtres anormales ont été détectées par rapport à la norme GAITEX.</strong>'
            "Le résultat signale un écart statistique, pas un diagnostic clinique.</div>",
            unsafe_allow_html=True,
        )

    st.caption(f"Source scorée : {source_name}")
    if score_summary.get("ignored_windows", 0) > 0:
        st.caption(
            f"{score_summary['ignored_windows']} fenêtre(s) non périodiques ignorées : "
            f"le scoring ne porte que sur les {score_summary['periodic_windows']} fenêtres périodiques."
        )
    selected_sequence, display_df, display_scores = choose_display_sequence(
        source_df,
        scores,
        key="scored_display_sequence",
    )
    st.markdown('<div class="section-band"><strong>4. Graphiques</strong></div>', unsafe_allow_html=True)
    explain(
        "Score d'anomalie",
        "Ce graphique suit le score fenêtre par fenêtre périodique. Les points rouges sont les fenêtres rejetées par le modèle ; la ligne pointillée correspond au seuil.",
    )
    plot_scores(display_scores if display_scores is not None else scores)

    explain(
        "Signal IMU nettoyé",
        "Nettoyé signifie : temps trié et remis à zéro, doublons supprimés, angles dépliés, interpolation à 50 Hz et léger lissage. Les courbes restent les angles mesurés par la semelle gauche.",
    )
    plot_signal(display_df, axes=artifact["axes"], title="Signal IMU nettoyé")

    explain(
        "Overlay signal + anomalies",
        "Les bandes rouges montrent uniquement les intervalles temporels des fenêtres anormales. Les fenêtres non périodiques sont exclues du calcul et ne sont plus dessinées dans ce graphique.",
    )
    plot_overlay(display_df, display_scores if display_scores is not None else scores, axes=artifact["axes"])

    st.subheader("Table scores.csv")
    public_columns = ["window_id", "start_s", "end_s", "center_s", "anomaly_score", "is_periodic", "is_anomaly"]
    public_columns = [col for col in public_columns if col in scores.columns]
    optional_columns = [col for col in ["sequence_id", "subject_id", "scenario"] if col in scores.columns]
    table = scores[public_columns + optional_columns].copy()
    table = humanize_scenarios(table)
    explain(
        "Contenu de la table",
        "start_s et end_s donnent les bornes de la fenêtre, is_periodic indique si une marche rythmique est détectée, anomaly_score reste vide si la fenêtre est non périodique, et is_anomaly donne le verdict final.",
    )
    st.dataframe(table, width="stretch", hide_index=True)
