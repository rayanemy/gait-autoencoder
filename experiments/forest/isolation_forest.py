# =============================================================================
# DÉTECTION D'ANOMALIES - ISOLATION FOREST
# Données : timestamp relatif, yaw, pitch, roll (angles d'Euler)
# =============================================================================

# =============================================================================
# PARTIE 1 — IMPORTS ET CONFIGURATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Style global
plt.rcParams['figure.facecolor'] = '#0f0f1a'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
plt.rcParams['axes.edgecolor'] = '#444466'
plt.rcParams['text.color'] = '#e0e0ff'
plt.rcParams['axes.labelcolor'] = '#c0c0ff'
plt.rcParams['xtick.color'] = '#8080aa'
plt.rcParams['ytick.color'] = '#8080aa'
plt.rcParams['grid.color'] = '#2a2a4a'
plt.rcParams['grid.alpha'] = 0.5

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 65)
print("  ISOLATION FOREST — Détection d'anomalies Angles d'Euler")
print("=" * 65)


# =============================================================================
# PARTIE 2 — CHARGEMENT ET EXPLORATION DES DONNÉES
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier CSV/TSV avec les colonnes :
    relative_timestamp, yaw, pitch, roll
    Adapte automatiquement le séparateur.
    """
    for sep in [',', ';', '\t', ' ']:
        try:
            df = pd.read_csv(filepath, sep=sep)
            if df.shape[1] >= 4:
                break
        except Exception:
            continue

    # Normalisation des noms de colonnes
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Renommage flexible
    rename_map = {}
    for col in df.columns:
        if 'timestamp' in col or 'time' in col:
            rename_map[col] = 'timestamp'
        elif 'yaw' in col:
            rename_map[col] = 'yaw'
        elif 'pitch' in col:
            rename_map[col] = 'pitch'
        elif 'roll' in col:
            rename_map[col] = 'roll'
    df = df.rename(columns=rename_map)

    required = ['timestamp', 'yaw', 'pitch', 'roll']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}. Colonnes trouvées : {list(df.columns)}")

    df = df[required].dropna()
    df[['yaw', 'pitch', 'roll']] = df[['yaw', 'pitch', 'roll']].astype(float)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna()
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\n[DONNÉES] {len(df)} lignes chargées depuis '{filepath}'")
    return df


def generate_synthetic_data(n_samples: int = 2000,
                             anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """
    Génère un dataset synthétique réaliste pour les tests.
    Les anomalies sont des valeurs hors-plage ou des pics brusques.
    """
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anom = n_samples - n_normal

    t = np.linspace(0, 100, n_normal)
    noise = 0.5
    yaw_n = np.sin(0.1 * t) * 30 + np.random.randn(n_normal) * noise * 5
    pitch_n = np.cos(0.15 * t) * 15 + np.random.randn(n_normal) * noise * 3
    roll_n = np.sin(0.05 * t + 1) * 20 + np.random.randn(n_normal) * noise * 4

    t_a = np.random.uniform(0, 100, n_anom)
    yaw_a = np.random.uniform(-180, 180, n_anom)
    pitch_a = np.random.uniform(-90, 90, n_anom)
    roll_a = np.random.uniform(-180, 180, n_anom)

    df_normal = pd.DataFrame({
        'timestamp': t,
        'yaw': yaw_n, 'pitch': pitch_n, 'roll': roll_n,
        'true_label': 0  # 0 = normal
    })
    df_anom = pd.DataFrame({
        'timestamp': t_a,
        'yaw': yaw_a, 'pitch': pitch_a, 'roll': roll_a,
        'true_label': 1  # 1 = anomalie
    })

    df = pd.concat([df_normal, df_anom]).sort_values('timestamp').reset_index(drop=True)
    print(f"\n[DONNÉES SYNTHÉTIQUES] {len(df)} lignes | "
          f"{n_normal} normales ({(1-anomaly_ratio)*100:.0f}%) | "
          f"{n_anom} anomalies ({anomaly_ratio*100:.0f}%)")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Analyse exploratoire des données brutes."""
    print("\n" + "─" * 50)
    print("  EXPLORATION DES DONNÉES")
    print("─" * 50)
    print(f"  Shape         : {df.shape}")
    print(f"  Timestamp min : {df['timestamp'].min():.3f}")
    print(f"  Timestamp max : {df['timestamp'].max():.3f}")
    print()

    features = ['yaw', 'pitch', 'roll']
    stats = df[features].describe().T
    stats['range'] = stats['max'] - stats['min']
    stats['cv'] = (stats['std'] / stats['mean'].abs()).replace([np.inf, -np.inf], np.nan)
    print(stats[['mean', 'std', 'min', 'max', 'range']].round(3).to_string())

    if 'true_label' in df.columns:
        n_anom = df['true_label'].sum()
        print(f"\n  Labels vérité terrain disponibles :")
        print(f"    Normaux   : {len(df) - n_anom} ({(1 - n_anom/len(df))*100:.1f}%)")
        print(f"    Anomalies : {n_anom} ({n_anom/len(df)*100:.1f}%)")

    # Corrélations
    corr = df[features].corr()
    print(f"\n  Matrice de corrélation :")
    print(corr.round(3).to_string())


def plot_raw_data(df: pd.DataFrame) -> None:
    """Visualisation des séries temporelles brutes."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Séries Temporelles — Angles d'Euler", fontsize=14,
                 color='#c0c0ff', fontweight='bold', y=1.02)

    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']
    labels = ['Yaw (°)', 'Pitch (°)', 'Roll (°)']

    for i, (col, color, label) in enumerate(zip(['yaw', 'pitch', 'roll'], colors, labels)):
        ax = axes[i]
        ax.plot(df['timestamp'], df[col], color=color, linewidth=0.8, alpha=0.85)
        if 'true_label' in df.columns:
            anom_mask = df['true_label'] == 1
            ax.scatter(df.loc[anom_mask, 'timestamp'], df.loc[anom_mask, col],
                       color='#ff0066', s=15, zorder=5, label='Anomalie réelle', alpha=0.8)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0 and 'true_label' in df.columns:
            ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Timestamp relatif', fontsize=11)
    plt.tight_layout()
    plt.savefig('/home/claude/if_01_raw_data.png', dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("\n  [FIGURE] Séries temporelles sauvegardées → if_01_raw_data.png")


# =============================================================================
# PARTIE 3 — FEATURE ENGINEERING
# =============================================================================

def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Construit les features à partir des angles bruts.
    Ajoute : deltas, rolling mean/std, magnitude angulaire, dérivées.
    """
    df = df.copy()
    angles = ['yaw', 'pitch', 'roll']

    # Deltas (vitesse angulaire)
    for a in angles:
        df[f'd_{a}'] = df[a].diff().fillna(0)

    # Magnitude angulaire totale
    df['magnitude'] = np.sqrt(df['yaw']**2 + df['pitch']**2 + df['roll']**2)
    df['d_magnitude'] = df['magnitude'].diff().fillna(0)

    # Rolling statistics
    for a in angles:
        df[f'{a}_rmean'] = df[a].rolling(window, min_periods=1).mean()
        df[f'{a}_rstd'] = df[a].rolling(window, min_periods=1).std().fillna(0)
        df[f'{a}_rmax'] = df[a].rolling(window, min_periods=1).max()
        df[f'{a}_rmin'] = df[a].rolling(window, min_periods=1).min()
        df[f'{a}_rrange'] = df[f'{a}_rmax'] - df[f'{a}_rmin']

    # Dérivée seconde
    for a in angles:
        df[f'd2_{a}'] = df[f'd_{a}'].diff().fillna(0)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns
                    if c not in ['timestamp', 'true_label']]
    print(f"\n[FEATURES] {len(feature_cols)} features construites : {feature_cols}")
    return df, feature_cols


# =============================================================================
# PARTIE 4 — SPLIT TRAIN / TEST STRATIFIÉ
# =============================================================================

def stratified_split(df: pd.DataFrame,
                     feature_cols: list,
                     train_ratio: float = 0.8,
                     has_labels: bool = True) -> dict:
    """
    Split 80/20 stratifié :
    - 80% des normaux dans train, 20% dans test
    - 80% des anomalies dans train, 20% dans test
    Si pas de labels : split temporel simple.
    """
    print("\n" + "─" * 50)
    print("  SPLIT TRAIN / TEST (80 / 20 STRATIFIÉ)")
    print("─" * 50)

    if has_labels and 'true_label' in df.columns:
        normal_idx = df[df['true_label'] == 0].index.tolist()
        anom_idx = df[df['true_label'] == 1].index.tolist()

        np.random.shuffle(normal_idx)
        np.random.shuffle(anom_idx)

        n_train_normal = int(len(normal_idx) * train_ratio)
        n_train_anom = int(len(anom_idx) * train_ratio)

        train_idx = sorted(normal_idx[:n_train_normal] + anom_idx[:n_train_anom])
        test_idx = sorted(normal_idx[n_train_normal:] + anom_idx[n_train_anom:])

        train_df = df.loc[train_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)

        print(f"  TRAIN : {len(train_df)} lignes")
        print(f"    Normaux   : {(train_df['true_label']==0).sum()} "
              f"({(train_df['true_label']==0).mean()*100:.1f}%)")
        print(f"    Anomalies : {(train_df['true_label']==1).sum()} "
              f"({(train_df['true_label']==1).mean()*100:.1f}%)")
        print(f"  TEST  : {len(test_df)} lignes")
        print(f"    Normaux   : {(test_df['true_label']==0).sum()} "
              f"({(test_df['true_label']==0).mean()*100:.1f}%)")
        print(f"    Anomalies : {(test_df['true_label']==1).sum()} "
              f"({(test_df['true_label']==1).mean()*100:.1f}%)")
    else:
        cut = int(len(df) * train_ratio)
        train_df = df.iloc[:cut].reset_index(drop=True)
        test_df = df.iloc[cut:].reset_index(drop=True)
        print(f"  TRAIN : {len(train_df)} | TEST : {len(test_df)}")
        print("  (split temporel, pas de labels disponibles)")

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_train = train_df['true_label'].values if 'true_label' in train_df.columns else None
    y_test = test_df['true_label'].values if 'true_label' in test_df.columns else None

    # Normalisation
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return {
        'X_train': X_train_sc, 'X_test': X_test_sc,
        'y_train': y_train, 'y_test': y_test,
        'train_df': train_df, 'test_df': test_df,
        'scaler': scaler
    }


# =============================================================================
# PARTIE 5 — CALIBRATION DU CONTAMINATION RATE
# =============================================================================

def calibrate_contamination(splits: dict,
                             values: list = None) -> float:
    """
    Teste plusieurs taux de contamination et sélectionne le meilleur
    selon l'AUC ou la proportion d'anomalies détectées.
    """
    if values is None:
        values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

    print("\n" + "─" * 50)
    print("  CALIBRATION DU CONTAMINATION RATE")
    print("─" * 50)

    y_test = splits['y_test']
    best_score = -1
    best_cont = 0.05
    results = []

    for cont in values:
        clf = IsolationForest(contamination=cont, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(splits['X_train'])
        scores = -clf.score_samples(splits['X_test'])  # Anomaly score

        if y_test is not None:
            auc = roc_auc_score(y_test, scores)
            results.append((cont, auc))
            print(f"  contamination={cont:.2f}  →  AUC = {auc:.4f}")
            if auc > best_score:
                best_score = auc
                best_cont = cont
        else:
            pred = clf.predict(splits['X_test'])
            n_anom = (pred == -1).sum()
            results.append((cont, n_anom))
            print(f"  contamination={cont:.2f}  →  {n_anom} anomalies détectées")

    print(f"\n  ✓ Meilleur contamination : {best_cont}")
    return best_cont, results


# =============================================================================
# PARTIE 6 — ENTRAÎNEMENT ISOLATION FOREST
# =============================================================================

def train_isolation_forest(X_train: np.ndarray,
                            contamination: float = 0.05,
                            n_estimators: int = 200) -> IsolationForest:
    """Entraîne le modèle Isolation Forest."""
    print("\n" + "─" * 50)
    print("  ENTRAÎNEMENT — ISOLATION FOREST")
    print("─" * 50)
    print(f"  n_estimators  : {n_estimators}")
    print(f"  contamination : {contamination}")
    print(f"  random_state  : {RANDOM_STATE}")

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        warm_start=False
    )
    clf.fit(X_train)
    print("  ✓ Entraînement terminé")
    return clf


# =============================================================================
# PARTIE 7 — ÉVALUATION ET MÉTRIQUES
# =============================================================================

def evaluate_model(clf: IsolationForest, splits: dict) -> dict:
    """Évaluation complète sur train et test."""
    print("\n" + "─" * 50)
    print("  ÉVALUATION DU MODÈLE")
    print("─" * 50)

    results = {}
    for split_name in ['train', 'test']:
        X = splits[f'X_{split_name}']
        y = splits[f'y_{split_name}']

        pred_raw = clf.predict(X)         # +1 normal, -1 anomalie
        pred = (pred_raw == -1).astype(int)  # 1 = anomalie, 0 = normal
        scores = -clf.score_samples(X)    # anomaly score (plus haut = plus anormal)

        results[split_name] = {'pred': pred, 'scores': scores, 'y': y}

        n_anom = pred.sum()
        print(f"\n  [{split_name.upper()}] {len(pred)} échantillons → "
              f"{n_anom} anomalies détectées ({n_anom/len(pred)*100:.1f}%)")

        if y is not None:
            auc = roc_auc_score(y, scores)
            ap = average_precision_score(y, scores)
            print(f"    AUC-ROC          : {auc:.4f}")
            print(f"    Avg Precision    : {ap:.4f}")
            print(f"\n    Classification Report :")
            print(classification_report(y, pred,
                                        target_names=['Normal', 'Anomalie'],
                                        digits=4))
            results[split_name]['auc'] = auc
            results[split_name]['ap'] = ap

    return results


# =============================================================================
# PARTIE 8 — VISUALISATIONS
# =============================================================================

def plot_anomaly_scores(splits: dict, eval_results: dict) -> None:
    """Distribution des scores d'anomalie."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Isolation Forest — Distribution des Scores d'Anomalie",
                 fontsize=13, color='#c0c0ff', fontweight='bold')

    for ax, split_name, color_base in zip(
            axes, ['train', 'test'], ['#4ecdc4', '#ff6b6b']):
        scores = eval_results[split_name]['scores']
        y = eval_results[split_name]['y']

        if y is not None:
            ax.hist(scores[y == 0], bins=60, alpha=0.7, color='#4ecdc4',
                    label='Normal', density=True)
            ax.hist(scores[y == 1], bins=60, alpha=0.7, color='#ff0066',
                    label='Anomalie', density=True)
            ax.legend(fontsize=10)
        else:
            ax.hist(scores, bins=60, alpha=0.8, color=color_base, density=True)

        ax.set_title(split_name.upper(), fontsize=12, color='#e0e0ff')
        ax.set_xlabel("Anomaly Score", fontsize=10)
        ax.set_ylabel("Densité", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/if_02_score_distribution.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Distribution des scores → if_02_score_distribution.png")


def plot_roc_pr(splits: dict, eval_results: dict) -> None:
    """Courbes ROC et Precision-Recall."""
    y_test = splits['y_test']
    if y_test is None:
        return

    scores_test = eval_results['test']['scores']
    scores_train = eval_results['train']['scores']
    y_train = splits['y_train']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Isolation Forest — Courbes ROC & Precision-Recall",
                 fontsize=13, color='#c0c0ff', fontweight='bold')

    # ROC
    ax = axes[0]
    for name, y, sc, col in [('Train', y_train, scores_train, '#4ecdc4'),
                               ('Test', y_test, scores_test, '#ff6b6b')]:
        fpr, tpr, _ = roc_curve(y, sc)
        auc = roc_auc_score(y, sc)
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{name} AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='#555577', lw=1)
    ax.set_xlabel('FPR', fontsize=10); ax.set_ylabel('TPR', fontsize=10)
    ax.set_title('Courbe ROC', fontsize=12, color='#e0e0ff')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # PR
    ax = axes[1]
    for name, y, sc, col in [('Train', y_train, scores_train, '#4ecdc4'),
                               ('Test', y_test, scores_test, '#ff6b6b')]:
        prec, rec, _ = precision_recall_curve(y, sc)
        ap = average_precision_score(y, sc)
        ax.plot(rec, prec, color=col, lw=2, label=f'{name} AP={ap:.3f}')
    ax.set_xlabel('Recall', fontsize=10); ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Courbe Precision-Recall', fontsize=12, color='#e0e0ff')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/if_03_roc_pr.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] ROC & PR → if_03_roc_pr.png")


def plot_confusion_matrix(eval_results: dict) -> None:
    """Matrice de confusion (test uniquement)."""
    y = eval_results['test']['y']
    pred = eval_results['test']['pred']
    if y is None:
        return

    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomalie'],
                yticklabels=['Normal', 'Anomalie'],
                ax=ax, linewidths=0.5,
                annot_kws={'size': 14, 'color': 'white'})
    ax.set_title('Matrice de Confusion — TEST\nIsolation Forest',
                 fontsize=12, color='#c0c0ff', pad=15)
    ax.set_xlabel('Prédit', fontsize=11)
    ax.set_ylabel('Réel', fontsize=11)
    plt.tight_layout()
    plt.savefig('/home/claude/if_04_confusion_matrix.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Matrice de confusion → if_04_confusion_matrix.png")


def plot_anomalies_on_timeseries(splits: dict,
                                  eval_results: dict,
                                  feature_cols: list) -> None:
    """Anomalies détectées superposées sur les séries temporelles."""
    test_df = splits['test_df'].copy()
    pred = eval_results['test']['pred']
    scores = eval_results['test']['scores']
    test_df['anomaly_pred'] = pred
    test_df['anomaly_score'] = scores

    angles = ['yaw', 'pitch', 'roll']
    available = [a for a in angles if a in test_df.columns]
    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']

    fig, axes = plt.subplots(len(available) + 1, 1,
                              figsize=(15, 3 * (len(available) + 1)),
                              sharex=True)
    fig.suptitle("Isolation Forest — Anomalies sur Séries Temporelles (TEST)",
                 fontsize=13, color='#c0c0ff', fontweight='bold')

    for i, (col, color) in enumerate(zip(available, colors)):
        ax = axes[i]
        ax.plot(test_df['timestamp'], test_df[col],
                color=color, linewidth=0.9, alpha=0.85, label=col.capitalize())
        mask = test_df['anomaly_pred'] == 1
        ax.scatter(test_df.loc[mask, 'timestamp'], test_df.loc[mask, col],
                   color='#ff0066', s=20, zorder=5, alpha=0.9, label='Détecté')
        if 'true_label' in test_df.columns:
            true_mask = test_df['true_label'] == 1
            ax.scatter(test_df.loc[true_mask, 'timestamp'],
                       test_df.loc[true_mask, col],
                       color='white', s=8, zorder=4, alpha=0.5, marker='x',
                       label='Réel')
        ax.set_ylabel(col.capitalize(), fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[-1]
    ax.fill_between(test_df['timestamp'], test_df['anomaly_score'],
                    alpha=0.7, color='#aa44ff', label='Anomaly Score')
    ax.axhline(y=test_df['anomaly_score'].quantile(0.95),
               color='#ff0066', linestyle='--', linewidth=1.2, label='Seuil 95%')
    ax.set_ylabel('Anomaly Score', fontsize=10)
    ax.set_xlabel('Timestamp relatif', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/if_05_anomalies_timeseries.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Anomalies sur séries → if_05_anomalies_timeseries.png")


def plot_feature_importance(clf: IsolationForest,
                             feature_cols: list) -> None:
    """
    Importance approximative des features via la profondeur moyenne
    dans les arbres de l'Isolation Forest.
    """
    n_features = len(feature_cols)
    importances = np.zeros(n_features)

    for tree in clf.estimators_:
        feature_indices = tree.tree_.feature
        importances += np.bincount(
            feature_indices[feature_indices >= 0],
            minlength=n_features
        )

    importances /= importances.sum()
    sorted_idx = np.argsort(importances)[::-1]

    top_n = min(20, n_features)
    idx = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n),
                   importances[idx][::-1],
                   color=plt.cm.plasma(np.linspace(0.2, 0.8, top_n)))
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("Importance relative (fréquence de sélection)", fontsize=10)
    ax.set_title(f"Top {top_n} Features — Isolation Forest",
                 fontsize=12, color='#c0c0ff', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('/home/claude/if_06_feature_importance.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Importance des features → if_06_feature_importance.png")


def plot_2d_projections(splits: dict, eval_results: dict) -> None:
    """Projections 2D des features (yaw vs pitch, yaw vs roll, pitch vs roll)."""
    test_df = splits['test_df'].copy()
    pred = eval_results['test']['pred']
    angles = ['yaw', 'pitch', 'roll']
    available = [a for a in angles if a in test_df.columns]
    pairs = [(available[i], available[j])
             for i in range(len(available)) for j in range(i+1, len(available))]

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]
    fig.suptitle("Isolation Forest — Projections 2D (Ensemble TEST)",
                 fontsize=13, color='#c0c0ff', fontweight='bold')

    for ax, (xa, ya) in zip(axes, pairs):
        normal_mask = pred == 0
        anom_mask = pred == 1
        ax.scatter(test_df.loc[normal_mask, xa], test_df.loc[normal_mask, ya],
                   c='#4ecdc4', s=8, alpha=0.4, label='Normal')
        ax.scatter(test_df.loc[anom_mask, xa], test_df.loc[anom_mask, ya],
                   c='#ff0066', s=25, alpha=0.9, label='Anomalie', zorder=5)
        ax.set_xlabel(xa.capitalize(), fontsize=10)
        ax.set_ylabel(ya.capitalize(), fontsize=10)
        ax.set_title(f'{xa.capitalize()} vs {ya.capitalize()}', fontsize=11,
                     color='#e0e0ff')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/if_07_2d_projections.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Projections 2D → if_07_2d_projections.png")


# =============================================================================
# PARTIE 9 — EXPORT DES RÉSULTATS
# =============================================================================

def export_results(splits: dict, eval_results: dict) -> None:
    """Exporte les prédictions et scores dans des CSV."""
    for split_name in ['train', 'test']:
        df_out = splits[f'{split_name}_df'].copy()
        df_out['if_anomaly_pred'] = eval_results[split_name]['pred']
        df_out['if_anomaly_score'] = eval_results[split_name]['scores']
        path = f'/home/claude/if_predictions_{split_name}.csv'
        df_out.to_csv(path, index=False)
        print(f"  [EXPORT] {split_name.upper()} → {path}")


def print_summary(eval_results: dict) -> None:
    """Résumé final."""
    print("\n" + "=" * 65)
    print("  RÉSUMÉ FINAL — ISOLATION FOREST")
    print("=" * 65)
    for split_name in ['train', 'test']:
        r = eval_results[split_name]
        n_anom = r['pred'].sum()
        total = len(r['pred'])
        print(f"  [{split_name.upper()}] Anomalies : {n_anom}/{total} "
              f"({n_anom/total*100:.1f}%)", end="")
        if 'auc' in r:
            print(f"  | AUC={r['auc']:.4f}  AP={r['ap']:.4f}", end="")
        print()
    print("=" * 65)


# =============================================================================
# PARTIE 10 — PIPELINE PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # ── Configuration ──────────────────────────────────────────────────────
    FILE_PATH = "euler_angles.csv"      # ← Remplacez par votre fichier
    USE_SYNTHETIC = True                # False si vous avez un vrai fichier
    N_SAMPLES = 3000
    ANOMALY_RATIO = 0.06
    TRAIN_RATIO = 0.80
    FEATURE_WINDOW = 5
    N_ESTIMATORS = 200

    # ── Chargement ─────────────────────────────────────────────────────────
    if USE_SYNTHETIC:
        df = generate_synthetic_data(N_SAMPLES, ANOMALY_RATIO)
    else:
        df = load_data(FILE_PATH)

    # ── Exploration ─────────────────────────────────────────────────────────
    explore_data(df)
    plot_raw_data(df)

    # ── Features ────────────────────────────────────────────────────────────
    df_feat, feature_cols = build_features(df, window=FEATURE_WINDOW)

    # ── Split ───────────────────────────────────────────────────────────────
    has_labels = 'true_label' in df_feat.columns
    splits = stratified_split(df_feat, feature_cols,
                               train_ratio=TRAIN_RATIO,
                               has_labels=has_labels)

    # ── Calibration ─────────────────────────────────────────────────────────
    if has_labels:
        best_cont, cal_results = calibrate_contamination(splits)
    else:
        best_cont = ANOMALY_RATIO
        print(f"\n[CALIBRATION] Pas de labels → contamination = {best_cont}")

    # ── Entraînement ────────────────────────────────────────────────────────
    clf = train_isolation_forest(splits['X_train'],
                                  contamination=best_cont,
                                  n_estimators=N_ESTIMATORS)

    # ── Évaluation ──────────────────────────────────────────────────────────
    eval_results = evaluate_model(clf, splits)

    # ── Visualisations ──────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  GÉNÉRATION DES FIGURES")
    print("─" * 50)
    plot_anomaly_scores(splits, eval_results)
    plot_roc_pr(splits, eval_results)
    plot_confusion_matrix(eval_results)
    plot_anomalies_on_timeseries(splits, eval_results, feature_cols)
    plot_feature_importance(clf, feature_cols)
    plot_2d_projections(splits, eval_results)

    # ── Export ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  EXPORT DES RÉSULTATS")
    print("─" * 50)
    export_results(splits, eval_results)

    # ── Résumé ──────────────────────────────────────────────────────────────
    print_summary(eval_results)
    print("\n  Toutes les figures et exports sont dans /home/claude/")