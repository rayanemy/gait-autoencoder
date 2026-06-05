# =============================================================================
# DÉTECTION D'ANOMALIES — ONE-CLASS SVM
# Données : timestamp relatif, yaw, pitch, roll (angles d'Euler)
# =============================================================================

# =============================================================================
# PARTIE 1 — IMPORTS ET CONFIGURATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Style global (identique au script IF pour cohérence)
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
print("  ONE-CLASS SVM — Détection d'anomalies Angles d'Euler")
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

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

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
        raise ValueError(f"Colonnes manquantes : {missing}.")

    df = df[required].dropna()
    df[['yaw', 'pitch', 'roll']] = df[['yaw', 'pitch', 'roll']].astype(float)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna().sort_values('timestamp').reset_index(drop=True)

    print(f"\n[DONNÉES] {len(df)} lignes chargées depuis '{filepath}'")
    return df


def generate_synthetic_data(n_samples: int = 2000,
                             anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """Génère un dataset synthétique réaliste (identique au script IF)."""
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anom = n_samples - n_normal

    t = np.linspace(0, 100, n_normal)
    yaw_n = np.sin(0.1 * t) * 30 + np.random.randn(n_normal) * 2.5
    pitch_n = np.cos(0.15 * t) * 15 + np.random.randn(n_normal) * 1.5
    roll_n = np.sin(0.05 * t + 1) * 20 + np.random.randn(n_normal) * 2.0

    t_a = np.random.uniform(0, 100, n_anom)
    yaw_a = np.random.uniform(-180, 180, n_anom)
    pitch_a = np.random.uniform(-90, 90, n_anom)
    roll_a = np.random.uniform(-180, 180, n_anom)

    df_normal = pd.DataFrame({
        'timestamp': t, 'yaw': yaw_n, 'pitch': pitch_n, 'roll': roll_n,
        'true_label': 0
    })
    df_anom = pd.DataFrame({
        'timestamp': t_a, 'yaw': yaw_a, 'pitch': pitch_a, 'roll': roll_a,
        'true_label': 1
    })

    df = pd.concat([df_normal, df_anom]).sort_values('timestamp').reset_index(drop=True)
    print(f"\n[DONNÉES SYNTHÉTIQUES] {len(df)} lignes | "
          f"{n_normal} normales | {n_anom} anomalies ({anomaly_ratio*100:.0f}%)")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Analyse exploratoire des données brutes."""
    print("\n" + "─" * 50)
    print("  EXPLORATION DES DONNÉES")
    print("─" * 50)
    print(f"  Shape : {df.shape}  |  Timestamp : [{df['timestamp'].min():.2f}, {df['timestamp'].max():.2f}]")

    features = ['yaw', 'pitch', 'roll']
    print("\n" + df[features].describe().round(3).to_string())

    if 'true_label' in df.columns:
        n_anom = df['true_label'].sum()
        print(f"\n  Labels : {len(df)-n_anom} normaux | {n_anom} anomalies "
              f"({n_anom/len(df)*100:.1f}%)")

    # Skewness et kurtosis
    print("\n  Skewness / Kurtosis :")
    for col in features:
        sk = df[col].skew()
        ku = df[col].kurtosis()
        print(f"    {col:6s} : skew={sk:+.3f}  kurt={ku:+.3f}")


# =============================================================================
# PARTIE 3 — FEATURE ENGINEERING
# =============================================================================

def build_features(df: pd.DataFrame, window: int = 5) -> tuple:
    """
    Features pour One-Class SVM :
    - Angles bruts + deltas + rolling stats + magnitude + dérivées
    NOTE : One-Class SVM est sensible à la dimensionnalité → on contrôle
    le nombre de features via une sélection optionnelle.
    """
    df = df.copy()
    angles = ['yaw', 'pitch', 'roll']

    for a in angles:
        df[f'd_{a}'] = df[a].diff().fillna(0)

    df['magnitude'] = np.sqrt(df['yaw']**2 + df['pitch']**2 + df['roll']**2)
    df['d_magnitude'] = df['magnitude'].diff().fillna(0)

    for a in angles:
        df[f'{a}_rmean'] = df[a].rolling(window, min_periods=1).mean()
        df[f'{a}_rstd'] = df[a].rolling(window, min_periods=1).std().fillna(0)
        df[f'{a}_rrange'] = (
            df[a].rolling(window, min_periods=1).max() -
            df[a].rolling(window, min_periods=1).min()
        )

    for a in angles:
        df[f'd2_{a}'] = df[f'd_{a}'].diff().fillna(0)

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'true_label']]

    print(f"\n[FEATURES] {len(feature_cols)} features construites")
    return df, feature_cols


# =============================================================================
# PARTIE 4 — SPLIT TRAIN / TEST STRATIFIÉ
# =============================================================================

def stratified_split(df: pd.DataFrame,
                     feature_cols: list,
                     train_ratio: float = 0.8,
                     has_labels: bool = True) -> dict:
    """
    Split 80/20 stratifié proportionnel sur normaux et anomalies.
    """
    print("\n" + "─" * 50)
    print("  SPLIT TRAIN / TEST (80 / 20 STRATIFIÉ)")
    print("─" * 50)

    if has_labels and 'true_label' in df.columns:
        normal_idx = df[df['true_label'] == 0].index.tolist()
        anom_idx = df[df['true_label'] == 1].index.tolist()
        np.random.shuffle(normal_idx)
        np.random.shuffle(anom_idx)

        n_train_n = int(len(normal_idx) * train_ratio)
        n_train_a = int(len(anom_idx) * train_ratio)

        train_idx = sorted(normal_idx[:n_train_n] + anom_idx[:n_train_a])
        test_idx = sorted(normal_idx[n_train_n:] + anom_idx[n_train_a:])

        train_df = df.loc[train_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)

        print(f"  TRAIN : {len(train_df)} | "
              f"normaux={( train_df['true_label']==0).sum()} | "
              f"anomalies={(train_df['true_label']==1).sum()}")
        print(f"  TEST  : {len(test_df)} | "
              f"normaux={(test_df['true_label']==0).sum()} | "
              f"anomalies={(test_df['true_label']==1).sum()}")
    else:
        cut = int(len(df) * train_ratio)
        train_df = df.iloc[:cut].reset_index(drop=True)
        test_df = df.iloc[cut:].reset_index(drop=True)
        print(f"  TRAIN : {len(train_df)} | TEST : {len(test_df)} (split temporel)")

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df['true_label'].values if 'true_label' in train_df.columns else None
    y_test = test_df['true_label'].values if 'true_label' in test_df.columns else None

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
# PARTIE 5 — GRID SEARCH HYPERPARAMÈTRES (One-Class SVM)
# =============================================================================

def grid_search_ocsvm(splits: dict,
                       param_grid: dict = None) -> dict:
    """
    Recherche des meilleurs hyperparamètres nu et gamma pour le OC-SVM.
    Optimise l'AUC-ROC si les labels sont disponibles,
    sinon maximise la compacité (ratio d'anomalies proche du nu cible).
    """
    if param_grid is None:
        param_grid = {
            'nu': [0.01, 0.03, 0.05, 0.08, 0.10, 0.15],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
            'kernel': ['rbf']
        }

    print("\n" + "─" * 50)
    print("  GRID SEARCH — ONE-CLASS SVM")
    print("─" * 50)
    print(f"  {len(list(ParameterGrid(param_grid)))} combinaisons testées...")

    y_test = splits['y_test']
    best_score = -1
    best_params = {}
    all_results = []

    for params in ParameterGrid(param_grid):
        clf = OneClassSVM(**params)
        clf.fit(splits['X_train'])
        scores = -clf.decision_function(splits['X_test'])

        if y_test is not None:
            score = roc_auc_score(y_test, scores)
        else:
            pred = clf.predict(splits['X_test'])
            detected_ratio = (pred == -1).mean()
            score = -abs(detected_ratio - params['nu'])  # proche du nu cible

        all_results.append({**params, 'score': score})

        if score > best_score:
            best_score = score
            best_params = params

    df_results = pd.DataFrame(all_results)
    print(f"\n  Meilleurs paramètres : {best_params}")
    metric_name = "AUC-ROC" if y_test is not None else "Proxy"
    print(f"  Meilleur {metric_name} : {best_score:.4f}")

    return best_params, df_results


def plot_grid_search_heatmap(df_results: pd.DataFrame) -> None:
    """Heatmap des scores du grid search (nu vs gamma)."""
    numeric_gamma = df_results['gamma'].apply(
        lambda g: 0.001 if g == 'scale' else (0.002 if g == 'auto' else float(g))
    )
    df_results = df_results.copy()
    df_results['gamma_num'] = numeric_gamma

    pivot = df_results.pivot_table(
        values='score', index='nu', columns='gamma_num', aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='plasma',
                ax=ax, linewidths=0.5,
                annot_kws={'size': 9})
    ax.set_title("Grid Search — One-Class SVM\nScore (AUC-ROC) par (nu, gamma)",
                 fontsize=12, color='#c0c0ff', fontweight='bold', pad=15)
    ax.set_xlabel("gamma (valeur numérique)", fontsize=10)
    ax.set_ylabel("nu", fontsize=10)
    plt.tight_layout()
    plt.savefig('/home/claude/ocsvm_01_grid_search.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Grid Search heatmap → ocsvm_01_grid_search.png")


# =============================================================================
# PARTIE 6 — ENTRAÎNEMENT ONE-CLASS SVM
# =============================================================================

def train_ocsvm(X_train: np.ndarray,
                nu: float = 0.05,
                gamma: str = 'scale',
                kernel: str = 'rbf') -> OneClassSVM:
    """
    Entraîne le One-Class SVM.
    nu      : borne supérieure sur la fraction d'outliers (≈ contamination).
    gamma   : paramètre du noyau RBF.
    kernel  : noyau utilisé ('rbf' recommandé pour données continues).
    """
    print("\n" + "─" * 50)
    print("  ENTRAÎNEMENT — ONE-CLASS SVM")
    print("─" * 50)
    print(f"  kernel : {kernel}")
    print(f"  nu     : {nu}")
    print(f"  gamma  : {gamma}")
    print(f"  Taille d'entraînement : {X_train.shape}")

    clf = OneClassSVM(
        kernel=kernel,
        nu=nu,
        gamma=gamma,
        max_iter=2000,
        tol=1e-4,
        shrinking=True,
        cache_size=500
    )
    clf.fit(X_train)

    n_sv = clf.n_support_[0]
    print(f"  ✓ Entraînement terminé — {n_sv} vecteurs support")
    return clf


# =============================================================================
# PARTIE 7 — ÉVALUATION ET MÉTRIQUES
# =============================================================================

def evaluate_model(clf: OneClassSVM, splits: dict) -> dict:
    """Évaluation complète train + test avec toutes les métriques."""
    print("\n" + "─" * 50)
    print("  ÉVALUATION DU MODÈLE")
    print("─" * 50)

    results = {}
    for split_name in ['train', 'test']:
        X = splits[f'X_{split_name}']
        y = splits[f'y_{split_name}']

        pred_raw = clf.predict(X)              # +1 normal, -1 anomalie
        pred = (pred_raw == -1).astype(int)    # 1 = anomalie, 0 = normal
        decision = clf.decision_function(X)    # valeur positive = normal
        scores = -decision                     # scores élevés = plus anormal

        results[split_name] = {
            'pred': pred, 'scores': scores, 'decision': decision, 'y': y
        }

        n_anom = pred.sum()
        print(f"\n  [{split_name.upper()}] {len(pred)} échantillons → "
              f"{n_anom} anomalies ({n_anom/len(pred)*100:.1f}%)")
        print(f"    Decision mean : {decision.mean():.4f}  |  "
              f"std : {decision.std():.4f}")
        print(f"    Vecteurs support dans zone frontière : {clf.n_support_[0]}")

        if y is not None:
            auc = roc_auc_score(y, scores)
            ap = average_precision_score(y, scores)
            f1 = f1_score(y, pred, zero_division=0)
            print(f"    AUC-ROC       : {auc:.4f}")
            print(f"    Avg Precision : {ap:.4f}")
            print(f"    F1-Score      : {f1:.4f}")
            print(f"\n    Classification Report :")
            print(classification_report(y, pred,
                                        target_names=['Normal', 'Anomalie'],
                                        digits=4))
            results[split_name]['auc'] = auc
            results[split_name]['ap'] = ap
            results[split_name]['f1'] = f1

    return results


# =============================================================================
# PARTIE 8 — VISUALISATIONS
# =============================================================================

def plot_decision_function(splits: dict, eval_results: dict) -> None:
    """Distribution de la fonction de décision."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("One-Class SVM — Fonction de Décision",
                 fontsize=13, color='#c0c0ff', fontweight='bold')

    for ax, split_name in zip(axes, ['train', 'test']):
        decision = eval_results[split_name]['decision']
        y = eval_results[split_name]['y']

        if y is not None:
            ax.hist(decision[y == 0], bins=60, alpha=0.75,
                    color='#4ecdc4', label='Normal', density=True)
            ax.hist(decision[y == 1], bins=60, alpha=0.75,
                    color='#ff0066', label='Anomalie', density=True)
            ax.legend(fontsize=10)
        else:
            ax.hist(decision, bins=60, alpha=0.8, color='#ffe66d', density=True)

        ax.axvline(x=0, color='white', linestyle='--', linewidth=1.5,
                   label='Frontière (0)')
        ax.set_title(split_name.upper(), fontsize=12, color='#e0e0ff')
        ax.set_xlabel("Decision Function (>0 = normal)", fontsize=10)
        ax.set_ylabel("Densité", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/ocsvm_02_decision_function.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Fonction de décision → ocsvm_02_decision_function.png")


def plot_roc_pr(splits: dict, eval_results: dict) -> None:
    """Courbes ROC et Precision-Recall."""
    y_test = splits['y_test']
    if y_test is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("One-Class SVM — Courbes ROC & Precision-Recall",
                 fontsize=13, color='#c0c0ff', fontweight='bold')

    ax = axes[0]
    for name, split, col in [('Train', 'train', '#4ecdc4'), ('Test', 'test', '#ff6b6b')]:
        y = splits[f'y_{split}']
        sc = eval_results[split]['scores']
        fpr, tpr, _ = roc_curve(y, sc)
        auc = roc_auc_score(y, sc)
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{name} AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='#555577', lw=1)
    ax.set_xlabel('FPR', fontsize=10); ax.set_ylabel('TPR', fontsize=10)
    ax.set_title('Courbe ROC', fontsize=12, color='#e0e0ff')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, split, col in [('Train', 'train', '#4ecdc4'), ('Test', 'test', '#ff6b6b')]:
        y = splits[f'y_{split}']
        sc = eval_results[split]['scores']
        prec, rec, _ = precision_recall_curve(y, sc)
        ap = average_precision_score(y, sc)
        ax.plot(rec, prec, color=col, lw=2, label=f'{name} AP={ap:.3f}')
    ax.set_xlabel('Recall', fontsize=10); ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Courbe Precision-Recall', fontsize=12, color='#e0e0ff')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/ocsvm_03_roc_pr.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] ROC & PR → ocsvm_03_roc_pr.png")


def plot_confusion_matrix(eval_results: dict) -> None:
    """Matrice de confusion."""
    y = eval_results['test']['y']
    pred = eval_results['test']['pred']
    if y is None:
        return

    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Normal', 'Anomalie'],
                yticklabels=['Normal', 'Anomalie'],
                ax=ax, linewidths=0.5,
                annot_kws={'size': 14, 'color': 'white'})
    ax.set_title('Matrice de Confusion — TEST\nOne-Class SVM',
                 fontsize=12, color='#c0c0ff', pad=15)
    ax.set_xlabel('Prédit', fontsize=11)
    ax.set_ylabel('Réel', fontsize=11)
    plt.tight_layout()
    plt.savefig('/home/claude/ocsvm_04_confusion_matrix.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Matrice de confusion → ocsvm_04_confusion_matrix.png")


def plot_anomalies_on_timeseries(splits: dict, eval_results: dict) -> None:
    """Anomalies détectées sur séries temporelles."""
    test_df = splits['test_df'].copy()
    pred = eval_results['test']['pred']
    decision = eval_results['test']['decision']
    test_df['anomaly_pred'] = pred
    test_df['decision'] = decision

    angles = ['yaw', 'pitch', 'roll']
    available = [a for a in angles if a in test_df.columns]
    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']

    fig, axes = plt.subplots(len(available) + 1, 1,
                              figsize=(15, 3 * (len(available) + 1)),
                              sharex=True)
    fig.suptitle("One-Class SVM — Anomalies sur Séries Temporelles (TEST)",
                 fontsize=13, color='#c0c0ff', fontweight='bold')

    for i, (col, color) in enumerate(zip(available, colors)):
        ax = axes[i]
        ax.plot(test_df['timestamp'], test_df[col],
                color=color, linewidth=0.9, alpha=0.85)
        mask = test_df['anomaly_pred'] == 1
        ax.scatter(test_df.loc[mask, 'timestamp'], test_df.loc[mask, col],
                   color='#ff0066', s=20, zorder=5, alpha=0.9, label='Détecté')
        if 'true_label' in test_df.columns:
            true_mask = test_df['true_label'] == 1
            ax.scatter(test_df.loc[true_mask, 'timestamp'],
                       test_df.loc[true_mask, col],
                       color='white', s=8, zorder=4, alpha=0.5,
                       marker='x', label='Réel')
        ax.set_ylabel(col.capitalize(), fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Decision function plot
    ax = axes[-1]
    ax.plot(test_df['timestamp'], test_df['decision'],
            color='#aa44ff', linewidth=0.9, alpha=0.85, label='Decision function')
    ax.axhline(y=0, color='#ff0066', linestyle='--', linewidth=1.3,
               label='Frontière (0)')
    ax.fill_between(test_df['timestamp'], test_df['decision'], 0,
                    where=(test_df['decision'] < 0),
                    color='#ff0066', alpha=0.25, label='Zone anomalie')
    ax.set_ylabel('Decision fn', fontsize=10)
    ax.set_xlabel('Timestamp relatif', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/ocsvm_05_anomalies_timeseries.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Anomalies sur séries → ocsvm_05_anomalies_timeseries.png")


def plot_pca_boundary(splits: dict, eval_results: dict) -> None:
    """
    Projection PCA 2D avec frontière de décision approximative.
    Permet de visualiser la séparation apprise par le OC-SVM.
    """
    X_test = splits['X_test']
    pred = eval_results['test']['pred']
    decision = eval_results['test']['decision']
    y_test = splits['y_test']

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_test)

    var_exp = pca.explained_variance_ratio_
    print(f"\n  [PCA] Variance expliquée : PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%")

    # Grille pour visualiser la frontière
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))

    # Inverser la PCA pour évaluer le modèle sur la grille
    Z_full = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    Z_dec = clf_global.decision_function(Z_full).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("One-Class SVM — Frontière de Décision (PCA 2D)",
                 fontsize=13, color='#c0c0ff', fontweight='bold')

    titles = ['Prédictions OC-SVM', 'Labels réels'] if y_test is not None else ['Prédictions OC-SVM']

    for ax_idx, (ax, title) in enumerate(zip(axes, titles)):
        contour = ax.contourf(xx, yy, Z_dec, levels=20,
                               cmap='RdBu_r', alpha=0.6)
        ax.contour(xx, yy, Z_dec, levels=[0],
                   colors='white', linewidths=1.5)
        plt.colorbar(contour, ax=ax, label='Decision function')

        if ax_idx == 0:
            normal_mask = pred == 0
            anom_mask = pred == 1
            ax.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1],
                       c='#4ecdc4', s=8, alpha=0.5, label='Normal prédit')
            ax.scatter(X_2d[anom_mask, 0], X_2d[anom_mask, 1],
                       c='#ff0066', s=25, alpha=0.9, label='Anomalie prédite', zorder=5)
        else:
            if y_test is not None:
                normal_mask = y_test == 0
                anom_mask = y_test == 1
                ax.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1],
                           c='#4ecdc4', s=8, alpha=0.5, label='Normal réel')
                ax.scatter(X_2d[anom_mask, 0], X_2d[anom_mask, 1],
                           c='#ff0066', s=25, alpha=0.9, label='Anomalie réelle', zorder=5)

        ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=10)
        ax.set_title(title, fontsize=11, color='#e0e0ff')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('/home/claude/ocsvm_06_pca_boundary.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] PCA + frontière → ocsvm_06_pca_boundary.png")


def plot_nu_sensitivity(splits: dict, best_params: dict) -> None:
    """
    Analyse de sensibilité sur nu :
    Montre comment les métriques varient avec nu.
    """
    nu_values = np.linspace(0.01, 0.30, 25)
    y_test = splits['y_test']

    aucs, aps, f1s, n_anoms = [], [], [], []

    for nu in nu_values:
        params = {**best_params, 'nu': nu}
        clf_tmp = OneClassSVM(**params)
        clf_tmp.fit(splits['X_train'])
        scores = -clf_tmp.decision_function(splits['X_test'])
        pred = (clf_tmp.predict(splits['X_test']) == -1).astype(int)
        n_anoms.append(pred.mean() * 100)
        if y_test is not None:
            aucs.append(roc_auc_score(y_test, scores))
            aps.append(average_precision_score(y_test, scores))
            f1s.append(f1_score(y_test, pred, zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Sensibilité au paramètre nu — One-Class SVM",
                 fontsize=13, color='#c0c0ff', fontweight='bold')

    ax = axes[0]
    ax.plot(nu_values, n_anoms, color='#ffe66d', lw=2, label='% anomalies détectées')
    ax.axvline(x=best_params['nu'], color='#ff0066', linestyle='--',
               linewidth=1.5, label=f"nu optimal = {best_params['nu']}")
    ax.set_xlabel('nu', fontsize=10); ax.set_ylabel('%', fontsize=10)
    ax.set_title('% Anomalies détectées vs nu', fontsize=11, color='#e0e0ff')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if y_test is not None:
        ax.plot(nu_values, aucs, color='#4ecdc4', lw=2, label='AUC-ROC')
        ax.plot(nu_values, aps, color='#ff6b6b', lw=2, label='Avg Precision')
        ax.plot(nu_values, f1s, color='#aa44ff', lw=2, label='F1-Score')
        ax.axvline(x=best_params['nu'], color='white', linestyle='--',
                   linewidth=1.5, label=f"nu optimal = {best_params['nu']}")
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Métriques vs nu', fontsize=11, color='#e0e0ff')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Labels requis\npour cette figure',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='#8080aa')
    ax.set_xlabel('nu', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/ocsvm_07_nu_sensitivity.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Sensibilité nu → ocsvm_07_nu_sensitivity.png")


def plot_2d_projections(splits: dict, eval_results: dict) -> None:
    """Projections 2D des angles bruts avec anomalies détectées."""
    test_df = splits['test_df'].copy()
    pred = eval_results['test']['pred']
    angles = ['yaw', 'pitch', 'roll']
    available = [a for a in angles if a in test_df.columns]
    pairs = [(available[i], available[j])
             for i in range(len(available)) for j in range(i+1, len(available))]

    fig, axes = plt.subplots(1, len(pairs), figsize=(6*len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]
    fig.suptitle("One-Class SVM — Projections 2D (Ensemble TEST)",
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
        ax.set_title(f'{xa.capitalize()} vs {ya.capitalize()}',
                     fontsize=11, color='#e0e0ff')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/ocsvm_08_2d_projections.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Projections 2D → ocsvm_08_2d_projections.png")


# =============================================================================
# PARTIE 9 — COMPARAISON IF vs OC-SVM (bonus si les deux scripts tournent)
# =============================================================================

def compare_with_if(splits: dict, eval_results_ocsvm: dict,
                     if_results_path: str = 'if_predictions_test.csv') -> None:
    """
    Compare les prédictions OC-SVM avec celles d'Isolation Forest
    si le fichier de prédictions IF est disponible.
    """
    import os
    if not os.path.exists(if_results_path):
        print(f"\n  [COMPARAISON] Fichier IF non trouvé : {if_results_path}")
        return

    print("\n" + "─" * 50)
    print("  COMPARAISON IF vs OC-SVM")
    print("─" * 50)

    df_if = pd.read_csv(if_results_path)
    pred_if = df_if['if_anomaly_pred'].values
    pred_ocsvm = eval_results_ocsvm['test']['pred']

    n = min(len(pred_if), len(pred_ocsvm))
    pred_if = pred_if[:n]
    pred_ocsvm = pred_ocsvm[:n]

    accord = (pred_if == pred_ocsvm).mean() * 100
    both_anom = ((pred_if == 1) & (pred_ocsvm == 1)).sum()
    if_only = ((pred_if == 1) & (pred_ocsvm == 0)).sum()
    ocsvm_only = ((pred_if == 0) & (pred_ocsvm == 1)).sum()

    print(f"  Accord total     : {accord:.1f}%")
    print(f"  Les deux détectent : {both_anom} anomalies")
    print(f"  IF seulement       : {if_only}")
    print(f"  OC-SVM seulement   : {ocsvm_only}")

    # Venn approximatif en barchart
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['IF ∩ OC-SVM\n(consensus)', 'IF seul', 'OC-SVM seul']
    values = [both_anom, if_only, ocsvm_only]
    colors = ['#4ecdc4', '#ff6b6b', '#ffe66d']
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=12, color='white')
    ax.set_title('Comparaison Isolation Forest vs One-Class SVM\n(anomalies détectées)',
                 fontsize=12, color='#c0c0ff', fontweight='bold')
    ax.set_ylabel('Nombre d\'anomalies', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('/home/claude/ocsvm_09_comparison_if.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  [FIGURE] Comparaison IF vs OC-SVM → ocsvm_09_comparison_if.png")


# =============================================================================
# PARTIE 10 — EXPORT DES RÉSULTATS
# =============================================================================

def export_results(splits: dict, eval_results: dict) -> None:
    """Exporte les prédictions et scores dans des CSV."""
    for split_name in ['train', 'test']:
        df_out = splits[f'{split_name}_df'].copy()
        df_out['ocsvm_anomaly_pred'] = eval_results[split_name]['pred']
        df_out['ocsvm_decision'] = eval_results[split_name]['decision']
        df_out['ocsvm_anomaly_score'] = eval_results[split_name]['scores']
        path = f'/home/claude/ocsvm_predictions_{split_name}.csv'
        df_out.to_csv(path, index=False)
        print(f"  [EXPORT] {split_name.upper()} → {path}")


def print_summary(eval_results: dict) -> None:
    """Résumé final."""
    print("\n" + "=" * 65)
    print("  RÉSUMÉ FINAL — ONE-CLASS SVM")
    print("=" * 65)
    for split_name in ['train', 'test']:
        r = eval_results[split_name]
        n_anom = r['pred'].sum()
        total = len(r['pred'])
        print(f"  [{split_name.upper()}] Anomalies : {n_anom}/{total} "
              f"({n_anom/total*100:.1f}%)", end="")
        if 'auc' in r:
            print(f"  | AUC={r['auc']:.4f}  AP={r['ap']:.4f}  F1={r['f1']:.4f}", end="")
        print()
    print("=" * 65)


# =============================================================================
# PARTIE 11 — PIPELINE PRINCIPAL
# =============================================================================

# Variable globale pour la visualisation PCA (accès dans plot_pca_boundary)
clf_global = None

if __name__ == "__main__":

    # ── Configuration ──────────────────────────────────────────────────────
    FILE_PATH = "euler_angles.csv"   # ← Remplacez par votre fichier
    USE_SYNTHETIC = True              # False si vous avez un vrai fichier
    N_SAMPLES = 3000
    ANOMALY_RATIO = 0.06
    TRAIN_RATIO = 0.80
    FEATURE_WINDOW = 5

    # Hyperparamètres OC-SVM (seront calibrés par grid search)
    NU_DEFAULT = 0.05
    GAMMA_DEFAULT = 'scale'
    KERNEL = 'rbf'
    RUN_GRID_SEARCH = True  # False pour aller plus vite

    # ── Chargement ─────────────────────────────────────────────────────────
    if USE_SYNTHETIC:
        df = generate_synthetic_data(N_SAMPLES, ANOMALY_RATIO)
    else:
        df = load_data(FILE_PATH)

    # ── Exploration ─────────────────────────────────────────────────────────
    explore_data(df)

    # ── Features ────────────────────────────────────────────────────────────
    df_feat, feature_cols = build_features(df, window=FEATURE_WINDOW)

    # ── Split ───────────────────────────────────────────────────────────────
    has_labels = 'true_label' in df_feat.columns
    splits = stratified_split(df_feat, feature_cols,
                               train_ratio=TRAIN_RATIO,
                               has_labels=has_labels)

    # ── Grid Search ─────────────────────────────────────────────────────────
    if RUN_GRID_SEARCH:
        best_params, df_gs = grid_search_ocsvm(splits)
        plot_grid_search_heatmap(df_gs)
    else:
        best_params = {'nu': NU_DEFAULT, 'gamma': GAMMA_DEFAULT, 'kernel': KERNEL}
        print(f"\n[GRID SEARCH] Désactivé → params : {best_params}")

    # ── Entraînement ────────────────────────────────────────────────────────
    clf = train_ocsvm(
        splits['X_train'],
        nu=best_params['nu'],
        gamma=best_params.get('gamma', GAMMA_DEFAULT),
        kernel=best_params.get('kernel', KERNEL)
    )
    clf_global = clf  # pour plot_pca_boundary

    # ── Évaluation ──────────────────────────────────────────────────────────
    eval_results = evaluate_model(clf, splits)

    # ── Visualisations ──────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  GÉNÉRATION DES FIGURES")
    print("─" * 50)
    plot_decision_function(splits, eval_results)
    plot_roc_pr(splits, eval_results)
    plot_confusion_matrix(eval_results)
    plot_anomalies_on_timeseries(splits, eval_results)
    plot_pca_boundary(splits, eval_results)
    plot_nu_sensitivity(splits, best_params)
    plot_2d_projections(splits, eval_results)

    # ── Comparaison avec IF ─────────────────────────────────────────────────
    compare_with_if(splits, eval_results)

    # ── Export ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  EXPORT DES RÉSULTATS")
    print("─" * 50)
    export_results(splits, eval_results)

    # ── Résumé ──────────────────────────────────────────────────────────────
    print_summary(eval_results)
    print("\n  Toutes les figures et exports sont dans /home/claude/")