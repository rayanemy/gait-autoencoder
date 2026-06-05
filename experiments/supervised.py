"""
=============================================================================
PIPELINE ML - SEMELLE CONNECTÉE
Détection d'anomalies de démarche à partir de données IMU + Flexiforce
=============================================================================
Variables :
  - yaw, pitch, roll       : angles IMU (degrés)
  - flex1, flex2, flex3    : état Flexiforce (0=non appuyé, 1=appuyé)
  - anomalie               : False (normal) | True (anomalie)

Fichier 1 : demarche_normale.txt   → anomalie = False
Fichier 2 : demarche_anomalie.txt  → anomalie = True
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, f1_score, accuracy_score
)

# Modèles
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# =============================================================================
# SECTION 1 — DONNÉES SYNTHÉTIQUES (simulation fichiers texte)
# =============================================================================
# Si tu as de vrais fichiers, remplace les StringIO par :
#   df1 = pd.read_csv("demarche_normale.txt")
#   df2 = pd.read_csv("demarche_anomalie.txt")

print("=" * 65)
print("  PIPELINE ML — SEMELLE CONNECTÉE")
print("=" * 65)

np.random.seed(42)
N_NORMAL  = 500   # lignes demarche normale
N_ANOMALY = 200   # lignes demarche avec anomalie

def gen_normal(n):
    """Demarche normale : yaw centré, flex3 activé (talon), flex1/2 variables."""
    return pd.DataFrame({
        "yaw":   np.random.normal(0,   8,  n),       # peu de déviation latérale
        "pitch": np.random.normal(5,   4,  n),       # légère flexion dorsale
        "roll":  np.random.normal(0,   5,  n),       # centré → neutre
        "flex1": np.random.choice([0, 1], n, p=[0.3, 0.7]),  # avant-pied
        "flex2": np.random.choice([0, 1], n, p=[0.4, 0.6]),  # milieu
        "flex3": np.random.choice([0, 1], n, p=[0.1, 0.9]),  # talon toujours actif
    })

def gen_anomaly(n):
    """Anomalie : roll élevé (pronation), flex3 souvent absent (pas de talon)."""
    return pd.DataFrame({
        "yaw":   np.random.normal(15,  12, n),       # déviation latérale forte
        "pitch": np.random.normal(-3,  7,  n),       # hyperextension
        "roll":  np.random.normal(18,  10, n),       # pronation marquée
        "flex1": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "flex2": np.random.choice([0, 1], n, p=[0.5, 0.5]),
        "flex3": np.random.choice([0, 1], n, p=[0.8, 0.2]),  # talon absent → pathologie
    })

# --- Créer les fichiers texte simulés ---
df_normal_raw  = gen_normal(N_NORMAL)
df_anomaly_raw = gen_anomaly(N_ANOMALY)

buf1 = StringIO()
buf2 = StringIO()
df_normal_raw.to_csv(buf1,  index=False)
df_anomaly_raw.to_csv(buf2, index=False)
buf1.seek(0); buf2.seek(0)

# =============================================================================
# SECTION 2 — LECTURE ET CONSTRUCTION DES TABLEAUX PANDAS
# =============================================================================
print("\n[1/6] Lecture des fichiers texte et construction des DataFrames...")

# Fichier 1 : demarche normale → Anomalie = False
df_normal = pd.read_csv(buf1)
df_normal["Anomalie"] = False
print(f"  ✓ Fichier 1 — {len(df_normal)} lignes | colonnes : {list(df_normal.columns)}")

# Fichier 2 : demarche avec anomalie → Anomalie = True
df_anomaly = pd.read_csv(buf2)
df_anomaly["Anomalie"] = True
print(f"  ✓ Fichier 2 — {len(df_anomaly)} lignes | colonnes : {list(df_anomaly.columns)}")

print("\n  Aperçu — Demarche normale :")
print(df_normal.head(3).to_string(index=False))
print("\n  Aperçu — Demarche anomalie :")
print(df_anomaly.head(3).to_string(index=False))

# =============================================================================
# SECTION 3 — FUSION ET SPLIT TRAIN/TEST PROPORTIONNEL 80/20
# =============================================================================
print("\n[2/6] Fusion et split train/test proportionnel (80/20)...")

df_full = pd.concat([df_normal, df_anomaly], ignore_index=True)
df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Table fusionnée : {len(df_full)} lignes")
print(f"  Répartition anomalie : {df_full['Anomalie'].value_counts().to_dict()}")

X = df_full[["yaw", "pitch", "roll", "flex1", "flex2", "flex3"]]
y = df_full["Anomalie"].astype(int)   # False→0, True→1

# stratify=y garantit la proportionnalité 80/20 sur chaque classe
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n  Train : {len(X_train)} lignes "
      f"(anomalie={y_train.sum()}, normal={len(y_train)-y_train.sum()})")
print(f"  Test  : {len(X_test)}  lignes "
      f"(anomalie={y_test.sum()}, normal={len(y_test)-y_test.sum()})")

# =============================================================================
# SECTION 4 — MODÈLES RECOMMANDÉS ET LEUR RATIONALE
# =============================================================================
print("\n[3/6] Modèles sélectionnés...")

"""
MODÈLES RECOMMANDÉS POUR CE PROJET :
══════════════════════════════════════════════════════════════════
1. Random Forest      — robuste aux valeurs aberrantes, interprétable
                        (importance des features → quel capteur décide ?)
2. Gradient Boosting  — précision élevée sur petits datasets tabulaires
3. XGBoost            — version optimisée GB, très utilisée en compétition
4. SVM (RBF kernel)   — efficace en haute dimension, bon avec IMU
5. MLP (réseau dense) — capture des interactions non-linéaires complexes
6. KNN                — baseline simple, sensible à la localité des patterns
7. Logistic Reg.      — baseline linéaire, rapide, interprétable
8. AdaBoost           — boosting léger, bon complément à RF

NON recommandés ici :
- Naive Bayes : suppose indépendance des capteurs (faux ici)
- Decision Tree seul : sur-apprentissage sans ensemble
══════════════════════════════════════════════════════════════════
"""

# =============================================================================
# SECTION 5 — PRÉPARATION DES DONNÉES PAR MODÈLE + ENTRAÎNEMENT
# =============================================================================
print("[4/6] Entraînement des modèles...\n")

# --- Données brutes (arbres, KNN, AdaBoost ne nécessitent pas de scaling)
X_train_raw = X_train.copy()
X_test_raw  = X_test.copy()

# --- Données normalisées (SVM, MLP, Logistic Regression, KNN avec scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Dictionnaire modèle → (X_train adapté, X_test adapté, params initiaux)
models_config = {
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=200, max_depth=8,
                                        class_weight="balanced", random_state=42),
        "X_train": X_train_raw,
        "X_test":  X_test_raw,
        "note": "Features brutes — arbres insensibles au scaling"
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                            max_depth=4, random_state=42),
        "X_train": X_train_raw,
        "X_test":  X_test_raw,
        "note": "Features brutes — boosting natif"
    },
    "XGBoost": {
        "model": XGBClassifier(n_estimators=200, learning_rate=0.05,
                               max_depth=4, scale_pos_weight=N_NORMAL/N_ANOMALY,
                               eval_metric="logloss", random_state=42),
        "X_train": X_train_raw,
        "X_test":  X_test_raw,
        "note": "scale_pos_weight compense le déséquilibre de classes"
    },
    "SVM (RBF)": {
        "model": SVC(kernel="rbf", C=10, gamma="scale",
                     class_weight="balanced", probability=True, random_state=42),
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
        "note": "StandardScaler appliqué — SVM sensible aux échelles"
    },
    "MLP": {
        "model": MLPClassifier(hidden_layer_sizes=(64, 32, 16),
                               activation="relu", max_iter=500,
                               early_stopping=True, random_state=42),
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
        "note": "StandardScaler appliqué — réseau de neurones dense"
    },
    "KNN": {
        "model": KNeighborsClassifier(n_neighbors=7, weights="distance",
                                      metric="minkowski"),
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
        "note": "StandardScaler appliqué — KNN très sensible aux distances"
    },
    "Logistic Regression": {
        "model": LogisticRegression(C=1.0, class_weight="balanced",
                                    max_iter=500, random_state=42),
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
        "note": "StandardScaler appliqué — modèle linéaire baseline"
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(n_estimators=100, learning_rate=0.5,
                                    random_state=42),
        "X_train": X_train_raw,
        "X_test":  X_test_raw,
        "note": "Features brutes — boosting léger"
    },
}

results = {}

for name, cfg in models_config.items():
    print(f"  ▶ {name} ({cfg['note']})")
    clf = cfg["model"]
    clf.fit(cfg["X_train"], y_train)
    y_pred = clf.predict(cfg["X_test"])

    cm  = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0   # recall anomalie
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0   # recall normal

    results[name] = {
        "model":       clf,
        "y_pred":      y_pred,
        "cm":          cm,
        "accuracy":    acc,
        "f1":          f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }
    print(f"     Accuracy={acc:.3f}  F1={f1:.3f}  "
          f"Sensibilité={sensitivity:.3f}  Spécificité={specificity:.3f}")

# =============================================================================
# SECTION 6 — MATRICES DE CONFUSION (visualisation)
# =============================================================================
print("\n[5/6] Génération des matrices de confusion...")

n_models = len(results)
cols = 4
rows = (n_models + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
axes = axes.flatten()

for i, (name, res) in enumerate(results.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=res["cm"],
        display_labels=["Normal", "Anomalie"]
    )
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(
        f"{name}\nAcc={res['accuracy']:.2f}  F1={res['f1']:.2f}",
        fontsize=9, fontweight="bold"
    )

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Matrices de Confusion — Semelle Connectée", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ confusion_matrices.png sauvegardé")

# =============================================================================
# SECTION 7 — RÉADAPTATION / OPTIMISATION DES HYPERPARAMÈTRES
# =============================================================================
print("\n[5b/6] Réadaptation des modèles (hyperparamètre tuning ciblé)...")

"""
LOGIQUE DE RÉADAPTATION :
  - Si sensibilité faible (beaucoup de faux négatifs → anomalies manquées) :
      → augmenter class_weight, baisser threshold, augmenter n_estimators
  - Si spécificité faible (faux positifs → trop d'alarmes) :
      → augmenter C (SVM), max_depth, ou regularisation
  - Si F1 < 0.85 : grille d'hyperparamètres plus large
"""

# Exemple de réadaptation automatique basée sur F1
readapted_results = {}

for name, res in results.items():
    f1_score_val = res["f1"]
    sensitivity  = res["sensitivity"]

    if f1_score_val < 0.90:
        print(f"  ↺ Réadaptation de {name} (F1={f1_score_val:.3f})...")

        if name == "Random Forest":
            clf2 = RandomForestClassifier(
                n_estimators=400, max_depth=None,
                min_samples_leaf=1, class_weight={0: 1, 1: 3},
                random_state=42
            )
        elif name == "Gradient Boosting":
            clf2 = GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.03,
                max_depth=5, subsample=0.8, random_state=42
            )
        elif name == "XGBoost":
            clf2 = XGBClassifier(
                n_estimators=300, learning_rate=0.03,
                max_depth=6, scale_pos_weight=N_NORMAL/N_ANOMALY,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42
            )
        elif name == "SVM (RBF)":
            clf2 = SVC(kernel="rbf", C=50, gamma="auto",
                       class_weight={0: 1, 1: 3},
                       probability=True, random_state=42)
        elif name == "MLP":
            clf2 = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu", alpha=0.001,
                max_iter=1000, early_stopping=True, random_state=42
            )
        elif name == "KNN":
            clf2 = KNeighborsClassifier(n_neighbors=5, weights="distance")
        elif name == "Logistic Regression":
            clf2 = LogisticRegression(C=5.0, class_weight={0: 1, 1: 3},
                                      max_iter=1000, random_state=42)
        elif name == "AdaBoost":
            clf2 = AdaBoostClassifier(n_estimators=200, learning_rate=0.3,
                                      random_state=42)
        else:
            clf2 = res["model"]

        cfg = models_config[name]
        clf2.fit(cfg["X_train"], y_train)
        y_pred2 = clf2.predict(cfg["X_test"])
        cm2   = confusion_matrix(y_test, y_pred2)
        acc2  = accuracy_score(y_test, y_pred2)
        f1_2  = f1_score(y_test, y_pred2, average="weighted")
        tn2, fp2, fn2, tp2 = cm2.ravel()
        sens2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
        spec2 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0

        readapted_results[name] = {
            "model": clf2, "y_pred": y_pred2, "cm": cm2,
            "accuracy": acc2, "f1": f1_2,
            "sensitivity": sens2, "specificity": spec2,
            "tn": tn2, "fp": fp2, "fn": fn2, "tp": tp2
        }
        delta = f1_2 - f1_score_val
        sign  = "+" if delta >= 0 else ""
        print(f"     F1 : {f1_score_val:.3f} → {f1_2:.3f} ({sign}{delta:.3f})")
    else:
        readapted_results[name] = res
        print(f"  ✓ {name} déjà bon (F1={f1_score_val:.3f}) — pas de réadaptation")

# =============================================================================
# SECTION 8 — SÉLECTION DU MEILLEUR MODÈLE
# =============================================================================
print("\n[6/6] Sélection du meilleur modèle...")

print("\n  ┌─────────────────────────┬──────────┬──────────┬─────────────┬─────────────┐")
print("  │ Modèle                  │ Accuracy │    F1    │ Sensibilité │ Spécificité │")
print("  ├─────────────────────────┼──────────┼──────────┼─────────────┼─────────────┤")

scores = {}
for name, res in readapted_results.items():
    # Score composite : compromis précision + sensibilité (anomalie = priorité)
    # On pénalise les faux négatifs (anomalies manquées = dangereux)
    composite = 0.3 * res["accuracy"] + 0.4 * res["sensitivity"] + 0.3 * res["f1"]
    scores[name] = composite
    flag = ""
    print(f"  │ {name:<23} │  {res['accuracy']:.3f}   │  {res['f1']:.3f}   │    {res['sensitivity']:.3f}    │    {res['specificity']:.3f}    │{flag}")

print("  └─────────────────────────┴──────────┴──────────┴─────────────┴─────────────┘")

best_model_name = max(scores, key=scores.get)
best_res = readapted_results[best_model_name]

print(f"\n  🏆 MEILLEUR MODÈLE : {best_model_name}")
print(f"     Score composite = {scores[best_model_name]:.4f}")
print(f"     (score = 0.3×accuracy + 0.4×sensibilité + 0.3×F1)")
print(f"\n     Justification :")
print(f"     → Détection d'anomalies médicales/sportives nécessite")
print(f"       une haute sensibilité (anomalies manquées = risque blessure)")
print(f"     → Le F1 pondéré gère le déséquilibre des classes")
print(f"     → L'accuracy confirme la généralisation globale")

print("\n  Rapport de classification complet :")
print(classification_report(y_test, best_res["y_pred"],
                             target_names=["Normal", "Anomalie"]))

# --- Figure finale : comparaison des scores
fig2, ax = plt.subplots(figsize=(10, 5))
names  = list(readapted_results.keys())
accs   = [readapted_results[n]["accuracy"]    for n in names]
f1s    = [readapted_results[n]["f1"]          for n in names]
senss  = [readapted_results[n]["sensitivity"] for n in names]
comps  = [scores[n]                            for n in names]

x = np.arange(len(names))
w = 0.2
ax.bar(x - 1.5*w, accs,  w, label="Accuracy",     color="#4C9BE8")
ax.bar(x - 0.5*w, f1s,   w, label="F1",           color="#50C878")
ax.bar(x + 0.5*w, senss, w, label="Sensibilité",  color="#FF7F50")
ax.bar(x + 1.5*w, comps, w, label="Score composite", color="#9B59B6", alpha=0.85)

ax.axvline(names.index(best_model_name), color="gold", linewidth=2,
           linestyle="--", alpha=0.6, label=f"Meilleur : {best_model_name}")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Comparaison des modèles — Semelle Connectée", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✓ model_comparison.png sauvegardé")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 65)
print("  RÉSUMÉ PIPELINE")
print("=" * 65)
print(f"  Données : {len(df_full)} lignes totales")
print(f"  Train   : {len(X_train)} | Test : {len(X_test)}")
print(f"  Modèles entraînés : {len(models_config)}")
print(f"  Meilleur modèle   : {best_model_name}")
print(f"  Fichiers générés  : confusion_matrices.png, model_comparison.png")
print("=" * 65)
print("""
INTERPRÉTATION PODOLOGIQUE DES VARIABLES :
  yaw  fort  → déviation latérale du pied (supination/pronation)
  roll fort  → rotation excessive (risque entorse)
  flex3 = 0  → pas d'appui talon → pathologie possible (avant-pied exclusif)
  flex1 = 0  → pas d'appui avant-pied → gait atypique
  flex1+flex2+flex3 = 0 → pied en l'air (phase aérienne) — normal en course

EXTENSIONS POSSIBLES :
  • LSTM / TCN → modéliser les séquences temporelles de pas
  • Autoencoder → détection d'anomalie non supervisée
  • Feature engineering → cadence, durée de contact, ratio pied/talon
  • Dashboard temps réel → Streamlit + Bluetooth BLE
""")