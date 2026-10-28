"""
=============================================================================
ENTRAÎNEMENT DU CLASSIFIEUR SVM — Personne B
=============================================================================

Ce module charge le fichier `features.npz` produit par la Personne A,
entraîne un SVM (avec recherche d'hyperparamètres), évalue ses performances,
et sauvegarde le modèle final pour la démo temps réel.

Pipeline :
    features.npz  →  StandardScaler  →  SVM (RBF)  →  model.pkl + scaler.pkl

Usage :
    python src/entrainement.py
=============================================================================
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, GridSearchCV, learning_curve, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from config import DATA_DIR, MODELS_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
DATA_FILE    = DATA_DIR / "features.npz"
MODEL_FILE   = MODELS_DIR / "model.pkl"
SCALER_FILE  = MODELS_DIR / "scaler.pkl"
RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def charger_donnees(fichier: str = DATA_FILE):
    """
    Charge le fichier features.npz et retourne (X, y).
    X : vecteurs MFCC de shape (n_exemples, 26)
    y : labels de shape (n_exemples,)
    """
    data = np.load(fichier, allow_pickle=True)
    X, y = data['X'], data['y']
    print(f"  Données chargées : {X.shape[0]} exemples × {X.shape[1]} features")
    print(f"  Classes : {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# ---------------------------------------------------------------------------
# Split + normalisation
# ---------------------------------------------------------------------------

def preparer_donnees(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Découpe en train/test (stratifié) puis normalise via StandardScaler.
    Retourne X_train, X_test, y_train, y_test, scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # garantit une répartition équilibrée des classes
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# Recherche des meilleurs hyperparamètres (GridSearch)
# ---------------------------------------------------------------------------

def chercher_meilleurs_parametres(X_train, y_train, cv=5, verbose=1):
    """
    GridSearchCV sur un SVM à noyau RBF.
    On balaie C (régularisation) et gamma (largeur du noyau).
    """
    param_grid = {
        'C':     [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }

    grid = GridSearchCV(
        estimator  = SVC(kernel='rbf', random_state=RANDOM_STATE),
        param_grid = param_grid,
        cv         = cv,
        scoring    = 'accuracy',
        n_jobs     = -1,
        verbose    = verbose
    )
    grid.fit(X_train, y_train)

    print(f"\n  Meilleurs paramètres : {grid.best_params_}")
    print(f"  Meilleure accuracy (CV) : {grid.best_score_ * 100:.2f}%")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluer_modele(modele, X_test, y_test, classes=None):
    """
    Calcule accuracy, matrice de confusion, et rapport de classification.
    """
    y_pred = modele.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Précision sur le jeu de test : {acc * 100:.2f}%\n")
    print("  Rapport de classification :")
    print(classification_report(y_test, y_pred))

    if classes is None:
        classes = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("  Matrice de confusion :")
    print(cm)

    return y_pred, acc, cm


# ---------------------------------------------------------------------------
# Figures d'évaluation
# ---------------------------------------------------------------------------

def tracer_matrice_confusion(cm, classes, chemin=FIGURES_DIR / 'fig5_matrice_confusion.png'):
    """Affiche une matrice de confusion lisible avec annotations."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vraie classe')
    ax.set_title('Matrice de confusion — SVM (noyau RBF)',
                 fontsize=12, fontweight='bold')

    # Annotation des valeurs
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', color=color, fontsize=14)

    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    print(f"  ✓ {chemin} sauvegardée")
    plt.close()


def tracer_courbe_apprentissage(modele, X, y, chemin=FIGURES_DIR / 'fig6_courbe_apprentissage.png'):
    """
    Courbe d'apprentissage : score (train/test) en fonction du nombre d'exemples.
    Permet de diagnostiquer sous-apprentissage / sur-apprentissage.
    """
    # IMPORTANT : learning_curve prend les N premiers exemples de chaque fold CV
    # sans re-stratifier. On mélange pour garantir la présence de toutes les classes
    # même dans les petites sous-tailles.
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(X))
    X_shuffle, y_shuffle = X[idx], y[idx]

    # Minimum 0.25 pour garantir suffisamment d'exemples de chaque classe
    tailles = np.linspace(0.25, 1.0, 7)
    train_sizes, train_scores, test_scores = learning_curve(
        modele, X_shuffle, y_shuffle,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        train_sizes=tailles,
        scoring='accuracy',
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    test_mean  = test_scores.mean(axis=1)
    test_std   = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label="Score d'entraînement")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='steelblue')
    ax.plot(train_sizes, test_mean, 'o-', color='darkorange', label='Score de validation (CV)')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.15, color='darkorange')

    ax.set_xlabel("Nombre d'exemples d'entraînement")
    ax.set_ylabel('Accuracy')
    ax.set_title("Courbe d'apprentissage du SVM",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    print(f"  ✓ {chemin} sauvegardée")
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def entrainer_pipeline_complet(fichier_donnees=DATA_FILE, sauvegarder=True,avec_gridsearch=True,tracer_figures=True):
    """
    Pipeline complet : charge → split → normalise → entraîne → évalue → sauvegarde.
    """
    print("\n" + "=" * 60)
    print("  ENTRAÎNEMENT DU CLASSIFIEUR — Personne B")
    print("=" * 60)

    # 1. Chargement
    print("\n[1/5] Chargement des données")
    X1, y1 = charger_donnees(fichier_donnees)
    X2,y2 = charger_donnees(DATA_DIR / "features-2.npz")
    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2.T))


    # 2. Split + normalisation
    print("\n[2/5] Split train/test et normalisation")
    X_train, X_test, y_train, y_test, scaler = preparer_donnees(X, y)
    print(f"  Train : {X_train.shape[0]} exemples | Test : {X_test.shape[0]} exemples")

    # 3. Recherche d'hyperparamètres
    print("\n[3/5] Recherche des meilleurs hyperparamètres (GridSearchCV)")
    if avec_gridsearch:
        modele, meilleurs, score_cv = chercher_meilleurs_parametres(X_train, y_train, verbose=0)
    else:
        modele = SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE)
        modele.fit(X_train, y_train)
        meilleurs = {'C': 10, 'gamma': 'scale'}
        score_cv  = None

    # 4. Évaluation
    print("\n[4/5] Évaluation sur le jeu de test")
    classes = sorted(np.unique(y))
    y_pred, acc, cm = evaluer_modele(modele, X_test, y_test, classes)

    if tracer_figures:
        tracer_matrice_confusion(cm, classes)
        # Pour la courbe d'apprentissage, on utilise TOUTES les données normalisées
        X_all = scaler.transform(X)
        tracer_courbe_apprentissage(modele, X_all, y)

    # 5. Sauvegarde
    print("\n[5/5] Sauvegarde du modèle")
    if sauvegarder:
        joblib.dump(modele, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print(f"  ✓ Modèle sauvegardé  : {MODEL_FILE}")
        print(f"  ✓ Scaler sauvegardé  : {SCALER_FILE}")

    print("\n" + "=" * 60)
    print(f"  TERMINÉ — Accuracy finale : {acc * 100:.2f}%")
    print("=" * 60)

    return {
        'modele'      : modele,
        'scaler'      : scaler,
        'accuracy'    : acc,
        'confusion'   : cm,
        'classes'     : classes,
        'meilleurs'   : meilleurs,
        'score_cv'    : score_cv,
        'X_test'      : X_test,
        'y_test'      : y_test,
        'y_pred'      : y_pred,
    }


if __name__ == '__main__':
    entrainer_pipeline_complet()
