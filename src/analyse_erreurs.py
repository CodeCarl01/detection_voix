"""

=============================================================================
ANALYSE DES ERREURS ET SÉPARABILITÉ DES CLASSES — Personne B
=============================================================================

Ce module fournit des outils d'analyse post-entraînement :
    - visualisation des vecteurs MFCC par classe (moyennes, écarts-types)
    - projection en 2D via PCA pour visualiser la séparabilité
    - analyse détaillée des erreurs de classification

Usage depuis un script / notebook :
    from analyse_erreurs import analyser_separabilite, analyser_erreurs
=============================================================================


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import FIGURES_DIR


# ---------------------------------------------------------------------------
# 1. Moyennes MFCC par classe
# ---------------------------------------------------------------------------

def tracer_mfcc_moyens_par_classe(X, y, chemin=FIGURES_DIR / 'fig7_mfcc_moyens.png'):
    """
    
    Affiche la moyenne (± écart-type) de chaque feature MFCC par classe.
    X est de shape (n, 26) : 13 moyennes MFCC + 13 écarts-types.
    
    """
    classes = sorted(np.unique(y))
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Signature MFCC moyenne par classe', fontsize=13, fontweight='bold')

    # À gauche : les 13 coefficients "moyenne"
    # À droite : les 13 coefficients "écart-type"
    for idx_classe, classe in enumerate(classes):
        X_classe = X[y == classe]
        moy = X_classe.mean(axis=0)
        std = X_classe.std(axis=0)
        couleur = couleurs[idx_classe % len(couleurs)]

        axes[0].errorbar(range(13), moy[:13], yerr=std[:13],
                         label=classe, color=couleur, marker='o',
                         capsize=3, linewidth=1.5)
        axes[1].errorbar(range(13), moy[13:], yerr=std[13:],
                         label=classe, color=couleur, marker='o',
                         capsize=3, linewidth=1.5)

    axes[0].set_title('13 coefficients « moyenne »')
    axes[0].set_xlabel('Index MFCC')
    axes[0].set_ylabel('Valeur')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title('13 coefficients « écart-type »')
    axes[1].set_xlabel('Index MFCC')
    axes[1].set_ylabel('Valeur')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    print(f"  ✓ {chemin} sauvegardée")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Projection PCA — visualiser la séparabilité dans l'espace 2D
# ---------------------------------------------------------------------------

def projection_pca_2d(X, y, scaler=None,
                      chemin=FIGURES_DIR / 'fig8_pca_2d.png'):
    """
    Projette les vecteurs MFCC normalisés en 2D via PCA et affiche
    la répartition des classes.

    Si un `scaler` est fourni, on l'applique avant la PCA.
    """
    X_norm = scaler.transform(X) if scaler is not None else X

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_norm)

    classes = sorted(np.unique(y))
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    marqueurs = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, classe in enumerate(classes):
        masque = (y == classe)
        ax.scatter(X_2d[masque, 0], X_2d[masque, 1],
                   c=couleurs[idx % len(couleurs)],
                   marker=marqueurs[idx % len(marqueurs)],
                   label=classe, s=60, alpha=0.7, edgecolors='white')

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}% de variance)')
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}% de variance)')
    ax.set_title('Projection PCA 2D des vecteurs MFCC\n'
                 f'Variance expliquée cumulée : {var_exp.sum()*100:.1f}%',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Mot')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    print(f"  ✓ {chemin} sauvegardée")
    plt.close()

    return var_exp


# ---------------------------------------------------------------------------
# 3. Analyse détaillée des erreurs
# ---------------------------------------------------------------------------

def analyser_erreurs(y_true, y_pred):
    """
    Identifie les erreurs de prédiction et retourne un dictionnaire :
        {
            ('vrai_mot', 'predit_mot') : nombre d'occurrences,
            ...
        }
    """
    erreurs = {}
    for vrai, pred in zip(y_true, y_pred):
        if vrai != pred:
            cle = (str(vrai), str(pred))
            erreurs[cle] = erreurs.get(cle, 0) + 1

    if not erreurs:
        print("  ✓ Aucune erreur détectée — le modèle est parfait sur ce jeu de test !")
        return erreurs

    print(f"\n  {sum(erreurs.values())} erreur(s) détectée(s) :")
    for (vrai, pred), nb in sorted(erreurs.items(), key=lambda x: -x[1]):
        print(f"    • « {vrai} » confondu avec « {pred} » : {nb} fois")

    return erreurs


# ---------------------------------------------------------------------------
# 4. Rapport complet (point d'entrée pratique)
# ---------------------------------------------------------------------------

def rapport_complet(resultats, X, y):
    """
    Produit l'ensemble des figures + l'analyse d'erreurs.
    `resultats` est le dictionnaire renvoyé par entrainer_pipeline_complet.
    """
    print("\n  ANALYSE DES ERREURS ET DE LA SÉPARABILITÉ")
    print("  " + "-" * 50)

    tracer_mfcc_moyens_par_classe(X, y)
    var_exp = projection_pca_2d(X, y, scaler=resultats['scaler'])
    erreurs = analyser_erreurs(resultats['y_test'], resultats['y_pred'])

    return {
        'variance_expliquee': var_exp,
        'erreurs'           : erreurs,
    }