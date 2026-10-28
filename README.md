# Détection de Mots-Clés par Traitement du Signal Audio

**Polytech Dijon — Traitement du Signal — 2025/2026**

Reconnaissance en temps réel des mots-clés `oui`, `non`, `stop` à partir du microphone, avec une chaîne complète de traitement du signal et un classifieur SVM.

```
Micro → Prétraitement → Fenêtrage → FFT + Mel → MFCC → SVM → Mot détecté
```

---

## Structure du projet

```
detection_voix/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                          ← code source
│   ├── config.py                 ← chemins du projet (data/, models/, figures/)
│   ├── traitement_signal.py      ← étapes 1 à 4 (prétraitement, fenêtrage, FFT, MFCC)
│   ├── enregistrer_dataset.py    ← collecte du dataset au micro → data/features-2.npz
│   ├── visualiser.py             ← figures 1 à 4 (étapes 1 à 4)
│   ├── entrainement.py           ← split + normalisation + SVM + GridSearch
│   ├── analyse_erreurs.py        ← PCA, signatures MFCC, analyse des confusions
│   └── detection_temps_reel.py   ← démo live au micro
│
├── data/
│   ├── features.npz              ← dataset MFCC (120 exemples)
│   └── features-2.npz            ← second enregistrement (120 exemples)
│
├── models/
│   ├── model.pkl                 ← SVM entraîné
│   └── scaler.pkl                ← StandardScaler
│
├── figures/                      ← fig1 à fig8 (pour le PPT)
│
└── notebooks/
    └── demo_projet.ipynb         ← notebook de démonstration (pipeline complète)
```

Répartition du travail : **Daniella KAMDEM-NANA** = traitement du signal (`traitement_signal.py`, `enregistrer_dataset.py`, `visualiser.py`) ;
**Carl MENSAH** = classifieur et démo (`entrainement.py`, `analyse_erreurs.py`, `detection_temps_reel.py`).

---

## Installation rapide

```bash
# 1. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 2. PortAudio (requis pour sounddevice — uniquement si enregistrement ou démo live)
sudo apt install portaudio19-dev     # Linux
brew install portaudio               # macOS

# 3. Dépendances Python
pip install -r requirements.txt
```

---

## Utilisation

### Option 1 — Parcours guidé via le notebook (recommandé)

Le notebook `demo_projet.ipynb` couvre la **pipeline complète** de A à Z : traitement du signal, entraînement, évaluation, analyse des erreurs.

```bash
jupyter notebook notebooks/demo_projet.ipynb
```

Il utilise les datasets `data/features.npz` et `data/features-2.npz` déjà fournis et s'exécute sans microphone.

### Option 2 — Utilisation en ligne de commande

```bash
# 1. Enregistrer un dataset au micro (40 × oui / non / stop)
python src/enregistrer_dataset.py
# → produit data/features-2.npz (écrase le fichier existant)

# 2. Entraîner le SVM (GridSearch, évaluation, matrice de confusion)
python src/entrainement.py
# → produit models/model.pkl + models/scaler.pkl + figures 5 et 6

# 3. Démo temps réel
python src/detection_temps_reel.py             # boucle continue
python src/detection_temps_reel.py --unique    # une seule détection

# 4. Générer les figures signal pour le PPT (optionnel)
python src/visualiser.py mon_fichier.wav       # avec un .wav
python src/visualiser.py                        # avec un signal synthétique
```

---

## La chaîne de traitement en détail

| Étape | Concept | Paramètres |
|---|---|---|
| **1. Prétraitement** | Normalisation, trim des silences, pré-accentuation (filtre RIF $H(z) = 1 - 0{,}97\,z^{-1}$) | `top_db = 20 dB` |
| **2. Fenêtrage** | Fenêtre de Hamming, trames de 25 ms avec chevauchement de 10 ms | `frame = 400`, `hop = 160` |
| **3. FFT + Mel** | STFT puis application d'un banc de 26 filtres triangulaires sur l'échelle de Mel | `n_mels = 26` |
| **4. MFCC** | log + DCT → 13 coefficients par trame, puis agrégation (moyenne + écart-type) → vecteur de 26 valeurs | `n_mfcc = 13` |
| **5. SVM** | Noyau RBF, hyperparamètres trouvés par GridSearchCV sur 5 folds | `C ∈ {0.1, 1, 10, 100}`, `γ ∈ {auto, scale, 0.001, 0.01, 0.1}` |

---

## Interface entre les deux rôles

Les fichiers `.npz` du dossier `data/` sont le **point d'interface unique** :

```python
# (production)
np.savez('features.npz', X=X, y=y)
# X : (n_exemples, 26) — vecteurs MFCC
# y : (n_exemples,)    — labels

# (consommation)
data = np.load('features.npz', allow_pickle=True)
X, y = data['X'], data['y']
```

Tant que ce contrat est respecté, les deux rôles peuvent travailler en parallèle dès la semaine 2.

---

## Résultats actuels

Avec les datasets fournis (2 × 40 × 3 mots = 240 exemples) :

| Métrique | Valeur |
|---|---|
| Précision (test, 20% hold-out) | ~87-93% |
| Score CV (GridSearch, 5 folds) | ~93% |
| Temps de prédiction | < 10 ms |
| Latence totale démo live | ~1.5 s (dominée par la durée d'enregistrement) |

---

## Pistes d'amélioration

- **Plus de données** : viser 80-100 exemples par mot (la courbe d'apprentissage est encore montante)
- **Plus de locuteurs** : diversifier les voix pour rendre le modèle robuste
- **Deltas MFCC** : ajouter la dérivée temporelle pour capturer la dynamique
- **VAD (Voice Activity Detection)** : déclencher l'enregistrement au lieu de traiter systématiquement 1.5 s
- **CNN léger sur le mel-spectrogramme** : alternative moderne qui atteint ≥ 95 % avec le même volume de données

---

*Projet réalisé dans le cadre du cours de Traitement du Signal — Polytech Dijon 2025/2026*
