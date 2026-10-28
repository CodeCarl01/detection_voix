"""
=============================================================================
DÉTECTION EN TEMPS RÉEL — Personne B
=============================================================================

Script de démonstration live : enregistre 1.5s de voix au micro,
extrait les MFCC, normalise, et prédit le mot-clé via le SVM entraîné.

Deux modes de fonctionnement :
    1. MODE UNIQUE  : une seule détection
    2. MODE BOUCLE  : détection continue jusqu'à Ctrl+C

Usage :
    python src/detection_temps_reel.py           # mode boucle (par défaut)
    python src/detection_temps_reel.py --unique  # mode unique
=============================================================================
"""

import sys
import argparse
import numpy as np
import joblib
import sounddevice as sd

from traitement_signal import traiter_signal_brut, SR
from config import MODELS_DIR


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
MODEL_FILE   = MODELS_DIR / "model.pkl"
SCALER_FILE  = MODELS_DIR / "scaler.pkl"
DUREE_SEC    = 1.5        # durée d'enregistrement
SEUIL_CONF   = 0.60       # seuil de confiance minimal pour valider la prédiction


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

def charger_modele(model_file=MODEL_FILE, scaler_file=SCALER_FILE):
    """Charge le modèle SVM et le scaler depuis le disque."""
    try:
        modele = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        print(f"  ✓ Modèle chargé : {model_file}")
        print(f"  ✓ Scaler chargé : {scaler_file}")
        print(f"  ✓ Classes connues : {list(modele.classes_)}")
        return modele, scaler
    except FileNotFoundError as e:
        print(f"\n  ✗ Fichier introuvable : {e.filename}")
        print("    → Lancez d'abord `python entrainement.py` pour créer le modèle.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Enregistrement + prédiction
# ---------------------------------------------------------------------------

def enregistrer(duree=DUREE_SEC, sr=SR):
    """Enregistre un signal audio depuis le micro, retourne un array 1D."""
    audio = sd.rec(int(duree * sr),
                   samplerate=sr,
                   channels=1,
                   dtype='float32')
    sd.wait()
    return audio.flatten()


def predire(signal, modele, scaler, sr=SR):
    """
    Pipeline : signal brut → MFCC → normalisation → prédiction.
    Retourne (mot_predit, confiance).

    Si le modèle supporte predict_proba, on renvoie la probabilité associée ;
    sinon, on renvoie la distance à la frontière via decision_function.
    """
    features = traiter_signal_brut(signal, sr).reshape(1, -1)
    features_scaled = scaler.transform(features)

    mot_predit = modele.predict(features_scaled)[0]

    # Confiance :
    #   - SVC(probability=True) → predict_proba disponible
    #   - sinon → on utilise decision_function (distance à la marge)
    if hasattr(modele, 'predict_proba') and getattr(modele, 'probability', False):
        probas = modele.predict_proba(features_scaled)[0]
        confiance = float(np.max(probas))
    else:
        scores = modele.decision_function(features_scaled)[0]
        # Pour du multi-classe 1-vs-1/1-vs-rest : on normalise via softmax
        exp_scores = np.exp(scores - np.max(scores))
        confiance  = float(np.max(exp_scores / exp_scores.sum()))

    return mot_predit, confiance


# ---------------------------------------------------------------------------
# Boucle de démonstration
# ---------------------------------------------------------------------------

def une_detection(modele, scaler, duree=DUREE_SEC):
    """Réalise une détection unique, affiche joliment le résultat."""
    print("\n  🎤 Parlez maintenant (1 mot)...")
    signal = enregistrer(duree)
    mot, conf = predire(signal, modele, scaler)

    symbole = "✓" if conf >= SEUIL_CONF else "?"
    print(f"  {symbole}  Mot détecté : « {mot.upper()} »  (confiance : {conf*100:.1f}%)")

    if conf < SEUIL_CONF:
        print("     ⚠ Confiance faible — probablement un bruit ou un mot inconnu.")

    return mot, conf


def boucle_detection(modele, scaler):
    """Boucle infinie jusqu'à Ctrl+C."""
    print("\n  Mode détection continue — Ctrl+C pour arrêter\n")
    try:
        while True:
            input("  [Entrée] pour lancer une détection...")
            une_detection(modele, scaler)
    except KeyboardInterrupt:
        print("\n\n  Détection arrêtée. À bientôt !\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Détection de mots-clés en temps réel")
    parser.add_argument('--unique', action='store_true',
                        help='Ne faire qu\'une seule détection et quitter')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DÉTECTION EN TEMPS RÉEL")
    print("=" * 60)

    modele, scaler = charger_modele()

    # Vérification du micro
    try:
        default_in = sd.default.device[0]
        devices = sd.query_devices()
        print(f"  ✓ Microphone : {devices[default_in]['name']}")
    except Exception as e:
        print(f"  ✗ Micro indisponible : {e}")
        return

    if args.unique:
        une_detection(modele, scaler)
    else:
        boucle_detection(modele, scaler)


if __name__ == '__main__':
    main()
