"""
=============================================================================
ENREGISTREMENT DU DATASET — Personne A
=============================================================================

Ce script guide l'enregistrement des mots-clés via le microphone,
extrait les vecteurs MFCC de chaque enregistrement, et exporte
le fichier `features.npz` pour la Personne B.

Usage :
    python src/enregistrer_dataset.py

Le fichier features.npz contiendra :
    X : tableau (n_échantillons, 26) — vecteurs MFCC
    y : tableau (n_échantillons,)    — labels ('oui', 'non', 'stop')
=============================================================================
"""

import numpy as np
import sounddevice as sd
import librosa
import os
from traitement_signal import traiter_signal_brut, SR
from config import DATA_DIR

# ---------------------------------------------------------------------------
# Paramètres d'enregistrement
# ---------------------------------------------------------------------------
DUREE_SEC   = 1.5    # Durée d'un enregistrement (secondes)
N_FOIS      = 40     # Nombre de répétitions par mot
MOTS_CLES   = ['oui', 'non', 'stop']
OUTPUT_FILE = DATA_DIR / 'features-2.npz'


def enregistrer_un_mot(nom_mot: str,
                        n_fois: int = N_FOIS,
                        sr: int = SR,
                        duree: float = DUREE_SEC) -> np.ndarray:
    """
    Guide l'utilisateur pour enregistrer `n_fois` répétitions de `nom_mot`.
    Extrait le vecteur MFCC de chaque enregistrement.

    Retourne un tableau de shape (n_fois, 26).
    """
    print(f"\n{'='*60}")
    print(f"  MOT À ENREGISTRER : « {nom_mot.upper()} »")
    print(f"  {n_fois} enregistrements de {duree}s chacun")
    print(f"{'='*60}")
    input("  → Appuyez sur Entrée quand vous êtes prêt(e)...")

    features = []

    for i in range(n_fois):
        input(f"\n  [{i+1:02d}/{n_fois}] Appuyez sur Entrée puis dites « {nom_mot} »")
        print("  ● Enregistrement en cours...")

        audio = sd.rec(int(duree * sr),
                       samplerate=sr,
                       channels=1,
                       dtype='float32')
        sd.wait()
        signal = audio.flatten()

        try:
            feat = traiter_signal_brut(signal, sr)
            features.append(feat)
            print("  ✓ Enregistré")
        except Exception as e:
            print(f"  ✗ Erreur sur cet enregistrement ({e}), on recommence.")
            n_fois += 1  # compensate for this failed attempt

    return np.array(features)


def main():
    print("\n" + "="*60)
    print("  COLLECTE DU DATASET — Détection de mots-clés")
    print("  Mots : " + ", ".join(f'« {m} »' for m in MOTS_CLES))
    print("="*60)

    # Vérification du microphone
    print("\n  Vérification du microphone...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        print(f"  ✓ Microphone détecté : {devices[default_in]['name']}")
    except Exception as e:
        print(f"  ✗ Problème avec le microphone : {e}")
        print("    → Vérifiez votre installation de sounddevice/PortAudio.")
        return

    # Enregistrement de chaque mot
    X_list, y_list = [], []

    for mot in MOTS_CLES:
        features = enregistrer_un_mot(mot, n_fois=N_FOIS)
        X_list.append(features)
        y_list.extend([mot] * len(features))
        print(f"\n  ✓ « {mot} » : {len(features)} enregistrements OK")

    # Assemblage et export
    X = np.vstack(X_list)
    y = np.array(y_list)

    np.savez(OUTPUT_FILE, X=X, y=y)
    print(f"\n{'='*60}")
    print(f"  Dataset sauvegardé dans : {OUTPUT_FILE}")
    print(f"  Shape de X : {X.shape}  →  {X.shape[0]} exemples × 26 features")
    print(f"  Labels y   : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"{'='*60}")
    print("\n  → Transmettez features.npz à la Personne B !")


if __name__ == '__main__':
    main()
