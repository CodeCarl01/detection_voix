import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import FIGURES_DIR
from traitement_signal import (
    SR, FRAME_LENGTH, HOP_LENGTH, N_MELS, N_MFCC,
    normaliser, supprimer_silences, pre_accentuer,
    pretraiter, extraire_mfcc
)


# ---------------------------------------------------------------------------
# Utilitaire : charger ou générer un signal de démonstration
# ---------------------------------------------------------------------------

def charger_signal(chemin: str = None):
    if chemin and chemin != 'demo':
        signal, sr = librosa.load(chemin, sr=SR)
        titre = chemin.split('/')[-1]
    else:
        # Signal de démo : sinusoïde modulée simulant un mot
        t = np.linspace(0, 1.0, SR)
        signal = (0.5 * np.sin(2 * np.pi * 200 * t) +
                  0.3 * np.sin(2 * np.pi * 800 * t) +
                  0.1 * np.random.randn(len(t)))
        signal = signal.astype(np.float32)
        titre = "signal_demo"
    return signal, SR, titre


# ---------------------------------------------------------------------------
# Figure 1 — Forme d'onde avant / après prétraitement
# ---------------------------------------------------------------------------

def fig_formes_donde(signal_brut: np.ndarray, sr: int, titre: str):
    signal_norm  = normaliser(signal_brut)
    signal_trim  = supprimer_silences(signal_norm)
    signal_final = pre_accentuer(signal_trim)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
    fig.suptitle(f'Étape 1 — Prétraitement du signal  [{titre}]',
                 fontsize=13, fontweight='bold')

    t_brut  = np.linspace(0, len(signal_brut)  / sr, len(signal_brut))
    t_trim  = np.linspace(0, len(signal_trim)  / sr, len(signal_trim))
    t_final = np.linspace(0, len(signal_final) / sr, len(signal_final))

    axes[0].plot(t_brut,  signal_brut,  color='steelblue', linewidth=0.6)
    axes[0].set_title('Signal brut (normalisé)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(t_trim,  signal_trim,  color='darkorange', linewidth=0.6)
    axes[1].set_title(f'Après suppression des silences (top_db=20)')
    axes[1].set_ylabel('Amplitude')

    axes[2].plot(t_final, signal_final, color='seagreen', linewidth=0.6)
    axes[2].set_title('Après pré-accentuation  [H(z) = 1 − 0.97·z⁻¹]')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Temps (s)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_formes_donde.png', dpi=150)
    print('  ✓ fig1_formes_donde.png sauvegardée')
    plt.show()


# ---------------------------------------------------------------------------
# Figure 2 — Spectrogramme (énergie en fonction du temps et de la fréquence)
# ---------------------------------------------------------------------------

def fig_spectrogramme(signal: np.ndarray, sr: int, titre: str):
    signal_p = pretraiter(signal, sr)

    D = librosa.amplitude_to_db(
            np.abs(librosa.stft(signal_p,
                                n_fft=FRAME_LENGTH,
                                hop_length=HOP_LENGTH,
                                win_length=FRAME_LENGTH,
                                window='hamming')),
            ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f'Étape 2-3 — Spectrogramme (FFT + fenêtre de Hamming)  [{titre}]',
                 fontsize=13, fontweight='bold')

    img = librosa.display.specshow(D,
                                   sr=sr,
                                   hop_length=HOP_LENGTH,
                                   x_axis='time',
                                   y_axis='hz',
                                   ax=ax,
                                   cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Trames de 25 ms — chevauchement de 10 ms')
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Fréquence (Hz)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_spectrogramme.png', dpi=150)
    print('  ✓ fig2_spectrogramme.png sauvegardée')
    plt.show()


# ---------------------------------------------------------------------------
# Figure 3 — Banc de filtres de Mel
# ---------------------------------------------------------------------------

def fig_banc_mel(sr: int):
    mel_filters = librosa.filters.mel(sr=sr,
                                      n_fft=FRAME_LENGTH,
                                      n_mels=N_MELS)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle('Étape 3 — Banc de filtres de Mel (26 filtres triangulaires)',
                 fontsize=13, fontweight='bold')

    for i, filt in enumerate(mel_filters):
        ax.plot(freqs, filt, alpha=0.7)

    ax.set_xlabel('Fréquence (Hz)')
    ax.set_ylabel('Amplitude du filtre')
    ax.set_title('Répartition non-linéaire selon l\'échelle de Mel\n'
                 'Mel = 2595 × log₁₀(1 + f/700)')
    ax.set_xlim(0, sr / 2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_banc_mel.png', dpi=150)
    print('  ✓ fig3_banc_mel.png sauvegardée')
    plt.show()


# ---------------------------------------------------------------------------
# Figure 4 — Heatmap des MFCC
# ---------------------------------------------------------------------------

def fig_mfcc_heatmap(signal: np.ndarray, sr: int, titre: str):
    signal_p = pretraiter(signal, sr)

    mfcc = librosa.feature.mfcc(y=signal_p, sr=sr,
                                 n_mfcc=N_MFCC,
                                 n_mels=N_MELS,
                                 hop_length=HOP_LENGTH,
                                 win_length=FRAME_LENGTH)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f'Étape 4 — Heatmap des 13 coefficients MFCC  [{titre}]',
                 fontsize=13, fontweight='bold')

    img = librosa.display.specshow(mfcc,
                                   sr=sr,
                                   hop_length=HOP_LENGTH,
                                   x_axis='time',
                                   ax=ax,
                                   cmap='coolwarm')
    fig.colorbar(img, ax=ax)
    ax.set_ylabel('Coefficient MFCC')
    ax.set_xlabel('Temps (s)')
    ax.set_title('Chaque ligne = un coefficient MFCC | Chaque colonne = une trame de 25 ms')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_mfcc_heatmap.png', dpi=150)
    print('  ✓ fig4_mfcc_heatmap.png sauvegardée')
    plt.show()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    chemin = sys.argv[1] if len(sys.argv) > 1 else 'demo'
    signal, sr, titre = charger_signal(chemin)

    print(f'\n  Signal chargé : {titre}')
    print(f'  Durée : {len(signal)/sr:.2f}s | SR : {sr} Hz\n')

    print('  Génération des figures...')
    fig_formes_donde(signal, sr, titre)
    fig_spectrogramme(signal, sr, titre)
    fig_banc_mel(sr)
    fig_mfcc_heatmap(signal, sr, titre)

    print('\n  Toutes les figures ont été sauvegardées dans le dossier figures/.')


if __name__ == '__main__':
    main()
