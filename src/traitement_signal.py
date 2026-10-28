import numpy as np
import librosa
import librosa.filters
import matplotlib.pyplot as plt
import os

SR          = 16000          # Fréquence d'échantillonnage (Hz)
FRAME_MS    = 0.025          # Durée d'une trame : 25 ms
HOP_MS      = 0.010          # Décalage entre trames : 10 ms
N_MELS      = 26             # Nombre de filtres Mel
N_MFCC      = 13             # Nombre de coefficients MFCC conservés
PREEMPH     = 0.97           # Coefficient de pré-accentuation
TOP_DB      = 20             # Seuil de silence pour le trim (en dB)

FRAME_LENGTH = int(FRAME_MS * SR)   # 400 échantillons
HOP_LENGTH   = int(HOP_MS  * SR)    # 160 échantillons


# ÉTAPE 1 — Prétraitement du signal

def normaliser(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal 
    return signal / max_val

def supprimer_silences(signal: np.ndarray, top_db: int = TOP_DB) -> np.ndarray:
    signal_trim, _ = librosa.effects.trim(signal, top_db=top_db)
    return signal_trim


def pre_accentuer(signal: np.ndarray, coeff: float = PREEMPH) -> np.ndarray:
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def pretraiter(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    signal = normaliser(signal)
    signal = supprimer_silences(signal)
    signal = pre_accentuer(signal)
    return signal


# ÉTAPE 2 — Fenêtrage

def decouper_en_trames(signal: np.ndarray,frame_length: int = FRAME_LENGTH,hop_length: int   = HOP_LENGTH) -> np.ndarray:
    
    window = np.hamming(frame_length)
    trames = librosa.util.frame(signal,frame_length=frame_length, hop_length=hop_length).T  
    trames = trames * window  # application de la fenêtre de Hamming
    return trames

# ÉTAPE 3 — FFT et banc de filtres de Mel

def calculer_spectre_mel(signal: np.ndarray,sr: int = SR) -> np.ndarray:
    stft = np.abs(librosa.stft(signal,n_fft=FRAME_LENGTH,hop_length=HOP_LENGTH,win_length=FRAME_LENGTH,window='hamming')) ** 2
    mel_filters = librosa.filters.mel(sr=sr,n_fft=FRAME_LENGTH,n_mels=N_MELS)
    mel_spectrum = mel_filters @ stft  
    return mel_spectrum



# ÉTAPE 4 — Extraction des MFCC

def extraire_mfcc(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y          = signal,
        sr         = sr,
        n_mfcc     = N_MFCC,
        n_mels     = N_MELS,
        hop_length = HOP_LENGTH,
        win_length = FRAME_LENGTH
    )
    moyenne    = np.mean(mfcc, axis=1)  
    ecart_type = np.std(mfcc,  axis=1)   
    features   = np.concatenate([moyenne, ecart_type])  

    return features


def traiter_fichier(chemin_wav: str) -> np.ndarray:
    signal, sr = librosa.load(chemin_wav, sr=SR)
    signal = pretraiter(signal, sr)
    return extraire_mfcc(signal, sr)


def traiter_signal_brut(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    signal = pretraiter(signal, sr)
    return extraire_mfcc(signal, sr)
