"""Chemins du projet, calculés par rapport à la racine du dépôt."""

from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
