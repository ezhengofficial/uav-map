import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MERGED_LAS_FILE = DATA_DIR / "final/merged.las"