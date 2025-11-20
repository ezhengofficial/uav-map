import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


def setup_data_dir():
    DATA_DIR.makedir(exist_ok=True)
