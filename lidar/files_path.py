import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "logs"
MERGED_LAS_FILE = DATA_DIR / "final/merged.las"

TRACKER_FILE = DATA_DIR / "final/merge_tracker.json"

SETTINGS_JSON_FILE = Path.home() / "Documents" / "AirSim" / "settings.json",