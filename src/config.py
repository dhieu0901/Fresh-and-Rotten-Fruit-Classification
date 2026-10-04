"""Paths, label names and constants used by all scripts.

Labels: fruit_label 0-7 (order of FRUITS), freshness_label 0 = fresh / 1 = spoiled,
combined_label = fruit_label * 2 + freshness_label (0-15).
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_DIR = RAW_DIR / "FRUIT-16K"
METADATA_CSV = DATA_DIR / "metadata.csv"
SPLITS_DIR = DATA_DIR / "splits"
CACHE_DIR = DATA_DIR / "cache"

CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

DATASET_PAGE = "https://data.mendeley.com/datasets/6ps7gtp2wg/1"
DATASET_ZIP_URL = "https://data.mendeley.com/public-api/zip/6ps7gtp2wg/download/1"
DATASET_DOI = "10.17632/6ps7gtp2wg.1"
DATASET_LICENSE = "CC BY 4.0"

FRUITS = ["Banana", "Lemon", "Lulo", "Mango", "Orange", "Strawberry", "Tamarillo", "Tomato"]
FRESHNESS = ["fresh", "spoiled"]
FOLDER_PREFIX = {"F": "fresh", "S": "spoiled"}
NUM_FRUITS = len(FRUITS)
NUM_COMBINED = NUM_FRUITS * len(FRESHNESS)
COMBINED_NAMES = [f"{fruit}_{state}" for fruit in FRUITS for state in FRESHNESS]

IMG_SIZE = 224
SEED = 42


def combined_label(fruit_label, freshness_label):
    return fruit_label * 2 + freshness_label


def decode_combined(label):
    """Split a combined (16-class) label into (fruit_label, freshness_label)."""
    return label // 2, label % 2


def parse_folder(folder_name):
    """'F_Banana' -> ('Banana', 'fresh'); 'S_Tomato' -> ('Tomato', 'spoiled')."""
    prefix, fruit = folder_name.split("_", 1)
    return fruit, FOLDER_PREFIX[prefix]


def load_json(name):
    with open(CONFIGS_DIR / name, encoding="utf-8") as f:
        return json.load(f)
