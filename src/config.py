from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_DIR = RAW_DIR / "FRUIT-16K"
METADATA_CSV = DATA_DIR / "metadata.csv"
SPLITS_DIR = DATA_DIR / "splits"
CACHE_DIR = DATA_DIR / "cache"

REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

DATASET_PAGE = "https://data.mendeley.com/datasets/6ps7gtp2wg/1"
DATASET_ZIP_URL = "https://data.mendeley.com/public-api/zip/6ps7gtp2wg/download/1"

FRUITS = ["Banana", "Lemon", "Lulo", "Mango", "Orange", "Strawberry", "Tamarillo", "Tomato"]
FRESHNESS = ["fresh", "spoiled"]
NUM_FRUITS = len(FRUITS)
NUM_COMBINED = NUM_FRUITS * len(FRESHNESS)
# nhãn: fruit_label 0-7, freshness_label 0 = tươi / 1 = hỏng,
# combined_label = fruit_label * 2 + freshness_label (16 lớp, dùng cho Model 1)
COMBINED_NAMES = [f"{fruit}_{state}" for fruit in FRUITS for state in FRESHNESS]

IMG_SIZE = 224
SEED = 42


def combined_label(fruit_label, freshness_label):
    return fruit_label * 2 + freshness_label


def parse_folder(folder_name):
    prefix, fruit = folder_name.split("_", 1)
    return fruit, {"F": "fresh", "S": "spoiled"}[prefix]
