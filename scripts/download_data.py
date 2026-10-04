"""Download and unzip the Spoiled and Fresh Fruit Inspection Dataset (Mendeley 6ps7gtp2wg, version 1).

python scripts/download_data.py            # skips the download if the data is complete
python scripts/download_data.py --force
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATASET_DIR, DATASET_ZIP_URL, FRUITS, RAW_DIR  # noqa: E402

ZIP_PATH = RAW_DIR / "6ps7gtp2wg-1.zip"
EXPECTED_FOLDERS = [f"{p}_{fruit}" for p in ("F", "S") for fruit in FRUITS]
EXPECTED_PER_FOLDER = 1000


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Mendeley rejects the default urllib user agent.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def verify():
    missing = [f for f in EXPECTED_FOLDERS if not (DATASET_DIR / f).is_dir()]
    if missing:
        return f"missing folders: {missing}"
    counts = {f: len(list((DATASET_DIR / f).glob("*.jpg"))) for f in EXPECTED_FOLDERS}
    wrong = {f: n for f, n in counts.items() if n != EXPECTED_PER_FOLDER}
    if wrong:
        return f"unexpected image counts: {wrong}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = parser.parse_args()

    if not args.force and verify() is None:
        print(f"Dataset already present and complete: {DATASET_DIR}")
        return

    if args.force or not ZIP_PATH.exists():
        print(f"Downloading {DATASET_ZIP_URL}")
        download(DATASET_ZIP_URL, ZIP_PATH)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    print(f"Extracting {ZIP_PATH.name} -> {RAW_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)

    problem = verify()
    if problem:
        sys.exit(f"Dataset check failed: {problem}")
    print(f"OK: {len(EXPECTED_FOLDERS)} folders x {EXPECTED_PER_FOLDER} images in {DATASET_DIR}")


if __name__ == "__main__":
    main()
