"""Data cleaning and leakage-aware train/val/test splitting.

Why this module exists: the dataset was captured as bursts of frames, so consecutive files
in a class folder are often near-copies of each other (and 1,273 files are byte-identical
copies). A plain random split would put near-identical frames in both train and test and
inflate test accuracy. The split is therefore made over *groups* of related images.

Steps (used by scripts/audit_data.py and scripts/make_splits.py):
    scan_dataset              one row per file: labels, file stats, SHA-256, pHash, colour stats
    mark_exact_duplicates     keep the first file of every SHA-256 group
    find_label_conflicts      near-identical images stored under two different labels
    near_duplicate_pairs      near-identical images inside one class folder
    build_groups              blocks of consecutive frames, merged through near-duplicate pairs
    grouped_stratified_split  whole groups -> train/val/test, separately per class
    leakage_summary           share of evaluation images that have a near-copy in train
"""
import hashlib
import time
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from src.config import CACHE_DIR, DATASET_DIR, FRUITS, PROJECT_ROOT, combined_label, load_json, parse_folder

THUMB_SIZE = 16
SCAN_CSV = CACHE_DIR / "scan.csv"
THUMBS_NPY = CACHE_DIR / "thumbs.npy"


# ----------------------------------------------------------------------------- scanning
def scan_dataset(dataset_dir):
    """Read every image once.

    Returns (df, thumbs): df has one row per file; thumbs[i] is the 16x16 RGB thumbnail of
    row i flattened to 768 values in [0, 1] (NaN for unreadable files).
    """
    rows, thumbs = [], []
    folders = sorted(p for p in Path(dataset_dir).iterdir() if p.is_dir())
    for folder in folders:
        fruit, freshness = parse_folder(folder.name)
        fruit_label = FRUITS.index(fruit)
        freshness_label = int(freshness == "spoiled")
        for path in sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem)):
            data = path.read_bytes()
            row = {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "folder": folder.name,
                "fruit": fruit,
                "freshness": freshness,
                "fruit_label": fruit_label,
                "freshness_label": freshness_label,
                "combined_label": combined_label(fruit_label, freshness_label),
                "file_index": int(path.stem),
                "file_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            try:
                with Image.open(path) as im:
                    im.load()
                    row.update(width=im.width, height=im.height, mode=im.mode)
                    rgb = im.convert("RGB")
                pixels = np.asarray(rgb, dtype=np.float32) / 255.0
                row["phash"] = str(imagehash.phash(rgb))
                for i, channel in enumerate("rgb"):
                    row[f"mean_{channel}"] = float(pixels[..., i].mean())
                    row[f"meansq_{channel}"] = float((pixels[..., i] ** 2).mean())
                row["readable"] = True
                thumb = np.asarray(rgb.resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR), np.float32) / 255.0
            except Exception as exc:  # corrupt or non-image file
                row.update(readable=False, error=repr(exc))
                thumb = np.full((THUMB_SIZE, THUMB_SIZE, 3), np.nan, np.float32)
            rows.append(row)
            thumbs.append(thumb.reshape(-1))
    return pd.DataFrame(rows), np.stack(thumbs)


def load_or_scan(rescan=False):
    """Cached scan_dataset(): data/cache/scan.csv + data/cache/thumbs.npy."""
    if not rescan and SCAN_CSV.exists() and THUMBS_NPY.exists():
        return pd.read_csv(SCAN_CSV, keep_default_na=False, na_values=[""]), np.load(THUMBS_NPY)
    t0 = time.time()
    print(f"Scanning {DATASET_DIR} ...")
    df, thumbs = scan_dataset(DATASET_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCAN_CSV, index=False)
    np.save(THUMBS_NPY, thumbs)
    print(f"  scanned {len(df)} files in {time.time() - t0:.0f}s")
    return df, thumbs


def prepare(rescan=False):
    """Scan + exact duplicates + label conflicts + cleaning decisions (configs/data_config.json)."""
    cfg = load_json("data_config.json")
    df, thumbs = load_or_scan(rescan)
    df = mark_exact_duplicates(df)
    lc = cfg["label_conflict"]
    conflicts = find_label_conflicts(df, thumbs, lc["phash_max"], lc["thumb_max"])
    df = apply_cleaning(df, conflicts, cfg["manual_exclusions"])
    return cfg, df, thumbs, conflicts


# ----------------------------------------------------------------------------- distances
def phash_array(df):
    return np.array([int(h, 16) for h in df["phash"]], dtype=np.uint64)


def hamming(a, b):
    """Pairwise Hamming distance between two arrays of 64-bit perceptual hashes."""
    return np.bitwise_count(a[:, None] ^ b[None, :]).astype(np.int16)


def thumb_distance(a, b, chunk=64):
    """Pairwise mean absolute difference between flattened thumbnails (0 = identical, 1 = opposite)."""
    out = np.empty((len(a), len(b)), np.float32)
    for start in range(0, len(a), chunk):
        out[start:start + chunk] = np.abs(a[start:start + chunk, None, :] - b[None, :, :]).mean(-1)
    return out


# ----------------------------------------------------------------------------- cleaning
def mark_exact_duplicates(df):
    """Flag byte-identical files; the first file (lowest index in folder order) is kept."""
    first = df.groupby("sha256", sort=False)["path"].transform("first")
    df["duplicate_of"] = np.where(df["path"] != first, first, "")
    df["is_exact_duplicate"] = df["duplicate_of"] != ""
    return df


def find_label_conflicts(df, thumbs, phash_max, thumb_max, chunk=2000):
    """Near-identical images (pHash <= phash_max and thumbnail distance <= thumb_max) that are
    stored under different class labels. Exact duplicates are ignored (their kept copy is checked)."""
    mask = (df["readable"] & ~df["is_exact_duplicate"]).to_numpy()
    idx = np.where(mask)[0]
    ph = phash_array(df.iloc[idx])
    labels = df["combined_label"].to_numpy()[idx]
    found = []
    for start in range(0, len(idx), chunk):
        block = slice(start, start + chunk)
        dist = hamming(ph[block], ph)
        ii, jj = np.where(dist <= phash_max)
        ii = ii + start
        keep = (jj > ii) & (labels[ii] != labels[jj])
        for i, j in zip(ii[keep], jj[keep]):
            d = float(np.abs(thumbs[idx[i]] - thumbs[idx[j]]).mean())
            if d <= thumb_max:
                found.append((df["path"].iat[idx[i]], df["path"].iat[idx[j]], int(dist[i - start, j]), round(d, 4)))
    return pd.DataFrame(found, columns=["path_a", "path_b", "phash_distance", "thumb_distance"])


def apply_cleaning(df, conflicts, manual_exclusions):
    """Decide which files are excluded and record why (columns `excluded`, `exclude_reason`).

    Order of rules: unreadable -> exact duplicate copy -> manual exclusion (reviewed label
    conflicts) -> any label conflict not resolved manually (both images dropped).
    """
    reason = pd.Series("", index=df.index, dtype=object)
    reason[~df["readable"]] = "unreadable file"
    is_dup = (reason == "") & df["is_exact_duplicate"]
    reason[is_dup] = "exact duplicate of " + df.loc[is_dup, "duplicate_of"]
    manual = {m["path"]: m["reason"] for m in manual_exclusions}
    for path, why in manual.items():
        reason[df["path"] == path] = "manual review: " + why
    for a, b in conflicts[["path_a", "path_b"]].itertuples(index=False):
        if a not in manual and b not in manual:
            for p, other in ((a, b), (b, a)):
                reason[df["path"] == p] = f"unresolved label conflict with {other}"
    df["exclude_reason"] = reason
    df["excluded"] = reason != ""
    return df


def near_duplicate_pairs(df, thumbs, phash_max, thumb_max, gap_max=0):
    """Positional index pairs (i < j) of near-identical images inside the same class.

    A pair qualifies when pHash distance <= phash_max AND either the thumbnail distance is
    <= thumb_max (near-identical pixels) or the two frames are at most `gap_max` files apart
    (same fruit re-shot a few seconds later under slightly different light).
    `thumbs` must be aligned with the rows of `df`.
    """
    ph = phash_array(df)
    file_index = df["file_index"].to_numpy()
    pairs = []
    for idx in df.groupby("combined_label").indices.values():
        dist = hamming(ph[idx], ph[idx])
        ii, jj = np.where(np.triu(dist <= phash_max, k=1))
        if len(ii) == 0:
            continue
        d = np.abs(thumbs[idx[ii]] - thumbs[idx[jj]]).mean(1)
        gap = np.abs(file_index[idx[ii]] - file_index[idx[jj]])
        keep = (d <= thumb_max) | (gap <= gap_max)
        pairs.append(np.stack([idx[ii[keep]], idx[jj[keep]]], axis=1))
    return np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=int)


# ----------------------------------------------------------------------------- grouping & splitting
def build_groups(df, pairs, block_size):
    """Union-find over (a) blocks of `block_size` consecutive frames of one class folder and
    (b) near-duplicate pairs. Returns a 0-based group id per row of `df`."""
    parent = np.arange(len(df))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    block = df["folder"] + ":" + ((df["file_index"] - 1) // block_size).astype(str)
    for idx in df.groupby(block).indices.values():
        for x in idx[1:]:
            union(idx[0], x)
    for a, b in pairs:
        union(a, b)
    roots = np.array([find(x) for x in range(len(df))])
    return pd.factorize(roots)[0]


def grouped_stratified_split(df, ratios, seed):
    """Assign whole groups to splits separately inside every class (stratification).

    Groups are processed largest first (random order among equal sizes) and each goes to the
    split with the most room left (target count minus current count), so large near-duplicate
    clusters land in train and cannot unbalance the validation or test set.
    """
    names = list(ratios)
    target = np.array([ratios[n] for n in names], dtype=float)
    rng = np.random.default_rng(seed)
    groups_all = df["group_id"].to_numpy()
    split = np.empty(len(df), dtype=object)
    for _, idx in sorted(df.groupby("combined_label").indices.items()):
        groups = groups_all[idx]
        ids, sizes = np.unique(groups, return_counts=True)
        order = rng.permutation(len(ids))
        order = order[np.argsort(-sizes[order], kind="stable")]
        want, have = target * len(idx), np.zeros(len(names))
        for k in order:
            s = int(np.argmax(want - have))
            have[s] += sizes[k]
            split[idx[groups == ids[k]]] = names[s]
    return split


def random_stratified_split(df, ratios, seed):
    """Naive image-level stratified split - used only as the leaky baseline in the audit."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    y = df["combined_label"].to_numpy()
    train, rest = train_test_split(idx, train_size=ratios["train"], stratify=y, random_state=seed)
    val_share = ratios["val"] / (ratios["val"] + ratios["test"])
    val, test = train_test_split(rest, train_size=val_share, stratify=y[rest], random_state=seed)
    split = np.empty(len(df), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    return split


# ----------------------------------------------------------------------------- leakage audit
def nearest_train_match(df, thumbs, split, eval_split, phash_max, thumb_max):
    """For each image of `eval_split`: distance to the closest train image of the same class.

    Returns a DataFrame with min_phash, min_thumb and near_duplicate (a train image exists with
    pHash <= phash_max AND thumbnail distance <= thumb_max - the grouping criterion).
    """
    ph = phash_array(df)
    records = []
    for _, idx in df.groupby("combined_label").indices.items():
        e = idx[split[idx] == eval_split]
        r = idx[split[idx] == "train"]
        if len(e) == 0 or len(r) == 0:
            continue
        h = hamming(ph[e], ph[r])
        t = thumb_distance(thumbs[e], thumbs[r])
        near = ((h <= phash_max) & (t <= thumb_max)).any(axis=1)
        records.append(pd.DataFrame({"row": e, "min_phash": h.min(1), "min_thumb": t.min(1), "near_duplicate": near}))
    return pd.concat(records).sort_values("row").reset_index(drop=True)


def leakage_summary(match, phash_max, thumb_max):
    n = len(match)
    return {
        "n_images": n,
        f"near-duplicate in train (pHash<={phash_max} & thumb<={thumb_max})": match["near_duplicate"].mean(),
        f"pHash<={phash_max} match in train": (match["min_phash"] <= phash_max).mean(),
        "pHash<=10 match in train": (match["min_phash"] <= 10).mean(),
        f"thumb<={thumb_max} match in train": (match["min_thumb"] <= thumb_max).mean(),
        "median nearest pHash distance": float(match["min_phash"].median()),
    }
