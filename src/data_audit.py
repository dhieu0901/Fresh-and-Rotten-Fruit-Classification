# Ảnh được chụp liên tiếp (burst) nên các file cạnh nhau thường gần như giống hệt. Chia ngẫu nhiên theo ảnh
# sẽ để ảnh gần trùng nằm cả ở train và test (rò rỉ dữ liệu), vì vậy phải chia theo nhóm ảnh liên quan.

import hashlib
import time
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from src.config import CACHE_DIR, DATASET_DIR, FRUITS, PROJECT_ROOT, combined_label, parse_folder

THUMB_SIZE = 16
SCAN_CSV = CACHE_DIR / "scan.csv"
THUMBS_NPY = CACHE_DIR / "thumbs.npy"


def scan_dataset(dataset_dir):
    # SHA-256: tìm file trùng từng byte; pHash: tìm ảnh nhìn giống nhau; thumbnail 16x16: so sánh pixel
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
            except Exception as exc:
                row.update(readable=False, error=repr(exc))
                thumb = np.full((THUMB_SIZE, THUMB_SIZE, 3), np.nan, np.float32)
            rows.append(row)
            thumbs.append(thumb.reshape(-1))
    return pd.DataFrame(rows), np.stack(thumbs)


def load_or_scan(rescan=False):
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


def prepare(conflict_phash_max, conflict_thumb_max, manual_exclusions, rescan=False):
    df, thumbs = load_or_scan(rescan)
    df = mark_exact_duplicates(df)
    conflicts = find_label_conflicts(df, thumbs, conflict_phash_max, conflict_thumb_max)
    df = apply_cleaning(df, conflicts, manual_exclusions)
    return df, thumbs, conflicts


def phash_array(df):
    return np.array([int(h, 16) for h in df["phash"]], dtype=np.uint64)


def hamming(a, b):
    # khoảng cách Hamming = số bit khác nhau giữa hai pHash 64 bit, càng nhỏ ảnh càng giống
    return np.bitwise_count(a[:, None] ^ b[None, :]).astype(np.int16)


def thumb_distance(a, b, chunk=64):
    out = np.empty((len(a), len(b)), np.float32)
    for start in range(0, len(a), chunk):
        out[start : start + chunk] = np.abs(a[start : start + chunk, None, :] - b[None, :, :]).mean(-1)
    return out


def mark_exact_duplicates(df):
    # file trùng từng byte: giữ file đầu tiên, bỏ các bản sao
    first = df.groupby("sha256", sort=False)["path"].transform("first")
    df["duplicate_of"] = np.where(df["path"] != first, first, "")
    df["is_exact_duplicate"] = df["duplicate_of"] != ""
    return df


def find_label_conflicts(df, thumbs, phash_max, thumb_max, chunk=2000):
    # cùng một ảnh nhưng nằm ở hai nhãn khác nhau
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
    reason = pd.Series("", index=df.index, dtype=object)
    reason[~df["readable"]] = "unreadable file"
    is_dup = (reason == "") & df["is_exact_duplicate"]
    reason[is_dup] = "exact duplicate of " + df.loc[is_dup, "duplicate_of"]
    manual = dict(manual_exclusions)
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
    # cặp gần trùng trong cùng lớp: pHash gần nhau và (thumbnail gần nhau hoặc cách nhau <= gap_max file)
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


def build_groups(df, pairs, block_size):
    # mỗi block_size ảnh liên tiếp là một nhóm, các nhóm có cặp gần trùng được gộp lại (union-find)
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
    # chia nguyên nhóm vào một tập, làm riêng cho từng lớp (stratified); nhóm lớn xếp trước,
    # mỗi nhóm vào tập đang thiếu nhiều ảnh nhất so với tỉ lệ 70/15/15
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
    # chia ngẫu nhiên theo từng ảnh, chỉ dùng để so sánh mức rò rỉ
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    y = df["combined_label"].to_numpy()
    train, rest = train_test_split(idx, train_size=ratios["train"], stratify=y, random_state=seed)
    val_share = ratios["val"] / (ratios["val"] + ratios["test"])
    val, test = train_test_split(rest, train_size=val_share, stratify=y[rest], random_state=seed)
    split = np.empty(len(df), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    return split


def nearest_train_match(df, thumbs, split, eval_split, phash_max, thumb_max):
    # với mỗi ảnh val/test: ảnh train cùng lớp giống nó nhất
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
