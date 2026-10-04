"""Leakage-aware stratified train / validation / test split (70 / 15 / 15).

Run after scripts/audit_data.py (re-uses its cached scan):
    python scripts/make_splits.py

Writes
    data/metadata.csv                 every file: labels, hashes, cleaning decision, group, split
    data/splits/{train,val,test}.csv  the files each model trains / validates / tests on
    reports/class_distribution.csv    per-class counts: raw, removed, clean, train, val, test
    reports/leakage_comparison.csv    near-copy rates for the grouped split vs a naive random split
    reports/split_audit.txt           integrity checks + summary
    figures/class_distribution.png
    figures/split_leakage.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_audit as da  # noqa: E402
from src.config import COMBINED_NAMES, FIGURES_DIR, METADATA_CSV, REPORTS_DIR, SPLITS_DIR  # noqa: E402

SPLITS = ["train", "val", "test"]
METADATA_COLUMNS = [
    "path", "folder", "fruit", "freshness", "fruit_label", "freshness_label", "combined_label",
    "combined_name", "file_index", "width", "height", "mode", "file_bytes", "sha256", "phash",
    "is_exact_duplicate", "duplicate_of", "excluded", "exclude_reason", "group_id", "split",
]
SPLIT_COLUMNS = ["path", "fruit", "freshness", "fruit_label", "freshness_label", "combined_label", "combined_name", "group_id"]


def class_distribution(df, kept):
    rows = []
    for label in range(len(COMBINED_NAMES)):
        raw = df[df.combined_label == label]
        k = kept[kept.combined_label == label]
        row = {"combined_label": label, "class": COMBINED_NAMES[label], "folder": raw.folder.iat[0],
               "raw": len(raw), "removed": int(raw.excluded.sum()), "clean": len(k)}
        for s in SPLITS:
            row[s] = int((k.split == s).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    total = out[["raw", "removed", "clean", *SPLITS]].sum()
    out = pd.concat([out, pd.DataFrame([{"combined_label": "", "class": "TOTAL", "folder": "", **total}])], ignore_index=True)
    for s in SPLITS:
        out[f"{s}_pct"] = (100 * out[s] / out["clean"]).round(1)
    return out


def leakage_table(kept, thumbs, grouped, random, cfg):
    g = cfg["grouping"]
    pmax, tmax = g["near_duplicate_phash_max"], g["near_duplicate_thumb_max"]
    rows, matches = [], {}
    for name, split in (("random split", random), ("grouped split", grouped)):
        for part in ("val", "test"):
            m = da.nearest_train_match(kept, thumbs, split, part, pmax, tmax)
            matches[(name, part)] = m
            rows.append({"split_method": name, "evaluated_on": part, **da.leakage_summary(m, pmax, tmax)})
    return pd.DataFrame(rows), matches


def integrity_checks(kept, pairs):
    split = kept["split"].to_numpy()
    by_sha = kept.groupby("sha256")["split"].nunique()
    by_group = kept.groupby("group_id")["split"].nunique()
    return {
        "files assigned to more than one split": int(kept["path"].duplicated().sum()),
        "SHA-256 shared between splits": int((by_sha > 1).sum()),
        "groups spanning more than one split": int((by_group > 1).sum()),
        "near-duplicate pairs crossing splits": int((split[pairs[:, 0]] != split[pairs[:, 1]]).sum()) if len(pairs) else 0,
    }


def write_audit(path, cfg, df, kept, pairs, checks, dist, leak):
    g, r = cfg["grouping"], cfg["split_ratios"]
    sizes = kept.groupby("group_id").size()
    per_class = kept.groupby("combined_label").agg(groups=("group_id", "nunique"))
    largest = kept.groupby(["combined_label", "group_id"]).size().groupby("combined_label").max() / kept.groupby("combined_label").size()
    reasons = df.loc[df.excluded, "exclude_reason"].str.split(" of | with |:", n=1, regex=True).str[0].value_counts()
    t = dist[dist["class"] == "TOTAL"].iloc[0]
    body = dist[dist["class"] != "TOTAL"]
    dev = max(abs(body[f"{s}_pct"] - 100 * r[s]).max() for s in SPLITS)
    lines = [
        "SPLIT AUDIT - Fresh and Rotten Fruit Classification",
        "=" * 60,
        f"Seed {cfg['seed']} | target train/val/test = {r['train']:.0%}/{r['val']:.0%}/{r['test']:.0%}",
        f"Grouping: blocks of {g['block_size']} consecutive frames per class folder, merged whenever two images of the class",
        f"are near-duplicates: pHash distance <= {g['near_duplicate_phash_max']} AND (16x16 thumbnail distance <= "
        f"{g['near_duplicate_thumb_max']} OR at most {g['near_duplicate_gap_max']} files apart).",
        "Whole groups are assigned to one split, separately inside each of the 16 classes (stratified),",
        "largest groups first, each to the split with the most room left.",
        "",
        "1. Cleaning",
        f"   raw files: {len(df)} | excluded: {int(df.excluded.sum())} | kept: {len(kept)}",
        *[f"   - {k}: {v}" for k, v in reasons.items()],
        "",
        "2. Groups",
        f"   near-duplicate pairs inside classes: {len(pairs)}",
        f"   groups: {len(sizes)} | per class min/median/max: {per_class.groups.min()}/{int(per_class.groups.median())}/{per_class.groups.max()}",
        f"   group size min/median/max: {sizes.min()}/{int(sizes.median())}/{sizes.max()}",
        f"   largest group as share of its class: max {largest.max():.1%} ({COMBINED_NAMES[int(largest.idxmax())]})",
        "",
        "3. Split sizes",
        *[f"   {s:5s}: {int(t[s]):5d} images ({t[s] / t['clean']:.1%})" for s in SPLITS],
        f"   largest per-class deviation from the target share: {dev:.1f} percentage points",
        "   per-class counts: reports/class_distribution.csv",
        "",
        "4. Integrity checks (all must be 0)",
        *[f"   {k}: {v}" for k, v in checks.items()],
        "",
        "5. Leakage: share of evaluation images with a near-copy of the same class in TRAIN",
        "   (the naive random split is shown only for comparison; it is not used)",
    ]
    cols = [c for c in leak.columns if c not in ("split_method", "evaluated_on")]
    header = f"   {'metric':50s}" + "".join(f"{m.split()[0] + ' ' + p:>14s}" for m, p in zip(leak.split_method, leak.evaluated_on))
    lines.append(header)
    for c in cols:
        vals = leak[c].to_numpy()
        fmt = (lambda v: f"{v:14.0f}") if c in ("n_images", "median nearest pHash distance") else (lambda v: f"{v:14.1%}")
        lines.append(f"   {c:50s}" + "".join(fmt(v) for v in vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- figures
def figure_distribution(dist):
    import matplotlib.pyplot as plt
    from src import viz
    viz.apply_style()
    body = dist[dist["class"] != "TOTAL"].iloc[::-1]
    y = np.arange(len(body))
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    left = np.zeros(len(body))
    segments = [("train", viz.SERIES[0]), ("val", viz.SERIES[1]), ("test", viz.SERIES[2]), ("removed", viz.BASELINE)]
    labels = {"train": "Train", "val": "Validation", "test": "Test", "removed": "Removed (duplicates / conflict)"}
    for key, color in segments:
        vals = body[key].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.62, color=color, edgecolor=viz.SURFACE, linewidth=1.5, label=labels[key])
        left += vals
    for yi, (clean, raw) in enumerate(zip(body["clean"], body["raw"])):
        ax.text(raw + 12, yi, f"{clean}", va="center", fontsize=8.5, color=viz.INK_2)
    ax.set_yticks(y, body["class"].str.replace("_", " "))
    ax.set_xlim(0, 1120)
    ax.set_xlabel("Images")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=9, handlelength=1.2)
    ax.set_title("Images per class after cleaning, by split (label = clean total)", pad=34, loc="left")
    viz.save(fig, FIGURES_DIR / "class_distribution.png")


def figure_leakage(matches, cfg):
    import matplotlib.pyplot as plt
    from src import viz
    viz.apply_style()
    pmax = cfg["grouping"]["near_duplicate_phash_max"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    # imagehash's pHash always has 32 one-bits, so Hamming distances are even numbers only
    bins = np.arange(0, 33, 2)
    ax.axvspan(-1, pmax + 1, color=viz.GRID, alpha=0.6, lw=0)
    ax.text(pmax / 2, 0.97, "near-copy\nzone", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=viz.INK_2)
    for (name, color) in (("random split", viz.SERIES[1]), ("grouped split", viz.SERIES[0])):
        m = matches[(name, "test")]
        counts = np.bincount(m["min_phash"].astype(int), minlength=33)[bins]
        share = 100 * counts / len(m)
        near = (m["min_phash"] <= pmax).mean()
        ax.plot(bins, share, color=color, marker="o", markersize=5, markeredgecolor=viz.SURFACE, markeredgewidth=1.5,
                label=f"{name.capitalize()}: {near:.1%} of test images have a pHash<={pmax} copy in train")
    ax.set_xlim(-1, 32)
    ax.set_xticks(bins)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("pHash distance from each test image to its closest training image of the same class")
    ax.set_ylabel("Share of test images (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5, ncol=1)
    ax.set_title("A naive random split leaks near-copies into the test set; the grouped split does not",
                 loc="left", pad=44)
    viz.save(fig, FIGURES_DIR / "split_leakage.png")


def main():
    cfg, df, thumbs, _ = da.prepare()
    kept = df[~df.excluded].reset_index()          # column "index" = row in df / thumbs
    kthumbs = thumbs[kept["index"].to_numpy()]
    g = cfg["grouping"]

    pairs = da.near_duplicate_pairs(kept, kthumbs, g["near_duplicate_phash_max"], g["near_duplicate_thumb_max"],
                                    g["near_duplicate_gap_max"])
    kept["group_id"] = da.build_groups(kept, pairs, g["block_size"])
    kept["split"] = da.grouped_stratified_split(kept, cfg["split_ratios"], cfg["seed"])
    random = da.random_stratified_split(kept, cfg["split_ratios"], cfg["seed"])

    df["combined_name"] = [COMBINED_NAMES[c] for c in df.combined_label]
    kept["combined_name"] = [COMBINED_NAMES[c] for c in kept.combined_label]
    df["group_id"] = -1
    df["split"] = "excluded"
    df.loc[kept["index"], "group_id"] = kept["group_id"].to_numpy()
    df.loc[kept["index"], "split"] = kept["split"].to_numpy()

    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df[METADATA_COLUMNS].to_csv(METADATA_CSV, index=False)
    for s in SPLITS:
        kept.loc[kept.split == s, SPLIT_COLUMNS].to_csv(SPLITS_DIR / f"{s}.csv", index=False)

    dist = class_distribution(df, kept)
    dist.to_csv(REPORTS_DIR / "class_distribution.csv", index=False)
    leak, matches = leakage_table(kept, kthumbs, kept["split"].to_numpy(), random, cfg)
    leak.to_csv(REPORTS_DIR / "leakage_comparison.csv", index=False)
    checks = integrity_checks(kept, pairs)
    write_audit(REPORTS_DIR / "split_audit.txt", cfg, df, kept, pairs, checks, dist, leak)
    figure_distribution(dist)
    figure_leakage(matches, cfg)

    print((REPORTS_DIR / "split_audit.txt").read_text(encoding="utf-8"))
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        sys.exit(f"INTEGRITY CHECK FAILED: {bad}")


if __name__ == "__main__":
    main()
