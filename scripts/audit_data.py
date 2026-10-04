"""Data audit: readability, image statistics, exact duplicates and label conflicts.

python scripts/audit_data.py            # uses the cached scan if there is one
python scripts/audit_data.py --rescan   # read all 16,000 images again
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_audit as da  # noqa: E402
from src.config import (  # noqa: E402
    COMBINED_NAMES,
    DATASET_DIR,
    DATASET_DOI,
    DATASET_LICENSE,
    DATASET_PAGE,
    FIGURES_DIR,
    FRUITS,
    IMG_SIZE,
    PROJECT_ROOT,
    REPORTS_DIR,
    SEED,
)


def write_structure(df):
    lines = [
        "Dataset : Spoiled and fresh fruit inspection dataset (Pachon Suescun, Pinzon Arenas, Jimenez-Moreno, 2020)",
        f"Source  : {DATASET_PAGE}  (DOI {DATASET_DOI}, license {DATASET_LICENSE})",
        f"Root    : {DATASET_DIR.relative_to(PROJECT_ROOT).as_posix()}",
        "Layout  : <F|S>_<Fruit>/<n>.jpg   F = fresh, S = spoiled, n = 1..1000 (capture order)",
        "Labels  : fruit_label 0-7 (alphabetical), freshness_label 0 = fresh / 1 = spoiled (positive class),",
        "          combined_label = fruit_label * 2 + freshness_label (0-15)",
        "",
        f"{'folder':14s} {'fruit':11s} {'freshness':9s} {'fruit_label':>11s} {'fresh_label':>11s} {'combined':>8s}  {'combined_name':18s} {'files':>5s}",
    ]
    for folder, g in df.groupby("folder"):
        r = g.iloc[0]
        lines.append(
            f"{folder:14s} {r.fruit:11s} {r.freshness:9s} {r.fruit_label:11d} {r.freshness_label:11d} "
            f"{r.combined_label:8d}  {COMBINED_NAMES[r.combined_label]:18s} {len(g):5d}"
        )
    lines.append(f"\nTotal files: {len(df)}")
    (REPORTS_DIR / "data_structure.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_report(df, conflicts):
    rows = []
    for r in df.itertuples():
        if not r.readable:
            rows.append((r.path, "unreadable", r.error, "excluded"))
            continue
        if (r.width, r.height) != (IMG_SIZE, IMG_SIZE):
            rows.append((r.path, "unexpected_size", f"{r.width}x{r.height}", "kept (resized in pipeline)"))
        if r.mode != "RGB":
            rows.append((r.path, "non_rgb_mode", r.mode, "kept (converted to RGB)"))
        if r.is_exact_duplicate:
            rows.append((r.path, "exact_duplicate", f"identical to {r.duplicate_of}", "excluded (copy)"))
    for c in conflicts.itertuples():
        for p, other in ((c.path_a, c.path_b), (c.path_b, c.path_a)):
            excluded = bool(df.loc[df.path == p, "excluded"].iat[0])
            rows.append(
                (
                    p,
                    "label_conflict",
                    f"near-identical to {other} (pHash {c.phash_distance}, thumb {c.thumb_distance})",
                    "excluded (manual review)" if excluded else "kept (conflicting copy excluded)",
                )
            )
    report = pd.DataFrame(rows, columns=["path", "issue", "details", "action"])
    report.to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)
    return report


def write_image_statistics(df):
    def stats(g, name):
        means = g[["mean_r", "mean_g", "mean_b"]].mean().to_numpy()
        meansq = g[["meansq_r", "meansq_g", "meansq_b"]].mean().to_numpy()
        std = np.sqrt(np.maximum(meansq - means**2, 0))
        aspect = g["width"] / g["height"]
        return {
            "group": name,
            "n_files": len(g),
            "width_min": int(g.width.min()),
            "width_max": int(g.width.max()),
            "height_min": int(g.height.min()),
            "height_max": int(g.height.max()),
            "aspect_min": round(float(aspect.min()), 3),
            "aspect_max": round(float(aspect.max()), 3),
            "modes": ";".join(sorted(g["mode"].unique())),
            "file_kb_mean": round(g.file_bytes.mean() / 1024, 2),
            "file_kb_min": round(g.file_bytes.min() / 1024, 2),
            "file_kb_max": round(g.file_bytes.max() / 1024, 2),
            "mean_r": round(means[0], 4),
            "mean_g": round(means[1], 4),
            "mean_b": round(means[2], 4),
            "std_r": round(std[0], 4),
            "std_g": round(std[1], 4),
            "std_b": round(std[2], 4),
        }

    ok = df[df.readable]
    rows = [stats(g, folder) for folder, g in ok.groupby("folder")]
    rows.append(stats(ok, "ALL (raw)"))
    rows.append(stats(ok[~ok.excluded], "ALL (after cleaning)"))
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "image_statistics.csv", index=False)
    return out


def write_duplicate_groups(df):
    dup = df[df.duplicated("sha256", keep=False)]
    rows = [
        {
            "sha256": h,
            "n_files": len(g),
            "folder": g.folder.iat[0],
            "kept": g.path.iat[0],
            "removed": ";".join(g.path.iloc[1:]),
        }
        for h, g in dup.groupby("sha256", sort=False)
    ]
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "duplicate_groups.csv", index=False)
    return out


def _show(ax, path, caption):
    from PIL import Image

    from src import viz

    ax.imshow(Image.open(PROJECT_ROOT / path))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(caption, fontsize=8, color=viz.TEXT_LIGHT, labelpad=3)


def figure_samples(df):
    import matplotlib.pyplot as plt

    from src import viz

    viz.apply_style()
    rng = np.random.default_rng(SEED)
    kept = df[~df.excluded]
    fig, axes = plt.subplots(4, len(FRUITS), figsize=(13.5, 8.3))
    for col, fruit in enumerate(FRUITS):
        for state_row, state in enumerate(["fresh", "spoiled"]):
            pool = kept[(kept.fruit == fruit) & (kept.freshness == state)]
            picks = pool.iloc[rng.choice(len(pool), 2, replace=False)]
            for k, r in enumerate(picks.itertuples()):
                ax = axes[state_row * 2 + k, col]
                _show(ax, r.path, f"{r.folder}/{r.file_index}.jpg")
        axes[0, col].set_title(fruit, fontsize=11)
    for row, label in enumerate(["Fresh", "Fresh", "Spoiled", "Spoiled"]):
        axes[row, 0].set_ylabel(label, fontsize=11, color=viz.TEXT, labelpad=8)
    fig.suptitle(
        "FRUIT-16K: two random examples per class (8 fruits x fresh / spoiled, 224x224 RGB)",
        fontsize=12.5,
        fontweight="semibold",
        color=viz.TEXT,
    )
    fig.subplots_adjust(left=0.04, right=0.995, top=0.9, bottom=0.04, hspace=0.3, wspace=0.06)
    viz.save(fig, FIGURES_DIR / "dataset_samples.png")


def figure_duplicates(df, thumbs, cfg, conflicts):
    import matplotlib.pyplot as plt

    from src import viz

    viz.apply_style()
    by_path = df.set_index("path")
    # (1) exact duplicates: first two duplicate groups of two different folders
    groups = pd.read_csv(REPORTS_DIR / "duplicate_groups.csv")
    exact = []
    for folder in ["F_Lemon", "F_Tamarillo"]:
        g = groups[groups.folder == folder].iloc[0]
        exact.append((g.kept, g.removed.split(";")[0]))
    # (2) four consecutive kept frames where every neighbouring pair is a near-duplicate
    kept = df[~df.excluded].reset_index(drop=True)
    g = cfg["grouping"]
    pairs = da.near_duplicate_pairs(
        kept, thumbs[df.index[~df.excluded]], g["near_duplicate_phash_max"], g["near_duplicate_thumb_max"]
    )
    # (gap rule not used here on purpose: this panel shows pixel-level near-copies only)
    linked = {(int(a), int(b)) for a, b in pairs}
    starts = [
        i
        for i in range(len(kept) - 3)
        if kept.folder.iat[i] == kept.folder.iat[i + 3]
        and kept.folder.iat[i].startswith("S_")
        and all((i + k, i + k + 1) in linked for k in range(3))
    ]
    start = starts[len(starts) // 2] if starts else int(pairs[0, 0])
    run = kept.iloc[start : start + 4]

    fig, axes = plt.subplots(3, 4, figsize=(9.4, 8.0))
    for k, (a, b) in enumerate(exact):
        _show(axes[0, 2 * k], a, a.split("FRUIT-16K/")[1])
        _show(axes[0, 2 * k + 1], b, b.split("FRUIT-16K/")[1] + "  (identical)")
    ph = da.phash_array(run)
    for k, r in enumerate(run.itertuples()):
        cap = f"{r.folder}/{r.file_index}.jpg"
        if k:
            cap += f"\npHash dist. to previous: {int(np.bitwise_count(ph[k] ^ ph[k - 1]))}"
        _show(axes[1, k], r.path, cap)
    c = conflicts.iloc[0]
    _show(axes[2, 0], c.path_a, c.path_a.split("FRUIT-16K/")[1] + f"\nlabel: {by_path.loc[c.path_a, 'freshness']}")
    _show(axes[2, 1], c.path_b, c.path_b.split("FRUIT-16K/")[1] + f"\nlabel: {by_path.loc[c.path_b, 'freshness']}")
    for ax in axes[2, 2:]:
        ax.axis("off")
    axes[2, 2].text(
        0.02,
        0.55,
        f"Same photo stored under both labels\n(pHash distance {c.phash_distance}, thumbnail distance {c.thumb_distance}).\n"
        "F_Banana/1.jpg is excluded as mislabelled.",
        fontsize=9.5,
        color=viz.TEXT_LIGHT,
        va="center",
        transform=axes[2, 2].transAxes,
    )
    titles = [
        "A. Exact duplicates (1,273 byte-identical copies in 4 folders)",
        "B. Consecutive burst frames are near-copies",
        "C. Label conflict between folders",
    ]
    for row, title in enumerate(titles):
        axes[row, 0].set_title(title, loc="left", fontsize=11, pad=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.06, wspace=0.08, hspace=0.42)
    viz.save(fig, FIGURES_DIR / "duplicate_examples.png")


def main():
    parser = argparse.ArgumentParser(description="Audit the FRUIT-16K dataset")
    parser.add_argument("--rescan", action="store_true", help="re-read all images instead of using the cache")
    args = parser.parse_args()
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    cfg, df, thumbs, conflicts = da.prepare(args.rescan)
    write_structure(df)
    quality = write_quality_report(df, conflicts)
    stats = write_image_statistics(df)
    dups = write_duplicate_groups(df)
    conflicts.to_csv(REPORTS_DIR / "label_conflicts.csv", index=False)
    figure_samples(df)
    figure_duplicates(df, thumbs, cfg, conflicts)

    print(f"files scanned        : {len(df)}")
    print(f"unreadable           : {(~df.readable).sum()}")
    print(f"sizes / modes        : {sorted(set(zip(df.width, df.height)))} / {sorted(df['mode'].unique())}")
    print(
        f"exact duplicate copies: {df.is_exact_duplicate.sum()} in {len(dups)} groups "
        f"({df[df.is_exact_duplicate].folder.value_counts().to_dict()})"
    )
    print(f"label conflicts      : {len(conflicts)} -> {conflicts.to_dict('records')}")
    print(f"excluded in total    : {df.excluded.sum()}  -> kept {(~df.excluded).sum()}")
    print(f"quality report rows  : {len(quality)} ({quality.issue.value_counts().to_dict()})")
    print(
        "overall mean/std RGB (after cleaning):",
        stats.iloc[-1][["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]].to_dict(),
    )


if __name__ == "__main__":
    main()
