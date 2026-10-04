"""Grad-CAM figures for the three models (which part of the fruit drives the fresh/spoiled decision).
Run after evaluate.py, the examples are picked from its test predictions.

    python scripts/gradcam.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import viz  # noqa: E402
from src.config import FIGURES_DIR, FRUITS, MODELS_DIR, PREDICTIONS_DIR, PROJECT_ROOT, SEED  # noqa: E402
from src.data_pipeline import load_image  # noqa: E402
from src.evaluation.gradcam import gradcam, overlay  # noqa: E402

RUNS = [("model1_simple_cnn", "Model 1"), ("model2_multitask_cnn", "Model 2"), ("model3_mobilenet_v2", "Model 3")]
OUT = FIGURES_DIR / "gradcam"


def spoiled_prob(model, x):
    out = model.predict(x, verbose=0)
    if isinstance(out, dict):
        return out["freshness"][:, 0]
    return out.reshape(len(x), -1, 2)[:, :, 1].sum(axis=1)


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure_examples(models, preds, state, path):
    import matplotlib.pyplot as plt

    viz.apply_style()
    rng = np.random.default_rng(SEED)
    target = int(state == "spoiled")
    ok = np.logical_and.reduce([(p["joint_pred"] == p["combined_label"]).to_numpy() for p in preds.values()])
    base = next(iter(preds.values()))
    picks = []
    for f, fruit in enumerate(FRUITS):
        pool = base[ok & (base.fruit_label == f).to_numpy() & (base.freshness_label == target).to_numpy()]
        if len(pool):
            picks.append(pool.iloc[rng.integers(len(pool))])
    images = np.stack([load_image(PROJECT_ROOT / r.path) for r in picks])
    fig, axes = plt.subplots(1 + len(models), len(picks), figsize=(1.75 * len(picks), 1.95 * (1 + len(models))))
    for c, r in enumerate(picks):
        axes[0, c].imshow(images[c].astype(np.uint8))
        axes[0, c].set_title(FRUITS[r.fruit_label], fontsize=10)
        _clean(axes[0, c])
    for row, (run, model) in enumerate(models.items(), start=1):
        heat = gradcam(model, images, "freshness")
        p = spoiled_prob(model, images)
        for c in range(len(picks)):
            axes[row, c].imshow(overlay(images[c], heat[c]))
            axes[row, c].set_xlabel(f"P(spoiled) {p[c]:.2f}", fontsize=7.5, color=viz.TEXT_LIGHT, labelpad=2)
            _clean(axes[row, c])
        axes[row, 0].set_ylabel(dict(RUNS)[run], fontsize=10, color=viz.TEXT)
    axes[0, 0].set_ylabel("Image", fontsize=10, color=viz.TEXT)
    fig.suptitle(
        f"Grad-CAM of the freshness decision - {state} test images (bright = strong evidence for the predicted class)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="semibold",
        color=viz.TEXT,
    )
    fig.tight_layout()
    viz.save(fig, path)


def figure_errors(models, preds, path, per_model=6):
    import matplotlib.pyplot as plt

    viz.apply_style()
    fig, axes = plt.subplots(2 * len(models), per_model, figsize=(1.75 * per_model, 2.0 * 2 * len(models)))
    for ax in axes.ravel():
        ax.axis("off")
    for m, (run, model) in enumerate(models.items()):
        p = preds[run]
        wrong = p[p["freshness_pred"] != p["freshness_label"]]
        if wrong.empty:
            continue
        wrong = wrong.sample(min(per_model, len(wrong)), random_state=SEED)
        images = np.stack([load_image(PROJECT_ROOT / path_) for path_ in wrong["path"]])
        heat = gradcam(model, images, "freshness")
        for c, r in enumerate(wrong.itertuples()):
            true = "spoiled" if r.freshness_label else "fresh"
            axes[2 * m, c].imshow(images[c].astype(np.uint8))
            axes[2 * m, c].set_title(
                f"{FRUITS[r.fruit_label]}, true {true}\nP(spoiled) {r.spoiled_prob:.2f}",
                fontsize=7.5,
                color=viz.TEXT_LIGHT,
            )
            axes[2 * m + 1, c].imshow(overlay(images[c], heat[c]))
        axes[2 * m, 0].text(
            -0.1,
            0.5,
            dict(RUNS)[run],
            transform=axes[2 * m, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=10,
            color=viz.TEXT,
        )
    fig.suptitle(
        "Freshness errors on the test set and their Grad-CAM heatmaps",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="semibold",
        color=viz.TEXT,
    )
    fig.tight_layout()
    viz.save(fig, path)


def figure_heads(model, preds, path, n=8):
    import matplotlib.pyplot as plt

    viz.apply_style()
    p = preds.sample(n=n, random_state=SEED)
    images = np.stack([load_image(PROJECT_ROOT / path_) for path_ in p["path"]])
    rows = [
        ("Image", None),
        ("Fruit head", gradcam(model, images, "fruit")),
        ("Freshness head", gradcam(model, images, "freshness")),
    ]
    fig, axes = plt.subplots(3, n, figsize=(1.75 * n, 6.0))
    for r, (label, heat) in enumerate(rows):
        for c in range(n):
            axes[r, c].imshow(images[c].astype(np.uint8) if heat is None else overlay(images[c], heat[c]))
            _clean(axes[r, c])
        axes[r, 0].set_ylabel(label, fontsize=10, color=viz.TEXT)
    fig.suptitle(
        "Model 3: what the fruit head and the freshness head look at",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="semibold",
        color=viz.TEXT,
    )
    fig.tight_layout()
    viz.save(fig, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ready = [
        run
        for run, _ in RUNS
        if (MODELS_DIR / f"{run}.keras").exists() and (PREDICTIONS_DIR / f"{run}_test.csv").exists()
    ]  # evaluated by scripts/evaluate.py
    models = {run: keras.models.load_model(MODELS_DIR / f"{run}.keras") for run in ready}
    preds = {run: pd.read_csv(PREDICTIONS_DIR / f"{run}_test.csv") for run in ready}
    print("models:", ready)
    figure_examples(models, preds, "fresh", OUT / "gradcam_fresh.png")
    figure_examples(models, preds, "spoiled", OUT / "gradcam_spoiled.png")
    figure_errors(models, preds, OUT / "gradcam_errors.png")
    if "model3_mobilenet_v2" in models:
        figure_heads(models["model3_mobilenet_v2"], preds["model3_mobilenet_v2"], OUT / "gradcam_heads_model3.png")
    print("Grad-CAM figures written to", OUT)


if __name__ == "__main__":
    main()
