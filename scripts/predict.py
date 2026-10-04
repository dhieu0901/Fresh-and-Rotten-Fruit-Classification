"""Predict fruit type and freshness for new images with the trained models.

python scripts/predict.py demo/demo_images/*.jpg
python scripts/predict.py photo.jpg --models model3_mobilenet_v2 --gradcam figures/demo_gradcam.png
"""

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import numpy as np  # noqa: E402

from src.config import FRUITS, MODELS_DIR  # noqa: E402
from src.data_pipeline import load_image  # noqa: E402
from src.evaluation.metrics import decode_outputs  # noqa: E402

DEFAULT_MODELS = ["model1_simple_cnn", "model2_multitask_cnn", "model3_mobilenet_v2"]


def main():
    parser = argparse.ArgumentParser(description="Predict fruit and freshness for images")
    parser.add_argument("images", nargs="+", help="image files (wildcards allowed)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--gradcam", help="save a figure with Grad-CAM heatmaps of the last model to this path")
    args = parser.parse_args()

    paths = [p for pattern in args.images for p in sorted(glob.glob(pattern))]
    if not paths:
        sys.exit("no images found")
    images = np.stack([load_image(p) for p in paths])
    models = {
        name: keras.models.load_model(MODELS_DIR / f"{name}.keras")
        for name in args.models
        if (MODELS_DIR / f"{name}.keras").exists()
    }
    if not models:
        sys.exit(f"no trained models found in {MODELS_DIR}")

    results = {name: decode_outputs(model.predict(images, verbose=0)) for name, model in models.items()}
    for i, path in enumerate(paths):
        print(f"\n{Path(path).name}")
        for name, r in results.items():
            fruit = FRUITS[r["fruit_pred"][i]]
            state = "SPOILED" if r["fresh_pred"][i] else "fresh"
            print(
                f"  {name:22s} {fruit:10s} (p={r['fruit_prob'][i].max():.2f})   {state:7s} (P(spoiled)={r['spoiled_prob'][i]:.2f})"
            )

    if args.gradcam:
        import matplotlib.pyplot as plt

        from src import viz
        from src.evaluation.gradcam import gradcam, overlay

        viz.apply_style()
        name, model = list(models.items())[-1]
        heat = gradcam(model, images, "freshness")
        r = results[name]
        fig, axes = plt.subplots(2, len(paths), figsize=(2.2 * len(paths), 4.8), squeeze=False)
        for i in range(len(paths)):
            axes[0, i].imshow(images[i].astype(np.uint8))
            axes[0, i].set_title(
                f"{FRUITS[r['fruit_pred'][i]]}\n{'spoiled' if r['fresh_pred'][i] else 'fresh'} "
                f"({r['spoiled_prob'][i]:.2f})",
                fontsize=8,
            )
            axes[1, i].imshow(overlay(images[i], heat[i]))
            for ax in axes[:, i]:
                ax.axis("off")
        fig.suptitle(f"{name}: prediction and Grad-CAM of the freshness decision", fontsize=10, x=0.01, ha="left")
        fig.tight_layout()
        viz.save(fig, Path(args.gradcam))
        print(f"\nGrad-CAM figure saved to {args.gradcam}")


if __name__ == "__main__":
    main()
