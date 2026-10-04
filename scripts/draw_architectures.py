"""Architecture diagrams of the three models.

python scripts/draw_architectures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src import viz  # noqa: E402
from src.config import FIGURES_DIR  # noqa: E402

STYLE = {  # box colour, text colour
    "io": (viz.BG, viz.TEXT),
    "pre": ("#f0efec", viz.TEXT),
    "conv": ("#b7d3f6", viz.TEXT),
    "pool": ("#3987e5", "white"),
    "fc": ("#ffffff", viz.TEXT),
    "drop": ("#f0efec", viz.TEXT),
    "res": ("#86b6ef", viz.TEXT),
    "frozen": ("#e1e0d9", viz.TEXT_LIGHT),
    "tuned": ("#2a78d6", "white"),
    "fruit": (viz.COLORS[0], "white"),
    "fresh": (viz.COLORS[1], "white"),
    "gap": ("#cde2fb", viz.TEXT),
}


def box(ax, x, y, w, h, text, kind, rotate=True, fontsize=8.5):
    fill, ink = STYLE[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", facecolor=fill, edgecolor=viz.AXIS, linewidth=0.9
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        rotation=90 if rotate else 0,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=ink,
        linespacing=1.15,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=viz.GREY, lw=1.1, shrinkA=0, shrinkB=0, mutation_scale=9),
    )


def chain(ax, items, x0, y, h, w=0.62, gap=0.28, shapes=None):
    """Draw boxes left to right; returns the x coordinate of the right edge."""
    x = x0
    for i, (text, kind) in enumerate(items):
        box(ax, x, y, w, h, text, kind)
        if shapes and shapes[i]:
            ax.text(
                x + w / 2, y - 0.16, shapes[i], ha="center", va="top", fontsize=7, color=viz.TEXT_LIGHT, rotation=90
            )
        if i < len(items) - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
        x += w + gap
    return x - gap


def finish(fig, ax, title, path, xmax, ymin, ymax):
    ax.set_xlim(-0.2, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, pad=6)
    viz.save(fig, path)


def heads(ax, x, y_mid):
    """Fruit head (top) and freshness head (bottom) starting at (x, y_mid)."""
    h, w, gap = 1.9, 0.72, 0.28
    fork = x + 0.3
    ax.plot([x, fork], [y_mid, y_mid], color=viz.GREY, lw=1.1)
    for y, items, label in (
        (
            y_mid + 0.6,
            [("Dense 128\nReLU", "fc"), ("Dropout 0.3", "drop"), ("Dense 8\nsoftmax", "fruit")],
            "FRUIT: P(8 fruit classes)",
        ),
        (
            y_mid - 0.6 - h,
            [("Dense 64\nReLU", "fc"), ("Dropout 0.3", "drop"), ("Dense 1\nsigmoid", "fresh")],
            "FRESHNESS: P(spoiled)",
        ),
    ):
        ax.plot([fork, fork], [y_mid, y + h / 2], color=viz.GREY, lw=1.1)
        arrow(ax, fork, y + h / 2, fork + 0.3, y + h / 2)
        right = chain(ax, items, fork + 0.3, y, h, w=w, gap=gap)
        ax.text(
            right - w / 2,
            y + h + 0.15 if y > y_mid else y - 0.2,
            label,
            fontsize=8,
            color=viz.TEXT,
            ha="center",
            va="bottom" if y > y_mid else "top",
            fontweight="semibold",
        )
    return fork + 0.3 + 3 * w + 2 * gap


def model1():
    fig, ax = plt.subplots(figsize=(15, 4.2))
    items, shapes = [("Input\n224x224x3", "io"), ("Rescaling\n1/255", "pre")], ["", ""]
    for f, s in [(32, 112), (64, 56), (128, 28), (128, 14), (256, 7)]:
        items += [(f"Conv 3x3 ({f})\nReLU", "conv"), ("MaxPool 2x2", "pool")]
        shapes += ["", f"{s}x{s}x{f}"]
    items += [("Flatten", "pre"), ("Dense 256\nReLU", "fc"), ("Dropout 0.5", "drop"), ("Dense 16\nsoftmax", "fruit")]
    shapes += ["12,544", "256", "", "16 classes"]
    right = chain(ax, items, 0, 0, 2.6, shapes=shapes)
    finish(
        fig,
        ax,
        "Model 1 - simple sequential CNN (3.75 M parameters): 5 x [Conv-ReLU-MaxPool] + fully connected layers",
        FIGURES_DIR / "model_1_architecture.png",
        right + 0.3,
        -1.6,
        2.9,
    )


def model2():
    fig, ax = plt.subplots(figsize=(15, 6.4))
    items = [
        ("Input\n224x224x3", "io"),
        ("Rescaling\n1/255", "pre"),
        ("Conv 3x3 /2 (32)\nBN - ReLU", "conv"),
        ("MaxPool 2x2", "pool"),
        ("ResBlock\n64", "res"),
        ("ResBlock\n128, /2", "res"),
        ("ResBlock\n256, /2", "res"),
        ("ResBlock\n512, /2", "res"),
        ("Global\nAvgPool", "gap"),
    ]
    shapes = ["", "", "112x112x32", "56x56x32", "56x56x64", "28x28x128", "14x14x256", "7x7x512", "512"]
    right = chain(ax, items, 0, 0.4, 2.6, w=0.7, shapes=shapes)
    end = heads(ax, right, 0.4 + 1.3)
    ax.text(right - 4.0, 3.55, "shared trunk (residual CNN blocks)", fontsize=9, color=viz.TEXT_LIGHT, ha="center")
    # inset: residual block (horizontal boxes, shortcut drawn underneath)
    y0, x0, bh = -3.6, 0.0, 1.0
    ax.text(x0, y0 + bh + 0.35, "ResBlock(f, s)", fontsize=9.5, color=viz.TEXT, fontweight="semibold")
    x = x0
    for i, (text, kind, w) in enumerate(
        [
            ("x", "io", 0.6),
            ("Conv 3x3 /s", "conv", 1.5),
            ("BN", "pre", 0.6),
            ("ReLU", "pre", 0.8),
            ("Conv 3x3", "conv", 1.3),
            ("BN", "pre", 0.6),
            ("+", "io", 0.6),
            ("ReLU", "pre", 0.8),
        ]
    ):
        box(ax, x, y0, w, bh, text, kind, rotate=False, fontsize=11 if text == "+" else 8.5)
        if i == 0:
            x_in = x + w / 2
        if text == "+":
            x_add = x + w / 2
        if i < 7:
            arrow(ax, x + w, y0 + bh / 2, x + w + 0.25, y0 + bh / 2)
        x += w + 0.25
    y_sc = y0 - 0.45
    ax.plot([x_in, x_in, x_add], [y0, y_sc, y_sc], color=viz.GREY, lw=1.1)
    arrow(ax, x_add, y_sc, x_add, y0)
    ax.text(
        (x_in + x_add) / 2,
        y_sc - 0.12,
        "shortcut: identity, or Conv 1x1 /s + BN when the shape changes",
        fontsize=8,
        color=viz.TEXT_LIGHT,
        ha="center",
        va="top",
    )
    finish(
        fig,
        ax,
        "Model 2 - multi-task CNN (Functional API, 4.99 M parameters): residual trunk shared by two heads",
        FIGURES_DIR / "model_2_architecture.png",
        end + 0.3,
        -4.8,
        6.2,
    )


def model3():
    fig, ax = plt.subplots(figsize=(15, 6.0))
    stages = [
        ("Conv 3x3 /2\n32", "frozen"),
        ("Block 0\n16", "frozen"),
        ("Blocks 1-2\n24, /2", "frozen"),
        ("Blocks 3-5\n32, /2", "frozen"),
        ("Blocks 6-9\n64, /2", "frozen"),
        ("Blocks 10-12\n96", "frozen"),
        ("Blocks 13-15\n160, /2", "tuned"),
        ("Block 16\n320", "tuned"),
        ("Conv 1x1\n1280", "tuned"),
    ]
    shapes = ["112x112", "112x112", "56x56", "28x28", "14x14", "14x14", "7x7", "7x7", "7x7x1280"]
    items = [("Input\n224x224x3", "io"), ("Rescaling\nx/127.5 - 1", "pre")] + stages + [("Global\nAvgPool", "gap")]
    right = chain(ax, items, 0, 0.4, 2.6, w=0.7, gap=0.25, shapes=["", ""] + shapes + ["1280"])
    end = heads(ax, right, 0.4 + 1.3)
    x_frozen0, step = 2 * 0.95, 0.95
    for x1, x2, label, color in (
        (x_frozen0, x_frozen0 + 6 * step - 0.25, "frozen in both stages (ImageNet weights)", viz.GREY),
        (x_frozen0 + 6 * step, x_frozen0 + 9 * step - 0.25, "fine-tuned in stage 2", viz.COLORS[0]),
    ):
        ax.plot([x1, x1, x2, x2], [3.25, 3.4, 3.4, 3.25], color=color, lw=1.2)
        ax.text((x1 + x2) / 2, 3.55, label, ha="center", fontsize=8.5, color=viz.TEXT_LIGHT)
    ax.text(
        x_frozen0 + 4.2,
        -2.35,
        "MobileNetV2 backbone (include_top=False): inverted residual blocks with depthwise convolutions; "
        "batch-norm layers stay frozen",
        ha="center",
        fontsize=8,
        color=viz.TEXT_LIGHT,
    )
    finish(
        fig,
        ax,
        "Model 3 - transfer learning: MobileNetV2 (ImageNet) + the same two heads; stage 1 trains the heads, "
        "stage 2 also fine-tunes blocks 13-16",
        FIGURES_DIR / "model_3_architecture.png",
        end + 0.3,
        -2.8,
        5.4,
    )


if __name__ == "__main__":
    viz.apply_style()
    model1()
    model2()
    model3()
    print("architecture diagrams written to", FIGURES_DIR)
