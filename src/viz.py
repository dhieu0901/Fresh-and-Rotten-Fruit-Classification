"""Matplotlib settings shared by all figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_LIGHT = "#52514e"
GREY = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "text.color": TEXT,
            "axes.labelcolor": TEXT_LIGHT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "xtick.labelcolor": TEXT_LIGHT,
            "ytick.labelcolor": TEXT_LIGHT,
            "legend.frameon": False,
            "legend.labelcolor": TEXT_LIGHT,
            "lines.linewidth": 2,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
