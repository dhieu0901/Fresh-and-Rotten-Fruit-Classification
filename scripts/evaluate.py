"""Evaluate the trained models on the test split and make the result tables and figures.

python scripts/evaluate.py                     # all finished runs (run it with an idle CPU for the timings)
python scripts/evaluate.py --runs model3_mobilenet_v2 --no-timing
python scripts/evaluate.py --tables-only       # rebuild tables and figures from reports/final_metrics.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import viz  # noqa: E402
from src.config import (  # noqa: E402
    COMBINED_NAMES,
    FIGURES_DIR,
    FRUITS,
    IMG_SIZE,
    LOGS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TABLES_DIR,
)
from src.data_pipeline import load_split, make_dataset  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_metrics,
    confusion_matrices,
    measure_inference_time,
    per_class_report,
    predict,
)
from src.models import label_mode_of  # noqa: E402

MAIN = [
    ("model1_simple_cnn", "Model 1 - Simple CNN (16 classes)"),
    ("model2_multitask_cnn", "Model 2 - Multi-task residual CNN"),
    ("model3_mobilenet_v2", "Model 3 - MobileNetV2 fine-tuned"),
]
AUG_PAIRS = [(run, f"{run}_noaug", label) for run, label in MAIN]
FINETUNE = [
    ("model3_ft_none", "none (frozen)"),
    ("model3_ft_block16", "block 16"),
    ("model3_mobilenet_v2", "block 13 (main)"),
    ("model3_ft_block10", "block 10"),
    ("model3_ft_block6", "block 6"),
    ("model3_ft_all", "all layers"),
]
SHORT = {"model1_simple_cnn": "Model 1", "model2_multitask_cnn": "Model 2", "model3_mobilenet_v2": "Model 3"}
METRIC_COLS = [
    "joint_accuracy",
    "joint_precision",
    "joint_recall",
    "joint_f1",
    "fruit_accuracy",
    "fruit_f1",
    "freshness_accuracy",
    "freshness_precision",
    "freshness_recall",
    "freshness_f1",
    "freshness_auc",
]


def finished_runs():
    log = pd.read_csv(REPORTS_DIR / "experiment_log.csv")
    return [r for r in log["run"] if (MODELS_DIR / f"{r}.keras").exists()], log.set_index("run")


def save_predictions(run, df, pred):
    out = df[["path", "fruit_label", "freshness_label", "combined_label"]].copy()
    out["fruit_pred"], out["freshness_pred"], out["joint_pred"] = (
        pred["fruit_pred"],
        pred["fresh_pred"],
        pred["joint_pred"],
    )
    out["spoiled_prob"] = pred["spoiled_prob"].round(5)
    for i, fruit in enumerate(FRUITS):
        out[f"p_{fruit}"] = pred["fruit_prob"][:, i].round(5)
    path = PREDICTIONS_DIR / f"{run}_test.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def plot_confusion(cm, labels, title, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    viz.apply_style()
    n = len(labels)
    size = {2: 3.6, 8: 6.2, 16: 9.8}.get(n, 6)
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    rate = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("seq", [viz.BG] + viz.BLUES)
    ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7.5 if n > 8 else 9,
                    color="white" if rate[i, j] > 0.55 else viz.TEXT,
                )
    ticks = [name.replace("_", " ") for name in labels]
    ax.set_xticks(range(n), ticks, rotation=90 if n > 2 else 0)
    ax.set_yticks(range(n), ticks)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    viz.save(fig, path)


def plot_learning_curves(run, path):
    import matplotlib.pyplot as plt

    viz.apply_style()
    files = sorted((LOGS_DIR / run).glob("history*.csv"))
    if not files:
        return
    hist = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    hist["epoch"] = hist["epoch"] + 1
    stage2 = LOGS_DIR / run / "history_stage2.csv"
    split_epoch = (
        pd.read_csv(stage2)["epoch"].min() + 0.5
        if stage2.exists() and (LOGS_DIR / run / "history_stage1.csv").exists()
        else None
    )
    panels = [("loss", "Loss")]
    panels += (
        [("accuracy", "Accuracy (16 classes)")]
        if "accuracy" in hist
        else [("fruit_accuracy", "Fruit accuracy"), ("freshness_accuracy", "Freshness accuracy")]
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6))
    for ax, (key, title) in zip(np.atleast_1d(axes), panels):
        ax.plot(hist["epoch"], hist[key], color=viz.COLORS[0], label="Train")
        ax.plot(hist["epoch"], hist[f"val_{key}"], color=viz.COLORS[1], label="Validation")
        if split_epoch:
            ax.axvline(split_epoch, color=viz.GREY, lw=1)
            ax.text(
                split_epoch + 0.3,
                0.97,
                "fine-tuning",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                color=viz.TEXT_LIGHT,
                va="top",
            )
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Epoch")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(run, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=viz.TEXT)
    fig.tight_layout()
    viz.save(fig, path)


# The comparison figures plot error rates from 0: all accuracies are close to 100 %, and accuracy bars
# on a cut axis would exaggerate the differences.
def plot_main_comparison(table, path):
    import matplotlib.pyplot as plt

    viz.apply_style()
    metrics = [
        ("joint_accuracy", "Joint (fruit and freshness)"),
        ("fruit_accuracy", "Fruit type"),
        ("freshness_accuracy", "Freshness"),
    ]
    runs = [(run, label) for run, label in MAIN if run in table.index]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    width = 0.26
    x = np.arange(len(metrics))
    errors = {run: [100 * (1 - table.loc[run, m]) for m, _ in metrics] for run, _ in runs}
    top = max(max(e) for e in errors.values())
    for k, (run, label) in enumerate(runs):
        bars = ax.bar(x + (k - (len(runs) - 1) / 2) * width, errors[run], width * 0.9, color=viz.COLORS[k], label=label)
        for b, v in zip(bars, errors[run]):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + top * 0.015,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=viz.TEXT_LIGHT,
            )
    ax.set_ylim(0, top * 1.2)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Test error rate (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_title("Test error rate of the three models (lower is better)", loc="left")
    viz.save(fig, path)


def plot_augmentation(table, path):
    import matplotlib.pyplot as plt

    viz.apply_style()
    rows = [
        (SHORT[a], 100 * (1 - table.loc[a, "joint_accuracy"]), 100 * (1 - table.loc[b, "joint_accuracy"]))
        for a, b, _ in AUG_PAIRS
        if a in table.index and b in table.index
    ]
    if not rows:
        return
    top = max(max(r[1], r[2]) for r in rows)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.arange(len(rows))
    for k, (label, idx) in enumerate([("With augmentation", 1), ("Without augmentation", 2)]):
        vals = [r[idx] for r in rows]
        bars = ax.bar(x + (k - 0.5) * 0.34, vals, 0.3, color=viz.COLORS[k], label=label)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + top * 0.015,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=viz.TEXT_LIGHT,
            )
    ax.set_ylim(0, top * 1.2)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("Joint test error rate (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_title("Ablation: data augmentation (lower is better)", loc="left")
    viz.save(fig, path)


def plot_finetuning(ft, path):
    import matplotlib.pyplot as plt

    viz.apply_style()
    if ft.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(ft))
    series = [("val_joint_accuracy", "Validation"), ("joint_accuracy", "Test")]
    top = max(100 * (1 - ft[col]).max() for col, _ in series)
    for k, (col, label) in enumerate(series):
        err = 100 * (1 - ft[col])
        ax.plot(
            x,
            err,
            color=viz.COLORS[k],
            marker="o",
            markersize=7,
            markeredgecolor=viz.BG,
            markeredgewidth=1.5,
            label=label,
        )
    for xi, (_, r) in zip(x, ft.iterrows()):  # test errors as image counts
        ax.annotate(
            f"{int(r['test_errors'])} img",
            (xi, 100 * (1 - r["joint_accuracy"])),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=viz.TEXT_LIGHT,
        )
    ax.set_ylim(0, top * 1.2)
    ax.set_xticks(x, [f"{r.setting}\n{int(r.unfrozen_layers)} layers" for r in ft.itertuples()], fontsize=8.5)
    ax.set_xlabel("MobileNetV2 unfrozen from")
    ax.set_ylabel("Joint error rate (%)")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_title("Ablation: number of fine-tuned layers, Model 3 (lower is better)", loc="left")
    viz.save(fig, path)


def plot_misclassified(run, preds, path, n=16):
    import matplotlib.pyplot as plt
    from PIL import Image

    viz.apply_style()
    wrong = preds[preds["joint_pred"] != preds["combined_label"]]
    if wrong.empty:
        return
    fresh_err = wrong[wrong["freshness_pred"] != wrong["freshness_label"]]
    fruit_err = wrong[wrong["fruit_pred"] != wrong["fruit_label"]]
    picks = (
        pd.concat([fresh_err.head(n // 2), fruit_err.head(n - min(len(fresh_err), n // 2))])
        .drop_duplicates("path")
        .head(n)
    )
    cols = 8
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.25))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, r in zip(np.atleast_1d(axes).ravel(), picks.itertuples()):
        ax.imshow(Image.open(PROJECT_ROOT / r.path))
        ax.set_title(
            f"true {COMBINED_NAMES[r.combined_label].replace('_', ' ')}\npred {COMBINED_NAMES[r.joint_pred].replace('_', ' ')}",
            fontsize=7,
            color=viz.TEXT_LIGHT,
            fontweight="normal",
        )
    fig.suptitle(
        f"Misclassified test images - {run} ({len(wrong)} of {len(preds)} wrong)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="semibold",
        color=viz.TEXT,
    )
    fig.tight_layout()
    viz.save(fig, path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test split")
    parser.add_argument("--runs", nargs="*", help="run names (default: every finished run)")
    parser.add_argument("--no-timing", action="store_true", help="skip inference-time measurement")
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="rebuild the tables and comparison figures from reports/final_metrics.csv (no inference)",
    )
    args = parser.parse_args()
    if args.tables_only:
        return build_outputs(pd.read_csv(REPORTS_DIR / "final_metrics.csv").set_index("run"))
    runs, log = finished_runs()
    runs = args.runs or runs
    test_df = load_split("test")
    rows, preds_by_run = [], {}
    for run in runs:
        model = keras.models.load_model(MODELS_DIR / f"{run}.keras")
        mode = label_mode_of(log.loc[run, "model"])
        pred = predict(model, make_dataset("test", mode, batch_size=64))
        metrics = compute_metrics(test_df, pred)
        preds_by_run[run] = save_predictions(run, test_df, pred)
        per_class_report(test_df, pred).round(4).to_csv(_mkdir(REPORTS_DIR / "per_class") / f"{run}.csv", index=False)
        val = json.loads((LOGS_DIR / run / "val_metrics.json").read_text())
        row = {
            "run": run,
            **metrics,
            **{f"val_{k}": v for k, v in val.items()},
            "joint_errors": int((preds_by_run[run]["joint_pred"] != preds_by_run[run]["combined_label"]).sum()),
            "params": model.count_params(),
            "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
            "train_minutes": log.loc[run, "train_minutes"],
        }
        if not args.no_timing:
            row["infer_ms_bs1"] = measure_inference_time(model, IMG_SIZE, batch_size=1)
            row["infer_ms_bs32"] = measure_inference_time(model, IMG_SIZE, batch_size=32, n_runs=10)
        for key in ("unfrozen_layers", "fine_tune_from", "augmentation"):
            if key in log.columns:
                row[key] = log.loc[run, key]
        rows.append(row)
        if run in dict(MAIN):
            for view, (cm, labels) in confusion_matrices(test_df, pred).items():
                plot_confusion(
                    cm,
                    labels,
                    f"{SHORT[run]} - {view} (test, counts; colour = row share)",
                    FIGURES_DIR / "confusion_matrices" / f"{run}_{view}.png",
                )
        plot_learning_curves(run, FIGURES_DIR / "learning_curves" / f"{run}.png")
        print(
            f"{run:32s} joint acc {metrics['joint_accuracy']:.4f}  joint F1 {metrics['joint_f1']:.4f}  "
            f"fruit acc {metrics['fruit_accuracy']:.4f}  fresh F1 {metrics['freshness_f1']:.4f}",
            flush=True,
        )
        keras.backend.clear_session()

    table = pd.DataFrame(rows).set_index("run")
    old = REPORTS_DIR / "final_metrics.csv"
    if args.runs and old.exists():  # update only the evaluated rows
        prev = pd.read_csv(old).set_index("run")
        table = pd.concat([prev.drop(index=[r for r in table.index if r in prev.index]), table])
    table.to_csv(old)  # not rounded, so --tables-only does not round twice
    build_outputs(table)


def build_outputs(table):
    """Metric tables (csv / xlsx / markdown) and comparison figures from the per-run test metrics."""
    table[[c for c in table.columns if c.startswith("fruit_")]].round(5).to_csv(REPORTS_DIR / "fruit_metrics.csv")
    table[[c for c in table.columns if c.startswith("freshness_")]].round(5).to_csv(
        REPORTS_DIR / "freshness_metrics.csv"
    )
    write_tables(table)
    plot_main_comparison(table, FIGURES_DIR / "model_comparison.png")
    plot_augmentation(table, FIGURES_DIR / "ablation_augmentation.png")
    plot_finetuning(finetune_table(table), FIGURES_DIR / "ablation_finetuning.png")
    best = max((r for r, _ in MAIN if r in table.index), key=lambda r: table.loc[r, "val_joint_accuracy"], default=None)
    if best and (PREDICTIONS_DIR / f"{best}_test.csv").exists():
        plot_misclassified(
            best, pd.read_csv(PREDICTIONS_DIR / f"{best}_test.csv"), FIGURES_DIR / "misclassified_samples.png"
        )


def finetune_table(table):
    # training cost from model_report.py (same conditions for every depth); the train_minutes of the
    # runs are not comparable because the machine load was different
    cost_path = REPORTS_DIR / "training_cost.csv"
    cost = pd.read_csv(cost_path).set_index("run")["train_s_per_step"] if cost_path.exists() else pd.Series(dtype=float)
    rows = []
    for run, setting in FINETUNE:
        if run in table.index:
            r = table.loc[run]
            rows.append(
                {
                    "setting": setting,
                    "run": run,
                    "unfrozen_layers": 0 if run == "model3_ft_none" else r.get("unfrozen_layers", np.nan),
                    "trainable_params": r["trainable_params"],
                    "val_joint_accuracy": r["val_joint_accuracy"],
                    "joint_accuracy": r["joint_accuracy"],
                    "joint_f1": r["joint_f1"],
                    "freshness_f1": r["freshness_f1"],
                    "test_errors": r.get("joint_errors", np.nan),
                    "train_s_per_step": cost.get(run, np.nan),
                }
            )
    return pd.DataFrame(rows)


def write_tables(table):
    main = pd.DataFrame(
        [
            {
                "Model": label,
                "Accuracy": table.loc[r, "joint_accuracy"],
                "Precision": table.loc[r, "joint_precision"],
                "Recall": table.loc[r, "joint_recall"],
                "F1-score": table.loc[r, "joint_f1"],
                "Fruit acc.": table.loc[r, "fruit_accuracy"],
                "Freshness F1": table.loc[r, "freshness_f1"],
                "Freshness AUC": table.loc[r, "freshness_auc"],
                "Params": int(table.loc[r, "params"]),
                "ms/img (CPU, bs 1)": table.loc[r].get("infer_ms_bs1", np.nan),
            }
            for r, label in MAIN
            if r in table.index
        ]
    )
    aug = pd.DataFrame(
        [
            {
                "Model": label,
                "Joint acc. (aug)": table.loc[a, "joint_accuracy"],
                "Joint acc. (no aug)": table.loc[b, "joint_accuracy"],
                "Joint F1 (aug)": table.loc[a, "joint_f1"],
                "Joint F1 (no aug)": table.loc[b, "joint_f1"],
                "Errors (aug)": table.loc[a].get("joint_errors", np.nan),
                "Errors (no aug)": table.loc[b].get("joint_errors", np.nan),
            }
            for a, b, label in AUG_PAIRS
            if a in table.index and b in table.index
        ]
    )
    ft = (
        finetune_table(table)
        .drop(columns="run")
        .rename(
            columns={
                "setting": "Unfrozen from",
                "unfrozen_layers": "Unfrozen layers",
                "trainable_params": "Trainable params",
                "val_joint_accuracy": "Val joint acc.",
                "joint_accuracy": "Test joint acc.",
                "joint_f1": "Test joint F1",
                "freshness_f1": "Test freshness F1",
                "test_errors": "Test errors",
                "train_s_per_step": "Train s/step (CPU)",
            }
        )
    )
    path = TABLES_DIR / "model_comparison.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as xl:
        main.round(4).to_excel(xl, sheet_name="main_comparison", index=False)
        aug.round(4).to_excel(xl, sheet_name="ablation_augmentation", index=False)
        ft.round(4).to_excel(xl, sheet_name="ablation_finetuning", index=False)
        table.round(5).to_excel(xl, sheet_name="all_runs_test_metrics")
    md = [
        "# Results (test split, 2,203 images)",
        "",
        "Joint = fruit and freshness both correct (16 classes, macro precision/recall/F1). "
        "Freshness metrics use spoiled as the positive class.",
        "",
        "## Main comparison",
        "",
        md_table(main),
        "",
        "## Ablation: data augmentation",
        "",
        md_table(aug) if len(aug) else "(not run yet)",
        "",
        "## Ablation: fine-tuning depth (Model 3)",
        "",
        md_table(ft) if len(ft) else "(not run yet)",
        "",
    ]
    (REPORTS_DIR / "results_summary.md").write_text("\n".join(md), encoding="utf-8")


def md_table(df):
    def fmt(v, col):
        if not isinstance(v, (int, float, np.integer, np.floating)) or isinstance(v, bool):
            return str(v)
        if pd.isna(v):
            return "-"
        if col in counts:
            return f"{int(v):,}"
        if "s/step" in col:
            return f"{v:.3f}"
        if col.startswith("ms/"):
            return f"{v:.2f}"
        return f"{v:.4f}"

    # counts (parameters, layers, errors) are printed as integers
    counts = {
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and df[c].dropna().apply(lambda x: float(x).is_integer()).all()
        and df[c].max() > 1
    }
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += [
        "| " + " | ".join(fmt(v, c) for v, c in zip(row, df.columns)) + " |" for row in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()
