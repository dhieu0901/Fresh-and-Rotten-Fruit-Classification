"""Model summaries, parameter counts and CPU cost of the three models.

Usage:
    python scripts/model_report.py              # all models
    python scripts/model_report.py --steps 10   # fewer timing steps (quicker)

Writes
    reports/model_summaries/<model>.txt   Keras summary (layers, output shapes, parameters)
    reports/model_parameters.csv          total / trainable / non-trainable parameters
    reports/training_cost.csv             the same CPU costs per experiment run, incl. every fine-tuning depth of Model 3
    reports/resource_report.md            training time per step, estimated epoch time, inference latency

All variants are timed under the same conditions (idle CPU, same batches), so their costs can be
compared - unlike the wall-clock training times in reports/experiment_log.csv.
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.config import IMG_SIZE, REPORTS_DIR, load_json  # noqa: E402
from src.data_pipeline import load_split, make_dataset  # noqa: E402
from src.evaluation.metrics import measure_inference_time  # noqa: E402
from src.models import build_model, compile_model, label_mode_of, set_backbone_trainable  # noqa: E402
from src.training.cpu import use_all_cores  # noqa: E402

VARIANTS = [  # (report name, model name, transfer fine-tuning point, experiment run, write a summary file)
    ("model1_simple_cnn", "simple_cnn", None, "model1_simple_cnn", True),
    ("model2_multitask_cnn", "multitask_cnn", None, "model2_multitask_cnn", True),
    ("model3_mobilenet_v2_frozen", "transfer", None, "model3_ft_none", True),
    ("model3_ft_block16", "transfer", "block_16_expand", "model3_ft_block16", False),
    ("model3_mobilenet_v2_finetune", "transfer", "block_13_expand", "model3_mobilenet_v2", True),
    ("model3_ft_block10", "transfer", "block_10_expand", "model3_ft_block10", False),
    ("model3_ft_block6", "transfer", "block_6_expand", "model3_ft_block6", False),
    ("model3_ft_all", "transfer", "all", "model3_ft_all", False),
]


def cpu_name():
    if sys.platform == "win32":  # platform.processor() only gives the CPU family on Windows
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def count_params(model):
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return trainable + non_trainable, trainable, non_trainable


def time_training(model, label_mode, batch_size, steps):
    """Seconds per training step (batch) on real, augmented training data; first epoch = warm-up."""
    ds = make_dataset("train", label_mode=label_mode, batch_size=batch_size, augment=True,
                      limit=batch_size * steps // 16 + 1).take(steps).repeat()
    times = []

    class Timer(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.t = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            times.append(time.perf_counter() - self.t)

    model.fit(ds, epochs=2, steps_per_epoch=steps, verbose=0, callbacks=[Timer()])
    return times[-1] / steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20, help="timed training steps per model")
    parser.add_argument("--no-timing", action="store_true", help="only summaries and parameter counts (CPU busy)")
    args = parser.parse_args()
    use_all_cores()
    cfg = load_json("training_config.json")
    batch_size = cfg["common"]["batch_size"]
    n_train, n_val = len(load_split("train")), len(load_split("val"))
    out_dir = REPORTS_DIR / "model_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for report_name, name, fine_tune_from, run, write_summary in VARIANTS:
        if args.no_timing and not write_summary:
            continue
        keras.utils.set_random_seed(0)
        model = build_model(name)
        if name == "transfer":
            set_backbone_trainable(model, fine_tune_from)
        mode = label_mode_of(name)
        compile_model(model, mode, 1e-3)
        if write_summary:
            lines = []
            model.summary(print_fn=lambda s, line_break=None: lines.append(s), expand_nested=False, show_trainable=True)
            (out_dir / f"{report_name}.txt").write_text("\n".join(lines), encoding="utf-8")
        total, trainable, non_trainable = count_params(model)
        if args.no_timing:
            rows.append({"model": report_name, "run": run, "total_params": total, "trainable_params": trainable,
                         "non_trainable_params": non_trainable})
            print(rows[-1], flush=True)
            keras.backend.clear_session()
            continue

        step = time_training(model, mode, batch_size, args.steps)
        steps_per_epoch = int(np.ceil(n_train / batch_size))
        val_batches = int(np.ceil(n_val / batch_size))
        lat1 = measure_inference_time(model, IMG_SIZE, batch_size=1)
        lat32 = measure_inference_time(model, IMG_SIZE, batch_size=32, n_runs=10)
        epoch_est = step * steps_per_epoch + val_batches * lat32 * 32 / 1000
        rows.append({"model": report_name, "run": run, "total_params": total, "trainable_params": trainable,
                     "non_trainable_params": non_trainable, "train_s_per_step": round(step, 3),
                     "est_epoch_min": round(epoch_est / 60, 1), "infer_ms_per_img_bs1": round(lat1, 2),
                     "infer_ms_per_img_bs32": round(lat32, 2)})
        print(rows[-1], flush=True)
        keras.backend.clear_session()

    table = pd.DataFrame(rows)
    main = table[table["model"].isin([v[0] for v in VARIANTS if v[4]])]
    main[["model", "total_params", "trainable_params", "non_trainable_params"]].to_csv(
        REPORTS_DIR / "model_parameters.csv", index=False)
    if args.no_timing:
        return
    table.drop(columns="model").to_csv(REPORTS_DIR / "training_cost.csv", index=False)
    cpu = cpu_name()
    md = [
        "# Resource report (CPU)",
        "",
        f"- Hardware: {cpu}, {os.cpu_count()} logical cores; no GPU used (TensorFlow {tf.__version__} on Windows is CPU-only)",
        f"- Keras {keras.__version__}, batch size {batch_size}, input {IMG_SIZE}x{IMG_SIZE}x3, augmentation on",
        f"- Training split: {n_train} images ({int(np.ceil(n_train / batch_size))} steps/epoch); validation: {n_val} images",
        f"- Step time measured over {args.steps} steps after one warm-up epoch; epoch estimate = training steps + one validation pass",
        "- Inference latency: median over repeated `predict_on_batch` calls on random input",
        "",
        "Model 3 rows: fine-tuning cost of each unfreezing depth (stage 2); `frozen` = only the heads are trained.",
        "",
        "| Model | Total params | Trainable | Non-trainable | s / step | est. min / epoch | ms / image (batch 1) | ms / image (batch 32) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['model']} | {r['total_params']:,} | {r['trainable_params']:,} | {r['non_trainable_params']:,} | "
                  f"{r['train_s_per_step']} | {r['est_epoch_min']} | {r['infer_ms_per_img_bs1']} | {r['infer_ms_per_img_bs32']} |")
    (REPORTS_DIR / "resource_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
