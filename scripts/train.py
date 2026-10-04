"""Train one model on the train split; the val split is used for checkpoints and early stopping.

    python scripts/train.py --model simple_cnn
    python scripts/train.py --model multitask_cnn --no-aug --run-name model2_multitask_cnn_noaug
    python scripts/train.py --model transfer
    python scripts/train.py --model transfer --fine-tune-from block_16_expand --run-name model3_ft_block16 --init-from models/model3_mobilenet_v2_stage1.keras
    python scripts/train.py --model multitask_cnn --resume

The best model is saved to models/<run>.keras, the history to logs/<run>/ and a summary row to
reports/experiment_log.csv.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.config import LOGS_DIR, MODELS_DIR, REPORTS_DIR, load_json  # noqa: E402
from src.data_pipeline import load_split, make_dataset  # noqa: E402
from src.evaluation.metrics import compute_metrics, predict  # noqa: E402
from src.models import build_model, compile_model, label_mode_of, set_backbone_trainable  # noqa: E402
from src.training.callbacks import make_callbacks  # noqa: E402

DEFAULT_RUN_NAMES = {
    "simple_cnn": "model1_simple_cnn",
    "multitask_cnn": "model2_multitask_cnn",
    "transfer": "model3_mobilenet_v2",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train one model (train/val only)")
    p.add_argument("--model", required=True, choices=sorted(DEFAULT_RUN_NAMES))
    p.add_argument(
        "--run-name", help="default: model1_simple_cnn / model2_multitask_cnn / model3_mobilenet_v2 (+ _noaug)"
    )
    p.add_argument("--no-aug", action="store_true", help="disable data augmentation (ablation)")
    p.add_argument("--epochs", type=int, help="override max epochs (transfer: stage 2)")
    p.add_argument("--stage1-epochs", type=int, help="transfer: override stage-1 epochs")
    p.add_argument("--fine-tune-from", help="transfer: backbone layer to unfreeze from, 'all', or 'none' (frozen only)")
    p.add_argument("--init-from", help="transfer: start stage 2 from this stage-1 .keras checkpoint (skips stage 1)")
    p.add_argument("--limit", type=int, help="use only N images per class (smoke test)")
    p.add_argument("--threads", type=int, help="number of TensorFlow threads")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run from its best checkpoint")
    return p.parse_args()


def resume_point(history_path, ckpt):
    """Where an interrupted run continues: (epochs kept, best val_loss, seconds spent), or None.
    Training restarts after the best epoch (the one in the checkpoint), so later rows are removed."""
    if not (history_path.exists() and ckpt.exists()):
        return None
    hist = pd.read_csv(history_path)
    if hist.empty:
        return None
    best = int(hist["val_loss"].idxmin())
    kept = hist.iloc[: best + 1]
    kept.to_csv(history_path, index=False)
    return best + 1, float(kept["val_loss"].iloc[-1]), float(kept["epoch_time"].sum())


def fit(model, train_ds, val_ds, epochs, run_dir, ckpt, common, history_name, initial_epoch=0, resume_best=None):
    callbacks = make_callbacks(
        run_dir,
        ckpt,
        monitor=common["monitor"],
        early_stopping_patience=common["early_stopping_patience"],
        reduce_lr_patience=common["reduce_lr_patience"],
        reduce_lr_factor=common["reduce_lr_factor"],
        min_lr=common["min_lr"],
        log_name=history_name,
        resume_best=resume_best,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epoch + epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=2,
    )
    best = keras.models.load_model(ckpt)  # reload the best epoch explicitly
    hist = pd.read_csv(run_dir / history_name)  # all epochs, also the ones before a resume
    return best, hist


def main():
    args = parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    cfg = load_json("training_config.json")
    common, mcfg = cfg["common"], cfg["models"][args.model]
    keras.utils.set_random_seed(common["seed"])

    augment = not args.no_aug
    run = args.run_name or DEFAULT_RUN_NAMES[args.model] + ("" if augment else "_noaug")
    run_dir = LOGS_DIR / run
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt = MODELS_DIR / f"{run}.keras"
    mode = label_mode_of(args.model)
    bs = common["batch_size"]

    train_ds = make_dataset(
        "train", mode, bs, augment=augment, aug_config=cfg["augmentation"], limit=args.limit, seed=common["seed"]
    )
    val_ds = make_dataset("val", mode, bs, limit=args.limit)
    record = {
        "run": run,
        "model": args.model,
        "augmentation": augment,
        "batch_size": bs,
        "seed": common["seed"],
        "limit_per_class": args.limit,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    t0 = time.time()

    if args.model != "transfer":
        epochs = args.epochs or mcfg["epochs"]
        resume = resume_point(run_dir / "history.csv", ckpt) if args.resume else None
        if resume:
            model = keras.models.load_model(ckpt)  # weights + optimizer state at the end of the best epoch
        else:
            model = build_model(args.model, **mcfg.get("build", {}))
            compile_model(model, mode, mcfg["learning_rate"], mcfg.get("loss_weights"))
        done, best, seconds = resume or (0, None, 0.0)
        t0 -= seconds
        record.update(learning_rate=mcfg["learning_rate"], max_epochs=epochs)
        model, hist = fit(
            model,
            train_ds,
            val_ds,
            epochs - done,
            run_dir,
            ckpt,
            common,
            "history.csv",
            initial_epoch=done,
            resume_best=best,
        )
        record.update(epochs_trained=len(hist), best_epoch=int(hist["val_loss"].idxmin()) + 1)
    else:
        s1, s2 = mcfg["stage1"], mcfg["stage2"]
        fine_tune_from = args.fine_tune_from or s2["fine_tune_from"]
        fine_tune_from = None if fine_tune_from == "none" else fine_tune_from
        stage1_ckpt = MODELS_DIR / f"{run}_stage1.keras"
        resume = None
        if args.resume and fine_tune_from is not None:
            resume = resume_point(run_dir / "history_stage2.csv", ckpt)
        if resume:  # stage 1 had finished: continue stage 2 from its best checkpoint
            model = keras.models.load_model(ckpt)
            stage1_epochs = 0
            if args.init_from:
                record.update(init_from=args.init_from)
            else:
                hist1 = pd.read_csv(run_dir / "history_stage1.csv")
                stage1_epochs = len(hist1)
                t0 -= float(hist1["epoch_time"].sum())
                record.update(
                    stage1_lr=s1["learning_rate"],
                    stage1_epochs=stage1_epochs,
                    stage1_best_epoch=int(hist1["val_loss"].idxmin()) + 1,
                )
        elif args.init_from:  # reuse a trained stage-1 model (fair comparison of fine-tuning depths)
            model = keras.models.load_model(args.init_from)
            stage1_epochs = 0
            record.update(init_from=args.init_from)
        else:
            model = build_model("transfer", **mcfg.get("build", {}))
            set_backbone_trainable(model, None)
            compile_model(model, mode, s1["learning_rate"], mcfg.get("loss_weights"))
            e1 = args.stage1_epochs or s1["epochs"]
            model, hist1 = fit(model, train_ds, val_ds, e1, run_dir, stage1_ckpt, common, "history_stage1.csv")
            stage1_epochs = len(hist1)
            record.update(
                stage1_lr=s1["learning_rate"],
                stage1_epochs=stage1_epochs,
                stage1_best_epoch=int(hist1["val_loss"].idxmin()) + 1,
            )
        if fine_tune_from is None:
            model.save(ckpt)
            record.update(fine_tune_from="none (frozen backbone)")
        else:
            info = set_backbone_trainable(model, fine_tune_from)  # a resumed model already has these flags
            if not resume:
                compile_model(model, mode, s2["learning_rate"], mcfg.get("loss_weights"))  # recompile after unfreezing
            done, best, seconds = resume or (0, None, 0.0)
            t0 -= seconds
            e2 = args.epochs or s2["epochs"]
            model, hist2 = fit(
                model,
                train_ds,
                val_ds,
                e2 - done,
                run_dir,
                ckpt,
                common,
                "history_stage2.csv",
                initial_epoch=stage1_epochs + done,
                resume_best=best,
            )
            record.update(
                fine_tune_from=fine_tune_from,
                unfrozen_layers=info["unfrozen_layers"],
                unfrozen_layers_with_weights=info["unfrozen_layers_with_weights"],
                trainable_params_stage2=info["trainable_params"],
                stage2_lr=s2["learning_rate"],
                stage2_epochs=len(hist2),
                stage2_best_epoch=int(hist2["val_loss"].idxmin()) + 1,
            )

    if resume:
        record["resumed_after_epoch"] = stage1_epochs + done if args.model == "transfer" else done
    record["train_minutes"] = round(
        (time.time() - t0) / 60, 1
    )  # resumed run: includes the epochs before the interruption
    val_df = load_split("val")
    if args.limit:
        val_df = val_df.groupby("combined_label", sort=False).head(args.limit).reset_index(drop=True)
    metrics = compute_metrics(val_df, predict(model, val_ds))
    record.update({f"val_{k}": round(v, 4) for k, v in metrics.items()})
    record["params"] = model.count_params()

    (run_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "training_config": cfg}, indent=2), encoding="utf-8"
    )
    (run_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = []
    model.summary(print_fn=lambda s, line_break=None: lines.append(s), show_trainable=True)
    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    if not args.limit:
        log_path = REPORTS_DIR / "experiment_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        log = pd.concat(
            [log[log.get("run", pd.Series(dtype=str)) != run] if len(log) else log, pd.DataFrame([record])],
            ignore_index=True,
        )
        log.to_csv(log_path, index=False)
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
