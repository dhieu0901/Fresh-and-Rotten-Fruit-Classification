"""Train all runs of the experiment plan one after another: the three main models, then the ablations.

    python scripts/run_experiments.py --list
    python scripts/run_experiments.py              # everything that has not finished yet
    python scripts/run_experiments.py --only main

Ablations: every model again without augmentation, and Model 3 stage 2 started from the same
stage-1 checkpoint with the backbone unfrozen from different blocks. Finished runs are skipped and
interrupted runs continue from their checkpoint (train.py --resume).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = "models/model3_mobilenet_v2_stage1.keras"  # written by the main Model 3 run

MAIN = [
    ("model1_simple_cnn", ["--model", "simple_cnn"]),
    ("model2_multitask_cnn", ["--model", "multitask_cnn"]),
    ("model3_mobilenet_v2", ["--model", "transfer"]),
]
ABLATIONS = [
    ("model1_simple_cnn_noaug", ["--model", "simple_cnn", "--no-aug"]),
    ("model2_multitask_cnn_noaug", ["--model", "multitask_cnn", "--no-aug"]),
    ("model3_mobilenet_v2_noaug", ["--model", "transfer", "--no-aug"]),
    ("model3_ft_none", ["--model", "transfer", "--fine-tune-from", "none", "--init-from", STAGE1]),
    ("model3_ft_block16", ["--model", "transfer", "--fine-tune-from", "block_16_expand", "--init-from", STAGE1]),
    ("model3_ft_block10", ["--model", "transfer", "--fine-tune-from", "block_10_expand", "--init-from", STAGE1]),
    ("model3_ft_block6", ["--model", "transfer", "--fine-tune-from", "block_6_expand", "--init-from", STAGE1]),
    ("model3_ft_all", ["--model", "transfer", "--fine-tune-from", "all", "--init-from", STAGE1]),
]
PLAN = {"main": MAIN, "ablations": ABLATIONS, "all": MAIN + ABLATIONS}


def main():
    parser = argparse.ArgumentParser(description="Run the experiment plan")
    parser.add_argument("--only", choices=sorted(PLAN), default="all")
    parser.add_argument("--threads", type=int, help="number of TensorFlow threads (default: all cores)")
    parser.add_argument("--list", action="store_true", help="only print the plan")
    args = parser.parse_args()
    runs = PLAN[args.only]
    if args.list:
        for run, extra in runs:
            print(f"{run:30s} python scripts/train.py {' '.join(extra)} --run-name {run}")
        return

    for run, extra in runs:
        log_dir = ROOT / "logs" / run
        if (log_dir / "val_metrics.json").exists():
            print(f"[skip] {run} (finished)", flush=True)
            continue
        if "--init-from" in extra and not (ROOT / STAGE1).exists():
            print(f"[skip] {run}: {STAGE1} not found, train model3_mobilenet_v2 first", flush=True)
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        if (log_dir / "train.log").exists():  # keep the log of an interrupted attempt
            (log_dir / "train.log").replace(log_dir / "train_interrupted.log")
        cmd = [sys.executable, "scripts/train.py", *extra, "--run-name", run, "--resume"]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        print(f"[start] {run}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        with open(log_dir / "train.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"[done ] {run}: exit {code}, {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
