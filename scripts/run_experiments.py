"""Run the whole experiment plan, one training at a time: the three main models first, then the ablations.

    python scripts/run_experiments.py --list          # show the plan
    python scripts/run_experiments.py                 # run every experiment that has not finished yet
    python scripts/run_experiments.py --only main     # only the three main models

Every run is one call to scripts/train.py --resume; its console output goes to logs/<run>/train.log.
A run is skipped when it has finished before (logs/<run>/val_metrics.json exists) and an interrupted
run continues from its best checkpoint, so the plan can simply be started again after a shutdown.
While it runs, Windows is kept awake (the screen may still turn off).

Ablations
    augmentation : each model trained again with --no-aug
    fine-tuning  : Model 3 stage 2 restarted from the same stage-1 checkpoint with the backbone
                   unfrozen from different depths (none / block 16 / block 13 = main run / block 10 / block 6 / all)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = "models/model3_mobilenet_v2_stage1.keras"   # stage-1 checkpoint written by the main Model 3 run

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


def keep_awake(on=True):
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main():
    parser = argparse.ArgumentParser(description="Run the experiment plan, one training at a time")
    parser.add_argument("--only", choices=sorted(PLAN), default="all", help="part of the plan to run (default: all)")
    parser.add_argument("--threads", type=int, help="TensorFlow intra-op threads (default: every CPU core)")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    args = parser.parse_args()
    runs = PLAN[args.only]
    if args.list:
        for run, extra in runs:
            print(f"{run:30s} python scripts/train.py {' '.join(extra)} --run-name {run}")
        return

    keep_awake(True)
    try:
        for run, extra in runs:
            if (ROOT / "logs" / run / "val_metrics.json").exists():
                print(f"[skip] {run} (already finished)", flush=True)
                continue
            if "--init-from" in extra and not (ROOT / STAGE1).exists():
                print(f"[skip] {run}: {STAGE1} not found - train model3_mobilenet_v2 first", flush=True)
                continue
            log_dir = ROOT / "logs" / run
            log_dir.mkdir(parents=True, exist_ok=True)
            if (log_dir / "train.log").exists():  # keep the console output of an interrupted attempt
                (log_dir / "train.log").replace(log_dir / "train_interrupted.log")
            cmd = [sys.executable, "scripts/train.py", *extra, "--run-name", run, "--resume"]
            if args.threads:
                cmd += ["--threads", str(args.threads)]
            print(f"[start] {run}: {' '.join(cmd[1:])}", flush=True)
            t0 = time.time()
            with open(log_dir / "train.log", "w", encoding="utf-8") as log:
                code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            print(f"[done ] {run}: exit {code}, {(time.time() - t0) / 60:.0f} min", flush=True)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
