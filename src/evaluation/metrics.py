"""Evaluation shared by the three models: every output is turned into fruit, freshness and joint predictions."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from src.config import COMBINED_NAMES, FRESHNESS, FRUITS, NUM_FRUITS


def decode_outputs(outputs):
    """Fruit / spoiled probabilities and the fruit, freshness and joint (16-class) predictions."""
    if isinstance(outputs, dict):  # models 2 and 3
        fruit_prob = np.asarray(outputs["fruit"], dtype=np.float64)
        spoiled_prob = np.asarray(outputs["freshness"], dtype=np.float64).reshape(-1)
        fruit_pred = fruit_prob.argmax(axis=1)
        fresh_pred = (spoiled_prob >= 0.5).astype(int)
        joint_pred = fruit_pred * 2 + fresh_pred
    else:  # model 1: one softmax over 16 classes, class = fruit * 2 + spoiled
        p = np.asarray(outputs, dtype=np.float64).reshape(-1, NUM_FRUITS, 2)
        fruit_prob = p.sum(axis=2)
        spoiled_prob = p[:, :, 1].sum(axis=1)
        fruit_pred = fruit_prob.argmax(axis=1)
        fresh_pred = (spoiled_prob >= 0.5).astype(int)
        joint_pred = p.reshape(len(p), -1).argmax(axis=1)
    return {
        "fruit_prob": fruit_prob,
        "spoiled_prob": spoiled_prob,
        "fruit_pred": fruit_pred,
        "fresh_pred": fresh_pred,
        "joint_pred": joint_pred,
    }


def predict(model, dataset):
    return decode_outputs(model.predict(dataset, verbose=0))


def compute_metrics(df, pred):
    """Accuracy and macro precision/recall/F1 for fruit and joint; binary metrics (spoiled = 1) for freshness."""
    yf, ys, yj = df["fruit_label"].to_numpy(), df["freshness_label"].to_numpy(), df["combined_label"].to_numpy()
    out = {}
    p, r, f, _ = precision_recall_fscore_support(yf, pred["fruit_pred"], average="macro", zero_division=0)
    out.update(fruit_accuracy=accuracy_score(yf, pred["fruit_pred"]), fruit_precision=p, fruit_recall=r, fruit_f1=f)
    p, r, f, _ = precision_recall_fscore_support(ys, pred["fresh_pred"], average="binary", pos_label=1, zero_division=0)
    out.update(
        freshness_accuracy=accuracy_score(ys, pred["fresh_pred"]),
        freshness_precision=p,
        freshness_recall=r,
        freshness_f1=f,
        freshness_macro_f1=f1_score(ys, pred["fresh_pred"], average="macro"),
        freshness_auc=roc_auc_score(ys, pred["spoiled_prob"]),
    )
    p, r, f, _ = precision_recall_fscore_support(yj, pred["joint_pred"], average="macro", zero_division=0)
    out.update(joint_accuracy=accuracy_score(yj, pred["joint_pred"]), joint_precision=p, joint_recall=r, joint_f1=f)
    return {k: float(v) for k, v in out.items()}


def confusion_matrices(df, pred):
    return {
        "fruit": (confusion_matrix(df["fruit_label"], pred["fruit_pred"], labels=range(len(FRUITS))), FRUITS),
        "freshness": (confusion_matrix(df["freshness_label"], pred["fresh_pred"], labels=[0, 1]), FRESHNESS),
        "joint": (
            confusion_matrix(df["combined_label"], pred["joint_pred"], labels=range(len(COMBINED_NAMES))),
            COMBINED_NAMES,
        ),
    }


def per_class_report(df, pred):
    p, r, f, s = precision_recall_fscore_support(
        df["combined_label"], pred["joint_pred"], labels=range(len(COMBINED_NAMES)), zero_division=0
    )
    return pd.DataFrame({"class": COMBINED_NAMES, "precision": p, "recall": r, "f1": f, "support": s})


def measure_inference_time(model, img_size, batch_size=1, n_warmup=5, n_runs=30, seed=0):
    """Median time per image in ms (predict_on_batch on random input)."""
    x = np.random.default_rng(seed).uniform(0, 255, (batch_size, img_size, img_size, 3)).astype("float32")
    for _ in range(n_warmup):
        model.predict_on_batch(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict_on_batch(x)
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times)) / batch_size
