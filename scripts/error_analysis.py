"""Error analysis of the three main models on the test split (run after scripts/evaluate.py).

    python scripts/error_analysis.py

Writes reports/error_analysis.md: error types, freshness false alarms and misses, errors per fruit,
the most frequent confusions, errors shared between the models, how confident the wrong
predictions were, and every test error of Model 3. Example images and Grad-CAM heatmaps of the
errors: figures/misclassified_samples.png and figures/gradcam/gradcam_errors.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import COMBINED_NAMES, FRUITS, PREDICTIONS_DIR, REPORTS_DIR  # noqa: E402

MAIN = [("model1_simple_cnn", "Model 1"), ("model2_multitask_cnn", "Model 2"), ("model3_mobilenet_v2", "Model 3")]
UNSURE = (0.2, 0.8)  # P(spoiled) range counted as an uncertain freshness decision


def load(run):
    p = pd.read_csv(PREDICTIONS_DIR / f"{run}_test.csv")
    p["fruit_wrong"] = p["fruit_pred"] != p["fruit_label"]
    p["fresh_wrong"] = p["freshness_pred"] != p["freshness_label"]
    p["wrong"] = p["joint_pred"] != p["combined_label"]
    return p


def md(df):
    cells = [[str(c) for c in df.columns]] + [[str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in cells[1:]])


def class_name(label):
    return COMBINED_NAMES[label].replace("_", " ")


def error_types(preds):
    rows = []
    for name, p in preds.items():
        fresh = p["freshness_label"] == 0
        rows.append({"Model": name, "Joint errors": int(p["wrong"].sum()),
                     "Fruit only": int((p["fruit_wrong"] & ~p["fresh_wrong"]).sum()),
                     "Freshness only": int((p["fresh_wrong"] & ~p["fruit_wrong"]).sum()),
                     "Both": int((p["fruit_wrong"] & p["fresh_wrong"]).sum()),
                     "False alarms (fresh -> spoiled)": int((p["fresh_wrong"] & fresh).sum()),
                     "Misses (spoiled -> fresh)": int((p["fresh_wrong"] & ~fresh).sum())})
    return pd.DataFrame(rows)


def errors_per_fruit(preds):
    base = next(iter(preds.values()))
    table = pd.DataFrame({"Fruit": FRUITS, "Test images": base.groupby("fruit_label").size().reindex(range(len(FRUITS))).values})
    for name, p in preds.items():
        table[name] = p[p["wrong"]].groupby("fruit_label").size().reindex(range(len(FRUITS)), fill_value=0).values
    return table


def top_confusions(preds, n=5):
    rows = []
    for name, p in preds.items():
        pairs = p[p["wrong"]].groupby(["combined_label", "joint_pred"]).size().sort_values(ascending=False).head(n)
        for (true, pred), count in pairs.items():
            rows.append({"Model": name, "True class": class_name(true), "Predicted": class_name(pred), "Images": int(count)})
    return pd.DataFrame(rows)


def shared_errors(preds):
    wrong = pd.DataFrame({name: p["wrong"] for name, p in preds.items()})
    n_models = wrong.sum(axis=1)
    rows = [{"Misclassified by": f"exactly {k} model{'s' if k > 1 else ''}", "Images": int((n_models == k).sum())}
            for k in range(1, len(preds) + 1)]
    names = list(preds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"Misclassified by": f"both {a} and {b}", "Images": int((wrong[a] & wrong[b]).sum())})
    return pd.DataFrame(rows)


def confidence(preds):
    rows = []
    for name, p in preds.items():
        e = p[p["fresh_wrong"]]
        unsure = e["spoiled_prob"].between(*UNSURE)
        rows.append({"Model": name, "Freshness errors": len(e),
                     f"Uncertain (P(spoiled) {UNSURE[0]}-{UNSURE[1]})": int(unsure.sum()),
                     "Confidently wrong": int((~unsure).sum()),
                     "Median P(spoiled), false alarms": round(e.loc[e["freshness_label"] == 0, "spoiled_prob"].median(), 2)
                     if (e["freshness_label"] == 0).any() else "-",
                     "Median P(spoiled), misses": round(e.loc[e["freshness_label"] == 1, "spoiled_prob"].median(), 2)
                     if (e["freshness_label"] == 1).any() else "-"})
    return pd.DataFrame(rows)


def error_list(p):
    e = p[p["wrong"]].copy()
    return pd.DataFrame({"Image": e["path"].str.replace("data/raw/FRUIT-16K/", "", regex=False),
                         "True class": e["combined_label"].map(class_name), "Predicted": e["joint_pred"].map(class_name),
                         "P(spoiled)": e["spoiled_prob"].round(2),
                         "P(true fruit)": [round(r[f"p_{FRUITS[r['fruit_label']]}"], 2) for _, r in e.iterrows()]})


def main():
    preds = {name: load(run) for run, name in MAIN if (PREDICTIONS_DIR / f"{run}_test.csv").exists()}
    n = len(next(iter(preds.values())))
    md_lines = [
        f"# Error analysis (test split, {n:,} images)", "",
        "Joint error = fruit type or freshness (or both) wrong. Freshness: spoiled is the positive class, so a",
        "false alarm is a fresh fruit predicted spoiled and a miss is a spoiled fruit predicted fresh.",
        "Images of the errors: figures/misclassified_samples.png; Grad-CAM of the freshness errors:",
        "figures/gradcam/gradcam_errors.png.", "",
        "## Error types", "", md(error_types(preds)), "",
        "## Joint errors per fruit", "", md(errors_per_fruit(preds)), "",
        "## Most frequent confusions", "", md(top_confusions(preds)), "",
        "## Errors shared between the models", "", md(shared_errors(preds)), "",
        "## Confidence of the freshness errors", "", md(confidence(preds)), "",
    ]
    if "Model 3" in preds:
        md_lines += ["## Every test error of Model 3", "", md(error_list(preds["Model 3"])), ""]
    path = REPORTS_DIR / "error_analysis.md"
    path.write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))
    print("written to", path)


if __name__ == "__main__":
    main()
