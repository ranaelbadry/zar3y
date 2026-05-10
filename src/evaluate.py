"""
Zar3y — Requirement 2: Honest Evaluation (WEIGHTS ONLY VERSION)
"""

import os, json, sys, argparse
sys.path.insert(0, "src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)

from data_prep import get_datasets, CLASS_DISPLAY_NAMES
from train import build_model   # IMPORTANT: reuse architecture

BATCH_SIZE = 32


# ─────────────────────────────────────────────
# Preprocessing (same as training)
# ─────────────────────────────────────────────
def mobilenet_preprocess(img, label):
    img = img * 255.0
    img = keras.applications.mobilenet_v3.preprocess_input(img)
    return img, label


def wrap_preprocess(ds):
    return (ds.map(
                mobilenet_preprocess,
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE))


# ─────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────
def get_predictions(model, ds):
    y_true, y_pred = [], []

    for batch_x, batch_y in ds:
        preds = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(batch_y.numpy())

    return np.array(y_true), np.array(y_pred)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def build_eval_report(y_true, y_pred, class_names):
    overall_acc  = float(accuracy_score(y_true, y_pred))
    macro_f1     = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec   = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    pc_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    pc_rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    pc_f1   = f1_score(y_true, y_pred, average=None, zero_division=0)

    support = np.bincount(y_true, minlength=len(class_names))

    per_class = {
        cls: {
            "display_name": CLASS_DISPLAY_NAMES.get(cls, cls),
            "precision": float(pc_prec[i]),
            "recall":    float(pc_rec[i]),
            "f1":        float(pc_f1[i]),
            "support":   int(support[i]),
        }
        for i, cls in enumerate(class_names)
    }

    return {
        "overall": {
            "accuracy": overall_acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_recall,
            "n_samples": int(len(y_true)),
        },
        "per_class": per_class,
    }


# ─────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────
def print_report(report, class_names):
    ov = report["overall"]

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS — TEST SET")
    print("=" * 70)
    print(f"  Samples          : {ov['n_samples']:,}")
    print(f"  Accuracy         : {ov['accuracy']:.4f}")
    print(f"  Macro F1         : {ov['macro_f1']:.4f}")
    print(f"  Macro Precision  : {ov['macro_precision']:.4f}")
    print(f"  Macro Recall     : {ov['macro_recall']:.4f}")
    print("=" * 70)

    print(f"\n{'Class':<45} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    print("-" * 75)

    for cls in class_names:
        d = report["per_class"][cls]
        print(f"{d['display_name']:<45} "
              f"{d['precision']:.3f} {d['recall']:.3f} {d['f1']:.3f} {d['support']}")

    print("-" * 75)


# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    display_names = [CLASS_DISPLAY_NAMES.get(c, c) for c in class_names]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        linewidths=0.5
    )

    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved -> {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading datasets...")
    _, _, test_ds_raw, class_names, _ = get_datasets(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
    )

    test_ds = wrap_preprocess(test_ds_raw)

    # ── IMPORTANT: build model + load WEIGHTS ONLY ──
    print("Building model architecture...")
    model = build_model(len(class_names), trainable_base=False)

    print(f"Loading weights from {args.weights_path} ...")
    model.load_weights(args.weights_path)

    print("Running inference...")
    y_true, y_pred = get_predictions(model, test_ds)

    report = build_eval_report(y_true, y_pred, class_names)

    print_report(report, class_names)

    # Save JSON
    report_path = os.path.join(args.output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, class_names, cm_path)

    print("\nSaved:")
    print(report_path)
    print(cm_path)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/plant_village")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--weights_path", default="models/best_model.weights.h5")
    parser.add_argument("--output_dir", default="outputs")

    main(parser.parse_args())
