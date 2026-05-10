"""
Zar3y — Requirement 2: Transfer-Learning Training
==================================================
Usage:
    python src/train.py --data_dir data/plant_village --splits_dir data/splits
"""

import os
import json
import sys
import argparse

sys.path.insert(0, "src")   # makes data_prep importable when run from project root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random


from data_prep import get_datasets, CLASS_DISPLAY_NAMES   # Req-1 pipeline

# ── Locked constants ──────────────────────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 32
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 10
PATIENCE      = 3
LR_STAGE1     = 1e-3
LR_STAGE2     = 1e-4
UNFREEZE_LAST = 30
WEIGHTS_PATH  = "models/best_model.weights.h5"
MODEL_PATH    = "models/best_model.h5"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# ─────────────────────────────────────────────────────────────────────────────


def mobilenet_preprocess_batch(imgs: tf.Tensor, labels: tf.Tensor):
    """
    Apply MobileNetV3 preprocessing to a BATCH of images.
    """
    imgs = imgs * 255.0
    imgs = keras.applications.mobilenet_v3.preprocess_input(imgs)
    return imgs, labels


def wrap_preprocess(ds: tf.data.Dataset) -> tf.data.Dataset:
    """
    Apply MobileNetV3 preprocessing to an already-batched dataset.
    """
    return ds.map(mobilenet_preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE) \
             .prefetch(tf.data.AUTOTUNE)


def build_model(n_classes: int, trainable_base: bool = False, unfreeze_last: int = 0):
    """
    MobileNetV3-Small backbone + GlobalAveragePooling + Dropout(0.2) + Dense(softmax).

    trainable_base=False  -> fully frozen backbone (Phase 1).
    trainable_base=True   -> unfreeze only the last `unfreeze_last` layers (Phase 2).
    """
    base = keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base
    if trainable_base and unfreeze_last > 0:
        for layer in base.layers[:-unfreeze_last]:
            layer.trainable = False

    inputs  = keras.Input(shape=(224, 224, 3))
    x       = base(inputs, training=trainable_base)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.2, seed=SEED)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def save_training_curves(h1: dict, h2: dict, out_path: str):
    """Plot accuracy + loss curves for both phases and save as PNG."""
    train_acc  = h1["accuracy"]     + h2["accuracy"]
    val_acc    = h1["val_accuracy"] + h2["val_accuracy"]
    train_loss = h1["loss"]         + h2["loss"]
    val_loss   = h1["val_loss"]     + h2["val_loss"]
    boundary   = len(h1["accuracy"])
    epochs_x   = list(range(1, len(train_acc) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs_x, train_acc,  label="Train",      linewidth=2)
    ax1.plot(epochs_x, val_acc,    label="Validation", linewidth=2)
    if boundary < len(epochs_x):
        ax1.axvline(x=boundary + 0.5, color="gray", linestyle="--",
                    linewidth=1.5, label="Fine-tune start")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1)

    ax2.plot(epochs_x, train_loss, label="Train",      linewidth=2)
    ax2.plot(epochs_x, val_loss,   label="Validation", linewidth=2)
    if boundary < len(epochs_x):
        ax2.axvline(x=boundary + 0.5, color="gray", linestyle="--",
                    linewidth=1.5, label="Fine-tune start")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("MobileNetV3-Small — Transfer Learning", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved -> {out_path}")


def run_phase(model, train_ds, val_ds, class_weights,
              n_epochs, phase_label,
              global_best_val_loss, global_best_weights):
    """
    Run one training phase with early stopping.
    Returns (global_best_val_loss, global_best_weights, metrics_dict).
    """
    metrics = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    patience_count = 0

    for epoch in range(n_epochs):
        print(f"  [{phase_label}] Epoch {epoch+1}/{n_epochs}", flush=True)
        hist = model.fit(
            train_ds, validation_data=val_ds,
            epochs=1, class_weight=class_weights, verbose=1,
        )

        acc  = hist.history["accuracy"][0]
        vacc = hist.history["val_accuracy"][0]
        tl   = hist.history["loss"][0]
        vl   = hist.history["val_loss"][0]

        metrics["accuracy"].append(acc)
        metrics["val_accuracy"].append(vacc)
        metrics["loss"].append(tl)
        metrics["val_loss"].append(vl)

        if vl < global_best_val_loss:
            global_best_val_loss = vl
            global_best_weights  = model.get_weights()
            patience_count       = 0
            model.save_weights(WEIGHTS_PATH)
            print(f"    -> New best val_loss={vl:.4f} — checkpoint saved")
        else:
            patience_count += 1
            print(f"    – No improvement ({patience_count}/{PATIENCE})  "
                  f"val_loss={vl:.4f}  best={global_best_val_loss:.4f}")
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return global_best_val_loss, global_best_weights, metrics


def main(args):
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ── 1. Load datasets from Req-1 pipeline  ──────────
    print("Loading datasets from data_prep.get_datasets() ...")
    train_ds_raw, val_ds_raw, _, class_names, class_weights_raw = get_datasets(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
    )

    class_weights = (
        {int(k): float(v) for k, v in class_weights_raw.items()}
        if class_weights_raw else None
    )

    # Apply MobileNetV3 preprocessing 
    train_ds = wrap_preprocess(train_ds_raw)
    val_ds   = wrap_preprocess(val_ds_raw)

    n_classes = len(class_names)
    label2id  = {label: idx for idx, label in enumerate(sorted(class_names))}

    with open("models/label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)

    print("=" * 60)
    print("  TRANSFER LEARNING — MobileNetV3-Small")
    print("=" * 60)
    print(f"  Classes  : {n_classes}")
    print(f"  Patience : {PATIENCE} (global across both phases)")
    print(f"  Weights  : {'yes' if class_weights else 'no'}")
    print("=" * 60)

    global_best_val_loss = float("inf")
    global_best_weights  = None

    # ── 2. Phase 1 — frozen backbone ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1  —  Frozen backbone  (LR=1e-3)")
    print("=" * 60)

    model = build_model(n_classes, trainable_base=False)
    model.compile(
        optimizer=keras.optimizers.Adam(LR_STAGE1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    tp = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"  Trainable params: {tp:,}\n")

    global_best_val_loss, global_best_weights, h1 = run_phase(
        model, train_ds, val_ds, class_weights,
        EPOCHS_STAGE1, "Phase 1",
        global_best_val_loss, global_best_weights,
    )
    model.set_weights(global_best_weights)
    print(f"\n  Phase 1 done — global best val_loss={global_best_val_loss:.4f}")

    # ── 3. Phase 2 — fine-tune last 30 backbone layers ────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2  —  Fine-tuning last 30 layers  (LR=1e-4)")
    print(f"  Inheriting global best val_loss={global_best_val_loss:.4f}")
    print("=" * 60)

    model_ft = build_model(n_classes, trainable_base=True, unfreeze_last=UNFREEZE_LAST)
    model_ft.compile(
        optimizer=keras.optimizers.Adam(LR_STAGE2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model_ft.set_weights(global_best_weights)

    tp2 = sum(tf.size(w).numpy() for w in model_ft.trainable_weights)
    print(f"  Trainable params: {tp2:,}\n")

    global_best_val_loss, global_best_weights, h2 = run_phase(
        model_ft, train_ds, val_ds, class_weights,
        EPOCHS_STAGE2, "Phase 2",
        global_best_val_loss, global_best_weights,
    )

    model_ft.set_weights(global_best_weights)
    model_ft.save(MODEL_PATH, include_optimizer=False)
    print(f"\n  Full model saved -> {MODEL_PATH}")

    save_training_curves(h1, h2, out_path="outputs/training_curves.png")

    with open("models/training_meta.json", "w") as f:
        json.dump({
            "phase1_epochs":        len(h1["accuracy"]),
            "phase2_epochs":        len(h2["accuracy"]),
            "global_best_val_loss": float(global_best_val_loss),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Phase 1 epochs       : {len(h1['accuracy'])}")
    print(f"  Phase 2 epochs       : {len(h2['accuracy'])}")
    print(f"  Global best val_loss : {global_best_val_loss:.4f}")
    print(f"  Weights  -> {WEIGHTS_PATH}")
    print(f"  Model    -> {MODEL_PATH}")
    print(f"  Curves   -> outputs/training_curves.png")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zar3y — Req 2: Transfer-learning training")
    parser.add_argument("--data_dir",   default="data/plant_village")
    parser.add_argument("--splits_dir", default="data/splits")
    main(parser.parse_args())
