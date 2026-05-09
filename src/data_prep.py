"""
Zar3y — Requirement 1: Data Preparation & Augmentation
========================================================
Usage (after downloading PlantVillage to data/plant_village/):
    python src/data_prep.py --data_dir data/plant_village --output_dir data/splits

Outputs:
    - data/splits/train.csv, val.csv, test.csv   (stratified 70/15/15, seed=42)
    - outputs/augmentation_samples.png
    - Prints per-class count table + class weights (if any class < 500 images)
"""

import os
import argparse
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ─── Locked constants (do NOT change) ────────────────────────────────────────
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
MIN_CLASS_SAMPLES = 500          # threshold for class-weighted loss

LOCKED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Pepper,_bell___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Corn_(maize)___Common_rust_",
]

# Friendly display names (for printing / README table)
CLASS_DISPLAY_NAMES = {
    "Tomato___healthy":                "Tomato — Healthy",
    "Tomato___Early_blight":           "Tomato — Early Blight",
    "Tomato___Late_blight":            "Tomato — Late Blight",
    "Tomato___Leaf_Mold":              "Tomato — Leaf Mold",
    "Potato___healthy":                "Potato — Healthy",
    "Potato___Early_blight":           "Potato — Early Blight",
    "Potato___Late_blight":            "Potato — Late Blight",
    "Pepper,_bell___healthy":          "Pepper — Healthy",
    "Pepper,_bell___Bacterial_spot":   "Pepper — Bacterial Spot",
    "Corn_(maize)___Common_rust_":     "Corn — Common Rust",
}
# ─────────────────────────────────────────────────────────────────────────────


def _normalize(name: str) -> str:
    """Lowercase + remove spaces, commas, parentheses, underscores for fuzzy comparison."""
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower())


def collect_image_paths(data_dir: Path) -> pd.DataFrame:
    """
    Walk data_dir and map every image folder to one of the 10 locked classes
    using normalized fuzzy matching — robust against:
      • Commas vs underscores  (Pepper,_bell  vs  Pepper__bell)
      • Spaces                 (Corn (maize)  vs  Corn_(maize))
      • Trailing underscores   (Common_rust_  vs  Common_rust)
      • Nested layouts         color/<Class>/ or segmented/<Class>/

    Returns DataFrame with columns [path, label].
    """
    records = []
    data_dir = Path(data_dir)

    # Pre-compute normalised keys for each locked class
    norm2class = {_normalize(c): c for c in LOCKED_CLASSES}

    # Collect every directory anywhere under data_dir
    all_dirs = [p for p in data_dir.rglob("*") if p.is_dir()]

    matched_dirs: dict[str, str] = {}   # dir_path_str → locked_class_name
    for d in all_dirs:
        norm_name = _normalize(d.name)
        if norm_name in norm2class:
            matched_dirs[str(d)] = norm2class[norm_name]

    if not matched_dirs:
        # Print what we actually found so the user can debug
        found = [p.name for p in all_dirs]
        raise FileNotFoundError(
            f"No matching folders found under '{data_dir}'.\n"
            f"Folders found: {found[:30]}\n"
            f"Expected (normalised): {list(norm2class.keys())}"
        )

    for dir_path, class_name in matched_dirs.items():
        for img_path in Path(dir_path).iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({"path": str(img_path), "label": class_name})

    df = pd.DataFrame(records).drop_duplicates(subset="path")
    return df


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70 / 15 / 15 split, seed=42 (locked).
    Returns (train_df, val_df, test_df).
    """
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["label"], random_state=SEED
    )
    # Split temp into val and test (50/50 of the 30 %)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df["label"], random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_class_weights(train_df: pd.DataFrame) -> dict:
    """
    Compute inverse-frequency class weights.
    Returns a dict {class_index: weight}.
    Applied only when at least one class has < MIN_CLASS_SAMPLES images.
    """
    counts = train_df["label"].value_counts()
    classes = sorted(train_df["label"].unique())
    n_samples = len(train_df)
    n_classes = len(classes)

    weights = {}
    for i, cls in enumerate(classes):
        # sklearn-style balanced weight: n_samples / (n_classes * count)
        weights[i] = n_samples / (n_classes * counts[cls])

    return weights


def print_class_table(train_df, val_df, test_df):
    """
    Print a Markdown-friendly per-class count table and determine if
    class-weighted loss is needed.
    Returns (needs_weighting: bool, class_weights: dict | None)
    """
    classes = sorted(train_df["label"].unique())
    label2idx = {c: i for i, c in enumerate(classes)}

    print("\n## Per-Class Image Counts\n")
    print(f"{'Class':<45} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>8}")
    print("-" * 78)

    train_counts = Counter(train_df["label"])
    val_counts   = Counter(val_df["label"])
    test_counts  = Counter(test_df["label"])

    needs_weighting = False
    for cls in classes:
        tr = train_counts[cls]
        va = val_counts[cls]
        te = test_counts[cls]
        tot = tr + va + te
        flag = " < 500" if tr < MIN_CLASS_SAMPLES else ""
        if tr < MIN_CLASS_SAMPLES:
            needs_weighting = True
        display = CLASS_DISPLAY_NAMES.get(cls, cls)
        print(f"{display:<45} {tr:>7} {va:>7} {te:>7} {tot:>8}{flag}")

    print("-" * 78)
    total_tr = len(train_df)
    total_va = len(val_df)
    total_te = len(test_df)
    print(f"{'TOTAL':<45} {total_tr:>7} {total_va:>7} {total_te:>7} {total_tr+total_va+total_te:>8}")
    print()

    if needs_weighting:
        print("One or more classes have < 500 training images → class-weighted loss WILL be applied.")
        weights = compute_class_weights(train_df)
        print("Class weights (index → weight):")
        for cls in classes:
            idx = label2idx[cls]
            print(f"  [{idx}] {CLASS_DISPLAY_NAMES.get(cls, cls):<40} → {weights[idx]:.4f}")
        print()
        return True, weights
    else:
        print("All classes have ≥ 500 training images — standard (unweighted) loss will be used.")
        print()
        return False, None


# ─── tf.data pipeline ────────────────────────────────────────────────────────

def load_and_resize(path: tf.Tensor, label: tf.Tensor):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def augment(img: tf.Tensor, label: tf.Tensor):
    """
    Augmentation pipeline (training only):
      • Random horizontal flip
      • Random rotation ±15° (~0.26 rad)
      • Random brightness ±20%
      • Random contrast ±20%
      • Random zoom (crop + resize, equivalent to ~10% zoom)
    """
    # 1. Horizontal flip
    img = tf.image.random_flip_left_right(img)

    # 2. Random rotation ±15°
    angle_rad = tf.random.uniform([], minval=-0.2618, maxval=0.2618)   # ±15° in radians
    img = _rotate(img, angle_rad)

    # 3. Random brightness ±20%
    img = tf.image.random_brightness(img, max_delta=0.2)

    # 4. Random contrast ±20%
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

    # 5. Random zoom (simulate via random crop + resize)
    img = _random_zoom(img, zoom_range=0.10)

    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def _rotate(img: tf.Tensor, angle_rad: tf.Tensor) -> tf.Tensor:
    """
    Rotate image by angle_rad using a pure-TF affine transform.
    Works inside tf.function / tf.data.map — no Keras layers, no Variables.
    """
    h, w = IMG_SIZE[0], IMG_SIZE[1]
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)
    cx, cy = w / 2.0, h / 2.0

    # Affine transform matrix in TF's flat 8-element format:
    # [a0, a1, a2, b0, b1, b2, c0, c1]
    # where: x' = (a0*x + a1*y + a2) / k,  y' = (b0*x + b1*y + b2) / k,  k = c0*x + c1*y + 1
    a2 = (1 - cos_a) * cx + sin_a * cy
    b2 = -sin_a * cx + (1 - cos_a) * cy
    transform = [cos_a, -sin_a, a2,
                 sin_a,  cos_a, b2,
                 0.0,    0.0]
    transform = tf.stack(transform)           # shape (8,)
    transform = tf.expand_dims(transform, 0)  # shape (1, 8)  — batch dim

    img4 = tf.expand_dims(img, 0)            # (1, H, W, 3)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=tf.cast(transform, tf.float32),
        output_shape=tf.constant([h, w], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(rotated, 0)            # (H, W, 3)


def _random_zoom(img: tf.Tensor, zoom_range: float = 0.10) -> tf.Tensor:
    """Random zoom by cropping a [1-zoom_range, 1] fraction and resizing back."""
    h, w = IMG_SIZE
    scale = tf.random.uniform([], 1.0 - zoom_range, 1.0)
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    img = tf.image.random_crop(img, [new_h, new_w, 3])
    img = tf.image.resize(img, [h, w])
    return img


def build_dataset(
    df: pd.DataFrame,
    class_names: list[str],
    augment_data: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a DataFrame with columns [path, label].
    Returns batched, prefetched dataset.
    """
    label2idx = {c: i for i, c in enumerate(class_names)}
    paths  = df["path"].values
    labels = np.array([label2idx[l] for l in df["label"].values], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Augmentation visualisation ──────────────────────────────────────────────

def save_augmentation_samples(
    train_df: pd.DataFrame,
    class_names: list[str],
    output_path: str = "outputs/augmentation_samples.png",
    n_classes: int = 5,
    n_augments: int = 4,
):
    """
    Pick one image per class (first n_classes classes), apply augmentation
    n_augments times, and save a grid: rows = classes, cols = [original, aug×n].
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(SEED)

    selected_classes = class_names
    n_rows = len(selected_classes)
    cols = 1 + n_augments   # original + augmented versions


    fig = plt.figure(figsize=(cols * 3, n_rows * 3 + 1))

    fig.suptitle("Augmentation Samples", fontsize=14, fontweight="bold", y=0.95)

    gs = gridspec.GridSpec(n_rows, cols, figure=fig)

    fig.subplots_adjust(top=0.92, hspace=0.3, wspace=0.1)

    for row_idx, cls in enumerate(selected_classes):
        cls_paths = train_df[train_df["label"] == cls]["path"].tolist()
        img_path = random.choice(cls_paths)

        # Load original
        raw = tf.io.read_file(img_path)
        orig_img = tf.image.decode_jpeg(raw, channels=3)
        orig_img = tf.image.resize(orig_img, IMG_SIZE)
        orig_img = tf.cast(orig_img, tf.float32) / 255.0

        # Plot original
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(orig_img.numpy())
        ax.set_title(f"{CLASS_DISPLAY_NAMES.get(cls, cls)}\n(original)", fontsize=7)
        ax.axis("off")

        # Plot augmented versions
        for col_idx in range(1, cols):
            aug_img, _ = augment(orig_img, 0)
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(aug_img.numpy())
            ax.set_title(f"aug #{col_idx}", fontsize=7)
            ax.axis("off")

    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Augmentation samples saved → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect paths
    print(f"Scanning {data_dir} for the 10 locked classes …")
    df = collect_image_paths(data_dir)
    if df.empty:
        raise FileNotFoundError(
            f"No images found under {data_dir} for the locked classes.\n"
            "Make sure you've downloaded PlantVillage from Kaggle and unzipped it.\n"
            "Expected layout: <data_dir>/<ClassName>/<image.jpg>"
        )
    print(f"Found {len(df):,} images across {df['label'].nunique()} classes.")

    # 2. Stratified split
    train_df, val_df, test_df = stratified_split(df)

    # 3. Per-class table + class weights
    needs_weighting, class_weights = print_class_table(train_df, val_df, test_df)

    # 4. Save splits
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir  / "val.csv",   index=False)
    test_df.to_csv(output_dir / "test.csv",  index=False)
    print(f"Splits saved → {output_dir}/{{train,val,test}}.csv")

    # 5. Save class weights if needed
    if needs_weighting and class_weights:
        import json
        weights_path = output_dir / "class_weights.json"
        json.dump(class_weights, open(weights_path, "w"), indent=2)
        print(f"Class weights saved → {weights_path}")

    # 6. Build datasets (smoke-test)
    class_names = sorted(df["label"].unique())
    train_ds = build_dataset(train_df, class_names, augment_data=True,  shuffle=True)
    val_ds   = build_dataset(val_df,   class_names, augment_data=False, shuffle=False)
    test_ds  = build_dataset(test_df,  class_names, augment_data=False, shuffle=False)

    print(f"\ntf.data datasets ready:")
    print(f"  train_ds : {len(train_df):>6,} images  ({len(train_ds):>4} batches)")
    print(f"  val_ds   : {len(val_df):>6,} images  ({len(val_ds):>4} batches)")
    print(f"  test_ds  : {len(test_df):>6,} images  ({len(test_ds):>4} batches)")

    # 7. Save augmentation samples
    aug_out = args.aug_output
    os.makedirs(os.path.dirname(aug_out), exist_ok=True)
    save_augmentation_samples(train_df, class_names, output_path=aug_out)

    # 8. Return datasets (usable when imported from train.py)
    return train_ds, val_ds, test_ds, class_names, class_weights


def get_datasets(
    data_dir: str = "data/plant_village",
    splits_dir: str = "data/splits",
):
    """
    Convenience function for import from train.py / evaluate.py.
    Loads pre-saved CSVs if they exist; otherwise runs the full pipeline.
    Returns (train_ds, val_ds, test_ds, class_names, class_weights_or_None).
    """
    import json
    splits_dir = Path(splits_dir)
    train_csv  = splits_dir / "train.csv"
    val_csv    = splits_dir / "val.csv"
    test_csv   = splits_dir / "test.csv"
    weights_f  = splits_dir / "class_weights.json"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(
            f"Split CSVs not found in {splits_dir}. "
            "Run `python src/data_prep.py` first."
        )

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)
    class_names   = sorted(train_df["label"].unique())
    class_weights = json.load(open(weights_f)) if weights_f.exists() else None

    train_ds = build_dataset(train_df, class_names, augment_data=True,  shuffle=True)
    val_ds   = build_dataset(val_df,   class_names, augment_data=False, shuffle=False)
    test_ds  = build_dataset(test_df,  class_names, augment_data=False, shuffle=False)

    return train_ds, val_ds, test_ds, class_names, class_weights


# ─── Entry-point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zar3y — Requirement 1: Data prep & augmentation")
    parser.add_argument(
        "--data_dir", default="data/plant_village",
        help="Root directory of the downloaded PlantVillage dataset"
    )
    parser.add_argument(
        "--output_dir", default="data/splits",
        help="Where to save train/val/test CSV splits"
    )
    parser.add_argument(
        "--aug_output", default="outputs/augmentation_samples.png",
        help="Path for augmentation sample grid PNG"
    )
    args = parser.parse_args()
    main(args)