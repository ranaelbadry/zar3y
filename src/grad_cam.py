"""
src/grad_cam.py
Zar3y – Crop Disease Detection
Requirement 4: Grad-CAM Explainability + OOD Field-Photo Test

What this file does:
  1. Loads best_model.keras
  2. Auto-detects the last Conv2D layer inside MobileNetV3Small
  3. Generates Grad-CAM overlays for 5 representative test images
  4. Runs inference on the OOD field-photo set
  5. Saves all outputs to outputs/grad_cam_examples/ and outputs/ood_report.json
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ──────────────────────────────────────────────
# 0. PATHS  (all relative to project root)
# ──────────────────────────────────────────────
PROJECT_ROOT      = Path('/content/Zar3y')
MODEL_PATH        = PROJECT_ROOT / 'models' / 'best_model.keras'
LABEL_JSON        = PROJECT_ROOT / 'models' / 'label2id.json'
DATA_DIR          = PROJECT_ROOT / 'data' / 'plant_village'
FIELD_PHOTOS_DIR  = PROJECT_ROOT / 'data' / 'field_photos'
OUT_GRADCAM_DIR   = PROJECT_ROOT / 'outputs' / 'grad_cam_examples'
OOD_REPORT_PATH   = PROJECT_ROOT / 'outputs' / 'ood_report.json'

IMG_SIZE   = (224, 224)
SEED       = 42

# ──────────────────────────────────────────────
# 1. LOAD MODEL + CLASS NAMES
# ──────────────────────────────────────────────
print("Loading model …")
model = tf.keras.models.load_model(str(MODEL_PATH))

with open(str(LABEL_JSON), 'r') as f:
    label2id = json.load(f)

# id2label: index → human-readable name
id2label = {v: k for k, v in label2id.items()}
CLASS_NAMES = [id2label[i] for i in range(len(id2label))]
print("Classes:", CLASS_NAMES)

# ──────────────────────────────────────────────
# 2. AUTO-DETECT LAST CONV LAYER
# ──────────────────────────────────────────────
def get_last_conv_layer(model):
    """
    MobileNetV3Small is nested as a sub-model.
    We dig into it to find the last Conv2D layer.
    """
    # Try to find nested MobileNetV3 sub-model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    print(f"Auto-detected last conv layer: '{sub_layer.name}' "
                          f"inside '{layer.name}'")
                    return layer.name, sub_layer.name

    # Fallback: search top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Auto-detected last conv layer (top-level): '{layer.name}'")
            return None, layer.name

    raise ValueError("No Conv2D layer found in the model.")

BACKBONE_NAME, LAST_CONV_NAME = get_last_conv_layer(model)

def backbone_has_builtin_preprocessing(model, backbone_name):
    """Return True if the nested backbone already rescales raw 0-255 pixels."""
    backbone = model.get_layer(backbone_name)
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            break
        if isinstance(layer, (tf.keras.layers.Rescaling,
                              tf.keras.layers.Normalization)):
            print(f"Detected built-in preprocessing layer: '{layer.name}'. "
                  "Feeding raw 0-255 pixels.")
            return True
    print("No built-in preprocessing detected before the first conv layer. "
          "Feeding pixels normalized to 0-1.")
    return False

USE_RAW_PIXELS = backbone_has_builtin_preprocessing(model, BACKBONE_NAME)

# ──────────────────────────────────────────────
# 3. IMAGE PREPROCESSING
# ──────────────────────────────────────────────
def preprocess(img_path):
    """Load an image from disk and return (display_img, model_input_tensor)."""
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)          # float32, 0-255
    display   = img_array.astype('uint8')
    model_input = img_array if USE_RAW_PIXELS else img_array / 255.0
    tensor    = tf.expand_dims(model_input, axis=0)
    return display, tensor

# ──────────────────────────────────────────────
# 4. GRAD-CAM CORE
# ──────────────────────────────────────────────
def make_gradcam_heatmap(img_tensor, model, backbone_name, last_conv_name):
    """
    Returns a (224,224) float32 heatmap in [0,1].
    Works with a nested sub-model (MobileNetV3Small inside our model).
    """
    backbone = model.get_layer(backbone_name)
    last_conv_layer = backbone.get_layer(last_conv_name)

    # Keras 3 can raise a Functional.call tensor_dict KeyError if we try to
    # expose a nested sub-model tensor directly from the outer model graph.
    # Instead, run the backbone as a feature extractor and then apply the
    # top-level classifier head layers that come after the backbone.
    feature_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv_layer.output, backbone.output],
        name='gradcam_feature_model'
    )

    head_layers = []
    found_backbone = False
    for layer in model.layers:
        if layer.name == backbone_name:
            found_backbone = True
            continue
        if found_backbone:
            head_layers.append(layer)

    if not head_layers:
        raise ValueError(
            f"No classifier head layers found after backbone '{backbone_name}'."
        )

    with tf.GradientTape() as tape:
        conv_outputs, x = feature_model(img_tensor, training=False)
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x

        # Gradient w.r.t. the top predicted class
        pred_index  = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)                 # (1,H,W,C)
    if grads is None:
        raise RuntimeError(
            'Could not compute Grad-CAM gradients. Check that the detected '
            f"layer '{last_conv_name}' is connected to the classifier output."
        )

    pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))               # (C,)
    conv_out   = conv_outputs[0]                                     # (H,W,C)
    heatmap    = conv_out @ pooled[..., tf.newaxis]                  # (H,W,1)
    heatmap    = tf.squeeze(heatmap)
    heatmap    = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy()), predictions.numpy()[0]


def overlay_gradcam(display_img, heatmap, alpha=0.4):
    """Resize heatmap to image size and blend as a colour overlay."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_color,
                                  (display_img.shape[1], display_img.shape[0]))
    overlay = (display_img * (1 - alpha) + heatmap_resized * alpha).astype('uint8')
    return overlay


def run_gradcam(img_path, model, backbone_name, last_conv_name):
    """Full pipeline: load → predict → Grad-CAM. Returns dict with all info."""
    display_img, tensor = preprocess(img_path)
    heatmap, pred_idx, probs = make_gradcam_heatmap(
        tensor, model, backbone_name, last_conv_name)
    overlay  = overlay_gradcam(display_img, heatmap)
    confidence = float(probs[pred_idx])
    label      = CLASS_NAMES[pred_idx]
    return {
        'display_img': display_img,
        'overlay':     overlay,
        'heatmap':     heatmap,
        'pred_idx':    pred_idx,
        'label':       label,
        'confidence':  confidence,
        'probs':       probs,
    }

# ──────────────────────────────────────────────
# 5. SAVE SIDE-BY-SIDE FIGURE
# ──────────────────────────────────────────────
def save_side_by_side(result, save_path, true_label=None, annotate_failure=False):
    """Save original | Grad-CAM overlay as a single PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(result['display_img'])
    axes[0].set_title('Original', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(result['overlay'])
    title = (f"Grad-CAM\nPred: {result['label']}\n"
             f"Conf: {result['confidence']*100:.1f}%")
    color = 'black'
    if true_label is not None:
        match = (true_label == result['label'])
        title += f"\nTrue: {true_label}"
        if annotate_failure and not match:
            title += "  ✗ FAILURE"
            color  = 'red'
    axes[1].set_title(title, fontsize=11, color=color)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

# ──────────────────────────────────────────────
# 6. REQUIREMENT 4-A: 5 REPRESENTATIVE TEST IMAGES
# ──────────────────────────────────────────────
def get_one_image_per_class(data_dir, class_names, seed=42):
    """
    Pick one image per class from the plant_village folder.
    Returns list of (img_path, true_label) tuples.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for cls in class_names:
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            print(f"  WARNING: folder not found for class '{cls}', skipping.")
            continue
        imgs = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.JPG')) + \
               list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.jpeg'))
        if not imgs:
            print(f"  WARNING: no images found in {cls_dir}, skipping.")
            continue
        chosen = imgs[rng.integers(len(imgs))]
        samples.append((str(chosen), cls))
    return samples


def run_representative_gradcam():
    print("\n── PART A: Representative Grad-CAM Examples ──")
    OUT_GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    samples = get_one_image_per_class(DATA_DIR, CLASS_NAMES, seed=SEED)
    print(f"  Selected {len(samples)} representative images (one per class).")

    for img_path, true_label in samples:
        result = run_gradcam(img_path, model, BACKBONE_NAME, LAST_CONV_NAME)
        safe_name = true_label.replace('/', '_').replace(',', '').replace(' ', '_')
        save_path = OUT_GRADCAM_DIR / f"{safe_name}.png"
        save_side_by_side(result, str(save_path), true_label=true_label)

    print(f"  All Grad-CAM examples saved to {OUT_GRADCAM_DIR}/")

# ──────────────────────────────────────────────
# 7. REQUIREMENT 4-B: OOD FIELD-PHOTO TEST
# ──────────────────────────────────────────────
# Plain-language disease descriptions (static dict — no LLM)
DISEASE_INFO = {
    'Corn_(maize)___Common_rust_': {
        'description': 'Common rust causes small, powdery, reddish-brown pustules on both leaf surfaces.',
        'next_step':   'Apply fungicide (e.g. propiconazole) early. Remove severely infected leaves.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial spot produces water-soaked, dark lesions on leaves and fruit.',
        'next_step':   'Use copper-based bactericide. Avoid overhead irrigation.'
    },
    'Pepper,_bell___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'next_step':   'Continue regular monitoring and good agronomic practices.'
    },
    'Potato___Early_blight': {
        'description': 'Early blight shows dark, target-like concentric rings on older leaves.',
        'next_step':   'Apply chlorothalonil or mancozeb fungicide. Remove infected foliage.'
    },
    'Potato___Late_blight': {
        'description': 'Late blight causes large, irregular, water-soaked dark lesions.',
        'next_step':   'Apply metalaxyl fungicide immediately. Destroy infected plants.'
    },
    'Potato___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'next_step':   'Continue regular monitoring and good agronomic practices.'
    },
    'Tomato___Early_blight': {
        'description': 'Early blight causes dark concentric rings on lower, older leaves.',
        'next_step':   'Apply mancozeb or copper fungicide. Improve air circulation.'
    },
    'Tomato___Late_blight': {
        'description': 'Late blight appears as dark, water-soaked lesions on leaves and stems.',
        'next_step':   'Apply metalaxyl or cymoxanil immediately. Remove and destroy infected parts.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Leaf mold causes yellow patches on upper leaf surface with olive mold below.',
        'next_step':   'Improve ventilation. Apply copper-based fungicide.'
    },
    'Tomato___healthy': {
        'description': 'No disease detected. The plant appears healthy.',
        'next_step':   'Continue regular monitoring and good agronomic practices.'
    },
}


def collect_field_photos(field_dir):
    """
    Collect all images from field_photos/.
    Structure can be flat or in subfolders (by crop name).
    Returns list of (img_path, true_label_or_None).
    """
    field_dir = Path(field_dir)
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    photos = []

    for p in sorted(field_dir.rglob('*')):
        if p.suffix in extensions:
            # If stored in a subfolder named after a class, use it as true label
            parent = p.parent.name
            true_label = parent if parent in label2id else None
            photos.append((str(p), true_label))

    return photos


def run_ood_test():
    print("\n── PART B: OOD Field-Photo Test ──")
    OUT_GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    if not FIELD_PHOTOS_DIR.exists():
        print(f"  ERROR: {FIELD_PHOTOS_DIR} does not exist.")
        print("  Please create it and add 30-50 field photos, then re-run.")
        return

    photos = collect_field_photos(FIELD_PHOTOS_DIR)
    print(f"  Found {len(photos)} field photos.")

    if len(photos) == 0:
        print("  No photos found. Add images to data/field_photos/ and re-run.")
        return

    results      = []
    correct      = 0
    labeled      = 0
    failures     = []   # (result, true_label, img_path)

    for img_path, true_label in photos:
        try:
            result = run_gradcam(img_path, model, BACKBONE_NAME, LAST_CONV_NAME)
        except Exception as e:
            print(f"  SKIP {img_path}: {e}")
            continue

        is_correct = None
        if true_label is not None:
            labeled  += 1
            is_correct = (result['label'] == true_label)
            if is_correct:
                correct += 1
            else:
                failures.append((result, true_label, img_path))

        results.append({
            'image':      img_path,
            'true_label': true_label,
            'pred_label': result['label'],
            'confidence': round(result['confidence'], 4),
            'correct':    is_correct,
            'description': DISEASE_INFO.get(result['label'], {}).get('description', ''),
            'next_step':   DISEASE_INFO.get(result['label'], {}).get('next_step', ''),
        })

        true_text = f" true={true_label:35s}" if true_label is not None else ""
        mark_text = f"  {'OK' if is_correct else 'FAIL'}" if is_correct is not None else ""

        print(f"  {Path(img_path).name:40s} -> pred={result['label']:35s} "
              f"({result['confidence']*100:.1f}%){true_text}{mark_text}")

    # ── OOD accuracy
    ood_accuracy = correct / labeled if labeled > 0 else None
    print(f"\n  OOD Top-1 Accuracy: {correct}/{labeled} = "
          f"{ood_accuracy*100:.1f}%" if ood_accuracy is not None
          else "\n  No labeled photos — accuracy not computed.")

    # ── Save 3 annotated failure cases
    failure_dir = OUT_GRADCAM_DIR / 'failure_cases'
    failure_dir.mkdir(parents=True, exist_ok=True)
    for i, (result, true_label, img_path) in enumerate(failures[:3]):
        save_path = failure_dir / f"failure_{i+1}_{Path(img_path).stem}.png"
        save_side_by_side(result, str(save_path),
                          true_label=true_label, annotate_failure=True)

    # ── Save OOD report JSON
    report = {
        'total_field_photos':   len(results),
        'labeled_photos':       labeled,
        'correct_predictions':  correct,
        'ood_top1_accuracy':    round(ood_accuracy, 4) if ood_accuracy else None,
        'failure_cases_saved':  min(3, len(failures)),
        'per_image_results':    results,
    }
    with open(str(OOD_REPORT_PATH), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  OOD report saved → {OOD_REPORT_PATH}")
    print(f"  Failure cases saved → {failure_dir}/")

    return report

# ──────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────
if __name__ == '__main__':
    run_representative_gradcam()
    run_ood_test()
    print("\n✓ Requirement 4 complete.")
