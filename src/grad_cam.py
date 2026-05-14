import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import CLASS_NAMES, predict
from settings import FIELD_PHOTOS_DIR, OUTPUTS_DIR, PLANT_VILLAGE_DIR


def save_side_by_side(image_path: Path, result: dict, output_path: Path, true_label: str | None = None) -> None:
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    overlay_b64 = result.get("gradcam_overlay")
    if overlay_b64 is None:
        return

    import base64
    import io

    overlay = Image.open(io.BytesIO(base64.b64decode(overlay_b64)))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    title = f"Grad-CAM\nPred: {result['label']}\nConf: {result['confidence'] * 100:.1f}%"
    if true_label:
        title += f"\nTrue: {true_label}"
    axes[1].set_title(title)
    axes[1].axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def first_image(folder: Path) -> Path | None:
    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
        images = sorted(folder.glob(pattern))
        if images:
            return images[0]
    return None


def run_representative_examples(data_dir: Path, output_dir: Path) -> None:
    gradcam_dir = output_dir / "grad_cam_examples"
    for label in CLASS_NAMES:
        image_path = first_image(data_dir / label)
        if image_path is None:
            continue
        result = predict(Image.open(image_path))
        save_side_by_side(image_path, result, gradcam_dir / f"{label.replace(',', '')}.png", label)


def run_ood(field_dir: Path, output_dir: Path) -> None:
    results = []
    failures = []
    correct = 0
    labeled = 0
    for image_path in sorted(field_dir.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        true_label = image_path.parent.name if image_path.parent.name in CLASS_NAMES else None
        result = predict(Image.open(image_path))
        is_correct = None
        if true_label is not None:
            labeled += 1
            is_correct = result["label"] == true_label
            correct += int(is_correct)
            if not is_correct:
                failures.append((image_path, result, true_label))
        results.append({
            "image": str(image_path),
            "true_label": true_label,
            "pred_label": result["label"],
            "confidence": round(result["confidence"], 4),
            "correct": is_correct,
            "description": result["description"],
            "next_step": result["action"],
        })

    failure_dir = output_dir / "failure_cases"
    for index, (image_path, result, true_label) in enumerate(failures[:3], start=1):
        save_side_by_side(image_path, result, failure_dir / f"failure_{index}_{image_path.stem}.png", true_label)

    report = {
        "total_field_photos": len(results),
        "labeled_photos": labeled,
        "correct_predictions": correct,
        "ood_top1_accuracy": round(correct / labeled, 4) if labeled else None,
        "failure_cases_saved": min(3, len(failures)),
        "per_image_results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ood_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(PLANT_VILLAGE_DIR))
    parser.add_argument("--field_dir", default=str(FIELD_PHOTOS_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUTS_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    run_representative_examples(Path(args.data_dir), output_dir)
    run_ood(Path(args.field_dir), output_dir)
    print("Grad-CAM and OOD reports saved.")


if __name__ == "__main__":
    main()
