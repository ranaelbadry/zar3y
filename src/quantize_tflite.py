import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DYNAMIC_MODEL_PATH, KERAS_MODEL_PATH, PLANT_VILLAGE_DIR, QUANTIZED_MODEL_PATH

IMG_SIZE = (224, 224)


def representative_data_gen(data_dir: Path, limit: int = 200):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=1,
        shuffle=True,
        seed=42,
    )
    for images, _ in dataset.take(limit):
        yield [tf.cast(images, tf.float32) / 255.0]


def export_saved_model(model: tf.keras.Model, export_dir: Path) -> None:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    model.export(str(export_dir))


def convert_dynamic(saved_model_dir: Path, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    out_path.write_bytes(converter.convert())


def convert_int8(saved_model_dir: Path, out_path: Path, data_dir: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(data_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    out_path.write_bytes(converter.convert())


def _predict_tflite(model_path: Path, images: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    preds = []
    for image in images:
        batch = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)
        if input_detail["dtype"] in (np.uint8, np.int8):
            scale, zero_point = input_detail["quantization"]
            batch = batch / scale + zero_point if scale else batch
            batch = np.clip(batch, np.iinfo(input_detail["dtype"]).min, np.iinfo(input_detail["dtype"]).max)
        interpreter.set_tensor(input_detail["index"], batch.astype(input_detail["dtype"]))
        started = time.perf_counter()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - started) * 1000
        output = interpreter.get_tensor(output_detail["index"])[0]
        if output_detail["dtype"] in (np.uint8, np.int8):
            scale, zero_point = output_detail["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale if scale else output
        preds.append((output, elapsed_ms))
    return preds


def benchmark(model: tf.keras.Model, model_path: Path, dynamic_path: Path, int8_path: Path, data_dir: Path, output_dir: Path) -> None:
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=100,
        shuffle=True,
        seed=42,
    )
    images, labels = next(iter(ds))
    images_np = images.numpy()
    labels_np = labels.numpy()

    started = time.perf_counter()
    float_preds = model.predict(images_np / 255.0, verbose=0)
    float_latency = ((time.perf_counter() - started) * 1000) / len(images_np)
    float_acc = float(np.mean(np.argmax(float_preds, axis=1) == labels_np))

    dyn_preds = _predict_tflite(dynamic_path, images_np)
    int8_preds = _predict_tflite(int8_path, images_np)
    dyn_acc = float(np.mean([np.argmax(p[0]) for p in dyn_preds] == labels_np))
    int8_acc = float(np.mean([np.argmax(p[0]) for p in int8_preds] == labels_np))

    report = {
        "float_model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 3),
        "dynamic_tflite_size_mb": round(dynamic_path.stat().st_size / (1024 * 1024), 3),
        "int8_tflite_size_mb": round(int8_path.stat().st_size / (1024 * 1024), 3),
        "float_latency_ms": round(float_latency, 3),
        "dynamic_tflite_latency_ms": round(float(np.mean([p[1] for p in dyn_preds])), 3),
        "int8_tflite_latency_ms": round(float(np.mean([p[1] for p in int8_preds])), 3),
        "float_accuracy_sample": round(float_acc, 4),
        "dynamic_tflite_accuracy_sample": round(dyn_acc, 4),
        "int8_accuracy_sample": round(int8_acc, 4),
        "int8_accuracy_delta_sample": round(int8_acc - float_acc, 4),
        "sample_size": int(len(images_np)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "quantization_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=str(KERAS_MODEL_PATH))
    parser.add_argument("--data_dir", default=str(PLANT_VILLAGE_DIR))
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    os.makedirs("models", exist_ok=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    saved_model_dir = Path("models") / "saved_model_export"
    export_saved_model(model, saved_model_dir)
    convert_dynamic(saved_model_dir, DYNAMIC_MODEL_PATH)
    convert_int8(saved_model_dir, QUANTIZED_MODEL_PATH, data_dir)
    benchmark(model, model_path, DYNAMIC_MODEL_PATH, QUANTIZED_MODEL_PATH, data_dir, output_dir)
    print(f"Saved {DYNAMIC_MODEL_PATH}")
    print(f"Saved {QUANTIZED_MODEL_PATH}")
    print(f"Saved {output_dir / 'quantization_benchmark.json'}")


if __name__ == "__main__":
    main()
