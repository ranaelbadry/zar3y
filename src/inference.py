import base64
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from settings import (
    CLASS_DISPLAY_NAMES,
    DISEASE_INFO,
    DYNAMIC_MODEL_PATH,
    IMG_SIZE,
    KERAS_MODEL_PATH,
    LABELS_PATH,
    QUANTIZED_MODEL_PATH,
)


def _load_class_names() -> list[str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    return [label for label, _ in sorted(label2id.items(), key=lambda item: item[1])]


CLASS_NAMES = _load_class_names()

_model_path = Path(os.getenv("ZAR3Y_MODEL_PATH", str(QUANTIZED_MODEL_PATH)))
if not _model_path.exists():
    _model_path = Path(os.getenv("ZAR3Y_FALLBACK_MODEL_PATH", str(DYNAMIC_MODEL_PATH)))
if not _model_path.exists():
    raise FileNotFoundError(
        f"No TFLite model found. Expected {QUANTIZED_MODEL_PATH} or {DYNAMIC_MODEL_PATH}."
    )

try:
    interpreter = tf.lite.Interpreter(model_path=str(_model_path))
    interpreter.allocate_tensors()
    _input_details = interpreter.get_input_details()
    _output_details = interpreter.get_output_details()
    _tflite_error = None
except Exception as exc:
    interpreter = None
    _input_details = None
    _output_details = None
    _tflite_error = str(exc)

_keras_model = None
_backbone_name = None
_last_conv_name = None
_keras_uses_raw_pixels = True


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    if _input_details is None:
        return arr

    input_detail = _input_details[0]
    if input_detail["dtype"] in (np.uint8, np.int8):
        scale, zero_point = input_detail["quantization"]
        if scale:
            arr = arr / scale + zero_point
        arr = np.clip(arr, np.iinfo(input_detail["dtype"]).min, np.iinfo(input_detail["dtype"]).max)
        arr = arr.astype(input_detail["dtype"])
    else:
        arr = arr.astype(input_detail["dtype"])

    return arr


def _dequantize_output(output: np.ndarray) -> np.ndarray:
    if _output_details is None:
        return output.astype(np.float32)
    output_detail = _output_details[0]
    if output_detail["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_detail["quantization"]
        if scale:
            output = (output.astype(np.float32) - zero_point) * scale
    return output.astype(np.float32)


def _find_last_conv_layer(model: tf.keras.Model) -> tuple[str | None, str]:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer.name, sub_layer.name
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def _get_keras_model() -> tf.keras.Model:
    global _keras_model, _backbone_name, _last_conv_name, _keras_uses_raw_pixels
    if _keras_model is None:
        keras_path = Path(os.getenv("ZAR3Y_KERAS_MODEL_PATH", str(KERAS_MODEL_PATH)))
        if not keras_path.exists():
            raise FileNotFoundError(f"Keras model not found for Grad-CAM: {keras_path}")
        _keras_model = tf.keras.models.load_model(str(keras_path), compile=False)
        _backbone_name, _last_conv_name = _find_last_conv_layer(_keras_model)
        if _backbone_name is not None:
            backbone = _keras_model.get_layer(_backbone_name)
            _keras_uses_raw_pixels = any(
                isinstance(layer, (tf.keras.layers.Rescaling, tf.keras.layers.Normalization))
                for layer in backbone.layers[:5]
            )
    return _keras_model


def _keras_input(image: Image.Image) -> tuple[np.ndarray, tf.Tensor]:
    _get_keras_model()
    rgb = image.convert("RGB").resize(IMG_SIZE)
    display = np.asarray(rgb).astype("uint8")
    arr = display.astype(np.float32)
    if not _keras_uses_raw_pixels:
        arr = arr / 255.0
    tensor = tf.expand_dims(arr, axis=0)
    return display, tensor


def _make_gradcam_heatmap(img_tensor: tf.Tensor, model: tf.keras.Model) -> tuple[np.ndarray, int, np.ndarray]:
    if _backbone_name is None:
        layer = model.get_layer(_last_conv_name)
        grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]
    else:
        backbone = model.get_layer(_backbone_name)
        conv_layer = backbone.get_layer(_last_conv_name)
        feature_model = tf.keras.Model(backbone.inputs, [conv_layer.output, backbone.output])
        head_layers = []
        found_backbone = False
        for layer in model.layers:
            if layer.name == _backbone_name:
                found_backbone = True
                continue
            if found_backbone:
                head_layers.append(layer)
        with tf.GradientTape() as tape:
            conv_outputs, x = feature_model(img_tensor, training=False)
            for layer in head_layers:
                x = layer(x, training=False)
            predictions = x
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy()), predictions.numpy()[0]


def _overlay_gradcam(display_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_color, (display_img.shape[1], display_img.shape[0]))
    overlay = (display_img * (1 - alpha) + heatmap_resized * alpha).astype("uint8")
    return Image.fromarray(overlay)


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_gradcam_overlay(image: Image.Image) -> str | None:
    try:
        model = _get_keras_model()
        display, tensor = _keras_input(image)
        heatmap, _, _ = _make_gradcam_heatmap(tensor, model)
        return _image_to_base64(_overlay_gradcam(display, heatmap))
    except Exception:
        return None


def predict(image: Image.Image) -> dict:
    if interpreter is not None:
        arr = preprocess_image(image)
        interpreter.set_tensor(_input_details[0]["index"], arr)
        interpreter.invoke()
        output = _dequantize_output(interpreter.get_tensor(_output_details[0]["index"])[0])
        runtime_model = str(_model_path)
    else:
        model = _get_keras_model()
        _, tensor = _keras_input(image)
        output = model.predict(tensor, verbose=0)[0].astype(np.float32)
        runtime_model = f"{KERAS_MODEL_PATH} (TFLite fallback: {_tflite_error})"

    idx = int(np.argmax(output))
    label = CLASS_NAMES[idx]
    confidence = float(output[idx])
    info = DISEASE_INFO.get(label, {})

    return {
        "label": label,
        "class": CLASS_DISPLAY_NAMES.get(label, label),
        "confidence": confidence,
        "description": info.get("description", "No information available."),
        "action": info.get("action", "Please consult an agricultural expert."),
        "gradcam_overlay": generate_gradcam_overlay(image),
        "model_path": runtime_model,
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    result = predict(Image.open(path))
    print(f"Class: {result['class']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
