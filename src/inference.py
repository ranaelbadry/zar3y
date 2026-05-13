import numpy as np
import tensorflow as tf
from PIL import Image

# Class names ordered by index from label2id.json
CLASS_NAMES = [
    "Corn Common Rust",            # 0
    "Pepper Bell Bacterial Spot",  # 1
    "Pepper Bell Healthy",         # 2
    "Potato Early Blight",         # 3
    "Potato Late Blight",          # 4
    "Potato Healthy",              # 5
    "Tomato Early Blight",         # 6
    "Tomato Late Blight",          # 7
    "Tomato Leaf Mold",            # 8
    "Tomato Healthy",              # 9
]

# Load TFLite model once at startup
interpreter = tf.lite.Interpreter(model_path="models/model_dynamic.tflite")
interpreter.allocate_tensors()

_input_details  = interpreter.get_input_details()
_output_details = interpreter.get_output_details()

print("Model loaded successfully!")


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image) -> dict:
    arr = preprocess_image(image)
    interpreter.set_tensor(_input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(_output_details[0]["index"])[0]
    idx = int(np.argmax(output))
    return {
        "class": CLASS_NAMES[idx],
        "confidence": float(output[idx]),
    }


if __name__ == "__main__":
    # Quick test: python src/inference.py path/to/leaf.jpg
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    result = predict(Image.open(path))
    print(f"Class:      {result['class']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")