import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image


SEED = 42


def build_model(n_classes=10):

    base = keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))

    x = base(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.2, seed=SEED)(x)

    outputs = layers.Dense(
        n_classes,
        activation="softmax"
    )(x)

    model = keras.Model(inputs, outputs)

    return model


# Load model
model = build_model()

model.load_weights("models/best_model.weights.h5")

print("Model loaded successfully!")


CLASS_NAMES = [
    "Corn Common Rust",
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold"
]


def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32)

    image = np.expand_dims(image, axis=0)

    return image


def predict(image_path):

    image = preprocess_image(image_path)

    prediction = model.predict(image)

    predicted_index = np.argmax(prediction)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        "confidence": confidence
    }


if __name__ == "__main__":

    result = predict("test.jpg")

    print(result)
