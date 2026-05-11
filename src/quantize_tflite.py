import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

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


# Build model
model = build_model()

# Load weights
model.load_weights("models/best_model.weights.h5")

print("Model loaded successfully!")

# Create TFLite converter
#converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert model
#tflite_model = converter.convert()

# Representative dataset for INT8 quantization
def representative_data_gen():

    dataset = tf.keras.utils.image_dataset_from_directory(
        "data/plant_village",
        image_size=(224, 224),
        batch_size=1,
        shuffle=True
    )

    for images, _ in dataset.take(200):
        images = tf.cast(images, tf.float32)
        yield [images]


# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset
converter.representative_dataset = representative_data_gen

# INT8 quantization
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8


converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]


# Convert model
tflite_model = converter.convert()


# Save TFLite model
os.makedirs("models", exist_ok=True)

with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")
