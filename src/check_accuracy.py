import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

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


# Load original model
model = build_model()
model.load_weights("models/best_model.weights.h5")

print("Original model loaded")


# Random test image
test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Original prediction
original_pred = model.predict(test_image)

print("\nOriginal model prediction:")
print(original_pred)


# Load TFLite model
#interpreter = tf.lite.Interpreter(
#    model_path="models/model.tflite"
#)


interpreter = tf.lite.Interpreter(
    model_path="models/model.tflite",
    experimental_delegates=[]
)



interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input
interpreter.set_tensor(
    input_details[0]['index'],
    test_image
)

# Run inference
interpreter.invoke()

# Get output
tflite_pred = interpreter.get_tensor(
    output_details[0]['index']
)

print("\nTFLite model prediction:")
print(tflite_pred)


# Compare predictions
difference = np.mean(
    np.abs(original_pred - tflite_pred)
)

print(f"\nAverage difference: {difference}")