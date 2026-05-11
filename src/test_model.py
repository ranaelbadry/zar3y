import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


model = build_model()

model.load_weights("models/best_model.weights.h5")

print("Model + weights loaded successfully!")

model.summary()
