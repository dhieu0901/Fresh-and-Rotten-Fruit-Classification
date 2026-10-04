"""Model 1 - simple sequential CNN baseline.

Stacked Conv2D-ReLU-MaxPool blocks followed by fully connected layers (Keras Sequential API).
It predicts the 16 combined classes (8 fruits x fresh/spoiled) with one softmax; fruit type
and freshness are decoded from the combined label afterwards (fruit = c // 2, spoiled = c % 2).

    224x224x3 -> Rescaling(1/255)
      -> [Conv 3x3 (32) -> ReLU -> MaxPool 2x2]   112x112x32
      -> [Conv 3x3 (64) -> ReLU -> MaxPool 2x2]    56x56x64
      -> [Conv 3x3 (128) -> ReLU -> MaxPool 2x2]   28x28x128
      -> [Conv 3x3 (128) -> ReLU -> MaxPool 2x2]   14x14x128
      -> [Conv 3x3 (256) -> ReLU -> MaxPool 2x2]    7x7x256
      -> Flatten (12544) -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(16, softmax)
"""
import keras
from keras import layers

from src.config import IMG_SIZE, NUM_COMBINED


def build_simple_cnn(img_size=IMG_SIZE, num_classes=NUM_COMBINED, filters=(32, 64, 128, 128, 256),
                     dense_units=256, dropout=0.5):
    model = keras.Sequential(name="model1_simple_cnn")
    model.add(keras.Input(shape=(img_size, img_size, 3), name="image"))
    model.add(layers.Rescaling(1.0 / 255, name="rescale"))
    for i, n_filters in enumerate(filters, start=1):
        model.add(layers.Conv2D(n_filters, 3, padding="same", activation="relu", name=f"conv{i}"))
        model.add(layers.MaxPooling2D(2, name=f"pool{i}"))
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(dense_units, activation="relu", name="fc1"))
    model.add(layers.Dropout(dropout, name="dropout"))
    model.add(layers.Dense(num_classes, activation="softmax", name="combined"))
    return model
