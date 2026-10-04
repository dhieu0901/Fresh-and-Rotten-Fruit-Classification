"""Model 1: plain Sequential CNN (5 x Conv-ReLU-MaxPool, then dense layers) with one softmax over the 16 classes."""

import keras
from keras import layers

from src.config import IMG_SIZE, NUM_COMBINED


def build_simple_cnn(
    img_size=IMG_SIZE, num_classes=NUM_COMBINED, filters=(32, 64, 128, 128, 256), dense_units=256, dropout=0.5
):
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
