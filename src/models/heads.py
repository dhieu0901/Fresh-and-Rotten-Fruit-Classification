"""The two output heads shared by Model 2 and Model 3 (multi-task learning).

    feature map -> GlobalAveragePooling2D -+-> Dense(128, ReLU) -> Dropout -> Dense(8, softmax)  "fruit"
                                           +-> Dense(64, ReLU)  -> Dropout -> Dense(1, sigmoid)  "freshness"

The freshness output is P(spoiled); spoiled is the positive class.
"""
from keras import layers

from src.config import NUM_FRUITS

HEAD_LAYER_NAMES = ("gap", "fruit_fc", "fruit_dropout", "fruit", "freshness_fc", "freshness_dropout", "freshness")


def build_heads(features, dropout=0.3, fruit_units=128, freshness_units=64):
    pooled = layers.GlobalAveragePooling2D(name="gap")(features)

    fruit = layers.Dense(fruit_units, activation="relu", name="fruit_fc")(pooled)
    fruit = layers.Dropout(dropout, name="fruit_dropout")(fruit)
    fruit = layers.Dense(NUM_FRUITS, activation="softmax", name="fruit")(fruit)

    fresh = layers.Dense(freshness_units, activation="relu", name="freshness_fc")(pooled)
    fresh = layers.Dropout(dropout, name="freshness_dropout")(fresh)
    fresh = layers.Dense(1, activation="sigmoid", name="freshness")(fresh)
    return fruit, fresh
