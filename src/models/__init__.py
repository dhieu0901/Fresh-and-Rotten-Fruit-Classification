"""Model factory and compile settings shared by training, evaluation and tests."""
import keras

from src.models.multitask_cnn import build_multitask_cnn
from src.models.simple_cnn import build_simple_cnn
from src.models.transfer_model import build_transfer_model, set_backbone_trainable

# model name -> (label mode, builder)
MODELS = {
    "simple_cnn": ("combined", build_simple_cnn),
    "multitask_cnn": ("multitask", build_multitask_cnn),
    "transfer": ("multitask", build_transfer_model),
}


def label_mode_of(name):
    return MODELS[name][0]


def build_model(name, **kwargs):
    return MODELS[name][1](**kwargs)


def compile_model(model, label_mode, learning_rate, loss_weights=None):
    """Model 1: sparse categorical cross-entropy over 16 classes.
    Models 2/3: cross-entropy (fruit) + binary cross-entropy (freshness), weighted sum."""
    optimizer = keras.optimizers.Adam(learning_rate)
    if label_mode == "combined":
        model.compile(optimizer=optimizer,
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    else:
        model.compile(optimizer=optimizer,
                      loss={"fruit": keras.losses.SparseCategoricalCrossentropy(),
                            "freshness": keras.losses.BinaryCrossentropy()},
                      loss_weights=loss_weights or {"fruit": 1.0, "freshness": 1.0},
                      metrics={"fruit": [keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
                               "freshness": [keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")]})
    return model


__all__ = ["MODELS", "build_model", "compile_model", "label_mode_of", "set_backbone_trainable"]
