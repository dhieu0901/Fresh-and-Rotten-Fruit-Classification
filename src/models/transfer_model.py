"""Model 3 - transfer learning: ImageNet-pretrained backbone + the two heads of Model 2.

    224x224x3 -> backbone preprocessing -> MobileNetV2 or EfficientNetB0 (include_top=False)
              -> GAP -> fruit (8, softmax) | freshness (1, sigmoid)

Training happens in two stages (scripts/train.py):
  1. feature extraction - the whole backbone is frozen, only the heads learn;
  2. fine-tuning        - the backbone is unfrozen from a chosen block onwards (early blocks
                          stay frozen) and trained with a low learning rate.
Batch-normalisation layers of the backbone always stay frozen (inference mode), the standard
practice when fine-tuning on a small dataset.

Preprocessing matters: MobileNetV2 expects inputs in [-1, 1] (x / 127.5 - 1), while Keras'
EfficientNetB0 rescales internally and expects raw [0, 255] pixels.
"""
import keras
from keras import layers

from src.config import IMG_SIZE
from src.models.heads import HEAD_LAYER_NAMES, build_heads

BACKBONES = {
    "mobilenet_v2": {
        "builder": keras.applications.MobileNetV2,
        "preprocess": lambda: layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_v2_preprocess"),
        "last_conv": "out_relu",
        # fine-tuning start points, from the deepest block to the whole network
        "fine_tune_points": ["block_16_expand", "block_13_expand", "block_10_expand", "block_6_expand", "all"],
    },
    "efficientnet_b0": {
        "builder": keras.applications.EfficientNetB0,
        "preprocess": None,  # rescaling + normalisation are built into the Keras model
        "last_conv": "top_activation",
        "fine_tune_points": ["block7a_expand_conv", "block6a_expand_conv", "block5a_expand_conv", "block4a_expand_conv", "all"],
    },
}
NON_BACKBONE = set(HEAD_LAYER_NAMES) | {"image", "mobilenet_v2_preprocess"}


def build_transfer_model(backbone="mobilenet_v2", img_size=IMG_SIZE, dropout=0.3, weights="imagenet"):
    spec = BACKBONES[backbone]
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = spec["preprocess"]()(inputs) if spec["preprocess"] else inputs
    # input_tensor keeps the backbone layers flat inside this model (needed for Grad-CAM and
    # for freezing individual blocks).
    base = spec["builder"](include_top=False, weights=weights, input_tensor=x)
    fruit, freshness = build_heads(base.output, dropout)
    model = keras.Model(inputs, {"fruit": fruit, "freshness": freshness}, name=f"model3_{backbone}")
    set_backbone_trainable(model, None)
    return model


def backbone_layers(model):
    return [layer for layer in model.layers if layer.name not in NON_BACKBONE]


def set_backbone_trainable(model, fine_tune_from=None, train_batchnorm=False):
    """Freeze the backbone, then unfreeze it from layer `fine_tune_from` onwards.

    fine_tune_from: None (fully frozen), "all", a layer name, or an int N (last N layers).
    Returns a summary dict (unfrozen layers, unfrozen layers with weights, trainable params).
    """
    layers_ = backbone_layers(model)
    if fine_tune_from is None:
        start = len(layers_)
    elif fine_tune_from == "all":
        start = 0
    elif isinstance(fine_tune_from, int):
        start = max(len(layers_) - fine_tune_from, 0)
    else:
        names = [layer.name for layer in layers_]
        if fine_tune_from not in names:
            raise ValueError(f"{fine_tune_from!r} is not a backbone layer")
        start = names.index(fine_tune_from)
    for i, layer in enumerate(layers_):
        trainable = i >= start
        if isinstance(layer, layers.BatchNormalization) and not train_batchnorm:
            trainable = False
        layer.trainable = trainable
    unfrozen = layers_[start:]
    return {
        "fine_tune_from": fine_tune_from,
        "backbone_layers": len(layers_),
        "unfrozen_layers": len(unfrozen),
        "unfrozen_layers_with_weights": sum(1 for layer in unfrozen if layer.trainable_weights),
        "trainable_params": int(sum(keras.ops.size(w) for w in model.trainable_weights)),
    }
