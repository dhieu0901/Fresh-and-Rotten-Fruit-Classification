"""Model 3: MobileNetV2 pretrained on ImageNet with the same two heads as Model 2.

Stage 1 trains only the heads (frozen backbone), stage 2 unfreezes the backbone from a chosen block
and fine-tunes it with a small learning rate. Batch-norm layers stay frozen in both stages.
"""

import keras
from keras import layers

from src.config import IMG_SIZE
from src.models.heads import HEAD_LAYER_NAMES, build_heads

NON_BACKBONE = set(HEAD_LAYER_NAMES) | {"image", "mobilenet_v2_preprocess"}


def build_transfer_model(img_size=IMG_SIZE, dropout=0.3, weights="imagenet"):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    # MobileNetV2 expects inputs in [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_v2_preprocess")(inputs)
    # input_tensor keeps the backbone layers inside this model, so single blocks can be frozen
    base = keras.applications.MobileNetV2(include_top=False, weights=weights, input_tensor=x)
    fruit, freshness = build_heads(base.output, dropout)
    model = keras.Model(inputs, {"fruit": fruit, "freshness": freshness}, name="model3_mobilenet_v2")
    set_backbone_trainable(model, None)
    return model


def set_backbone_trainable(model, fine_tune_from=None):
    """Freeze the backbone, then unfreeze it from layer `fine_tune_from` on (None = all frozen, "all" = everything)."""
    backbone = [layer for layer in model.layers if layer.name not in NON_BACKBONE]
    if fine_tune_from is None:
        start = len(backbone)
    elif fine_tune_from == "all":
        start = 0
    else:
        names = [layer.name for layer in backbone]
        if fine_tune_from not in names:
            raise ValueError(f"{fine_tune_from} is not a backbone layer")
        start = names.index(fine_tune_from)
    for i, layer in enumerate(backbone):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    unfrozen = backbone[start:]
    return {
        "unfrozen_layers": len(unfrozen),
        "unfrozen_layers_with_weights": sum(1 for layer in unfrozen if layer.trainable_weights),
        "trainable_params": int(sum(keras.ops.size(w) for w in model.trainable_weights)),
    }
