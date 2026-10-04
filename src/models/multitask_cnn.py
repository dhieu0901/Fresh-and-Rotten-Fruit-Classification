"""Model 2 - multi-branch, multi-task CNN built with the Keras Functional API.

A shared convolutional trunk made of residual blocks (ResNet-style "CNN blocks" with batch
normalisation, Chapter 5.2; one block per stage like ResNet-10) splits into two parallel heads
(src/models/heads.py): an 8-way softmax for the fruit type and a sigmoid for freshness (P(spoiled)).

    224x224x3 -> Rescaling(1/255)
      stem    : Conv 3x3/2 (32) - BN - ReLU -> MaxPool 2x2               56x56x32
      stage 1 : ResBlock(64)                                             56x56x64
      stage 2 : ResBlock(128, stride 2)                                  28x28x128
      stage 3 : ResBlock(256, stride 2)                                  14x14x256
      stage 4 : ResBlock(512, stride 2)                                   7x7x512
      heads   : GAP -> fruit (8, softmax) | freshness (1, sigmoid)

ResBlock(f, s): Conv 3x3/s - BN - ReLU - Conv 3x3 - BN, added to the shortcut (identity, or
1x1 conv/s + BN when the shape changes), then ReLU.
"""
import keras
from keras import layers

from src.config import IMG_SIZE
from src.models.heads import build_heads


def conv_bn_relu(x, filters, kernel_size=3, strides=1, name="conv"):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def residual_block(x, filters, strides=1, name="res"):
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    if strides != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, use_bias=False, name=f"{name}_proj")(x)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    y = layers.Add(name=f"{name}_add")([y, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(y)


def build_trunk(inputs, stem_filters=32, widths=(64, 128, 256, 512), blocks_per_stage=1):
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    x = conv_bn_relu(x, stem_filters, 3, strides=2, name="stem")
    x = layers.MaxPooling2D(2, name="stem_pool")(x)
    for stage, width in enumerate(widths, start=1):
        for block in range(1, blocks_per_stage + 1):
            strides = 2 if (block == 1 and stage > 1) else 1
            x = residual_block(x, width, strides, name=f"stage{stage}_block{block}")
    return x


def build_multitask_cnn(img_size=IMG_SIZE, dropout=0.3, stem_filters=32, widths=(64, 128, 256, 512), blocks_per_stage=1):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    features = build_trunk(inputs, stem_filters, widths, blocks_per_stage)
    fruit, freshness = build_heads(features, dropout)
    return keras.Model(inputs, {"fruit": fruit, "freshness": freshness}, name="model2_multitask_cnn")
