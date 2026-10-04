"""Grad-CAM (Selvaraju et al., 2017) for the three models, following the Keras Grad-CAM example.

heatmap = ReLU( sum_k  alpha_k * A_k ),   alpha_k = mean over pixels of d(score) / d(A_k)
where A_k are the feature maps of the last convolutional layer.

Score explained (computed from the pre-activation logits of the output Dense layers, so very
confident predictions - sigmoid/softmax saturated at 1.0 - still have non-zero gradients)
    target="freshness": log-odds of the predicted freshness class
        Models 2/3: logit of the freshness neuron;  Model 1: logsumexp(spoiled logits) - logsumexp(fresh logits).
        The sign is flipped when the model predicts "fresh", so the map always shows the evidence
        for the decision the model actually made.
    target="fruit": logit of the predicted fruit (Model 1: logsumexp of the fruit's two logits).
"""
import numpy as np
import tensorflow as tf
import keras

from src.config import NUM_FRUITS

LAST_CONV = {
    "model1_simple_cnn": "conv5",              # 14 x 14 x 256
    "model2_multitask_cnn": "stage4_block1_out",  # 7 x 7 x 512
    "model3_mobilenet_v2": "out_relu",           # 7 x 7 x 1280
}
EPS = 1e-7


def last_conv_layer(model):
    """Name of the last convolutional feature map (model names are set in src/models/*)."""
    if model.name in LAST_CONV:
        return LAST_CONV[model.name]
    feature_maps = [l.name for l in model.layers if len(l.output.shape) == 4 and not isinstance(l, keras.layers.Rescaling)]
    return feature_maps[-1]


def _score(logits, target):
    """Scalar score per image from the output-layer logits (dict: layer name -> [N, units])."""
    if "freshness" in logits:                          # Models 2 / 3
        fresh_logit = logits["freshness"][:, 0]
        fruit_logits = logits["fruit"]
    else:                                              # Model 1: 16 logits, index = fruit*2 + spoiled
        z = tf.reshape(logits["combined"], (-1, NUM_FRUITS, 2))
        fresh_logit = tf.reduce_logsumexp(z[:, :, 1], axis=1) - tf.reduce_logsumexp(z[:, :, 0], axis=1)
        fruit_logits = tf.reduce_logsumexp(z, axis=2)
    if target == "freshness":
        sign = tf.where(fresh_logit >= 0, 1.0, -1.0)   # explain the predicted class
        return sign * fresh_logit
    if target == "fruit":
        idx = tf.argmax(fruit_logits, axis=1)
        return tf.gather(fruit_logits, idx, batch_dims=1)
    raise ValueError(target)


def _forward(model, x, layer_name, heads, tape):
    """One forward pass returning the feature maps of `layer_name` and the inputs of the output Dense layers."""
    if isinstance(model, keras.Sequential):
        # layer by layer: a reloaded Sequential model holds several graphs per layer, so a sub-model
        # built from layer.output / layer.input could mix them (the gradient would be None)
        feature_maps = None
        for layer in model.layers:
            if layer is heads[0]:
                return feature_maps, [x]
            x = layer(x, training=False)
            if layer.name == layer_name:
                feature_maps = x
                tape.watch(feature_maps)
        raise ValueError(f"output layer {heads[0].name!r} not found")
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output] + [h.input for h in heads])
    feature_maps, *hidden = grad_model(x, training=False)
    return feature_maps, hidden


def gradcam(model, images, target="freshness", layer_name=None):
    """images: float32 [N, H, W, 3] raw pixels in [0, 255]. Returns heatmaps [N, H, W] in [0, 1]."""
    layer_name = layer_name or last_conv_layer(model)
    names = [layer.name for layer in model.layers]
    heads = [model.get_layer(n) for n in (("fruit", "freshness") if "freshness" in names else ("combined",))]
    x = tf.convert_to_tensor(images, dtype=tf.float32)
    with tf.GradientTape() as tape:
        # inputs of the output Dense layers -> logits computed with the layers' own weights
        feature_maps, hidden = _forward(model, x, layer_name, heads, tape)
        logits = {h.name: tf.matmul(z, h.kernel) + h.bias for h, z in zip(heads, hidden)}
        score = _score(logits, target)
    grads = tape.gradient(score, feature_maps)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * feature_maps, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + EPS)
    cam = tf.image.resize(cam[..., None], images.shape[1:3], method="bilinear")[..., 0]
    return cam.numpy()


def overlay(image, heatmap, alpha=0.45, cmap="inferno"):
    """Blend a [0, 1] heatmap onto a [0, 255] RGB image; returns uint8 RGB."""
    import matplotlib

    colored = matplotlib.colormaps[cmap](heatmap)[..., :3] * 255.0
    return np.clip((1 - alpha) * image + alpha * colored, 0, 255).astype(np.uint8)
