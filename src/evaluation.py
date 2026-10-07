import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from src.config import COMBINED_NAMES, FRESHNESS, FRUITS, NUM_FRUITS


def decode_outputs(outputs):
    # đổi output của mỗi model thành 3 dự đoán: loại quả, tươi/hỏng và joint (16 lớp) để so sánh được
    if isinstance(outputs, dict):  # Model 2 và 3: hai output
        fruit_prob = np.asarray(outputs["fruit"], dtype=np.float64)
        spoiled_prob = np.asarray(outputs["freshness"], dtype=np.float64).reshape(-1)
        fruit_pred = fruit_prob.argmax(axis=1)
        fresh_pred = (spoiled_prob >= 0.5).astype(int)  # hỏng nếu P(spoiled) >= 0.5
        joint_pred = fruit_pred * 2 + fresh_pred
    else:  # Model 1: softmax 16 lớp, xác suất một loại quả = P(tươi) + P(hỏng) của loại đó
        p = np.asarray(outputs, dtype=np.float64).reshape(-1, NUM_FRUITS, 2)
        fruit_prob = p.sum(axis=2)
        spoiled_prob = p[:, :, 1].sum(axis=1)
        fruit_pred = fruit_prob.argmax(axis=1)
        fresh_pred = (spoiled_prob >= 0.5).astype(int)
        joint_pred = p.reshape(len(p), -1).argmax(axis=1)
    return {
        "fruit_prob": fruit_prob,
        "spoiled_prob": spoiled_prob,
        "fruit_pred": fruit_pred,
        "fresh_pred": fresh_pred,
        "joint_pred": joint_pred,
    }


def predict(model, dataset):
    return decode_outputs(model.predict(dataset, verbose=0))


def compute_metrics(df, pred):
    # macro = trung bình cộng các lớp (mỗi lớp như nhau); freshness: lớp dương là "hỏng"
    yf, ys, yj = df["fruit_label"].to_numpy(), df["freshness_label"].to_numpy(), df["combined_label"].to_numpy()
    out = {}
    p, r, f, _ = precision_recall_fscore_support(yf, pred["fruit_pred"], average="macro", zero_division=0)
    out.update(fruit_accuracy=accuracy_score(yf, pred["fruit_pred"]), fruit_precision=p, fruit_recall=r, fruit_f1=f)
    p, r, f, _ = precision_recall_fscore_support(ys, pred["fresh_pred"], average="binary", pos_label=1, zero_division=0)
    out.update(
        freshness_accuracy=accuracy_score(ys, pred["fresh_pred"]),
        freshness_precision=p,
        freshness_recall=r,
        freshness_f1=f,
        freshness_macro_f1=f1_score(ys, pred["fresh_pred"], average="macro"),
        freshness_auc=roc_auc_score(ys, pred["spoiled_prob"]),
    )
    p, r, f, _ = precision_recall_fscore_support(yj, pred["joint_pred"], average="macro", zero_division=0)
    out.update(joint_accuracy=accuracy_score(yj, pred["joint_pred"]), joint_precision=p, joint_recall=r, joint_f1=f)
    return {k: float(v) for k, v in out.items()}


def confusion_matrices(df, pred):
    return {
        "fruit": (confusion_matrix(df["fruit_label"], pred["fruit_pred"], labels=range(len(FRUITS))), FRUITS),
        "freshness": (confusion_matrix(df["freshness_label"], pred["fresh_pred"], labels=[0, 1]), FRESHNESS),
        "joint": (
            confusion_matrix(df["combined_label"], pred["joint_pred"], labels=range(len(COMBINED_NAMES))),
            COMBINED_NAMES,
        ),
    }


def predictions_table(df, pred):
    out = df[["path", "fruit_label", "freshness_label", "combined_label"]].copy()
    out["fruit_pred"], out["freshness_pred"], out["joint_pred"] = (
        pred["fruit_pred"],
        pred["fresh_pred"],
        pred["joint_pred"],
    )
    out["spoiled_prob"] = pred["spoiled_prob"].round(5)
    for i, fruit in enumerate(FRUITS):
        out[f"p_{fruit}"] = pred["fruit_prob"][:, i].round(5)
    return out


# Grad-CAM (Selvaraju et al., 2017): vùng nào của ảnh làm model đưa ra dự đoán, tính trên layer conv cuối.
# Lấy điểm trước softmax/sigmoid (logit) vì khi model rất tự tin thì softmax/sigmoid bão hoà, gradient gần 0.

LAST_CONV = {
    "model1_simple_cnn": "conv5",  # 14 x 14 x 256
    "model2_multitask_cnn": "stage4_block1_out",  # 7 x 7 x 512
    "model3_mobilenet_v2": "out_relu",  # 7 x 7 x 1280
}


def last_conv_layer(model):
    if model.name in LAST_CONV:
        return LAST_CONV[model.name]
    return [layer.name for layer in model.layers if len(layer.output.shape) == 4][-1]


def _score(logits, target):
    if "freshness" in logits:
        fresh_logit = logits["freshness"][:, 0]
        fruit_logits = logits["fruit"]
    else:
        z = tf.reshape(logits["combined"], (-1, NUM_FRUITS, 2))
        fresh_logit = tf.reduce_logsumexp(z[:, :, 1], axis=1) - tf.reduce_logsumexp(z[:, :, 0], axis=1)
        fruit_logits = tf.reduce_logsumexp(z, axis=2)
    if target == "freshness":
        return tf.where(fresh_logit >= 0, 1.0, -1.0) * fresh_logit
    if target == "fruit":
        idx = tf.argmax(fruit_logits, axis=1)
        return tf.gather(fruit_logits, idx, batch_dims=1)
    raise ValueError(target)


def _forward(model, x, layer_name, heads, tape):
    if isinstance(model, keras.Sequential):
        feature_maps = None
        for layer in model.layers:
            if layer is heads[0]:
                return feature_maps, [x]
            x = layer(x, training=False)
            if layer.name == layer_name:
                feature_maps = x
                tape.watch(feature_maps)
        raise ValueError(f"output layer {heads[0].name} not found")
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output] + [h.input for h in heads])
    feature_maps, *hidden = grad_model(x, training=False)
    return feature_maps, hidden


def gradcam(model, images, target="freshness", layer_name=None):
    layer_name = layer_name or last_conv_layer(model)
    names = [layer.name for layer in model.layers]
    heads = [model.get_layer(n) for n in (("fruit", "freshness") if "freshness" in names else ("combined",))]
    x = tf.convert_to_tensor(images, dtype=tf.float32)
    with tf.GradientTape() as tape:
        feature_maps, hidden = _forward(model, x, layer_name, heads, tape)
        logits = {h.name: tf.matmul(z, h.kernel) + h.bias for h, z in zip(heads, hidden)}
        score = _score(logits, target)
    grads = tape.gradient(score, feature_maps)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # trung bình gradient = độ quan trọng của từng kênh
    cam = tf.nn.relu(tf.reduce_sum(weights * feature_maps, axis=-1))  # tổng có trọng số các feature map, giữ phần dương
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-7)
    cam = tf.image.resize(cam[..., None], images.shape[1:3], method="bilinear")[..., 0]  # phóng to về 224 x 224
    return cam.numpy()


def spoiled_prob(model, images):
    return decode_outputs(model.predict(images, verbose=0))["spoiled_prob"]
