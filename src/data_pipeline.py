"""Shared tf.data input pipeline for the three models.

* Reads the fixed split files data/splits/{train,val,test}.csv (see scripts/make_splits.py).
* Decodes JPEG -> RGB float32 in [0, 255] at IMG_SIZE x IMG_SIZE (the dataset is already
  224 x 224, so resizing only matters if another input size is configured).
* Normalisation is NOT applied here. Every model starts with its own normalisation layer
  (Rescaling 1/255 for the custom CNNs, the backbone's own preprocessing for transfer
  learning), so a saved .keras model can be fed raw RGB images directly (demo, Grad-CAM).
* Augmentation is applied to the training split only and can be switched off (ablation):
  random flip, rotation, zoom, brightness and contrast. No hue/saturation changes - colour
  is a key freshness cue (browning, dark spots).
* Label formats
      "combined"   y = combined_label (0..15)                              -> Model 1
      "multitask"  y = {"fruit": fruit_label, "freshness": 0./1.}           -> Models 2 and 3
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import IMG_SIZE, PROJECT_ROOT, SEED, SPLITS_DIR

AUTOTUNE = tf.data.AUTOTUNE
LABEL_MODES = ("combined", "multitask")

DEFAULT_AUGMENTATION = {"flip": "horizontal_and_vertical", "rotation": 0.1, "zoom": 0.15, "brightness": 0.15, "contrast": 0.15}


def load_split(name):
    return pd.read_csv(SPLITS_DIR / f"{name}.csv")


def labels_for(df, label_mode):
    """Labels in dataset order as numpy arrays (combined) or a dict of arrays (multitask)."""
    if label_mode == "combined":
        return df["combined_label"].to_numpy(np.int32)
    if label_mode == "multitask":
        return {"fruit": df["fruit_label"].to_numpy(np.int32), "freshness": df["freshness_label"].to_numpy(np.float32)}
    raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}")


def decode_image(jpeg_bytes, img_size=IMG_SIZE):
    """Encoded JPEG bytes -> uint8 tensor (img_size, img_size, 3)."""
    image = tf.io.decode_jpeg(jpeg_bytes, channels=3)
    if img_size != IMG_SIZE:  # source images are 224 x 224
        image = tf.cast(tf.image.resize(image, (img_size, img_size), antialias=True), tf.uint8)
    return tf.ensure_shape(image, (img_size, img_size, 3))


def build_augmenter(cfg=None, seed=SEED):
    """Random training-time transformations, implemented with Keras preprocessing layers."""
    import keras

    cfg = {**DEFAULT_AUGMENTATION, **(cfg or {})}
    return keras.Sequential([
        keras.layers.RandomFlip(cfg["flip"], seed=seed),
        keras.layers.RandomRotation(cfg["rotation"], fill_mode="reflect", seed=seed + 1),
        keras.layers.RandomZoom(cfg["zoom"], fill_mode="reflect", seed=seed + 2),
        keras.layers.RandomBrightness(cfg["brightness"], value_range=(0, 255), seed=seed + 3),
        keras.layers.RandomContrast(cfg["contrast"], value_range=(0, 255), seed=seed + 4),
    ], name="augmentation")


def make_dataset(split, label_mode="multitask", batch_size=32, augment=False, aug_config=None,
                 shuffle=None, cache=True, img_size=IMG_SIZE, limit=None, seed=SEED):
    """tf.data.Dataset of (image float32 [0, 255], label) batches for one split.

    shuffle defaults to True for the training split only; evaluation splits keep CSV order so
    predictions line up with labels_for(load_split(split), ...).
    `limit` keeps the first `limit` rows per class (smoke tests).
    """
    df = load_split(split)
    if limit:
        df = df.groupby("combined_label", sort=False).head(limit).reset_index(drop=True)
    paths = [str(PROJECT_ROOT / p) for p in df["path"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_for(df, label_mode)))
    ds = ds.map(lambda p, y: (tf.io.read_file(p), y), num_parallel_calls=AUTOTUNE)
    if cache:  # keep the encoded JPEG bytes in memory (~6 KB per image); decoding each epoch is cheap
        ds = ds.cache()
    if shuffle if shuffle is not None else split == "train":
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda b, y: (decode_image(b, img_size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if augment:
        augmenter = build_augmenter(aug_config, seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def load_image(path, img_size=IMG_SIZE):
    """Single image file -> float32 array (img_size, img_size, 3) in [0, 255] (demo / Grad-CAM).

    Decoded with the same TensorFlow decoder as the training pipeline, so dataset images give
    exactly the same predictions as in evaluation. Non-square photos are centre-cropped to a
    square first (the dataset images are square), then resized.
    """
    image = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    h, w = int(image.shape[0]), int(image.shape[1])
    if h != w:
        side = min(h, w)
        image = tf.image.crop_to_bounding_box(image, (h - side) // 2, (w - side) // 2, side, side)
    if image.shape[0] != img_size:
        image = tf.image.resize(image, (img_size, img_size), antialias=True)
    return tf.cast(image, tf.float32).numpy()
