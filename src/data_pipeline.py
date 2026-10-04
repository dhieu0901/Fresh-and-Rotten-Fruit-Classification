"""tf.data pipeline shared by the three models.

Images come out as float32 in [0, 255]; each model rescales its own input. Augmentation is only
used on the training split and does not touch hue, since colour matters for freshness.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import IMG_SIZE, PROJECT_ROOT, SEED, SPLITS_DIR

AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_AUGMENTATION = {
    "flip": "horizontal_and_vertical",
    "rotation": 0.1,
    "zoom": 0.15,
    "brightness": 0.15,
    "contrast": 0.15,
}


def load_split(name):
    return pd.read_csv(SPLITS_DIR / f"{name}.csv")


def labels_for(df, label_mode):
    """16-class labels for Model 1 ("combined"), fruit + freshness labels for Models 2 and 3 ("multitask")."""
    if label_mode == "combined":
        return df["combined_label"].to_numpy(np.int32)
    if label_mode == "multitask":
        return {"fruit": df["fruit_label"].to_numpy(np.int32), "freshness": df["freshness_label"].to_numpy(np.float32)}
    raise ValueError(f"unknown label_mode: {label_mode}")


def decode_image(jpeg_bytes, img_size=IMG_SIZE):
    image = tf.io.decode_jpeg(jpeg_bytes, channels=3)
    if img_size != IMG_SIZE:  # the dataset images are already 224 x 224
        image = tf.cast(tf.image.resize(image, (img_size, img_size), antialias=True), tf.uint8)
    return tf.ensure_shape(image, (img_size, img_size, 3))


def build_augmenter(cfg=None, seed=SEED):
    import keras

    cfg = {**DEFAULT_AUGMENTATION, **(cfg or {})}
    return keras.Sequential(
        [
            keras.layers.RandomFlip(cfg["flip"], seed=seed),
            keras.layers.RandomRotation(cfg["rotation"], fill_mode="reflect", seed=seed + 1),
            keras.layers.RandomZoom(cfg["zoom"], fill_mode="reflect", seed=seed + 2),
            keras.layers.RandomBrightness(cfg["brightness"], value_range=(0, 255), seed=seed + 3),
            keras.layers.RandomContrast(cfg["contrast"], value_range=(0, 255), seed=seed + 4),
        ],
        name="augmentation",
    )


def make_dataset(
    split,
    label_mode="multitask",
    batch_size=32,
    augment=False,
    aug_config=None,
    shuffle=None,
    cache=True,
    img_size=IMG_SIZE,
    limit=None,
    seed=SEED,
):
    """Batches of (image, label) for one split. Only train is shuffled by default, so predictions on
    val/test stay in CSV order. `limit` keeps the first `limit` images of each class (quick tests)."""
    df = load_split(split)
    if limit:
        df = df.groupby("combined_label", sort=False).head(limit).reset_index(drop=True)
    if shuffle is None:
        shuffle = split == "train"
    paths = [str(PROJECT_ROOT / p) for p in df["path"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_for(df, label_mode)))
    ds = ds.map(lambda p, y: (tf.io.read_file(p), y), num_parallel_calls=AUTOTUNE)
    if cache:  # cache the JPEG bytes instead of decoded images, which would not fit in RAM
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda b, y: (decode_image(b, img_size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if augment:
        augmenter = build_augmenter(aug_config, seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def load_image(path, img_size=IMG_SIZE):
    """One image file as float32 in [0, 255], decoded the same way as in the pipeline.
    Non-square photos are centre-cropped first."""
    image = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    h, w = image.shape[:2]
    if h != w:
        side = min(h, w)
        image = tf.image.crop_to_bounding_box(image, (h - side) // 2, (w - side) // 2, side, side)
    if image.shape[0] != img_size:
        image = tf.image.resize(image, (img_size, img_size), antialias=True)
    return tf.cast(image, tf.float32).numpy()
