"""Tests for the split files and the tf.data pipeline. Needs the dataset (download_data, audit_data, make_splits)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import METADATA_CSV, SPLITS_DIR  # noqa: E402

SPLITS = ("train", "val", "test")


@pytest.fixture(scope="module")
def splits():
    return {s: pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in SPLITS}


def test_splits_are_disjoint_and_cover_the_clean_data(splits):
    meta = pd.read_csv(METADATA_CSV, keep_default_na=False)
    kept = meta[meta["split"] != "excluded"]
    all_paths = pd.concat([df["path"] for df in splits.values()])
    assert all_paths.is_unique
    assert set(all_paths) == set(kept["path"])
    assert not meta.loc[meta["excluded"].astype(str) == "True", "path"].isin(all_paths).any()


def test_groups_and_hashes_never_cross_splits(splits):
    meta = pd.read_csv(METADATA_CSV, keep_default_na=False).set_index("path")
    owner = {}
    for name, df in splits.items():
        for group in df["group_id"].unique():
            assert owner.setdefault(("group", group), name) == name
        for sha in meta.loc[df["path"], "sha256"]:
            assert owner.setdefault(("sha", sha), name) == name


def test_label_columns_are_consistent(splits):
    for df in splits.values():
        assert (df["combined_label"] == df["fruit_label"] * 2 + df["freshness_label"]).all()
        assert df["combined_label"].nunique() == 16


def test_split_proportions_are_close_to_70_15_15(splits):
    n = {s: len(df) for s, df in splits.items()}
    total = sum(n.values())
    for s, target in zip(SPLITS, (0.70, 0.15, 0.15)):
        assert abs(n[s] / total - target) < 0.01


@pytest.mark.parametrize("label_mode", ["combined", "multitask"])
def test_batches_have_expected_shape_range_and_labels(label_mode):
    from src.data_pipeline import make_dataset

    x, y = next(iter(make_dataset("val", label_mode=label_mode, batch_size=8, limit=2)))
    assert tuple(x.shape) == (8, 224, 224, 3) and x.dtype.name == "float32"
    assert float(np.min(x)) >= 0.0 and float(np.max(x)) <= 255.0
    if label_mode == "combined":
        assert y.shape == (8,) and int(np.max(y)) < 16
    else:
        assert set(y) == {"fruit", "freshness"} and y["fruit"].shape == (8,)
        assert set(np.unique(y["freshness"])) <= {0.0, 1.0}


def test_evaluation_order_matches_csv():
    from src.data_pipeline import labels_for, load_split, make_dataset

    labels = np.concatenate([y.numpy() for _, y in make_dataset("test", label_mode="combined", batch_size=256)])
    np.testing.assert_array_equal(labels, labels_for(load_split("test"), "combined"))


def test_augmentation_changes_images_but_keeps_range():
    from src.data_pipeline import make_dataset

    plain = next(iter(make_dataset("train", batch_size=8, limit=1, shuffle=False)))[0].numpy()
    aug = next(iter(make_dataset("train", batch_size=8, limit=1, shuffle=False, augment=True)))[0].numpy()
    assert not np.allclose(plain, aug)
    assert aug.min() >= 0.0 and aug.max() <= 255.0
