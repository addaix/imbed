"""Tests for imbed.data_prep (clustering helpers) and the small pure parts of imbed.base."""

import numpy as np
import pandas as pd
import pytest

from imbed import base
from imbed.base import ComputedValuesMapping, LocalSavesMixin, batches
from imbed.data_prep import clusters_df, kmeans_cluster_indices, re_clusters


def _blobs(n_per=20, centers=((0, 0), (10, 10), (-10, 10)), seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.3, size=(n_per, 2)) for c in centers])
    truth = np.repeat(np.arange(len(centers)), n_per)
    return X, truth


def _same_partition(a, b):
    """Whether two labelings induce the same grouping, whatever the label names."""
    pairs = set(zip(a, b))
    return len(pairs) == len(set(a)) == len(set(b))


def test_kmeans_on_array_finds_blobs():
    X, truth = _blobs()
    labels = kmeans_cluster_indices(X, n_clusters=3, random_state=0, n_init=3)
    assert _same_partition(labels, truth)


def test_kmeans_on_batches_uses_minibatch():
    X, truth = _blobs()
    batch_list = [X[i::2] for i in range(2)]
    labels = kmeans_cluster_indices(
        lambda: iter(batch_list), n_clusters=3, random_state=0, n_init=3
    )
    assert len(labels) == len(X)
    assert len(set(labels)) == 3
    # a plain (re-traversable) list of batches works too
    assert len(kmeans_cluster_indices(batch_list, n_clusters=3, random_state=0, n_init=3)) == len(X)


def test_kmeans_rejects_bad_sources():
    with pytest.raises(ValueError, match="numpy arrays"):
        kmeans_cluster_indices([[1, 2], [3, 4]], n_clusters=2)
    with pytest.raises(ValueError, match="iterable"):
        kmeans_cluster_indices(lambda: 5, n_clusters=2)


def test_clusters_df_accepts_arrays_mappings_and_frames():
    X, _ = _blobs(n_per=10)
    out = clusters_df(X, n_clusters=[2, 3])
    assert list(out.columns) == ["cluster_02", "cluster_03"]
    assert len(out) == len(X)
    as_map = {f"k{i}": row for i, row in enumerate(X)}
    assert list(clusters_df(as_map, n_clusters=[2]).index)[:2] == ["k0", "k1"]
    frame = pd.DataFrame({"embedding": list(X)}, index=[f"r{i}" for i in range(len(X))])
    assert clusters_df(frame, n_clusters=[3]).index[0] == "r0"


def test_re_clusters_equal_merge_and_split():
    X, truth = _blobs()
    assert re_clusters(X, truth, 3) is truth
    merged = re_clusters(X, truth, 2)
    assert len(set(merged)) == 2


def test_re_clusters_split():
    X, truth = _blobs()
    split = re_clusters(X, truth.copy(), 5)
    assert len(set(split)) >= 5


def test_computed_values_mapping():
    m = ComputedValuesMapping(("a", "bb"), value_of_key=len)
    assert list(m) == ["a", "bb"] and m["bb"] == 2
    assert list(ComputedValuesMapping(lambda: ["x", "y"])) == ["x", "y"]


def test_computed_values_mapping_default_value():
    assert ComputedValuesMapping(("x",))["x"] is None


def test_batches():
    df = pd.DataFrame({"a": range(5)})
    whole = list(batches(df, chk_size=None))
    assert len(whole) == 1 and len(whole[0]) == 5
    chunks = list(batches(df, chk_size=2))
    assert len(chunks) == 3


def test_temp_folder_and_ensure_dir(tmp_path):
    import os

    d = base.get_empty_temporary_folder()
    assert os.path.isdir(d) and os.listdir(d) == []
    target = str(tmp_path / "new")
    assert base._ensure_dir_exists(target) == target and os.path.isdir(target)
    assert base._ensure_dir_exists(target) == target  # idempotent


def test_local_saves_mixin_stores(tmp_path):
    class Dacc(LocalSavesMixin):
        saves_dir = str(tmp_path / "saves")

    d = Dacc()
    assert (tmp_path / "saves").is_dir() is False  # lazy
    assert d.saves is d.saves and (tmp_path / "saves").is_dir()
    d.saves_bytes_store["note.bin"] = b"hi"
    assert "note.bin" in list(d.saves_bytes_store)
    assert d.embeddings_chunks_store is d.embeddings_chunks_store
    assert (tmp_path / "saves" / "embeddings_chunks").is_dir()
