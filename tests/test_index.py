"""Tests for the FAISS vector index. Random small vectors; no model, no dataset."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.encoders.fusion import l2_normalize
from src.index.faiss_index import VectorIndex
from src.index.stock import Stock
from src.rerank.availability import rerank_with_availability

DIM = 8
N = 40


@pytest.fixture
def vecs() -> np.ndarray:
    rng = np.random.default_rng(0)
    return l2_normalize(rng.normal(size=(N, DIM)).astype(np.float32), axis=1)


@pytest.fixture
def ids() -> list[str]:
    return [f"i{n}" for n in range(N)]


@pytest.fixture
def index(vecs, ids) -> VectorIndex:
    idx = VectorIndex(dim=DIM)
    idx.build(vecs, ids)
    return idx


def test_search_is_descending(index, vecs):
    _, _, scores = index.search(vecs[3], 10)
    assert np.all(np.diff(scores) <= 1e-6)


def test_nearest_neighbour_of_stored_vector_is_itself(index, vecs, ids):
    got, _, scores = index.search(vecs[7], 1)
    assert got == [ids[7]]
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_returned_vectors_match_returned_ids(index):
    got, out_vecs, _ = index.search(index.get_vector("i5"), 6)
    for iid, v in zip(got, out_vecs):
        np.testing.assert_allclose(v, index.get_vector(iid))


def test_depth_larger_than_index_is_clipped(index):
    got, out_vecs, scores = index.search(index.get_vector("i0"), 10 * N)
    assert len(got) == N and out_vecs.shape == (N, DIM) and scores.shape == (N,)


def test_depth_zero_is_empty(index):
    got, out_vecs, scores = index.search(index.get_vector("i0"), 0)
    assert got == [] and out_vecs.shape == (0, DIM) and scores.shape == (0,)


def test_get_vector_round_trip(index, vecs):
    np.testing.assert_allclose(index.get_vector("i12"), vecs[12])


def test_all_item_ids_is_a_copy_in_index_order(index, ids):
    got = index.all_item_ids()
    assert got == ids
    got.append("intruder")
    assert len(index) == len(ids)


def test_build_rejects_duplicate_ids(vecs, ids):
    with pytest.raises(ValueError, match="duplicate"):
        VectorIndex(dim=DIM).build(vecs, ids[:-1] + [ids[0]])


def test_build_rejects_misaligned_input(vecs, ids):
    with pytest.raises(ValueError):
        VectorIndex(dim=DIM).build(vecs, ids[:-1])


def test_from_embeddings_fuses_at_alpha(tmp_path, vecs, ids):
    rng = np.random.default_rng(1)
    txt = l2_normalize(rng.normal(size=(N, DIM)).astype(np.float32), axis=1)
    np.save(tmp_path / "img.npy", vecs)
    np.save(tmp_path / "txt.npy", txt)
    (tmp_path / "ids.json").write_text(json.dumps(ids))

    image_only = VectorIndex.from_embeddings(
        tmp_path / "img.npy", tmp_path / "txt.npy", tmp_path / "ids.json", alpha=0.0
    )
    text_only = VectorIndex.from_embeddings(
        tmp_path / "img.npy", tmp_path / "txt.npy", tmp_path / "ids.json", alpha=1.0
    )
    np.testing.assert_allclose(image_only.get_vector("i3"), vecs[3], atol=1e-6)
    np.testing.assert_allclose(text_only.get_vector("i3"), txt[3], atol=1e-6)


def test_from_embeddings_rejects_mismatched_files(tmp_path, vecs, ids):
    """Manifest and embeddings from different runs would silently pair images
    with the wrong products; fail loudly instead."""
    np.save(tmp_path / "img.npy", vecs)
    np.save(tmp_path / "txt.npy", vecs)
    (tmp_path / "ids.json").write_text(json.dumps(ids[:-1]))
    with pytest.raises(ValueError, match="same encode run"):
        VectorIndex.from_embeddings(
            tmp_path / "img.npy", tmp_path / "txt.npy", tmp_path / "ids.json", 0.5
        )


def test_composes_with_stage_c(index, ids):
    """search() is the retrieve_fn contract; Stage C must accept it as is."""
    stock = Stock(ids)
    stock.mark_sold("i1")
    res = rerank_with_availability(
        index.get_vector("i0"), index.search, stock,
        k=5, n_candidates=10, exclude_ids=["i0"],
    )
    assert len(res) == 5
    assert "i0" not in res.item_ids and "i1" not in res.item_ids
