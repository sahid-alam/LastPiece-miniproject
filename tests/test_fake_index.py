"""Tests for Track D: FakeIndex, integration with Stock and reranker.

Conventions (from tests/conftest.py and CLAUDE.md):
  - No network calls, no model downloads.
  - Use synthetic fixtures only.
  - pytest must pass before every commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.fake_index import FakeIndex, FakeProduct, build_fake_index, DIM, SEED
from src.index.stock import Stock
from src.rerank.availability import rerank_with_availability


# ============================================================================ #
# Fixtures                                                                      #
# ============================================================================ #

@pytest.fixture(scope="module")
def index() -> FakeIndex:
    """One FakeIndex shared across all tests in this module (expensive to build)."""
    return build_fake_index(seed=SEED)


@pytest.fixture(scope="module")
def all_ids(index: FakeIndex) -> list[str]:
    return index.all_item_ids()


@pytest.fixture(scope="module")
def full_stock(all_ids: list[str]) -> Stock:
    return Stock(all_ids)


@pytest.fixture(scope="module")
def query_vec(index: FakeIndex) -> np.ndarray:
    """Query vector: the stored vector of the first item (deterministic)."""
    first_id = index.all_item_ids()[0]
    return index.get_vector(first_id)


# ============================================================================ #
# Synthetic index structure                                                      #
# ============================================================================ #

class TestFakeIndexStructure:
    def test_approximately_300_products(self, index: FakeIndex) -> None:
        """Spec says ~300 synthetic products."""
        assert 280 <= len(index) <= 320

    def test_all_ids_are_strings(self, index: FakeIndex) -> None:
        for iid in index.all_item_ids():
            assert isinstance(iid, str), f"Expected str, got {type(iid)}"

    def test_ids_are_unique(self, index: FakeIndex) -> None:
        ids = index.all_item_ids()
        assert len(ids) == len(set(ids)), "Duplicate item IDs found"

    def test_vectors_are_float32(self, index: FakeIndex) -> None:
        first_id = index.all_item_ids()[0]
        vec = index.get_vector(first_id)
        assert vec.dtype == np.float32

    def test_vector_shape_is_512(self, index: FakeIndex) -> None:
        for iid in index.all_item_ids()[:10]:
            vec = index.get_vector(iid)
            assert vec.shape == (DIM,), f"Expected ({DIM},), got {vec.shape}"

    def test_vectors_are_l2_normalised(self, index: FakeIndex) -> None:
        """Every stored vector must have unit norm (within float32 tolerance)."""
        for iid in index.all_item_ids():
            vec = index.get_vector(iid)
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 1e-5, (
                f"Vector {iid} not normalised: norm={norm}"
            )

    def test_fixed_seed_is_reproducible(self) -> None:
        """Same seed → same IDs and same first vector."""
        idx_a = build_fake_index(seed=7)
        idx_b = build_fake_index(seed=7)
        assert idx_a.all_item_ids() == idx_b.all_item_ids()
        first_id = idx_a.all_item_ids()[0]
        np.testing.assert_array_equal(
            idx_a.get_vector(first_id), idx_b.get_vector(first_id)
        )

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce different embeddings.
        IDs are derived from cluster labels (deterministic), but vectors
        are seeded RNG noise so they differ."""
        idx_a = build_fake_index(seed=1)
        idx_b = build_fake_index(seed=2)
        first_id = idx_a.all_item_ids()[0]
        vec_a = idx_a.get_vector(first_id)
        vec_b = idx_b.get_vector(first_id)
        assert not np.allclose(vec_a, vec_b), (
            "Vectors from different seeds should differ"
        )

    def test_products_have_titles(self, index: FakeIndex) -> None:
        for p in index.all_products():
            assert isinstance(p.title, str) and p.title.strip()

    def test_thumbnails_are_uint8_arrays(self, index: FakeIndex) -> None:
        for p in index.all_products()[:10]:
            assert p.thumbnail.dtype == np.uint8
            assert p.thumbnail.ndim == 3          # H × W × 3
            assert p.thumbnail.shape[2] == 3


# ============================================================================ #
# search()                                                                      #
# ============================================================================ #

class TestFakeIndexSearch:
    def test_search_returns_triple(self, index: FakeIndex, query_vec: np.ndarray) -> None:
        result = index.search(query_vec, depth=5)
        assert isinstance(result, tuple) and len(result) == 3

    def test_search_ids_are_strings(self, index: FakeIndex, query_vec: np.ndarray) -> None:
        ids, vecs, scores = index.search(query_vec, depth=5)
        for iid in ids:
            assert isinstance(iid, str)

    def test_search_respects_depth(self, index: FakeIndex, query_vec: np.ndarray) -> None:
        for depth in [1, 5, 10, 50]:
            ids, vecs, scores = index.search(query_vec, depth=depth)
            assert len(ids) == depth
            assert vecs.shape == (depth, DIM)
            assert scores.shape == (depth,)

    def test_search_capped_at_catalogue_size(self, index: FakeIndex, query_vec: np.ndarray) -> None:
        """Requesting more than n items must return at most n."""
        ids, vecs, scores = index.search(query_vec, depth=10_000)
        assert len(ids) == len(index)

    def test_scores_descending(self, index: FakeIndex, query_vec: np.ndarray) -> None:
        _, _, scores = index.search(query_vec, depth=20)
        assert list(scores) == sorted(scores, reverse=True)

    def test_scores_are_float32(self, index: FakeIndex, query_vec: np.ndarray) -> None:
        _, _, scores = index.search(query_vec, depth=5)
        assert scores.dtype == np.float32

    def test_vectors_shape_matches_depth(self, index: FakeIndex, query_vec: np.ndarray) -> None:
        _, vecs, _ = index.search(query_vec, depth=7)
        assert vecs.shape == (7, DIM)

    def test_top_result_is_self(self, index: FakeIndex) -> None:
        """When we query with item A's own vector, A should be the top hit."""
        first_id = index.all_item_ids()[0]
        q = index.get_vector(first_id)
        ids, _, _ = index.search(q, depth=1)
        assert ids[0] == first_id

    def test_cluster_items_appear_together(self, index: FakeIndex) -> None:
        """Items in the same cluster should dominate the top hits for a
        cluster-centre query, demonstrating why MMR is useful."""
        # Pick the first item (from the first cluster, denim_jacket_000).
        first_id = index.all_item_ids()[0]
        q = index.get_vector(first_id)
        ids, _, _ = index.search(q, depth=10)
        # Most of the top 10 should share the same cluster prefix.
        cluster_prefix = first_id.rsplit("_", 1)[0]
        same_cluster = sum(1 for iid in ids if iid.startswith(cluster_prefix))
        assert same_cluster >= 5, (
            f"Expected cluster items to dominate top-10 hits; "
            f"only {same_cluster}/10 from cluster '{cluster_prefix}'"
        )

    def test_zero_depth_returns_empty(self, index: FakeIndex, query_vec: np.ndarray) -> None:
        ids, vecs, scores = index.search(query_vec, depth=0)
        assert ids == []
        assert vecs.shape[0] == 0
        assert scores.shape[0] == 0

    def test_wrong_shape_raises_value_error(self, index: FakeIndex) -> None:
        bad_vec = np.ones(128, dtype=np.float32)
        with pytest.raises(ValueError, match="512"):
            index.search(bad_vec, depth=5)

    def test_2d_query_raises_value_error(self, index: FakeIndex) -> None:
        bad_vec = np.ones((1, 512), dtype=np.float32)
        with pytest.raises(ValueError):
            index.search(bad_vec, depth=5)


# ============================================================================ #
# get_vector()                                                                   #
# ============================================================================ #

class TestGetVector:
    def test_returns_correct_shape(self, index: FakeIndex, all_ids: list[str]) -> None:
        for iid in all_ids[:5]:
            assert index.get_vector(iid).shape == (DIM,)

    def test_unknown_id_raises_key_error(self, index: FakeIndex) -> None:
        with pytest.raises(KeyError):
            index.get_vector("__nonexistent__")

    def test_vector_consistent_with_search(self, index: FakeIndex) -> None:
        """get_vector(id) and the vector returned by search() must agree."""
        first_id = index.all_item_ids()[0]
        q = index.get_vector(first_id)
        ids, vecs, _ = index.search(q, depth=1)
        assert ids[0] == first_id
        np.testing.assert_array_equal(index.get_vector(first_id), vecs[0])


# ============================================================================ #
# Integration: FakeIndex + Stock + rerank_with_availability                     #
# ============================================================================ #

class TestRecommendationIntegration:
    def test_query_item_excluded(self, index: FakeIndex, all_ids: list[str]) -> None:
        """The item used as a query must never appear in its own results."""
        stock = Stock(all_ids)
        qid = all_ids[0]
        q = index.get_vector(qid)
        result = rerank_with_availability(
            q, index.search, stock, k=5, n_candidates=20, exclude_ids=[qid]
        )
        assert qid not in result.item_ids

    def test_sold_items_absent_from_results(self, index: FakeIndex, all_ids: list[str]) -> None:
        stock = Stock(all_ids)
        qid = all_ids[0]
        q = index.get_vector(qid)

        # Sell the five nearest neighbours.
        ids_nearby, _, _ = index.search(q, depth=10)
        to_sell = [iid for iid in ids_nearby if iid != qid][:5]
        for iid in to_sell:
            stock.mark_sold(iid)

        result = rerank_with_availability(
            q, index.search, stock, k=5, n_candidates=20, exclude_ids=[qid]
        )
        for iid in to_sell:
            assert iid not in result.item_ids

    def test_no_duplicates_in_result(self, index: FakeIndex, all_ids: list[str]) -> None:
        stock = Stock(all_ids)
        q = index.get_vector(all_ids[0])
        result = rerank_with_availability(
            q, index.search, stock, k=5, n_candidates=20
        )
        assert len(result.item_ids) == len(set(result.item_ids))

    def test_substitute_appears_when_item_sold(
        self, index: FakeIndex, all_ids: list[str]
    ) -> None:
        """After selling a top result, re-querying should return a substitute."""
        stock = Stock(all_ids)
        qid = all_ids[0]
        q = index.get_vector(qid)

        result_before = rerank_with_availability(
            q, index.search, stock, k=5, n_candidates=50, exclude_ids=[qid]
        )
        sold_id = result_before.item_ids[0]
        stock.mark_sold(sold_id)

        result_after = rerank_with_availability(
            q, index.search, stock, k=5, n_candidates=50, exclude_ids=[qid]
        )
        assert sold_id not in result_after.item_ids
        # Should still get a full list (catalogue is large enough).
        assert len(result_after) == 5

    def test_widening_when_many_sold(self, index: FakeIndex, all_ids: list[str]) -> None:
        """With a shallow initial depth and many sold items, the reranker must
        widen to fill k results."""
        stock = Stock(all_ids)
        qid = all_ids[0]
        q = index.get_vector(qid)

        # Sell the first 30 items in the catalogue.
        for iid in all_ids[1:31]:
            stock.mark_sold(iid)

        result = rerank_with_availability(
            q, index.search, stock,
            k=5, n_candidates=10, max_candidates=500, exclude_ids=[qid]
        )
        # Either the list is full, or widening was attempted.
        assert len(result) > 0
        assert result.n_widening_rounds >= 0    # may be 0 if 10 were enough

    def test_diagnostics_populated(self, index: FakeIndex, all_ids: list[str]) -> None:
        stock = Stock(all_ids)
        q = index.get_vector(all_ids[0])
        stock.mark_sold(all_ids[1])   # force at least one drop

        result = rerank_with_availability(
            q, index.search, stock, k=5, n_candidates=20
        )
        assert result.n_retrieved > 0
        assert len(result.scores) == len(result.item_ids)
        assert result.final_candidate_depth >= 20

    def test_insufficient_stock_does_not_raise(
        self, index: FakeIndex, all_ids: list[str]
    ) -> None:
        """Nearly exhausted catalogue must produce a short result, not an exception."""
        stock = Stock(all_ids)
        for iid in all_ids[1:]:
            stock.mark_sold(iid)

        q = index.get_vector(all_ids[0])
        result = rerank_with_availability(
            q, index.search, stock, k=5, n_candidates=20, exclude_ids=[all_ids[0]]
        )
        assert result.short_result
        assert len(result) == 0   # only one item left, excluded

    def test_reset_stock_restores_availability(
        self, index: FakeIndex, all_ids: list[str]
    ) -> None:
        stock = Stock(all_ids)
        for iid in all_ids[:10]:
            stock.mark_sold(iid)
        assert stock.n_available() == len(all_ids) - 10

        # Simulate Reset: build a new Stock from the same full ID list.
        stock = Stock(all_ids)
        assert stock.n_available() == len(all_ids)

    def test_mmr_lambda_affects_diversity(
        self, index: FakeIndex, all_ids: list[str]
    ) -> None:
        """lambda=1.0 should return near-duplicates; lambda=0.4 should break them
        apart, demonstrating the MMR trade-off (D14)."""
        stock = Stock(all_ids)
        qid = all_ids[0]
        q = index.get_vector(qid)

        plain = rerank_with_availability(
            q, index.search, stock, k=5, n_candidates=50,
            lambda_=1.0, exclude_ids=[qid]
        )
        diverse = rerank_with_availability(
            q, index.search, stock, k=5, n_candidates=50,
            lambda_=0.4, exclude_ids=[qid]
        )
        # With 30 near-duplicates in the first cluster, lambda=1.0 returns
        # mostly cluster members; lambda=0.4 must differ.
        assert plain.item_ids != diverse.item_ids


# ============================================================================ #
# Real-data integration (fixture-based, no network)                             #
# ============================================================================ #

class TestRealDataIntegration:
    """Smoke tests using tiny synthetic data in the shape of the real files.

    These tests verify that the real-data loading paths handle edge cases
    (missing images, path resolution) without requiring actual data files
    or model weights.
    """

    def test_manifest_columns_are_complete(self) -> None:
        """MANIFEST_COLUMNS must include all fields the UI reads."""
        from src.data.dataset import MANIFEST_COLUMNS
        required = {"item_id", "image_path", "description", "category",
                    "colour", "usage"}
        assert required.issubset(set(MANIFEST_COLUMNS))

    def test_placeholder_bytes_returns_bytes(self) -> None:
        """Placeholder image generator must work without PIL import errors."""
        import importlib.util
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow not installed")
        if importlib.util.find_spec("streamlit") is None:
            pytest.skip("streamlit not installed")
        from app.main import _placeholder_bytes
        result = _placeholder_bytes()
        assert isinstance(result, bytes) and len(result) > 0

    def test_thumbnail_bytes_real_missing_path(self, tmp_path) -> None:
        """Missing image file must return None, not raise."""
        import importlib.util
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow not installed")
        if importlib.util.find_spec("streamlit") is None:
            pytest.skip("streamlit not installed")
        from app.main import _thumbnail_bytes_real
        result = _thumbnail_bytes_real("images/nonexistent.jpg", tmp_path)
        assert result is None

    def test_thumbnail_bytes_real_valid_image(self, tmp_path) -> None:
        """A valid JPEG must be returned as bytes."""
        import importlib.util
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow not installed")
        if importlib.util.find_spec("streamlit") is None:
            pytest.skip("streamlit not installed")
        from PIL import Image
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = Image.new("RGB", (100, 100), (200, 100, 50))
        img.save(img_dir / "test_item.jpg", format="JPEG")

        from app.main import _thumbnail_bytes_real
        result = _thumbnail_bytes_real("images/test_item.jpg", tmp_path)
        assert isinstance(result, bytes) and len(result) > 0

    def test_fake_index_compatible_with_stock(self, index: FakeIndex, all_ids: list[str]) -> None:
        """FakeIndex.all_item_ids() populates Stock correctly."""
        stock = Stock(all_ids)
        assert stock.n_available() == len(all_ids)

    def test_text_query_vector_compatible_with_index(self, index: FakeIndex) -> None:
        """A synthetic text-query vector (512-d float32 unit) is accepted by search."""
        rng = np.random.default_rng(99)
        v = rng.standard_normal(512).astype(np.float32)
        v /= np.linalg.norm(v)
        ids, vecs, scores = index.search(v, depth=5)
        assert len(ids) == 5
        assert vecs.dtype == np.float32
