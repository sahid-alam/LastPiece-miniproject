"""Tests for the experiment runner: replay protocol and D15 scoring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.runner import content_only, ours, replay
from src.eval.simulation import Step
from src.index.stock import Stock

K = 2


@pytest.fixture
def index(clustered_candidates, retrieve_fn):
    vecs, ids = clustered_candidates
    return SimpleNamespace(search=retrieve_fn, get_vector=lambda i: vecs[ids.index(i)])


@pytest.fixture
def ids(clustered_candidates) -> list[str]:
    return clustered_candidates[1]


def fixed(*lists: list[str]):
    """A system that returns pre-set lists in order, ignoring the stock."""
    it = iter(lists)
    return lambda query_id, stock: next(it)


def run(steps, ids, system, index, relevant=frozenset()):
    return replay(steps, ids, system, lambda q: set(relevant), index.get_vector, K)


def test_sold_item_shown_is_a_miss_and_not_coverage(ids, index):
    """D15: dup_b sold after step 0, so showing it at step 1 earns nothing."""
    steps = [Step("u0", "far", ("dup_b",)), Step("u0", "far", ())]
    system = fixed(["mid", "near"], ["dup_b", "dup_c"])
    r = run(steps, ids, system, index, relevant={"dup_b", "dup_c"})

    # Step 0: 0/2 relevant. Step 1: dup_b sold (miss), dup_c hit -> 1/2.
    assert r.precision_at_k == pytest.approx(0.25)
    # mid, near, dup_c were available when shown; dup_b was not.
    assert r.coverage == pytest.approx(3 / len(ids))


def test_sold_shown_pct_hand_checked(ids, index):
    steps = [Step("u0", "far", ("dup_a", "dup_b")), Step("u0", "far", ())]
    system = fixed(["dup_a", "near"], ["dup_a", "dup_b"])
    r = run(steps, ids, system, index)
    # 4 shown slots; 2 of them (both in step 1) were already sold.
    assert r.sold_shown_pct == pytest.approx(50.0)
    assert [s.n_sold_shown for s in r.steps] == [0, 2]


def test_each_replay_gets_a_fresh_full_stock(ids, index):
    steps = [Step("u0", "far", ("dup_a", "dup_b", "dup_c"))]
    seen: list[int] = []

    def spy(query_id: str, stock: Stock) -> list[str]:
        seen.append(stock.n_available())
        return []

    run(steps, ids, spy, index)
    run(steps, ids, spy, index)
    assert seen == [len(ids), len(ids)]


def test_all_systems_see_identical_steps_and_stock(ids, index):
    steps = [Step("u0", "far", ("dup_a",)), Step("u1", "mid", ("near", "dup_b"))]
    log: dict[str, list] = {"a": [], "b": []}

    def spy(name):
        return lambda q, stock: log[name].append((q, stock.available_ids())) or []

    run(steps, ids, spy("a"), index)
    run(steps, ids, spy("b"), index)
    assert log["a"] == log["b"]
    assert log["a"][1][1] == set(ids) - {"dup_a"}


def test_ours_never_shows_sold_or_query_item(ids, index):
    steps = [Step("u0", "dup_a", ("dup_b",)), Step("u0", "dup_a", ("near",)), Step("u0", "dup_a", ())]
    r = run(steps, ids, ours(index, 0.7, K, 3, len(ids)), index)
    assert r.sold_shown_pct == 0.0
    assert r.short_list_rate == 0.0
    assert all("dup_a" not in s.shown for s in r.steps)


def test_content_only_shows_sold_items(ids, index):
    """No stock filter: that is the baseline's point, and D15 must catch it."""
    steps = [Step("u0", "dup_a", ("dup_b",)), Step("u0", "dup_a", ())]
    r = run(steps, ids, content_only(index, K), index)
    assert r.steps[1].shown == ("dup_b", "dup_c")
    assert "dup_a" not in r.steps[1].shown
    assert r.sold_shown_pct == pytest.approx(25.0)


def test_latency_recorded_per_step(ids, index):
    steps = [Step("u0", "far", ()) for _ in range(3)]
    r = run(steps, ids, content_only(index, K), index)
    assert len(r.steps) == 3
    assert all(s.latency_ms >= 0.0 for s in r.steps)
    assert np.isfinite(r.latency_mean_ms) and np.isfinite(r.latency_p95_ms)


def test_stock_remaining_and_short_lists(ids, index):
    steps = [Step("u0", "far", ("dup_a", "dup_b", "dup_c")), Step("u0", "far", ())]
    r = run(steps, ids, fixed(["near"], ["near", "mid"]), index)
    assert [s.stock_remaining for s in r.steps] == pytest.approx([1.0, 0.5])
    assert r.short_list_rate == pytest.approx(0.5)


def test_after_step_hook_sees_only_the_past(ids, index):
    """The CF baseline (#14) learns via after_step; it must run after the
    step is served, never before."""
    order: list[str] = []

    def system(q, stock):
        order.append(f"serve {q}")
        return []

    system.after_step = lambda st: order.append(f"learn {st.query_id}")
    run([Step("u0", "far", ()), Step("u0", "mid", ())], ids, system, index)
    assert order == ["serve far", "learn far", "serve mid", "learn mid"]
