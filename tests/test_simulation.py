"""Tests for the simulated session."""

from __future__ import annotations

import pytest

from src.eval.simulation import simulate_session

IDS = [f"i{n}" for n in range(200)]


def _run(**kw):
    args = dict(n_users=10, queries_per_user=4, deplete_fraction=0.9, seed=42)
    args.update(kw)
    return simulate_session(IDS, **args)


def test_same_seed_same_session():
    """Every system under test must replay the identical session."""
    assert _run() == _run()


def test_different_seed_different_session():
    assert _run(seed=1) != _run(seed=2)


def test_query_item_is_still_for_sale_when_queried():
    sold: set[str] = set()
    for step in _run():
        assert step.query_id not in sold
        sold.update(step.sold_after)


def test_each_item_sells_at_most_once():
    all_sold = [iid for step in _run() for iid in step.sold_after]
    assert len(all_sold) == len(set(all_sold))


def test_depletes_the_requested_fraction():
    """Late-session state must actually be reached, or substitution is never
    exercised."""
    all_sold = [iid for step in _run() for iid in step.sold_after]
    assert len(all_sold) == int(0.9 * len(IDS))


def test_users_issue_consecutive_queries():
    steps = _run(n_users=3, queries_per_user=2)
    assert [s.user_id for s in steps] == ["u0", "u0", "u1", "u1", "u2", "u2"]


@pytest.mark.parametrize("bad", [-0.1, 1.0])
def test_bad_deplete_fraction_raises(bad):
    with pytest.raises(ValueError, match="deplete_fraction"):
        _run(deplete_fraction=bad)
