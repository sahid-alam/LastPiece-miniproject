import pandas as pd
import pytest

from src.data.dataset import MANIFEST_COLUMNS, relevant_items


def make_manifest():
    rows = [
        # item_id, category, colour, usage
        ("1", "Tshirts", "Blue", "Casual"),   # the query item
        ("2", "Tshirts", "Blue", "Formal"),   # matches on colour only
        ("3", "Tshirts", "Red", "Casual"),    # matches on usage only
        ("4", "Tshirts", "Blue", "Casual"),   # matches on both
        ("5", "Tshirts", "Green", "Sports"),  # matches on neither
        ("6", "Shirts", "Blue", "Casual"),    # wrong category
        ("7", "Tshirts", "", ""),             # empty values
    ]
    df = pd.DataFrame(
        [
            {
                "item_id": i,
                "image_path": f"images/{i}.jpg",
                "description": f"item {i}",
                "category": cat,
                "colour": col,
                "usage": use,
            }
            for i, cat, col, use in rows
        ]
    )
    return df[MANIFEST_COLUMNS]


ATTRS = ["colour", "usage"]


def test_one_match_is_enough():
    result = relevant_items(make_manifest(), "1", ATTRS, 1)
    assert result == {"2", "3", "4"}


def test_two_matches_is_stricter():
    result = relevant_items(make_manifest(), "1", ATTRS, 2)
    assert result == {"4"}


def test_query_item_is_excluded():
    result = relevant_items(make_manifest(), "1", ATTRS, 1)
    assert "1" not in result


def test_other_category_never_relevant():
    result = relevant_items(make_manifest(), "1", ATTRS, 1)
    assert "6" not in result


def test_empty_values_never_match():
    # Item 7 has empty colour/usage, so it matches nothing, even
    # though the query's other fields exist.
    result = relevant_items(make_manifest(), "1", ATTRS, 1)
    assert "7" not in result


def test_empty_query_attributes_match_nothing():
    # Query item 7 has empty colour and usage, so nothing can match.
    result = relevant_items(make_manifest(), "7", ATTRS, 1)
    assert result == set()


def test_unknown_query_id_raises():
    with pytest.raises(KeyError):
        relevant_items(make_manifest(), "999", ATTRS, 1)