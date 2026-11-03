from __future__ import annotations

import string

from utils.gtfs_helpers import stop_pattern_id


def test_stop_pattern_id_is_pinned_value() -> None:
    # A fixed value proves the ID doesn't depend on PYTHONHASHSEED or the process.
    assert stop_pattern_id(["S1", "S2", "S3", "S4", "S5", "S6"]) == "92da08bf"


def test_stop_pattern_id_is_8_lowercase_hex_chars() -> None:
    result = stop_pattern_id(("S1", "S2"))
    assert isinstance(result, str)
    assert len(result) == 8
    assert set(result) <= set(string.hexdigits.lower())


def test_stop_pattern_id_accepts_any_iterable() -> None:
    stops = ["S1", "S2", "S3"]
    assert stop_pattern_id(stops) == stop_pattern_id(tuple(stops)) == stop_pattern_id(iter(stops))


def test_stop_pattern_id_order_matters() -> None:
    assert stop_pattern_id(["S1", "S2"]) != stop_pattern_id(["S2", "S1"])


def test_stop_pattern_id_separator_in_stop_id_does_not_collide() -> None:
    assert stop_pattern_id(["a,b"]) != stop_pattern_id(["a", "b"])
    assert stop_pattern_id(["a|b"]) != stop_pattern_id(["a", "b"])


def test_stop_pattern_id_coerces_non_strings() -> None:
    assert stop_pattern_id([1001, 1002]) == stop_pattern_id(["1001", "1002"])


def test_stop_pattern_id_empty_pattern_is_still_an_id() -> None:
    assert len(stop_pattern_id([])) == 8
