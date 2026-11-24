"""Tests for the TODS supplement helpers in utils/gtfs_helpers.py."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from utils.gtfs_helpers import apply_supplement, drop_trips_on_deleted_routes


def _trips() -> pd.DataFrame:
    return pd.DataFrame(
        {"trip_id": ["T1", "T2"], "route_id": ["R1", "R1"], "block_id": ["B1", "B1"]}
    )


def test_apply_supplement_updates_deletes_and_adds_rows() -> None:
    supplement = pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R2", "", "R1"],
            "block_id": ["", "", "B2"],
            "TODS_delete": ["", "1", ""],
        }
    )
    out = apply_supplement(_trips(), supplement, ["trip_id"])
    assert out.to_dict("records") == [
        # Blank supplement values leave the GTFS value in place.
        {"trip_id": "T1", "route_id": "R2", "block_id": "B1"},
        {"trip_id": "T3", "route_id": "R1", "block_id": "B2"},
    ]


def test_apply_supplement_without_supplement_returns_a_copy() -> None:
    trips = _trips()
    out = apply_supplement(trips, None, ["trip_id"])
    pd.testing.assert_frame_equal(out, trips)
    assert out is not trips


def test_apply_supplement_rejects_missing_keys_and_delete_then_readd() -> None:
    with pytest.raises(ValueError, match="missing primary-key"):
        apply_supplement(_trips(), pd.DataFrame({"route_id": ["R1"]}), ["trip_id"])
    conflict = pd.DataFrame({"trip_id": ["T1", "T1"], "TODS_delete": ["1", ""]})
    with pytest.raises(ValueError, match="both deletes and re-adds"):
        apply_supplement(_trips(), conflict, ["trip_id"])


def test_apply_supplement_warns_on_delete_of_unknown_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        out = apply_supplement(
            _trips(), pd.DataFrame({"trip_id": ["T9"], "TODS_delete": ["1"]}), ["trip_id"]
        )
    assert len(out) == 2
    assert "matches no GTFS row" in caplog.text


def test_drop_trips_on_deleted_routes_keeps_reassigned_trips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    routes = pd.DataFrame({"route_id": ["R1", "R2"]})
    routes_after = apply_supplement(
        routes, pd.DataFrame({"route_id": ["R1"], "TODS_delete": ["1"]}), ["route_id"]
    )
    # T1 was moved to R2 by trips_supplement.txt before routes were checked.
    trips = pd.DataFrame({"trip_id": ["T1", "T2"], "route_id": ["R2", "R1"]})
    with caplog.at_level(logging.WARNING):
        kept = drop_trips_on_deleted_routes(trips, routes, routes_after)
    assert kept["trip_id"].tolist() == ["T1"]
    assert "Ignoring 1 trip(s) on route(s) deleted in routes_supplement.txt (R1)" in caplog.text


def test_drop_trips_on_deleted_routes_without_deletions_is_a_no_op() -> None:
    routes = pd.DataFrame({"route_id": ["R1"]})
    trips = _trips()
    assert drop_trips_on_deleted_routes(trips, routes, routes) is trips
