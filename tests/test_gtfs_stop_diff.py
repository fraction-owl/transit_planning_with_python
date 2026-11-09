from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd
import pytest

from scripts.stop_analysis import gtfs_stop_diff as gsd
from scripts.stop_analysis.gtfs_stop_diff import (
    Feed,
    addition_output_tables,
    candidate_matches,
    cli_pairing_options,
    compare_shared,
    comparison_inventory,
    haversine_feet,
    join_labels,
    load_route_context,
    load_stops,
    main,
    make_scope,
    normalize_route_names,
    normalize_route_pairs,
    normalize_text,
    normalized_attribute,
    parse_args,
    prompt_route_pair,
    read_gtfs_table,
    read_pairing_file,
    route_counterparts,
    route_sort_key,
    run_compare,
    service_day_types,
    stop_id_code_changes,
    write_outputs,
)

# ---------------------------------------------------------------------------
# Synthetic feeds
# ---------------------------------------------------------------------------

# Current feed: route 101 serves B1-B3; route 200 serves only B4.
BEFORE_STOPS = [
    {"stop_id": "B1", "stop_code": "1001", "stop_name": "Main St & 1st St",
     "stop_lat": "38.8000", "stop_lon": "-77.1000", "location_type": ""},
    {"stop_id": "B2", "stop_code": "1002", "stop_name": "Main St & 2nd St",
     "stop_lat": "38.8100", "stop_lon": "-77.1000", "location_type": ""},
    {"stop_id": "B3", "stop_code": "1003", "stop_name": "Oak Ave",
     "stop_lat": "38.8200", "stop_lon": "-77.1000", "location_type": ""},
    {"stop_id": "B4", "stop_code": "1004", "stop_name": "Elm St",
     "stop_lat": "38.8300", "stop_lon": "-77.1000", "location_type": ""},
]  # fmt: skip
BEFORE_ROUTES = [
    {"route_id": "R101", "route_short_name": "101", "route_long_name": "Main"},
    {"route_id": "R200", "route_short_name": "200", "route_long_name": "Elm"},
]
BEFORE_TRIPS = [
    {"trip_id": "T1", "route_id": "R101"},
    {"trip_id": "T2", "route_id": "R200"},
]
BEFORE_STOP_TIMES = [
    {"trip_id": "T1", "stop_id": "B1", "stop_sequence": "1"},
    {"trip_id": "T1", "stop_id": "B2", "stop_sequence": "2"},
    {"trip_id": "T1", "stop_id": "B3", "stop_sequence": "3"},
    {"trip_id": "T2", "stop_id": "B4", "stop_sequence": "1"},
]

# Concept feed: route "101" under a generated route_id and new stop_ids.
# 1001 unchanged (location_type "" vs "0" is equivalent), 1002 renamed,
# 1003 dropped from the route, 1004 newly used, 2005 a brand-new code
# placed ~36 ft from B3.
AFTER_STOPS = [
    {"stop_id": "A1", "stop_code": "1001", "stop_name": "Main St & 1st St",
     "stop_lat": "38.8000", "stop_lon": "-77.1000", "location_type": "0"},
    {"stop_id": "A2", "stop_code": "1002", "stop_name": "Main St & Second St",
     "stop_lat": "38.8100", "stop_lon": "-77.1000", "location_type": "0"},
    {"stop_id": "A4", "stop_code": "1004", "stop_name": "Elm St",
     "stop_lat": "38.8300", "stop_lon": "-77.1000", "location_type": "0"},
    {"stop_id": "A5", "stop_code": "2005", "stop_name": "Oak Ave (new)",
     "stop_lat": "38.8201", "stop_lon": "-77.1000", "location_type": "0"},
]  # fmt: skip
AFTER_ROUTES = [{"route_id": "X1", "route_short_name": "101", "route_long_name": ""}]
AFTER_TRIPS = [{"trip_id": "AT1", "route_id": "X1"}]
AFTER_STOP_TIMES = [
    {"trip_id": "AT1", "stop_id": stop_id, "stop_sequence": str(seq)}
    for seq, stop_id in enumerate(["A1", "A2", "A4", "A5"], start=1)
]


def write_feed(
    root: Path,
    stops: list[dict[str, str]],
    routes: list[dict[str, str]] | None = None,
    trips: list[dict[str, str]] | None = None,
    stop_times: list[dict[str, str]] | None = None,
) -> Path:
    """Write a minimal GTFS folder of string columns."""
    root.mkdir(parents=True, exist_ok=True)
    tables = {"stops": stops, "routes": routes, "trips": trips, "stop_times": stop_times}
    for name, rows in tables.items():
        if rows is not None:
            pd.DataFrame(rows).to_csv(root / f"{name}.txt", index=False)
    return root


@pytest.fixture
def before_feed(tmp_path: Path) -> Path:
    return write_feed(
        tmp_path / "before", BEFORE_STOPS, BEFORE_ROUTES, BEFORE_TRIPS, BEFORE_STOP_TIMES
    )


@pytest.fixture
def after_feed(tmp_path: Path) -> Path:
    return write_feed(tmp_path / "after", AFTER_STOPS, AFTER_ROUTES, AFTER_TRIPS, AFTER_STOP_TIMES)


@pytest.fixture(autouse=True)
def _reset_logger() -> Iterator[None]:
    """run_compare attaches file handlers; close them so tmp dirs can be removed."""
    yield
    for handler in list(gsd.LOGGER.handlers):
        gsd.LOGGER.removeHandler(handler)
        handler.close()


def write_calendar(root: Path, service_days: dict[str, str]) -> None:
    """Write calendar.txt where each service_id runs weekdays, saturday, or sunday."""
    rows = []
    for service_id, day_type in service_days.items():
        row = {"service_id": service_id, "start_date": "20260101", "end_date": "20261231"}
        for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
            row[day] = "1" if day_type == "weekday" else "0"
        row["saturday"] = "1" if day_type == "saturday" else "0"
        row["sunday"] = "1" if day_type == "sunday" else "0"
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "calendar.txt", index=False)


def stops_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a loaded-stops-like frame with numeric coordinates and status."""
    frame = pd.DataFrame(rows)
    if "coordinate_status" not in frame:
        frame["coordinate_status"] = "valid"
    frame["coordinate_status"] = frame["coordinate_status"].fillna("valid")
    return frame


def make_feed(stop_ids: list[str], routes: dict[str, str], pairs: list[tuple[str, str]]) -> Feed:
    """Build a Feed directly: routes maps route_id -> label; pairs are (stop_id, route_id)."""
    route_frame = pd.DataFrame({"route_id": list(routes), "route_label": list(routes.values())})
    pair_frame = pd.DataFrame(pairs, columns=["stop_id", "route_id"]).merge(
        route_frame, on="route_id"
    )
    return Feed(pd.DataFrame({"stop_id": stop_ids}), route_frame, pair_frame, Path("."))


# ---------------------------------------------------------------------------
# Text and sorting helpers
# ---------------------------------------------------------------------------


def test_normalize_text_preserves_literal_ids_and_strips() -> None:
    result = normalize_text(pd.Series(["  0042 ", "NA", None]))
    assert result.tolist() == ["0042", "NA", ""]


def test_normalized_attribute_fills_gtfs_defaults_only() -> None:
    frame = pd.DataFrame({"location_type": ["", "1"], "stop_name": ["", "Main"]})
    assert normalized_attribute(frame, "location_type").tolist() == ["0", "1"]
    assert normalized_attribute(frame, "stop_name").tolist() == ["", "Main"]


def test_normalized_attribute_absent_column_uses_default() -> None:
    frame = pd.DataFrame({"stop_id": ["S1"]})
    assert normalized_attribute(frame, "wheelchair_boarding").tolist() == ["0"]


def test_route_sort_key_and_join_labels_numeric_first() -> None:
    assert sorted(["10", "2", "B", "a"], key=route_sort_key) == ["2", "10", "a", "B"]
    assert join_labels(["10", "2", "2", "Express"]) == "2; 10; Express"


def test_haversine_feet_one_degree_latitude() -> None:
    expected = 6_371_000.0 * 3.280839895013123 * math.pi / 180
    result = haversine_feet(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert result[0] == pytest.approx(expected, rel=1e-9)


def test_haversine_feet_identical_points_is_zero() -> None:
    point = np.array([38.8])
    assert haversine_feet(point, point, point, point)[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reading GTFS (folders and ZIPs)
# ---------------------------------------------------------------------------


def test_read_gtfs_table_folder_keeps_strings(tmp_path: Path) -> None:
    write_feed(tmp_path, [{"stop_id": "007", "stop_name": "NA"}])
    frame = read_gtfs_table(tmp_path, "stops.txt")
    assert frame.loc[0, "stop_id"] == "007"
    assert frame.loc[0, "stop_name"] == "NA"


def test_read_gtfs_table_nested_zip_ignores_macosx(tmp_path: Path) -> None:
    archive = tmp_path / "feed.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("wrapper/stops.txt", "stop_id,stop_lat,stop_lon\nS1,38.8,-77.1\n")
        zf.writestr("__MACOSX/wrapper/stops.txt", "junk")
    frame = read_gtfs_table(archive, "stops.txt")
    assert frame["stop_id"].tolist() == ["S1"]


def test_read_gtfs_table_ambiguous_zip_raises(tmp_path: Path) -> None:
    archive = tmp_path / "feed.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/stops.txt", "stop_id\nS1\n")
        zf.writestr("b/stops.txt", "stop_id\nS2\n")
    with pytest.raises(ValueError, match="ambiguous"):
        read_gtfs_table(archive, "stops.txt")


def test_read_gtfs_table_missing_member_raises(tmp_path: Path) -> None:
    archive = tmp_path / "feed.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("routes.txt", "route_id\nR1\n")
    with pytest.raises(FileNotFoundError, match="missing stops.txt"):
        read_gtfs_table(archive, "stops.txt")


def test_read_gtfs_table_rejects_non_gtfs_file(tmp_path: Path) -> None:
    source = tmp_path / "stops.csv"
    source.write_text("stop_id\nS1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="folder or ZIP"):
        read_gtfs_table(source, "stops.txt")


def test_read_gtfs_table_usecols_requires_columns(tmp_path: Path) -> None:
    write_feed(tmp_path, [{"stop_id": "S1"}], stop_times=[{"trip_id": "T1", "x": "1"}])
    with pytest.raises(ValueError, match="missing columns"):
        read_gtfs_table(tmp_path, "stop_times.txt", ["trip_id", "stop_id"])


# ---------------------------------------------------------------------------
# load_stops / load_route_context
# ---------------------------------------------------------------------------


def test_load_stops_classifies_coordinate_status(tmp_path: Path) -> None:
    rows = [
        {"stop_id": "OK", "stop_lat": "38.8", "stop_lon": "-77.1", "location_type": ""},
        {"stop_id": "NODE", "stop_lat": "", "stop_lon": "", "location_type": "3"},
        {"stop_id": "PART", "stop_lat": "", "stop_lon": "-77.1", "location_type": ""},
        {"stop_id": "BAD", "stop_lat": "95", "stop_lon": "-77.1", "location_type": ""},
    ]
    stops = load_stops(write_feed(tmp_path, rows), "test").set_index("stop_id")
    assert stops.loc["OK", "coordinate_status"] == "valid"
    assert stops.loc["NODE", "coordinate_status"] == "not_provided_optional"
    assert stops.loc["PART", "coordinate_status"] == "missing_required_or_partial"
    assert stops.loc["BAD", "coordinate_status"] == "invalid_or_out_of_range"
    assert stops.loc["BAD", "raw_stop_lat"] == "95"


def test_load_stops_rejects_duplicate_stop_ids(tmp_path: Path) -> None:
    rows = [{"stop_id": "S1", "stop_lat": "38.8", "stop_lon": "-77.1"}] * 2
    with pytest.raises(ValueError, match="duplicate stop_id"):
        load_stops(write_feed(tmp_path, rows), "test")


def test_load_stops_rejects_blank_stop_ids(tmp_path: Path) -> None:
    rows = [{"stop_id": "", "stop_lat": "38.8", "stop_lon": "-77.1"}]
    with pytest.raises(ValueError, match="blank IDs"):
        load_stops(write_feed(tmp_path, rows), "test")


def test_load_route_context_pairs_and_label_fallback(tmp_path: Path) -> None:
    routes = [
        {"route_id": "R1", "route_short_name": "", "route_long_name": "Crosstown"},
        {"route_id": "R2", "route_short_name": "", "route_long_name": ""},
    ]
    trips = [{"trip_id": "T1", "route_id": "R1"}, {"trip_id": "T2", "route_id": "R2"}]
    stop_times = [
        {"trip_id": "T1", "stop_id": "B1"},
        {"trip_id": "T1", "stop_id": "B1"},
        {"trip_id": "T2", "stop_id": "B2"},
    ]
    feed = write_feed(tmp_path, BEFORE_STOPS, routes, trips, stop_times)
    routes_out, pairs = load_route_context(feed, load_stops(feed, "test"))
    assert routes_out["route_label"].tolist() == ["Crosstown", "R2"]
    assert sorted(map(tuple, pairs[["stop_id", "route_label"]].to_numpy())) == [
        ("B1", "Crosstown"),
        ("B2", "R2"),
    ]


def test_service_day_types_uses_calendar_flags_and_added_dates() -> None:
    calendar = pd.DataFrame(
        [
            {"service_id": "WK", "monday": "1", "tuesday": "1", "wednesday": "1",
             "thursday": "1", "friday": "1", "saturday": "0", "sunday": "0"},
            {"service_id": "SA", "monday": "0", "tuesday": "0", "wednesday": "0",
             "thursday": "0", "friday": "0", "saturday": "1", "sunday": "0"},
        ]
    )  # fmt: skip
    calendar_dates = pd.DataFrame(
        [
            {"service_id": "HOL", "date": "20261115", "exception_type": "1"},  # Sunday
            {"service_id": "WK", "date": "20261114", "exception_type": "2"},  # removal
            {"service_id": "BAD", "date": "2026-11-15", "exception_type": "1"},
        ]
    )
    assert service_day_types(calendar, calendar_dates) == {
        "WK": frozenset({"weekday"}),
        "SA": frozenset({"saturday"}),
        "HOL": frozenset({"sunday"}),
    }


def test_load_route_context_adds_day_types_only_with_calendars(tmp_path: Path) -> None:
    trips = [{"trip_id": "T1", "route_id": "R101", "service_id": "WK"},
             {"trip_id": "T2", "route_id": "R101", "service_id": "SA"}]  # fmt: skip
    stop_times = [{"trip_id": "T1", "stop_id": "B1"}, {"trip_id": "T2", "stop_id": "B1"},
                  {"trip_id": "T1", "stop_id": "B2"}]  # fmt: skip
    feed = write_feed(tmp_path, BEFORE_STOPS, BEFORE_ROUTES, trips, stop_times)
    _, pairs = load_route_context(feed, load_stops(feed, "test"))
    assert "day_types" not in pairs
    write_calendar(feed, {"WK": "weekday", "SA": "saturday"})
    _, pairs = load_route_context(feed, load_stops(feed, "test"))
    days = dict(zip(pairs["stop_id"], pairs["day_types"]))
    assert days == {"B1": {"weekday", "saturday"}, "B2": {"weekday"}}


def test_route_counterparts_all_stops_pairs_by_label_or_route_id() -> None:
    before = make_feed(["B1"], {"R1": "10", "R2": "20", "R3": "30"}, [("B1", "R1")])
    after = make_feed(["A1"], {"Q1": "10", "Q2": "11", "R3": "30X"}, [("A1", "Q1")])
    scope = make_scope(before, after, "all_stops", [], {}, [], "error")
    assert route_counterparts(before, after, scope, "all_stops") == {("R1", "Q1"), ("R3", "R3")}


def test_load_route_context_rejects_unknown_stop(tmp_path: Path) -> None:
    stop_times = [{"trip_id": "T1", "stop_id": "GHOST"}]
    feed = write_feed(tmp_path, BEFORE_STOPS, BEFORE_ROUTES, BEFORE_TRIPS, stop_times)
    with pytest.raises(ValueError, match="unknown stop_id"):
        load_route_context(feed, load_stops(feed, "test"))


# ---------------------------------------------------------------------------
# comparison_inventory / stop_id_code_changes
# ---------------------------------------------------------------------------


def test_comparison_inventory_sets_comparison_key() -> None:
    out = comparison_inventory(
        pd.DataFrame({"stop_id": ["S1"], "stop_code": [" 0042 "]}), "stop_code", "test"
    )
    assert out.loc[0, "comparison_key"] == "0042"
    assert out.loc[0, "primary_stop_key"] == "stop_code"


def test_comparison_inventory_blank_key_raises_without_fallback() -> None:
    frame = pd.DataFrame({"stop_id": ["S1", "S2"], "stop_code": ["1", ""]})
    with pytest.raises(ValueError, match="blank for 1 records"):
        comparison_inventory(frame, "stop_code", "test")


def test_comparison_inventory_duplicate_key_raises() -> None:
    frame = pd.DataFrame({"stop_id": ["S1", "S2"], "stop_code": ["1", "1"]})
    with pytest.raises(ValueError, match="must be unique"):
        comparison_inventory(frame, "stop_code", "test")


def test_comparison_inventory_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="primary stop key"):
        comparison_inventory(pd.DataFrame({"stop_name": ["x"]}), "stop_name", "test")


def test_stop_id_code_changes_flags_reused_ids() -> None:
    before = pd.DataFrame({"stop_id": ["S1", "S2"], "stop_code": ["1", "2"]})
    after = pd.DataFrame({"stop_id": ["S1", "S2"], "stop_code": ["9", "2"]})
    out = stop_id_code_changes(before, after)
    assert out["stop_id"].tolist() == ["S1"]
    assert (out.loc[0, "stop_code_before"], out.loc[0, "stop_code_after"]) == ("1", "9")


def test_stop_id_code_changes_without_codes_is_empty() -> None:
    out = stop_id_code_changes(pd.DataFrame({"stop_id": ["S1"]}), pd.DataFrame({"stop_id": ["S1"]}))
    assert out.empty
    assert "review_note" in out.columns


# ---------------------------------------------------------------------------
# compare_shared
# ---------------------------------------------------------------------------


def shared_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    before = stops_frame(
        [
            {"stop_id": "B1", "stop_code": "1", "stop_name": "A", "stop_lat": 38.8,
             "stop_lon": -77.1, "location_type": ""},
            {"stop_id": "B2", "stop_code": "2", "stop_name": "B", "stop_lat": 38.8,
             "stop_lon": -77.1, "location_type": ""},
            {"stop_id": "B3", "stop_code": "3", "stop_name": "C", "stop_lat": 38.8,
             "stop_lon": -77.1, "location_type": ""},
            {"stop_id": "B4", "stop_code": "4", "stop_name": "D", "stop_lat": 38.8,
             "stop_lon": -77.1, "location_type": ""},
        ]
    )  # fmt: skip
    after = stops_frame(
        [
            # unchanged (equivalent location_type default, new stop_id)
            {"stop_id": "A1", "stop_code": "1", "stop_name": "A", "stop_lat": 38.8,
             "stop_lon": -77.1, "location_type": "0"},
            # moved ~364 ft
            {"stop_id": "B2", "stop_code": "2", "stop_name": "B", "stop_lat": 38.801,
             "stop_lon": -77.1, "location_type": "0"},
            # renamed
            {"stop_id": "B3", "stop_code": "3", "stop_name": "C2", "stop_lat": 38.8,
             "stop_lon": -77.1, "location_type": "0"},
            # coordinates unusable on the after side
            {"stop_id": "B4", "stop_code": "4", "stop_name": "D", "stop_lat": np.nan,
             "stop_lon": np.nan, "location_type": "0",
             "coordinate_status": "missing_required_or_partial"},
        ]
    )  # fmt: skip
    return before, after


def test_compare_shared_classifies_change_types() -> None:
    before, after = shared_pair()
    result, _, _, _ = compare_shared(before, after, 25.0, [])
    by_code = result.set_index("stop_code")
    assert by_code.loc["1", "change_type"] == "unchanged"
    assert by_code.loc["2", "change_type"] == "coordinates"
    assert by_code.loc["2", "distance_ft"] == pytest.approx(364.8, abs=1.0)
    assert by_code.loc["3", "change_type"] == "attributes"
    assert by_code.loc["3", "changed_fields"] == "stop_name"
    assert by_code.loc["4", "change_type"] == "needs_review"
    assert by_code.loc["4", "location_assessment"] == "unable_to_assess"


def test_compare_shared_secondary_id_change_is_not_an_attribute_change() -> None:
    before, after = shared_pair()
    result, _, _, _ = compare_shared(before, after, 25.0, [])
    row = result.set_index("stop_code").loc["1"]
    assert bool(row["secondary_id_changed"]) is True
    assert bool(row["attributes_changed"]) is False


def test_compare_shared_attribute_summary_counts_equivalent_raw_changes() -> None:
    before, after = shared_pair()
    _, summary, details, schema = compare_shared(before, after, 25.0, [])
    location = summary.set_index("field").loc["location_type"]
    assert location["changed_count"] == 0
    assert location["equivalent_raw_changes"] == 4
    assert details["field"].tolist() == ["stop_name"]
    assert set(schema["schema_change"]) == {"same"}


def test_compare_shared_ignored_attribute_is_not_counted() -> None:
    before, after = shared_pair()
    result, _, _, schema = compare_shared(before, after, 25.0, ["stop_name"])
    assert result.set_index("stop_code").loc["3", "change_type"] == "unchanged"
    included = schema.set_index("field")["included_in_classification"]
    assert bool(included["stop_name"]) is False


def test_compare_shared_threshold_is_respected() -> None:
    before, after = shared_pair()
    result, _, _, _ = compare_shared(before, after, 1000.0, [])
    assert result.set_index("stop_code").loc["2", "change_type"] == "unchanged"


@pytest.mark.parametrize(
    ("threshold", "ignored", "match"),
    [(-1.0, [], "nonnegative"), (math.inf, [], "finite"), (25.0, ["bogus"], "IGNORE")],
)
def test_compare_shared_rejects_bad_settings(
    threshold: float, ignored: list[str], match: str
) -> None:
    before, after = shared_pair()
    with pytest.raises(ValueError, match=match):
        compare_shared(before, after, threshold, ignored)


# ---------------------------------------------------------------------------
# candidate_matches
# ---------------------------------------------------------------------------


def test_candidate_matches_finds_nearby_stop() -> None:
    before = stops_frame(
        [{"stop_id": "B3", "stop_code": "1003", "stop_name": "Oak", "stop_lat": 38.82,
          "stop_lon": -77.1}]
    )  # fmt: skip
    after = stops_frame(
        [{"stop_id": "A5", "stop_code": "2005", "stop_name": "Oak new", "stop_lat": 38.8201,
          "stop_lon": -77.1}]
    )  # fmt: skip
    out = candidate_matches(before, after, 500.0, 3)
    assert len(out) == 1
    assert out.loc[0, "before_comparison_key"] == "1003"
    assert bool(out.loc[0, "within_distance"]) is True
    assert bool(out.loc[0, "same_stop_code"]) is False
    assert out.loc[0, "after_candidate_count"] == 1


def test_candidate_matches_same_code_far_away_with_stop_id_key() -> None:
    before = stops_frame(
        [{"stop_id": "OLD", "stop_code": "77", "stop_name": "X", "stop_lat": 38.8,
          "stop_lon": -77.1}]
    )  # fmt: skip
    after = stops_frame(
        [{"stop_id": "NEW", "stop_code": "77", "stop_name": "X", "stop_lat": 39.8,
          "stop_lon": -77.1}]
    )  # fmt: skip
    out = candidate_matches(before, after, 500.0, 3, "stop_id")
    assert len(out) == 1
    assert bool(out.loc[0, "same_stop_code"]) is True
    assert bool(out.loc[0, "within_distance"]) is False


def test_candidate_matches_never_crosses_location_types() -> None:
    before = stops_frame(
        [{"stop_id": "STN", "stop_code": "1", "stop_name": "S", "stop_lat": 38.8,
          "stop_lon": -77.1, "location_type": "1"}]
    )  # fmt: skip
    after = stops_frame(
        [{"stop_id": "PLT", "stop_code": "2", "stop_name": "S", "stop_lat": 38.8,
          "stop_lon": -77.1, "location_type": "0"}]
    )  # fmt: skip
    assert candidate_matches(before, after, 500.0, 3).empty


def test_candidate_matches_rejects_bad_parameters() -> None:
    empty = stops_frame([])
    with pytest.raises(ValueError, match="Candidate"):
        candidate_matches(empty, empty, 500.0, 0)


# ---------------------------------------------------------------------------
# addition_output_tables
# ---------------------------------------------------------------------------


def additions() -> pd.DataFrame:
    return pd.DataFrame(
        {"stop_code": ["1004", "2005"], "present_in_before_inventory": [True, False]}
    )


def test_addition_output_tables_separate_mode() -> None:
    tables = addition_output_tables(additions(), "separate")
    assert set(tables) == {"new_stop_codes", "existing_stops_new_to_scope"}
    assert tables["new_stop_codes"]["stop_code"].tolist() == ["2005"]
    assert tables["existing_stops_new_to_scope"]["stop_code"].tolist() == ["1004"]


def test_addition_output_tables_names_new_ids_for_stop_id_key() -> None:
    tables = addition_output_tables(additions(), "new_only", "stop_id")
    assert list(tables) == ["new_stop_ids"]
    assert tables["new_stop_ids"]["addition_category"].tolist() == ["new_stop_id"]


def test_addition_output_tables_combined_mode() -> None:
    tables = addition_output_tables(additions(), "combined")
    assert tables["added_to_scope"]["addition_category"].tolist() == [
        "existing_stop_new_to_scope",
        "new_stop_code",
    ]


def test_addition_output_tables_reads_csv_text_flags() -> None:
    added = additions().assign(present_in_before_inventory=["True", "False"])
    tables = addition_output_tables(added, "existing_only")
    assert tables["existing_stops_new_to_scope"]["stop_code"].tolist() == ["1004"]


def test_addition_output_tables_rejects_invalid_flags_and_modes() -> None:
    with pytest.raises(ValueError, match="inventory-presence"):
        addition_output_tables(
            additions().assign(present_in_before_inventory=["", "x"]), "separate"
        )
    with pytest.raises(ValueError, match="additions output mode"):
        addition_output_tables(additions(), "everything")


# ---------------------------------------------------------------------------
# Route pairing decisions
# ---------------------------------------------------------------------------


def test_normalize_route_pairs_accepts_name_or_list() -> None:
    assert normalize_route_pairs({" New ": "101", "Merged": ["101", "202", "101"]}) == {
        "new": ["101"],
        "merged": ["101", "202"],
    }


@pytest.mark.parametrize(
    ("mapping", "error"),
    [
        ({"A": []}, ValueError),
        ({"A": [""]}, ValueError),
        ({"A": 5}, TypeError),
        ({"A": "1", "a": "2"}, ValueError),
        ({"": "1"}, ValueError),
    ],
)
def test_normalize_route_pairs_rejects_invalid(mapping: dict[str, Any], error: type) -> None:
    with pytest.raises(error):
        normalize_route_pairs(mapping)


def test_normalize_route_names_rejects_bare_string() -> None:
    with pytest.raises(ValueError, match="list of nonblank"):
        normalize_route_names("101", "BEFORE_ROUTE_NAMES")


def test_prompt_route_pair_retries_until_known_route(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["?", "", "nope", "101, 200"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    lookup = {"101": ("R101", "101"), "200": ("R200", "200")}
    assert prompt_route_pair("101X", lookup) == ["101", "200"]


def test_prompt_route_pair_none_and_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = {"101": ("R101", "101")}
    monkeypatch.setattr("builtins.input", lambda _prompt: "None")
    assert prompt_route_pair("101X", lookup) is None
    monkeypatch.setattr("builtins.input", lambda _prompt: "cancel")
    with pytest.raises(ValueError, match="cancelled"):
        prompt_route_pair("101X", lookup)


def test_prompt_route_pair_without_input_explains_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_input(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", no_input)
    with pytest.raises(ValueError, match="--route-pair"):
        prompt_route_pair("101X", {"101": ("R101", "101")})


# ---------------------------------------------------------------------------
# make_scope
# ---------------------------------------------------------------------------


def scope_feeds() -> tuple[Feed, Feed]:
    before = make_feed(
        ["B1", "B2", "B3"],
        {"R101": "101", "R200": "200"},
        [("B1", "R101"), ("B2", "R101"), ("B3", "R200")],
    )
    after = make_feed(
        ["A1", "A2", "A9"],
        {"X1": "101", "X9": "999"},
        [("A1", "X1"), ("A2", "X1"), ("A9", "X9")],
    )
    return before, after


def test_make_scope_same_name_match_and_unresolved_error() -> None:
    before, after = scope_feeds()
    with pytest.raises(ValueError, match="Unresolved concept routes"):
        make_scope(before, after, "after_routes", [], {}, [], "error")


def test_make_scope_warn_keeps_unresolved_route_provisional() -> None:
    before, after = scope_feeds()
    scope = make_scope(before, after, "after_routes", [], {}, [], "warn")
    assert scope.before_ids == {"B1", "B2"}
    assert scope.after_ids == {"A1", "A2", "A9"}
    assert scope.unresolved_routes == ["999"]
    assert scope.resolved_route_pairs == {"101": ["101"]}
    statuses = dict(zip(scope.route_table["after_route"], scope.route_table["status"]))
    assert statuses["101"] == "same_name"
    assert statuses["999"] == "unresolved_concept_route_review"
    assert any("provisional" in warning for warning in scope.warnings)


def test_make_scope_no_match_route_stays_in_after_scope() -> None:
    before, after = scope_feeds()
    scope = make_scope(before, after, "after_routes", [], {}, ["999"], "error")
    assert scope.confirmed_no_match_routes == ["999"]
    assert "A9" in scope.after_ids
    assert scope.unresolved_routes == []


def test_make_scope_merger_pair_unions_before_routes() -> None:
    before, after = scope_feeds()
    scope = make_scope(before, after, "after_routes", [], {"999": ["101", "200"]}, [], "error")
    assert scope.before_ids == {"B1", "B2", "B3"}
    assert scope.before_route_ids == {"R101", "R200"}
    assert scope.resolved_route_pairs["999"] == ["101", "200"]


def test_make_scope_explicit_before_route_and_outside_scope_status() -> None:
    before, after = scope_feeds()
    scope = make_scope(before, after, "after_routes", ["200"], {}, ["999"], "error")
    statuses = dict(zip(scope.route_table["before_route"], scope.route_table["status"]))
    assert statuses["200"] == "explicit_before_only"
    assert "B3" in scope.before_ids
    scope = make_scope(before, after, "after_routes", [], {}, ["999"], "error")
    statuses = dict(zip(scope.route_table["before_route"], scope.route_table["status"]))
    assert statuses["200"] == "outside_scope"


@pytest.mark.parametrize(
    ("names", "mapping", "no_match", "match"),
    [
        ([], {"999": "101"}, ["999"], "both paired and confirmed"),
        ([], {"nope": "101"}, [], "Unknown concept route"),
        ([], {"999": "nope"}, [], "Unknown before route"),
        (["nope"], {}, ["999"], "Unknown before route name"),
    ],
)
def test_make_scope_rejects_inconsistent_decisions(
    names: list[str], mapping: dict[str, str], no_match: list[str], match: str
) -> None:
    before, after = scope_feeds()
    with pytest.raises(ValueError, match=match):
        make_scope(before, after, "after_routes", names, mapping, no_match, "error")


def test_make_scope_all_stops_rejects_route_selections() -> None:
    before, after = scope_feeds()
    scope = make_scope(before, after, "all_stops", [], {}, [], "error")
    assert scope.before_ids == {"B1", "B2", "B3"}
    with pytest.raises(ValueError, match="require scope='after_routes'"):
        make_scope(before, after, "all_stops", ["101"], {}, [], "error")


# ---------------------------------------------------------------------------
# CLI pairing options and saved pairing files
# ---------------------------------------------------------------------------


def test_cli_pairing_options_route_pairs_form_merger() -> None:
    args = parse_args(
        ["--route-pair", "353New", "353", "--route-pair", "353New", "354",
         "--no-route-match", "400X"]
    )  # fmt: skip
    mapping, no_match, names = cli_pairing_options(args)
    assert mapping == {"353new": ["353", "354"]}
    assert no_match == ["400x"]
    assert names == []


def test_cli_pairing_options_conflict_raises() -> None:
    args = parse_args(["--route-pair", "A", "1", "--no-route-match", "A"])
    with pytest.raises(ValueError, match="Conflicting"):
        cli_pairing_options(args)


def test_cli_pairing_options_cli_overrides_saved_file(tmp_path: Path) -> None:
    saved = tmp_path / "route_pairings.json"
    saved.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "route_name_map": {"A": ["1"]},
                "after_routes_with_no_match": ["B"],
                "before_route_names": ["9"],
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(["--route-pairs-file", str(saved), "--route-pair", "B", "2"])
    mapping, no_match, names = cli_pairing_options(args)
    assert mapping == {"a": ["1"], "b": ["2"]}
    assert no_match == []
    assert names == ["9"]


def test_read_pairing_file_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"schema_version": 2, "route_name_map": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_pairing_file(path)
    path.write_text(json.dumps({"routes": {}}), encoding="utf-8")
    with pytest.raises(TypeError, match="route_name_map"):
        read_pairing_file(path)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def test_write_outputs_keeps_leading_equals_as_text(tmp_path: Path) -> None:
    settings = {
        "route_name_map": {},
        "after_routes_with_no_match": [],
        "before_route_names": [],
        "unresolved_routes": [],
    }
    write_outputs(tmp_path, {"before": pd.DataFrame({"stop_name": ["=1+1"]})}, {}, settings)
    cell = openpyxl.load_workbook(tmp_path / "stops_comparison.xlsx")["before"]["A2"]
    assert cell.value == "=1+1"
    assert cell.data_type == "s"
    assert (tmp_path / "before.csv").exists()
    assert json.loads((tmp_path / "route_pairings.json").read_text())["schema_version"] == 1


# ---------------------------------------------------------------------------
# run_compare / main (end to end)
# ---------------------------------------------------------------------------


def test_run_compare_after_routes_end_to_end(
    before_feed: Path, after_feed: Path, tmp_path: Path
) -> None:
    summary = run_compare(
        before_dir=before_feed,
        after_dir=after_feed,
        out_dir=tmp_path / "out",
        unmatched_policy="error",
    )
    assert summary["before_scope_stops"] == 3
    assert summary["after_scope_stops"] == 4
    assert summary["before_outside_scope"] == 1
    assert summary["shared_scope_keys"] == 2
    assert summary["modified_count"] == 1
    assert summary["unchanged_count"] == 1
    assert summary["removed_from_scope_count"] == 1
    assert summary["added_to_scope_count"] == 2
    assert summary["added_new_key_count"] == 1
    assert summary["added_existing_inventory_count"] == 1
    assert summary["reused_stops_modified_count"] == 0
    assert summary["shared_secondary_id_changes"] == 2
    assert summary["candidate_id_pairs"] >= 1
    assert summary["route_pairing_complete"] is True

    run_dir = Path(summary["output_directory"])
    for name in (
        "stops_comparison.xlsx",
        "summary.json",
        "run_settings.json",
        "route_pairings.json",
        "gtfs_stop_diff.log",
        "modified.csv",
        "removed_from_scope.csv",
        "new_stop_codes.csv",
        "existing_stops_new_to_scope.csv",
    ):
        assert (run_dir / name).exists(), name
    modified = pd.read_csv(run_dir / "modified.csv", dtype=str)
    assert modified["stop_code"].tolist() == ["1002"]
    removed = pd.read_csv(run_dir / "removed_from_scope.csv", dtype=str)
    assert removed["stop_code"].tolist() == ["1003"]
    # 2005 sits ~36 ft away: beyond the 25 ft threshold, so relocation, not renumber.
    assert removed.loc[0, "loses_all_service"] == "True"
    assert removed.loc[0, "possible_relocation"] == "True"
    assert removed.loc[0, "possible_renumber"] == "False"
    new_codes = pd.read_csv(run_dir / "new_stop_codes.csv", dtype=str)
    assert new_codes["stop_code"].tolist() == ["2005"]
    assert new_codes.loc[0, "has_id_match_candidate"] == "True"
    pairings = json.loads((run_dir / "route_pairings.json").read_text(encoding="utf-8"))
    assert pairings["route_name_map"] == {"101": ["101"]}


def test_run_compare_reuses_saved_route_pairings(
    before_feed: Path, after_feed: Path, tmp_path: Path
) -> None:
    first = run_compare(
        before_dir=before_feed,
        after_dir=after_feed,
        out_dir=tmp_path / "out",
        route_name_map={"101": ["101", "200"]},
        unmatched_policy="error",
    )
    saved = Path(first["output_directory"]) / "route_pairings.json"
    second = run_compare(
        before_dir=before_feed,
        after_dir=after_feed,
        out_dir=tmp_path / "out",
        route_pairs_file=saved,
        unmatched_policy="error",
    )
    assert first["before_scope_stops"] == second["before_scope_stops"] == 4
    assert second["added_existing_inventory_count"] == 0


def test_run_compare_all_stops_with_stop_id_key(
    before_feed: Path, after_feed: Path, tmp_path: Path
) -> None:
    summary = run_compare(
        before_dir=before_feed,
        after_dir=after_feed,
        out_dir=tmp_path / "out",
        scope_mode="all_stops",
        primary_stop_key="stop_id",
        additions_output_mode="combined",
    )
    assert summary["shared_scope_keys"] == 0
    assert summary["removed_from_scope_count"] == 4
    assert summary["added_new_key_count"] == 4
    assert summary["stop_id_code_change_count"] == 0
    assert (Path(summary["output_directory"]) / "added_to_scope.csv").exists()


def test_run_compare_blank_primary_key_stops_run(before_feed: Path, tmp_path: Path) -> None:
    rows = [dict(row, stop_code="") for row in AFTER_STOPS]
    after = write_feed(tmp_path / "blank", rows, AFTER_ROUTES, AFTER_TRIPS, AFTER_STOP_TIMES)
    with pytest.raises(ValueError, match="no fallback was applied"):
        run_compare(before_dir=before_feed, after_dir=after, out_dir=tmp_path / "out")


def test_run_compare_after_routes_requires_route_context(
    before_feed: Path, after_feed: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="requires route context"):
        run_compare(
            before_dir=before_feed,
            after_dir=after_feed,
            out_dir=tmp_path / "out",
            include_route_context=False,
        )


def test_run_compare_classifies_service_route_day_and_identity_changes(tmp_path: Path) -> None:
    before_stops = [
        {"stop_id": "B1", "stop_code": "1001", "stop_name": "Main St & 1st St",
         "stop_lat": "38.8000", "stop_lon": "-77.1000"},
        {"stop_id": "B2", "stop_code": "1002", "stop_name": "Main St & 2nd St",
         "stop_lat": "38.8100", "stop_lon": "-77.1000"},
        {"stop_id": "B3", "stop_code": "1003", "stop_name": "Oak Ave",
         "stop_lat": "38.8200", "stop_lon": "-77.1000"},
        {"stop_id": "B4", "stop_code": "1004", "stop_name": "Elm St",
         "stop_lat": "38.8300", "stop_lon": "-77.1000"},
        {"stop_id": "B5", "stop_code": "1005", "stop_name": "Pine St",
         "stop_lat": "38.8400", "stop_lon": "-77.1000"},
    ]  # fmt: skip
    before = write_feed(
        tmp_path / "before",
        before_stops,
        [{"route_id": "R101", "route_short_name": "101"},
         {"route_id": "R200", "route_short_name": "200"},
         {"route_id": "R300", "route_short_name": "300"}],
        [{"trip_id": "T1", "route_id": "R101", "service_id": "WK"},
         {"trip_id": "T2", "route_id": "R101", "service_id": "SA"},
         {"trip_id": "T3", "route_id": "R200", "service_id": "WK"},
         {"trip_id": "T4", "route_id": "R300", "service_id": "WK"}],
        [{"trip_id": trip, "stop_id": stop}
         for trip, stop in [("T1", "B1"), ("T1", "B2"), ("T1", "B3"), ("T2", "B1"),
                            ("T2", "B2"), ("T3", "B4"), ("T3", "B2"), ("T4", "B5")]],
    )  # fmt: skip
    write_calendar(before, {"WK": "weekday", "SA": "saturday"})
    # 1001 moves ~109 ft; 1002 renamed; 1003 dropped; 2006 a new code ~11 ft from 1003.
    # Route 300 is renamed "300 New" and gains a Saturday trip (calendar_dates only).
    after_stops = [
        {"stop_id": "A1", "stop_code": "1001", "stop_name": "Main St & 1st St",
         "stop_lat": "38.8003", "stop_lon": "-77.1000"},
        {"stop_id": "A2", "stop_code": "1002", "stop_name": "Main St & Second St",
         "stop_lat": "38.8100", "stop_lon": "-77.1000"},
        {"stop_id": "A4", "stop_code": "1004", "stop_name": "Elm St",
         "stop_lat": "38.8300", "stop_lon": "-77.1000"},
        {"stop_id": "A5", "stop_code": "1005", "stop_name": "Pine St",
         "stop_lat": "38.8400", "stop_lon": "-77.1000"},
        {"stop_id": "A6", "stop_code": "2006", "stop_name": "Oak Ave (new)",
         "stop_lat": "38.82003", "stop_lon": "-77.1000"},
    ]  # fmt: skip
    after = write_feed(
        tmp_path / "after",
        after_stops,
        [{"route_id": "X1", "route_short_name": "101"},
         {"route_id": "X3", "route_short_name": "300 New"}],
        [{"trip_id": "AT1", "route_id": "X1", "service_id": "WK"},
         {"trip_id": "AT3", "route_id": "X3", "service_id": "WK"},
         {"trip_id": "AT4", "route_id": "X3", "service_id": "SA"}],
        [{"trip_id": trip, "stop_id": stop}
         for trip, stop in [("AT1", "A1"), ("AT1", "A2"), ("AT1", "A4"), ("AT1", "A6"),
                            ("AT3", "A5"), ("AT4", "A5")]],
    )  # fmt: skip
    write_calendar(after, {"WK": "weekday"})
    pd.DataFrame([{"service_id": "SA", "date": "20261114", "exception_type": "1"}]).to_csv(
        after / "calendar_dates.txt", index=False
    )

    summary = run_compare(
        before_dir=before,
        after_dir=after,
        out_dir=tmp_path / "out",
        route_name_map={"300 New": "300"},
        unmatched_policy="error",
    )
    run_dir = Path(summary["output_directory"])
    changes = pd.read_csv(run_dir / "stop_changes.csv", dtype=str, keep_default_na=False)
    rows = changes.set_index("comparison_key")

    assert rows.loc["1001", "change_summary"] == (
        "loses service days; a continuing route loses days; relocated"
    )
    assert rows.loc["1001", "days_lost"] == "saturday"
    assert rows.loc["1001", "route_days_lost"] == "101: saturday"
    # Route 200 (outside the concept) keeps serving 1002 on weekdays only.
    assert rows.loc["1002", "change_summary"] == (
        "loses service days; a continuing route loses days; renamed"
    )
    assert rows.loc["1002", "other_current_routes"] == "200"
    assert rows.loc["1003", "comparison_status"] == "removed_from_scope"
    assert rows.loc["1003", "change_summary"] == "loses all service; possible renumber (review)"
    assert rows.loc["1003", "routes_lost"] == "101"
    assert rows.loc["1004", "change_summary"] == "gains routes"
    assert rows.loc["1004", "routes_gained"] == "101"
    # The renamed route is paired, so it is neither lost nor gained.
    assert rows.loc["1005", "change_summary"] == (
        "gains service days; a continuing route gains days"
    )
    assert rows.loc["1005", "routes_lost"] == rows.loc["1005", "routes_gained"] == ""
    assert rows.loc["1005", "route_days_gained"] == "300 New: saturday"
    assert rows.loc["2006", "change_summary"] == "new stop; possible renumber (review)"

    assert summary["day_context_available"] is True
    assert summary["loses_all_service_count"] == 1
    assert summary["loses_days_count"] == 2
    assert summary["gains_days_count"] == 1
    assert summary["relocated_count"] == 1
    assert summary["renamed_count"] == 1
    assert summary["possible_renumber_count"] == 2
    assert summary["route_context_changed_count"] == 0
    removed = pd.read_csv(run_dir / "removed_from_scope.csv", dtype=str)
    assert removed.loc[0, "loses_all_service"] == "True"
    assert removed.loc[0, "possible_renumber"] == "True"


def test_run_compare_all_stops_flags_unserved_stop_and_renumber(tmp_path: Path) -> None:
    stops = [
        {"stop_id": "S1", "stop_code": "100", "stop_name": "First",
         "stop_lat": "38.80", "stop_lon": "-77.10"},
        {"stop_id": "S2", "stop_code": "200", "stop_name": "Second",
         "stop_lat": "38.81", "stop_lon": "-77.10"},
    ]  # fmt: skip
    routes = [{"route_id": "R1", "route_short_name": "1"}]
    trips = [{"trip_id": "T1", "route_id": "R1"}]
    before = write_feed(
        tmp_path / "before", stops, routes, trips,
        [{"trip_id": "T1", "stop_id": "S1"}, {"trip_id": "T1", "stop_id": "S2"}],
    )  # fmt: skip
    # S1 gets a new public code; S2 stays in stops.txt but no trip serves it.
    after = write_feed(
        tmp_path / "after", [dict(stops[0], stop_code="101"), stops[1]], routes, trips,
        [{"trip_id": "T1", "stop_id": "S1"}],
    )  # fmt: skip
    summary = run_compare(
        before_dir=before,
        after_dir=after,
        out_dir=tmp_path / "out",
        scope_mode="all_stops",
        primary_stop_key="stop_id",
    )
    changes = pd.read_csv(
        Path(summary["output_directory"]) / "stop_changes.csv", dtype=str, keep_default_na=False
    ).set_index("comparison_key")
    assert changes.loc["S1", "change_summary"] == "renumbered"
    assert changes.loc["S2", "comparison_status"] == "shared"
    assert changes.loc["S2", "change_summary"] == "loses all service"
    assert summary["removed_from_scope_count"] == 0
    assert summary["day_context_available"] is False
    assert any("day impact was not assessed" in warning for warning in summary["warnings"])


def test_main_returns_zero_on_success(before_feed: Path, after_feed: Path, tmp_path: Path) -> None:
    argv = ["--before", str(before_feed), "--after", str(after_feed), "--out", str(tmp_path)]
    assert main([*argv, "--unmatched-routes", "error"]) == 0


def test_main_returns_one_for_missing_source(tmp_path: Path) -> None:
    argv = ["--before", str(tmp_path / "missing"), "--after", str(tmp_path / "missing")]
    assert main([*argv, "--out", str(tmp_path / "out")]) == 1
