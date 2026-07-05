from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.facilities_tools.stop_improvement_coverage_summary as target

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _routes_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route_id": ["R1", "R2", "R3"],
            "route_short_name": ["101", "202", "303"],
        }
    )


def _trips_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1", "R2", "R3"],
        }
    )


def _stop_times_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2", "T3"],
            "stop_id": ["S1", "S2", "S2", "S3"],
        }
    )


def _stops_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3"],
            "stop_code": ["C1", "C2", "C1"],
            "stop_name": ["Main & 1st", "Main & 2nd", "Main & 1st (far side)"],
        }
    )


# ---------------------------------------------------------------------------
# _standardise_yn
# ---------------------------------------------------------------------------


def test_standardise_yn_maps_truthy_tokens_to_y() -> None:
    s = pd.Series(["yes", "TRUE", "1", " y "])
    assert list(target._standardise_yn(s)) == ["Y", "Y", "Y", "Y"]


def test_standardise_yn_maps_falsy_tokens_to_n() -> None:
    s = pd.Series(["no", "FALSE", "0", None])
    assert list(target._standardise_yn(s)) == ["N", "N", "N", "N"]


# ---------------------------------------------------------------------------
# resolve_route_ids_by_short_name
# ---------------------------------------------------------------------------


def test_resolve_route_ids_matches_short_names() -> None:
    ids = target.resolve_route_ids_by_short_name(_routes_df(), {"101", "303"})
    assert ids == {"R1", "R3"}


def test_resolve_route_ids_empty_input_returns_empty() -> None:
    assert target.resolve_route_ids_by_short_name(_routes_df(), set()) == set()


def test_resolve_route_ids_missing_token_logs_warning(caplog) -> None:
    with caplog.at_level("WARNING"):
        ids = target.resolve_route_ids_by_short_name(_routes_df(), {"101", "888"})
    assert ids == {"R1"}
    assert "888" in caplog.text


def test_resolve_route_ids_no_short_name_column_returns_empty() -> None:
    routes = pd.DataFrame({"route_id": ["R1"]})
    assert target.resolve_route_ids_by_short_name(routes, {"101"}) == set()


# ---------------------------------------------------------------------------
# build_stop_to_routes
# ---------------------------------------------------------------------------


def test_build_stop_to_routes_maps_each_stop_to_serving_routes() -> None:
    out = target.build_stop_to_routes(_stop_times_df(), _trips_df(), _routes_df())
    lookup = dict(zip(out["stop_id"], out["route_ids"]))
    assert lookup["S1"] == "R1"
    assert lookup["S2"] == "R1,R2"
    assert lookup["S3"] == "R3"


def test_build_stop_to_routes_includes_short_names() -> None:
    out = target.build_stop_to_routes(_stop_times_df(), _trips_df(), _routes_df())
    lookup = dict(zip(out["stop_id"], out["route_short_names"]))
    assert lookup["S2"] == "101,202"


# ---------------------------------------------------------------------------
# collapse_to_logical_stops
# ---------------------------------------------------------------------------


def test_collapse_merges_platforms_sharing_stop_code() -> None:
    stop_to_routes = target.build_stop_to_routes(_stop_times_df(), _trips_df(), _routes_df())
    out = target.collapse_to_logical_stops(_stops_df(), stop_to_routes, "stop_code")
    assert len(out) == 2  # C1 (S1+S3) and C2 (S2)
    c1 = out[out["stop_code"] == "C1"].iloc[0]
    assert c1["stop_ids"] == "S1,S3"
    assert c1["route_ids"] == "R1,R3"  # union across platforms


def test_collapse_blank_key_falls_back_to_stop_id() -> None:
    stops = _stops_df()
    stops.loc[stops["stop_id"] == "S3", "stop_code"] = ""
    stop_to_routes = target.build_stop_to_routes(_stop_times_df(), _trips_df(), _routes_df())
    out = target.collapse_to_logical_stops(stops, stop_to_routes, "stop_code")
    assert "S3" in set(out["stop_code"])


def test_collapse_missing_key_field_raises() -> None:
    stops = _stops_df().drop(columns=["stop_code"])
    with pytest.raises(ValueError, match="stop_code"):
        target.collapse_to_logical_stops(stops, pd.DataFrame(), "stop_code")


# ---------------------------------------------------------------------------
# load_improvements
# ---------------------------------------------------------------------------


def _write_improvements_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_load_improvements_normalises_aliases_and_values(tmp_path: Path) -> None:
    csv_path = _write_improvements_csv(
        tmp_path / "improvements.csv",
        "stop_code,bus_shelter,bench,trash_can,pad",
        ["C1,yes,n, Y ,1", "C2,N,Y,N,0"],
    )
    df, cols = target.load_improvements(
        csv_path,
        "stop_code",
        target.IMPROVEMENT_COLUMNS,
        target.IMPROVEMENT_ALIASES,
    )
    assert cols == ["SHELTER", "BENCH", "TRASHCAN", "PAD"]
    c1 = df[df["stop_code"] == "C1"].iloc[0]
    assert c1["SHELTER"] == "Y"
    assert c1["BENCH"] == "N"
    assert c1["TRASHCAN"] == "Y"
    assert c1["PAD"] == "Y"


def test_load_improvements_missing_column_defaults_to_n(tmp_path: Path) -> None:
    csv_path = _write_improvements_csv(
        tmp_path / "improvements.csv",
        "stop_code,SHELTER",
        ["C1,Y"],
    )
    df, _ = target.load_improvements(
        csv_path,
        "stop_code",
        target.IMPROVEMENT_COLUMNS,
        target.IMPROVEMENT_ALIASES,
    )
    assert df["BENCH"].iloc[0] == "N"


def test_load_improvements_missing_join_field_raises(tmp_path: Path) -> None:
    csv_path = _write_improvements_csv(
        tmp_path / "improvements.csv",
        "wrong_key,SHELTER",
        ["C1,Y"],
    )
    with pytest.raises(ValueError, match="stop_code"):
        target.load_improvements(
            csv_path,
            "stop_code",
            target.IMPROVEMENT_COLUMNS,
            target.IMPROVEMENT_ALIASES,
        )


def test_load_improvements_drops_duplicate_join_keys(tmp_path: Path) -> None:
    csv_path = _write_improvements_csv(
        tmp_path / "improvements.csv",
        "stop_code,SHELTER,BENCH,TRASHCAN,PAD",
        ["C1,Y,N,N,N", "C1,N,Y,N,N"],
    )
    df, _ = target.load_improvements(
        csv_path,
        "stop_code",
        target.IMPROVEMENT_COLUMNS,
        target.IMPROVEMENT_ALIASES,
    )
    assert len(df) == 1


# ---------------------------------------------------------------------------
# attach_improvements
# ---------------------------------------------------------------------------


def test_attach_improvements_joins_on_logical_key() -> None:
    logical = pd.DataFrame(
        {
            "stop_code": ["C1", "C2", "C9"],
            "route_ids": ["R1", "R2", "R3"],
        }
    )
    improvements = pd.DataFrame(
        {
            "stop_code": ["C1", "C2"],
            "SHELTER": ["Y", "N"],
        }
    )
    out = target.attach_improvements(logical, improvements, "stop_code", "stop_code", ["SHELTER"])
    lookup = dict(zip(out["stop_code"], out["SHELTER"]))
    assert lookup["C1"] == "Y"
    assert lookup["C2"] == "N"
    assert lookup["C9"] == "N"  # unmatched → normalised to 'N'


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------


def _logical_with_improvements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_code": ["C1", "C2", "C3", "C4"],
            "route_ids": ["R1", "R1,R2", "R2", "R3"],
            "SHELTER": ["Y", "Y", "N", "N"],
            "BENCH": ["N", "N", "N", "Y"],
            "TRASHCAN": ["N", "N", "N", "N"],
            "PAD": ["Y", "Y", "Y", "Y"],
        }
    )


def test_compute_summary_system_counts_and_percentages() -> None:
    summary = target.compute_summary(
        _logical_with_improvements(),
        ["SHELTER", "BENCH", "TRASHCAN", "PAD"],
        target.IMPROVEMENT_COLUMNS,
        whitelist_route_ids=set(),
        whitelist_short_names=set(),
    )
    assert summary["system_total_stops"] == 4
    assert summary["per_improvement"]["Shelter"]["system_count"] == 2
    assert summary["per_improvement"]["Shelter"]["system_pct"] == 50.0
    assert summary["per_improvement"]["ADA Pad"]["system_pct"] == 100.0


def test_compute_summary_whitelist_coverage() -> None:
    summary = target.compute_summary(
        _logical_with_improvements(),
        ["SHELTER", "BENCH", "TRASHCAN", "PAD"],
        target.IMPROVEMENT_COLUMNS,
        whitelist_route_ids={"R1"},
        whitelist_short_names={"101"},
    )
    assert summary["whitelist_total_stops"] == 2  # C1, C2
    assert summary["whitelist_pct_of_system"] == 50.0
    assert summary["per_improvement"]["Shelter"]["whitelist_count"] == 2
    assert summary["per_improvement"]["Shelter"]["whitelist_pct"] == 100.0


def test_compute_summary_empty_universe_is_zero_safe() -> None:
    empty = _logical_with_improvements().iloc[0:0]
    summary = target.compute_summary(
        empty,
        ["SHELTER"],
        {"Shelter": "SHELTER"},
        whitelist_route_ids=set(),
        whitelist_short_names=set(),
    )
    assert summary["system_total_stops"] == 0
    assert summary["whitelist_pct_of_system"] == 0.0
    assert summary["per_improvement"]["Shelter"]["system_pct"] == 0.0


def test_compute_summary_no_improvements_supplied_flag() -> None:
    logical = pd.DataFrame({"stop_code": ["C1"], "route_ids": ["R1"]})
    summary = target.compute_summary(
        logical,
        [],
        target.IMPROVEMENT_COLUMNS,
        whitelist_route_ids=set(),
        whitelist_short_names=set(),
    )
    assert summary["improvements_supplied"] is False
    assert summary["per_improvement"] == {}


# ---------------------------------------------------------------------------
# write_summary_txt
# ---------------------------------------------------------------------------


def test_write_summary_txt_reports_counts(tmp_path: Path) -> None:
    summary = target.compute_summary(
        _logical_with_improvements(),
        ["SHELTER", "BENCH", "TRASHCAN", "PAD"],
        target.IMPROVEMENT_COLUMNS,
        whitelist_route_ids={"R1"},
        whitelist_short_names={"101"},
    )
    out = tmp_path / "summary.txt"
    target.write_summary_txt(summary, {"9999A"}, out)
    content = out.read_text(encoding="utf-8")
    assert "Total logical stops (post-blacklist): 4" in content
    assert "Blacklist routes excluded: 9999A" in content
    assert "Whitelist routes: 101" in content
    assert "Shelter" in content


def test_write_summary_txt_without_improvements(tmp_path: Path) -> None:
    logical = pd.DataFrame({"stop_code": ["C1"], "route_ids": ["R1"]})
    summary = target.compute_summary(
        logical,
        [],
        target.IMPROVEMENT_COLUMNS,
        whitelist_route_ids=set(),
        whitelist_short_names=set(),
    )
    out = tmp_path / "summary.txt"
    target.write_summary_txt(summary, set(), out)
    content = out.read_text(encoding="utf-8")
    assert "no improvements CSV supplied" in content
