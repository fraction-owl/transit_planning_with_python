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
    s = pd.Series(["no", "FALSE", "0", None, "", "  "])
    assert list(target._standardise_yn(s)) == ["N", "N", "N", "N", "N", "N"]


# ---------------------------------------------------------------------------
# load_gtfs_data
# ---------------------------------------------------------------------------


def test_load_gtfs_data_keeps_literal_na_values(tmp_path: Path) -> None:
    (tmp_path / "routes.txt").write_text("route_id,route_short_name\nR1,NA\nR2,\n")
    routes = target.load_gtfs_data(str(tmp_path), files=("routes.txt",))["routes"]
    assert list(routes["route_short_name"]) == ["NA", ""]
    assert target.resolve_route_ids_by_short_name(routes, {"NA"}) == {"R1"}


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
    assert len(out) == 3  # C1 (S1), C2 (S2), and S3 on its own
    s3 = out[out["stop_ids"] == "S3"].iloc[0]
    assert s3["stop_code"] == ""  # real (blank) code is kept for the improvements join
    assert s3["route_ids"] == "R3"


def test_collapse_fallback_stop_id_does_not_merge_with_matching_stop_code() -> None:
    # S3's blank code must not fall back into the same key as stop_code "S1".
    stops = pd.DataFrame(
        {
            "stop_id": ["S1", "S3"],
            "stop_code": ["S3", ""],
            "stop_name": ["First", "Unrelated"],
        }
    )
    stop_to_routes = target.build_stop_to_routes(_stop_times_df(), _trips_df(), _routes_df())
    out = target.collapse_to_logical_stops(stops, stop_to_routes, "stop_code")
    assert sorted(out["stop_ids"]) == ["S1", "S3"]
    assert dict(zip(out["stop_ids"], out["stop_code"])) == {"S1": "S3", "S3": ""}
    assert dict(zip(out["stop_ids"], out["route_ids"])) == {"S1": "R1", "S3": "R3"}


def test_collapse_by_stop_id_keeps_each_physical_stop() -> None:
    stop_to_routes = target.build_stop_to_routes(_stop_times_df(), _trips_df(), _routes_df())
    out = target.collapse_to_logical_stops(_stops_df(), stop_to_routes, "stop_id")
    assert list(out.columns) == [
        "stop_id",
        "stop_ids",
        "stop_name",
        "route_ids",
        "route_short_names",
    ]
    assert list(out["stop_id"]) == ["S1", "S2", "S3"]
    assert dict(zip(out["stop_id"], out["route_ids"])) == {"S1": "R1", "S2": "R1,R2", "S3": "R3"}


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


def test_load_improvements_collapses_duplicate_keys_any_y_wins(tmp_path: Path) -> None:
    csv_path = _write_improvements_csv(
        tmp_path / "improvements.csv",
        "stop_code,SHELTER,BENCH,TRASHCAN,PAD",
        ["C1,N,Y,N,", "C1,Y,N,N,N"],
    )
    df, _ = target.load_improvements(
        csv_path,
        "stop_code",
        target.IMPROVEMENT_COLUMNS,
        target.IMPROVEMENT_ALIASES,
    )
    assert len(df) == 1
    c1 = df.iloc[0]
    assert (c1["SHELTER"], c1["BENCH"], c1["TRASHCAN"], c1["PAD"]) == ("Y", "Y", "N", "N")


def test_load_improvements_keeps_literal_na_key_and_drops_blank_keys(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = _write_improvements_csv(
        tmp_path / "improvements.csv",
        "stop_code,SHELTER",
        ["NA,Y", ",Y", "nan,N"],
    )
    with caplog.at_level("WARNING"):
        df, _ = target.load_improvements(csv_path, "stop_code", {"Shelter": "SHELTER"}, {})
    assert dict(zip(df["stop_code"], df["SHELTER"])) == {"NA": "Y", "nan": "N"}
    assert "1 improvements rows have a blank stop_code" in caplog.text


# ---------------------------------------------------------------------------
# attach_improvements
# ---------------------------------------------------------------------------


def test_attach_improvements_joins_on_logical_key(caplog: pytest.LogCaptureFixture) -> None:
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
    with caplog.at_level("WARNING"):
        out = target.attach_improvements(
            logical, improvements, "stop_code", "stop_code", ["SHELTER"]
        )
    lookup = dict(zip(out["stop_code"], out["SHELTER"]))
    assert lookup["C1"] == "Y"
    assert lookup["C2"] == "N"
    assert lookup["C9"] == "N"  # unmatched → normalised to 'N'
    assert "_merge" not in out.columns
    assert "1 of 3 logical stops did not match" in caplog.text


def test_attach_improvements_all_matched_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logical = pd.DataFrame({"stop_code": ["C1"], "route_ids": ["R1"]})
    improvements = pd.DataFrame({"code": ["C1"], "SHELTER": ["Y"]})
    with caplog.at_level("WARNING"):
        out = target.attach_improvements(logical, improvements, "stop_code", "code", ["SHELTER"])
    assert list(out.columns) == ["stop_code", "route_ids", "SHELTER"]
    assert "did not match" not in caplog.text


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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_stop_code", [True, False])
def test_main_runs_with_only_required_gtfs_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_stop_code: bool
) -> None:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    for name, df in {
        "stops": _stops_df(),
        "routes": _routes_df(),
        "trips": _trips_df(),
        "stop_times": _stop_times_df(),
    }.items():
        df.to_csv(gtfs_dir / f"{name}.txt", index=False)
    join_field = "stop_code" if use_stop_code else "stop_id"
    first_key = "C1" if use_stop_code else "S1"
    csv_path = _write_improvements_csv(
        tmp_path / "improvements.csv",
        f"{join_field},SHELTER",
        [f"{first_key},N", f"{first_key},Y"],
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(target, "GTFS_DIR", gtfs_dir)
    monkeypatch.setattr(target, "USE_STOP_CODE", use_stop_code)
    monkeypatch.setattr(target, "IMPROVEMENTS_CSV", csv_path)
    monkeypatch.setattr(target, "IMPROVEMENTS_JOIN_FIELD", join_field)
    monkeypatch.setattr(target, "ROUTE_WHITELIST", {"101"})
    monkeypatch.setattr(target, "ROUTE_BLACKLIST", set())
    monkeypatch.setattr(target, "OUTPUT_DIR", out_dir)

    assert target.main() == 0

    detail = pd.read_csv(out_dir / target.DETAIL_CSV_NAME, dtype=str)
    shelter = dict(zip(detail[join_field], detail["SHELTER"]))
    assert shelter[first_key] == "Y"
    assert len(detail) == (2 if use_stop_code else 3)
    assert (out_dir / target.SUMMARY_TXT_NAME).exists()
