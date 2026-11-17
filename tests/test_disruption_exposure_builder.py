"""Tests for scripts/modeling/disruption_exposure_builder.py."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from scripts.modeling import disruption_exposure_builder as deb
from scripts.modeling import prep_features_public as pfp

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADER = "event_id,start_date,end_date,shape,scope,target,category,severity,source,notes"

# ---------------------------------------------------------------------------
# Fixtures: a three-route feed. Stop S2 is shared by routes 10 and 20.
#   route 10: S1, S2         (2 stops)
#   route 20: S2, S3, S4, S5 (4 stops)
#   route 30: S6             (1 stop, route_short_name blank -> label from route_id)
# ---------------------------------------------------------------------------


@pytest.fixture
def gtfs() -> dict[str, pd.DataFrame]:
    routes = pd.DataFrame(
        {"route_id": ["R10", "R20", "R30"], "route_short_name": ["10", "20 ", ""]}
    )
    trips = pd.DataFrame({"trip_id": ["t1", "t2", "t3"], "route_id": ["R10", "R20", "R30"]})
    stop_times = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t2", "t2", "t2", "t2", "t3"],
            "stop_id": ["S1", "S2", "S2", "S3", "S4", "S5", "S6"],
        }
    )
    return {"routes": routes, "trips": trips, "stop_times": stop_times}


def _events(*rows: str) -> pd.DataFrame:
    text = "\n".join([HEADER, *rows]) + "\n"
    return pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)


def _build(
    gtfs: dict[str, pd.DataFrame],
    *rows: str,
    start: str = "2024-01",
    end: str = "2024-06",
    districts: pd.DataFrame | None = None,
    **kwargs: object,
) -> deb.ExposureTables:
    events = deb.validate_events(_events(*rows))
    labels = deb.route_label_map(gtfs["routes"])
    stop_routes = deb.stop_route_crosswalk(gtfs["stop_times"], gtfs["trips"], labels)
    return deb.build_exposure_tables(
        events, labels, stop_routes, districts, start_month=start, end_month=end, **kwargs
    )


def _cell(df: pd.DataFrame, route: str, period: str, col: str) -> float:
    return float(df.set_index(["route_id", "period"]).loc[(route, period), col])


# ---------------------------------------------------------------------------
# Exposure arithmetic
# ---------------------------------------------------------------------------


def test_pulse_exposure_is_affected_days_over_days_in_month(gtfs) -> None:
    out = _build(gtfs, "DET,2024-01-16,2024-02-10,pulse,route,10,construction,,,")
    by_route = out.by_route
    assert _cell(by_route, "10", "2024-01", "ev_det") == pytest.approx(16 / 31)
    assert _cell(by_route, "10", "2024-02", "ev_det") == pytest.approx(10 / 29)  # leap year
    assert _cell(by_route, "10", "2024-03", "ev_det") == 0.0
    assert _cell(by_route, "20", "2024-01", "ev_det") == 0.0


def test_grid_is_full_and_zero_filled(gtfs) -> None:
    out = _build(gtfs, "DET,2024-02-01,2024-02-29,pulse,route,10,construction,,,")
    assert len(out.by_route) == 3 * 6  # three routes x six months
    assert set(out.by_route["route_id"]) == {"10", "20", "R30"}
    assert not out.by_route.isna().any().any()


@pytest.mark.parametrize(
    ("transition", "feb", "mar"),
    [("fractional", 15 / 29, 1.0), ("start_month", 1.0, 1.0), ("next_month", 0.0, 1.0)],
)
def test_step_transition_conventions(gtfs, transition, feb, mar) -> None:
    out = _build(
        gtfs,
        "NEW,2024-02-15,,step,route,20,service,,,",
        step_transition=transition,
    )
    assert _cell(out.by_route, "20", "2024-01", "ev_new") == 0.0
    assert _cell(out.by_route, "20", "2024-02", "ev_new") == pytest.approx(feb)
    assert _cell(out.by_route, "20", "2024-03", "ev_new") == mar
    assert _cell(out.by_route, "20", "2024-06", "ev_new") == 1.0


def test_ongoing_pulse_runs_to_window_end(gtfs) -> None:
    out = _build(gtfs, "OPEN,2024-05-01,,pulse,route,10,construction,,,")
    assert _cell(out.by_route, "10", "2024-06", "ev_open") == 1.0


def test_system_events_go_to_period_table_only(gtfs) -> None:
    out = _build(gtfs, "SL2,2024-03-16,,step,system,ALL,wmata,,,")
    assert list(out.system.columns) == ["period", "sys_ev_sl2"]
    sys_by_period = out.system.set_index("period")["sys_ev_sl2"]
    assert sys_by_period["2024-03"] == pytest.approx(16 / 31)
    assert sys_by_period["2024-04"] == 1.0
    assert list(out.by_route.columns) == ["route_id", "period"]  # nothing local
    assert set(out.long["route_label"]) == {"ALL"}


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


def test_stop_scope_rolls_up_to_every_serving_route(gtfs) -> None:
    out = _build(gtfs, "CLOSE,2024-03-01,2024-03-31,pulse,stop,S2,construction,,,")
    assert _cell(out.by_route, "10", "2024-03", "ev_close") == 1.0
    assert _cell(out.by_route, "20", "2024-03", "ev_close") == 1.0
    assert _cell(out.by_route, "R30", "2024-03", "ev_close") == 0.0
    shares = out.long.set_index("route_label")["stop_share"]
    assert shares["10"] == pytest.approx(1 / 2)
    assert shares["20"] == pytest.approx(1 / 4)
    assert any("S2 -> routes 10 (1/2 stops), 20 (1/4 stops)" in r for r in out.resolution)


def test_weight_by_stop_share_scales_local_exposure(gtfs) -> None:
    out = _build(
        gtfs,
        "CLOSE,2024-03-01,2024-03-31,pulse,stop,S2,construction,,,",
        weight_by_stop_share=True,
    )
    assert _cell(out.by_route, "10", "2024-03", "ev_close") == pytest.approx(0.5)
    assert _cell(out.by_route, "20", "2024-03", "ev_close") == pytest.approx(0.25)


def test_district_scope_uses_crosswalk(gtfs) -> None:
    districts = pd.DataFrame(
        {"stop_id": ["S3", "S4", "S6"], "district": ["Lee", "Lee", "Mason"]}
    ).assign(district_key=lambda d: d["district"].str.upper())
    out = _build(
        gtfs,
        "FLOOD,2024-04-01,2024-04-15,pulse,district,lee,flood,3,,",
        districts=districts,
    )
    assert _cell(out.by_route, "20", "2024-04", "ev_flood") == pytest.approx(0.5)
    assert _cell(out.by_route, "10", "2024-04", "ev_flood") == 0.0
    assert out.long["severity"].tolist() == [3.0]


def test_district_scope_without_crosswalk_raises(gtfs) -> None:
    with pytest.raises(ValueError, match="STOP_DISTRICTS_CSV"):
        _build(gtfs, "FLOOD,2024-04-01,2024-04-15,pulse,district,Lee,flood,,,")


def test_unknown_route_is_kept_in_grid_with_warning(gtfs, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        out = _build(gtfs, "GONE,2024-01-01,2024-01-31,pulse,route,99,service,,,")
    assert "99" in set(out.by_route["route_id"])
    assert _cell(out.by_route, "99", "2024-01", "ev_gone") == 1.0
    assert any("not in the GTFS feed" in w for w in out.warnings)


def test_unknown_stop_resolves_to_nothing(gtfs) -> None:
    out = _build(
        gtfs,
        "KEEP,2024-01-01,2024-01-31,pulse,route,10,service,,,",
        "MISS,2024-01-01,2024-01-31,pulse,stop,NOPE,service,,,",
    )
    assert "ev_miss" not in out.by_route.columns
    assert any("(no match)" in r for r in out.resolution)
    assert any("'MISS' has no exposure" in w for w in out.warnings)


def test_extra_routes_join_the_grid(gtfs) -> None:
    out = _build(
        gtfs, "DET,2024-01-01,2024-01-31,pulse,route,10,construction,,,", extra_routes=["7a"]
    )
    assert "7A" in set(out.by_route["route_id"])


# ---------------------------------------------------------------------------
# Pivoting and dropped columns
# ---------------------------------------------------------------------------


def test_category_pivot_counts_overlapping_days_once(gtfs) -> None:
    out = _build(
        gtfs,
        "A,2024-01-01,2024-01-20,pulse,route,10,construction,,,",
        "B,2024-01-11,2024-01-31,pulse,route,10,construction,,,",
        pivot_by="category",
    )
    assert list(out.by_route.columns) == ["route_id", "period", "evcat_construction"]
    assert _cell(out.by_route, "10", "2024-01", "evcat_construction") == 1.0
    assert set(out.long["event_id"]) == {"A", "B"}  # the audit table stays per event


def test_constant_columns_are_dropped(gtfs, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        out = _build(
            gtfs,
            "OLD,2020-01-01,,step,system,ALL,wmata,,,",
            "NEW,2024-03-01,,step,system,ALL,wmata,,,",
        )
    assert out.dropped_columns == ["sys_ev_old"]
    assert list(out.system.columns) == ["period", "sys_ev_new"]
    assert "collinear with the intercept" in caplog.text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("X,2024-01-01,2024-02-01,step,route,10,c,,,", "end_date must be blank"),
        ("X,2024-02-01,2024-01-01,pulse,route,10,c,,,", "end_date is before start_date"),
        ("X,2024-01-01,,pulse,corridor,10,c,,,", "scope must be"),
        ("X,2024-01-01,,ramp,route,10,c,,,", "shape must be"),
        ("X,2024-01-01,,pulse,system,10,c,,,", "system scope needs target ALL"),
        ("X,2024-01-01,,pulse,route,ALL,c,,,", "only valid with system scope"),
        ("X,01/05/2024,,pulse,route,10,c,,,", "start_date is not a YYYY-MM-DD"),
        ("X,2024-01-01,,pulse,route,10,c,high,,", "severity must be a number"),
        ("X,2024-01-01,,pulse,route,10,,,,", "blank category"),
    ],
)
def test_validation_rejects_bad_rows(row: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        deb.validate_events(_events(row))


def test_validation_rejects_event_level_conflicts() -> None:
    with pytest.raises(ValueError) as exc:
        deb.validate_events(
            _events(
                "E,2024-01-01,,pulse,system,ALL,wmata,,,",
                "E,2024-01-01,,pulse,route,10,wmata,,,",
                "F,2024-01-01,,pulse,route,10,a,,,",
                "F,2024-01-01,,step,route,20,b,,,",
                "F,2024-01-01,,pulse,route,10,a,,,",
                "G-1,2024-01-01,,pulse,route,10,a,,,",
                "G_1,2024-01-01,,pulse,route,10,a,,,",
            )
        )
    text = str(exc.value)
    assert "mixes system scope" in text
    assert "disagree on shape" in text
    assert "disagree on category" in text
    assert "duplicate (event_id, scope, target)" in text
    assert "column suffix 'g_1'" in text


def test_validation_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        deb.validate_events(pd.DataFrame({"event_id": ["X"]}))


def test_sample_events_file_validates() -> None:
    events = deb.validate_events(deb.load_events(REPO_ROOT / "sample_data/disruptions_example.csv"))
    assert {"SL2-OPEN", "7000-STANDDOWN"} <= set(events["event_id"])


# ---------------------------------------------------------------------------
# Orchestrator contract: wide tables bundle, long table does not
# ---------------------------------------------------------------------------


def test_outputs_describe_as_intended_for_prep_features_public(gtfs, tmp_path) -> None:
    out = _build(
        gtfs,
        "SL2,2024-03-16,,step,system,ALL,wmata,,,",
        "DET,2024-01-16,2024-02-10,pulse,route,10,construction,,,",
    )
    paths = {}
    for name, frame in (("route", out.by_route), ("system", out.system), ("long", out.long)):
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    run = pfp.ScriptRun("disruption_exposure_builder.py", [], 0, 0.0, "", False, [])

    route_ft = pfp.describe_output(paths["route"], run, {}, {})
    system_ft = pfp.describe_output(paths["system"], run, {}, {})
    assert route_ft is not None and route_ft.join_keys == ("route_id", "period")
    assert system_ft is not None and system_ft.join_keys == ("period",)
    assert pfp.describe_output(paths["long"], run, {}, {}) is None
    pfp.validate_config([route_ft, system_ft])  # no forbidden column names


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_returns_2_for_placeholders() -> None:
    assert deb.main([]) == 2


def test_main_returns_2_without_months(tmp_path) -> None:
    args = ["--events-csv", "e.csv", "--gtfs-path", "g", "--output-dir", str(tmp_path)]
    assert deb.main(args) == 2
    assert deb.main([*args, "--start-month", "2024-1", "--end-month", "2024-06"]) == 2


def test_main_end_to_end_on_sample_feed(tmp_path) -> None:
    out_dir = tmp_path / "out"
    code = deb.main(
        [
            "--events-csv",
            str(REPO_ROOT / "sample_data/disruptions_example.csv"),
            "--gtfs-path",
            str(REPO_ROOT / "sample_data/gtfs_current.zip"),
            "--start-month",
            "2021-01",
            "--end-month",
            "2024-12",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    by_route = pd.read_csv(out_dir / deb.ROUTE_FILENAME, dtype={"route_id": str})
    system = pd.read_csv(out_dir / deb.SYSTEM_FILENAME)
    assert len(system) == 48
    assert system.set_index("period").loc["2022-11", "sys_ev_sl2_open"] == pytest.approx(16 / 30)
    assert _cell(by_route, "50H", "2024-06", "ev_example_stop_closure") == pytest.approx(0.7)

    log = (out_dir / deb.RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "DC_NS_020 -> routes 10 (1/70 stops), 50H (1/66 stops)" in log
    assert "sha256=" in log
    assert "STEP_TRANSITION" in log  # verbatim config block
