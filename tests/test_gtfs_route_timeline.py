"""Tests for scripts.gtfs_exports.gtfs_route_timeline."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless: no display needed for savefig

from scripts.gtfs_exports.gtfs_route_timeline import (
    FeedPeriod,
    FeedSummary,
    Lineage,
    build_events_table,
    build_lineages,
    build_timeline_table,
    dedupe_labels,
    match_consecutive,
    natural_key,
    parse_date,
    presence_spans,
    resolve_periods,
    run_timeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _routes(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a routes table from [(route_id, route_short_name), ...] rows."""
    return pd.DataFrame(
        [
            {"route_id": rid, "route_short_name": short, "route_long_name": f"{short} Line"}
            for rid, short in rows
        ]
    )


def _summary(
    routes: list[tuple[str, str]],
    stop_sets: dict[str, frozenset[str]] | None = None,
    bounds: tuple[str, str, str] = ("", "", "unknown"),
) -> FeedSummary:
    """Build an in-memory feed summary matching load_feed_summary's shape."""
    return FeedSummary(routes=_routes(routes), stop_sets=stop_sets or {}, date_bounds=bounds)


def _write_feed(
    root: Path,
    name: str,
    routes: list[tuple[str, str]],
    trips: list[tuple[str, str]],
    stop_times: list[tuple[str, str, str]],
    start: str,
    end: str,
) -> Path:
    """Write a minimal on-disk GTFS feed and return its folder."""
    feed_dir = root / name
    feed_dir.mkdir()
    routes_lines = ["route_id,route_short_name,route_long_name"]
    routes_lines += [f"{rid},{short},{short} Line" for rid, short in routes]
    (feed_dir / "routes.txt").write_text("\n".join(routes_lines) + "\n", encoding="utf-8")
    trips_lines = ["route_id,trip_id,service_id"]
    trips_lines += [f"{rid},{tid},WK" for rid, tid in trips]
    (feed_dir / "trips.txt").write_text("\n".join(trips_lines) + "\n", encoding="utf-8")
    st_lines = ["trip_id,stop_id,stop_sequence,departure_time"]
    st_lines += [f"{tid},{sid},{seq},08:00:00" for tid, sid, seq in stop_times]
    (feed_dir / "stop_times.txt").write_text("\n".join(st_lines) + "\n", encoding="utf-8")
    (feed_dir / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n"
        f"WK,1,1,1,1,1,0,0,{start},{end}\n",
        encoding="utf-8",
    )
    return feed_dir


def _periods(starts_ends: list[tuple[str, str]]) -> list[FeedPeriod]:
    return [
        FeedPeriod(
            index=i,
            label=f"feed{i + 1}",
            gtfs_dir=f"dir{i + 1}",
            start=dt.date.fromisoformat(s),
            end=dt.date.fromisoformat(e),
            start_source="change_dates",
            route_count=0,
        )
        for i, (s, e) in enumerate(starts_ends)
    ]


# ---------------------------------------------------------------------------
# parse_date / natural_key / dedupe_labels
# ---------------------------------------------------------------------------


def test_parse_date_accepts_both_formats() -> None:
    assert parse_date("2024-07-01") == dt.date(2024, 7, 1)
    assert parse_date("20240701") == dt.date(2024, 7, 1)


def test_parse_date_rejects_garbage() -> None:
    assert parse_date("July 2024") is None
    assert parse_date("20241350") is None  # month 13
    assert parse_date("") is None


def test_natural_key_orders_numbers_numerically() -> None:
    labels = ["10", "2", "1A", "1"]
    assert sorted(labels, key=natural_key) == ["1", "1A", "2", "10"]


def test_dedupe_labels_suffixes_repeats() -> None:
    assert dedupe_labels(["gtfs", "gtfs", "other", "gtfs"]) == [
        "gtfs",
        "gtfs_2",
        "other",
        "gtfs_3",
    ]


# ---------------------------------------------------------------------------
# resolve_periods
# ---------------------------------------------------------------------------


def test_resolve_periods_chains_change_dates() -> None:
    summaries = [_summary([("R1", "1")]), _summary([("R1", "1")]), _summary([("R1", "1")])]
    periods = resolve_periods(
        ["d1", "d2", "d3"],
        ["f1", "f2", "f3"],
        summaries,
        ["2024-01-01", "2024-07-01", "2025-01-01"],
        "2025-06-30",
        90,
    )
    assert [p.start.isoformat() for p in periods] == ["2024-01-01", "2024-07-01", "2025-01-01"]
    assert [p.end.isoformat() for p in periods] == ["2024-07-01", "2025-01-01", "2025-06-30"]
    assert all(p.start_source == "change_dates" for p in periods)


def test_resolve_periods_infers_starts_from_feed_bounds() -> None:
    summaries = [
        _summary([("R1", "1")], bounds=("20240101", "20240630", "calendar")),
        _summary([("R1", "1")], bounds=("20240701", "20241231", "feed_info")),
    ]
    periods = resolve_periods(["d1", "d2"], ["f1", "f2"], summaries, None, None, 90)
    assert periods[0].start == dt.date(2024, 1, 1)
    assert periods[0].end == dt.date(2024, 7, 1)
    assert periods[1].end == dt.date(2024, 12, 31)  # last feed's own declared end
    assert periods[0].start_source == "calendar"
    assert periods[1].start_source == "feed_info"


def test_resolve_periods_mixed_override_and_inferred() -> None:
    summaries = [
        _summary([("R1", "1")], bounds=("20240101", "20240630", "calendar")),
        _summary([("R1", "1")], bounds=("", "", "unknown")),
    ]
    periods = resolve_periods(
        ["d1", "d2"], ["f1", "f2"], summaries, [None, "2024-07-01"], "2024-12-31", 90
    )
    assert periods[0].start_source == "calendar"
    assert periods[1].start_source == "change_dates"


def test_resolve_periods_final_end_falls_back_to_fixed_days() -> None:
    summaries = [
        _summary([("R1", "1")], bounds=("20240101", "20240630", "calendar")),
        _summary([("R1", "1")], bounds=("20240701", "", "calendar")),
    ]
    periods = resolve_periods(["d1", "d2"], ["f1", "f2"], summaries, None, None, 90)
    assert periods[1].end == dt.date(2024, 7, 1) + dt.timedelta(days=90)


def test_resolve_periods_rejects_single_feed() -> None:
    with pytest.raises(ValueError, match="at least two feeds"):
        resolve_periods(["d1"], ["f1"], [_summary([("R1", "1")])], None, None, 90)


def test_resolve_periods_rejects_non_increasing_starts() -> None:
    summaries = [_summary([("R1", "1")]), _summary([("R1", "1")])]
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_periods(
            ["d1", "d2"], ["f1", "f2"], summaries, ["2024-07-01", "2024-01-01"], None, 90
        )


def test_resolve_periods_rejects_length_mismatch() -> None:
    summaries = [_summary([("R1", "1")]), _summary([("R1", "1")])]
    with pytest.raises(ValueError, match="CHANGE_DATES"):
        resolve_periods(["d1", "d2"], ["f1", "f2"], summaries, ["2024-01-01"], None, 90)


def test_resolve_periods_requires_inferable_start() -> None:
    summaries = [_summary([("R1", "1")]), _summary([("R1", "1")])]
    with pytest.raises(ValueError, match="CHANGE_DATES"):
        resolve_periods(["d1", "d2"], ["f1", "f2"], summaries, None, None, 90)


# ---------------------------------------------------------------------------
# match_consecutive
# ---------------------------------------------------------------------------


def test_match_consecutive_matches_shared_ids() -> None:
    mapping = match_consecutive(_routes([("R1", "1")]), _routes([("R1", "1")]), {}, {}, 0.5)
    assert mapping == {"R1": "R1"}


def test_match_consecutive_rekeys_on_short_name_and_stops() -> None:
    stops = {"R2": frozenset({"a", "b", "c"})}
    stops_new = {"R2A": frozenset({"a", "b", "c"})}
    mapping = match_consecutive(
        _routes([("R2", "2")]), _routes([("R2A", "2")]), stops, stops_new, 0.5
    )
    assert mapping == {"R2": "R2A"}


def test_match_consecutive_no_rekey_when_short_names_differ() -> None:
    stops = {"R2": frozenset({"a", "b"})}
    stops_new = {"R9": frozenset({"a", "b"})}
    mapping = match_consecutive(
        _routes([("R2", "2")]), _routes([("R9", "9")]), stops, stops_new, 0.5
    )
    assert mapping == {}


def test_match_consecutive_skips_rekey_without_stop_sets() -> None:
    mapping = match_consecutive(_routes([("R2", "2")]), _routes([("R2A", "2")]), {}, {}, 0.5)
    assert mapping == {}


def test_match_consecutive_rekey_target_claimed_once() -> None:
    stops = {"R2": frozenset({"a", "b"}), "R3": frozenset({"a", "b"})}
    stops_new = {"RX": frozenset({"a", "b"})}
    mapping = match_consecutive(
        _routes([("R2", "2"), ("R3", "2")]), _routes([("RX", "2")]), stops, stops_new, 0.5
    )
    assert list(mapping.values()).count("RX") == 1


# ---------------------------------------------------------------------------
# build_lineages / presence_spans
# ---------------------------------------------------------------------------


def test_build_lineages_continues_added_and_removed() -> None:
    summaries = [
        _summary([("R1", "1"), ("R2", "2")]),
        _summary([("R1", "1"), ("R3", "3")]),
    ]
    lineages = build_lineages(summaries, 0.5)
    by_name = {ln.display_name(): ln for ln in lineages}
    assert set(by_name) == {"1", "2", "3"}
    assert by_name["1"].ids == {0: "R1", 1: "R1"}
    assert by_name["2"].ids == {0: "R2"}
    assert by_name["3"].ids == {1: "R3"}


def test_build_lineages_rekey_keeps_one_lineage() -> None:
    stops = {"R1": frozenset({"a"}), "R2": frozenset({"x", "y", "z"})}
    stops_new = {"R1": frozenset({"a"}), "R2A": frozenset({"x", "y", "z"})}
    summaries = [
        _summary([("R1", "1"), ("R2", "2")], stops),
        _summary([("R1", "1"), ("R2A", "2")], stops_new),
    ]
    lineages = build_lineages(summaries, 0.5)
    assert len(lineages) == 2
    r2 = next(ln for ln in lineages if ln.display_name() == "2")
    assert r2.ids == {0: "R2", 1: "R2A"}


def test_build_lineages_gap_rejoins_same_route_id() -> None:
    summaries = [
        _summary([("R1", "1"), ("R2", "2")]),
        _summary([("R1", "1")]),
        _summary([("R1", "1"), ("R2", "2")]),
    ]
    lineages = build_lineages(summaries, 0.5)
    assert len(lineages) == 2
    r2 = next(ln for ln in lineages if ln.display_name() == "2")
    assert r2.ids == {0: "R2", 2: "R2"}


def test_build_lineages_sorted_by_first_appearance_then_name() -> None:
    summaries = [
        _summary([("R10", "10"), ("R2", "2")]),
        _summary([("R10", "10"), ("R2", "2"), ("R1", "1")]),
    ]
    lineages = build_lineages(summaries, 0.5)
    assert [ln.display_name() for ln in lineages] == ["2", "10", "1"]


def test_presence_spans_merges_contiguous_periods() -> None:
    periods = _periods(
        [("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-06-30")]
    )
    lineage = Lineage(ids={0: "R1", 1: "R1"}, short_name="1")
    assert presence_spans(lineage, periods) == [(dt.date(2024, 1, 1), dt.date(2025, 1, 1))]
    gappy = Lineage(ids={0: "R2", 2: "R2"}, short_name="2")
    assert presence_spans(gappy, periods) == [
        (dt.date(2024, 1, 1), dt.date(2024, 7, 1)),
        (dt.date(2025, 1, 1), dt.date(2025, 6, 30)),
    ]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def test_build_timeline_table_columns_and_gaps() -> None:
    periods = _periods(
        [("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-06-30")]
    )
    lineages = [Lineage(ids={0: "R2", 2: "R2"}, short_name="2", long_name="2 Line")]
    table = build_timeline_table(lineages, periods)
    row = table.iloc[0]
    assert row["route"] == "2"
    assert row["first_active"] == "2024-01-01"
    assert row["last_active"] == "2025-06-30"
    assert row["feeds_present"] == 2
    assert row["gap_feeds"] == 1
    assert (row["feed1"], row["feed2"], row["feed3"]) == ("R2", "", "R2")


def test_build_events_table_all_event_kinds() -> None:
    periods = _periods(
        [("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-06-30")]
    )
    lineages = [
        Lineage(ids={1: "R3", 2: "R3"}, short_name="3"),  # added in feed2
        Lineage(ids={0: "R2", 2: "R2"}, short_name="2"),  # removed then reappears
        Lineage(ids={0: "R4", 1: "R4A"}, short_name="4"),  # rekeyed, then removed
    ]
    events = build_events_table(lineages, periods)
    kinds = dict(zip(events["route"] + "|" + events["date"], events["event"]))
    assert kinds["3|2024-07-01"] == "added"
    assert kinds["2|2024-07-01"] == "removed"
    assert kinds["2|2025-01-01"] == "reappeared"
    assert kinds["4|2024-07-01"] == "rekeyed"
    assert kinds["4|2025-01-01"] == "removed"
    removed_with_return = events[(events["route"] == "2") & (events["event"] == "removed")].iloc[0]
    assert removed_with_return["detail"] == "returns in feed3"
    rekey = events[events["event"] == "rekeyed"].iloc[0]
    assert rekey["detail"] == "R4 -> R4A"


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


def test_run_timeline_end_to_end(tmp_path: Path) -> None:
    # feed1: routes 1 and 2. feed2: route 2 rekeyed to R2A, route 3 added.
    # feed3: route 2 (R2A) removed.
    f1 = _write_feed(
        tmp_path,
        "gtfs_2024_winter",
        routes=[("R1", "1"), ("R2", "2")],
        trips=[("R1", "t1"), ("R2", "t2")],
        stop_times=[("t1", "a", "1"), ("t1", "b", "2"), ("t2", "x", "1"), ("t2", "y", "2")],
        start="20240101",
        end="20240630",
    )
    f2 = _write_feed(
        tmp_path,
        "gtfs_2024_summer",
        routes=[("R1", "1"), ("R2A", "2"), ("R3", "3")],
        trips=[("R1", "t1"), ("R2A", "t2"), ("R3", "t3")],
        stop_times=[
            ("t1", "a", "1"),
            ("t1", "b", "2"),
            ("t2", "x", "1"),
            ("t2", "y", "2"),
            ("t3", "g", "1"),
            ("t3", "h", "2"),
        ],
        start="20240701",
        end="20241231",
    )
    f3 = _write_feed(
        tmp_path,
        "gtfs_2025_winter",
        routes=[("R1", "1"), ("R3", "3")],
        trips=[("R1", "t1"), ("R3", "t3")],
        stop_times=[("t1", "a", "1"), ("t1", "b", "2"), ("t3", "g", "1"), ("t3", "h", "2")],
        start="20250101",
        end="20250630",
    )
    out_dir = tmp_path / "out"

    timeline = run_timeline(
        feed_dirs=[str(f1), str(f2), str(f3)],
        labels=["winter24", "summer24", "winter25"],
        change_dates=None,
        out_dir=out_dir,
    )

    assert len(timeline) == 3  # routes 1, 2, 3 -- the rekey did NOT create a 4th row
    r2 = timeline[timeline["route"] == "2"].iloc[0]
    assert (r2["winter24"], r2["summer24"], r2["winter25"]) == ("R2", "R2A", "")
    assert r2["first_active"] == "2024-01-01"
    assert r2["last_active"] == "2025-01-01"

    events = pd.read_csv(out_dir / "route_timeline_events.csv", keep_default_na=False)
    assert set(events["event"]) == {"added", "removed", "rekeyed"}

    periods = pd.read_csv(out_dir / "feed_periods.csv")
    assert periods["start"].tolist() == ["2024-01-01", "2024-07-01", "2025-01-01"]
    assert periods["end"].tolist() == ["2024-07-01", "2025-01-01", "2025-06-30"]

    for name in (
        "route_timeline.csv",
        "route_timeline_events.csv",
        "feed_periods.csv",
        "route_timeline.xlsx",
        "route_timeline.png",
    ):
        assert (out_dir / name).exists(), name
    assert (out_dir / "route_timeline.png").stat().st_size > 0

    runlog = (out_dir / "route_timeline_runlog.txt").read_text(encoding="utf-8")
    assert "GTFS_FEEDS" in runlog  # config block captured
    assert "RESOLVED FEED PERIODS" in runlog


def test_run_timeline_rejects_label_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FEED_LABELS"):
        run_timeline(feed_dirs=["a", "b"], labels=["only-one"], out_dir=tmp_path)
