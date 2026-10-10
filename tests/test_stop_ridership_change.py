import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add the script directory to sys.path to allow importing the module
script_dir = Path("scripts/ridership_tools").resolve()
sys.path.append(str(script_dir))

import stop_ridership_change as target  # noqa: E402

COLUMNS = ["TIME_PERIOD", "ROUTE_NAME", "STOP", "STOP_ID", "BOARD_ALL", "ALIGHT_ALL"]

# Three synthetic signups:
#   1001 on routes 10 + 20 throughout (route 20 skips signup 2)
#   1002 on route 10 throughout; route 20 serves it only in signup 2
#   1003 present, absent, present again (intermittent)
#   1004 added in signup 2
#   2001 present (zero ridership) in signup 1, removed in signup 3
SIGNUP_1 = [
    ["AM PEAK", "10", "Main & 1st", 1001, 10, 2],
    ["PM PEAK", "10", "Main & 1st", 1001, 5, 8],
    ["AM PEAK", "20", "Main & 1st", 1001, 4, 1],
    ["AM PEAK", "10", "Main & 2nd", 1002, 3, 3],
    ["AM PEAK", "10", "Old Stop", 1003, 6, 0],
    ["AM PEAK", "2", "Elm", 2001, 0, 0],
    ["AM PEAK", None, None, None, 999, 999],  # grand-total row
]
SIGNUP_2 = [
    ["AM PEAK", "10", "Main & 1st", 1001, 12, 2],
    ["PM PEAK", "10", "Main & 1st", 1001, 6, 8],
    ["AM PEAK", "10", "Main & 2nd", 1002, 1.5, 3],
    ["AM PEAK", "20", "Main & 2nd", 1002, 2, 2],
    ["AM PEAK", "10", "New Stop", 1004, 7, 1],
    ["AM PEAK", "2", "Elm", 2001, 1, 0],
]
SIGNUP_3 = [
    ["AM PEAK", "10", "Main & 1st", 1001.0, 15, 2],
    ["PM PEAK", "10", "Main & 1st", 1001.0, 6, 9],
    ["AM PEAK", "20", "Main & 1st", 1001.0, 3, 1],
    ["AM PEAK", "10", "Main & 2nd", 1002.0, 2, 3],
    ["AM PEAK", "10", "New Stop", 1004.0, 9, 1],
    ["AM PEAK", "10", "Old Stop", 1003.0, 2, 0],
]
LABELS = ["Fall 25", "Spring 26", "Fall 26"]


def _signup(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def _panel(aggregate_routes_together: bool, **kwargs: bool) -> pd.DataFrame:
    keys = target.key_columns(aggregate_routes_together)
    tables = [
        target.aggregate_signup(
            target.prepare_signup(_signup(rows), source=f"s{i}"),
            aggregate_routes_together=aggregate_routes_together,
        )
        for i, rows in enumerate((SIGNUP_1, SIGNUP_2, SIGNUP_3), start=1)
    ]
    return target.build_panel(tables, keys, **kwargs)


def _write_inputs(tmp_path: Path) -> list:
    paths = []
    for name, rows in (("s1", SIGNUP_1), ("s2", SIGNUP_2), ("s3", SIGNUP_3)):
        path = tmp_path / f"{name}.csv"
        _signup(rows).to_csv(path, index=False)
        paths.append(path)
    return paths


# --- pure helpers -----------------------------------------------------------


def test_normalize_id_unifies_excel_number_formats() -> None:
    raw = pd.Series([1001, 1001.0, " 1001 ", "10A", None, "", float("nan")])
    out = target.normalize_id(raw)
    assert out.iloc[:4].tolist() == ["1001", "1001", "1001", "10A"]
    assert out.iloc[4:].isna().all()


def test_pct_change_blanks_zero_missing_and_small_bases() -> None:
    old = pd.Series([10.0, 0.0, None, 2.0, 50.0])
    new = pd.Series([15.0, 5.0, 5.0, 8.0, None])
    out = target.pct_change(old, new)
    assert out.iloc[0] == pytest.approx(50.0)
    assert math.isnan(out.iloc[1]) and math.isnan(out.iloc[2]) and math.isnan(out.iloc[4])
    assert out.iloc[3] == pytest.approx(300.0)

    guarded = target.pct_change(old, new, min_base=5.0)
    assert guarded.iloc[0] == pytest.approx(50.0)
    assert math.isnan(guarded.iloc[3])  # 2 → 8 is below the 5-rider base


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([True, True, True], "present in all"),
        ([False, True, True], "added"),
        ([False, False, True], "added"),
        ([True, True, False], "removed"),
        ([True, False, True], "intermittent"),
        ([False, True, False], "intermittent"),
    ],
)
def test_classify_presence(flags: list, expected: str) -> None:
    assert target.classify_presence(flags) == expected


def test_derive_signup_labels() -> None:
    same_name = [Path("Fall25/RIDERSHIP.XLSX"), Path("Spring26/RIDERSHIP.XLSX")]
    assert target.derive_signup_labels(same_name, []) == ["Fall25", "Spring26"]
    assert target.derive_signup_labels([Path("a.xlsx"), Path("b.xlsx")], []) == ["a", "b"]
    assert target.derive_signup_labels([Path("x/r.xlsx"), Path("x/r.xlsx")], []) == [
        "Signup 1",
        "Signup 2",
    ]
    assert target.derive_signup_labels(same_name, [" F ", "S"]) == ["F", "S"]


@pytest.mark.parametrize(
    "labels",
    [["only one"], ["dup", "dup"], ["ok", " "], ["STATUS", "ok"]],
)
def test_derive_signup_labels_rejects_bad_labels(labels: list) -> None:
    with pytest.raises(ValueError):
        target.derive_signup_labels([Path("a.xlsx"), Path("b.xlsx")], labels)


def test_prepare_signup_drops_keyless_rows_and_filters() -> None:
    df = target.prepare_signup(_signup(SIGNUP_1), source="s1")
    assert len(df) == 6  # grand-total row dropped
    assert df["TOTAL"].tolist() == (df["BOARD_ALL"] + df["ALIGHT_ALL"]).tolist()

    routes = target.prepare_signup(_signup(SIGNUP_1), source="s1", routes=["10"])
    assert set(routes["ROUTE_NAME"]) == {"10"}

    stops = target.prepare_signup(_signup(SIGNUP_1), source="s1", stop_ids=[1001])
    assert set(stops["STOP_ID"]) == {"1001"}

    am = target.prepare_signup(_signup(SIGNUP_1), source="s1", time_periods=["am peak"])
    assert "PM PEAK" not in am.get("TIME_PERIOD", pd.Series(dtype=object)).tolist()
    assert len(am) == 5


def test_prepare_signup_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="BOARD_ALL"):
        target.prepare_signup(_signup(SIGNUP_1).drop(columns=["BOARD_ALL"]), source="s1")


# --- comparison tables -----------------------------------------------------


def test_stop_level_metric_table() -> None:
    panel = _panel(True)
    keys = target.key_columns(True)
    attrs = target.build_key_attributes(panel, keys, LABELS)
    table = target.build_metric_table(panel, attrs, "BOARD_ALL", LABELS, keys).set_index("STOP_ID")

    assert table.loc["1001", LABELS].tolist() == [19.0, 18.0, 24.0]
    assert table.loc["1001", "ROUTES"] == "10, 20"
    assert bool(table.loc["1001", "ROUTES_CHANGED"]) is True
    assert table.loc["1001", "% Chg Fall 25 to Fall 26"] == pytest.approx(26.32)
    assert table.loc["1003", "STATUS"] == "intermittent"
    assert table.loc["1004", "STATUS"] == "added"
    assert table.loc["2001", "STATUS"] == "removed"
    # Absent signups leave change cells blank rather than showing -100%.
    assert math.isnan(table.loc["1004", "Chg Fall 25 to Spring 26"])
    # A zero base still gets an absolute change but no % change.
    assert table.loc["2001", "Chg Fall 25 to Spring 26"] == pytest.approx(1.0)
    assert math.isnan(table.loc["2001", "% Chg Fall 25 to Spring 26"])


def test_two_signups_have_no_separate_first_to_last_columns() -> None:
    keys = target.key_columns(True)
    tables = [
        target.aggregate_signup(
            target.prepare_signup(_signup(rows), source="s"), aggregate_routes_together=True
        )
        for rows in (SIGNUP_1, SIGNUP_2)
    ]
    panel = target.build_panel(tables, keys)
    attrs = target.build_key_attributes(panel, keys, ["A", "B"])
    table = target.build_metric_table(panel, attrs, "TOTAL", ["A", "B"], keys)
    assert [c for c in table.columns if c.startswith("Chg ")] == ["Chg A to B"]


def test_route_stop_level_sorts_naturally_and_flags_pattern_changes() -> None:
    panel = _panel(False)
    keys = target.key_columns(False)
    attrs = target.build_key_attributes(panel, keys, LABELS)
    table = target.build_metric_table(panel, attrs, "BOARD_ALL", LABELS, keys)
    assert table["ROUTE_NAME"].iloc[0] == "2"  # natural order: 2 before 10

    changes = target.build_added_removed(panel, keys, LABELS)
    pair = changes[(changes["ROUTE_NAME"] == "20") & (changes["STOP_ID"] == "1001")]
    assert pair["CHANGE"].tolist() == ["removed", "added"]
    # Stop 1001 itself stayed in service on route 10: a route-pattern change.
    assert pair["STOP_IN_BOTH_SIGNUPS"].all()
    new_stop = changes[changes["STOP_ID"] == "1004"].iloc[0]
    assert new_stop["CHANGE"] == "added"
    assert not new_stop["STOP_IN_BOTH_SIGNUPS"]
    assert new_stop["RIDERSHIP_FROM"] == "Spring 26"
    assert new_stop["BOARD_ALL"] == pytest.approx(7.0)


def test_change_summary_separates_all_keys_from_keys_in_both() -> None:
    panel = _panel(True)
    summary = target.build_change_summary(panel, target.key_columns(True), LABELS, ["BOARD_ALL"])
    assert summary["Comparison"].tolist() == ["consecutive", "consecutive", "first to last"]
    first = summary.iloc[0]
    assert (first["Keys in Both"], first["Added"], first["Removed"]) == (3, 1, 1)
    assert first["Boardings From"] == pytest.approx(28.0)
    assert first["Boardings To"] == pytest.approx(29.5)
    assert first["Boardings % Chg"] == pytest.approx(5.36)
    # Keys in both (1001, 1002, 2001): 22 → 22.5.
    assert first["Boardings % Chg (in both)"] == pytest.approx(2.27)


def test_treat_zero_as_absent_changes_presence() -> None:
    panel = _panel(True, treat_zero_as_absent=True)
    attrs = target.build_key_attributes(panel, target.key_columns(True), LABELS)
    assert attrs.loc["2001", "STATUS"] == "intermittent"  # zero in signup 1 → absent


def test_build_panel_raises_when_everything_filtered() -> None:
    empty = target.aggregate_signup(
        target.prepare_signup(_signup(SIGNUP_1), source="s1", routes=["999"]),
        aggregate_routes_together=True,
    )
    with pytest.raises(ValueError, match="No ridership rows"):
        target.build_panel([empty, empty], ["STOP_ID"])


def test_restore_numeric_ids() -> None:
    df = pd.DataFrame({"ROUTE_NAME": ["10", "10A"], "STOP_ID": ["1001", "1002"]})
    out = target.restore_numeric_ids(df)
    assert out["ROUTE_NAME"].tolist() == ["10", "10A"]
    assert out["STOP_ID"].tolist() == [1001, 1002]
    leading_zero = target.restore_numeric_ids(pd.DataFrame({"STOP_ID": ["0123", "456"]}))
    assert leading_zero["STOP_ID"].tolist() == ["0123", "456"]


# --- end to end ------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "suffix", "expected_keys"),
    [
        ("--aggregate-routes-together", "stop", 5),
        ("--no-aggregate-routes-together", "route_stop", 7),
    ],
)
def test_main_writes_workbook_and_run_log(
    tmp_path: Path, flag: str, suffix: str, expected_keys: int
) -> None:
    inputs = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    code = target.main(
        ["--inputs", *map(str, inputs), "--labels", *LABELS, "--output-dir", str(out_dir), flag]
    )
    assert code == 0

    workbook = out_dir / f"ridership_change_{suffix}.xlsx"
    sheets = pd.read_excel(workbook, sheet_name=None)
    assert list(sheets) == [
        "Change Summary",
        "Signup Totals",
        "Boardings",
        "Alightings",
        "Total (Board+Alight)",
        "Added & Removed",
        "Long",
    ]
    assert len(sheets["Boardings"]) == expected_keys
    assert len(sheets["Long"]) == expected_keys * len(LABELS)

    run_log = (out_dir / f"ridership_change_{suffix}_runlog.txt").read_text(encoding="utf-8")
    assert "BEGIN CONFIG" not in run_log
    assert "INPUT_FILES" in run_log  # verbatim CONFIG block
    assert f"Comparison level:     {suffix}" in run_log
    assert "SHA256:" in run_log and "Fall 26" in run_log


def test_main_returns_2_for_placeholders_and_single_input(tmp_path: Path) -> None:
    assert target.main([]) == 2
    one = _write_inputs(tmp_path)[:1]
    assert target.main(["--inputs", str(one[0]), "--output-dir", str(tmp_path)]) == 2


def test_main_returns_2_for_label_mismatch(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    code = target.main(
        ["--inputs", *map(str, inputs), "--labels", "A", "B", "--output-dir", str(tmp_path)]
    )
    assert code == 2


def test_main_returns_1_for_missing_input(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    code = target.main(
        [
            "--inputs",
            str(inputs[0]),
            str(tmp_path / "missing.xlsx"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1


# --- input hardening and output publishing ----------------------------------


def test_stop_ids_file_keeps_leading_zeros(tmp_path: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("﻿0123, 1001 # inline note\n# whole-line comment\n1001.0\n", encoding="utf-8")
    assert target.resolve_stop_ids([], ids) == ["0123", "1001"]
    assert target.resolve_stop_ids(["0123", 1001], None) == ["0123", "1001"]


def test_prepare_signup_parses_grouped_numbers_and_blank_cells() -> None:
    raw = pd.DataFrame(
        [["AM PEAK", "NA", "A", "0123", "1,234.5", ""], ["AM PEAK", "10", "B", "7", "", "2"]],
        columns=COLUMNS,
    )
    df = target.prepare_signup(raw, source="s1")
    assert df["ROUTE_NAME"].tolist() == ["NA", "10"]
    assert df["STOP_ID"].tolist() == ["0123", "7"]
    assert df["BOARD_ALL"].tolist() == [1234.5, 0.0]
    assert df["ALIGHT_ALL"].tolist() == [0.0, 2.0]


def test_prepare_signup_rejects_infinite_values() -> None:
    raw = _signup(SIGNUP_1)
    raw["BOARD_ALL"] = raw["BOARD_ALL"].astype(object)
    raw.loc[0, "BOARD_ALL"] = "inf"
    with pytest.raises(ValueError, match="infinite"):
        target.prepare_signup(raw, source="s1")


def test_labels_cannot_collide_with_generated_change_columns() -> None:
    with pytest.raises(ValueError, match="duplicate output column"):
        target.derive_signup_labels(
            [Path("a.xlsx"), Path("b.xlsx"), Path("c.xlsx")], ["A", "B", "Chg A to B"]
        )


def test_restore_numeric_ids_keeps_long_ids_as_text() -> None:
    out = target.restore_numeric_ids(pd.DataFrame({"STOP_ID": ["1234567890123456", "1"]}))
    assert out["STOP_ID"].tolist() == ["1234567890123456", "1"]


def test_resolve_script_source_prefers_module_file() -> None:
    _, label = target._resolve_script_source()
    assert label.endswith("stop_ridership_change.py")


def test_main_overwrites_cleanly_without_leftover_staging(tmp_path: Path) -> None:
    inputs = [str(p) for p in _write_inputs(tmp_path)]
    out_dir = tmp_path / "out"
    for _ in range(2):
        assert target.main(["--inputs", *inputs, "--output-dir", str(out_dir)]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "ridership_change_stop.xlsx",
        "ridership_change_stop_runlog.txt",
    ]


def test_main_restores_previous_workbook_when_publishing_fails(tmp_path: Path) -> None:
    inputs = [str(p) for p in _write_inputs(tmp_path)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "ridership_change_stop.xlsx").write_bytes(b"previous")
    # A directory where the run log should go makes the second replacement fail.
    (out_dir / "ridership_change_stop_runlog.txt").mkdir()

    assert target.main(["--inputs", *inputs, "--output-dir", str(out_dir)]) == 1
    assert (out_dir / "ridership_change_stop.xlsx").read_bytes() == b"previous"
    assert not any(p.name.startswith(".ridership_change_") for p in out_dir.iterdir())


@pytest.mark.parametrize(
    "extra",
    [["--output-filename", "change.csv"], ["--min-base-for-pct", "-1"]],
)
def test_main_returns_2_for_bad_output_name_or_min_base(tmp_path: Path, extra: list) -> None:
    inputs = [str(p) for p in _write_inputs(tmp_path)]
    assert target.main(["--inputs", *inputs, "--output-dir", str(tmp_path), *extra]) == 2


def test_main_refuses_to_overwrite_an_input(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    clash = tmp_path / "ridership_change_stop.xlsx"
    _signup(SIGNUP_1).to_excel(clash, index=False)
    code = target.main(["--inputs", str(clash), str(inputs[1]), "--output-dir", str(tmp_path)])
    assert code == 2
