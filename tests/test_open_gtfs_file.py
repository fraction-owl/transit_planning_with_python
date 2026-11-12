from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from utils.gtfs_helpers import open_gtfs_file

STOPS = "stop_id,stop_code,stop_name\n001,NA,Main St\n002,,Oak Ave\n"


def _mk_zip(tmp_path: Path, members: dict[str, str], name: str = "gtfs.zip") -> Path:
    zip_path = tmp_path / name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, contents in members.items():
            zf.writestr(arcname, contents)
    return zip_path


def test_open_gtfs_file_folder_keeps_caller_read_options(tmp_path) -> None:
    """The caller's read_csv options apply, e.g. keep_default_na=False keeps "NA"."""
    (tmp_path / "stops.txt").write_text(STOPS, encoding="utf-8")
    with open_gtfs_file(tmp_path, "stops.txt") as handle:
        df = pd.read_csv(handle, dtype=str, keep_default_na=False)
    assert df["stop_id"].tolist() == ["001", "002"]
    assert df["stop_code"].tolist() == ["NA", ""]


def test_open_gtfs_file_accepts_str_path(tmp_path) -> None:
    (tmp_path / "stops.txt").write_text(STOPS, encoding="utf-8")
    with open_gtfs_file(str(tmp_path), "stops.txt") as handle:
        assert handle.read().decode("utf-8") == STOPS


@pytest.mark.parametrize("arcname", ["stops.txt", "feed/stops.txt"])
def test_open_gtfs_file_zip_root_and_wrapper_folder(tmp_path, arcname) -> None:
    zip_path = _mk_zip(tmp_path, {arcname: STOPS, "routes.txt": "route_id\n1\n"})
    with open_gtfs_file(zip_path, "stops.txt") as handle:
        df = pd.read_csv(handle, dtype=str, usecols=["stop_id"])
    assert df["stop_id"].tolist() == ["001", "002"]


def test_open_gtfs_file_zip_handle_works_with_csv_module(tmp_path) -> None:
    zip_path = _mk_zip(tmp_path, {"stops.txt": "﻿" + STOPS})
    with open_gtfs_file(zip_path, "stops.txt") as handle:
        rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")))
    assert [row["stop_id"] for row in rows] == ["001", "002"]


def test_open_gtfs_file_ignores_macos_metadata_member(tmp_path) -> None:
    zip_path = _mk_zip(tmp_path, {"feed/stops.txt": STOPS, "__MACOSX/feed/._stops.txt": "junk"})
    with open_gtfs_file(zip_path, "stops.txt") as handle:
        assert handle.read().decode("utf-8") == STOPS


def test_open_gtfs_file_mock_dc_fixture() -> None:
    """The shared fixture nests its files in a ``dc/`` wrapper folder."""
    fixture = Path(__file__).parent / "fixtures" / "mock_gtfs_dc.zip"
    with open_gtfs_file(fixture, "stops.txt") as handle:
        df = pd.read_csv(handle, dtype=str)
    assert {"stop_id", "stop_lat", "stop_lon"} <= set(df.columns)
    assert not df.empty


def test_open_gtfs_file_missing_file_in_folder_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="shapes.txt"):
        with open_gtfs_file(tmp_path, "shapes.txt"):
            pass


def test_open_gtfs_file_missing_file_in_zip_raises(tmp_path) -> None:
    zip_path = _mk_zip(tmp_path, {"stops.txt": STOPS})
    with pytest.raises(FileNotFoundError, match="shapes.txt"):
        with open_gtfs_file(zip_path, "shapes.txt"):
            pass


def test_open_gtfs_file_missing_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        with open_gtfs_file(tmp_path / "nope.zip", "stops.txt"):
            pass


def test_open_gtfs_file_ambiguous_zip_member_raises(tmp_path) -> None:
    zip_path = _mk_zip(tmp_path, {"a/stops.txt": STOPS, "b/stops.txt": STOPS})
    with pytest.raises(ValueError, match="Ambiguous"):
        with open_gtfs_file(zip_path, "stops.txt"):
            pass


def test_open_gtfs_file_bad_zip_raises(tmp_path) -> None:
    bad = tmp_path / "gtfs.zip"
    bad.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid zip"):
        with open_gtfs_file(bad, "stops.txt"):
            pass


def test_open_gtfs_file_non_zip_file_raises(tmp_path) -> None:
    other = tmp_path / "stops.csv"
    other.write_text(STOPS, encoding="utf-8")
    with pytest.raises(ValueError, match="neither a directory nor a .zip"):
        with open_gtfs_file(other, "stops.txt"):
            pass
