from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pytest

from utils.gtfs_helpers import validate_gtfs_files_exist


def test_validate_warns_for_each_missing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Each requested file that is absent triggers a 'Missing GTFS file' warning."""
    folder = tmp_path / "gtfs"
    folder.mkdir()
    (folder / "stops.txt").write_text("stop_id\n001\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        validate_gtfs_files_exist(str(folder), files=("stops.txt", "trips.txt", "routes.txt"))

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Missing GTFS file: trips.txt" in m for m in messages)
    assert any("Missing GTFS file: routes.txt" in m for m in messages)
    assert not any("Missing GTFS file: stops.txt" in m for m in messages)


def test_validate_no_warnings_when_all_present(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When every requested file exists, no warnings are emitted."""
    folder = tmp_path / "gtfs"
    folder.mkdir()
    (folder / "stops.txt").write_text("stop_id\n001\n", encoding="utf-8")
    (folder / "trips.txt").write_text("trip_id\nA\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        validate_gtfs_files_exist(str(folder), files=("stops.txt", "trips.txt"))

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_validate_missing_directory_warns_and_returns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing directory yields a single directory warning and skips file checks."""
    missing = tmp_path / "no_such_folder"

    with caplog.at_level(logging.WARNING):
        validate_gtfs_files_exist(str(missing), files=("stops.txt", "trips.txt"))

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(missing) in warnings[0]
    assert "does not exist" in warnings[0]
    assert not any("Missing GTFS file" in m for m in warnings)


def test_validate_default_file_list_flags_standard_gtfs_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With files=None, the standard GTFS set is checked against the folder."""
    folder = tmp_path / "gtfs"
    folder.mkdir()
    (folder / "stops.txt").write_text("stop_id\n001\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        validate_gtfs_files_exist(str(folder))

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    # stops.txt is present, so it should not be flagged.
    assert not any("Missing GTFS file: stops.txt" in m for m in messages)
    # A few standard files we know are absent should be flagged.
    for expected in ("trips.txt", "routes.txt", "stop_times.txt"):
        assert any(f"Missing GTFS file: {expected}" in m for m in messages)


# ---------------------------------------------------------------------------
# zip archive support
# ---------------------------------------------------------------------------


def test_validate_zip_wrapper_folder_no_false_positives(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A zip nesting files in one wrapper folder is not flagged as missing."""
    zip_path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dc/stops.txt", "stop_id\n001\n")
        zf.writestr("dc/trips.txt", "trip_id\nA\n")

    with caplog.at_level(logging.WARNING):
        validate_gtfs_files_exist(str(zip_path), files=("stops.txt", "trips.txt"))

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_validate_zip_missing_file_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A file absent from the zip triggers the same warning as a missing folder file."""
    zip_path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("stops.txt", "stop_id\n001\n")

    with caplog.at_level(logging.WARNING):
        validate_gtfs_files_exist(str(zip_path), files=("stops.txt", "trips.txt"))

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Missing GTFS file: trips.txt" in m for m in messages)
    assert not any("Missing GTFS file: stops.txt" in m for m in messages)


def test_validate_bad_zip_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A .zip path that isn't a valid archive warns instead of raising."""
    bad_zip = tmp_path / "not_really_a.zip"
    bad_zip.write_text("this is not a zip file", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        validate_gtfs_files_exist(str(bad_zip), files=("stops.txt",))

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not a valid zip archive" in m for m in warnings)
