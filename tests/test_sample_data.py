"""Check that each copy in sample_data/ still matches its tests/fixtures/ original.

sample_data/ holds copies of the mock DC fixtures for users to try scripts on. The tests
read the originals, so a regenerated fixture that isn't copied again would leave the two
apart without any other test noticing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA = REPO_ROOT / "sample_data"
FIXTURES = REPO_ROOT / "tests" / "fixtures"

# sample_data/ copy -> tests/fixtures/ original
COPIES = {
    "gtfs_current.zip": "mock_gtfs_dc.zip",
    "gtfs_archive.zip": "mock_gtfs_dc_archive.zip",
    "tods_current.zip": "mock_tods_dc.zip",
    "tides/stop_visits.csv": "stop_visits.csv",
    "tides/trips_performed.csv": "trips_performed.csv",
    "disruptions_example.csv": "disruptions_example.csv",
}


@pytest.mark.parametrize(("copy", "original"), COPIES.items())
def test_sample_copy_matches_fixture(copy: str, original: str) -> None:
    copy_path = SAMPLE_DATA / copy
    original_path = FIXTURES / original
    assert copy_path.read_bytes() == original_path.read_bytes(), (
        f"sample_data/{copy} differs from tests/fixtures/{original}; "
        f"copy the fixture over it again."
    )
