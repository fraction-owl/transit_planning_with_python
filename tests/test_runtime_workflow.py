"""Tests for runtime_workflow's GTFS schedule loader."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.operations_tools import runtime_workflow

GTFS_FIXTURE = Path("tests/fixtures/gtfs_basic")


@pytest.mark.parametrize("short_names", ["absent", "blank"])
def test_load_schedule_selects_route_by_id_without_short_names(
    tmp_path: Path, short_names: str
) -> None:
    gtfs = tmp_path / "gtfs"
    shutil.copytree(GTFS_FIXTURE, gtfs)
    routes = pd.read_csv(gtfs / "routes.txt", dtype=str)
    if short_names == "absent":
        routes = routes.drop(columns="route_short_name")
    else:
        routes["route_short_name"] = ""
    routes.to_csv(gtfs / "routes.txt", index=False)

    cfg = runtime_workflow.Settings(
        gtfs_path=gtfs,
        gtfs_service_date="2026-03-02",
        routes=("R2",),
        output_dir=tmp_path / "out",
    )
    trips = runtime_workflow.load_schedule(cfg)["trips"]
    assert len(trips) == 3
    assert set(trips["route"]) == {"R2"}
