from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.service_coverage.private_shuttle_coverage_by_route_gpd as shuttle_mod
from scripts.service_coverage.points_of_interest_coverage_gpd import (
    _find_layer_sources,
    _load_layers,
)
from scripts.service_coverage.private_shuttle_coverage_by_route_gpd import (
    CATEGORY_SHUTTLE,
    CATEGORY_TRANSIT_FEEDER,
    CATEGORY_UNSPECIFIED,
    REASON_INVALID_COORDS,
    REASON_MISSING_COORDS,
    categorize_notes,
    clean_registry,
    load_registry_csv,
    run,
)

FIXTURE_CSV = Path("tests/fixtures/private_shuttles_sample.csv")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_gtfs_files(gtfs_dir: Path) -> None:
    """Write a two-route DC-area GTFS feed whose R1 stops sit on fixture sites.

    R1's stops land exactly on three fixture operators (Consulting Unlimited,
    Grand Hotel, Riverside Apartments), so its ¼-mile catchment captures those
    three and nothing else; R2's single stop is far from every fixture site.
    """
    (gtfs_dir / "routes.txt").write_text(
        "route_id,route_short_name\nR1,101\nR2,202\n", encoding="utf-8"
    )
    (gtfs_dir / "trips.txt").write_text("route_id,trip_id\nR1,T1\nR2,T2\n", encoding="utf-8")
    (gtfs_dir / "stop_times.txt").write_text(
        "trip_id,stop_id,stop_sequence\nT1,S1,1\nT1,S2,2\nT1,S3,3\nT2,S4,1\n", encoding="utf-8"
    )
    (gtfs_dir / "stops.txt").write_text(
        "stop_id,stop_lat,stop_lon\n"
        "S1,38.9442712599487,-77.0264059709359\n"  # Consulting Unlimited (unspecified)
        "S2,38.9236,-77.0523\n"  # Grand Hotel (shuttle)
        "S3,38.8734,-76.9948\n"  # Riverside Apartments (transit_feeder)
        "S4,38.9900,-77.1900\n",  # far from every fixture site
        encoding="utf-8",
    )


def _cleaned_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean the shipped sample registry."""
    return clean_registry(load_registry_csv(FIXTURE_CSV))


# ---------------------------------------------------------------------------
# load_registry_csv
# ---------------------------------------------------------------------------


def test_load_registry_matches_headers_case_insensitively(tmp_path: Path) -> None:
    """Lower-case headers still resolve to the canonical column names."""
    path = tmp_path / "registry.csv"
    path.write_text(
        "company,address,city,state,zip,x,y,notes\n"
        "Acme,1 Main St,Springfield,VA,22150,-77.18,38.77,Shuttle\n",
        encoding="utf-8",
    )
    out = load_registry_csv(path)
    assert list(out.columns) == [
        "company",
        "address",
        "city",
        "state",
        "zip",
        "notes",
        "lon_raw",
        "lat_raw",
    ]
    assert out.loc[0, "company"] == "Acme"
    assert out.loc[0, "lon_raw"] == "-77.18"


def test_load_registry_missing_company_column_raises(tmp_path: Path) -> None:
    """A registry without the company column fails with an actionable error."""
    path = tmp_path / "registry.csv"
    path.write_text("Address,X,Y\n1 Main St,-77.0,38.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Company"):
        load_registry_csv(path)


def test_load_registry_without_coordinate_columns(tmp_path: Path) -> None:
    """A never-geocoded registry (no X/Y columns) loads with empty coordinates."""
    path = tmp_path / "registry.csv"
    path.write_text("Company,Address\nAcme,1 Main St\n", encoding="utf-8")
    clean, needs = clean_registry(load_registry_csv(path))
    assert clean.empty
    assert list(needs["reason"]) == [REASON_MISSING_COORDS]


# ---------------------------------------------------------------------------
# categorize_notes
# ---------------------------------------------------------------------------


def test_categorize_notes_feeder_shuttle_and_unspecified() -> None:
    """Notes split into transit_feeder / shuttle / unspecified as documented."""
    notes = pd.Series(["Shuttle to Metro", "Guest shuttle", "Other", "", "Rail station link"])
    assert list(categorize_notes(notes)) == [
        CATEGORY_TRANSIT_FEEDER,
        CATEGORY_SHUTTLE,
        CATEGORY_UNSPECIFIED,
        CATEGORY_UNSPECIFIED,
        CATEGORY_TRANSIT_FEEDER,
    ]


# ---------------------------------------------------------------------------
# clean_registry (against the shipped fixture)
# ---------------------------------------------------------------------------


def test_clean_registry_fixture_drops_blanks_and_dedupes() -> None:
    """The sample registry cleans to 8 geolocated rows + 5 worklist rows."""
    clean, needs = _cleaned_fixture()
    assert len(clean) == 8
    assert len(needs) == 5
    # The duplicated Building Co row collapses to one.
    assert (clean["company"] == "Building Co").sum() == 1


def test_clean_registry_trims_whitespace_and_uppercases_state() -> None:
    """Padded names are trimmed and the state code is normalized."""
    clean, _ = _cleaned_fixture()
    row = clean.set_index("company").loc["Riverside Apartments"]
    assert row["state"] == "DC"


def test_clean_registry_routes_bad_coordinates_to_worklist() -> None:
    """Missing and out-of-range coordinates land on the worklist with reasons."""
    _, needs = _cleaned_fixture()
    reasons = needs.set_index("company")["reason"]
    assert reasons["Furniture LLC"] == REASON_MISSING_COORDS
    assert reasons["Warehouse Partners"] == REASON_MISSING_COORDS
    assert reasons["Tech Campus North"] == REASON_INVALID_COORDS  # latitude 138.9593


def test_clean_registry_fixture_categories() -> None:
    """The clean fixture rows carry the expected category mix."""
    clean, _ = _cleaned_fixture()
    counts = clean["category"].value_counts()
    assert counts[CATEGORY_TRANSIT_FEEDER] == 4
    assert counts[CATEGORY_SHUTTLE] == 1
    assert counts[CATEGORY_UNSPECIFIED] == 3


# ---------------------------------------------------------------------------
# run (prep-only mode)
# ---------------------------------------------------------------------------


def test_run_prep_only_writes_registry_worklist_layer_and_runlog(tmp_path: Path) -> None:
    """Without a GTFS folder, run() preps the registry but writes no rollup."""
    result = run(shuttles_csv=FIXTURE_CSV, output_dir=tmp_path)

    assert (tmp_path / "private_shuttles_clean.csv").exists()
    assert (tmp_path / "private_shuttles_needs_geocoding.csv").exists()
    assert (tmp_path / "Private_Shuttle_Stops.zip").exists()
    assert not (tmp_path / "private_shuttle_coverage_by_route.csv").exists()
    assert result.coverage is None

    runlog = (tmp_path / "private_shuttles_runlog.txt").read_text(encoding="utf-8")
    assert "SHUTTLES_CSV" in runlog  # config block captured verbatim


def test_run_poi_layer_matches_coverage_tool_layer_spec(tmp_path: Path) -> None:
    """The emitted zip is discovered and loaded by points_of_interest_coverage_gpd."""
    run(shuttles_csv=FIXTURE_CSV, output_dir=tmp_path)

    sources = _find_layer_sources("Private_Shuttle_Stops.shp", tmp_path)
    assert len(sources) == 1
    assert sources[0].startswith("zip://")

    layers = _load_layers([("Private_Shuttle_Stops.shp", "NAME")], tmp_path)
    layer = layers["Private_Shuttle_Stops.shp"]
    assert len(layer) == 8
    assert "Building Co" in set(layer["NAME"])


def test_run_can_skip_poi_layer(tmp_path: Path) -> None:
    """write_poi_layer=False suppresses the zip but keeps the CSVs."""
    run(shuttles_csv=FIXTURE_CSV, output_dir=tmp_path, write_poi_layer=False)
    assert not (tmp_path / "Private_Shuttle_Stops.zip").exists()
    assert (tmp_path / "private_shuttles_clean.csv").exists()


# ---------------------------------------------------------------------------
# run (with GTFS: route rollup)
# ---------------------------------------------------------------------------


def test_run_with_gtfs_writes_route_keyed_coverage(tmp_path: Path) -> None:
    """The rollup counts shuttle sites and the transit-feeder subset per route."""
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    out_dir = tmp_path / "out"

    result = run(shuttles_csv=FIXTURE_CSV, gtfs_dir=gtfs_dir, output_dir=out_dir)

    out_csv = out_dir / "private_shuttle_coverage_by_route.csv"
    assert out_csv.exists()
    written = pd.read_csv(out_csv, dtype={"route_id": str})
    assert {
        "route_id",
        "route_short_name",
        "shuttle_sites_served",
        "shuttle_feeder_sites_served",
    } <= set(written.columns)

    by_route = written.set_index("route_id")
    assert by_route.loc["R1", "shuttle_sites_served"] == 3
    assert by_route.loc["R1", "shuttle_feeder_sites_served"] == 1
    assert by_route.loc["R2", "shuttle_sites_served"] == 0
    assert by_route.loc["R2", "shuttle_feeder_sites_served"] == 0
    assert result.coverage is not None


def test_run_with_gtfs_and_empty_clean_registry(tmp_path: Path) -> None:
    """A registry with no usable coordinates still yields an all-zero rollup."""
    registry = tmp_path / "registry.csv"
    registry.write_text("Company,Address,X,Y\nAcme,1 Main St,,\n", encoding="utf-8")
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    _write_gtfs_files(gtfs_dir)
    out_dir = tmp_path / "out"

    result = run(shuttles_csv=registry, gtfs_dir=gtfs_dir, output_dir=out_dir)

    assert result.coverage is not None
    assert result.coverage["shuttle_sites_served"].sum() == 0


# ---------------------------------------------------------------------------
# main (placeholder guard)
# ---------------------------------------------------------------------------


def test_main_blocks_unedited_placeholder_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CONFIG untouched and no flags, main() warns and does not run."""
    calls: list[dict] = []
    monkeypatch.setattr(shuttle_mod, "run", lambda **kw: calls.append(kw))
    assert shuttle_mod.main([]) == 2
    assert calls == []


def test_main_runs_after_config_edit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The documented edit-CONFIG-then-run workflow must reach run()."""
    calls: list[dict] = []
    monkeypatch.setattr(shuttle_mod, "run", lambda **kw: calls.append(kw))
    monkeypatch.setattr(shuttle_mod, "SHUTTLES_CSV", tmp_path / "registry.csv")
    assert shuttle_mod.main([]) == 0
    assert len(calls) == 1


def test_main_reports_missing_input_as_error(tmp_path: Path) -> None:
    """A nonexistent registry path exits 1 with an error, not a traceback."""
    assert shuttle_mod.main(["--shuttles-csv", str(tmp_path / "nope.csv")]) == 1


def test_gpd_gtfs_dir_flag_reaches_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--gtfs-dir is forwarded so the rollup can be enabled from the CLI."""
    calls: list[dict] = []
    monkeypatch.setattr(shuttle_mod, "run", lambda **kw: calls.append(kw))
    assert (
        shuttle_mod.main(["--shuttles-csv", str(FIXTURE_CSV), "--gtfs-dir", str(tmp_path / "gtfs")])
        == 0
    )
    assert calls[0]["gtfs_dir"] == tmp_path / "gtfs"
