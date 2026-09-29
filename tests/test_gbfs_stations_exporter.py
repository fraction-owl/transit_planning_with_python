from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.gbfs_tools.gbfs_stations_exporter as target

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _station_info(stations: list[dict]) -> dict:
    return {"data": {"stations": stations}}


def _valid_stations() -> list[dict]:
    return [
        {"station_id": "A", "name": "First St", "lat": 38.9, "lon": -77.0},
        {"station_id": "B", "name": "Second St", "lat": 38.8, "lon": -77.1},
    ]


# ---------------------------------------------------------------------------
# _is_url
# ---------------------------------------------------------------------------


def test_is_url_accepts_http_and_https() -> None:
    assert target._is_url("https://example.com/gbfs.json") is True
    assert target._is_url("http://example.com/gbfs.json") is True


def test_is_url_rejects_local_paths() -> None:
    assert target._is_url("/data/gbfs.json") is False
    assert target._is_url(r"C:\data\gbfs.json") is False


# ---------------------------------------------------------------------------
# fetch_json (local files only — no network access in tests)
# ---------------------------------------------------------------------------


def test_fetch_json_reads_local_file(tmp_path: Path) -> None:
    doc = {"data": {"stations": []}}
    path = tmp_path / "station_information.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert target.fetch_json(str(path)) == doc


def test_fetch_json_missing_file_raises_ioerror(tmp_path: Path) -> None:
    with pytest.raises(IOError):
        target.fetch_json(str(tmp_path / "nope.json"))


def test_fetch_json_invalid_json_raises_valueerror(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        target.fetch_json(str(path))


# ---------------------------------------------------------------------------
# resolve_station_information_url
# ---------------------------------------------------------------------------


def test_resolve_direct_station_information_source_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "station_information.json"
    path.write_text(json.dumps(_station_info(_valid_stations())), encoding="utf-8")
    assert target.resolve_station_information_url(str(path)) == str(path)


def test_resolve_station_file_recognized_by_content_not_name(tmp_path: Path) -> None:
    # A station file saved under an ordinary name must not be mistaken for a
    # discovery document.
    path = tmp_path / "stations.json"
    path.write_text(json.dumps(_station_info(_valid_stations())), encoding="utf-8")
    assert target.resolve_station_information_url(str(path)) == str(path)
    assert target.load_station_information(str(path))["data"]["stations"][0]["station_id"] == "A"


def test_load_station_information_follows_discovery(tmp_path: Path) -> None:
    stations_path = tmp_path / "si.json"
    stations_path.write_text(json.dumps(_station_info(_valid_stations())), encoding="utf-8")
    doc = {"data": {"feeds": [{"name": "station_information", "url": str(stations_path)}]}}
    path = tmp_path / "gbfs.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = target.load_station_information(str(path))
    assert len(loaded["data"]["stations"]) == 2


def test_resolve_gbfs_2x_language_nested_feeds(tmp_path: Path) -> None:
    doc = {
        "data": {
            "en": {
                "feeds": [
                    {"name": "system_information", "url": "https://x/sys.json"},
                    {"name": "station_information", "url": "https://x/si.json"},
                ]
            }
        }
    }
    path = tmp_path / "gbfs.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert target.resolve_station_information_url(str(path)) == "https://x/si.json"


def test_resolve_gbfs_3x_flat_feeds(tmp_path: Path) -> None:
    doc = {"data": {"feeds": [{"name": "station_information", "url": "https://x/si.json"}]}}
    path = tmp_path / "gbfs.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert target.resolve_station_information_url(str(path)) == "https://x/si.json"


def test_resolve_discovery_without_station_information_raises(tmp_path: Path) -> None:
    doc = {"data": {"feeds": [{"name": "system_information", "url": "https://x/sys.json"}]}}
    path = tmp_path / "gbfs.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="station_information"):
        target.resolve_station_information_url(str(path))


def test_resolve_non_discovery_document_returned_as_is(tmp_path: Path) -> None:
    path = tmp_path / "something.json"
    path.write_text(json.dumps({"data": []}), encoding="utf-8")
    assert target.resolve_station_information_url(str(path)) == str(path)


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


def test_extract_text_plain_string_passthrough() -> None:
    assert target._extract_text("Union Station") == "Union Station"


def test_extract_text_prefers_configured_language() -> None:
    localized = [
        {"text": "Gare Union", "language": "fr"},
        {"text": "Union Station", "language": "en"},
    ]
    assert target._extract_text(localized) == "Union Station"


def test_extract_text_falls_back_to_first_text() -> None:
    localized = [{"text": "Gare Union", "language": "fr"}]
    assert target._extract_text(localized) == "Gare Union"


def test_extract_text_unusable_values_return_none() -> None:
    assert target._extract_text(None) is None
    assert target._extract_text([]) is None
    assert target._extract_text([{"language": "en"}]) is None


# ---------------------------------------------------------------------------
# build_stations_gdf
# ---------------------------------------------------------------------------


def test_build_stations_gdf_creates_wgs84_points() -> None:
    gdf = target.build_stations_gdf(_station_info(_valid_stations()))
    assert len(gdf) == 2
    assert gdf.crs.to_string() == target.GBFS_CRS
    first = gdf.iloc[0]
    assert first.geometry.x == pytest.approx(-77.0)
    assert first.geometry.y == pytest.approx(38.9)


def test_build_stations_gdf_missing_stations_list_raises() -> None:
    with pytest.raises(ValueError, match="data.stations"):
        target.build_stations_gdf({"data": {}})


def test_build_stations_gdf_empty_station_list_returns_empty_gdf() -> None:
    gdf = target.build_stations_gdf(_station_info([]))
    assert gdf.empty


def test_build_stations_gdf_missing_required_fields_raises() -> None:
    stations = [{"station_id": "A", "name": "No coords"}]
    with pytest.raises(ValueError, match="lat"):
        target.build_stations_gdf(_station_info(stations))


def test_build_stations_gdf_drops_invalid_coordinates() -> None:
    stations = _valid_stations()
    stations.append({"station_id": "C", "name": "Bad", "lat": "oops", "lon": -77.2})
    gdf = target.build_stations_gdf(_station_info(stations))
    assert len(gdf) == 2
    assert "C" not in set(gdf["station_id"])


def test_build_stations_gdf_drops_out_of_range_coordinates() -> None:
    stations = _valid_stations()
    stations.append({"station_id": "C", "name": "Off map", "lat": 138.9, "lon": -77.0})
    stations.append({"station_id": "D", "name": "Off map", "lat": 38.9, "lon": -277.0})
    gdf = target.build_stations_gdf(_station_info(stations))
    assert set(gdf["station_id"]) == {"A", "B"}


def test_build_stations_gdf_keeps_localized_short_name() -> None:
    # GBFS 3.x localizes short_name; it must survive as a flat string column
    # because it is the Capital Bikeshare trip-history station number.
    stations = [
        {
            "station_id": "08263b1f",
            "name": [{"text": "Union Station", "language": "en"}],
            "short_name": [{"text": "31623", "language": "en"}],
            "lat": 38.9,
            "lon": -77.0,
        }
    ]
    gdf = target.build_stations_gdf(_station_info(stations))
    assert gdf["short_name"].iloc[0] == "31623"


def test_build_stations_gdf_numeric_short_name_stays_text() -> None:
    stations = _valid_stations()
    stations[0]["short_name"] = 31000
    gdf = target.build_stations_gdf(_station_info(stations))
    assert gdf["short_name"].iloc[0] == "31000"


def test_build_stations_gdf_normalises_localized_names() -> None:
    stations = [
        {
            "station_id": "A",
            "name": [{"text": "Union Station", "language": "en"}],
            "lat": 38.9,
            "lon": -77.0,
        }
    ]
    gdf = target.build_stations_gdf(_station_info(stations))
    assert gdf["name"].iloc[0] == "Union Station"


def test_build_stations_gdf_drops_nested_columns() -> None:
    stations = [
        {
            "station_id": "A",
            "name": "First St",
            "lat": 38.9,
            "lon": -77.0,
            "rental_methods": ["KEY", "CREDITCARD"],
        }
    ]
    gdf = target.build_stations_gdf(_station_info(stations))
    assert "rental_methods" not in gdf.columns


# ---------------------------------------------------------------------------
# export_geojson / export_shapefile
# ---------------------------------------------------------------------------


def test_export_geojson_writes_file(tmp_path: Path) -> None:
    gdf = target.build_stations_gdf(_station_info(_valid_stations()))
    out = tmp_path / "gbfs_stations.geojson"
    target.export_geojson(gdf, out)
    assert out.exists()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["features"]) == 2


def test_export_geojson_empty_gdf_writes_nothing(tmp_path: Path) -> None:
    gdf = target.build_stations_gdf(_station_info([]))
    out = tmp_path / "gbfs_stations.geojson"
    target.export_geojson(gdf, out)
    assert not out.exists()


def test_export_shapefile_writes_file(tmp_path: Path) -> None:
    gdf = target.build_stations_gdf(_station_info(_valid_stations()))
    out = tmp_path / "gbfs_stations.shp"
    target.export_shapefile(gdf, out)
    assert out.exists()
