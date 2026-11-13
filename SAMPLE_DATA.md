# Sample data

> **Draft.** Where sample data should live is still open; see issue #326.

Every sample input in this repository lives in [`tests/fixtures/`](tests/fixtures/). The files exist so the tests can run, but most of them also work as sample inputs: run a script on the sample first to see what it produces, then swap in your own data.

**Point the script at your file; don't overwrite the sample.** The tests check the contents of these files, so replacing one breaks them. It also leaves you with local changes that conflict when you `git pull`. Change the path setting in the script's CONFIGURATION block instead, or pass the matching command-line flag where the script has one.

Most scripts accept a GTFS feed as either the `.zip` or the unzipped folder, and a shapefile as either the `.shp` or a `.zip`. `_arcpy` scripts need shapefiles unzipped. Some older scripts still accept only a folder or a `.shp`; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Checklist: your agency's data

The samples below describe a fictional DC-area agency. Most were written by the `dev_tools/generate_mock_*.py` scripts and line up with each other: stop IDs, trip IDs and routes match across files.

### Core

Start here. GTFS, TODS and TIDES are published data standards, and many scripts read one or more of them.

- [ ] **GTFS feed.** Sample: `tests/fixtures/mock_gtfs_dc.zip` (five routes, from `dev_tools/generate_mock_gtfs.py`). Yours: your agency's published GTFS `.zip`, or an export from your scheduling system. Read by any script with a `GTFS_*` path setting (58 scripts).
- [ ] **TODS supplement.** Sample: `tests/fixtures/mock_tods_dc.zip` (`*_supplement.txt` files for the mock feed). Yours: TODS files for your GTFS feed. Read by `printable_block_schedules.py` and `scheduled_service_hours.py`.
- [ ] **TIDES stop visits and trips performed.** Sample: `tests/fixtures/stop_visits.csv` and `tests/fixtures/trips_performed.csv` (from `dev_tools/generate_mock_tides.py`, matching the mock feed). Yours: TIDES tables from your AVL/APC data; `scripts/conversion_tools/convert_to_tides_stop_visits.py` and `convert_to_tides_trips_performed.py` build them from CLEVER exports. Read by `ridership_from_tides.py`, `ridership_by_hour.py`, `ridership_by_time_period.py`, `max_load_from_tides.py`, `otp_by_route.py`, `otp_by_stop.py`, `otp_by_trip_and_hour.py`, `otp_by_timepoint_from_avl.py`, `otp_monthly_panel.py`, `runtime_by_route.py`, `runtime_by_segment.py`, `runtime_by_trip.py`, `trip_runtime_diagnostics_from_avl.py` and `prep_features_private.py`.

### Optional: exports from your ridership system

Each of these is read by only the scripts named. These samples are hand-made, and their route and stop IDs don't match the mock agency's.

- [ ] **Ridership by route and stop.** Sample: `tests/fixtures/ridership_by_route_and_stop.csv`. Yours: your ridership system's route-and-stop export, with the same column names. Read by `data_request_by_stop_processor.py` and `ntd_monthly_summary.py`.
- [ ] **Max load by trip.** Sample: `tests/fixtures/statistics_by_route_and_trip.csv`. Yours: your ridership system's statistics-by-route-and-trip export. Read by `load_factor_monitor.py`.
- [ ] **Stop usage by stop ID.** Sample: `tests/fixtures/stop_usage_by_stop_id_sample.csv`. Yours: your ridership system's stop-usage export. Read by `flag_stop_upgrades.py`.
- [ ] **Monthly NTD ridership.** Sample: `tests/fixtures/ntd_monthly_multi_month.csv`. Yours: monthly NTD ridership workbooks with `ROUTE_NAME`, `SERVICE_PERIOD`, `MTH_BOARD` and `DAYS` columns. Read by `ntd_monthly_trends_export.py`.

### Optional: other agency data

Each of these is read by only the scripts named.

- [ ] **Past editions of your GTFS feed.** Sample: `tests/fixtures/mock_gtfs_dc_archive.zip` (three yearly editions of the mock feed, from `dev_tools/generate_mock_gtfs.py`). Yours: archived copies of earlier feeds. Read by `gtfs_route_diff.py` and `gtfs_service_change_dates.py`.
- [ ] **Road centerlines.** Sample: `dc_road_centerlines.shp` inside `tests/fixtures/output_road_shps_dc.zip` (mock DC, matching `dev_tools/generate_mock_roads.py` output). Yours: your jurisdiction's road centerline layer, with the road-name fields `FULLNAME`, `RW_PREFIX`, `RW_TYPE_US`, `RW_SUFFIX` and `RW_SUFFIX_`. Read by `stop_vs_roadname_checker_gpd.py` and `stop_vs_roadname_checker_arcpy.py`.
- [ ] **Road surface polygons.** Sample: `dc_road_asphalt.shp` inside the same zip. Yours: edge-of-pavement polygons; centerlines won't work, because a stop point almost never lies exactly on a line. Read by `stop_v_conflict_checker_gpd.py`, which can also check driveway and building-footprint polygons (no samples for those), and, optionally, by `walking_network_qa_gpd.py` and `walking_network_qa_arcpy.py`.
- [ ] **Downtown (CBD) boundary.** Sample: `tests/fixtures/mock_cbd_dc.zip` (from `dev_tools/generate_mock_cbd.py`). Yours: your CBD polygon. Read by `gtfs_route_features_gpd.py`.
- [ ] **Transit center outlines, for bay planning.** Sample: `tests/fixtures/stop_clusters_sample.zip`. Yours: one polygon around each transit center or bus station whose bays you want to plan. Read by `cluster_stops_from_zones_gpd.py`, which lists the GTFS stops in each outline for `block_status_timeline_exporter.py`, `bay_usage_analyzer.py` and `bay_change_sweep.py`.
- [ ] **Points of interest.** Sample: `tests/fixtures/points_of_interest/`, ten zipped point layers (hospitals, libraries, schools, park-and-rides and so on) from `dev_tools/generate_mock_points_of_interest.py`. Yours: local open-data layers for each category. Read by `points_of_interest_coverage_gpd.py`, `points_of_interest_coverage_arcpy.py` and `private_shuttle_coverage_by_route_gpd.py`.
- [ ] **Private shuttle registry.** Sample: `tests/fixtures/private_shuttles_sample.csv`. Yours: your hand-maintained operator spreadsheet. Read by `private_shuttle_coverage_by_route_gpd.py`.
- [ ] **Express routes and their origin stops.** Sample: `tests/fixtures/express_routes.txt` and `tests/fixtures/express_origin_stops.txt`. Yours: lists you write, one ID per line, `#` for comments. Read by `gtfs_service_demographics_gpd.py`.
- [ ] **Large employers missing from LODES.** Sample: `tests/fixtures/supplemental_jobs.csv`. Yours: a list you compile. Read by `uscensus_tiger_join_gpd.py`.

## Checklist: public data you download (all optional)

These samples are trimmed copies or mock versions of public files, cut down to a few DC, Maryland and Virginia geographies.

- [ ] **Census tables.** Sample: `tests/fixtures/ACSST5Y2024.S0801_*.zip` and `tests/fixtures/ACSDT5Y2024.C16001_*.zip`. Yours: table downloads from [data.census.gov](https://data.census.gov/); `dev_tools/build_uscensus_fixtures.py` builds a trimmed set like this one from full downloads. Read by `uscensus_table_build.py` and `uscensus_tiger_join_gpd.py`.
- [ ] **Census blocks.** Sample: `tests/fixtures/tl_2025_11_tabblock20_sample.zip` (DC), `tl_2025_24_…` (MD) and `tl_2025_51_…` (VA); the blocks picked are listed in `sample_block_geoids.csv`. Yours: TIGER/Line `tabblock20` shapefiles for your states from census.gov. Read by `uscensus_tiger_join_gpd.py`.
- [ ] **Schools.** Sample: `tests/fixtures/EDGE_GEOCODE_*_1920_sample.zip`, `ccd_sch_052_1920_sample.zip`, `ELSI_csv_export_*_1920_sample.csv` and `effy2019_sample.csv` (from `dev_tools/build_schools_fixtures.py`). Yours: NCES EDGE school locations, CCD school membership, ELSI table-generator exports and IPEDS 12-month enrollment. Read by `schools_prep_join_gpd.py`.
- [ ] **Unemployment rate.** Sample: `tests/fixtures/UNRATE_sample.csv`. Yours: a [FRED](https://fred.stlouisfed.org/series/UNRATE) CSV export. Read by `clean_fred_unrate.py`.
- [ ] **Gas prices.** Sample: `tests/fixtures/PET_PRI_GND_DCUS_NUS_W_Data_1__sample.csv`. Yours: the [EIA weekly retail price](https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm) "Data 1" CSV. Read by `clean_eia_gas_prices.py`.
- [ ] **Weather.** Sample: `tests/fixtures/4331222_sample.csv`. Yours: a NOAA [Climate Data Online](https://www.ncei.noaa.gov/cdo-web/search) daily-summaries export. Read by `clean_noaa_weather.py`.
- [ ] **Bikeshare trips.** Sample: `tests/fixtures/capitalbikeshare_fixtures_24mo.zip`, 24 mock monthly files from `dev_tools/generate_mock_bikeshare_trips.py`. Yours: Capital Bikeshare's monthly `YYYYMM-capitalbikeshare-tripdata.csv` files. Read by `bikeshare_ridership_trends.py`.

## Test-only fixtures

These are small, hand-built cases for specific behavior. They aren't meant as samples.

| Fixture | What it tests |
|---|---|
| `tests/fixtures/gtfs_basic/` | A three-route feed for "Your Transit Agency", read by 14 test files. Issue #326 asks whether it should also count as a sample. |
| `tests/fixtures/gtfs_holiday_negation/` | A `calendar.txt` service whose every date is removed in `calendar_dates.txt`. |
| `tests/fixtures/gtfs_split_weekday/` | Weekday service split across several `service_id`s (Monday, Tuesday–Thursday, Friday). |
| `tests/fixtures/gtfs_route_diff_replacement_day/` | Before and after feeds where a one-day replacement service swaps in for the weekday service. |
| `tests/fixtures/stop_ids.txt` | Parsing of `data_request_by_stop_processor.py`'s optional `STOP_IDS_FILE` (comments, blank lines, mixed commas and spaces). That setting is a per-request filter, an alternative to listing IDs inline in `STOP_IDS`, not a standing input. |

## Not read by any test

No test reads these by name or glob. Some tests use the same filenames, but only for synthetic files they write to a temporary folder. Issue #331 proposes a test for the Census and LODES files; issue #326 asks what to do with the rest.

- `tests/fixtures/ACSDT5Y2024.B01001_*.zip`, `ACSDT5Y2024.B08201_*.zip`, `ACSDT5Y2024.B19001_*.zip`
- `tests/fixtures/DECENNIALDHC2020.H1_*.zip`, `DECENNIALDHC2020.P9_*.zip`, `DECENNIALPL2020.P1_*.zip`
- `tests/fixtures/dc_wac_S000_JT00_2023.csv.gz`, `md_wac_…`, `va_wac_…`
- `tests/fixtures/CLEVER_Runtime_and_OTP_by_Month_sample.csv`
- `tests/fixtures/sample_block_geoids.csv`: read only by `dev_tools/build_uscensus_fixtures.py`, in its replay mode.
