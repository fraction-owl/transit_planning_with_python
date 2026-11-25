# Sample data

Sample inputs for trying a script before you point it at your own data. They describe Mock DC Transit, an invented five-route agency, and their IDs match across files: TIDES trips and stops resolve against the GTFS feed and the TODS supplements.

The GTFS, TODS and TIDES files are copies of fixtures in [`tests/fixtures/`](../tests/fixtures/), written by `dev_tools/generate_mock_gtfs.py` and `dev_tools/generate_mock_tides.py`. `tests/test_sample_data.py` fails if a copy differs from its original, so after regenerating a fixture, copy it here again.

**Point the script at your file; don't overwrite the sample.** Change the path setting in the script's CONFIGURATION block, or pass the matching command-line flag where the script has one. Overwriting a sample fails that test and leaves you with local changes that conflict when you `git pull`.

| File | Contents | Source |
|---|---|---|
| `gtfs_current.zip` | GTFS feed for 2026: route 10 (local), 20 (peak-only express), 30 (crosstown), 40 (diagonal, close stop spacing) and 50H (holiday-only loop). | Copy of `tests/fixtures/mock_gtfs_dc.zip` |
| `gtfs_archive.zip` | Three yearly editions of that feed, in folders `dc_2024/`, `dc_2025/` and `dc_2026/`. See [Archive editions](#archive-editions). | Copy of `tests/fixtures/mock_gtfs_dc_archive.zip` |
| `tods_current.zip` | TODS 2.1 supplements for `gtfs_current.zip`: garages, pull-outs, pull-backs and deadheads between trips. | Copy of `tests/fixtures/mock_tods_dc.zip` |
| `tides/stop_visits.csv`, `tides/trips_performed.csv` | TIDES tables for a sample of service dates from January to March 2026 on routes 10, 20 and 30, plus the 50H loop on January 1. Pull-outs, pull-ins and deadheads carry the TODS trip IDs. | Copies of `tests/fixtures/stop_visits.csv` and `tests/fixtures/trips_performed.csv` |
| `time_periods_example.txt` | Time-of-day periods for `ridership_from_tides.py` and `otp_runtime_from_tides.py` (`TIME_PERIODS_FILE`). | Moved from `scripts/ridership_tools/` |
| `county_fips_example.txt` | County FIPS list for the `uscensus_*` scripts (`COUNTY_FIPS_FILE` / `FIPS_FILE`). The Washington, DC area counties are commented out, so a script pointed at an unedited copy stops with a message instead of filtering to DC by mistake. | Written by hand; codes from the [FCC county FIPS list](https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt) |
| `disruptions_example.csv` | Dated service disruptions: two real WMATA system-wide events and two fictional Mock DC Transit events (`EXAMPLE-` ids, source `illustrative`) on route 30 and a stop in `gtfs_current.zip`. No script reads it yet. See [Disruption events](#disruption-events). | Copy of `tests/fixtures/disruptions_example.csv`; written by hand |

## Disruption events

`disruptions_example.csv` is a format for a hand-maintained list of dated disruptions, kept for future ridership modeling. It has one row per event and affected target; when one event hits several targets, repeat its `event_id` on each row.

| Column | Notes |
|---|---|
| `event_id` | Shared across rows when one event affects several targets. |
| `start_date`, `end_date` | `YYYY-MM-DD`, inclusive. A blank `end_date` means ongoing. |
| `shape` | `step` (a permanent level shift, with a blank `end_date`) or `pulse` (a bounded event). |
| `scope` | `system`, `route`, `stop`, or `district`. |
| `target` | `ALL` for system scope, a `route_short_name` for route scope, a GTFS `stop_id` for stop scope, or a district name for district scope. |
| `category` | Free text, e.g. `wmata`, `flood`, `weather`, `construction`. |
| `severity` | Optional number, e.g. a 1–3 scale or inches of snow. |
| `source`, `notes` | Where the dates came from. |

Only dated, one-off events belong here. Holidays come from the GTFS calendar, and routine weather is a monthly covariate from `clean_noaa_weather.py`.

The two real rows were checked against news coverage in November 2026. Silver Line Phase 2 opened November 15, 2022. The Washington Metrorail Safety Commission ordered every 7000-series car out of service by 5 a.m. on October 18, 2021. The first eight trains returned June 16, 2022, and the rest came back in phases after that, so the `2022-06-15` end date marks the first return, not the full one.

## Route lists

These optional examples are lists of Mock DC Transit routes. `express_routes_example.txt` lists route 20 by its `gtfs_current.zip` route ID, `DC_R20`. In `gtfs_service_demographics_gpd.py`, point `EXPRESS_ROUTES_FILE` (or `--express-routes-file`) at it to label route 20 "express" in route mode; a relative path is read from the folder you run the script in. The non-revenue and event examples use invented route IDs that `gtfs_current.zip` does not contain. In `trip_cut_savings.py`, point `NON_REVENUE_ROUTES` at `non_revenue_routes_example.txt` to remove test routes from baseline totals and cut screening. Point `OTHER_EXCLUDED_ROUTES` at `event_routes_example.txt` when event shuttles should remain in passenger-service totals and be excluded from cut proposals. Both settings accept absolute paths or paths relative to the script. Scripts can read these files with `load_express_route_ids`, `load_non_revenue_route_ids` or `load_event_route_ids` from `utils/gtfs_helpers.py`.

## Archive editions

Scripts that compare feeds read each edition as its own feed, so unzip `gtfs_archive.zip` first. Point `gtfs_route_diff.py` or `gtfs_route_timeline.py` at the edition folders, and `gtfs_service_change_dates.py` at the folder that holds them.

- `dc_2024` → `dc_2025`: service change only. Sunday service starts on routes 10, 30 and 40, and route 30 goes from every 20 to every 15 minutes on weekdays.
- `dc_2025` → `dc_2026`: fare change only. The local fare goes from $1.75 to $2.00, and route 50H moves to a new free fare.
- `dc_2026` is the same feed as `gtfs_current.zip`.

Stops, shapes and routes are the same in every edition, so `gtfs_stop_diff.py` and `gtfs_linear_diff_gpd.py` find no differences between them.

## Seeded errors

Each GTFS feed folder also holds `fixture_manifest.json`, which lists errors planted on purpose for the stop data-quality scripts to find: stop names misspelled from the road they sit on (`stop_vs_roadname_checker_*.py`), and stops left in the roadbed (`stop_v_conflict_checker_*.py`). The road layers those scripts check against are in `tests/fixtures/output_road_shps_dc.zip`. GTFS readers ignore the file.
