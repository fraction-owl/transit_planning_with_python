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
| `ntd_monthly_config_example.txt` | Settings template for `ntd_monthly_summary.py`. Its paths are placeholders, and the workbooks it lists aren't included. | Moved from `scripts/ridership_tools/` |

## Archive editions

Scripts that compare feeds read each edition as its own feed, so unzip `gtfs_archive.zip` first. Point `gtfs_route_diff.py` or `gtfs_route_timeline.py` at the edition folders, and `gtfs_service_change_dates.py` at the folder that holds them.

- `dc_2024` → `dc_2025`: service change only. Sunday service starts on routes 10, 30 and 40, and route 30 goes from every 20 to every 15 minutes on weekdays.
- `dc_2025` → `dc_2026`: fare change only. The local fare goes from $1.75 to $2.00, and route 50H moves to a new free fare.
- `dc_2026` is the same feed as `gtfs_current.zip`.

Stops, shapes and routes are the same in every edition, so `gtfs_stop_diff.py` and `gtfs_linear_diff_gpd.py` find no differences between them.

## Seeded errors

Each GTFS feed folder also holds `fixture_manifest.json`, which lists errors planted on purpose for the stop data-quality scripts to find: stop names misspelled from the road they sit on (`stop_vs_roadname_checker_*.py`), and stops left in the roadbed (`stop_v_conflict_checker_*.py`). The road layers those scripts check against are in `tests/fixtures/output_road_shps_dc.zip`. GTFS readers ignore the file.
