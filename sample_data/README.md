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
| `time_periods_example.txt` | Time-of-day periods for `ridership_from_tides.py` (`TIME_PERIODS_FILE`). | Moved from `scripts/ridership_tools/` |
| `ntd_monthly_config_example.txt` | Settings template for `ntd_monthly_summary.py`. Its paths are placeholders, and the workbooks it lists aren't included. | Moved from `scripts/ridership_tools/` |
| `disruptions_example.csv` | Starter events file for `disruption_exposure_builder.py`: two real WMATA system-wide events plus two fictional Mock DC Transit events (`EXAMPLE-` ids, source `illustrative`) that show route and stop scope against `gtfs_current.zip`. See [Disruption events](#disruption-events). | Hand-written |

## Archive editions

Scripts that compare feeds read each edition as its own feed, so unzip `gtfs_archive.zip` first. Point `gtfs_route_diff.py` or `gtfs_route_timeline.py` at the edition folders, and `gtfs_service_change_dates.py` at the folder that holds them.

- `dc_2024` → `dc_2025`: service change only. Sunday service starts on routes 10, 30 and 40, and route 30 goes from every 20 to every 15 minutes on weekdays.
- `dc_2025` → `dc_2026`: fare change only. The local fare goes from $1.75 to $2.00, and route 50H moves to a new free fare.
- `dc_2026` is the same feed as `gtfs_current.zip`.

Stops, shapes and routes are the same in every edition, so `gtfs_stop_diff.py` and `gtfs_linear_diff_gpd.py` find no differences between them.

## Seeded errors

Each GTFS feed folder also holds `fixture_manifest.json`, which lists errors planted on purpose for the stop data-quality scripts to find: stop names misspelled from the road they sit on (`stop_vs_roadname_checker_*.py`), and stops left in the roadbed (`stop_v_conflict_checker_*.py`). The road layers those scripts check against are in `tests/fixtures/output_road_shps_dc.zip`. GTFS readers ignore the file.

## Disruption events

`disruptions_example.csv` shows the format of the hand-maintained events file that `scripts/modeling/disruption_exposure_builder.py` turns into monthly exposure columns for the ridership panel. Copy it, delete the `EXAMPLE-` rows, and add your own events.

One row per event and affected target. When one event hits several targets, repeat the `event_id` on each row.

| Column | Notes |
|---|---|
| `event_id` | Shared across rows when one event affects several targets. Becomes the column name (`ev_<event_id>` or `sys_ev_<event_id>`, lowercased, punctuation to `_`). |
| `start_date`, `end_date` | `YYYY-MM-DD`, inclusive. A blank `end_date` means ongoing. |
| `shape` | `step` (a permanent level shift; `end_date` must be blank) or `pulse` (a bounded event, or an ongoing one with a blank `end_date`). |
| `scope` | `system`, `route`, `stop`, or `district`. |
| `target` | `ALL` for system scope; a route label (`route_short_name`, as in the NTD anchor) for route scope; a GTFS `stop_id` for stop scope; a district name from the `stop_id,district` crosswalk for district scope. |
| `category` | Free text, e.g. `wmata`, `flood`, `weather`, `construction`. Rows of one event must share it. `--pivot-by category` makes one column per category. |
| `severity` | Optional number (e.g. a 1–3 scale or snow inches). Carried into the long audit table, not into the exposure value. |
| `source`, `notes` | Provenance. Not used in the calculation. |

Points to keep in mind when adding rows:

- **Exceptional events only.** Put dated, identifiable shocks here. Routine weather is already a monthly covariate from `clean_noaa_weather.py`, and repeating it here would count ordinary seasonality twice against the month effects.
- **No sign.** The exposure value says only how much of a month an event covered. Whether it raised or lowered ridership is left to the model's coefficient (a Metrorail outage could push riders onto buses or keep them home).
- **Step transition month.** By default a step's start month gets the fraction of days remaining; `STEP_TRANSITION` can make it 1 or 0 instead. The run log records the setting used.
- **System events stay system-wide.** They vary only by month, so the builder writes them to a `period`-keyed table rather than copying them to every route. Their effect is identified from timing alone, against anything else that changed system-wide in the same months.

The two real rows were checked against news coverage in November 2026. Silver Line Phase 2 opened November 15, 2022. The Washington Metrorail Safety Commission ordered every 7000-series car out of service by 5 a.m. on October 18, 2021. The first eight trains returned June 16, 2022, and the rest came back in phases after that, so the `2022-06-15` end date marks the first return, not the full one.
