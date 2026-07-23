# Which Script Do I Need?

<!-- TODO: 2-3 sentence intro. Audience: planners/analysts who are NOT programmers.
     Promise: "Find your task below, and this page tells you which script(s) to use,
     what data you'll need, and what you'll get out." -->

> **New to Python or this repository?** See the [README](README.md) for how to
> install Python and run a script. This page only helps you pick the *right*
> script — you don't need to understand any code to use it.

**How to use this page:** Find the question that sounds like yours in the table
of contents below. Each section lists the scripts for that task, what data they
need, and what they produce.

<!-- TODO: note on arcpy vs. gpd — many spatial tools come in two flavors:
     `*_arcpy.py` (run inside ArcGIS Pro) and `*_gpd.py` (open-source Python).
     They do the same thing; pick the one that matches your setup. -->

---

## Table of Contents

1. [I'm planning a service change — what's impacted?](#1-im-planning-a-service-change--whats-impacted)
2. [Who does our service reach? (demographics, equity / Title VI)](#2-who-does-our-service-reach-demographics-equity--title-vi)
3. [What destinations does our service connect to? (points of interest, schools, sites)](#3-what-destinations-does-our-service-connect-to)
4. [How is our ridership doing?](#4-how-is-our-ridership-doing)
5. [Are our buses on time? Where are they slow?](#5-are-our-buses-on-time-where-are-they-slow)
6. [Is our GTFS feed correct? (data quality checks)](#6-is-our-gtfs-feed-correct-data-quality-checks)
7. [I need a map, spreadsheet, or schedule from our GTFS](#7-i-need-a-map-spreadsheet-or-schedule-from-our-gtfs)
8. [Bus stop amenities and facilities](#8-bus-stop-amenities-and-facilities)
9. [Bikeshare (GBFS) analysis](#9-bikeshare-gbfs-analysis)
10. [Advanced: ridership modeling](#10-advanced-ridership-modeling)
11. [Helper scripts: preparing Census and other national datasets](#11-helper-scripts-preparing-census-and-other-national-datasets)

---

## 1. I'm planning a service change — what's impacted?

<!-- TODO: scenario framing. E.g. "You have a current GTFS feed and a proposed one
     (or a list of stops/routes being changed) and need to answer: what changes,
     who's affected, and what do we tell the public and the FTA?" -->

**Folder:** `scripts/stop_analysis/`

| If you want to… | Use this script | You'll need |
|---|---|---|
| Compare two GTFS feeds and see which **routes** were added, eliminated, renumbered, or changed | `stop_analysis/gtfs_route_diff.py` | Before + after GTFS feeds |
| Compare two GTFS feeds and see which **stops** were added, removed, or moved | `stop_analysis/gtfs_stop_diff.py` | Before + after GTFS feeds |
| Map **where on the street** a change adds, keeps, or removes coverage | `stop_analysis/gtfs_linear_diff_gpd.py` | Before + after GTFS feeds |
| See which stops (and how many riders) are affected by changes to **specific routes** | `stop_analysis/stop_impacts_target_routes.py` | GTFS feed + list of target routes <!-- TODO: verify inputs --> |
| Estimate **walk-access impacts** of removing specific stops (who loses easy access) | `stop_analysis/stop_removal_impact_gpd.py` | GTFS feed + sidewalk/street network data |

**Related:**
- Checking whether stop spacing is reasonable on a changed route → see [Section 6](#6-is-our-gtfs-feed-correct-data-quality-checks) (`stop_spacing_flagger_*`).
- Demographics of who the change affects → see [Section 2](#2-who-does-our-service-reach-demographics-equity--title-vi).
- A before/after picture of routes over many years → `gtfs_exports/gtfs_route_timeline.py` ([Section 7](#7-i-need-a-map-spreadsheet-or-schedule-from-our-gtfs)).

<!-- TODO: maybe add a short "typical workflow" walkthrough: run route diff first,
     then stop diff, then impacts on the routes that changed. -->

---

## 2. Who does our service reach? (demographics, equity / Title VI)

<!-- TODO: scenario framing. E.g. "Title VI analysis, service equity reporting,
     or just understanding the population within walking distance of service —
     system-wide or route by route." -->

**Folder:** `scripts/service_coverage/` (with data prep in `scripts/national_data_tools/`)

| If you want to… | Use this script | You'll need |
|---|---|---|
| Population/demographics within walking distance of stops — **system-wide or by route** | `service_coverage/gtfs_service_demographics_gpd.py` (or `_arcpy`) | GTFS feed + Census demographic shapefile |
| A matrix of **which routes serve which districts** (wards, counties, council districts…) | `service_coverage/gtfs_service_by_district_gpd.py` (or `_arcpy`) | GTFS feed + district boundaries |
| Each district's **share of ridership** | `ridership_tools/district_ridership_share_gpd.py` (or `_arcpy`) | Stop-level ridership + district boundaries |

**Before you start:** the demographic inputs usually come from the Census. The
scripts in [Section 11](#11-helper-scripts-preparing-census-and-other-national-datasets)
download-and-prep those tables into the shape these tools expect.

<!-- TODO: clarify which demographic fields are expected / how buffers are configured -->

---

## 3. What destinations does our service connect to?

<!-- TODO: scenario framing. E.g. "A developer, school district, or elected official
     asks: 'which routes serve this site?' Or you want a system-wide inventory of
     the key destinations each route connects to." -->

**Folder:** `scripts/service_coverage/`

| If you want to… | Use this script | You'll need |
|---|---|---|
| Which routes/stops are near **a specific site or list of sites** (site planning, developer requests) | `service_coverage/site_route_proximity_gpd.py` (or `_arcpy`) | GTFS feed + site locations (points/parcels) |
| Coverage of **points of interest** (hospitals, grocery stores, major employers…) — system-wide or by route | `service_coverage/points_of_interest_coverage_gpd.py` (or `_arcpy`) | GTFS feed + POI layer |
| **Schools** (and enrollment) served by each route | `service_coverage/school_coverage_by_route_gpd.py` (or `_arcpy`) | GTFS feed + school locations (prep: `national_data_tools/schools_prep_join_gpd.py`) |
| Where riders can **transfer** from each route, and to what | `gtfs_exports/route_transfer_calculator.py` | One or more GTFS feeds |
| **Bikeshare stations** near each route | `service_coverage/cabi_coverage_by_route_gpd.py` | GTFS feed + bikeshare station/trip data |

---

## 4. How is our ridership doing?

<!-- TODO: scenario framing. E.g. "Monthly reporting, board slides, spotting routes
     that are growing or shrinking, mapping ridership by stop." -->

**Folder:** `scripts/ridership_tools/`

| If you want to… | Use this script | You'll need |
|---|---|---|
| Summarize **NTD monthly ridership** (system and route trends) | `ridership_tools/ntd_monthly_summary.py` | NTD-format monthly workbook |
| Trend charts/tables for **selected routes** | `ridership_tools/ntd_monthly_trends_export.py` | NTD-format monthly workbook |
| Put ridership numbers **on a map** (join to stop locations) | `ridership_tools/stops_ridership_joiner_gpd.py` (or `_arcpy`) | Stop-level ridership + GTFS stops or stop shapefile |
| Respond to a **data request for ridership by stop** | `ridership_tools/data_request_by_stop_processor.py` | Stop-level ridership export (Excel) |
| Flag **overcrowded trips** (load factor) | `ridership_tools/load_factor_monitor.py` | Ridecheck/APC output |
| Ridership by **district** | `ridership_tools/district_ridership_share_gpd.py` (or `_arcpy`) | Stop-level ridership + district boundaries |

**If your ridership data lives in TIDES / AVL-APC stop events** (no vendor ridecheck software):

| If you want to… | Use this script |
|---|---|
| Boardings/alightings by route and time period | `ridership_tools/ridership_from_tides.py` |
| Peak (maximum) load per trip | `ridership_tools/max_load_from_tides.py` |

<!-- TODO: brief plain-language note on what TIDES is and the converters in
     operations_tools/ (convert_to_tides_*) for getting AVL exports into TIDES shape. -->

---

## 5. Are our buses on time? Where are they slow?

<!-- TODO: scenario framing. E.g. "Riders complain a route is always late; ops wants
     to know where to add running time; you need OTP numbers for a board report." -->

**Folder:** `scripts/operations_tools/`

| If you want to… | Use this script | You'll need |
|---|---|---|
| **On-time performance by route**, month over month | `operations_tools/otp_monthly_panel.py` | TIDES `stop_visits` data |
| OTP at **specific stops** (find problem stops that hurt multiple routes) | `operations_tools/otp_by_stop.py` | TIDES `stop_visits` data |
| OTP by **timepoint** from an AVL export | `operations_tools/otp_by_timepoint_from_avl.py` | AVL OTP-by-timepoint export |
| **Running time by trip** (which trips need schedule adjustments) | `operations_tools/runtime_by_trip.py` | TIDES `stop_visits` data |
| Running time by **segment** (where along the route time is lost) | `operations_tools/runtime_by_segment.py` | TIDES `stop_visits` data |
| Trip-level **runtime diagnostics** from a raw AVL export | `operations_tools/trip_runtime_diagnostics_from_avl.py` | AVL runtime export |
| **Scheduled** speeds by segment and time of day (from GTFS, no AVL needed) | `gtfs_exports/segment_speed_exporter.py` | GTFS feed only |

**Getting your data into TIDES format:** if your AVL vendor exports raw CSVs,
`operations_tools/convert_to_tides_stop_visits.py` and
`operations_tools/convert_to_tides_trips_performed.py` convert them into the
standard format the tools above expect.

<!-- TODO: note that otp_by_route.py / runtime_by_route.py are mainly feeder steps
     for the modeling pipeline (Section 10), not standalone reports -->

---

## 6. Is our GTFS feed correct? (data quality checks)

<!-- TODO: scenario framing. E.g. "Before a pick or a feed publication, catch stop
     placement errors, name typos, and calendar surprises before riders do." -->

**Folder:** `scripts/gtfs_data_quality/`

| If you want to… | Use this script | You'll need |
|---|---|---|
| A plain-language summary of the feed's **service calendar** (what runs when) | `gtfs_data_quality/gtfs_calendar_inspector.py` | GTFS feed |
| Find stops placed **on the wrong side of the road / conflicting with road geometry** | `gtfs_data_quality/stop_v_conflict_checker_gpd.py` (or `_arcpy`) | GTFS feed + road centerlines |
| Find likely **typos in stop names** (vs. nearby street names) | `gtfs_data_quality/stop_vs_roadname_checker_gpd.py` (or `_arcpy`) | GTFS feed + road centerlines |
| Find **stops too close together / too far apart** | `stop_analysis/stop_spacing_flagger_gpd.py` (or `_arcpy`) | GTFS feed |
| Find stops that are suspiciously **close to another stop** (duplicates) | `gtfs_data_quality/gtfs_stop_proximity_qc.py` | GTFS feed |
| Find trips that **skip stops** their pattern normally serves | `gtfs_data_quality/gtfs_skipped_stop_flagger_gpd.py` | GTFS feed |
| Check that route **direction labels** are consistent | `gtfs_data_quality/route_direction_classifier.py` | GTFS feed |

---

## 7. I need a map, spreadsheet, or schedule from our GTFS

<!-- TODO: scenario framing. E.g. "Turn the GTFS feed into things humans actually
     look at: shapefiles for maps, Excel schedules, printable field documents." -->

**Folders:** `scripts/gtfs_exports/`, `scripts/field_tools/`

**Maps / GIS:**

| If you want to… | Use this script |
|---|---|
| Turn GTFS stops and routes into **shapefiles** for mapping | `gtfs_exports/gtfs_to_shapefile_gpd.py` (or `_arcpy`) |

**Spreadsheets & schedules:**

| If you want to… | Use this script |
|---|---|
| A **one-row-per-route desk reference** (span, headways, stats) | `field_tools/gtfs_route_summary.py` |
| **Headway and span of service** by route | `gtfs_exports/headway_span_exporter.py` |
| Public-timetable-style **schedules by timepoint** in Excel | `gtfs_exports/timepoint_schedule_exporter.py` |
| Every unique **stop pattern** per route | `gtfs_exports/stop_pattern_exporter.py` |
| Trips summarized into **time bands** (peak/off-peak…) | `gtfs_exports/time_band_exporter.py` |
| **Printable schedules for each vehicle block** (for operators/field staff) | `field_tools/printable_block_schedules.py` |
| Minute-by-minute **vehicle block** schedules | `gtfs_exports/bus_block_exporter.py` |
| Minute-by-minute **block status timelines** (in service, deadhead, layover) | `gtfs_exports/block_status_timeline_exporter.py` |
| A multi-year **timeline of when each route existed** (across many feeds) | `gtfs_exports/gtfs_route_timeline.py` |

---

## 8. Bus stop amenities and facilities

<!-- TODO: scenario framing. E.g. "Prioritizing shelter/bench investments; checking
     whether a transit center's bays can handle the schedule." -->

**Folder:** `scripts/facilities_tools/`

| If you want to… | Use this script | You'll need |
|---|---|---|
| Flag stops that **warrant amenity upgrades** (shelters, benches…) based on ridership | `facilities_tools/flag_stop_upgrades.py` | Stop ridership + amenity inventory |
| A **system-wide summary** of stop improvement coverage | `facilities_tools/stop_improvement_coverage_summary.py` | Amenity inventory <!-- TODO: verify inputs --> |
| Detect **bay/berth scheduling conflicts** at transit centers | `facilities_tools/bay_usage_analyzer.py` | GTFS feed <!-- TODO: verify inputs --> |

---

## 9. Bikeshare (GBFS) analysis

<!-- TODO: scenario framing — first/last-mile analysis, bikeshare-transit integration.
     Note: ridership tools are currently built around Capital Bikeshare's trip export format. -->

**Folder:** `scripts/gbfs_tools/`

| If you want to… | Use this script |
|---|---|
| Export bikeshare **station locations** to shapefile/GeoJSON | `gbfs_tools/gbfs_stations_exporter.py` |
| Bikeshare **ridership trends** over time | `gbfs_tools/bikeshare_ridership_trends.py` |
| Join bikeshare **ridership onto station locations** for mapping | `gbfs_tools/gbfs_ridership_join.py` |
| Bikeshare stations **near each transit route** | `service_coverage/cabi_coverage_by_route_gpd.py` |

---

## 10. Advanced: ridership modeling

<!-- TODO: keep this section short — one paragraph. Point technical users to the
     detailed writeup in the README ("Demand Modeling" section) and make clear this
     is the one part of the repo that assumes statistics/modeling comfort. -->

**Folder:** `scripts/modeling/` — answers questions like *"which routes carry
more or fewer riders than their service and demographics would predict?"* and
*"what drives our monthly ridership?"* These pipelines chain several scripts
together and assume comfort with regression output. See the
[README's Demand Modeling section](README.md#demand-modeling-advanced) before starting.

---

## 11. Helper scripts: preparing Census and other national datasets

<!-- TODO: scenario framing. E.g. "These don't answer planning questions by
     themselves — they download-and-clean the public datasets that Sections 2, 3,
     and 10 consume." -->

**Folder:** `scripts/national_data_tools/`

| Dataset | Prep script |
|---|---|
| US Census tables (ACS/Decennial) | `national_data_tools/uscensus_table_build.py` |
| US Census joined to TIGER geographies | `national_data_tools/uscensus_tiger_join_gpd.py` (or `_arcpy`) |
| Canadian census (StatCan) by dissemination area | `national_data_tools/cacensus_da_join_gpd.py` |
| Schools + enrollment (NCES EDGE) | `national_data_tools/schools_prep_join_gpd.py` |
| Gas prices (EIA) | `national_data_tools/clean_eia_gas_prices.py` |
| Unemployment (FRED) | `national_data_tools/clean_fred_unrate.py` |
| Weather (NOAA) | `national_data_tools/clean_noaa_weather.py` |

---

<!-- TODO: closing section — "Didn't find your task?" Open an issue, or browse the
     folder list in the README. Maybe repeat the pointer back to README setup docs. -->
