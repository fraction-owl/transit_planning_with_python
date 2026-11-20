# Transit Planning With Python

**Transit Planning With Python** is a collection of standalone scripts for transit planners and analysts at medium-sized agencies. Use the tools you need to automate reports, check data, and evaluate service. Inputs include GTFS schedules, TIDES operational data, ridership exports, and GIS layers. An optional [demand-modeling workflow](#advanced-demand-modeling) builds on these tools' outputs; ordinary use requires no modeling knowledge.

**New to Python?** Start with [GETTING_STARTED.md](GETTING_STARTED.md), which walks through setting up Python, getting the scripts, and running your first one.

## 🚀 Planning Tools

Start with the task you need. Read the selected script's opening comments for inputs, outputs, and assumptions, then edit its CONFIGURATION section. Most spatial tools offer both ArcPy and GeoPandas versions; choose the one for your environment.

| I need to… | Start here | Main input |
|---|---|---|
| Summarize monthly ridership | [ntd_monthly_summary.py](scripts/ridership_tools/ntd_monthly_summary.py) | Monthly ridership workbooks |
| Answer a stop-ridership request | [data_request_by_stop_processor.py](scripts/ridership_tools/data_request_by_stop_processor.py) | Stop-ridership export |
| Export timetables | [gtfs_schedule_exporter.py](scripts/gtfs_exports/gtfs_schedule_exporter.py) | GTFS |
| Check service dates | [gtfs_calendar_inspector.py](scripts/gtfs_data_quality/gtfs_calendar_inspector.py) | GTFS |
| Compare route changes | [gtfs_route_diff.py](scripts/stop_analysis/gtfs_route_diff.py) | Before/after GTFS |
| Summarize observed OTP and runtime | [otp_runtime_from_tides.py](scripts/operations_tools/otp_runtime_from_tides.py) | TIDES |
| Calculate planned service hours | [scheduled_service_hours.py](scripts/gtfs_exports/scheduled_service_hours.py) | GTFS + optional TODS supplements |
| Review bay assignments | [bay_assignment_optimizer.py](scripts/facilities_tools/bay_assignment_optimizer.py) | GTFS and facility settings |

The sections below describe the wider collection. Scripts can also form workflows by passing their output files to another tool.

### Ridership & NTD Trend Analysis *(arcpy + GeoPandas)*
Turn NTD-format monthly summaries into route-level trend reports, process ridership by stop, and join ridership figures to stop geometries. Load factor monitoring is also included, built on ridecheck outputs — the data shape varies between agencies, but the underlying structure is robust and adaptable. For agencies without vendor ridecheck software, `ridership_from_tides.py` builds the inputs the rest of this folder expects directly from raw TIDES stop-event data. In one run it exports boardings, alightings and maximum load for every grain switched on — system, service type, route, route-direction, stop, route-stop, route-direction-stop, trip and trip-stop, each by day type and optionally by time period, hour of the service day, month, or month and time period — plus engine-ready trip tables for `service_cut_impact_gpd.py`, low-ridership trip flags and charts. Day types come from a weekday mapping, with holidays (their own day type) read from the GTFS calendar — a date whose scheduled services include none of its weekday's regular service — or from a manual list; a date table records why each date got its day type and flags dates whose trips don't match the GTFS schedule, were heavily canceled, are missing more than half their scheduled trips, or have conflicting duplicate stop visits; with GTFS, a trip table also flags scheduled trips missing from TIDES on most of their dates. A setting decides whether trips with missing APC counts are left out. Averages divide by the dates with observed data, and the time periods can be set inline, in a `.txt` file, or with command-line flags; the run stops if they leave a gap or overlap anywhere in the 24-hour day. `ridership_by_hour.py` rolls a vendor Route-and-Trip database export (one workbook per day type) up to average boardings by hour of the service day, draws per-route and system bar charts for each day type, and writes engine-ready `ridership_by_trip_<day>.csv` tables that plug directly into `service_cut_impact_gpd.py` when the export's trip numbers are GTFS trip_ids. `stop_ridership_change.py` takes two or more of the stop-level exports that `data_request_by_stop_processor.py` reads, one per signup in temporal order, and reports the change and % change at each stop (or each route × stop pair) from one signup to the next and from first to last, flagging stops that were added, removed, or appear only intermittently, and also writes a per-stop CSV for joining to stop locations plus charts of the largest gainers and losers and of first-vs-last ridership.
*Folder:* `ridership_tools/`

### OTP & Runtime Diagnostics
Choose the tool for your input and intended output:

- **TIDES events:** `otp_runtime_from_tides.py` exports OTP, trip runtimes, and timepoint-segment runtimes by route, direction, trip, and other levels, split by day type and time period. It also reports data coverage and flags unusual stop performance and runtime variability. With `--monthly-reports`, it adds monthly OTP tables and charts for routes, directions, service types, named corridors, and the system, with each month's share of trips that have scored visits.
- **Schedule drafting:** `runtime_workflow.py` compares prepared CLEVER trip and timepoint exports with scheduled runtimes and provides segment recommendations for planner review. `trip_runtime_diagnostics_from_avl.py` provides trip-level diagnostics and suggested time bands from vendor CSVs.
- **Vendor timepoint summaries:** `otp_by_timepoint_from_avl.py` produces monthly outputs by route, direction, and variation.

CleverWorks-to-TIDES converters prepare stop visits and performed trips. `tsde_to_tods.py` converts Trapeze TSDE into TODS pull-out, pull-back, and interlining-deadhead supplements for a matching GTFS feed.

*Folders:* `operations_tools/`, `conversion_tools/`

### Service Coverage & Catchments *(arcpy + GeoPandas)*
Generate GTFS-to-Census catchment areas and calculate service coverage by district — built for **Title VI** equity/demographic analysis and service planning. Site-to-route proximity tools support site planning collaboration with developers, elected officials, and partner agencies. A private-shuttle tool (`private_shuttle_coverage_by_route_gpd.py`, with an `_arcpy` twin for ArcGIS Pro environments) turns the hand-maintained operator spreadsheets agencies actually have — partial addresses, partial coordinates, free-text notes — into a clean registry, a geocoding worklist, a `Private_Shuttle_Stops` point layer for the strategic-site coverage counts, and an optional per-route rollup of shuttle sites (splitting transit *feeders* from potential *competitors*) that plugs into the demand-modeling bundles, so agencies can see where employer/university/residential shuttles run and test whether they move ridership. Service start/end dates are normalized and carried through, and the rollup's `--as-of` flag counts only the shuttles active on a chosen date — the groundwork for modeling dated shuttle openings and closures against the ridership panel. For corridor work, a cordon tool (`cordon_crossing_routes_gpd.py`) reads a layer of cordon lines, each drawn across a roadway with a little extra length past each curb, and lists the GTFS routes that cross each one, with crossings per trip and the share of each route's trips that cross, so ridership and service can be summed by corridor. It flags near misses for review — routes that come within a set distance (default 50 ft) of a cordon without crossing it, or that cross it in that extra length near its end — and can draw a PNG map of each cordon.
*Folders:* `service_coverage/`, `national_data_tools/`

### GTFS Data Quality & Validation *(arcpy + GeoPandas)*
Catch errors before they go live. Cross-check GTFS stops against road centerlines, flag skipped stops, validate stop spacing and proximity, check USPS suffix conventions, and identify stops that conflict with road geometry.
*Folder:* `gtfs_data_quality/`

### Service Changes & Stop Impacts *(arcpy + GeoPandas)*
Quantify what changes when you remove or relocate a stop. Score the downstream impact of stop removals on target routes, flag stop-spacing outliers, and compare GTFS stop sets across versions. For service changes, a companion `gtfs_route_diff.py` compares two feeds at the *route* level — classifying each route as eliminated, added, rekeyed, or changed (major/minor alignment, major/minor schedule, reblocking, name/attribute, calendar-approach, day-type, and fare changes), reconciling `route_id` renumbering so renamed routes aren't double-counted, flagging candidate route splits and merges, and reporting each feed's declared date range. Every threshold is a configurable knob. A scenario-based cut engine (`service_cut_impact_gpd.py`, with an `_arcpy` twin for ArcGIS Pro environments) quantifies what is lost when routes, trips, or stops are cut: each named scenario (whole routes, explicit trips, departure-window thinning, stop truncations, or any combination) is diffed against the baseline feed to report stops eliminated or degraded by time of day, walk-buffer coverage lost (with optional demographics), riders affected from optional ridership inputs (a trip-level table, a route×stop stop-usage export, or both), GTFS-approximated revenue hours and miles saved, and ready-to-map shapefiles (remaining and removed route alignments, remaining-network and lost-coverage walk buffers). It can also write each scenario's surviving network as a standalone GTFS feed, so any other tool in this repository can run baseline-vs-scenario comparisons on ordinary feeds, and a screening mode (`--each-route`) evaluates the elimination of every route in one run, ranking routes by the riders and revenue hours at stake. To check a feed against the agency's own route shapefile(s), `gtfs_shp_linear_diff_gpd.py` (with an `_arcpy` twin) picks each route's busiest GTFS pattern per direction on a typical weekday, merges the directions into one line (both directions on one street count once), and maps where the GTFS line and the shapefile line differ by more than a set distance (default 80 ft), listing routes that are missing on either side.
*Folder:* `stop_analysis/`

### Fieldwork & Ridechecks
Manual data collection that supplements or backstops modern AVL and APC systems when they go down or produce questionable data. Generate printable block schedules (optionally with each block's TODS pull-out, deadheads and pull-in alongside its layovers), build ridecheck cluster checklists for stationary surveyors, and process ridecheck results back into analysis-ready data. Per-pattern route maps (`pattern_map_exporter_gpd.py`) draw every distinct stop/path variant of each route in its own panel, with direction-of-travel arrows, stops, and each variant's trip counts and first/last trips by day type.
*Folder:* `field_tools/`

### Plus
- **GTFS Exports** (`gtfs_exports/`) — GTFS-to-shapefile, segment speeds, time-band summaries, stop patterns, timepoint schedule exports, an interlined-routes list (`interlined_routes_exporter.py`) that reads `block_id` to show which routes share vehicles on a representative weekday, Saturday and Sunday, with every route-to-route hand-off and its layover, a plain stops-by-route list (`route_stops_exporter.py`) that takes a list of route short names or ids and writes the stops each serves — one CSV per route/direction/stop and one of distinct stops, each noting the other routes that also serve it — per-route supply/competition features (`gtfs_route_features_gpd.py`, with an optional serves-the-CBD flag) for demand modeling, a multi-agency feed merger (`gtfs_feed_merger.py`) that combines several agencies' feeds (e.g. Fairfax Connector + WMATA + Ride On) into one valid GTFS feed — the first feed listed is dominant and keeps its IDs and route names, lower-priority IDs that collide are prefixed, route names another agency already uses get the agency name appended (`10_WMATA`), duplicate street stops within a configurable distance (default 50 ft) merge when their buses travel the same direction, bus bays stay separate unless configured otherwise, and calendars are clipped to the dates every feed covers — and a route-to-route transfer calculator (`route_transfer_calculator.py`) that, within one feed (a merged feed for transfers between agencies), lists and counts, for each target route, the routes it can transfer to within a configurable walking distance and (optionally) a scheduled wait window, a scheduled-hours calculator (`scheduled_service_hours.py`) that applies TODS supplements to a GTFS feed and totals revenue trip, layover, pull-out, pull-in and deadhead hours by service, by day type (weekday, Saturday, Sunday, holiday) and by period such as a month, using the GTFS calendar or a configured holiday list, and a longitudinal route timeline (`gtfs_route_timeline.py`) that takes a whole *sequence* of feeds in chronological order (with optional service-change dates) and charts when each route begins and ends over time — a Gantt-style timeline with routes down the left and year/month along the bottom, chaining route identity through renumberings and service gaps
- **GBFS Tools** (`gbfs_tools/`) — export GBFS static stations to shapefile and GeoJSON
- **Facility Tools** (`facilities_tools/`) — bay conflict reports, assignment optimization, candidate bay/time-shift comparisons, stop clustering, bus-bay cluster maps, amenity reviews, and pedestrian-network QA. `bus_bay_cluster_mapper.py` draws a map of each bus-bay cluster polygon with its numbered stops and the routes and directions serving each one, from one or more agencies' GTFS feeds, and lists stops just outside a cluster and close stops from different feeds for review rather than merging them. These tools support operational and facilities decisions independently of demand modeling.
- **National Data Tools** (`national_data_tools/`) — prep and join national/external reference datasets that feed coverage and demand work: US Census/TIGER geographies, plus exogenous context (EIA gas prices, FRED unemployment, NOAA weather, NCES schools)
- **Shared utilities** (`utils/gtfs_helpers.py`) and contributor tooling (`dev_tools/`)

## Advanced Demand Modeling

This optional workflow combines outputs from the planning tools to investigate ridership patterns and compare model estimates with observed results. The planning tools remain usable independently.

- **Route benchmarking:** `route_performance_model.py` compares observed ridership with a regression fitted to service and route context. Express routes can be excluded from the fit and reported separately.
- **Route or route-by-period analysis:** `monthly_ridership_model.py` fits OLS regression and reports coefficients, fitted values, residuals, and diagnostics.
- **Method comparison:** `ridership_ml_model.py` compares linear regression, random forests, and gradient boosting using cross-validation. It requires `scikit-learn`.

`prep_features_public.py` and `prep_features_private.py` assemble feature bundles for workflows that separate public inputs from restricted agency data. Related preparation scripts include `ntd_anchor_builder.py`, `otp_by_route.py`, and `runtime_by_route.py`, currently in the ridership and operations folders. They prepare modeling inputs and are not required for ordinary reporting.

Model results depend on the data and specification. Validation must match the intended use; fitted relationships alone do not establish the ridership effect of a proposed service change. Read each model's opening comments and configuration before use.

*Folder:* [scripts/modeling/](scripts/modeling/)

## 📂 Repository Structure

The repository is organized for ease of use, with:

- ***Standalone scripts:*** Each tool is fully documented with comments explaining field name assumptions, file formats, and usage instructions.
- ***Standard data formats:*** The preferred formats are the open transit data standards — GTFS (schedules), GBFS (bikeshare), TIDES (operational data such as AVL and APC), and TODS (scheduled operations such as runs and deadheads) — plus CSV and shapefiles. Most scripts read or write at least one of these; other formats, such as vendor exports, GeoJSON, and Excel, are used alongside them where a script needs them. Specific requirements are outlined in the script comments.

## 🛠️ Requirements

- Python — the required version depends on which scripts you run:
  - **Scripts that run inside ArcGIS Pro** — the `*_arcpy` variants plus any script that only needs the Pro-bundled stack (pandas, networkx, matplotlib, …): **Python 3.9+**, i.e. the Python that ships with ArcGIS Pro 3.0 or later. See `requirements-arcpro.txt` for what Pro already bundles.
  - **Scripts using the open-source geospatial stack** — anything importing geopandas, shapely, rapidfuzz, etc. (most `*_gpd` scripts): **Python 3.12+** in your own environment, installed with `pip install -r requirements.txt`. These cannot run in ArcGIS Pro's bundled Python.
- Common libraries like pandas, geopandas, matplotlib, networkx, and others.
  - **ArcGIS Pro users:** check the selected script's dependencies. Optimization and machine-learning tools may require additional packages, such as PuLP or scikit-learn.
  - **Non-ArcGIS / home users:** install the open-source stack with `pip install -r requirements.txt`.

## 🧑‍💻 How to Use

New to Python or Jupyter? [GETTING_STARTED.md](GETTING_STARTED.md) walks through setting up Python on a work computer with ArcGIS Pro or on a home computer, getting the scripts, and running one in a notebook, with troubleshooting steps for common installation problems.

Depending on the script, you can give it its settings in up to three ways:

- **Edit the CONFIGURATION section** at the top of the script. Every script has one.
- **Pass command-line flags.** About half the scripts (those that import `argparse`) also accept their settings as flags, so you can change inputs for one run without editing the file; see **Running from the Shell** below.
- **Point to a settings file.** `ntd_monthly_summary.py` keeps its settings in a separate `.txt` file, named in its CONFIGURATION section or passed with `--config`, and can repeat an earlier run from its `_runlog.txt`. `bay_assignment_optimizer.py` accepts a JSON file of setting overrides with `--config`, and `ridership_from_tides.py` can read its time periods from a `.txt` file. Other scripts don't read settings from a file yet.

### 🐚 Running from the Shell (Command Line)

Every script is also a normal command-line program — nothing about the notebook workflow is required. This suits users who prefer a terminal, or who want to automate a script in a batch file, Makefile, or scheduled task.

1. **Run a script directly**
   - Edit the CONFIGURATION section at the top of the script, then run it with Python:
     ```bash
     python scripts/gtfs_exports/gtfs_schedule_exporter.py
     ```
   - ArcGIS Pro users can run the `*_arcpy` variants with Pro's bundled interpreter instead:
     ```bat
     "C:\Program Files\ArcGIS\Pro\bin\Python\Scripts\propy.bat" scripts\gtfs_exports\gtfs_to_shapefile_arcpy.py
     ```

2. **Override the configuration with flags (no file editing needed)**
   - Many scripts mirror their CONFIGURATION constants as command-line flags, so you can point at new inputs per run without touching the file. See what a script accepts (and each flag's current default) with:
     ```bash
     python scripts/gtfs_data_quality/gtfs_calendar_inspector.py --help
     ```
     ```bash
     python scripts/gtfs_data_quality/gtfs_calendar_inspector.py --gtfs data/feed.zip --output-dir out
     ```
   - Flags are parsed strictly: a mistyped flag stops the script with an error instead of being silently ignored.

3. **Exit codes for automation**
   - Scripts return `0` on success, `1` on a runtime failure, and `2` when required CONFIGURATION values are still placeholders (or flags are invalid) — so shell scripts, cron jobs, and CI pipelines can detect failures reliably (`&&`, `set -e`, `if errorlevel`, etc.).
   - Every output file still gets its `_runlog.txt` sidecar recording the exact settings used, which serves as the audit trail for automated runs.

## 🤝 Contributing

Contributions are welcome - bug reports, new scripts, or improvements to existing ones. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request; it covers branch naming, commit message conventions, testing requirements, and code style expectations.

## 📄 License

This repository is licensed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).
You are free to use, modify, and distribute the tools within this repository, as long as you adhere to the license terms.

We hope these tools speed up and simplify your transit planning workflows! If you encounter issues or have questions, feel free to open an issue or contact us. 🚍
