# Transit Planning With Python

Welcome to **Transit Planning With Python**, a toolkit of off‑the‑shelf Python scripts designed for data‑driven transit planning at medium‑sized agencies. Each tool focuses on accomplishing a common task or solving a common challenge in transit analysis and planning. The most common input is static ***General Transit Feed Specification (GTFS)*** data, but TIDES operational data, Census data, ridership data, and road network data are also commonly used.

## 🚀 Features

A toolkit for transit planners and analysts. Most spatial tools ship in both `arcpy` and GeoPandas flavors, so the repo is usable whether or not your agency licenses ArcGIS Pro.

### Ridership & NTD Trend Analysis *(arcpy + GeoPandas)*
Turn NTD-format monthly summaries into route-level trend reports, process ridership by stop, and join ridership figures to stop geometries. Load factor monitoring is also included, built on ridecheck outputs — the data shape varies between agencies, but the underlying structure is robust and adaptable. For agencies without vendor ridecheck software, two TIDES-derived tools build the inputs the rest of this folder expects directly from raw stop-event data: `ridership_from_tides.py` sums boardings/alightings, and `max_load_from_tides.py` derives each trip's peak load — both at a configurable time-period × month grain, mirroring the TIDES OTP calculator.
*Folder:* `ridership_tools/`

### Demand Modeling *(advanced)*
For analysts comfortable assembling their own modeling tables. The core is a two-step ridership regression that keeps proprietary NTD data on one machine: `prep_features_public.py` loads, validates, and bundles the *non-NTD* feature tables — the service supplied (scheduled hours/miles), exogenous context (gas, unemployment, weather), and demographic and points-of-interest coverage — anywhere, writing a manifest with a per-bundle SHA-256, and `monthly_ridership_model.py` then runs only where the NTD anchor lives, verifying and joining each bundle before fitting a hand-rolled OLS — coefficients, robust standard errors, VIFs, AIC/BIC, and the usual diagnostics — entirely on the ArcGIS Pro `numpy`/`scipy` stack, so it needs no machine-learning package. A route-level anchor fits a cross-sectional model; a route×period anchor fits a panel that surfaces monthly and exogenous effects. A machine-learning counterpart (`ridership_ml_model.py`) ingests the *same* modeling table but swaps OLS for random-forest and gradient-boosting ensembles, comparing both against the linear baseline on cross-validated, out-of-sample accuracy and adding permutation importance and partial-dependence plots for interpretability. It captures the nonlinearities and interactions OLS can't — at the cost of a `scikit-learn` dependency (not bundled with ArcGIS Pro). For a more turnkey, opinionated route read, a two-step *cross-sectional route* pipeline pairs `gtfs_route_features.py` (in `gtfs_exports/`), which reads a GTFS feed and emits one row per route of supply and competition features, with `route_performance_model.py`, which joins those to route demographics and the NTD boardings anchor to fit a single cross-sectional OLS — reporting each route's fitted *potential* and its productivity-adjusted over/under residual, so planners can see which routes carry more or fewer riders than their fundamentals predict. Express routes can be held out of the fit and benchmarked descriptively instead (their park-and-ride catchment isn't what the buffer-based features measure), and an optional supply-free companion fit surfaces demand drivers that the dominant service term would otherwise mask. On the secured box, a parallel orchestrator (`prep_features_private.py`) assembles the proprietary side — the NTD ridership anchor plus the TIDES-derived on-time performance (`otp_by_route.py`) and running time (`runtime_by_route.py`), each rolled up to the route over a configurable trailing window (e.g. 12 or 36 months, naive or month-normalized) — into the same bundle-plus-manifest those models already ingest.
*Folder:* `modeling/`

### OTP & Runtime Diagnostics
Stop guessing where buses are getting delayed. Pivot route-level OTP trends, diagnose runtime by trip event, and analyze segment-level runtimes. Roll on-time performance (`otp_by_route.py`) and running time (`runtime_by_route.py`) up to the route over a trailing window for demand modeling. A stop-level counterpart (`otp_by_stop.py`) inverts the route view — computing OTP and AVL data coverage per stop and natively flagging stops that fall well below the baselines of *multiple* routes serving them — a signature of stop-specific operational or data issues (mis-placed geofences, relocated stops, chronic congestion) that route rollups average away. Includes CleverWorks-to-TIDES converters for stop visits and trips performed.
*Folder:* `operations_tools/`

### Service Coverage & Catchments *(arcpy + GeoPandas)*
Generate GTFS-to-Census catchment areas and calculate service coverage by district — built for **Title VI** equity/demographic analysis and service planning. Site-to-route proximity tools support site planning collaboration with developers, elected officials, and partner agencies.
*Folders:* `service_coverage/`, `national_data_tools/`

### GTFS Data Quality & Validation *(arcpy + GeoPandas)*
Catch errors before they go live. Cross-check GTFS stops against road centerlines, flag skipped stops, validate stop spacing and proximity, check USPS suffix conventions, and identify stops that conflict with road geometry.
*Folder:* `gtfs_data_quality/`

### Stop Placement & Impact Analysis *(arcpy + GeoPandas)*
Quantify what changes when you remove or relocate a stop. Score the downstream impact of stop removals on target routes, flag stop-spacing outliers, and compare GTFS stop sets across versions. For service changes, a companion `gtfs_route_diff.py` compares two feeds at the *route* level — classifying each route as eliminated, added, rekeyed, or changed (major/minor alignment, major/minor schedule, reblocking, name/attribute, calendar-approach, day-type, and fare changes), reconciling `route_id` renumbering so renamed routes aren't double-counted, flagging candidate route splits and merges, and reporting each feed's declared date range. Every threshold is a configurable knob.
*Folder:* `stop_analysis/`

### Fieldwork & Ridechecks
Manual data collection that supplements or backstops modern AVL and APC systems when they go down or produce questionable data. Generate printable block schedules, build ridecheck cluster checklists for stationary surveyors, and process ridecheck results back into analysis-ready data.
*Folder:* `field_tools/`

### Plus
- **GTFS Exports** (`gtfs_exports/`) — GTFS-to-shapefile, segment speeds, time-band summaries, stop patterns, timepoint schedule exports, per-route supply/competition features (`gtfs_route_features.py`) for demand modeling, and a route-to-route transfer calculator (`route_transfer_calculator.py`) that pools one or more feeds and, for each target route, lists and counts the routes it can transfer to within a configurable walking distance and (optionally) a scheduled wait window, and a longitudinal route timeline (`gtfs_route_timeline.py`) that takes a whole *sequence* of feeds in chronological order (with optional service-change dates) and charts when each route begins and ends over time — a Gantt-style timeline with routes down the left and year/month along the bottom, chaining route identity through renumberings and service gaps
- **GBFS Tools** (`gbfs_tools/`) — export GBFS static stations to shapefile and GeoJSON
- **Facility Tools** (`facilities_tools/`) — bay usage analysis and stop upgrade flagging
- **National Data Tools** (`national_data_tools/`) — prep and join national/external reference datasets that feed coverage and demand work: US Census/TIGER and Canadian census geographies, plus exogenous context (EIA gas prices, FRED unemployment, NOAA weather, NCES schools)
- **Shared utilities** (`utils/gtfs_helpers.py`) and contributor tooling (`dev_tools/`)

## 📂 Repository Structure

The repository is organized for ease of use, with:

- ***Standalone scripts:*** Each tool is fully documented with comments explaining field name assumptions, file formats, and usage instructions.
- ***Standard data formats:*** Most scripts are designed to work with commonly used data types like GTFS files and shapefiles. Specific requirements are outlined in the script comments.

## 🛠️ Requirements

- Python — the required version depends on which scripts you run:
  - **Scripts that run inside ArcGIS Pro** — the `*_arcpy` variants plus any script that only needs the Pro-bundled stack (pandas, networkx, matplotlib, …): **Python 3.9+**, i.e. the Python that ships with ArcGIS Pro 3.0 or later. Nothing extra to install — see `requirements-arcpro.txt` for what Pro already bundles.
  - **Scripts using the open-source geospatial stack** — anything importing geopandas, shapely, rapidfuzz, etc. (most `*_gpd` scripts): **Python 3.12+** in your own environment, installed with `pip install -r requirements.txt`. These cannot run in ArcGIS Pro's bundled Python.
- Common libraries like pandas, geopandas, matplotlib, networkx, and others.
  - **ArcGIS Pro users:** required libraries already ship with ArcGIS Pro — see `requirements-arcpro.txt` for a reference list.
  - **Non-ArcGIS / home users:** install the open-source stack with `pip install -r requirements.txt`.

## 🧑‍💻 How to Use

The **transit_planning_with_python** tools are designed to work on most systems with Python installed. Here are some key considerations based on your setup:
1. **Work PC with ArcPro Installed:**
   - If ArcPro is installed, libraries such as `arcpy` and other useful dependencies are already included.
   - However, your organization may restrict the installation of additional libraries like `geopandas` or `rapidfuzz`.
   - If unrestricted, note that `geopandas` conflicts with `arcpy`, so you will need to create a separate Python environment to use it.
2. **Home Computer with Python Installed:**
   - On a personal system, you can install Python and any libraries using `pip` without organization restrictions.
   - Keep in mind that `arcpy` is unavailable outside of ArcPro/ArcMap environments, so certain features relying on `arcpy` won't work.

Where possible, we will provide both `arcpy` and `geopandas` versions of geospatial scripts to accommodate these different setups.

---

### 🖥️ Option A: Using Python on a Work Computer with ArcPro

1. **Open a Notebook**
   - If ArcPro is installed on your computer, then Python is as well.
   - You can launch ArcPro and then open, create, or save a Notebook file (.ipynb) within that program.
   - You can also find "Jupyter Notebook" like any other program on your computer. Clicking it will open your default browser with the notebook interface.
   - Alternately, you can run this command in the Command Prompt to launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```

2. **Get the Script(s) You Need**
   - Navigate to scripts that are useful to you and your agency. Then copy and paste their contents into an active notebook file or download them as .py files. You do not need a GitHub account to do this.
   - Alternately, you can clone or download the whole repository from GitHub.
 
3. **Run the Script**
   - Confirm that desired code is present in notebook file and locate "CONFIGURATION" section.
   - "Update file paths:" At a minimum, you will need to update the folder and file paths to point to your data and specify where to save any
     output. There may also be additional configuration choices (e.g. choice of CRS, list of routes or stops to analyze)
   - You do not need to modify any code outside of the configuration section, which contains the core script logic.
   - Run the script, follow any printed instructions or error messages, and check the output for reasonableness and/or accuracy.

---

### ❓ What is Jupyter Notebook?

Jupyter Notebook is a powerful tool for running Python scripts in an interactive environment that opens in your web browser (or within ArcPro). It allows you to write, test, and visualize Python code in a user-friendly way.

---

### 🏡 Option B: Setting Up Python on a Home Computer

1. **Download and Install Python**
   - Visit the [official Python website](https://www.python.org/downloads/) and download Python 3.12 or later.
   - During installation, ensure you check the option to **"Add Python to PATH"**. This step is crucial for the command-line tools to work correctly.

2. **Install JupyterLab and Required Libraries**
   - Open the Command Prompt (search for "cmd" in your Start menu) and run the following command:
     ```bash
     pip install jupyterlab -r requirements.txt
     ```
   - Wait for the installation to complete. If you see warnings about scripts not being on the PATH, don't worry - you can still use these tools.

3. **Launch JupyterLab**
   - After installation, search for "Command Prompt" and open it like any other program on your computer.
   - From the Command Prompt, type the following and press Enter:
     ```bash
     jupyter lab
     ```
   - Once you have JupyterLab open in your web browser, you can browse your local files and create new notebook files.

4. **Get the Script(s) You Need**
   - Navigate to scripts that are useful to you and your agency. Then copy and paste their contents into an active notebook file or download them as .py files. You do not need a GitHub account to do this.
   - Alternately, you can clone or download the whole repository from GitHub.
 
5. **Run the Script**
   - Confirm that desired code is present in notebook file and locate "CONFIGURATION" section.
   - "Update file paths:" At a minimum, you will need to update the folder and file paths to point to your data and specify where to save any
     output. There may also be additional configuration choices (e.g. choice of CRS, list of routes or stops to analyze)
   - You do not need to modify any code outside of the configuration section, which contains the core script logic.
   - Run the script, follow any printed instructions or error messages, and check the output for reasonableness and/or accuracy.

---

## 💡 Notes for Beginners

If you encounter any issues while installing Python or libraries, or if JupyterLab doesn't open as expected, here are some common troubleshooting steps:

1. **Verify Python Installation**

   Open the Command Prompt and type:
     ```bash
     python --version
     ```

   If this command doesn't show a Python version (e.g., `Python 3.12.5`), Python may not be installed correctly, or it isn't added to your PATH. In that case:
   - Reinstall Python from the [official website](https://www.python.org/downloads/).
   - During installation, ensure the option **"Add Python to PATH"** is checked.
   - Or proceed to **Step 3: Fixing PATH Issues**

2. **Check JupyterLab Installation**

   To confirm JupyterLab is installed, type:
     ```bash
     jupyter lab --version
     ```

   If this fails with an error like `'jupyter' is not recognized as an internal or external command`, JupyterLab is either not installed or the PATH variable isn't set correctly.  
   Try reinstalling JupyterLab by running:
     ```bash
     pip install jupyterlab
     ```

   If the issue persists, follow the steps in **Fixing PATH Issues** below.

3. **Fixing PATH Issues**

   If Python or Jupyter commands aren't recognized, you may need to manually add the Python Scripts directory to your PATH:

   - Locate the directory where Python installed its scripts. For most users, this will be:
        ```makefile
     C:\Users\<YourUsername>\AppData\Roaming\Python\Python311\Scripts
     ```

   - Add this directory to your PATH:
      - Press <kbd>Win + S</kbd>, search for **Environment Variables**, and open **Edit the system environment variables**.
      - In the dialog, click **Environment Variables**.
      - Under **User variables**, find and edit the **Path** variable.
      - Add the Scripts directory to the list by clicking **New** and pasting the path above.

   - Restart your Command Prompt and verify by typing:
     ```bash
     jupyter lab --version
     ```   

4. **Reinstalling or Updating Pip**

   If the installation commands fail with errors like `pip is not recognized` or `ModuleNotFoundError`, try reinstalling or upgrading pip:
     ```bash
     python -m ensurepip --upgrade
     python -m pip install --upgrade pip
     ```   

5. **Alternative Way to Open JupyterLab**

   Instead of relying on PATH, you can directly use Python to run JupyterLab:
     ```bash
     python -m jupyter lab
     ```
     
   This bypasses PATH issues entirely.

By following these steps, you should have Python and the necessary tools ready to run your scripts efficiently. If you continue to face issues, feel free to consult the official documentation for [Python](https://docs.python.org/3/) or [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/).

---

### ❓ What is JupyterLab?

JupyterLab is an advanced interactive development environment for data science and computational research that runs in your web browser. It builds on the functionality of Jupyter Notebook, providing a flexible interface where you can write, test, and visualize Python code alongside other tasks such as file browsing, markdown editing, and interactive data exploration.

---

### 🌟 Tips for Success

- **File Paths:** Ensure all file paths are updated in the scripts to match your local system.
- **Permissions:** If IT restrictions block library installations on your work computer, setting up Python on a home computer is recommended.

## 🤝 Contributing

Contributions are welcome - bug reports, new scripts, or improvements to existing ones. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request; it covers branch naming, commit message conventions, testing requirements, and code style expectations.

## 📄 License

This repository is licensed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).
You are free to use, modify, and distribute the tools within this repository, as long as you adhere to the license terms.

We hope these tools speed up and simplify your transit planning workflows! If you encounter issues or have questions, feel free to open an issue or contact us. 🚍
