r"""Analyze NTD monthly ridership using a plain-text config or a previous run log.

Set CONFIG_PATH below, then run the script from a notebook, IDE, or shell:
    python ntd_monthly_summary.py
    python ntd_monthly_summary.py --config "C:\\reports\\ntd_monthly_config.txt"
    python ntd_monthly_summary.py --config "C:\\reports\\ntd_monthly_summary_runlog.txt"
    python ntd_monthly_summary.py --validate-config

From a notebook or ArcGIS Python window:
    import ntd_monthly_summary as ntd
    ntd.main(r"C:\reports\ntd_monthly_config.txt")

The config uses INI sections, read by Python's standard-library configparser.
No configuration content is executed. Requires Python 3.9+ and the original
analysis dependencies: pandas, matplotlib, and an Excel engine such as openpyxl.
Relative paths resolve against the configuration file's directory. Replaying a
run log retains that original directory, even if the log has been moved.
Run-log input creates a new timestamped output folder beside the recorded output
folder, preserving the earlier results. The workbooks themselves must still exist.

Outputs retain the original master CSV, monthly workbook, route summaries,
service-type summaries, plots, trend flags, and time-window folders. The run log
records the loaded configuration text and resolved paths. Optional garage
assignments add metadata to detailed data and route summaries.
"""

from __future__ import annotations

import argparse
import calendar
import configparser
import logging
import math
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Final, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import FixedFormatter, FixedLocator

# Full config OR previous run-log location. Join filenames with /.
# A UNC path starts with exactly two backslashes. No __file__ is needed in notebooks.
CONFIG_PATH: Final[Path] = Path(r"Path\To\Your\Config_Folder") / "ntd_monthly_config.txt"

RUN_LOG_TITLE: Final[str] = "NTD MONTHLY SUMMARY RUN LOG"
REPLAY_CONFIG_BEGIN: Final[str] = "# === BEGIN REPLAY CONFIG ==="
REPLAY_CONFIG_END: Final[str] = "# === END REPLAY CONFIG ==="

# Workbook schema and supported service periods belong to the analysis code.
REQUIRED_NUMERIC_COLS: Final[list[str]] = [
    "MTH_BOARD",
    "MTH_REV_HOURS",
    "MTH_PASS_MILES",
    "ASCH_TRIPS",
    "DAYS",
    "REV_MILES",
]
SERVICE_PERIODS: Final[list[str]] = ["Weekday", "Saturday", "Sunday"]
MONTH_NAMES: Final[tuple[str, ...]] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
PLOT_FLAGS: Final[tuple[str, ...]] = (
    "plot_total_ridership",
    "plot_weekday_avg",
    "plot_saturday_avg",
    "plot_sunday_avg",
    "plot_revenue_hours",
    "plot_trips",
    "plot_revenue_miles",
    "plot_pph",
    "plot_ppt",
    "plot_ppm",
)


@dataclass(frozen=True)
class PeriodSpec:
    """Workbook filename and source worksheet."""

    filename: str
    sheet: str


@dataclass(frozen=True)
class TimeWindow:
    """Inclusive whole-month span to aggregate."""

    label: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class PlotStyle:
    """Validated presentation settings."""

    figsize: tuple[float, float]
    marker: str
    linestyle: str
    rotation: float
    grid: bool


@dataclass(frozen=True)
class AppConfig:
    """Parsed settings passed explicitly to the analysis functions."""

    source_path: Path
    source_text: str
    data_root: Path
    output_dir: Path
    periods: dict[str, PeriodSpec]
    time_windows: list[TimeWindow]
    holidays: list[datetime]
    service_types: dict[str, list[str]]
    corridors: dict[str, list[str]]
    garage_assignments: dict[str, list[str]]
    rolling_window: int
    min_coverage: float
    decline_thresh_pct: float
    require_two_months: bool
    exclude_data: dict[str, str | list[str]]
    plots: dict[str, bool]
    plot_style: PlotStyle
    log_level: int
    require_run_log: bool
    from_run_log: bool = False
    config_base_dir: Path | None = None

    @property
    def ordered_periods(self) -> list[str]:
        """Return catalogue months in chronological order."""
        return list(self.periods)

    @property
    def timeline(self) -> list[str]:
        """Keep absent catalogue months as gaps, not compressed time."""
        start = parse_period(self.ordered_periods[0])
        end = parse_period(self.ordered_periods[-1])
        months = pd.date_range(start, end, freq="MS")
        return [f"{MONTH_NAMES[d.month - 1]}-{d.year:04d}" for d in months]


class ConfigError(ValueError):
    """An invalid analyst-supplied setting."""


class CasePreservingConfigParser(configparser.ConfigParser):
    """Keep month keys, group labels, and option names exactly as written."""

    def optionxform(self, optionstr: str) -> str:
        """Return *optionstr* unchanged instead of lower-casing it."""
        return optionstr


def parse_period(value: str) -> datetime:
    """Parse an English MMM-YYYY month without relying on the host locale."""
    match = re.fullmatch(r"([A-Z][a-z]{2})-(\d{4})", value)
    if not match or match[1] not in MONTH_NAMES:
        raise ConfigError(f"Invalid period {value!r}; use MMM-YYYY, e.g. Jul-2024.")
    try:
        return datetime(int(match[2]), MONTH_NAMES.index(match[1]) + 1, 1)
    except ValueError as exc:
        raise ConfigError(f"Invalid period {value!r}: {exc}") from exc


def parse_date(value: str, context: str) -> datetime:
    """Read a strict ISO date and identify the offending setting on failure."""
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("use YYYY-MM-DD")
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ConfigError(f"{context}: invalid date {value!r}; use YYYY-MM-DD.") from exc


def split_list(value: str, context: str) -> list[str]:
    """Read a comma-separated list; an entirely blank value means no entries."""
    if not value.strip():
        return []
    items = [item.strip() for item in value.split(",")]
    if any(not item for item in items):
        raise ConfigError(f"{context}: empty list entry; remove the extra comma.")
    return items


def split_pair(value: str, context: str) -> tuple[str, str]:
    """Read exactly two nonempty pipe-separated values."""
    parts = [part.strip() for part in value.split("|")]
    if len(parts) != 2 or not all(parts):
        raise ConfigError(f"{context}: expected two nonempty values separated by |.")
    return parts[0], parts[1]


def normalize_route(value: str) -> str:
    """Match the workbook route normalization without converting IDs to numbers."""
    return re.sub(r"\.0$", "", value.strip().upper().replace(" ", ""))


def route_list(value: str, context: str) -> list[str]:
    """Read route IDs and reject duplicates after normalization."""
    routes = [normalize_route(item) for item in split_list(value, context)]
    if any(not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]*", route) for route in routes):
        raise ConfigError(f"{context}: route IDs must be unquoted letters/numbers, ., _, or -.")
    if len(routes) != len(set(routes)):
        raise ConfigError(f"{context}: duplicate route ID after normalization.")
    return routes


def read_groups(
    parser: configparser.ConfigParser, section: str, *, exclusive: bool
) -> dict[str, list[str]]:
    """Read classification lists, permitting corridor overlap only."""
    if not parser.has_section(section):
        return {}
    result: dict[str, list[str]] = {}
    assigned: dict[str, str] = {}
    for label, value in parser.items(section):
        if label.casefold() in {"unknown", "total"}:
            raise ConfigError(f"[{section}]: label {label!r} is reserved.")
        routes = route_list(value, f"[{section}] {label}")
        if not routes:
            raise ConfigError(f"[{section}] {label}: supply routes or remove this entry.")
        for route in routes:
            if exclusive and route in assigned:
                raise ConfigError(
                    f"[{section}]: route {route} appears in both {assigned[route]!r} and {label!r}."
                )
            assigned[route] = label
        result[label] = routes
    return result


def config_path(value: str, base: Path) -> Path:
    """Preserve Windows/UNC literals and resolve relative paths beside the config."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    if not value or "\n" in value or "\r" in value:
        raise ConfigError("[paths]: paths must be nonempty, single-line values.")
    windows = PureWindowsPath(value)
    if windows.drive and not windows.is_absolute():
        raise ConfigError(f"[paths]: drive-relative path {value!r} is ambiguous.")
    path = Path(value).expanduser()
    if path.is_absolute() or windows.is_absolute():
        return path
    return (base / path).resolve()


def extract_replay_config(text: str) -> tuple[str, dict[str, str]]:
    """Extract INI settings from a new log or the previous unmarked log format."""
    lines = text.splitlines()
    header = "CONFIGURATION (text loaded for this run)"
    if lines.count(header) != 1:
        raise ConfigError("Run log must contain one INI configuration section.")
    heading = lines.index(header)
    metadata = {}
    for line in lines[:heading]:
        key, separator, value = line.partition(":")
        if separator:
            key = key.strip()
            if key in metadata:
                raise ConfigError(f"Run log contains duplicate metadata: {key}.")
            metadata[key] = value.strip()

    if REPLAY_CONFIG_BEGIN in lines or REPLAY_CONFIG_END in lines:
        if lines.count(REPLAY_CONFIG_BEGIN) != 1 or lines.count(REPLAY_CONFIG_END) != 1:
            raise ConfigError("Run log has missing or duplicate configuration markers.")
        start = lines.index(REPLAY_CONFIG_BEGIN) + 1
        end = lines.index(REPLAY_CONFIG_END)
        if not heading < start <= end:
            raise ConfigError("Run-log configuration markers are out of order.")
    else:
        # Backward compatibility with the log produced before replay was supported.
        if heading + 1 >= len(lines) or lines[heading + 1] != "-" * 72:
            raise ConfigError("Run-log configuration header is incomplete.")
        start = heading + 2
        footers = [
            i
            for i in range(start, len(lines))
            if lines[i] == "=" * 72
            or (
                lines[i] == "-" * 72
                and i + 1 < len(lines)
                and lines[i + 1] == "SERVICE-DAY COUNTS PER MONTH"
            )
        ]
        if not footers:
            raise ConfigError("Run log is truncated: no end of the configuration was found.")
        end = footers[0]

    config_text = "\n".join(lines[start:end]).strip("\n") + "\n"
    if not config_text.strip():
        raise ConfigError("Run-log configuration section is empty.")
    return config_text, metadata


def replay_base_directory(metadata: dict[str, str]) -> Path:
    """Recover the original base for relative paths without needing the old config file."""
    base_text = metadata.get("Config base directory")
    if base_text:
        base = config_path(base_text, Path.cwd())
        if not Path(base_text).is_absolute() and not PureWindowsPath(base_text).is_absolute():
            raise ConfigError("Run-log config base directory must be absolute.")
        return base
    original = metadata.get("Config file", "")
    if PureWindowsPath(original).is_absolute():
        return Path(str(PureWindowsPath(original).parent))
    if Path(original).is_absolute():
        return Path(original).parent
    raise ConfigError("Run log is missing the original absolute Config file path.")


def load_config(path: str | Path) -> AppConfig:
    """Load and validate all settings without opening Excel files or writing output."""
    source_path = Path(path).expanduser().resolve()
    try:
        source_text = source_path.read_text(encoding="utf-8-sig")
        from_run_log = RUN_LOG_TITLE in source_text.splitlines()[:3]
        base_dir = source_path.parent
        if from_run_log:
            source_text, metadata = extract_replay_config(source_text)
            base_dir = replay_base_directory(metadata)
        parser = CasePreservingConfigParser(
            interpolation=None,
            delimiters=("=",),
            strict=True,
            inline_comment_prefixes=None,
            empty_lines_in_values=False,
        )
        parser.read_string(source_text, source=str(source_path))
        config = parse_config(parser, source_path, source_text, base_dir=base_dir)
        return replace(config, from_run_log=from_run_log, config_base_dir=base_dir)
    except (OSError, UnicodeError, configparser.Error, ValueError) as exc:
        raise ConfigError(f"{source_path}: {exc}") from exc


def parse_config(
    parser: configparser.ConfigParser,
    source_path: Path,
    source_text: str,
    *,
    base_dir: Path | None = None,
) -> AppConfig:
    """Convert a ConfigParser into a typed configuration with no silent defaults."""
    fixed_keys = {
        "paths": {"data_root", "output_dir"},
        "holidays": {"dates"},
        "trend": {"rolling_window", "min_coverage", "decline_thresh_pct", "require_two_months"},
        "plots": set(PLOT_FLAGS),
        "plot_style": {"figsize", "marker", "linestyle", "rotation", "grid"},
        "logging": {"level", "require_run_log"},
    }
    required_sections = set(fixed_keys) | {
        "periods",
        "time_windows",
        "service_types",
        "corridors",
        "exclude_data",
    }
    if parser.defaults():
        raise ConfigError("[DEFAULT] entries are not supported; use the named sections.")
    missing_sections = required_sections - set(parser.sections())
    extra_sections = set(parser.sections()) - required_sections - {"garage_assignments"}
    if missing_sections or extra_sections:
        raise ConfigError(
            f"Sections: missing={sorted(missing_sections)}, unknown={sorted(extra_sections)}."
        )
    for section, expected in fixed_keys.items():
        actual = set(parser[section])
        if actual != expected:
            raise ConfigError(
                f"[{section}]: missing settings={sorted(expected - actual)}, "
                f"unknown settings={sorted(actual - expected)}."
            )

    periods: dict[str, PeriodSpec] = {}
    for label, value in sorted(parser.items("periods"), key=lambda item: parse_period(item[0])):
        filename, sheet = split_pair(value, f"[periods] {label}")
        periods[label] = PeriodSpec(filename, sheet)
    if not periods:
        raise ConfigError("[periods]: configure at least one workbook.")

    windows: list[TimeWindow] = []
    seen_labels: set[str] = set()
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
        "PLOTS",
    }
    for label, value in parser.items("time_windows"):
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label)
            or label.upper() in reserved
            or label.casefold() in seen_labels
        ):
            raise ConfigError(
                f"[time_windows]: unsafe, reserved, or duplicate folder label {label!r}."
            )
        seen_labels.add(label.casefold())
        start_text, end_text = split_pair(value, f"[time_windows] {label}")
        start = parse_date(start_text, label)
        end = parse_date(end_text, label)
        if start > end:
            raise ConfigError(f"[time_windows] {label}: start must be on or before end.")
        if start.day != 1 or end.day != calendar.monthrange(end.year, end.month)[1]:
            raise ConfigError(
                f"[time_windows] {label}: use complete months (first day to last day)."
            )
        windows.append(TimeWindow(label, start, end))

    holidays = [
        parse_date(item, "[holidays] dates")
        for item in split_list(parser.get("holidays", "dates"), "[holidays] dates")
    ]
    if len(holidays) != len(set(holidays)):
        raise ConfigError("[holidays]: duplicate dates would subtract service days twice.")

    exclusions: dict[str, str | list[str]] = {}
    for label, value in parser.items("exclude_data"):
        parse_period(label)
        if label not in periods:
            raise ConfigError(f"[exclude_data] {label}: period is absent from [periods].")
        exclusions[label] = (
            "*" if value.strip() == "*" else route_list(value, f"[exclude_data] {label}")
        )
        if not exclusions[label]:
            raise ConfigError(f"[exclude_data] {label}: supply * or route IDs.")

    window = parser.getint("trend", "rolling_window")
    coverage = parser.getfloat("trend", "min_coverage")
    threshold = parser.getfloat("trend", "decline_thresh_pct")
    if window < 1 or not 0 < coverage <= 1 or not 0 < threshold <= 100:
        raise ConfigError("[trend]: require window >= 1, 0 < coverage <= 1, 0 < decline <= 100.")

    size = split_list(parser.get("plot_style", "figsize"), "[plot_style] figsize")
    if len(size) != 2:
        raise ConfigError("[plot_style] figsize: supply width, height in inches.")
    width, height = float(size[0]), float(size[1])
    rotation = parser.getfloat("plot_style", "rotation")
    if not all(math.isfinite(x) and x > 0 for x in (width, height)) or not math.isfinite(rotation):
        raise ConfigError("[plot_style]: size must be positive/finite and rotation must be finite.")
    marker = parser.get("plot_style", "marker")
    MarkerStyle(marker)
    linestyle = parser.get("plot_style", "linestyle")
    if linestyle not in {"-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted", "None", ""}:
        raise ConfigError(f"[plot_style]: unsupported linestyle {linestyle!r}.")

    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = parser.get("logging", "level").upper()
    if level not in levels:
        raise ConfigError(f"[logging] level: choose one of {', '.join(levels)}.")

    return AppConfig(
        source_path=source_path,
        source_text=source_text,
        data_root=config_path(parser.get("paths", "data_root"), base_dir or source_path.parent),
        output_dir=config_path(parser.get("paths", "output_dir"), base_dir or source_path.parent),
        periods=periods,
        time_windows=windows,
        holidays=holidays,
        service_types=read_groups(parser, "service_types", exclusive=True),
        corridors=read_groups(parser, "corridors", exclusive=False),
        garage_assignments=read_groups(parser, "garage_assignments", exclusive=True),
        rolling_window=window,
        min_coverage=coverage,
        decline_thresh_pct=threshold,
        require_two_months=parser.getboolean("trend", "require_two_months"),
        exclude_data=exclusions,
        plots={flag: parser.getboolean("plots", flag) for flag in PLOT_FLAGS},
        plot_style=PlotStyle(
            (width, height), marker, linestyle, rotation, parser.getboolean("plot_style", "grid")
        ),
        log_level=levels[level],
        require_run_log=parser.getboolean("logging", "require_run_log"),
    )


def check_input_files(config: AppConfig) -> None:
    """Report all missing workbook paths before creating the output folder."""
    missing = [
        f"{period}: {config.data_root / spec.filename}"
        for period, spec in config.periods.items()
        if not (config.data_root / spec.filename).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing input workbooks:\n" + "\n".join(missing))


def _is_excluded(period: str, route: str, config: AppConfig) -> bool:
    """Return True if (*period*, *route*) is in EXCLUDE_DATA."""
    spec = config.exclude_data.get(period)
    if spec is None:  # period not excluded
        return False
    if spec == "*":  # whole system excluded
        return True
    return route in spec  # route‑specific exclusion


def slice_for_window(df: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    """Return rows in *df* whose month-start date falls inside *window*."""
    if "period_dt" not in df.columns:
        # Parse once – assumes "Jul-2024", "Aug-2024", …
        df = df.assign(period_dt=pd.to_datetime(df["period"].map(parse_period), errors="coerce"))
    mask = (df["period_dt"] >= window.start) & (df["period_dt"] <= window.end)
    return df.loc[mask].copy()


def safe_float(value: Any) -> float | None:
    """Return float if *value* looks numeric; else ``None``."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def safe_div(
    numerator: float | int,
    denominator: float | int,
    precision: int = 1,
) -> float | None:
    """Divide with protective zero handling."""
    try:
        # NumPy scalars divide by zero without raising ZeroDivisionError
        # (they warn and yield nan/inf), so guard the denominator explicitly.
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return None
        # MyPy may complain about mixed float/int division but it's safe here
        result = round(numerator / denominator, precision)
    except (ZeroDivisionError, TypeError):
        return None
    return result if math.isfinite(result) else None


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case, trim, and replace spaces with underscores in columns."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    return df


def read_excel_data(config: AppConfig) -> dict[str, pd.DataFrame]:
    """Load, clean, and filter every monthly workbook."""
    # Cast converters to Any to bypass MyPy strictness on Mapping/Dict variance
    converters: Any = {col: safe_float for col in REQUIRED_NUMERIC_COLS}
    data: dict[str, pd.DataFrame] = {}

    for period in config.ordered_periods:
        spec = config.periods[period]
        path: Path = config.data_root / spec.filename
        logging.info("→ Reading %s (%s) … ", period, spec.filename)

        # --- load ---------------------------------------------------------- #
        try:
            df = pd.read_excel(path, sheet_name=spec.sheet, converters=converters)
        except ValueError as exc:
            raise ValueError(
                f"{period}: cannot read {spec.filename!r}, sheet {spec.sheet!r}: {exc}"
            ) from exc
        df = normalise_columns(df)

        # --- guard-rail ---------------------------------------------------- #
        required = [*REQUIRED_NUMERIC_COLS, "ROUTE_NAME", "SERVICE_PERIOD"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"{period}: workbook '{spec.filename}' is missing "
                f"required column(s): {', '.join(missing)}"
            )

        # --- cleansing ----------------------------------------------------- #
        df = df.dropna(subset=["ROUTE_NAME", "MTH_BOARD"])
        df = df[df["MTH_BOARD"] != 0]
        df = df[df["SERVICE_PERIOD"].isin(SERVICE_PERIODS)].copy()
        df["ROUTE_NAME"] = (
            df["ROUTE_NAME"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(" ", "", regex=False)
            .apply(lambda x: re.sub(r"\.0$", "", x))
        )

        data[period] = df
        logging.info("%d rows", len(df))

    return data


def classify_route(route_name: str, config: AppConfig) -> str:
    """Return the first matching service-type label for *route_name*."""
    for stype, lst in config.service_types.items():
        if route_name in lst:
            return stype
    return "unknown"


def classify_corridor(route_name: str, config: AppConfig) -> list[str]:
    """Return all corridor labels containing *route_name*."""
    matches = [c for c, lst in config.corridors.items() if route_name in lst]
    return matches or ["other"]


def calculate_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append trip- and mileage-based indicators to *df*."""
    df = df.copy()
    df["TOTAL_TRIPS"] = df["ASCH_TRIPS"] * df["DAYS"]
    df["BOARDS_PER_HOUR"] = safe_div_vec(df["MTH_BOARD"], df["MTH_REV_HOURS"])
    df["MTH_REV_MILES"] = df["REV_MILES"] * df["DAYS"]
    df["PASSENGERS_PER_TRIP"] = safe_div_vec(df["MTH_BOARD"], df["TOTAL_TRIPS"])
    df["PASSENGERS_PER_MILE"] = safe_div_vec(df["MTH_BOARD"], df["MTH_REV_MILES"], 3)

    df["TOTAL_TRIPS"] = df["TOTAL_TRIPS"].round(1)
    df["BOARDS_PER_HOUR"] = df["BOARDS_PER_HOUR"].round(1)
    df["PASSENGERS_PER_TRIP"] = df["PASSENGERS_PER_TRIP"].round(1)
    df["PASSENGERS_PER_MILE"] = df["PASSENGERS_PER_MILE"].round(3)
    return df


def safe_div_vec(a: pd.Series, b: pd.Series, precision: int = 1) -> pd.Series:
    """Vectorised safe_div for Series."""
    result = a.astype(float) / b.astype(float)
    result = result.replace([float("inf"), float("-inf")], float("nan"))
    return result.round(precision)


def aggregate_by_service_type(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise performance at the service-type level."""
    grouped = (
        df.groupby("service_type")
        .agg(
            {
                "MTH_BOARD": "sum",
                "MTH_REV_HOURS": "sum",
                "MTH_PASS_MILES": "sum",
                "MTH_REV_MILES": "sum",
                "TOTAL_TRIPS": "sum",
            }
        )
        .reset_index()
    )

    grouped["BOARDS_PER_HOUR"] = safe_div_vec(grouped["MTH_BOARD"], grouped["MTH_REV_HOURS"])
    grouped["PASSENGERS_PER_TRIP"] = safe_div_vec(grouped["MTH_BOARD"], grouped["TOTAL_TRIPS"])
    grouped["PASSENGERS_PER_MILE"] = safe_div_vec(grouped["MTH_BOARD"], grouped["MTH_REV_MILES"], 3)

    totals = grouped[
        ["MTH_BOARD", "MTH_REV_HOURS", "MTH_PASS_MILES", "MTH_REV_MILES", "TOTAL_TRIPS"]
    ].sum()

    total_row = {
        "service_type": "TOTAL",
        "MTH_BOARD": totals["MTH_BOARD"],
        "MTH_REV_HOURS": totals["MTH_REV_HOURS"],
        "MTH_PASS_MILES": totals["MTH_PASS_MILES"],
        "MTH_REV_MILES": totals["MTH_REV_MILES"],
        "TOTAL_TRIPS": totals["TOTAL_TRIPS"],
        "BOARDS_PER_HOUR": safe_div(totals["MTH_BOARD"], totals["MTH_REV_HOURS"]),
        "PASSENGERS_PER_TRIP": safe_div(totals["MTH_BOARD"], totals["TOTAL_TRIPS"]),
        "PASSENGERS_PER_MILE": safe_div(totals["MTH_BOARD"], totals["MTH_REV_MILES"], 3),
    }

    grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)
    return grouped


def route_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a per-route performance table."""
    agg_cols = {
        "MTH_BOARD": "sum",
        "DAYS": "sum",
        "MTH_REV_HOURS": "sum",
        "MTH_PASS_MILES": "sum",
        "MTH_REV_MILES": "sum",
        "TOTAL_TRIPS": "sum",
    }
    group_columns = ["service_type", "ROUTE_NAME"]
    if "garage" in df.columns:
        group_columns.append("garage")
    totals = df.groupby(group_columns, as_index=False).agg(agg_cols)

    totals["BOARDS_PER_HOUR"] = safe_div_vec(totals["MTH_BOARD"], totals["MTH_REV_HOURS"])
    totals["PASSENGERS_PER_TRIP"] = safe_div_vec(totals["MTH_BOARD"], totals["TOTAL_TRIPS"])
    totals["PASSENGERS_PER_MILE"] = safe_div_vec(totals["MTH_BOARD"], totals["MTH_REV_MILES"], 3)
    totals["DAILY_AVG"] = safe_div_vec(totals["MTH_BOARD"], totals["DAYS"])

    totals = totals.sort_values(["ROUTE_NAME", "service_type"], ignore_index=True)
    return totals


def with_weekday_ex_holiday_columns(
    weekday_summary: pd.DataFrame,
    weekday_df: pd.DataFrame,
    holiday_counts: dict[str, int],
) -> pd.DataFrame:
    """Append holiday-free weekday columns to a weekday route summary.

    *weekday_summary* is the output of :func:`route_level_summary` for the
    weekday subset; *weekday_df* is that same subset (filtered to
    ``SERVICE_PERIOD == "Weekday"``). Adds two columns:

    * ``DAYS_EX_HOLIDAYS`` — weekday service days minus the weekday holidays that
      fell in the months each route operated.
    * ``DAILY_AVG_EX_HOLIDAYS`` — boardings over those holiday-free days.

    The summary is returned unchanged when no weekday holiday falls inside the
    covered months, so the extra columns appear only when they carry information.
    """
    if weekday_df.empty or not holiday_counts:
        return weekday_summary

    # Holidays removed per route: sum over each route's distinct service months.
    removed = (
        weekday_df.drop_duplicates(["ROUTE_NAME", "period"])
        .assign(_hol=lambda d: d["period"].map(holiday_counts).fillna(0).astype(float))
        .groupby("ROUTE_NAME")["_hol"]
        .sum()
    )
    if removed.sum() == 0:
        return weekday_summary

    out = weekday_summary.merge(
        removed.rename("_removed").reset_index(), on="ROUTE_NAME", how="left"
    )
    out["_removed"] = out["_removed"].fillna(0)
    out["DAYS_EX_HOLIDAYS"] = (out["DAYS"] - out["_removed"]).clip(lower=0)
    out["DAILY_AVG_EX_HOLIDAYS"] = safe_div_vec(out["MTH_BOARD"], out["DAYS_EX_HOLIDAYS"])
    return out.drop(columns="_removed")


def detect_negative_trends_12m(
    df_time: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """Flag routes with two‑month declines vs. 12‑month baseline.

    Returns one row per route flagged.
    """
    window = config.rolling_window
    pct_threshold = config.decline_thresh_pct
    min_coverage = config.min_coverage
    confirm_prev = config.require_two_months
    flags: list[dict[str, Any]] = []

    metrics = ["weekday_avg", "pph", "ppt", "ppm"]  # adjust if desired
    periods = config.timeline  # include missing calendar months as gaps

    for route, grp in df_time.groupby("route"):
        grp = grp.set_index("period").reindex(periods)  # align to master timeline

        for metric in metrics:
            vals = pd.to_numeric(grp[metric], errors="coerce")

            # Build exclusion mask
            excl_mask = [_is_excluded(p, str(route), config) for p in periods]
            vals = vals.mask(pd.Series(excl_mask, index=vals.index))  # convert excluded to NaN

            if vals.notna().sum() < window + 1:
                continue  # not enough data overall

            latest = vals.iloc[-1]
            prev = vals.iloc[-2]

            # Rolling baseline excludes latest month
            baseline_window = vals.iloc[-(window + 1) : -1]  # previous 12
            valid_fraction = baseline_window.notna().mean()

            if latest is None or pd.isna(latest) or valid_fraction < min_coverage:
                continue  # insufficient baseline

            baseline_mean = baseline_window.mean(skipna=True)
            pct_change = 100 * (latest - baseline_mean) / baseline_mean

            # Optionally require previous month below baseline too
            prev_ok = not confirm_prev or (
                prev is not None and not pd.isna(prev) and prev < baseline_mean
            )

            if pct_change <= -pct_threshold and prev_ok:
                flags.append(
                    {
                        "route": route,
                        "metric": metric,
                        "latest_value": latest,
                        "baseline_mean": baseline_mean,
                        "pct_change": pct_change,
                        "window_months": int(baseline_window.notna().sum()),
                    }
                )

    return pd.DataFrame(flags)


def write_trend_log(df_flags: pd.DataFrame, output_dir: Path) -> Path:
    """Write a plain‑text summary of flagged routes. Return the file path."""
    out_path = output_dir / "NegativeTrendFlags.txt"

    if df_flags.empty:
        logging.info("No negative trends detected.")
        # create/overwrite an empty file to avoid downstream errors
        out_path.write_text("# No negative trends detected.\n", encoding="utf-8")
        return out_path

    lines: list[str] = []
    header = (
        f"# Routes flagged for negative trends\n# Generated {datetime.now():%Y-%m-%d %H:%M}\n\n"
    )
    lines.append(header)

    for route, grp in df_flags.groupby("route"):
        lines.append(f"Route {route}:")
        for _, row in grp.iterrows():
            lines.append(
                f"  • {row['metric']} ↓ {abs(row['pct_change']):.1f}% "
                f"(latest = {row['latest_value']:.1f}, "
                f"baseline = {row['baseline_mean']:.1f})"
            )
        lines.append("")  # blank line between routes

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Trend log written → %s", out_path)
    return out_path


def weekday_holiday_counts(holidays: Iterable[datetime]) -> dict[str, int]:
    """Count weekday holidays per month.

    Returns a mapping ``period -> n`` where *period* is a ``"MMM-YYYY"`` key and
    *n* is the number of holidays falling on a weekday (Monday–Friday) that
    month. Holidays landing on a Saturday or Sunday are ignored.
    """
    counts: dict[str, int] = {}
    for holiday in holidays:
        if holiday.weekday() >= 5:  # Saturday=5, Sunday=6
            continue
        period = f"{MONTH_NAMES[holiday.month - 1]}-{holiday.year:04d}"
        counts[period] = counts.get(period, 0) + 1
    return counts


def summarize_service_days(
    all_data: pd.DataFrame,
    config: AppConfig,
    holiday_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build one systemwide row per month of reported service-day counts.

    The Weekday/Saturday/Sunday columns hold the representative (most common)
    ``DAYS`` value reported across routes for that service period — i.e. the
    number of service days filed for the month. They are also the denominators
    of the SYSTEMWIDE daily averages in :func:`build_monthly_timeseries`.
    ``Holidays`` is the number of weekday holidays configured for that month.
    """
    rows: list[dict[str, Any]] = []
    for period in config.ordered_periods:
        dfp = all_data[all_data["period"] == period]
        row: dict[str, Any] = {"period": period}
        for service_period in SERVICE_PERIODS:
            days = dfp.loc[dfp["SERVICE_PERIOD"] == service_period, "DAYS"].dropna()
            # The most frequently reported count ignores routes with partial-month
            # or no service. A tie goes to the larger (full-month) count rather than
            # to whichever route happens to be listed first.
            counts = days.value_counts()
            modal = counts[counts == counts.max()].index
            row[service_period] = int(modal.max()) if not days.empty else 0
        row["Holidays"] = int((holiday_counts or {}).get(period, 0))
        rows.append(row)
    return pd.DataFrame(rows, columns=pd.Index(["period", *SERVICE_PERIODS, "Holidays"]))


def build_monthly_timeseries(all_data: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Convert the year-to-date table into a plotting-ready time series.

    The result has one row per (period, route) plus a *SYSTEMWIDE* row,
    with these columns:

        period | route | total_ridership | weekday_avg | saturday_avg |
        sunday_avg | revenue_hours | trips | revenue_miles |
        pph | ppt | ppm

    SYSTEMWIDE daily averages divide each day type's total boardings by the
    month's service days from :func:`summarize_service_days`. Summing ``DAYS``
    across routes would instead give boardings per route-day, because every
    route reports the same calendar days.
    """
    # 1. Aggregate to (period, route, service_period)
    group_cols = ["period", "ROUTE_NAME", "SERVICE_PERIOD"]
    agg = all_data.groupby(group_cols, as_index=False).agg(
        {
            "MTH_BOARD": "sum",
            "DAYS": "sum",
            "MTH_REV_HOURS": "sum",
            "TOTAL_TRIPS": "sum",
            "MTH_REV_MILES": "sum",
        }
    )

    def _sum_pair(dfsub: pd.DataFrame, daytype: str) -> tuple[float, float]:
        row = dfsub.loc[dfsub["SERVICE_PERIOD"] == daytype]
        if not row.empty:
            # iat[0] returns a scalar that MyPy sees as very broad.
            # Explicit cast or float conversion helps.
            board_val = row["MTH_BOARD"].iloc[0]
            days_val = row["DAYS"].iloc[0]
            # Ensure we are converting something float-compatible
            return float(board_val), float(days_val)
        return 0.0, 0.0

    rows: list[dict[str, Any]] = []
    for (period, route), grp in agg.groupby(["period", "ROUTE_NAME"]):
        tr = grp["MTH_BOARD"].sum()
        hrs = grp["MTH_REV_HOURS"].sum()
        trips = grp["TOTAL_TRIPS"].sum()
        miles = grp["MTH_REV_MILES"].sum()

        wd_b, wd_d = _sum_pair(grp, "Weekday")
        sa_b, sa_d = _sum_pair(grp, "Saturday")
        su_b, su_d = _sum_pair(grp, "Sunday")

        rows.append(
            {
                "period": period,
                "route": route,
                "total_ridership": tr,
                "weekday_avg": safe_div(wd_b, wd_d),
                "saturday_avg": safe_div(sa_b, sa_d),
                "sunday_avg": safe_div(su_b, su_d),
                "revenue_hours": hrs,
                "trips": trips,
                "revenue_miles": miles,
                "pph": safe_div(tr, hrs),
                "ppt": safe_div(tr, trips),
                "ppm": safe_div(tr, miles, 3),
            }
        )

    df_time = pd.DataFrame(rows)

    # 2. Add systemwide rows
    service_days = summarize_service_days(all_data, config).set_index("period").to_dict("index")
    sys_rows: list[dict[str, Any]] = []
    for period in config.ordered_periods:
        dfp = df_time[df_time["period"] == period]
        tr = dfp["total_ridership"].sum()
        hrs = dfp["revenue_hours"].sum()
        trips = dfp["trips"].sum()
        miles = dfp["revenue_miles"].sum()
        days = service_days[period]

        agg_wd = agg[(agg["period"] == period) & (agg["SERVICE_PERIOD"] == "Weekday")]
        agg_sa = agg[(agg["period"] == period) & (agg["SERVICE_PERIOD"] == "Saturday")]
        agg_su = agg[(agg["period"] == period) & (agg["SERVICE_PERIOD"] == "Sunday")]

        sys_rows.append(
            {
                "period": period,
                "route": "SYSTEMWIDE",
                "total_ridership": tr,
                "weekday_avg": safe_div(agg_wd["MTH_BOARD"].sum(), days["Weekday"]),
                "saturday_avg": safe_div(agg_sa["MTH_BOARD"].sum(), days["Saturday"]),
                "sunday_avg": safe_div(agg_su["MTH_BOARD"].sum(), days["Sunday"]),
                "revenue_hours": hrs,
                "trips": trips,
                "revenue_miles": miles,
                "pph": safe_div(tr, hrs),
                "ppt": safe_div(tr, trips),
                "ppm": safe_div(tr, miles, 3),
            }
        )

    return pd.concat([df_time, pd.DataFrame(sys_rows)], ignore_index=True)


def plot_metric_over_time(df_time: pd.DataFrame, metric: str, config: AppConfig) -> None:
    """Create a line plot of *metric* by month for every route."""
    plot_dir = config.output_dir / "plots" / metric
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_m = df_time[["period", "route", metric]].copy()
    df_m[metric] = pd.to_numeric(df_m[metric], errors="coerce")
    # Month strings passed as x-values become Matplotlib categories, which logs
    # INFO messages for every chart. Plot against positions instead and label them
    # with a fixed locator/formatter; plt.xticks(ticks, labels) would also stretch
    # the x-axis to every month, even for a route with no data at the start or end.
    periods = config.timeline
    x_pos = range(len(periods))

    for route in sorted(df_m["route"].unique()):
        df_r = df_m[df_m["route"] == route]
        y_vals = df_r.set_index("period")[metric].reindex(periods).to_numpy(dtype=float)
        if pd.isna(y_vals).all():
            continue  # nothing to plot

        plt.figure(figsize=config.plot_style.figsize)
        plt.plot(
            x_pos,
            y_vals,
            marker=config.plot_style.marker,
            linestyle=config.plot_style.linestyle,
        )
        plt.gca().xaxis.set_major_locator(FixedLocator(x_pos))
        plt.gca().xaxis.set_major_formatter(FixedFormatter(periods))
        plt.title(f"{metric.replace('_', ' ').title()} – Route {route}")
        plt.xlabel("Month")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=config.plot_style.rotation)
        plt.grid(config.plot_style.grid)

        numeric = [v for v in y_vals if v is not None and not pd.isna(v)]
        if numeric:
            plt.ylim(0, max(numeric) * 1.1)

        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}_route_{route}.png", dpi=150)
        plt.close()


def generate_all_plots(df_time: pd.DataFrame, config: AppConfig) -> None:
    """Generate the plots enabled in the external configuration."""
    metric_map = {
        "plot_total_ridership": "total_ridership",
        "plot_weekday_avg": "weekday_avg",
        "plot_saturday_avg": "saturday_avg",
        "plot_sunday_avg": "sunday_avg",
        "plot_revenue_hours": "revenue_hours",
        "plot_trips": "trips",
        "plot_revenue_miles": "revenue_miles",
        "plot_pph": "pph",
        "plot_ppt": "ppt",
        "plot_ppm": "ppm",
    }
    for flag, col in metric_map.items():
        if config.plots[flag]:
            logging.info("Plotting %s …", col)
            plot_metric_over_time(df_time, col, config)


# =============================================================================
# RUN LOG
# =============================================================================


def resolve_source_file() -> Path | None:
    """Best-effort path to this script's source (``None`` in notebooks)."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def write_run_log(
    config: AppConfig,
    service_days: pd.DataFrame | None = None,
    *,
    status: str = "complete",
) -> bool:
    """Record the loaded external configuration and processing status.

    If *service_days* is provided (see :func:`summarize_service_days`), a table
    of the number of weekdays, Saturdays, Sundays, and weekday holidays in
    service for each month is appended so the counts can be verified against
    another source.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = config.output_dir / "ntd_monthly_summary_runlog.txt"

    source_file = resolve_source_file()
    source_display = str(source_file) if source_file else "<interactive>"

    lines: list[str] = [
        "=" * 72,
        RUN_LOG_TITLE,
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output directory: {config.output_dir}",
        f"Source script:    {source_display}",
        f"Run status:       {status}",
        f"Config file:      {config.source_path}",
        f"Config base directory: {config.config_base_dir or config.source_path.parent}",
        f"Configuration source: {'previous run log' if config.from_run_log else 'config file'}",
        f"Input directory:  {config.data_root}",
        f"Period order:     {', '.join(config.ordered_periods)}",
        "",
        "-" * 72,
        "CONFIGURATION (text loaded for this run)",
        "-" * 72,
        REPLAY_CONFIG_BEGIN,
        config.source_text,
        REPLAY_CONFIG_END,
    ]

    if service_days is not None and not service_days.empty:
        lines += [
            "",
            "-" * 72,
            "SERVICE-DAY COUNTS PER MONTH",
            "-" * 72,
            "Weekday/Saturday/Sunday = representative reported DAYS per month.",
            "Holidays = holidays falling on a weekday in the month.",
            "",
            service_days.to_string(index=False),
        ]

    lines.append("=" * 72)

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


# =============================================================================
# MAIN
# =============================================================================


def run_analysis(config: AppConfig) -> int:
    """Execute the original reports using one validated external configuration."""
    check_input_files(config)
    if config.from_run_log:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = config.output_dir.with_name(f"{config.output_dir.name}_rerun_{stamp}")
        config = replace(config, output_dir=output_dir)
        # Exclusive creation prevents a rerun from overwriting an existing run.
        config.output_dir.mkdir(parents=True, exist_ok=False)
        logging.info("Replaying configuration from %s", config.source_path)
        logging.info("New rerun output folder: %s", config.output_dir)
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    if not write_run_log(config, status="started") and config.require_run_log:
        return 1

    # === STEP 1: READ EXCEL FILES ============================================
    logging.info("=== STEP 1: READ EXCEL FILES ===")
    data_dict = read_excel_data(config)

    # === STEP 2: CLASSIFY & DERIVE ===========================================
    logging.info("\n=== STEP 2: CLASSIFY & DERIVE ===")
    for period, df in data_dict.items():
        df["service_type"] = df["ROUTE_NAME"].apply(lambda route: classify_route(route, config))
        df["corridors"] = df["ROUTE_NAME"].apply(lambda route: classify_corridor(route, config))
        if config.garage_assignments:
            garages = {
                route: garage
                for garage, routes in config.garage_assignments.items()
                for route in routes
            }
            df["garage"] = df["ROUTE_NAME"].map(garages).fillna("unknown")
        df = calculate_derived_columns(df)
        df["period"] = period
        data_dict[period] = df

    all_data = pd.concat(data_dict.values(), ignore_index=True)

    if all_data.empty:
        raise ValueError("No rows remain after the original workbook filters.")

    unknown = (
        all_data.loc[all_data["service_type"] == "unknown", "ROUTE_NAME"]
        .drop_duplicates()
        .sort_values()
    )
    if not unknown.empty:
        logging.info("Unclassified routes: %s", ", ".join(unknown))

    # === STEP 3: EXPORT COMPLETE DATASETS ====================================
    logging.info("\n=== STEP 3: EXPORT COMPLETE DATASETS ===")

    # 3.1  Master CSV (all periods) – single file for pipelines
    all_data.to_csv(
        config.output_dir / "DetailedAllPeriods_for_plotting.csv",
        index=False,
    )
    logging.info("Combined CSV exported.")

    # 3.2  Convenience workbook – one sheet per month (no giant sheet)
    with pd.ExcelWriter(config.output_dir / "MonthlySheets.xlsx") as xw:
        for period in config.ordered_periods:
            data_dict[period].to_excel(xw, sheet_name=period, index=False)
    logging.info("Monthly workbook exported.")

    # Weekday holidays per month, reused for the run log and the holiday-free
    # weekday summaries below.
    holiday_counts = weekday_holiday_counts(config.holidays)

    # === STEP 4: ROUTE-LEVEL SUMMARIES (ALL LOADED MONTHS) ==========================
    logging.info("\n=== STEP 4: ROUTE-LEVEL SUMMARIES ===")
    subsets = {
        "Combined": all_data,
        "Weekday": all_data[all_data["SERVICE_PERIOD"] == "Weekday"],
        "Saturday": all_data[all_data["SERVICE_PERIOD"] == "Saturday"],
        "Sunday": all_data[all_data["SERVICE_PERIOD"] == "Sunday"],
    }
    for label, subset in subsets.items():
        out = route_level_summary(subset)
        # The weekday summary also carries holiday-free columns when any weekday
        # holiday falls inside the covered months.
        if label == "Weekday":
            out = with_weekday_ex_holiday_columns(out, subset, holiday_counts)
        with pd.ExcelWriter(config.output_dir / f"RouteLevelSummary_{label}.xlsx") as xw:
            out.to_excel(xw, sheet_name=f"{label}_Route_Level", index=False)
        logging.info("%s summary exported.", label)

    # === STEP 5: TIME-SERIES PLOTS ===========================================
    logging.info("\n=== STEP 5: TIME-SERIES PLOTS ===")
    ts = build_monthly_timeseries(all_data, config)
    generate_all_plots(ts, config)

    # === STEP 5B: TREND FLAGGING (12‑mo baseline) ============================
    logging.info("\n=== TREND FLAGGING (12‑mo baseline) ===")
    flags = detect_negative_trends_12m(ts, config)
    write_trend_log(flags, config.output_dir)

    # === STEP 6: USER-DEFINED TIME WINDOWS ===================================
    logging.info("\n=== STEP 6: TIME-WINDOW OUTPUTS ===")
    for tw in config.time_windows:
        w_dir = config.output_dir / tw.label

        subset = slice_for_window(all_data, tw)
        if subset.empty:
            logging.info(
                "⚠︎ %s: no rows inside %s → %s",
                tw.label,
                tw.start.strftime("%Y-%m-%d"),
                tw.end.strftime("%Y-%m-%d"),
            )
            continue

        w_dir.mkdir(parents=True, exist_ok=True)
        expected = (tw.end.year - tw.start.year) * 12 + tw.end.month - tw.start.month + 1
        available = subset["period"].nunique()
        if available < expected:
            logging.warning(
                "%s is partial: %d of %d months available.", tw.label, available, expected
            )

        # 6.1  Raw slice (complete FY etc.) → CSV only
        subset.to_csv(
            w_dir / f"detailed_{tw.label}_for_plotting.csv",
            index=False,
        )

        # 6.2  Route-level summaries (Combined + day-type splits) → XLSX only
        subsets_tw = {
            "Combined": subset,
            "Weekday": subset[subset["SERVICE_PERIOD"] == "Weekday"],
            "Saturday": subset[subset["SERVICE_PERIOD"] == "Saturday"],
            "Sunday": subset[subset["SERVICE_PERIOD"] == "Sunday"],
        }
        for lbl, df_sub in subsets_tw.items():
            rl = route_level_summary(df_sub)
            rl.to_excel(
                w_dir / f"RouteLevelSummary_{lbl}.xlsx",
                index=False,
            )

        # 6.3  Service-type aggregation for the window → XLSX only
        st = aggregate_by_service_type(subset)
        st.to_excel(
            w_dir / f"AggByServiceType_{tw.label}.xlsx",
            index=False,
        )

        logging.info(
            "%s: %d rows → %s", tw.label, len(subset), w_dir.relative_to(config.output_dir)
        )

    service_days = summarize_service_days(all_data, config, holiday_counts)
    if not write_run_log(config, service_days) and config.require_run_log:
        logging.error(
            "Run log could not be written. Set [logging] require_run_log = false to "
            "suppress this error when a sidecar file is genuinely impossible."
        )
        return 1

    logging.info("All processing complete. Script completed successfully.")
    return 0


def main(config_file: str | Path = CONFIG_PATH, *, validate_only: bool = False) -> int:
    """Load a config and run; return 0 on success, 1 on analysis error, 2 on config error."""
    try:
        config = load_config(config_file)
    except ConfigError as exc:
        logging.error("Configuration error: %s", exc)
        return 2

    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Honor the configured level when an IDE or notebook already has handlers.
    logging.getLogger().setLevel(config.log_level)
    if validate_only:
        # print, not logging: the result must show even when [logging] level hides INFO.
        print(  # noqa: T201
            f"Configuration valid: {len(config.periods)} months, "
            f"{len(config.time_windows)} time windows. No workbook files were checked."
        )
        return 0
    try:
        return run_analysis(config)
    except Exception:
        logging.exception("NTD analysis failed.")
        return 1


def cli() -> int:
    """Parse shell arguments separately from programmatic main() calls."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="INI-style .txt config or previous run log (default: CONFIG_PATH near the top)",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate settings only; do not access workbooks or create output",
    )
    args = parser.parse_args()
    return main(args.config, validate_only=args.validate_config)


def launched_by_notebook() -> bool:
    """Recognize a kernel's arguments while retaining explicit %run script arguments."""
    # %run changes argv[0] to the script name, so its --config arguments still work.
    launcher = PureWindowsPath(sys.argv[0]).name if sys.argv else ""
    return launcher in {"ipykernel_launcher.py", "ipykernel_launcher"}


if __name__ == "__main__":
    if launched_by_notebook():
        # Kernel arguments (including -f kernel-....json) do not belong to this script.
        # Calling main directly also avoids a SystemExit traceback in a notebook cell.
        main(CONFIG_PATH)
    else:
        raise SystemExit(cli())
