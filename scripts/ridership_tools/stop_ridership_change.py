"""Compare stop-level ridership across two or more signups and flag added or removed stops.

Reads the same ``RIDERSHIP_BY_ROUTE_AND_STOP_(ALL_TIME_PERIODS)`` export that
``data_request_by_stop_processor.py`` processes, but several of them (one per signup,
listed in temporal order), and reports how ridership changed from each signup to the
next and from the earliest to the latest. Every input is filtered and summed across
time periods the same way before the signups are lined up on the comparison key.

The comparison grain is a switch: ``AGGREGATE_ROUTES_TOGETHER = True`` sums every route
at a stop and compares by STOP_ID; ``False`` compares each (ROUTE_NAME, STOP_ID) pair.
A key is "present" in a signup when that export has a row for it (or, with
``TREAT_ZERO_AS_ABSENT``, a row with nonzero ridership). Changes are computed only
between signups where the key is present, so a stop that appears or disappears is
labelled added / removed / intermittent instead of showing a +/-100% swing.

Inputs
------
- Two or more ridership exports in temporal order (``.xlsx``/``.xls`` first sheet, or
  ``.csv``) with columns ROUTE_NAME, STOP, STOP_ID, BOARD_ALL, ALIGHT_ALL, plus
  TIME_PERIOD when ``TIME_PERIODS`` is set.

Outputs
-------
- ``<OUTPUT_FILENAME stem>_<stop|route_stop>.xlsx`` in ``OUTPUT_DIR``, with sheets:
  ``Change Summary`` (totals, added/removed counts and % change per transition, for
  all keys and for keys present in both signups), ``Signup Totals``, one sheet per
  metric (value per signup, change and % change for each consecutive pair and for
  first to last, plus a presence STATUS), ``Added & Removed`` and ``Long`` (optional).
- A ``_runlog.txt`` sidecar with the verbatim CONFIGURATION block, the settings
  actually used (including CLI overrides), and a SHA-256 hash of each input file.

Limitations
-----------
- "Added" / "removed" only means a key gained or lost rows in the export. A renumbered
  STOP_ID, a renamed route, or a sampling gap looks the same. Cross-check stop changes
  against the GTFS feeds with ``stop_analysis/gtfs_stop_diff.py``.

Typical usage
-------------
Update INPUT_FILES (earliest first) and OUTPUT_DIR in the CONFIGURATION section (or pass
``--inputs`` and ``--output-dir``) and run from a shell, ArcGIS Pro's Python window, or a
Jupyter notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

# Sentinel markers used by write_run_log to identify the configuration block within
# this file's source. Each string must appear exactly once in this file as a
# stand-alone comment line (other than these constant definitions themselves).
CONFIG_BEGIN_MARKER: str = "# === BEGIN CONFIG ==="
CONFIG_END_MARKER: str = "# === END CONFIG ==="

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# Ridership exports to compare, in TEMPORAL ORDER (earliest first), one per signup
# (months or any other periods work too). Each is the same
# RIDERSHIP_BY_ROUTE_AND_STOP_(ALL_TIME_PERIODS) export that
# data_request_by_stop_processor.py reads: .xlsx/.xls (first sheet) or .csv.
INPUT_FILES: List[Path] = [
    Path(r"Path\To\Signup_1\RIDERSHIP_BY_ROUTE_AND_STOP_(ALL_TIME_PERIODS).XLSX"),
    Path(r"Path\To\Signup_2\RIDERSHIP_BY_ROUTE_AND_STOP_(ALL_TIME_PERIODS).XLSX"),
]

# Short label for each input, same order and length as INPUT_FILES (e.g. "Fall 2025").
# Labels become column headers. Empty ⇒ use the file names, or the parent folder names
# when the file names repeat, or "Signup 1", "Signup 2", ... as a last resort.
SIGNUP_LABELS: List[str] = []

OUTPUT_DIR: Path = Path(r"Path\To\Output\Folder")
# The comparison level is appended to the stem (ridership_change_stop.xlsx or
# ridership_change_route_stop.xlsx), so the two kinds of run don't overwrite each other.
OUTPUT_FILENAME: str = r"ridership_change.xlsx"

# Comparison grain:
#   True  → stop only: all routes at a stop are summed first and stops are compared by
#           STOP_ID. A ROUTES column lists every route seen at the stop.
#   False → route × stop: each (ROUTE_NAME, STOP_ID) pair is compared separately.
AGGREGATE_ROUTES_TOGETHER: bool = True

# Measures to compare; each gets its own sheet. TOTAL = BOARD_ALL + ALIGHT_ALL.
METRICS: List[str] = ["BOARD_ALL", "ALIGHT_ALL", "TOTAL"]

# Filters applied identically to every input before summing.
# ROUTES = keep-only list   |  ROUTES_EXCLUDE = toss-out list   (empty → no filter)
ROUTES: List[str] = []
ROUTES_EXCLUDE: List[str] = []
STOP_IDS: List[int] = []  # keep these (empty → keep all)
# Optional plain-text file of stop IDs (newline/comma/space separated, "#" comments).
# When set it replaces the inline STOP_IDS list.
STOP_IDS_FILE: Path | None = None  # e.g. Path(r"C:\Data\stop_ids.txt")
# Keep only these TIME_PERIOD values (e.g. ["AM PEAK", "PM PEAK"]). Empty → all periods.
TIME_PERIODS: List[str] = []

# A % change is reported only when the earlier value is at least this large; below it
# the % cell is left blank while the absolute change is still shown. Guards against
# small-base swings such as 1 → 4 = +300%. 0 ⇒ blank only when the earlier value is 0.
MIN_BASE_FOR_PCT: float = 0.0

# When a stop (or route × stop pair) counts as present in a signup:
#   False → whenever that export has a row for it, even with zero ridership.
#   True  → only when its BOARD_ALL + ALIGHT_ALL is above zero. Use this if your exports
#           list every stop on a route's pattern, including stops nobody used.
TREAT_ZERO_AS_ABSENT: bool = False

# Write an "Added & Removed" sheet: one row per key whose presence changes between
# consecutive signups, with its ridership in the signup where it is present.
FLAG_ADDED_REMOVED: bool = True

# Write a "Long" sheet (one row per key × signup) for pivot tables and charts.
EXPORT_LONG_SHEET: bool = True

# If True → round ridership and changes to 1 decimal place and % changes to 2.
APPLY_ROUNDING: bool = True

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# When True, a failed run-log write aborts the script so the analyst is never left
# with an output workbook that lacks a matching configuration record.
REQUIRE_RUN_LOG: bool = True

# === END CONFIG ===

REQUIRED_COLUMNS: Tuple[str, ...] = ("ROUTE_NAME", "STOP", "STOP_ID", "BOARD_ALL", "ALIGHT_ALL")
VALID_METRICS: Tuple[str, ...] = ("BOARD_ALL", "ALIGHT_ALL", "TOTAL")
METRIC_NAMES: Dict[str, str] = {
    "BOARD_ALL": "Boardings",
    "ALIGHT_ALL": "Alightings",
    "TOTAL": "Total (Board+Alight)",
}

STATUS_ALL: str = "present in all"
STATUS_ADDED: str = "added"
STATUS_REMOVED: str = "removed"
STATUS_INTERMITTENT: str = "intermittent"

# Column names the script writes next to the signup labels; a label may not reuse one.
RESERVED_COLUMNS: frozenset = frozenset(
    {
        "ROUTE_NAME",
        "STOP_ID",
        "STOP",
        "ROUTES",
        "ROUTES_CHANGED",
        "STATUS",
        "SIGNUPS_PRESENT",
        "FIRST_SIGNUP",
        "LAST_SIGNUP",
    }
)

EXCEL_MAX_ROWS: int = 1_048_576
PLACEHOLDER_PREFIX: str = "Path\\To\\"

# =============================================================================
# FUNCTIONS
# =============================================================================


def key_columns(aggregate_routes_together: bool) -> List[str]:
    """Return the columns that identify one comparison key."""
    return ["STOP_ID"] if aggregate_routes_together else ["ROUTE_NAME", "STOP_ID"]


def comparison_level(aggregate_routes_together: bool) -> str:
    """Return ``"stop"`` or ``"route_stop"`` for file names and logs."""
    return "stop" if aggregate_routes_together else "route_stop"


def _natural_key(value: Any) -> Tuple[Any, ...]:
    """Sort key that orders embedded numbers numerically ("2" < "10" < "10A")."""
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))
    )


def normalize_id(values: pd.Series) -> pd.Series:
    """Normalize route or stop identifiers to stripped strings.

    Excel can hand the same ID back as ``1001``, ``1001.0`` or ``" 1001"`` depending on
    the rest of the column, which would break the join between signups. Missing and
    blank values become NaN.

    Args:
        values: Raw identifier column.

    Returns:
        An object Series of strings (NaN where missing), with a trailing ``.0`` removed
        from integer-valued numbers.
    """
    text = pd.Series(
        [None if pd.isna(v) else str(v).strip() for v in values],
        index=values.index,
        dtype=object,
    )
    text = text.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    return text.where(text.notna() & (text != ""))


def derive_signup_labels(input_files: Sequence[Path], labels: Sequence[str]) -> List[str]:
    """Return one unique, non-blank label per input file.

    Args:
        input_files: Input paths in temporal order.
        labels: User-supplied labels; empty to derive them from the paths.

    Returns:
        A list of labels the same length as ``input_files``.

    Raises:
        ValueError: If supplied labels don't match the inputs one-to-one, repeat, are
            blank, or collide with an output column name.
    """
    if labels:
        cleaned = [str(label).strip() for label in labels]
        if len(cleaned) != len(input_files):
            raise ValueError(
                f"SIGNUP_LABELS has {len(cleaned)} label(s) but INPUT_FILES has "
                f"{len(input_files)} file(s); give one label per file or leave it empty."
            )
        if any(not label for label in cleaned):
            raise ValueError("SIGNUP_LABELS contains a blank label.")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"SIGNUP_LABELS must be unique; got {cleaned}.")
        clashes = sorted(set(cleaned) & RESERVED_COLUMNS)
        if clashes:
            raise ValueError(f"SIGNUP_LABELS may not reuse output column names: {clashes}.")
        return cleaned

    for candidate in ([p.stem for p in input_files], [p.parent.name for p in input_files]):
        if all(candidate) and len(set(candidate)) == len(candidate):
            if not set(candidate) & RESERVED_COLUMNS:
                return list(candidate)
    return [f"Signup {i}" for i in range(1, len(input_files) + 1)]


def load_stop_ids_from_file(stop_ids_file: Path) -> List[int]:
    """Read a plain-text list of integer stop IDs from *stop_ids_file*.

    The file may separate IDs by newlines, commas, or whitespace. Blank lines and lines
    beginning with ``#`` are ignored. Non-integer tokens are skipped with a warning.
    Duplicates are collapsed while preserving first-seen order.

    Args:
        stop_ids_file: Path to the text file of stop IDs.

    Returns:
        A list of unique integer stop IDs in first-seen order.

    Raises:
        ValueError: If the file cannot be read or contains no valid integer stop IDs.
    """
    try:
        raw_text: str = stop_ids_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Error reading STOP_IDS_FILE '{stop_ids_file}': {exc}") from exc

    stop_ids: List[int] = []
    bad_tokens: List[str] = []
    seen: set[int] = set()
    for raw_line in raw_text.splitlines():
        line: str = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for token in line.replace(",", " ").split():
            try:
                value: int = int(token)
            except ValueError:
                bad_tokens.append(token)
                continue
            if value not in seen:
                seen.add(value)
                stop_ids.append(value)

    if bad_tokens:
        logging.warning(
            "Ignored %d non-integer token(s) in STOP_IDS_FILE '%s': %s",
            len(bad_tokens),
            stop_ids_file,
            bad_tokens,
        )
    if not stop_ids:
        raise ValueError(f"STOP_IDS_FILE '{stop_ids_file}' contained no valid integer stop IDs.")

    logging.info("Loaded %d stop ID(s) from STOP_IDS_FILE '%s'.", len(stop_ids), stop_ids_file)
    return stop_ids


def resolve_stop_ids(stop_ids: Sequence[int], stop_ids_file: Path | None) -> List[int]:
    """Return the effective STOP_IDS filter; ``stop_ids_file`` wins when both are set."""
    if stop_ids_file is not None:
        loaded: List[int] = load_stop_ids_from_file(stop_ids_file)
        if stop_ids:
            logging.warning(
                "Both STOP_IDS and STOP_IDS_FILE are set; using STOP_IDS_FILE "
                "and ignoring the inline STOP_IDS list."
            )
        return loaded
    return list(stop_ids)


def read_ridership_table(path: Path) -> pd.DataFrame:
    """Load one ridership export (first sheet of an Excel workbook, or a CSV).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be parsed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: '{path}'.")
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_excel(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read '{path}': {exc}") from exc


def prepare_signup(
    raw: pd.DataFrame,
    *,
    source: str,
    routes: Sequence[str] = (),
    routes_exclude: Sequence[str] = (),
    stop_ids: Sequence[Any] = (),
    time_periods: Sequence[str] = (),
) -> pd.DataFrame:
    """Validate, clean and filter one signup's ridership rows.

    Identifiers are normalized with :func:`normalize_id`; rows without a ROUTE_NAME or
    STOP_ID (e.g. grand-total rows) are dropped; non-numeric ridership becomes 0. The
    route, stop and time-period filters are then applied.

    Args:
        raw: Rows as read from the export.
        source: Name used in log messages and errors.
        routes: Route names to keep (empty → all).
        routes_exclude: Route names to drop.
        stop_ids: Stop IDs to keep (empty → all).
        time_periods: TIME_PERIOD values to keep, case-insensitive (empty → all).

    Returns:
        A DataFrame with ROUTE_NAME, STOP, STOP_ID, BOARD_ALL, ALIGHT_ALL and TOTAL.

    Raises:
        ValueError: If a required column is missing.
    """
    required: List[str] = list(REQUIRED_COLUMNS) + (["TIME_PERIOD"] if time_periods else [])
    missing: List[str] = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"'{source}' is missing required column(s): {missing}")

    df: pd.DataFrame = raw[required].copy()
    df["ROUTE_NAME"] = normalize_id(df["ROUTE_NAME"])
    df["STOP_ID"] = normalize_id(df["STOP_ID"])
    stop_names = pd.Series(
        [None if pd.isna(v) else str(v).strip() for v in df["STOP"]], index=df.index, dtype=object
    )
    df["STOP"] = stop_names.where(stop_names.notna() & (stop_names != ""))

    no_key = df["ROUTE_NAME"].isna() | df["STOP_ID"].isna()
    if no_key.any():
        logging.warning(
            "%s: dropped %d row(s) with no ROUTE_NAME or STOP_ID (e.g. total rows).",
            source,
            int(no_key.sum()),
        )
        df = df.loc[~no_key].copy()

    for col in ("BOARD_ALL", "ALIGHT_ALL"):
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = numeric.isna() & df[col].notna()
        if bad.any():
            logging.warning(
                "%s: %d non-numeric %s value(s) treated as 0.", source, int(bad.sum()), col
            )
        df[col] = numeric.fillna(0.0).astype(float)

    if routes:
        df = df.loc[df["ROUTE_NAME"].isin(set(normalize_id(pd.Series(list(routes)))))]
    if routes_exclude:
        df = df.loc[~df["ROUTE_NAME"].isin(set(normalize_id(pd.Series(list(routes_exclude)))))]
    if stop_ids:
        df = df.loc[df["STOP_ID"].isin(set(normalize_id(pd.Series(list(stop_ids)))))]
    if time_periods:
        wanted = {str(p).strip().upper() for p in time_periods}
        period_text = pd.Series(
            [str(v).strip().upper() for v in df["TIME_PERIOD"]], index=df.index, dtype=object
        )
        df = df.loc[period_text.isin(wanted)]
        df = df.drop(columns=["TIME_PERIOD"])

    if df.empty:
        logging.warning("%s: no rows left after filtering.", source)

    df = df.assign(TOTAL=df["BOARD_ALL"] + df["ALIGHT_ALL"])
    return df.reset_index(drop=True)


def _join_routes(values: Any) -> str:
    """Join unique, non-null route names in natural order."""
    unique = {str(v) for v in pd.Series(values).dropna()}
    return ", ".join(sorted(unique, key=_natural_key))


def _union_route_lists(values: Any) -> str:
    """Union several ``", "``-joined route lists into one, in natural order."""
    routes: set[str] = set()
    for value in pd.Series(values).dropna():
        routes.update(part for part in str(value).split(", ") if part)
    return ", ".join(sorted(routes, key=_natural_key))


def aggregate_signup(df: pd.DataFrame, *, aggregate_routes_together: bool) -> pd.DataFrame:
    """Sum one signup's rows (all time periods) to one row per comparison key.

    Args:
        df: Output of :func:`prepare_signup`.
        aggregate_routes_together: True → one row per STOP_ID with a ``ROUTES`` column;
            False → one row per (ROUTE_NAME, STOP_ID).

    Returns:
        Key columns plus STOP, BOARD_ALL, ALIGHT_ALL, TOTAL (and ROUTES at stop level).
    """
    named: Dict[str, Tuple[str, Any]] = {
        "STOP": ("STOP", "first"),
        "BOARD_ALL": ("BOARD_ALL", "sum"),
        "ALIGHT_ALL": ("ALIGHT_ALL", "sum"),
        "TOTAL": ("TOTAL", "sum"),
    }
    if aggregate_routes_together:
        named["ROUTES"] = ("ROUTE_NAME", _join_routes)
    return df.groupby(key_columns(aggregate_routes_together), as_index=False, sort=False).agg(
        **named
    )


def build_panel(
    signup_tables: Sequence[pd.DataFrame],
    key_cols: Sequence[str],
    *,
    treat_zero_as_absent: bool = False,
) -> pd.DataFrame:
    """Stack per-signup aggregates into a complete key × signup panel.

    Every key seen in any signup gets one row per signup. ``PRESENT`` is False (and the
    ridership columns NaN) for signups where the key has no row, or, with
    ``treat_zero_as_absent``, where its TOTAL is zero.

    Args:
        signup_tables: Outputs of :func:`aggregate_signup`, in temporal order.
        key_cols: Comparison key columns.
        treat_zero_as_absent: Treat zero-ridership rows as absent.

    Returns:
        A DataFrame with key columns, ``SIGNUP_INDEX`` (0-based), ``PRESENT`` and the
        aggregated columns.

    Raises:
        ValueError: If no signup has any rows left.
    """
    keys: List[str] = list(key_cols)
    frames: List[pd.DataFrame] = []
    for idx, table in enumerate(signup_tables):
        frame = table.copy()
        if treat_zero_as_absent:
            frame = frame.loc[frame["TOTAL"] > 0]
        frame["SIGNUP_INDEX"] = idx
        frame["PRESENT"] = True
        frames.append(frame)

    observed: pd.DataFrame = pd.concat(frames, ignore_index=True)
    if observed.empty:
        raise ValueError("No ridership rows are left in any signup after filtering.")

    grid: pd.DataFrame = (
        observed[keys]
        .drop_duplicates()
        .merge(pd.DataFrame({"SIGNUP_INDEX": range(len(signup_tables))}), how="cross")
    )
    panel: pd.DataFrame = grid.merge(observed, on=keys + ["SIGNUP_INDEX"], how="left")
    panel["PRESENT"] = panel["PRESENT"].eq(True)
    return panel


def _wide(panel: pd.DataFrame, key_cols: Sequence[str], column: str) -> pd.DataFrame:
    """Pivot ``column`` to one row per key and one column per SIGNUP_INDEX."""
    # pivot, not pivot_table: the panel has exactly one row per (key, signup), and
    # pivot_table(dropna=False) would expand a two-column key into every route × stop combo.
    return panel.pivot(index=list(key_cols), columns="SIGNUP_INDEX", values=column)  # noqa: PD010


def classify_presence(flags: Sequence[bool]) -> str:
    """Label a key's presence pattern across signups (in temporal order).

    Returns:
        ``"present in all"``; ``"added"`` (absent at first, then present through the
        last signup); ``"removed"`` (present at first, then absent through the last);
        otherwise ``"intermittent"``.
    """
    pattern = "".join("1" if flag else "0" for flag in flags)
    if re.fullmatch(r"1+", pattern):
        return STATUS_ALL
    if re.fullmatch(r"0+1+", pattern):
        return STATUS_ADDED
    if re.fullmatch(r"1+0+", pattern):
        return STATUS_REMOVED
    return STATUS_INTERMITTENT


def pct_change(old: pd.Series, new: pd.Series, min_base: float = 0.0) -> pd.Series:
    """Percent change from ``old`` to ``new`` (in percent units, e.g. 12.5).

    NaN where either side is missing, where ``old`` is zero, or where ``old`` is below
    ``min_base``.
    """
    valid = old.notna() & new.notna() & (old > 0) & (old >= min_base)
    return (new - old) / old.where(valid) * 100.0


def _pct_scalar(old: float, new: float) -> float:
    """Percent change between two totals; NaN when ``old`` is not positive."""
    return (new - old) / old * 100.0 if old > 0 else float("nan")


def _comparison_pairs(n_signups: int) -> List[Tuple[int, int]]:
    """Consecutive (from, to) index pairs, plus (first, last) when there are 3+ signups."""
    pairs = [(i - 1, i) for i in range(1, n_signups)]
    if n_signups > 2:
        pairs.append((0, n_signups - 1))
    return pairs


def build_key_attributes(
    panel: pd.DataFrame, key_cols: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    """Build the descriptive columns shared by every metric sheet.

    Returns:
        A DataFrame indexed by the key columns with STOP (name from the latest signup
        that has one), ROUTES and ROUTES_CHANGED at stop level, STATUS,
        SIGNUPS_PRESENT, FIRST_SIGNUP and LAST_SIGNUP.
    """
    keys: List[str] = list(key_cols)
    present: pd.DataFrame = _wide(panel, keys, "PRESENT").astype(bool)
    seen: pd.DataFrame = panel.loc[panel["PRESENT"]].sort_values("SIGNUP_INDEX")
    grouped = seen.groupby(keys)

    attrs = pd.DataFrame(index=present.index)
    attrs["STOP"] = grouped["STOP"].last()
    if "ROUTES" in panel.columns:
        attrs["ROUTES"] = grouped["ROUTES"].agg(_union_route_lists)
        attrs["ROUTES_CHANGED"] = grouped["ROUTES"].agg(lambda s: len(set(s.dropna())) > 1)
    attrs["STATUS"] = [classify_presence(row) for row in present.to_numpy().tolist()]
    attrs["SIGNUPS_PRESENT"] = present.sum(axis=1).astype(int)
    label_list: List[str] = list(labels)
    attrs["FIRST_SIGNUP"] = [label_list[i] for i in present.to_numpy().argmax(axis=1)]
    last_idx = present.shape[1] - 1 - present.to_numpy()[:, ::-1].argmax(axis=1)
    attrs["LAST_SIGNUP"] = [label_list[i] for i in last_idx]
    return attrs


def build_metric_table(
    panel: pd.DataFrame,
    attributes: pd.DataFrame,
    metric: str,
    labels: Sequence[str],
    key_cols: Sequence[str],
    *,
    min_base: float = 0.0,
    apply_rounding: bool = True,
) -> pd.DataFrame:
    """Build one metric's wide change table (one row per key).

    Columns: the key columns and :func:`build_key_attributes` columns, the metric's
    value in each signup (blank where absent), then ``Chg <a> to <b>`` and
    ``% Chg <a> to <b>`` for each consecutive pair and, with 3+ signups, first to last.

    Returns:
        The table in natural key order.
    """
    keys: List[str] = list(key_cols)
    values: pd.DataFrame = _wide(panel, keys, metric)
    out: pd.DataFrame = attributes.copy()
    for idx, label in enumerate(labels):
        out[label] = values[idx]

    for a, b in _comparison_pairs(len(labels)):
        old, new = values[a], values[b]
        out[f"Chg {labels[a]} to {labels[b]}"] = new - old
        out[f"% Chg {labels[a]} to {labels[b]}"] = pct_change(old, new, min_base)

    if apply_rounding:
        for col in out.columns:
            if str(col).startswith("% Chg "):
                out[col] = out[col].round(2)
            elif col in labels or str(col).startswith("Chg "):
                out[col] = out[col].round(1)

    return sort_by_key(out.reset_index(), keys)


def build_added_removed(
    panel: pd.DataFrame,
    key_cols: Sequence[str],
    labels: Sequence[str],
    *,
    apply_rounding: bool = True,
) -> pd.DataFrame:
    """List keys whose presence changes between consecutive signups.

    Each row is one key in one transition: ``CHANGE`` is ``"added"`` (absent in the
    earlier signup, present in the later) or ``"removed"`` (the reverse), and the
    ridership columns come from the signup where the key is present (``RIDERSHIP_FROM``).
    At route × stop level, ``STOP_IN_BOTH_SIGNUPS`` is True when the STOP_ID itself
    appears (on any route) in both signups: a route-pattern change rather than a new
    or closed stop.

    Returns:
        One row per (transition, key); empty (with headers) when nothing changed.
    """
    keys: List[str] = list(key_cols)
    route_level: bool = "ROUTE_NAME" in keys
    present: pd.DataFrame = _wide(panel, keys, "PRESENT").astype(bool)
    stops_by_signup: Dict[int, set] = {
        int(idx): set(group["STOP_ID"])
        for idx, group in panel.loc[panel["PRESENT"]].groupby("SIGNUP_INDEX")
    }

    detail_cols: List[str] = [
        c for c in ("STOP", "ROUTES", "BOARD_ALL", "ALIGHT_ALL", "TOTAL") if c in panel.columns
    ]
    lead_cols: List[str] = ["TRANSITION", "CHANGE"] + keys
    tail_cols: List[str] = ["RIDERSHIP_FROM"] + (["STOP_IN_BOTH_SIGNUPS"] if route_level else [])
    columns: List[str] = lead_cols + detail_cols[:1] + tail_cols + detail_cols[1:]

    pieces: List[pd.DataFrame] = []
    for b in range(1, len(labels)):
        a = b - 1
        flips = (
            (STATUS_ADDED, present[b] & ~present[a], b),
            (STATUS_REMOVED, present[a] & ~present[b], a),
        )
        for change, mask, source_idx in flips:
            changed_keys: pd.DataFrame = mask[mask].index.to_frame(index=False)
            if changed_keys.empty:
                continue
            rows = changed_keys.merge(
                panel.loc[panel["SIGNUP_INDEX"] == source_idx], on=keys, how="left"
            )
            rows["TRANSITION"] = f"{labels[a]} to {labels[b]}"
            rows["CHANGE"] = change
            rows["RIDERSHIP_FROM"] = labels[source_idx]
            if route_level:
                in_both = stops_by_signup.get(a, set()) & stops_by_signup.get(b, set())
                rows["STOP_IN_BOTH_SIGNUPS"] = rows["STOP_ID"].isin(in_both)
            pieces.append(sort_by_key(rows, keys)[columns])

    if not pieces:
        return pd.DataFrame(columns=columns)
    out = pd.concat(pieces, ignore_index=True)
    if apply_rounding:
        for col in ("BOARD_ALL", "ALIGHT_ALL", "TOTAL"):
            if col in out.columns:
                out[col] = out[col].round(1)
    return out


def build_change_summary(
    panel: pd.DataFrame,
    key_cols: Sequence[str],
    labels: Sequence[str],
    metrics: Sequence[str],
    *,
    apply_rounding: bool = True,
) -> pd.DataFrame:
    """Summarize each transition: key counts, added/removed, and total % change.

    ``% Chg`` compares the sum over all keys present in each signup; ``% Chg (in both)``
    compares only keys present in both signups, so additions and removals don't drive
    it.

    Returns:
        One row per consecutive transition, plus first to last with 3+ signups.
    """
    keys: List[str] = list(key_cols)
    present: pd.DataFrame = _wide(panel, keys, "PRESENT").astype(bool)
    values: Dict[str, pd.DataFrame] = {m: _wide(panel, keys, m).fillna(0.0) for m in metrics}
    n: int = len(labels)

    rows: List[Dict[str, Any]] = []
    for a, b in _comparison_pairs(n):
        pa, pb = present[a], present[b]
        both = pa & pb
        row: Dict[str, Any] = {
            "From": labels[a],
            "To": labels[b],
            "Comparison": "first to last" if (b - a) > 1 else "consecutive",
            "Keys From": int(pa.sum()),
            "Keys To": int(pb.sum()),
            "Keys in Both": int(both.sum()),
            "Added": int((pb & ~pa).sum()),
            "Removed": int((pa & ~pb).sum()),
        }
        for metric in metrics:
            name = METRIC_NAMES[metric]
            v = values[metric]
            total_a, total_b = float(v[a][pa].sum()), float(v[b][pb].sum())
            both_a, both_b = float(v[a][both].sum()), float(v[b][both].sum())
            row[f"{name} From"] = total_a
            row[f"{name} To"] = total_b
            row[f"{name} % Chg"] = _pct_scalar(total_a, total_b)
            row[f"{name} % Chg (in both)"] = _pct_scalar(both_a, both_b)
        rows.append(row)

    out = pd.DataFrame(rows)
    if apply_rounding and not out.empty:
        for col in out.columns:
            if "% Chg" in col:
                out[col] = out[col].round(2)
            elif col.endswith((" From", " To")) and not col.startswith("Keys"):
                out[col] = out[col].round(1)
    return out


def build_signup_totals(
    panel: pd.DataFrame,
    key_cols: Sequence[str],
    labels: Sequence[str],
    input_files: Sequence[Path],
    *,
    apply_rounding: bool = True,
) -> pd.DataFrame:
    """One row per signup: source file, keys and stops present, and ridership totals."""
    route_level: bool = "ROUTE_NAME" in key_cols
    rows: List[Dict[str, Any]] = []
    for idx, label in enumerate(labels):
        seen = panel.loc[(panel["SIGNUP_INDEX"] == idx) & panel["PRESENT"]]
        row: Dict[str, Any] = {
            "Signup #": idx + 1,
            "Signup": label,
            "Source File": str(input_files[idx]),
            "Stops Present": int(seen["STOP_ID"].nunique()),
        }
        if route_level:
            row["Route-Stop Pairs Present"] = len(seen)
            row["Routes Present"] = int(seen["ROUTE_NAME"].nunique())
        for metric in VALID_METRICS:
            total = float(seen[metric].sum())
            row[METRIC_NAMES[metric]] = round(total, 1) if apply_rounding else total
        rows.append(row)
    return pd.DataFrame(rows)


def build_long_table(
    panel: pd.DataFrame,
    key_cols: Sequence[str],
    labels: Sequence[str],
    *,
    apply_rounding: bool = True,
) -> pd.DataFrame:
    """Return the panel as a tidy key × signup table with labels in place of indexes."""
    keys: List[str] = list(key_cols)
    out = panel.copy()
    # Absent rows carry no STOP name; borrow the key's latest one so every row is labelled.
    out["STOP"] = out["STOP"].fillna(out.groupby(keys)["STOP"].transform("last"))
    out["SIGNUP"] = [labels[int(i)] for i in out["SIGNUP_INDEX"]]
    out["SIGNUP #"] = out["SIGNUP_INDEX"].astype(int) + 1
    cols = keys + [c for c in ("STOP", "ROUTES") if c in out.columns]
    cols += ["SIGNUP #", "SIGNUP", "PRESENT", "BOARD_ALL", "ALIGHT_ALL", "TOTAL"]
    out = sort_by_key(out, keys + ["SIGNUP #"])[cols]
    if apply_rounding:
        for col in ("BOARD_ALL", "ALIGHT_ALL", "TOTAL"):
            out[col] = out[col].round(1)
    return out


def sort_by_key(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Sort rows by ``columns`` in natural order (route "2" before "10")."""
    column_values: List[List[Any]] = [df[c].tolist() for c in columns]
    order = sorted(
        range(len(df)),
        key=lambda i: tuple(_natural_key(values[i]) for values in column_values),
    )
    return df.iloc[order].reset_index(drop=True)


def restore_numeric_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Write all-digit ROUTE_NAME / STOP_ID columns back out as integers.

    IDs are compared as text, but Excel users expect numeric IDs to stay numeric so
    they can join or filter against other tables. Columns holding any ID with a
    leading zero (e.g. "0123") stay text so the zero isn't lost.
    """
    out = df.copy()
    for col in ("ROUTE_NAME", "STOP_ID"):
        if col not in out.columns or out.empty:
            continue
        text = out[col].dropna().astype(str)
        if not text.empty and text.str.fullmatch(r"-?(?:0|[1-9]\d*)").all():
            out[col] = pd.to_numeric(out[col]).astype("Int64")
    return out


def write_workbook(output_file: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    """Write each DataFrame to its own sheet, then bold headers and size columns.

    Raises:
        OSError: If the workbook cannot be written (e.g. it is open in Excel).
    """
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for name, df in sheets.items():
            restore_numeric_ids(df).to_excel(writer, sheet_name=name, index=False)
    format_workbook(output_file)


def format_workbook(output_file: Path, sample_rows: int = 200, max_width: int = 50) -> None:
    """Bold the header row, freeze it, and size columns from the first rows of each sheet."""
    workbook = load_workbook(output_file)
    for sheet in workbook.worksheets:
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        sheet.freeze_panes = "A2"
        for col_idx, column_cells in enumerate(
            sheet.iter_cols(max_row=min(sheet.max_row, sample_rows + 1)), start=1
        ):
            longest = max(
                (len(str(c.value)) for c in column_cells if c.value is not None), default=0
            )
            sheet.column_dimensions[get_column_letter(col_idx)].width = min(longest + 2, max_width)
    workbook.save(output_file)


def extract_config_block_from_text(source_text: str, source_label: str) -> str:
    r"""Return the text between the CONFIG markers in *source_text*.

    Slices out the lines strictly *between* the first occurrence of
    :data:`CONFIG_BEGIN_MARKER` and the first subsequent occurrence of
    :data:`CONFIG_END_MARKER`. The marker lines themselves are excluded;
    whitespace and inline comments inside the block are preserved verbatim.

    Args:
        source_text: Raw source text to scan (file contents or notebook cell).
        source_label: Human-readable identifier used in error messages
            (e.g. a file path or ``"<Jupyter cell>"``).

    Returns:
        The verbatim text of the configuration block, joined with ``\n``.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
    """
    lines: List[str] = source_text.splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == CONFIG_BEGIN_MARKER:
            begin_idx = i
        elif begin_idx is not None and stripped == CONFIG_END_MARKER:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_label}'. "
            f"Expected '{CONFIG_BEGIN_MARKER}' and '{CONFIG_END_MARKER}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def _notebook_path(ip: Any) -> str | None:
    """Return the notebook file path from the IPython kernel namespace, or None.

    Tries the known frontend-specific variables in order:
      * ``__vsc_ipynb_file__``  – VS Code / Pylance Jupyter extension
      * ``__session__``         – JupyterLab ≥ 4 kernel session path

    Returns None when running in an environment that doesn't expose a path
    (classic Notebook, Colab, plain IPython console, etc.).
    """
    for var in ("__vsc_ipynb_file__", "__session__"):
        val = ip.user_ns.get(var)
        if val:
            return str(val)
    return None


def _resolve_script_source() -> Tuple[str, str]:
    """Return ``(source_text, source_label)`` for the running configuration.

    Resolution order:

    1. **Jupyter / IPython kernel** – walks ``In`` history in reverse and
       returns the most-recent cell that contains both CONFIG markers.
       Label is the notebook path when detectable, otherwise ``"<Jupyter cell>"``.

    2. **Plain script / imported module** – reads ``__file__`` from the
       module's own globals. Label is the resolved file path.

    Raises:
        RuntimeError: If neither source can be located.
    """
    _ipython = sys.modules.get("IPython")
    ip = _ipython.get_ipython() if _ipython is not None else None  # type: ignore[attr-defined]

    if ip is not None:
        history: List[str] = ip.user_ns.get("In", [])
        for cell in reversed(history):
            if CONFIG_BEGIN_MARKER in cell and CONFIG_END_MARKER in cell:
                label: str = _notebook_path(ip) or "<Jupyter cell>"
                return cell, label

    file_attr: str | None = globals().get("__file__")
    if file_attr is not None:
        source_path: Path = Path(file_attr).resolve()
        return source_path.read_text(encoding="utf-8"), str(source_path)

    raise RuntimeError(
        "Cannot locate script source for the run log: __file__ is not defined "
        "and no Jupyter cell containing the config markers was found in In[]. "
        "Run the cell containing the CONFIGURATION block before writing output."
    )


def _hash_file_sha256(path: Path) -> str | None:
    """Return the SHA-256 hex digest of *path*'s contents, or None if unreadable."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def write_run_log(
    output_file: Path,
    *,
    settings_lines: Sequence[str],
    input_files: Sequence[Path],
    labels: Sequence[str],
) -> bool:
    """Write a ``<output stem>_runlog.txt`` sidecar next to *output_file*.

    The log records the CONFIGURATION block verbatim, the settings actually used
    (CLI flags can override the block), and each input file's SHA-256 hash, so a past
    output can be traced to the exact files it was built from.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path: Path = output_file.with_name(f"{output_file.stem}_runlog.txt")

    try:
        source_text, source_label = _resolve_script_source()
        config_text: str = extract_config_block_from_text(source_text, source_label)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("Could not extract config block for run log: %s", exc)
        return False

    lines: List[str] = [
        "=" * 72,
        "STOP RIDERSHIP CHANGE RUN LOG",
        "=" * 72,
        f"Run timestamp:   {datetime.now().isoformat(timespec='seconds')}",
        f"Output workbook: {output_file}",
        f"Source script:   {source_label}",
        "",
        "-" * 72,
        "SETTINGS USED (after any command-line overrides)",
        "-" * 72,
        *settings_lines,
        "",
        "-" * 72,
        "INPUT FILES (temporal order)",
        "-" * 72,
    ]
    for idx, (path, label) in enumerate(zip(input_files, labels), start=1):
        lines += [
            f"{idx}. {label}",
            f"   Path:   {path}",
            f"   SHA256: {_hash_file_sha256(path) or '<unreadable at log time>'}",
        ]
    lines += [
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "=" * 72,
    ]

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.info("Run log saved to '%s'.", log_path)
        return True
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False


def run(
    input_files: Sequence[Path],
    labels: Sequence[str],
    output_file: Path,
    *,
    aggregate_routes_together: bool = AGGREGATE_ROUTES_TOGETHER,
    metrics: Sequence[str] = tuple(METRICS),
    routes: Sequence[str] = (),
    routes_exclude: Sequence[str] = (),
    stop_ids: Sequence[Any] = (),
    time_periods: Sequence[str] = (),
    min_base_for_pct: float = MIN_BASE_FOR_PCT,
    treat_zero_as_absent: bool = TREAT_ZERO_AS_ABSENT,
    flag_added_removed: bool = FLAG_ADDED_REMOVED,
    export_long_sheet: bool = EXPORT_LONG_SHEET,
    apply_rounding: bool = APPLY_ROUNDING,
) -> Dict[str, pd.DataFrame]:
    """Read every signup, build the change tables, and write the workbook.

    Returns:
        The sheets written, keyed by sheet name.

    Raises:
        FileNotFoundError: If an input file is missing.
        ValueError: If an input is unreadable or lacks required columns, or nothing
            is left after filtering.
        OSError: If the workbook cannot be written.
    """
    keys: List[str] = key_columns(aggregate_routes_together)

    tables: List[pd.DataFrame] = []
    for path, label in zip(input_files, labels):
        prepared = prepare_signup(
            read_ridership_table(path),
            source=label,
            routes=routes,
            routes_exclude=routes_exclude,
            stop_ids=stop_ids,
            time_periods=time_periods,
        )
        table = aggregate_signup(prepared, aggregate_routes_together=aggregate_routes_together)
        logging.info(
            "Loaded %s from '%s': %d row(s) → %d key(s).", label, path, len(prepared), len(table)
        )
        tables.append(table)

    panel = build_panel(tables, keys, treat_zero_as_absent=treat_zero_as_absent)
    attributes = build_key_attributes(panel, keys, labels)

    summary = build_change_summary(panel, keys, labels, metrics, apply_rounding=apply_rounding)
    sheets: Dict[str, pd.DataFrame] = {
        "Change Summary": summary,
        "Signup Totals": build_signup_totals(
            panel, keys, labels, input_files, apply_rounding=apply_rounding
        ),
    }
    for metric in metrics:
        sheets[METRIC_NAMES[metric]] = build_metric_table(
            panel,
            attributes,
            metric,
            labels,
            keys,
            min_base=min_base_for_pct,
            apply_rounding=apply_rounding,
        )
    if flag_added_removed:
        sheets["Added & Removed"] = build_added_removed(
            panel, keys, labels, apply_rounding=apply_rounding
        )
    if export_long_sheet:
        long_table = build_long_table(panel, keys, labels, apply_rounding=apply_rounding)
        if len(long_table) < EXCEL_MAX_ROWS:
            sheets["Long"] = long_table
        else:
            logging.warning(
                "Skipped the Long sheet: %d rows exceeds Excel's row limit.", len(long_table)
            )

    for _, row in summary.iterrows():
        logging.info(
            "%s → %s (%s): %d key(s) in both, %d added, %d removed.",
            row["From"],
            row["To"],
            row["Comparison"],
            row["Keys in Both"],
            row["Added"],
            row["Removed"],
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_workbook(output_file, sheets)
    logging.info("The comparison workbook has been saved as '%s'.", output_file)
    return sheets


# =============================================================================
# CLI / MAIN
# =============================================================================


def notebook_safe_argv(argv: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Return the argv to parse, shielding notebook kernels from stray flags.

    When a script's ``main()`` runs with no explicit ``argv`` inside a
    Jupyter/IPython kernel, ``sys.argv`` holds kernel plumbing (for example
    ``-f /path/kernel.json``) rather than flags meant for the script, and
    strict ``argparse.parse_args`` would reject it and abort.  This helper
    detects the notebook case and substitutes an empty argument list so the
    CONFIGURATION constants stay in charge, while shell runs keep strict
    parsing (a typo in a flag fails loudly instead of being silently ignored).

    Canonical implementation: ``utils/cli_helpers.py``.

    Args:
        argv: Explicit argument list passed to ``main()``, or ``None`` to
            fall back to ``sys.argv``.

    Returns:
        ``list(argv)`` when *argv* was provided; ``[]`` when running inside a
        notebook kernel; otherwise ``None`` so argparse reads ``sys.argv[1:]``.
    """
    if argv is not None:
        return list(argv)
    if "ipykernel" in sys.modules:
        return []
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser (defaults are the CONFIGURATION values)."""
    p = argparse.ArgumentParser(
        description="Compare stop-level ridership across two or more signups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[str(p) for p in INPUT_FILES],
        help="Ridership exports in temporal order (earliest first).",
    )
    p.add_argument("--labels", nargs="+", default=list(SIGNUP_LABELS), help="One label per input.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output folder.")
    p.add_argument(
        "--output-filename",
        default=OUTPUT_FILENAME,
        help="Workbook name; _stop or _route_stop is appended to the stem.",
    )
    p.add_argument(
        "--aggregate-routes-together",
        action=argparse.BooleanOptionalAction,
        default=AGGREGATE_ROUTES_TOGETHER,
        help="Compare by stop (routes summed) instead of by route × stop.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=VALID_METRICS,
        default=list(METRICS),
        help="Measures to compare.",
    )
    p.add_argument("--routes", nargs="+", default=list(ROUTES), help="Routes to keep.")
    p.add_argument(
        "--routes-exclude", nargs="+", default=list(ROUTES_EXCLUDE), help="Routes to drop."
    )
    p.add_argument(
        "--stop-ids", nargs="+", type=int, default=list(STOP_IDS), help="Stop IDs to keep."
    )
    p.add_argument(
        "--stop-ids-file",
        type=Path,
        default=STOP_IDS_FILE,
        help="Text file of stop IDs to keep (replaces --stop-ids).",
    )
    p.add_argument(
        "--time-periods",
        nargs="+",
        default=list(TIME_PERIODS),
        help="TIME_PERIOD values to keep (e.g. 'AM PEAK').",
    )
    p.add_argument(
        "--min-base-for-pct",
        type=float,
        default=MIN_BASE_FOR_PCT,
        help="Leave %% change blank when the earlier value is below this.",
    )
    p.add_argument(
        "--treat-zero-as-absent",
        action=argparse.BooleanOptionalAction,
        default=TREAT_ZERO_AS_ABSENT,
        help="Count zero-ridership rows as absent when flagging added/removed.",
    )
    p.add_argument(
        "--flag-added-removed",
        action=argparse.BooleanOptionalAction,
        default=FLAG_ADDED_REMOVED,
        help="Write the Added & Removed sheet.",
    )
    p.add_argument(
        "--long-sheet",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_LONG_SHEET,
        help="Write the Long (key × signup) sheet.",
    )
    return p


def _is_placeholder(path: Path) -> bool:
    """True when *path* is still one of the CONFIGURATION placeholders."""
    return str(path).startswith(PLACEHOLDER_PREFIX)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read → align → compare → write pipeline.

    Returns:
        Process exit code: 0 on success, 1 on failure, 2 on a configuration error
        (placeholder paths, fewer than two inputs, or bad labels/metrics).
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))

    input_files: List[Path] = [Path(p).expanduser() for p in args.inputs]
    output_dir: Path = Path(args.output_dir).expanduser()
    if any(_is_placeholder(p) for p in input_files) or _is_placeholder(output_dir):
        logging.warning(
            "File paths are still set to their defaults. Update INPUT_FILES and OUTPUT_DIR "
            "in the CONFIGURATION section (or pass --inputs and --output-dir) before running."
        )
        return 2
    if len(input_files) < 2:
        logging.error("At least two input files are needed; got %d.", len(input_files))
        return 2
    if len({p.resolve() for p in input_files}) != len(input_files):
        logging.warning("The same input file is listed more than once.")

    metrics: List[str] = list(dict.fromkeys(args.metrics))  # drop repeats, keep order
    unknown = [m for m in metrics if m not in VALID_METRICS]
    if unknown or not metrics:
        logging.error("METRICS must be drawn from %s; got %s.", list(VALID_METRICS), metrics)
        return 2

    try:
        labels: List[str] = derive_signup_labels(input_files, args.labels)
        stop_ids: List[int] = resolve_stop_ids(args.stop_ids, args.stop_ids_file)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2

    level = comparison_level(args.aggregate_routes_together)
    base = Path(args.output_filename)
    output_file: Path = output_dir / f"{base.stem}_{level}{base.suffix or '.xlsx'}"
    logging.info("Comparing %d signups by %s: %s.", len(labels), level, " → ".join(labels))

    try:
        run(
            input_files,
            labels,
            output_file,
            aggregate_routes_together=args.aggregate_routes_together,
            metrics=metrics,
            routes=args.routes,
            routes_exclude=args.routes_exclude,
            stop_ids=stop_ids,
            time_periods=args.time_periods,
            min_base_for_pct=args.min_base_for_pct,
            treat_zero_as_absent=args.treat_zero_as_absent,
            flag_added_removed=args.flag_added_removed,
            export_long_sheet=args.long_sheet,
            apply_rounding=APPLY_ROUNDING,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        logging.error("%s", exc)
        return 1

    settings_lines: List[str] = [
        f"Comparison level:     {level}",
        f"Signup labels:        {labels}",
        f"Metrics:              {metrics}",
        f"Routes kept:          {list(args.routes) or 'all'}",
        f"Routes excluded:      {list(args.routes_exclude) or 'none'}",
        f"Stop IDs kept:        {stop_ids or 'all'}",
        f"Time periods kept:    {list(args.time_periods) or 'all'}",
        f"Min base for %:       {args.min_base_for_pct}",
        f"Zero counts absent:   {args.treat_zero_as_absent}",
    ]
    if (
        not write_run_log(
            output_file, settings_lines=settings_lines, input_files=input_files, labels=labels
        )
        and REQUIRE_RUN_LOG
    ):
        logging.error(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )
        return 1

    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
