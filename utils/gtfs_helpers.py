"""General-purpose helper functions for GTFS and transit data workflows.

Includes reusable utilities for loading GTFS files and other common tasks used
across transit data processing scripts.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def validate_gtfs_files_exist(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
) -> None:
    """Check that specific GTFS text files exist and log a warning if missing.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to check. If ``None``,
            a standard set of GTFS files is checked.
    """
    if not os.path.exists(gtfs_folder_path):
        logging.warning("The directory '%s' does not exist.", gtfs_folder_path)
        return

    if files is None:
        files = (
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        )

    for file_name in files:
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name)):
            logging.warning("Missing GTFS file: %s", file_name)


def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Folder missing or one of *files* not present.
        ValueError: Empty file or CSV parser failure.
        RuntimeError: Generic OS error while reading a file.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = (
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        )

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info("Loaded %s (%d records).", file_name, len(df))

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


def load_express_route_ids(
    inline_ids: Optional[Sequence[str]] = None,
    txt_path: Optional[str] = None,
) -> set[str]:
    """Resolve the set of express-route ``route_id`` values from two sources.

    Express/commuter routes (long highway corridors with few stops) need to be
    treated differently from local routes in coverage and ridership work. This
    helper lets callers name those routes either inline or in an external file,
    and returns the union so a script can accept both without duplicating the
    parsing logic.

    Args:
        inline_ids: ``route_id`` values supplied directly (e.g. an
            ``EXPRESS_ROUTE_IDS`` config list). ``None`` is treated as empty.
        txt_path: Path to a text file with one ``route_id`` per line. Blank
            lines are skipped and ``#`` starts a comment (whole-line or inline).
            ``None`` skips the file. A path that is set but missing is logged as
            a warning and skipped — the inline ids are still returned.

    Returns:
        The unioned set of express ``route_id`` strings (possibly empty). Every
        id is coerced to a trimmed ``str`` so it matches GTFS ``route_id`` values,
        which are read as strings.
    """
    express: set[str] = set()

    for raw in inline_ids or ():
        text = str(raw).strip()
        if text:
            express.add(text)

    if txt_path:
        if not os.path.exists(txt_path):
            logging.warning(
                "Express-routes file '%s' not found; using inline route ids only.",
                txt_path,
            )
        else:
            with open(txt_path, encoding="utf-8") as handle:
                for line in handle:
                    text = line.split("#", 1)[0].strip()
                    if text:
                        express.add(text)
            logging.info("Loaded express route ids from '%s'.", txt_path)

    logging.info("Resolved %d express route_id(s).", len(express))
    return express
