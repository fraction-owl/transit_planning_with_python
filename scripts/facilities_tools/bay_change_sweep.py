"""Rank candidate bay reassignments and time shifts at a transit center (Step 3).

Reads Step 1 timelines (one scenario folder per occupancy standard, for
example a "direct conflict" and a "likely conflict" run), inventories what
could move, and scores candidate changes with the same conflict rule Step 2
uses: buses occupying the same bay in the same minute. Nothing moves unless
the CONFIGURATION lists it.

Discover pass (``DISCOVER_ONLY = True``)
----------------------------------------
Run this first. It writes ``<label>_discover.xlsx`` with every route-end found
at the cluster (``<route> arrive``, ``<route> depart``, ``<route> through
<direction>``) and the bay(s) it uses, every block's chain of trips with the
gap between them and where it is spent, and the interline hand-offs whose
gaps a time shift has to respect. It also logs a ``BAY_CANDIDATES`` /
``SHIFT_CANDIDATES`` stub to paste into this CONFIGURATION.

Sweep (``DISCOVER_ONLY = False``)
---------------------------------
Every listed single change (a route-end to another bay, a route-end or route
shifted by k minutes) is applied to each standard's occupancy table and
re-scored. Each change is also checked against the block chains: a shift that
reduces any gap on any affected block below the minimums is rejected with the
reason. Counts are shown per standard and never summed across them; conflicts
are broken out by kind (boarding/boarding, boarding/waiting, waiting/waiting)
rather than weighted.

Inputs
------
- One Step 1 output folder per standard (``TIMELINES``): the combined CSV
  ``block_status_timeline_exporter.py`` writes, or its ``block_*.xlsx``
  workbooks.

Outputs
-------
- ``<SCENARIO_LABEL>_discover.xlsx`` (discover pass): "Route-ends", "Block
  chains" and "Interlines" sheets per standard, plus the "Config stub".
- ``<SCENARIO_LABEL>_sweep.xlsx`` (sweep): "Baseline" conflict minutes per
  bay and by kind; "Easy wins" (feasible single changes that do not raise the
  count on the standards named in ``REQUIRE_NO_WORSE_ON``, with before/after,
  the conflicts removed and created, the tightest gap after the change and
  an off-clockface note); "Pairs" (the best pairs, including pairs whose
  members are not wins on their own, such as a bay swap between two routes);
  "Packages" and "Package finalists" (a beam search over singles and the
  best pairs, up to ``MAX_CHANGES_PER_PACKAGE`` steps); "Rejected"
  (infeasible or non-improving changes and why); and "Config used".
- A ``_runlog.txt`` sidecar next to each workbook capturing the verbatim
  CONFIGURATION block.

Typical usage
-------------
Update the paths in the CONFIGURATION section, run once with
``DISCOVER_ONLY = True``, paste the candidates you are willing to consider
into ``BAY_CANDIDATES`` / ``SHIFT_CANDIDATES``, set ``DISCOVER_ONLY = False``
and run again, from a shell, ArcGIS Pro's Python window, or a Jupyter
notebook.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
# === BEGIN CONFIG ===

# --- Inputs ---------------------------------------------------------------------------------------
# One Step 1 scenario folder per standard. Every candidate is scored on all of
# them; the report shows each standard side by side and never sums across them.
# The comments give the Step 1 occupancy preset each folder was produced with.
TIMELINES: Dict[str, str] = {
    "Direct conflict": r"Path\To\block_status_output\direct_conflict",  # 1 / 0 / 0 / 10
    "Likely conflict": r"Path\To\block_status_output\likely_conflict",  # 2 / 5 / 2 / 10
}
COMBINED_TIMELINE_FILE = r"all_blocks_timeline.csv"  # read if present, else block_*.xlsx
OUTPUT_FOLDER = r"Path\To\bay_change_sweep_output"
SCENARIO_LABEL = "baseline"

# Same cluster as Step 2. Stops are GTFS stop_ids; the bay label is what the
# report shows. Bays not listed in CLUSTER_CAPACITY hold one bus.
CLUSTER_NAME = "Metro"
CLUSTER_STOPS: Dict[str, str] = {  # stop_id -> bay label
    "2373": "A",
    "2832": "B",
}
CLUSTER_CAPACITY: Dict[str, int] = {"B": 2}  # bay label -> capacity; default 1

# --- Discover pass --------------------------------------------------------------------------------
# Run with DISCOVER_ONLY = True first. The script logs every route-end it
# finds, its current bay(s), visits per day, the interline hand-offs on its
# blocks, and a stub of BAY_CANDIDATES / SHIFT_CANDIDATES to paste below.
# Nothing moves unless it is listed.
DISCOVER_ONLY = True

# --- Bay reassignment candidates -----------------------------------------------------------------
# A route-end is "<route> arrive" (trips ending here), "<route> depart"
# (trips starting here) or "<route> through <direction_id>" (mid-trip
# visits). A route-end moves as a whole, all day. Anything not listed is
# locked.
BAY_CANDIDATES: List[Dict[str, Any]] = [
    # {"route_end": "101 arrive", "bays": ["A", "B"]},
]

# Route-ends that must share a bay (moved together, or not at all).
SAME_BAY_GROUPS: List[List[str]] = [
    # ["101 arrive", "101 depart"],
]

# --- Time-shift candidates ------------------------------------------------------------------------
# Shifts in minutes, per route-end: only the trips that make up that
# route-end move ("101 arrive" shifts the trips that end at the cluster).
# Use the route name alone ("101") to shift every trip of the route. Route
# names must not contain spaces. Every gap on every affected block is checked
# against the minimums below.
SHIFT_CANDIDATES: Dict[str, List[int]] = {
    # "101 arrive": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    # "102": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
}

# Route-ends (or routes) that must shift together, e.g. an interline the
# discover pass reports as a hand-off of two minutes.
SHIFT_TOGETHER: List[List[str]] = [
    # ["101 arrive", "102 depart"],
]

# --- Feasibility (hard constraints, from the block chains) ---------------------------------------
MIN_INTERLINE_TURN_MINUTES = 2  # gap between two different routes on one block
MIN_LAYOVER_MINUTES = 3  # any other gap between trips on one block
FLAG_OFF_CLOCKFACE = True  # note, do not reject, when a shift takes departures off :00/:30

# --- Search ---------------------------------------------------------------------------------------
REQUIRE_NO_WORSE_ON = "all"  # "all": no standard may get worse; "any": at least one must not
MIN_IMPROVEMENT_MINUTES = 1
MAX_CHANGES_PER_PACKAGE = 4  # beam-search steps; a step adds a single change or a best pair
BEAM_WIDTH = 8  # partial packages kept alive at each step
ENUMERATE_PAIRS = True  # score every compatible pair of feasible singles
TOP_N = 30

# Every output must be traceable: a failed run-log write aborts the script.
# Set to False only when writing to a genuinely read-only location.
REQUIRE_RUN_LOG: bool = True

LOG_LEVEL = logging.INFO

# === END CONFIG ===

# Statuses (from Step 1)
BAY_STATUSES = {"ARRIVE", "DEPART", "ARRIVE/DEPART", "LOADING", "DWELL"}
BOARDING_STATUSES = {"ARRIVE", "DEPART", "ARRIVE/DEPART"}
WAITING_STATUSES = {"LOADING", "DWELL"}

BIG = 100000  # minute key spacing per bay (bay index * BIG + minute)

TRIP_COLUMNS = [
    "trip_id",
    "block",
    "route",
    "headsign",
    "direction",
    "start",
    "end",
    "first_stop",
    "first_bay",
    "last_stop",
    "last_bay",
    "through_bays",
]
CHAIN_COLUMNS = [
    "block",
    "gap_min",
    "where",
    "from_route",
    "from_headsign",
    "from_trip",
    "arrives",
    "to_route",
    "to_headsign",
    "to_trip",
    "departs",
    "interline",
    "from_route_end",
    "to_route_end",
]

# ==================================================================================================
# LOADING
# ==================================================================================================


def timestamp_to_minutes(ts: object) -> Optional[int]:
    """Convert an ``HH:MM`` timestamp (hours may exceed 24) to minutes past midnight."""
    if ts is None or pd.isna(ts):
        return None
    parts = str(ts).strip().split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return None


def minutes_to_hhmm(minutes: Optional[float], missing: str = "") -> str:
    """Convert minutes past midnight to a zero-padded ``HH:MM`` string.

    GTFS service days may exceed 24 hours, so values of 1440 minutes or more
    format with hours >= 24 (e.g. ``1590`` -> ``"26:30"``).

    Args:
        minutes: Minutes since midnight (may be fractional; rounded to the
            nearest minute). ``None`` and NaN yield ``missing``.
        missing: String returned for missing values, e.g. ``""`` or a
            sentinel such as ``"–"``.

    Returns:
        Zero-padded ``HH:MM`` string, or ``missing`` when *minutes* is
        ``None``/NaN.
    """
    if minutes is None or pd.isna(minutes):
        return missing
    hours, mins = divmod(int(round(minutes)), 60)
    return f"{hours:02d}:{mins:02d}"


def load_timeline(folder: str) -> DataFrame:
    """Load a Step 1 scenario folder: the combined CSV if present, else the block workbooks.

    Adds ``Minute`` (integer minutes past midnight) and ``Bay`` (the label
    from ``CLUSTER_STOPS``, or ``""`` for stops outside the cluster).
    """
    combined = os.path.join(folder, COMBINED_TIMELINE_FILE)
    if COMBINED_TIMELINE_FILE and os.path.isfile(combined):
        df = pd.read_csv(combined, dtype=str, keep_default_na=False)
    else:
        frames = []
        for name in sorted(os.listdir(folder)):
            if name.lower().startswith("block_") and name.lower().endswith(".xlsx"):
                frames.append(pd.read_excel(os.path.join(folder, name), dtype=str).fillna(""))
        if not frames:
            raise FileNotFoundError(f"No {COMBINED_TIMELINE_FILE} or block_*.xlsx in {folder}")
        df = pd.concat(frames, ignore_index=True)
    for col in (
        "Route Short Name",
        "Trip Headsign",
        "Direction",
        "Prev Trip ID",
        "Next Trip ID",
        "Layover Location",
    ):
        if col not in df.columns:
            df[col] = ""
    df["Stop ID"] = df["Stop ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["Minute"] = df["Timestamp"].apply(timestamp_to_minutes)
    df = df.dropna(subset=["Minute"]).copy()
    df["Minute"] = df["Minute"].astype(int)
    df["Bay"] = df["Stop ID"].map(CLUSTER_STOPS).fillna("")
    logging.info("Loaded %d rows, %d blocks from %s", len(df), df["Block"].nunique(), folder)
    return df


# ==================================================================================================
# TRIPS, ROUTE-ENDS, BLOCK CHAINS
# ==================================================================================================


def build_trips(df: DataFrame) -> DataFrame:
    """One row per trip: route, headsign, direction, block, scheduled start/end and end stops.

    Start is the first stop's scheduled departure (the DEPART row); end is the
    last stop's scheduled arrival (the largest Arrival Time on the trip).
    """
    rows = []
    on_trip = df[
        (df["Trip ID"] != "") & df["Status"].isin(BAY_STATUSES | {"TRAVELING BETWEEN STOPS"})
    ]
    for tid, g in on_trip.groupby("Trip ID", sort=False):
        dep = g[g["Status"] == "DEPART"]
        arr_minutes = g["Arrival Time"].map(timestamp_to_minutes)
        arr_times = [t for t in arr_minutes if t is not None]
        dep_times = [t for t in g["Departure Time"].map(timestamp_to_minutes) if t is not None]
        if not arr_times:
            continue
        start = int(dep["Minute"].iloc[0]) if not dep.empty else min(dep_times + arr_times)
        end = max(arr_times)
        first = dep.iloc[0] if not dep.empty else g.sort_values("Minute").iloc[0]
        last_rows = g[arr_minutes == end]
        if last_rows.empty:
            last_rows = g
        last = last_rows.sort_values("Minute").iloc[-1]
        through = g[(g["Status"] == "ARRIVE/DEPART") & (g["Bay"] != "")]["Bay"]
        rows.append(
            {
                "trip_id": tid,
                "block": first["Block"],
                "route": first["Route Short Name"],
                "headsign": first["Trip Headsign"],
                "direction": first["Direction"],
                "start": start,
                "end": end,
                "first_stop": first["Stop ID"],
                "first_bay": CLUSTER_STOPS.get(first["Stop ID"], ""),
                "last_stop": last["Stop ID"],
                "last_bay": CLUSTER_STOPS.get(last["Stop ID"], ""),
                "through_bays": ",".join(sorted(set(through))),
            }
        )
    trips = DataFrame(rows, columns=TRIP_COLUMNS)
    return trips.sort_values(["block", "start"]).reset_index(drop=True)


def route_end_of(trip: pd.Series) -> List[Tuple[str, str]]:
    """Which route-ends this trip contributes at the cluster: ``[(route_end, bay), ...]``."""
    out = []
    if trip["first_bay"]:
        out.append((f"{trip['route']} depart", trip["first_bay"]))
    if trip["last_bay"]:
        out.append((f"{trip['route']} arrive", trip["last_bay"]))
    for bay in [b for b in str(trip["through_bays"]).split(",") if b]:
        out.append((f"{trip['route']} through {trip['direction']}", bay))
    return out


def build_block_chains(trips: DataFrame) -> DataFrame:
    """Consecutive trip pairs on each block with the gap and where it is spent."""
    rows = []
    for block, g in trips.groupby("block", sort=False):
        g = g.sort_values("start")
        prev = None
        for _, t in g.iterrows():
            if prev is not None:
                gap = int(t["start"] - prev["end"])
                if prev["last_stop"] == t["first_stop"]:
                    bay = f" (Bay {prev['last_bay']})" if prev["last_bay"] else ""
                    where = f"same stop{bay}"
                elif prev["last_bay"] and t["first_bay"]:
                    where = f"cluster (Bay {prev['last_bay']} to Bay {t['first_bay']})"
                else:
                    where = "elsewhere"
                rows.append(
                    {
                        "block": block,
                        "gap_min": gap,
                        "where": where,
                        "from_route": prev["route"],
                        "from_headsign": prev["headsign"],
                        "from_trip": prev["trip_id"],
                        "arrives": minutes_to_hhmm(int(prev["end"])),
                        "to_route": t["route"],
                        "to_headsign": t["headsign"],
                        "to_trip": t["trip_id"],
                        "departs": minutes_to_hhmm(int(t["start"])),
                        "interline": prev["route"] != t["route"],
                        "from_route_end": f"{prev['route']} arrive" if prev["last_bay"] else "",
                        "to_route_end": f"{t['route']} depart" if t["first_bay"] else "",
                    }
                )
            prev = t
    return DataFrame(rows, columns=CHAIN_COLUMNS)


def build_route_ends(trips: DataFrame, chains: DataFrame) -> DataFrame:
    """Inventory of route-ends at the cluster with bays, visits, times and tightest gaps."""
    visits: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)  # -> [(bay, minute, trip)]
    for _, t in trips.iterrows():
        for route_end, bay in route_end_of(t):
            minute = t["start"] if route_end.endswith("depart") else t["end"]
            visits[route_end].append((bay, int(minute), t["trip_id"]))
    rows = []
    for route_end, lst in sorted(visits.items()):
        bays = sorted({b for b, _, _ in lst})
        mins = [m for _, m, _ in lst]
        tids = {tid for _, _, tid in lst}
        before = chains[chains["to_trip"].isin(tids)]
        after = chains[chains["from_trip"].isin(tids)]
        partners = set(before.loc[before["interline"], "from_route"]) | set(
            after.loc[after["interline"], "to_route"]
        )
        rows.append(
            {
                "route_end": route_end,
                "bays": ",".join(bays),
                "visits_per_day": len(lst),
                "first": minutes_to_hhmm(min(mins)),
                "last": minutes_to_hhmm(max(mins)),
                "min_gap_before": int(before["gap_min"].min()) if not before.empty else "",
                "min_gap_after": int(after["gap_min"].min()) if not after.empty else "",
                "interlines_with": ",".join(sorted(partners)),
            }
        )
    cols = [
        "route_end",
        "bays",
        "visits_per_day",
        "first",
        "last",
        "min_gap_before",
        "min_gap_after",
        "interlines_with",
    ]
    return DataFrame(rows, columns=cols)


def interline_summary(chains: DataFrame) -> DataFrame:
    """Minimum hand-off gap per ordered route pair, with the count and the tightest example."""
    cols = [
        "from_route",
        "to_route",
        "hand_offs_per_day",
        "min_gap_min",
        "max_gap_min",
        "tightest_example",
        "shift_together_needed",
    ]
    if chains.empty:
        return DataFrame(columns=cols)
    inter = chains[chains["interline"]]
    rows = []
    for (a, b), g in inter.groupby(["from_route", "to_route"]):
        tight = g.sort_values("gap_min").iloc[0]
        rows.append(
            {
                "from_route": a,
                "to_route": b,
                "hand_offs_per_day": len(g),
                "min_gap_min": int(g["gap_min"].min()),
                "max_gap_min": int(g["gap_min"].max()),
                "tightest_example": (
                    f"{tight['arrives']} {tight['from_headsign']} -> "
                    f"{tight['departs']} {tight['to_headsign']} ({tight['where']})"
                ),
                "shift_together_needed": bool(g["gap_min"].min() <= MIN_INTERLINE_TURN_MINUTES + 3),
            }
        )
    return DataFrame(rows, columns=cols).sort_values("min_gap_min")


# ==================================================================================================
# DISCOVER
# ==================================================================================================


def config_stub(route_ends: DataFrame, inter: DataFrame) -> str:
    """Text the user can paste into CONFIGURATION."""
    lines = ["BAY_CANDIDATES: List[Dict[str, Any]] = ["]
    for _, r in route_ends.iterrows():
        first_bay = r["bays"].split(",")[0]
        lines.append(
            f'    # {{"route_end": "{r["route_end"]}", "bays": ["{first_bay}"]}},'
            f"  # now {r['bays']}, {r['visits_per_day']} visits/day"
        )
    lines.append("]")
    lines.append("SHIFT_CANDIDATES: Dict[str, List[int]] = {")
    for _, r in route_ends.iterrows():
        lines.append(f'    # "{r["route_end"]}": [-3, -2, -1, 1, 2, 3],')
    for route in sorted({str(re_).split(" ")[0] for re_ in route_ends["route_end"]}):
        lines.append(f'    # "{route}": [-3, -2, -1, 1, 2, 3],  # whole route')
    lines.append("}")
    lines.append("SHIFT_TOGETHER: List[List[str]] = [")
    if not inter.empty:
        for _, r in inter[inter["shift_together_needed"]].iterrows():
            lines.append(
                f'    # ["{r["from_route"]} arrive", "{r["to_route"]} depart"],'
                f"  # hand-off as tight as {r['min_gap_min']} min"
            )
    lines.append("]")
    return "\n".join(lines)


def run_discover() -> str:
    """Write the discover workbook and log the CONFIGURATION stub.

    Returns:
        Path of the workbook written.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_path = os.path.join(OUTPUT_FOLDER, f"{SCENARIO_LABEL}_discover.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        stub = ""
        for tier, folder in TIMELINES.items():
            df = load_timeline(folder)
            trips = build_trips(df)
            chains = build_block_chains(trips)
            route_ends = build_route_ends(trips, chains)
            inter = interline_summary(chains)
            tag = "".join(ch for ch in tier if ch.isalnum())[:12]
            route_ends.to_excel(writer, sheet_name=f"Route-ends {tag}", index=False)
            chains.to_excel(writer, sheet_name=f"Block chains {tag}", index=False)
            if not inter.empty:
                inter.to_excel(writer, sheet_name=f"Interlines {tag}", index=False)
            logging.info(
                "%s: %d trips, %d route-ends, %d block hand-offs (%d interline)",
                tier,
                len(trips),
                len(route_ends),
                len(chains),
                int(chains["interline"].sum()) if not chains.empty else 0,
            )
            if not stub:
                stub = config_stub(route_ends, inter)
        DataFrame({"config_stub": stub.split("\n")}).to_excel(
            writer, sheet_name="Config stub", index=False
        )
    logging.info(
        "Paste into CONFIGURATION (uncomment what may move) -- also on the "
        "'Config stub' sheet:\n%s",
        stub,
    )
    logging.info("Discover workbook written to %s", out_path)
    require_run_log(write_run_log(Path(out_path)))
    return out_path


# ==================================================================================================
# SWEEP
# ==================================================================================================


class Change:
    """A candidate: bay moves ``{route_end: bay}`` and/or shifts ``{route_end_or_route: min}``."""

    def __init__(
        self, moves: Optional[Dict[str, str]] = None, shifts: Optional[Dict[str, int]] = None
    ) -> None:
        """Store the moves and shifts (copies; empty when omitted)."""
        self.moves = dict(moves or {})
        self.shifts = dict(shifts or {})

    def label(self) -> str:
        """Human-readable description, e.g. ``101 arrive to Bay B; 102 +2 min``."""
        parts = [f"{re_} to Bay {b}" for re_, b in self.moves.items()]
        parts += [f"{k} {v:+d} min" for k, v in self.shifts.items()]
        return "; ".join(parts)

    def combine(self, other: Change) -> Change:
        """Union of two changes; shifts on the same key add up and zero shifts drop out."""
        moves = dict(self.moves)
        moves.update(other.moves)
        shifts = dict(self.shifts)
        for k, v in other.shifts.items():
            shifts[k] = shifts.get(k, 0) + v
        return Change(moves, {k: v for k, v in shifts.items() if v})

    def compatible(self, other: Change) -> bool:
        """Two singles can be paired unless they move the same route-end or shift the same key."""
        return not (set(self.moves) & set(other.moves)) and not (
            set(self.shifts) & set(other.shifts)
        )


def expand_groups(change: Change) -> Change:
    """Apply ``SAME_BAY_GROUPS`` and ``SHIFT_TOGETHER`` so grouped members move together."""
    moves = dict(change.moves)
    for group in SAME_BAY_GROUPS:
        for re_ in list(moves):
            if re_ in group:
                for member in group:
                    moves[member] = moves[re_]
    shifts = dict(change.shifts)
    for group in SHIFT_TOGETHER:
        for k in list(shifts):
            if k in group:
                for member in group:
                    shifts[member] = shifts[k]
    return Change(moves, shifts)


class Standard:
    """Occupancy table for one standard (tier), as arrays, with a fast re-scorer."""

    def __init__(self, name: str, df: DataFrame, trips: DataFrame) -> None:
        """Build the per-row bay / minute / route-end arrays from a loaded timeline.

        Args:
            name: Label of the standard (a ``TIMELINES`` key).
            df: Output of :func:`load_timeline` for that standard.
            trips: Output of :func:`build_trips` for the same timeline.
        """
        self.name = name
        occ = df[df["Status"].isin(BAY_STATUSES) & (df["Bay"] != "")].copy()
        last_bay = trips.set_index("trip_id")["last_bay"].to_dict()
        route_of = trips.set_index("trip_id")["route"].to_dict()
        direction_of = trips.set_index("trip_id")["direction"].to_dict()

        def route_end(row: pd.Series) -> str:
            tid, st = row["Trip ID"], row["Status"]
            route = route_of.get(tid, row["Route Short Name"])
            if st == "ARRIVE/DEPART":
                return f"{route} through {direction_of.get(tid, row['Direction'])}"
            if st in ("DEPART", "LOADING"):
                return f"{route} depart"
            if st == "ARRIVE":
                return f"{route} arrive"
            # DWELL: in the arrival bay after a trip, else a hold at the departure stop
            return f"{route} arrive" if last_bay.get(tid, "") == row["Bay"] else f"{route} depart"

        occ["route_end"] = occ.apply(route_end, axis=1) if not occ.empty else ""
        occ["route"] = occ["route_end"].astype(str).str.split(" ").str[0]
        occ = occ.reset_index(drop=True)
        self.occ = occ
        self.bay_names = sorted(set(CLUSTER_STOPS.values()))
        self.bay_index = {b: i for i, b in enumerate(self.bay_names)}
        self.cap = np.array([CLUSTER_CAPACITY.get(b, 1) for b in self.bay_names])
        self.bays0 = occ["Bay"].map(self.bay_index).to_numpy(dtype=int)
        self.minutes0 = occ["Minute"].to_numpy(dtype=int)
        self.block = occ["Block"].astype("category").cat.codes.to_numpy()
        self.prefer = occ["Status"].isin({"DEPART", "LOADING"}).to_numpy()
        self.waiting = occ["Status"].isin(WAITING_STATUSES).to_numpy()
        self.route_end_arr = occ["route_end"].to_numpy()
        self.mask_route_end = {
            re_: (occ["route_end"] == re_).to_numpy() for re_ in set(occ["route_end"])
        }
        self.mask_route = {r: (occ["route"] == r).to_numpy() for r in set(occ["route"])}

    def apply(self, change: Change) -> Tuple[np.ndarray, np.ndarray]:
        """Bay index (-1 = not in a bay) and minute per row under the change.

        A shift of one route-end moves only that end's rows, so a bus's
        in-bay layover rows (its arriving end) can slide onto its own
        departure rows; those duplicates are dropped, keeping the departing
        trip's rows. An arrival moved earlier is not extended, so such a
        change is scored slightly optimistically (by at most the shift).
        Whole-route shifts are exact.
        """
        bays = self.bays0.copy()
        minutes = self.minutes0.copy()
        for route_end, bay in change.moves.items():
            m = self.mask_route_end.get(route_end)
            if m is not None:
                bays[m] = self.bay_index[bay]
        if change.shifts:
            for key, k in change.shifts.items():
                m = self.mask_route.get(key) if " " not in key else self.mask_route_end.get(key)
                if m is not None:
                    minutes[m] += k
            key_arr = self.block.astype(np.int64) * BIG + minutes
            order = np.lexsort((~self.prefer, key_arr))  # by key, departing rows first
            dup = np.zeros(len(key_arr), dtype=bool)
            sorted_keys = key_arr[order]
            dup_sorted = np.concatenate([[False], sorted_keys[1:] == sorted_keys[:-1]])
            dup[order] = dup_sorted
            bays = np.where(dup, -1, bays)
        return bays, minutes

    def score(
        self, bays: np.ndarray, minutes: np.ndarray
    ) -> Tuple[int, Dict[str, int], np.ndarray]:
        """Total conflict minutes, per-bay counts, and the conflict keys (``bay*BIG+minute``)."""
        valid = bays >= 0
        keys = bays[valid].astype(np.int64) * BIG + minutes[valid]
        uniq, counts = np.unique(keys, return_counts=True)
        bay_of = uniq // BIG
        conflict = uniq[counts > self.cap[bay_of]]
        per_bay: Dict[str, int] = {}
        for b_idx, n in zip(*np.unique(conflict // BIG, return_counts=True)):
            per_bay[self.bay_names[int(b_idx)]] = int(n)
        return int(len(conflict)), per_bay, conflict

    def detail(
        self, bays: np.ndarray, minutes: np.ndarray, conflict: np.ndarray
    ) -> Tuple[Dict[int, Tuple[str, ...]], Dict[str, int]]:
        """Route-ends present at each conflict key, and conflict minutes by kind."""
        valid = bays >= 0
        keys = np.where(valid, bays.astype(np.int64) * BIG + minutes, -1)
        in_conf = np.isin(keys, conflict)
        who: Dict[int, Tuple[str, ...]] = {}
        kinds: Dict[str, int] = defaultdict(int)
        sub = DataFrame(
            {"k": keys[in_conf], "re": self.route_end_arr[in_conf], "w": self.waiting[in_conf]}
        )
        for k, g in sub.groupby("k", sort=False):
            who[int(k)] = tuple(sorted(set(g["re"])))
            n_wait = int(g["w"].sum())
            if n_wait == len(g):
                kinds["waiting/waiting"] += 1
            elif n_wait == 0:
                kinds["boarding/boarding"] += 1
            else:
                kinds["boarding/waiting"] += 1
        return who, dict(kinds)

    def key_label(self, k: int) -> Tuple[str, int]:
        """Bay label and minute for a conflict key."""
        return self.bay_names[int(k // BIG)], int(k % BIG)


class Feasibility:
    """Vectorized gap check over the block chains for a change's shifts."""

    def __init__(self, trips: DataFrame, chains: DataFrame) -> None:
        """Index the chains by trip so a change's shifts can be applied to every gap at once."""
        self.trips = trips.reset_index(drop=True)
        self.chains = chains.reset_index(drop=True)
        idx = {tid: i for i, tid in enumerate(self.trips["trip_id"])}
        empty = np.array([], dtype=int)
        if chains.empty:
            self.from_idx = self.to_idx = self.gap = self.floor = empty
        else:
            self.from_idx = self.chains["from_trip"].map(idx).to_numpy(dtype=int)
            self.to_idx = self.chains["to_trip"].map(idx).to_numpy(dtype=int)
            self.gap = self.chains["gap_min"].to_numpy(dtype=int)
            self.floor = np.where(
                self.chains["interline"].to_numpy(dtype=bool),
                MIN_INTERLINE_TURN_MINUTES,
                MIN_LAYOVER_MINUTES,
            )
        ends = [dict(route_end_of(t)) for _, t in self.trips.iterrows()]
        self.masks: Dict[str, np.ndarray] = {}
        for r in set(self.trips["route"]):
            self.masks[r] = (self.trips["route"] == r).to_numpy()
        for re_ in {k for e in ends for k in e}:
            self.masks[re_] = np.array([re_ in e for e in ends])
        self.first_bay = self.trips["first_bay"].to_numpy()
        self.start = self.trips["start"].to_numpy(dtype=int)
        self.route = self.trips["route"].to_numpy()

    def trip_shift(self, shifts: Dict[str, int]) -> np.ndarray:
        """Minutes each trip moves under *shifts* (route-end and whole-route keys)."""
        out = np.zeros(len(self.trips), dtype=int)
        for key, k in shifts.items():
            m = self.masks.get(key)
            if m is not None:
                out += k * m
        return out

    def check(self, change: Change) -> Tuple[bool, str, Optional[int]]:
        """Whether the change keeps every affected gap at or above its minimum.

        Returns:
            ``(feasible, reason, tightest_gap_after)``. A gap already below
            its minimum is only rejected if the change shrinks it further.
        """
        if not change.shifts or len(self.gap) == 0:
            return True, "", None
        ts = self.trip_shift(change.shifts)
        new = self.gap + ts[self.to_idx] - ts[self.from_idx]
        changed = new != self.gap
        bad = changed & (new < self.gap) & (new < self.floor)
        if bad.any():
            i = int(np.argmax(bad))
            c = self.chains.iloc[i]
            reason = (
                f"gap {c['from_route']} {c['arrives']} -> {c['to_route']} {c['departs']} "
                f"({c['where']}) goes {int(self.gap[i])} -> {int(new[i])} min, "
                f"below {int(self.floor[i])}"
            )
            return False, reason, None
        return True, "", (int(new[changed].min()) if changed.any() else None)

    def off_clockface(self, change: Change) -> str:
        """Note naming routes whose :00/:30 cluster departures a shift takes off the clockface."""
        if not FLAG_OFF_CLOCKFACE or not change.shifts:
            return ""
        ts = self.trip_shift(change.shifts)
        m = (
            (self.first_bay != "")
            & (ts != 0)
            & (self.start % 30 == 0)
            & ((self.start + ts) % 30 != 0)
        )
        return f"departures off :00/:30: {', '.join(sorted(set(self.route[m])))}" if m.any() else ""


def describe_delta(
    std: Standard, before: Dict[int, Tuple[str, ...]], after: Dict[int, Tuple[str, ...]]
) -> Tuple[str, str]:
    """Conflicts removed and created, collapsed into windows with the route pair."""

    def windows(keys: List[int], src: Dict[int, Tuple[str, ...]]) -> str:
        out = []
        labelled = sorted((std.key_label(k), k) for k in keys)
        by_bay: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for (bay, minute), k in labelled:
            by_bay[bay].append((minute, k))
        for bay, lst in by_bay.items():
            start = prev = lst[0][0]
            pr = src[lst[0][1]]
            for minute, k in lst[1:] + [(None, None)]:
                if minute is not None and k is not None and minute == prev + 1 and src[k] == pr:
                    prev = minute
                    continue
                span = minutes_to_hhmm(start)
                if prev != start:
                    span += f"-{minutes_to_hhmm(prev)}"
                out.append(f"{bay} {span} {'/'.join(x.split(' ')[0] for x in pr)}")
                if minute is not None and k is not None:
                    start = prev = minute
                    pr = src[k]
        return "; ".join(out)

    removed = [k for k in before if k not in after]
    created = [k for k in after if k not in before]
    return windows(removed, before) if removed else "", windows(created, after) if created else ""


def single_candidates(standards: Dict[str, Standard]) -> List[Change]:
    """Every listed single change that exists in the timelines: one bay move or one shift."""
    known = set()
    for std in standards.values():
        known |= set(std.mask_route_end)
    first = next(iter(standards.values()))
    out: List[Change] = []
    for c in BAY_CANDIDATES:
        re_ = c["route_end"]
        if re_ not in known:
            logging.warning(
                "BAY_CANDIDATES: route-end '%s' not found in the timelines; skipped.", re_
            )
            continue
        current = set(first.occ.loc[first.occ["route_end"] == re_, "Bay"])
        for bay in c["bays"]:
            if bay not in current and bay in first.bay_index:
                out.append(Change(moves={re_: bay}))
    for key, ks in SHIFT_CANDIDATES.items():
        if key not in known and not any(re_.startswith(key + " ") for re_ in known):
            logging.warning("SHIFT_CANDIDATES: '%s' not found in the timelines; skipped.", key)
            continue
        for k in ks:
            if k:
                out.append(Change(shifts={key: k}))
    return out


class Scorer:
    """Scores changes against a baseline (or a package-in-progress) on every standard."""

    def __init__(self, standards: Dict[str, Standard], feas: Feasibility) -> None:
        """Score the untouched baseline of every standard."""
        self.standards = standards
        self.feas = feas
        self.base: Dict[str, Tuple[int, Dict[str, int], np.ndarray]] = {}
        self.base_change = Change()
        self.rebase(Change())

    def rebase(self, change: Change) -> None:
        """Make *change* the baseline further changes are measured against."""
        self.base_change = change
        self.base = {}
        for n, s in self.standards.items():
            b, m = s.apply(change)
            self.base[n] = s.score(b, m)

    def quick(self, change: Change) -> Optional[Dict[str, int]]:
        """Improvement per standard for ``base_change + change``, or ``None`` if infeasible."""
        full = expand_groups(self.base_change.combine(change))
        ok, _, _ = self.feas.check(full)
        if not ok:
            return None
        imp = {}
        for n, s in self.standards.items():
            b, m = s.apply(full)
            total, _, _ = s.score(b, m)
            imp[n] = self.base[n][0] - total
        return imp

    def full(self, change: Change) -> Dict[str, Any]:
        """Complete row for the report (keys starting with ``_`` are internal)."""
        combined = expand_groups(self.base_change.combine(change))
        ok, reason, tightest = self.feas.check(combined)
        row: Dict[str, Any] = {
            "change": expand_groups(change).label(),
            "feasible": ok,
            "reason": reason,
            "tightest gap after": tightest if tightest is not None else "",
            "note": self.feas.off_clockface(combined) if ok else "",
        }
        imp: Dict[str, int] = {}
        for n, s in self.standards.items():
            b_total, _, b_conf = self.base[n]
            row[f"{n} before"] = b_total
            if not ok:
                row[f"{n} after"] = ""
                continue
            b, m = s.apply(combined)
            total, per, conf = s.score(b, m)
            imp[n] = b_total - total
            b0, m0 = s.apply(self.base_change)
            who_before, _ = s.detail(b0, m0, b_conf)
            who_after, kinds = s.detail(b, m, conf)
            removed, created = describe_delta(s, who_before, who_after)
            row[f"{n} after"] = total
            row[f"{n} per bay after"] = " ".join(f"{k}{v}" for k, v in sorted(per.items()))
            row[f"{n} removed"] = removed
            row[f"{n} created"] = created
            row[f"{n} kinds after"] = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        row["_improvement"] = imp
        row["_change"] = change
        return row


def is_win(imp: Optional[Dict[str, int]]) -> bool:
    """Feasible, no worse on the required standards, and better by the minimum on at least one."""
    if not imp:
        return False
    if REQUIRE_NO_WORSE_ON == "any":
        no_worse = any(v >= 0 for v in imp.values())
    else:
        no_worse = all(v >= 0 for v in imp.values())
    return no_worse and max(imp.values()) >= MIN_IMPROVEMENT_MINUTES


def rank_key(imp: Dict[str, int]) -> tuple:
    """Sort key: biggest improvement first, judged on the last-listed standard first."""
    names = list(imp)
    return tuple(-imp[n] for n in reversed(names))


def _public(rows: List[Dict[str, Any]]) -> DataFrame:
    """Report rows without the internal ``_``-prefixed keys."""
    return DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])


def _search_packages(
    scorer: Scorer, pool: List[Change], standards: Dict[str, Standard]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Beam search over *pool*: best package per step, and the finalists at the last step.

    A greedy walk cannot assemble a change whose parts are individually
    neutral (a bay swap among three routes); a beam keeps several partial
    packages alive and lets those emerge.
    """
    beam: List[Tuple[tuple, Change]] = [((), Change())]
    best_per_step: List[Dict[str, Any]] = []
    finalists: List[Tuple[tuple, Change]] = []
    for step in range(1, MAX_CHANGES_PER_PACKAGE + 1):
        expanded: Dict[str, Change] = {}
        for _, pkg in beam:
            scorer.rebase(pkg)
            for c in pool:
                if not pkg.compatible(c):
                    continue
                if not is_win(scorer.quick(c)):
                    continue
                cand = expand_groups(pkg.combine(c))
                expanded.setdefault(cand.label(), cand)
        # rank against the untouched baseline so steps are comparable
        scorer.rebase(Change())
        if not expanded:
            break
        ranked = []
        for cand in expanded.values():
            imp0 = scorer.quick(cand)
            if imp0 is not None:
                ranked.append((rank_key(imp0), cand))
        ranked.sort(key=lambda x: x[0])
        beam = ranked[:BEAM_WIDTH]
        finalists = ranked[:5]
        best = beam[0][1]
        row = scorer.full(best)
        pkg_row: Dict[str, Any] = {"step": step, "package": best.label()}
        for n in standards:
            pkg_row[f"{n} conflict minutes"] = row[f"{n} after"]
            pkg_row[f"{n} per bay"] = row[f"{n} per bay after"]
            pkg_row[f"{n} kinds"] = row[f"{n} kinds after"]
        pkg_row["tightest gap after"] = row["tightest gap after"]
        pkg_row["note"] = row["note"]
        best_per_step.append(pkg_row)
    finalist_rows = [scorer.full(c) for _, c in finalists]
    return best_per_step, finalist_rows


def run_sweep() -> str:
    """Score every listed change, pair and package and write the sweep workbook.

    Feasibility is judged on the block chains of the first-listed standard:
    the schedule is the same across standards, only the occupancy assumptions
    differ.

    Returns:
        Path of the workbook written.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    standards: Dict[str, Standard] = {}
    trips: Optional[DataFrame] = None
    chains: Optional[DataFrame] = None
    for name, folder in TIMELINES.items():
        df = load_timeline(folder)
        t = build_trips(df)
        if trips is None or chains is None:
            trips, chains = t, build_block_chains(t)
        standards[name] = Standard(name, df, t)
    if trips is None or chains is None:
        raise ValueError("TIMELINES is empty; list at least one Step 1 scenario folder.")
    feas = Feasibility(trips, chains)
    scorer = Scorer(standards, feas)
    for n, (total, per, _) in scorer.base.items():
        logging.info("%s baseline: %d conflict minutes %s", n, total, dict(sorted(per.items())))

    # ---- singles
    singles = single_candidates(standards)
    logging.info("Scoring %d single changes.", len(singles))
    single_imp = [(c, scorer.quick(c)) for c in singles]
    win_rows = sorted(
        [scorer.full(c) for c, imp in single_imp if is_win(imp)],
        key=lambda r: rank_key(r["_improvement"]),
    )
    rejected_rows = []
    for c, imp in single_imp:
        if is_win(imp):
            continue
        r = scorer.full(c)
        if r["feasible"] and not r["reason"] and imp is not None:
            if max(imp.values()) < MIN_IMPROVEMENT_MINUTES:
                r["reason"] = "no improvement"
            else:
                r["reason"] = "worsens " + ", ".join(n for n, v in imp.items() if v < 0)
        rejected_rows.append(r)
    logging.info("%d single wins, %d rejected.", len(win_rows), len(rejected_rows))

    # ---- pairs: every compatible pair of feasible singles, scored quickly, best ones described
    pair_rows: List[Dict[str, Any]] = []
    if ENUMERATE_PAIRS:
        feasible = [c for c, imp in single_imp if imp is not None]
        scored = []
        n_pairs = 0
        for i in range(len(feasible)):
            for j in range(i + 1, len(feasible)):
                if not feasible[i].compatible(feasible[j]):
                    continue
                n_pairs += 1
                pair = feasible[i].combine(feasible[j])
                imp = scorer.quick(pair)
                if is_win(imp) and imp is not None:
                    scored.append((rank_key(imp), pair))
        scored.sort(key=lambda x: x[0])
        logging.info("%d pairs scored; %d are wins.", n_pairs, len(scored))
        pair_rows = [scorer.full(pair) for _, pair in scored[:TOP_N]]

    # ---- packages
    pool = [c for c, imp in single_imp if imp is not None] + [r["_change"] for r in pair_rows]
    packages, finalist_rows = _search_packages(scorer, pool, standards)

    out_path = os.path.join(OUTPUT_FOLDER, f"{SCENARIO_LABEL}_sweep.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        base_rows = []
        for n, s in standards.items():
            total, per, conf = scorer.base[n]
            _, kinds = s.detail(s.bays0, s.minutes0, conf)
            base_rows.append(
                {
                    "standard": n,
                    "total": total,
                    **{f"Bay {b}": per.get(b, 0) for b in s.bay_names},
                    **kinds,
                }
            )
        DataFrame(base_rows).to_excel(writer, sheet_name="Baseline", index=False)
        _public(win_rows[:TOP_N]).to_excel(writer, sheet_name="Easy wins", index=False)
        if ENUMERATE_PAIRS:
            _public(pair_rows).to_excel(writer, sheet_name="Pairs", index=False)
        DataFrame(packages).to_excel(writer, sheet_name="Packages", index=False)
        _public(finalist_rows).to_excel(writer, sheet_name="Package finalists", index=False)
        _public(rejected_rows).to_excel(writer, sheet_name="Rejected", index=False)
        cfg = {
            "CLUSTER_NAME": CLUSTER_NAME,
            "CLUSTER_STOPS": CLUSTER_STOPS,
            "CLUSTER_CAPACITY": CLUSTER_CAPACITY,
            "TIMELINES": TIMELINES,
            "BAY_CANDIDATES": BAY_CANDIDATES,
            "SAME_BAY_GROUPS": SAME_BAY_GROUPS,
            "SHIFT_CANDIDATES": SHIFT_CANDIDATES,
            "SHIFT_TOGETHER": SHIFT_TOGETHER,
            "MIN_INTERLINE_TURN_MINUTES": MIN_INTERLINE_TURN_MINUTES,
            "MIN_LAYOVER_MINUTES": MIN_LAYOVER_MINUTES,
            "REQUIRE_NO_WORSE_ON": REQUIRE_NO_WORSE_ON,
            "MIN_IMPROVEMENT_MINUTES": MIN_IMPROVEMENT_MINUTES,
            "MAX_CHANGES_PER_PACKAGE": MAX_CHANGES_PER_PACKAGE,
            "BEAM_WIDTH": BEAM_WIDTH,
            "ENUMERATE_PAIRS": ENUMERATE_PAIRS,
        }
        DataFrame({"setting": list(cfg), "value": [str(v) for v in cfg.values()]}).to_excel(
            writer, sheet_name="Config used", index=False
        )
    logging.info("Sweep workbook for %s written to %s", CLUSTER_NAME, out_path)
    require_run_log(write_run_log(Path(out_path)))
    return out_path


# --------------------------------------------------------------------------------------------------
# RUN LOG
# --------------------------------------------------------------------------------------------------


class RunLogError(RuntimeError):
    """Raised when the required ``_runlog.txt`` sidecar could not be written."""


# Canonical version lives in utils/run_log.py -- keep this copy in sync.
def extract_config_block(source_file: Path) -> str:
    r"""Return the text between the CONFIG markers in *source_file*.

    Reads ``source_file`` as UTF-8 text and slices out the lines strictly
    *between* the first occurrence of ``# === BEGIN CONFIG ===`` and the first
    subsequent occurrence of ``# === END CONFIG ===``.  The marker lines
    themselves are excluded; whitespace and inline comments inside the block
    are preserved verbatim.

    Args:
        source_file: Path to the Python source file to scan (typically
            ``Path(__file__)`` from the calling script).

    Returns:
        The verbatim text of the configuration block, joined with ``\n``.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    _BEGIN = "# === BEGIN CONFIG ==="
    _END = "# === END CONFIG ==="

    lines: list[str] = source_file.read_text(encoding="utf-8").splitlines()

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped: str = line.strip()
        if begin_idx is None and stripped == _BEGIN:
            begin_idx = i
        elif begin_idx is not None and stripped == _END:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_file}'. Expected '{_BEGIN}' and '{_END}'."
        )

    return "\n".join(lines[begin_idx + 1 : end_idx])


def resolve_source_file() -> Optional[Path]:
    """Path of this script on disk, or ``None`` in an interactive session without ``__file__``."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def write_run_log(output_file: Path) -> bool:
    """Write the ``_runlog.txt`` sidecar for *output_file* (same folder, same stem).

    The log captures this script's CONFIGURATION block verbatim, between the
    ``# === BEGIN CONFIG ===`` / ``# === END CONFIG ===`` markers, so it can
    never drift from the values actually used.

    Returns:
        ``True`` if the log was written successfully, ``False`` otherwise.
    """
    log_path = output_file.with_name(f"{output_file.stem}_runlog.txt")

    source_file = resolve_source_file()
    if source_file is None:
        config_text = "(config block unavailable: interactive session, no __file__ on disk)"
        source_display = "<interactive>"
    else:
        try:
            config_text = extract_config_block(source_file)
        except (OSError, ValueError) as exc:
            logging.error("Could not extract config block for run log: %s", exc)
            return False
        source_display = str(source_file)

    lines: List[str] = [
        "=" * 72,
        "BAY CHANGE SWEEP RUN LOG",
        "=" * 72,
        f"Run timestamp:    {datetime.now().isoformat(timespec='seconds')}",
        f"Output file:      {output_file}",
        f"Source script:    {source_display}",
        "",
        "-" * 72,
        "CONFIGURATION (verbatim from source)",
        "-" * 72,
        config_text,
        "=" * 72,
    ]

    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logging.error("Error writing run log: %s", exc)
        return False
    logging.info("Run log saved to %s", log_path)
    return True


def require_run_log(written: bool) -> None:
    """Raise :class:`RunLogError` when a run log failed and ``REQUIRE_RUN_LOG`` is set."""
    if not written and REQUIRE_RUN_LOG:
        raise RunLogError(
            "Run log could not be written. Set REQUIRE_RUN_LOG = False to suppress this "
            "error when a sidecar file is genuinely impossible."
        )


# ==================================================================================================
# MAIN
# ==================================================================================================

_PLACEHOLDER_MARKERS: Tuple[str, ...] = (
    "path\\to\\",
    "path/to/",
)


def _is_placeholder_path(p: str) -> bool:
    """Return True if *p* still points at a default placeholder location."""
    s = str(p).lower()
    return any(marker in s for marker in _PLACEHOLDER_MARKERS)


def main() -> int:
    """Run the discover pass or the sweep, depending on ``DISCOVER_ONLY``.

    Returns:
        Process exit code: 0 on success, 1 if the required run log could not
        be written, 2 if required CONFIGURATION values are still placeholders
        or a Step 1 folder does not exist.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    paths = dict(TIMELINES)
    paths["OUTPUT_FOLDER"] = OUTPUT_FOLDER
    unset = [name for name, p in paths.items() if _is_placeholder_path(p)]
    if unset:
        logging.warning(
            "Default placeholder paths detected for: %s. Update the CONFIGURATION section "
            "before running.",
            ", ".join(unset),
        )
        return 2
    missing = [name for name, folder in TIMELINES.items() if not os.path.isdir(folder)]
    if missing:
        logging.warning(
            "Step 1 folder not found for: %s. Run block_status_timeline_exporter.py with the "
            "matching SCENARIO_NAME first.",
            ", ".join(missing),
        )
        return 2
    try:
        if DISCOVER_ONLY:
            run_discover()
        else:
            run_sweep()
    except RunLogError as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Script completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
