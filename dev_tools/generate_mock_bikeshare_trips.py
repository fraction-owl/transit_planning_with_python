r"""Generate small, deterministic Capital Bikeshare trip-data fixtures over time.

Writes one CSV per month (named like the real vendor files,
``YYYYMM-capitalbikeshare-tripdata.csv``) so tests can exercise scripts that
build ridership-over-time charts against a realistic, lightweight stand-in for
the public Capital Bikeshare (Lyft GBFS) extracts.

------------------------------------------------------------------------------
RUNNING IT
------------------------------------------------------------------------------
Notebook / manual: edit the CONFIG block below, then run the file (or call
``run()``). No command-line arguments needed -- pasting the module into a cell
or ``%run``-ing it uses the CONFIG values.

Command line: every CONFIG value has a matching flag that overrides it, e.g.
    python dev_tools/generate_bikeshare_fixture.py --months 6 --rows 600
    python dev_tools/generate_bikeshare_fixture.py \\
        --input-extract "C:\\data\\202604-capitalbikeshare-tripdata.zip" \\
        --output-dir tests/fixtures/capitalbikeshare

------------------------------------------------------------------------------
WHAT IT PRODUCES
------------------------------------------------------------------------------
The total trip budget is spread across the months with structure, so an
over-time chart has something to show rather than a flat line:
  * Seasonality: a smooth summer peak / winter trough (~3x peak-to-trough).
  * Trend: mild year-over-year growth.

Per-trip distributions and quirks are calibrated against a profiled full-month
extract (202604, 604,565 rows):
  * 13-column schema, canonical order, CRLF line endings, no BOM, ms timestamps.
  * ``ride_id`` is a 16-character uppercase hex string (unique across all files).
  * ~67.5% electric / 32.5% classic (flat over time); ~70.5% member.
  * Dockless e-bike trips: ~15% of trips have a BLANK station name+id (start or
    end). Classic bikes always dock, so they always have a station.
  * Coordinate precision varies 1-15 dp, different mix per rideable_type (classic
    never below 3 dp). start coords never null; a tiny fraction of end coords are.
  * Heavy-tailed durations: median ~9 min, tail to multi-hour; always > 0.
  * ~3% round trips; trailing whitespace appears verbatim in some station names.

A handful of rare cases (null end coords, multi-hour and sub-minute durations,
dockless start/end, round trip, name-present/id-blank) are guaranteed present
somewhere in the span so the suite covers them; set GUARANTEE_EDGE_CASES = False
(or pass --no-guarantee) to turn that off.

Determinism: SEED (default 0) gives byte-identical files for stable commits.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import io
import math
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from random import Random
from typing import NamedTuple, TextIO

# ===========================================================================
# CONFIG  --  notebook users edit these; CLI flags override them
# ===========================================================================

# Optional path to a REAL Capital Bikeshare extract (.zip or .csv). If set, the
# station pool AND coordinate bounding box are harvested from it, so fixtures
# use stations from your own data. Leave "" to use the built-in DC pool.
# Raw string (r"...") so Windows backslash paths paste in safely.
INPUT_EXTRACT = r""

# Directory the per-month fixture CSVs are written to.
OUTPUT_DIR = r"tests/fixtures/capitalbikeshare"

START_MONTH = "2024-05"  # first month, YYYY-MM
NUM_MONTHS = 24  # number of consecutive months
TOTAL_ROWS = 3000  # total trips, apportioned across the months
SEED = 0  # RNG seed for reproducible output
GUARANTEE_EDGE_CASES = True  # force rare edge cases to appear in the span

# ===========================================================================
# Calibrated constants (from the 202604 full-month profile)
# ===========================================================================

COLUMNS: tuple[str, ...] = (
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
)

# Built-in DC stations (id, name, lat, lng). Names are VERBATIM, including
# trailing whitespace where the source has it. Used unless INPUT_EXTRACT is set.
DEFAULT_STATIONS: tuple[tuple[str, str, float, float], ...] = (
    ("31016", "Clarendon Blvd & Pierce St", 38.893438, -77.076389),
    ("31022", "Clarendon Metro / Wilson Blvd & N Highland St", 38.887010, -77.095257),
    ("31028", "N Veitch St & Key Blvd", 38.893237, -77.086063),
    ("31031", "15th St & N Scott St", 38.890540, -77.080950),
    ("31062", "Roosevelt Island", 38.896553, -77.067140),
    ("31096", "Columbia Pike & S Highland St", 38.862398, -77.089133),
    ("31130", "7th & S St NW", 38.914247, -77.021556),
    ("31138", "4th & College St NW", 38.921233, -77.018135),
    ("31201", "15th & P St NW", 38.909801, -77.034427),
    ("31226", "34th St & Wisconsin Ave NW", 38.916442, -77.068200),
    ("31236", "37th & O St NW / Georgetown University", 38.907837, -77.071660),
    ("31241", "Thomas Circle", 38.905900, -77.032500),
    ("31248", "Smithsonian-National Mall / Jefferson Dr & 12th St SW", 38.888774, -77.028694),
    ("31249", "Jefferson Memorial", 38.879819, -77.037413),
    ("31266", "11th & M St NW", 38.905578, -77.027313),
    ("31273", "Hains Point/Buckeye & Ohio Dr SW", 38.878433, -77.030230),
    ("31285", "22nd & P ST NW", 38.909394, -77.048728),
    ("31286", "11th & O St NW", 38.908431, -77.027088),
    ("31288", "4th St & Madison Dr NW", 38.890496, -77.017246),
    ("31291", "Vermont Ave & I St NW", 38.901136, -77.034451),
    ("31327", "14th & Q St NW", 38.910674, -77.031880),
    ("31404", "9th & Upshur St NW", 38.941800, -77.025100),
    ("31628", "1st & I St SE", 38.878939, -77.005833),
    ("31649", "14th & Newton St NW", 38.931991, -77.032956),
    ("31654", "King Greenleaf Rec Center", 38.876211, -77.012443),
    ("31677", "1st & L St NW", 38.903819, -77.011987),
    ("31680", "Half & I St SW ", 38.879262, -77.011016),  # trailing ws is real
)


class Pool(NamedTuple):
    """Station pool plus the bounding box used for dockless (free-float) points."""

    stations: tuple[tuple[str, str, float, float], ...]
    lat_min: float
    lat_max: float
    lng_min: float
    lng_max: float


DEFAULT_POOL = Pool(DEFAULT_STATIONS, 38.760000, 39.125828, -77.420000, -76.825535)

P_ELECTRIC = 0.6753  # flat over time, per request
MEMBER_CASUAL: tuple[tuple[str, float], ...] = (
    ("member", 0.7051),
    ("casual", 0.2949),
)

# P(dockless | electric) tuned so overall blank-station rate matches the profile
# (~14.7% start, ~15.5% end across all trips). Classic bikes never go dockless.
P_DOCKLESS_START_GIVEN_E = 0.2175
P_DOCKLESS_END_GIVEN_E = 0.2370
P_ROUNDTRIP_GIVEN_DOCKED = 0.0363  # ~3.1% overall
P_NULL_END_COORDS = 0.0007  # rare; guarantee step covers small fixtures

# Coordinate-string precision (decimal places) per rideable_type, from the
# profile's start_lat tallies. Lat & lng in one row share a precision.
PRECISION_WEIGHTS: dict[str, dict[int, int]] = {
    "electric_bike": {
        1: 19237,
        2: 69551,
        3: 1337,
        4: 18219,
        5: 21406,
        6: 167775,
        7: 11903,
        8: 3825,
        9: 37950,
        12: 1459,
        13: 5586,
        14: 33208,
        15: 16836,
    },
    "classic_bike": {
        3: 800,
        4: 11393,
        5: 18604,
        6: 116112,
        7: 9295,
        12: 1477,
        13: 2803,
        14: 23386,
        15: 12403,
    },
}

# Lognormal duration model (seconds): median ~541s, heavy right tail.
DURATION_LOG_MU = math.log(541)
DURATION_LOG_SIGMA = 0.90
DURATION_MIN_S = 1.0
DURATION_MAX_S = 90000.0

# Temporal shape for distributing trips across months.
SEASONAL_AMPLITUDE = 0.50  # 1 +/- this; peak month ~3x the trough
SEASONAL_PEAK_MONTH = 7  # July
ANNUAL_GROWTH = 0.12  # ~12% year-over-year

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S.%f"


# ---------------------------------------------------------------------------
# Loading a station pool from a real extract (optional INPUT_EXTRACT)
# ---------------------------------------------------------------------------


def _iter_extract_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield dict-rows from a .zip of CSVs or a single .csv (streaming)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No .csv members inside {path}")
            for member in members:
                with zf.open(member) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                    yield from csv.DictReader(text)
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            yield from csv.DictReader(fh)


def load_pool_from_extract(path: Path) -> Pool:
    """Harvest a station pool and coordinate bbox from a real extract."""
    best: dict[str, tuple[str, float, float, int]] = {}  # id -> (name, lat, lng, len)
    lat_lo = lng_lo = float("inf")
    lat_hi = lng_hi = float("-inf")
    for row in _iter_extract_rows(path):
        for side in ("start", "end"):
            sid = (row.get(f"{side}_station_id") or "").strip()
            name = row.get(f"{side}_station_name") or ""  # keep trailing ws
            lat_s = (row.get(f"{side}_lat") or "").strip()
            lng_s = (row.get(f"{side}_lng") or "").strip()
            if not lat_s or not lng_s:
                continue
            try:
                lat, lng = float(lat_s), float(lng_s)
            except ValueError:
                continue
            lat_lo, lat_hi = min(lat_lo, lat), max(lat_hi, lat)
            lng_lo, lng_hi = min(lng_lo, lng), max(lng_hi, lng)
            if sid and name.strip():
                # keep the shortest-precision (cleanest, dock-like) coordinate
                weight = len(lat_s) + len(lng_s)
                if sid not in best or weight < best[sid][3]:
                    best[sid] = (name, lat, lng, weight)
    stations = tuple((sid, nm, lat, lng) for sid, (nm, lat, lng, _) in sorted(best.items()))
    if len(stations) < 2:
        raise ValueError(f"Could not harvest >=2 stations from {path}")
    return Pool(stations, lat_lo, lat_hi, lng_lo, lng_hi)


def _resolve_pool(input_extract: str) -> Pool:
    if input_extract:
        path = Path(input_extract)
        if not path.exists():
            raise FileNotFoundError(f"INPUT_EXTRACT not found: {path}")
        pool = load_pool_from_extract(path)
        print(f"Harvested {len(pool.stations)} stations from {path}", file=sys.stderr)
        return pool
    return DEFAULT_POOL


# ---------------------------------------------------------------------------
# Temporal distribution
# ---------------------------------------------------------------------------


def month_span(start_year: int, start_month: int, n: int) -> list[tuple[int, int]]:
    """Return ``n`` consecutive ``(year, month)`` pairs starting at the given month."""
    span = []
    y, m = start_year, start_month
    for _ in range(n):
        span.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return span


def _month_weight(index: int, month: int) -> float:
    seasonal = 1.0 + SEASONAL_AMPLITUDE * math.cos(2 * math.pi * (month - SEASONAL_PEAK_MONTH) / 12)
    trend = (1.0 + ANNUAL_GROWTH) ** (index / 12)
    return seasonal * trend


def allocate_counts(total: int, span: list[tuple[int, int]]) -> list[int]:
    """Apportion `total` across months by seasonal x trend weight (sums to total)."""
    weights = [_month_weight(i, m) for i, (_, m) in enumerate(span)]
    wsum = sum(weights)
    raw = [total * w / wsum for w in weights]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _weighted(rng: Random, pairs: Iterable[tuple[object, float]]) -> object:
    values, weights = zip(*pairs)
    return rng.choices(values, weights=weights, k=1)[0]


def _make_ride_id(rng: Random) -> str:
    return f"{rng.getrandbits(64):016X}"


def _format_timestamp(dt: datetime) -> str:
    return dt.strftime(TIMESTAMP_FMT)[:-3]  # microseconds -> milliseconds


def _sample_precision(rng: Random, rideable_type: str) -> int:
    weights = PRECISION_WEIGHTS[rideable_type]
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def _sample_duration(rng: Random) -> float:
    val = math.exp(rng.gauss(DURATION_LOG_MU, DURATION_LOG_SIGMA))
    return min(DURATION_MAX_S, max(DURATION_MIN_S, val))


def _docked_point(rng: Random, lat: float, lng: float) -> tuple[float, float]:
    """A jittered point near a dock (full-precision floats for realistic digits)."""
    return lat + rng.uniform(-3e-4, 3e-4), lng + rng.uniform(-3e-4, 3e-4)


def _dockless_point(rng: Random, pool: Pool) -> tuple[float, float]:
    return rng.uniform(pool.lat_min, pool.lat_max), rng.uniform(pool.lng_min, pool.lng_max)


def _pick_station(rng: Random, pool: Pool, exclude_id: str = "") -> tuple[str, str, float, float]:
    """Choose a station, optionally excluding one id.

    Excluding ``exclude_id`` avoids accidental round trips from the small fixture
    pool, which a full 800+ station source would not produce.
    """
    if not exclude_id:
        return rng.choice(pool.stations)
    return rng.choice([st for st in pool.stations if st[0] != exclude_id])


def _fmt_coords(rng: Random, lat: float, lng: float, rideable_type: str) -> tuple[str, str]:
    p = _sample_precision(rng, rideable_type)
    return f"{lat:.{p}f}", f"{lng:.{p}f}"


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _build_row(
    rng: Random, year: int, month: int, seen_ids: set[str], pool: Pool
) -> dict[str, str]:
    ride_id = _make_ride_id(rng)
    while ride_id in seen_ids:
        ride_id = _make_ride_id(rng)
    seen_ids.add(ride_id)

    rideable = "electric_bike" if rng.random() < P_ELECTRIC else "classic_bike"
    is_e = rideable == "electric_bike"

    # --- stations & coordinates ------------------------------------------
    start_dockless = is_e and rng.random() < P_DOCKLESS_START_GIVEN_E
    if start_dockless:
        s_name = s_id = ""
        s_lat, s_lng = _dockless_point(rng, pool)
    else:
        s_id, s_name, base_lat, base_lng = rng.choice(pool.stations)
        s_lat, s_lng = _docked_point(rng, base_lat, base_lng)

    round_trip = (not start_dockless) and rng.random() < P_ROUNDTRIP_GIVEN_DOCKED
    if round_trip:
        e_id, e_name = s_id, s_name
        base = next(st for st in pool.stations if st[0] == s_id)
        e_lat, e_lng = _docked_point(rng, base[2], base[3])
    else:
        end_dockless = is_e and rng.random() < P_DOCKLESS_END_GIVEN_E
        if end_dockless:
            e_name = e_id = ""
            e_lat, e_lng = _dockless_point(rng, pool)
        else:
            e_id, e_name, base_lat, base_lng = _pick_station(rng, pool, exclude_id=s_id)
            e_lat, e_lng = _docked_point(rng, base_lat, base_lng)

    s_lat_str, s_lng_str = _fmt_coords(rng, s_lat, s_lng, rideable)
    if rng.random() < P_NULL_END_COORDS:
        e_lat_str = e_lng_str = ""
    else:
        e_lat_str, e_lng_str = _fmt_coords(rng, e_lat, e_lng, rideable)

    # --- timing (rides may legitimately spill past the month end) ---------
    days = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1) + timedelta(
        seconds=rng.uniform(0, days * 86400 - 1), milliseconds=rng.randint(0, 999)
    )
    ended = start + timedelta(seconds=_sample_duration(rng))

    return {
        "ride_id": ride_id,
        "rideable_type": rideable,
        "started_at": _format_timestamp(start),
        "ended_at": _format_timestamp(ended),
        "start_station_name": s_name,
        "start_station_id": s_id,
        "end_station_name": e_name,
        "end_station_id": e_id,
        "start_lat": s_lat_str,
        "start_lng": s_lng_str,
        "end_lat": e_lat_str,
        "end_lng": e_lng_str,
        "member_casual": _weighted(rng, MEMBER_CASUAL),
    }


def _force_duration(row: dict[str, str], seconds: float) -> None:
    start = datetime.strptime(row["started_at"], TIMESTAMP_FMT)
    row["ended_at"] = _format_timestamp(start + timedelta(seconds=seconds))


def guarantee_edge_cases(rng: Random, rows: list[dict[str, str]], pool: Pool) -> None:
    """Ensure rare-but-important cases appear somewhere in the span (in place)."""
    e_idx = [i for i, r in enumerate(rows) if r["rideable_type"] == "electric_bike"]
    if len(e_idx) < 6:
        return
    rng.shuffle(e_idx)
    a, b, c, d, e, f = e_idx[:6]

    rows[a]["start_station_name"] = rows[a]["start_station_id"] = ""  # dockless start
    rows[a]["start_lat"], rows[a]["start_lng"] = _fmt_coords(
        rng, *_dockless_point(rng, pool), "electric_bike"
    )
    rows[b]["end_station_name"] = rows[b]["end_station_id"] = ""  # dockless end
    rows[c]["end_lat"] = rows[c]["end_lng"] = ""  # null end coords
    _force_duration(rows[d], 21600 + rng.uniform(0, 3600))  # ~6h ride
    _force_duration(rows[e], rng.uniform(3, 29))  # sub-30s ride
    st = rng.choice(pool.stations)  # name, blank id
    rows[f]["end_station_name"], rows[f]["end_station_id"] = st[1], ""


def generate_dataset(
    rng: Random, total_rows: int, span: list[tuple[int, int]], guarantee: bool, pool: Pool
) -> list[dict[str, str]]:
    """Build every trip row across the span, optionally forcing edge cases in."""
    counts = allocate_counts(total_rows, span)
    seen_ids: set[str] = set()
    rows: list[dict[str, str]] = []
    for (year, month), n in zip(span, counts):
        rows.extend(_build_row(rng, year, month, seen_ids, pool) for _ in range(n))
    if guarantee:
        guarantee_edge_cases(rng, rows, pool)
    rows.sort(key=lambda r: r["started_at"])
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_rows(rows: list[dict[str, str]], span: list[tuple[int, int]]) -> None:
    """Assert invariants the source guarantees. Raises on any violation."""
    name_to_id: dict[str, str] = {}
    months_present: set[str] = set()
    for i, row in enumerate(rows):
        assert set(row) == set(COLUMNS), f"row {i}: column set mismatch"
        assert len(row["ride_id"]) == 16, f"row {i}: ride_id not 16 chars"

        started = datetime.strptime(row["started_at"], TIMESTAMP_FMT)
        ended = datetime.strptime(row["ended_at"], TIMESTAMP_FMT)
        assert ended > started, f"row {i}: ended_at not after started_at"
        months_present.add(f"{started.year}-{started.month:02d}")

        assert row["start_lat"] and row["start_lng"], f"row {i}: null start coords"
        if row["rideable_type"] == "classic_bike":
            assert row["start_station_id"], f"row {i}: classic with blank station"

        for side in ("start", "end"):
            name, sid = row[f"{side}_station_name"], row[f"{side}_station_id"]
            if name.strip() and sid.strip():
                prev = name_to_id.setdefault(name, sid)
                assert prev == sid, f"row {i}: station '{name}' has inconsistent id"

    ride_ids = [r["ride_id"] for r in rows]
    assert len(ride_ids) == len(set(ride_ids)), "duplicate ride_id found"
    expected_months = {f"{y}-{m:02d}" for y, m in span}
    assert months_present <= expected_months, "rows fall outside the requested span"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict[str, str]], handle: TextIO) -> None:
    """Write ``rows`` to ``handle`` as CSV in canonical column order with CRLF."""
    writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)


def group_by_start_month(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group rows by their start month 'YYYYMM' (matches vendor file naming)."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["started_at"][:7].replace("-", "")].append(row)
    return groups


def write_monthly_files(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    """Write one vendor-named CSV per start month into ``out_dir``; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for yyyymm, mrows in sorted(group_by_start_month(rows).items()):
        mrows.sort(key=lambda r: r["started_at"])
        path = out_dir / f"{yyyymm}-capitalbikeshare-tripdata.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            write_csv(mrows, handle)
        written.append(path)
    return written


def _print_summary(rows: list[dict[str, str]], written: list[Path], out_dir: Path) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["started_at"][:7].replace("-", "")] += 1
    peak = max(counts.values()) if counts else 1
    print(f"Wrote {len(rows)} trips across {len(written)} files -> {out_dir}", file=sys.stderr)
    for yyyymm in sorted(counts):
        bar = "#" * round(counts[yyyymm] / peak * 30)
        print(f"  {yyyymm[:4]}-{yyyymm[4:]}  {counts[yyyymm]:>4}  {bar}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def generate_and_write(
    *,
    input_extract: str,
    output_dir: str,
    start_month: str,
    num_months: int,
    total_rows: int,
    seed: int,
    guarantee: bool,
    to_stdout: bool = False,
) -> list[Path]:
    """Core run used by both the notebook config path and the CLI."""
    if num_months < 1:
        raise ValueError("num_months must be >= 1")
    if total_rows < num_months:
        raise ValueError(
            f"total_rows ({total_rows}) must be >= num_months ({num_months}) "
            "so every month gets at least one trip."
        )
    year, month = parse_month(start_month)
    span = month_span(year, month, num_months)
    pool = _resolve_pool(input_extract)
    rng = Random(seed)
    rows = generate_dataset(rng, total_rows, span, guarantee, pool)
    validate_rows(rows, span)

    if to_stdout:
        write_csv(rows, sys.stdout)
        return []
    out_dir = Path(output_dir)
    written = write_monthly_files(rows, out_dir)
    _print_summary(rows, written, out_dir)
    return written


def run() -> list[Path]:
    """Run using the CONFIG block above. Intended for notebook / manual use."""
    return generate_and_write(
        input_extract=INPUT_EXTRACT,
        output_dir=OUTPUT_DIR,
        start_month=START_MONTH,
        num_months=NUM_MONTHS,
        total_rows=TOTAL_ROWS,
        seed=SEED,
        guarantee=GUARANTEE_EDGE_CASES,
    )


def parse_month(value: str) -> tuple[int, int]:
    """Parse a ``YYYY-MM`` string into a ``(year, month)`` pair."""
    try:
        dt = datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"month must be in YYYY-MM format, got {value!r}") from exc
    return dt.year, dt.month


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, defaulting to the CONFIG block values."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic Capital Bikeshare fixtures over time. "
        "Defaults come from the CONFIG block in this file.",
    )
    parser.add_argument(
        "--input-extract",
        default=INPUT_EXTRACT,
        help="Real .zip/.csv extract to harvest stations+bbox from.",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR, help="Directory for the per-month CSV files."
    )
    parser.add_argument("--start", default=START_MONTH, metavar="YYYY-MM", help="First month.")
    parser.add_argument(
        "--months", type=int, default=NUM_MONTHS, help="Number of consecutive months."
    )
    parser.add_argument(
        "--rows", type=int, default=TOTAL_ROWS, help="Total trips across all months."
    )
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed.")
    parser.add_argument(
        "--no-guarantee",
        dest="guarantee",
        action="store_false",
        default=GUARANTEE_EDGE_CASES,
        help="Do not force rare edge cases into the output.",
    )
    parser.add_argument(
        "--stdout", action="store_true", help="Write one combined CSV to stdout instead of files."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code."""
    args = parse_args(argv)
    try:
        generate_and_write(
            input_extract=args.input_extract,
            output_dir=args.output_dir,
            start_month=args.start,
            num_months=args.months,
            total_rows=args.rows,
            seed=args.seed,
            guarantee=args.guarantee,
            to_stdout=args.stdout,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _in_ipython() -> bool:
    try:
        get_ipython  # type: ignore[name-defined]  # noqa: B018
        return True
    except NameError:
        return "ipykernel" in sys.modules


if __name__ == "__main__":
    # In a notebook (pasted cell or %run), use the CONFIG block instead of
    # argparse, which would otherwise try to parse the kernel's own argv.
    if _in_ipython():
        run()
    else:
        raise SystemExit(main())
