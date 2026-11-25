"""Pilot-test AI passenger counting on bus door-camera video against ground-truth counts.

Requires two packages beyond ArcGIS Pro's arcgispro-py3 environment (install them
into a clone of it; the version cap keeps numpy 1.x, which pandas 1.x needs):
    pip install anthropic
    pip install "opencv-python<4.12"

Sends each door-open window from onboard door-camera clips to a Claude model
through the Anthropic API, asks for boardings and alightings, and compares the
answers with ground-truth counts (a Ridecheck export, or manual counts taken
from the same clips). It measures accuracy, run-to-run repeatability and cost;
it is not a production counter. Door-open windows come from CLEVER AVL door
times on purpose, so the pilot measures counting alone. MODE = "stage2" tests
whether the outside camera alone can tell which stops the bus served or passed.

The clips show riders: confirm agency policy on sending footage to an outside
API before running on real footage. Frames stay in memory or on a local disk.

Inputs
------
- VIDEO_DIR: clips exported from the onboard recorder (MP4 or any format OpenCV
  reads), one per camera per trip, with bus, camera and start time in the name.
- STOP_EVENTS_PATH: CLEVER Stop Visit Events export (.csv or .xlsx).
- MANUAL_COUNTS_CSV (start from MODE = "template") or RIDECHECK_PATH.
- GTFS_PATH (Stage 2 only): stop_times.txt and stops.txt, folder or .zip.

Outputs
-------
- ``<prefix>_<model>_<timestamp>.xlsx`` with Events, Trip summary, Accuracy,
  Repeatability, Cost and Reconciliation sheets (Stage 2: Detections, Stop
  visits, Scores, Cost and Reconciliation).
- A ``_runlog.txt`` sidecar with the verbatim CONFIGURATION block, resolved
  parameters, input SHA-256 fingerprints, model ID, prompt hash and run totals.
- MODE = "template": a blank manual-count CSV. MODE = "calibrate": still frames
  at door-open times, saved to a local folder, for checking the clock offset.

Typical usage
-------------
Set the paths in the CONFIGURATION section (or pass the matching CLI flags),
run MODE = "calibrate" to check CLOCK_OFFSET_SECONDS, run with DRY_RUN = True to
see the estimated cost, then run for real. Works from a shell, ArcGIS Pro's
Python window or a Jupyter notebook cell. ``--self-test`` (or SELF_TEST = True)
runs the built-in tests on synthetic clips with a stub API client.
"""

from __future__ import annotations

import argparse
import ast
import base64
import builtins
import concurrent.futures
import datetime as _dt
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================
# The clips show riders. Before running on real footage, confirm agency policy on
# sending video frames to an outside API (here, Anthropic's). Frames are kept in
# memory. KEEP_FRAMES copies and calibration stills go to FRAMES_DIR, which must be
# a local disk, never a shared drive. OUTPUT_DIR only receives tables and run logs.
#
# Modes:
#   "count"      Stage 1: count boardings and alightings per door-open window, score them.
#   "template"   write a blank manual-count CSV, one row per event and door with video.
#   "calibrate"  save a still at each of the first CALIBRATION_STILLS door-open times,
#                to check by eye that the doors are opening, then fix the clock offset.
#   "stage2"     Stage 2: name the stops served or passed, from the outside camera alone.
#
# Where the brief left a choice open, the simpler option was taken and noted in a
# comment beside the code that makes it.
# === BEGIN CONFIG ===

# --- Run mode -----------------------------------------------------------------
MODE: str = "count"  # "count", "template", "calibrate" or "stage2" (see above)
DRY_RUN: bool = False  # True = do everything except the API calls; logs the cost estimate
SELF_TEST: bool = False  # True = run the built-in tests on synthetic data instead

# --- Inputs -------------------------------------------------------------------
VIDEO_DIR: str = r"Path\To\Your\Video_Clips"  # exported clips, one per camera per trip
STOP_EVENTS_PATH: str = r"Path\To\Your\stop_visit_events.csv"  # CLEVER export, .csv/.xlsx
# Ground truth. "manual" = counts you take by watching the same clips (start from
# the MODE = "template" CSV); differences then measure model error alone.
# "ridecheck" = an on-board ride check of the same trips; differences then also
# include what a person on board sees that the camera does not (riders hidden from
# the lens, a doorway the camera only half covers) and the checker's own misses.
GROUND_TRUTH_SOURCE: str = "manual"  # "manual" or "ridecheck"
MANUAL_COUNTS_CSV: str = r"Path\To\Your\manual_counts.csv"  # filled-in copy of the template
RIDECHECK_PATH: str = r"Path\To\Your\ridecheck_export.csv"  # Ridecheck export, .csv/.xlsx
GTFS_PATH: str = r"Path\To\Your\GTFS"  # Stage 2 only: GTFS feed folder or .zip

# --- Outputs ------------------------------------------------------------------
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"  # workbook, template CSV and run log
FRAMES_DIR: str = ""  # local folder for kept frames and stills; "" = this PC's temp folder
OUTPUT_PREFIX: str = "video_count_pilot"  # file stem; model ID and timestamp are appended
TEMPLATE_PREFIX: str = "manual_counts_template"  # MODE = "template" file stem
REQUIRE_RUN_LOG: bool = True  # True = a failed run-log write aborts the run

# --- Video file names and clocks ----------------------------------------------
# Named groups <bus>, <camera> and <start>, searched in the file name without its
# extension. The default matches names like 1234_cam1_20260302_070000.mp4.
FILENAME_REGEX: str = r"^(?P<bus>\d+)_(?P<camera>[A-Za-z0-9]+)_(?P<start>\d{8}_\d{6})"  # name
FILENAME_TIME_FORMAT: str = "%Y%m%d_%H%M%S"  # strptime format of the <start> group
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")  # files to open
CAMERA_DOORS: Dict[str, str] = {"cam1": "front", "cam2": "rear"}  # <camera> code -> door
OUTSIDE_CAMERA: str = "cam3"  # Stage 2: <camera> code of the forward-facing outside camera
# The recorder and AVL clocks drift apart. AVL time = recorder (video) time + this
# offset, so use a positive value when the recorder clock runs behind AVL. Check it
# with MODE = "calibrate": if the doors open d seconds AFTER a still, subtract d;
# if they were already open d seconds before the still, add d.
CLOCK_OFFSET_SECONDS: float = 0.0  # added to recorder time to get AVL time

# --- CLEVER Stop Visit Events columns -----------------------------------------
# Scheduled times in these exports carry a placeholder 1899 date. The script never
# uses them; it uses the actual times, which carry the real service date.
EVENT_TRIP_COL: str = "TRIP_ID"  # trip ID; Stage 2 also matches it to GTFS trip_id
EVENT_BUS_COL: str = "VEHICLE_ID"  # bus number, matched to <bus> in the file names
EVENT_STOP_CODE_COL: str = "STOP_CODE"  # public stop code (= Ridecheck STOP_ID)
EVENT_SEQ_COL: str = "STOP_SEQUENCE"  # stop sequence within the trip
EVENT_DOOR_OPEN_COL: str = "ACT_DOOR_OPEN_TIME"  # actual door-open timestamp
EVENT_DOOR_CLOSE_COL: str = "ACT_DOOR_CLOSE_TIME"  # actual door-close timestamp
EVENT_ARRIVAL_COL: str = "ACT_ARRIVAL_TIME"  # Stage 2 arrival time; "" = door open only
EVENT_TIME_FORMAT: str = ""  # strptime format of the timestamps; "" = let pandas infer
TRIPS_TO_INCLUDE: Tuple[str, ...] = ()  # limit the run to these trip IDs; () = all trips

# --- Ridecheck columns (one row per trip and stop, on/off columns per door) ---
RIDECHECK_TRIP_COL: str = "TRIP_ID"  # trip ID, same values as the CLEVER trip column
RIDECHECK_STOP_COL: str = "STOP_ID"  # Ridecheck STOP_ID = CLEVER stop_code
RIDECHECK_SEQ_COL: str = ""  # stop sequence column; "" = join on trip and stop only
RIDECHECK_FRONT_ON_COL: str = "FRONT_ON"  # front-door boardings
RIDECHECK_FRONT_OFF_COL: str = "FRONT_OFF"  # front-door alightings
RIDECHECK_REAR_ON_COL: str = "REAR_ON"  # rear-door boardings
RIDECHECK_REAR_OFF_COL: str = "REAR_OFF"  # rear-door alightings

# --- Frames -------------------------------------------------------------------
PAD_SECONDS: float = 3.0  # window = door open - pad to door close + pad
FRAME_RATE: float = 2.0  # frames per second of video sent to the model
FRAME_WIDTH: int = 640  # pixels; each frame costs ceil(w/28) x ceil(h/28) tokens
FRAME_HEIGHT: int = 480  # pixels
JPEG_QUALITY: int = 85  # JPEG quality, 1-100
KEEP_FRAMES: bool = False  # True = also save the JPEGs sent to the API in FRAMES_DIR
CALIBRATION_STILLS: int = 10  # MODE = "calibrate": door-open events to grab stills for

# --- Model and requests -------------------------------------------------------
# The models' own effort defaults differ (medium on Opus 5.5 and Haiku 5.5, high on
# Sonnet 5.5), so EFFORT is set explicitly to keep runs comparable across models.
MODEL_ID: str = "claude-haiku-5-5"  # claude-haiku-5-5, claude-sonnet-5-5 or claude-opus-5-5
EFFORT: str = "medium"  # low, medium, high, xhigh or max; "" = the model's default
MAX_OUTPUT_TOKENS: int = 16000  # cap per response, thinking tokens included
REQUEST_TOKEN_CEILING: int = 90_000  # split a clip whose request would exceed this
MAX_IMAGES_PER_REQUEST: int = 600  # API limit for 1M-context models (docs, Nov 2026)
TEXT_CHARS_PER_TOKEN: float = 3.0  # rough text-token estimate (conservative)
REQUEST_OVERHEAD_TOKENS: int = 100  # estimate allowance for formatting and output schema
CACHE_MIN_TOKENS: int = 512  # shortest prompt prefix the 5.5 models will cache
REPEATS: int = 3  # times each request is sent, to measure run-to-run variation
MAX_API_RETRIES: int = 4  # SDK retries with exponential backoff (429, 5xx, network)
MAX_PARSE_RETRIES: int = 2  # re-sends of a request whose answer can't be parsed
MAX_CONSECUTIVE_API_ERRORS: int = 10  # stop the run after this many API errors in a row
MAX_CONCURRENT_REQUESTS: int = 4  # clips processed at the same time

# --- Cost ---------------------------------------------------------------------
# USD per million tokens (claude.com/pricing, Nov 2026); check before a run.
# cache_read = price of cached input tokens. Haiku 5.5 charges more for prompts
# over 100,000 tokens, which REQUEST_TOKEN_CEILING keeps requests under.
MODEL_PRICES: Dict[str, Dict[str, float]] = {  # model ID -> USD per million tokens
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},  # Opus 5.5
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10},  # Sonnet 5.5
    "claude-haiku-5-5": {  # Haiku 5.5, prompts up to 100,000 tokens:
        "input": 0.10,  # input
        "output": 0.50,  # output
        "cache_read": 0.01,  # cached input (assumed 0.1x input)
        "long_prompt_tokens": 100_000.0,  # prompts above this many tokens cost:
        "long_input": 0.50,  # input
        "long_output": 2.50,  # output
    },
}
CACHE_WRITE_MULTIPLIER: float = 1.25  # 5-minute cache writes cost 1.25x the input price
EST_OUTPUT_TOKENS_PER_REQUEST: int = 1500  # assumed output incl. thinking, for estimates
BUDGET_USD: float = 25.0  # stop before any API call if the estimate is higher
OVERRIDE_BUDGET: bool = False  # True = run even when the estimate exceeds BUDGET_USD
DOOR_OPEN_SHARE: float = 0.10  # share of revenue time that falls in a padded door window
DOOR_CAMERAS_PER_BUS: int = 2  # door cameras a production system would process

# --- Stage 2 (MODE = "stage2") ------------------------------------------------
STAGE2_FRAME_RATE: float = 1.0  # frames per second of outside-camera footage
STAGE2_WINDOW_SECONDS: float = 120.0  # consecutive footage per request
STAGE2_MATCH_TOLERANCE_SECONDS: float = 60.0  # max gap from a detection to the real visit
STAGE2_DEDUPE_SECONDS: float = 60.0  # same stop reported again within this = one detection

LOG_LEVEL: int = logging.INFO  # logging verbosity

# === END CONFIG ===

# =============================================================================
# CONSTANTS AND PROMPTS
# =============================================================================

MODES: Tuple[str, ...] = ("count", "template", "calibrate", "stage2")
EFFORT_LEVELS: Tuple[str, ...] = ("", "low", "medium", "high", "xhigh", "max")
CONFIDENCE_LEVELS: Tuple[str, ...] = ("low", "medium", "high")
CROWDING_BANDS: Tuple[str, ...] = ("0", "1-2", "3-5", "6+")
# Images on Claude 4.7 and later models are downscaled past these limits, which
# would make the token estimate wrong, so the frame size must stay inside them.
MAX_IMAGE_LONG_EDGE: int = 2576
MAX_IMAGE_TOKENS: int = 4784
_CONFIG_BEGIN = "# === BEGIN CONFIG ==="
_CONFIG_END = "# === END CONFIG ==="

COUNT_SYSTEM_PROMPT: str = """\
You count passengers at one door of a transit bus. You will see still frames from \
a fixed security camera inside the bus that looks at the doorway. The frames are in \
time order, and a label before each frame gives its time in seconds from the start \
of the clip. A clip covers one stop: it starts a few seconds before the doors open \
and ends a few seconds after they close.

Your task: count the people who BOARDED (moved from outside the bus, across the door \
threshold, into the bus) and the people who ALIGHTED (moved from inside the bus, \
across the threshold, out of the bus) through this door during the clip.

Counting rules:
1. Count a person only when they fully cross the threshold (the door line at the \
bottom step) during the clip. If someone is partway across in the first or last \
frame and you cannot see the crossing completed inside the clip, do not count them, \
and say so in the notes.
2. Count each person once. The same person appears in several consecutive frames; \
follow people by position, clothing and what they carry.
3. Do not count the bus operator, even if the operator steps into or through the \
doorway, for example to deploy a ramp or help a rider.
4. Do not count anyone who steps out and back in, or in and back out, during the \
clip (for example a rider who steps off to let others leave). Their net movement \
is zero.
5. A person using a wheelchair, mobility device or stroller counts as one boarding \
or alighting like anyone else. Children count as people. Do not count bags, carts \
or strollers separately.
6. People outside who never cross, and riders inside who move toward the door but \
do not leave, are not counted.
7. Fast movement may appear in only one or two frames, so judge from the whole \
sequence.

Confidence:
- "high": the doorway was clearly visible throughout and every crossing was clear.
- "medium": some uncertainty, such as a brief occlusion or a crowd at the door, but \
the counts are probably right.
- "low": the view of the threshold was blocked for part of the clip, the image was \
too dark or blurred, a crowd made people hard to tell apart, or you are guessing.

Notes: one or two short sentences on anything that affected the count, for example \
view blocked (by what, and when), person lingering at the door, wheelchair ramp or \
lift deployed, operator crossing the door, person stepping out and back in, \
crowding, poor lighting, or a crossing cut off at the start or end of the clip. \
Always say when the view of the threshold was blocked. Write "none" if nothing \
affected the count."""

CLIP_HEADER: str = "Clip: trip {trip}, stop {stop_code} (stop sequence {seq}), {door} door camera."
CHUNK_NOTE: str = (
    "This clip is long, so it was split into parts. These frames are part {part} of "
    "{parts}, in time order. Count only crossings completed within these frames."
)
FRAME_LABEL: str = "t = {t:.1f} s"
COUNT_QUESTION: str = (
    "How many people boarded and how many alighted through this door in these "
    "frames? Follow the counting rules and answer in the requested JSON format."
)
COUNT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "boardings": {"type": "integer"},
        "alightings": {"type": "integer"},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
        "notes": {"type": "string"},
    },
    "required": ["boardings", "alightings", "confidence", "notes"],
    "additionalProperties": False,
}

STAGE2_SYSTEM_PROMPT: str = """\
You review still frames from a forward-facing camera on a transit bus. The frames \
are in time order, and a label before each frame gives its time in seconds from the \
start of the recording. You are also given the ordered list of stops on the bus's \
route pattern, as stop code and stop name.

Your task: name each stop on the list that the bus ARRIVED at (came to a stop at the \
bus stop) or PASSED (drove past the bus stop without stopping) during these frames. \
Use what you can see: bus stop signs and poles, shelters, benches, people waiting, \
street name signs, intersections and landmarks, together with the order of stops in \
the list.

For each stop you identify, give its stop code exactly as listed, the time label of \
the frame where the bus is at or alongside the stop, and whether it stopped or \
passed. Report only stops you see evidence for; do not add a stop just because it \
comes next in the list. If you cannot identify any stop in these frames, return an \
empty list. In the notes, say in one short sentence what limited your view, such \
as glare, darkness or a blocked windshield, or write "none"."""

STAGE2_STOPS_INTRO: str = "Stops on this trip's pattern, in order (stop code - stop name):\n"
STAGE2_FRAME_LABEL: str = "t = {t:.0f} s"
STAGE2_QUESTION: str = (
    "Which stops on the list did the bus arrive at or pass in these frames? Answer in "
    "the requested JSON format."
)
STAGE2_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "stops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stop_code": {"type": "string"},
                    "frame_time_s": {"type": "number"},
                    "action": {"type": "string", "enum": ["stopped", "passed"]},
                },
                "required": ["stop_code", "frame_time_s", "action"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["stops", "notes"],
    "additionalProperties": False,
}

EVENT_COLUMNS: List[str] = [
    "trip",
    "bus",
    "stop_code",
    "stop_sequence",
    "door",
    "repeat",
    "video_file",
    "clip_start_s",
    "clip_end_s",
    "window_start",
    "window_end",
    "frames_sent",
    "model_boardings",
    "gt_boardings",
    "boardings_error",
    "boardings_abs_error",
    "model_alightings",
    "gt_alightings",
    "alightings_error",
    "alightings_abs_error",
    "confidence",
    "notes",
    "split_flag",
    "chunks",
    "status",
    "detail",
    "input_tokens",
    "output_tokens",
    "cache_write_tokens",
    "cache_read_tokens",
    "est_input_tokens",
    "cost_usd",
    "est_cost_usd",
    "api_calls",
    "model_id",
    "prompt_hash",
]
RECON_COLUMNS: List[str] = [
    "category",
    "trip",
    "bus",
    "stop_code",
    "stop_sequence",
    "door",
    "video_file",
    "repeat",
    "detail",
]
VIDEO_COLUMNS: List[str] = [
    "path",
    "file",
    "bus",
    "camera",
    "kind",
    "door",
    "start_recorder",
    "start_avl",
    "fps",
    "frames",
    "last_pos",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class PilotError(Exception):
    """A problem that stops the run with exit code 1 and a clear message."""


class FatalApiError(Exception):
    """An API error that every later request would hit too (key, model or request)."""


class Settings(NamedTuple):
    """Resolved run settings: the CONFIGURATION values after any CLI overrides."""

    mode: str
    dry_run: bool
    self_test: bool
    video_dir: str
    stop_events_path: str
    ground_truth_source: str
    manual_counts_csv: str
    ridecheck_path: str
    gtfs_path: str
    output_dir: str
    frames_dir: str
    output_prefix: str
    template_prefix: str
    require_run_log: bool
    filename_regex: str
    filename_time_format: str
    video_extensions: Tuple[str, ...]
    camera_doors: Dict[str, str]
    outside_camera: str
    clock_offset_seconds: float
    event_trip_col: str
    event_bus_col: str
    event_stop_code_col: str
    event_seq_col: str
    event_door_open_col: str
    event_door_close_col: str
    event_arrival_col: str
    event_time_format: str
    trips_to_include: Tuple[str, ...]
    ridecheck_trip_col: str
    ridecheck_stop_col: str
    ridecheck_seq_col: str
    ridecheck_front_on_col: str
    ridecheck_front_off_col: str
    ridecheck_rear_on_col: str
    ridecheck_rear_off_col: str
    pad_seconds: float
    frame_rate: float
    frame_width: int
    frame_height: int
    jpeg_quality: int
    keep_frames: bool
    calibration_stills: int
    model_id: str
    effort: str
    max_output_tokens: int
    request_token_ceiling: int
    max_images_per_request: int
    text_chars_per_token: float
    request_overhead_tokens: int
    cache_min_tokens: int
    repeats: int
    max_api_retries: int
    max_parse_retries: int
    max_consecutive_api_errors: int
    max_concurrent_requests: int
    model_prices: Dict[str, Dict[str, float]]
    cache_write_multiplier: float
    est_output_tokens_per_request: int
    budget_usd: float
    override_budget: bool
    door_open_share: float
    door_cameras_per_bus: int
    stage2_frame_rate: float
    stage2_window_seconds: float
    stage2_match_tolerance_seconds: float
    stage2_dedupe_seconds: float
    log_level: int


class ClipJob(NamedTuple):
    """One door-open event seen by one door camera: the unit sent to the model."""

    trip: str
    bus: str
    stop_code: str
    stop_sequence: str
    door: str
    video_path: Path
    video_fps: float
    video_frames: int
    clip_start_avl: _dt.datetime
    start_pos: float
    end_pos: float
    door_open: _dt.datetime
    door_close: _dt.datetime
    gt_boardings: Optional[int] = None
    gt_alightings: Optional[int] = None


class PlannedJob(NamedTuple):
    """A clip job with its frame times, request chunks and token estimates."""

    job: ClipJob
    frame_times: List[float]
    chunks: List[Tuple[int, int]]
    chunk_tokens: List[int]


class PlannedWindow(NamedTuple):
    """One Stage 2 window of consecutive outside-camera frames for one trip."""

    trip: str
    bus: str
    video_path: Path
    video_fps: float
    video_frames: int
    clip_start_avl: _dt.datetime
    window_index: int
    frame_times: List[float]
    chunks: List[Tuple[int, int]]
    chunk_tokens: List[int]
    stop_list_text: str
    pattern_codes: Tuple[str, ...]


class CallResult(NamedTuple):
    """Outcome of one request to the model, including any parse retries."""

    status: str
    data: Optional[Dict[str, Any]]
    detail: str
    usage: Dict[str, int]
    cost_usd: float
    api_calls: int


class RunControl:
    """Shared stop flag and error counter for the worker threads of one run."""

    def __init__(self, max_consecutive_errors: int) -> None:
        """Create the controller.

        Args:
            max_consecutive_errors: API errors in a row (across all workers) that
                stop the run, so a dead network or missing key does not produce
                thousands of failed rows.
        """
        self.abort = threading.Event()
        self.fatal_message = ""
        self._max_errors = max_consecutive_errors
        self._consecutive_errors = 0
        self._lock = threading.Lock()

    def stop(self, message: str) -> None:
        """Stop the run, keeping the first reason given.

        Args:
            message: Why the run stopped.
        """
        with self._lock:
            if not self.fatal_message:
                self.fatal_message = message
        self.abort.set()

    def record(self, api_error: bool, message: str = "") -> None:
        """Count consecutive API errors and stop the run when there are too many.

        Args:
            api_error: True if the request failed with an API error.
            message: Error text, used as the stop reason.
        """
        with self._lock:
            self._consecutive_errors = self._consecutive_errors + 1 if api_error else 0
            too_many = self._consecutive_errors >= self._max_errors
        if too_many:
            self.stop(f"{self._max_errors} API errors in a row; last one: {message}")


# =============================================================================
# GENERAL HELPERS
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


@contextmanager
def open_gtfs_file(
    gtfs_path: str | os.PathLike[str], file_name: str
) -> Generator[IO[bytes], None, None]:
    """Open one GTFS text file from a feed folder or a ``.zip`` of the feed.

    The counterpart of :func:`load_gtfs_data` for scripts that read a file with
    their own options (``pandas.read_csv`` with ``usecols`` or
    ``keep_default_na=False``, or the :mod:`csv` module), so they accept a
    zipped feed without extracting it. Zip members are found the way
    :func:`load_gtfs_data` finds them: at the archive root or inside a single
    wrapper folder.

    Args:
        gtfs_path: Folder containing the GTFS feed, or a ``.zip`` archive of it.
        file_name: GTFS file to open, e.g. ``"stops.txt"``.

    Yields:
        A binary file handle. ``pandas.read_csv`` reads it directly; wrap it in
        :class:`io.TextIOWrapper` for the :mod:`csv` module.

    Raises:
        FileNotFoundError: *gtfs_path* does not exist, or the feed has no
            *file_name*.
        ValueError: *gtfs_path* is neither a directory nor a valid ``.zip``
            file, or *file_name* is found in more than one place in the zip.
    """
    path = os.fspath(gtfs_path)
    if os.path.isdir(path):
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing GTFS file in '{path}': {file_name}")
        with open(file_path, "rb") as handle:
            yield handle
        return

    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    if not (os.path.isfile(path) and path.lower().endswith(".zip")):
        raise ValueError(f"'{path}' is neither a directory nor a .zip file.")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid zip archive.") from exc
    with archive:
        matches = [name for name in archive.namelist() if os.path.basename(name) == file_name]
        if not matches:
            raise FileNotFoundError(f"Missing GTFS file in '{path}': {file_name}")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous GTFS file in '{path}' (found in multiple locations): {file_name}"
            )
        with archive.open(matches[0]) as handle:
            yield handle


def _clean_id(value: Any) -> str:
    """Return an ID as a clean string, ready for a join.

    Strips whitespace and removes the trailing ".0" that Excel float conversion
    adds to numeric IDs, so "1234.0" becomes "1234". Blank or missing values
    become "".

    Args:
        value: Raw cell value (normally a string; tables are read with dtype=str).

    Returns:
        The cleaned ID.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _bus_key(bus: str) -> str:
    """Return a bus number without leading zeros, so "01234" matches "1234".

    Args:
        bus: Cleaned bus ID.

    Returns:
        The bus ID with leading zeros removed when it is all digits.
    """
    if bus.isdigit():
        return bus.lstrip("0") or "0"
    return bus


def _is_placeholder(value: str) -> bool:
    r"""Return True if a path setting is blank or still the shipped placeholder.

    Args:
        value: The path setting.

    Returns:
        True for "" and for values that start with or contain Path\To.
    """
    text = str(value).strip().replace("/", "\\").lower()
    return text == "" or text.startswith("path\\to") or "\\path\\to\\" in text


def _is_network_path(path: str) -> bool:
    """Return True if *path* is on a network share (UNC path or mapped drive).

    Args:
        path: Folder path to test.

    Returns:
        True for UNC paths and, on Windows, drive letters mapped to a share.
    """
    text = str(path).strip()
    if text.startswith("\\\\") or text.startswith("//"):
        return True
    if sys.platform == "win32":
        import ctypes

        drive = os.path.splitdrive(os.path.abspath(text))[0]
        if drive.endswith(":"):
            drive_remote = 4  # GetDriveTypeW code for a network drive
            return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == drive_remote
    return False


def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel table with every column as text.

    Args:
        path: .csv, .txt, .xlsx, .xlsm or .xls file.

    Returns:
        The table, all values as strings, blanks as "".
    """
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        table = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        try:
            table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
    table.columns = [str(col).strip() for col in table.columns]
    return table.fillna("")


def _parse_times(values: pd.Series, time_format: str) -> pd.Series:
    """Parse a text column of timestamps, turning blanks and bad values into NaT.

    Args:
        values: Text timestamps.
        time_format: strptime format, or "" to let pandas infer it.

    Returns:
        A datetime64 Series.
    """
    text = values.fillna("").astype(str).str.strip()
    text = text.where(~text.str.lower().isin(["", "nan", "nat", "none"]))
    if time_format:
        return pd.to_datetime(text, format=time_format, errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MB chunks.

    Args:
        path: File to fingerprint.

    Returns:
        The hex digest, or "unreadable: <reason>" if the file can't be read.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        return f"unreadable: {exc}"
    return digest.hexdigest()


def _text_tokens(text: str, chars_per_token: float) -> int:
    """Estimate the tokens in a text block.

    Args:
        text: The text.
        chars_per_token: Characters per token assumed.

    Returns:
        Estimated token count (rounded up).
    """
    return int(math.ceil(len(text) / chars_per_token))


def image_tokens(width: int, height: int) -> int:
    """Return the visual tokens an image costs: ceil(width/28) x ceil(height/28).

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Token count for one image.
    """
    return int(math.ceil(width / 28) * math.ceil(height / 28))


def frame_times(start_pos: float, end_pos: float, fps: float) -> List[float]:
    """Return evenly spaced sample times from *start_pos* to *end_pos* inclusive.

    Args:
        start_pos: First sample, seconds into the video file.
        end_pos: Last allowed sample, seconds into the video file.
        fps: Samples per second.

    Returns:
        Sample times in seconds.
    """
    if end_pos < start_pos:
        return []
    count = int(math.floor((end_pos - start_pos) * fps + 1e-9)) + 1
    return [round(start_pos + k / fps, 6) for k in range(count)]


def plan_chunks(
    n_frames: int, per_frame_tokens: int, fixed_tokens: int, ceiling: int, max_images: int
) -> List[Tuple[int, int]]:
    """Split a run of frames into consecutive requests under the token and image limits.

    Args:
        n_frames: Frames in the clip.
        per_frame_tokens: Tokens per frame, image plus its text label.
        fixed_tokens: Tokens per request that don't depend on the frames.
        ceiling: Largest allowed request, in tokens.
        max_images: Largest number of images allowed in one request.

    Returns:
        (start, end) index pairs, end exclusive, of near-equal size.

    Raises:
        PilotError: The ceiling can't fit even one frame.
    """
    if n_frames <= 0:
        return []
    room = ceiling - fixed_tokens
    per_request = min(max_images, room // per_frame_tokens) if per_frame_tokens > 0 else max_images
    if per_request < 1:
        raise PilotError(
            f"REQUEST_TOKEN_CEILING ({ceiling}) leaves no room for a frame "
            f"({per_frame_tokens} tokens) after the {fixed_tokens} fixed tokens per request."
        )
    n_chunks = int(math.ceil(n_frames / per_request))
    base, extra = divmod(n_frames, n_chunks)
    chunks: List[Tuple[int, int]] = []
    start = 0
    for index in range(n_chunks):
        size = base + (1 if index < extra else 0)
        chunks.append((start, start + size))
        start += size
    return chunks


def parse_video_name(
    file_name: str, pattern: re.Pattern[str], time_format: str
) -> Optional[Dict[str, Any]]:
    """Parse bus, camera and recording start time from a clip's file name.

    Args:
        file_name: File name, with or without folder and extension.
        pattern: Compiled FILENAME_REGEX with groups bus, camera and start.
        time_format: strptime format of the start group.

    Returns:
        {"bus", "camera", "start"} or None if the name doesn't match.
    """
    match = pattern.search(Path(file_name).stem)
    if match is None:
        return None
    groups = match.groupdict()
    try:
        start = _dt.datetime.strptime(str(groups["start"]), time_format)
    except (KeyError, ValueError, TypeError):
        return None
    return {
        "bus": _bus_key(_clean_id(groups.get("bus"))),
        "camera": str(groups.get("camera") or "").strip().lower(),
        "start": start,
    }


def door_window_positions(
    door_open: _dt.datetime,
    door_close: _dt.datetime,
    clip_start_avl: _dt.datetime,
    last_pos: float,
    pad: float,
) -> Tuple[str, float, float]:
    """Locate a padded door-open window inside a clip, in seconds into the file.

    Args:
        door_open: AVL door-open time.
        door_close: AVL door-close time.
        clip_start_avl: Clip start in AVL time (recorder start + clock offset).
        last_pos: Time of the clip's last frame, seconds into the file.
        pad: Seconds added before door open and after door close.

    Returns:
        (coverage, start_pos, end_pos). coverage is "full" when the clip holds the
        whole door-open interval (the padding is cut at the clip's edges),
        "partial" when it holds only part of it, and "none" otherwise.
    """
    open_pos = (door_open - clip_start_avl).total_seconds()
    close_pos = (door_close - clip_start_avl).total_seconds()
    if open_pos >= 0 and close_pos <= last_pos:
        return "full", max(0.0, open_pos - pad), min(last_pos, close_pos + pad)
    if close_pos < 0 or open_pos > last_pos:
        return "none", 0.0, 0.0
    return "partial", max(0.0, open_pos - pad), min(last_pos, close_pos + pad)


def _door_names(settings: Settings) -> List[str]:
    """Return the door names from CAMERA_DOORS, in order, without repeats.

    Args:
        settings: Run settings.

    Returns:
        Door names, e.g. ["front", "rear"].
    """
    doors: List[str] = []
    for door in settings.camera_doors.values():
        if door not in doors:
            doors.append(door)
    return doors


def _recon(category: str, detail: str = "", **fields: Any) -> Dict[str, Any]:
    """Build one reconciliation row.

    Args:
        category: Short machine-readable category.
        detail: Human-readable explanation.
        **fields: Any of trip, bus, stop_code, stop_sequence, door, video_file, repeat.

    Returns:
        A dict with every reconciliation column.
    """
    row: Dict[str, Any] = {col: "" for col in RECON_COLUMNS}
    row.update(fields)
    row["category"] = category
    row["detail"] = detail
    return row


def _seq_number(values: pd.Series) -> pd.Series:
    """Return stop sequences as numbers for sorting (text that isn't a number is NaN).

    Args:
        values: Stop sequence column.

    Returns:
        Numeric Series.
    """
    return pd.to_numeric(values, errors="coerce")


def _safe_name(*parts: Any) -> str:
    """Join ID parts into a file-name-safe stem (trip IDs may hold "/" or spaces).

    Args:
        *parts: IDs and labels.

    Returns:
        Parts joined with "_", each with unsafe characters replaced by "-".
    """
    return "_".join(re.sub(r"[^A-Za-z0-9.-]+", "-", str(part)) for part in parts)


def _import_cv2() -> Any:
    """Import OpenCV, with an install hint if it is missing.

    Returns:
        The cv2 module.

    Raises:
        PilotError: OpenCV is not installed.
    """
    try:
        import cv2
    except ImportError as exc:
        raise PilotError(
            'OpenCV is not installed. Install it with: pip install "opencv-python<4.12"'
        ) from exc
    return cv2


# =============================================================================
# PRICES
# =============================================================================


def _tier_prices(prompt_tokens: int, prices: Dict[str, float]) -> Tuple[float, float, float]:
    """Return (input, output, cached-input) prices for a prompt of this size.

    Args:
        prompt_tokens: All input tokens of one request, cached or not.
        prices: One model's MODEL_PRICES entry.

    Returns:
        USD per million tokens for input, output and cached input.
    """
    p_in = prices["input"]
    p_out = prices["output"]
    p_cache = prices.get("cache_read", p_in)
    threshold = prices.get("long_prompt_tokens", 0.0)
    if threshold and prompt_tokens > threshold:
        scale = prices.get("long_input", p_in) / p_in if p_in else 1.0
        return prices.get("long_input", p_in), prices.get("long_output", p_out), p_cache * scale
    return p_in, p_out, p_cache


def estimate_request_cost(input_tokens: int, output_tokens: int, prices: Dict[str, float]) -> float:
    """Estimate one request's cost, assuming no cache discount.

    Args:
        input_tokens: Estimated input tokens.
        output_tokens: Assumed output tokens.
        prices: One model's MODEL_PRICES entry.

    Returns:
        Cost in USD.
    """
    p_in, p_out, _ = _tier_prices(input_tokens, prices)
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


def usage_cost(usage: Dict[str, int], prices: Dict[str, float], write_multiplier: float) -> float:
    """Return the actual cost of one response from its usage counts.

    Args:
        usage: Token counts with keys input, output, cache_write and cache_read.
        prices: One model's MODEL_PRICES entry.
        write_multiplier: Cache-write price as a multiple of the input price.

    Returns:
        Cost in USD.
    """
    prompt = usage["input"] + usage["cache_write"] + usage["cache_read"]
    p_in, p_out, p_cache = _tier_prices(prompt, prices)
    return (
        usage["input"] * p_in
        + usage["cache_write"] * p_in * write_multiplier
        + usage["cache_read"] * p_cache
        + usage["output"] * p_out
    ) / 1e6


# =============================================================================
# LOADING INPUTS
# =============================================================================


def load_stop_events(path: Path, settings: Settings) -> pd.DataFrame:
    """Load the CLEVER Stop Visit Events export, cleaning IDs and parsing actual times.

    Args:
        path: CSV or Excel export.
        settings: Run settings (column names, time format, trip filter).

    Returns:
        One row per stop visit with trip, bus, stop_code, stop_sequence, raw and
        parsed door times, and the arrival time (NaT if not configured).

    Raises:
        PilotError: The file is empty or lacks a configured column.
    """
    raw = _read_table(path)
    if raw.empty:
        raise PilotError(f"The stop-events file has no rows: {path}")
    needed = [
        settings.event_trip_col,
        settings.event_bus_col,
        settings.event_stop_code_col,
        settings.event_seq_col,
        settings.event_door_open_col,
        settings.event_door_close_col,
    ]
    missing = [col for col in needed if col not in raw.columns]
    if missing:
        raise PilotError(
            f"The stop-events file lacks column(s) {missing}. Columns found: "
            f"{list(raw.columns)}. Fix the EVENT_*_COL settings."
        )
    fmt = settings.event_time_format
    events = pd.DataFrame(
        {
            "trip": raw[settings.event_trip_col].map(_clean_id),
            "bus": raw[settings.event_bus_col].map(_clean_id).map(_bus_key),
            "stop_code": raw[settings.event_stop_code_col].map(_clean_id),
            "stop_sequence": raw[settings.event_seq_col].map(_clean_id),
            "door_open_raw": raw[settings.event_door_open_col].astype(str).str.strip(),
            "door_close_raw": raw[settings.event_door_close_col].astype(str).str.strip(),
            "door_open": _parse_times(raw[settings.event_door_open_col], fmt),
            "door_close": _parse_times(raw[settings.event_door_close_col], fmt),
        }
    )
    if settings.event_arrival_col and settings.event_arrival_col in raw.columns:
        events["arrival"] = _parse_times(raw[settings.event_arrival_col], fmt)
    else:
        events["arrival"] = pd.NaT
    if settings.trips_to_include:
        keep = {_clean_id(trip) for trip in settings.trips_to_include}
        events = events[events["trip"].isin(keep)]
        if events.empty:
            raise PilotError(f"No stop-event rows belong to TRIPS_TO_INCLUDE {sorted(keep)}.")
    return events.reset_index(drop=True)


def classify_door_events(events: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], int]:
    """Keep rows with usable door-open and door-close times.

    Rows with both door times blank are stops passed without opening; they are
    counted but are not door events. Rows with only one time, unparseable times,
    a close before the open, or an 1899 placeholder date go to reconciliation.

    Args:
        events: Output of :func:`load_stop_events`.

    Returns:
        (door events, reconciliation rows, number of passed stops).
    """
    blank = {"", "nan", "nat", "none"}
    open_blank = events["door_open_raw"].str.lower().isin(blank)
    close_blank = events["door_close_raw"].str.lower().isin(blank)
    passed = open_blank & close_blank
    both = events["door_open"].notna() & events["door_close"].notna()
    ordered = both & (events["door_close"] >= events["door_open"])
    real_date = ordered & (events["door_open"].dt.year >= 1950)
    recon: List[Dict[str, Any]] = []
    for idx in events.index[~passed & ~real_date]:
        row = events.loc[idx]
        if not both[idx]:
            reason = "door-open or door-close time missing or unparseable"
        elif not ordered[idx]:
            reason = "door closes before it opens"
        else:
            reason = "placeholder 1899 date: is this a scheduled-time column?"
        recon.append(
            _recon(
                "bad_door_times",
                f"{reason} (open='{row['door_open_raw']}', close='{row['door_close_raw']}')",
                trip=row["trip"],
                bus=row["bus"],
                stop_code=row["stop_code"],
                stop_sequence=row["stop_sequence"],
            )
        )
    return events[real_date].reset_index(drop=True), recon, int(passed.sum())


def scan_videos(video_dir: Path, settings: Settings) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """List the clips in VIDEO_DIR with bus, camera, door, start time and length.

    Args:
        video_dir: Folder of exported clips.
        settings: Run settings (file-name regex, camera codes, clock offset).

    Returns:
        (one row per readable clip with VIDEO_COLUMNS, reconciliation rows for
        names that don't parse, unknown cameras and unreadable files).
    """
    cv2 = _import_cv2()
    pattern = re.compile(settings.filename_regex)
    extensions = {ext.lower() for ext in settings.video_extensions}
    rows: List[Dict[str, Any]] = []
    recon: List[Dict[str, Any]] = []
    offset = _dt.timedelta(seconds=settings.clock_offset_seconds)
    files = sorted(p for p in video_dir.iterdir() if p.is_file())
    skipped = [p for p in files if p.suffix.lower() not in extensions]
    if skipped:
        logging.info("Skipping %d file(s) in VIDEO_DIR without a video extension.", len(skipped))
    for path in (p for p in files if p.suffix.lower() in extensions):
        parsed = parse_video_name(path.name, pattern, settings.filename_time_format)
        if parsed is None:
            recon.append(
                _recon(
                    "video_name_unparsed", "name doesn't match FILENAME_REGEX", video_file=path.name
                )
            )
            continue
        camera = parsed["camera"]
        if camera == settings.outside_camera.lower():
            kind, door = "outside", ""
        elif camera in settings.camera_doors:
            kind, door = "door", settings.camera_doors[camera]
        else:
            recon.append(
                _recon(
                    "video_unknown_camera",
                    f"camera code '{camera}' is not in CAMERA_DOORS or OUTSIDE_CAMERA",
                    bus=parsed["bus"],
                    video_file=path.name,
                )
            )
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if capture.isOpened() else 0.0
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if capture.isOpened() else 0
        finally:
            capture.release()
        if fps <= 0 or frames <= 0:
            recon.append(
                _recon(
                    "video_unreadable",
                    "OpenCV could not read the frame rate or frame count",
                    bus=parsed["bus"],
                    video_file=path.name,
                )
            )
            continue
        rows.append(
            {
                "path": path,
                "file": path.name,
                "bus": parsed["bus"],
                "camera": camera,
                "kind": kind,
                "door": door,
                "start_recorder": parsed["start"],
                "start_avl": pd.Timestamp(parsed["start"] + offset),
                "fps": fps,
                "frames": frames,
                "last_pos": (frames - 1) / fps,
            }
        )
    return pd.DataFrame(rows, columns=VIDEO_COLUMNS), recon


def match_door_events(
    door_events: pd.DataFrame, clips: pd.DataFrame, settings: Settings
) -> Tuple[List[ClipJob], List[Dict[str, Any]]]:
    """Pair each door-open event with the door-camera clip that covers it, per door.

    A clip covers an event when it holds the whole door-open interval; padding is
    cut at the clip's edges. Partial coverage is reported, not sent (the simpler
    option). A trip with no coverage at all gets one reconciliation row instead of
    one per event.

    Args:
        door_events: Output of :func:`classify_door_events`.
        clips: Output of :func:`scan_videos`.
        settings: Run settings.

    Returns:
        (clip jobs, reconciliation rows including clips that matched no event).
    """
    door_clips = clips[clips["kind"] == "door"]
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for clip in door_clips.sort_values("start_avl").to_dict("records"):
        by_key.setdefault((clip["bus"], clip["door"]), []).append(clip)
    doors = _door_names(settings)
    jobs: List[ClipJob] = []
    recon: List[Dict[str, Any]] = []
    used: set = set()
    trip_has_video: Dict[str, bool] = {}
    no_video: Dict[str, List[Dict[str, Any]]] = {}
    for event in door_events.to_dict("records"):
        ids = {
            "trip": event["trip"],
            "bus": event["bus"],
            "stop_code": event["stop_code"],
            "stop_sequence": event["stop_sequence"],
        }
        for door in doors:
            partial_clip = ""
            chosen = None
            for clip in by_key.get((event["bus"], door), []):
                coverage, start_pos, end_pos = door_window_positions(
                    event["door_open"],
                    event["door_close"],
                    clip["start_avl"],
                    clip["last_pos"],
                    settings.pad_seconds,
                )
                if coverage == "full":
                    chosen = (clip, start_pos, end_pos)
                    break
                if coverage == "partial":
                    partial_clip = clip["file"]
            if chosen is not None:
                clip, start_pos, end_pos = chosen
                used.add(clip["file"])
                trip_has_video[event["trip"]] = True
                jobs.append(
                    ClipJob(
                        trip=event["trip"],
                        bus=event["bus"],
                        stop_code=event["stop_code"],
                        stop_sequence=event["stop_sequence"],
                        door=door,
                        video_path=clip["path"],
                        video_fps=clip["fps"],
                        video_frames=clip["frames"],
                        clip_start_avl=clip["start_avl"],
                        start_pos=start_pos,
                        end_pos=end_pos,
                        door_open=event["door_open"],
                        door_close=event["door_close"],
                    )
                )
            elif partial_clip:
                trip_has_video[event["trip"]] = True
                recon.append(
                    _recon(
                        "event_partial_video",
                        "the clip holds only part of the door-open interval; not sent",
                        door=door,
                        video_file=partial_clip,
                        **ids,
                    )
                )
            else:
                no_video.setdefault(event["trip"], []).append(
                    _recon(
                        "event_no_video",
                        "no clip for this bus and door covers it",
                        door=door,
                        **ids,
                    )
                )
    for trip, rows in no_video.items():
        if trip_has_video.get(trip):
            recon.extend(rows)
        else:
            n_events = len({(r["stop_code"], r["stop_sequence"]) for r in rows})
            recon.append(
                _recon(
                    "trip_no_video",
                    f"no clip covers any of this trip's {n_events} door-open event(s)",
                    trip=trip,
                    bus=rows[0]["bus"],
                )
            )
    for clip in door_clips.to_dict("records"):
        if clip["file"] not in used:
            recon.append(
                _recon(
                    "video_no_events",
                    "no door-open event for this bus falls inside the clip",
                    bus=clip["bus"],
                    door=clip["door"],
                    video_file=clip["file"],
                )
            )
    return jobs, recon


def _parse_count(value: Any) -> Tuple[Optional[int], bool]:
    """Parse one ground-truth count.

    Args:
        value: Cell text.

    Returns:
        (count, valid). Blank gives (None, True); text that isn't a whole number
        of zero or more gives (None, False).
    """
    text = _clean_id(value)
    if text == "":
        return None, True
    try:
        number = float(text)
    except ValueError:
        return None, False
    if number < 0 or not number.is_integer():
        return None, False
    return int(number), True


def _gt_rows_to_dict(
    rows: List[Dict[str, Any]], source: str
) -> Tuple[Dict[Tuple[str, str, str, str], Tuple[int, int]], List[Dict[str, Any]]]:
    """Index long-format ground-truth rows by (trip, stop_code, stop_sequence, door).

    Args:
        rows: Dicts with trip, stop_code, stop_sequence, door, boardings, alightings.
        source: "manual" or "ridecheck", for messages.

    Returns:
        (counts by key, reconciliation rows for invalid and duplicate rows).
    """
    counts: Dict[Tuple[str, str, str, str], Tuple[int, int]] = {}
    recon: List[Dict[str, Any]] = []
    for row in rows:
        key = (row["trip"], row["stop_code"], row["stop_sequence"], row["door"])
        ids = {"trip": key[0], "stop_code": key[1], "stop_sequence": key[2], "door": key[3]}
        boardings, ok_b = _parse_count(row["boardings"])
        alightings, ok_a = _parse_count(row["alightings"])
        if not (ok_b and ok_a):
            recon.append(
                _recon(
                    "ground_truth_invalid",
                    f"{source} count is not a whole number >= 0 "
                    f"(boardings='{row['boardings']}', alightings='{row['alightings']}')",
                    **ids,
                )
            )
            continue
        if boardings is None or alightings is None:
            continue  # not counted yet; the event shows up as event_no_ground_truth
        if key in counts:
            recon.append(
                _recon("ground_truth_duplicate", f"repeated {source} row; first one kept", **ids)
            )
            continue
        counts[key] = (boardings, alightings)
    return counts, recon


def load_manual_counts(
    path: Path,
) -> Tuple[Dict[Tuple[str, str, str, str], Tuple[int, int]], List[Dict[str, Any]]]:
    """Load the filled-in manual-count CSV (the MODE = "template" layout).

    Args:
        path: CSV or Excel file with trip, stop_code, stop_sequence, door,
            boardings and alightings columns.

    Returns:
        (counts by (trip, stop_code, stop_sequence, door), reconciliation rows).

    Raises:
        PilotError: A required column is missing.
    """
    raw = _read_table(path)
    needed = ["trip", "stop_code", "stop_sequence", "door", "boardings", "alightings"]
    missing = [col for col in needed if col not in raw.columns]
    if missing:
        raise PilotError(f"The manual-count file lacks column(s) {missing}: {path}")
    rows = [
        {
            "trip": _clean_id(rec["trip"]),
            "stop_code": _clean_id(rec["stop_code"]),
            "stop_sequence": _clean_id(rec["stop_sequence"]),
            "door": str(rec["door"]).strip().lower(),
            "boardings": rec["boardings"],
            "alightings": rec["alightings"],
        }
        for rec in raw.to_dict("records")
    ]
    return _gt_rows_to_dict(rows, "manual")


def load_ridecheck(
    path: Path, settings: Settings
) -> Tuple[Dict[Tuple[str, str, str, str], Tuple[int, int]], List[Dict[str, Any]]]:
    """Load a Ridecheck export with per-door on/off columns, one row per trip and stop.

    Ridecheck STOP_ID is the public stop code, so it is compared with CLEVER
    stop_code. When RIDECHECK_SEQ_COL is "", the stop sequence in each key is "".

    Args:
        path: CSV or Excel export.
        settings: Run settings (Ridecheck column names).

    Returns:
        (counts by (trip, stop_code, stop_sequence, door), reconciliation rows).

    Raises:
        PilotError: A configured column is missing.
    """
    raw = _read_table(path)
    door_cols = {
        "front": (settings.ridecheck_front_on_col, settings.ridecheck_front_off_col),
        "rear": (settings.ridecheck_rear_on_col, settings.ridecheck_rear_off_col),
    }
    needed = [settings.ridecheck_trip_col, settings.ridecheck_stop_col]
    needed += [col for pair in door_cols.values() for col in pair]
    if settings.ridecheck_seq_col:
        needed.append(settings.ridecheck_seq_col)
    missing = [col for col in needed if col not in raw.columns]
    if missing:
        raise PilotError(
            f"The Ridecheck file lacks column(s) {missing}. Columns found: {list(raw.columns)}. "
            "Fix the RIDECHECK_*_COL settings."
        )
    rows: List[Dict[str, Any]] = []
    for rec in raw.to_dict("records"):
        seq = _clean_id(rec[settings.ridecheck_seq_col]) if settings.ridecheck_seq_col else ""
        for door, (on_col, off_col) in door_cols.items():
            rows.append(
                {
                    "trip": _clean_id(rec[settings.ridecheck_trip_col]),
                    "stop_code": _clean_id(rec[settings.ridecheck_stop_col]),
                    "stop_sequence": seq,
                    "door": door,
                    "boardings": rec[on_col],
                    "alightings": rec[off_col],
                }
            )
    return _gt_rows_to_dict(rows, "ridecheck")


def attach_ground_truth(
    jobs: List[ClipJob],
    counts: Dict[Tuple[str, str, str, str], Tuple[int, int]],
    door_events: pd.DataFrame,
    settings: Settings,
    use_sequence: bool,
) -> Tuple[List[ClipJob], List[Dict[str, Any]]]:
    """Add ground-truth counts to each clip job; report what doesn't match.

    Jobs without ground truth are not sent to the model (the simpler option,
    and it saves money); they are listed in reconciliation instead.

    Args:
        jobs: Clip jobs from :func:`match_door_events`.
        counts: Ground truth by (trip, stop_code, stop_sequence, door).
        door_events: All usable door events, to find ground truth with no event.
        settings: Run settings.
        use_sequence: False when the ground truth has no stop sequence (Ridecheck
            without RIDECHECK_SEQ_COL); keys then use "" for the sequence.

    Returns:
        (jobs with ground truth, reconciliation rows).
    """
    recon: List[Dict[str, Any]] = []
    ambiguous: set = set()
    if not use_sequence:
        repeats = door_events.groupby(["trip", "stop_code"]).size()
        ambiguous = {key for key, size in repeats.items() if size > 1}
    keep: List[ClipJob] = []
    for job in jobs:
        ids = {
            "trip": job.trip,
            "bus": job.bus,
            "stop_code": job.stop_code,
            "stop_sequence": job.stop_sequence,
            "door": job.door,
            "video_file": job.video_path.name,
        }
        if (job.trip, job.stop_code) in ambiguous:
            recon.append(
                _recon(
                    "ground_truth_ambiguous",
                    "the trip visits this stop more than once and the ground truth has "
                    "no stop sequence (set RIDECHECK_SEQ_COL); not sent",
                    **ids,
                )
            )
            continue
        key = (job.trip, job.stop_code, job.stop_sequence if use_sequence else "", job.door)
        if key not in counts:
            recon.append(_recon("event_no_ground_truth", "not sent to the model", **ids))
            continue
        boardings, alightings = counts[key]
        keep.append(job._replace(gt_boardings=boardings, gt_alightings=alightings))
    event_keys = {
        (row["trip"], row["stop_code"], row["stop_sequence"] if use_sequence else "", door)
        for row in door_events.to_dict("records")
        for door in _door_names(settings)
    }
    trips = set(door_events["trip"])
    for key in counts:
        if key not in event_keys and key[0] in trips:
            recon.append(
                _recon(
                    "ground_truth_no_event",
                    "no usable door-open event has this trip, stop, sequence and door",
                    trip=key[0],
                    stop_code=key[1],
                    stop_sequence=key[2],
                    door=key[3],
                )
            )
    return keep, recon


# =============================================================================
# FRAMES
# =============================================================================


def _encode_jpeg(cv2: Any, frame: Any, width: int, height: int, quality: int) -> bytes:
    """Resize a frame and encode it as JPEG.

    Args:
        cv2: The OpenCV module.
        frame: BGR image array.
        width: Output width in pixels.
        height: Output height in pixels.
        quality: JPEG quality, 1-100.

    Returns:
        JPEG bytes.

    Raises:
        PilotError: OpenCV failed to encode the frame.
    """
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise PilotError("OpenCV failed to encode a frame as JPEG.")
    return buffer.tobytes()


def extract_frames(
    cv2: Any,
    path: Path,
    times: Sequence[float],
    video_fps: float,
    video_frames: int,
    width: int,
    height: int,
    quality: int,
) -> List[bytes]:
    """Read the frames nearest to the given times and return them as JPEGs.

    Seeks once to the first frame, then reads forward, which is faster and more
    exact than seeking for every frame. Frame numbers come from time x the file's
    frame rate, which assumes a constant frame rate (the simpler option). If the
    recorder writes a variable frame rate, timing drifts within long clips; the
    calibration stills and the timestamp burned into most recorder footage show it.

    Args:
        cv2: The OpenCV module.
        path: Video file.
        times: Sample times, seconds into the file, ascending.
        video_fps: The file's native frame rate.
        video_frames: Frames in the file.
        width: Output width.
        height: Output height.
        quality: JPEG quality.

    Returns:
        One JPEG per requested time (fewer if the file ends early or can't be read).
    """
    if not times:
        return []
    targets = [min(max(int(round(t * video_fps)), 0), video_frames - 1) for t in times]
    capture = cv2.VideoCapture(str(path))
    jpegs: List[bytes] = []
    try:
        if not capture.isOpened():
            return []
        capture.set(cv2.CAP_PROP_POS_FRAMES, targets[0])
        index = targets[0]
        next_target = 0
        while next_target < len(targets):
            ok, frame = capture.read()
            if not ok:
                break
            while next_target < len(targets) and targets[next_target] <= index:
                jpegs.append(_encode_jpeg(cv2, frame, width, height, quality))
                next_target += 1
            index += 1
    finally:
        capture.release()
    return jpegs


def _local_frames_root(settings: Settings) -> Path:
    """Return the local folder for kept frames and stills, refusing network shares.

    Args:
        settings: Run settings.

    Returns:
        FRAMES_DIR, or a folder under this PC's temp folder when FRAMES_DIR is "".

    Raises:
        PilotError: FRAMES_DIR is on a network share.
    """
    if settings.frames_dir.strip():
        root = Path(settings.frames_dir)
    else:
        root = Path(tempfile.gettempdir()) / "video_count_pilot_frames"
    if _is_network_path(str(root)):
        raise PilotError(
            f"FRAMES_DIR '{root}' is on a network share. Frames show riders and must stay on a "
            "local disk; pick a local folder or leave FRAMES_DIR blank."
        )
    return root


# =============================================================================
# REQUESTS
# =============================================================================


def prompt_hash() -> str:
    """Return the SHA-256 of every prompt text and schema the script sends.

    Returns:
        Hex digest; any change to the prompts changes it.
    """
    payload = {
        "count_system": COUNT_SYSTEM_PROMPT,
        "clip_header": CLIP_HEADER,
        "chunk_note": CHUNK_NOTE,
        "frame_label": FRAME_LABEL,
        "count_question": COUNT_QUESTION,
        "count_schema": COUNT_SCHEMA,
        "stage2_system": STAGE2_SYSTEM_PROMPT,
        "stage2_stops_intro": STAGE2_STOPS_INTRO,
        "stage2_frame_label": STAGE2_FRAME_LABEL,
        "stage2_question": STAGE2_QUESTION,
        "stage2_schema": STAGE2_SCHEMA,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _image_block(jpeg: bytes) -> Dict[str, Any]:
    """Return a base64 image content block for one JPEG.

    Args:
        jpeg: JPEG bytes.

    Returns:
        Messages API image block.
    """
    data = base64.standard_b64encode(jpeg).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def request_kwargs(
    settings: Settings, system_text: str, content: List[Dict[str, Any]], schema: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the keyword arguments for one ``client.messages.create`` call.

    Structured output uses ``output_config.format`` with a JSON schema, the method
    the current docs recommend (forced tool use returns a 400 on Opus 5.5 and
    Sonnet 5.5). The fixed instructions are marked for prompt caching when they
    are long enough to qualify.

    The brief asked for temperature 0, but Claude 4.7 and later models (all three
    5.5 models) accept only the default sampling settings and return a 400 for any
    other value, and API version 2026-10-21 (anthropic 2.x) rejects the field
    outright. So no temperature is sent, and REPEATS measures the variation.

    Args:
        settings: Run settings.
        system_text: Fixed instructions.
        content: User message content blocks.
        schema: JSON schema for the answer.

    Returns:
        Keyword arguments for the Messages API.
    """
    system: Any = system_text
    if _text_tokens(system_text, settings.text_chars_per_token) >= settings.cache_min_tokens:
        system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
    output_config: Dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
    if settings.effort:
        output_config["effort"] = settings.effort
    return {
        "model": settings.model_id,
        "max_tokens": settings.max_output_tokens,
        "system": system,
        "messages": [{"role": "user", "content": content}],
        "output_config": output_config,
    }


def _response_text(response: Any) -> Optional[str]:
    """Return the first text block of a response, or None.

    Args:
        response: Messages API response.

    Returns:
        The text, or None if there is no text block.
    """
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", "") == "text":
            return str(getattr(block, "text", ""))
    return None


def _response_usage(response: Any) -> Dict[str, int]:
    """Return a response's token counts.

    Args:
        response: Messages API response.

    Returns:
        Dict with input, output, cache_write and cache_read token counts.
    """
    usage = getattr(response, "usage", None)

    def _value(name: str) -> int:
        return int(getattr(usage, name, 0) or 0) if usage is not None else 0

    return {
        "input": _value("input_tokens"),
        "output": _value("output_tokens"),
        "cache_write": _value("cache_creation_input_tokens"),
        "cache_read": _value("cache_read_input_tokens"),
    }


def _is_fatal_api_error(exc: BaseException) -> bool:
    """Return True for errors every later request would hit too.

    Bad credentials, missing permission, an unknown model and malformed requests
    (HTTP 400) stop the run at once; the request sizes are already controlled,
    so a 400 almost always means a setting is wrong.

    Args:
        exc: Exception raised by the client.

    Returns:
        True if the run should stop.
    """
    try:
        import anthropic
    except ImportError:
        return False
    fatal = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.BadRequestError,
    )
    return isinstance(exc, fatal)


def call_model(
    client: Any,
    kwargs: Dict[str, Any],
    parse: Callable[[Any], Tuple[Optional[Dict[str, Any]], str]],
    settings: Settings,
    control: RunControl,
) -> CallResult:
    """Send one request, re-sending it while the answer can't be parsed.

    API errors are retried by the SDK itself (MAX_API_RETRIES, exponential
    backoff); an error that survives those retries is returned as "api_error".

    Args:
        client: Anthropic client (or a test stub with the same interface).
        kwargs: Request keyword arguments.
        parse: Function that turns a response into (data, problem).
        settings: Run settings.
        control: Run controller, told about API errors.

    Returns:
        The call's outcome, with token use and cost summed over every attempt.

    Raises:
        FatalApiError: The error would repeat for every request.
    """
    prices = settings.model_prices[settings.model_id]
    usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    cost = 0.0
    calls = 0
    problem = ""
    for attempt in range(1 + settings.max_parse_retries):
        calls += 1
        try:
            response = client.messages.create(**kwargs)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if _is_fatal_api_error(exc):
                raise FatalApiError(message) from exc
            control.record(api_error=True, message=message)
            return CallResult("api_error", None, message, usage, cost, calls)
        control.record(api_error=False)
        this_usage = _response_usage(response)
        cost += usage_cost(this_usage, prices, settings.cache_write_multiplier)
        for key, value in this_usage.items():
            usage[key] += value
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            return CallResult("refused", None, f"refusal ({category})", usage, cost, calls)
        data, problem = parse(response)
        if data is not None:
            return CallResult("ok", data, "", usage, cost, calls)
        logging.warning(
            "Unparseable response (attempt %d of %d): %s",
            attempt + 1,
            1 + settings.max_parse_retries,
            problem,
        )
    return CallResult("unparseable", None, problem, usage, cost, calls)


def parse_count_response(response: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """Check a Stage 1 answer and return its fields.

    Args:
        response: Messages API response.

    Returns:
        (data, "") on success, or (None, problem).
    """
    text = _response_text(response)
    if text is None:
        return None, f"no text block (stop_reason={getattr(response, 'stop_reason', None)})"
    try:
        data = json.loads(text)
    except ValueError as exc:
        return None, f"not valid JSON ({exc}): {text[:200]!r}"
    if not isinstance(data, dict):
        return None, "JSON is not an object"
    counts: Dict[str, int] = {}
    for key in ("boardings", "alightings"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"{key} is not a number: {value!r}"
        if value < 0 or not float(value).is_integer():
            return None, f"{key} is not a whole number >= 0: {value!r}"
        counts[key] = int(value)
    confidence = str(data.get("confidence", "")).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        return None, f"confidence is not low/medium/high: {data.get('confidence')!r}"
    notes = str(data.get("notes", "")).strip()
    return {**counts, "confidence": confidence, "notes": notes}, ""


def parse_stage2_response(response: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """Check a Stage 2 answer and return its detections.

    Args:
        response: Messages API response.

    Returns:
        ({"stops": [...], "notes": str}, "") on success, or (None, problem).
    """
    text = _response_text(response)
    if text is None:
        return None, f"no text block (stop_reason={getattr(response, 'stop_reason', None)})"
    try:
        data = json.loads(text)
    except ValueError as exc:
        return None, f"not valid JSON ({exc}): {text[:200]!r}"
    if not isinstance(data, dict) or not isinstance(data.get("stops"), list):
        return None, "JSON has no 'stops' list"
    stops: List[Dict[str, Any]] = []
    for item in data["stops"]:
        if not isinstance(item, dict):
            return None, f"stop entry is not an object: {item!r}"
        time_value = item.get("frame_time_s")
        if isinstance(time_value, bool) or not isinstance(time_value, (int, float)):
            return None, f"frame_time_s is not a number: {item!r}"
        action = str(item.get("action", "")).strip().lower()
        if action not in ("stopped", "passed"):
            return None, f"action is not stopped/passed: {item!r}"
        stops.append(
            {
                "stop_code": _clean_id(item.get("stop_code")),
                "frame_time_s": float(time_value),
                "action": action,
            }
        )
    return {"stops": stops, "notes": str(data.get("notes", "")).strip()}, ""


# =============================================================================
# STAGE 1: PLANNING AND PROCESSING
# =============================================================================


def _clip_header(job: ClipJob) -> str:
    """Return the per-clip text placed before the frames.

    Args:
        job: Clip job.

    Returns:
        Header text.
    """
    return CLIP_HEADER.format(
        trip=job.trip, stop_code=job.stop_code, seq=job.stop_sequence, door=job.door
    )


def plan_count_job(job: ClipJob, settings: Settings) -> PlannedJob:
    """Work out a clip's frame times, request chunks and token estimates.

    Args:
        job: Clip job.
        settings: Run settings.

    Returns:
        The planned job.
    """
    cpt = settings.text_chars_per_token
    times = frame_times(job.start_pos, job.end_pos, settings.frame_rate)
    per_frame = image_tokens(settings.frame_width, settings.frame_height) + _text_tokens(
        FRAME_LABEL.format(t=999.9), cpt
    )
    fixed_text = COUNT_SYSTEM_PROMPT + _clip_header(job) + COUNT_QUESTION + json.dumps(COUNT_SCHEMA)
    fixed = _text_tokens(fixed_text, cpt) + settings.request_overhead_tokens
    ceiling = settings.request_token_ceiling
    max_images = settings.max_images_per_request
    chunks = plan_chunks(len(times), per_frame, fixed, ceiling, max_images)
    if len(chunks) > 1:
        fixed += _text_tokens(CHUNK_NOTE.format(part=99, parts=99), cpt)
        chunks = plan_chunks(len(times), per_frame, fixed, ceiling, max_images)
    tokens = [fixed + (end - start) * per_frame for start, end in chunks]
    return PlannedJob(job=job, frame_times=times, chunks=chunks, chunk_tokens=tokens)


def build_count_requests(
    planned: PlannedJob, frames: List[bytes], settings: Settings
) -> List[Dict[str, Any]]:
    """Build one request per chunk: instructions, labelled frames, then the question.

    Args:
        planned: Planned job.
        frames: JPEGs, one per frame time.
        settings: Run settings.

    Returns:
        Request keyword arguments, one per chunk.
    """
    job = planned.job
    requests: List[Dict[str, Any]] = []
    n_chunks = len(planned.chunks)
    for index, (start, end) in enumerate(planned.chunks):
        content: List[Dict[str, Any]] = [{"type": "text", "text": _clip_header(job)}]
        if n_chunks > 1:
            content.append(
                {"type": "text", "text": CHUNK_NOTE.format(part=index + 1, parts=n_chunks)}
            )
        for t, jpeg in zip(planned.frame_times[start:end], frames[start:end]):
            content.append({"type": "text", "text": FRAME_LABEL.format(t=t - job.start_pos)})
            content.append(_image_block(jpeg))
        content.append({"type": "text", "text": COUNT_QUESTION})
        requests.append(request_kwargs(settings, COUNT_SYSTEM_PROMPT, content, COUNT_SCHEMA))
    return requests


def _est_job_cost(planned: PlannedJob, settings: Settings) -> float:
    """Estimate the cost of one repeat of a planned job (all its chunks).

    Args:
        planned: Planned job.
        settings: Run settings.

    Returns:
        Cost in USD.
    """
    prices = settings.model_prices[settings.model_id]
    out = settings.est_output_tokens_per_request
    return sum(estimate_request_cost(tokens, out, prices) for tokens in planned.chunk_tokens)


def _event_row(
    planned: PlannedJob, settings: Settings, repeat: int, status: str, **values: Any
) -> Dict[str, Any]:
    """Build one Events-sheet row.

    Args:
        planned: Planned job.
        settings: Run settings.
        repeat: Repeat number, from 1.
        status: ok, api_error, unparseable, refused, dry_run or not_sent.
        **values: Columns to set (counts, tokens, cost, notes, ...).

    Returns:
        A dict with every EVENT_COLUMNS key.
    """
    job = planned.job
    row: Dict[str, Any] = {col: None for col in EVENT_COLUMNS}
    row.update(
        {
            "trip": job.trip,
            "bus": job.bus,
            "stop_code": job.stop_code,
            "stop_sequence": job.stop_sequence,
            "door": job.door,
            "repeat": repeat,
            "video_file": job.video_path.name,
            "clip_start_s": round(job.start_pos, 2),
            "clip_end_s": round(job.end_pos, 2),
            "window_start": job.clip_start_avl + _dt.timedelta(seconds=job.start_pos),
            "window_end": job.clip_start_avl + _dt.timedelta(seconds=job.end_pos),
            "frames_sent": len(planned.frame_times),
            "gt_boardings": job.gt_boardings,
            "gt_alightings": job.gt_alightings,
            "confidence": "",
            "notes": "",
            "split_flag": "Y" if len(planned.chunks) > 1 else "N",
            "chunks": len(planned.chunks),
            "status": status,
            "detail": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "est_input_tokens": sum(planned.chunk_tokens),
            "est_cost_usd": _est_job_cost(planned, settings),
            "api_calls": 0,
            "model_id": settings.model_id,
            "prompt_hash": prompt_hash()[:12],
        }
    )
    row.update(values)
    for metric in ("boardings", "alightings"):
        model_value, gt_value = row[f"model_{metric}"], row[f"gt_{metric}"]
        if model_value is not None and gt_value is not None:
            row[f"{metric}_error"] = model_value - gt_value
            row[f"{metric}_abs_error"] = abs(model_value - gt_value)
    return row


def process_count_job(
    planned: PlannedJob,
    settings: Settings,
    client: Any,
    frames_dir: Optional[Path],
    control: RunControl,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract one clip's frames, send them REPEATS times and record every answer.

    A split clip's chunk counts are added up; its confidence is the lowest of the
    chunks and its notes are joined. If any chunk of a repeat fails, that repeat
    has no counts.

    Args:
        planned: Planned job.
        settings: Run settings.
        client: Anthropic client, or None in a dry run.
        frames_dir: Folder for KEEP_FRAMES copies, or None.
        control: Run controller.

    Returns:
        (Events rows, reconciliation rows).
    """
    job = planned.job
    ids = {
        "trip": job.trip,
        "bus": job.bus,
        "stop_code": job.stop_code,
        "stop_sequence": job.stop_sequence,
        "door": job.door,
        "video_file": job.video_path.name,
    }
    if control.abort.is_set():
        rows = [
            _event_row(planned, settings, r, "not_sent", detail="run stopped")
            for r in range(1, settings.repeats + 1)
        ]
        return rows, [_recon("not_sent", f"run stopped: {control.fatal_message}", **ids)]
    cv2 = _import_cv2()
    frames = extract_frames(
        cv2,
        job.video_path,
        planned.frame_times,
        job.video_fps,
        job.video_frames,
        settings.frame_width,
        settings.frame_height,
        settings.jpeg_quality,
    )
    if len(frames) != len(planned.frame_times):
        detail = f"read {len(frames)} of {len(planned.frame_times)} frames"
        return [], [_recon("frame_extraction_failed", detail, **ids)]
    # Frames live only in memory and are dropped after the requests are built, so
    # there is nothing to delete afterwards; KEEP_FRAMES writes copies to a local disk.
    if frames_dir is not None:
        stem = _safe_name(job.trip, job.stop_sequence, job.stop_code, job.door)
        for number, jpeg in enumerate(frames, 1):
            (frames_dir / f"{stem}_f{number:03d}.jpg").write_bytes(jpeg)
    requests = build_count_requests(planned, frames, settings)
    del frames
    rows: List[Dict[str, Any]] = []
    recon: List[Dict[str, Any]] = []
    for repeat in range(1, settings.repeats + 1):
        if settings.dry_run:
            rows.append(_event_row(planned, settings, repeat, "dry_run"))
            continue
        if control.abort.is_set():
            rows.append(_event_row(planned, settings, repeat, "not_sent", detail="run stopped"))
            continue
        totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
        cost, calls = 0.0, 0
        boardings, alightings = 0, 0
        confidences: List[str] = []
        notes: List[str] = []
        status, detail = "ok", ""
        for index, kwargs in enumerate(requests):
            try:
                result = call_model(client, kwargs, parse_count_response, settings, control)
            except FatalApiError as exc:
                control.stop(str(exc))
                status, detail = "api_error", f"run stopped: {exc}"
                break
            for key, value in result.usage.items():
                totals[key] += value
            cost += result.cost_usd
            calls += result.api_calls
            if result.status != "ok" or result.data is None:
                status, detail = result.status, result.detail
                break
            boardings += int(result.data["boardings"])
            alightings += int(result.data["alightings"])
            confidences.append(str(result.data["confidence"]))
            prefix = f"[part {index + 1}/{len(requests)}] " if len(requests) > 1 else ""
            notes.append(prefix + str(result.data["notes"]))
        values: Dict[str, Any] = {
            "input_tokens": totals["input"],
            "output_tokens": totals["output"],
            "cache_write_tokens": totals["cache_write"],
            "cache_read_tokens": totals["cache_read"],
            "cost_usd": cost,
            "api_calls": calls,
            "detail": detail,
        }
        if status == "ok":
            values.update(
                {
                    "model_boardings": boardings,
                    "model_alightings": alightings,
                    "confidence": min(confidences, key=CONFIDENCE_LEVELS.index),
                    "notes": " | ".join(notes),
                }
            )
        else:
            category = {
                "api_error": "api_failure",
                "unparseable": "unparseable_response",
                "refused": "refused",
            }.get(status, status)
            recon.append(_recon(category, detail, repeat=repeat, **ids))
        rows.append(_event_row(planned, settings, repeat, status, **values))
    return rows, recon


def run_workers(
    items: Sequence[Any],
    worker: Callable[[Any], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    max_workers: int,
    describe: Callable[[Any], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run a worker over items in a thread pool, keeping results in input order.

    An unexpected error in one item (for example a corrupt video) is logged and
    recorded in reconciliation, so it doesn't lose the rest of the run.

    Args:
        items: Planned jobs or windows.
        worker: Function returning (rows, reconciliation rows) for one item.
        max_workers: Threads.
        describe: Function returning reconciliation ID fields for an item.

    Returns:
        (all rows, all reconciliation rows).
    """
    rows: List[Dict[str, Any]] = []
    recon: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = [pool.submit(worker, item) for item in items]
        for number, (item, future) in enumerate(zip(items, futures), 1):
            try:
                item_rows, item_recon = future.result()
            except Exception as exc:
                logging.exception("Processing failed for %s", describe(item))
                recon.append(
                    _recon("processing_error", f"{type(exc).__name__}: {exc}", **describe(item))
                )
                continue
            rows.extend(item_rows)
            recon.extend(item_recon)
            if number % 25 == 0 or number == len(futures):
                logging.info("Processed %d of %d.", number, len(futures))
    return rows, recon


# =============================================================================
# STAGE 1: SUMMARY SHEETS
# =============================================================================


def _crowding_band(value: Any) -> str:
    """Return the crowding band of a ground-truth count: 0, 1-2, 3-5 or 6+.

    Args:
        value: Ground-truth count.

    Returns:
        Band label, or "" if missing.
    """
    if value is None or pd.isna(value):
        return ""
    count = float(value)
    if count == 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    return "6+"


def _accuracy_row(
    metric: str, group: str, value: str, model: pd.Series, truth: pd.Series
) -> Dict[str, Any]:
    """Return accuracy statistics for one metric and one group.

    Args:
        metric: "boardings" or "alightings".
        group: Breakdown name.
        value: Group value.
        model: Model counts.
        truth: Ground-truth counts, aligned with *model*.

    Returns:
        One Accuracy-sheet row.
    """
    row: Dict[str, Any] = {"metric": metric, "breakdown": group, "group": value, "n": len(model)}
    if len(model) == 0:
        return row
    error = model - truth
    total_truth = float(truth.sum())
    row.update(
        {
            "exact_share": float((error == 0).mean()),
            "within_1_share": float((error.abs() <= 1).mean()),
            "mean_abs_error": float(error.abs().mean()),
            "mean_signed_error": float(error.mean()),
            "model_total": float(model.sum()),
            "gt_total": total_truth,
            "model_to_gt_ratio": float(model.sum()) / total_truth if total_truth > 0 else None,
        }
    )
    return row


def accuracy_table(events: pd.DataFrame, doors: Sequence[str]) -> pd.DataFrame:
    """Build the Accuracy sheet: every successful repeat counts as one observation.

    Args:
        events: Events sheet.
        doors: Door names.

    Returns:
        Rows by metric and breakdown (overall, door, confidence, crowding band).
    """
    columns = [
        "metric",
        "breakdown",
        "group",
        "n",
        "exact_share",
        "within_1_share",
        "mean_abs_error",
        "mean_signed_error",
        "model_total",
        "gt_total",
        "model_to_gt_ratio",
    ]
    scored = events[(events["status"] == "ok") & events["gt_boardings"].notna()]
    rows: List[Dict[str, Any]] = []
    for metric in ("boardings", "alightings"):
        model = pd.to_numeric(scored[f"model_{metric}"], errors="coerce")
        truth = pd.to_numeric(scored[f"gt_{metric}"], errors="coerce")
        bands = truth.map(_crowding_band)
        groups: List[Tuple[str, str, pd.Series]] = [
            ("overall", "all", pd.Series(True, index=scored.index))
        ]
        groups += [("door", door, scored["door"] == door) for door in doors]
        groups += [("confidence", c, scored["confidence"] == c) for c in CONFIDENCE_LEVELS]
        groups += [("crowding (ground truth)", b, bands == b) for b in CROWDING_BANDS]
        for group, value, mask in groups:
            rows.append(_accuracy_row(metric, group, value, model[mask], truth[mask]))
    return pd.DataFrame(rows, columns=columns)


def _per_event_means(events: pd.DataFrame) -> pd.DataFrame:
    """Average the successful repeats of each event and door.

    Args:
        events: Events sheet.

    Returns:
        One row per trip, stop and door with model and ground-truth counts.
    """
    keys = ["trip", "stop_code", "stop_sequence", "door"]
    ok = events[events["status"] == "ok"].copy()
    for col in ("model_boardings", "model_alightings", "gt_boardings", "gt_alightings"):
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    if ok.empty:
        return pd.DataFrame(columns=keys + ["model_b", "model_a", "gt_b", "gt_a"])
    return ok.groupby(keys, as_index=False).agg(
        model_b=("model_boardings", "mean"),
        model_a=("model_alightings", "mean"),
        gt_b=("gt_boardings", "first"),
        gt_a=("gt_alightings", "first"),
    )


def trip_summary_table(events: pd.DataFrame) -> pd.DataFrame:
    """Build the Trip summary sheet.

    Model values are the mean of each event's successful repeats. Per trip and
    door: totals and their ratio. Per trip ("all doors" rows): the running load
    (cumulative ons minus offs) along the trip, model versus ground truth, over
    the events and doors that have both, and the largest gap between them.

    Args:
        events: Events sheet.

    Returns:
        The Trip summary table.
    """
    columns = [
        "trip",
        "door",
        "events_compared",
        "events_without_model",
        "model_boardings",
        "gt_boardings",
        "boardings_ratio",
        "model_alightings",
        "gt_alightings",
        "alightings_ratio",
        "max_load_diff",
        "max_load_diff_stop_code",
        "max_load_diff_stop_sequence",
        "gt_load_at_max_diff",
    ]
    per = _per_event_means(events)
    keys = ["trip", "stop_code", "stop_sequence", "door"]
    all_events = events.drop_duplicates(keys)[keys]
    rows: List[Dict[str, Any]] = []

    def _ratio(model_total: float, truth_total: float) -> Optional[float]:
        return model_total / truth_total if truth_total > 0 else None

    for trip in sorted(all_events["trip"].unique()):
        trip_per = per[per["trip"] == trip]
        trip_all = all_events[all_events["trip"] == trip]
        for door in list(dict.fromkeys(trip_all["door"])) + ["all doors"]:
            sub = trip_per if door == "all doors" else trip_per[trip_per["door"] == door]
            n_all = len(trip_all) if door == "all doors" else int((trip_all["door"] == door).sum())
            row: Dict[str, Any] = {
                "trip": trip,
                "door": door,
                "events_compared": len(sub),
                "events_without_model": n_all - len(sub),
                "model_boardings": float(sub["model_b"].sum()),
                "gt_boardings": float(sub["gt_b"].sum()),
                "boardings_ratio": _ratio(float(sub["model_b"].sum()), float(sub["gt_b"].sum())),
                "model_alightings": float(sub["model_a"].sum()),
                "gt_alightings": float(sub["gt_a"].sum()),
                "alightings_ratio": _ratio(float(sub["model_a"].sum()), float(sub["gt_a"].sum())),
            }
            if door == "all doors" and not sub.empty:
                stops = sub.groupby(["stop_sequence", "stop_code"], as_index=False)[
                    ["model_b", "model_a", "gt_b", "gt_a"]
                ].sum()
                stops = stops.assign(seq_num=_seq_number(stops["stop_sequence"])).sort_values(
                    ["seq_num", "stop_sequence"]
                )
                model_load = (stops["model_b"] - stops["model_a"]).cumsum()
                truth_load = (stops["gt_b"] - stops["gt_a"]).cumsum()
                gap = model_load - truth_load
                # Rounded so float noise in averaged repeats can't pick a later tie.
                worst = gap.round(6).abs().idxmax()
                row.update(
                    {
                        "max_load_diff": float(gap[worst]),
                        "max_load_diff_stop_code": stops.loc[worst, "stop_code"],
                        "max_load_diff_stop_sequence": stops.loc[worst, "stop_sequence"],
                        "gt_load_at_max_diff": float(truth_load[worst]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def repeatability_table(events: pd.DataFrame, repeats: int) -> Tuple[pd.DataFrame, List[str]]:
    """Build the Repeatability sheet: spread of each event's answers across repeats.

    Args:
        events: Events sheet.
        repeats: REPEATS setting.

    Returns:
        (per-event table, summary lines shown above it).
    """
    columns = [
        "trip",
        "stop_code",
        "stop_sequence",
        "door",
        "repeats_requested",
        "repeats_ok",
        "boardings_min",
        "boardings_max",
        "boardings_range",
        "alightings_min",
        "alightings_max",
        "alightings_range",
        "all_repeats_agree",
    ]
    rows: List[Dict[str, Any]] = []
    for (trip, stop, seq, door), group in events.groupby(
        ["trip", "stop_code", "stop_sequence", "door"], sort=False
    ):
        ok = group[group["status"] == "ok"]
        b = pd.to_numeric(ok["model_boardings"], errors="coerce")
        a = pd.to_numeric(ok["model_alightings"], errors="coerce")
        row: Dict[str, Any] = {
            "trip": trip,
            "stop_code": stop,
            "stop_sequence": seq,
            "door": door,
            "repeats_requested": len(group),
            "repeats_ok": len(ok),
        }
        if len(ok):
            row.update(
                {
                    "boardings_min": b.min(),
                    "boardings_max": b.max(),
                    "boardings_range": b.max() - b.min(),
                    "alightings_min": a.min(),
                    "alightings_max": a.max(),
                    "alightings_range": a.max() - a.min(),
                }
            )
        if len(ok) >= 2:
            agree = row["boardings_range"] == 0 and row["alightings_range"] == 0
            row["all_repeats_agree"] = "Y" if agree else "N"
        rows.append(row)
    table = pd.DataFrame(rows, columns=columns)
    compared = table[table["all_repeats_agree"].isin(["Y", "N"])]
    if compared.empty:
        share_text = "n/a (needs 2 or more successful repeats per event)"
    else:
        share = float((compared["all_repeats_agree"] == "Y").mean())
        share_text = f"{share:.1%} of {len(compared)} events"
    lines = [
        f"Repeats per request (REPEATS): {repeats}",
        f"Events where all successful repeats agreed: {share_text}",
    ]
    return table, lines


def count_cost_table(
    events: pd.DataFrame, planned: Sequence[PlannedJob], settings: Settings
) -> pd.DataFrame:
    """Build the Cost sheet for a Stage 1 run.

    Args:
        events: Events sheet.
        planned: Planned jobs (for footage hours).
        settings: Run settings.

    Returns:
        A metric / value / note table.
    """
    sent = events[events["status"].isin(["ok", "api_error", "unparseable", "refused"])]
    actual = float(pd.to_numeric(sent["cost_usd"], errors="coerce").fillna(0).sum())
    estimate = float(pd.to_numeric(events["est_cost_usd"], errors="coerce").fillna(0).sum())
    cost = estimate if settings.dry_run else actual
    basis = "estimated (dry run)" if settings.dry_run else "actual"
    footage_hours = sum(p.job.end_pos - p.job.start_pos for p in planned) / 3600.0
    stop_events = len({(p.job.trip, p.job.stop_code, p.job.stop_sequence) for p in planned})
    tokens_in = int(sent["input_tokens"].sum() + sent["cache_write_tokens"].sum())
    tokens_in += int(sent["cache_read_tokens"].sum())
    tokens_out = int(sent["output_tokens"].sum())
    if settings.dry_run:
        tokens_in = int(events["est_input_tokens"].sum())
        tokens_out = settings.est_output_tokens_per_request * int(
            sum(len(p.chunks) for p in planned) * settings.repeats
        )
    calls = int(pd.to_numeric(sent["api_calls"], errors="coerce").fillna(0).sum())
    single_pass = cost / settings.repeats if settings.repeats else cost
    per_footage_hour = single_pass / footage_hours if footage_hours else None
    per_revenue_hour = (
        per_footage_hour * settings.door_open_share * settings.door_cameras_per_bus
        if per_footage_hour is not None
        else None
    )

    def _per(value: float, count: float) -> Optional[float]:
        return value / count if count else None

    rows = [
        ("model_id", settings.model_id, ""),
        ("prompt_hash", prompt_hash(), "SHA-256 of all prompt texts and schemas"),
        ("cost_basis", basis, ""),
        ("estimated_cost_usd", estimate, "before the run, no cache discount"),
        ("actual_cost_usd", None if settings.dry_run else actual, "from response usage"),
        ("api_calls", calls, "includes parse retries"),
        ("stop_events", stop_events, "trip/stop/sequence combinations sent"),
        ("clips", len(planned), "stop events x door cameras"),
        ("repeats", settings.repeats, ""),
        ("input_tokens", tokens_in, "uncached + cache write + cache read"),
        ("uncached_input_tokens", int(sent["input_tokens"].sum()), ""),
        ("cache_write_tokens", int(sent["cache_write_tokens"].sum()), ""),
        ("cache_read_tokens", int(sent["cache_read_tokens"].sum()), ""),
        ("output_tokens", tokens_out, "thinking included"),
        ("input_tokens_per_stop_event", _per(tokens_in, stop_events), "all repeats"),
        ("output_tokens_per_stop_event", _per(tokens_out, stop_events), "all repeats"),
        ("cost_per_stop_event_usd", _per(cost, stop_events), "all repeats"),
        ("cost_per_stop_event_single_pass_usd", _per(single_pass, stop_events), "1 repeat"),
        ("footage_hours", footage_hours, "padded windows, all door cameras, one pass"),
        ("input_tokens_per_footage_hour", _per(tokens_in / settings.repeats, footage_hours), ""),
        ("cost_per_footage_hour_usd", per_footage_hour, "one pass"),
        ("door_open_share", settings.door_open_share, "DOOR_OPEN_SHARE"),
        ("door_cameras_per_bus", settings.door_cameras_per_bus, "DOOR_CAMERAS_PER_BUS"),
        (
            "cost_per_revenue_hour_usd",
            per_revenue_hour,
            "cost per footage hour x door-open share x door cameras",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


# =============================================================================
# OUTPUT FILES
# =============================================================================


def _excel_value(value: Any) -> Any:
    """Convert a cell value to something openpyxl writes cleanly.

    Args:
        value: Any value from a DataFrame.

    Returns:
        None for missing values, Python scalars and datetimes otherwise.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, set, Path)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def write_workbook(path: Path, sheets: Sequence[Tuple[str, pd.DataFrame, Sequence[str]]]) -> None:
    """Write sheets to an Excel workbook: Arial throughout, bold headers, frozen, filtered.

    Args:
        path: Output .xlsx path.
        sheets: (sheet name, table, summary lines shown above the table).
    """
    workbook = Workbook()
    default_sheet = workbook.active
    if default_sheet is not None:
        workbook.remove(default_sheet)
    normal = Font(name="Arial", size=10)
    bold = Font(name="Arial", size=10, bold=True)
    for name, table, preamble in sheets:
        sheet = workbook.create_sheet(title=name[:31])
        row_number = 1
        for line in preamble:
            sheet.cell(row=row_number, column=1, value=line).font = bold
            row_number += 1
        if preamble:
            row_number += 1
        header_row = row_number
        columns = [str(col) for col in table.columns]
        widths = [len(col) for col in columns]
        for col_number, col in enumerate(columns, 1):
            sheet.cell(row=header_row, column=col_number, value=col).font = bold
        for row_offset, record in enumerate(table.itertuples(index=False, name=None), 1):
            for col_number, raw in enumerate(record, 1):
                value = _excel_value(raw)
                cell = sheet.cell(row=header_row + row_offset, column=col_number, value=value)
                cell.font = normal
                col = columns[col_number - 1]
                if isinstance(value, _dt.datetime):
                    cell.number_format = "yyyy-mm-dd hh:mm:ss"
                elif isinstance(value, float) and col.endswith("share"):
                    cell.number_format = "0.0%"
                elif isinstance(value, float) and "cost" in col:
                    cell.number_format = "0.0000"
                elif isinstance(value, float):
                    cell.number_format = "0.00"
                text_length = 19 if isinstance(value, _dt.datetime) else len(str(value or ""))
                widths[col_number - 1] = max(widths[col_number - 1], text_length)
        last_row = header_row + max(len(table), 1)
        last_col = get_column_letter(max(len(columns), 1))
        sheet.freeze_panes = sheet.cell(row=header_row + 1, column=1)
        sheet.auto_filter.ref = f"A{header_row}:{last_col}{last_row}"
        for col_number, width in enumerate(widths, 1):
            sheet.column_dimensions[get_column_letter(col_number)].width = min(
                max(width + 2, 8), 60
            )
    workbook.save(path)


def _config_block_from_text(text: str) -> str:
    """Return the lines strictly between the CONFIG markers in *text*.

    Args:
        text: Source text (the script file or a notebook cell).

    Returns:
        The verbatim configuration block.

    Raises:
        ValueError: A marker is missing.
    """
    lines = text.splitlines()
    begin = next((i for i, line in enumerate(lines) if line.strip() == _CONFIG_BEGIN), None)
    if begin is None:
        raise ValueError("CONFIG begin marker not found")
    end = next((i for i in range(begin + 1, len(lines)) if lines[i].strip() == _CONFIG_END), None)
    if end is None:
        raise ValueError("CONFIG end marker not found")
    return "\n".join(lines[begin + 1 : end])


def _notebook_cell_source() -> str:
    """Return the latest notebook cell that holds the CONFIG block, or "".

    Returns:
        Cell source text from IPython's input history, or "" outside IPython.
    """
    getter = getattr(builtins, "get_ipython", None)
    if getter is None:
        return ""
    shell = getter()
    history = shell.user_ns.get("In", []) if shell is not None else []
    for cell in reversed(list(history)):
        if _CONFIG_BEGIN in str(cell) and _CONFIG_END in str(cell):
            return str(cell)
    return ""


def config_block_and_source() -> Tuple[str, str]:
    """Return the verbatim CONFIG block and where it came from.

    Works when run as a file and when pasted into a notebook cell, where
    ``__file__`` is undefined.

    Returns:
        (config block text, source label).
    """
    try:
        source_path: Optional[Path] = Path(__file__).resolve()
    except NameError:
        source_path = None
    if source_path is not None and source_path.is_file():
        try:
            return _config_block_from_text(source_path.read_text(encoding="utf-8")), str(
                source_path
            )
        except (OSError, ValueError) as exc:
            return f"(config block unavailable: {exc})", str(source_path)
    cell = _notebook_cell_source()
    if cell:
        try:
            return _config_block_from_text(cell), "<Jupyter notebook cell>"
        except ValueError as exc:
            return f"(config block unavailable: {exc})", "<Jupyter notebook cell>"
    return "(config block unavailable: no source file or notebook history)", "<interactive>"


def write_run_log(
    log_path: Path,
    settings: Settings,
    inputs: Sequence[Path],
    totals: Dict[str, Any],
    exit_code: int,
) -> bool:
    """Write the run-log sidecar.

    It records the CONFIG block verbatim, the resolved parameters, a SHA-256
    fingerprint of every input file, the model ID, the prompt hash and run totals.

    Args:
        log_path: Path of the ``_runlog.txt`` file.
        settings: Run settings.
        inputs: Input files to fingerprint.
        totals: Run totals to record.
        exit_code: The run's exit code.

    Returns:
        True if the file was written.
    """
    block, source = config_block_and_source()
    lines = [
        "Video passenger count pilot - run log",
        f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"Script: {source}",
        f"Mode: {settings.mode}   Dry run: {settings.dry_run}   Exit code: {exit_code}",
        "",
        "=== CONFIGURATION (verbatim) ===",
        _CONFIG_BEGIN,
        block,
        _CONFIG_END,
        "",
        "=== RESOLVED PARAMETERS ===",
    ]
    lines += [f"{name} = {value!r}" for name, value in settings._asdict().items()]
    lines += [
        "",
        "=== MODEL AND PROMPT ===",
        f"model_id = {settings.model_id}",
        f"prompt_sha256 = {prompt_hash()}",
        "",
        "=== INPUT FINGERPRINTS (SHA-256) ===",
    ]
    if inputs:
        logging.info("Fingerprinting %d input file(s)...", len(inputs))
    for path in inputs:
        size = path.stat().st_size if path.exists() else 0
        lines.append(f"{_sha256_file(path)}  {size:>14,} bytes  {path}")
    lines += ["", "=== RUN TOTALS ==="]
    lines += [f"{name} = {value}" for name, value in totals.items()]
    try:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logging.error("Could not write the run log %s: %s", log_path, exc)
        return False
    logging.info("Run log: %s", log_path)
    return True


def _finish(
    log_path: Path, settings: Settings, inputs: Sequence[Path], totals: Dict[str, Any], code: int
) -> int:
    """Write the run log and return the final exit code.

    Args:
        log_path: Run-log path.
        settings: Run settings.
        inputs: Input files to fingerprint.
        totals: Run totals.
        code: Exit code so far.

    Returns:
        *code*, or 1 if the run log could not be written and REQUIRE_RUN_LOG is set.
    """
    if not write_run_log(log_path, settings, inputs, totals, code) and settings.require_run_log:
        logging.error("REQUIRE_RUN_LOG is True and the run log could not be written.")
        return 1
    return code


# =============================================================================
# MODES
# =============================================================================


def _require_inputs(paths: Dict[str, str]) -> Dict[str, Path]:
    """Check that input files and folders exist.

    Args:
        paths: Setting name -> path.

    Returns:
        Setting name -> Path.

    Raises:
        PilotError: One or more inputs are missing.
    """
    resolved = {name: Path(value) for name, value in paths.items()}
    missing = [f"{name} = {path}" for name, path in resolved.items() if not path.exists()]
    if missing:
        raise PilotError("Input not found: " + "; ".join(missing))
    return resolved


def _video_inputs(video_dir: Path, settings: Settings) -> List[Path]:
    """Return the video files in VIDEO_DIR (for fingerprinting).

    Args:
        video_dir: Clip folder.
        settings: Run settings.

    Returns:
        Video files, sorted.
    """
    extensions = {ext.lower() for ext in settings.video_extensions}
    return sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def _load_door_jobs(
    settings: Settings,
) -> Tuple[Dict[str, Path], pd.DataFrame, List[ClipJob], List[Dict[str, Any]], Dict[str, Any]]:
    """Load events and clips and match them (shared by count, template and calibrate).

    Args:
        settings: Run settings.

    Returns:
        (input paths, usable door events, clip jobs, reconciliation rows, totals).

    Raises:
        PilotError: Inputs are missing or empty.
    """
    paths = _require_inputs(
        {"VIDEO_DIR": settings.video_dir, "STOP_EVENTS_PATH": settings.stop_events_path}
    )
    events = load_stop_events(paths["STOP_EVENTS_PATH"], settings)
    door_events, recon, n_passed = classify_door_events(events)
    logging.info(
        "Stop visits: %d rows, %d with usable door times, %d passed without opening.",
        len(events),
        len(door_events),
        n_passed,
    )
    if door_events.empty:
        raise PilotError(
            "No stop-visit row has usable door-open and door-close times; check the "
            "EVENT_DOOR_*_COL settings and EVENT_TIME_FORMAT."
        )
    clips, video_recon = scan_videos(paths["VIDEO_DIR"], settings)
    recon += video_recon
    n_door = int((clips["kind"] == "door").sum())
    n_outside = int((clips["kind"] == "outside").sum())
    logging.info("Clips: %d door camera, %d outside camera (not used here).", n_door, n_outside)
    if n_door == 0:
        raise PilotError(
            f"No readable door-camera clips in {paths['VIDEO_DIR']}; check VIDEO_EXTENSIONS, "
            "FILENAME_REGEX and CAMERA_DOORS."
        )
    jobs, match_recon = match_door_events(door_events, clips, settings)
    recon += match_recon
    totals = {
        "stop_visit_rows": len(events),
        "door_events": len(door_events),
        "passed_without_opening": n_passed,
        "door_clips": n_door,
        "clip_jobs_with_video": len(jobs),
    }
    return paths, door_events, jobs, recon, totals


def _recon_frame(recon: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return reconciliation rows as a table, logging a count per category.

    Args:
        recon: Reconciliation rows.

    Returns:
        Reconciliation table.
    """
    table = pd.DataFrame(recon, columns=RECON_COLUMNS)
    for category, size in table["category"].value_counts().items():
        logging.info("Reconciliation: %s = %d", category, size)
    return table


def run_count(settings: Settings, client: Any, stamp: str) -> int:
    """Stage 1: count passengers in every door-open window and score the counts.

    Args:
        settings: Run settings.
        client: Anthropic client, a test stub, or None to create one.
        stamp: Timestamp for output names.

    Returns:
        Exit code: 0 success, 1 failure (including unparseable answers).

    Raises:
        PilotError: Inputs missing, nothing matched, or the estimate is over budget.
    """
    gt_name = "MANUAL_COUNTS_CSV" if settings.ground_truth_source == "manual" else "RIDECHECK_PATH"
    gt_value = (
        settings.manual_counts_csv
        if settings.ground_truth_source == "manual"
        else settings.ridecheck_path
    )
    gt_path = _require_inputs({gt_name: gt_value})[gt_name]
    paths, door_events, jobs, recon, totals = _load_door_jobs(settings)
    if settings.ground_truth_source == "manual":
        counts, gt_recon = load_manual_counts(gt_path)
        use_sequence = True
    else:
        counts, gt_recon = load_ridecheck(gt_path, settings)
        use_sequence = bool(settings.ridecheck_seq_col)
    recon += gt_recon
    jobs, attach_recon = attach_ground_truth(jobs, counts, door_events, settings, use_sequence)
    recon += attach_recon
    if not jobs:
        _recon_frame(recon)
        raise PilotError(
            "Zero events matched: no door-open event has both a covering clip and ground "
            "truth. See the reconciliation counts above."
        )
    planned = [plan_count_job(job, settings) for job in jobs]
    n_requests = sum(len(p.chunks) for p in planned) * settings.repeats
    estimate = sum(_est_job_cost(p, settings) for p in planned) * settings.repeats
    est_tokens = sum(sum(p.chunk_tokens) for p in planned) * settings.repeats
    logging.info(
        "Plan: %d clips, %d requests (%d repeats), ~%s input tokens, estimated cost $%.2f "
        "(budget $%.2f).",
        len(planned),
        n_requests,
        settings.repeats,
        f"{est_tokens:,}",
        estimate,
        settings.budget_usd,
    )
    if estimate > settings.budget_usd and not settings.override_budget:
        if settings.dry_run:
            logging.warning("The estimate exceeds BUDGET_USD; a live run would stop here.")
        else:
            raise PilotError(
                f"Estimated cost ${estimate:.2f} exceeds BUDGET_USD ${settings.budget_usd:.2f}. "
                "Raise BUDGET_USD, narrow TRIPS_TO_INCLUDE, or set OVERRIDE_BUDGET = True."
            )
    if not settings.dry_run and client is None:
        client = make_client(settings)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{settings.output_prefix}_{settings.model_id}_{stamp}"
    if settings.dry_run:
        stem += "_DRYRUN"
    frames_dir: Optional[Path] = None
    if settings.keep_frames:
        frames_dir = _local_frames_root(settings) / stem
        frames_dir.mkdir(parents=True, exist_ok=True)
        logging.info("KEEP_FRAMES: frames will be kept in %s", frames_dir)
    control = RunControl(settings.max_consecutive_api_errors)

    def _worker(item: PlannedJob) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return process_count_job(item, settings, client, frames_dir, control)

    def _describe(item: PlannedJob) -> Dict[str, Any]:
        job = item.job
        return {
            "trip": job.trip,
            "bus": job.bus,
            "stop_code": job.stop_code,
            "stop_sequence": job.stop_sequence,
            "door": job.door,
            "video_file": job.video_path.name,
        }

    rows, run_recon = run_workers(planned, _worker, settings.max_concurrent_requests, _describe)
    recon += run_recon
    if control.fatal_message:
        recon.append(_recon("api_fatal_error", control.fatal_message))
    events_table = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if not events_table.empty:
        door_order = {door: i for i, door in enumerate(_door_names(settings))}
        events_table = (
            events_table.assign(
                _seq=_seq_number(events_table["stop_sequence"]),
                _door=events_table["door"].map(door_order),
            )
            .sort_values(["trip", "_seq", "stop_sequence", "_door", "repeat"])
            .drop(columns=["_seq", "_door"])
            .reset_index(drop=True)
        )
    repeat_table, repeat_lines = repeatability_table(events_table, settings.repeats)
    sheets = [
        ("Events", events_table, []),
        ("Trip summary", trip_summary_table(events_table), []),
        ("Accuracy", accuracy_table(events_table, _door_names(settings)), []),
        ("Repeatability", repeat_table, repeat_lines),
        ("Cost", count_cost_table(events_table, planned, settings), []),
        ("Reconciliation", _recon_frame(recon), []),
    ]
    workbook_path = output_dir / f"{stem}.xlsx"
    write_workbook(workbook_path, sheets)
    logging.info("Workbook: %s", workbook_path)
    status_counts = events_table["status"].value_counts().to_dict() if len(events_table) else {}
    n_ok = int(status_counts.get("ok", 0))
    n_unparseable = int(status_counts.get("unparseable", 0))
    code = 0
    if events_table.empty:
        logging.error("No clip could be processed; see Reconciliation for the reasons.")
        code = 1
    if control.fatal_message:
        logging.error("The run stopped early: %s", control.fatal_message)
        code = 1
    if n_unparseable:
        logging.error(
            "%d answer(s) could not be parsed after %d retries (see Reconciliation, "
            "unparseable_response). Results were written, but the run is marked failed.",
            n_unparseable,
            settings.max_parse_retries,
        )
        code = 1
    if not settings.dry_run and n_ok == 0:
        logging.error("No request returned a usable answer.")
        code = 1
    totals.update(
        {
            "clip_jobs_sent": len(planned),
            "requests_planned": n_requests,
            "estimated_cost_usd": round(estimate, 4),
            "actual_cost_usd": round(
                float(pd.to_numeric(events_table["cost_usd"], errors="coerce").fillna(0).sum()), 4
            ),
            "event_rows_by_status": status_counts,
            "reconciliation_rows": len(recon),
            "workbook": workbook_path,
        }
    )
    inputs = [paths["STOP_EVENTS_PATH"], gt_path] + _video_inputs(paths["VIDEO_DIR"], settings)
    code = _finish(output_dir / f"{stem}_runlog.txt", settings, inputs, totals, code)
    if code == 0:
        logging.info("Run completed successfully.")
    return code


def run_template(settings: Settings, stamp: str) -> int:
    """Write a blank manual-count CSV: one row per door-open event and door with video.

    The window columns give the same padded window the model sees, in seconds into
    the clip, so manual counts cover exactly what the model counts.

    Args:
        settings: Run settings.
        stamp: Timestamp for the file name.

    Returns:
        Exit code 0.

    Raises:
        PilotError: Inputs missing or no event has video.
    """
    paths, _, jobs, recon, totals = _load_door_jobs(settings)
    _recon_frame(recon)
    if not jobs:
        raise PilotError("Zero events matched a clip, so the template would be empty.")
    rows = [
        {
            "trip": job.trip,
            "bus": job.bus,
            "stop_code": job.stop_code,
            "stop_sequence": job.stop_sequence,
            "door": job.door,
            "video_file": job.video_path.name,
            "window_start_s": round(job.start_pos, 1),
            "window_end_s": round(job.end_pos, 1),
            "door_open_s": round((job.door_open - job.clip_start_avl).total_seconds(), 1),
            "door_close_s": round((job.door_close - job.clip_start_avl).total_seconds(), 1),
            "door_open_time": job.door_open,
            "boardings": "",
            "alightings": "",
            "counter_notes": "",
        }
        for job in jobs
    ]
    table = pd.DataFrame(rows)
    table = (
        table.assign(_seq=_seq_number(table["stop_sequence"]))
        .sort_values(["trip", "_seq", "stop_sequence", "door"])
        .drop(columns="_seq")
    )
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{settings.template_prefix}_{stamp}.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info(
        "Template with %d rows written to %s. Fill in boardings and alightings for the "
        "window given (seconds into the clip), then set MANUAL_COUNTS_CSV to the file.",
        len(table),
        out_path,
    )
    totals["template_rows"] = len(table)
    totals["template"] = out_path
    inputs = [paths["STOP_EVENTS_PATH"]] + _video_inputs(paths["VIDEO_DIR"], settings)
    code = _finish(out_path.with_name(out_path.stem + "_runlog.txt"), settings, inputs, totals, 0)
    if code == 0:
        logging.info("Template completed successfully.")
    return code


def _grab_still(
    cv2: Any, job: ClipJob, settings: Settings, out_path: Path, offset_note: str
) -> bool:
    """Save the frame at an event's AVL door-open time, labelled, as a JPEG.

    Args:
        cv2: The OpenCV module.
        job: Clip job.
        settings: Run settings.
        out_path: JPEG path.
        offset_note: Text drawn on the still.

    Returns:
        True if the still was saved.
    """
    position = (job.door_open - job.clip_start_avl).total_seconds()
    index = min(max(int(round(position * job.video_fps)), 0), job.video_frames - 1)
    capture = cv2.VideoCapture(str(job.video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return False
    frame = cv2.resize(frame, (settings.frame_width, settings.frame_height))
    height = settings.frame_height
    cv2.rectangle(frame, (0, height - 28), (settings.frame_width, height), (0, 0, 0), -1)
    cv2.putText(
        frame, offset_note, (6, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
    )
    return bool(cv2.imwrite(str(out_path), frame))


def run_calibrate(settings: Settings, stamp: str) -> int:
    """Save a still at each of the first CALIBRATION_STILLS door-open times, per door.

    Args:
        settings: Run settings.
        stamp: Timestamp for the folder name.

    Returns:
        Exit code 0, or 1 if no still could be saved.

    Raises:
        PilotError: Inputs missing or no event has video.
    """
    paths, _, jobs, recon, totals = _load_door_jobs(settings)
    if not jobs:
        _recon_frame(recon)
        raise PilotError("Zero events matched a clip, so there is nothing to calibrate.")
    jobs = sorted(jobs, key=lambda j: (j.door_open, j.door))
    chosen_events: List[Tuple[str, str, str]] = []
    for job in jobs:
        key = (job.trip, job.stop_code, job.stop_sequence)
        if key not in chosen_events and len(chosen_events) < settings.calibration_stills:
            chosen_events.append(key)
    folder = _local_frames_root(settings) / f"calibration_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    cv2 = _import_cv2()
    saved = 0
    for job in jobs:
        if (job.trip, job.stop_code, job.stop_sequence) not in chosen_events:
            continue
        note = (
            f"AVL door open {job.door_open:%H:%M:%S} | trip {job.trip} stop {job.stop_code} "
            f"{job.door} | offset {settings.clock_offset_seconds:+.1f}s"
        )
        name = _safe_name("calib", f"{job.door_open:%H%M%S}", job.trip, job.stop_sequence, job.door)
        name += ".jpg"
        if _grab_still(cv2, job, settings, folder / name, note):
            saved += 1
    if saved == 0:
        logging.error("No still could be read from the clips.")
        return 1
    logging.info(
        "Saved %d still(s) to %s. Check that the doors are just opening in each one. If they "
        "open d seconds after the still, subtract d from CLOCK_OFFSET_SECONDS; if they were "
        "already open d seconds earlier, add d. Delete the stills when done: they show riders.",
        saved,
        folder,
    )
    totals["stills_saved"] = saved
    totals["stills_folder"] = folder
    inputs = [paths["STOP_EVENTS_PATH"]] + _video_inputs(paths["VIDEO_DIR"], settings)
    code = _finish(folder.with_name(folder.name + "_runlog.txt"), settings, inputs, totals, 0)
    if code == 0:
        logging.info("Calibration completed successfully.")
    return code


# =============================================================================
# STAGE 2
# =============================================================================


def stage2_visits(
    events: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Turn stop-visit rows into scored visits with an actual time and a served flag.

    The actual time is the arrival time when EVENT_ARRIVAL_COL gives one, else the
    door-open time. Served = the doors opened.

    Args:
        events: Output of :func:`load_stop_events`.

    Returns:
        (visits with visit_time and served, reconciliation rows for visits with
        no usable time).
    """
    visits = events.copy()
    for col in ("door_open", "arrival"):
        visits[col] = visits[col].where(visits[col].dt.year >= 1950)
    visits["served"] = visits["door_open"].notna()
    visits["visit_time"] = visits["arrival"].fillna(visits["door_open"])
    no_time = visits["visit_time"].isna()
    recon = [
        _recon(
            "visit_no_time",
            "no usable arrival or door-open time; not scored",
            trip=row["trip"],
            bus=row["bus"],
            stop_code=row["stop_code"],
            stop_sequence=row["stop_sequence"],
        )
        for row in visits[no_time].to_dict("records")
    ]
    return visits[~no_time].reset_index(drop=True), recon


def load_gtfs_patterns(gtfs_path: str, trips: set) -> Dict[str, List[Tuple[str, str]]]:
    """Read each trip's ordered stops (stop_code, stop_name) from GTFS.

    stop_times and stops are joined on GTFS stop_id, which is internal to GTFS;
    what gets compared with CLEVER is stop_code.

    Args:
        gtfs_path: GTFS folder or .zip.
        trips: Trip IDs needed.

    Returns:
        trip_id -> [(stop_code, stop_name), ...] in stop_sequence order.

    Raises:
        PilotError: The feed can't be read or lacks a needed column.
    """
    try:
        with open_gtfs_file(gtfs_path, "stops.txt") as handle:
            stops = pd.read_csv(handle, dtype=str, keep_default_na=False)
        with open_gtfs_file(gtfs_path, "stop_times.txt") as handle:
            stop_times = pd.read_csv(
                handle,
                dtype=str,
                keep_default_na=False,
                usecols=["trip_id", "stop_id", "stop_sequence"],
            )
    except (OSError, ValueError) as exc:
        raise PilotError(f"Could not read GTFS from {gtfs_path}: {exc}") from exc
    for col in ("stop_id", "stop_code", "stop_name"):
        if col not in stops.columns:
            raise PilotError(f"GTFS stops.txt has no {col} column; Stage 2 needs stop_code.")
    stop_times["trip_id"] = stop_times["trip_id"].map(_clean_id)
    stop_times = stop_times[stop_times["trip_id"].isin(trips)]
    merged = stop_times.merge(
        stops[["stop_id", "stop_code", "stop_name"]], on="stop_id", how="left"
    )
    merged["seq_num"] = _seq_number(merged["stop_sequence"])
    patterns: Dict[str, List[Tuple[str, str]]] = {}
    for trip, group in merged.sort_values(["trip_id", "seq_num"]).groupby("trip_id", sort=False):
        patterns[str(trip)] = [
            (_clean_id(code), str(name))
            for code, name in zip(group["stop_code"], group["stop_name"])
        ]
    return patterns


def assign_outside_clips(
    visits: pd.DataFrame, outside: pd.DataFrame
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Match each outside-camera clip to the trip it covers most, one clip per trip.

    Only the trip's first and last visit times are used to pick the trip; the
    model never sees them.

    Args:
        visits: Output of :func:`stage2_visits`.
        outside: Outside-camera rows of :func:`scan_videos`.

    Returns:
        (trip -> clip record, reconciliation rows).
    """
    spans = (
        visits.groupby(["trip", "bus"])
        .agg(t0=("visit_time", "min"), t1=("visit_time", "max"))
        .reset_index()
    )
    recon: List[Dict[str, Any]] = []
    best_by_trip: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for clip in outside.to_dict("records"):
        clip_end = clip["start_avl"] + _dt.timedelta(seconds=clip["last_pos"])
        best: Optional[Tuple[float, str]] = None
        for span in spans[spans["bus"] == clip["bus"]].to_dict("records"):
            overlap = (
                min(clip_end, span["t1"]) - max(clip["start_avl"], span["t0"])
            ).total_seconds()
            if overlap >= 0 and (best is None or overlap > best[0]):
                best = (overlap, span["trip"])
        if best is None:
            recon.append(
                _recon(
                    "video_no_events",
                    "no trip of this bus falls inside the outside-camera clip",
                    bus=clip["bus"],
                    video_file=clip["file"],
                )
            )
            continue
        overlap, trip = best
        previous = best_by_trip.get(trip)
        if previous is None or overlap > previous[0]:
            if previous is not None:
                recon.append(_duplicate_clip_recon(trip, previous[1]))
            best_by_trip[trip] = (overlap, clip)
        else:
            recon.append(_duplicate_clip_recon(trip, clip))
    for row in spans.to_dict("records"):
        if row["trip"] not in best_by_trip:
            recon.append(
                _recon(
                    "trip_no_outside_video",
                    "no outside-camera clip covers this trip",
                    trip=row["trip"],
                    bus=row["bus"],
                )
            )
    return {trip: clip for trip, (_, clip) in best_by_trip.items()}, recon


def _duplicate_clip_recon(trip: str, clip: Dict[str, Any]) -> Dict[str, Any]:
    """Return the reconciliation row for an outside clip left out for a trip.

    Args:
        trip: Trip ID.
        clip: The clip left out.

    Returns:
        Reconciliation row.
    """
    return _recon(
        "video_duplicate_for_trip",
        "another outside-camera clip covers more of this trip; this one not used",
        trip=trip,
        bus=clip["bus"],
        video_file=clip["file"],
    )


def plan_stage2_windows(
    trip: str, clip: Dict[str, Any], pattern: List[Tuple[str, str]], settings: Settings
) -> List[PlannedWindow]:
    """Cut a trip's outside-camera clip into windows of consecutive frames.

    Args:
        trip: Trip ID.
        clip: Clip record from :func:`scan_videos`.
        pattern: The trip's (stop_code, stop_name) list.
        settings: Run settings.

    Returns:
        Planned windows, each split into chunks under the request limits.
    """
    cpt = settings.text_chars_per_token
    rate = settings.stage2_frame_rate
    times = [k / rate for k in range(int(math.floor(clip["last_pos"] * rate + 1e-9)) + 1)]
    stop_text = "\n".join(f"{i}. {code} - {name}" for i, (code, name) in enumerate(pattern, 1))
    per_frame = image_tokens(settings.frame_width, settings.frame_height) + _text_tokens(
        STAGE2_FRAME_LABEL.format(t=99999), cpt
    )
    fixed_text = (
        STAGE2_SYSTEM_PROMPT
        + STAGE2_STOPS_INTRO
        + stop_text
        + STAGE2_QUESTION
        + CHUNK_NOTE
        + json.dumps(STAGE2_SCHEMA)
    )
    fixed = _text_tokens(fixed_text, cpt) + settings.request_overhead_tokens
    by_window: Dict[int, List[float]] = {}
    for t in times:
        by_window.setdefault(int(t // settings.stage2_window_seconds), []).append(t)
    windows: List[PlannedWindow] = []
    for index, window_times in sorted(by_window.items()):
        chunks = plan_chunks(
            len(window_times),
            per_frame,
            fixed,
            settings.request_token_ceiling,
            settings.max_images_per_request,
        )
        windows.append(
            PlannedWindow(
                trip=trip,
                bus=clip["bus"],
                video_path=clip["path"],
                video_fps=clip["fps"],
                video_frames=clip["frames"],
                clip_start_avl=clip["start_avl"],
                window_index=index,
                frame_times=window_times,
                chunks=chunks,
                chunk_tokens=[fixed + (end - start) * per_frame for start, end in chunks],
                stop_list_text=stop_text,
                pattern_codes=tuple(code for code, _ in pattern),
            )
        )
    return windows


def process_stage2_window(
    window: PlannedWindow, settings: Settings, client: Any, control: RunControl
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Send one Stage 2 window (all its chunks) and record the detections.

    Args:
        window: Planned window.
        settings: Run settings.
        client: Anthropic client, or None in a dry run.
        control: Run controller.

    Returns:
        (one request row per chunk, each with its detections; reconciliation rows).
    """
    prices = settings.model_prices[settings.model_id]
    ids = {"trip": window.trip, "bus": window.bus, "video_file": window.video_path.name}
    base = {**ids, "window": window.window_index}
    if control.abort.is_set():
        return [], [_recon("not_sent", f"run stopped: {control.fatal_message}", **ids)]
    cv2 = _import_cv2()
    frames = extract_frames(
        cv2,
        window.video_path,
        window.frame_times,
        window.video_fps,
        window.video_frames,
        settings.frame_width,
        settings.frame_height,
        settings.jpeg_quality,
    )
    if len(frames) != len(window.frame_times):
        detail = f"window {window.window_index}: read {len(frames)} of {len(window.frame_times)}"
        return [], [_recon("frame_extraction_failed", detail, **ids)]
    rows: List[Dict[str, Any]] = []
    recon: List[Dict[str, Any]] = []
    n_chunks = len(window.chunks)
    for index, (start, end) in enumerate(window.chunks):
        est_cost = estimate_request_cost(
            window.chunk_tokens[index], settings.est_output_tokens_per_request, prices
        )
        row: Dict[str, Any] = {
            **base,
            "chunk": index + 1,
            "frames": end - start,
            "est_input_tokens": window.chunk_tokens[index],
            "est_cost_usd": est_cost,
            "detections": [],
        }
        if settings.dry_run:
            rows.append({**row, "status": "dry_run"})
            continue
        if control.abort.is_set():
            rows.append({**row, "status": "not_sent"})
            continue
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": STAGE2_STOPS_INTRO + window.stop_list_text}
        ]
        if n_chunks > 1:
            content.append(
                {"type": "text", "text": CHUNK_NOTE.format(part=index + 1, parts=n_chunks)}
            )
        for t, jpeg in zip(window.frame_times[start:end], frames[start:end]):
            content.append({"type": "text", "text": STAGE2_FRAME_LABEL.format(t=t)})
            content.append(_image_block(jpeg))
        content.append({"type": "text", "text": STAGE2_QUESTION})
        kwargs = request_kwargs(settings, STAGE2_SYSTEM_PROMPT, content, STAGE2_SCHEMA)
        try:
            result = call_model(client, kwargs, parse_stage2_response, settings, control)
        except FatalApiError as exc:
            control.stop(str(exc))
            rows.append({**row, "status": "api_error", "detail": str(exc)})
            break
        row.update(
            {
                "status": result.status,
                "detail": result.detail,
                "input_tokens": result.usage["input"],
                "output_tokens": result.usage["output"],
                "cache_write_tokens": result.usage["cache_write"],
                "cache_read_tokens": result.usage["cache_read"],
                "cost_usd": result.cost_usd,
                "api_calls": result.api_calls,
            }
        )
        if result.status == "ok" and result.data is not None:
            row["notes"] = result.data["notes"]
            for det in result.data["stops"]:
                row["detections"].append(
                    {
                        **ids,
                        "window": window.window_index,
                        "stop_code": det["stop_code"],
                        "frame_time_s": det["frame_time_s"],
                        "detected_time": window.clip_start_avl
                        + _dt.timedelta(seconds=det["frame_time_s"]),
                        "action": det["action"],
                        "in_pattern": "Y" if det["stop_code"] in window.pattern_codes else "N",
                    }
                )
        else:
            category = {"api_error": "api_failure", "unparseable": "unparseable_response"}.get(
                result.status, result.status
            )
            recon.append(_recon(category, f"window {window.window_index}: {result.detail}", **ids))
        rows.append(row)
    return rows, recon


def dedupe_detections(detections: List[Dict[str, Any]], seconds: float) -> None:
    """Mark repeat reports of the same stop within *seconds* as duplicates, in place.

    Args:
        detections: One trip's detections.
        seconds: STAGE2_DEDUPE_SECONDS.
    """
    kept_times: Dict[str, List[_dt.datetime]] = {}
    for det in sorted(detections, key=lambda d: d["detected_time"]):
        earlier = kept_times.setdefault(det["stop_code"], [])
        det["duplicate"] = (
            "Y"
            if any(abs((det["detected_time"] - t).total_seconds()) <= seconds for t in earlier)
            else "N"
        )
        if det["duplicate"] == "N":
            earlier.append(det["detected_time"])


def score_stage2_trip(
    visits: pd.DataFrame, detections: List[Dict[str, Any]], tolerance: float
) -> List[Dict[str, Any]]:
    """Match one trip's detections to its actual stop visits.

    First, each visit takes the nearest unused detection of the same stop code
    within the tolerance (correct stop). Then visits still unmatched take the
    nearest unused detection of any code within the tolerance (detected, wrong
    stop). Detections left over match no real stop. Time error is measured on
    correct-stop matches only.

    Args:
        visits: The trip's visits (visit_time, served, stop_code, stop_sequence).
        detections: The trip's detections, already deduplicated (updated in place
            with match results).
        tolerance: STAGE2_MATCH_TOLERANCE_SECONDS.

    Returns:
        One Stop-visits row per visit.
    """
    active = [d for d in detections if d.get("duplicate") != "Y"]
    for det in detections:
        det.update({"match": "duplicate" if det.get("duplicate") == "Y" else "false_detection"})
    used = [False] * len(active)
    records = visits.sort_values("visit_time").to_dict("records")
    matches: Dict[int, Tuple[int, bool]] = {}
    for same_code in (True, False):
        for v_index, visit in enumerate(records):
            if v_index in matches:
                continue
            best: Optional[Tuple[float, int]] = None
            for d_index, det in enumerate(active):
                if used[d_index] or (same_code and det["stop_code"] != visit["stop_code"]):
                    continue
                gap = abs((det["detected_time"] - visit["visit_time"]).total_seconds())
                if gap <= tolerance and (best is None or gap < best[0]):
                    best = (gap, d_index)
            if best is not None:
                used[best[1]] = True
                matches[v_index] = (best[1], same_code)
    rows: List[Dict[str, Any]] = []
    for v_index, visit in enumerate(records):
        row: Dict[str, Any] = {
            "trip": visit["trip"],
            "stop_code": visit["stop_code"],
            "stop_sequence": visit["stop_sequence"],
            "served": "doors opened" if visit["served"] else "passed without opening",
            "actual_time": visit["visit_time"],
            "detected": "N",
            "correct_stop": "N",
        }
        if v_index in matches:
            d_index, correct = matches[v_index]
            det = active[d_index]
            det["match"] = "correct" if correct else "wrong_stop"
            det["matched_stop_code"] = visit["stop_code"]
            det["matched_stop_sequence"] = visit["stop_sequence"]
            row.update(
                {
                    "detected": "Y",
                    "correct_stop": "Y" if correct else "N",
                    "detected_stop_code": det["stop_code"],
                    "detected_time": det["detected_time"],
                }
            )
            if correct:
                error = (det["detected_time"] - visit["visit_time"]).total_seconds()
                row["time_error_s"] = error
                det["time_error_s"] = error
        rows.append(row)
    return rows


def stage2_scores(
    visit_rows: pd.DataFrame, detections: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the Stage 2 Scores sheet.

    Args:
        visit_rows: Stop-visits sheet.
        detections: Detections sheet.

    Returns:
        (scores by group, summary lines on detections).
    """
    rows: List[Dict[str, Any]] = []
    groups = [
        ("all visits", visit_rows),
        ("doors opened", visit_rows[visit_rows["served"] == "doors opened"]),
        ("passed without opening", visit_rows[visit_rows["served"] == "passed without opening"]),
    ]
    for name, group in groups:
        errors = pd.to_numeric(group.get("time_error_s"), errors="coerce").dropna().abs()
        rows.append(
            {
                "group": name,
                "visits": len(group),
                "detected_share": float((group["detected"] == "Y").mean()) if len(group) else None,
                "correct_stop_share": (
                    float((group["correct_stop"] == "Y").mean()) if len(group) else None
                ),
                "median_abs_time_error_s": float(errors.median()) if len(errors) else None,
                "p90_abs_time_error_s": float(errors.quantile(0.9)) if len(errors) else None,
            }
        )
    active = detections[detections["match"] != "duplicate"] if len(detections) else detections
    n_false = int((active["match"] == "false_detection").sum()) if len(active) else 0
    share = f"{n_false / len(active):.1%}" if len(active) else "n/a"
    lines = [
        f"Detections (after removing repeats): {len(active)}",
        f"Detections that match no real stop visit: {n_false} ({share})",
        "Time errors are detection time minus actual time, on correct-stop matches only.",
    ]
    return pd.DataFrame(rows), lines


def run_stage2(settings: Settings, client: Any, stamp: str) -> int:
    """Stage 2: identify stops served or passed from the outside camera alone, and score.

    Args:
        settings: Run settings.
        client: Anthropic client, a test stub, or None to create one.
        stamp: Timestamp for output names.

    Returns:
        Exit code.

    Raises:
        PilotError: Inputs missing, nothing matched, or the estimate is over budget.
    """
    paths = _require_inputs(
        {
            "VIDEO_DIR": settings.video_dir,
            "STOP_EVENTS_PATH": settings.stop_events_path,
            "GTFS_PATH": settings.gtfs_path,
        }
    )
    events = load_stop_events(paths["STOP_EVENTS_PATH"], settings)
    visits, recon = stage2_visits(events)
    clips, video_recon = scan_videos(paths["VIDEO_DIR"], settings)
    recon += video_recon
    outside = clips[clips["kind"] == "outside"]
    if outside.empty:
        raise PilotError(
            f"No readable outside-camera clips (camera code '{settings.outside_camera}') in "
            f"{paths['VIDEO_DIR']}."
        )
    assignments, assign_recon = assign_outside_clips(visits, outside)
    recon += assign_recon
    patterns = load_gtfs_patterns(settings.gtfs_path, set(assignments))
    for trip in sorted(set(assignments) - set(patterns)):
        recon.append(_recon("trip_not_in_gtfs", "trip_id not found in GTFS stop_times", trip=trip))
        assignments.pop(trip)
    if not assignments:
        _recon_frame(recon)
        raise PilotError("Zero trips matched an outside-camera clip and a GTFS trip.")
    windows: List[PlannedWindow] = []
    for trip, clip in sorted(assignments.items()):
        windows += plan_stage2_windows(trip, clip, patterns[trip], settings)
    prices = settings.model_prices[settings.model_id]
    out = settings.est_output_tokens_per_request
    estimate = sum(estimate_request_cost(t, out, prices) for w in windows for t in w.chunk_tokens)
    n_requests = sum(len(w.chunks) for w in windows)
    logging.info(
        "Stage 2 plan: %d trips, %d windows, %d requests, estimated cost $%.2f (budget $%.2f).",
        len(assignments),
        len(windows),
        n_requests,
        estimate,
        settings.budget_usd,
    )
    if estimate > settings.budget_usd and not settings.override_budget:
        if settings.dry_run:
            logging.warning("The estimate exceeds BUDGET_USD; a live run would stop here.")
        else:
            raise PilotError(
                f"Estimated cost ${estimate:.2f} exceeds BUDGET_USD ${settings.budget_usd:.2f}. "
                "Raise BUDGET_USD, narrow TRIPS_TO_INCLUDE, or set OVERRIDE_BUDGET = True."
            )
    if not settings.dry_run and client is None:
        client = make_client(settings)
    control = RunControl(settings.max_consecutive_api_errors)

    def _worker(item: PlannedWindow) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return process_stage2_window(item, settings, client, control)

    def _describe(item: PlannedWindow) -> Dict[str, Any]:
        return {"trip": item.trip, "bus": item.bus, "video_file": item.video_path.name}

    request_rows, run_recon = run_workers(
        windows, _worker, settings.max_concurrent_requests, _describe
    )
    recon += run_recon
    if control.fatal_message:
        recon.append(_recon("api_fatal_error", control.fatal_message))
    detection_rows: List[Dict[str, Any]] = []
    visit_rows: List[Dict[str, Any]] = []
    for trip in sorted(assignments):
        trip_dets = [d for r in request_rows if r["trip"] == trip for d in r["detections"]]
        dedupe_detections(trip_dets, settings.stage2_dedupe_seconds)
        trip_visits = visits[visits["trip"] == trip]
        visit_rows += score_stage2_trip(
            trip_visits, trip_dets, settings.stage2_match_tolerance_seconds
        )
        detection_rows += sorted(trip_dets, key=lambda d: d["detected_time"])
    det_columns = [
        "trip",
        "bus",
        "video_file",
        "window",
        "stop_code",
        "frame_time_s",
        "detected_time",
        "action",
        "in_pattern",
        "duplicate",
        "match",
        "matched_stop_code",
        "matched_stop_sequence",
        "time_error_s",
    ]
    visit_columns = [
        "trip",
        "stop_code",
        "stop_sequence",
        "served",
        "actual_time",
        "detected",
        "correct_stop",
        "detected_stop_code",
        "detected_time",
        "time_error_s",
    ]
    detections = pd.DataFrame(detection_rows, columns=det_columns)
    visit_table = pd.DataFrame(visit_rows, columns=visit_columns)
    scores, score_lines = stage2_scores(visit_table, detections)
    requests = pd.DataFrame(request_rows)
    if requests.empty:
        requests = pd.DataFrame(columns=["status", "cost_usd", "input_tokens", "output_tokens"])
    sent = requests[requests["status"].isin(["ok", "api_error", "unparseable", "refused"])]
    sent = sent.reindex(columns=["cost_usd", "input_tokens", "output_tokens"])
    actual = float(pd.to_numeric(sent.get("cost_usd"), errors="coerce").fillna(0).sum())
    cost = estimate if settings.dry_run else actual
    footage_hours = sum(clip["last_pos"] for clip in assignments.values()) / 3600.0
    per_hour = cost / footage_hours if footage_hours else None
    cost_table = pd.DataFrame(
        [
            ("model_id", settings.model_id, ""),
            ("prompt_hash", prompt_hash(), ""),
            ("cost_basis", "estimated (dry run)" if settings.dry_run else "actual", ""),
            ("estimated_cost_usd", estimate, "before the run, no cache discount"),
            ("actual_cost_usd", None if settings.dry_run else actual, "from response usage"),
            ("requests", n_requests, ""),
            (
                "input_tokens",
                int(pd.to_numeric(sent.get("input_tokens"), errors="coerce").sum()),
                "",
            ),
            (
                "output_tokens",
                int(pd.to_numeric(sent.get("output_tokens"), errors="coerce").sum()),
                "",
            ),
            ("trips", len(assignments), ""),
            ("footage_hours", footage_hours, "whole outside-camera clips"),
            ("cost_per_footage_hour_usd", per_hour, ""),
            (
                "cost_per_revenue_hour_usd",
                per_hour,
                "the outside camera runs all the time, so one revenue hour ~ one footage hour",
            ),
        ],
        columns=["metric", "value", "note"],
    )
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{settings.output_prefix}_stage2_{settings.model_id}_{stamp}"
    if settings.dry_run:
        stem += "_DRYRUN"
    workbook_path = output_dir / f"{stem}.xlsx"
    write_workbook(
        workbook_path,
        [
            ("Detections", detections, []),
            ("Stop visits", visit_table, []),
            ("Scores", scores, score_lines),
            ("Cost", cost_table, []),
            ("Reconciliation", _recon_frame(recon), []),
        ],
    )
    logging.info("Stage 2 workbook: %s", workbook_path)
    statuses = requests["status"].value_counts().to_dict() if len(requests) else {}
    code = 0
    if not request_rows:
        logging.error("No window could be processed; see Reconciliation for the reasons.")
        code = 1
    if control.fatal_message:
        logging.error("The run stopped early: %s", control.fatal_message)
        code = 1
    if statuses.get("unparseable"):
        logging.error("%d answer(s) could not be parsed after retries.", statuses["unparseable"])
        code = 1
    if not settings.dry_run and not statuses.get("ok"):
        logging.error("No request returned a usable answer.")
        code = 1
    totals = {
        "trips": len(assignments),
        "windows": len(windows),
        "requests_by_status": statuses,
        "estimated_cost_usd": round(estimate, 4),
        "actual_cost_usd": round(actual, 4),
        "detections": len(detections),
        "visits_scored": len(visit_table),
        "workbook": workbook_path,
    }
    inputs = [paths["STOP_EVENTS_PATH"]] + _video_inputs(paths["VIDEO_DIR"], settings)
    gtfs = paths["GTFS_PATH"]
    inputs += [gtfs] if gtfs.is_file() else [gtfs / "stops.txt", gtfs / "stop_times.txt"]
    code = _finish(output_dir / f"{stem}_runlog.txt", settings, inputs, totals, code)
    if code == 0:
        logging.info("Stage 2 run completed successfully.")
    return code


# =============================================================================
# SETTINGS, CLIENT AND MAIN
# =============================================================================


def make_client(settings: Settings) -> Any:
    """Create the Anthropic client, retrying API errors with exponential backoff.

    Args:
        settings: Run settings.

    Returns:
        anthropic.Anthropic client.

    Raises:
        PilotError: The anthropic package is not installed.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise PilotError("The anthropic package is not installed: pip install anthropic") from exc
    return anthropic.Anthropic(max_retries=settings.max_api_retries)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; each flag's default is the matching CONFIGURATION value.

    Column names, the price table and the request-tuning values are set only in
    the CONFIGURATION section.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description="Pilot-test AI passenger counting on bus door-camera video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    flag = argparse.BooleanOptionalAction
    parser.add_argument("--mode", default=MODE, help="count, template, calibrate or stage2")
    parser.add_argument("--dry-run", action=flag, default=DRY_RUN, help="skip the API calls")
    parser.add_argument("--self-test", action=flag, default=SELF_TEST, help="run built-in tests")
    parser.add_argument("--video-dir", default=VIDEO_DIR)
    parser.add_argument("--stop-events-path", default=STOP_EVENTS_PATH)
    parser.add_argument("--ground-truth-source", default=GROUND_TRUTH_SOURCE)
    parser.add_argument("--manual-counts-csv", default=MANUAL_COUNTS_CSV)
    parser.add_argument("--ridecheck-path", default=RIDECHECK_PATH)
    parser.add_argument("--gtfs-path", default=GTFS_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--frames-dir", default=FRAMES_DIR, help="local folder only")
    parser.add_argument("--filename-regex", default=FILENAME_REGEX)
    parser.add_argument("--filename-time-format", default=FILENAME_TIME_FORMAT)
    parser.add_argument("--clock-offset-seconds", type=float, default=CLOCK_OFFSET_SECONDS)
    parser.add_argument("--trips", nargs="*", default=list(TRIPS_TO_INCLUDE), help="trip IDs")
    parser.add_argument("--pad-seconds", type=float, default=PAD_SECONDS)
    parser.add_argument("--frame-rate", type=float, default=FRAME_RATE)
    parser.add_argument("--frame-width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--keep-frames", action=flag, default=KEEP_FRAMES)
    parser.add_argument("--calibration-stills", type=int, default=CALIBRATION_STILLS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--effort", default=EFFORT)
    parser.add_argument("--request-token-ceiling", type=int, default=REQUEST_TOKEN_CEILING)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-concurrent-requests", type=int, default=MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--budget-usd", type=float, default=BUDGET_USD)
    parser.add_argument("--override-budget", action=flag, default=OVERRIDE_BUDGET)
    parser.add_argument("--stage2-frame-rate", type=float, default=STAGE2_FRAME_RATE)
    parser.add_argument("--stage2-window-seconds", type=float, default=STAGE2_WINDOW_SECONDS)
    parser.add_argument(
        "--stage2-match-tolerance-seconds", type=float, default=STAGE2_MATCH_TOLERANCE_SECONDS
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Combine parsed CLI flags with the CONFIGURATION-only values.

    Args:
        args: Parsed flags.

    Returns:
        Resolved settings.
    """
    return Settings(
        mode=str(args.mode).strip().lower(),
        dry_run=bool(args.dry_run),
        self_test=bool(args.self_test),
        video_dir=args.video_dir,
        stop_events_path=args.stop_events_path,
        ground_truth_source=str(args.ground_truth_source).strip().lower(),
        manual_counts_csv=args.manual_counts_csv,
        ridecheck_path=args.ridecheck_path,
        gtfs_path=args.gtfs_path,
        output_dir=args.output_dir,
        frames_dir=args.frames_dir,
        output_prefix=OUTPUT_PREFIX,
        template_prefix=TEMPLATE_PREFIX,
        require_run_log=REQUIRE_RUN_LOG,
        filename_regex=args.filename_regex,
        filename_time_format=args.filename_time_format,
        video_extensions=tuple(VIDEO_EXTENSIONS),
        camera_doors={
            str(k).strip().lower(): str(v).strip().lower() for k, v in CAMERA_DOORS.items()
        },
        outside_camera=str(OUTSIDE_CAMERA).strip().lower(),
        clock_offset_seconds=float(args.clock_offset_seconds),
        event_trip_col=EVENT_TRIP_COL,
        event_bus_col=EVENT_BUS_COL,
        event_stop_code_col=EVENT_STOP_CODE_COL,
        event_seq_col=EVENT_SEQ_COL,
        event_door_open_col=EVENT_DOOR_OPEN_COL,
        event_door_close_col=EVENT_DOOR_CLOSE_COL,
        event_arrival_col=EVENT_ARRIVAL_COL,
        event_time_format=EVENT_TIME_FORMAT,
        trips_to_include=tuple(args.trips or ()),
        ridecheck_trip_col=RIDECHECK_TRIP_COL,
        ridecheck_stop_col=RIDECHECK_STOP_COL,
        ridecheck_seq_col=RIDECHECK_SEQ_COL,
        ridecheck_front_on_col=RIDECHECK_FRONT_ON_COL,
        ridecheck_front_off_col=RIDECHECK_FRONT_OFF_COL,
        ridecheck_rear_on_col=RIDECHECK_REAR_ON_COL,
        ridecheck_rear_off_col=RIDECHECK_REAR_OFF_COL,
        pad_seconds=float(args.pad_seconds),
        frame_rate=float(args.frame_rate),
        frame_width=int(args.frame_width),
        frame_height=int(args.frame_height),
        jpeg_quality=JPEG_QUALITY,
        keep_frames=bool(args.keep_frames),
        calibration_stills=int(args.calibration_stills),
        model_id=str(args.model_id).strip(),
        effort=str(args.effort).strip().lower(),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        request_token_ceiling=int(args.request_token_ceiling),
        max_images_per_request=MAX_IMAGES_PER_REQUEST,
        text_chars_per_token=TEXT_CHARS_PER_TOKEN,
        request_overhead_tokens=REQUEST_OVERHEAD_TOKENS,
        cache_min_tokens=CACHE_MIN_TOKENS,
        repeats=int(args.repeats),
        max_api_retries=MAX_API_RETRIES,
        max_parse_retries=MAX_PARSE_RETRIES,
        max_consecutive_api_errors=MAX_CONSECUTIVE_API_ERRORS,
        max_concurrent_requests=int(args.max_concurrent_requests),
        model_prices=MODEL_PRICES,
        cache_write_multiplier=CACHE_WRITE_MULTIPLIER,
        est_output_tokens_per_request=EST_OUTPUT_TOKENS_PER_REQUEST,
        budget_usd=float(args.budget_usd),
        override_budget=bool(args.override_budget),
        door_open_share=DOOR_OPEN_SHARE,
        door_cameras_per_bus=DOOR_CAMERAS_PER_BUS,
        stage2_frame_rate=float(args.stage2_frame_rate),
        stage2_window_seconds=float(args.stage2_window_seconds),
        stage2_match_tolerance_seconds=float(args.stage2_match_tolerance_seconds),
        stage2_dedupe_seconds=STAGE2_DEDUPE_SECONDS,
        log_level=LOG_LEVEL,
    )


def config_problems(settings: Settings) -> List[str]:
    """List configuration errors, starting with paths still set to placeholders.

    Only the paths the chosen mode reads are checked (the simpler option: an
    unused RIDECHECK_PATH, say, may stay a placeholder).

    Args:
        settings: Run settings.

    Returns:
        Problems found; empty if the configuration is usable.
    """
    if settings.mode not in MODES:
        return [f"MODE must be one of {MODES}, not '{settings.mode}'."]
    needed = {"VIDEO_DIR": settings.video_dir, "STOP_EVENTS_PATH": settings.stop_events_path}
    if settings.mode in ("count", "template", "stage2"):
        needed["OUTPUT_DIR"] = settings.output_dir
    if settings.mode == "count":
        if settings.ground_truth_source == "manual":
            needed["MANUAL_COUNTS_CSV"] = settings.manual_counts_csv
        else:
            needed["RIDECHECK_PATH"] = settings.ridecheck_path
    if settings.mode == "stage2":
        needed["GTFS_PATH"] = settings.gtfs_path
    problems = [
        f"{name} is still a placeholder ({value!r}); set it in CONFIGURATION or by CLI flag."
        for name, value in needed.items()
        if _is_placeholder(value)
    ]
    if settings.ground_truth_source not in ("manual", "ridecheck"):
        problems.append("GROUND_TRUTH_SOURCE must be 'manual' or 'ridecheck'.")
    if settings.mode == "count" and settings.ground_truth_source == "ridecheck":
        doors = set(settings.camera_doors.values())
        if not doors <= {"front", "rear"}:
            problems.append("With Ridecheck ground truth, CAMERA_DOORS doors must be front/rear.")
    if settings.model_id not in settings.model_prices:
        problems.append(f"MODEL_ID '{settings.model_id}' has no entry in MODEL_PRICES.")
    if settings.effort not in EFFORT_LEVELS:
        problems.append(f"EFFORT must be one of {EFFORT_LEVELS}.")
    if settings.frame_rate <= 0 or settings.stage2_frame_rate <= 0:
        problems.append("FRAME_RATE and STAGE2_FRAME_RATE must be above 0.")
    if settings.frame_width < 1 or settings.frame_height < 1:
        problems.append("FRAME_WIDTH and FRAME_HEIGHT must be at least 1.")
    long_edge = max(settings.frame_width, settings.frame_height)
    if long_edge > MAX_IMAGE_LONG_EDGE or (
        image_tokens(settings.frame_width, settings.frame_height) > MAX_IMAGE_TOKENS
    ):
        problems.append(
            f"Frames larger than {MAX_IMAGE_LONG_EDGE} px or {MAX_IMAGE_TOKENS} tokens are "
            "downscaled by the API, which breaks the token estimate; use a smaller size."
        )
    if not 1 <= settings.jpeg_quality <= 100:
        problems.append("JPEG_QUALITY must be between 1 and 100.")
    if settings.pad_seconds < 0:
        problems.append("PAD_SECONDS must be 0 or more.")
    if settings.repeats < 1 or settings.max_concurrent_requests < 1:
        problems.append("REPEATS and MAX_CONCURRENT_REQUESTS must be at least 1.")
    if not 1 <= settings.max_images_per_request <= 600:
        problems.append("MAX_IMAGES_PER_REQUEST must be between 1 and 600.")
    if settings.request_token_ceiling < 1000:
        problems.append("REQUEST_TOKEN_CEILING must be at least 1,000.")
    if settings.calibration_stills < 1:
        problems.append("CALIBRATION_STILLS must be at least 1.")
    if settings.stage2_window_seconds <= 0:
        problems.append("STAGE2_WINDOW_SECONDS must be above 0.")
    if not settings.camera_doors:
        problems.append("CAMERA_DOORS must name at least one door camera.")
    try:
        groups = set(re.compile(settings.filename_regex).groupindex)
        if not {"bus", "camera", "start"} <= groups:
            problems.append("FILENAME_REGEX needs named groups bus, camera and start.")
    except re.error as exc:
        problems.append(f"FILENAME_REGEX is not a valid regular expression: {exc}")
    if settings.frames_dir.strip() and _is_network_path(settings.frames_dir):
        problems.append("FRAMES_DIR is on a network share; frames show riders, use a local disk.")
    return problems


def main(argv: Optional[Sequence[str]] = None, client: Any = None) -> int:
    """Run the pilot in the configured mode.

    Args:
        argv: CLI arguments; None reads sys.argv (or [] inside a notebook).
        client: Optional Anthropic client or test stub; created when needed.

    Returns:
        0 on success, 1 on a runtime failure (missing or empty inputs, nothing
        matched, over budget, unparseable answers), 2 on a configuration error
        (such as a placeholder path).
    """
    args = build_arg_parser().parse_args(notebook_safe_argv(argv))
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
    settings = settings_from_args(args)
    if settings.self_test:
        return run_self_test()
    problems = config_problems(settings)
    if problems:
        for problem in problems:
            logging.error("Configuration: %s", problem)
        return 2
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if settings.mode == "template":
            return run_template(settings, stamp)
        if settings.mode == "calibrate":
            return run_calibrate(settings, stamp)
        if settings.mode == "stage2":
            return run_stage2(settings, client, stamp)
        return run_count(settings, client, stamp)
    except PilotError as exc:
        logging.error("%s", exc)
        return 1


# =============================================================================
# SELF-TEST (synthetic clips and tables, stub API client)
# =============================================================================


class _StubBlock:
    """Content block of a stub response."""

    def __init__(self, block_type: str, text: str) -> None:
        """Store the block.

        Args:
            block_type: "text" or "thinking".
            text: Block text.
        """
        self.type = block_type
        self.text = text


class _StubUsage:
    """Usage counts of a stub response."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        """Store the counts.

        Args:
            input_tokens: Uncached input tokens.
            output_tokens: Output tokens.
        """
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class _StubResponse:
    """A Messages API response with a JSON text block."""

    def __init__(self, text: str, n_images: int) -> None:
        """Build the response.

        Args:
            text: Text of the answer.
            n_images: Images in the request, for plausible usage numbers.
        """
        self.content = [_StubBlock("thinking", ""), _StubBlock("text", text)]
        self.usage = _StubUsage(input_tokens=420 * n_images + 900, output_tokens=150)
        self.stop_reason = "end_turn"


class _StubMessages:
    """``client.messages`` of the stub: counts calls, hands each request to a handler."""

    def __init__(self, handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Store the handler.

        Args:
            handler: Function that turns request kwargs into a response.
        """
        self.handler = handler
        self.calls = 0
        self.requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> Any:
        """Record the request and return the handler's response.

        Args:
            **kwargs: Request keyword arguments.

        Returns:
            Stub response.
        """
        with self._lock:
            self.calls += 1
            self.requests.append(kwargs)
        return self.handler(kwargs)


class _StubClient:
    """Stands in for anthropic.Anthropic in the self-test."""

    def __init__(self, handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Create the stub.

        Args:
            handler: Function that turns request kwargs into a response.
        """
        self.messages = _StubMessages(handler)


class _Checks:
    """Collects self-test results."""

    def __init__(self) -> None:
        """Start with no results."""
        self.passed = 0
        self.failed: List[str] = []

    def check(self, name: str, condition: bool, detail: Any = "") -> None:
        """Record one check.

        Args:
            name: What is checked.
            condition: True if it passed.
            detail: Shown when it fails.
        """
        if condition:
            self.passed += 1
            logging.info("PASS  %s", name)
        else:
            self.failed.append(name)
            logging.error("FAIL  %s  %s", name, detail)

    def report(self) -> int:
        """Log the summary.

        Returns:
            0 if every check passed, else 1.
        """
        logging.info("Self-test: %d passed, %d failed.", self.passed, len(self.failed))
        for name in self.failed:
            logging.error("Failed: %s", name)
        return 0 if not self.failed else 1


# Known settings for the self-test, so it passes whatever the CONFIGURATION says.
_SELF_TEST_CONFIG: Dict[str, Any] = {
    "FILENAME_REGEX": r"^(?P<bus>\d+)_(?P<camera>[A-Za-z0-9]+)_(?P<start>\d{8}_\d{6})",
    "FILENAME_TIME_FORMAT": "%Y%m%d_%H%M%S",
    "VIDEO_EXTENSIONS": (".mp4", ".avi"),
    "CAMERA_DOORS": {"cam1": "front", "cam2": "rear"},
    "OUTSIDE_CAMERA": "cam3",
    "EVENT_TRIP_COL": "TRIP_ID",
    "EVENT_BUS_COL": "VEHICLE_ID",
    "EVENT_STOP_CODE_COL": "STOP_CODE",
    "EVENT_SEQ_COL": "STOP_SEQUENCE",
    "EVENT_DOOR_OPEN_COL": "ACT_DOOR_OPEN_TIME",
    "EVENT_DOOR_CLOSE_COL": "ACT_DOOR_CLOSE_TIME",
    "EVENT_ARRIVAL_COL": "ACT_ARRIVAL_TIME",
    "EVENT_TIME_FORMAT": "",
    "TRIPS_TO_INCLUDE": (),
    "RIDECHECK_TRIP_COL": "TRIP_ID",
    "RIDECHECK_STOP_COL": "STOP_ID",
    "RIDECHECK_SEQ_COL": "SEQ",
    "RIDECHECK_FRONT_ON_COL": "FRONT_ON",
    "RIDECHECK_FRONT_OFF_COL": "FRONT_OFF",
    "RIDECHECK_REAR_ON_COL": "REAR_ON",
    "RIDECHECK_REAR_OFF_COL": "REAR_OFF",
    "MODEL_PRICES": {
        "claude-haiku-5-5": {
            "input": 0.10,
            "output": 0.50,
            "cache_read": 0.01,
            "long_prompt_tokens": 100_000.0,
            "long_input": 0.50,
            "long_output": 2.50,
        }
    },
    "JPEG_QUALITY": 85,
    "TEXT_CHARS_PER_TOKEN": 3.0,
    "REQUEST_OVERHEAD_TOKENS": 100,
    "CACHE_MIN_TOKENS": 512,
    "MAX_IMAGES_PER_REQUEST": 600,
    "MAX_PARSE_RETRIES": 2,
    "MAX_CONSECUTIVE_API_ERRORS": 10,
    "EST_OUTPUT_TOKENS_PER_REQUEST": 1500,
    "CACHE_WRITE_MULTIPLIER": 1.25,
    "DOOR_OPEN_SHARE": 0.10,
    "DOOR_CAMERAS_PER_BUS": 2,
    "STAGE2_DEDUPE_SECONDS": 60.0,
    "REQUIRE_RUN_LOG": True,
    "OUTPUT_PREFIX": "video_count_pilot",
    "TEMPLATE_PREFIX": "manual_counts_template",
    "LOG_LEVEL": logging.INFO,
    "MODE": "count",
    "DRY_RUN": False,
    "SELF_TEST": False,
    "VIDEO_DIR": r"Path\To\Your\Video_Clips",
    "STOP_EVENTS_PATH": r"Path\To\Your\stop_visit_events.csv",
    "GROUND_TRUTH_SOURCE": "manual",
    "MANUAL_COUNTS_CSV": r"Path\To\Your\manual_counts.csv",
    "RIDECHECK_PATH": r"Path\To\Your\ridecheck_export.csv",
    "GTFS_PATH": r"Path\To\Your\GTFS",
    "OUTPUT_DIR": r"Path\To\Your\Output_Folder",
    "FRAMES_DIR": "",
    "CLOCK_OFFSET_SECONDS": 0.0,
    "PAD_SECONDS": 3.0,
    "FRAME_RATE": 2.0,
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 480,
    "KEEP_FRAMES": False,
    "CALIBRATION_STILLS": 10,
    "MODEL_ID": "claude-haiku-5-5",
    "EFFORT": "medium",
    "REQUEST_TOKEN_CEILING": 90_000,
    "REPEATS": 3,
    "MAX_CONCURRENT_REQUESTS": 4,
    "BUDGET_USD": 25.0,
    "OVERRIDE_BUDGET": False,
    "STAGE2_FRAME_RATE": 1.0,
    "STAGE2_WINDOW_SECONDS": 120.0,
    "STAGE2_MATCH_TOLERANCE_SECONDS": 60.0,
}


@contextmanager
def _temporary_config(values: Dict[str, Any]) -> Generator[None, None, None]:
    """Swap module-level CONFIGURATION values for the duration of a block.

    Args:
        values: Setting name -> temporary value.

    Yields:
        Nothing; the original values come back afterwards.
    """
    module_globals = globals()
    saved = {name: module_globals[name] for name in values}
    module_globals.update(values)
    try:
        yield
    finally:
        module_globals.update(saved)


def _write_test_clip(cv2: Any, path: Path, seconds: float, label: str) -> None:
    """Write a synthetic clip: a moving block and a burned-in timestamp, 10 fps.

    Args:
        cv2: The OpenCV module.
        path: Output .mp4 (falls back to MJPG .avi codec data if mp4v is missing).
        seconds: Length.
        label: Text burned into each frame.

    Raises:
        RuntimeError: OpenCV can't write video here.
    """
    import numpy as np

    fps, width, height = 10, 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write test video {path}")
    for number in range(int(seconds * fps)):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        x = (number * 3) % (width - 40)
        cv2.rectangle(image, (x, 90), (x + 30, 200), (40, 200, 40), -1)
        stamp = f"{label} t={number / fps:6.1f}s"
        cv2.putText(image, stamp, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(image)
    writer.release()


def _build_synthetic_inputs(cv2: Any, root: Path) -> Dict[str, Path]:
    """Write the synthetic clips, stop events, manual counts, Ridecheck file and GTFS.

    Timeline (recorder start 07:00:00, CLOCK_OFFSET_SECONDS = 2, so the clips run
    07:00:02-07:01:41.9 in AVL time): trip T1 on bus 1234 has door events at seq 1
    (stop 1001), 2 (1002, long, split), 4 (1234, ground truth keyed "1234.0"),
    5 (outside the clips), 6 (half outside), 7 (no ground truth), 8 (bad time),
    and seq 3 passed without opening. Trip T2 (bus 4321) has no video.

    Args:
        cv2: The OpenCV module.
        root: Temporary folder.

    Returns:
        Paths of the synthetic inputs.
    """
    video_dir = root / "videos"
    video_dir.mkdir()
    for camera in ("cam1", "cam2", "cam3"):
        _write_test_clip(cv2, video_dir / f"1234_{camera}_20260302_070000.mp4", 100, camera)
    _write_test_clip(cv2, video_dir / "5678_cam1_20260302_080000.mp4", 20, "no events")
    (video_dir / "badname.mp4").write_bytes(b"")
    (video_dir / "notes.txt").write_text("not a video", encoding="utf-8")
    day = "2026-03-02 "
    visit_rows = [
        ("T1", "1234", "1001", "1", "07:00:10", "07:00:20", "07:00:10"),
        ("T1", "1234", "1002", "2", "07:00:40", "07:01:20", "07:00:40"),
        ("T1", "1234", "1003", "3", "", "", "07:00:55"),
        ("T1", "1234", "1234", "4", "07:01:30", "07:01:35", "07:01:30"),
        ("T1", "1234", "1005", "5", "07:05:00", "07:05:10", "07:05:00"),
        ("T1", "1234", "1006", "6", "07:01:38", "07:01:50", "07:01:38"),
        ("T1", "1234", "1007", "7", "07:00:25", "07:00:30", "07:00:25"),
        ("T1", "1234", "1008", "8", "not a time", "07:00:31", ""),
        ("T2", "4321", "2001", "1", "08:00:10", "08:00:20", "08:00:10"),
    ]
    events = pd.DataFrame(
        [
            {
                "TRIP_ID": trip,
                "VEHICLE_ID": bus,
                "STOP_CODE": stop,
                "STOP_SEQUENCE": seq,
                "SCHED_TIME": "1899-12-30 07:00:00",
                "ACT_DOOR_OPEN_TIME": (day + o) if o and ":" in o else o,
                "ACT_DOOR_CLOSE_TIME": (day + c) if c else "",
                "ACT_ARRIVAL_TIME": (day + a) if a else "",
            }
            for trip, bus, stop, seq, o, c, a in visit_rows
        ]
    )
    events_path = root / "stop_visit_events.csv"
    events.to_csv(events_path, index=False)
    empty_events = root / "empty_events.csv"
    events.head(0).to_csv(empty_events, index=False)
    manual = pd.DataFrame(
        [
            ("T1", "1001", "1", "front", "2", "1"),
            ("T1", "1001", "1", "rear", "0", "1"),
            ("T1", "1002", "2", "front", "3", "0"),
            ("T1", "1002", "2", "rear", "1", "2"),
            ("T1", "1234.0", "4.0", "front", "4", "0"),
            ("T1", "1234.0", "4.0", "rear", "0", "0"),
            ("T1", "1234.0", "4.0", "rear", "9", "9"),
            ("T1", "9999", "9", "front", "1", "1"),
            ("T1", "1007", "7", "front", "", ""),
        ],
        columns=["trip", "stop_code", "stop_sequence", "door", "boardings", "alightings"],
    )
    manual_path = root / "manual_counts.csv"
    manual.to_csv(manual_path, index=False)
    ridecheck = pd.DataFrame(
        [
            {
                "TRIP_ID": "T1",
                "STOP_ID": "1234.0",
                "SEQ": "4.0",
                "FRONT_ON": "4.0",
                "FRONT_OFF": "0",
                "REAR_ON": "0",
                "REAR_OFF": "0",
            },
        ]
    )
    ridecheck_path = root / "ridecheck.csv"
    ridecheck.to_csv(ridecheck_path, index=False)
    gtfs_dir = root / "gtfs"
    gtfs_dir.mkdir()
    codes = ["1001", "1002", "1003", "1234", "1005", "1006", "1007"]
    pd.DataFrame(
        {
            "stop_id": [f"S{c}" for c in codes],
            "stop_code": codes,
            "stop_name": [f"Main St & {c} Ave" for c in codes],
        }
    ).to_csv(gtfs_dir / "stops.txt", index=False)
    pd.DataFrame(
        {
            "trip_id": ["T1"] * len(codes),
            "stop_id": [f"S{c}" for c in codes],
            "stop_sequence": [str(i) for i in range(1, len(codes) + 1)],
        }
    ).to_csv(gtfs_dir / "stop_times.txt", index=False)
    return {
        "video_dir": video_dir,
        "events": events_path,
        "empty_events": empty_events,
        "manual": manual_path,
        "ridecheck": ridecheck_path,
        "gtfs": gtfs_dir,
    }


def _count_handler(bad_json: bool) -> Callable[[Dict[str, Any]], Any]:
    """Return a stub handler with known Stage 1 answers.

    1001 front: 2/1 on repeats 1-2, 3/1 on repeat 3. 1001 rear: API error.
    1002: split into parts; the parts add up to the ground truth. 1234 rear:
    invalid JSON when *bad_json*. Everything else: the ground truth.

    Args:
        bad_json: Whether 1234 rear answers with invalid JSON.

    Returns:
        Handler for :class:`_StubClient`.
    """
    truth = {
        ("1001", "front"): (2, 1),
        ("1002", "rear"): (1, 2),
        ("1234", "front"): (4, 0),
        ("1234", "rear"): (0, 0),
    }
    seen: Dict[Tuple[str, str], int] = {}
    lock = threading.Lock()

    def handler(kwargs: Dict[str, Any]) -> Any:
        content = kwargs["messages"][0]["content"]
        header = re.search(r"stop (\S+) \(stop sequence (\S+)\), (\S+) door", content[0]["text"])
        stop, door = (header.group(1), header.group(3)) if header else ("?", "?")
        part = re.search(r"part (\d+) of (\d+)", content[1]["text"])
        n_images = sum(1 for block in content if block["type"] == "image")
        with lock:
            seen[(stop, door)] = seen.get((stop, door), 0) + 1
            call = seen[(stop, door)]
        if stop == "1001" and door == "rear":
            raise RuntimeError("simulated API failure")
        if stop == "1234" and door == "rear" and bad_json:
            return _StubResponse("not json {", n_images)
        if stop == "1002" and door == "front":
            answer = (1, 0)
        elif stop == "1002" and door == "rear":
            answer = truth[(stop, door)] if part is None or part.group(1) == "1" else (0, 0)
        elif stop == "1001" and door == "front":
            answer = (2, 1) if call % 3 else (3, 1)
        else:
            answer = truth.get((stop, door), (0, 0))
        text = json.dumps(
            {"boardings": answer[0], "alightings": answer[1], "confidence": "high", "notes": "none"}
        )
        return _StubResponse(text, n_images)

    return handler


def _stage2_handler(kwargs: Dict[str, Any]) -> Any:
    """Return known Stage 2 detections for the synthetic outside clip.

    Args:
        kwargs: Request keyword arguments.

    Returns:
        Stub response.
    """
    content = kwargs["messages"][0]["content"]
    labels = [
        float(m.group(1))
        for b in content
        if b["type"] == "text"
        for m in [re.fullmatch(r"t = (\d+) s", b["text"])]
        if m
    ]
    n_images = sum(1 for block in content if block["type"] == "image")
    if labels and labels[0] < 60:
        stops = [
            ("1001", 8, "stopped"),
            ("1002", 39, "stopped"),
            ("1002", 41, "stopped"),
            ("9999", 50, "passed"),
            ("1003", 55, "passed"),
        ]
    else:
        stops = [("1234", 88, "stopped"), ("1005", 96, "stopped")]
    answer = {
        "stops": [{"stop_code": c, "frame_time_s": t, "action": a} for c, t, a in stops],
        "notes": "none",
    }
    return _StubResponse(json.dumps(answer), n_images)


def _latest(folder: Path, pattern: str) -> Optional[Path]:
    """Return the newest file in *folder* matching *pattern*.

    Args:
        folder: Folder to search.
        pattern: Glob pattern.

    Returns:
        Path, or None.
    """
    found = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def _test_units(checks: _Checks, cv2: Any, data: Dict[str, Path]) -> None:
    """Unit checks: names, windows, frames, chunks, tokens, IDs, prices, argv, paths.

    Args:
        checks: Result collector.
        cv2: The OpenCV module.
        data: Synthetic input paths.
    """
    pattern = re.compile(_SELF_TEST_CONFIG["FILENAME_REGEX"])
    parsed = parse_video_name("01234_CAM1_20260302_070000.mp4", pattern, "%Y%m%d_%H%M%S")
    checks.check(
        "filename parsing: bus, camera, start",
        parsed == {"bus": "1234", "camera": "cam1", "start": _dt.datetime(2026, 3, 2, 7, 0, 0)},
        parsed,
    )
    checks.check(
        "filename parsing: bad name -> None", parse_video_name("x.mp4", pattern, "%Y") is None
    )
    checks.check(
        "filename parsing: bad date -> None",
        parse_video_name("1234_cam1_20261302_070000.mp4", pattern, "%Y%m%d_%H%M%S") is None,
    )
    start_avl = _dt.datetime(2026, 3, 2, 7, 0, 0) + _dt.timedelta(seconds=2.0)
    t = _dt.datetime(2026, 3, 2, 7, 0, 10)
    window = door_window_positions(t, t + _dt.timedelta(seconds=10), start_avl, 99.9, 3.0)
    checks.check(
        "window + clock offset: full window at 5-21 s", window == ("full", 5.0, 21.0), window
    )
    late = door_window_positions(
        _dt.datetime(2026, 3, 2, 7, 1, 38), _dt.datetime(2026, 3, 2, 7, 1, 50), start_avl, 99.9, 3.0
    )
    checks.check("window: partial coverage detected", late[0] == "partial", late)
    early = door_window_positions(t, t + _dt.timedelta(seconds=5), start_avl, 99.9, 9.0)
    checks.check("window: padding cut at clip start", early[1] == 0.0, early)
    gone = door_window_positions(
        _dt.datetime(2026, 3, 2, 7, 5), _dt.datetime(2026, 3, 2, 7, 5, 10), start_avl, 99.9, 3.0
    )
    checks.check("window: no coverage detected", gone[0] == "none", gone)
    times = frame_times(5.0, 21.0, 2.0)
    checks.check(
        "frame count: 16 s at 2 fps = 33 frames",
        len(times) == 33 and times[0] == 5.0 and times[-1] == 21.0,
        len(times),
    )
    clip = data["video_dir"] / "1234_cam1_20260302_070000.mp4"
    jpegs = extract_frames(cv2, clip, times, 10.0, 1000, 640, 480, 85)
    import numpy as np

    first = (
        cv2.imdecode(np.frombuffer(jpegs[0], dtype=np.uint8), cv2.IMREAD_COLOR) if jpegs else None
    )
    checks.check("frame extraction: 33 JPEGs", len(jpegs) == 33, len(jpegs))
    checks.check(
        "frame extraction: resized to 640x480",
        first is not None and tuple(first.shape) == (480, 640, 3),
        None if first is None else first.shape,
    )
    checks.check("token estimate: 640x480 = 23 x 18 = 414", image_tokens(640, 480) == 414)
    checks.check(
        "token estimate: 28x28 = 1, 29x28 = 2",
        (image_tokens(28, 28), image_tokens(29, 28)) == (1, 2),
    )
    chunks = plan_chunks(93, 420, 1000, 20000, 600)
    checks.check(
        "chunk splitting: 93 frames under a 20k ceiling -> 3 x 31",
        chunks == [(0, 31), (31, 62), (62, 93)],
        chunks,
    )
    capped = plan_chunks(10, 420, 1000, 90000, 4)
    checks.check(
        "chunk splitting: image limit 4 -> 4/3/3", [e - s for s, e in capped] == [4, 3, 3], capped
    )
    checks.check(
        "chunk splitting: fits in one request", plan_chunks(33, 420, 1000, 20000, 600) == [(0, 33)]
    )
    try:
        plan_chunks(5, 420, 1000, 1200, 600)
        raised = False
    except PilotError:
        raised = True
    checks.check("chunk splitting: ceiling too low raises", raised)
    checks.check(
        "_clean_id: float-converted and padded IDs",
        [_clean_id(v) for v in ("1234.0", " 77 ", 1234.0, "A12.0B", "", None, "0012")]
        == ["1234", "77", "1234", "A12.0B", "", "", "0012"],
    )
    haiku = _SELF_TEST_CONFIG["MODEL_PRICES"]["claude-haiku-5-5"]
    small = usage_cost(
        {"input": 50_000, "output": 1000, "cache_write": 0, "cache_read": 0}, haiku, 1.25
    )
    large = usage_cost(
        {"input": 150_000, "output": 1000, "cache_write": 0, "cache_read": 0}, haiku, 1.25
    )
    checks.check("price tier: Haiku <=100k tokens", abs(small - (0.005 + 0.0005)) < 1e-9, small)
    checks.check("price tier: Haiku >100k tokens", abs(large - (0.075 + 0.0025)) < 1e-9, large)
    cached = usage_cost(
        {"input": 0, "output": 0, "cache_write": 1000, "cache_read": 10_000}, haiku, 1.25
    )
    checks.check("price: cache write and read", abs(cached - (0.000125 + 0.0001)) < 1e-12, cached)
    had_kernel = "ipykernel" in sys.modules
    if not had_kernel:
        sys.modules["ipykernel"] = type(sys)("ipykernel")
    try:
        checks.check("notebook argv: kernel flags ignored", notebook_safe_argv(None) == [])
    finally:
        if not had_kernel:
            del sys.modules["ipykernel"]
    checks.check("network path: UNC share refused", _is_network_path(r"\\server\share\frames"))
    checks.check("network path: local temp allowed", not _is_network_path(tempfile.gettempdir()))
    checks.check(
        "placeholder detection",
        _is_placeholder(r"Path\To\Your\x.csv") and not _is_placeholder("x.csv"),
    )
    settings = settings_from_args(build_arg_parser().parse_args([]))
    counts, recon = load_ridecheck(data["ridecheck"], settings)
    checks.check(
        "Ridecheck join keys: float IDs cleaned",
        counts.get(("T1", "1234", "4", "front")) == (4, 0) and not recon,
        counts,
    )
    try:
        source = Path(__file__).read_text(encoding="utf-8")
        ast.parse(source)
        parsed_ok = True
    except NameError:
        parsed_ok = True  # pasted into a notebook: the cell already ran, so it parsed
    except SyntaxError:
        parsed_ok = False
    checks.check("script source parses (ast.parse)", parsed_ok)


def _test_end_to_end(checks: _Checks, data: Dict[str, Path], root: Path) -> None:
    """End-to-end checks of main(): exit codes, budget stop, sheets, joins, formatting.

    Args:
        checks: Result collector.
        data: Synthetic input paths.
        root: Temporary folder.
    """
    frames_dir = root / "frames"

    def argv(out: Path, *extra: str) -> List[str]:
        return [
            "--video-dir",
            str(data["video_dir"]),
            "--stop-events-path",
            str(data["events"]),
            "--manual-counts-csv",
            str(data["manual"]),
            "--output-dir",
            str(out),
            "--frames-dir",
            str(frames_dir),
            "--clock-offset-seconds",
            "2",
            "--request-token-ceiling",
            "20000",
            *extra,
        ]

    checks.check("placeholder paths -> exit 2", main(argv=[]) == 2)

    stub = _StubClient(_count_handler(bad_json=False))
    code = main(argv(root / "budget", "--budget-usd", "0.00001"), client=stub)
    checks.check(
        "over budget -> exit 1, no API calls", code == 1 and stub.messages.calls == 0, code
    )

    code = main(
        argv(root / "empty") + ["--stop-events-path", str(data["empty_events"])], client=stub
    )
    checks.check("empty stop-events file -> exit 1", code == 1, code)
    empty_videos = root / "no_videos"
    empty_videos.mkdir()
    code = main(argv(root / "empty") + ["--video-dir", str(empty_videos)], client=stub)
    checks.check("empty video folder -> exit 1", code == 1, code)
    code = main(
        argv(root / "empty") + ["--manual-counts-csv", str(root / "missing.csv")], client=stub
    )
    checks.check("missing ground-truth file -> exit 1", code == 1, code)
    code = main(argv(root / "empty", "--trips", "T2"), client=stub)
    checks.check("zero events matched -> exit 1", code == 1, code)

    code = main(argv(root / "dry", "--dry-run"), client=stub)
    dry_book = _latest(root / "dry", "*_DRYRUN.xlsx")
    checks.check("dry run -> exit 0, no API calls", code == 0 and stub.messages.calls == 0, code)
    if dry_book is not None:
        dry_events = pd.read_excel(dry_book, sheet_name="Events")
        checks.check(
            "dry run: 18 planned rows with estimated cost",
            len(dry_events) == 18
            and set(dry_events["status"]) == {"dry_run"}
            and dry_events["est_cost_usd"].gt(0).all(),
            len(dry_events),
        )
    else:
        checks.check("dry run: workbook written", False)

    stub = _StubClient(_count_handler(bad_json=True))
    code = main(argv(root / "run_a"), client=stub)
    book = _latest(root / "run_a", "*.xlsx")
    checks.check(
        "unparseable answers -> exit 1, results still written", code == 1 and book is not None, code
    )
    checks.check(
        "API calls: 36 (splits, parse retries, repeats)",
        stub.messages.calls == 36,
        stub.messages.calls,
    )
    first_request = stub.messages.requests[0]
    checks.check("request: no temperature sent", "temperature" not in first_request)
    checks.check(
        "request: instructions marked for caching",
        isinstance(first_request["system"], list)
        and first_request["system"][0].get("cache_control") == {"type": "ephemeral"},
    )
    checks.check(
        "request: structured output schema",
        first_request["output_config"]["format"]["type"] == "json_schema",
    )
    labels = [b["text"] for b in first_request["messages"][0]["content"] if b["type"] == "text"]
    checks.check(
        "request: frame labels give offsets", "t = 0.0 s" in labels and "t = 0.5 s" in labels
    )
    if book is None:
        return
    sheets = pd.read_excel(book, sheet_name=None, dtype={"stop_code": str, "stop_sequence": str})
    checks.check(
        "workbook sheets",
        list(sheets)
        == ["Events", "Trip summary", "Accuracy", "Repeatability", "Cost", "Reconciliation"],
        list(sheets),
    )
    events = sheets["Events"]
    checks.check("Events: 6 clips x 3 repeats = 18 rows", len(events) == 18, len(events))
    status = events["status"].value_counts().to_dict()
    checks.check(
        "Events: 12 ok, 3 api_error, 3 unparseable",
        status == {"ok": 12, "api_error": 3, "unparseable": 3},
        status,
    )
    joined = events[(events["stop_code"] == "1234") & (events["door"] == "front")]
    checks.check(
        "ground-truth join with '1234.0' / '4.0' IDs",
        len(joined) == 3
        and joined["gt_boardings"].eq(4).all()
        and joined["model_boardings"].eq(4).all(),
        joined[["gt_boardings", "model_boardings"]].to_dict("list"),
    )
    split = events[(events["stop_code"] == "1002")]
    checks.check(
        "split events flagged, chunk counts added up",
        split["split_flag"].eq("Y").all()
        and split["chunks"].eq(3).all()
        and split.loc[split["door"] == "front", "model_boardings"].eq(3).all(),
        split[["door", "chunks", "model_boardings"]].to_dict("list"),
    )
    recon_categories = set(sheets["Reconciliation"]["category"])
    expected = {
        "event_no_video",
        "video_no_events",
        "video_name_unparsed",
        "event_partial_video",
        "event_no_ground_truth",
        "ground_truth_no_event",
        "ground_truth_duplicate",
        "bad_door_times",
        "trip_no_video",
        "api_failure",
        "unparseable_response",
    }
    checks.check(
        "reconciliation row for every missing-data case",
        expected <= recon_categories,
        sorted(expected - recon_categories),
    )
    accuracy = sheets["Accuracy"]
    overall = accuracy[(accuracy["metric"] == "boardings") & (accuracy["breakdown"] == "overall")]
    checks.check(
        "Accuracy: overall boardings exact share 11/12, bias +1/12",
        len(overall) == 1
        and abs(float(overall["exact_share"].iloc[0]) - 11 / 12) < 1e-9
        and abs(float(overall["mean_signed_error"].iloc[0]) - 1 / 12) < 1e-9,
        overall.to_dict("records"),
    )
    raw_repeat = pd.read_excel(book, sheet_name="Repeatability", header=None)
    checks.check(
        "Repeatability: 75.0% of 4 events agreed",
        "75.0% of 4 events" in str(raw_repeat.iloc[1, 0]),
        raw_repeat.iloc[1, 0],
    )
    trip = sheets["Trip summary"]
    all_doors = trip[(trip["trip"] == "T1") & (trip["door"] == "all doors")]
    checks.check(
        "Trip summary: running-load gap 0.33 at stop 1001",
        len(all_doors) == 1
        and abs(float(all_doors["max_load_diff"].iloc[0]) - 1 / 3) < 1e-9
        and _clean_id(all_doors["max_load_diff_stop_code"].iloc[0]) == "1001",
        all_doors.to_dict("records"),
    )
    workbook = load_workbook(book)
    fonts_ok = all(
        cell.font.name == "Arial"
        for sheet in workbook.worksheets
        for row in sheet.iter_rows()
        for cell in row
        if cell.value is not None
    )
    headers_bold = all(cell.font.bold for cell in workbook["Events"][1])
    checks.check("Excel: Arial font throughout", fonts_ok)
    checks.check("Excel: bold headers", headers_bold)
    checks.check("Excel: freeze panes below header", workbook["Events"].freeze_panes == "A2")
    checks.check(
        "Excel: auto-filters on every sheet",
        all(sheet.auto_filter.ref for sheet in workbook.worksheets),
        [sheet.auto_filter.ref for sheet in workbook.worksheets],
    )
    checks.check(
        "Excel: Repeatability header below summary", workbook["Repeatability"].freeze_panes == "A5"
    )
    log = book.with_name(book.stem + "_runlog.txt")
    log_text = log.read_text(encoding="utf-8") if log.exists() else ""
    checks.check(
        "run log: config, parameters, fingerprints, model, prompt hash, totals",
        all(
            part in log_text
            for part in (
                "=== CONFIGURATION (verbatim) ===",
                "MODE: str =",
                "request_token_ceiling = 20000",
                "INPUT FINGERPRINTS (SHA-256)",
                "1234_cam1_20260302_070000.mp4",
                "model_id = claude-haiku-5-5",
                f"prompt_sha256 = {prompt_hash()}",
                "event_rows_by_status",
            )
        ),
    )
    checks.check(
        "frames not written to disk unless KEEP_FRAMES",
        not (frames_dir.exists() and any(frames_dir.rglob("*.jpg"))),
    )

    stub = _StubClient(_count_handler(bad_json=False))
    code = main(argv(root / "run_b", "--keep-frames", "--repeats", "1"), client=stub)
    checks.check("API failures recorded but not fatal -> exit 0", code == 0, code)
    checks.check("KEEP_FRAMES: JPEGs kept in local frames folder", any(frames_dir.rglob("*.jpg")))

    code = main(argv(root / "template", "--mode", "template"))
    template = _latest(root / "template", "manual_counts_template_*.csv")
    rows = pd.read_csv(template, dtype=str) if template is not None else pd.DataFrame()
    checks.check(
        "template: 4 events x 2 doors with clip windows",
        code == 0 and len(rows) == 8 and {"window_start_s", "boardings"} <= set(rows.columns),
        len(rows),
    )
    code = main(argv(root / "calib", "--mode", "calibrate", "--calibration-stills", "2"))
    stills = list(frames_dir.glob("calibration_*/*.jpg"))
    checks.check(
        "calibration: 2 events x 2 doors = 4 stills", code == 0 and len(stills) == 4, len(stills)
    )


def _test_stage2(checks: _Checks, data: Dict[str, Path], root: Path) -> None:
    """Stage 2 checks: detections deduplicated and scored against stop visits.

    Args:
        checks: Result collector.
        data: Synthetic input paths.
        root: Temporary folder.
    """
    stub = _StubClient(_stage2_handler)
    out = root / "stage2"
    code = main(
        [
            "--mode",
            "stage2",
            "--video-dir",
            str(data["video_dir"]),
            "--stop-events-path",
            str(data["events"]),
            "--gtfs-path",
            str(data["gtfs"]),
            "--output-dir",
            str(out),
            "--clock-offset-seconds",
            "2",
            "--stage2-window-seconds",
            "60",
            "--stage2-match-tolerance-seconds",
            "20",
        ],
        client=stub,
    )
    book = _latest(out, "*.xlsx")
    checks.check("stage 2: exit 0, 2 window requests", code == 0 and stub.messages.calls == 2, code)
    if book is None:
        return
    scores = pd.read_excel(book, sheet_name="Scores", header=4).set_index("group")
    expected = {
        "all visits": (7, 5 / 7, 4 / 7, 0.5, 1.7),
        "doors opened": (6, 4 / 6, 3 / 6, 0.0, 0.8),
        "passed without opening": (1, 1.0, 1.0, 2.0, 2.0),
    }
    for group, (n, detected, correct, median, p90) in expected.items():
        row = scores.loc[group] if group in scores.index else None
        ok = row is not None and int(row["visits"]) == n
        ok = ok and abs(row["detected_share"] - detected) < 1e-9
        ok = ok and abs(row["correct_stop_share"] - correct) < 1e-9
        ok = ok and abs(row["median_abs_time_error_s"] - median) < 1e-9
        ok = ok and abs(row["p90_abs_time_error_s"] - p90) < 1e-9
        checks.check(f"stage 2 scores: {group}", ok, None if row is None else row.to_dict())
    detections = pd.read_excel(book, sheet_name="Detections", dtype={"stop_code": str})
    match = dict(
        zip(
            detections["stop_code"] + "@" + detections["frame_time_s"].map(lambda v: f"{v:g}"),
            detections["match"],
        )
    )
    checks.check(
        "stage 2: duplicate, false and wrong-stop detections labelled",
        match.get("1002@41") == "duplicate"
        and match.get("9999@50") == "false_detection"
        and match.get("1005@96") == "wrong_stop",
        match,
    )
    raw = pd.read_excel(book, sheet_name="Scores", header=None)
    checks.check(
        "stage 2: 1 false detection reported", "1 (16.7%)" in str(raw.iloc[1, 0]), raw.iloc[1, 0]
    )


def run_self_test() -> int:
    """Run the built-in tests on synthetic clips and tables, with a stub API client.

    Returns:
        0 if every check passed, else 1.
    """
    try:
        cv2 = _import_cv2()
    except PilotError as exc:
        logging.error("%s", exc)
        return 1
    checks = _Checks()
    with _temporary_config(_SELF_TEST_CONFIG):
        with tempfile.TemporaryDirectory(prefix="video_count_selftest_") as tmp:
            root = Path(tmp)
            data = _build_synthetic_inputs(cv2, root)
            _test_units(checks, cv2, data)
            _test_end_to_end(checks, data, root)
            _test_stage2(checks, data, root)
    return checks.report()


if __name__ == "__main__":
    # In a notebook (ipykernel loaded) main() is called directly, so a failure shows
    # as a logged error and a return value instead of a SystemExit traceback. From a
    # shell, the exit code goes to sys.exit: 1 for failures, 2 for config errors.
    if "ipykernel" in sys.modules:
        main()
    else:
        sys.exit(main())
