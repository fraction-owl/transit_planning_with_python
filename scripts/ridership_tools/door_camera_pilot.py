"""Detect door cycles and count boardings/alightings locally on fixed bus cameras.

Install in a cloned ArcGIS Pro environment with pandas 1.x and openpyxl:
    python -m pip install "opencv-python>=4.8,<4.12" "numpy<2"
For passenger counts, place OpenCV Zoo's object_detection_yolox_2022nov.onnx
(FP32 YOLOX-S) on local disk. The program makes no network requests.
Model card: https://huggingface.co/opencv/object_detection_yolox

Use MODE="template" for an independent manual inventory, "calibrate" for an
on-screen check of the two reference frames and gate orientation, "doors" to
scan without weights, or "pilot" for both stages. CLI: --mode MODE.
Inside a notebook paste the whole script in one cell or use %run.

Reference seconds and truth seconds are relative to each file's first frame.
Rectangles use normalized (left, top, right, bottom); gate endpoints use
normalized (x, y), with origin at the upper left. The example gate runs left
to right: positive is below it. Choose the side in the bus for inside_sign.
Select a moving part of the door visible across stops, preferably above riders
and away from glass. Give each distinct camera mounting its own profile.
A profile named BUS:camera overrides the camera-wide profile.

Manual validation: watch the whole clip independently. Copy its template row
for each door cycle, set kind=event and supply open_s and close_s. Count fields
may be blank for door-only validation. For a fully reviewed clip with no cycles
use kind=none. Remove unreviewed rows or leave kind blank. The entire clip must
be reviewed before marking any events. Stop codes are optional manual labels.
Enter player times as seconds, mm:ss or hh:mm:ss. For a reviewed negative clip,
explicitly enter 0 in each count field you reviewed to score false alarms.
The manual counts and the external Ridecheck comparator are independent.

Ridecheck: defaults accept one row per clip, stop_code, sequence and door.
For shared visits, enter the same unique visit_id for every reopening and camera
in the manual sheet, then map visit_id (instead of clip) in RIDECHECK_COLUMNS.
Alternatively map service_date, bus, trip, stop_code and stop_sequence on both
tables. Enter the operating date explicitly for after-midnight service. Keys
match on YYYY-MM-DD dates (slash dates read month first) and on numeric labels
without leading zeros, so 0123 matches 123. The Ridecheck table is checked
before any clip is processed. For vehicle totals use a shared visit key and
omit the door mapping. Supply a
kind=no_cycle row with visit labels for an observed door that stayed closed.
For wide door columns, set RIDECHECK_WIDE to the actual board/alight headers
for each door or door group, and supply only visit keys in RIDECHECK_COLUMNS.
Door group values are configured in RIDECHECK_GROUPS. A vehicle or grouped
comparison requires explicit coverage of all its configured doors. Additional
unassigned detections with counts, missing coverage, or incomplete footage are
flagged and left unscored. APC comparator differences measure agreement.

Validation summaries exclude calibration-source clips and partial footage.
A door ROI needs visible spatial detail in both states. Check on independent
day/night clips. A stable but different mounting requires a separate profile.

Count semantics: a track must reach stable opposite sides of the finite gate.
Multiple crossings by the same surviving track cancel in pairs within a cycle.
Track breaks and operator movements require review. Door boundaries correspond
to confirmed visual state changes, backdated to the first agreeing sample.
Input must be constant-frame-rate video at its original speed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

# === BEGIN CONFIG ===
VIDEO_DIR: Path = Path("EDIT_ME/videos")  # One folder of trip-door videos.
OUTPUT_DIR: Path = Path("EDIT_ME/output")  # Tables and run logs; no frames are saved.
MODEL_PATH: Path = Path(
    "EDIT_ME/object_detection_yolox_2022nov.onnx"
)  # Local FP32 weights.
TRUTH_PATH: str = ""  # Optional independent cycle inventory, CSV or XLSX.
RIDECHECK_PATH: str = ""  # Optional door counts; requires the manual cycle inventory.
RIDECHECK_COLUMNS: dict = {
    "clip": "clip",
    "stop_code": "STOP_ID",
    "stop_sequence": "stop_sequence",
    "door": "door",
    "boardings": "boardings",
    "alightings": "alightings",
}  # Canonical key -> input column.
RIDECHECK_WIDE: dict = {}  # Example {"front": ("boarding_1","alighting_1"), "other": ("boarding_2","alighting_2")}; verify export headers.
RIDECHECK_GROUPS: dict = {
    "front": ("front",),
    "rear": ("rear",),
}  # For pooled other doors, e.g. "other": ("rear","middle").
VIDEO_EXTENSIONS: tuple = (
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".ts",
)  # Recorder exports accepted.
MODE: str = "doors"  # template, calibrate, doors, pilot.
FILENAME_REGEX: str = r"(?P<bus>[^_]+)_(?P<camera>front|rear)_(?P<start>\d{8}_\d{6})\.[^.]+$"  # Example 0123_front_20261103_080000.mp4; optional named trip.
START_FORMAT: str = "%Y%m%d_%H%M%S"  # Local recorder time.
PROFILES: dict = {
    "front": {
        "reference": "EDIT_ME/front.mp4",
        "closed_s": 1.0,
        "open_s": 6.0,
        "door_roi": (0.15, 0.10, 0.35, 0.28),
        "gate": ((0.12, 0.55), (0.88, 0.55)),
        "inside_sign": 1,
    },
    "rear": {
        "reference": "EDIT_ME/rear.mp4",
        "closed_s": 1.0,
        "open_s": 6.0,
        "door_roi": (0.15, 0.10, 0.35, 0.28),
        "gate": ((0.12, 0.55), (0.88, 0.55)),
        "inside_sign": 1,
    },
}  # Example geometry only; reference filenames relative to VIDEO_DIR or absolute.
DOOR_HZ: float = 5.0  # Samples/second for continuous door-state scan.
CONFIRM_SECONDS: float = 0.4  # Consecutive agreeing samples before transition.
MIN_REFERENCE_DIFFERENCE: float = (
    0.25  # Mean difference of standardized spatial patterns.
)
MIN_PATCH_STD: float = 2.0  # Reject flat or washed-out regions in raw 8-bit grayscale.
TIMING_TOLERANCE_FRAMES: float = (
    0.55  # Allowed sampled presentation-time deviation from nominal frame index.
)
STATE_MARGIN: float = (
    0.25  # Required distance advantage as fraction of reference separation.
)
MAX_REFERENCE_DISTANCE: float = (
    1.25  # Reject observations far from both reference states.
)
COUNT_HZ: float = 6.0  # Samples/second in crossing windows.
PADDING_SECONDS: float = (
    0.75  # Extra time before and after; adjacent windows never overlap.
)
PERSON_SCORE: float = 0.35  # Minimum YOLOX person probability times objectness.
NMS_IOU: float = 0.45  # Overlap threshold for duplicate person boxes.
ANCHOR_HEIGHT: float = (
    0.65  # From box top to tracking point, as fraction of box height.
)
TRACK_SPEED: float = 1.2  # Maximum normalized image distance per elapsed second.
TRACK_REACH: float = 0.32  # Cap association radius across longer gaps.
TRACK_GAP_SECONDS: float = 1.0  # Maximum missing time before starting a new track.
GATE_DEADBAND: float = 0.025  # Minimum normalized distance from line for a stable side.
MATCH_SECONDS: float = (
    2.0  # Maximum endpoint gap to allow non-overlapping validation matches.
)
# === END CONFIG ===

TRUTH_COLUMNS = (
    "clip",
    "kind",
    "visit_id",
    "service_date",
    "trip",
    "bus",
    "open_s",
    "close_s",
    "stop_code",
    "stop_sequence",
    "boardings",
    "alightings",
)
COUNTS = ("boardings", "alightings")


def config_source():
    """Return the literal config, including inline comments, in a file or pasted cell."""
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except (NameError, OSError):
        from IPython import get_ipython

        source = get_ipython().history_manager.input_hist_raw[-1]
    match = re.search(
        r"^# === BEGIN CONFIG ===$.*?^# === END CONFIG ===$",
        source,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError("Run the whole module in one notebook cell to log its config.")
    return match.group()


def _clean_id(value):
    """Preserve leading zeros and remove Excel's trailing decimal zero."""
    return re.sub(r"\.0$", "", str(value).strip())


def service_date(value, context):
    """Normalize an operating date to YYYY-MM-DD; slash dates read month first."""
    value = str(value).strip()
    if not value:
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: invalid service_date {value!r}") from exc


def _visit_key_part(name, value):
    """Match visit labels across systems: numeric labels ignore leading zeros."""
    value = _clean_id(value)
    if name != "clip" and value.isdigit():
        return value.lstrip("0") or "0"
    return value


def fingerprint(path):
    """Hash large clips without reading them entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def workbook(path, sheets):
    """Write literal cell text, filterable headers and Arial throughout."""
    wb = Workbook()
    wb.remove(wb.active)
    for name, (rows, base) in sheets.items():
        ws = wb.create_sheet(name)
        cols = list(dict.fromkeys([*base, *(k for row in rows for k in row)]))
        ws.append(cols)
        for row in rows:
            ws.append([row.get(k) for k in cols])
        for cells in ws.iter_rows():
            for cell in cells:
                if cell.data_type == "f":
                    cell.data_type = "s"
                cell.font = Font(name="Arial", bold=cell.row == 1)
                if isinstance(cell.value, _dt.datetime):
                    cell.number_format = "yyyy-mm-dd hh:mm:ss.000"
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    wb.save(path)


def clip_info(path):
    """Get recorder identifiers without inferring stop identity from video."""
    m = re.fullmatch(FILENAME_REGEX, path.name, re.IGNORECASE)
    if not m:
        raise ValueError("filename does not match FILENAME_REGEX")
    bus, door = _clean_id(m["bus"]), m["camera"].lower()
    start = _dt.datetime.strptime(m["start"], START_FORMAT)
    return {
        "clip": path.name,
        "bus": bus,
        "door": door,
        "trip": m.groupdict().get("trip") or "",
        "recording_start": start,
    }


def video(path):
    """Open a local constant-frame-rate video and validate timing metadata."""
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if (
        not cap.isOpened()
        or not math.isfinite(fps)
        or not math.isfinite(count)
        or min(fps, count) <= 0
    ):
        cap.release()
        raise ValueError(f"Invalid FPS or duration: {path}")
    return cap, fps, int(count)


def samples(cap, fps, n, start, end, hz, audit=None):
    """Yield sampled frame times from a half-open interval with one initial seek."""
    first = max(0, math.ceil(start * fps - 1e-7))
    end_index = min(n, math.ceil(end * fps - 1e-7))
    if first >= end_index:
        return
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
        raise ValueError(f"Cannot seek to frame {first}")
    stride = max(1.0, fps / hz)
    index = first
    k = 0
    origin_ms = None
    while index < end_index:
        ok, image = cap.read()
        if not ok:
            if audit is not None and index > 0:
                audit["decoded_frames"] = index
                return
            raise ValueError(
                f"Cannot decode frame {index} at {index / fps:.3f} seconds"
            )
        if audit is not None:
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if origin_ms is None:
                origin_ms = ms
            if (
                not math.isfinite(ms)
                or abs((ms - origin_ms) * fps / 1000 - index) > TIMING_TOLERANCE_FRAMES
            ):
                raise ValueError(
                    f"Nonuniform or unavailable video timestamps at frame {index}; re-export at constant frame rate"
                )
        yield index / fps, image
        k += 1
        nxt = first + round(k * stride)
        for skipped in range(index + 1, min(nxt, end_index)):
            if not cap.grab():
                if audit is not None:
                    audit["decoded_frames"] = skipped
                    return
                raise ValueError(f"Cannot advance after frame {index}")
        index = nxt


def patch(image, roi):
    """Extract the calibrated part of the door as a small grayscale image."""
    h, w = image.shape[:2]
    x, y, r, b = (round(v * s) for v, s in zip(roi, (w, h, w, h)))
    if r - x < 3 or b - y < 3:
        raise ValueError("Door rectangle too small")
    gray = cv2.cvtColor(image[y:b, x:r], cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    spread = float(thumb.std())
    if spread < MIN_PATCH_STD:
        return None
    return (thumb - thumb.mean()) / spread


def difference(a, b):
    """Return the mean absolute difference between two standardized patches."""
    return float(np.abs(a - b).mean())


def reference(cfg, preview):
    """Check profile geometry and load one closed and one open reference."""
    roi = cfg["door_roi"]
    line = np.asarray(cfg["gate"], dtype=float)
    if (
        len(roi) != 4
        or not 0 <= roi[0] < roi[2] <= 1
        or not 0 <= roi[1] < roi[3] <= 1
        or line.shape != (2, 2)
        or not np.all((line >= 0) & (line <= 1))
        or np.linalg.norm(line[1] - line[0]) < 0.02
        or cfg["inside_sign"] not in (-1, 1)
    ):
        raise ValueError("Invalid door ROI, gate, or inside_sign")
    path = VIDEO_DIR / Path(cfg["reference"])
    cap, fps, n = video(path)
    refs = []
    try:
        for label in ("closed", "open"):
            t = cfg[label + "_s"]
            if not 0 <= t < n / fps:
                raise ValueError(f"{path}: reference {label} time outside video")
            index = math.floor(t * fps)
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError("Reference seek failed")
            ok, img = cap.read()
            if not ok:
                raise ValueError("Cannot decode reference frame")
            shape = img.shape[:2]
            pattern = patch(img, roi)
            if pattern is None:
                raise ValueError(
                    f"{path}: {label} reference has insufficient texture; select a door edge or panel feature"
                )
            refs.append(pattern)
            if preview:
                scale = np.array([img.shape[1] - 1, img.shape[0] - 1])
                corners = (np.reshape(roi, (2, 2)) * scale).astype(int).tolist()
                points = (line * scale).astype(int).tolist()
                cv2.rectangle(img, tuple(corners[0]), tuple(corners[1]), (0, 255, 0), 3)
                cv2.arrowedLine(
                    img, tuple(points[0]), tuple(points[1]), (0, 180, 255), 3
                )
                cv2.imshow(f"{label}: {path.name} - press any key", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    finally:
        cap.release()
    sep = difference(*refs)
    if sep < MIN_REFERENCE_DIFFERENCE:
        raise ValueError(
            f"Open and closed references too similar ({sep:.2f} < "
            f"{MIN_REFERENCE_DIFFERENCE}): {path}"
        )
    return (*refs, sep, shape)


def scan(cap, fps, n, cfg, refs):
    """Debounce reference comparisons; expose gaps and clip-edge censoring."""
    closed, opened, sep, ref_shape = refs
    audit = {"decoded_frames": n}
    state = pending = None
    since = 0.0
    onset = None
    partial = False
    runs = []
    unknown = total = 0
    last_unknown_start = None
    uncertain = []
    for t, img in samples(cap, fps, n, 0, n / fps, DOOR_HZ, audit):
        if img.shape[:2] != ref_shape:
            raise ValueError(
                f"Clip dimensions {img.shape[:2]} differ from reference {ref_shape}; configure a matching profile"
            )
        total += 1
        p = patch(img, cfg["door_roi"])
        dc, do = (
            (difference(p, closed), difference(p, opened))
            if p is not None
            else (math.inf, math.inf)
        )
        if (
            min(dc, do) > MAX_REFERENCE_DISTANCE * sep
            or abs(dc - do) < STATE_MARGIN * sep
        ):
            unknown += 1
            if last_unknown_start is None:
                last_unknown_start = t
            pending = None
            continue
        if last_unknown_start is not None:
            uncertain.append((last_unknown_start, t))
            last_unknown_start = None
        obs = "open" if do < dc else "closed"
        if obs != pending:
            pending, since = obs, t
        if obs == state or t - since + 1e-7 < CONFIRM_SECONDS:
            continue
        if obs == "open":
            onset = since
            partial = state is None
        elif onset is not None:
            runs.append((onset, since, partial, False))
            onset = None
        state = obs
    duration = audit["decoded_frames"] / fps
    if last_unknown_start is not None:
        uncertain.append((last_unknown_start, duration))
    if onset is not None:
        runs.append((onset, duration, partial, True))
    if state is None:
        raise ValueError("No confirmed door state; review references and ROI")
    rows = []
    for a, b, left, right in runs:
        rows.append(
            {
                "open_s": a,
                "close_s": b,
                "left_censored": left,
                "right_censored": right,
                "uncertain_seconds": sum(
                    max(0, min(y, b) - max(x, a)) for x, y in uncertain
                ),
            }
        )
    return rows, unknown / total, audit["decoded_frames"]


class PersonDetector:
    """CPU inference for the specific raw-head OpenCV Zoo YOLOX-S model."""

    def __init__(self, path):
        if not path.is_file():
            raise FileNotFoundError(path)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        grids = []
        strides = []
        for step in (8, 16, 32):
            y, x = np.indices((640 // step, 640 // step))
            grids.append(np.column_stack((x.ravel(), y.ravel())))
            strides.extend([step] * x.size)
        self.grids = np.concatenate(grids)
        self.strides = np.asarray(strides)

    def __call__(self, img):
        """Return normalized person anchors after person-class suppression."""
        h, w = img.shape[:2]
        scale = 640 / max(w, h)
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[:sh, :sw] = cv2.resize(img, (sw, sh))
        self.net.setInput(cv2.dnn.blobFromImage(canvas, swapRB=True))
        out = self.net.forward()
        if out.shape != (1, 8400, 85):
            raise ValueError(
                f"Expected OpenCV Zoo YOLOX-S FP32 output; received {out.shape}"
            )
        pred = out[0]
        score = pred[:, 4] * pred[:, 5]
        idx = np.flatnonzero(
            (score >= PERSON_SCORE) & (np.argmax(pred[:, 5:], axis=1) == 0)
        )
        if not len(idx):
            return []
        stride = self.strides[idx, None]
        center = (pred[idx, :2] + self.grids[idx]) * stride
        wh = np.exp(pred[idx, 2:4]) * stride
        boxes = np.column_stack((center - wh / 2, wh))
        keep = cv2.dnn.NMSBoxes(
            boxes.tolist(), score[idx].tolist(), PERSON_SCORE, NMS_IOU
        )
        anchors = []
        for k in np.asarray(keep, dtype=int).ravel():
            x, y, bw, bh = boxes[k]
            p = ((x + bw / 2) / sw, (y + ANCHOR_HEIGHT * bh) / sh)
            if all(0 <= v <= 1 for v in p):
                anchors.append(p)
        return anchors


@dataclass
class Track:
    """Track state within a single door cycle."""

    point: tuple
    time: float
    stable: tuple | None = None
    side: int = 0
    net: int = 0
    bypass: bool = False


def observe(track, point, cfg):
    """Record a crossing only if the trajectory intersects the finite gate."""
    a, b = cfg["gate"]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)

    def signed(p):
        return ((p[1] - a[1]) * dx - (p[0] - a[0]) * dy) / length

    distance = signed(point)
    if abs(distance) <= GATE_DEADBAND:
        return
    side = (1 if distance > 0 else -1) * cfg["inside_sign"]
    if track.bypass:
        if side != track.side:
            return
        track.bypass = False
    if track.stable is not None and track.side != side:
        old = track.stable
        alpha = signed(old) / (signed(old) - distance)
        x = old[0] + alpha * (point[0] - old[0])
        y = old[1] + alpha * (point[1] - old[1])
        if 0 <= ((x - a[0]) * dx + (y - a[1]) * dy) / length**2 <= 1:
            track.net += side
        else:
            track.bypass = True
            return  # Crossing the extension does not establish the opposite gate side.
    track.side, track.stable = side, point


def crossings(cap, fps, n, start, end, cfg, detector):
    """Associate nearest eligible anchors one-to-one; flag ambiguous associations."""
    tracks = []
    ambiguous = frames = reacquired = 0
    last_sample = start
    for t, img in samples(cap, fps, n, start, end, COUNT_HZ):
        frames += 1
        pts = detector(img)
        edges = []
        for i, tr in enumerate(tracks):
            if t - tr.time <= TRACK_GAP_SECONDS + 1e-7:
                for j, p in enumerate(pts):
                    d = math.dist(tr.point, p)
                    if d <= min(TRACK_REACH, TRACK_SPEED * (t - tr.time)) + 1e-9:
                        edges.append((d, i, j))
        if len(edges) > len({i for _, i, _ in edges}) or len(edges) > len(
            {j for _, _, j in edges}
        ):
            ambiguous += 1
        used = set()
        matched = set()
        for _, i, j in sorted(edges):
            if i in used or j in matched:
                continue
            tr = tracks[i]
            reacquired += int(tr.time < last_sample - 1e-7)
            tr.point, tr.time = pts[j], t
            observe(tr, pts[j], cfg)
            used.add(i)
            matched.add(j)
        for j, p in enumerate(pts):
            if j not in matched:
                tr = Track(p, t)
                observe(tr, p, cfg)
                tracks.append(tr)
        last_sample = t
    return {
        "boardings": sum(max(0, tr.net) for tr in tracks),
        "alightings": sum(max(0, -tr.net) for tr in tracks),
        "person_tracks": len(tracks),
        "reacquired_tracks": reacquired,
        "tracks_lost_before_window_end": sum(
            tr.time < last_sample - 1e-7 for tr in tracks
        ),
        "count_frames": frames,
        "ambiguous_tracking_frames": ambiguous,
    }


def read_table(path):
    """Read literal strings from CSV or the first worksheet of XLSX."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return pd.read_excel(p, dtype=str, keep_default_na=False, engine="openpyxl")


def player_seconds(value, clip, field):
    """Parse elapsed seconds or a video-player timestamp with a useful row error."""
    try:
        parts = str(value).strip().split(":")
        if len(parts) > 3 or any(not p.isdigit() for p in parts[:-1]):
            raise ValueError
        seconds = float(parts[-1])
        if (
            not math.isfinite(seconds)
            or seconds < 0
            or (len(parts) > 1 and seconds >= 60)
            or (len(parts) == 3 and int(parts[1]) >= 60)
        ):
            raise ValueError
        return (
            sum(int(p) * 60**i for i, p in enumerate(reversed(parts[:-1]), 1)) + seconds
        )
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            f"{clip}: invalid {field} {value!r}; use seconds, mm:ss or hh:mm:ss"
        ) from exc


def optional_count(value, context):
    """Validate nonnegative counts without turning blanks into zeros."""
    value = _clean_id(value)
    if value and not value.isdigit():
        raise ValueError(f"{context}: invalid passenger count {value!r}")
    return int(value) if value else None


def load_truth():
    """Read independent manual events, per-direction coverage, and closed-door visit labels."""
    truth, coverage, closed = {}, {}, []
    if not TRUTH_PATH:
        if RIDECHECK_PATH:
            raise ValueError(
                "Ridecheck comparison requires a labeled manual visit inventory"
            )
        return truth, coverage, closed
    df = read_table(TRUTH_PATH)
    required = {"clip", "kind", "open_s", "close_s"}
    if required - set(df):
        raise ValueError(f"Truth table missing {sorted(required - set(df))}")
    for name, group in df.groupby("clip"):
        rows: list[dict[str, Any]] = []
        neg = None
        annotated = False
        for r in group.to_dict("records"):
            kind = r["kind"].strip().lower()
            if not kind:
                continue
            annotated = True
            row = {
                k: _clean_id(r.get(k, ""))
                for k in (
                    "visit_id",
                    "service_date",
                    "bus",
                    "trip",
                    "stop_code",
                    "stop_sequence",
                )
            }
            row["service_date"] = service_date(row["service_date"], name)
            counts = {c: optional_count(r.get(c, ""), name) for c in COUNTS}
            if kind == "none":
                if neg is not None or any(v not in (None, 0) for v in counts.values()):
                    raise ValueError(
                        f"{name}: none requires one row with zero or blank counts"
                    )
                neg = counts
            elif kind == "no_cycle":
                if any(v not in (None, 0) for v in counts.values()):
                    raise ValueError(f"{name}: no_cycle has nonzero counts")
                closed.append(dict(row, clip=name))
            elif kind == "event":
                a, b = (player_seconds(r[k], name, k) for k in ("open_s", "close_s"))
                if b <= a:
                    raise ValueError(f"{name}: close_s must follow open_s")
                rows.append(dict(row, open_s=a, close_s=b, **counts))
            else:
                raise ValueError(f"{name}: unknown kind {kind!r}")
        if not annotated:
            continue
        if rows and neg is not None:
            raise ValueError(f"{name}: cannot combine none and event")
        rows.sort(key=lambda x: x["open_s"])
        if any(rows[i - 1]["close_s"] > rows[i]["open_s"] for i in range(1, len(rows))):
            raise ValueError(f"{name}: overlapping truth cycles")
        truth[name] = rows
        coverage[name] = {
            c: all(r[c] is not None for r in rows) if rows else (neg or {}).get(c) == 0
            for c in COUNTS
        }
    return truth, coverage, closed


def load_ridecheck():
    """Check Ridecheck keys, door groups, headers, labels and counts before scanning."""
    keys = [k for k in RIDECHECK_COLUMNS if k not in (*COUNTS, "door")]
    if not (
        {"clip", "visit_id"} & set(keys)
        or {"service_date", "bus", "trip", "stop_sequence"} <= set(keys)
    ):
        raise ValueError(
            "Ridecheck needs clip, a unique visit_id, or service_date/bus/trip/stop_sequence"
        )
    if not keys or not set(keys) <= {
        "clip",
        "visit_id",
        "service_date",
        "trip",
        "bus",
        "stop_code",
        "stop_sequence",
    }:
        raise ValueError("Invalid Ridecheck visit keys")
    groups = {k: set(v) for k, v in RIDECHECK_GROUPS.items()}
    doors = set().union(*groups.values())
    if (
        not doors
        or any(not g for g in groups.values())
        or sum(map(len, groups.values())) != len(doors)
    ):
        raise ValueError("Ridecheck door groups must be nonempty and disjoint")
    totals = "door" not in RIDECHECK_COLUMNS and not RIDECHECK_WIDE
    if (totals or any(len(g) > 1 for g in groups.values())) and "clip" in keys:
        raise ValueError("Use a shared visit key instead of clip for pooled doors")
    if any(len(cols) != 2 for cols in RIDECHECK_WIDE.values()):
        raise ValueError(
            "Each RIDECHECK_WIDE entry needs a boarding and an alighting header"
        )
    data = read_table(RIDECHECK_PATH)
    selected = list(RIDECHECK_COLUMNS.values()) + [
        col for pair in RIDECHECK_WIDE.values() for col in pair
    ]
    if set(selected) - set(data):
        raise ValueError(
            f"Missing Ridecheck headers: {sorted(set(selected) - set(data))}"
        )
    if data.empty:
        raise ValueError(f"Ridecheck table has no rows: {RIDECHECK_PATH}")
    entries = []
    seen = set()
    for number, source in enumerate(data.to_dict("records"), 1):
        where = f"Ridecheck data row {number}"
        labels = {k: _clean_id(source[RIDECHECK_COLUMNS[k]]) for k in keys}
        if "service_date" in labels:
            labels["service_date"] = service_date(labels["service_date"], where)
        if not all(labels.values()):
            raise ValueError(f"{where}: blank visit key {labels}")
        key = tuple(_visit_key_part(k, labels[k]) for k in keys)
        columns = RIDECHECK_WIDE or {
            (
                "vehicle"
                if totals
                else source[RIDECHECK_COLUMNS["door"]].strip().lower()
            ): tuple(RIDECHECK_COLUMNS[c] for c in COUNTS)
        }
        for label, cols in columns.items():
            if (key, label) in seen:
                raise ValueError(
                    f"{where}: duplicate visit and door group {labels} {label!r}"
                )
            seen.add((key, label))
            expected = doors if totals else groups.get(label)
            if not expected:
                raise ValueError(
                    f"{where}: door {label!r} is not a RIDECHECK_GROUPS key {sorted(groups)}"
                )
            entries.append(
                {
                    "key": key,
                    "labels": labels,
                    "label": label,
                    "expected": expected,
                    "counts": {
                        c: optional_count(source[col], f"{where} {label!r}")
                        for c, col in zip(COUNTS, cols)
                    },
                }
            )
    return {"keys": keys, "entries": entries}


def compare_visits(events, summaries, closed, issues, ridecheck):
    """Aggregate reopening cycles before comparing external counts once per visit."""
    if not ridecheck:
        return []
    keys = ridecheck["keys"]
    door_of = {r["clip"]: r["door"] for r in summaries}
    known = {r["clip"] for r in summaries if r["eligible_validation"] and r["reviewed"]}
    excluded = {
        r["clip"] for r in summaries if r["reviewed"] and not r["eligible_validation"]
    }
    extra = {
        r["clip"]
        for r in events
        if r["match"] == "extra" and any(r.get(c, 0) for c in COUNTS)
    }
    by_visit = {}
    excluded_visits = {}
    unassigned = set()

    def add(r, labels, cycle):
        clip = r["clip"]
        if clip not in known and clip not in excluded:
            return
        values = [_clean_id(labels.get(k, "")) for k in keys]
        if not all(values):
            if clip in excluded:
                return
            if any(r.get(c, 0) for c in COUNTS):
                unassigned.add(clip)
            issues.append(
                {
                    "clip": clip,
                    "issue": "missing Ridecheck visit key",
                    "detail": str(dict(zip(keys, values))),
                }
            )
            return
        key = tuple(_visit_key_part(k, v) for k, v in zip(keys, values))
        if clip in excluded:
            excluded_visits.setdefault(key, set()).add(door_of[clip])
            return
        by_visit.setdefault(key, {}).setdefault(door_of[clip], []).append((r, cycle))

    for r in events:
        if r["match"] in ("matched", "missed"):
            add(
                r,
                {
                    k: r.get("truth_" + k, r["clip"] if k == "clip" else "")
                    for k in keys
                },
                True,
            )
    for r in closed:
        add(r, r, False)
    output = []
    used = set()
    for entry in ridecheck["entries"]:
        key, labels, label = entry["key"], entry["labels"], entry["label"]
        expected = entry["expected"]
        members = by_visit.get(key, {})
        records = [item for door in expected for item in members.get(door, [])]
        used.update((key, d) for d in expected if d in members)
        status = "compared"
        if expected & excluded_visits.get(key, set()):
            status = "footage excluded from validation (partial or calibration clip)"
        elif expected - members.keys():
            status = "missing reviewed door coverage"
        elif any(
            len({r["clip"] for r, _ in members[d]}) > 1
            or (len(members[d]) > 1 and not all(cycle for _, cycle in members[d]))
            for d in expected
        ):
            status = "ambiguous visit coverage"
        elif any(r["clip"] in unassigned for r, _ in records):
            status = "unassigned visit label with passenger counts"
        elif any(r["clip"] in extra for r, _ in records):
            status = "unassigned extra cycle with passenger counts"
        row = dict(
            labels,
            door_group=label,
            status=status,
            manual_cycles=sum(cycle for _, cycle in records),
        )
        for c in COUNTS:
            comparator = entry["counts"][c]
            row["ridecheck_" + c] = comparator
            if status == "compared":
                model = sum(r.get(c, 0) for r, _ in records)
                row["model_" + c] = model
                if comparator is not None:
                    row["difference_" + c] = model - comparator
        if status != "compared" or any(row["ridecheck_" + c] is None for c in COUNTS):
            issues.append(
                dict(
                    labels,
                    door_group=label,
                    issue=status if status != "compared" else "blank Ridecheck count",
                )
            )
        output.append(row)
    for key, members in by_visit.items():
        for door in members:
            if (key, door) not in used:
                issues.append(
                    dict(
                        zip(keys, key),
                        door=door,
                        issue="manual visit lacks Ridecheck row",
                    )
                )
    return output


def pair_cycles(actual, manual):
    """Monotone one-to-one alignment: maximize pairs, then minimize endpoint error."""
    m, n = len(actual), len(manual)
    scores = [[(0, 0.0)] * (n + 1) for _ in range(m + 1)]
    moves = {}
    for i in range(m + 1):
        for j in range(n + 1):
            if not (i or j):
                continue
            options = []
            if i:
                options.append((scores[i - 1][j], "extra"))
            if j:
                options.append((scores[i][j - 1], "missed"))
            if i and j:
                p, q = actual[i - 1], manual[j - 1]
                gap = max(p["open_s"], q["open_s"]) - min(p["close_s"], q["close_s"])
                delta = abs(p["open_s"] - q["open_s"]) + abs(
                    p["close_s"] - q["close_s"]
                )
                if (
                    gap <= 0
                    or max(abs(p[k] - q[k]) for k in ("open_s", "close_s"))
                    <= MATCH_SECONDS
                ):
                    hits, merit = scores[i - 1][j - 1]
                    options.append(((hits + 1, merit - delta), "matched"))
            scores[i][j], moves[i, j] = max(options, key=lambda pair: pair[0])
    pairs = []
    i, j = m, n
    while i or j:
        move = moves[i, j]
        p = actual[i - 1] if move != "missed" else None
        q = manual[j - 1] if move != "extra" else None
        pairs.append((move, p, q))
        i -= int(p is not None)
        j -= int(q is not None)
    # Preserve event order even where independent missed and extra cycles tie.
    return sorted(
        pairs,
        key=lambda pair: min(r["open_s"] for r in pair[1:] if r is not None),
    )


def safe_div(a, b):
    """Leave an undefined ratio blank."""
    return a / b if b else None


def report_clip(info, actual, manual, counted, coverage):
    """Retain false alarms, missed cycles, and unscored footage in the ledger."""
    reviewed = manual is not None
    pairs = (
        pair_cycles(actual, manual)
        if reviewed
        else [("unreviewed", p, None) for p in actual]
    )
    rows = []
    for status, p, q in pairs:
        r: dict[str, Any] = dict(info, match=status)
        if p:
            r.update(p)
            for k in ("open", "close"):
                r[k + "_time"] = info["recording_start"] + _dt.timedelta(
                    seconds=p[k + "_s"]
                )
        if q:
            r.update({"truth_" + k: v for k, v in q.items()})
        if p and q:
            for k in ("open_s", "close_s"):
                r[k + "_error"] = p[k] - q[k]
        if reviewed and counted:
            for c in COUNTS:
                if coverage.get(c):
                    truth = q[c] if q else 0
                elif status == "matched" and q[c] is not None:
                    # Incomplete count review: misses and false alarms stay unscored.
                    truth = q[c]
                    r["count_scope_" + c] = "matched cycles only"
                else:
                    truth = None
                if truth is not None:
                    r["error_" + c] = (p[c] if p else 0) - truth
                    r["abs_error_" + c] = abs(r["error_" + c])
        rows.append(r)
    hits = sum(r["match"] == "matched" for r in rows)
    summary = dict(info, reviewed=reviewed, detected_cycles=len(actual))
    if reviewed:
        summary.update(
            truth_cycles=len(manual),
            matched_cycles=hits,
            missed_cycles=len(manual) - hits,
            extra_cycles=len(actual) - hits,
            door_precision=safe_div(hits, len(actual)),
            door_recall=safe_div(hits, len(manual)),
        )
    if counted:
        for c in COUNTS:
            summary[c] = sum(p[c] for p in actual)
            if reviewed and coverage.get(c):
                total = sum(q[c] for q in manual)
                summary["truth_" + c] = total
                summary["ratio_" + c] = safe_div(summary[c], total)
        if reviewed and all("truth_" + c in summary for c in COUNTS):
            drift = peak = 0
            for r in rows:
                drift += r["error_boardings"] - r["error_alightings"]
                peak = max(peak, abs(drift))
            summary["max_abs_door_load_difference"] = peak
    return rows, summary


def accuracy(rows):
    """Aggregate crossing errors, retaining separate matched and end-to-end scores."""
    from collections import defaultdict

    buckets = defaultdict(list)
    for r in rows:
        for c in COUNTS:
            if "error_" + c not in r:
                continue
            truth = r.get("truth_" + c, 0)
            band = (
                "0"
                if truth == 0
                else "1–2"
                if truth <= 2
                else "3–5"
                if truth <= 5
                else "6+"
            )
            if r["match"] != "matched":
                scopes = ["end-to-end"]
            elif r.get("count_scope_" + c) == "matched cycles only":
                scopes = ["matched cycles"]
            else:
                scopes = ["end-to-end", "matched cycles"]
            for scope in scopes:
                for dimension, group in [
                    ("overall", "all"),
                    ("door", r["door"]),
                    ("crowding", band),
                ]:
                    buckets[scope, c, dimension, group].append((r["error_" + c], truth))
    result = []
    for (scope, c, dimension, group), pairs in buckets.items():
        errors = [e for e, _ in pairs]
        n = len(errors)
        total = sum(v for _, v in pairs)
        result.append(
            {
                "scope": scope,
                "count": c,
                "breakdown": dimension,
                "group": group,
                "observations": n,
                "exact_share": errors.count(0) / n,
                "within_one_share": sum(abs(e) <= 1 for e in errors) / n,
                "mae": sum(map(abs, errors)) / n,
                "bias": sum(errors) / n,
                "total_ratio": safe_div(total + sum(errors), total),
            }
        )
    return result


def analyze(path, info, cfg, refs, detector, truth, coverage):
    """Scan a complete trip-door file, then count only around detected cycles."""
    cap, fps, n = video(path)
    try:
        duration = n / fps
        if truth is not None and any(q["close_s"] > duration + 1e-6 for q in truth):
            raise ValueError("Truth interval extends beyond the end of the video")
        events, unknown, decoded = scan(cap, fps, n, cfg, refs)
        metadata_duration = duration
        duration = decoded / fps
        partial = bool(decoded < n)
        tail = []
        if partial and truth is not None:
            tail = [q for q in truth if q["close_s"] > duration]
            truth = [q for q in truth if q["close_s"] <= duration]
            coverage = {}
        if detector is not None:
            for i, r in enumerate(events):
                a, b = r["open_s"], r["close_s"]
                left = (events[i - 1]["close_s"] + a) / 2 if i else 0
                right = (
                    (b + events[i + 1]["open_s"]) / 2
                    if i + 1 < len(events)
                    else duration
                )
                start = max(left, a - PADDING_SECONDS)
                end = min(right, b + PADDING_SECONDS)
                r.update(count_start_s=start, count_end_s=end)
                r.update(crossings(cap, fps, decoded, start, end, cfg, detector))
        rows, summary = report_clip(info, events, truth, detector is not None, coverage)
        for q in tail:
            rows.append(
                dict(
                    info,
                    match="unobservable tail",
                    **{"truth_" + k: v for k, v in q.items()},
                )
            )
        summary.update(
            truth_cycles_outside_decoded_prefix=len(tail),
            partial_clip=partial,
            metadata_seconds=metadata_duration,
            video_seconds=duration,
            uncertain_sample_share=unknown,
        )
        return rows, summary
    finally:
        cap.release()


def validate(mode):
    """Validate settings for the active mode only."""
    if mode not in ("template", "calibrate", "doors", "pilot"):
        raise ValueError(f"Invalid mode: {mode}")
    paths = [VIDEO_DIR, OUTPUT_DIR]
    if mode == "pilot":
        paths.append(MODEL_PATH)
    if any("EDIT_ME" in str(p) for p in paths):
        raise ValueError("Replace EDIT_ME paths used by this mode")
    if not VIDEO_DIR.is_dir():
        raise FileNotFoundError(VIDEO_DIR)
    if (
        min(
            DOOR_HZ,
            COUNT_HZ,
            TRACK_SPEED,
            TRACK_REACH,
            MIN_PATCH_STD,
            TIMING_TOLERANCE_FRAMES,
            TRACK_GAP_SECONDS,
            MIN_REFERENCE_DIFFERENCE,
            MAX_REFERENCE_DISTANCE,
        )
        <= 0
        or min(CONFIRM_SECONDS, PADDING_SECONDS, GATE_DEADBAND, MATCH_SECONDS) < 0
    ):
        raise ValueError("Invalid sampling, tracking, or duration threshold")
    if not all(0 < v < 1 for v in (STATE_MARGIN, PERSON_SCORE, NMS_IOU, ANCHOR_HEIGHT)):
        raise ValueError("Fractional thresholds must lie between zero and one")


def run(mode):
    """Run the configured local workflow and write an auditable sidecar."""
    if "EDIT_ME" in str(OUTPUT_DIR):
        raise ValueError("Replace EDIT_ME in OUTPUT_DIR before running")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # noqa: UP017 - supports Python 3.9
    stem = OUTPUT_DIR / f"door_local_{mode}_{stamp}"
    source = config_source()
    params = {k: globals()[k] for k in re.findall(r"^(\w+):", source, re.MULTILINE)}
    params["MODE"] = mode
    params.update(
        {k: str(v.resolve()) for k, v in params.items() if isinstance(v, Path)}
    )
    log = {
        "config_block": source,
        "parameters": params,
        "opencv": cv2.__version__,
        "fingerprints": {},
        "totals": {},
    }
    started = time.perf_counter()
    try:
        validate(mode)
        return process(mode, stem, log)
    except Exception as e:
        log["error"] = str(e)
        log["traceback"] = traceback.format_exc()
        raise
    finally:
        log["elapsed_seconds"] = time.perf_counter() - started
        stem.with_suffix(".json").write_text(
            json.dumps(log, indent=2, default=str), encoding="utf-8"
        )


def process(mode, stem, log):
    """Reconcile per-clip failures and save usable results before a nonzero exit."""
    info_by_path = {}
    issues = []
    for path in sorted(p for p in VIDEO_DIR.iterdir() if p.is_file()):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            info_by_path[path] = clip_info(path)
        except (KeyError, ValueError) as e:
            issues.append({"clip": path.name, "issue": str(e)})
    if not info_by_path:
        raise ValueError(f"Zero video filenames matched in {VIDEO_DIR}: {issues}")
    if issues:
        print(
            f"Skipped {len(issues)} video file(s) not matching FILENAME_REGEX: "
            + ", ".join(i["clip"] for i in issues),
            file=sys.stderr,
        )
    output = stem.with_suffix(".xlsx")
    if mode == "template":
        dest = stem.with_suffix(".csv")
        pd.DataFrame(
            [
                {k: info[k] for k in ("clip", "bus", "trip")}
                for info in info_by_path.values()
            ],
            columns=np.asarray(TRUTH_COLUMNS),
        ).to_csv(dest, index=False)
        log["unmatched_files"] = issues
        log["totals"]["template_clips"] = len(info_by_path)
        return dest
    failures = 0
    truth, coverage, closed = {}, {}, []
    detector = None
    ridecheck = None
    if mode in ("doors", "pilot"):
        for name in (TRUTH_PATH, RIDECHECK_PATH):
            if name:
                log["fingerprints"][str(Path(name).resolve())] = fingerprint(Path(name))
        truth, coverage, closed = load_truth()
    if mode == "pilot":
        if RIDECHECK_PATH:
            ridecheck = load_ridecheck()  # Fail before hours of scanning, not after.
        log["fingerprints"][str(MODEL_PATH.resolve())] = fingerprint(MODEL_PATH)
        detector = PersonDetector(MODEL_PATH)
    refs = {}
    bad_profiles = {}
    ref_hashes = set()
    events = []
    summaries = []
    for path, info in info_by_path.items():
        start = time.perf_counter()
        camera = info["door"]
        key = f"{info['bus']}:{camera}"
        if key not in PROFILES:
            key = camera
        try:
            cfg = PROFILES[key]
            if key in bad_profiles:
                raise ValueError(bad_profiles[key])
            if key not in refs:
                if "EDIT_ME" in str(cfg["reference"]):
                    raise ValueError(
                        f"Replace EDIT_ME in PROFILES[{key!r}]['reference']"
                    )
                ref_path = (VIDEO_DIR / Path(cfg["reference"])).resolve()
                if str(ref_path) not in log["fingerprints"]:
                    log["fingerprints"][str(ref_path)] = fingerprint(ref_path)
                refs[key] = reference(cfg, mode == "calibrate")
                ref_hashes.add(log["fingerprints"][str(ref_path)])
            if mode == "calibrate":
                continue
            resolved = str(path.resolve())
            if resolved not in log["fingerprints"]:
                log["fingerprints"][resolved] = fingerprint(path)
            rows, summary = analyze(
                path,
                info,
                cfg,
                refs[key],
                detector,
                truth.get(path.name),
                coverage.get(path.name, {}),
            )
            if summary["partial_clip"]:
                issues.append(
                    dict(
                        info,
                        issue="partial clip: decoded prefix retained",
                        decoded_seconds=summary["video_seconds"],
                        metadata_seconds=summary["metadata_seconds"],
                    )
                )
            events.extend(rows)
            summary["processing_seconds"] = time.perf_counter() - start
            summary["compute_hours_per_camera_hour"] = (
                summary["processing_seconds"] / summary["video_seconds"]
            )
            summaries.append(summary)
            if TRUTH_PATH and path.name not in truth:
                issues.append(dict(info, issue="clip has no independent review"))
            for r in rows:
                if r["match"] in ("extra", "missed"):
                    issues.append(
                        dict(
                            info,
                            issue=r["match"] + " door cycle",
                            open_s=r.get("open_s", r.get("truth_open_s")),
                        )
                    )
        except (OSError, ValueError, KeyError, cv2.error) as e:
            failures += 1
            if key not in refs:
                bad_profiles[key] = str(e)
            log.setdefault("clip_tracebacks", {})[path.name] = traceback.format_exc()
            issues.append(dict(info, issue="clip failed", detail=str(e), profile=key))
    for name in truth.keys() - {path.name for path in info_by_path}:
        issues.append({"clip": name, "issue": "truth has no matching video"})
    log["totals"].update(
        input_clips=len(info_by_path),
        completed_clips=len(summaries),
        profile_count=len(refs),
        detected_cycles=sum(r["detected_cycles"] for r in summaries),
        failures=failures,
    )
    if mode == "calibrate":
        log["issues"] = issues
        if failures:
            raise ValueError(f"Calibration failed. See {stem.with_suffix('.json')}")
        return stem.with_suffix(".json")
    eligible = set()
    for row in summaries:
        digest = log["fingerprints"][str((VIDEO_DIR / row["clip"]).resolve())]
        row["calibration_source"] = digest in ref_hashes
        row["eligible_validation"] = not (
            row["partial_clip"] or row["calibration_source"]
        )
        if row["eligible_validation"]:
            eligible.add(row["clip"])
    for row in events:
        row["eligible_validation"] = row["clip"] in eligible
    scored = [r for r in events if r["eligible_validation"]]
    reviewed = [r for r in summaries if r["reviewed"] and r["eligible_validation"]]
    ride_rows = []
    try:
        ride_rows = compare_visits(events, summaries, closed, issues, ridecheck)
    except (KeyError, ValueError) as e:  # Keep the clip results; report the failure.
        failures += 1
        log["ridecheck_traceback"] = traceback.format_exc()
        issues.append({"issue": "Ridecheck comparison failed", "detail": str(e)})
    hits = sum(r["matched_cycles"] for r in reviewed)
    manual = sum(r["truth_cycles"] for r in reviewed)
    detected = sum(r["detected_cycles"] for r in reviewed)
    overall = {
        "reviewed_clips": len(reviewed),
        "truth_cycles": manual,
        "detected_on_reviewed_clips": detected,
        "matches": hits,
        "misses": manual - hits,
        "false_alarms": detected - hits,
        "precision": safe_div(hits, detected),
        "recall": safe_div(hits, manual),
    }
    for k in ("open", "close"):
        censored = "left_censored" if k == "open" else "right_censored"
        errors = [
            abs(r[k + "_s_error"])
            for r in scored
            if k + "_s_error" in r and not r[censored]
        ]
        overall[k + "_timed_pairs"] = len(errors)
        for percentile in (50, 90):
            overall[f"{k}_abs_error_p{percentile}_s"] = (
                float(np.percentile(errors, percentile)) if errors else None
            )
    workbook(
        output,
        {
            "Events": (events, ["clip", "door", "match", "open_s", "close_s", *COUNTS]),
            "Trip summary": (
                summaries,
                ["clip", "bus", "trip", "door", "reviewed", "detected_cycles"],
            ),
            "Door accuracy": ([overall], list(overall)),
            "Count accuracy": (
                accuracy(scored),
                [
                    "scope",
                    "count",
                    "breakdown",
                    "group",
                    "observations",
                    "exact_share",
                    "within_one_share",
                    "mae",
                    "bias",
                    "total_ratio",
                ],
            ),
            **(
                {"Ridecheck agreement": (ride_rows, ["status", "door_group"])}
                if RIDECHECK_PATH and mode == "pilot"
                else {}
            ),
            "Reconciliation": (issues, ["clip", "issue", "detail"]),
        },
    )
    if failures or not summaries or (TRUTH_PATH and not reviewed):
        raise ValueError(
            "Input failure, or no reviewed clip eligible for validation (partial and "
            f"calibration-source clips are excluded). See {output}"
        )
    return output


def main(argv=None):
    """Run from a command line or a pasted notebook cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", default=MODE, choices=("template", "calibrate", "doors", "pilot")
    )
    if argv is None and "ipykernel" in sys.modules and "__file__" not in globals():
        argv = []
    args = parser.parse_args(argv)
    try:
        print(run(args.mode))
    except Exception as e:  # noqa: BLE001 - preserve clear notebook/CLI failure.
        traceback.print_exc()
        print(f"Door-camera pilot failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
