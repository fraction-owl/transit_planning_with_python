"""Smoke tests for the secured-box feature-prep orchestrator (prep_features_private.py).

Two layers:
  * synthetic feature scripts exercise the orchestrator logic in isolation (the
    governance inversion vs the public half, the ignore-intermediate convention,
    join-key grouping, and the hygiene guard);
  * one end-to-end run drives the REAL private scripts (ntd_anchor_builder,
    otp_monthly_tides -> otp_by_route, route_runtime_tides) through the actual
    jobs.private.json registry and the repo TIDES fixtures, catching registry /
    CLI drift.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from scripts.modeling import prep_features_private as pf

STOP_VISITS = Path("tests/fixtures/stop_visits.csv")
TRIPS_PERFORMED = Path("tests/fixtures/trips_performed.csv")
PRIVATE_REGISTRY = Path("scripts/modeling/orchestrator_jobs_private.json")


def _write_script(scripts_dir: Path, name: str, body: str) -> Path:
    """Write a synthetic feature script and return its path."""
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# Emits a route_id-keyed table that INCLUDES the NTD dependent variable. On the
# public side this would be forbidden; on the private side it is expected.
_ANCHOR_SCRIPT = """
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
args, _ = parser.parse_known_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {"route_id": [101, 102, 103], "ntd_boardings": [5000, 6000, 7000],
     "revenue_hours": [800.0, 700.0, 600.0]}
).to_csv(out / "ntd_anchor.csv", index=False)
"""

# Emits a route x month panel (a different join-key signature).
_PANEL_SCRIPT = """
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
args, _ = parser.parse_known_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {"route_id": [101, 101], "month": ["2025-01", "2025-02"],
     "runtime_mean_min": [30.0, 31.0]}
).to_csv(out / "route_runtime_monthly.csv", index=False)
"""


def _dirs(tmp_path: Path) -> dict[str, Path]:
    """Return a standard set of orchestrator directories under tmp_path."""
    layout = {
        "scripts": tmp_path / "scripts",
        "input": tmp_path / "input",
        "output": tmp_path / "private_features",
        "work": tmp_path / "work",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _run(layout: dict[str, Path], **kwargs: object) -> list[pf.BundleResult]:
    """Invoke the orchestrator with no zip extraction by default."""
    return pf.orchestrate(
        scripts_dir=layout["scripts"],
        input_dir=layout["input"],
        output_dir=layout["output"],
        work_dir=layout["work"],
        extract_zips_flag=kwargs.pop("extract_zips_flag", False),
        **kwargs,
    )


def test_dependent_variable_is_allowed_in_a_bundle(tmp_path: Path) -> None:
    """The inversion vs the public half: an NTD column ships instead of aborting."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_anchor.py", _ANCHOR_SCRIPT)

    bundles = _run(layout)

    assert [b.filename for b in bundles] == ["features__route_id.csv"]
    bundle = pd.read_csv(layout["output"] / "features__route_id.csv")
    # The dependent variable crosses into the bundle (no forbidden-column denylist).
    assert "ntd_boardings" in bundle.columns
    assert "revenue_hours" in bundle.columns


def test_groups_by_join_key_signature(tmp_path: Path) -> None:
    """Route-level and route x month outputs land in separate bundles."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_anchor.py", _ANCHOR_SCRIPT)
    _write_script(layout["scripts"], "gen_panel.py", _PANEL_SCRIPT)

    bundles = _run(layout)

    assert {b.filename for b in bundles} == {
        "features__route_id.csv",
        "features__route_id__month.csv",
    }


def test_empty_keepcols_spec_ignores_intermediate(tmp_path: Path) -> None:
    """A registry spec with empty keep_cols drops an intermediate from bundling."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_anchor.py", _ANCHOR_SCRIPT)
    _write_script(layout["scripts"], "gen_panel.py", _PANEL_SCRIPT)

    registry = tmp_path / "jobs.private.json"
    registry.write_text(
        json.dumps(
            {
                "scripts": [
                    {
                        "script": "gen_anchor.py",
                        "cmd": ["{python}", "{script}", "--output-dir", "{output}"],
                        "outputs": [
                            {
                                "file": "ntd_anchor.csv",
                                "join_keys": ["route_id"],
                                "keep_cols": ["ntd_boardings", "revenue_hours"],
                            }
                        ],
                    },
                    {
                        "script": "gen_panel.py",
                        "cmd": ["{python}", "{script}", "--output-dir", "{output}"],
                        "outputs": [
                            {
                                "file": "route_runtime_monthly.csv",
                                "join_keys": [],
                                "keep_cols": [],
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    bundles = _run(layout, registry_path=registry)
    # Only the anchor bundle survives; the panel was registry-marked ignore.
    assert [b.filename for b in bundles] == ["features__route_id.csv"]


def test_hygiene_fails_when_join_key_listed_in_keepcols(tmp_path: Path) -> None:
    """A spec that ships a join key as a value column aborts before writing."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_anchor.py", _ANCHOR_SCRIPT)

    registry = tmp_path / "jobs.private.json"
    registry.write_text(
        json.dumps(
            {
                "scripts": [
                    {
                        "script": "gen_anchor.py",
                        "cmd": ["{python}", "{script}", "--output-dir", "{output}"],
                        "outputs": [
                            {
                                "file": "ntd_anchor.csv",
                                "join_keys": ["route_id"],
                                "keep_cols": ["route_id", "ntd_boardings"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="join key"):
        _run(layout, registry_path=registry)
    assert list(layout["output"].glob("features__*.csv")) == []


def _write_ntd_workbook(path: Path) -> None:
    """Write a minimal NTD monthly workbook matching ntd_anchor_builder's schema.

    Carries all three service days so the wide anchor exercises the weekday /
    saturday / sunday breakout end to end.
    """
    routes = ["101", "202", "303"]
    rows: dict[str, list[object]] = {
        "ROUTE_NAME": [],
        "SERVICE_PERIOD": [],
        "MTH_BOARD": [],
        "MTH_REV_HOURS": [],
        "REV_MILES": [],
        "DAYS": [],
    }
    # (service_period, days, board/hours/miles scale) per weekday/saturday/sunday.
    day_specs = [("Weekday", 22.0, 1.0), ("Saturday", 4.0, 0.4), ("Sunday", 5.0, 0.35)]
    base = {
        "101": (6000.0, 800.0, 500.0),
        "202": (4000.0, 600.0, 400.0),
        "303": (2000.0, 400.0, 300.0),
    }
    for route in routes:
        b, h, m = base[route]
        for period, days, scale in day_specs:
            rows["ROUTE_NAME"].append(route)
            rows["SERVICE_PERIOD"].append(period)
            rows["MTH_BOARD"].append(round(b * scale, 1))
            rows["MTH_REV_HOURS"].append(round(h * scale, 1))
            rows["REV_MILES"].append(round(m * scale, 1))
            rows["DAYS"].append(days)
    pd.DataFrame(rows).to_excel(path, index=False, sheet_name="Sheet1")


def test_end_to_end_real_scripts_via_registry(tmp_path: Path) -> None:
    """Drive the real private scripts through the actual jobs.private.json registry."""
    input_dir = tmp_path / "input"
    (input_dir / "tides").mkdir(parents=True, exist_ok=True)
    shutil.copy(STOP_VISITS, input_dir / "tides" / "stop_visits.csv")
    shutil.copy(TRIPS_PERFORMED, input_dir / "tides" / "trips_performed.csv")
    # The NTD workbooks sit loose in the input root (the registry passes
    # --data-root {input}), unlike the TIDES exports in their topic subfolder.
    _write_ntd_workbook(input_dir / "JULY 2025 NTD.xlsx")

    bundles = pf.orchestrate(
        scripts_dir=Path("scripts"),
        input_dir=input_dir,
        output_dir=tmp_path / "private_features",
        work_dir=tmp_path / "work",
        registry_path=PRIVATE_REGISTRY,
        extract_zips_flag=False,
    )

    names = {b.filename for b in bundles}
    assert "features__route_id.csv" in names

    route_bundle = pd.read_csv(tmp_path / "private_features" / "features__route_id.csv")
    # The route-level table carries the weekday DV, the broken-out saturday/sunday
    # daily averages, and the OTP and runtime features.
    for col in (
        "weekday_avg_ntd_boardings",
        "weekday_avg_revenue_hours",
        "weekday_avg_revenue_miles",
        "saturday_avg_ntd_boardings",
        "sunday_avg_ntd_boardings",
        "weekday_service_days",
        "pct_on_time",
        "runtime_mean_min",
    ):
        assert col in route_bundle.columns, f"missing {col} in {list(route_bundle.columns)}"

    # Weekday boardings average = 6000 / 22 for route 101 (single month).
    r101 = route_bundle[route_bundle["route_id"].astype(str) == "101"].iloc[0]
    assert r101["weekday_avg_ntd_boardings"] == pytest.approx(round(6000.0 / 22.0, 2))
