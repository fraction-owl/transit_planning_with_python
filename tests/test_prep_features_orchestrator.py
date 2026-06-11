"""Smoke tests for the drop-folder feature-prep orchestrator (prep_features.py).

Mirrors the validation checklist in the orchestrator brief (§11): synthetic
feature scripts + (optionally zipped) inputs are run end to end, and the emitted
bundles + manifest are round-tripped into the unmodified Part B (fit_model.py).
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from scripts.modeling import fit_model as fm
from scripts.modeling import prep_features as pf


def _write_script(scripts_dir: Path, name: str, body: str) -> Path:
    """Write a synthetic feature script that takes --output-dir and emits a CSV."""
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# Emits a route_id-keyed table.
_ROUTE_SCRIPT = """
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
args, _ = parser.parse_known_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {"route_id": [101, 102, 103], "avg_headway_min": [10.0, 20.0, 30.0],
     "span_hrs": [18.0, 16.0, 14.0]}
).to_csv(out / "headway_span_by_route.csv", index=False)
"""

# Emits a period-keyed table.
_PERIOD_SCRIPT = """
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
args, _ = parser.parse_known_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {"period": ["2024-01", "2024-02"], "gas_price": [3.1, 3.2],
     "unemployment_rate": [4.0, 4.1]}
).to_csv(out / "exogenous_monthly.csv", index=False)
"""

# Exits non-zero without producing anything.
_FAIL_SCRIPT = """
import sys
sys.exit(1)
"""

# Emits an NTD-derived (forbidden) column.
_LEAK_SCRIPT = """
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
args, _ = parser.parse_known_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {"route_id": [101, 102], "ntd_boardings": [5000, 6000]}
).to_csv(out / "leaky.csv", index=False)
"""


def _dirs(tmp_path: Path) -> dict[str, Path]:
    """Return a standard set of orchestrator directories under tmp_path."""
    layout = {
        "scripts": tmp_path / "scripts",
        "input": tmp_path / "input",
        "output": tmp_path / "prepped",
        "work": tmp_path / "work",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _run(layout: dict[str, Path], **kwargs: object) -> list[pf.BundleResult]:
    """Invoke the orchestrator with no registry and no zip extraction by default."""
    return pf.orchestrate(
        scripts_dir=layout["scripts"],
        input_dir=layout["input"],
        output_dir=layout["output"],
        work_dir=layout["work"],
        extract_zips_flag=kwargs.pop("extract_zips_flag", False),
        **kwargs,
    )


def test_collect_and_bundle_by_join_key(tmp_path: Path) -> None:
    """§11.1: two scripts produce one route-keyed and one period-keyed bundle."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_route.py", _ROUTE_SCRIPT)
    _write_script(layout["scripts"], "gen_period.py", _PERIOD_SCRIPT)

    bundles = _run(layout)

    names = {b.filename for b in bundles}
    assert names == {"features__route_id.csv", "features__period.csv"}
    assert (layout["output"] / "features__route_id.csv").exists()
    assert (layout["output"] / "features__period.csv").exists()


def test_manifest_required_fields_and_real_hash(tmp_path: Path) -> None:
    """§11.2: every manifest bundle has the four required fields and a true SHA-256."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_route.py", _ROUTE_SCRIPT)
    _write_script(layout["scripts"], "gen_period.py", _PERIOD_SCRIPT)
    _run(layout)

    manifest = json.loads((layout["output"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundles"]
    for entry in manifest["bundles"]:
        assert isinstance(entry["filename"], str)
        assert isinstance(entry["join_keys"], list)
        assert all(isinstance(k, str) for k in entry["join_keys"])
        assert isinstance(entry["sha256"], str)
        assert isinstance(entry["n_rows"], int)
        # Additive provenance (§8) is present and records the producing script.
        assert isinstance(entry["produced_by"], list) and entry["produced_by"]
        assert entry["produced_by"][0]["exit_code"] == 0

        digest = hashlib.sha256((layout["output"] / entry["filename"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"]


def test_round_trip_panel_anchor_joins_both(tmp_path: Path) -> None:
    """§11.3: a panel anchor (route_id+period) joins both bundles via Part B."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_route.py", _ROUTE_SCRIPT)
    _write_script(layout["scripts"], "gen_period.py", _PERIOD_SCRIPT)
    _run(layout)

    anchor = tmp_path / "anchor_panel.csv"
    pd.DataFrame(
        {
            "route_id": [101, 101, 102, 102],
            "period": ["2024-01", "2024-02", "2024-01", "2024-02"],
            "ntd_boardings": [100, 110, 200, 210],
        }
    ).to_csv(anchor, index=False)

    merged, provenance = fm.assemble_model_table(
        anchor, 0, layout["output"], layout["output"] / "manifest.json", True
    )
    # Both bundles joined: route feature and period feature both present.
    assert "avg_headway_min" in merged.columns
    assert "gas_price" in merged.columns
    assert len(provenance) == 2


def test_round_trip_cross_sectional_anchor_skips_period(tmp_path: Path) -> None:
    """§11.3: a cross-sectional anchor (route_id only) auto-skips the period bundle."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_route.py", _ROUTE_SCRIPT)
    _write_script(layout["scripts"], "gen_period.py", _PERIOD_SCRIPT)
    _run(layout)

    anchor = tmp_path / "anchor_xs.csv"
    pd.DataFrame({"route_id": [101, 102, 103], "ntd_boardings": [100, 200, 300]}).to_csv(
        anchor, index=False
    )

    merged, provenance = fm.assemble_model_table(
        anchor, 0, layout["output"], layout["output"] / "manifest.json", True
    )
    assert "avg_headway_min" in merged.columns  # route bundle joined
    assert "gas_price" not in merged.columns  # period bundle skipped
    assert [name for name, _ in provenance] == ["features__route_id.csv"]


def test_failing_script_is_skipped_run_continues(tmp_path: Path) -> None:
    """§11.4: a non-zero-exit script is skipped and the run still completes."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_route.py", _ROUTE_SCRIPT)
    _write_script(layout["scripts"], "gen_fail.py", _FAIL_SCRIPT)

    bundles = _run(layout)

    assert [b.filename for b in bundles] == ["features__route_id.csv"]


def test_argparse_error_surfaces_reason(tmp_path: Path, caplog) -> None:
    """A script that rejects the passed flags (exit 2) logs the hint + captured log tail."""
    layout = _dirs(tmp_path)
    # Mimics the real scripts: requires a specific flag, so DEFAULT_CMD_TEMPLATE's
    # --input-dir/--output-dir trigger argparse's exit code 2.
    _write_script(
        layout["scripts"],
        "needs_flags.py",
        """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--gtfs-folder", required=True)
        p.parse_args()
        """,
    )
    _write_script(layout["scripts"], "gen_route.py", _ROUTE_SCRIPT)  # so a bundle still results

    with caplog.at_level("WARNING"):
        _run(layout)

    text = caplog.text
    assert "exited 2" in text
    assert "argument error" in text  # the exit-2 hint
    assert "Last log lines" in text  # the captured per-script log tail


def test_forbidden_column_hard_fails_before_write(tmp_path: Path) -> None:
    """§11.5: a script emitting a forbidden column aborts before any bundle is written."""
    layout = _dirs(tmp_path)
    _write_script(layout["scripts"], "gen_route.py", _ROUTE_SCRIPT)
    _write_script(layout["scripts"], "gen_leak.py", _LEAK_SCRIPT)

    with pytest.raises(ValueError, match="forbidden"):
        _run(layout)

    # Pre-write hard-fail: no bundle CSVs exist.
    assert list(layout["output"].glob("features__*.csv")) == []


def test_mixed_type_join_keys_match_after_canonicalization(tmp_path: Path) -> None:
    """§11.6: int route_id in one output and float/str in another still align."""
    layout = _dirs(tmp_path)
    _write_script(
        layout["scripts"],
        "gen_int.py",
        """
        import argparse
        from pathlib import Path
        import pandas as pd

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", required=True)
        args, _ = parser.parse_known_args()
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"route_id": [101, 102], "a_val": [1.0, 2.0]}).to_csv(
            out / "a.csv", index=False
        )
        """,
    )
    _write_script(
        layout["scripts"],
        "gen_float.py",
        """
        import argparse
        from pathlib import Path
        import pandas as pd

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", required=True)
        args, _ = parser.parse_known_args()
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # route_id as float (101.0) on this side; _canonical_key strips the .0.
        pd.DataFrame({"route_id": [101.0, 102.0], "b_val": [3.0, 4.0]}).to_csv(
            out / "b.csv", index=False
        )
        """,
    )

    _run(layout)

    bundle = pd.read_csv(layout["output"] / "features__route_id.csv")
    # One row per route with BOTH value columns populated => keys matched.
    assert len(bundle) == 2
    assert bundle["a_val"].notna().all()
    assert bundle["b_val"].notna().all()


def test_extract_zips_unpacks_into_input_root(tmp_path: Path) -> None:
    """The 'both' input model: a dropped *.zip is extracted (zip-slip guarded)."""
    layout = _dirs(tmp_path)
    # Drop a zip containing a CSV into an input subfolder.
    import zipfile

    sub = layout["input"] / "gtfs"
    sub.mkdir(parents=True, exist_ok=True)
    zip_path = sub / "feed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("stops.txt", "stop_id\n1\n")

    written = pf.extract_zips(layout["input"], None)

    assert (sub / "feed" / "stops.txt").exists()
    assert (sub / "feed") in written
