from pathlib import Path

import pandas as pd
import pytest

# Import the script module
from scripts.operations_tools import otp_by_timepoint_from_avl


@pytest.fixture
def tides_input_csv(tmp_path: Path) -> Path:
    """Creates a temporary TIDES-style CSV input file."""
    # Load fixture
    fixture_path = Path("tests/fixtures/stop_visits.csv")
    df = pd.read_csv(fixture_path)

    # Add simulated joined columns
    # Pattern is PAT_30_WB or PAT_30_EB.
    df["pattern_id"] = df["pattern_id"].map({"shp-101-01": "PAT_30_WB", "shp-101-51": "PAT_30_EB"})
    df = df.dropna(subset=["pattern_id"])
    df["route_id"] = "30"
    df["direction_id"] = df["pattern_id"].apply(lambda x: "0" if "WB" in x else "1")

    # Ensure TIDES columns are present
    # They should be there: schedule_relationship, service_date, actual_departure_time, ...

    input_path = tmp_path / "tides_input.csv"
    df.to_csv(input_path, index=False)
    return input_path


def test_tides_data_processing(tides_input_csv: Path, tmp_path: Path) -> None:
    """Test that the script correctly processes TIDES-style data."""
    output_dir = tmp_path / "output"

    # Arguments for the script (passed explicitly; strict parsing would
    # otherwise see pytest's own argv).
    argv = [
        "--input",
        str(tides_input_csv),
        "--outdir",
        str(output_dir),
        "--start-month",
        "2025-03",
        "--end-month",
        "2025-03",
    ]

    assert otp_by_timepoint_from_avl.main(argv) == 0

    # Verification
    assert output_dir.exists()

    # Check for variation index
    variation_index = output_dir / "variation_index.csv"
    assert variation_index.exists()

    df_var = pd.read_csv(variation_index)
    # Expect rows for PAT_30_WB and PAT_30_EB
    assert "PAT_30_WB" in df_var["Variation"].to_numpy()
    assert "PAT_30_EB" in df_var["Variation"].to_numpy()

    # Check for specific output file
    # Format: {route}_{direction}_{variation_slug}_n{count}_pct.csv
    # Direction is "0" or "1" (not normalized to text)

    # Get N for one variation
    wb_row = df_var[df_var["Variation"] == "PAT_30_WB"].iloc[0]
    n_wb = int(wb_row["N"])

    expected_pct_file = output_dir / f"30_0_PAT_30_WB_n{n_wb}_pct.csv"
    assert expected_pct_file.exists()

    # Read output and verify some content
    df_pct = pd.read_csv(expected_pct_file)
    assert "Year-Month" in df_pct.columns
    assert "2025-03" in df_pct["Year-Month"].to_numpy()


# ---------------------------------------------------------------------------
# add_year_month_column
# ---------------------------------------------------------------------------


def _months(**cols: list) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"Route": ["101"] * n, "Direction": ["NORTHBOUND"] * n, **cols})


def test_year_month_from_date_without_month_column() -> None:
    df = _months(Date=["2025-03-04", "2025-04-09"])
    out = otp_by_timepoint_from_avl.add_year_month_column(df)
    assert out["Year-Month"].tolist() == ["2025-03", "2025-04"]


def test_partial_year_month_is_filled_from_date() -> None:
    df = _months(**{"Year-Month": ["2025-03", ""], "Date": ["", "2025-04-09"]})
    out = otp_by_timepoint_from_avl.add_year_month_column(df)
    assert out["Year-Month"].tolist() == ["2025-03", "2025-04"]


def test_month_backfill_uses_the_one_year_that_month_was_seen() -> None:
    df = _months(Date=["2025-03-04", "", "2025-04-01"], Month=["Mar", "March", "Apr"])
    out = otp_by_timepoint_from_avl.add_year_month_column(df)
    assert out["Year-Month"].tolist() == ["2025-03", "2025-03", "2025-04"]


@pytest.mark.parametrize(
    ("dates", "months", "reason"),
    [
        # April never appears on a dated row: no year to take.
        (["2025-03-04", ""], ["Mar", "Apr"], "no dated row"),
        # March appears in two years: the undated March row is ambiguous.
        (["2024-03-04", "2025-03-04", ""], ["Mar", "Mar", "Mar"], "more than one year"),
    ],
)
def test_month_backfill_rejects_unsupported_or_ambiguous_years(
    dates: list, months: list, reason: str
) -> None:
    df = _months(Date=dates, Month=months)
    with pytest.raises(SystemExit, match=reason):
        otp_by_timepoint_from_avl.add_year_month_column(df)
