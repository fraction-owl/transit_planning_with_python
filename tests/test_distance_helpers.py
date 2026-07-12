from __future__ import annotations

import numpy as np
import pytest

from utils.distance_helpers import convert_distance

# ---------------------------------------------------------------------------
# unit conversions
# ---------------------------------------------------------------------------


def test_convert_distance_meters_to_miles() -> None:
    assert convert_distance(1609.344, "meters") == pytest.approx(1.0)


def test_convert_distance_feet_to_miles() -> None:
    assert convert_distance(5280.0, "feet") == pytest.approx(1.0)


def test_convert_distance_km_to_miles() -> None:
    assert convert_distance(1.609344, "km") == pytest.approx(1.0, rel=1e-6)


def test_convert_distance_miles_to_miles_is_identity() -> None:
    assert convert_distance(2.5, "miles") == pytest.approx(2.5)


def test_convert_distance_meters_to_km() -> None:
    assert convert_distance(2500.0, "meters", "km") == pytest.approx(2.5)


def test_convert_distance_feet_to_km() -> None:
    assert convert_distance(1000.0, "feet", "km") == pytest.approx(0.3048)


def test_convert_distance_unit_case_insensitive() -> None:
    assert convert_distance(1609.344, "Meters", "Miles") == pytest.approx(1.0)  # type: ignore[arg-type]


def test_convert_distance_numeric_string_value() -> None:
    assert convert_distance("1609.344", "meters") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# missing / invalid values
# ---------------------------------------------------------------------------


def test_convert_distance_none_returns_none() -> None:
    assert convert_distance(None, "meters") is None


def test_convert_distance_nan_returns_none() -> None:
    assert convert_distance(float("nan"), "meters") is None
    assert convert_distance(np.nan, "meters") is None


def test_convert_distance_empty_string_returns_none() -> None:
    assert convert_distance("", "meters") is None
    assert convert_distance("   ", "meters") is None


def test_convert_distance_non_numeric_returns_none() -> None:
    assert convert_distance("abc", "meters") is None


# ---------------------------------------------------------------------------
# error conditions
# ---------------------------------------------------------------------------


def test_convert_distance_unknown_input_unit_raises() -> None:
    with pytest.raises(ValueError, match="input_unit"):
        convert_distance(1.0, "furlongs")


def test_convert_distance_unknown_output_unit_raises() -> None:
    with pytest.raises(ValueError, match="output_unit"):
        convert_distance(1.0, "meters", "feet")  # type: ignore[arg-type]
