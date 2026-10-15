from io import StringIO

import pandas as pd
import pytest

from sensors import flag_unusual, load_sensor_csv, make_simulated_csv


def test_load_sensor_csv_sorts_and_reports_rows():
    csv = StringIO(
        "timestamp,turbidity_ntu\n"
        "2026-01-01 10:05:00,22\n"
        "2026-01-01 10:00:00,20\n"
    )
    df, present, report = load_sensor_csv(csv)
    assert present == ["turbidity_ntu"]
    assert list(df["turbidity_ntu"]) == [20, 22]
    assert report["rows_read"] == 2
    assert df["t_s"].iloc[0] == 0


def test_load_sensor_csv_requires_timestamp():
    with pytest.raises(ValueError, match="timestamp"):
        load_sensor_csv(StringIO("turbidity_ntu\n20\n21\n"))


def test_load_sensor_csv_requires_recognised_sensor_column():
    with pytest.raises(ValueError, match="No recognised sensor columns"):
        load_sensor_csv(StringIO("timestamp,unknown\n2026-01-01,1\n"))


def test_simulated_data_is_deterministic_and_labelled_by_api_contract():
    first = make_simulated_csv()
    second = make_simulated_csv()
    assert first == second
    df = pd.read_csv(StringIO(first))
    assert "timestamp" in df.columns
    assert "turbidity_ntu" in df.columns
    assert len(df) > 0


def test_flag_unusual_returns_boolean_series():
    df = pd.DataFrame({"x": [10.0] * 20 + [100.0] + [10.0] * 20})
    flags = flag_unusual(df, "x")
    assert len(flags) == len(df)
    assert flags.dtype == bool
