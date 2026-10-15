import io

import cv2
import numpy as np
import pytest

from rivet.controller import investigate
from rivet.sensors import flag_unusual, load_sensor_csv, make_simulated_csv
from rivet.video import analyse, read_frames


def v(t, changed, quality=200.0):
    return {"t": t, "frame_index": int(t), "changed": changed, "quality": quality}


def test_normal_when_nothing_changes():
    assert investigate([v(0, 0.0)])["status"] == "normal"


def test_visual_only_is_candidate():
    assert investigate([v(30, 0.1)], sensor=None)["status"] == "anomaly candidate"


def test_agreement_is_candidate():
    r = investigate([v(30, 0.1)], sensor=[{"t": 40, "flagged": True}])
    assert r["status"] == "anomaly candidate"


def test_conflict_is_inconclusive():
    r = investigate([v(30, 0.1)], sensor=[{"t": 500, "flagged": True}])
    assert r["status"] == "inconclusive"


def test_blurry_peak_is_inconclusive():
    r = investigate([v(0, 0.1, quality=10.0)])
    assert r["status"] == "inconclusive"
    assert r["log"][0]["action"] == "inspect_nearby_frame"


def test_action_limit_respected():
    r = investigate([v(0, 0.1, quality=10.0)], max_actions=1)
    assert len(r["log"]) <= 1


def test_csv_requires_timestamp():
    with pytest.raises(ValueError):
        load_sensor_csv(io.StringIO("ph\n7.1\n"))


def test_csv_reports_bad_timestamps():
    df, present, report = load_sensor_csv(io.StringIO("timestamp,ph\n2026-01-01 10:00,7\nnot-a-date,7.1\n"))
    assert report["bad_timestamps"] == 1
    assert len(df) == 1


def test_spike_is_flagged():
    text = "timestamp,turbidity_ntu\n" + "".join(
        f"2026-01-01 10:{i:02d}:00,{80 if i == 20 else 10}\n" for i in range(40))
    df, _, _ = load_sensor_csv(io.StringIO(text))
    flags = flag_unusual(df, "turbidity_ntu")
    assert bool(flags.iloc[20]) and not bool(flags.iloc[5])


def test_simulated_csv_loads():
    df, present, _ = load_sensor_csv(io.StringIO(make_simulated_csv()))
    assert len(df) > 0 and "turbidity_ntu" in present


def test_synthetic_video_detects_change(tmp_path):
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(40):
        frame = np.zeros((120, 160, 3), np.uint8)
        if i >= 20:
            cv2.rectangle(frame, (40, 30), (120, 90), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    frames, _ = read_frames(path, every_n=5)
    visual = analyse(frames, sensitivity=0.5)
    assert max(x["changed"] for x in visual) > 0.02
    assert visual[0]["changed"] < 0.02


def test_corrupt_video_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        read_frames(str(bad))
