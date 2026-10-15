import cv2
import numpy as np
import pytest

from video import analyse, frame_diff, make_baseline, pixel_threshold, read_frames


def make_frame(gray, t=0.0, frame_index=0):
    gray = np.asarray(gray, dtype=np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return {"t": t, "frame_index": frame_index, "bgr": bgr, "gray": gray}


def test_pixel_threshold_maps_sensitivity_range():
    assert pixel_threshold(0.0) == 60
    assert pixel_threshold(1.0) == 20
    assert pixel_threshold(0.5) == 40


def test_frame_diff_detects_large_bright_region():
    base = np.zeros((80, 80), dtype=np.uint8)
    current = base.copy()
    current[20:60, 20:60] = 255

    diff, mask = frame_diff(current, base, 20)

    assert diff.shape == base.shape
    assert mask.shape == base.shape
    assert np.count_nonzero(mask) > 0


def test_make_baseline_uses_first_sampled_frames():
    frames = [
        make_frame(np.zeros((16, 16), dtype=np.uint8)),
        make_frame(np.full((16, 16), 20, dtype=np.uint8)),
        make_frame(np.full((16, 16), 40, dtype=np.uint8)),
    ]
    baseline = make_baseline(frames, n=3)
    assert baseline.shape == (16, 16)
    # Median of 0, 20, 40 is 20; blur should preserve a constant image.
    assert np.all(baseline == 20)


def test_analyse_returns_expected_fields():
    baseline = np.zeros((80, 80), dtype=np.uint8)
    changed = baseline.copy()
    changed[20:60, 20:60] = 255
    frames = [make_frame(baseline, 0.0, 0), make_frame(changed, 1.0, 15)]

    result = analyse(frames, sensitivity=1.0)

    assert len(result) == 2
    assert {"t", "frame_index", "changed", "boxes", "quality", "brightness"} <= set(result[0])
    assert result[1]["changed"] > result[0]["changed"]
    assert isinstance(result[1]["boxes"], list)


def test_read_frames_rejects_non_video(tmp_path):
    path = tmp_path / "not-a-video.mp4"
    path.write_bytes(b"not a real video")
    with pytest.raises(ValueError):
        read_frames(str(path))
