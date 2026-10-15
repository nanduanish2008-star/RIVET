from controller import investigate


def v(t, changed, quality=100.0, frame_index=0):
    return {"t": t, "frame_index": frame_index, "changed": changed, "quality": quality}


def test_no_visual_or_sensor_flags_is_normal():
    result = investigate([v(0, 0.001), v(1, 0.005)], sensor=[])
    assert result["status"] == "normal"


def test_visual_candidate_without_sensor_is_candidate():
    result = investigate([v(0, 0.0), v(1, 0.05)], sensor=None)
    assert result["status"] == "anomaly candidate"
    assert any(row["action"] == "request_sensor_data" for row in result["log"])


def test_aligned_sensor_flag_supports_candidate_not_proof():
    result = investigate(
        [v(0, 0.0), v(10, 0.05)],
        sensor=[{"t": 12, "flagged": True}],
        align_tol_s=60,
    )
    assert result["status"] == "anomaly candidate"
    assert "agree" in result["rationale"].lower()


def test_unaligned_sensor_and_visual_evidence_is_inconclusive():
    result = investigate(
        [v(0, 0.0), v(10, 0.05)],
        sensor=[{"t": 500, "flagged": True}],
        align_tol_s=60,
    )
    assert result["status"] == "inconclusive"


def test_blurry_visual_candidate_is_inconclusive_when_neighbours_blurry():
    result = investigate(
        [v(0, 0.0, 10), v(1, 0.05, 12), v(2, 0.0, 8)],
        sensor=None,
    )
    assert result["status"] == "inconclusive"


def test_investigation_log_is_sequential():
    result = investigate([v(0, 0.0), v(1, 0.05)], sensor=None)
    steps = [row["step"] for row in result["log"]]
    assert steps == list(range(1, len(steps) + 1))
