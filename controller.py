"""Rule-based investigation controller. Every decision is an explicit, logged rule."""

TRIGGER = 0.02       # changed-pixel fraction counted as a visual candidate
MIN_QUALITY = 50.0   # Laplacian variance below this is treated as a blurry frame
ALIGN_TOL_S = 60.0   # max seconds between visual peak and a sensor flag to count as aligned


def investigate(visual, sensor=None, max_actions=5, align_tol_s=ALIGN_TOL_S):
    """visual: list of {t, frame_index, changed, quality}.
    sensor: None when no sensor data was supplied, else list of {t, flagged}."""
    log = []

    def step(action, reason, result, next_decision):
        log.append({"step": len(log) + 1, "action": action, "reason": reason,
                    "result": result, "next_decision": next_decision})

    peak = max(visual, key=lambda v: v["changed"], default=None)
    strong = peak is not None and peak["changed"] >= TRIGGER
    hits = [s for s in (sensor or []) if s.get("flagged")]
    blurry = False
    aligned = False

    # Rule 1: a blurry peak weakens visual evidence, so look for a sharper neighbouring frame.
    if strong and peak["quality"] < MIN_QUALITY and len(log) < max_actions:
        i = visual.index(peak)
        best = max(visual[max(0, i - 2): i + 3], key=lambda v: v["quality"])
        reason = f"Peak frame is blurry (quality {peak['quality']:.0f} < {MIN_QUALITY:.0f})"
        if best["quality"] >= MIN_QUALITY:
            step("inspect_nearby_frame", reason,
                 f"Sharper frame at t={best['t']} s (quality {best['quality']:.0f})",
                 "Use the sharper frame as the peak")
            peak = best
        else:
            blurry = True
            step("inspect_nearby_frame", reason, "No sharp frame within two samples",
                 "Mark visual evidence as low quality")

    # Rule 2: cross-check the visual peak against sensor flags.
    if len(log) < max_actions:
        if strong and sensor is None:
            step("request_sensor_data", "No sensor readings supplied",
                 "Visual candidate cannot be cross-checked", "Decide on visual evidence alone")
        elif strong:
            near = [h for h in hits if abs(h["t"] - peak["t"]) <= align_tol_s]
            aligned = bool(near)
            step("check_sensor_alignment",
                 f"Compare sensor flags within ±{align_tol_s:.0f} s of the visual peak",
                 f"{len(near)} aligned flag(s), {len(hits)} flag(s) total",
                 "Agreement or conflict decides status")
        elif hits:
            step("check_sensor_alignment", "Sensor flags exist but no visual candidate",
                 f"{len(hits)} sensor flag(s)", "Evidence is one-sided")

    # Decision rules, applied in order.
    if not strong and not hits:
        status, why = "normal", "No visual change above trigger and no sensor flags"
    elif strong and blurry:
        status, why = "inconclusive", "Visual change found but nearby frames are too blurry to trust"
    elif strong and sensor is None:
        status, why = "anomaly candidate", "Visual change only; no sensor data to corroborate it"
    elif strong and aligned:
        status, why = "anomaly candidate", "Visual change and sensor flag agree within tolerance"
    else:
        status, why = "inconclusive", "Visual and sensor evidence conflict or are one-sided"

    recommendations = {
        "normal": "No action needed; keep routine monitoring.",
        "anomaly candidate": "Inspect the highlighted area on site and confirm with independent measurements before drawing conclusions.",
        "inconclusive": "Recommend human inspection and collection of more frames or sensor data.",
    }
    return {"status": status, "rationale": why,
            "recommendation": recommendations[status], "log": log}
