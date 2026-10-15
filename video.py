"""Video decoding and visual change analysis (OpenCV). Frames are kept at most 640 px wide."""
import cv2
import numpy as np

MAX_MB = 200


def pixel_threshold(sensitivity):
    """Map sensitivity 0..1 to a grey-level difference threshold."""
    return int(round(60 - 40 * sensitivity))


def read_frames(path, every_n=15, max_frames=120, width=640):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(
            "Could not open the video. It may be corrupt or in an unsupported format."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, idx = [], 0

    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break

        if idx % every_n == 0:
            h, w = bgr.shape[:2]
            if w > width:
                bgr = cv2.resize(
                    bgr, (width, int(h * width / w))
                )

            frames.append({
                "t": round(idx / fps, 2),
                "frame_index": idx,
                "bgr": bgr,
                "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
            })

        idx += 1

    cap.release()

    if not frames:
        raise ValueError(
            "No readable frames were found. The video may be corrupt."
        )

    return frames, idx


def make_baseline(frames, n=5):
    """Build a baseline that preserves changes in short video clips."""
    if not frames:
        raise ValueError("At least one frame is required to build a baseline.")

    
if len(frames) < n:
    baseline = frames[0]["gray"]
else:
    stack = np.stack([f["gray"] for f in frames[:n]])
    baseline = np.median(stack, axis=0).astype(np.uint8)
    return cv2.GaussianBlur(baseline, (5, 5), 0)


def frame_diff(gray, base, thresh):
    """Return the absolute difference image and cleaned change mask."""
    cur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(cur, base)

    _, mask = cv2.threshold(
        diff, thresh, 255, cv2.THRESH_BINARY
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    return diff, mask


def analyse(frames, sensitivity=0.5):
    """Summarize visual changes, candidate boxes, sharpness, and brightness."""
    base = make_baseline(frames)
    thr = pixel_threshold(sensitivity)
    out = []

    for f in frames:
        _, mask = frame_diff(f["gray"], base, thr)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = [
            [int(v) for v in cv2.boundingRect(c)]
            for c in contours
            if cv2.contourArea(c) >= 200
        ]

        out.append({
            "t": f["t"],
            "frame_index": f["frame_index"],
            "changed": float(np.count_nonzero(mask)) / mask.size,
            "boxes": boxes,
            "quality": float(
                cv2.Laplacian(f["gray"], cv2.CV_64F).var()
            ),
            "brightness": float(f["gray"].mean()),
        })

    return out
