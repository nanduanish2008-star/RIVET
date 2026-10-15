# RIVET — Environmental Incident Investigator (prototype)

Observe. Investigate. Verify.

RIVET analyses a short video for visual changes against a baseline, cross-checks them with
sensor CSV readings, runs a transparent rule-based investigation, and exports a report.
It does not identify pollutants or sources.

## Setup (Windows)

Use Python 3.10 to 3.12. Open PowerShell in this folder, then:

    py -3.11 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

## Run tests

    python -m pytest -q

## Launch

    streamlit run app.py

Your browser opens at http://localhost:8501. Use "Use simulated demo data" on the Sensor
Analysis page for a labelled demo dataset.

## How to use

1. Video Investigation: upload an mp4/avi/mov (max 200 MB), press Analyse, then inspect frames.
2. Sensor Analysis: upload a CSV with a `timestamp` column and any of `turbidity_ntu`,
   `water_level_cm`, `ph`, `temperature_c`, or load the simulated demo.
3. Investigation & Report: set the sensor time offset, review the status and timeline,
   download the JSON report or timeline CSV.

## Method

- Baseline: pixel-wise median of the first 5 sampled frames (grayscale, blurred).
- Change: absolute difference thresholded at a sensitivity-dependent level (60 to 20 grey levels).
- Candidate: a frame whose changed-pixel fraction is at least 2%.
- Sharpness: variance of the Laplacian. Below 50 is treated as blurry.
- Sensor flags: a reading more than 3.5 robust standard deviations from its local median.
- Controller: fixed rules, logged in order. Status is normal, anomaly candidate, or inconclusive.

## Limitations

- Lighting, reflections, turbidity, and camera movement can all create visual change.
- Colour is not interpreted chemically; no chemical identification is attempted.
- An event present in the first sampled frames will be absorbed into the baseline.
- Sensor flags are statistical outliers, not calibrated limit exceedances.
- Timestamp alignment uses a manual offset only.

## Not yet built

Optical flow, region-of-interest selection, image-sequence upload, the evaluation lab
(precision/recall against labelled data), AWS S3/CloudWatch integration, PDF export,
ESP32 interface, and the Overview / Evidence Explorer / Settings pages.

## Not verified

The code has been syntax-checked only. It has not been run against a real video or a
browser session, and the tests have not been executed on Windows.
