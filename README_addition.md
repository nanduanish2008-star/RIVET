## Automated test checks

The repository includes a pytest suite for video-processing helpers, rule-based investigation decisions, and sensor CSV validation. GitHub Actions runs the suite on Python 3.11 and 3.12 for pushes and pull requests.

Run locally from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

A passing test suite verifies the covered software behaviours only. It does not establish environmental accuracy, validate real-world sensor calibration, or replace field evaluation against labelled video and sensor datasets.
