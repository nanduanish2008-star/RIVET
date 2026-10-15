import datetime as dt
import io
import json
import os
import tempfile

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from controller import TRIGGER, investigate
from sensors import FIELDS, flag_unusual, load_sensor_csv, make_simulated_csv
from video import MAX_MB, analyse, frame_diff, make_baseline, pixel_threshold, read_frames

SIM_LABEL = "SIMULATED DATA — NOT LIVE SENSOR MEASUREMENTS"
LIMITATIONS = [
    "Visual changes may come from lighting, reflections, turbidity, or camera movement.",
    "The baseline uses the first sampled frames; an event present from the start will be missed.",
    "This tool does not identify pollutants or sources.",
    "Sensor flags are statistical outliers, not calibrated threshold breaches.",
]

st.set_page_config(page_title="RIVET", layout="wide")
st.title("RIVET")
st.caption("Environmental Incident Investigator · Observe. Investigate. Verify.")
st.warning("Prototype. Results are evidence to review, not verified conclusions.")

ss = st.session_state
section = st.sidebar.radio("Section", ["Video Investigation", "Sensor Analysis", "Investigation & Report"])

if section == "Video Investigation":
    st.header("Video Investigation")
    up = st.file_uploader("Upload a video (mp4, avi, mov)", type=["mp4", "avi", "mov"])
    every = st.slider("Sample every N frames", 5, 60, 15)
    if up is not None and up.size > MAX_MB * 1024 * 1024:
        st.error(f"File is larger than {MAX_MB} MB.")
    elif up is not None and st.button("Analyse video"):
        suffix = os.path.splitext(up.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(up.getvalue())
            path = tmp.name
        try:
            with st.spinner("Decoding and sampling frames..."):
                ss.frames, ss.frames_read = read_frames(path, every_n=every)
            ss.video_name = up.name
            ss.incident_id = f"RIVET-{dt.datetime.now(dt.timezone.utc):%Y%m%d%H%M%S}"
        except ValueError as exc:
            st.error(str(exc))
        finally:
            os.remove(path)

    if not ss.get("frames"):
        st.info("No analysed video yet. Upload a short clip and press Analyse.")
    else:
        frames = ss.frames
        sensitivity = st.slider("Visual sensitivity", 0.0, 1.0, 0.5)
        visual = analyse(frames, sensitivity)
        ss.visual = visual
        st.success(f"Processed {len(frames)} sampled frames ({ss.frames_read} frames read) from {ss.video_name}.")

        fig = go.Figure(go.Scatter(x=[v["t"] for v in visual], y=[v["changed"] for v in visual],
                                   mode="lines+markers", name="Changed fraction"))
        fig.add_hline(y=TRIGGER, line_dash="dot", annotation_text="candidate trigger")
        fig.update_layout(template="plotly_dark", height=300,
                          xaxis_title="Time (s)", yaxis_title="Changed pixel fraction")
        st.plotly_chart(fig, use_container_width=True)

        idx = st.slider("Inspect frame", 0, len(frames) - 1, 0)
        f, v = frames[idx], visual[idx]
        base = make_baseline(frames)
        diff, _ = frame_diff(f["gray"], base, pixel_threshold(sensitivity))
        overlay = f["bgr"].copy()
        for x, y, w, h in v["boxes"]:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
        c1, c2, c3, c4 = st.columns(4)
        c1.image(cv2.cvtColor(f["bgr"], cv2.COLOR_BGR2RGB), caption="Original")
        c2.image(base, caption="Baseline")
        c3.image(diff, caption="Difference")
        c4.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Candidate regions")
        st.caption(f"t = {v['t']} s · changed {v['changed']:.2%} · regions {len(v['boxes'])} · "
                   f"sharpness {v['quality']:.0f} · brightness {v['brightness']:.0f}")

        st.subheader("Observed changes")
        changes = [{"t (s)": x["t"], "changed %": round(x["changed"] * 100, 2), "regions": len(x["boxes"])}
                   for x in visual if x["changed"] >= TRIGGER]
        if changes:
            st.dataframe(pd.DataFrame(changes), use_container_width=True)
        else:
            st.write("No frames exceeded the candidate trigger at this sensitivity.")

elif section == "Sensor Analysis":
    st.header("Sensor Analysis")
    up = st.file_uploader("Sensor CSV (timestamp plus any of: " + ", ".join(FIELDS) + ")", type=["csv"])
    if up is not None:
        ss.sensor_src = (up.getvalue().decode("utf-8", "replace"), "UPLOADED", up.name)
    if st.button("Use simulated demo data"):
        ss.sensor_src = (make_simulated_csv(), SIM_LABEL, "simulated_sensors.csv")

    if not ss.get("sensor_src"):
        st.info("Upload a sensor CSV or load the simulated demo.")
    else:
        text, label, name = ss.sensor_src
        st.caption(f"Source: {label} · {name}")
        try:
            df, present, report = load_sensor_csv(io.StringIO(text))
        except ValueError as exc:
            st.error(str(exc))
            ss.sensor = None
        else:
            flags = pd.DataFrame({f: flag_unusual(df, f) for f in present})
            ss.sensor = {"df": df, "flags": flags, "label": label, "name": name}
            st.json(report)
            st.caption("Flagged = more than 3.5 robust standard deviations from the local median (15-sample window).")
            for f in present:
                fig = go.Figure(go.Scatter(x=df["timestamp"], y=df[f], mode="lines", name=f))
                m = flags[f]
                fig.add_trace(go.Scatter(x=df.loc[m, "timestamp"], y=df.loc[m, f], mode="markers",
                                         name="flagged", marker=dict(color="red", size=8)))
                fig.update_layout(template="plotly_dark", height=260, title=f)
                st.plotly_chart(fig, use_container_width=True)

else:
    st.header("Investigation & Report")
    if not ss.get("visual"):
        st.info("Analyse a video first.")
    else:
        offset = st.number_input("Sensor time offset vs video start (s)", value=0.0, step=1.0)
        sensor, label, name = None, "none", "none"
        if ss.get("sensor"):
            s = ss.sensor
            any_flag = s["flags"].any(axis=1)
            sensor = [{"t": float(t) + offset, "flagged": bool(h)}
                      for t, h in zip(s["df"]["t_s"], any_flag)]
            label, name = s["label"], s["name"]

        res = investigate(ss.visual, sensor)
        st.metric("Status", res["status"])
        st.write(res["rationale"])
        st.write(f"**Recommendation:** {res['recommendation']}")

        st.subheader("Investigation timeline")
        st.dataframe(pd.DataFrame(res["log"]), use_container_width=True)

        report = {
            "incident_id": ss.incident_id,
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "video_file": ss.video_name,
            "sensor_source": label,
            "sensor_file": name,
            "sensor_time_offset_s": offset,
            "status": res["status"],
            "rationale": res["rationale"],
            "recommendation": res["recommendation"],
            "visual_candidates": [{"t": x["t"], "frame_index": x["frame_index"],
                                   "changed": round(x["changed"], 4), "regions": x["boxes"]}
                                  for x in ss.visual if x["changed"] >= TRIGGER],
            "investigation_log": res["log"],
            "limitations": LIMITATIONS,
        }
        c1, c2 = st.columns(2)
        c1.download_button("Download report (JSON)", json.dumps(report, indent=2, default=str),
                           f"{ss.incident_id}.json", "application/json")
        c2.download_button("Download timeline (CSV)", pd.DataFrame(res["log"]).to_csv(index=False),
                           f"{ss.incident_id}_timeline.csv", "text/csv")
