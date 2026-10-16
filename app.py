from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from train_model import (
    AREAS,
    DATA_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    ROAD_TYPES,
    TARGET_COLUMN,
    WEATHER_OPTIONS,
    prepare_app_dataset,
    train_and_save,
    training_data_warnings,
    validate_training_data,
)
from video_analysis import (
    DETECTIONS_PATH,
    LOW_TRAFFIC_MAX_DETECTIONS,
    MEDIUM_TRAFFIC_MAX_DETECTIONS,
    PROCESSED_VIDEO_PATH,
    REPORT_PATH,
    SUMMARY_PATH,
    VIDEO_PATH,
    analyze_video,
    build_report_text,
    frame_summary_table,
    load_detection_rows,
    required_assets_available,
    vehicle_count_table,
)


st.set_page_config(
    page_title="TrafficSight AI",
    page_icon="TS",
    layout="wide",
    initial_sidebar_state="expanded",
)


DAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

PAGE_DESCRIPTIONS = {
    "Dashboard": "Explore historical traffic patterns, predictions, and model performance.",
    "Traffic Analysis": "Explore historical records by date, hour, area, and weather.",
    "Vehicle Prediction": "Generate vehicle-count estimates from the trained tabular ML pipeline.",
    "Model Performance": "Inspect evaluation metrics for the tabular vehicle-count model.",
    "Data Upload": "Validate a traffic CSV and retrain the tabular model without leaving the website.",
    "Real-Traffic Demo": "Analyze one prerecorded road-traffic clip with YOLO vehicle detection.",
    "About": "Understand the project architecture, workflow, and deployment path.",
}

PALETTE = {
    "navy": "#102033",
    "muted": "#657084",
    "line": "#dde5ef",
    "teal": "#0f8f8c",
    "blue": "#2563eb",
    "amber": "#d97706",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #0f1f33;
            --ink: #223047;
            --muted: #66758a;
            --line: #e2e8f0;
            --page: #f7f9fc;
            --surface: #ffffff;
            --teal: #0f8f8c;
            --teal-soft: #e9f7f6;
            --blue: #2563eb;
            --amber: #d97706;
        }

        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            letter-spacing: 0;
        }

        .stApp {
            background: var(--page);
            color: var(--ink);
        }

        .block-container {
            padding: 1rem 1.35rem 1.6rem;
            max-width: 1240px;
        }

        h1, h2, h3, h4 {
            color: var(--navy);
            letter-spacing: 0;
        }

        p, li, div, label {
            font-size: 0.92rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
            width: 15.5rem !important;
            min-width: 15.5rem !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1rem 0.8rem;
        }

        [data-testid="stSidebar"] * {
            color: var(--navy);
        }

        [data-testid="stSidebar"] .stRadio > label {
            display: none;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            color: var(--navy);
            border-radius: 8px;
            padding: 0.38rem 0.5rem;
            margin-bottom: 0.08rem;
            border: 1px solid transparent;
            min-height: 2.1rem;
            font-weight: 650;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: var(--teal-soft);
            border-color: #c7e9e6;
        }

        .brand {
            display: flex;
            gap: 0.6rem;
            align-items: center;
            padding: 0 0 0.75rem;
        }

        .brand-mark {
            width: 34px;
            height: 34px;
            border-radius: 8px;
            background: #102033;
            color: white;
            display: grid;
            place-items: center;
            font-weight: 850;
            font-size: 0.86rem;
            letter-spacing: 0;
        }

        .brand-title {
            font-size: 0.98rem;
            font-weight: 850;
            line-height: 1.1;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 0.74rem;
            margin-top: 0.12rem;
        }

        .top-header {
            margin: 0 0 0.7rem;
            padding-bottom: 0.2rem;
        }

        .eyebrow {
            color: var(--teal);
            font-size: 0.72rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.2rem;
        }

        .top-header h1 {
            font-size: 1.75rem;
            line-height: 1.15;
            margin: 0;
            font-weight: 850;
        }

        .top-header p {
            color: var(--muted);
            margin: 0.28rem 0 0;
            max-width: 720px;
            font-size: 0.93rem;
        }

        .section-title {
            color: var(--navy);
            font-size: 0.97rem;
            font-weight: 820;
            margin: 0.85rem 0 0.42rem;
        }

        .section-note {
            color: var(--muted);
            font-size: 0.84rem;
            margin: -0.2rem 0 0.5rem;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.72rem 0.82rem;
            min-height: 92px;
            box-shadow: none;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.7rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: 0.055em;
        }

        .metric-value {
            color: var(--navy);
            font-size: 1.75rem;
            font-weight: 850;
            line-height: 1.05;
            margin-top: 0.28rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.3rem;
        }

        .level-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.62rem;
            font-weight: 800;
            font-size: 0.86rem;
            border: 1px solid transparent;
        }

        .level-low { background: #ecfdf3; color: #067647; border-color: #abefc6; }
        .level-moderate { background: #fffaeb; color: #b54708; border-color: #fedf89; }
        .level-high { background: #fff4ed; color: #c2410c; border-color: #fed7aa; }
        .level-congested { background: #fef3f2; color: #b42318; border-color: #fecdca; }

        .callout {
            background: var(--teal-soft);
            border: 1px solid #c9e9e8;
            border-radius: 8px;
            padding: 0.72rem 0.85rem;
            color: #164e63;
            font-size: 0.86rem;
        }

        .step-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.5rem;
            margin: 0.3rem 0 0.3rem;
        }

        .step-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.68rem;
            min-height: 96px;
        }

        .step-number {
            width: 24px;
            height: 24px;
            display: grid;
            place-items: center;
            background: var(--teal-soft);
            color: var(--teal);
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 850;
            margin-bottom: 0.42rem;
        }

        .step-title {
            color: var(--navy);
            font-weight: 820;
            margin-bottom: 0.25rem;
        }

        .step-text {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.34;
        }

        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: none;
        }

        div[data-testid="stExpander"] details summary p {
            font-weight: 760;
            color: var(--navy);
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 8px;
            border: 1px solid var(--teal);
            background: var(--teal);
            color: white;
            font-weight: 780;
            min-height: 2.35rem;
            box-shadow: none;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: #0b7472;
            border-color: #0b7472;
            color: white;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            background: white;
        }

        div[data-testid="stVerticalBlock"] > div:has(.metric-card) {
            gap: 0.55rem;
        }

        div[data-testid="stDateInput"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stTimeInput"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stFileUploader"] label {
            color: var(--navy);
            font-size: 0.8rem;
            font-weight: 700;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        [data-testid="stDateInput"] input,
        [data-testid="stTimeInput"] input {
            border-radius: 8px;
            border-color: var(--line);
            min-height: 2.35rem;
        }

        [data-testid="stSlider"] {
            padding-top: 0.15rem;
        }

        .stDataFrame {
            font-size: 0.84rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .step-row {
                grid-template-columns: 1fr;
            }
            .top-header h1 {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def traffic_level(vehicle_count: float) -> tuple[str, str]:
    if vehicle_count < 450:
        return "Low", "level-low"
    if vehicle_count < 750:
        return "Moderate", "level-moderate"
    if vehicle_count < 1050:
        return "High", "level-high"
    return "Congested", "level-congested"


def level_html(label: str, css_class: str) -> str:
    return f"<span class='level-pill {css_class}'>{label}</span>"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Historical traffic dataset was not found: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    return prepare_app_dataset(raw)


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, object]:
    if not METRICS_PATH.exists():
        train_and_save()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_video_analysis(force: bool = False) -> dict[str, object]:
    return analyze_video(force=force)


@st.cache_data(show_spinner=False)
def load_video_detections() -> pd.DataFrame:
    return load_detection_rows()


def reset_caches() -> None:
    load_data.clear()
    load_model.clear()
    load_metrics.clear()
    load_video_analysis.clear()
    load_video_detections.clear()


def page_header(title: str, subtitle: str, eyebrow: str = "TrafficSight AI") -> None:
    st.markdown(
        f"""
        <div class="top-header">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, note: str | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if note:
        st.markdown(f"<div class='section-note'>{note}</div>", unsafe_allow_html=True)


def file_size_label(path: Path) -> str:
    if not path.exists():
        return "missing"
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.1f} MB"


def video_level_css(level: str) -> str:
    if level == "Low":
        return "level-low"
    if level == "Medium":
        return "level-moderate"
    return "level-high"


def video_analysis_or_message() -> dict[str, object] | None:
    available, missing = required_assets_available()
    if not available:
        st.error("Traffic video analysis cannot run until the fixed video and YOLO model are available.")
        for item in missing:
            st.warning(item)
        st.code(
            "mkdir assets\\videos models\\vision\n"
            "Place the fixed traffic clip at assets\\videos\\traffic_sample.mp4\n"
            "Place the YOLO weights at models\\vision\\yolo11n.pt",
            language="powershell",
        )
        return None

    try:
        with st.spinner("Loading the fixed-video analysis..."):
            return load_video_analysis(False)
    except Exception as exc:
        st.error(f"Video analysis failed: {exc}")
        return None


def video_summary_cards(summary: dict[str, object]) -> None:
    level = str(summary["traffic_level"])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Unique Track IDs", f"{int(summary['unique_vehicle_count']):,}", "Centroid tracking; occlusion may split tracks")
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Traffic Level</div>
                <div class="metric-value">{level_html(level, video_level_css(level))}</div>
                <div class="metric-note">Derived from video detections</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        metric_card("Processed Frames", f"{int(summary['processed_frames']):,}", f"Source-frame step: {summary['frame_step']}")
    with col4:
        metric_card("Peak Frame", f"{int(summary['peak_frame_detections']):,}", f"At {float(summary['peak_time_seconds']):.1f}s")


def ensure_processed_video(summary: dict[str, object]) -> Path | None:
    path = Path(str(summary.get("processed_video", PROCESSED_VIDEO_PATH)))
    if path.exists():
        return path
    st.warning("The analysis summary exists, but the processed video file is missing. Rebuild the analysis to recreate it.")
    return None


def video_interval_table(
    detections: pd.DataFrame,
    frame_summary: pd.DataFrame,
    interval_seconds: int = 5,
) -> pd.DataFrame:
    """Summarize track observations and average simultaneous detections per interval."""
    columns = ["interval_start_seconds", "vehicles_observed", "average_frame_detections", "traffic_level"]
    if frame_summary.empty or not {"time_seconds", "detections"}.issubset(frame_summary.columns):
        return pd.DataFrame(columns=columns)
    frames = frame_summary[["time_seconds", "detections"]].dropna().copy()
    frames["interval_start_seconds"] = (frames["time_seconds"] // interval_seconds * interval_seconds).astype(int)
    interval = frames.groupby("interval_start_seconds")["detections"].mean().rename("average_frame_detections").reset_index()
    if not detections.empty and {"time_seconds", "track_id"}.issubset(detections.columns):
        observed = detections[["time_seconds", "track_id"]].dropna().copy()
        observed["interval_start_seconds"] = (observed["time_seconds"] // interval_seconds * interval_seconds).astype(int)
        unique_tracks = observed.groupby("interval_start_seconds")["track_id"].nunique().rename("vehicles_observed")
        interval = interval.merge(unique_tracks, on="interval_start_seconds", how="left")
    else:
        interval["vehicles_observed"] = 0
    interval["vehicles_observed"] = interval["vehicles_observed"].fillna(0).astype(int)
    interval["traffic_level"] = interval["average_frame_detections"].map(
        lambda count: "Low" if count < LOW_TRAFFIC_MAX_DETECTIONS else "Medium" if count < MEDIUM_TRAFFIC_MAX_DETECTIONS else "High"
    )
    return interval


def page_video_demo() -> None:
    page_header(
        "Real-Traffic Demo",
        "Analyze one prerecorded traffic clip. Results are observed detections, not forecasts or live camera data.",
        "Video Analysis",
    )
    available, missing = required_assets_available()
    if not available:
        st.error("The fixed-video demo is missing required local assets.")
        for item in missing:
            st.warning(item)
        st.code("assets\\videos\\traffic_sample.mp4\nmodels\\vision\\yolo11n.pt", language="text")
        return

    st.video(str(VIDEO_PATH), format="video/mp4")
    st.caption("Sample: assets/videos/traffic_sample.mp4 · YOLO11n COCO vehicle classes")
    if st.button("Analyze Traffic", type="primary", key="analyze_traffic"):
        progress = st.progress(0, text="Preparing video analysis…")

        def update_progress(completed: int, total: int) -> None:
            value = min(100, int(completed / max(total, 1) * 100))
            progress.progress(value, text=f"Analyzing sampled frames… {value}%")

        try:
            with st.spinner("Running vehicle detection and tracking on the sample clip…"):
                summary = analyze_video(force=True, progress_callback=update_progress)
            st.session_state.video_analysis_summary = summary
            st.session_state.video_analysis_error = None
            progress.progress(100, text="Video analysis complete.")
        except Exception as exc:
            st.session_state.video_analysis_error = str(exc)
            st.session_state.pop("video_analysis_summary", None)
            progress.empty()

    if st.session_state.get("video_analysis_error"):
        st.error(f"Video analysis failed: {st.session_state.video_analysis_error}")
    summary = st.session_state.get("video_analysis_summary")
    if not summary:
        st.info("Choose **Analyze Traffic** to run YOLO detection. Results appear after analysis finishes.")
        return

    st.success("Analysis completed from the selected prerecorded video.")
    video_summary_cards(summary)
    detections = load_video_detections()
    interval_df = video_interval_table(detections, frame_summary_table(summary))
    category_df = vehicle_count_table(summary)

    left, right = st.columns([1.4, 1])
    with left:
        section_title("Observed vehicles over time", "Unique track IDs per 5-second interval; level color uses average simultaneous detections in sampled frames.")
        if interval_df.empty:
            st.info("No supported vehicles were detected in this clip.")
        else:
            line = alt.Chart(interval_df).mark_line(point=True, color=PALETTE["teal"]).encode(
                x=alt.X("interval_start_seconds:Q", title="Time in clip (seconds)"),
                y=alt.Y("vehicles_observed:Q", title="Tracked vehicles observed", scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip("interval_start_seconds:Q", title="Interval starts (s)", format=".0f"),
                    alt.Tooltip("vehicles_observed:Q", title="Tracked vehicles"),
                    alt.Tooltip("average_frame_detections:Q", title="Average detections per sampled frame", format=".1f"),
                    alt.Tooltip("traffic_level:N", title="Traffic level"),
                ],
            )
            st.altair_chart(base_chart(line, 280), width="stretch")
            bars = alt.Chart(interval_df).mark_bar().encode(
                x=alt.X("interval_start_seconds:Q", title="Time in clip (seconds)"),
                y=alt.Y("vehicles_observed:Q", title="Tracked vehicles observed"),
                color=alt.Color("traffic_level:N", title="Traffic level", scale=alt.Scale(domain=["Low", "Medium", "High"], range=["#12b76a", "#f79009", "#f04438"])),
                tooltip=["interval_start_seconds:Q", "vehicles_observed:Q", "average_frame_detections:Q", "traffic_level:N"],
            )
            st.altair_chart(base_chart(bars, 180), width="stretch")
    with right:
        section_title("Detected vehicle categories")
        st.dataframe(category_df, width="stretch", hide_index=True)
        if int(summary.get("unique_vehicle_count", 0)):
            most_common = category_df.sort_values("Count", ascending=False).iloc[0]
            st.markdown(
                f"The tracker recorded **{int(summary['unique_vehicle_count'])} unique track IDs**. "
                f"{most_common['Vehicle Type']} was the largest category ({int(most_common['Count'])} tracks). "
                f"Overall measured traffic level: **{summary['traffic_level']}**, based on average simultaneous detections in sampled frames."
            )
        else:
            st.markdown("No supported vehicles were detected in the sampled frames.")

    processed = ensure_processed_video(summary)
    if processed:
        section_title("Annotated video", "Boxes show detector labels, confidence, and the tracker ID assigned in each frame.")
        st.video(str(processed), format="video/mp4")
    st.download_button("Download analysis report", build_report_text(summary), "traffic_video_report.txt", "text/plain")
    if not detections.empty:
        st.download_button(
            "Download detection records",
            detections.to_csv(index=False).encode("utf-8"),
            "traffic_video_detections.csv",
            "text/csv",
        )


def base_chart(chart: alt.Chart, height: int = 320) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_axis(
            labelColor=PALETTE["muted"],
            titleColor=PALETTE["navy"],
            gridColor="#edf1f7",
            domainColor="#d7e0ea",
            tickColor="#d7e0ea",
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor=PALETTE["muted"],
            titleColor=PALETTE["navy"],
            orient="bottom",
        )
        .configure_view(strokeWidth=0)
    )


def filter_bar(df: pd.DataFrame, key_prefix: str = "") -> pd.DataFrame:
    section_title("Filters")
    col1, col2, col3, col4 = st.columns([1.15, 1, 1, 0.9])
    date_range = col1.date_input(
        "Date range",
        value=(df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()),
        min_value=df["timestamp"].dt.date.min(),
        max_value=df["timestamp"].dt.date.max(),
        key=f"{key_prefix}_dates",
    )
    area = col2.multiselect(
        "Junction / area",
        sorted(df["area"].unique()),
        default=sorted(df["area"].unique()),
        key=f"{key_prefix}_area",
    )
    road_type = col3.multiselect(
        "Road type",
        sorted(df["road_type"].unique()),
        default=sorted(df["road_type"].unique()),
        key=f"{key_prefix}_road",
    )
    hour_range = col4.slider(
        "Hour range",
        min_value=0,
        max_value=23,
        value=(0, 23),
        key=f"{key_prefix}_hours",
    )

    filtered = df[df["area"].isin(area) & df["road_type"].isin(road_type)].copy()
    filtered = filtered[(filtered["hour"] >= hour_range[0]) & (filtered["hour"] <= hour_range[1])]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start)
            & (filtered["timestamp"].dt.date <= end)
        ]
    return filtered


def add_predictions(model, rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    result = rows.copy()
    result["predicted_vehicle_count"] = model.predict(result[FEATURE_COLUMNS]).round(0).astype(int)
    result["prediction_error"] = result[TARGET_COLUMN] - result["predicted_vehicle_count"]
    return result


def prediction_input_frame(
    selected_date,
    selected_time: time,
    weather: str,
    road_type: str,
    area: str,
    lane_count: int,
    speed_limit: int,
    accident_reported: bool,
    event_nearby: bool,
) -> pd.DataFrame:
    selected_dt = datetime.combine(selected_date, selected_time)
    return pd.DataFrame(
        [
            {
                "hour": selected_dt.hour,
                "day_of_week": selected_dt.weekday(),
                "month": selected_dt.month,
                "is_weekend": int(selected_dt.weekday() >= 5),
                "weather": weather,
                "road_type": road_type,
                "area": area,
                "lane_count": lane_count,
                "speed_limit": speed_limit,
                "accident_reported": int(accident_reported),
                "event_nearby": int(event_nearby),
            }
        ]
    )


def option_values(df: pd.DataFrame, column: str, fallback: list[str]) -> list[str]:
    if column not in df.columns:
        return fallback
    values = sorted(value for value in df[column].dropna().astype(str).str.strip().unique() if value)
    return values or fallback


def validate_prediction_inputs(input_df: pd.DataFrame, valid_options: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    row = input_df.iloc[0]

    if not 0 <= int(row["hour"]) <= 23:
        errors.append("Choose an hour between 0 and 23.")
    if not 1 <= int(row["lane_count"]) <= 12:
        errors.append("Lane count must be between 1 and 12.")
    if not 5 <= int(row["speed_limit"]) <= 140:
        errors.append("Speed limit must be between 5 and 140.")

    for column, label in {"area": "junction or area", "road_type": "road type", "weather": "weather"}.items():
        if row[column] not in valid_options[column]:
            errors.append(f"Choose a valid {label}.")

    return errors


def show_how_it_works(metrics: dict[str, object]) -> None:
    features = ", ".join(FEATURE_COLUMNS[:5]) + ", and road conditions"
    with st.expander("How the model works", expanded=False):
        st.markdown(
            """
            <div class="step-row">
                <div class="step-card">
                    <div class="step-number">1</div>
                    <div class="step-title">Data</div>
                    <div class="step-text">Traffic observations provide time, location, road, weather, and count fields.</div>
                </div>
                <div class="step-card">
                    <div class="step-number">2</div>
                    <div class="step-title">Processing</div>
                    <div class="step-text">The saved pipeline scales numeric values and encodes categorical fields.</div>
                </div>
                <div class="step-card">
                    <div class="step-number">3</div>
                    <div class="step-title">Learning</div>
                    <div class="step-text">A RandomForest regressor learns traffic patterns from training records.</div>
                </div>
                <div class="step-card">
                    <div class="step-number">4</div>
                    <div class="step-title">Output</div>
                    <div class="step-text">The app estimates vehicle count and maps it to a traffic level.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            - **Predicted vehicle count:** the model's estimated number of vehicles for the selected conditions.
            - **MAE:** the model misses by about `{float(metrics["mae"]):,.1f}` vehicles on average in the evaluation sample.
            - **RMSE:** larger misses are penalized more strongly; current RMSE is `{float(metrics["rmse"]):,.1f}`.
            - **R2:** current R2 is `{float(metrics["r2"]):.3f}`. Values closer to 1 mean the model explains more variation in traffic counts.
            - **Features used:** {features}.
            - **Limitations:** predictions are estimates and can be affected by unusual events, sensor errors, or traffic patterns not represented in training data.
            """
        )


def page_dashboard(df: pd.DataFrame, model, metrics: dict[str, object]) -> None:
    page_header(
        "Traffic Overview",
        "Monitor flow, congestion, and model accuracy across selected traffic conditions.",
        "Dashboard",
    )

    filtered = filter_bar(df, "dash")
    if filtered.empty:
        st.warning("No rows match the selected filters. Adjust the filters to refresh the dashboard.")
        return

    average_count = filtered[TARGET_COLUMN].mean()
    peak_row = filtered.loc[filtered[TARGET_COLUMN].idxmax()]
    level, css_class = traffic_level(average_count)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Average Vehicles", f"{average_count:,.0f}", "Mean for current filters")
    with col2:
        metric_card("Peak Vehicles", f"{peak_row[TARGET_COLUMN]:,.0f}", f"{peak_row['area']} at {peak_row['hour']}:00")
    with col3:
        metric_card("R2 Score", f"{float(metrics['r2']):.3f}", f"MAE {float(metrics['mae']):,.1f} vehicles")
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Traffic Level</div>
                <div class="metric-value">{level_html(level, css_class)}</div>
                <div class="metric-note">Based on average count</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title("Traffic Trend", "Average vehicle count over time.")
    daily = filtered.groupby(filtered["timestamp"].dt.date)[TARGET_COLUMN].mean().reset_index()
    daily.columns = ["date", "average_vehicle_count"]
    trend = (
        alt.Chart(daily)
        .mark_line(point=False, strokeWidth=3, color=PALETTE["teal"])
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("average_vehicle_count:Q", title="Average vehicle count"),
            tooltip=["date:T", alt.Tooltip("average_vehicle_count:Q", format=",.0f", title="Average vehicles")],
        )
    )
    st.altair_chart(base_chart(trend, 305), width="stretch")

    section_title("Hourly Pattern")
    hourly = filtered.groupby("hour", as_index=False)[TARGET_COLUMN].mean()
    chart = (
        alt.Chart(hourly)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=PALETTE["blue"])
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y(f"{TARGET_COLUMN}:Q", title="Average vehicle count"),
            tooltip=["hour:O", alt.Tooltip(f"{TARGET_COLUMN}:Q", format=",.0f", title="Average vehicles")],
        )
    )
    st.altair_chart(base_chart(chart, 220), width="stretch")

    section_title("Recent Observations")
    recent = filtered.sort_values("timestamp", ascending=False).head(12)
    recent = add_predictions(model, recent)
    table = recent[
        [
            "timestamp",
            "area",
            "road_type",
            "weather",
            "vehicle_count",
            "predicted_vehicle_count",
            "prediction_error",
            "traffic_level",
        ]
    ].rename(
        columns={
            "timestamp": "Timestamp",
            "area": "Junction / Area",
            "road_type": "Road Type",
            "weather": "Weather",
            "vehicle_count": "Actual Vehicles",
            "predicted_vehicle_count": "Predicted Vehicles",
            "prediction_error": "Error",
            "traffic_level": "Traffic Level",
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

    show_how_it_works(metrics)


def page_analysis(df: pd.DataFrame) -> None:
    page_header(
        "Traffic Analysis",
        "Find congestion patterns across time, junctions, weather, and road infrastructure.",
        "Analysis",
    )
    filtered = filter_bar(df, "analysis")
    if filtered.empty:
        st.warning("No matching rows were found.")
        return

    section_title("Day and Hour Congestion Heatmap", "Darker cells indicate heavier average vehicle volume.")
    heat = filtered.groupby(["day_name", "hour"], as_index=False)[TARGET_COLUMN].mean()
    heatmap = (
        alt.Chart(heat)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("hour:O", title="Hour"),
            y=alt.Y("day_name:N", title="Day", sort=list(DAY_NAMES.values())),
            color=alt.Color(
                f"{TARGET_COLUMN}:Q",
                title="Avg vehicles",
                scale=alt.Scale(range=["#e0f2fe", "#38bdf8", "#2563eb", "#102033"]),
            ),
            tooltip=["day_name:N", "hour:O", alt.Tooltip(f"{TARGET_COLUMN}:Q", format=",.0f", title="Average vehicles")],
        )
    )
    st.altair_chart(base_chart(heatmap, 360), width="stretch")

    left, right = st.columns(2)
    with left:
        section_title("Road Type Comparison")
        road_df = filtered.groupby("road_type", as_index=False)[TARGET_COLUMN].mean().sort_values(TARGET_COLUMN)
        chart = (
            alt.Chart(road_df)
            .mark_bar(color=PALETTE["teal"], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("road_type:N", title=None, sort="-y"),
                y=alt.Y(f"{TARGET_COLUMN}:Q", title="Average vehicles"),
                tooltip=["road_type:N", alt.Tooltip(f"{TARGET_COLUMN}:Q", format=",.0f", title="Average vehicles")],
            )
        )
        st.altair_chart(base_chart(chart, 300), width="stretch")
    with right:
        section_title("Weather Impact")
        weather_df = filtered.groupby("weather", as_index=False)[TARGET_COLUMN].mean().sort_values(TARGET_COLUMN)
        chart = (
            alt.Chart(weather_df)
            .mark_bar(color=PALETTE["amber"], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("weather:N", title=None, sort="-y"),
                y=alt.Y(f"{TARGET_COLUMN}:Q", title="Average vehicles"),
                tooltip=["weather:N", alt.Tooltip(f"{TARGET_COLUMN}:Q", format=",.0f", title="Average vehicles")],
            )
        )
        st.altair_chart(base_chart(chart, 300), width="stretch")

    section_title("Highest Traffic Records")
    table = filtered.sort_values(TARGET_COLUMN, ascending=False).head(14)
    st.dataframe(
        table[["timestamp", "area", "road_type", "weather", "hour", "vehicle_count", "traffic_level"]].rename(
            columns={
                "timestamp": "Timestamp",
                "area": "Junction / Area",
                "road_type": "Road Type",
                "weather": "Weather",
                "hour": "Hour",
                "vehicle_count": "Vehicle Count",
                "traffic_level": "Traffic Level",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def page_prediction(df: pd.DataFrame, model, metrics: dict[str, object]) -> None:
    page_header(
        "Vehicle Prediction",
        "Enter traffic conditions and generate a model-backed estimate for vehicle count.",
        "Prediction",
    )

    st.markdown(
        """
        <div class="callout">
            The model requires time, weather, road type, junction or area, lane count, speed limit,
            accident status, and nearby-event status. Predictions are estimates, not guaranteed counts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "prediction_form_version" not in st.session_state:
        st.session_state.prediction_form_version = 0

    reset_col, _ = st.columns([0.22, 0.78])
    if reset_col.button("Reset Inputs", width="stretch"):
        st.session_state.prediction_form_version += 1
        st.rerun()

    form_version = st.session_state.prediction_form_version
    area_options = option_values(df, "area", AREAS)
    road_type_options = option_values(df, "road_type", ROAD_TYPES)
    weather_options = option_values(df, "weather", WEATHER_OPTIONS)
    valid_options = {"area": area_options, "road_type": road_type_options, "weather": weather_options}

    with st.form("prediction_form"):
        section_title("Prediction Inputs")
        col1, col2, col3 = st.columns(3)
        selected_date = col1.date_input("Date", value=datetime.now().date(), key=f"prediction_date_{form_version}")
        selected_time = col2.time_input(
            "Time",
            value=time(8, 0),
            step=timedelta(minutes=30),
            key=f"prediction_time_{form_version}",
        )
        area = col3.selectbox("Junction / area", area_options, key=f"prediction_area_{form_version}")

        col4, col5, col6 = st.columns(3)
        road_type = col4.selectbox("Road type", road_type_options, key=f"prediction_road_{form_version}")
        weather = col5.selectbox("Weather", weather_options, key=f"prediction_weather_{form_version}")
        lane_count = col6.slider("Lane count", min_value=1, max_value=12, value=4, key=f"prediction_lanes_{form_version}")

        col7, col8, col9 = st.columns(3)
        speed_limit = col7.select_slider(
            "Speed limit",
            options=[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            value=60,
            key=f"prediction_speed_{form_version}",
        )
        accident_reported = col8.checkbox("Accident reported", key=f"prediction_accident_{form_version}")
        event_nearby = col9.checkbox("Event nearby", key=f"prediction_event_{form_version}")

        submitted = st.form_submit_button("Generate Prediction", width="stretch")

    if not submitted:
        st.info("Fill the form and generate a prediction to see the model output and comparison chart.")
        show_how_it_works(metrics)
        return

    input_df = prediction_input_frame(
        selected_date,
        selected_time,
        weather,
        road_type,
        area,
        lane_count,
        speed_limit,
        accident_reported,
        event_nearby,
    )
    input_errors = validate_prediction_inputs(input_df, valid_options)
    if input_errors:
        st.error("Please fix the prediction inputs before trying again.")
        for error in input_errors:
            st.warning(error)
        return

    with st.spinner("Generating prediction from the trained pipeline..."):
        try:
            prediction = float(model.predict(input_df[FEATURE_COLUMNS])[0])
        except Exception as exc:
            st.error(f"The model could not generate a prediction: {exc}")
            return
    label, css_class = traffic_level(prediction)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        metric_card("Predicted Vehicle Count", f"{prediction:,.0f}", "Estimated vehicles for selected inputs")
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Traffic Level</div>
                <div class="metric-value">{level_html(label, css_class)}</div>
                <div class="metric-note">Derived from the predicted count</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        recommendation = "Extend green phase" if prediction >= 750 else "Use normal signal timing"
        metric_card("Operational Note", recommendation, "Simple demonstration recommendation")

    st.markdown(
        f"""
        <div class="callout">
            Under these conditions, TrafficSight AI estimates about <strong>{prediction:,.0f}</strong>
            vehicles. This suggests a <strong>{label.lower()}</strong> traffic state for the selected
            time and junction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison = df[(df["area"] == area) & (df["road_type"] == road_type) & (df["hour"] == input_df.loc[0, "hour"])].copy()
    if not comparison.empty:
        historical_mean = comparison[TARGET_COLUMN].mean()
        comparison_rows = pd.DataFrame(
            {
                "Series": ["Historical average", "Current prediction"],
                "Vehicle Count": [historical_mean, prediction],
            }
        )
        section_title("Historical Comparison", "Compare the new prediction with similar historical observations.")
        chart = (
            alt.Chart(comparison_rows)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Series:N", title=None),
                y=alt.Y("Vehicle Count:Q", title="Vehicles"),
                color=alt.Color("Series:N", scale=alt.Scale(range=[PALETTE["blue"], PALETTE["teal"]]), legend=None),
                tooltip=["Series:N", alt.Tooltip("Vehicle Count:Q", format=",.0f")],
            )
        )
        st.altair_chart(base_chart(chart, 260), width="stretch")

    horizon_rows = []
    start_dt = datetime.combine(selected_date, selected_time)
    for offset in range(12):
        future_dt = start_dt + timedelta(hours=offset)
        row = prediction_input_frame(
            future_dt.date(),
            future_dt.time(),
            weather,
            road_type,
            area,
            lane_count,
            speed_limit,
            accident_reported,
            event_nearby,
        )
        future_errors = validate_prediction_inputs(row, valid_options)
        if future_errors:
            continue
        horizon_rows.append(
            {
                "datetime": future_dt,
                "predicted_vehicle_count": float(model.predict(row[FEATURE_COLUMNS])[0]),
            }
        )
    horizon = pd.DataFrame(horizon_rows)
    if not horizon.empty:
        section_title("12-Hour Prediction Trend")
        chart = (
            alt.Chart(horizon)
            .mark_line(point=True, color=PALETTE["teal"], strokeWidth=3)
            .encode(
                x=alt.X("datetime:T", title="Time"),
                y=alt.Y("predicted_vehicle_count:Q", title="Predicted vehicles"),
                tooltip=["datetime:T", alt.Tooltip("predicted_vehicle_count:Q", format=",.0f", title="Predicted vehicles")],
            )
        )
        st.altair_chart(base_chart(chart, 310), width="stretch")


def page_performance(metrics: dict[str, object]) -> None:
    page_header(
        "Model Performance",
        "Evaluate how well the trained model estimates vehicle counts on held-out data.",
        "Evaluation",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("R2 Score", f"{float(metrics['r2']):.3f}", "Explained variation; closer to 1 is better")
    with col2:
        metric_card("MAE", f"{float(metrics['mae']):,.1f}", "Average absolute error in vehicles")
    with col3:
        metric_card("RMSE", f"{float(metrics['rmse']):,.1f}", "More sensitive to larger misses")
    with col4:
        metric_card("Training Rows", f"{int(metrics['rows']):,}", f"Trained {metrics['trained_on']}")

    eval_df = pd.DataFrame(metrics["evaluation_sample"])
    left, right = st.columns([1.05, 0.95])
    with left:
        section_title("Actual Versus Predicted", "Each dot compares a real evaluation value with the model estimate.")
        scatter = (
            alt.Chart(eval_df)
            .mark_circle(size=54, opacity=0.68, color=PALETTE["teal"])
            .encode(
                x=alt.X("actual:Q", title="Actual vehicles"),
                y=alt.Y("predicted:Q", title="Predicted vehicles"),
                tooltip=[
                    alt.Tooltip("actual:Q", format=",.0f"),
                    alt.Tooltip("predicted:Q", format=",.0f"),
                    alt.Tooltip("error:Q", format=",.1f"),
                ],
            )
        )
        st.altair_chart(base_chart(scatter, 340), width="stretch")
    with right:
        section_title("Feature Influence")
        importance = pd.DataFrame(metrics["feature_importance"])
        bars = (
            alt.Chart(importance.head(12))
            .mark_bar(color=PALETTE["blue"], cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", title=None, sort="-x"),
                tooltip=["feature:N", alt.Tooltip("importance:Q", format=".4f")],
            )
        )
        st.altair_chart(base_chart(bars, 340), width="stretch")

    section_title("Residual Distribution", "Errors near zero indicate better predictions.")
    hist = (
        alt.Chart(eval_df)
        .mark_bar(color=PALETTE["teal"], cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("error:Q", bin=alt.Bin(maxbins=28), title="Prediction error"),
            y=alt.Y("count():Q", title="Records"),
            tooltip=[alt.Tooltip("count():Q", title="Records")],
        )
    )
    st.altair_chart(base_chart(hist, 260), width="stretch")

    show_how_it_works(metrics)


def page_upload(df: pd.DataFrame) -> None:
    page_header(
        "Data Upload",
        "Upload a CSV, validate the schema, and retrain the TrafficSight AI model.",
        "Dataset",
    )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    st.markdown(
        f"""
        <div class="callout">
            Required columns: <strong>{", ".join(required_columns)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([0.95, 1.05])
    with col1:
        section_title("Upload CSV")
        sample = df.head(40).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Sample CSV",
            data=sample,
            file_name="traffic_sight_sample.csv",
            mime="text/csv",
            width="stretch",
        )
        uploaded = st.file_uploader("Choose a traffic dataset", type=["csv"])
    with col2:
        section_title("Current Dataset Snapshot")
        st.dataframe(df.head(10), width="stretch", hide_index=True)

    if uploaded is None:
        if st.session_state.get("training_success"):
            st.success(st.session_state.pop("training_success"))
        st.info("Upload a CSV when you are ready to validate and retrain the model.")
        return

    try:
        upload_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    section_title("Uploaded Preview")
    st.dataframe(upload_df.head(20), width="stretch", hide_index=True)

    errors = validate_training_data(upload_df)
    if errors:
        st.error("The uploaded file needs a few fixes before training.")
        for error in errors:
            st.warning(error)
        return

    warnings = training_data_warnings(upload_df)
    for warning in warnings:
        st.info(warning)

    st.success(f"Validated {len(upload_df):,} rows. The dataset is ready for model training.")
    if st.button("Train Model With Uploaded Data", width="stretch"):
        with st.spinner("Training and evaluating the model..."):
            try:
                metrics = train_and_save(upload_df)
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return
        reset_caches()
        st.session_state.training_success = (
            f"Model retrained successfully. R2: {float(metrics['r2']):.3f}, "
            f"MAE: {float(metrics['mae']):,.1f}"
        )
        st.rerun()


def page_about(metrics: dict[str, object]) -> None:
    page_header(
        "About TrafficSight AI",
        "A portfolio-ready Streamlit machine learning application for traffic flow intelligence.",
        "Project",
    )

    st.markdown(
        """
        <div class="callout">
            TrafficSight AI combines a Streamlit frontend with a scikit-learn RandomForest pipeline.
            The website loads the saved model artifact from the models folder and uses it for live predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        section_title("What This Website Includes")
        st.markdown(
            """
            - Dashboard with filtered KPIs, charts, and recent model estimates
            - Traffic analysis by hour, day, junction, road type, and weather
            - Vehicle prediction form connected to the trained model
            - Model evaluation charts and simple metric explanations
            - CSV upload and retraining workflow
            - Responsive SaaS-style visual design
            """
        )
    with right:
        section_title("Run and Deploy")
        st.code("streamlit run app.py", language="bash")
        st.markdown(
            """
            Deploy online by pushing this folder to GitHub and creating a Streamlit Community Cloud app
            with `app.py` as the main file. Keep `requirements.txt`, `data/`, and `models/` in the repository.
            """
        )

    show_how_it_works(metrics)


def sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="brand">
            <div class="brand-mark">TS</div>
            <div>
                <div class="brand-title">TrafficSight AI</div>
                <div class="brand-subtitle">Traffic intelligence dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Traffic Analysis", "Vehicle Prediction", "Model Performance", "Data Upload", "Real-Traffic Demo", "About"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption(PAGE_DESCRIPTIONS[page])
    if st.sidebar.button("Retrain Demo Model", width="stretch"):
        with st.spinner("Retraining demo model..."):
            train_and_save(load_data())
        reset_caches()
        st.sidebar.success("Demo model retrained.")
        st.rerun()
    st.sidebar.caption("Local command: streamlit run app.py")
    return page


def main() -> None:
    inject_css()
    page = sidebar()
    if page == "Real-Traffic Demo":
        page_video_demo()
        return

    with st.spinner("Loading TrafficSight AI..."):
        try:
            df = load_data()
            model = load_model()
            metrics = load_metrics()
        except Exception as exc:
            st.error(f"Could not load the historical dataset or prediction model: {exc}")
            st.info("Check data/traffic_data.csv and the models artifacts, then refresh the page.")
            return

    if page == "Dashboard":
        page_dashboard(df, model, metrics)
    elif page == "Traffic Analysis":
        page_analysis(df)
    elif page == "Vehicle Prediction":
        page_prediction(df, model, metrics)
    elif page == "Model Performance":
        page_performance(metrics)
    elif page == "Data Upload":
        page_upload(df)
    else:
        page_about(metrics)


if __name__ == "__main__":
    main()
