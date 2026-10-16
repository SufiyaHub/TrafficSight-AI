from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).parent
ASSET_DIR = ROOT / "assets" / "videos"
OUTPUT_DIR = ROOT / "outputs" / "video_analysis"
MODEL_DIR = ROOT / "models" / "vision"

VIDEO_PATH = ASSET_DIR / "traffic_sample.mp4"
VISION_MODEL_PATH = MODEL_DIR / "yolo11n.pt"
PROCESSED_VIDEO_PATH = OUTPUT_DIR / "traffic_sample_detected.mp4"
SUMMARY_PATH = OUTPUT_DIR / "traffic_video_summary.json"
DETECTIONS_PATH = OUTPUT_DIR / "traffic_video_detections.csv"
REPORT_PATH = OUTPUT_DIR / "traffic_video_report.txt"

VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
LOW_TRAFFIC_MAX_DETECTIONS = 4
MEDIUM_TRAFFIC_MAX_DETECTIONS = 9
DISPLAY_LABELS = {
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
}
BOX_COLORS = {
    "car": (15, 143, 140),
    "motorcycle": (217, 119, 6),
    "bus": (37, 99, 235),
    "truck": (180, 35, 24),
}


@dataclass
class Track:
    track_id: int
    label: str
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    first_frame: int
    last_frame: int
    best_confidence: float
    missing_frames: int = 0
    hits: int = 1
    history: list[tuple[float, float]] = field(default_factory=list)


def required_assets_available() -> tuple[bool, list[str]]:
    missing = []
    if not VIDEO_PATH.exists():
        missing.append(f"Missing predefined traffic video: {VIDEO_PATH}")
    if not VISION_MODEL_PATH.exists():
        missing.append(f"Missing YOLO model weights: {VISION_MODEL_PATH}")
    return len(missing) == 0, missing


def _traffic_level(average_detections_per_frame: float) -> str:
    """Classify observed density using average simultaneous vehicles per sampled frame."""
    if average_detections_per_frame >= MEDIUM_TRAFFIC_MAX_DETECTIONS:
        return "High"
    if average_detections_per_frame >= LOW_TRAFFIC_MAX_DETECTIONS:
        return "Medium"
    return "Low"


def _level_from_frame_count(vehicle_count: int) -> str:
    if vehicle_count >= MEDIUM_TRAFFIC_MAX_DETECTIONS:
        return "High"
    if vehicle_count >= LOW_TRAFFIC_MAX_DETECTIONS:
        return "Medium"
    return "Low"


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _assign_tracks(
    detections: list[dict[str, Any]],
    tracks: dict[int, Track],
    next_track_id: int,
    frame_number: int,
    max_distance: float = 95,
    max_missing_frames: int = 9,
) -> tuple[list[dict[str, Any]], int]:
    for track in tracks.values():
        track.missing_frames += 1

    assignments: list[dict[str, Any]] = []
    used_tracks: set[int] = set()

    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        candidates = [
            (track_id, _distance(detection["centroid"], track.centroid))
            for track_id, track in tracks.items()
            if track.label == detection["label"] and track_id not in used_tracks
        ]
        candidates.sort(key=lambda item: item[1])

        if candidates and candidates[0][1] <= max_distance:
            track_id = candidates[0][0]
            track = tracks[track_id]
            track.centroid = detection["centroid"]
            track.bbox = detection["bbox"]
            track.last_frame = frame_number
            track.best_confidence = max(track.best_confidence, detection["confidence"])
            track.missing_frames = 0
            track.hits += 1
            track.history.append(detection["centroid"])
            used_tracks.add(track_id)
        else:
            track_id = next_track_id
            next_track_id += 1
            tracks[track_id] = Track(
                track_id=track_id,
                label=detection["label"],
                centroid=detection["centroid"],
                bbox=detection["bbox"],
                first_frame=frame_number,
                last_frame=frame_number,
                best_confidence=detection["confidence"],
                history=[detection["centroid"]],
            )
            used_tracks.add(track_id)

        assigned = dict(detection)
        assigned["track_id"] = track_id
        assignments.append(assigned)

    stale_ids = [track_id for track_id, track in tracks.items() if track.missing_frames > max_missing_frames]
    for track_id in stale_ids:
        tracks.pop(track_id, None)

    return assignments, next_track_id


def _draw_detection(frame: np.ndarray, detection: dict[str, Any]) -> None:
    x1, y1, x2, y2 = detection["bbox"]
    label = detection["label"]
    color = BOX_COLORS.get(label, (15, 143, 140))
    text = f"#{detection['track_id']} {DISPLAY_LABELS[label]} {detection['confidence']:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_width, text_height = text_size
    cv2.rectangle(frame, (x1, max(0, y1 - text_height - 10)), (x1 + text_width + 8, y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_overlay(frame: np.ndarray, frame_vehicle_count: int, level: str, elapsed_seconds: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (390, 96), (16, 32, 51), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    lines = [
        f"TrafficSight AI | {elapsed_seconds:05.1f}s",
        f"Detected vehicles in frame: {frame_vehicle_count} | Level: {level}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (32, 48 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _detect_frame(model: YOLO, frame: np.ndarray, confidence_threshold: float) -> list[dict[str, Any]]:
    result = model.predict(frame, imgsz=640, conf=confidence_threshold, verbose=False, device="cpu")[0]
    detections: list[dict[str, Any]] = []
    for box in result.boxes or []:
        class_id = int(box.cls[0])
        if class_id not in VEHICLE_CLASS_IDS:
            continue

        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        detections.append(
            {
                "label": VEHICLE_CLASS_IDS[class_id],
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "centroid": centroid,
            }
        )
    return detections


@lru_cache(maxsize=2)
def _load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def analyze_video(
    force: bool = False,
    frame_step: int = 3,
    confidence_threshold: float = 0.28,
    max_frames: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    available, missing = required_assets_available()
    if not available:
        raise FileNotFoundError("; ".join(missing))

    if not force and SUMMARY_PATH.exists() and PROCESSED_VIDEO_PATH.exists() and DETECTIONS_PATH.exists():
        cached = load_video_summary()
        source_stat = VIDEO_PATH.stat()
        model_stat = VISION_MODEL_PATH.stat()
        if (
            cached.get("source_size_bytes") == source_stat.st_size
            and cached.get("source_mtime_ns") == source_stat.st_mtime_ns
            and cached.get("model_size_bytes") == model_stat.st_size
            and cached.get("model_mtime_ns") == model_stat.st_mtime_ns
        ):
            cached.update(
                {
                    "source_video": str(VIDEO_PATH),
                    "processed_video": str(PROCESSED_VIDEO_PATH),
                    "model_path": str(VISION_MODEL_PATH),
                }
            )
            return cached

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    output_fps = max(1.0, source_fps / max(frame_step, 1))

    writer = cv2.VideoWriter(
        str(PROCESSED_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write processed video: {PROCESSED_VIDEO_PATH}")

    model = _load_model(str(VISION_MODEL_PATH))
    tracks: dict[int, Track] = {}
    next_track_id = 1
    rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []

    processed_frames = 0
    frame_number = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_number % frame_step != 0:
            frame_number += 1
            continue
        if max_frames is not None and processed_frames >= max_frames:
            break

        elapsed_seconds = frame_number / source_fps
        detections = _detect_frame(model, frame, confidence_threshold)
        assigned, next_track_id = _assign_tracks(detections, tracks, next_track_id, frame_number)
        level = _level_from_frame_count(len(assigned))

        for detection in assigned:
            _draw_detection(frame, detection)
            x1, y1, x2, y2 = detection["bbox"]
            rows.append(
                {
                    "frame": frame_number,
                    "time_seconds": round(elapsed_seconds, 2),
                    "track_id": detection["track_id"],
                    "vehicle_type": detection["label"],
                    "confidence": round(detection["confidence"], 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        _draw_overlay(frame, len(assigned), level, elapsed_seconds)
        writer.write(frame)
        frame_summaries.append(
            {
                "frame": frame_number,
                "time_seconds": round(elapsed_seconds, 2),
                "detections": len(assigned),
                "traffic_level": level,
            }
        )

        processed_frames += 1
        if progress_callback and (processed_frames % max(1, (max_frames or max(1, total_frames // frame_step)) // 20) == 0):
            progress_callback(processed_frames, max_frames or max(1, total_frames // frame_step))
        frame_number += 1

    capture.release()
    writer.release()

    detection_df = pd.DataFrame(rows)
    if detection_df.empty:
        count_by_type = {label: 0 for label in DISPLAY_LABELS}
        unique_total = 0
    else:
        track_df = detection_df.groupby(["track_id", "vehicle_type"], as_index=False).size()
        count_by_type = (
            track_df.groupby("vehicle_type")["track_id"]
            .nunique()
            .reindex(list(DISPLAY_LABELS.keys()), fill_value=0)
            .astype(int)
            .to_dict()
        )
        unique_total = int(track_df["track_id"].nunique())

    duration_seconds = (total_frames / source_fps) if total_frames else 0
    average_frame_detections = float(np.mean([item["detections"] for item in frame_summaries])) if frame_summaries else 0.0
    peak_frame = max(frame_summaries, key=lambda item: item["detections"], default={"detections": 0, "time_seconds": 0})
    overall_level = _traffic_level(average_frame_detections)

    summary: dict[str, Any] = {
        "source_video": str(VIDEO_PATH),
        "processed_video": str(PROCESSED_VIDEO_PATH),
        "model_path": str(VISION_MODEL_PATH),
        "model_name": "Ultralytics YOLO11n COCO detector",
        "source_size_bytes": VIDEO_PATH.stat().st_size,
        "source_mtime_ns": VIDEO_PATH.stat().st_mtime_ns,
        "model_size_bytes": VISION_MODEL_PATH.stat().st_size,
        "model_mtime_ns": VISION_MODEL_PATH.stat().st_mtime_ns,
        "confidence_threshold": confidence_threshold,
        "frame_step": frame_step,
        "source_fps": round(source_fps, 2),
        "output_fps": round(output_fps, 2),
        "total_source_frames": total_frames,
        "processed_frames": processed_frames,
        "duration_seconds": round(duration_seconds, 2),
        "unique_vehicle_count": unique_total,
        "count_by_type": count_by_type,
        "average_detections_per_processed_frame": round(average_frame_detections, 2),
        "peak_frame_detections": int(peak_frame["detections"]),
        "peak_time_seconds": float(peak_frame["time_seconds"]),
        "traffic_level": overall_level,
        "traffic_level_thresholds": {
            "Low": f"fewer than {LOW_TRAFFIC_MAX_DETECTIONS} vehicles per sampled frame",
            "Medium": f"{LOW_TRAFFIC_MAX_DETECTIONS} to fewer than {MEDIUM_TRAFFIC_MAX_DETECTIONS}",
            "High": f"{MEDIUM_TRAFFIC_MAX_DETECTIONS} or more",
        },
        "frame_summary": frame_summaries,
    }

    detection_df.to_csv(DETECTIONS_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(build_report_text(summary), encoding="utf-8")
    return summary


def load_video_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Video analysis has not been generated: {SUMMARY_PATH}")
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def load_detection_rows() -> pd.DataFrame:
    if not DETECTIONS_PATH.exists():
        return pd.DataFrame(
            columns=["frame", "time_seconds", "track_id", "vehicle_type", "confidence", "x1", "y1", "x2", "y2"]
        )
    return pd.read_csv(DETECTIONS_PATH)


def vehicle_count_table(summary: dict[str, Any]) -> pd.DataFrame:
    count_by_type = summary.get("count_by_type", {})
    total = max(int(summary.get("unique_vehicle_count", 0)), 1)
    rows = []
    for vehicle_type in DISPLAY_LABELS:
        count = int(count_by_type.get(vehicle_type, 0))
        rows.append(
            {
                "Vehicle Type": DISPLAY_LABELS[vehicle_type],
                "Count": count,
                "Percentage": round((count / total) * 100, 1) if total else 0,
            }
        )
    return pd.DataFrame(rows)


def frame_summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(summary.get("frame_summary", []))


def build_report_text(summary: dict[str, Any]) -> str:
    counts = vehicle_count_table(summary)
    lines = [
        "TrafficSight AI Video Analysis Report",
        "=" * 39,
        f"Source video: {summary['source_video']}",
        f"Detection model: {summary['model_name']}",
        f"Processed video: {summary['processed_video']}",
        "",
        "Summary",
        "-------",
        f"Traffic level: {summary['traffic_level']}",
        f"Unique tracked IDs (approximate): {summary['unique_vehicle_count']}",
        f"Processed frames: {summary['processed_frames']} of {summary['total_source_frames']}",
        f"Video duration: {summary['duration_seconds']} seconds",
        f"Average detections per processed frame: {summary['average_detections_per_processed_frame']}",
        f"Peak frame detections: {summary['peak_frame_detections']} at {summary['peak_time_seconds']} seconds",
        "",
        "Vehicle Counts",
        "--------------",
    ]
    for row in counts.to_dict(orient="records"):
        lines.append(f"{row['Vehicle Type']}: {row['Count']} ({row['Percentage']}%)")
    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            (
                "Centroid tracking can split one real vehicle into multiple track IDs after occlusion. TrafficSight AI classified this clip as "
                f"{summary['traffic_level']} based on the average number of simultaneous YOLO vehicle detections "
                "per sampled frame in the predefined video."
            ),
        ]
    )
    return "\n".join(lines)
