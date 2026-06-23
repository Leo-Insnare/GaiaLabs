from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import A1_CLASSES, A2_CLASSES


class PreviewVideoWriter:
    def __init__(self, output_path: str | Path, fps: float = 4.0):
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None

    def write(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if self.writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, self.size)
            if not self.writer.isOpened():
                raise RuntimeError(f"preview_video_open_failed: {self.output_path}")
        self.writer.write(frame_bgr)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def annotate_frame(
    frame_bgr: np.ndarray,
    a_detections: list[dict[str, Any]],
    d_detections: list[dict[str, Any]],
    c_result: dict[str, Any],
    observation: dict[str, Any],
) -> np.ndarray:
    image = frame_bgr.copy()

    for det in a_detections:
        label = str(det.get("class_name", ""))
        conf = float(det.get("conf", 0.0))
        bbox = det.get("bbox_xyxy") or []

        if label in A1_CLASSES:
            draw_box(image, bbox, f"A1 {label} {conf:.2f}", (40, 180, 40))
        elif label in A2_CLASSES:
            draw_box(image, bbox, f"A2 {label} {conf:.2f}", (190, 120, 30))

    for det in d_detections:
        label = str(det.get("class_name", ""))
        conf = float(det.get("conf", 0.0))
        bbox = det.get("bbox_xyxy") or []
        draw_box(image, bbox, f"{label} {conf:.2f}", (40, 90, 210))

    c_bbox = c_result.get("c1_bbox")
    if c_bbox:
        text = f"C1 anesthesia {float(c_result.get('c1_conf', 0.0)):.2f}"
        draw_box(image, c_bbox, text, (120, 60, 180))

    draw_status_panel(image, observation, c_result)

    return image


def draw_box(image: np.ndarray, bbox: list[float], label: str, color: tuple[int, int, int]) -> None:
    if len(bbox) != 4:
        return

    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    y_text = max(20, y1 - 6)
    cv2.putText(
        image,
        label,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status_panel(image: np.ndarray, observation: dict[str, Any], c_result: dict[str, Any]) -> None:
    c_reason = str(c_result.get("c1_reason", ""))
    if len(c_reason) > 34:
        c_reason = c_reason[:31] + "..."

    lines = [
        f"t={float(observation.get('timestamp_sec', 0.0)):.1f}s frame={observation.get('frame_idx', 0)}",
        (
            f"A1={int(bool(observation.get('a1_raw')))} "
            f"A2={int(bool(observation.get('a2_raw')))} "
            f"C1={int(bool(observation.get('c1_raw')))} "
            f"D={int(bool(observation.get('d_present_any')))}"
        ),
        (
            f"S1raw={int(bool(observation.get('room_occupied_candidate')))} "
            f"S2raw={int(bool(observation.get('simple_waiting_candidate')))} "
            f"S3raw={int(bool(observation.get('anesthesia_candidate')))} "
            f"S5raw={int(bool(observation.get('device_present_candidate')))}"
        ),
        "CRM states are generated after debounce",
    ]

    if c_reason and not bool(observation.get("c1_raw")):
        lines.append(f"C1 reason: {c_reason}")

    x, y = 12, 22
    box_h = 24 * len(lines) + 10

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.48, image, 0.52, 0, image)

    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def convert_to_h264(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_h264.mp4")

    output_path = Path(output_path)
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        return input_path

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0 or not output_path.exists():
        return input_path

    return output_path