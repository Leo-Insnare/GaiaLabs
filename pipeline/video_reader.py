from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass
class SampledFrame:
    frame_idx: int
    source_frame_idx: int
    timestamp_sec: float
    frame_bgr: np.ndarray


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"video_open_failed: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    cap.release()
    return VideoMetadata(str(path), fps, frame_count, duration, width, height)


def iter_sampled_frames(
    video_path: str | Path,
    sampling_fps: float = 1.0,
    mode: str = "seek",
    max_sampled_frames: int | None = None,
) -> Iterator[SampledFrame]:
    if sampling_fps <= 0:
        raise ValueError("sampling_fps must be greater than 0")
    path = Path(video_path)
    if mode == "sequential":
        yield from _iter_sequential(path, sampling_fps, max_sampled_frames)
    else:
        yield from _iter_seek(path, sampling_fps, max_sampled_frames)


def _iter_seek(path: Path, sampling_fps: float, max_sampled_frames: int | None) -> Iterator[SampledFrame]:
    meta = get_video_metadata(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"video_open_failed: {path}")

    interval = 1.0 / sampling_fps
    target_ts = 0.0
    frame_idx = 0
    duration = meta.duration_sec if meta.duration_sec > 0 else float("inf")

    while target_ts <= duration + 1e-6:
        if max_sampled_frames is not None and frame_idx >= max_sampled_frames:
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, target_ts * 1000.0)
        ok, frame = cap.read()

        if not ok:
            break

        source_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1
        timestamp_sec = round(float(target_ts), 6)

        yield SampledFrame(
            frame_idx=frame_idx,
            source_frame_idx=max(source_frame_idx, 0),
            timestamp_sec=timestamp_sec,
            frame_bgr=frame,
        )

        frame_idx += 1
        target_ts += interval

    cap.release()


def _iter_sequential(path: Path, sampling_fps: float, max_sampled_frames: int | None) -> Iterator[SampledFrame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"video_open_failed: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    interval = 1.0 / sampling_fps
    next_ts = 0.0
    frame_idx = 0
    source_frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if timestamp <= 0 and fps > 0:
            timestamp = source_frame_idx / fps
        if timestamp + 1e-9 >= next_ts:
            yield SampledFrame(frame_idx, source_frame_idx, timestamp, frame)
            frame_idx += 1
            next_ts += interval
            if max_sampled_frames is not None and frame_idx >= max_sampled_frames:
                break
        source_frame_idx += 1
    cap.release()