from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from config import CRM_STATUS_MAP, D_CLASS_MAP, RuntimeConfig


@dataclass
class EventSegment:
    event_type: str
    start_sec: float
    end_sec: float
    duration_sec: float
    start_frame: int
    end_frame: int
    supporting_variables: dict[str, Any]
    confidence_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "start_sec": round(float(self.start_sec), 3),
            "end_sec": round(float(self.end_sec), 3),
            "duration_sec": round(float(self.duration_sec), 3),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "supporting_variables": self.supporting_variables,
            "confidence_summary": self.confidence_summary,
        }


class DebouncedEventBuilder:
    def __init__(
        self,
        event_type: str,
        start_sec: float,
        end_sec: float,
        predicate: Callable[[pd.Series], bool],
        support_fn: Callable[[pd.DataFrame], dict[str, Any]],
        confidence_fn: Callable[[pd.DataFrame], dict[str, Any]],
    ):
        self.event_type = event_type
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)
        self.predicate = predicate
        self.support_fn = support_fn
        self.confidence_fn = confidence_fn

    def build(self, df: pd.DataFrame) -> list[EventSegment]:
        if df.empty:
            return []

        events: list[EventSegment] = []
        candidate_start_ts: float | None = None
        candidate_start_frame: int | None = None
        active = False
        active_start_ts: float | None = None
        active_start_frame: int | None = None
        false_start_ts: float | None = None
        false_start_frame: int | None = None
        last_true_ts: float | None = None
        last_true_frame: int | None = None

        for _, row in df.iterrows():
            ts = float(row["timestamp_sec"])
            frame_idx = int(row["frame_idx"])
            is_true = bool(self.predicate(row))

            if is_true:
                if candidate_start_ts is None:
                    candidate_start_ts = ts
                    candidate_start_frame = frame_idx
                last_true_ts = ts
                last_true_frame = frame_idx
                false_start_ts = None
                false_start_frame = None
                if not active and ts - candidate_start_ts >= self.start_sec:
                    active = True
                    active_start_ts = candidate_start_ts
                    active_start_frame = candidate_start_frame
                continue

            if candidate_start_ts is not None and not active:
                candidate_start_ts = None
                candidate_start_frame = None
                last_true_ts = None
                last_true_frame = None
                continue

            if active:
                if false_start_ts is None:
                    false_start_ts = ts
                    false_start_frame = frame_idx
                if ts - false_start_ts >= self.end_sec:
                    end_ts = false_start_ts
                    end_frame = false_start_frame if false_start_frame is not None else frame_idx
                    if active_start_ts is not None and active_start_frame is not None:
                        events.append(self._make_segment(df, active_start_ts, end_ts, active_start_frame, end_frame))
                    active = False
                    active_start_ts = None
                    active_start_frame = None
                    candidate_start_ts = None
                    candidate_start_frame = None
                    false_start_ts = None
                    false_start_frame = None
                    last_true_ts = None
                    last_true_frame = None

        if active and active_start_ts is not None and active_start_frame is not None:
            end_ts = float(last_true_ts if last_true_ts is not None else df.iloc[-1]["timestamp_sec"])
            end_frame = int(last_true_frame if last_true_frame is not None else df.iloc[-1]["frame_idx"])
            events.append(self._make_segment(df, active_start_ts, end_ts, active_start_frame, end_frame))

        return events

    def _make_segment(
        self,
        df: pd.DataFrame,
        start_ts: float,
        end_ts: float,
        start_frame: int,
        end_frame: int,
    ) -> EventSegment:
        duration = max(0.0, float(end_ts - start_ts))
        window = df[(df["timestamp_sec"] >= start_ts) & (df["timestamp_sec"] <= end_ts)].copy()
        return EventSegment(
            event_type=self.event_type,
            start_sec=float(start_ts),
            end_sec=float(end_ts),
            duration_sec=duration,
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            supporting_variables=self.support_fn(window),
            confidence_summary=self.confidence_fn(window),
        )


class CRMStateMachine:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def build_segments(self, observations: pd.DataFrame) -> list[dict[str, Any]]:
        df = observations.sort_values("timestamp_sec").reset_index(drop=True)
        builders = [
            DebouncedEventBuilder(
                CRM_STATUS_MAP["room_occupied"],
                self.config.debounce.s1_start_sec,
                self.config.debounce.s1_end_sec,
                lambda r: bool(r["room_occupied_candidate"]),
                lambda w: self._support(w, ["a1_raw", "a2_raw"]),
                lambda w: self._confidence(w, ["a1_conf", "a2_conf"]),
            ),
            DebouncedEventBuilder(
                CRM_STATUS_MAP["simple_waiting"],
                self.config.debounce.s2_start_sec,
                self.config.debounce.s2_end_sec,
                lambda r: bool(r["simple_waiting_candidate"]),
                lambda w: self._support(w, ["a1_raw", "a2_raw", "c1_raw"]),
                lambda w: self._confidence(w, ["a1_conf", "a2_conf", "c1_conf"]),
            ),
            DebouncedEventBuilder(
                CRM_STATUS_MAP["anesthesia"],
                self.config.debounce.s3_start_sec,
                self.config.debounce.s3_end_sec,
                lambda r: bool(r["anesthesia_candidate"]),
                lambda w: self._support(w, ["c1_raw"]),
                lambda w: self._confidence(w, ["c1_conf"]),
            ),
            DebouncedEventBuilder(
                CRM_STATUS_MAP["room_empty"],
                self.config.debounce.s6_start_sec,
                self.config.debounce.s6_end_sec,
                lambda r: not bool(r["room_occupied_candidate"]),
                lambda w: self._support(w, ["a1_raw", "a2_raw"]),
                lambda w: self._confidence(w, ["a1_conf", "a2_conf"]),
            ),
        ]

        segments: list[dict[str, Any]] = []
        for builder in builders:
            segments.extend([seg.to_dict() for seg in builder.build(df)])

        for class_name in D_CLASS_MAP.values():
            builder = DebouncedEventBuilder(
                f"{CRM_STATUS_MAP['device_present']}:{class_name}",
                self.config.debounce.s5_start_sec,
                self.config.debounce.s5_end_sec,
                lambda r, cn=class_name: cn in _json_list(r.get("d_classes_present", "[]")),
                lambda w, cn=class_name: {"device_class": cn, "d_present_ratio": _presence_ratio(w, cn)},
                lambda w, cn=class_name: {"device_conf_mean": _device_conf_mean(w, cn), "device_conf_max": _device_conf_max(w, cn)},
            )
            segments.extend([seg.to_dict() for seg in builder.build(df)])

        return sorted(segments, key=lambda x: (x["start_sec"], x["event_type"]))

    def build_summary(
        self,
        observations: pd.DataFrame,
        segments: list[dict[str, Any]],
        video_metadata: dict[str, Any],
        model_meta: dict[str, Any],
    ) -> dict[str, Any]:
        def total(event_type: str) -> float:
            return round(sum(float(s["duration_sec"]) for s in segments if s["event_type"] == event_type), 3)

        device_duration: dict[str, float] = {}
        for class_name in D_CLASS_MAP.values():
            prefix = f"{CRM_STATUS_MAP['device_present']}:{class_name}"
            device_duration[class_name] = round(sum(float(s["duration_sec"]) for s in segments if s["event_type"] == prefix), 3)

        a2_duration = _duration_from_candidate(observations, "a2_raw")
        return {
            "room_occupied_total_sec": total(CRM_STATUS_MAP["room_occupied"]),
            "simple_waiting_total_sec": total(CRM_STATUS_MAP["simple_waiting"]),
            "anesthesia_total_sec": total(CRM_STATUS_MAP["anesthesia"]),
            "device_present_duration_by_class_sec": device_duration,
            "a2_personnel_presence_duration_sec": round(float(a2_duration), 3),
            "room_empty_total_sec": total(CRM_STATUS_MAP["room_empty"]),
            "room_turnover_count": int(
                sum(1 for s in segments if s["event_type"] == CRM_STATUS_MAP["room_empty"])
            ),            
            "processed_frame_count": int(len(observations)),
            "video_length_sec": round(float(video_metadata.get("duration_sec", 0.0)), 3),
            "sampling_fps": float(self.config.sampling_fps),
            "model_path_version": model_meta,
            "config": self.config.to_dict(),
        }

    @staticmethod
    def _support(window: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
        if window.empty:
            return {c: 0.0 for c in columns}
        return {c: round(float(window[c].astype(bool).mean()), 4) for c in columns if c in window.columns}

    @staticmethod
    def _confidence(window: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for col in columns:
            if col not in window.columns or window.empty:
                result[col] = {"mean": 0.0, "max": 0.0}
                continue
            vals = pd.to_numeric(window[col], errors="coerce").fillna(0.0)
            result[col] = {"mean": round(float(vals.mean()), 4), "max": round(float(vals.max()), 4)}
        return result


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _json_dict(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _presence_ratio(window: pd.DataFrame, class_name: str) -> float:
    if window.empty:
        return 0.0
    return round(float(window["d_classes_present"].apply(lambda v: class_name in _json_list(v)).mean()), 4)


def _device_conf_mean(window: pd.DataFrame, class_name: str) -> float:
    vals = [_json_dict(v).get(class_name, 0.0) for v in window.get("d_conf_by_class", [])]
    vals = [float(v) for v in vals if float(v) > 0]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _device_conf_max(window: pd.DataFrame, class_name: str) -> float:
    vals = [_json_dict(v).get(class_name, 0.0) for v in window.get("d_conf_by_class", [])]
    vals = [float(v) for v in vals if float(v) > 0]
    return round(max(vals), 4) if vals else 0.0


def _duration_from_candidate(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    total = 0.0
    rows = df.sort_values("timestamp_sec").reset_index(drop=True)
    for idx in range(len(rows) - 1):
        if bool(rows.loc[idx, column]):
            delta = float(rows.loc[idx + 1, "timestamp_sec"] - rows.loc[idx, "timestamp_sec"])
            if delta > 0:
                total += delta
    return total