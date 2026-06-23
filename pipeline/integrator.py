from __future__ import annotations

import json
from typing import Any

from config import CRM_STATUS_MAP, RuntimeConfig


OBSERVATION_COLUMNS = [
    "frame_idx",
    "source_frame_idx",
    "timestamp_sec",
    "a1_raw",
    "a1_conf",
    "a1_source_class",
    "a2_raw",
    "a2_conf",
    "a2_source_class",
    "c1_raw",
    "c1_conf",
    "c1_source",
    "c1_reason",
    "d_present_any",
    "d_classes_present",
    "d_conf_by_class",
    "room_occupied_candidate",
    "simple_waiting_candidate",
    "anesthesia_candidate",
    "device_present_candidate",
    "s4_treatment_or_procedure_candidate",
]


def build_observation_row(
    frame_idx: int,
    source_frame_idx: int,
    timestamp_sec: float,
    a_result: dict[str, Any],
    c_result: dict[str, Any],
    d_result: dict[str, Any],
    config: RuntimeConfig,
) -> dict[str, Any]:
    a1 = bool(a_result.get("a1_raw", False))
    a2 = bool(a_result.get("a2_raw", False))
    c1 = bool(c_result.get("c1_raw", False))
    d_any = bool(d_result.get("d_present_any", False))

    s4_candidate = bool(
    config.s4_candidate_enabled
    and config.b_group_enabled
    and a1
    and a2
    and d_any
)

    return {
        "frame_idx": int(frame_idx),
        "source_frame_idx": int(source_frame_idx),
        "timestamp_sec": float(timestamp_sec),
        "a1_raw": a1,
        "a1_conf": float(a_result.get("a1_conf", 0.0)),
        "a1_source_class": a_result.get("a1_source_class", ""),
        "a2_raw": a2,
        "a2_conf": float(a_result.get("a2_conf", 0.0)),
        "a2_source_class": a_result.get("a2_source_class", ""),
        "c1_raw": c1,
        "c1_conf": float(c_result.get("c1_conf", 0.0)),
        "c1_source": c_result.get("c1_source", ""),
        "c1_reason": c_result.get("c1_reason", ""),
        "d_present_any": d_any,
        "d_classes_present": json.dumps(d_result.get("d_classes_present", []), ensure_ascii=False),
        "d_conf_by_class": json.dumps(d_result.get("d_conf_by_class", {}), ensure_ascii=False),
        "room_occupied_candidate": a1 or a2,
        "simple_waiting_candidate": a1 and (not a2) and (not c1),
        "anesthesia_candidate": c1,
        "device_present_candidate": d_any,
        "s4_treatment_or_procedure_candidate": s4_candidate,
    }


def empty_a_result() -> dict[str, Any]:
    return {
        "a1_raw": False,
        "a1_conf": 0.0,
        "a1_source_class": "",
        "a1_patient_bbox": None,
        "a2_raw": False,
        "a2_conf": 0.0,
        "a2_source_class": "",
        "a_detections": [],
    }


def empty_d_result() -> dict[str, Any]:
    return {
        "d_present_any": False,
        "d_classes_present": [],
        "d_conf_by_class": {},
        "d_detections": [],
    }


def output_schema() -> dict[str, Any]:
    return {
        "frame_observations": OBSERVATION_COLUMNS,
        "crm_status_map": CRM_STATUS_MAP,
    }